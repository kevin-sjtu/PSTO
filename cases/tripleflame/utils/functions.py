import torch
import torch.nn.functional as F
import cantera as ct
import numpy as np
from multiprocessing import Pool, cpu_count, get_context


STATE_SPECIES = ('H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'N2')
_coefficient_gases = {}


def boundary_halo(state, padding=1):
    return F.pad(state, (padding,) * 4, mode='replicate')


def normalize_species(species):
    positive = species.clamp_min(0)
    total = positive.sum(dim=1, keepdim=True)
    fallback = torch.cat((torch.zeros_like(positive[:, :-1]),
                          torch.ones_like(positive[:, -1:])), dim=1)
    return torch.where(total > 0, positive / total.clamp_min(1e-12), fallback)


def finish_transport(output, inlet, data_min, data_max):
    output = enforce_boundaries(output.clamp(min=data_min, max=data_max), inlet)
    return torch.cat((output[:, :1], normalize_species(output[:, 1:-2]),
                      output[:, -2:]), dim=1)


def channel_bounds(data, channels, start=0, stop=None, chunk_size=16):
    stop = len(data) if stop is None else stop
    if not 0 <= start < stop <= len(data):
        raise ValueError('Require a nonempty frame interval inside the dataset')
    lower = torch.full((len(channels), 1, 1), float('inf'))
    upper = torch.full_like(lower, -float('inf'))
    for first in range(start, stop, chunk_size):
        block = torch.as_tensor(data[first:min(first + chunk_size, stop)]).float()[:, channels]
        lower = torch.minimum(lower, block.amin((0, 2, 3)).view(-1, 1, 1))
        upper = torch.maximum(upper, block.amax((0, 2, 3)).view(-1, 1, 1))
    return lower, upper


def rollout_channel_bounds(initial, reference):
    lower = torch.minimum(initial.amin((-2, -1)), reference.amin((1, 3, 4)))
    upper = torch.maximum(initial.amax((-2, -1)), reference.amax((1, 3, 4)))
    return lower[..., None, None], upper[..., None, None]


def normalize_error_channels(field, data_min, data_max):
    shape = (1,) * (field.ndim - 3) + (field.shape[-3], 1, 1)
    if torch.is_tensor(field):
        lower = torch.as_tensor(data_min, device=field.device, dtype=field.dtype).reshape(shape)
        upper = torch.as_tensor(data_max, device=field.device, dtype=field.dtype).reshape(shape)
        scale = (upper - lower).clamp_min(1e-30)
    else:
        lower = np.asarray(data_min, dtype=field.dtype).reshape(shape)
        upper = np.asarray(data_max, dtype=field.dtype).reshape(shape)
        scale = np.maximum(upper - lower, 1e-30)
    return (field - lower) / scale


def _physical_state_array(state):
    array = state.detach().cpu().numpy() if torch.is_tensor(state) else np.asarray(state)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 4 or array.shape[1] not in (10, 12):
        raise ValueError(f'Expected physical NCHW [T, nine Y, optional u,v], got {array.shape}')
    if not np.isfinite(array).all() or np.any(array[:, 0] <= 0):
        raise ValueError('Cantera input must have finite fields and positive temperature')
    if np.any(array[:, 1:10] < -1e-12) or np.any(array[:, 1:10].sum(1) <= 0):
        raise ValueError('Cantera input has negative mass fractions or an empty mixture')
    return array


def _species_indices(gas):
    return [gas.species_index(name) for name in STATE_SPECIES]


def cantera_coefficients(state, p=101325.0, mech_file='mech/h2o2.yaml'):
    if not np.isfinite(p) or p <= 0:
        raise ValueError('Cantera pressure must be positive and finite')
    array = _physical_state_array(state)
    gas = _coefficient_gases.get(mech_file)
    if gas is None:
        gas = ct.Solution(mech_file)
        _coefficient_gases[mech_file] = gas
    batch, _, height, width = array.shape
    points = array[:, :10].transpose(0, 2, 3, 1).reshape(-1, 10)
    species = np.zeros((len(points), gas.n_species), dtype=np.float64)
    species[:, _species_indices(gas)] = np.maximum(points[:, 1:], 0.0)
    states = ct.SolutionArray(gas, shape=(len(points),))
    try:
        states.TPY = points[:, 0], float(p), species
        values = np.column_stack((states.density, states.viscosity,
                                  states.thermal_conductivity / states.cp_mass))
    except ct.CanteraError as exc:
        raise ValueError(f'Cantera coefficient evaluation failed: {exc}') from exc
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Cantera returned nonpositive or nonfinite coefficients')
    result = values.reshape(batch, height, width, 3).transpose(0, 3, 1, 2).copy()
    if torch.is_tensor(state):
        return torch.as_tensor(result, dtype=state.dtype, device=state.device)
    return result

def fast_physics_compute(data_batch, p, mech_file, n_procs=None):
    array = _physical_state_array(data_batch)
    rho, mu, kappa = np.split(cantera_coefficients(array, p, mech_file), 3, axis=1)
    return np.concatenate((kappa / rho, mu / rho), axis=1).astype(np.float32)

def _init_T_worker(mech_file, pressure):
    global _T_worker_gas, _T_worker_p
    _T_worker_gas = ct.Solution(mech_file)
    _T_worker_p = pressure

def _compute_T_task(args):
    T_chunk, Y_prev_chunk, Y_next_chunk = args
    gas, p = _T_worker_gas, _T_worker_p

    original_shape = T_chunk.shape
    T_chunk = np.asarray(T_chunk)

    T_out = T_chunk.copy()
    mask = T_chunk > 310.0

    if not np.any(mask):
        return T_out.reshape(original_shape)

    Y_prev = np.moveaxis(Y_prev_chunk, 0, -1)
    Y_next = np.moveaxis(Y_next_chunk, 0, -1)

    T_masked = T_chunk[mask]
    Y_prev_masked = Y_prev[mask]
    Y_next_masked = Y_next[mask]

    try:
        states_prev = ct.SolutionArray(gas, shape=T_masked.shape)
        states_prev.TPY = T_masked, p, Y_prev_masked
        states_next = ct.SolutionArray(gas, shape=T_masked.shape)
        states_next.HPY = states_prev.enthalpy_mass, p, Y_next_masked
        T_out[mask] = states_next.T
    except ct.CanteraError as exc:
        raise ValueError(f'Cantera temperature recovery failed: {exc}') from exc

    return T_out.reshape(original_shape)

def compute_temperature_from_Y(T_prev, Y_prev, Y_next, p=101325.0, mech_file=None, n_procs=None):
    T_prev, Y_prev, Y_next = np.asarray(T_prev), np.asarray(Y_prev), np.asarray(Y_next)

    B = T_prev.shape[0]
    num_workers = min(B, cpu_count()) if n_procs is None else min(B, n_procs, cpu_count())
    chunks = [(T_prev[i], Y_prev[i], Y_next[i]) for i in range(B)]

    if num_workers <= 1:
        _init_T_worker(mech_file, p)
        results = [_compute_T_task(chunk) for chunk in chunks]
    else:
        with Pool(processes=num_workers, initializer=_init_T_worker, initargs=(mech_file, p)) as pool:
            results = pool.map(_compute_T_task, chunks)

    return np.stack(results, axis=0)

def _advance_reaction_points(points, gas, pressure, time_step, rtol, atol):
    indices = _species_indices(gas)
    out = np.empty_like(points)
    for i, row in enumerate(points):
        mixture = np.zeros(gas.n_species)
        mixture[indices] = np.maximum(row[1:], 0.0)
        try:
            gas.TPY = float(row[0]), pressure, mixture
            reactor = ct.IdealGasConstPressureReactor(gas, clone=True)
            solver = ct.ReactorNet([reactor])
            solver.rtol, solver.atol = rtol, atol
            solver.advance(time_step)
            out[i, 0] = reactor.T
            out[i, 1:] = reactor.phase.Y[indices]
        except ct.CanteraError as exc:
            raise ValueError(f'Cantera reaction integration failed at local point {i}: {exc}') from exc
    return out


def _init_reaction_worker(mech_file, pressure, time_step, rtol, atol):
    global _reaction_worker
    _reaction_worker = (ct.Solution(mech_file), pressure, time_step, rtol, atol)


def _reaction_chunk(points):
    return _advance_reaction_points(points, *_reaction_worker)


class cantera_source:
    def __init__(self, n_procs=None, time_step=1e-6, mech_file='mech/h2o2.yaml',
                 pressure=101325.0, rtol=1e-9, atol=1e-15):
        if not np.isfinite(time_step) or time_step < 0 or not np.isfinite(pressure) or pressure <= 0:
            raise ValueError('Reaction time step must be nonnegative and pressure positive')
        self.mech_file, self.pressure = mech_file, float(pressure)
        self.n_procs, self.time_step = n_procs, float(time_step)
        self.rtol, self.atol = rtol, atol
        self._gas = ct.Solution(mech_file)
        self._pool = None

    def start_workers(self):
        workers = min(self.n_procs or 1, cpu_count())
        if workers > 1 and self._pool is None:
            self._pool = get_context('fork').Pool(
                workers, initializer=_init_reaction_worker,
                initargs=(self.mech_file, self.pressure, self.time_step, self.rtol, self.atol))
            self._pool.map(_reaction_chunk, [np.empty((0, 10)) for _ in range(workers)])

    def __call__(self, state):
        array = _physical_state_array(state)
        if array.shape[1] != 12:
            raise ValueError('Reaction interface expects [T, nine Y, u,v]')
        if self.time_step == 0:
            return state.clone() if torch.is_tensor(state) else array.copy()
        batch, _, height, width = array.shape
        points = array[:, :10].transpose(0, 2, 3, 1).reshape(-1, 10)
        workers = min(self.n_procs or 1, len(points), cpu_count())
        if workers <= 1:
            out = _advance_reaction_points(points, self._gas, self.pressure,
                                           self.time_step, self.rtol, self.atol)
        else:
            if self._pool is None:
                self.start_workers()
            chunks = [chunk for chunk in np.array_split(points, workers) if len(chunk)]
            out = np.concatenate(self._pool.map(_reaction_chunk, chunks))
        if not np.isfinite(out).all() or np.any(out[:, 0] <= 0):
            raise ValueError('Cantera returned invalid reaction states')
        result = out.reshape(batch, height, width, 10).transpose(0, 3, 1, 2)
        if torch.is_tensor(state):
            physical = torch.as_tensor(result.copy(), device=state.device, dtype=state.dtype)
            return torch.cat((physical, state[:, -2:].detach()), dim=1)
        return np.concatenate((result, array[:, -2:]), axis=1)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

class EfficientDataset(torch.utils.data.Dataset):
    def __init__(self, lf_data, hfx_data, hfy_data, n_lf, n_hf, chs, stats,
                 pressure_reference=None, normalize_state=True):
        self.lf_data = lf_data
        self.hfx_data = hfx_data
        self.hfy_data = hfy_data
        self.n_lf = n_lf
        self.n_hf = n_hf
        self.chs = chs
        self.pressure_reference = pressure_reference
        self.normalize_state = normalize_state
        if len(stats) == 2:
            self.data_max, self.data_min = stats
            self.return_physics = False
        elif len(stats) == 4:
            self.data_max, self.data_min, _, _ = stats
            self.return_physics = True
        else:
            raise ValueError("stats must be (data_max,data_min) or (data_max,data_min,*,*)")

    def __len__(self):
        return self.n_lf + self.n_hf

    def __getitem__(self, idx):
        rx, ry = (self.lf_data[idx], self.lf_data[idx + 1]) if idx < self.n_lf else (self.hfx_data[idx - self.n_lf], self.hfy_data[idx - self.n_lf])
        if self.normalize_state:
            x = normalization(rx[self.chs], self.data_max, self.data_min)
            y = normalization(ry[self.chs], self.data_max, self.data_min)
        else:
            x, y = rx[self.chs], ry[self.chs]
        if not self.return_physics:
            return x, y
        rho, cp, k, mu = rx[-4:-3], rx[-3:-2], rx[-2:-1], rx[-1:]
        if self.pressure_reference is not None:
            rho = rho * (self.pressure_reference / rx[10:11])
        kappa_T = k / (cp + 1e-12)
        p = torch.cat([rho, mu, kappa_T], dim=0)
        return x, y, p

def get_parameter_net(net):
    total_num = sum(p.numel() for p in net.parameters())
    train_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    real_dof = sum(
        p.numel() * (2 if p.is_complex() else 1)
        for p in net.parameters()
        if p.requires_grad
    )
    print(
        f"parameter elements: {train_num:,} | real degrees of freedom: {real_dof:,} "
        f"(all parameters: {total_num:,})"
    )
    return total_num, train_num

def normalization(data, max, min):
    span = max - min
    span = span.clamp_min(1e-30) if torch.is_tensor(span) else np.maximum(span, 1e-30)
    return 2 * (data - min) / span - 1

def denormalization(data, max, min):
    return ((data + 1) * (max - min) / 2) + min

def guided_coeff_preminmax(rho, mu, kappa_T):
    return rho, torch.log(mu.clamp_min(1e-12)), torch.log(kappa_T.clamp_min(1e-12))

def standardization(data, mean, std):
    return (data - mean) / std

def destandardization(data, mean, std):
    return (data * std) + mean

def BCT(x, lam = 0.1):
    if lam == 0:
        return torch.log(x)
    return (torch.pow(x, lam) - 1)/lam

def inBCT(x, lam = 0.1):
    if lam == 0:
        return torch.exp(x)
    return torch.pow((lam*x+1), 1/lam)

def Y_correct(Y):
    Y = torch.clamp(Y, min=0, max=1.0)
    Y_sum = Y.sum(dim=1, keepdim=True)
    return Y / Y_sum

def compute_relative_error(pred, truth):
    pred_t = pred if isinstance(pred, torch.Tensor) else torch.as_tensor(pred)
    truth_t = truth if isinstance(truth, torch.Tensor) else torch.as_tensor(truth)
    rel = torch.norm(pred_t - truth_t) / (torch.norm(truth_t)) * 100.0
    return rel

def compute_lewis_numbers(mech_file):
    gas = ct.Solution(mech_file)
    gas.TPX = 1000.0, ct.one_atm, 'H2:1.0, O2:0.5, N2:1.88'
    alpha = gas.thermal_conductivity / (gas.density * gas.cp_mass)
    D_im = gas.mix_diff_coeffs_mass
    lewis_numbers = alpha / D_im
    results = {}
    for i, name in enumerate(gas.species_names):
        results[name] = lewis_numbers[i]

    return results


def enforce_boundaries(output, reference):
    rows = torch.cat((output[..., 1:2, :], output[..., 1:-1, :], output[..., -2:-1, :]), dim=-2)
    return torch.cat((reference[..., :1], rows[..., 1:-1], rows[..., -2:-1]), dim=-1)


def diffusive_flux_divergence(state, coefficient, padding=0):
    halo = boundary_halo(state, padding + 1)
    a = boundary_halo(coefficient, padding + 1)
    fx = .5 * (a[..., 1:] + a[..., :-1]) * (halo[..., 1:] - halo[..., :-1])
    fy = .5 * (a[..., 1:, :] + a[..., :-1, :]) * (halo[..., 1:, :] - halo[..., :-1, :])
    return fx[..., 1:-1, 1:] - fx[..., 1:-1, :-1] + fy[..., 1:, 1:-1] - fy[..., :-1, 1:-1]


class ContinuityFlux:
    @staticmethod
    def face_flux(velocity, density, inlet_velocity=None, inlet_density=None):
        mass = density * velocity
        inlet = mass[:, :1, :, :1]
        if inlet_velocity is not None:
            inlet = inlet_density * inlet_velocity[:, :1, :, :1]
        fx = torch.cat((inlet, 0.5 * (mass[:, :1, :, :-1] + mass[:, :1, :, 1:]),
                        mass[:, :1, :, -1:]), dim=-1)
        fy = torch.cat((mass[:, 1:, :1],
                        .5*(mass[:, 1:, :-1] + mass[:, 1:, 1:]), mass[:, 1:, -1:]), dim=-2)
        return fx, fy

    @staticmethod
    def divergence(flux_x, flux_y, dx):
        return (flux_x[..., 1:] - flux_x[..., :-1] +
                flux_y[..., 1:, :] - flux_y[..., :-1, :]) / dx


def load_transport_state_dict(model, state, strict=True):
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    state = dict(state)
    backend_keys = ('coeff_input_mean', 'coeff_input_std', 'coeff_target_mean',
                    'coeff_target_std', 'coefficient_pressure',
                    'species_molecular_weights', 'gas_constant')
    current = model.state_dict()
    if model.coefficient_backend == 'cantera' or model._coefficient_checkpoint_loaded:
        state = {key: value for key, value in state.items()
                 if not key.startswith(('coefficients.', 'coefficient_model.')) and key not in backend_keys}
        state.update({key: value for key, value in current.items()
                      if key.startswith(('coefficients.', 'coefficient_model.')) or key in backend_keys})
    return model.load_state_dict(state, strict=strict)


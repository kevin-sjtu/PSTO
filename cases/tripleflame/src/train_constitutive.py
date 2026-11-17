

import argparse
import json
import os
import shutil
from datetime import datetime

import cantera as ct
import h5py
import numpy as np
import torch
from torch import nn

if __package__:
    from .DNN import NN_MLP, COEFFICIENT_LAYERS
else:
    from DNN import NN_MLP, COEFFICIENT_LAYERS


class ConstitutiveTransport(nn.Module):


    def _init_constitutive(self, num_phys_vars, data_min, data_max, dt, dx, lewis,
                          coefficient_checkpoint, coefficient_backend, mechanism,
                          pressure):
        from utils.functions import STATE_SPECIES

        if coefficient_backend not in ('cantera', 'dnn'):
            raise ValueError('coefficient_backend must be cantera or dnn')
        self.coefficient_backend = coefficient_backend
        self.mechanism = mechanism
        self._coefficient_checkpoint_loaded = coefficient_backend == 'dnn' and coefficient_checkpoint is not None
        self.register_buffer('coefficient_pressure', torch.tensor(float(pressure)))
        gas = ct.Solution(mechanism)
        self.register_buffer('species_molecular_weights', torch.tensor(
            [gas.molecular_weights[gas.species_index(name)] for name in STATE_SPECIES],
            dtype=torch.float64))
        self.register_buffer('gas_constant', torch.tensor(ct.gas_constant, dtype=torch.float64))
        self.N = int(num_phys_vars)
        self.Ns = self.N - 2
        self.register_buffer('data_min', torch.as_tensor(
            torch.full((self.N, 1, 1), -torch.inf) if data_min is None else data_min
        ).float().reshape(self.N, 1, 1).clone())
        self.register_buffer('data_max', torch.as_tensor(
            torch.full((self.N, 1, 1), torch.inf) if data_max is None else data_max
        ).float().reshape(self.N, 1, 1).clone())
        self.register_buffer('time_step', torch.tensor(float(dt)))
        self.register_buffer('grid_step', torch.tensor(float(dx)))
        le = [1.0] + (list(lewis) if lewis is not None else [1.0] * 9)
        self.register_buffer('lewis', torch.tensor(le, dtype=self.time_step.dtype).reshape(1, self.Ns, 1, 1))
        self.coefficient_model = NN_MLP(COEFFICIENT_LAYERS) if coefficient_backend == 'dnn' else None
        self.register_buffer('coeff_input_mean', torch.zeros(10))
        self.register_buffer('coeff_input_std', torch.ones(10))
        self.register_buffer('coeff_target_mean', torch.zeros(3))
        self.register_buffer('coeff_target_std', torch.ones(3))
        if coefficient_backend == 'dnn' and coefficient_checkpoint is not None:
            saved = torch.load(coefficient_checkpoint, map_location='cpu', weights_only=False)
            if saved.get('preprocessing') != 'BCT_0.1_then_train_standardization':
                raise ValueError('Expected a reaction-style BCT coefficient checkpoint')
            if 'net' not in saved:
                raise ValueError('Expected a single three-output constitutive DNN checkpoint; retrain old net0/net1/net2 models')
            self.coefficient_model.load_state_dict(saved['net'])
            for buffer_name, saved_name in (
                ('coeff_input_mean', 'data_in_mean'),
                ('coeff_input_std', 'data_in_std'),
                ('coeff_target_mean', 'data_target_mean'),
                ('coeff_target_std', 'data_target_std'),
            ):
                getattr(self, buffer_name).copy_(torch.as_tensor(saved[saved_name]))
        if self.coefficient_model is not None:
            self.coefficient_model.requires_grad_(False)
            self.coefficient_model.eval()
        self.transport_velocity = None
        self.transport_coefficients = None


    def train(self, mode=True):
        super().train(mode)
        if self.coefficient_model is not None:
            self.coefficient_model.eval()
        return self


    def set_stats(self, data_max, data_min):
        self.data_max.copy_(data_max.to(self.data_max))
        self.data_min.copy_(data_min.to(self.data_min))


    def physical_density(self, physical):

        from src.thermodynamics import ideal_gas_density

        return ideal_gas_density(physical, self.species_molecular_weights,
                                 self.gas_constant, self.coefficient_pressure)


    def physical_coefficients(self, physical, supplied=None):
        if supplied is not None:
            if self.coefficient_backend != 'cantera':
                raise ValueError('Precomputed Cantera coefficients require coefficient_backend=cantera')
            expected = (physical.shape[0], 3, *physical.shape[2:])
            if tuple(supplied.shape) != expected:
                raise ValueError(f'Expected physical [rho,mu,kappa] coefficients {expected}')
            if not torch.isfinite(supplied).all() or torch.any(supplied <= 0):
                raise ValueError('Supplied physical coefficients must be finite and positive')
            return physical, supplied.detach().to(device=physical.device, dtype=physical.dtype)
        if self.coefficient_backend == 'cantera':
            from utils.functions import cantera_coefficients
            coeff = cantera_coefficients(physical, p=float(self.coefficient_pressure),
                                         mech_file=self.mechanism)
            return physical, coeff
        batch, _, height, width = physical.shape
        points = physical[:, :self.Ns].permute(0, 2, 3, 1).reshape(-1, self.Ns)
        
        outputs = []
        for chunk in points.split(16384):
            outputs.append(self._coefficient_chunk(chunk))
        coefficients = torch.cat(outputs).reshape(batch, height, width, 3).permute(0, 3, 1, 2)
        if not torch.isfinite(coefficients).all() or torch.any(coefficients <= 0):
            raise ValueError('Constitutive DNN returned nonpositive or nonfinite physical coefficients')
        return physical, coefficients


    def _coefficient_chunk(self, chunk):
        with torch.autocast(device_type=chunk.device.type, enabled=False):
            local = chunk.float()
            species = local[:, 1:].abs()
            safe_species = species.clamp_min(torch.finfo(species.dtype).tiny)
            species_bct = torch.where(species > 0,
                                      (safe_species.pow(0.1) - 1.0) / 0.1,
                                      torch.full_like(species, -10.0))
            local_bct = torch.cat((local[:, :1], species_bct), dim=1)
            local_standard = (local_bct - self.coeff_input_mean) / self.coeff_input_std
            predicted_standard = self.coefficient_model(local_standard)
            predicted_bct = predicted_standard * self.coeff_target_std + self.coeff_target_mean
            return (1.0 + 0.1 * predicted_bct).clamp_min(0.0).pow(10.0)


def bct(values, lam=0.1):

    return (np.power(values, lam) - 1.0) / lam


def prepare_zero_dimensional_data(features, labels):

    cut = int(0.8 * len(features))
    input_bct = np.abs(features).astype(np.float64, copy=True)
    input_bct[:, 1:] = bct(input_bct[:, 1:])
    target_bct = bct(np.abs(labels).astype(np.float64))
    input_mean = input_bct[:cut].mean(0)
    input_std = input_bct[:cut].std(0, ddof=1)
    target_mean = target_bct[:cut].mean(0)
    target_std = target_bct[:cut].std(0, ddof=1)
    input_std[input_std == 0] = 1.0
    target_std[target_std == 0] = 1.0
    x = ((input_bct - input_mean) / input_std).astype(np.float32)
    y = ((target_bct - target_mean) / target_std).astype(np.float32)
    stats = dict(data_in_mean=input_mean.astype(np.float32),
                 data_in_std=input_std.astype(np.float32),
                 data_target_mean=target_mean.astype(np.float32),
                 data_target_std=target_std.astype(np.float32))
    return x[:cut], y[:cut], x[cut:], y[cut:], stats


def sample_initial_states(data, n_samples, sampling, seed):

    if data.ndim != 2 or data.shape[1] != 44:
        raise ValueError(f"Expected 44-column initial/next-state data, got {data.shape}")
    if not 30000 <= n_samples <= 60000:
        raise ValueError("Use 30000--60000 constitutive pairs")
    if sampling not in ('inverse_group_bins', 'uniform_bins'):
        raise ValueError(f"Unknown sampling mode: {sampling}")
    states = np.asarray(data[:, :12])
    if not np.isfinite(states).all() or np.any(states[:, 0] <= 0):
        raise ValueError("Source has invalid initial states")
    if np.any(states[:, 2:] < 0) or not np.allclose(states[:, 2:].sum(1), 1, atol=1e-6):
        raise ValueError("Source mass fractions must be nonnegative and sum to one")
    if not np.allclose(states[:, 1], ct.one_atm, rtol=0, atol=1e-3):
        raise ValueError("The current 10-input constitutive MLP requires fixed 101325 Pa")
    if np.any(states[:, 10] != 0):
        raise ValueError("The current nine-species input requires Ar=0")
    temperature = states[:, 0]
    bin_ids = np.floor(temperature / 200).astype(int)
    bins = np.unique(bin_ids)
    rng = np.random.default_rng(seed)
    pools = {int(b): np.flatnonzero(bin_ids == b) for b in bins}
    groups = [bins[bins < 4], bins[bins >= 4]]
    if any(len(g) == 0 for g in groups):
        raise ValueError("Both temperature groups must have samples")
    if sampling == 'inverse_group_bins':
        
        
        weights = np.array([0.5 / len(groups[int(b >= 4)]) for b in bins])
    else:
        weights = np.full(len(bins), 1 / len(bins))
    def quotas(total):
        raw = total * weights
        counts = np.floor(raw).astype(int)
        order = np.argsort(-(raw - counts), kind='stable')
        counts[order[:total - counts.sum()]] += 1
        return counts
    n_train = int(0.8 * n_samples)
    requested = [quotas(n_train), quotas(n_samples - n_train)]
    selected = [[], []]
    rows = []
    for k, b in enumerate(bins):
        ids = rng.permutation(pools[int(b)])
        if len(ids) < 2:
            raise ValueError(f"Bin {b*200}--{(b+1)*200} K cannot have disjoint train/val pools")
        cut = min(max(int(0.8 * len(ids)), 1), len(ids) - 1)
        split_pools = [ids[:cut], ids[cut:]]
        for split, pool in enumerate(split_pools):
            count = int(requested[split][k])
            selected[split].append(rng.choice(pool, count, replace=count > len(pool)))
        rows.append(dict(lower_K=int(b*200), upper_K=int((b+1)*200),
                         source_count=len(ids), source_train_unique=cut,
                         source_val_unique=len(ids)-cut, train_count=int(requested[0][k]),
                         val_count=int(requested[1][k]),
                         count=int(requested[0][k]+requested[1][k]),
                         per_source_state_probability=float(weights[k]/len(ids))))
    train_ids = rng.permutation(np.concatenate(selected[0]))
    val_ids = rng.permutation(np.concatenate(selected[1]))
    if np.intersect1d(train_ids, val_ids).size:
        raise RuntimeError("Sampled source rows overlap between train and validation")
    ids = np.concatenate((train_ids, val_ids))
    report = dict(sampling=sampling, seed=seed, n_source=len(data), n_samples=len(ids),
                  n_train=len(train_ids), n_val=len(val_ids), bin_width_K=200,
                  source_temperature_range_K=[float(temperature.min()), float(temperature.max())],
                  sampled_temperature_range_K=[float(temperature[ids].min()), float(temperature[ids].max())],
                  source_below_800=int((temperature < 800).sum()),
                  source_equal_800=int((temperature == 800).sum()),
                  source_above_800=int((temperature > 800).sum()),
                  sampled_below_800=int((temperature[ids] < 800).sum()),
                  sampled_equal_800=int((temperature[ids] == 800).sum()),
                  sampled_above_800=int((temperature[ids] > 800).sum()),
                  unique_sampled_rows=int(np.unique(ids).size),
                  train_val_shared_source_rows=0, temperature_bins=rows,
                  split_note='Source-row-disjoint, not an independent CFD-trajectory split; '
                             'rare bins are resampled within each split.')
    print(json.dumps(report, indent=2), flush=True)
    return ids, report


def build_labels(source, mechanism, n_samples=45000, sampling='inverse_group_bins', seed=2048):
    data = np.load(source, mmap_mode="r")
    indices, report = sample_initial_states(data, n_samples, sampling, seed)
    data = data[indices]
    
    temperature = data[:, 0]
    species = np.concatenate((data[:, 2:10], data[:, 11:12]), axis=1)
    pressure = data[:, 1]
    features = np.column_stack((temperature, species)).astype(np.float32)
    full_species = np.maximum(data[:, 2:12], 0).astype(np.float32)
    pressures = pressure.astype(np.float32)
    gas = ct.Solution(mechanism)
    labels = np.empty((len(features), 3), dtype=np.float32)
    for i in range(len(features)):
        gas.TPY = float(features[i, 0]), float(pressures[i]), full_species[i]
        labels[i] = [gas.density, gas.viscosity,
                     gas.thermal_conductivity / gas.cp_mass]
    if not np.isfinite(labels).all() or np.any(labels <= 0):
        raise ValueError("Cantera produced invalid property labels")
    return features, labels, indices, report


def load_or_create_h5(path, source, mechanism, rebuild=False, n_samples=45000,
                      sampling='inverse_group_bins', seed=2048):
    if os.path.exists(path) and not rebuild:
        with h5py.File(path, "r") as h5:
            features = h5["input_T_Y"][:]
            labels = h5["target_rho_mu_kappa_T"][:]
        if features.ndim != 2 or features.shape[1] != 10 or labels.shape != (len(features), 3):
            raise ValueError(f"Invalid constitutive data shapes: {features.shape}, {labels.shape}")
        if not np.isfinite(features).all() or not np.isfinite(labels).all() or np.any(labels <= 0):
            raise ValueError("Constitutive dataset contains invalid states or property labels")
        return features, labels
    features, labels, indices, report = build_labels(source, mechanism, n_samples, sampling, seed)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    temporary = path + '.building'
    with h5py.File(temporary, "w") as h5:
        h5.create_dataset("input_T_Y", data=features, compression="gzip")
        h5.create_dataset("target_rho_mu_kappa_T", data=labels, compression="gzip")
        h5.attrs["source"] = source
        h5.attrs["mechanism"] = mechanism
        h5.attrs["pressure_Pa"] = float(ct.one_atm)
        h5.attrs["source_field_data_used"] = False
        h5.attrs['sampling'] = sampling
        h5.attrs['seed'] = seed
        h5.attrs['n_train'] = report['n_train']
        h5.attrs['n_val'] = report['n_val']
        h5.attrs['species_order'] = 'H2,H,O,O2,OH,H2O,HO2,H2O2,N2'
        h5.attrs['target_order'] = 'rho,mu,lambda/cp'
        h5.attrs['sampling_report'] = json.dumps(report)
        h5.create_dataset('source_row_index', data=indices, compression='gzip')
    with h5py.File(temporary, 'r') as h5:
        x, y = h5['input_T_Y'][:], h5['target_rho_mu_kappa_T'][:]
        if x.shape != (n_samples, 10) or y.shape != (n_samples, 3):
            raise ValueError('Generated H5 shape validation failed')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y <= 0):
            raise ValueError('Generated H5 physical label validation failed')
        if not np.allclose(x[:, 1:].sum(1), 1, atol=1e-6):
            raise ValueError('Generated H5 mass fractions do not sum to one')
        for row in report['temperature_bins']:
            count = ((x[:, 0] >= row['lower_K']) & (x[:, 0] < row['upper_K'])).sum()
            if int(count) != row['count']:
                raise ValueError('Saved temperature-bin quota validation failed')
    if os.path.exists(path):
        archive = os.path.join(os.path.dirname(path), 'archive')
        os.makedirs(archive, exist_ok=True)
        backup = os.path.join(archive, 'constitutive_data_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.h5')
        shutil.copy2(path, backup)
        print(f'Preserved previous dataset: {backup}', flush=True)
    os.replace(temporary, path)
    with open(os.path.splitext(path)[0] + '_sampling.json', 'w') as stream:
        json.dump(report, stream, indent=2)
    print(f"Generated zero-dimensional Cantera property data: {path} ({len(features)} states)", flush=True)
    return features, labels


@torch.no_grad()
def evaluate_validation(checkpoint, dataset_h5):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = NN_MLP(COEFFICIENT_LAYERS).eval()
    model.load_state_dict(saved['net'])
    with h5py.File(dataset_h5, 'r') as h5:
        cut = int(h5.attrs.get('n_train', 0.8 * len(h5['input_T_Y'])))
        inputs = h5['input_T_Y'][cut:]
        exact = h5['target_rho_mu_kappa_T'][cut:]
    transformed = inputs.astype(np.float64)
    transformed[:, 1:] = bct(transformed[:, 1:])
    standardized = ((transformed - saved['data_in_mean']) / saved['data_in_std']).astype(np.float32)
    output = np.concatenate([model(torch.from_numpy(standardized[i:i+2048])).numpy()
                             for i in range(0, len(inputs), 2048)])
    predicted_bct = output * saved['data_target_std'] + saved['data_target_mean']
    predicted = np.maximum(1.0 + 0.1 * predicted_bct, 0) ** 10
    if not np.isfinite(predicted).all() or np.any(predicted <= 0):
        raise ValueError('Validation produced nonpositive or nonfinite physical coefficients')
    relative = np.abs(predicted / exact - 1)
    def metrics(mask):
        return dict(n=int(mask.sum()), mean_relative_error=relative[mask].mean(0).tolist(),
                    p95_relative_error=np.percentile(relative[mask], 95, axis=0).tolist(),
                    max_relative_error=relative[mask].max(0).tolist())
    report = dict(checkpoint=checkpoint, dataset_h5=dataset_h5, targets=['rho', 'mu', 'lambda/cp'],
                  overall=metrics(np.ones(len(inputs), dtype=bool)), temperature_bins=[])
    for lower in np.unique(np.floor(inputs[:, 0] / 200).astype(int)) * 200:
        mask = (inputs[:, 0] >= lower) & (inputs[:, 0] < lower+200)
        report['temperature_bins'].append(dict(lower_K=int(lower), upper_K=int(lower+200), **metrics(mask)))
    output_path = os.path.splitext(checkpoint)[0] + '_validation.json'
    with open(output_path, 'w') as stream:
        json.dump(report, stream, indent=2)
    print('Physical coefficient validation:', json.dumps(report['overall']), flush=True)
    print(f'Per-temperature validation saved: {output_path}', flush=True)


@torch.no_grad()
def evaluate_cases(checkpoint, mechanism, n_points, cases):
    model = NN_MLP(COEFFICIENT_LAYERS).eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "net" not in saved:
        raise ValueError("Expected a single three-output constitutive DNN checkpoint; retrain old net0/net1/net2 models")
    model.load_state_dict(saved["net"])
    gas = ct.Solution(mechanism)
    rng = np.random.default_rng(88)
    species_channels = [0, 4, 5, 1, 3, 2, 6, 7, 8]
    for case, path, times in (
        ("CD", "data/H2_tripleflow.h5", [0, 749, 3000, 4500]),
        ("CDR", "data/H2_tripleflame.h5", [0, 749, 3000, 5000]),
    ):
        if case not in cases:
            continue
        samples = []
        with h5py.File(path, "r") as h5:
            for time_index in times:
                field = h5["data"][time_index]
                positions = rng.choice(field.shape[-2] * field.shape[-1],
                                       size=n_points // len(times), replace=False)
                flat = field.reshape(field.shape[0], -1)
                local = np.column_stack((flat[9, positions],
                                         flat[species_channels][:, positions].T))
                samples.append(local)
        inputs = np.concatenate(samples).astype(np.float32)
        transformed = np.abs(inputs).astype(np.float64)
        transformed[:, 1:] = bct(transformed[:, 1:])
        standardized = (transformed - saved["data_in_mean"]) / saved["data_in_std"]
        network_input = torch.from_numpy(standardized.astype(np.float32))
        predicted_standard = model(network_input).numpy()
        predicted_bct = predicted_standard * saved["data_target_std"] + saved["data_target_mean"]
        predicted = np.maximum(1.0 + 0.1 * predicted_bct, 0.0) ** 10
        exact = np.empty((len(inputs), 3), dtype=np.float64)
        for i, row in enumerate(inputs):
            y = np.insert(np.maximum(row[1:], 0.0), 8, 0.0)
            gas.TPY = float(row[0]), ct.one_atm, y
            exact[i] = (gas.density, gas.viscosity,
                        gas.thermal_conductivity / gas.cp_mass)
        relative = np.abs(predicted / exact - 1.0)
        print(f"{case} vs Cantera, n={len(inputs)}, T=[{inputs[:, 0].min():.1f},"
              f" {inputs[:, 0].max():.1f}] K: mean relative error"
              f" rho={relative[:,0].mean():.5f}, mu={relative[:,1].mean():.5f},"
              f" kappa_T={relative[:,2].mean():.5f};"
              f" 95th percentile={np.percentile(relative,95,axis=0).tolist()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--print_every", type=int, default=2)
    parser.add_argument("--early_stop_patience", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source", default="data/0d_dataset_cfd.npy",
                        help="Source used only when the existing constitutive H5 is absent.")
    parser.add_argument("--mechanism", default="mech/h2o2.yaml")
    parser.add_argument("--output", default="checkpoints/models/model_coeff/constitutive_DNN_BCT.pt")
    parser.add_argument("--dataset_h5", default="data/constitutive_data.h5")
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument('--rebuild_dataset', action='store_true')
    parser.add_argument('--n_samples', type=int, default=45000)
    parser.add_argument('--sampling', choices=('inverse_group_bins', 'uniform_bins'),
                        default='inverse_group_bins')
    parser.add_argument('--seed', type=int, default=2048)
    parser.add_argument("--dry_run", action="store_true",
                        help="Check existing 0-D data and one forward/backward pass; do not train or save.")
    parser.add_argument("--evaluate_cases", action="store_true")
    parser.add_argument('--evaluate_validation', action='store_true')
    parser.add_argument("--case_samples", type=int, default=2048)
    parser.add_argument("--cases", nargs="+", choices=("CD", "CDR"),
                        default=["CD"], help="CD is training-domain diagnostics; reserve CDR for final tests.")
    args = parser.parse_args()
    if args.evaluate_validation:
        evaluate_validation(args.output, args.dataset_h5)
        return
    if args.evaluate_cases:
        evaluate_cases(args.output, args.mechanism, args.case_samples, args.cases)
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    features, labels = load_or_create_h5(args.dataset_h5, args.source, args.mechanism,
                                        args.rebuild_dataset, args.n_samples, args.sampling, args.seed)
    
    if args.generate_only:
        return
    x_train_np, y_train_np, x_val_np, y_val_np, stats = prepare_zero_dimensional_data(features, labels)
    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    x_val = torch.from_numpy(x_val_np)
    y_val = torch.from_numpy(y_val_np)
    model = NN_MLP(COEFFICIENT_LAYERS).to(args.device)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if args.dry_run:
        x, y = x_train[:16].to(args.device), y_train[:16].to(args.device)
        pred = model(x)
        loss = torch.nn.functional.l1_loss(pred, y)
        loss.backward()
        print(f"0-D dry run: n={len(features)}, input={tuple(x.shape)}, "
              f"output={tuple(pred.shape)}, BCT-standardized L1={loss.item():.6g}; "
              "no optimizer step or checkpoint", flush=True)
        return
    if os.path.exists(args.output):
        archive = os.path.join(os.path.dirname(args.output), 'archive')
        os.makedirs(archive, exist_ok=True)
        backup = os.path.join(archive, datetime.now().strftime('%Y%m%d_%H%M%S_%f_') + os.path.basename(args.output))
        shutil.copy2(args.output, backup)
        print(f'Preserved previous checkpoint: {backup}', flush=True)
    best = float('inf')
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        permutation = np.random.permutation(len(x_train))
        train_losses = []
        for start in range(0, len(x_train), args.batch_size):
            index = permutation[start:start + args.batch_size]
            x = x_train[index].to(args.device)
            y = y_train[index].to(args.device)
            predicted = model(x)
            loss = torch.nn.functional.l1_loss(predicted, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        if (epoch + 1) % args.print_every == 0:
            model.eval()
            total_abs = total_sq = count = 0.0
            with torch.no_grad():
                for start in range(0, len(x_val), args.batch_size):
                    x = x_val[start:start + args.batch_size].to(args.device)
                    y = y_val[start:start + args.batch_size].to(args.device)
                    predicted = model(x)
                    error = predicted - y
                    total_abs += error.abs().sum().item()
                    total_sq += error.square().sum().item()
                    count += error.numel()
            val_l1, val_mse = total_abs / count, total_sq / count
            print(f"epoch {epoch+1}: train L1={np.mean(train_losses):.5e}, "
                  f"val L1={val_l1:.5e}, val MSE={val_mse:.5e}, "
                  f"val RMSE={np.sqrt(val_mse):.5e}", flush=True)
            if val_l1 < best:
                best, best_epoch = val_l1, epoch + 1
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                torch.save({**stats,
                            'net': {key: value.detach().cpu()
                                    for key, value in model.state_dict().items()},
                            'preprocessing': 'BCT_0.1_then_train_standardization',
                            'architecture': list(COEFFICIENT_LAYERS),
                            'source': args.source, 'n_train': len(x_train),
                            'dataset_h5': args.dataset_h5, 'seed': args.seed,
                            'n_val': len(x_val), 'epoch': best_epoch,
                            'val_l1': val_l1, 'val_mse': val_mse}, args.output)
            if args.early_stop_patience and epoch + 1 - best_epoch >= args.early_stop_patience:
                print(f"Early stop after {args.early_stop_patience} epochs without improvement", flush=True)
                break
        if epoch + 1 in (1000, 2000):
            lr *= 0.4
            for group in optimizer.param_groups:
                group['lr'] = lr
            print(f"LR decay at epoch {epoch+1}: {lr:.2e}", flush=True)
    print(f"Best standardized validation L1: {best:.6g} at epoch {best_epoch}; saved {args.output}")
    evaluate_validation(args.output, args.dataset_h5)


if __name__ == "__main__":
    main()

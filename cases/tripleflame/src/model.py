import os
import torch
import torch.nn as nn
from src.thermodynamics import EnthalpyTemperature
from src.DNN import NN_MLP
from src.FNO import FNO
from src.Unet import Unet
from src.UFNO import UFNO
from src.PSTO import PSTO

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from utils.functions import BCT, inBCT, standardization, destandardization, normalization, denormalization
from utils.functions import finish_transport, normalize_species


class source_Y(nn.Module):
    def __init__(self, model_list, p=101325.0, mech_file="h2o2.yaml"):
        super().__init__()
        assert isinstance(model_list, list)
        self.model_list = nn.ModuleList(model_list)
        self.p = p
        self.Y_to_T = EnthalpyTemperature(mech_file)

    def load_stats(self, meanstd_stats):
        self.register_buffer('data_in_mean', meanstd_stats['data_in_mean'])
        self.register_buffer('data_in_std', meanstd_stats['data_in_std'])
        self.register_buffer('data_target_mean', meanstd_stats['data_target_mean'])
        self.register_buffer('data_target_std', meanstd_stats['data_target_std'])

    def forward(self, input):
        
        active = input[:, 0] > 800.0
        normalized = torch.cat((input[:, :1], normalize_species(input[:, 1:-2]),
                                input[:, -2:]), dim=1)
        if not active.any():
            return normalized
        points = input.permute(0, 2, 3, 1)
        prediction = self._predict(points[active].T.unsqueeze(0).unsqueeze(2))
        output = normalized.permute(0, 2, 3, 1).clone()
        output[active] = prediction[0, :, 0, :].T
        return output.permute(0, 3, 1, 2).contiguous()

    def _predict(self, input):
        T_in = input[:, 0:1]
        Y_in = input[:, 1:-2]
        U_in = input[:, -2:]
        Y_in_BCT = BCT(Y_in)

        it = torch.cat((T_in, torch.full_like(T_in, self.p), Y_in_BCT), dim=1)
        it = standardization(it, self.data_in_mean, self.data_in_std)
        B, _, H, W = it.shape
        out = []
        for i in range(len(self.model_list)):
            out.append(self.model_list[i](it.permute(0, 2, 3, 1).flatten(0, 2)))
        out = torch.cat(out, dim=1)
        out = out.unflatten(0, (B, H, W)).permute(0, 3, 1, 2)
        Y_res = destandardization(out, self.data_target_mean, self.data_target_std)
        Y_out_BCT = Y_res + Y_in_BCT[:, :-1]

        Y_out = inBCT(Y_out_BCT)
        Y_out = torch.clamp(Y_out, min=0.0)
        Y_sum = Y_out.sum(dim=1, keepdim=True).clamp(min=1e-8)
        Y_out = (1 - Y_in[:, -1:]) * Y_out / Y_sum
        Y_out = torch.cat((Y_out, Y_in[:, -1:]), dim=1)

        Y_out = normalize_species(Y_out)
        T_out = self.Y_to_T(T_in, Y_in, Y_out)

        return torch.cat((T_out, Y_out, U_in), dim=1)


class transport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def load_stats(self, maxmin_stats, clamp_stats=None):
        self.register_buffer('data_max', maxmin_stats['data_max'])
        self.register_buffer('data_min', maxmin_stats['data_min'])
        if hasattr(self.model, "set_stats"):
            self.model.set_stats(maxmin_stats['data_max'], maxmin_stats['data_min'])

    def forward(self, input):
        if getattr(self.model, 'physical_io', False):
            TYU_out = self.model(input)
        else:
            TYU_in_normalized = normalization(input, self.data_max, self.data_min)
            TYU_out_normalized = self.model(TYU_in_normalized)
            TYU_out = denormalization(TYU_out_normalized, self.data_max, self.data_min)

        if getattr(self.model, 'finalizes_transport_output', False):
            return TYU_out
        return finish_transport(TYU_out, input, self.data_min, self.data_max)


def build_dnn_source_model(root, n_species, device, pressure=101325.0,
                           mechanism='mech/h2o2.yaml', checkpoint=None):
    saved = torch.load(checkpoint or os.path.join(root, 'checkpoints/models/model_DNN/dnn_model_Y.pt'),
                       map_location=device, weights_only=False)
    models = [NN_MLP([2 + n_species] + [400] * 4 + [1])
              for _ in range(n_species - 1)]
    for i, model in enumerate(models):
        model.load_state_dict(saved[f'net{i}'])
    stats = {name: torch.as_tensor(saved[name], device=device, dtype=torch.float32).view(1, -1, 1, 1)
             for name in ('data_in_mean', 'data_in_std', 'data_target_mean', 'data_target_std')}
    reaction = source_Y(models, p=pressure, mech_file=mechanism)
    reaction.load_stats(stats)
    return reaction.requires_grad_(False).eval().to(device)


def infer_hidden_channels(state: dict, fallback: int) -> int:

    weight = state.get("lift.weight")
    if weight is None:
        return fallback
    return int(weight.shape[0])


def infer_latent_per_var(model_type: str, state: dict, fallback: int) -> int:

    if model_type == "PSTO":
        for key in ('transport_branch.lift.weight',):
            if key in state:
                return int(state[key].shape[0])
    return fallback


def build_backbone(
    model_type: str,
    in_ch: int,
    args,
    hidden_channels=None,
    latent_per_var=None,
    data_max=None,
    data_min=None,
    state=None,
) -> torch.nn.Module:
    hidden = args.hidden_channels if hidden_channels is None else hidden_channels
    latent = getattr(args, 'latent_per_var', 8) if latent_per_var is None else latent_per_var
    layers, modes = args.n_layers, args.modes
    if state is not None:
        latent = infer_latent_per_var(model_type, state, latent)
        if model_type != 'PSTO':
            prefix = 'core.encoders.' if model_type == 'Unet' else 'blocks.'
            indices = [int(key[len(prefix):].split('.')[0]) for key in state
                       if key.startswith(prefix) and key[len(prefix):].split('.')[0].isdigit()]
            if indices: layers = max(indices)+1
        spectral = next((value for key,value in state.items()
                         if key.endswith('weights1')), None)
        if spectral is not None: modes = int(spectral.shape[-1])
    if model_type == "PSTO":
        return PSTO(
            in_ch,
            latent_per_var=latent,
            fno_modes=modes,
            n_layers=layers,
            data_max=data_max,
            data_min=data_min,
            dt=args.dt,
            dx=args.dx,
            coefficient_backend=getattr(args, 'coefficient_backend', 'dnn'),
            coefficient_checkpoint=getattr(args, 'coefficient_checkpoint',
                                           'checkpoints/models/model_coeff/constitutive_DNN_BCT.pt'),
            mechanism=getattr(args, 'mechanism', os.path.join(root, 'mech/h2o2.yaml')),
            pressure=getattr(args, 'pressure', 101325.0),
        )
    if model_type == "FNO":
        return FNO(
            in_ch,
            hidden_channels=hidden,
            modes=modes,
            n_layers=layers,
            dt=args.dt,
        )
    if model_type == "Unet":
        return Unet(in_ch, hidden_channels=hidden, n_layers=layers, dt=args.dt)
    if model_type == 'UFNO':
        return UFNO(in_ch, hidden_channels=hidden, modes=modes, n_layers=layers, dt=args.dt)
    raise ValueError(model_type)


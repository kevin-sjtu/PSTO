import torch
from torch import nn
import torch.nn.functional as F

from FNO import SpectralConv2d, xavier_conv_
from train_constitutive import ConstitutiveTransport
from utils.functions import boundary_halo, diffusive_flux_divergence, finish_transport


class AdvectionDiffusionBlock(nn.Module):
    def __init__(self, channels, modes):
        super().__init__()
        self.advection_x = SpectralConv2d(channels, channels, modes, modes)
        self.advection_y = SpectralConv2d(channels, channels, modes, modes)
        self.diffusion = SpectralConv2d(channels, channels, modes, modes)
        self.adv_x_local = nn.Conv2d(channels, channels, 3, bias=False)
        self.adv_y_local = nn.Conv2d(channels, channels, 3, bias=False)
        self.dissipative_local = nn.Conv2d(channels, channels, 1, bias=False)
        with torch.no_grad():
            self.dissipative_local.weight.copy_(
                .1 * torch.eye(channels).reshape(channels, channels, 1, 1))
        self.update_logit = nn.Parameter(torch.tensor(-.8))
        for local in (self.adv_x_local, self.adv_y_local):
            xavier_conv_(local, gain=.2)

    def diffusion_weight(self, diffusion_scale, relaxation, bounds):
        diffusivity, density = bounds
        weight = self.dissipative_local.weight[:, :, 0, 0]
        norm = torch.linalg.matrix_norm(weight, ord=2)
        strength = (8 * diffusion_scale * diffusivity / density * relaxation
                    / (2 - relaxation)).clamp_min(torch.finfo(weight.dtype).tiny).sqrt()
        effective = weight / (norm * strength / .95).clamp_min(1)
        return (effective.T @ effective)[:, :, None, None]

    @staticmethod
    def padding_size(state):
        height, width = state.shape[-2:]
        return min(max(4, width // 16), max(1, height - 2), max(1, width - 2))

    def forward(self, state, advection, diffusivity, diffusion_scale, density, bounds):
        height, width = state.shape[-2:]
        padding = self.padding_size(state)
        halo = F.pad(state, (padding + 1,) * 4, mode='replicate')
        grad_x = .5 * (halo[..., 1:-1, 2:] - halo[..., 1:-1, :-2])
        grad_y = .5 * (halo[..., 2:, 1:-1] - halo[..., :-2, 1:-1])
        divergence = diffusive_flux_divergence(state, diffusivity, padding)
        rho = boundary_halo(density, padding)
        diffusion_increment = diffusion_scale * divergence / rho

        def response(spectral, local, field):
            fourier = spectral(field)[..., padding:padding + height, padding:padding + width]
            spatial = local(field)[..., padding - 1:padding - 1 + height,
                                   padding - 1:padding - 1 + width]
            return fourier + spatial

        gx = response(self.advection_x, self.adv_x_local, grad_x)
        gy = response(self.advection_y, self.adv_y_local, grad_y)
        lap = self.diffusion(diffusion_increment)[
            ..., padding:padding + height, padding:padding + width]
        advection_term = -advection[:, :1] * gx - advection[:, 1:2] * gy
        relaxation = .5 * torch.sigmoid(self.update_logit)
        local_diffusion = F.conv2d(
            diffusion_increment, self.diffusion_weight(diffusion_scale, relaxation, bounds))[
                ..., padding:padding + height, padding:padding + width]
        return advection_term + lap + local_diffusion

    def update(self, state, state0, advection, diffusivity, diffusion_scale, density, bounds):
        contribution = self(
            state, advection, diffusivity, diffusion_scale, density, bounds)
        relaxation = .5 * torch.sigmoid(self.update_logit)
        return state + relaxation * (state0 + contribution - state)


class SharedTransportBranch(nn.Module):
    def __init__(self, channels, modes):
        super().__init__()
        self.lift = nn.Conv2d(1, channels, 1, bias=False)
        self.block = AdvectionDiffusionBlock(channels, modes)
        self.project = nn.Conv2d(channels, 1, 1, bias=False)
        xavier_conv_(self.lift)
        xavier_conv_(self.project, gain=.01)

    def forward(self, state, state0, velocity, coefficient, dt, dx, density):
        batch, _, height, width = velocity.shape
        variables = state.shape[0] // batch
        uv = velocity[:, None].expand(-1, variables, -1, -1, -1)
        uv = uv.reshape(batch * variables, 2, height, width)
        bounds = (coefficient.amax(), density.amin())
        return self.block.update(state, state0, uv * (dt / dx), coefficient,
                                 dt / dx.square(), density, bounds)


class PSTO(ConstitutiveTransport):
    def __init__(self, num_phys_vars, latent_per_var=8, fno_modes=12,
                 n_layers=4, data_min=None, data_max=None, dt=1e-6,
                 dx=1e-4, lewis=None,
                 coefficient_checkpoint='checkpoints/models/model_coeff/constitutive_DNN_BCT.pt',
                 coefficient_backend='dnn', mechanism='mech/h2o2.yaml', pressure=101325.0):
        super().__init__()
        self.transport_branch = SharedTransportBranch(latent_per_var, fno_modes)
        self.physical_io = True
        self.finalizes_transport_output = True
        self.n_layers = int(n_layers)
        self._init_constitutive(
            num_phys_vars, data_min, data_max, dt, dx, lewis,
            coefficient_checkpoint, coefficient_backend, mechanism,
            pressure)

    def forward(self, x, coefficients=None):
        if x.ndim != 4 or x.shape[1] != self.N or min(x.shape[-2:]) < 3:
            raise ValueError(f'Expected NCHW with {self.N} channels and H,W >= 3, got {tuple(x.shape)}')
        physical, coeff = self.physical_coefficients(x, coefficients)
        rho, mu, kappa = coeff.split(1, dim=1)
        self.transport_coefficients = coeff
        dt, dx = self.time_step, self.grid_step
        batch, _, height, width = physical.shape
        velocity0 = physical[:, -2:]
        flat0 = physical.reshape(batch * self.N, 1, height, width)
        branch = self.transport_branch
        z0 = branch.lift(flat0)
        z = z0
        coefficient = torch.cat((kappa / self.lewis,
                                 mu.expand(-1, 2, -1, -1)), dim=1).reshape_as(flat0)
        channel_rho = rho[:, None].expand(-1, self.N, -1, -1, -1).reshape_as(flat0)
        for _ in range(self.n_layers):
            z = branch(z, z0, velocity0, coefficient, dt, dx, channel_rho)
        self.transport_velocity = velocity0
        output = (flat0 + branch.project(z - z0)).reshape_as(physical)
        return finish_transport(output, x, self.data_min, self.data_max)

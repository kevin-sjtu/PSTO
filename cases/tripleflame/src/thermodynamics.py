

import cantera as ct
import torch
from torch import nn

from utils.functions import STATE_SPECIES


class EnthalpyTemperature(nn.Module):


    def __init__(self, mechanism):
        super().__init__()
        gas = ct.Solution(mechanism)
        indices = [gas.species_index(name) for name in STATE_SPECIES]
        coefficients = []
        for index in indices:
            thermo = gas.species(index).thermo
            if not isinstance(thermo, ct.NasaPoly2):
                raise ValueError('GPU temperature recovery requires NASA7 species')
            coefficients.append(thermo.coeffs)
        nasa = torch.stack([torch.as_tensor(values, dtype=torch.float64)
                            for values in coefficients])
        self.register_buffer('midpoint', nasa[:, 0])
        self.register_buffer('high', nasa[:, 1:8])
        self.register_buffer('low', nasa[:, 8:15])
        self.register_buffer('specific_R', torch.tensor(
            ct.gas_constant / gas.molecular_weights[indices], dtype=torch.float64))

    def enthalpy_cp(self, temperature, mass_fractions):
        t = temperature.double().reshape(-1, 1)
        y = mass_fractions.double()
        y = y / y.sum(-1, keepdim=True)
        a = torch.where((t > self.midpoint).unsqueeze(-1), self.high, self.low)
        cp_R = a[..., 0] + t * (a[..., 1] + t * (
            a[..., 2] + t * (a[..., 3] + t * a[..., 4])))
        h_R = a[..., 5] + t * (a[..., 0] + t * (
            a[..., 1] / 2 + t * (a[..., 2] / 3 + t * (
                a[..., 3] / 4 + t * a[..., 4] / 5))))
        return ((y * h_R * self.specific_R).sum(-1),
                (y * cp_R * self.specific_R).sum(-1))

    def _recover(self, previous_t, previous_y, next_y):
        target, _ = self.enthalpy_cp(previous_t, previous_y)
        lower = (previous_t.double() * 0.25).clamp_min(1.0)
        upper = previous_t.double() * 2 + 1000
        for _ in range(12):
            low_h, _ = self.enthalpy_cp(lower, next_y)
            high_h, _ = self.enthalpy_cp(upper, next_y)
            below, above = low_h > target, high_h < target
            if not (below | above).any():
                break
            lower = torch.where(below, lower * 0.5, lower)
            upper = torch.where(above, upper * 1.5, upper)
        if (below | above).any():
            raise ValueError('GPU enthalpy temperature recovery could not bracket T')
        t = previous_t.double()
        tolerance = 1e-7 + target.abs() * 1e-11
        for _ in range(50):
            h, cp = self.enthalpy_cp(t, next_y)
            residual = h - target
            converged = residual.abs() <= tolerance
            if converged.all():
                return t
            if not torch.isfinite(h).all() or (cp <= 0).any():
                raise ValueError('GPU thermodynamics returned invalid h or cp')
            lower = torch.where((residual < 0) & ~converged, t, lower)
            upper = torch.where((residual > 0) & ~converged, t, upper)
            newton = t - residual / cp
            inside = (newton > lower) & (newton < upper)
            proposed = torch.where(inside, newton, (lower + upper) / 2)
            t = torch.where(converged, t, proposed)
        raise ValueError('GPU enthalpy temperature recovery did not converge')

    def forward(self, previous_t, previous_y, next_y):
        if self.midpoint.device != previous_t.device:
            raise ValueError('Thermodynamic buffers and state must share a device')
        t = previous_t.reshape(-1)
        y0 = previous_y.movedim(1, -1).reshape(-1, len(STATE_SPECIES))
        y1 = next_y.movedim(1, -1).reshape(-1, len(STATE_SPECIES))
        if not torch.isfinite(y0).all() or not torch.isfinite(y1).all():
            raise ValueError('Nonfinite composition in GPU temperature recovery')
        if (y0 < 0).any() or (y1 < 0).any() or (y0.sum(-1) <= 0).any() or (y1.sum(-1) <= 0).any():
            raise ValueError('Invalid composition in GPU temperature recovery')
        result = []
        for first in range(0, t.numel(), 16384):
            old_t = t[first:first + 16384]
            active = old_t > 310
            recovered = old_t.double().clone()
            if active.any():
                recovered[active] = self._recover(
                    old_t[active], y0[first:first + 16384][active],
                    y1[first:first + 16384][active])
            result.append(recovered)
        return torch.cat(result).reshape_as(previous_t).to(previous_t.dtype)


def ideal_gas_density(physical, molecular_weights, gas_constant, pressure):

    temperature = physical[:, :1].double()
    species = physical[:, 1:1 + molecular_weights.numel()].double()
    total = species.sum(dim=1, keepdim=True)
    molar = (species / molecular_weights.double().view(1, -1, 1, 1)).sum(1, keepdim=True)
    density = pressure.double() * total / (gas_constant.double() * temperature * molar)
    if ((temperature <= 0).any() or (total <= 0).any() or (molar <= 0).any()
            or not torch.isfinite(density).all() or (density <= 0).any()):
        raise ValueError('Ideal-gas EOS requires positive finite density and valid T/Y')
    return density


class IdealGasDensity(nn.Module):


    def __init__(self, mechanism='mech/h2o2.yaml', pressure=101325.0):
        super().__init__()
        gas = ct.Solution(mechanism)
        indices = [gas.species_index(name) for name in STATE_SPECIES]
        self.register_buffer('molecular_weights', torch.tensor(
            gas.molecular_weights[indices], dtype=torch.float64))
        self.register_buffer('gas_constant', torch.tensor(ct.gas_constant, dtype=torch.float64))
        self.register_buffer('pressure', torch.tensor(pressure, dtype=torch.float64))

    def forward(self, physical):
        return ideal_gas_density(physical, self.molecular_weights,
                                 self.gas_constant, self.pressure)

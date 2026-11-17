
import torch
from torch import nn

class RobertsonSource(nn.Module):

    def __init__(self, dt, rtol=1e-8, density=1.0, molecular_weight=25.9):
        super().__init__()
        self.dt, self.rtol = (dt, rtol)
        
        self.k1 = 4.0
        self.k2 = 3e7 * density / molecular_weight
        self.k3 = 1e4 * density / molecular_weight
        self.register_buffer('atol', torch.tensor([1e-13, 1e-17], dtype=torch.float64).reshape(2, 1, 1, 1))

    def rhs(self, y, total):
        a, b = y.unbind(0)
        c = total - a - b
        k2, k3 = self.k2, self.k3
        transfer = self.k1 * a - k3 * b * c
        return torch.stack((-transfer, transfer - k2 * b * b))

    def implicit(self, base, h, total, initial):
        y = initial.clone()
        k2, k3 = self.k2, self.k3
        for _ in range(16):
            a, b = y.unbind(0)
            c = total - a - b
            f = self.rhs(y, total)
            r = y - base - h * f
            j00 = 1 + h * (self.k1 + k3 * b)
            j01 = -h * k3 * (c - b)
            j10 = -h * (self.k1 + k3 * b)
            j11 = 1 + h * (k3 * (c - b) + 2 * k2 * b)
            determinant = j00 * j11 - j01 * j10
            da = (j11 * r[0] - j01 * r[1]) / determinant
            db = (-j10 * r[0] + j00 * r[1]) / determinant
            delta = torch.stack((da, db))
            y = y - delta
            if (delta.abs() / (self.atol + self.rtol * y.abs())).amax() < 0.01:
                break
        residual = y - base - h * self.rhs(y, total)
        if (residual.abs() / (self.atol + self.rtol * y.abs())).amax() > 0.1:
            raise ValueError("Robertson Newton iteration did not converge")
        if not torch.isfinite(y).all():
            raise ValueError('Robertson implicit source produced nonfinite states')
        return y

    def sdirk(self, y, h, total):
        gamma = 1 - 2 ** (-0.5)
        first = self.implicit(y, gamma * h, total, y)
        base = y + (1 - gamma) * h * self.rhs(first, total)
        return self.implicit(base, gamma * h, total, first)

    def integrate(self, old):
        if not torch.isfinite(old).all():
            raise ValueError('Robertson source received nonfinite input species; check upstream transport')
        old = old.double()
        total = old.sum(0)
        y = old[:2].clone()
        elapsed = 0.0
        h = self.dt
        for _ in range(4096):
            h = min(h, self.dt - elapsed)
            try:
                full = self.sdirk(y, h, total)
                half = self.sdirk(y, h * 0.5, total)
                fine = self.sdirk(half, h * 0.5, total)
            except ValueError:
                h *= 0.5
                if h < 1e-16:
                    raise ValueError('Robertson source step size underflow')
                continue
            error = ((fine - full).abs() / (3 * (self.atol + self.rtol * torch.maximum(fine.abs(), y.abs())))).amax().item()
            valid = (fine >= -1e-13).all() and (total - fine.sum(0) >= -1e-13).all()
            if error <= 1 and valid:
                y = fine
                elapsed += h
                if elapsed >= self.dt * (1 - 1e-12):
                    break
            h *= max(0.1, min(2.0, 0.8 * max(error, 1e-12) ** (-1 / 3))) if valid else 0.5
        else:
            raise ValueError('Robertson source adaptation did not converge')
        return torch.cat((y, (total - y.sum(0))[None]), 0)

    def forward(self, state):
        old = state[:, :3].movedim(1, 0)
        abc = self.integrate(old).clamp_min(0).to(state.dtype)
        species = state[:, :4].clone()
        species[:, :3] = abc.movedim(0, 1)
        result = torch.cat((species, state[:, 4:]), 1)
        
        return torch.cat((state[..., :1], result[..., 1:-1], state[..., -1:]), -1)


class AssimuloSource(nn.Module):


    def __init__(self, dt, density=1.0, molecular_weight=25.9):
        super().__init__()
        self.dt = dt
        self.k1, self.k2, self.k3 = 4.0, 3e7 * density / molecular_weight, 1e4 * density / molecular_weight
        from assimulo.problem import Explicit_Problem
        from assimulo.solvers import CVode
        self.problem_class, self.solver_class = Explicit_Problem, CVode

    def rhs(self, t, flat):
        import numpy as np
        a, b, c = flat.reshape(3, -1)
        transfer = self.k1 * a - self.k3 * b * c
        conversion = self.k2 * b * b
        return np.stack((-transfer, transfer - conversion, conversion)).ravel()

    def jacv(self, t, flat, fy, vector):
        import numpy as np
        a, b, c = flat.reshape(3, -1)
        va, vb, vc = vector.reshape(3, -1)
        transfer = self.k1 * va - self.k3 * (c * vb + b * vc)
        conversion = 2 * self.k2 * b * vb
        return np.stack((-transfer, transfer - conversion, conversion)).ravel()

    def forward(self, state):
        import numpy as np
        old = state[:, :3].movedim(1, 0).double().cpu().numpy()
        points = old.reshape(3, -1)
        problem = self.problem_class(self.rhs, points.ravel(), 0.0)
        problem.jacv = self.jacv
        solver = self.solver_class(problem)
        solver.discr, solver.iter = 'BDF', 'Newton'
        solver.linear_solver = 'SPGMR'
        solver.rtol = 1e-8
        solver.atol = np.repeat([1e-10, 1e-14, 1e-10], points.shape[1])
        solver.verbosity = 50
        _, trajectory = solver.simulate(self.dt, ncp=1)
        result = trajectory[-1].reshape(old.shape)
        if not np.isfinite(result).all() or result.min() < -1e-12:
            raise ValueError('Assimulo returned invalid mass fractions')
        if np.max(np.abs(result.sum(0) - old.sum(0))) > 1e-9:
            raise ValueError('Assimulo violated species conservation')
        species = state[:, :4].clone()
        species[:, :3] = torch.as_tensor(result, device=state.device, dtype=state.dtype).movedim(0, 1).clamp_min(0)
        result = torch.cat((species, state[:, 4:]), 1)
        return torch.cat((state[..., :1], result[..., 1:-1], state[..., -1:]), -1)

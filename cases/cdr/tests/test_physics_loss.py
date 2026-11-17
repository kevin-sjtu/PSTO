"""Check the transport residual against analytic periodic Fourier modes."""
from pathlib import Path
import math
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from train import PhysicsLoss


class PhysicsLossTests(unittest.TestCase):
    def setUp(self):
        self.dx, self.dt = 0.125, 0.25
        self.theta_x, self.theta_y = 2 * torch.pi * 2 / 16, 2 * torch.pi * 3 / 20
        i, j = torch.meshgrid(torch.arange(16, dtype=torch.float64),
                              torch.arange(20, dtype=torch.float64), indexing='ij')
        self.phase_x, self.phase_y = self.theta_x * i, self.theta_y * j
        self.lower = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        self.upper = torch.tensor([2., 3., 4.], dtype=torch.float64).view(1, 3, 1, 1)

    def objective(self, diffusivity=0.):
        config = dict(channels=['u', 'v', 'phi'], dt=self.dt, dx=self.dx,
                      nu=diffusivity, alpha=diffusivity)
        return PhysicsLoss(config, self.lower, self.upper).double()

    def test_centered_advection_uses_input_velocity_for_all_channels(self):
        sx, cy = self.phase_x.sin(), self.phase_y.cos()
        state = torch.stack((1.25 + .2 * sx, -.75 + .1 * cy,
                             sx + .6 * cy)).unsqueeze(0)
        prediction = state.clone()
        prediction[:, 0] -= 4.
        prediction[:, 1] += 3.
        prediction.requires_grad_(True)

        # A centered derivative multiplies a Fourier mode by sin(theta)/dx.
        gx = math.sin(self.theta_x) / self.dx * self.phase_x.cos()
        gy = -math.sin(self.theta_y) / self.dx * self.phase_y.sin()
        u, v = state[:, 0], state[:, 1]
        residual = torch.stack((-4. / self.dt + u * .2 * gx,
                                3. / self.dt + v * .1 * gy,
                                u * gx + v * .6 * gy), dim=1)
        expected = (residual / self.upper).square().mean()
        actual = self.objective()(state, prediction)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad.abs().sum((0, 2, 3)) > 0).all())

    def test_fourth_order_periodic_diffusion_is_preserved(self):
        state = torch.zeros(1, 3, 16, 20, dtype=torch.float64)
        state[:, 2] = self.phase_x.sin() + .6 * self.phase_y.cos()
        diffusivity = .125

        def eigenvalue(theta):
            sine_squared = torch.sin(torch.tensor(theta / 2, dtype=torch.float64)) ** 2
            return -4 * sine_squared * (1 + sine_squared / 3) / self.dx**2

        laplacian = (eigenvalue(self.theta_x) * self.phase_x.sin()
                     + .6 * eigenvalue(self.theta_y) * self.phase_y.cos())
        expected = (diffusivity * laplacian / 4).square().mean() / 3
        actual = self.objective(diffusivity)(state, state)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()

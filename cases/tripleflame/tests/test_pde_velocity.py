"""Check the frozen input velocity in scalar and continuity residuals."""
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from train import trainer
from utils.functions import normalization


class ConstantDensity(nn.Module):
    def forward(self, state):
        return torch.ones_like(state[:, :1])


class PDEVelocityTests(unittest.TestCase):
    def setUp(self):
        self.dx = 0.5
        self.yy, self.xx = torch.meshgrid(
            torch.arange(7, dtype=torch.float64)*self.dx,
            torch.arange(9, dtype=torch.float64)*self.dx, indexing='ij')
        self.state = torch.zeros(1, 12, 7, 9, dtype=torch.float64)
        self.state[:, 0] = 1000
        self.state[:, 1:10] = 1/9
        self.coefficients = torch.ones(1, 3, 7, 9, dtype=torch.float64)

    def objective(self, state, physical_io):
        # Exercise the shared loss without training or loading any checkpoints.
        loss = trainer.__new__(trainer)
        nn.Module.__init__(loss)
        loss.model = nn.Identity()
        if physical_io:
            # PSTO caches the input velocity; FNO/U-Net/U-FNO use the fallback.
            loss.model.transport_velocity = state[:, -2:]
            loss.model.transport_coefficients = self.coefficients
        loss.physical_io = physical_io
        loss.physics_weight = 1.0
        loss.dt, loss.dx = 0.25, self.dx
        loss.data_min = torch.tensor([900]+[0]*9+[-100]*2, dtype=torch.float64).view(12, 1, 1)
        loss.data_max = torch.tensor([1100]+[1]*9+[100]*2, dtype=torch.float64).view(12, 1, 1)
        loss.lewis = torch.ones(1, 9, 1, 1, dtype=torch.float64)
        loss.baseline_constitutive = None
        loss.continuity_density = ConstantDensity()
        return loss

    def evaluate(self, state, prediction, physical_io):
        loss = self.objective(state, physical_io)
        if not physical_io:
            state = normalization(state, loss.data_max, loss.data_min)
            prediction = normalization(prediction, loss.data_max, loss.data_min)
        prediction = prediction.detach().requires_grad_(True)
        return loss.pde_loss(state, prediction, self.coefficients), prediction

    def test_scalar_advection_uses_input_velocity(self):
        state = self.state.clone()
        state[:, 0] += 4*self.xx+6*self.yy
        state[:, 1] += 0.01*self.xx+0.02*self.yy
        state[:, -2], state[:, -1] = 2, 1
        prediction = state.clone()
        prediction[:, -2], prediction[:, -1] = 7, -4
        # Linear scalars: zero time derivative and interior diffusion. Constant
        # density/velocity: zero continuity residual. Average over 10 scalars + rho.
        expected = ((14/200)**2 + 0.04**2)/11
        for physical_io in (True, False):
            with self.subTest(physical_io=physical_io):
                actual, pred = self.evaluate(state, prediction, physical_io)
                self.assertAlmostEqual(actual.item(), expected, places=12)
                actual.backward()
                self.assertTrue(torch.isfinite(pred.grad).all())
                self.assertGreater(pred.grad[:, :10].abs().max().item(), 0)
                torch.testing.assert_close(pred.grad[:, -2:], torch.zeros_like(pred.grad[:, -2:]),
                                           rtol=0, atol=0)

    def test_continuity_flux_uses_input_velocity(self):
        state = self.state.clone()
        state[:, -2], state[:, -1] = 2+3*self.xx, -1+4*self.yy
        prediction = state.clone()
        prediction[:, -2], prediction[:, -1] = -5+11*self.xx, 8-13*self.yy
        # Constant scalars and density isolate div(u^n) = 3+4 = 7;
        # using the predicted velocity would instead give 11-13 = -2.
        for physical_io in (True, False):
            with self.subTest(physical_io=physical_io):
                actual, _ = self.evaluate(state, prediction, physical_io)
                self.assertAlmostEqual(actual.item(), 49/11, places=12)


if __name__ == '__main__':
    unittest.main()

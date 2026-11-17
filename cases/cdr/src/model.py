
import torch
from torch import nn

class CDRSource(nn.Module):

    def __init__(self, dt, beta=1.0, rate=1.0, threshold=0.5):
        super().__init__()
        self.dt, self.beta, self.rate, self.threshold = (dt, beta, rate, threshold)

    def rhs(self, state):
        u, v, phi = state.split(1, 1)
        s = u * u + v * v
        return torch.cat(((1 - s) * u + self.beta * s * v, (1 - s) * v - self.beta * s * u, self.rate * phi * (1 - phi) * (phi - self.threshold)), 1)

    def forward(self, state):
        k1 = self.rhs(state)
        k2 = self.rhs(state + self.dt * k1)
        return state + 0.5 * self.dt * (k1 + k2)

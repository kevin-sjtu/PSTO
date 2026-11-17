import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_complex_parameter(shape):
    real = torch.empty(*shape)
    imag = torch.empty(*shape)
    nn.init.xavier_normal_(real)
    nn.init.xavier_normal_(imag)
    return torch.complex(real, imag) / (2.0**0.5)


def xavier_conv_(layer, gain=1.0):
    nn.init.xavier_normal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        shape = (in_channels, out_channels, modes1, modes2)
        self.weights1 = nn.Parameter(xavier_complex_parameter(shape))
        self.weights2 = nn.Parameter(xavier_complex_parameter(shape))

    @staticmethod
    def compl_mul2d(x_ft, weights):
        return torch.einsum("bihw,iohw->bohw", x_ft, weights)

    def forward(self, x):
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x.float())
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
        )
        mx = min(self.modes1, height // 2)
        my = min(self.modes2, width // 2 + 1)
        out_ft[:, :, :mx, :my] = self.compl_mul2d(
            x_ft[:, :, :mx, :my], self.weights1[:, :, :mx, :my]
        )
        out_ft[:, :, -mx:, :my] = self.compl_mul2d(
            x_ft[:, :, -mx:, :my], self.weights2[:, :, :mx, :my]
        )
        return torch.fft.irfft2(out_ft, s=(height, width)).to(x.dtype)


class FNOBlock(nn.Module):
    def __init__(self, channels, modes):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes, modes)
        self.local = nn.Conv2d(channels, channels, 1)
        xavier_conv_(self.local)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.local(x))


class FNO(nn.Module):
    def __init__(self, num_phys_vars, hidden_channels=54, modes=16, n_layers=4, dt=1.0e-6):
        super().__init__()
        self.num_phys_vars = num_phys_vars
        self.register_buffer("time_step", torch.tensor(float(dt)))
        self.lift = nn.Conv2d(num_phys_vars, hidden_channels, 1)
        self.blocks = nn.ModuleList([FNOBlock(hidden_channels, modes) for _ in range(n_layers)])
        self.projection = nn.Conv2d(hidden_channels, num_phys_vars, 1)
        xavier_conv_(self.lift)
        xavier_conv_(self.projection)

    def forward(self, x):

        z = self.lift(x)
        for block in self.blocks:
            z = block(z)
        prediction = x + self.time_step.to(dtype=x.dtype) * self.projection(z)
        return prediction.clamp(-1.0, 1.0)

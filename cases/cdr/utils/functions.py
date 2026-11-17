import torch.nn.functional as F
import torch


def boundary_halo(state, padding=1):
    return F.pad(state, (padding,) * 4, mode='circular')


def diffusive_flux_divergence(state, coefficient):
    gx = torch.roll(state, -1, -1) - state
    gy = torch.roll(state, -1, -2) - state
    ax = .5 * (torch.roll(coefficient, -1, -1) + coefficient)
    ay = .5 * (torch.roll(coefficient, -1, -2) + coefficient)
    fx, fy = ax * gx, ay * gy
    return fx - torch.roll(fx, 1, -1) + fy - torch.roll(fy, 1, -2)


def finish_transport(output, data_min, data_max):
    return output.clamp(min=data_min, max=data_max)

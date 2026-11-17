import torch
import torch.nn.functional as F


def boundary_halo(state, padding=1):
    return F.pad(state, (padding,) * 4, mode='replicate')


def diffusive_flux_divergence(state, coefficient, padding=0):
    halo = boundary_halo(state, padding + 1)
    a = boundary_halo(coefficient, padding + 1)
    flux_x = .5 * (a[..., 1:] + a[..., :-1]) * (halo[..., 1:] - halo[..., :-1])
    flux_y = .5 * (a[..., 1:, :] + a[..., :-1, :]) * (halo[..., 1:, :] - halo[..., :-1, :])
    return (flux_x[..., 1:-1, 1:] - flux_x[..., 1:-1, :-1]
            + flux_y[..., 1:, 1:-1] - flux_y[..., :-1, 1:-1])


def enforce_boundaries(output, inlet):
    rows = torch.cat((output[..., 1:2, :], output[..., 1:-1, :],
                      output[..., -2:-1, :]), dim=-2)
    return torch.cat((inlet[..., :1], rows[..., 1:-1], inlet[..., -1:]), dim=-1)


def normalize_species(species):
    positive = species.clamp_min(0)
    total = positive.sum(1, keepdim=True)
    fallback = torch.cat((torch.zeros_like(positive[:, :-1]),
                          torch.ones_like(positive[:, -1:])), dim=1)
    return torch.where(total > 0, positive / total.clamp_min(1e-12), fallback)


def finish_transport(output, inlet, data_min, data_max):
    output = enforce_boundaries(output.clamp(min=data_min, max=data_max), inlet)
    return torch.cat((normalize_species(output[:, :4]), output[:, 4:]), dim=1)


def denormalization(data, data_max, data_min):
    return (data + 1) * (data_max - data_min) / 2 + data_min

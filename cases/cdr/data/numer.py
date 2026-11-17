import os
import argparse
from pathlib import Path

import h5py
import scipy.io as sio
import torch
import numpy as np
import torch.nn.functional as F


dx_2d_op  = [[[[    0,   0,  1/12,   0,    0],
               [    0,   0, -8/12,   0,    0],
               [    0,   0,     0,   0,    0],
               [    0,   0,  8/12,   0,    0],
               [    0,   0, -1/12,   0,    0]]]]

dy_2d_op  = [[[[    0,     0,    0,     0,      0],
               [    0,     0,    0,     0,      0],
               [ 1/12, -8/12,    0,  8/12,  -1/12],
               [    0,     0,    0,     0,      0],
               [    0,     0,    0,     0,      0]]]]

lap_2d_op = [[[[    0,   0, -1/12,   0,     0],
               [    0,   0,   4/3,   0,     0],
               [-1/12, 4/3,   - 5, 4/3, -1/12],
               [    0,   0,   4/3,   0,     0],
               [    0,   0, -1/12,   0,     0]]]]


laplacian = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=0, bias=False)
laplacian.weight.data = torch.tensor(lap_2d_op).float()
laplacian.weight.requires_grad = False

nabla_x = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=0, bias=False)
nabla_x.weight.data = torch.tensor(dx_2d_op).float()
nabla_x.weight.requires_grad = False

nabla_y = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=0, bias=False)
nabla_y.weight.data = torch.tensor(dy_2d_op).float()
nabla_y.weight.requires_grad = False

DX = 1.0 / 100.0


def delta(u):

    u = torch.cat((u[..., :, -2:], u, u[..., :, 0:2]), dim=3)
    u = torch.cat((u[..., -2:, :], u, u[..., 0:2, :]), dim=2)

    return laplacian(u) / DX ** 2


def nabla_u(u, v):
    
    u_px = torch.cat((u[..., -2:, :], u, u[..., :2, :]), dim=2)
    du_dx_back = (3*u - 4*u_px[..., 1:-3, :] + u_px[..., :-4, :]) / (2*DX)
    du_dx_fwd  = (-3*u + 4*u_px[..., 3:-1, :] - u_px[..., 4:, :]) / (2*DX)

    
    u_py = torch.cat((u[..., :, -2:], u, u[..., :, :2]), dim=3)
    du_dy_back = (3*u - 4*u_py[..., :, 1:-3] + u_py[..., :, :-4]) / (2*DX)
    du_dy_fwd  = (-3*u + 4*u_py[..., :, 3:-1] - u_py[..., :, 4:]) / (2*DX)

    u_pos = torch.clamp(u, min=0)
    u_neg = torch.clamp(u, max=0)
    v_pos = torch.clamp(v, min=0)
    v_neg = torch.clamp(v, max=0)

    return u_pos * du_dx_back + u_neg * du_dx_fwd + v_pos * du_dy_back + v_neg * du_dy_fwd

def nabla_v(u, v):
    
    v_px = torch.cat((v[..., -2:, :], v, v[..., :2, :]), dim=2)
    dv_dx_back = (3*v - 4*v_px[..., 1:-3, :] + v_px[..., :-4, :]) / (2*DX)
    dv_dx_fwd  = (-3*v + 4*v_px[..., 3:-1, :] - v_px[..., 4:, :]) / (2*DX)

    
    v_py = torch.cat((v[..., :, -2:], v, v[..., :, :2]), dim=3)
    dv_dy_back = (3*v - 4*v_py[..., :, 1:-3] + v_py[..., :, :-4]) / (2*DX)
    dv_dy_fwd  = (-3*v + 4*v_py[..., :, 3:-1] - v_py[..., :, 4:]) / (2*DX)

    u_pos = torch.clamp(u, min=0)
    u_neg = torch.clamp(u, max=0)
    v_pos = torch.clamp(v, min=0)
    v_neg = torch.clamp(v, max=0)

    return u_pos * dv_dx_back + u_neg * dv_dx_fwd + v_pos * dv_dy_back + v_neg * dv_dy_fwd


def u_t(U, V):
    u_t = 1/200 * delta(U) - nabla_u(U, V)
    return u_t

def v_t(U, V):
    v_t = 1/200 * delta(V) - nabla_v(U, V)
    return v_t


def scalar_advection(phi, u, v, dx=DX):


    phi_px = torch.cat((phi[..., -2:, :], phi, phi[..., :2, :]), dim=2)
    dphi_dx_back = (3 * phi - 4 * phi_px[..., 1:-3, :] + phi_px[..., :-4, :]) / (2 * dx)
    dphi_dx_fwd = (-3 * phi + 4 * phi_px[..., 3:-1, :] - phi_px[..., 4:, :]) / (2 * dx)

    phi_py = torch.cat((phi[..., :, -2:], phi, phi[..., :, :2]), dim=3)
    dphi_dy_back = (3 * phi - 4 * phi_py[..., :, 1:-3] + phi_py[..., :, :-4]) / (2 * dx)
    dphi_dy_fwd = (-3 * phi + 4 * phi_py[..., :, 3:-1] - phi_py[..., :, 4:]) / (2 * dx)

    return (
        torch.clamp(u, min=0) * dphi_dx_back
        + torch.clamp(u, max=0) * dphi_dx_fwd
        + torch.clamp(v, min=0) * dphi_dy_back
        + torch.clamp(v, max=0) * dphi_dy_fwd
    )


def scalar_laplacian(phi, dx=DX):

    d2x = (
        -torch.roll(phi, shifts=2, dims=2)
        + 16 * torch.roll(phi, shifts=1, dims=2)
        - 30 * phi
        + 16 * torch.roll(phi, shifts=-1, dims=2)
        - torch.roll(phi, shifts=-2, dims=2)
    ) / (12 * dx**2)
    d2y = (
        -torch.roll(phi, shifts=2, dims=3)
        + 16 * torch.roll(phi, shifts=1, dims=3)
        - 30 * phi
        + 16 * torch.roll(phi, shifts=-1, dims=3)
        - torch.roll(phi, shifts=-2, dims=3)
    ) / (12 * dx**2)
    return d2x + d2y


def scalar_transport_rhs(phi, u, v, diffusivity=1.0 / 200.0, dx=DX):

    return diffusivity * scalar_laplacian(phi, dx=dx) - scalar_advection(
        phi, u, v, dx=dx
    )


def scalar_reaction_rhs(phi, reaction_rate=1.0, reaction_threshold=0.5):

    return reaction_rate * phi * (1.0 - phi) * (phi - reaction_threshold)


def transport_rhs(state, viscosity=1.0 / 200.0, scalar_diffusivity=1.0 / 200.0, dx=DX):

    u, v, phi = state[:, 0:1], state[:, 1:2], state[:, 2:3]
    rhs_u = viscosity * scalar_laplacian(u, dx) - scalar_advection(u, u, v, dx)
    rhs_v = viscosity * scalar_laplacian(v, dx) - scalar_advection(v, u, v, dx)
    rhs_phi = scalar_transport_rhs(phi, u, v, scalar_diffusivity, dx)
    return torch.cat((rhs_u, rhs_v, rhs_phi), dim=1)


def reaction_rhs(
    state,
    velocity_reaction_beta=1.0,
    scalar_reaction_rate=1.0,
    scalar_reaction_threshold=0.5,
):

    u, v, phi = state[:, 0:1], state[:, 1:2], state[:, 2:3]
    speed_squared = u.square() + v.square()
    growth = 1.0 - speed_squared
    rotation = velocity_reaction_beta * speed_squared
    rhs_u = growth * u + rotation * v
    rhs_v = -rotation * u + growth * v
    rhs_phi = scalar_reaction_rhs(
        phi, scalar_reaction_rate, scalar_reaction_threshold
    )
    return torch.cat((rhs_u, rhs_v, rhs_phi), dim=1)


def cdr_rhs(
    state,
    viscosity=1.0 / 200.0,
    scalar_diffusivity=1.0 / 200.0,
    velocity_reaction_beta=1.0,
    scalar_reaction_rate=1.0,
    scalar_reaction_threshold=0.5,
    dx=DX,
):

    return transport_rhs(state, viscosity, scalar_diffusivity, dx) + reaction_rhs(
        state,
        velocity_reaction_beta,
        scalar_reaction_rate,
        scalar_reaction_threshold,
    )


def initial_scalar(ny, nx, dtype=torch.float64):

    y = torch.arange(ny, dtype=dtype) / float(ny)
    x = torch.arange(nx, dtype=dtype) / float(nx)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phi0 = 0.5 + 0.25 * torch.sin(2.0 * np.pi * xx) * torch.cos(2.0 * np.pi * yy)
    return phi0.unsqueeze(0).unsqueeze(0)


def advance_scalar_transport_rk2(phi, uv_now, uv_next, dt, diffusivity, dx):

    u_now, v_now = uv_now[:, 0:1], uv_now[:, 1:2]
    u_next, v_next = uv_next[:, 0:1], uv_next[:, 1:2]
    k1 = scalar_transport_rhs(phi, u_now, v_now, diffusivity, dx)
    phi_predictor = phi + dt * k1
    k2 = scalar_transport_rhs(phi_predictor, u_next, v_next, diffusivity, dx)
    return phi + 0.5 * dt * (k1 + k2)


def advance_transport_rk2(state, dt, viscosity, scalar_diffusivity, dx):

    k1 = transport_rhs(state, viscosity, scalar_diffusivity, dx)
    predictor = state + dt * k1
    k2 = transport_rhs(predictor, viscosity, scalar_diffusivity, dx)
    return state + 0.5 * dt * (k1 + k2)


def advance_reaction_rk2(
    state,
    dt,
    velocity_reaction_beta,
    scalar_reaction_rate,
    scalar_reaction_threshold,
):

    k1 = reaction_rhs(
        state,
        velocity_reaction_beta,
        scalar_reaction_rate,
        scalar_reaction_threshold,
    )
    predictor = state + dt * k1
    k2 = reaction_rhs(
        predictor,
        velocity_reaction_beta,
        scalar_reaction_rate,
        scalar_reaction_threshold,
    )
    return state + 0.5 * dt * (k1 + k2)


def advance_cdr_rk2(
    state,
    dt,
    viscosity,
    scalar_diffusivity,
    velocity_reaction_beta,
    scalar_reaction_rate,
    scalar_reaction_threshold,
    dx,
):

    k1 = cdr_rhs(
        state,
        viscosity,
        scalar_diffusivity,
        velocity_reaction_beta,
        scalar_reaction_rate,
        scalar_reaction_threshold,
        dx,
    )
    predictor = state + dt * k1
    k2 = cdr_rhs(
        predictor,
        viscosity,
        scalar_diffusivity,
        velocity_reaction_beta,
        scalar_reaction_rate,
        scalar_reaction_threshold,
        dx,
    )
    return state + 0.5 * dt * (k1 + k2)


def generate_cd_and_cdr_h5(
    transport_input_mat,
    cd_output_h5,
    cdr_output_h5,
    cdr_steps=1000,
    dt=2.5e-4,
    viscosity=1.0 / 200.0,
    scalar_diffusivity=1.0 / 200.0,
    velocity_reaction_beta=1.0,
    scalar_reaction_rate=1.0,
    scalar_reaction_threshold=0.5,
    compression="lzf",
    overwrite=False,
):


    transport_input_mat = Path(transport_input_mat).expanduser().resolve()
    cd_output_h5 = Path(cd_output_h5).expanduser().resolve()
    cdr_output_h5 = Path(cdr_output_h5).expanduser().resolve()
    if not transport_input_mat.is_file():
        raise FileNotFoundError(
            f"Reaction-free velocity trajectory not found: {transport_input_mat}"
        )
    if cdr_steps < 1:
        raise ValueError("cdr_steps must be positive")
    for output_path in (cd_output_h5, cdr_output_h5):
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )

    transport_mat = sio.loadmat(transport_input_mat)
    if "uv" not in transport_mat:
        raise KeyError(f"{transport_input_mat} does not contain an 'uv' array")
    uv = np.asarray(transport_mat["uv"])
    if uv.ndim != 4 or uv.shape[1] != 2:
        raise ValueError(f"Expected uv with shape [time, 2, ny, nx], got {uv.shape}")
    if not np.isfinite(uv).all():
        raise ValueError("The supplied velocity trajectory contains non-finite values")

    n_time, _, ny, nx = uv.shape
    n_cdr_time = cdr_steps + 1
    dx = 1.0 / float(nx)
    torch_dtype = torch.float64 if uv.dtype == np.float64 else torch.float32
    phi_transport = initial_scalar(ny, nx, dtype=torch_dtype)

    cd_output_h5.parent.mkdir(parents=True, exist_ok=True)
    cdr_output_h5.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with h5py.File(cd_output_h5, mode) as cd_h5:
        cd_data = cd_h5.create_dataset(
            "data",
            shape=(n_time, 3, ny, nx),
            dtype=uv.dtype,
            chunks=(1, 3, ny, nx),
            compression=compression,
        )
        transport_initial_state = torch.cat(
            (torch.from_numpy(uv[0:1]), phi_transport), dim=1
        )
        cd_data[0] = transport_initial_state[0].numpy()

        
        for step in range(n_time - 1):
            uv_now = torch.from_numpy(uv[step : step + 1])
            uv_next = torch.from_numpy(uv[step + 1 : step + 2])
            phi_transport = advance_scalar_transport_rk2(
                phi_transport,
                uv_now,
                uv_next,
                dt,
                scalar_diffusivity,
                dx,
            )
            if not torch.isfinite(phi_transport).all():
                raise FloatingPointError(
                    f"Non-finite reaction-free scalar state at step {step + 1}"
                )
            cd_data[step + 1, :2] = uv[step + 1]
            cd_data[step + 1, 2] = phi_transport[0, 0].numpy()

        cd_h5.attrs["channels"] = np.asarray(
            ["u", "v", "phi"], dtype=h5py.string_dtype()
        )
        cd_h5.attrs["equation"] = "convection-diffusion (source-free)"
        cd_h5.attrs["source_transport_uv_file"] = str(transport_input_mat)
        cd_h5.attrs["dt"] = dt
        cd_h5.attrs["dx"] = dx
        cd_h5.attrs["viscosity"] = viscosity
        cd_h5.attrs["scalar_diffusivity"] = scalar_diffusivity
        cd_h5.attrs["learned_operator"] = "source-free transport T(dt)"
        cd_h5.attrs["scalar_equation"] = (
            "phi_t + u*phi_x + v*phi_y = D_phi*Laplacian(phi)"
        )
        cd_h5.attrs["scalar_initial_condition"] = (
            "0.5 + 0.25*sin(2*pi*x)*cos(2*pi*y)"
        )

    with h5py.File(cdr_output_h5, mode) as cdr_h5:
        cdr_data = cdr_h5.create_dataset(
            "data",
            shape=(n_cdr_time, 3, ny, nx),
            dtype=uv.dtype,
            chunks=(1, 3, ny, nx),
            compression=compression,
        )
        
        
        
        cdr_state = transport_initial_state.clone()
        cdr_data[0] = cdr_state[0].numpy()
        for step in range(n_cdr_time - 1):
            cdr_state = advance_cdr_rk2(
                cdr_state,
                dt,
                viscosity,
                scalar_diffusivity,
                velocity_reaction_beta,
                scalar_reaction_rate,
                scalar_reaction_threshold,
                dx,
            )
            if not torch.isfinite(cdr_state).all():
                raise FloatingPointError(
                    f"Non-finite CDR state at step {step + 1}"
                )
            cdr_data[step + 1] = cdr_state[0].numpy()
            if (step + 1) % 200 == 0 or step + 1 == n_cdr_time - 1:
                print(
                    f"CDR step {step + 1:4d}/{n_cdr_time - 1}: "
                    f"phi in [{cdr_state[:, 2:3].min().item():.6f}, "
                    f"{cdr_state[:, 2:3].max().item():.6f}]"
                )

        cdr_h5.attrs["channels"] = np.asarray(
            ["u", "v", "phi"], dtype=h5py.string_dtype()
        )
        cdr_h5.attrs["equation"] = "convection-diffusion-reaction"
        cdr_h5.attrs["initial_state_source"] = "CD.h5/data[0]"
        cdr_h5.attrs["dt"] = dt
        cdr_h5.attrs["dx"] = dx
        cdr_h5.attrs["viscosity"] = viscosity
        cdr_h5.attrs["scalar_diffusivity"] = scalar_diffusivity
        cdr_h5.attrs["velocity_reaction_beta"] = velocity_reaction_beta
        cdr_h5.attrs["scalar_reaction_rate"] = scalar_reaction_rate
        cdr_h5.attrs["scalar_reaction_threshold"] = scalar_reaction_threshold
        cdr_h5.attrs["reference_integrator"] = "RK2 applied to the complete CDR equation"
        cdr_h5.attrs["surrogate_operator_splitting"] = (
            "Strang: R(dt/2) o T_hat(dt) o R(dt/2)"
        )
        cdr_h5.attrs["velocity_source"] = (
            "lambda=(1-u^2-v^2), omega=beta*(u^2+v^2), "
            "R_u=lambda*u+omega*v, R_v=-omega*u+lambda*v"
        )
        cdr_h5.attrs["scalar_equation"] = (
            "phi_t + u*phi_x + v*phi_y = D_phi*Laplacian(phi) "
            "+ k*phi*(1-phi)*(phi-gamma)"
        )
        cdr_h5.attrs["scalar_initial_condition"] = (
            "0.5 + 0.25*sin(2*pi*x)*cos(2*pi*y)"
        )

    print(
        f"Saved {cd_output_h5}: data={(n_time, 3, ny, nx)}; "
        f"saved {cdr_output_h5}: data={(n_cdr_time, 3, ny, nx)}; "
        f"dtype={uv.dtype}"
    )
    return cd_output_h5, cdr_output_h5


def Runge_kutta(u, v, dt, u_t, v_t):
    
    
    
    
    k1u = dt * u_t(u, v)
    k1v = dt * v_t(u, v)
    u_mid = u + k1u
    v_mid = v + k1v
    k2u = dt * u_t(u_mid, v_mid)
    k2v = dt * v_t(u_mid, v_mid)
    return u + (k1u + k2u) / 2, v + (k1v + k2v) / 2

def get_concentra(U_0, V_0, steps, dt=0.00025):
    
    U = torch.from_numpy(U_0).float()
    V = torch.from_numpy(V_0).float()

    U_list, V_list = [U], [V]

    for _ in range(steps):
        U_next, V_next = Runge_kutta(U, V, dt, u_t, v_t)
        U = U_next
        V = V_next
        U_list.append(U)
        V_list.append(V)

    return U_list, V_list


def max_abs_field_error(a: torch.Tensor, b: torch.Tensor) -> float:

    return torch.max(torch.abs(a - b)).item()


def stack_uv_nchw(U: torch.Tensor, V: torch.Tensor) -> np.ndarray:

    out = np.concatenate(
        (U.detach().cpu().numpy(), V.detach().cpu().numpy()), axis=1
    )
    return out.astype(np.float32, copy=False)


def set_dx_from_n(n_grid: int):

    global DX
    DX = 1.0 / float(n_grid)


def resize_nchw_np(arr: np.ndarray, out_hw):

    ten = torch.from_numpy(arr).float()
    ten_r = F.interpolate(ten, size=out_hw, mode="bilinear", align_corners=False)
    return ten_r.detach().cpu().numpy()


def load_uv_data(mat_path):
    uv = sio.loadmat(mat_path)["uv"]
    print("UV.shape:", uv.shape)
    return uv


def load_initial_uv(uv_data, ic_index, out_n=None):
    uv = uv_data
    u0 = uv[ic_index : ic_index + 1, 0:1, :, :]
    v0 = uv[ic_index : ic_index + 1, 1:2, :, :]
    if out_n is not None and u0.shape[-1] != out_n:
        u0 = resize_nchw_np(u0, (out_n, out_n))
        v0 = resize_nchw_np(v0, (out_n, out_n))
    return u0, v0


def get_last_uv(u0, v0, steps, dt):
    u_list, v_list = get_concentra(u0, v0, steps, dt=dt)
    return u_list[-1], v_list[-1]


def load_cached_stack(cache_path, expected_shape):
    if not os.path.isfile(cache_path):
        return None
    uv_stack = np.load(cache_path)
    if uv_stack.shape == expected_shape:
        print("Using saved", cache_path, uv_stack.shape)
        return uv_stack
    print(
        f"Shape mismatch in {cache_path} ({uv_stack.shape}), recompute. "
        f"Expected {expected_shape}."
    )
    return None


def save_uv_stack(cache_path, uv_stack, label_text):
    np.save(cache_path, uv_stack)
    print(f"Saved {cache_path} {uv_stack.shape} - {label_text}")


def loglog_ols_line(dt, err):

    dt = np.asarray(dt, dtype=np.float64).ravel()
    err = np.maximum(np.asarray(err, dtype=np.float64).ravel(), 1e-30)
    a = np.polyfit(np.log10(dt), np.log10(err), 1)
    return float(a[0]), float(a[1])


def max_err_vs_ref(uv_stack: np.ndarray):

    u0, v0 = uv_stack[0, 0], uv_stack[0, 1]
    err_u, err_v = [], []
    for i in range(1, uv_stack.shape[0]):
        err_u.append(float(np.max(np.abs(uv_stack[i, 0] - u0))))
        err_v.append(float(np.max(np.abs(uv_stack[i, 1] - v0))))
    return np.asarray(err_u, np.float64), np.asarray(err_v, np.float64)


def plot_temporal_ols(dt_vals, err_u, err_v, save_path, title=None):
    _plot_ols_with_scatter(
        x_vals=dt_vals,
        err_u=err_u,
        err_v=err_v,
        save_path=save_path,
        x_label=r"$\Delta t$ (s)",
        title=title,
        log_prefix="Temporal OLS",
    )


def _plot_ols_with_scatter(x_vals, err_u, err_v, save_path, x_label, title=None, log_prefix="OLS"):
    import matplotlib.pyplot as plt

    x_vals = np.asarray(x_vals, dtype=np.float64)
    err_u = np.maximum(np.asarray(err_u, float), 1e-30)
    err_v = np.maximum(np.asarray(err_v, float), 1e-30)
    p_u, c_u = loglog_ols_line(x_vals, err_u)
    p_v, c_v = loglog_ols_line(x_vals, err_v)
    xf = np.logspace(np.log10(np.min(x_vals)), np.log10(np.max(x_vals)), 100, dtype=np.float64)
    e_u = 10.0 ** (c_u + p_u * np.log10(xf))
    e_v = 10.0 ** (c_v + p_v * np.log10(xf))
    print(f"{log_prefix} log10–log10: p_u = {p_u:.6f}, p_v = {p_v:.6f}")
    _rc = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Nimbus Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
    }
    with plt.rc_context(_rc):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.loglog(xf, e_u, "b-", linewidth=1.4, zorder=1)
        ax.loglog(xf, e_v, "r-", linewidth=1.4, zorder=1)
        ax.scatter(
            x_vals,
            err_u,
            c="tab:blue",
            s=32,
            zorder=3,
            label="u",
            edgecolors="white",
            linewidths=0.6,
        )
        ax.scatter(
            x_vals,
            err_v,
            c="tab:red",
            s=32,
            zorder=3,
            label="v",
            edgecolors="white",
            linewidths=0.6,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("max absolute error")
        if title is None:
            title = f"$p_u = {p_u:.3f}$,  $p_v = {p_v:.3f}$"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close("all")
    print("Saved", save_path)


def plot_spatial_ols(dx_vals, err_u, err_v, save_path, title=None):
    _plot_ols_with_scatter(
        x_vals=dx_vals,
        err_u=err_u,
        err_v=err_v,
        save_path=save_path,
        x_label=r"$\Delta x$",
        title=title,
        log_prefix="Spatial OLS",
    )


def build_temporal_convergence(mat_path, ic_index, t_final, dt_baseline, dt_compared):
    uv_data = load_uv_data(mat_path)
    u0, v0 = load_initial_uv(uv_data, ic_index)
    steps_ref = int(round(t_final / dt_baseline))
    assert abs(steps_ref * dt_baseline - t_final) < 1e-9
    u_ref, v_ref = get_last_uv(u0, v0, steps_ref, dt_baseline)
    uv_rows = [stack_uv_nchw(u_ref, v_ref)]
    for dtc in dt_compared:
        n_steps = int(round(t_final / dtc))
        assert abs(n_steps * dtc - t_final) < 1e-9, dtc
        u_cmp, v_cmp = get_last_uv(u0, v0, n_steps, dtc)
        uv_rows.append(stack_uv_nchw(u_cmp, v_cmp))
        eu = max_abs_field_error(u_cmp, u_ref)
        ev = max_abs_field_error(v_cmp, v_ref)
        print(
            f"dt={dtc:g} s, steps={n_steps}, max|u-ur|={eu:.6e}, max|v-vr|={ev:.6e}"
        )
    return np.concatenate(uv_rows, axis=0)


def build_spatial_convergence(mat_path, ic_index, t_final, dt_fixed, n_baseline, n_compared):
    steps = int(round(t_final / dt_fixed))
    assert abs(steps * dt_fixed - t_final) < 1e-9
    uv_data = load_uv_data(mat_path)
    u0_ref, v0_ref = load_initial_uv(uv_data, ic_index, out_n=n_baseline)
    set_dx_from_n(n_baseline)
    u_ref, v_ref = get_last_uv(u0_ref, v0_ref, steps, dt_fixed)
    u_ref_np = u_ref.detach().cpu().numpy()
    v_ref_np = v_ref.detach().cpu().numpy()
    uv_rows = [stack_uv_nchw(u_ref, v_ref)]
    for n_grid in n_compared:
        set_dx_from_n(n_grid)
        u0_cmp, v0_cmp = load_initial_uv(uv_data, ic_index, out_n=n_grid)
        u_cmp, v_cmp = get_last_uv(u0_cmp, v0_cmp, steps, dt_fixed)
        u_cmp_up = resize_nchw_np(u_cmp.detach().cpu().numpy(), (n_baseline, n_baseline))
        v_cmp_up = resize_nchw_np(v_cmp.detach().cpu().numpy(), (n_baseline, n_baseline))
        uv_rows.append(
            np.concatenate((u_cmp_up, v_cmp_up), axis=1).astype(np.float32, copy=False)
        )
        eu = float(np.max(np.abs(u_cmp_up - u_ref_np)))
        ev = float(np.max(np.abs(v_cmp_up - v_ref_np)))
        print(
            f"N={n_grid}, dx={1.0/n_grid:.6f}, max|u-ur|={eu:.6e}, max|v-vr|={ev:.6e}"
        )
    set_dx_from_n(n_baseline)
    return np.concatenate(uv_rows, axis=0)


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate separate three-field CD and CDR trajectories."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "Burgers_2001x2x100x100_[FWE,dt=00025].mat",
        help="MAT file containing the prescribed uv trajectory.",
    )
    parser.add_argument(
        "--cd-output",
        type=Path,
        default=project_root / "data" / "CD.h5",
        help="Output HDF5 file for source-free convection--diffusion data.",
    )
    parser.add_argument(
        "--cdr-output",
        type=Path,
        default=project_root / "data" / "CDR.h5",
        help="Output HDF5 file for convection--diffusion--reaction data.",
    )
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--viscosity", type=float, default=1.0 / 200.0)
    parser.add_argument("--scalar-diffusivity", type=float, default=1.0 / 200.0)
    parser.add_argument("--velocity-reaction-beta", type=float, default=1.0)
    parser.add_argument("--scalar-reaction-rate", type=float, default=1.0)
    parser.add_argument("--scalar-reaction-threshold", type=float, default=0.5)
    parser.add_argument("--cdr-steps", type=int, default=1000)
    parser.add_argument(
        "--compression",
        choices=("lzf", "gzip", "none"),
        default="lzf",
        help="Lossless HDF5 compression.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_cd_and_cdr_h5(
        transport_input_mat=args.input,
        cd_output_h5=args.cd_output,
        cdr_output_h5=args.cdr_output,
        cdr_steps=args.cdr_steps,
        dt=args.dt,
        viscosity=args.viscosity,
        scalar_diffusivity=args.scalar_diffusivity,
        velocity_reaction_beta=args.velocity_reaction_beta,
        scalar_reaction_rate=args.scalar_reaction_rate,
        scalar_reaction_threshold=args.scalar_reaction_threshold,
        compression=None if args.compression == "none" else args.compression,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

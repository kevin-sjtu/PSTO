import os
import sys
import pickle
import h5py
import torch
import argparse
import numpy as np


root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)
sys.path.insert(0, os.path.dirname(__file__))
from src.model import transport, build_dnn_source_model, build_backbone
from tripleflame import configuration
from utils.functions import load_transport_state_dict, rollout_channel_bounds

torch.set_default_dtype(torch.float32)


CHANNEL_NAMES = ["T", "H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2", "u", "v"]


def build_reaction_source(args, device):
    if not np.isclose(args.dt, 1e-6, rtol=0, atol=1e-15):
        raise ValueError('Existing reaction DNN checkpoints represent dt=1e-6 s')
    return build_dnn_source_model(root, 9, device,
                                  pressure=args.pressure, mechanism=args.mechanism,
                                  checkpoint=getattr(args, 'source_checkpoint', None))

def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def _safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        return torch.load(path, map_location=device, weights_only=False)

def validate_args(args):
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.data_idx is None:
        args.data_idx = [2001]
    if any(idx <= 0 for idx in args.data_idx):
        raise ValueError("--data_idx must be 1-based positive integers.")

    args.print_steps = list(range(1, args.steps + 1)) if args.print_steps is None else args.print_steps
    if any(step < 1 or step > args.steps for step in args.print_steps):
        raise ValueError("--print_steps must lie in [1, --steps].")


    args.hf = resolve_path(args.hf)
    args.save_dir = resolve_path(args.save_dir)
    args.ckpt_dir = resolve_path(args.ckpt_dir)
    args.mechanism = resolve_path(args.mechanism)
    args.coefficient_checkpoint = resolve_path(args.coefficient_checkpoint)
    if getattr(args, 'source_checkpoint', None):
        args.source_checkpoint = resolve_path(args.source_checkpoint)
    return args


def evaluate(args):
    torch.manual_seed(args.seed)
    data_idx = [x - 1 for x in args.data_idx]
    print_steps = [step - 1 for step in args.print_steps]

    os.makedirs(args.save_dir, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())

    load_channels = configuration()['load_channels']
    n_ch = len(load_channels)

    with h5py.File(args.hf, "r") as f:
        total = f["data"].shape[0]
        max_needed = max(data_idx) + args.steps + 1
        if max_needed > total:
            raise ValueError(
                f"Dataset has {total} frames but requested rollout needs frame {max_needed - 1}."
            )
        x_raw = torch.from_numpy(f["data"][data_idx]).float()
        y_raw = []
        for start_idx in data_idx:
            y_raw.append(torch.from_numpy(f["data"][start_idx + 1 : start_idx + args.steps + 1]).float())
        y_raw = torch.stack(y_raw, dim=0)

    TYU_in = x_raw[:, load_channels].to(device)
    TYU_out = y_raw[:, :, load_channels].to(device)
    del x_raw, y_raw

    source_model = build_reaction_source(args, device)
    window_min, window_max = rollout_channel_bounds(TYU_in, TYU_out)
    data_min, data_max = window_min.amin(0), window_max.amax(0)

    preds = {}
    for name in args.models:
        directory = f'model_{name}'
        default_ckpt_path = os.path.join(args.ckpt_dir, directory, 'checkpoint_best.pt')
        direct_ckpt_path = os.path.join(args.ckpt_dir, "checkpoint_best.pt")
        if os.path.exists(default_ckpt_path):
            ckpt_path = default_ckpt_path
        else:
            ckpt_path = direct_ckpt_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[{name}] checkpoint not found: {ckpt_path}")

        ckpt = _safe_torch_load(ckpt_path, device)
        if isinstance(ckpt, dict):
            state = ckpt["model"]
        else:
            state = ckpt

        model_max, model_min = data_max, data_min
        backbone = build_backbone(
            name,
            n_ch,
            args,
            data_max=model_max,
            data_min=model_min,
            state=state,
        )
        if name == "PSTO":
            load_transport_state_dict(backbone, state, strict=True)
        else:
            backbone.load_state_dict(state, strict=True)
        backbone.eval().to(device)
        transport_model = transport(backbone).eval().to(device)
        transport_model.load_stats({'data_max': model_max, 'data_min': model_min})

        pred = []
        with torch.no_grad():
            for i in range(len(data_idx)):
                
                transport_model.load_stats({'data_min': window_min[i], 'data_max': window_max[i]})
                cur = TYU_in[i : i + 1].clone().to(device)
                step_out = []
                for t in range(args.steps):
                    mid = transport_model(cur)
                    cur = source_model(mid)
                    if not torch.isfinite(cur).all():
                        raise ValueError(f'{name} produced nonfinite output at step {t + 1}')
                    step_out.append(cur.detach())
                pred.append(torch.cat(step_out, dim=0))
        preds[name] = torch.stack(pred, dim=0)
        del backbone, transport_model, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_print_ch = TYU_out.shape[2]
    ch_names = CHANNEL_NAMES[:n_print_ch]
    for name, stack in preds.items():
        print(f"\n{'=' * 12} {name} {'=' * 12}", flush=True)
        for s in print_steps:
            pred_s = stack[:, s]
            gt_s = TYU_out[:, s]
            pred_error = pred_s.double()
            gt_error = gt_s.double()
            parts = []
            field_rels = []
            for c, ch_name in enumerate(ch_names):
                num = torch.sqrt(torch.sum((pred_error[:, c] - gt_error[:, c]) ** 2))
                den = torch.sqrt(torch.sum(gt_error[:, c] ** 2))
                ratio = torch.where((num==0)&(den==0),0.,num/den)
                rel = ratio.item()
                field_rels.append(ratio)
                parts.append(f"{ch_name}={rel:.6e}")
            mean_rel = torch.stack(field_rels).mean().item()
            print(
                f"  step {s + 1:3d}/{args.steps}: mean_physical_channel_rel_L2={mean_rel:.6e}  "
                + "  ".join(parts),
                flush=True,
            )

    results = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in preds.items()}
    results["ground_truth"] = TYU_out.detach().cpu().numpy()
    results["input"] = TYU_in.detach().cpu().numpy()
    npz_path = os.path.join(args.save_dir, f"{args.save_name}_{args.steps}.npz")
    
    
    temporary = f'{npz_path}.{os.getpid()}.tmp'
    try:
        with open(temporary, 'wb') as stream:
            np.savez_compressed(stream, **results)
        os.replace(temporary, npz_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return npz_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, nargs="+",
                    default=["PSTO", "FNO", "Unet", "UFNO"])
    ap.add_argument("--data_idx", type=int, nargs="+", default=None, help="1-based start indices; default 2001 (zero-based frame 2000)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--print_steps", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--hf", type=str, default=os.path.join(root, "data/H2_tripleflame.h5"))
    ap.add_argument("--save_dir", type=str, default=os.path.join(root, "analysis/figures"))
    ap.add_argument("--ckpt_dir", type=str, default=os.path.join(root, "checkpoints/models"))
    ap.add_argument("--latent_per_var", type=int, default=8)
    ap.add_argument("--save_name", type=str, default="test_multi")
    ap.add_argument('--source_checkpoint', default=None, help='Species MLP checkpoint; defaults to checkpoints/models/model_DNN/dnn_model_Y.pt')
    ap.add_argument('--coefficient_backend', choices=('cantera', 'dnn'), default='dnn')
    ap.add_argument('--coefficient_checkpoint', default='checkpoints/models/model_coeff/constitutive_DNN_BCT.pt')
    ap.add_argument('--mechanism', default=os.path.join(root, 'mech/h2o2.yaml'))
    ap.add_argument('--pressure', type=float, default=101325.0)
    ap.add_argument("--dt", type=float, default=1e-6)
    ap.add_argument("--dx", type=float, default=1e-4)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--hidden_channels", type=int, default=20)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = validate_args(ap.parse_args())
    evaluate(args)

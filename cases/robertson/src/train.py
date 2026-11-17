from robertson import configuration, args_parser

import os
import torch.distributed as dist
from pathlib import Path
import h5py, numpy as np
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler, Subset
import torch
from torch.distributed.nn.functional import all_reduce
from torch import nn
from pytorch_wavelets import DWTForward
import torch.nn.functional as F
from FNO import FNO
from PSTO import PSTO
from utils.functions import enforce_boundaries
from Unet import Unet
from UFNO import UFNO
def global_sum(value):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    if torch.is_grad_enabled() and value.requires_grad:
        return all_reduce(value, op=dist.ReduceOp.SUM)
    result = value.clone()
    dist.all_reduce(result)
    return result


MODELS = ('PSTO', 'FNO', 'Unet', 'UFNO')

def normalize(x, lower, upper):
    return 2 * (x - lower) / (upper - lower).clamp_min(1e-30) - 1

def denormalize(x, lower, upper):
    return (x + 1) * 0.5 * (upper - lower) + lower

def error_fields(x, lower, upper):
    return (x - lower) / (upper - lower).clamp_min(1e-30)

def channel_rel_l2(pred, target):
    dims = (0, 2, 3)
    floor = (0.0001 * target.shape[0] * target.shape[2] * target.shape[3]) ** 0.5
    return torch.linalg.vector_norm(pred - target, dim=dims) / torch.linalg.vector_norm(target, dim=dims).clamp_min(floor)

def build_model(name, config, data_min, data_max, modes=12, hidden_channels=20,
                n_layers=4, latent_per_var=8):
    input_channels = len(config['channels'])
    output_channels = 4
    if name == 'PSTO':
        return PSTO(input_channels, latent_per_var=latent_per_var, fno_modes=modes,
                    n_layers=n_layers, data_min=data_min, data_max=data_max,
                    dt=config['dt'], dx=config['dx'],
                    species_diffusivity=config['species_diffusivity'])
    if name == 'FNO':
        return FNO(input_channels, output_channels, hidden_channels, modes,
                   n_layers, config['dt'])
    if name == 'Unet':
        return Unet(input_channels, output_channels, hidden_channels,
                    n_layers, config['dt'])
    if name == 'UFNO':
        return UFNO(input_channels, output_channels, hidden_channels, modes,
                    n_layers, config['dt'])
    raise ValueError(name)

class Transport(nn.Module):
    def __init__(self, model, lower, upper):
        super().__init__()
        self.model = model
        self.register_buffer('lower', lower.clone())
        self.register_buffer('upper', upper.clone())
        if getattr(model, 'physical_io', False):
            model.set_stats(upper, lower)

    def forward(self, state):
        physical = getattr(self.model, 'physical_io', False)
        pred = (self.model(state) if physical else
                denormalize(self.model(normalize(state, self.lower, self.upper)),
                            self.lower[:4], self.upper[:4]))
        if getattr(self.model, 'finalizes_transport_output', False):
            return pred
        species = pred.clamp(min=self.lower[:4], max=self.upper[:4]).clamp_min(0)
        species = species / species.sum(1, keepdim=True).clamp_min(1e-12)
        return enforce_boundaries(species, state[:, :4])

class PhysicsLoss(nn.Module):
    def __init__(self, config, lower, upper):
        super().__init__()
        self.config = config
        self.register_buffer('ranges', (upper[:4] - lower[:4]).clamp_min(1e-30))

    def forward(self, x, pred):
        c = self.config
        dx = c['dx']
        previous = x[:, :4]
        center = pred[..., 1:-1, 1:-1]
        gx = (pred[..., 1:-1, 2:] - pred[..., 1:-1, :-2]) / (2 * dx)
        gy = (pred[..., 2:, 1:-1] - pred[..., :-2, 1:-1]) / (2 * dx)
        lap = (pred[..., 1:-1, 2:] + pred[..., 1:-1, :-2] + pred[..., 2:, 1:-1]
               + pred[..., :-2, 1:-1] - 4 * center) / dx**2
        u = x[:, 4:5, 1:-1, 1:-1]
        v = x[:, 5:6, 1:-1, 1:-1]
        residual = ((center - previous[..., 1:-1, 1:-1]) / c['dt']
                    + u * gx + v * gy - c['species_diffusivity'] * lap)
        return (residual / self.ranges).square().mean()

ROOT = Path(__file__).resolve().parents[1]


def load_frames(path, start, count):
    with h5py.File(path, 'r') as f:
        if start < 0 or start + count > len(f['data']):
            raise ValueError('Requested data window exceeds the dataset')
        return torch.from_numpy(f['data'][start:start + count]).float()


def model_args(args):
    keys = ['modes', 'latent_per_var', 'n_layers'] if args.model_type == 'PSTO' else ['hidden_channels', 'n_layers']
    if args.model_type in ('FNO', 'UFNO'):
        keys.append('modes')
    return {key: getattr(args, key) for key in keys}

def predict(model, state, lo, hi):
    return (model(state) if getattr(model, 'physical_io', False) else
            denormalize(model(normalize(state, lo, hi)), lo[:4], hi[:4]))

class DataLoss(nn.Module):
    def __init__(self, args, roi_channel, device):
        super().__init__()
        self.roi_weight = args.roi_weight
        self.weighted_power = args.weighted_power
        self.wavelet_weight = args.wavelet_weight
        self.wavelet_level = args.wavelet_level
        self.roi_channel = roi_channel
        self.dwt = DWTForward(J=args.wavelet_level, wave="haar", mode="zero").to(device).float()

    def _error_fields(self, pred, target):
        return pred, target

    def rel_l2_loss(self, pred, y):
            pred, y = self._error_fields(pred, y)
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                squared = global_sum(torch.stack(((pred-y).square().sum((0, 2, 3)),
                                                  y.square().sum((0, 2, 3)))))
                error_norm = squared[0].clamp_min(torch.finfo(pred.dtype).tiny).sqrt()
                target_norm = squared[1].sqrt()
                batch_size = y.shape[0] * torch.distributed.get_world_size()
            else:
                error_norm = torch.linalg.vector_norm(pred - y, dim=(0, 2, 3))
                target_norm = torch.linalg.vector_norm(y, dim=(0, 2, 3))
                batch_size = y.shape[0]
            floor = (1e-4 * batch_size * y.shape[2] * y.shape[3]) ** 0.5
            return (error_norm / target_norm.clamp_min(floor)).mean()

    def weighted_roi_loss(self, pred, y, x=None, eps=1e-8):
            if self.roi_weight == 0:
                return torch.zeros((), device=pred.device, dtype=pred.dtype)
            pred, y = self._error_fields(pred, y)
            diff = (pred - y) ** 2
            T = y[:, self.roi_channel:self.roi_channel + 1]
            dTdy = F.pad(T[:, :, 1:, :] - T[:, :, :-1, :], (0, 0, 0, 1))
            dTdx = F.pad(T[:, :, :, 1:] - T[:, :, :, :-1], (0, 1, 0, 0))
            g = torch.sqrt(dTdx * dTdx + dTdy * dTdy + 1e-12)
            g = g / (g.amax(dim=(2, 3), keepdim=True) + eps)
            w_roi = g ** self.weighted_power
            num = global_sum((w_roi * diff).sum(dim=(0, 2, 3)))
            den = global_sum(((y ** 2) * w_roi).sum(dim=(0, 2, 3))) + eps
            den = den.clamp_min(1e-4 * global_sum(w_roi.sum()))
            return self.roi_weight * (num / den).mean()

    def _wavelet_discrepancy(self, a, b):

            all_, alh = self.dwt(a)
            bll, blh = self.dwt(b)

            lam11, lam12 = 1.0, 0.25
            lam1, lam2, lam3, lam4 = 40.0, 2.5, 0.3, 0.2

            h1a, h1b = alh[0], blh[0]
            ma = torch.sqrt((h1a ** 2).sum(dim=2) + 1e-8)
            mb = torch.sqrt((h1b ** 2).sum(dim=2) + 1e-8)
            loss1 = lam11 * F.mse_loss(ma, mb) + lam12 * torch.mean(torch.abs(h1a - h1b))

            loss2 = torch.mean(torch.abs(alh[1] - blh[1])) if self.wavelet_level >= 2 else torch.zeros_like(loss1)
            loss3 = torch.mean(torch.abs(alh[2] - blh[2])) if self.wavelet_level >= 3 else torch.zeros_like(loss1)
            loss4 = F.mse_loss(all_, bll)

            return lam1 * loss1 + lam2 * loss2 + lam3 * loss3 + lam4 * loss4

    def wavelet_data_loss(self, pred, y, x=None, eps=1e-12):
            if self.wavelet_weight == 0:
                return torch.zeros((), device=pred.device, dtype=pred.dtype)
            pred, y = self._error_fields(pred, y)

            with torch.autocast(device_type=pred.device.type, enabled=False):
                loss = self._wavelet_discrepancy(pred.float(), y.float())
                return (self.wavelet_weight * loss).to(pred.dtype)

class ValidationTotals:

    def __init__(self, channels, device):
        self.c = channels
        
        self.values = torch.zeros(4 * channels + 6, device=device, dtype=torch.float64)

    def add(self, a, b, objective, conservation, raw_pde):
        c = self.c
        aa, bb = a.double(), b.double()
        self.values[:c] += (aa-bb).square().sum((0,2,3))
        self.values[c:2*c] += bb.square().sum((0,2,3))
        target = b[:, objective.roi_channel:objective.roi_channel+1]
        dy = F.pad(target[:,:,1:,:]-target[:,:,:-1,:], (0,0,0,1))
        dx = F.pad(target[:,:,:,1:]-target[:,:,:,:-1], (0,1,0,0))
        g = (dx.square()+dy.square()+1e-12).sqrt()
        weights = (g/(g.amax((2,3),keepdim=True)+1e-8)).pow(objective.weighted_power).double()
        self.values[2*c:3*c] += (weights*(aa-bb).square()).sum((0,2,3))
        self.values[3*c:4*c] += (weights*bb.square()).sum((0,2,3))
        n = a.shape[0]
        self.values[4*c:] += torch.stack((weights.sum(),
            objective.wavelet_data_loss(a,b).double()*n, conservation.double()*n,
            raw_pde.double()*n, a.new_tensor(n).double(),
            aa.new_tensor(n*a.shape[-2]*a.shape[-1])))

    def finish(self, objective, physics_weight):
        v, c = self.values, self.c
        if dist.is_initialized():
            dist.all_reduce(v)
        roi_mass, wave, cons, pde, count, pixels = v[4*c:]
        if count.item() == 0:
            raise ValueError('Validation dataset is empty')
        rel = (v[:c].sqrt()/v[c:2*c].sqrt().clamp_min((1e-4*pixels).sqrt())).mean()
        roi = objective.roi_weight * (v[2*c:3*c]/(v[3*c:4*c]+1e-8).clamp_min(1e-4*roi_mass)).mean()
        wave, cons, raw = wave/count, cons/count, pde/count
        data = rel+roi+wave
        
        weighted_pde = physics_weight * data * raw / raw.clamp_min(torch.finfo(torch.float32).eps)
        values = torch.stack((data+cons+weighted_pde, rel, roi, wave, cons, weighted_pde))
        return values, raw


def train_main(arguments=None):
    args = args_parser().parse_args(arguments)
    c = configuration()
    if min(args.batch_size, args.n_iters, args.grad_accum_steps, args.save_every) < 1:
        raise ValueError('Batch size, iterations, accumulation and save interval must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world > 1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl')
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    else:
        device = torch.device(args.device)

    def log(*values):
        if rank == 0:
            print(*values, flush=True)

    def reduce_mean(value):
        value = value.detach().clone()
        if world > 1:
            dist.all_reduce(value)
            value /= world
        return value

    with h5py.File(ROOT / c['lf'], 'r') as stream:
        n_pairs = len(stream['data']) - c['start'] - 1
    if args.max_pairs > 0:
        n_pairs = min(n_pairs, args.max_pairs)
    if (args.train_pairs is None) != (args.val_pairs is None):
        raise ValueError('Specify both train_pairs and val_pairs, or neither')
    train_pairs = max(1, int(0.8 * n_pairs)) if args.train_pairs is None else args.train_pairs
    val_pairs = n_pairs - train_pairs if args.val_pairs is None else args.val_pairs
    if min(train_pairs, val_pairs) < 1 or train_pairs + val_pairs > n_pairs:
        raise ValueError('Invalid consecutive-pair split')
    frames = load_frames(ROOT / c['lf'], c['start'], train_pairs + val_pairs + 1)
    train = frames[:train_pairs + 1]
    lo = train.amin((0, 2, 3)).reshape(-1, 1, 1).to(device)
    hi = train.amax((0, 2, 3)).reshape(-1, 1, 1).to(device)
    model = build_model(args.model_type, c, lo, hi, **model_args(args)).to(device)
    physics = PhysicsLoss(c, lo, hi).to(device)
    data_objective = DataLoss(args, roi_channel=1, device=device)
    if world > 1:
        for parameter in model.parameters():
            dist.broadcast(torch.view_as_real(parameter.data) if parameter.is_complex() else parameter.data, 0)
    path = Path(args.checkpoint_dir) if args.checkpoint_dir else ROOT / 'checkpoints/models' / f'model_{args.model_type}'
    path.mkdir(parents=True, exist_ok=True)
    training_data = TensorDataset(frames[:train_pairs], frames[1:train_pairs + 1])
    validation_data = TensorDataset(frames[train_pairs:-1], frames[train_pairs + 1:])
    if args.max_val_pairs > 0 and len(validation_data) > args.max_val_pairs:
        indices = torch.linspace(0, len(validation_data) - 1, args.max_val_pairs).round().long().tolist()
        validation_data = Subset(validation_data, indices)
    sampler = DistributedSampler(training_data, num_replicas=world, rank=rank, seed=args.seed) if world > 1 else None
    if world > 1:
        validation_data = Subset(validation_data, range(rank, len(validation_data), world))
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(training_data, shuffle=sampler is None, sampler=sampler,
                              generator=torch.Generator().manual_seed(args.seed), **loader_args)
    val_loader_args = dict(loader_args)
    if args.val_batch_size is not None:
        if args.val_batch_size < 1:
            raise ValueError('Validation batch size must be positive')
        val_loader_args['batch_size'] = args.val_batch_size
    val_loader = DataLoader(validation_data, shuffle=False, **val_loader_args)
    spectral, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (spectral if name.endswith(('weights1', 'weights2', 'logits')) else other).append(parameter)
    optimizer = torch.optim.AdamW([{'params': other, 'weight_decay': args.weight_decay},
                                   {'params': spectral, 'weight_decay': 0.0}], lr=args.learning_rate)
    start_step = 0
    best = float('inf')
    if args.restart:
        checkpoint = torch.load(path / 'checkpoint_best.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        start_step = checkpoint['step']
        best = checkpoint['val_loss']
        if not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
    remaining = args.n_iters - start_step
    if remaining < 1:
        raise ValueError('No remaining training steps')
    updates = (remaining + args.grad_accum_steps - 1) // args.grad_accum_steps
    
    warm = max(1, (args.warmup_iters + args.grad_accum_steps - 1) // args.grad_accum_steps)
    warm = min(warm, max(1, updates - 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, 0.001 if not args.restart else 0.01, 1.0, total_iters=warm),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, updates - warm), eta_min=1e-6)], [warm])

    def losses(x, y):
        target = y[:, :4]
        if getattr(model, 'physical_io', False):
            pred = model(x)
            a = error_fields(pred, lo[:4], hi[:4])
            b = error_fields(target, lo[:4], hi[:4])
        else:
            normalized_pred = model(normalize(x, lo, hi))
            a = 0.5 * (normalized_pred + 1)
            b = 0.5 * (normalize(target, lo[:4], hi[:4]) + 1)
            pred = denormalize(normalized_pred, lo[:4], hi[:4])
        rel_l2 = data_objective.rel_l2_loss(a, b)
        roi_loss = data_objective.weighted_roi_loss(a, b)
        wavelet_loss = data_objective.wavelet_data_loss(a, b)
        data_loss = rel_l2 + roi_loss + wavelet_loss
        cons_loss = (args.mass_fraction_weight * (pred.sum(1) - 1).square().mean()
                     + args.nonneg_weight * F.relu(-pred).square().mean(dim=(0, 2, 3)).mean())
        pde_loss_raw = physics(x, pred) if args.physics_weight else rel_l2.new_zeros(())
        scales = torch.stack((data_loss.detach(), pde_loss_raw.detach()))
        if world > 1 and torch.is_grad_enabled():
            dist.all_reduce(scales)
            scales /= world
        denominator = scales[1].clamp_min(torch.finfo(scales.dtype).eps)
        pde_weight = torch.where(scales[1] > 0, scales[0] / denominator, torch.zeros_like(scales[0]))
        pde_loss = args.physics_weight * pde_weight * pde_loss_raw
        return data_loss + cons_loss + pde_loss, (rel_l2, roi_loss, wavelet_loss, cons_loss, pde_loss)

    def print_losses(step, phase, values):
        log(f'Step {step}, {phase} rel_L2:{values[1].item():.6e} roi:{values[2].item():.6e} '
            f'wav:{values[3].item():.6e} cons:{values[4].item():.6e} '
            f'pde:{values[5].item():.6e} total:{values[0].item():.6e}')

    history = []
    epoch = 0
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.n_iters + 1):
        model.train()
        try:
            x, y = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            x, y = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        total, terms = losses(x, y)
        if not torch.isfinite(total):
            raise ValueError(f'Nonfinite loss at step {step}')
        (total / args.grad_accum_steps).backward()
        if step % args.grad_accum_steps == 0 or step == args.n_iters:
            if world > 1:
                
                gradients = [torch.view_as_real(p.grad) if p.grad.is_complex() else p.grad
                             for p in model.parameters() if p.grad is not None]
                bucket = torch.cat([g.reshape(-1) for g in gradients])
                dist.all_reduce(bucket)
                bucket /= world
                offset = 0
                for gradient in gradients:
                    size = gradient.numel()
                    gradient.copy_(bucket[offset:offset + size].reshape_as(gradient))
                    offset += size
                del bucket, gradients, gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise ValueError(f'Nonfinite gradient at step {step}')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        if step % args.print_every == 0 or step == 1:
            print_losses(step, 'Train', reduce_mean(torch.stack((total, *terms))))
        if step % args.save_every == 0 or step == args.n_iters:
            model.eval()
            totals = ValidationTotals(4, device)
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vp = predict(model, vx, lo, hi)
                    va = error_fields(vp, lo[:4], hi[:4])
                    vb = error_fields(vy[:, :4], lo[:4], hi[:4])
                    conservation = (args.mass_fraction_weight * (vp.sum(1)-1).square().mean()
                                    + args.nonneg_weight * F.relu(-vp).square().mean())
                    raw_pde = physics(vx, vp) if args.physics_weight else vp.new_zeros(())
                    totals.add(va, vb, data_objective, conservation, raw_pde)
            values, _ = totals.finish(data_objective, args.physics_weight)
            print_losses(step, 'Val', values)
            reported = reduce_mean(torch.stack((total, *terms)))
            if rank == 0:
                history.append([step, reported[1].item(), values[1].item(),
                                reported[0].item(), values[0].item()])
                np.savetxt(path / 'train_val_loss.csv', history, delimiter=',',
                           header='step,train_rel_L2,val_rel_L2,train_loss,val_loss')
            if not torch.isfinite(values).all():
                raise ValueError(f'Nonfinite validation loss at step {step}')
            if rank == 0 and values[0].item() < best and step >= start_step + args.start_save_step:
                best = values[0].item()
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                step=step, val_loss=best), path / 'checkpoint_best.pt')
            if world > 1:
                dist.barrier()
            del values
        del x, y, total, terms
    if world > 1:
        dist.destroy_process_group()
    return path

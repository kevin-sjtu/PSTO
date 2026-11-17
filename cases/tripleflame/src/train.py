import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWTForward
from accelerate.data_loader import prepare_data_loader
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.functions import denormalization, normalize_error_channels
from utils.functions import diffusive_flux_divergence, ContinuityFlux
from DNN import NN_MLP
from train_constitutive import COEFFICIENT_LAYERS
from src.thermodynamics import IdealGasDensity

def global_sum(value):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    if torch.is_grad_enabled() and value.requires_grad:
        return all_reduce(value, op=dist.ReduceOp.SUM)
    result = value.clone()
    dist.all_reduce(result)
    return result


class ValidationTotals:


    def __init__(self, channels, device):
        self.c = channels
        
        self.values = torch.zeros(4 * channels + 7, device=device, dtype=torch.float64)

    def add(self, pred, target, objective, mass_fraction, nonnegative, raw_pde):
        a, b = objective._error_fields(pred, target)
        aa, bb = a.double(), b.double()
        c = self.c
        self.values[:c] += (aa - bb).square().sum((0, 2, 3))
        self.values[c:2*c] += bb.square().sum((0, 2, 3))
        temperature = b[:, :1]
        dy = F.pad(temperature[:, :, 1:, :] - temperature[:, :, :-1, :], (0, 0, 0, 1))
        dx = F.pad(temperature[:, :, :, 1:] - temperature[:, :, :, :-1], (0, 1, 0, 0))
        gradient = (dx.square() + dy.square() + 1e-12).sqrt()
        weights = (gradient / (gradient.amax((2, 3), keepdim=True) + 1e-8))
        weights = weights.pow(objective.weighted_power).double()
        self.values[2*c:3*c] += (weights * (aa - bb).square()).sum((0, 2, 3))
        self.values[3*c:4*c] += (weights * bb.square()).sum((0, 2, 3))
        n = a.shape[0]
        self.values[4*c:] += torch.stack((
            weights.sum(), objective.wavelet_data_loss(pred, target).double() * n,
            mass_fraction.double() * n, nonnegative.double() * n, raw_pde.double() * n,
            aa.new_tensor(n), aa.new_tensor(n * a.shape[-2] * a.shape[-1]),
        ))

    def finish(self, objective):
        v = objective.accelerator.reduce(self.values, 'sum')
        c = self.c
        roi_mass, wave, mf, nn_loss, pde, count, pixels = v[4*c:]
        if count.item() == 0:
            raise ValueError('Validation dataset is empty')
        rel = (v[:c].sqrt() / v[c:2*c].sqrt().clamp_min((1e-4 * pixels).sqrt())).mean()
        roi = objective.roi_weight * (v[2*c:3*c] /
              (v[3*c:4*c] + 1e-8).clamp_min(1e-4 * roi_mass)).mean()
        wave, mf, nn_loss, raw = wave / count, mf / count, nn_loss / count, pde / count
        data = rel + roi + wave
        weighted_pde = (objective.physics_weight * data * raw /
                        raw.clamp_min(torch.finfo(torch.float32).eps))
        terms = (rel, roi, mf, nn_loss, wave, weighted_pde)
        return (data + mf + nn_loss + weighted_pde).item(), tuple(x.item() for x in terms), raw.item()


class trainer(nn.Module):
    def __init__(self, model, dt, dx, data_tuple, learning_rate, n_iters,
                print_every, save_every, start_save_step, grad_accum_steps, warmup_iters, weight_decay,
                accelerator=None, data_max=None, data_min=None,
                val_data_max=None, val_data_min=None,
                roi_weight=0.0, weighted_power=2, wavelet_weight=0.0, wavelet_level=3,
                mass_fraction_weight=0.0, nonneg_weight=0.0, physics_weight=0.0,
                lewis=None, scheduler_units='updates',
                restart=False, reset_optimizer=False, ckpt_dir="checkpoints/models/",
                coefficient_checkpoint='checkpoints/models/model_coeff/constitutive_DNN_BCT.pt',
                mechanism='mech/h2o2.yaml', pressure=101325.0, run_metadata=None):
        super().__init__()
        self.model = model
        self.run_metadata = run_metadata or {}
        self.physical_io = getattr(model, 'physical_io', False)
        self.train_loader, self.val_loader = data_tuple
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.print_every = print_every
        self.save_every = save_every
        self.warmup_iters = warmup_iters
        self.weight_decay = weight_decay
        self.accelerator = accelerator
        self.dt, self.dx = dt, dx
        self.train_data_max = data_max.to(accelerator.device)
        self.train_data_min = data_min.to(accelerator.device)
        self.val_data_max = val_data_max.to(accelerator.device) if val_data_max is not None\
        else data_max.to(accelerator.device)
        self.val_data_min = val_data_min.to(accelerator.device) if val_data_min is not None\
        else data_min.to(accelerator.device)
        self.data_max = self.train_data_max
        self.data_min = self.train_data_min
        self.mass_fraction_weight = mass_fraction_weight
        self.nonneg_weight = nonneg_weight
        self.roi_weight = roi_weight
        self.weighted_power = weighted_power
        self.physics_weight = physics_weight
        self.continuity_density = IdealGasDensity(mechanism, pressure).to(accelerator.device)
        self.baseline_constitutive = None
        if physics_weight and not hasattr(model, 'physical_coefficients'):
            saved = torch.load(coefficient_checkpoint, map_location='cpu', weights_only=False)
            if saved.get('preprocessing') != 'BCT_0.1_then_train_standardization':
                raise ValueError('Expected a BCT-standardized constitutive checkpoint')
            self.baseline_constitutive = NN_MLP(COEFFICIENT_LAYERS).to(accelerator.device)
            self.baseline_constitutive.load_state_dict(saved['net'])
            self.baseline_constitutive.requires_grad_(False).eval()
            for name in ('data_in_mean', 'data_in_std', 'data_target_mean', 'data_target_std'):
                self.register_buffer('physics_' + name, torch.as_tensor(saved[name],
                                      device=accelerator.device, dtype=torch.float32))
        self.wavelet_weight = wavelet_weight
        self.wavelet_level = wavelet_level
        self.lewis = torch.as_tensor(lewis, device=accelerator.device).view(1, -1, 1, 1)
        self.scheduler_units = scheduler_units
        self.restart = restart
        self.reset_optimizer = reset_optimizer
        self.ckpt_dir = ckpt_dir
        self.start_step = 0
        self.start_save_step = start_save_step
        self.grad_accum_steps = grad_accum_steps

        self.dwt = DWTForward(J=wavelet_level, wave="haar", mode="zero").to(accelerator.device).float()


    def _run_validation(self, model, val_loader, device):
        accelerator = self.accelerator
        model.eval()
        totals = ValidationTotals(self.data_min.shape[0], device)

        with torch.no_grad(), accelerator.autocast():
            for b_idx, batch_v in enumerate(val_loader):
                if isinstance(batch_v, (list, tuple)) and len(batch_v) == 3:
                    x_v, y_v, p_v = batch_v
                else:
                    x_v, y_v = batch_v
                    p_v = None

                v_pred = model(x_v, coefficients=p_v) if self.physical_io and p_v is not None else model(x_v)
                b_mf = self.specie_mass_fraction_sum_loss(v_pred)
                b_nn = self.nonneg_mass_fraction_loss(v_pred)
                batch_pde_loss_raw = self.pde_loss(x_v, v_pred, p_v)
                totals.add(v_pred, y_v, self, b_mf, b_nn, batch_pde_loss_raw)

        total, terms, self.validation_raw_pde = totals.finish(self)
        return total, terms

    def data_generator(self, loader):
        while True:
            for batch in loader:
                yield batch

    def _error_fields(self, pred, y):

        if self.physical_io:
            pred = normalize_error_channels(pred, self.data_min, self.data_max)
            y = normalize_error_channels(y, self.data_min, self.data_max)
        else:
            
            pred, y = 0.5 * (pred + 1), 0.5 * (y + 1)
        return pred, y

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
        T = y[:, 0:1]
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

    def specie_mass_fraction_sum_loss(self, pred, eps=1e-8):
        
        if self.mass_fraction_weight == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        y_phys = pred[:, 1:-2] if self.physical_io else denormalization(
            pred[:, 1:-2], self.data_max[1:-2], self.data_min[1:-2])
        s = y_phys.sum(dim=1)
        ones = torch.ones_like(s)
        raw = ((s - ones) ** 2).sum() / ((ones ** 2).sum() + eps)
        return self.mass_fraction_weight * raw

    def nonneg_mass_fraction_loss(self, pred, eps=1e-8):

        if self.nonneg_weight == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        y_phys = pred[:, 1:-2] if self.physical_io else denormalization(
            pred[:, 1:-2], self.data_max[1:-2], self.data_min[1:-2])
        raw = (F.relu(-y_phys) ** 2).mean(dim=(0, 2, 3)).mean()
        return self.nonneg_weight * raw


    def _physics_coefficients(self, physical):
        net = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(net, 'physical_coefficients'):
            return net.physical_coefficients(physical)[1]
        outputs = []
        points = physical[:, :10].permute(0, 2, 3, 1).reshape(-1, 10)
        for chunk in points.split(16384):
            def evaluate(local):
                with torch.autocast(device_type=local.device.type, enabled=False):
                    species = local[:, 1:].float().abs()
                    transformed = torch.where(species > 0,
                        (species.clamp_min(torch.finfo(torch.float32).tiny).pow(0.1)-1)/0.1,
                        torch.full_like(species, -10))
                    features = torch.cat((local[:, :1].float(), transformed), dim=1)
                    features = (features-self.physics_data_in_mean)/self.physics_data_in_std
                    values = self.baseline_constitutive(features)
                    values = values*self.physics_data_target_std+self.physics_data_target_mean
                    return (1+0.1*values).clamp_min(0).pow(10)
            outputs.append(evaluate(chunk))
        coefficients = torch.cat(outputs).reshape(physical.shape[0], *physical.shape[2:], 3).permute(0, 3, 1, 2)
        if not torch.isfinite(coefficients).all() or (coefficients <= 0).any():
            raise ValueError("Frozen coefficient DNN returned nonpositive or nonfinite properties")
        return coefficients

    def pde_loss(self, x, pred, p=None):
        if self.physics_weight == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        if not self.physical_io:
            x = denormalization(x, self.data_max, self.data_min)
            pred = denormalization(pred, self.data_max, self.data_min)
        net = self.model.module if hasattr(self.model, 'module') else self.model
        
        if getattr(net, 'transport_coefficients', None) is not None:
            p = net.transport_coefficients
        elif hasattr(net, 'physical_coefficients') or self.baseline_constitutive is not None:
            p = self._physics_coefficients(x)
        elif p is None:
            raise ValueError('PDE loss requires physical coefficients')
        rho, mu, kappa_T = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        pred_t = (pred - x) / self.dt
        
        scalar_channels = pred.shape[1] - 2
        advecting = getattr(net, 'transport_velocity', None)
        if advecting is None:
            # Match PSTO: freeze the advecting velocity at the input time for
            # both scalar advection and the continuity flux below.
            advecting = x[:, -2:]
        u = torch.cat((advecting[:, :1].expand(-1, scalar_channels, -1, -1),
                       x[:, -2:-1].expand(-1, 2, -1, -1)), dim=1)
        v = torch.cat((advecting[:, 1:].expand(-1, scalar_channels, -1, -1),
                       x[:, -1:].expand(-1, 2, -1, -1)), dim=1)

        dfdx = (pred[:, :, 1:-1, 2:] - pred[:, :, 1:-1, :-2]) / (2.0 * self.dx)
        dfdy = (pred[:, :, 2:, 1:-1] - pred[:, :, :-2, 1:-1]) / (2.0 * self.dx)
        adv = F.pad(u[:, :, 1:-1, 1:-1] * dfdx + v[:, :, 1:-1, 1:-1] * dfdy, (1, 1, 1, 1))

        inv_le = 1.0 / self.lewis.to(dtype=kappa_T.dtype, device=kappa_T.device)
        kappa_scalar = torch.cat([kappa_T, kappa_T * inv_le], dim=1)
        flux_coefficient = torch.cat((kappa_scalar, mu.expand(-1, 2, -1, -1)), dim=1)
        inlet = x[..., :1]
        diffusion = diffusive_flux_divergence(
            pred, flux_coefficient) / self.dx ** 2
        inv_rho = 1.0 / rho

        transport_rhs = -adv + inv_rho * diffusion
        res = pred_t - transport_rhs
        
        scale = (self.data_max - self.data_min).clamp_min(1e-30)
        res = res / scale
        rho_continuity_old = self.continuity_density(x)
        rho_future = self.continuity_density(pred)
        
        raw_flux = ContinuityFlux.face_flux(
            advecting, rho_future, inlet[:, -2:], self.continuity_density(inlet))
        raw_residual = (rho_future - rho_continuity_old) / self.dt + ContinuityFlux.divergence(
            raw_flux[0], raw_flux[1], self.dx)
        continuity_residual = (raw_residual / rho_continuity_old).to(res.dtype)
        
        
        
        residuals = torch.cat((res[:, :scalar_channels], continuity_residual), dim=1)
        return residuals[:, :, 1:-1, 1:-1].square().mean()

    def _build_optimizer(self, model):

        spectral, other = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (spectral if name.endswith(("weights1", "weights2", "spectral_weight")) else other).append(param)
        groups = [{"params": other, "weight_decay": self.weight_decay},
                  {"params": spectral, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=self.learning_rate)

    def _adaptive_pde_weight(self, data_loss, pde_loss):
        if self.physics_weight == 0:
            return data_loss.new_zeros(())
        
        scales = torch.stack((data_loss.detach(), pde_loss.detach()))
        scales = self.accelerator.reduce(scales, 'mean')
        denominator = scales[1].clamp_min(torch.finfo(scales.dtype).eps)
        return torch.where(scales[1] > 0, scales[0] / denominator, torch.zeros_like(scales[0]))

    def train(self):
        accelerator = self.accelerator
        model = self.model
        device = accelerator.device

        val_loss_min = float('inf')

        if self.restart:
            ckpt = torch.load(f"{self.ckpt_dir}/checkpoint_best.pt", map_location=device, weights_only=False)
            if not isinstance(ckpt, dict):
                raise ValueError("Loaded checkpoint is not a dictionary")
            model.load_state_dict(ckpt.get("model", ckpt), strict=True)
            if hasattr(model, 'set_stats'):
                model.set_stats(self.train_data_max, self.train_data_min)
            if hasattr(model, 'coefficient_model') and model.coefficient_model is not None:
                model.coefficient_model.requires_grad_(False).eval()
            self.start_step = ckpt.get("step", 0)
            remaining_iters = self.n_iters - self.start_step

            if "val_loss" in ckpt:
                val_loss_min = ckpt["val_loss"]

            optimizer = self._build_optimizer(model)
            if not self.reset_optimizer and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])

            if self.scheduler_units == 'microsteps':
                remaining_updates = max(1, remaining_iters)
                warmup_restart = min(self.warmup_iters, max(1, remaining_updates - 1))
            else:
                remaining_updates = max(1, (remaining_iters + self.grad_accum_steps - 1) // self.grad_accum_steps)
                warmup_restart = min(max(1, (self.warmup_iters + self.grad_accum_steps - 1)
                                         // self.grad_accum_steps), max(1, remaining_updates - 1))
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=warmup_restart)
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, max(1, remaining_updates - warmup_restart), eta_min=1e-6)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, main], [warmup_restart])
        else:
            optimizer = self._build_optimizer(model)
            if self.scheduler_units == 'microsteps':
                total_updates = max(1, self.n_iters)
                warmup_updates = min(self.warmup_iters, max(1, total_updates - 1))
            else:
                total_updates = max(1, (self.n_iters + self.grad_accum_steps - 1) // self.grad_accum_steps)
                warmup_updates = min(max(1, (self.warmup_iters + self.grad_accum_steps - 1)
                                         // self.grad_accum_steps), max(1, total_updates - 1))
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 0.001, 1.0, total_iters=warmup_updates)
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, max(1, total_updates - warmup_updates), eta_min=1e-6)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, main], [warmup_updates])

        
        
        if dist.is_initialized():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    dist.broadcast(torch.view_as_real(parameter.data) if parameter.is_complex()
                                   else parameter.data, 0)
        optimizer, train_loader = accelerator.prepare(optimizer, self.train_loader)
        
        
        val_loader = prepare_data_loader(
            self.val_loader, device=device, num_processes=accelerator.num_processes,
            process_index=accelerator.process_index, put_on_device=True,
            even_batches=False, dispatch_batches=False,
        )

        train_iter = self.data_generator(train_loader)
        record = []
        if self.restart and accelerator.is_main_process:
            history = os.path.join(self.ckpt_dir, 'train_val_loss.csv')
            if os.path.exists(history):
                previous = np.loadtxt(history, delimiter=',', skiprows=1, ndmin=2)
                record = [row.tolist() for row in previous if row[0] <= self.start_step]

        optimizer.zero_grad()
        for step in range(self.start_step, self.n_iters):
            model.train()
            batch = next(train_iter)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, y, p = batch
            else:
                x, y = batch
                p = None

            with accelerator.autocast():
                pred = model(x, coefficients=p) if self.physical_io and p is not None else model(x)
                rel_l2 = self.rel_l2_loss(pred, y)
                roi_loss = self.weighted_roi_loss(pred, y, x)
                w_loss = self.wavelet_data_loss(pred, y, x)
                mf_loss = self.specie_mass_fraction_sum_loss(pred)
                nn_loss = self.nonneg_mass_fraction_loss(pred)
                pde_loss_raw = self.pde_loss(x, pred, p)
                data_loss = rel_l2 + roi_loss + w_loss
                cons_loss = mf_loss + nn_loss
                pde_weight = self._adaptive_pde_weight(data_loss, pde_loss_raw)
                pde_loss = self.physics_weight * pde_weight * pde_loss_raw
                loss_total = data_loss + cons_loss + pde_loss

            if not torch.isfinite(loss_total):
                raise ValueError(f'Nonfinite training loss at step {step+1}')
            group_size = min(self.grad_accum_steps,
                             self.n_iters - (step // self.grad_accum_steps)*self.grad_accum_steps)
            accelerator.backward(loss_total / group_size)

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == self.n_iters:
                if dist.is_initialized():
                    gradients = [torch.view_as_real(p.grad) if p.grad.is_complex() else p.grad
                                 for p in model.parameters() if p.grad is not None]
                    bucket = torch.cat([g.reshape(-1) for g in gradients])
                    dist.all_reduce(bucket)
                    bucket /= dist.get_world_size()
                    offset = 0
                    for gradient in gradients:
                        size = gradient.numel()
                        gradient.copy_(bucket[offset:offset+size].reshape_as(gradient))
                        offset += size
                    del gradients, gradient, bucket
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    raise ValueError(f'Nonfinite gradient at step {step+1}')
                optimizer.step()
                optimizer.zero_grad()
                if self.scheduler_units == 'updates':
                    scheduler.step()

            if self.scheduler_units == 'microsteps':
                scheduler.step()
            event_step = step + 1
            if event_step % self.print_every == 0:
                train_rel_l2 = accelerator.reduce(rel_l2, "mean").item()
                lroi = accelerator.reduce(roi_loss, "mean").item()
                lmf = accelerator.reduce(mf_loss, "mean").item()
                lnn = accelerator.reduce(nn_loss, "mean").item()
                wv = accelerator.reduce(w_loss, "mean").item()
                train_pde_loss = accelerator.reduce(pde_loss, "mean").item()
                t = accelerator.reduce(loss_total, "mean").item()

                if accelerator.is_main_process:
                    prefix = f"Step {step+1}"
                    accelerator.print(f"{prefix}, Train rel_L2:{train_rel_l2:.6e} roi:{lroi:.6f} "
                                      f"mf:{lmf:.6f} nn:{lnn:.6f} wav:{wv:.6f} pde:{train_pde_loss:.6e} total:{t:.6f}")

            if event_step % self.save_every == 0 or step + 1 == self.n_iters:
                self.data_max, self.data_min = self.val_data_max, self.val_data_min
                vt, (val_rel_l2, vroi, vmf, vnn, vw, val_pde_loss) = self._run_validation(
                    model=model, val_loader=val_loader, device=device)

                
                train_at_val = accelerator.reduce(loss_total, "mean").item()

                if accelerator.is_main_process:
                    prefix = f"Step {step+1}"
                    pad = " " * len(prefix)
                    accelerator.print(f"{pad}, Val rel_L2:{val_rel_l2:.6e} roi:{vroi:.6f} "
                                      f"mf:{vmf:.6f} nn:{vnn:.6f} wav:{vw:.6f} pde:{val_pde_loss:.6e} total:{vt:.6f}")

                    if not np.isfinite(vt):
                        raise ValueError(f'Nonfinite validation loss at step {step+1}')
                    record.append([step + 1, train_at_val, vt])
                    np.savetxt(f"{self.ckpt_dir}/train_val_loss.csv", np.asarray(record),
                               header='step,train_loss,val_loss', delimiter=',')
                    if vt < val_loss_min and (step + 1) >= self.start_step + self.start_save_step:
                        torch.save({
                            "model": (model.module if hasattr(model, "module") else model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step + 1,
                            "val_loss": vt,
                            "dt": float(self.dt),
                            "dx": float(self.dx),
                            "run_metadata": self.run_metadata,
                            },
                            f"{self.ckpt_dir}/checkpoint_best.pt"
                        )

                        val_loss_min = vt

                self.data_max, self.data_min = self.train_data_max, self.train_data_min
                accelerator.wait_for_everyone()

            
            del batch, x, y, p, pred, rel_l2, roi_loss, w_loss, mf_loss, nn_loss
            del pde_loss_raw, data_loss, cons_loss, pde_loss, loss_total
            net = model.module if hasattr(model, "module") else model
            if self.physical_io:
                net.transport_velocity = None
                net.transport_coefficients = None
            
            if (step + 1) % self.save_every == 0 and device.type == "cuda":
                torch.cuda.empty_cache()



import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

MODELS = ("PSTO", "FNO", "Unet", "UFNO")

def configuration():
    return {
        'channels': ['u', 'v', 'phi'],
        'dt': 0.00025,
        'dx': 0.01,
        'nu': 0.005,
        'alpha': 0.005,
        'lf': 'data/CD.h5',
        'hf': 'data/CDR.h5',
        'start': 0,
        'train_pairs': 1600,
        'val_pairs': 400,
        'n_iters': 10000,
        'batch_size': 16,
    }

def midpoint_index(path):

    import h5py
    with h5py.File(path, 'r') as data:
        return len(data['data']) // 2

def args_parser():
    c = configuration()
    p = argparse.ArgumentParser()
    p.add_argument('--model_type', choices=MODELS, default='PSTO')
    p.add_argument('--latent_per_var', type=int, default=8)
    p.add_argument('--modes', type=int, default=12)
    p.add_argument('--hidden_channels', type=int, default=20)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=c['batch_size'])
    p.add_argument('--n_iters', type=int, default=c['n_iters'])
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-05)
    p.add_argument('--warmup_iters', type=int, default=200)
    p.add_argument('--print_every', type=int, default=50)
    p.add_argument('--save_every', type=int, default=100)
    p.add_argument('--start_save_step', type=int, default=50)
    p.add_argument('--physics_weight', type=float, default=0.1)
    p.add_argument('--roi_weight', type=float, default=0.1)
    p.add_argument('--train_pairs', type=int, default=None)
    p.add_argument('--val_pairs', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=88)
    p.add_argument('--device', default='cuda')
    p.add_argument('--ckpt_dir', '--checkpoint_dir', dest='checkpoint_dir', default=None)
    p.add_argument('--max_pairs', type=int, default=0)
    p.add_argument('--max_val_pairs', type=int, default=0)
    p.add_argument('--grad_accum_steps', type=int, default=2)
    p.add_argument('--scheduler_units', choices=('updates', 'microsteps'), default='updates')
    p.add_argument('--weighted_power', type=float, default=1.0)
    p.add_argument('--wavelet_weight', type=float, default=0.02)
    p.add_argument('--wavelet_level', type=int, default=2)
    p.add_argument('--restart', action='store_true')
    p.add_argument('--reset_optimizer', action='store_true')
    return p

if __name__ == "__main__":
    from train import train_main
    train_main()

"""
train_pinn.py  —  TMC-PINN Unified Training Script (self-contained)
====================================================================
All model architectures and utilities are embedded — no external
imports from chlorwu/tmc-pinn needed. Just this one file + your data.

USAGE (Jupyter):
    import importlib.util
    spec = importlib.util.spec_from_file_location('train_pinn', './train_pinn.py')
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    tp.main(pde='reaction', dtype_start='fp64', optim_start='lbfgs', total_epochs=2000)

7 CONDITIONS:
    Static:    fp64lbfgs, fp64adam, fp32lbfgs, fp32adam
    Switching: fp32adam_to_fp64adam, fp32lbfgs_to_fp64lbfgs, fp32adam_to_fp64lbfgs
"""

# ===========================================================================
# DEPENDENCIES — install if missing
# ===========================================================================
import subprocess, sys
for pkg in ['tqdm', 'pandas', 'matplotlib', 'scipy']:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q', '--break-system-packages'], check=False)

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import time
import json
import random
import argparse
import types
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import LBFGS, Adam
from tqdm import tqdm


# ===========================================================================
# CUDA SETUP  (TF32 disabled — critical for FP32 vs FP64 paper validity)
# ===========================================================================

def configure_cuda():
    if not torch.cuda.is_available():
        print('[CUDA] No GPU — running on CPU.')
        return
    print(f'[CUDA] Device  : {torch.cuda.get_device_name(0)}')
    print(f'[CUDA] CUDA    : {torch.version.cuda}')
    print(f'[CUDA] PyTorch : {torch.__version__}')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False
    torch.backends.cudnn.benchmark        = True
    torch.backends.cudnn.deterministic    = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    print('[CUDA] TF32 disabled — true IEEE 754 FP32/FP64')
    print('[CUDA] Deterministic mode ON')

configure_cuda()


# ===========================================================================
# PINN MODEL  (hidden=512, layers=4 — scaled for A10 GPU; note in paper methods)
# ===========================================================================

class PINN_Model(nn.Module):
    """
    Fully-connected PINN. hidden=512, layers=4 chosen for A10 GPU compute budget.
    Xu et al. use 1024x6 on faster GPUs; we note this in the methods section.
    Input: (x, t)  →  Output: u scalar
    """
    def __init__(self, in_dim=2, hidden_dim=512, out_dim=1, num_layer=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layer - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


def build_model(dtype, device):
    """Build and initialise the PINN. Xavier init matches baseline."""
    model = PINN_Model(in_dim=2, hidden_dim=1024, out_dim=1, num_layer=6)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    model.apply(init_weights)
    model = model.to(dtype).to(device)
    return model


# ===========================================================================
# DATA UTILITIES  (from chlorwu/tmc-pinn util.py — embedded)
# ===========================================================================

def get_data(x_range, t_range, x_num, t_num):
    """
    Returns collocation grid + 4 boundary slices.
    Layout matches the original util.get_data exactly.
    b_left  = t=0   (IC),  b_upper/lower = x walls (BC periodic)
    """
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(t_range[0], t_range[1], t_num)
    X, T = np.meshgrid(x, t)          # shape (t_num, x_num)
    res  = np.stack([X.ravel(), T.ravel()], axis=1)  # (N, 2)

    # Boundary slices
    b_left  = np.stack([x,                        np.zeros_like(x)           ], axis=1)  # t=0
    b_right = np.stack([x,                        np.full_like(x, t_range[1])], axis=1)  # t=T
    b_upper = np.stack([np.full_like(t, x_range[1]), t                       ], axis=1)  # x=x_max
    b_lower = np.stack([np.full_like(t, x_range[0]), t                       ], axis=1)  # x=x_min

    return res, b_left, b_right, b_upper, b_lower


# ===========================================================================
# PDE DEFINITIONS
# ===========================================================================

def _to_tensor(arr, dtype, device, requires_grad=True):
    return torch.tensor(arr, dtype=dtype, requires_grad=requires_grad).to(device)


def build_reaction_pde(device, dtype):
    """u_t = 5 u(1-u),  u(x,0) = Gaussian,  periodic BC"""
    res, b_left, b_right, b_upper, b_lower = get_data([0, 2*np.pi], [0, 1], 101, 101)
    res_test = res.copy()

    T = {
        'x_res':   _to_tensor(res[:,0:1],     dtype, device),
        't_res':   _to_tensor(res[:,1:2],     dtype, device),
        'x_left':  _to_tensor(b_left[:,0:1],  dtype, device),
        't_left':  _to_tensor(b_left[:,1:2],  dtype, device),
        'x_upper': _to_tensor(b_upper[:,0:1], dtype, device),
        't_upper': _to_tensor(b_upper[:,1:2], dtype, device),
        'x_lower': _to_tensor(b_lower[:,0:1], dtype, device),
        't_lower': _to_tensor(b_lower[:,1:2], dtype, device),
    }

    def loss_fn(model, T):
        pred_res   = model(T['x_res'],   T['t_res'])
        pred_upper = model(T['x_upper'], T['t_upper'])
        pred_lower = model(T['x_lower'], T['t_lower'])
        pred_left  = model(T['x_left'],  T['t_left'])

        u_t = torch.autograd.grad(
            pred_res, T['t_res'],
            grad_outputs=torch.ones_like(pred_res),
            retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_t - 5 * pred_res * (1 - pred_res)) ** 2)
        loss_bc  = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic  = torch.mean(
            (pred_left[:, 0:1] -
             torch.exp(-(T['x_left'] - torch.pi)**2 /
                       (2 * (torch.pi / 4)**2))) ** 2)
        return loss_res, loss_bc, loss_ic

    def exact_fn(xy):
        x, t = xy[:, 0], xy[:, 1]
        h = np.exp(-(x - np.pi)**2 / (2*(np.pi/4)**2))
        return (h * np.exp(5*t) / (h*np.exp(5*t) + 1 - h)).reshape(101, 101)

    return T, loss_fn, exact_fn, res_test


def build_wave_pde(device, dtype):
    """u_tt = u_xx,  u(x,0)=sin(πx), Dirichlet BC=0"""
    res, b_left, b_right, b_upper, b_lower = get_data([0, 1], [0, 1], 101, 101)
    res_test = res.copy()

    keys = ['x_res','t_res','x_left','t_left','x_upper','t_upper','x_lower','t_lower']
    arrs = [res[:,0:1], res[:,1:2],
            b_left[:,0:1], b_left[:,1:2],
            b_upper[:,0:1], b_upper[:,1:2],
            b_lower[:,0:1], b_lower[:,1:2]]
    T = {k: _to_tensor(a, dtype, device) for k, a in zip(keys, arrs)}

    def loss_fn(model, T):
        pred_res = model(T['x_res'], T['t_res'])
        u_t = torch.autograd.grad(pred_res, T['t_res'],
            grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, T['t_res'],
            grad_outputs=torch.ones_like(u_t), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(pred_res, T['x_res'],
            grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, T['x_res'],
            grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_tt - u_xx)**2)
        pred_upper = model(T['x_upper'], T['t_upper'])
        pred_lower = model(T['x_lower'], T['t_lower'])
        loss_bc  = torch.mean(pred_upper**2) + torch.mean(pred_lower**2)
        pred_left = model(T['x_left'], T['t_left'])
        loss_ic  = torch.mean((pred_left[:,0:1] - torch.sin(np.pi * T['x_left']))**2)
        return loss_res, loss_bc, loss_ic

    def exact_fn(xy):
        x, t = xy[:, 0], xy[:, 1]
        return (np.sin(np.pi*x) * np.cos(np.pi*t)).reshape(101, 101)

    return T, loss_fn, exact_fn, res_test


def build_convection_pde(device, dtype):
    """u_t + 30 u_x = 0,  u(x,0)=sin(x),  periodic BC"""
    beta = 30.0
    res, b_left, b_right, b_upper, b_lower = get_data([0, 2*np.pi], [0, 1], 101, 101)
    res_test = res.copy()

    keys = ['x_res','t_res','x_left','t_left','x_upper','t_upper','x_lower','t_lower']
    arrs = [res[:,0:1], res[:,1:2],
            b_left[:,0:1], b_left[:,1:2],
            b_upper[:,0:1], b_upper[:,1:2],
            b_lower[:,0:1], b_lower[:,1:2]]
    T = {k: _to_tensor(a, dtype, device) for k, a in zip(keys, arrs)}

    def loss_fn(model, T):
        pred_res = model(T['x_res'], T['t_res'])
        u_t = torch.autograd.grad(pred_res, T['t_res'],
            grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(pred_res, T['x_res'],
            grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        loss_res = torch.mean((u_t + beta * u_x)**2)

        # periodic BC: x walls
        pred_upper = model(T['x_upper'], T['t_upper'])
        pred_lower = model(T['x_lower'], T['t_lower'])
        loss_bc = torch.mean((pred_upper - pred_lower)**2)

        # IC: t=0 slice
        pred_left = model(T['x_left'], T['t_left'])
        loss_ic = torch.mean((pred_left[:,0:1] - torch.sin(T['x_left']))**2)
        return loss_res, loss_bc, loss_ic

    def exact_fn(xy):
        x, t = xy[:, 0], xy[:, 1]
        return np.sin(x - beta*t).reshape(101, 101)

    return T, loss_fn, exact_fn, res_test


def build_ac_pde(device, dtype):
    """Allen-Cahn: u_t - 0.0001 u_xx + 5(u^3-u) = 0"""
    try:
        import scipy.io
        data = scipy.io.loadmat('allen_cahn.mat')
        t_arr = data['t'].flatten()
        x_arr = data['x'].flatten()
        usol  = data['usol']  # (len(x), len(t))
    except Exception as e:
        print(f'[AC] allen_cahn.mat not found ({e}), using synthetic domain.')
        return _build_ac_synthetic(device, dtype)

    T_g, X_g = np.meshgrid(t_arr, x_arr)
    res = np.stack([X_g.ravel(), T_g.ravel()], axis=1)
    b_left  = np.stack([x_arr, np.zeros_like(x_arr)],        axis=1)
    b_upper = np.stack([np.full_like(t_arr, x_arr[-1]), t_arr], axis=1)
    b_lower = np.stack([np.full_like(t_arr, x_arr[0]),  t_arr], axis=1)
    res_test = res.copy()
    ic_vals  = torch.tensor(usol[:, 0], dtype=dtype).to(device)

    keys = ['x_res','t_res','x_left','t_left','x_upper','t_upper','x_lower','t_lower']
    arrs = [res[:,0:1], res[:,1:2],
            b_left[:,0:1], b_left[:,1:2],
            b_upper[:,0:1], b_upper[:,1:2],
            b_lower[:,0:1], b_lower[:,1:2]]
    T = {k: _to_tensor(a, dtype, device) for k, a in zip(keys, arrs)}

    d = 0.0001

    def loss_fn(model, T):
        pred_res = model(T['x_res'], T['t_res'])
        u_t = torch.autograd.grad(pred_res, T['t_res'],
            grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(pred_res, T['x_res'],
            grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, T['x_res'],
            grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        loss_res = torch.mean((u_t - d*u_xx + 5*(pred_res**3 - pred_res))**2)

        pred_upper = model(T['x_upper'], T['t_upper'])
        pred_lower = model(T['x_lower'], T['t_lower'])
        ux_upper = torch.autograd.grad(pred_upper, T['x_upper'],
            grad_outputs=torch.ones_like(pred_upper), retain_graph=True, create_graph=True)[0]
        ux_lower = torch.autograd.grad(pred_lower, T['x_lower'],
            grad_outputs=torch.ones_like(pred_lower), retain_graph=True, create_graph=True)[0]
        loss_bc = (torch.mean((pred_upper - pred_lower)**2) +
                   torch.mean((ux_upper  - ux_lower )**2))

        pred_left = model(T['x_left'], T['t_left'])
        loss_ic = torch.mean((pred_left[:,0:1] - ic_vals.unsqueeze(1))**2)
        return loss_res, loss_bc, loss_ic

    def exact_fn(xy):
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((x_arr, t_arr), usol,
                                         method='linear', bounds_error=False)
        return interp(xy).reshape(len(x_arr), len(t_arr))

    return T, loss_fn, exact_fn, res_test


def _build_ac_synthetic(device, dtype):
    res, b_left, b_right, b_upper, b_lower = get_data([-1, 1], [0, 1], 101, 101)
    res_test = res.copy()
    keys = ['x_res','t_res','x_left','t_left','x_upper','t_upper','x_lower','t_lower']
    arrs = [res[:,0:1], res[:,1:2],
            b_left[:,0:1], b_left[:,1:2],
            b_upper[:,0:1], b_upper[:,1:2],
            b_lower[:,0:1], b_lower[:,1:2]]
    T = {k: _to_tensor(a, dtype, device) for k, a in zip(keys, arrs)}
    d = 0.0001

    def loss_fn(model, T):
        pred_res = model(T['x_res'], T['t_res'])
        u_t = torch.autograd.grad(pred_res, T['t_res'],
            grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(pred_res, T['x_res'],
            grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, T['x_res'],
            grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        loss_res = torch.mean((u_t - d*u_xx + 5*(pred_res**3 - pred_res))**2)
        pred_upper = model(T['x_upper'], T['t_upper'])
        pred_lower = model(T['x_lower'], T['t_lower'])
        loss_bc = torch.mean((pred_upper - pred_lower)**2)
        pred_left = model(T['x_left'], T['t_left'])
        loss_ic = torch.mean((pred_left[:,0:1] -
            (T['x_left']**2 * torch.cos(np.pi * T['x_left'])))**2)
        return loss_res, loss_bc, loss_ic

    return T, loss_fn, None, res_test


PDE_BUILDERS = {
    'reaction':   build_reaction_pde,
    'wave':       build_wave_pde,
    'convection': build_convection_pde,
    'ac':         build_ac_pde,
}

DTYPE_MAP = {
    'fp32': torch.float32,
    'fp64': torch.float64,
}


# ===========================================================================
# OPTIMIZER FACTORY
# ===========================================================================

def make_optimizer(optim_key, model, adam_lr=1e-3,
                   lbfgs_tol_grad=1e-8, lbfgs_tol_change=1e-10):
    if optim_key == 'adam':
        return Adam(model.parameters(), lr=adam_lr)
    return LBFGS(model.parameters(),
                 line_search_fn='strong_wolfe',
                 tolerance_grad=lbfgs_tol_grad,
                 tolerance_change=lbfgs_tol_change)


# ===========================================================================
# TENSOR CAST
# ===========================================================================

def cast_tensors(tensors, dtype, device):
    return {k: v.detach().to(dtype=dtype, device=device).requires_grad_(v.requires_grad)
            for k, v in tensors.items()}


# ===========================================================================
# LOGGING
# ===========================================================================

def open_logs(run_dir, condition, pde, model_name):
    os.makedirs(run_dir, exist_ok=True)
    prefix = f'{run_dir}/{pde}_{model_name}_{condition}'
    paths = {
        'loss':   f'{prefix}_loss.csv',
        'timing': f'{prefix}_timing.csv',
        'grad':   f'{prefix}_grad.csv',
        'eval':   f'{prefix}_eval.csv',
        'switch': f'{prefix}_switch.csv',
    }
    with open(paths['loss'], 'w') as f:
        f.write('epoch,loss_res,loss_bc,loss_ic,total_loss,dtype,optim,phase,condition,pde\n')
    with open(paths['timing'], 'w') as f:
        f.write('epoch,fwd_time_s,bwd_time_s,total_time_s,mem_allocated_mb,mem_reserved_mb,dtype,optim,phase,condition,pde\n')
    with open(paths['grad'], 'w') as f:
        f.write('epoch,grad_norm,dtype,optim,phase,condition,pde\n')
    with open(paths['eval'], 'w') as f:
        f.write('condition,pde,model,L1_rel,L2_rel,final_loss,switch_epoch,total_epochs,total_time_s,peak_mem_mb\n')
    with open(paths['switch'], 'w') as f:
        f.write('condition,pde,model,switch_epoch,from_dtype,to_dtype,from_optim,to_optim,loss_at_switch,grad_norm_at_switch\n')
    return paths


def log_epoch(paths, epoch, lr_v, lbc_v, lic_v, tot_v,
              fwd_t, bwd_t, mem_alloc, mem_res,
              grad_norm, dtype_str, optim_str, phase, condition, pde):
    with open(paths['loss'], 'a') as f:
        f.write(f'{epoch},{lr_v:.8e},{lbc_v:.8e},{lic_v:.8e},{tot_v:.8e},'
                f'{dtype_str},{optim_str},{phase},{condition},{pde}\n')
    with open(paths['timing'], 'a') as f:
        f.write(f'{epoch},{fwd_t:.6f},{bwd_t:.6f},{fwd_t+bwd_t:.6f},'
                f'{mem_alloc:.2f},{mem_res:.2f},'
                f'{dtype_str},{optim_str},{phase},{condition},{pde}\n')
    with open(paths['grad'], 'a') as f:
        f.write(f'{epoch},{grad_norm:.8e},{dtype_str},{optim_str},{phase},{condition},{pde}\n')


# ===========================================================================
# TRAINING STEP
# ===========================================================================

def train_one_epoch(model, optim, tensors, loss_fn):
    timing = [0.0, 0.0]
    last_losses = [None]

    def closure():
        optim.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        lr, lbc, lic = loss_fn(model, tensors)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing[0] += time.perf_counter() - t0

        loss = lr + lbc + lic
        last_losses[0] = (lr.item(), lbc.item(), lic.item(), loss.item())

        t1 = time.perf_counter()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing[1] += time.perf_counter() - t1
        return loss

    optim.step(closure)

    total_norm = sum(
        p.grad.data.norm(2).item()**2
        for p in model.parameters() if p.grad is not None
    ) ** 0.5

    lr_v, lbc_v, lic_v, tot_v = last_losses[0]
    return lr_v, lbc_v, lic_v, tot_v, timing[0], timing[1], total_norm


# ===========================================================================
# EVALUATION
# ===========================================================================

def evaluate(model, res_test, exact_fn, dtype, device):
    if exact_fn is None:
        return float('nan'), float('nan')
    x_t = torch.tensor(res_test, dtype=dtype, requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(x_t[:, 0:1], x_t[:, 1:2])[:, 0:1].cpu().numpy()
    u = exact_fn(res_test)
    pred_r = pred.reshape(u.shape)
    rl1 = np.sum(np.abs(u - pred_r)) / np.sum(np.abs(u))
    rl2 = np.sqrt(np.sum((u - pred_r)**2) / np.sum(u**2))
    return float(rl1), float(rl2)


# ===========================================================================
# PLOTS
# ===========================================================================

def save_plots(run_dir, condition, pde, paths, switch_epoch):
    prefix = f'{run_dir}/{pde}_PINN_{condition}'
    try:
        df = pd.read_csv(paths['loss'])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(df['epoch'], df['total_loss'], 'b-',  lw=2,   label='Total')
        ax.semilogy(df['epoch'], df['loss_res'],   'r--', lw=1.5, label='Residual', alpha=0.7)
        ax.semilogy(df['epoch'], df['loss_bc'],    'g--', lw=1.5, label='BC',       alpha=0.7)
        ax.semilogy(df['epoch'], df['loss_ic'],    'm--', lw=1.5, label='IC',       alpha=0.7)
        if switch_epoch:
            ax.axvline(switch_epoch, color='k', ls=':', lw=1.5, label=f'switch@{switch_epoch}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
        ax.set_title(f'{pde} | {condition}')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{prefix}_loss_curve.pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[plot] loss: {e}')

    try:
        df_g = pd.read_csv(paths['grad'])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(df_g['epoch'], df_g['grad_norm'], 'b-', lw=1.5)
        if switch_epoch:
            ax.axvline(switch_epoch, color='k', ls=':', lw=1.5, label=f'switch@{switch_epoch}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Grad norm (log)')
        ax.set_title(f'Gradient norm — {pde} | {condition}')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{prefix}_grad_curve.pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[plot] grad: {e}')

    try:
        df_t = pd.read_csv(paths['timing'])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(df_t['epoch'], df_t['total_time_s'],  label='total')
        axes[0].plot(df_t['epoch'], df_t['fwd_time_s'],    label='fwd', alpha=0.7)
        axes[0].plot(df_t['epoch'], df_t['bwd_time_s'],    label='bwd', alpha=0.7)
        axes[1].plot(df_t['epoch'], df_t['mem_allocated_mb'], label='allocated MB')
        for ax in axes:
            if switch_epoch:
                ax.axvline(switch_epoch, color='k', ls=':', lw=1.5)
            ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
        axes[0].set_title('Time/epoch'); axes[1].set_title('GPU memory (MB)')
        fig.suptitle(f'{pde} | {condition}')
        fig.tight_layout()
        fig.savefig(f'{prefix}_timing_curve.pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[plot] timing: {e}')


# ===========================================================================
# ARGS — works from both CLI and Jupyter
# ===========================================================================

def parse_args(
    pde='reaction', model='PINN', device='cuda:0', seed=1,
    dtype_start='fp64', optim_start='lbfgs',
    dtype_switch=None, optim_switch=None, switch_epoch=None,
    total_epochs=2000, adam_lr=1e-3,
    lbfgs_tol_grad=1e-8, lbfgs_tol_change=1e-10,
    out_dir='./results',
):
    _in_notebook = any(
        x in ' '.join(sys.argv)
        for x in ['ipykernel', 'jupyter', 'ipython', 'kernel']
    ) or not any(a.startswith('--') for a in sys.argv[1:])

    if not _in_notebook:
        p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--pde',              type=str,   default=pde)
        p.add_argument('--model',            type=str,   default=model)
        p.add_argument('--device',           type=str,   default=device)
        p.add_argument('--seed',             type=int,   default=seed)
        p.add_argument('--dtype_start',      type=str,   default=dtype_start,  choices=['fp32','fp64'])
        p.add_argument('--optim_start',      type=str,   default=optim_start,  choices=['adam','lbfgs'])
        p.add_argument('--dtype_switch',     type=str,   default=dtype_switch, choices=['fp32','fp64'])
        p.add_argument('--optim_switch',     type=str,   default=optim_switch, choices=['adam','lbfgs'])
        p.add_argument('--switch_epoch',     type=int,   default=switch_epoch)
        p.add_argument('--total_epochs',     type=int,   default=total_epochs)
        p.add_argument('--adam_lr',          type=float, default=adam_lr)
        p.add_argument('--lbfgs_tol_grad',   type=float, default=lbfgs_tol_grad)
        p.add_argument('--lbfgs_tol_change', type=float, default=lbfgs_tol_change)
        p.add_argument('--out_dir',          type=str,   default=out_dir)
        args = p.parse_args()
    else:
        args = types.SimpleNamespace(
            pde=pde, model=model, device=device, seed=seed,
            dtype_start=dtype_start, optim_start=optim_start,
            dtype_switch=dtype_switch, optim_switch=optim_switch,
            switch_epoch=switch_epoch, total_epochs=total_epochs,
            adam_lr=adam_lr, lbfgs_tol_grad=lbfgs_tol_grad,
            lbfgs_tol_change=lbfgs_tol_change, out_dir=out_dir,
        )

    if args.dtype_switch is None:
        args.dtype_switch = args.dtype_start
    if args.optim_switch is None:
        args.optim_switch = args.optim_start

    is_switching = (
        args.switch_epoch is not None
        and args.switch_epoch < args.total_epochs
        and (args.dtype_switch != args.dtype_start
             or args.optim_switch != args.optim_start)
    )
    args.condition   = (f'{args.dtype_start}{args.optim_start}_to_{args.dtype_switch}{args.optim_switch}'
                        if is_switching else f'{args.dtype_start}{args.optim_start}')
    args.is_switching = is_switching
    return args


# ===========================================================================
# MAIN
# ===========================================================================

def main(**kwargs):
    args = parse_args(**kwargs)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[WARN] No GPU — falling back to CPU.')
        args.device = 'cpu'
    device = args.device

    dtype     = DTYPE_MAP[args.dtype_start]
    dtype_str = args.dtype_start
    optim_str = args.optim_start
    phase     = 'pre_switch' if args.is_switching else 'static'

    gpu_label = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'
    print(f'\n{"="*60}')
    print(f'  PDE       : {args.pde}')
    print(f'  Condition : {args.condition}')
    print(f'  Switch at : epoch {args.switch_epoch} ({"active" if args.is_switching else "no switch"})')
    print(f'  Epochs    : {args.total_epochs}')
    print(f'  Device    : {device} ({gpu_label})')
    print('  TF32      : disabled')
    print(f'{"="*60}\n')

    run_dir = os.path.join(args.out_dir, args.pde, args.condition)
    paths   = open_logs(run_dir, args.condition, args.pde, 'PINN')

    tensors, loss_fn, exact_fn, res_test = PDE_BUILDERS[args.pde](device, dtype)
    model = build_model(dtype, device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    optim = make_optimizer(args.optim_start, model, args.adam_lr,
                           args.lbfgs_tol_grad, args.lbfgs_tol_change)

    switch_epoch_actual = None
    total_wall_start    = time.perf_counter()
    peak_mem_mb         = 0.0

    for epoch in tqdm(range(1, args.total_epochs + 1),
                      desc=args.condition, ncols=100, unit='ep'):

        # ---- switch ----
        if args.is_switching and switch_epoch_actual is None and epoch == args.switch_epoch:
            switch_epoch_actual = epoch
            from_dtype, from_optim = dtype_str, optim_str
            to_dtype,   to_optim   = args.dtype_switch, args.optim_switch

            last_loss = last_grad = 0.0
            try:
                last_loss = pd.read_csv(paths['loss'])['total_loss'].iloc[-1]
                last_grad = pd.read_csv(paths['grad'])['grad_norm'].iloc[-1]
            except Exception:
                pass

            print(f'\n>>> SWITCH epoch {epoch}: {from_dtype}/{from_optim} → {to_dtype}/{to_optim}\n')
            new_dtype = DTYPE_MAP[to_dtype]
            model   = model.to(new_dtype)
            tensors = cast_tensors(tensors, new_dtype, device)
            optim   = make_optimizer(to_optim, model, args.adam_lr,
                                     args.lbfgs_tol_grad, args.lbfgs_tol_change)
            dtype, dtype_str, optim_str, phase = new_dtype, to_dtype, to_optim, 'post_switch'

            with open(paths['switch'], 'a') as f:
                f.write(f'{args.condition},{args.pde},PINN,{switch_epoch_actual},'
                        f'{from_dtype},{to_dtype},{from_optim},{to_optim},'
                        f'{last_loss:.8e},{last_grad:.8e}\n')

        # ---- train step ----
        lr_v, lbc_v, lic_v, tot_v, fwd_t, bwd_t, grad_norm = \
            train_one_epoch(model, optim, tensors, loss_fn)

        if torch.cuda.is_available():
            mem_alloc   = torch.cuda.memory_allocated(device) / 1e6
            mem_res     = torch.cuda.memory_reserved(device)  / 1e6
            peak_mem_mb = max(peak_mem_mb, mem_alloc)
        else:
            mem_alloc = mem_res = 0.0

        log_epoch(paths, epoch, lr_v, lbc_v, lic_v, tot_v,
                  fwd_t, bwd_t, mem_alloc, mem_res,
                  grad_norm, dtype_str, optim_str, phase,
                  args.condition, args.pde)

    # ---- eval ----
    total_wall = time.perf_counter() - total_wall_start
    rl1, rl2   = evaluate(model, res_test, exact_fn, dtype, device)
    print(f'\nRelative L1 : {rl1:.6f}')
    print(f'Relative L2 : {rl2:.6f}')
    print(f'Wall time   : {total_wall:.1f} s')
    print(f'Peak GPU mem: {peak_mem_mb:.1f} MB')

    try:
        final_loss = pd.read_csv(paths['loss'])['total_loss'].iloc[-1]
    except Exception:
        final_loss = float('nan')

    with open(paths['eval'], 'a') as f:
        f.write(f'{args.condition},{args.pde},PINN,'
                f'{rl1:.8e},{rl2:.8e},{final_loss:.8e},'
                f'{switch_epoch_actual},{args.total_epochs},'
                f'{total_wall:.2f},{peak_mem_mb:.2f}\n')

    prefix = f'{run_dir}/{args.pde}_PINN_{args.condition}'
    torch.save(model.state_dict(), f'{prefix}_model.pt')
    config = {**vars(args),
              'gpu_name':     torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU',
              'torch_version': torch.__version__,
              'cuda_version':  str(torch.version.cuda),
              'tf32_enabled':  False}
    with open(f'{prefix}_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    save_plots(run_dir, args.condition, args.pde, paths, switch_epoch_actual)
    print(f'\nOutputs → {os.path.abspath(run_dir)}')


if __name__ == '__main__':
    main()

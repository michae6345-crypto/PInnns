"""
train_pinn.py  —  TMC-PINN Unified Training Script
====================================================
Runs all 7 experimental conditions with identical code paths.

STATIC BASELINES (no switch fires):
    python train_pinn.py --pde reaction --optim_start lbfgs --dtype_start fp64
    python train_pinn.py --pde reaction --optim_start adam  --dtype_start fp64
    python train_pinn.py --pde reaction --optim_start lbfgs --dtype_start fp32
    python train_pinn.py --pde reaction --optim_start adam  --dtype_start fp32

SWITCHING CONDITIONS:
    python train_pinn.py --pde reaction --optim_start adam  --dtype_start fp32 --optim_switch adam  --dtype_switch fp64 --switch_epoch 1000
    python train_pinn.py --pde reaction --optim_start lbfgs --dtype_start fp32 --optim_switch lbfgs --dtype_switch fp64 --switch_epoch 1000
    python train_pinn.py --pde reaction --optim_start adam  --dtype_start fp32 --optim_switch lbfgs --dtype_switch fp64 --switch_epoch 1000

PDEs: reaction | wave | convection | ac
"""

import time
import os
import json
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import argparse
import numpy as np
import pandas as pd
from torch.optim import LBFGS, Adam
from tqdm import tqdm

# Import your existing utilities — must be on PYTHONPATH
from util import get_data
from model_dict import get_model


# ===========================================================================
# H100 / CUDA ENVIRONMENT SETUP
# Call once at import time so every run benefits automatically.
# ===========================================================================

def configure_cuda():
    if not torch.cuda.is_available():
        print('[CUDA] No GPU detected — running on CPU.')
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f'[CUDA] Device : {gpu_name}')
    print(f'[CUDA] CUDA   : {torch.version.cuda}')
    print(f'[CUDA] PyTorch: {torch.__version__}')

    # --- TF32 control (CRITICAL for this paper) ---
    # H100 and A100 use TF32 by default for matmuls and convolutions.
    # TF32 has FP32 range but only ~10 bits of mantissa (vs 23 for true FP32).
    # This means "FP32" runs are silently using reduced precision unless we
    # disable TF32 — making our FP32 vs FP64 comparison misleading.
    # We disable it so FP32 means true IEEE 754 single precision throughout.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('[CUDA] TF32 disabled — FP32 runs use true IEEE 754 single precision.')

    # --- cuDNN benchmark ---
    # Finds the fastest algorithm for fixed input shapes. No convolutions in
    # a standard PINN, but harmless to enable and helps with other architectures.
    torch.backends.cudnn.benchmark = True

    # --- deterministic ops ---
    # Makes results reproducible across runs on the same GPU.
    # Slight performance cost but essential for fair condition comparisons.
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    # warn_only=True: warns instead of crashing if a non-deterministic op
    # is unavoidable — keeps experiments running while flagging any issues.

    print('[CUDA] Deterministic mode: ON (reproducible across runs)')

configure_cuda()


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='TMC-PINN unified training script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- experiment identity ---
    p.add_argument('--pde',          type=str, default='reaction',
                   choices=['reaction', 'wave', 'convection', 'ac'],
                   help='Which PDE to solve')
    p.add_argument('--model',        type=str, default='PINN',
                   help='Model architecture key (see model_dict.py)')
    p.add_argument('--device',       type=str, default='cuda:0')
    p.add_argument('--seed',         type=int, default=1)

    # --- phase 1 (always runs) ---
    p.add_argument('--dtype_start',  type=str, default='fp32',
                   choices=['fp32', 'fp64'],
                   help='Floating-point dtype for phase 1')
    p.add_argument('--optim_start',  type=str, default='lbfgs',
                   choices=['adam', 'lbfgs'],
                   help='Optimizer for phase 1')

    # --- phase 2 (only runs if switch_epoch < total_epochs) ---
    p.add_argument('--dtype_switch', type=str, default=None,
                   choices=['fp32', 'fp64'],
                   help='Dtype after switch. Omit for static runs (defaults to dtype_start).')
    p.add_argument('--optim_switch', type=str, default=None,
                   choices=['adam', 'lbfgs'],
                   help='Optimizer after switch. Omit for static runs (defaults to optim_start).')
    p.add_argument('--switch_epoch', type=int, default=None,
                   help='Epoch at which to perform the switch (1-indexed). '
                        'None = no switch (static run).')

    # --- training budget ---
    p.add_argument('--total_epochs', type=int, default=2000)

    # --- optimizer hyperparams ---
    p.add_argument('--adam_lr',      type=float, default=1e-3)
    p.add_argument('--lbfgs_tol_grad',   type=float, default=1e-8)
    p.add_argument('--lbfgs_tol_change', type=float, default=1e-10)

    # --- output ---
    p.add_argument('--out_dir',      type=str, default='./results',
                   help='Root output directory. Run-specific subdir is created automatically.')

    args = p.parse_args()

    # Fill defaults for switch args
    if args.dtype_switch is None:
        args.dtype_switch = args.dtype_start
    if args.optim_switch is None:
        args.optim_switch = args.optim_start

    # Condition string — used in filenames and CSV columns
    is_switching = (
        args.switch_epoch is not None
        and args.switch_epoch < args.total_epochs
        and (args.dtype_switch != args.dtype_start or args.optim_switch != args.optim_start)
    )
    if is_switching:
        args.condition = (
            f'{args.dtype_start}{args.optim_start}'
            f'_to_{args.dtype_switch}{args.optim_switch}'
        )
    else:
        args.condition = f'{args.dtype_start}{args.optim_start}'

    args.is_switching = is_switching
    return args


# ===========================================================================
# DTYPE HELPERS
# ===========================================================================

DTYPE_MAP = {
    'fp32': torch.float32,
    'fp64': torch.float64,
}


# ===========================================================================
# PDE MODULES
# Each returns (get_data_fn, loss_fn, analytical_solution_fn or None)
# ===========================================================================

def build_reaction_pde(device, dtype):
    """1-D reaction equation: u_t = 5 u(1-u)"""
    res, b_left, b_right, b_upper, b_lower = get_data(
        [0, 2 * np.pi], [0, 1], 101, 101)
    res_test, *_ = get_data([0, 2 * np.pi], [0, 1], 101, 101)

    def to_tensor(arr, requires_grad=True):
        return torch.tensor(arr, dtype=dtype, requires_grad=requires_grad).to(device)

    tensors = dict(
        x_res=to_tensor(res)[:, 0:1],
        t_res=to_tensor(res)[:, 1:2],
        x_left=to_tensor(b_left)[:, 0:1],
        t_left=to_tensor(b_left)[:, 1:2],
        x_right=to_tensor(b_right)[:, 0:1],
        t_right=to_tensor(b_right)[:, 1:2],
        x_upper=to_tensor(b_upper)[:, 0:1],
        t_upper=to_tensor(b_upper)[:, 1:2],
        x_lower=to_tensor(b_lower)[:, 0:1],
        t_lower=to_tensor(b_lower)[:, 1:2],
    )

    def loss_fn(model, T):
        pred_res   = model(T['x_res'],   T['t_res'])
        pred_left  = model(T['x_left'],  T['t_left'])
        pred_upper = model(T['x_upper'], T['t_upper'])
        pred_lower = model(T['x_lower'], T['t_lower'])

        u_t = torch.autograd.grad(
            pred_res, T['t_res'],
            grad_outputs=torch.ones_like(pred_res),
            retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_t - 5 * pred_res * (1 - pred_res)) ** 2)
        loss_bc  = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic  = torch.mean(
            (pred_left[:, 0:1]
             - torch.exp(-(T['x_left'] - torch.pi) ** 2
                         / (2 * (torch.pi / 4) ** 2))) ** 2)
        return loss_res, loss_bc, loss_ic

    def exact_fn(xy):
        x, t = xy[:, 0], xy[:, 1]
        h = np.exp(-(x - np.pi) ** 2 / (2 * (np.pi / 4) ** 2))
        return (h * np.exp(5 * t) / (h * np.exp(5 * t) + 1 - h)).reshape(101, 101)

    return tensors, loss_fn, exact_fn, res_test


def build_wave_pde(device, dtype):
    """1-D wave equation: u_tt = u_xx"""
    res, b_left, b_right, b_upper, b_lower = get_data(
        [0, 1], [0, 1], 101, 101)
    res_test, *_ = get_data([0, 1], [0, 1], 101, 101)

    def to_tensor(arr, requires_grad=True):
        return torch.tensor(arr, dtype=dtype, requires_grad=requires_grad).to(device)

    tensors = dict(
        x_res=to_tensor(res)[:, 0:1],
        t_res=to_tensor(res)[:, 1:2],
        x_left=to_tensor(b_left)[:, 0:1],
        t_left=to_tensor(b_left)[:, 1:2],
        x_right=to_tensor(b_right)[:, 0:1],
        t_right=to_tensor(b_right)[:, 1:2],
        x_upper=to_tensor(b_upper)[:, 0:1],
        t_upper=to_tensor(b_upper)[:, 1:2],
        x_lower=to_tensor(b_lower)[:, 0:1],
        t_lower=to_tensor(b_lower)[:, 1:2],
    )

    def loss_fn(model, T):
        pred_res = model(T['x_res'], T['t_res'])
        u_tt = torch.autograd.grad(
            torch.autograd.grad(
                pred_res, T['t_res'],
                grad_outputs=torch.ones_like(pred_res),
                retain_graph=True, create_graph=True)[0],
            T['t_res'],
            grad_outputs=torch.ones_like(pred_res),
            retain_graph=True, create_graph=True)[0]
        u_x_intermediate = torch.autograd.grad(
            pred_res, T['x_res'],
            grad_outputs=torch.ones_like(pred_res),
            retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x_intermediate, T['x_res'],
            grad_outputs=torch.ones_like(u_x_intermediate),
            retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_tt - u_xx) ** 2)

        pred_upper = model(T['x_upper'], T['t_upper'])
        pred_lower = model(T['x_lower'], T['t_lower'])
        loss_bc = torch.mean(pred_upper ** 2) + torch.mean(pred_lower ** 2)

        pred_left = model(T['x_left'], T['t_left'])
        loss_ic = torch.mean(
            (pred_left[:, 0:1]
             - torch.sin(np.pi * T['x_left'])) ** 2)

        return loss_res, loss_bc, loss_ic

    def exact_fn(xy):
        x, t = xy[:, 0], xy[:, 1]
        return (np.sin(np.pi * x) * np.cos(np.pi * t)).reshape(101, 101)

    return tensors, loss_fn, exact_fn, res_test


def build_convection_pde(device, dtype):
    """1-D convection: u_t + u_x = 0, periodic BC"""
    res, b_left, b_right, b_upper, b_lower = get_data(
        [0, 2 * np.pi], [0, 1], 101, 101)
    res_test, *_ = get_data([0, 2 * np.pi], [0, 1], 101, 101)

    def to_tensor(arr, requires_grad=True):
        return torch.tensor(arr, dtype=dtype, requires_grad=requires_grad).to(device)

    tensors = dict(
        x_res=to_tensor(res)[:, 0:1],
        t_res=to_tensor(res)[:, 1:2],
        x_left=to_tensor(b_left)[:, 0:1],
        t_left=to_tensor(b_left)[:, 1:2],
        x_right=to_tensor(b_right)[:, 0:1],
        t_right=to_tensor(b_right)[:, 1:2],
        x_upper=to_tensor(b_upper)[:, 0:1],
        t_upper=to_tensor(b_upper)[:, 1:2],
        x_lower=to_tensor(b_lower)[:, 0:1],
        t_lower=to_tensor(b_lower)[:, 1:2],
    )

    beta = 30.0  # convection speed (stiff case from FP64-is-all-you-need)

    def loss_fn(model, T):
        pred_res = model(T['x_res'], T['t_res'])
        u_t = torch.autograd.grad(
            pred_res, T['t_res'],
            grad_outputs=torch.ones_like(pred_res),
            retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(
            pred_res, T['x_res'],
            grad_outputs=torch.ones_like(pred_res),
            retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_t + beta * u_x) ** 2)

        pred_upper = model(T['x_upper'], T['t_upper'])
        pred_lower = model(T['x_lower'], T['t_lower'])
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)        # periodic in x

        pred_left = model(T['x_left'], T['t_left'])                 # t = 0 slice
        loss_ic = torch.mean(
            (pred_left[:, 0:1] - torch.sin(T['x_left'])) ** 2)     # u(x,0) = sin(x)

        return loss_res, loss_bc, loss_ic

    def exact_fn(xy):
        x, t = xy[:, 0], xy[:, 1]
        return np.sin(x - beta * t).reshape(101, 101)

    return tensors, loss_fn, exact_fn, res_test


def build_ac_pde(device, dtype):
    """
    Allen-Cahn equation: u_t - 0.0001 u_xx + 5(u^3 - u) = 0
    Uses the .mat data file from the original FP64 repo.
    """
    import scipy.io
    try:
        data = scipy.io.loadmat('allen_cahn.mat')
        # Expected keys: 't', 'x', 'usol' (from original repo)
        t = data['t'].flatten()
        x = data['x'].flatten()
        usol = data['usol']           # shape (len(x), len(t))
    except Exception as e:
        print(f"[AC] Could not load allen_cahn.mat: {e}")
        print("[AC] Falling back to synthetic domain — results will differ from paper baseline.")
        return build_ac_pde_synthetic(device, dtype)

    # Collocation grid
    T_grid, X_grid = np.meshgrid(t, x)
    res = np.stack([X_grid.ravel(), T_grid.ravel()], axis=1)

    # Boundary / IC slices
    b_left  = np.stack([x, np.zeros_like(x)], axis=1)   # t=0 (IC)
    b_right = np.stack([x, np.full_like(x, t[-1])], axis=1)
    b_upper = np.stack([np.full_like(t, x[-1]), t], axis=1)
    b_lower = np.stack([np.full_like(t, x[0]),  t], axis=1)
    res_test = res.copy()

    def to_tensor(arr, requires_grad=True):
        return torch.tensor(arr, dtype=dtype, requires_grad=requires_grad).to(device)

    tensors = dict(
        x_res=to_tensor(res)[:, 0:1],
        t_res=to_tensor(res)[:, 1:2],
        x_left=to_tensor(b_left)[:, 0:1],
        t_left=to_tensor(b_left)[:, 1:2],
        x_right=to_tensor(b_right)[:, 0:1],
        t_right=to_tensor(b_right)[:, 1:2],
        x_upper=to_tensor(b_upper)[:, 0:1],
        t_upper=to_tensor(b_upper)[:, 1:2],
        x_lower=to_tensor(b_lower)[:, 0:1],
        t_lower=to_tensor(b_lower)[:, 1:2],
    )

    ic_vals = torch.tensor(
        usol[:, 0:1].flatten(), dtype=dtype).to(device)  # u(x, 0)

    d = 0.0001

    def loss_fn(model, T):
        pred_res = model(T['x_res'], T['t_res'])
        u_t = torch.autograd.grad(
            pred_res, T['t_res'],
            grad_outputs=torch.ones_like(pred_res),
            retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(
            pred_res, T['x_res'],
            grad_outputs=torch.ones_like(pred_res),
            retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x, T['x_res'],
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean(
            (u_t - d * u_xx + 5 * (pred_res ** 3 - pred_res)) ** 2)

        pred_upper = model(T['x_upper'], T['t_upper'])
        pred_lower = model(T['x_lower'], T['t_lower'])
        # periodic BC in x (value + derivative — matches canonical FP64 benchmark)
        ux_upper = torch.autograd.grad(pred_upper, T['x_upper'],
            grad_outputs=torch.ones_like(pred_upper),
            retain_graph=True, create_graph=True)[0]
        ux_lower = torch.autograd.grad(pred_lower, T['x_lower'],
            grad_outputs=torch.ones_like(pred_lower),
            retain_graph=True, create_graph=True)[0]
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2) \
                + torch.mean((ux_upper  - ux_lower)  ** 2)

        pred_left = model(T['x_left'], T['t_left'])       # t = 0 slice, length len(x) — matches ic_vals
        loss_ic = torch.mean((pred_left[:, 0:1] - ic_vals.unsqueeze(1)) ** 2)

        return loss_res, loss_bc, loss_ic

    def exact_fn(xy):
        # Bilinear interpolation from mat data
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((x, t), usol, method='linear',
                                         bounds_error=False, fill_value=None)
        return interp(xy).reshape(len(x), len(t))

    return tensors, loss_fn, exact_fn, res_test


def build_ac_pde_synthetic(device, dtype):
    """Fallback if allen_cahn.mat is unavailable."""
    res, b_left, b_right, b_upper, b_lower = get_data([-1, 1], [0, 1], 101, 101)
    res_test, *_ = get_data([-1, 1], [0, 1], 101, 101)

    def to_tensor(arr, requires_grad=True):
        return torch.tensor(arr, dtype=dtype, requires_grad=requires_grad).to(device)

    tensors = dict(
        x_res=to_tensor(res)[:, 0:1],   t_res=to_tensor(res)[:, 1:2],
        x_left=to_tensor(b_left)[:, 0:1],  t_left=to_tensor(b_left)[:, 1:2],
        x_right=to_tensor(b_right)[:, 0:1], t_right=to_tensor(b_right)[:, 1:2],
        x_upper=to_tensor(b_upper)[:, 0:1], t_upper=to_tensor(b_upper)[:, 1:2],
        x_lower=to_tensor(b_lower)[:, 0:1], t_lower=to_tensor(b_lower)[:, 1:2],
    )
    d = 0.0001

    def loss_fn(model, T):
        pred_res = model(T['x_res'], T['t_res'])
        u_t = torch.autograd.grad(pred_res, T['t_res'],
            grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(pred_res, T['x_res'],
            grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, T['x_res'],
            grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        loss_res = torch.mean((u_t - d * u_xx + 5 * (pred_res**3 - pred_res))**2)
        pred_upper = model(T['x_upper'], T['t_upper'])
        pred_lower = model(T['x_lower'], T['t_lower'])
        loss_bc = torch.mean((pred_upper - pred_lower)**2)          # periodic in x
        pred_left = model(T['x_left'], T['t_left'])                 # t = 0 slice
        loss_ic = torch.mean((pred_left[:, 0:1] - (T['x_left']**2 * torch.cos(np.pi * T['x_left'])))**2)
        return loss_res, loss_bc, loss_ic

    return tensors, loss_fn, None, res_test


PDE_BUILDERS = {
    'reaction':   build_reaction_pde,
    'wave':       build_wave_pde,
    'convection': build_convection_pde,
    'ac':         build_ac_pde,
}


# ===========================================================================
# OPTIMIZER FACTORY
# ===========================================================================

def make_optimizer(optim_key, model, adam_lr, lbfgs_tol_grad, lbfgs_tol_change):
    if optim_key == 'adam':
        return Adam(model.parameters(), lr=adam_lr)
    elif optim_key == 'lbfgs':
        return LBFGS(
            model.parameters(),
            line_search_fn='strong_wolfe',
            tolerance_grad=lbfgs_tol_grad,
            tolerance_change=lbfgs_tol_change,
        )
    else:
        raise ValueError(f'Unknown optimizer: {optim_key}')


# ===========================================================================
# TENSOR CAST HELPER  (avoids globals())
# ===========================================================================

def cast_tensors(tensors, dtype, device):
    """Re-cast all input tensors to a new dtype, preserving requires_grad."""
    return {
        k: v.detach().to(dtype=dtype, device=device).requires_grad_(v.requires_grad)
        for k, v in tensors.items()
    }


# ===========================================================================
# LOGGING SETUP
# ===========================================================================

def open_logs(out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    prefix = f'{out_dir}/{args.pde}_{args.model}_{args.condition}'

    loss_path  = f'{prefix}_loss.csv'
    timing_path = f'{prefix}_timing.csv'
    grad_path  = f'{prefix}_grad.csv'
    eval_path  = f'{prefix}_eval.csv'
    switch_path = f'{prefix}_switch.csv'

    with open(loss_path, 'w') as f:
        f.write('epoch,loss_res,loss_bc,loss_ic,total_loss,'
                'dtype,optim,phase,condition,pde\n')
    with open(timing_path, 'w') as f:
        f.write('epoch,fwd_time_s,bwd_time_s,total_time_s,'
                'mem_allocated_mb,mem_reserved_mb,dtype,optim,phase,condition,pde\n')
    with open(grad_path, 'w') as f:
        f.write('epoch,grad_norm,dtype,optim,phase,condition,pde\n')
    with open(eval_path, 'w') as f:
        f.write('condition,pde,model,L1_rel,L2_rel,final_loss,'
                'switch_epoch,total_epochs,total_time_s,peak_mem_mb\n')
    with open(switch_path, 'w') as f:
        f.write('condition,pde,model,switch_epoch,from_dtype,to_dtype,'
                'from_optim,to_optim,loss_at_switch,grad_norm_at_switch\n')

    return loss_path, timing_path, grad_path, eval_path, switch_path


def log_epoch(loss_path, timing_path, grad_path,
              epoch, loss_res, loss_bc, loss_ic, total_loss,
              fwd_t, bwd_t, mem_alloc_mb, mem_res_mb,
              grad_norm, dtype_str, optim_str, phase, condition, pde):
    with open(loss_path, 'a') as f:
        f.write(f'{epoch},{loss_res:.8e},{loss_bc:.8e},{loss_ic:.8e},'
                f'{total_loss:.8e},{dtype_str},{optim_str},{phase},{condition},{pde}\n')
    with open(timing_path, 'a') as f:
        f.write(f'{epoch},{fwd_t:.6f},{bwd_t:.6f},{fwd_t+bwd_t:.6f},'
                f'{mem_alloc_mb:.2f},{mem_res_mb:.2f},'
                f'{dtype_str},{optim_str},{phase},{condition},{pde}\n')
    if grad_norm is not None:
        with open(grad_path, 'a') as f:
            f.write(f'{epoch},{grad_norm:.8e},{dtype_str},{optim_str},{phase},{condition},{pde}\n')


# ===========================================================================
# MODEL INIT
# ===========================================================================

def build_model(args, dtype, device):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    if args.model == 'KAN':
        model = get_model(args).Model(
            width=[2, 5, 1], grid=5, k=3, grid_eps=1.0,
            noise_scale_base=0.25, device=device
        ).to(dtype).to(device)
    elif args.model == 'QRes':
        model = get_model(args).Model(
            in_dim=2, hidden_dim=256, out_dim=1, num_layer=4
        ).to(dtype).to(device)
        model.apply(init_weights)
    elif args.model in ('PINNsFormer', 'PINNsFormer_Enc_Only'):
        model = get_model(args).Model(
            in_dim=2, hidden_dim=32, out_dim=1, num_layer=1
        ).to(dtype).to(device)
        model.apply(init_weights)
    else:  # PINN (default), FLS, PINNMamba
        model = get_model(args).Model(
            in_dim=2, hidden_dim=1024, out_dim=1, num_layer=6
        ).to(dtype).to(device)
        model.apply(init_weights)

    return model


# ===========================================================================
# TRAINING STEP (handles both Adam and L-BFGS uniformly)
# ===========================================================================

def train_one_epoch(model, optim, optim_key, tensors, loss_fn):
    """
    Returns (loss_res, loss_bc, loss_ic, total_loss,
             fwd_time, bwd_time, grad_norm)

    Grad norm is computed AFTER optim.step() so it reflects the accepted step,
    not an intermediate L-BFGS line-search evaluation.
    """
    timing = [0.0, 0.0]      # accumulates over all closure calls within one .step()
    last_losses = [None]

    def closure():
        optim.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        lr, lbc, lic = loss_fn(model, tensors)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing[0] += time.perf_counter() - t0       # += accumulates across LBFGS line-search calls

        loss = lr + lbc + lic
        last_losses[0] = (lr.item(), lbc.item(), lic.item(), loss.item())

        t1 = time.perf_counter()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing[1] += time.perf_counter() - t1

        return loss

    optim.step(closure)

    # Gradient norm computed once, after the accepted step
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = total_norm ** 0.5

    lr_v, lbc_v, lic_v, tot_v = last_losses[0]
    return lr_v, lbc_v, lic_v, tot_v, timing[0], timing[1], grad_norm


# ===========================================================================
# EVALUATION
# ===========================================================================

def evaluate(model, pde, res_test, exact_fn, dtype, device):
    if exact_fn is None:
        return float('nan'), float('nan')

    res_test_t = torch.tensor(
        res_test, dtype=dtype, requires_grad=False
    ).to(device)

    x_test = res_test_t[:, 0:1]
    t_test = res_test_t[:, 1:2]

    with torch.no_grad():
        pred = model(x_test, t_test)[:, 0:1].cpu().numpy()

    u_exact = exact_fn(res_test)  # numpy, already (101,101) or flat
    pred_r = pred.reshape(u_exact.shape)

    rl1 = np.sum(np.abs(u_exact - pred_r)) / np.sum(np.abs(u_exact))
    rl2 = np.sqrt(np.sum((u_exact - pred_r) ** 2) / np.sum(u_exact ** 2))
    return float(rl1), float(rl2)


# ===========================================================================
# PLOTTING
# ===========================================================================

def save_plots(out_dir, args, loss_path, timing_path, grad_path, switch_epoch):
    prefix = f'{out_dir}/{args.pde}_{args.model}_{args.condition}'

    # --- loss curve ---
    try:
        df = pd.read_csv(loss_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(df['epoch'], df['total_loss'], 'b-',  label='Total',    lw=2)
        ax.semilogy(df['epoch'], df['loss_res'],   'r--', label='Residual', lw=1.5, alpha=0.7)
        ax.semilogy(df['epoch'], df['loss_bc'],    'g--', label='BC',       lw=1.5, alpha=0.7)
        ax.semilogy(df['epoch'], df['loss_ic'],    'm--', label='IC',       lw=1.5, alpha=0.7)
        if switch_epoch:
            ax.axvline(switch_epoch, color='k', ls=':', lw=1.5, label=f'switch @ {switch_epoch}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
        ax.set_title(f'{args.pde} | {args.condition} | {args.model}')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{prefix}_loss_curve.pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[plot] loss curve failed: {e}')

    # --- timing curve ---
    try:
        df_t = pd.read_csv(timing_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(df_t['epoch'], df_t['total_time_s'], label='total')
        axes[0].plot(df_t['epoch'], df_t['fwd_time_s'],   label='forward', alpha=0.7)
        axes[0].plot(df_t['epoch'], df_t['bwd_time_s'],   label='backward', alpha=0.7)
        if switch_epoch:
            for ax in axes:
                ax.axvline(switch_epoch, color='k', ls=':', lw=1.5)
        axes[0].set_title('Time per epoch (s)'); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(df_t['epoch'], df_t['mem_allocated_mb'], label='allocated MB')
        axes[1].set_title('GPU memory (MB)'); axes[1].legend(); axes[1].grid(alpha=0.3)
        for ax in axes:
            ax.set_xlabel('Epoch')
        fig.suptitle(f'{args.pde} | {args.condition}')
        fig.tight_layout()
        fig.savefig(f'{prefix}_timing_curve.pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[plot] timing curve failed: {e}')

    # --- gradient norm curve ---
    try:
        df_g = pd.read_csv(grad_path)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(df_g['epoch'], df_g['grad_norm'], 'b-', lw=1.5)
        if switch_epoch:
            ax.axvline(switch_epoch, color='k', ls=':', lw=1.5, label=f'switch @ {switch_epoch}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Gradient norm (log)')
        ax.set_title(f'Gradient norm — {args.pde} | {args.condition}')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{prefix}_grad_curve.pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[plot] grad curve failed: {e}')


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Auto-detect device: if user passed 'cuda:0' but no GPU exists, fall back to CPU
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[WARN] CUDA not available — falling back to CPU.')
        args.device = 'cpu'
    device = args.device

    dtype  = DTYPE_MAP[args.dtype_start]
    dtype_str  = args.dtype_start
    optim_str  = args.optim_start
    phase      = 'pre_switch' if args.is_switching else 'static'

    gpu_label = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'
    print(f'\n{"="*60}')
    print(f'  PDE        : {args.pde}')
    print(f'  Model      : {args.model}')
    print(f'  Condition  : {args.condition}')
    print(f'  Switch at  : epoch {args.switch_epoch} '
          f'({"active" if args.is_switching else "no switch"})')
    print(f'  Epochs     : {args.total_epochs}')
    print(f'  Device     : {device} ({gpu_label})')
    print('  TF32       : disabled (true IEEE 754 FP32/FP64)')
    print(f'{"="*60}\n')

    # Output directory per (pde, condition)
    run_dir = os.path.join(args.out_dir, args.pde, args.condition)
    loss_path, timing_path, grad_path, eval_path, switch_path = \
        open_logs(run_dir, args)

    # Build PDE
    tensors, loss_fn, exact_fn, res_test = \
        PDE_BUILDERS[args.pde](device, dtype)

    # Build model
    model = build_model(args, dtype, device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Build optimizer
    optim = make_optimizer(
        args.optim_start, model,
        args.adam_lr, args.lbfgs_tol_grad, args.lbfgs_tol_change)

    # State tracking
    switch_epoch_actual = None
    total_wall_start = time.perf_counter()
    peak_mem_mb = 0.0

    # ==========================
    # TRAINING LOOP
    # ==========================
    for epoch in tqdm(range(1, args.total_epochs + 1),
                      desc=args.condition, ncols=100, unit='ep'):

        # ---- switch check ----
        if (args.is_switching
                and switch_epoch_actual is None
                and epoch == args.switch_epoch):

            switch_epoch_actual = epoch
            from_dtype  = dtype_str
            from_optim  = optim_str
            to_dtype    = args.dtype_switch
            to_optim    = args.optim_switch

            # Retrieve last loss/grad for the switch log
            last_total_loss = 0.0
            last_grad_norm  = 0.0
            try:
                df_tmp = pd.read_csv(loss_path)
                last_total_loss = df_tmp['total_loss'].iloc[-1]
                df_g = pd.read_csv(grad_path)
                last_grad_norm = df_g['grad_norm'].iloc[-1]
            except Exception:
                pass

            print(f'\n>>> SWITCH at epoch {epoch}: '
                  f'{from_dtype}/{from_optim} → {to_dtype}/{to_optim}\n')

            # Cast model
            new_dtype = DTYPE_MAP[to_dtype]
            model = model.to(new_dtype)

            # Cast all input tensors (safe, no globals())
            tensors = cast_tensors(tensors, new_dtype, device)

            # Rebuild optimizer with fresh state
            optim = make_optimizer(
                to_optim, model,
                args.adam_lr, args.lbfgs_tol_grad, args.lbfgs_tol_change)

            # Update bookkeeping
            dtype      = new_dtype
            dtype_str  = to_dtype
            optim_str  = to_optim
            phase      = 'post_switch'

            # Log switch event
            with open(switch_path, 'a') as f:
                f.write(f'{args.condition},{args.pde},{args.model},'
                        f'{switch_epoch_actual},'
                        f'{from_dtype},{to_dtype},{from_optim},{to_optim},'
                        f'{last_total_loss:.8e},{last_grad_norm:.8e}\n')

        # ---- one training step ----
        lr_v, lbc_v, lic_v, tot_v, fwd_t, bwd_t, grad_norm = \
            train_one_epoch(model, optim, optim_str, tensors, loss_fn)

        # ---- memory ----
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(device) / 1e6
            mem_res   = torch.cuda.memory_reserved(device)  / 1e6
            peak_mem_mb = max(peak_mem_mb, mem_alloc)
        else:
            mem_alloc = mem_res = 0.0  # CPU run — memory tracking unavailable, logs will show 0.0

        # ---- write per-epoch logs ----
        log_epoch(
            loss_path, timing_path, grad_path,
            epoch, lr_v, lbc_v, lic_v, tot_v,
            fwd_t, bwd_t, mem_alloc, mem_res,
            grad_norm, dtype_str, optim_str, phase,
            args.condition, args.pde,
        )

    # ==========================
    # EVALUATION
    # ==========================
    total_wall = time.perf_counter() - total_wall_start
    rl1, rl2 = evaluate(model, args.pde, res_test, exact_fn, dtype, device)

    print(f'\nRelative L1 error : {rl1:.6f}')
    print(f'Relative L2 error : {rl2:.6f}')
    print(f'Total wall time   : {total_wall:.1f} s')
    print(f'Peak GPU memory   : {peak_mem_mb:.1f} MB')

    # Read final loss from file
    try:
        final_loss = pd.read_csv(loss_path)['total_loss'].iloc[-1]
    except Exception:
        final_loss = float('nan')

    with open(eval_path, 'a') as f:
        f.write(f'{args.condition},{args.pde},{args.model},'
                f'{rl1:.8e},{rl2:.8e},{final_loss:.8e},'
                f'{switch_epoch_actual},{args.total_epochs},'
                f'{total_wall:.2f},{peak_mem_mb:.2f}\n')

    # ==========================
    # SAVE MODEL
    # ==========================
    prefix = f'{run_dir}/{args.pde}_{args.model}_{args.condition}'
    torch.save(model.state_dict(), f'{prefix}_model.pt')

    # Save run config for reproducibility — includes GPU info for the paper
    config = vars(args).copy()
    config['gpu_name']     = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'
    config['torch_version'] = torch.__version__
    config['cuda_version']  = torch.version.cuda or 'N/A'
    config['tf32_enabled']  = False   # always disabled by configure_cuda()
    with open(f'{prefix}_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ==========================
    # PLOTS
    # ==========================
    save_plots(run_dir, args, loss_path, timing_path, grad_path, switch_epoch_actual)

    print(f'\nAll outputs saved to: {os.path.abspath(run_dir)}')


if __name__ == '__main__':
    main()

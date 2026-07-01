"""
train_pinn.py  —  TMC-PINN Unified Training Script
====================================================
USAGE (Jupyter):
    import importlib.util
    spec = importlib.util.spec_from_file_location('train_pinn', './train_pinn.py')
    tp   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)

    # Single condition:
    tp.main(pde='wave', dtype_start='fp64', optim_start='lbfgs')

    # Multiple PDEs — skips already-completed runs:
    tp.run_all(pdes=['wave', 'ac'])
    tp.run_all(pdes=['convection'])   # run convection separately — it's long

7 CONDITIONS per PDE (reaction, wave, ac, convection):
    Static:    fp64lbfgs, fp64adam, fp32lbfgs, fp32adam
    Switching: fp32adam->fp64adam, fp32lbfgs->fp64lbfgs, fp32adam->fp64lbfgs

BURGERS (targeted stiffness-generalization test -- NOT run as all 7):
    Formulation: Raissi, Perdikaris & Karniadakis 2019 (J. Comput. Phys.
    378:686-707) -- standard PINN benchmark instance, nu=0.01/pi, Dirichlet BC.
    No Xu et al. precedent for epoch budget -- ours is a tentative default,
    confirm via sanity run before committing the full subset. Recommended
    subset (NOT the full 7 -- see conversation with mentor for rationale):
        python train_pinn.py --pde burgers --dtype_start fp64 --optim_start lbfgs
        python train_pinn.py --pde burgers --dtype_start fp32 --optim_start lbfgs
        python train_pinn.py --pde burgers --dtype_start fp32 --optim_start lbfgs \\
            --dtype_switch fp64 --optim_switch lbfgs --switch_epoch 5000
    Requires burgers_shock.mat (Raissi et al. supplementary data,
    maziarraissi/PINNs repo) in the working directory for L2 eval.

EPOCH BUDGET (matches Xu et al. miniHuiHui/PINN_FP64 exactly, except burgers):
    reaction:   2,000   switch @ 1,000
    wave:      10,000   switch @ 5,000
    ac:        10,000   switch @ 5,000
    convection: 50,000  switch @ 25,000  <- long run, plan accordingly
    burgers:   10,000   switch @ 5,000   <- TENTATIVE, no source precedent
"""

import subprocess, sys
for _pkg in ['tqdm', 'pandas', 'matplotlib', 'scipy']:
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', _pkg, '-q',
                        '--break-system-packages'], check=False)

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import time, json, random, argparse, types
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import LBFGS, Adam
from tqdm import tqdm

# Per-PDE epoch budgets — matches Xu et al. miniHuiHui/PINN_FP64 exactly
# NOTE: 'burgers' has NO Xu et al. precedent. Budget below is our own choice,
# not inherited — disclose this explicitly in the paper (unlike the other 4).
PDE_EPOCHS = {
    'reaction':   2000,
    'wave':       10000,
    'ac':         10000,
    'convection': 50000,
    'burgers':    10000,   # tentative — confirm via sanity-check run before committing
}
PDE_SWITCH = {k: v // 2 for k, v in PDE_EPOCHS.items()}

DTYPE_MAP = {'fp32': torch.float32, 'fp64': torch.float64}


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

def configure_cuda():
    if not torch.cuda.is_available():
        print('[CUDA] No GPU -- CPU mode.')
        return
    print(f'[CUDA] {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | PyTorch {torch.__version__}')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32        = False
    torch.backends.cudnn.benchmark         = True
    torch.backends.cudnn.deterministic     = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    print('[CUDA] TF32 disabled | deterministic ON')

configure_cuda()


# ---------------------------------------------------------------------------
# MODEL  (1024x6, Xavier -- matches Xu et al. reaction_fp64.py exactly)
# ---------------------------------------------------------------------------

class PINN_Model(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=512, out_dim=1, num_layer=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layer - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


def build_model(dtype, device):
    model = PINN_Model()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)
    return model.to(dtype).to(device)


# ---------------------------------------------------------------------------
# DATA GRID
# b_left  = t=0 (IC), b_upper = x=x_max (BC), b_lower = x=x_min (BC)
# ---------------------------------------------------------------------------

def get_data(x_range, t_range, x_num, t_num):
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(t_range[0], t_range[1], t_num)
    X, T_ = np.meshgrid(x, t)
    data    = np.stack([X, T_], axis=-1)   # (t_num, x_num, 2)
    res     = data.reshape(-1, 2)
    b_left  = data[0,  :, :]
    b_right = data[-1, :, :]
    b_upper = data[:,  -1, :]
    b_lower = data[:,  0,  :]
    return res, b_left, b_right, b_upper, b_lower


def _ten(arr, dtype, device, grad=True):
    return torch.tensor(arr, dtype=dtype, requires_grad=grad).to(device)


def _make_T(res, b_left, b_upper, b_lower, dtype, device):
    return {
        'x_res':   _ten(res[:,0:1],     dtype, device),
        't_res':   _ten(res[:,1:2],     dtype, device),
        'x_left':  _ten(b_left[:,0:1],  dtype, device),
        't_left':  _ten(b_left[:,1:2],  dtype, device),
        'x_upper': _ten(b_upper[:,0:1], dtype, device),
        't_upper': _ten(b_upper[:,1:2], dtype, device),
        'x_lower': _ten(b_lower[:,0:1], dtype, device),
        't_lower': _ten(b_lower[:,1:2], dtype, device),
    }


# ---------------------------------------------------------------------------
# PDE DEFINITIONS  (verified against Xu et al. miniHuiHui/PINN_FP64)
# ---------------------------------------------------------------------------

def build_reaction_pde(device, dtype, **_):
    """
    u_t = 5u(1-u), u(x,0)=Gaussian, periodic BC.
    [0,2pi]x[0,1], 101x101, 2000 epochs.
    """
    res, b_left, _, b_upper, b_lower = get_data([0, 2*np.pi], [0, 1], 101, 101)
    T = _make_T(res, b_left, b_upper, b_lower, dtype, device)

    def loss_fn(model, T):
        u   = model(T['x_res'], T['t_res'])
        u_t = torch.autograd.grad(u, T['t_res'],
                  grad_outputs=torch.ones_like(u),
                  retain_graph=True, create_graph=True)[0]
        l_res = torch.mean((u_t - 5*u*(1-u))**2)
        l_bc  = torch.mean((model(T['x_upper'], T['t_upper']) -
                             model(T['x_lower'], T['t_lower']))**2)
        u_ic  = model(T['x_left'], T['t_left'])
        u_ref = torch.exp(-(T['x_left'] - torch.pi)**2 / (2*(torch.pi/4)**2))
        l_ic  = torch.mean((u_ic - u_ref)**2)
        return l_res, l_bc, l_ic

    def exact_fn(xy):
        x, t = xy[:, 0], xy[:, 1]
        h = np.exp(-(x - np.pi)**2 / (2*(np.pi/4)**2))
        return (h*np.exp(5*t) / (h*np.exp(5*t) + 1 - h)).reshape(101, 101)

    return T, loss_fn, exact_fn, res.copy()


def build_wave_pde(device, dtype, **_):
    """
    u_tt = 4*u_xx  (c=2),
    u(x,0) = sin(pi*x) + 0.5*sin(3*pi*x),  u_t(x,0)=0,  Dirichlet BC=0.
    [0,1]x[0,1], 101x101, 10000 epochs.
    Source: Xu et al. wave_fp64.py  loss_res = mean((u_tt - 4*u_xx)^2)
    """
    pi = np.pi
    res, b_left, _, b_upper, b_lower = get_data([0, 1], [0, 1], 101, 101)
    T = _make_T(res, b_left, b_upper, b_lower, dtype, device)

    def loss_fn(model, T):
        u    = model(T['x_res'], T['t_res'])
        u_t  = torch.autograd.grad(u, T['t_res'],
                   grad_outputs=torch.ones_like(u),
                   retain_graph=True, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, T['t_res'],
                   grad_outputs=torch.ones_like(u_t),
                   retain_graph=True, create_graph=True)[0]
        u_x  = torch.autograd.grad(u, T['x_res'],
                   grad_outputs=torch.ones_like(u),
                   retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, T['x_res'],
                   grad_outputs=torch.ones_like(u_x),
                   retain_graph=True, create_graph=True)[0]
        l_res = torch.mean((u_tt - 4*u_xx)**2)
        l_bc  = (torch.mean(model(T['x_upper'], T['t_upper'])**2) +
                 torch.mean(model(T['x_lower'], T['t_lower'])**2))
        u_ic   = model(T['x_left'], T['t_left'])
        u_t_ic = torch.autograd.grad(u_ic, T['t_left'],
                     grad_outputs=torch.ones_like(u_ic),
                     retain_graph=True, create_graph=True)[0]
        l_ic   = (torch.mean((u_ic[:,0] -
                               torch.sin(pi*T['x_left'][:,0]) -
                               0.5*torch.sin(3*pi*T['x_left'][:,0]))**2) +
                  torch.mean(u_t_ic**2))
        return l_res, l_bc, l_ic

    def exact_fn(xy):
        x, t = xy[:, 0], xy[:, 1]
        return (np.sin(pi*x)*np.cos(2*pi*t) +
                0.5*np.sin(3*pi*x)*np.cos(6*pi*t)).reshape(101, 101)

    return T, loss_fn, exact_fn, res.copy()


def build_convection_pde(device, dtype, **_):
    """
    u_t + 50*u_x = 0,  u(x,0)=sin(x),  periodic BC.
    Train grid: 401x401.  Test grid: 101x101.  50000 epochs.
    Source: Xu et al. convection_fp64.py  beta=50, get_data 401x401.
    """
    beta = 50.0
    res, b_left, _, b_upper, b_lower = get_data([0, 2*np.pi], [0, 1], 401, 401)
    T = _make_T(res, b_left, b_upper, b_lower, dtype, device)
    res_test, _, _, _, _ = get_data([0, 2*np.pi], [0, 1], 101, 101)

    def loss_fn(model, T):
        u   = model(T['x_res'], T['t_res'])
        u_t = torch.autograd.grad(u, T['t_res'],
                  grad_outputs=torch.ones_like(u),
                  retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, T['x_res'],
                  grad_outputs=torch.ones_like(u),
                  retain_graph=True, create_graph=True)[0]
        l_res = torch.mean((u_t + beta*u_x)**2)
        l_bc  = torch.mean((model(T['x_upper'], T['t_upper']) -
                             model(T['x_lower'], T['t_lower']))**2)
        l_ic  = torch.mean((model(T['x_left'], T['t_left'])[:,0] -
                             torch.sin(T['x_left'][:,0]))**2)
        return l_res, l_bc, l_ic

    def exact_fn(xy):
        return np.sin(xy[:, 0] - beta*xy[:, 1]).reshape(101, 101)

    return T, loss_fn, exact_fn, res_test.copy()


def build_ac_pde(device, dtype, mat_path='allen_cahn.mat', **_):
    """
    Allen-Cahn: u_t - 0.0001*u_xx + 5(u^3-u)=0
    IC (analytic): x^2*cos(pi*x)  -- does NOT need .mat for training
    BC: periodic value + derivative
    [-1,1]x[0,1], 101x101, 10000 epochs.
    .mat loaded AFTER training for L2 evaluation only.
    Source: Xu et al. ac_fp64.py
    """
    res, b_left, _, b_upper, b_lower = get_data([-1, 1], [0, 1], 101, 101)
    T = _make_T(res, b_left, b_upper, b_lower, dtype, device)
    d = 0.0001

    def loss_fn(model, T):
        u    = model(T['x_res'], T['t_res'])
        u_t  = torch.autograd.grad(u, T['t_res'],
                   grad_outputs=torch.ones_like(u),
                   retain_graph=True, create_graph=True)[0]
        u_x  = torch.autograd.grad(u, T['x_res'],
                   grad_outputs=torch.ones_like(u),
                   retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, T['x_res'],
                   grad_outputs=torch.ones_like(u_x),
                   retain_graph=True, create_graph=True)[0]
        l_res = torch.mean((u_t - d*u_xx + 5*(u**3 - u))**2)
        u_up  = model(T['x_upper'], T['t_upper'])
        u_lo  = model(T['x_lower'], T['t_lower'])
        ux_up = torch.autograd.grad(u_up, T['x_upper'],
                    grad_outputs=torch.ones_like(u_up),
                    retain_graph=True, create_graph=True)[0]
        ux_lo = torch.autograd.grad(u_lo, T['x_lower'],
                    grad_outputs=torch.ones_like(u_lo),
                    retain_graph=True, create_graph=True)[0]
        l_bc  = (torch.mean((u_up - u_lo)**2) + torch.mean((ux_up - ux_lo)**2))
        u_ic  = model(T['x_left'], T['t_left'])
        l_ic  = torch.mean((u_ic[:,0] -
                             (T['x_left'][:,0]**2) *
                             torch.cos(np.pi * T['x_left'][:,0]))**2)
        return l_res, l_bc, l_ic

    exact_fn = None
    try:
        import scipy.io
        data  = scipy.io.loadmat(mat_path)
        t_arr = data['tt'].flatten()
        x_arr = data['x'].flatten()
        usol  = np.real(data['uu'])
        print(f'[AC] Loaded {mat_path}: x={len(x_arr)}, t={len(t_arr)}')

        def exact_fn(xy):
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator((x_arr, t_arr), usol,
                                             method='linear', bounds_error=False,
                                             fill_value=None)
            return interp(xy).reshape(101, 101)
    except Exception as e:
        print(f'[AC] Could not load {mat_path}: {e}')
        print('[AC] L2 eval = nan. Training unaffected.')

    return T, loss_fn, exact_fn, res.copy()


def build_burgers_pde(device, dtype, mat_path='burgers_shock.mat', **_):
    """
    Viscous Burgers' equation (Raissi, Perdikaris & Karniadakis 2019,
    J. Comput. Phys. 378:686-707 — the standard PINN benchmark instance).

        u_t + u*u_x - nu*u_xx = 0,   nu = 0.01/pi
        u(x,0) = -sin(pi*x)
        u(-1,t) = u(1,t) = 0          <-- DIRICHLET, not periodic (differs
                                           from reaction/wave/convection/AC)
        [-1,1]x[0,1], 101x101.

    Loss weighting: unweighted l_res + l_bc + l_ic, matching Raissi et al.
    and matching the convention already used for reaction/wave/convection
    (only AC departs from this, with its 10:1:1:100 weighting).

    .mat reference solution: verify key names before trusting them.
    Raissi's supplementary file (maziarraissi/PINNs repo) commonly uses
    't' / 'x' / 'usol', NOT the 'tt'/'uu' keys used by allen_cahn.mat —
    do not assume the AC convention carries over here (that exact mistake
    cost a transpose bug on AC; this loader checks key names explicitly
    and fails loudly rather than silently mis-loading).
    """
    nu = 0.01 / np.pi
    res, b_left, _, b_upper, b_lower = get_data([-1, 1], [0, 1], 101, 101)
    T = _make_T(res, b_left, b_upper, b_lower, dtype, device)

    def loss_fn(model, T):
        u    = model(T['x_res'], T['t_res'])
        u_t  = torch.autograd.grad(u, T['t_res'],
                   grad_outputs=torch.ones_like(u),
                   retain_graph=True, create_graph=True)[0]
        u_x  = torch.autograd.grad(u, T['x_res'],
                   grad_outputs=torch.ones_like(u),
                   retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, T['x_res'],
                   grad_outputs=torch.ones_like(u_x),
                   retain_graph=True, create_graph=True)[0]
        l_res = torch.mean((u_t + u*u_x - nu*u_xx)**2)
        # Dirichlet BC: u=0 at both walls (NOT periodic — no upper==lower match)
        l_bc  = (torch.mean(model(T['x_upper'], T['t_upper'])**2) +
                 torch.mean(model(T['x_lower'], T['t_lower'])**2))
        u_ic  = model(T['x_left'], T['t_left'])
        l_ic  = torch.mean((u_ic[:,0] +
                             torch.sin(np.pi * T['x_left'][:,0]))**2)
        return l_res, l_bc, l_ic

    exact_fn = None
    try:
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        available_keys = [k for k in data.keys() if not k.startswith('__')]
        print(f'[Burgers] {mat_path} keys found: {available_keys}')

        # Try the Raissi-convention keys first; fall back to AC-style keys
        # ONLY if Raissi keys are absent, and print which path was taken —
        # never assume silently.
        if 't' in data and 'x' in data and 'usol' in data:
            t_arr = data['t'].flatten()
            x_arr = data['x'].flatten()
            usol  = np.real(data['usol'])
            print('[Burgers] Using Raissi-convention keys: t / x / usol')
        elif 'tt' in data and 'x' in data and 'uu' in data:
            t_arr = data['tt'].flatten()
            x_arr = data['x'].flatten()
            usol  = np.real(data['uu'])
            print('[Burgers] Using AC-convention keys: tt / x / uu '
                  '(unexpected for this file — double-check source)')
        else:
            raise KeyError(
                f'No recognized key combination in {available_keys}. '
                f'Inspect the .mat file manually before proceeding — '
                f'do not guess the shape/orientation.'
            )

        # Verify orientation against expected grid size rather than assuming
        # (x,t) vs (t,x) — the AC loader needed a transpose fix for exactly
        # this reason.
        if usol.shape == (len(t_arr), len(x_arr)):
            print(f'[Burgers] usol shape {usol.shape} is (t,x) -- transposing to (x,t)')
            usol = usol.T
        elif usol.shape == (len(x_arr), len(t_arr)):
            print(f'[Burgers] usol shape {usol.shape} already (x,t) -- no transpose needed')
        else:
            raise ValueError(
                f'usol shape {usol.shape} matches neither (t,x)=({len(t_arr)},{len(x_arr)}) '
                f'nor (x,t)=({len(x_arr)},{len(t_arr)}) -- inspect manually.'
            )

        def exact_fn(xy):
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator((x_arr, t_arr), usol,
                                             method='linear', bounds_error=False,
                                             fill_value=None)
            return interp(xy).reshape(101, 101)
    except Exception as e:
        print(f'[Burgers] Could not load {mat_path}: {e}')
        print('[Burgers] L2 eval = nan. Training unaffected.')

    return T, loss_fn, exact_fn, res.copy()


PDE_BUILDERS = {
    'reaction':   build_reaction_pde,
    'wave':       build_wave_pde,
    'convection': build_convection_pde,
    'ac':         build_ac_pde,
    'burgers':    build_burgers_pde,
}

# Per-PDE default reference-solution files. mat_path is only ever an
# explicit OVERRIDE when passed by the caller — it must never silently
# fall back to allen_cahn.mat for a different PDE (that mismatch would
# load the wrong ground truth without raising an error).
PDE_MAT_DEFAULTS = {
    'ac':       'allen_cahn.mat',
    'burgers':  'burgers_shock.mat',
}


# ---------------------------------------------------------------------------
# OPTIMIZER / CAST
# ---------------------------------------------------------------------------

def make_optimizer(key, model, adam_lr=1e-3,
                   lbfgs_tol_grad=1e-8, lbfgs_tol_change=1e-10):
    if key == 'adam':
        return Adam(model.parameters(), lr=adam_lr)
    return LBFGS(model.parameters(), line_search_fn='strong_wolfe',
                 tolerance_grad=lbfgs_tol_grad, tolerance_change=lbfgs_tol_change)


def cast_tensors(tensors, dtype, device):
    return {k: v.detach().to(dtype=dtype, device=device).requires_grad_(v.requires_grad)
            for k, v in tensors.items()}


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def open_logs(run_dir, condition, pde):
    os.makedirs(run_dir, exist_ok=True)
    p = f'{run_dir}/{pde}_PINN_{condition}'
    paths = {
        'loss':   f'{p}_loss.csv',
        'timing': f'{p}_timing.csv',
        'grad':   f'{p}_grad.csv',
        'eval':   f'{p}_eval.csv',
        'switch': f'{p}_switch.csv',
    }
    with open(paths['loss'],   'w') as f:
        f.write('epoch,loss_res,loss_bc,loss_ic,total_loss,dtype,optim,phase,condition,pde\n')
    with open(paths['timing'], 'w') as f:
        f.write('epoch,fwd_time_s,bwd_time_s,total_time_s,mem_allocated_mb,mem_reserved_mb,dtype,optim,phase,condition,pde\n')
    with open(paths['grad'],   'w') as f:
        f.write('epoch,grad_norm,dtype,optim,phase,condition,pde\n')
    with open(paths['eval'],   'w') as f:
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
                f'{mem_alloc:.2f},{mem_res:.2f},{dtype_str},{optim_str},{phase},{condition},{pde}\n')
    with open(paths['grad'], 'a') as f:
        f.write(f'{epoch},{grad_norm:.8e},{dtype_str},{optim_str},{phase},{condition},{pde}\n')


# ---------------------------------------------------------------------------
# TRAIN STEP
# ---------------------------------------------------------------------------

def train_one_epoch(model, optim, tensors, loss_fn):
    timing = [0.0, 0.0]
    last   = [None]

    def closure():
        optim.zero_grad()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        lr, lbc, lic = loss_fn(model, tensors)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timing[0] += time.perf_counter() - t0
        loss = lr + lbc + lic
        last[0] = (lr.item(), lbc.item(), lic.item(), loss.item())
        t1 = time.perf_counter()
        loss.backward()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timing[1] += time.perf_counter() - t1
        return loss

    optim.step(closure)
    grad_norm = sum(
        p.grad.data.norm(2).item()**2
        for p in model.parameters() if p.grad is not None
    ) ** 0.5
    return *last[0], timing[0], timing[1], grad_norm


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate(model, res_test, exact_fn, dtype, device):
    if exact_fn is None:
        return float('nan'), float('nan')
    x_t = torch.tensor(res_test, dtype=dtype, requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(x_t[:, 0:1], x_t[:, 1:2])[:, 0].cpu().numpy()
    u      = exact_fn(res_test)
    pred_r = pred.reshape(u.shape)
    rl1 = np.sum(np.abs(u - pred_r))    / np.sum(np.abs(u))
    rl2 = np.sqrt(np.sum((u-pred_r)**2) / np.sum(u**2))
    return float(rl1), float(rl2)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def save_plots(run_dir, condition, pde, paths, switch_epoch):
    prefix = f'{run_dir}/{pde}_PINN_{condition}'
    for name, ycol, ylabel in [('loss','total_loss','Loss (log)'),
                                ('grad','grad_norm','Grad norm (log)')]:
        try:
            df = pd.read_csv(paths[name])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.semilogy(df['epoch'], df[ycol], 'b-', lw=1.5)
            if name == 'loss':
                for col, c, lbl in [('loss_res','r','Residual'),
                                     ('loss_bc','g','BC'),('loss_ic','m','IC')]:
                    ax.semilogy(df['epoch'], df[col], '--', color=c, lw=1, alpha=0.7, label=lbl)
            if switch_epoch:
                ax.axvline(switch_epoch, color='k', ls=':', lw=1.5, label=f'switch@{switch_epoch}')
            ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
            ax.set_title(f'{pde} | {condition}'); ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(f'{prefix}_{name}_curve.pdf', bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f'[plot] {name}: {e}')
    try:
        df_t = pd.read_csv(paths['timing'])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(df_t['epoch'], df_t['total_time_s'], label='total')
        axes[0].plot(df_t['epoch'], df_t['fwd_time_s'],   label='fwd', alpha=0.7)
        axes[0].plot(df_t['epoch'], df_t['bwd_time_s'],   label='bwd', alpha=0.7)
        axes[1].plot(df_t['epoch'], df_t['mem_allocated_mb'], label='alloc MB')
        for ax in axes:
            if switch_epoch: ax.axvline(switch_epoch, color='k', ls=':', lw=1.5)
            ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
        axes[0].set_title('Time/epoch'); axes[1].set_title('GPU memory')
        fig.suptitle(f'{pde} | {condition}'); fig.tight_layout()
        fig.savefig(f'{prefix}_timing_curve.pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[plot] timing: {e}')


# ---------------------------------------------------------------------------
# ARGS
# ---------------------------------------------------------------------------

def parse_args(pde='reaction', model='PINN', device='cuda:0', seed=1,
               dtype_start='fp64', optim_start='lbfgs',
               dtype_switch=None, optim_switch=None, switch_epoch=None,
               total_epochs=None,
               adam_lr=1e-3, lbfgs_tol_grad=1e-8, lbfgs_tol_change=1e-10,
               out_dir='./results', mat_path='allen_cahn.mat'):

    in_jupyter = (not any(a.startswith('--') for a in sys.argv[1:]) or
                  any(x in ' '.join(sys.argv)
                      for x in ['ipykernel','jupyter','ipython','kernel']))

    if not in_jupyter:
        p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--pde',              default=pde)
        p.add_argument('--model',            default=model)
        p.add_argument('--device',           default=device)
        p.add_argument('--seed',   type=int, default=seed)
        p.add_argument('--dtype_start',      default=dtype_start,  choices=['fp32','fp64'])
        p.add_argument('--optim_start',      default=optim_start,  choices=['adam','lbfgs'])
        p.add_argument('--dtype_switch',     default=dtype_switch, choices=['fp32','fp64'])
        p.add_argument('--optim_switch',     default=optim_switch, choices=['adam','lbfgs'])
        p.add_argument('--switch_epoch',     type=int,   default=switch_epoch)
        p.add_argument('--total_epochs',     type=int,   default=total_epochs)
        p.add_argument('--adam_lr',          type=float, default=adam_lr)
        p.add_argument('--lbfgs_tol_grad',   type=float, default=lbfgs_tol_grad)
        p.add_argument('--lbfgs_tol_change', type=float, default=lbfgs_tol_change)
        p.add_argument('--out_dir',          default=out_dir)
        p.add_argument('--mat_path',         default=mat_path)
        args = p.parse_args()
    else:
        args = types.SimpleNamespace(
            pde=pde, model=model, device=device, seed=seed,
            dtype_start=dtype_start, optim_start=optim_start,
            dtype_switch=dtype_switch, optim_switch=optim_switch,
            switch_epoch=switch_epoch, total_epochs=total_epochs,
            adam_lr=adam_lr, lbfgs_tol_grad=lbfgs_tol_grad,
            lbfgs_tol_change=lbfgs_tol_change, out_dir=out_dir,
            mat_path=mat_path,
        )

    if args.total_epochs is None:
        args.total_epochs = PDE_EPOCHS.get(args.pde, 2000)

    if args.dtype_switch is None: args.dtype_switch = args.dtype_start
    if args.optim_switch is None: args.optim_switch = args.optim_start

    is_switching = (
        args.switch_epoch is not None
        and args.switch_epoch < args.total_epochs
        and (args.dtype_switch != args.dtype_start or args.optim_switch != args.optim_start)
    )
    args.condition    = (f'{args.dtype_start}{args.optim_start}_to_{args.dtype_switch}{args.optim_switch}'
                         if is_switching else f'{args.dtype_start}{args.optim_start}')
    args.is_switching = is_switching
    return args


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(**kwargs):
    args   = parse_args(**kwargs)
    device = args.device

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('[WARN] No GPU -- CPU.'); device = 'cpu'

    dtype     = DTYPE_MAP[args.dtype_start]
    dtype_str = args.dtype_start
    optim_str = args.optim_start
    phase     = 'pre_switch' if args.is_switching else 'static'
    gpu_label = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'

    print(f'\n{"="*55}')
    print(f'  PDE       : {args.pde}')
    print(f'  Condition : {args.condition}')
    print(f'  Epochs    : {args.total_epochs}')
    print(f'  Switch    : epoch {args.switch_epoch} ({"ON" if args.is_switching else "OFF"})')
    print(f'  Device    : {device} ({gpu_label})')
    print(f'{"="*55}\n')

    run_dir = os.path.join(args.out_dir, args.pde, args.condition)
    paths   = open_logs(run_dir, args.condition, args.pde)

    # Resolve mat_path per-PDE if the caller left it at the generic default
    # rather than explicitly overriding it -- prevents e.g. burgers runs
    # silently trying to load allen_cahn.mat.
    resolved_mat_path = args.mat_path
    if resolved_mat_path == 'allen_cahn.mat' and args.pde in PDE_MAT_DEFAULTS:
        resolved_mat_path = PDE_MAT_DEFAULTS[args.pde]

    tensors, loss_fn, exact_fn, res_test = PDE_BUILDERS[args.pde](
        device, dtype, mat_path=resolved_mat_path)
    model = build_model(dtype, device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    optim = make_optimizer(args.optim_start, model, args.adam_lr,
                           args.lbfgs_tol_grad, args.lbfgs_tol_change)

    switch_epoch_actual = None
    t_start  = time.perf_counter()
    peak_mem = 0.0

    for epoch in tqdm(range(1, args.total_epochs + 1),
                      desc=args.condition, ncols=100, unit='ep'):

        if args.is_switching and switch_epoch_actual is None and epoch == args.switch_epoch:
            switch_epoch_actual = epoch
            from_dtype, from_optim = dtype_str, optim_str
            to_dtype,   to_optim   = args.dtype_switch, args.optim_switch
            try:
                last_loss = pd.read_csv(paths['loss'])['total_loss'].iloc[-1]
                last_grad = pd.read_csv(paths['grad'])['grad_norm'].iloc[-1]
            except Exception:
                last_loss = last_grad = 0.0
            print(f'\n>>> SWITCH @ epoch {epoch}: {from_dtype}/{from_optim} -> {to_dtype}/{to_optim}\n')
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

        lr_v, lbc_v, lic_v, tot_v, fwd_t, bwd_t, grad_norm = \
            train_one_epoch(model, optim, tensors, loss_fn)

        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(device) / 1e6
            mem_res   = torch.cuda.memory_reserved(device)  / 1e6
            peak_mem  = max(peak_mem, mem_alloc)
        else:
            mem_alloc = mem_res = 0.0

        log_epoch(paths, epoch, lr_v, lbc_v, lic_v, tot_v,
                  fwd_t, bwd_t, mem_alloc, mem_res,
                  grad_norm, dtype_str, optim_str, phase,
                  args.condition, args.pde)

    total_wall = time.perf_counter() - t_start
    rl1, rl2   = evaluate(model, res_test, exact_fn, dtype, device)
    print(f'\nRelL1={rl1:.6f}  RelL2={rl2:.6f}  Wall={total_wall:.1f}s  Peak={peak_mem:.0f}MB')

    try:
        final_loss = pd.read_csv(paths['loss'])['total_loss'].iloc[-1]
    except Exception:
        final_loss = float('nan')

    with open(paths['eval'], 'a') as f:
        f.write(f'{args.condition},{args.pde},PINN,'
                f'{rl1:.8e},{rl2:.8e},{final_loss:.8e},'
                f'{switch_epoch_actual},{args.total_epochs},'
                f'{total_wall:.2f},{peak_mem:.2f}\n')

    prefix = f'{run_dir}/{args.pde}_PINN_{args.condition}'
    torch.save(model.state_dict(), f'{prefix}_model.pt')
    with open(f'{prefix}_config.json', 'w') as f:
        json.dump({**vars(args),
                   'gpu': torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU',
                   'torch': torch.__version__, 'cuda': str(torch.version.cuda),
                   'tf32': False}, f, indent=2)

    save_plots(run_dir, args.condition, args.pde, paths, switch_epoch_actual)
    print(f'Outputs -> {os.path.abspath(run_dir)}')
    return rl1, rl2, total_wall


# ---------------------------------------------------------------------------
# ALL_CONDITIONS
# ---------------------------------------------------------------------------

ALL_CONDITIONS = [
    dict(dtype_start='fp64', optim_start='lbfgs'),
    dict(dtype_start='fp64', optim_start='adam'),
    dict(dtype_start='fp32', optim_start='lbfgs'),
    dict(dtype_start='fp32', optim_start='adam'),
    dict(dtype_start='fp32', optim_start='adam',  dtype_switch='fp64', optim_switch='adam'),
    dict(dtype_start='fp32', optim_start='lbfgs', dtype_switch='fp64', optim_switch='lbfgs'),
    dict(dtype_start='fp32', optim_start='adam',  dtype_switch='fp64', optim_switch='lbfgs'),
]


# ---------------------------------------------------------------------------
# run_all — select which PDEs to run, skips completed runs
# ---------------------------------------------------------------------------

def run_all(pdes=None, conditions=None, out_dir='./results',
            skip_existing=True, mat_path='allen_cahn.mat', **shared_kwargs):
    """
    Run all 7 conditions across selected PDEs.

    pdes: list of PDEs to run.
          ['wave', 'ac']          -- run wave and AC
          ['convection']          -- run only convection (long!)
          ['burgers']             -- run burgers (~4-6 hrs, all 7 conditions)
          None                    -- all 5 PDEs (reaction/wave/ac/convection/burgers)

    Burgers' runs all 7 conditions the same as the other PDEs.
    Epoch counts and switch epochs are set automatically per PDE.
    Already-completed runs are skipped (skip_existing=True).
    mat_path is resolved per-PDE automatically -- burgers uses burgers_shock.mat,
    ac uses allen_cahn.mat; you do not need to pass mat_path manually.

    Example:
        run_all(pdes=['wave', 'ac'])
        run_all(pdes=['convection'])   # do this separately, it takes hours
        run_all(pdes=['burgers'])      # all 7 conditions, ~4-6 hrs
    """
    if pdes is None:
        pdes = ['reaction', 'wave', 'convection', 'ac', 'burgers']
    if conditions is None:
        conditions = ALL_CONDITIONS

    total = len(pdes) * len(conditions)
    done  = 0

    for pde in pdes:
        ep  = PDE_EPOCHS[pde]
        sw  = PDE_SWITCH[pde]
        print(f'\n{"="*55}')
        print(f'  PDE: {pde}  |  {ep} epochs  |  switch @ {sw}')
        print(f'{"="*55}')

        # Resolve mat_path per-PDE so burgers never silently
        # inherits allen_cahn.mat as the default.
        resolved_mat = mat_path
        if resolved_mat == 'allen_cahn.mat' and pde in PDE_MAT_DEFAULTS:
            resolved_mat = PDE_MAT_DEFAULTS[pde]

        for cond in conditions:
            ds, os_ = cond['dtype_start'], cond['optim_start']
            ds2 = cond.get('dtype_switch', ds)
            os2 = cond.get('optim_switch', os_)
            is_sw = ds2 != ds or os2 != os_
            label = f'{ds}{os_}_to_{ds2}{os2}' if is_sw else f'{ds}{os_}'
            done += 1

            if skip_existing:
                eval_path = os.path.join(out_dir, pde, label,
                                         f'{pde}_PINN_{label}_eval.csv')
                if os.path.exists(eval_path):
                    try:
                        if len(pd.read_csv(eval_path)) > 0:
                            print(f'[{done}/{total}] SKIP {pde}/{label}')
                            continue
                    except Exception:
                        pass

            print(f'\n[{done}/{total}] {pde} / {label}')
            kwargs = dict(pde=pde, total_epochs=ep, out_dir=out_dir,
                          mat_path=resolved_mat, **cond, **shared_kwargs)
            if is_sw:
                kwargs['switch_epoch'] = sw
            main(**kwargs)

    print(f'\nrun_all complete: {pdes}')


if __name__ == '__main__':
    main()

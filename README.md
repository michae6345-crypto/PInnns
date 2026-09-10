# Precision and optimizer switching in physics-informed neural networks

Training a PINN in FP64 buys accuracy but costs time and memory; training in
FP32 is cheap but stalls at a much higher error floor. This repository holds the
code for a study of what happens if you do not have to choose — start a run in
FP32, then switch precision, optimizer, or both partway through, and see where
the resulting error and wall time land relative to the static baselines.

Seven conditions are compared per PDE:

| | Phase 1 | Phase 2 |
|---|---|---|
| **Static baselines** | | |
| `fp64lbfgs` | FP64 L-BFGS | — |
| `fp64adam` | FP64 Adam | — |
| `fp32lbfgs` | FP32 L-BFGS | — |
| `fp32adam` | FP32 Adam | — |
| **Switching** | | |
| `fp32adam_to_fp64adam` | FP32 Adam | FP64 Adam |
| `fp32lbfgs_to_fp64lbfgs` | FP32 L-BFGS | FP64 L-BFGS |
| `fp32adam_to_fp64lbfgs` | FP32 Adam | FP64 L-BFGS |

An eighth condition, `fp64adam_to_fp64lbfgs`, isolates the optimizer switch by
holding precision at FP64 throughout. It lives in its own notebook.

The static baselines and the epoch budgets follow Xu et al., *FP64 Is All You
Need* (2025), whose reference implementation is at `miniHuiHui/PINN_FP64`.

---

## Layout

```
train_pinn.py                                  training script, all conditions
aggregate_results.py                           builds the paper tables and figures
run_experiments.ipynb                          all seven conditions, per PDE
run_experiments_fp64adam_to_fp64lbfgs.ipynb    the FP64 optimizer-switch ablation
util.py, model_dict.py, models/                inherited from the baseline repo
allen_cahn.mat, burgers_shock.mat              reference solutions for L2 evaluation
results/                                       created on first run
```

Each run writes to `results/<pde>/<condition>/` — five CSVs (loss, timing,
gradient norm, final evaluation, switch event), three PDF curves, the model
weights, and a JSON config recording every argument used.

## Setup

```bash
pip install torch numpy matplotlib tqdm pandas scipy
```

Python 3.9+ and PyTorch 2.0+. A GPU is strongly recommended: L-BFGS evaluates
its closure several times per step under the strong Wolfe line search, so an
L-BFGS run takes roughly 3-10x as long as an Adam run at the same epoch count.

## Running

Every condition goes through `train_pinn.py`, so hyperparameters and logging are
identical across the board. A static baseline just omits the switch flags:

```bash
python train_pinn.py --pde reaction --dtype_start fp64 --optim_start lbfgs
```

Passing `--dtype_switch`, `--optim_switch` and `--switch_epoch` together turns a
run into a switching run:

```bash
python train_pinn.py \
  --pde reaction \
  --dtype_start fp32 --optim_start adam \
  --dtype_switch fp64 --optim_switch lbfgs \
  --switch_epoch 1000
```

From Python, `run_all` sweeps conditions across PDEs and picks the epoch and
switch budgets itself. It skips any condition that already has results, so an
interrupted sweep resumes by re-running the same call:

```python
tp.run_all(pdes=['reaction'], out_dir='./results', skip_existing=True)
```

The two notebooks wrap this for the full sweep and for the FP64 ablation. Set
`REPO_URL` in the environment first if you are running them on a fresh machine.

### PDEs and budgets

| `--pde` | Equation | Epochs | Switch at |
|---|---|---|---|
| `reaction` | u_t = 5u(1-u), periodic | 2,000 | 1,000 |
| `wave` | u_tt = 4u_xx, Dirichlet | 10,000 | 5,000 |
| `ac` | Allen-Cahn, unweighted loss | 10,000 | 5,000 |
| `ac_weighted` | Allen-Cahn, 10:1:100 loss weighting | 10,000 | 5,000 |
| `convection` | u_t + 50u_x = 0, periodic, 401x401 grid | 50,000 | 25,000 |
| `burgers` | u_t + uu_x = (0.01/pi)u_xx, Dirichlet | 10,000 | 5,000 |

Start with `reaction`: it is the fastest to converge and the most exercised.
`convection` is by far the most expensive — a 401x401 training grid for 50,000
epochs — so leave it until last if compute is tight. The budgets for the first
five come from Xu et al.; `burgers` has no counterpart there, so its budget is
our own choice and should be reported as such.

`ac` and `ac_weighted` differ only in loss weighting and write to separate
directories, so neither overwrites the other.

## Options

| Flag | Default | Description |
|---|---|---|
| `--pde` | `reaction` | Which PDE to solve |
| `--device` | `cuda:0` | PyTorch device string; falls back to CPU if unavailable |
| `--seed` | `1` | Random seed |
| `--dtype_start` | `fp32` | Phase 1 dtype: `fp32` or `fp64` |
| `--optim_start` | `lbfgs` | Phase 1 optimizer: `adam` or `lbfgs` |
| `--dtype_switch` | *(none)* | Phase 2 dtype; omit for a static run |
| `--optim_switch` | *(none)* | Phase 2 optimizer; omit for a static run |
| `--switch_epoch` | *(none)* | Epoch at which to switch, 1-indexed |
| `--total_epochs` | per-PDE | Defaults to the budget in the table above |
| `--adam_lr` | `1e-3` | Adam learning rate |
| `--lbfgs_tol_grad` | `1e-8` | L-BFGS gradient tolerance |
| `--lbfgs_tol_change` | `1e-10` | L-BFGS step change tolerance |
| `--mat_path` | per-PDE | Reference solution; resolved per PDE unless overridden |
| `--out_dir` | `./results` | Root output directory |

A switch only fires when `--switch_epoch` is set, is less than `--total_epochs`,
and at least one of dtype or optimizer actually changes. Otherwise the run is
recorded as static.

## Log schema

`*_loss.csv` — per epoch
```
epoch, loss_res, loss_bc, loss_ic, total_loss, dtype, optim, phase, condition, pde
```

`*_timing.csv` — per epoch
```
epoch, fwd_time_s, bwd_time_s, total_time_s, mem_allocated_mb, mem_reserved_mb, dtype, optim, phase, condition, pde
```

`*_grad.csv` — per epoch
```
epoch, grad_norm, dtype, optim, phase, condition, pde
```

`*_eval.csv` — one row per run
```
condition, pde, model, L1_rel, L2_rel, final_loss, switch_epoch, total_epochs, total_time_s, peak_mem_mb
```

`*_switch.csv` — one row, switching runs only
```
condition, pde, model, switch_epoch, from_dtype, to_dtype, from_optim, to_optim, loss_at_switch, grad_norm_at_switch
```

`phase` is `static` for baselines and `pre_switch` / `post_switch` either side of
a switch; it is what places the vertical switch line on the plots.

## Figures

```bash
python aggregate_results.py --results_dir ./results --out_dir ./paper_figures
```

This writes `table1_summary.csv` and `.tex`, per-PDE loss curves
(`fig_loss_<pde>.pdf`), and bar charts of relative L2 error and training time.
Conditions with no results appear as em-dashes, so a partial sweep still
produces a readable table.

## Reproducing a run

Every run saves its full argument list:

```bash
cat results/reaction/fp32adam_to_fp64lbfgs/reaction_PINN_fp32adam_to_fp64lbfgs_config.json
```

Re-running with those arguments reproduces the run. TF32 is disabled and
deterministic kernels are requested at import time, so FP32 results are not
silently computed in reduced precision.

## Limitations

- One seed per condition. Differences between nearby conditions should be read
  with that in mind.
- Allen-Cahn and Burgers report `nan` for L2 if their `.mat` file is missing.
  Training is unaffected; only the evaluation is.
- GPU memory columns are `0.0` on CPU runs.
- `--model` is accepted but ignored: every run uses the same 512-wide, 4-layer
  tanh network defined in `train_pinn.py`.

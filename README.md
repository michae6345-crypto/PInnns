# TMC-PINN: Dtype and Optimizer Switching in Physics-Informed Neural Networks

This repository accompanies the paper:

> **Training Physics-Informed Neural Networks (PINNs) often involves trade-offs between numerical precision and computational efficiency.**  
> We study the effects of switching both optimizer type and floating-point precision mid-training across 7 experimental conditions and multiple benchmark PDEs.

Mentor: Shivank  
Authors: Michael, Chloe, Trisha  
Baseline: [FP64 Is All You Need](https://github.com/XXX) (Xu et al., 2025)

---

## Repository Structure

```
tmc-pinn/
├── train_pinn.py          # unified training script — all 7 conditions
├── run_all.sh             # launches all experiments
├── aggregate_results.py   # builds paper tables and figures from logs
├── util.py                # data generation utilities (from baseline)
├── model_dict.py          # model registry (from baseline)
├── allen_cahn.mat         # Allen-Cahn reference data (from baseline)
└── results/               # created automatically on first run
    └── <pde>/
        └── <condition>/
            ├── *_loss.csv
            ├── *_timing.csv
            ├── *_grad.csv
            ├── *_eval.csv
            ├── *_switch.csv
            ├── *_loss_curve.pdf
            ├── *_timing_curve.pdf
            ├── *_grad_curve.pdf
            ├── *_model.pt
            └── *_config.json
```

---

## Setup

### Requirements

```bash
pip install torch numpy matplotlib tqdm pandas scipy
```

Python 3.9+ and PyTorch 2.0+ recommended. For GPU runs (strongly recommended for L-BFGS), CUDA 11.8+.

### Clone and install

```bash
git clone https://github.com/chlorwu/tmc-pinn.git
cd tmc-pinn
pip install torch numpy matplotlib tqdm pandas scipy
```

`util.py` and `model_dict.py` must be in the same directory as `train_pinn.py`, or on your `PYTHONPATH`. They are inherited from the baseline repo.

---

## Running Experiments

All 7 conditions are run through the **same script** (`train_pinn.py`) with different CLI flags. This ensures identical hyperparameters and logging across every condition.

### Quick smoke test (run this first)

Before launching full experiments, run one short test to confirm your setup works:

```bash
python train_pinn.py \
  --pde reaction \
  --dtype_start fp64 \
  --optim_start lbfgs \
  --total_epochs 50
```

You should see a `results/reaction/fp64lbfgs/` folder appear with 5 CSV files and 3 PDF plots. If that works, you're good to scale up.

---

### The 7 experimental conditions

#### Static baselines (no switch)

These are your performance bounds. Omit `--dtype_switch`, `--optim_switch`, and `--switch_epoch` — the switch never fires.

```bash
# Best expected — FP64 L-BFGS
python train_pinn.py --pde reaction --dtype_start fp64 --optim_start lbfgs

# FP64 Adam
python train_pinn.py --pde reaction --dtype_start fp64 --optim_start adam

# FP32 L-BFGS
python train_pinn.py --pde reaction --dtype_start fp32 --optim_start lbfgs

# Worst expected — FP32 Adam
python train_pinn.py --pde reaction --dtype_start fp32 --optim_start adam
```

#### Switching conditions (novel contribution)

Pass `--dtype_switch`, `--optim_switch`, and `--switch_epoch` to enable a mid-training switch.

```bash
# FP32 Adam → FP64 Adam  (same optimizer, precision upgrade)
python train_pinn.py \
  --pde reaction \
  --dtype_start fp32 --optim_start adam \
  --dtype_switch fp64 --optim_switch adam \
  --switch_epoch 1000

# FP32 L-BFGS → FP64 L-BFGS  (same optimizer, precision upgrade)
python train_pinn.py \
  --pde reaction \
  --dtype_start fp32 --optim_start lbfgs \
  --dtype_switch fp64 --optim_switch lbfgs \
  --switch_epoch 1000

# FP32 Adam → FP64 L-BFGS  (cross-type — key novel condition)
python train_pinn.py \
  --pde reaction \
  --dtype_start fp32 --optim_start adam \
  --dtype_switch fp64 --optim_switch lbfgs \
  --switch_epoch 1000
```

---

### Run all conditions on one PDE

The included shell script runs all 7 conditions sequentially. Edit `DEVICE`, `TOTAL_EPOCHS`, and `SWITCH_EPOCH` at the top before running.

```bash
# Run all 7 conditions on reaction only
bash run_all.sh reaction

# Run all 7 conditions on wave only
bash run_all.sh wave

# Run all 7 on all PDEs (28 total runs — needs significant compute)
bash run_all.sh
```

---

### Compute notes

**Which PDE to run first:** Start with `reaction`. It converges fastest and is the most thoroughly tested. `wave` is the next best option. Skip `convection` if compute is limited — β=30 makes it stiff and slow. `ac` requires `allen_cahn.mat` from the baseline repo.

**L-BFGS is slow:** Each L-BFGS epoch involves multiple closure evaluations (strong Wolfe line search). Expect L-BFGS runs to take 3–10× longer than Adam runs at the same epoch count. Plan accordingly on Lambda.

**Recommended run order if compute is limited:**

1. `fp64lbfgs` (upper bound baseline — validate against Xu et al.)
2. `fp32adam` (lower bound baseline)
3. `fp32adam_to_fp64lbfgs` (key novel condition)
4. Remaining 4 conditions

---

## All CLI Options

| Flag | Default | Description |
|---|---|---|
| `--pde` | `reaction` | PDE to solve: `reaction`, `wave`, `convection`, `ac` |
| `--model` | `PINN` | Model architecture (see `model_dict.py`) |
| `--device` | `cuda:0` | PyTorch device string |
| `--seed` | `1` | Random seed for reproducibility |
| `--dtype_start` | `fp32` | Phase 1 dtype: `fp32` or `fp64` |
| `--optim_start` | `lbfgs` | Phase 1 optimizer: `adam` or `lbfgs` |
| `--dtype_switch` | *(none)* | Phase 2 dtype. Omit for static runs |
| `--optim_switch` | *(none)* | Phase 2 optimizer. Omit for static runs |
| `--switch_epoch` | *(none)* | Epoch to perform switch (1-indexed). Omit for static runs |
| `--total_epochs` | `2000` | Total training epochs |
| `--adam_lr` | `1e-3` | Adam learning rate |
| `--lbfgs_tol_grad` | `1e-8` | L-BFGS gradient tolerance |
| `--lbfgs_tol_change` | `1e-10` | L-BFGS step change tolerance |
| `--out_dir` | `./results` | Root output directory |

---

## Log File Schema

Every run produces 5 CSV files in `results/<pde>/<condition>/`:

**`*_loss.csv`** — per-epoch loss breakdown
```
epoch, loss_res, loss_bc, loss_ic, total_loss, dtype, optim, phase, condition, pde
```

**`*_timing.csv`** — per-epoch wall-clock time and GPU memory
```
epoch, fwd_time_s, bwd_time_s, total_time_s, mem_allocated_mb, mem_reserved_mb, dtype, optim, phase, condition, pde
```

**`*_grad.csv`** — per-epoch gradient norm
```
epoch, grad_norm, dtype, optim, phase, condition, pde
```

**`*_eval.csv`** — final evaluation metrics (one row per run)
```
condition, pde, model, L1_rel, L2_rel, final_loss, switch_epoch, total_epochs, total_time_s, peak_mem_mb
```

**`*_switch.csv`** — switch event record (one row per run, switching conditions only)
```
condition, pde, model, switch_epoch, from_dtype, to_dtype, from_optim, to_optim, loss_at_switch, grad_norm_at_switch
```

The `phase` column is `static` for static baselines, `pre_switch` before the switch, and `post_switch` after. This is the column used to draw the vertical switch-epoch line on all plots.

---

## Generating Paper Figures

Once you have results, run:

```bash
python aggregate_results.py --results_dir ./results --out_dir ./paper_figures
```

This produces in `./paper_figures/`:
- `table1_summary.csv` — full results table
- `table1_summary.tex` — LaTeX Table 1, ready to paste into Overleaf
- `fig_loss_<pde>.pdf` — loss curves for all conditions, per PDE
- `fig_l2_bars.pdf` — L2 error bar chart
- `fig_timing_bars.pdf` — training time bar chart

---

## Reproducing a Specific Run

Every run saves a `*_config.json` with all CLI arguments used. To reproduce exactly:

```bash
# Read the config
cat results/reaction/fp32adam_to_fp64lbfgs/reaction_PINN_fp32adam_to_fp64lbfgs_config.json

# Re-run with identical settings
python train_pinn.py --pde reaction --dtype_start fp32 --optim_start adam \
  --dtype_switch fp64 --optim_switch lbfgs --switch_epoch 1000 \
  --seed 1 --total_epochs 2000
```

---

## Known Limitations

- Single random seed per condition. Results may vary across seeds. We acknowledge this in the paper's limitations section.
- Convection (β=30) is computationally expensive. Results are included for completeness but we recommend starting with `reaction` or `wave` when compute is limited.
- Allen-Cahn requires `allen_cahn.mat` from the original FP64 baseline repo. A synthetic fallback is provided but results will not match the paper baseline.
- GPU memory logging shows `0.0` on CPU runs — this is expected.

---

## Citation

If you use this code, please cite:

```bibtex
@article{tmc-pinn-2025,
  title   = {Dtype and Optimizer Switching in Physics-Informed Neural Networks},
  author  = {[Authors]},
  year    = {2025},
}
```

And the baseline this work builds on:

```bibtex
@article{xu2025fp64,
  title   = {FP64 Is All You Need},
  author  = {Xu et al.},
  year    = {2025},
}
```

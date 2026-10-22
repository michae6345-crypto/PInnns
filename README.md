# Optimizer or Precision? Understanding Optimizer–Precision Interactions in PINNs

Code and run logs for the OPT 2026 workshop paper (NeurIPS 2026 Workshop on
Optimization for Machine Learning):

> Michael Tarekegn, Chloe Wu, Trisha Sharma, Shivank Garg (Algoverse AI Research).
> *Optimizer or Precision? Understanding Optimizer–Precision Interactions in PINNs.*
> OPT 2026. [OpenReview](https://openreview.net/forum?id=uyYM0lHArz)

We train PINNs with Adam and L-BFGS in FP32 and FP64 on four PDEs (reaction,
wave, Burgers', Allen–Cahn), in eight configurations, some of which switch
optimizer, precision or both halfway through. Every low-error run ends in
L-BFGS. Going from FP32 to FP64 changes Adam's error by at most 1.25x and
L-BFGS's by up to 17.7x.

The version used for the paper is tagged `opt2026`.

---

## Layout

```
train_pinn.py            training script, all eight conditions and five PDE setups
make_paper_tables.py     regenerates Tables 2-4 of the paper from results_opt2026/
aggregate_results.py     summary table, loss curves and bar charts from any results dir
run_experiments.ipynb    notebook wrapper around run_all, one cell per PDE
results_opt2026/         logs of the 40 runs reported in the paper (see below)
allen_cahn.mat           Allen-Cahn reference solution (third-party, see DATA_SOURCES.md)
burgers_shock.mat        Burgers' reference solution (third-party, see DATA_SOURCES.md)
```

## Setup

```bash
pip install -r requirements.txt
```

The paper's runs used PyTorch 2.7.0 with CUDA 12.8 on one NVIDIA H100. Use a
GPU: an L-BFGS run takes roughly 8-20x as long as an Adam run with the same
number of epochs.

## Reproducing the paper

### From the released logs

```bash
python make_paper_tables.py --results_dir ./results_opt2026
```

This prints:

- **Table 2:** relative L2 error for the 32 main runs.
- **Table 3:** all 40 runs, with relative L1 error, final loss, terminal gradient
  L2 norm, stall epoch and wall time.
- **Table 4:** improvement factors.

Each run's per-epoch loss, gradient-norm and timing logs are in
`results_opt2026/<pde>/<condition>/`, alongside the JSON config that records every
argument plus the GPU, torch and CUDA versions. Model weights are not included.

### Re-running the experiments

```python
import train_pinn as tp
tp.run_all(pdes=['reaction', 'wave', 'burgers', 'ac', 'ac_weighted'],
           out_dir='./results', skip_existing=True)
```

This runs all eight conditions for each PDE with the paper's budgets (2,000
epochs for reaction, 10,000 for the rest, switching at the halfway epoch) and
skips runs that already have results. The 40 runs took about 41 H100-hours.
Then run `make_paper_tables.py --results_dir ./results`.

`ac` is Allen–Cahn with equal loss weights (the main-text setting).
`ac_weighted` uses the (10, 1, 100) weighting of Xu et al.'s released code
(paper Appendix E). `convection` is implemented but not used in the paper.

## Conditions

| Condition | Phase 1 | Phase 2 |
|---|---|---|
| `fp32adam` | FP32 Adam | — |
| `fp64adam` | FP64 Adam | — |
| `fp32lbfgs` | FP32 L-BFGS | — |
| `fp64lbfgs` | FP64 L-BFGS | — |
| `fp32adam_to_fp64adam` | FP32 Adam | FP64 Adam |
| `fp32lbfgs_to_fp64lbfgs` | FP32 L-BFGS | FP64 L-BFGS |
| `fp32adam_to_fp64lbfgs` | FP32 Adam | FP64 L-BFGS |
| `fp64adam_to_fp64lbfgs` | FP64 Adam | FP64 L-BFGS |

`fp64adam_to_fp64lbfgs` never changes precision, so it separates the effect of
switching optimizer from the effect of switching precision. At a switch, the
model and collocation tensors are cast to the new dtype and a fresh optimizer is
created, so L-BFGS starts its second phase with an empty curvature history.

## PDEs and budgets

| `--pde` | Equation | Boundary / initial conditions | Epochs | Switch at |
|---|---|---|---|---|
| `reaction` | u_t = 5u(1-u) | periodic in u | 2,000 | 1,000 |
| `wave` | u_tt = 4u_xx | Dirichlet u = 0; u_t(x,0) = 0 | 10,000 | 5,000 |
| `ac` | u_t - 1e-4 u_xx + 5(u^3 - u) = 0 | periodic in u and u_x | 10,000 | 5,000 |
| `ac_weighted` | as `ac`, loss weights 10:1:100 | as `ac` | 10,000 | 5,000 |
| `burgers` | u_t + uu_x = (0.01/pi) u_xx | Dirichlet u = 0 | 10,000 | 5,000 |
| `convection` | u_t + 50u_x = 0, 401x401 grid (not in paper) | periodic | 50,000 | 25,000 |

All paper runs use a 101x101 grid and a 4x512 tanh MLP with 790,017 parameters.
Adam runs at lr 1e-3. L-BFGS runs at lr 1.0 with history 100, max_iter 20,
strong-Wolfe line search, `tolerance_grad=1e-8` and `tolerance_change=1e-10`.
The seed is 1, and TF32 is disabled.

## Command-line use

A static baseline omits the switch flags:

```bash
python train_pinn.py --pde reaction --dtype_start fp64 --optim_start lbfgs
```

Passing `--dtype_switch`, `--optim_switch` and `--switch_epoch` together turns a
run into a switching run:

```bash
python train_pinn.py --pde reaction \
  --dtype_start fp32 --optim_start adam \
  --dtype_switch fp64 --optim_switch lbfgs \
  --switch_epoch 1000
```

| Flag | Default | Description |
|---|---|---|
| `--pde` | `reaction` | Which PDE to solve |
| `--device` | `cuda:0` | PyTorch device string; falls back to CPU if unavailable |
| `--seed` | `1` | Random seed |
| `--dtype_start` | `fp64` | Phase 1 dtype: `fp32` or `fp64` |
| `--optim_start` | `lbfgs` | Phase 1 optimizer: `adam` or `lbfgs` |
| `--dtype_switch` | *(none)* | Phase 2 dtype; omit for a static run |
| `--optim_switch` | *(none)* | Phase 2 optimizer; omit for a static run |
| `--switch_epoch` | *(none)* | Epoch at which to switch, 1-indexed; the switch is applied before that epoch's step |
| `--total_epochs` | per-PDE | Defaults to the budget in the table above |
| `--adam_lr` | `1e-3` | Adam learning rate |
| `--lbfgs_tol_grad` | `1e-8` | L-BFGS `tolerance_grad` |
| `--lbfgs_tol_change` | `1e-10` | L-BFGS `tolerance_change` |
| `--mat_path` | per-PDE | Reference solution; resolved per PDE unless overridden |
| `--out_dir` | `./results` | Root output directory |

`--model` is accepted but ignored: every run uses the network defined in
`train_pinn.py`.

## Log schema

Each run writes to `<out_dir>/<pde>/<condition>/`. The released logs keep the
CSVs and config; a fresh run also writes PDF curves and model weights.

| File | Rows | Columns |
|---|---|---|
| `*_loss.csv` | per epoch | epoch, loss_res, loss_bc, loss_ic, total_loss, dtype, optim, phase, condition, pde |
| `*_grad.csv` | per epoch | epoch, grad_norm (L2 norm of the full parameter gradient), dtype, optim, phase, condition, pde |
| `*_timing.csv` | per epoch | epoch, fwd_time_s, bwd_time_s, total_time_s, mem_allocated_mb, mem_reserved_mb, dtype, optim, phase, condition, pde |
| `*_eval.csv` | one per run | condition, pde, model, L1_rel, L2_rel, final_loss, switch_epoch, total_epochs, total_time_s, peak_mem_mb |
| `*_switch.csv` | one per run | condition, pde, model, switch_epoch, from_dtype, to_dtype, from_optim, to_optim, loss_at_switch, grad_norm_at_switch |

Losses and gradient norms come from the last objective evaluation of each epoch.
For `ac_weighted` the logged total loss is the weighted sum. `phase` is `static`
for baselines, and `pre_switch` / `post_switch` on either side of a switch.
Ignore `peak_mem_mb`: it records current, not peak, allocation.

`aggregate_results.py --results_dir <dir> --out_dir ./paper_figures` writes a
summary table, per-PDE loss curves and bar charts for any results directory.

## Limitations

- One seed per condition; the paper does not rank third-decimal differences.
- Allen–Cahn and Burgers' report `nan` for L2 if their `.mat` file is missing.
  Training is unaffected.
- Solution error is evaluated on the training grid.

## Citation

```bibtex
@inproceedings{tarekegn2026optimizer,
  title     = {Optimizer or Precision? Understanding Optimizer--Precision Interactions in {PINNs}},
  author    = {Tarekegn, Michael and Wu, Chloe and Sharma, Trisha and Garg, Shivank},
  booktitle = {OPT 2026: 18th Annual Workshop on Optimization for Machine Learning (NeurIPS 2026 Workshop)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=uyYM0lHArz}
}
```

The benchmark setups and reference data build on Xu et al., *FP64 is all you
need: rethinking failure modes in physics-informed neural networks* (NeurIPS
2025), and Raissi et al., *Physics-informed neural networks* (JCP 2019). See
DATA_SOURCES.md.

## License

The code is released under the MIT License (see LICENSE). The `.mat` reference
solutions are third-party data and are not covered by it (see DATA_SOURCES.md).

# Optimizer or Precision? Understanding Optimizer–Precision Interactions in PINNs

Code and run logs for a workshop paper at OPT 2026, the NeurIPS 2026 Workshop on
Optimization for Machine Learning. Michael Tarekegn, Chloe Wu, Trisha Sharma and
Shivank Garg (Algoverse AI Research). [OpenReview](https://openreview.net/forum?id=uyYM0lHArz)

## Finding

We trained physics-informed neural networks (PINNs) with Adam and L-BFGS in FP32
and FP64 on four PDEs. Every run with relative L2 error below 0.226 ended in
L-BFGS. Moving from FP32 to FP64 changed Adam's error by at most 1.25x and
L-BFGS's by up to 17.7x. All four FP32 L-BFGS runs stopped early: the training
loss last changed at epoch 51 of 2,000 (reaction), 829 of 10,000 (wave), 488 of
10,000 (Burgers') and 32 of 10,000 (Allen–Cahn). In the FP64 L-BFGS runs
the loss kept changing until epochs 368, 1,725, 2,841 and 6,110. The likely cause is that the
L-BFGS stopping tolerances used here (`tolerance_grad=1e-8`,
`tolerance_change=1e-10`) sit below FP32 machine epsilon. We have not tested
this directly.

## Experiment grid

40 runs: 8 configurations on 5 PDE setups, seed 1, one NVIDIA H100 PCIe
(PyTorch 2.7.0, CUDA 12.8, TF32 off). The four main PDEs give the 32 runs in the
paper's main table. The fifth setup, `ac_weighted`, is Allen–Cahn with loss
weights 10:1:100 on the residual, boundary and initial-condition terms. The 40 runs took about 41 H100-hours.

| Configuration | First half | Second half |
|---|---|---|
| `fp32adam`, `fp64adam`, `fp32lbfgs`, `fp64lbfgs` | as named | same |
| `fp32adam_to_fp64adam` | FP32 Adam | FP64 Adam |
| `fp32lbfgs_to_fp64lbfgs` | FP32 L-BFGS | FP64 L-BFGS |
| `fp32adam_to_fp64lbfgs` | FP32 Adam | FP64 L-BFGS |
| `fp64adam_to_fp64lbfgs` | FP64 Adam | FP64 L-BFGS |

| PDE (`--pde`) | Epochs | Switch at |
|---|---|---|
| `reaction` | 2,000 | 1,000 |
| `wave`, `burgers`, `ac`, `ac_weighted` | 10,000 | 5,000 |

Every run uses a 4-layer, 512-wide tanh MLP (790,017 parameters) on a 101x101
grid, with Adam at lr 1e-3 and L-BFGS with strong-Wolfe line search. At a switch
the model is cast to the new dtype and a fresh optimizer is created.
`fp64adam_to_fp64lbfgs` keeps FP64 throughout, so it isolates the optimizer
switch from the precision switch.

## Reproduce

```bash
pip install -r requirements.txt

# Tables 2-4 of the paper from the released logs (CPU is fine)
python make_paper_tables.py --results_dir ./results_opt2026

# Re-run all 40 runs (GPU; finished runs are skipped)
python -c "import train_pinn as tp; tp.run_all(pdes=['reaction','wave','burgers','ac','ac_weighted'], out_dir='./results')"
python make_paper_tables.py --results_dir ./results

# One run
python train_pinn.py --pde wave --dtype_start fp32 --optim_start adam \
  --dtype_switch fp64 --optim_switch lbfgs --switch_epoch 5000

# Loss curves and bar charts
python aggregate_results.py --results_dir ./results_opt2026 --out_dir ./paper_figures
```

`python train_pinn.py --help` lists every flag. The code used for the paper is
tagged `opt2026`.

## Logs and results

Each run is in `results_opt2026/<pde>/<configuration>/`:

- `*_loss.csv`, `*_grad.csv`, `*_timing.csv`: one row per epoch (loss terms,
  gradient L2 norm, time and memory).
- `*_eval.csv`: relative L1 and L2 error, final loss and wall time.
- `*_switch.csv`: loss and gradient norm at the switch.
- `*_config.json`: every argument plus the GPU, PyTorch and CUDA versions.

Model weights are not included. Ignore the `peak_mem_mb` column: it records
current, not peak, allocation.

## Limitations

- One seed per configuration.
- Error is measured on the training grid.
- Allen–Cahn and Burgers' report `nan` error if `allen_cahn.mat` or
  `burgers_shock.mat` is missing. These reference solutions are third-party data
  (see [DATA_SOURCES.md](DATA_SOURCES.md)).

## Citation

```bibtex
@misc{tarekegn2026optimizer,
  title  = {Optimizer or Precision? Understanding Optimizer--Precision Interactions in {PINNs}},
  author = {Tarekegn, Michael and Wu, Chloe and Sharma, Trisha and Garg, Shivank},
  note   = {OPT 2026: 18th Annual Workshop on Optimization for Machine Learning (NeurIPS 2026 Workshop)},
  year   = {2026},
  url    = {https://openreview.net/forum?id=uyYM0lHArz}
}
```

Code is MIT licensed (see [LICENSE](LICENSE)). The `.mat` files are not covered
by it.

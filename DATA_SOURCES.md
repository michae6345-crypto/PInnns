# Data sources

The two reference-solution files are third-party data. They are not covered by this repository's MIT license. Each file is byte-identical to its upstream copy; the git blob SHA-1 values below match the upstream repositories.

| File | Used for | Source | Upstream license | Git blob SHA-1 |
|---|---|---|---|---|
| `allen_cahn.mat` | Allen–Cahn reference solution | [miniHuiHui/PINN_FP64](https://github.com/miniHuiHui/PINN_FP64) (`allen_cahn.mat`), the released code of Xu et al., *FP64 is all you need: rethinking failure modes in physics-informed neural networks*, NeurIPS 2025 | none stated | `28ceae2c2d1fa872778035798618eeb1c1e0231a` |
| `burgers_shock.mat` | Burgers' reference solution | [maziarraissi/PINNs](https://github.com/maziarraissi/PINNs) (`appendix/Data/burgers_shock.mat`), Raissi, Perdikaris and Karniadakis, *Physics-informed neural networks*, J. Comput. Phys. 378 (2019) | MIT | `3df4f17279b519feede1e8c872aede1a0736cdc9` |

Reaction and wave use closed-form reference solutions computed in `train_pinn.py`.

If you use these files, please cite the original papers.

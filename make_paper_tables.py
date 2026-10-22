"""
Regenerate the numbers in Tables 2-4 of "Optimizer or Precision? Understanding
Optimizer-Precision Interactions in PINNs" (OPT 2026) from the per-run logs in
results_opt2026/.

    python make_paper_tables.py --results_dir ./results_opt2026

Prints Table 2 (relative L2 error, 32 main runs), Table 3 (all 40 runs with
relative L1, final loss, terminal gradient L2 norm, stall epoch and wall time)
and Table 4 (improvement factors). Stall is the last epoch at which the total
training loss changed; a dash means it was still changing at the final epoch.
"""

import os
import argparse
import pandas as pd

CONDITIONS = [
    ('fp32adam',               'FP32-Adam'),
    ('fp64adam',               'FP64-Adam'),
    ('fp32adam_to_fp64adam',   'FP32A->FP64A'),
    ('fp32lbfgs',              'FP32-LBFGS'),
    ('fp64lbfgs',              'FP64-LBFGS'),
    ('fp32lbfgs_to_fp64lbfgs', 'FP32L->FP64L'),
    ('fp32adam_to_fp64lbfgs',  'FP32A->FP64L'),
    ('fp64adam_to_fp64lbfgs',  'FP64A->FP64L'),
]
MAIN_PDES = [('reaction', 'Reaction'), ('wave', 'Wave'),
             ('burgers', "Burgers'"), ('ac', 'AC')]
ALL_PDES = MAIN_PDES + [('ac_weighted', 'AC-w')]

# (label, numerator condition, denominator condition): factor = err(num) / err(den),
# so values above one mean the change lowers error.
CONTRASTS = [
    ('FP32->FP64, Adam fixed',      'fp32adam',  'fp64adam'),
    ('FP32->FP64, L-BFGS fixed',    'fp32lbfgs', 'fp64lbfgs'),
    ('Adam->L-BFGS, FP64 fixed',    'fp64adam',  'fp64lbfgs'),
    ('FP64A->FP64L vs FP64-LBFGS',  'fp64lbfgs', 'fp64adam_to_fp64lbfgs'),
    ('FP32L->FP64L vs FP64-LBFGS',  'fp64lbfgs', 'fp32lbfgs_to_fp64lbfgs'),
    ('FP32A->FP64L vs FP64-LBFGS',  'fp64lbfgs', 'fp32adam_to_fp64lbfgs'),
]


def load_run(results_dir, pde, cond):
    prefix = os.path.join(results_dir, pde, cond, f'{pde}_PINN_{cond}')
    ev = pd.read_csv(prefix + '_eval.csv').iloc[0]
    loss = pd.read_csv(prefix + '_loss.csv')
    grad = pd.read_csv(prefix + '_grad.csv')

    changed = loss['total_loss'].diff().fillna(1.0) != 0
    last_change = loss.loc[changed, 'epoch'].iloc[-1]
    stall = None if last_change == loss['epoch'].iloc[-1] else int(last_change)

    ends_in_lbfgs = cond.endswith('lbfgs')
    return {
        'l2': ev['L2_rel'], 'l1': ev['L1_rel'], 'final_loss': ev['final_loss'],
        'grad_end': grad['grad_norm'].iloc[-1] if ends_in_lbfgs else None,
        'stall': stall if ends_in_lbfgs else None,
        'time_s': ev['total_time_s'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='./results_opt2026')
    args = ap.parse_args()

    runs = {(p, c): load_run(args.results_dir, p, c)
            for p, _ in ALL_PDES for c, _ in CONDITIONS}

    print('Table 2: relative L2 error (* = low-error group, < 0.226)')
    print(f"{'Configuration':<16}" + ''.join(f'{n:>11}' for _, n in MAIN_PDES))
    for c, name in CONDITIONS:
        cells = [runs[(p, c)]['l2'] for p, _ in MAIN_PDES]
        print(f'{name:<16}' + ''.join(f"{v:>10.4f}{'*' if v < 0.226 else ' '}" for v in cells))

    print('\nTable 3: all 40 runs')
    print(f"{'Eq':<9}{'Configuration':<15}{'rel L2':>8}{'rel L1':>8}"
          f"{'final loss':>12}{'|grad|_2 end':>14}{'stall':>7}{'time (s)':>10}")
    for p, pname in ALL_PDES:
        for c, name in CONDITIONS:
            r = runs[(p, c)]
            grad = f"{r['grad_end']:.2e}" if r['grad_end'] is not None else '--'
            stall = str(r['stall']) if r['stall'] is not None else '--'
            print(f"{pname:<9}{name:<15}{r['l2']:>8.4f}{r['l1']:>8.4f}"
                  f"{r['final_loss']:>12.2e}{grad:>14}{stall:>7}{r['time_s']:>10.0f}")

    print('\nTable 4: improvement factors (error before / error after)')
    print(f"{'Contrast':<30}" + ''.join(f'{n:>10}' for _, n in MAIN_PDES))
    for label, num, den in CONTRASTS:
        cells = [runs[(p, num)]['l2'] / runs[(p, den)]['l2'] for p, _ in MAIN_PDES]
        print(f'{label:<30}' + ''.join(f'{v:>10.3g}' for v in cells))


if __name__ == '__main__':
    main()

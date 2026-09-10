"""
Collect the per-run CSVs written by train_pinn.py into the tables and figures
used in the paper: a summary table of relative L2 error and wall time (as both
CSV and LaTeX), per-PDE loss curves with every condition overlaid, and bar
charts of final error and training time.

    python aggregate_results.py --results_dir ./results --out_dir ./paper_figures

Conditions or PDEs with no results are rendered as em-dashes rather than
dropped, so a partial sweep still produces a readable table.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDITION_ORDER = [
    'fp64lbfgs',
    'fp64adam',
    'fp32lbfgs',
    'fp32adam',
    'fp32adam_to_fp64adam',
    'fp32lbfgs_to_fp64lbfgs',
    'fp32adam_to_fp64lbfgs',
    'fp64adam_to_fp64lbfgs',
]

CONDITION_LABELS = {
    'fp64lbfgs':              'FP64 L-BFGS',
    'fp64adam':               'FP64 Adam',
    'fp32lbfgs':              'FP32 L-BFGS',
    'fp32adam':               'FP32 Adam',
    'fp32adam_to_fp64adam':   'FP32 Adam → FP64 Adam',
    'fp32lbfgs_to_fp64lbfgs': 'FP32 L-BFGS → FP64 L-BFGS',
    'fp32adam_to_fp64lbfgs':  'FP32 Adam → FP64 L-BFGS',
    'fp64adam_to_fp64lbfgs':  'FP64 Adam → FP64 L-BFGS',
}

PDE_ORDER  = ['reaction', 'wave', 'convection', 'ac', 'ac_weighted', 'burgers']
PDE_LABELS = {'reaction': 'Reaction', 'wave': 'Wave',
              'convection': 'Convection', 'ac': 'Allen-Cahn',
              'ac_weighted': 'Allen-Cahn (weighted)', 'burgers': "Burgers'"}

COLORS = {
    'fp64lbfgs':              '#1f4e79',   # dark blue
    'fp64adam':               '#2e75b6',   # blue
    'fp32lbfgs':              '#7f7f7f',   # gray
    'fp32adam':               '#bfbfbf',   # light gray
    'fp32adam_to_fp64adam':   '#7030a0',   # purple
    'fp32lbfgs_to_fp64lbfgs': '#833c00',   # brown
    'fp32adam_to_fp64lbfgs':  '#c00000',   # red
    'fp64adam_to_fp64lbfgs':  '#e07b39',   # orange
}


def load_eval_logs(results_dir):
    paths = glob.glob(os.path.join(results_dir, '**', '*_eval.csv'), recursive=True)
    if not paths:
        print(f'[WARN] No eval CSV files found under {results_dir}')
        return pd.DataFrame()
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f'[WARN] Could not read {p}: {e}')
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_loss_logs(results_dir):
    paths = glob.glob(os.path.join(results_dir, '**', '*_loss.csv'), recursive=True)
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f'[WARN] Could not read {p}: {e}')
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ===========================================================================
# Summary table — relative L2 error and wall time
# ===========================================================================

def build_summary_table(eval_df, out_dir):
    if eval_df.empty:
        print('[TABLE] No eval data found.')
        return

    # A condition may have been re-run; the last row wins.
    eval_df = eval_df.drop_duplicates(subset=['condition', 'pde'], keep='last')

    rows = []
    for cond in CONDITION_ORDER:
        row = {'Condition': CONDITION_LABELS.get(cond, cond)}
        for pde in PDE_ORDER:
            sub = eval_df[(eval_df['condition'] == cond) & (eval_df['pde'] == pde)]
            if sub.empty:
                row[f'{PDE_LABELS[pde]}\nL2'] = '—'
                row[f'{PDE_LABELS[pde]}\nTime(s)'] = '—'
            else:
                row[f'{PDE_LABELS[pde]}\nL2'] = f"{sub['L2_rel'].values[0]:.4e}"
                row[f'{PDE_LABELS[pde]}\nTime(s)'] = f"{sub['total_time_s'].values[0]:.0f}"
        rows.append(row)

    df_table = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, 'table1_summary.csv')
    df_table.to_csv(csv_path, index=False)
    print(f'[TABLE] Summary CSV: {csv_path}')

    latex_path = os.path.join(out_dir, 'table1_summary.tex')
    with open(latex_path, 'w') as f:
        n_pdes = len(PDE_ORDER)
        f.write('\\begin{table}[t]\n')
        f.write('\\centering\n')
        f.write('\\caption{Relative L2 error and total training time (seconds) for '
                'every condition and benchmark PDE. Switching conditions switch at '
                'the halfway point of each PDE\'s epoch budget.}\n')
        f.write('\\label{tab:main_results}\n')
        f.write('\\resizebox{\\textwidth}{!}{%\n')
        col_spec = 'l' + ('cc' * n_pdes)
        f.write(f'\\begin{{tabular}}{{{col_spec}}}\n')
        f.write('\\toprule\n')
        pde_headers = ' & '.join(
            f'\\multicolumn{{2}}{{c}}{{{PDE_LABELS[p]}}}' for p in PDE_ORDER)
        f.write(f'Condition & {pde_headers} \\\\\n')
        sub_headers = ' & '.join(['L2 Rel. & Time (s)'] * n_pdes)
        f.write(f'\\cmidrule(lr){{2-{1+2*n_pdes}}}')
        f.write(f' & {sub_headers} \\\\\n')
        f.write('\\midrule\n')
        f.write('\\multicolumn{' + str(1 + 2 * n_pdes) + '}{l}'
                '{\\textit{Static baselines}} \\\\\n')
        switch_header_written = False
        for cond in CONDITION_ORDER:
            if '_to_' in cond and not switch_header_written:
                switch_header_written = True
                f.write('\\midrule\n')
                f.write('\\multicolumn{' + str(1 + 2 * n_pdes) + '}{l}'
                        '{\\textit{Switching conditions}} \\\\\n')
            label = CONDITION_LABELS.get(cond, cond)
            cells = []
            for pde in PDE_ORDER:
                sub = eval_df[(eval_df['condition'] == cond) & (eval_df['pde'] == pde)]
                if sub.empty:
                    cells += ['—', '—']
                else:
                    cells += [f"{sub['L2_rel'].values[0]:.4e}",
                              f"{sub['total_time_s'].values[0]:.0f}"]
            f.write(f'{label} & ' + ' & '.join(cells) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}}\n')
        f.write('\\end{table}\n')

    print(f'[TABLE] LaTeX table: {latex_path}')


# ===========================================================================
# Loss curves, one figure per PDE with every condition overlaid
# ===========================================================================

def plot_loss_curves(loss_df, eval_df, out_dir):
    if loss_df.empty:
        print('[PLOT] No loss data found.')
        return

    for pde in PDE_ORDER:
        sub = loss_df[loss_df['pde'] == pde]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for cond in CONDITION_ORDER:
            cdf = sub[sub['condition'] == cond]
            if cdf.empty:
                continue

            color = COLORS.get(cond, 'black')
            # Switching conditions are the point of the paper, so draw them heavier.
            lw = 2.5 if '_to_' in cond else 1.5

            ax.semilogy(cdf['epoch'], cdf['total_loss'],
                        color=color, lw=lw,
                        label=CONDITION_LABELS.get(cond, cond))

            if '_to_' in cond:
                sw_rows = eval_df[
                    (eval_df['condition'] == cond) & (eval_df['pde'] == pde)]
                if not sw_rows.empty:
                    se = sw_rows['switch_epoch'].values[0]
                    if pd.notna(se) and se > 0:
                        ax.axvline(se, color=color, ls=':', lw=1.2, alpha=0.6)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Total loss (log scale)', fontsize=11)
        ax.set_title(f'{PDE_LABELS[pde]} — training loss by condition', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir, f'fig_loss_{pde}.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'[PLOT] Loss curve: {path}')


# ===========================================================================
# Bar charts of final error and training time
# ===========================================================================

def plot_l2_bars(eval_df, out_dir):
    if eval_df.empty:
        return

    eval_df = eval_df.drop_duplicates(subset=['condition', 'pde'], keep='last')
    present_conds = [c for c in CONDITION_ORDER if c in eval_df['condition'].values]

    fig, axes = plt.subplots(1, len(PDE_ORDER),
                             figsize=(2.6 * len(PDE_ORDER), 4), sharey=False)
    if len(PDE_ORDER) == 1:
        axes = [axes]

    for ax, pde in zip(axes, PDE_ORDER):
        vals, colors, labels = [], [], []
        for cond in present_conds:
            sub = eval_df[(eval_df['condition'] == cond) & (eval_df['pde'] == pde)]
            if not sub.empty:
                vals.append(sub['L2_rel'].values[0])
                colors.append(COLORS.get(cond, 'gray'))
                labels.append(CONDITION_LABELS.get(cond, cond))

        if not vals:
            continue

        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(PDE_LABELS[pde], fontsize=10)
        ax.set_ylabel('Rel. L2 error' if pde == PDE_ORDER[0] else '')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Relative L2 error by condition and PDE', fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_l2_bars.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'[PLOT] L2 bar chart: {path}')


def plot_timing_bars(eval_df, out_dir):
    if eval_df.empty:
        return

    eval_df = eval_df.drop_duplicates(subset=['condition', 'pde'], keep='last')
    present_conds = [c for c in CONDITION_ORDER if c in eval_df['condition'].values]

    fig, axes = plt.subplots(1, len(PDE_ORDER),
                             figsize=(2.6 * len(PDE_ORDER), 4), sharey=False)
    if len(PDE_ORDER) == 1:
        axes = [axes]

    for ax, pde in zip(axes, PDE_ORDER):
        vals, colors, labels = [], [], []
        for cond in present_conds:
            sub = eval_df[(eval_df['condition'] == cond) & (eval_df['pde'] == pde)]
            if not sub.empty:
                vals.append(sub['total_time_s'].values[0])
                colors.append(COLORS.get(cond, 'gray'))
                labels.append(CONDITION_LABELS.get(cond, cond))

        if not vals:
            continue

        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(PDE_LABELS[pde], fontsize=10)
        ax.set_ylabel('Wall time (s)' if pde == PDE_ORDER[0] else '')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Total training time by condition and PDE', fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_timing_bars.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'[PLOT] Timing bar chart: {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', type=str, default='./results')
    ap.add_argument('--out_dir',     type=str, default='./paper_figures')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading results from: {args.results_dir}')
    eval_df = load_eval_logs(args.results_dir)
    loss_df = load_loss_logs(args.results_dir)

    print(f'Found {len(eval_df)} eval rows across '
          f'{eval_df["condition"].nunique() if not eval_df.empty else 0} conditions '
          f'and {eval_df["pde"].nunique() if not eval_df.empty else 0} PDEs.')

    build_summary_table(eval_df, args.out_dir)
    plot_loss_curves(loss_df, eval_df, args.out_dir)
    plot_l2_bars(eval_df, args.out_dir)
    plot_timing_bars(eval_df, args.out_dir)

    print(f'\nAll paper figures saved to: {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()

"""
Plot Sweep Results — Publication-quality figures for the E4 sub_weight sweep.

Reads results saved by sub_weight_sweep.py and generates four figures:
  1. pareto_ppl_subsumption.png  — Pareto frontier (PPL vs subsumption_test)
  2. training_curves.png         — 2x2 training curves per weight
  3. sweep_comparison_bar.png    — Grouped bar chart across all metrics
  4. sweep_summary_table.png     — Matplotlib table for paper inclusion

All plots are saved to benchmarks/plots/ at 300 DPI.

Usage:
  python playground/plot_sweep_results.py
"""

import os
import sys
import json
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Paths & constants
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_DIR = os.path.join(PROJECT_ROOT, 'playground', 'results', 'sub_weight_sweep')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'benchmarks', 'plots')

SWEEP_WEIGHTS = [0.5, 1.0, 2.0, 5.0]

# Run 15 baselines (XL, tanh, no sub loss)
RUN15 = {
    'ppl':               7.69,
    'sub_train_rate':    0.92,
    'sub_test_rate':     0.00,
    'dead_bits':         15,
    'semantic_gap':      0.020,
}

# Consistent color palette across all plots
WEIGHT_COLORS = {
    0.5: '#2196F3',   # blue
    1.0: '#4CAF50',   # green
    2.0: '#FF9800',   # orange
    5.0: '#E91E63',   # pink
}

WEIGHT_MARKERS = {
    0.5: 'o',
    1.0: 's',
    2.0: 'D',
    5.0: '^',
}


# ============================================================
# Data loading
# ============================================================

def load_all_results():
    """Load results.json for every available weight. Returns {weight: dict}."""
    results = {}
    for w in SWEEP_WEIGHTS:
        path = os.path.join(SWEEP_DIR, f'weight_{w}', 'results.json')
        if os.path.exists(path):
            with open(path) as f:
                results[w] = json.load(f)
        else:
            print(f"  [skip] weight={w} — no results.json found")
    return results


def load_training_log(weight):
    """Load training_log.csv for a given weight. Returns dict of lists."""
    path = os.path.join(SWEEP_DIR, f'weight_{weight}', 'training_log.csv')
    if not os.path.exists(path):
        return None
    data = {
        'step': [], 'lang_loss': [], 'tri_loss': [], 'sub_loss': [],
        'lr': [], 'entropy': [], 'dead_bits': [], 'elapsed_s': [],
    }
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                try:
                    data[key].append(float(row[key]))
                except (KeyError, ValueError):
                    data[key].append(0.0)
    return data


def get_eval(result, stage='50k'):
    """Get eval dict, handling both old (eval_25k) and new (eval_mid) formats."""
    if stage == 'mid':
        return result.get('eval_mid') or result.get('eval_25k', {})
    return result.get('eval_50k', {})


# ============================================================
# Style setup
# ============================================================

def setup_style():
    """Configure matplotlib for publication-quality output."""
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        try:
            plt.style.use('seaborn-paper')
        except OSError:
            pass  # fall back to defaults
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# ============================================================
# Plot 1: Pareto frontier — PPL vs Subsumption Test
# ============================================================

def plot_pareto(results, output_dir):
    """Scatter plot: X=PPL, Y=subsumption_test rate. Annotated with weight values."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ppls, subs, weights = [], [], []
    for w in SWEEP_WEIGHTS:
        if w not in results:
            continue
        e = get_eval(results[w], '50k')
        ppl = e.get('ppl', 0)
        sub = e.get('sub_test_rate', 0) * 100
        ppls.append(ppl)
        subs.append(sub)
        weights.append(w)
        ax.scatter(ppl, sub, color=WEIGHT_COLORS[w], marker=WEIGHT_MARKERS[w],
                   s=120, zorder=5, edgecolors='black', linewidths=0.8)
        ax.annotate(f'w={w}', (ppl, sub), textcoords='offset points',
                    xytext=(10, 6), fontsize=10, fontweight='bold',
                    color=WEIGHT_COLORS[w])

    # Run 15 baseline as reference star
    ax.scatter(RUN15['ppl'], RUN15['sub_test_rate'] * 100,
               marker='*', s=250, color='#9E9E9E', zorder=5,
               edgecolors='black', linewidths=0.8, label='Run 15 (no sub loss)')
    ax.annotate('Run 15\n(baseline)', (RUN15['ppl'], RUN15['sub_test_rate'] * 100),
                textcoords='offset points', xytext=(-12, 12),
                fontsize=9, color='#666', ha='center')

    # Connect sweep points in weight order to show the frontier
    if len(ppls) > 1:
        order = np.argsort(ppls)
        ax.plot([ppls[i] for i in order], [subs[i] for i in order],
                '--', color='#BDBDBD', linewidth=1.2, zorder=2)

    ax.set_xlabel('Perplexity (PPL)')
    ax.set_ylabel('Subsumption Test Rate (%)')
    ax.set_title('E4: PPL vs Subsumption Test — Pareto Frontier')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()

    path = os.path.join(output_dir, 'pareto_ppl_subsumption.png')
    fig.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Plot 2: Training curves (2x2 grid)
# ============================================================

def plot_training_curves(results, output_dir):
    """2x2 subplot: lang_loss, tri_loss, sub_loss, dead_bits vs step."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ('lang_loss', 'Language Loss', axes[0, 0]),
        ('tri_loss', 'Triadic Loss', axes[0, 1]),
        ('sub_loss', 'Subsumption Loss', axes[1, 0]),
        ('dead_bits', 'Dead Bits (/64)', axes[1, 1]),
    ]

    any_plotted = False
    for w in SWEEP_WEIGHTS:
        if w not in results:
            continue
        log = load_training_log(w)
        if log is None:
            print(f"  [skip] weight={w} — no training_log.csv")
            continue

        any_plotted = True
        steps = log['step']

        for key, title, ax in metrics:
            vals = log[key]
            ax.plot(steps, vals, color=WEIGHT_COLORS[w], linewidth=1.2,
                    alpha=0.85, label=f'w={w}')

    if not any_plotted:
        print("  [skip] No training logs found, skipping training_curves.png")
        plt.close()
        return

    for key, title, ax in metrics:
        ax.set_xlabel('Step')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle('E4: Training Dynamics by Sub Weight',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, 'training_curves.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Plot 3: Grouped bar chart comparison
# ============================================================

def plot_comparison_bars(results, output_dir):
    """Grouped bar chart: all weights + Run 15 baseline across 5 metrics."""
    metric_keys = [
        ('ppl',              'PPL',             False),
        ('sub_train_rate',   'Sub Train (%)',    True),
        ('sub_test_rate',    'Sub Test (%)',     True),
        ('dead_bits',        'Dead Bits',        False),
        ('semantic_gap',     'Semantic Gap',     False),
    ]
    n_metrics = len(metric_keys)

    # Collect labels and values
    labels = ['Run 15']
    all_values = []

    # Run 15 baseline
    run15_vals = []
    for key, _, is_pct in metric_keys:
        v = RUN15.get(key, 0)
        run15_vals.append(v * 100 if is_pct else v)
    all_values.append(run15_vals)

    for w in SWEEP_WEIGHTS:
        if w not in results:
            continue
        labels.append(f'w={w}')
        e = get_eval(results[w], '50k')
        vals = []
        for key, _, is_pct in metric_keys:
            v = e.get(key, 0)
            vals.append(v * 100 if is_pct else v)
        all_values.append(vals)

    n_groups = len(labels)
    x = np.arange(n_metrics)
    width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color assignments: Run 15 is gray, rest are weight colors
    bar_colors = ['#9E9E9E']
    for w in SWEEP_WEIGHTS:
        if w in results:
            bar_colors.append(WEIGHT_COLORS[w])

    for i, (label, vals, color) in enumerate(zip(labels, all_values, bar_colors)):
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=label,
                      color=color, edgecolor='white', linewidth=0.5)
        # Value labels on bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            fmt = f'{val:.0f}' if val >= 10 else f'{val:.1f}'
            if val >= 1:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3,
                        fmt, ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in metric_keys], fontsize=11)
    ax.set_ylabel('Value')
    ax.set_title('E4: Metric Comparison Across Sub Weights')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'sweep_comparison_bar.png')
    fig.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Plot 4: Summary table as figure
# ============================================================

def plot_summary_table(results, output_dir):
    """Render a comparison table as a matplotlib figure for paper inclusion."""
    # Build table data
    col_labels = ['Weight', 'PPL\n@mid', 'PPL\n@50k',
                  'Sub Train\n@50k', 'Sub Test\n@50k',
                  'Dead Bits\n@50k', 'Sem Gap\n@50k',
                  'Time\n(min)']

    rows = []

    # Run 15 baseline
    rows.append([
        'Run 15',
        f'{RUN15["ppl"]:.2f}',
        f'{RUN15["ppl"]:.2f}',
        f'{RUN15["sub_train_rate"]*100:.0f}%',
        f'{RUN15["sub_test_rate"]*100:.0f}%',
        f'{RUN15["dead_bits"]}',
        f'{RUN15["semantic_gap"]:.4f}',
        '--',
    ])

    for w in SWEEP_WEIGHTS:
        if w not in results:
            continue
        r = results[w]
        e_mid = get_eval(r, 'mid')
        e_50k = get_eval(r, '50k')
        t_min = r.get('training_time_min', 0)

        rows.append([
            f'{w}',
            f'{e_mid.get("ppl", 0):.2f}',
            f'{e_50k.get("ppl", 0):.2f}',
            f'{e_50k.get("sub_train_rate", 0)*100:.0f}%',
            f'{e_50k.get("sub_test_rate", 0)*100:.1f}%',
            f'{e_50k.get("dead_bits", 0)}',
            f'{e_50k.get("semantic_gap", 0):.4f}',
            f'{t_min:.0f}',
        ])

    n_rows = len(rows)
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(10, 1.0 + 0.5 * n_rows))
    ax.axis('off')
    ax.set_title('E4: Subsumption Weight Sweep Results',
                 fontsize=13, fontweight='bold', pad=12)

    # Row colors: header gray, Run 15 light, alternating white/light-blue
    row_colors = []
    for i in range(n_rows):
        if i == 0:
            row_colors.append('#F5F5F5')  # Run 15
        elif i % 2 == 1:
            row_colors.append('#FFFFFF')
        else:
            row_colors.append('#E3F2FD')

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Header row styling
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor('#37474F')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        cell.set_edgecolor('white')

    # Data row styling
    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor(row_colors[i])
            cell.set_edgecolor('#E0E0E0')
            # Bold the Run 15 row
            if i == 0:
                cell.set_text_props(fontstyle='italic', color='#666')

    # Highlight best sub_test (column 4, index 4)
    best_sub_test = -1
    best_row = -1
    for i in range(1, n_rows):  # skip Run 15
        val_str = rows[i][4].replace('%', '')
        try:
            val = float(val_str)
            if val > best_sub_test:
                best_sub_test = val
                best_row = i
        except ValueError:
            pass

    if best_row >= 0:
        cell = table[best_row + 1, 4]
        cell.set_text_props(fontweight='bold', color='#1B5E20')

    plt.tight_layout()
    path = os.path.join(output_dir, 'sweep_summary_table.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Console summary
# ============================================================

def print_summary(results):
    """Print a summary table to the console."""
    print()
    print("=" * 100)
    print("  E4: SUBSUMPTION WEIGHT SWEEP — SUMMARY")
    print("=" * 100)

    header = (f"  {'Weight':>8s}  {'PPL@mid':>10s}  {'PPL@50k':>10s}  "
              f"{'SubTr@50k':>10s}  {'SubTe@50k':>10s}  "
              f"{'Dead':>6s}  {'SemGap':>10s}  {'Time(m)':>8s}")
    print(header)
    print("  " + "-" * 90)

    # Run 15 baseline
    print(f"  {'Run15':>8s}  {RUN15['ppl']:>10.2f}  {RUN15['ppl']:>10.2f}  "
          f"{RUN15['sub_train_rate']*100:>9.0f}%  {RUN15['sub_test_rate']*100:>9.0f}%  "
          f"{RUN15['dead_bits']:>6d}  {RUN15['semantic_gap']:>10.4f}  {'--':>8s}")

    for w in SWEEP_WEIGHTS:
        if w not in results:
            continue
        r = results[w]
        e_mid = get_eval(r, 'mid')
        e_50k = get_eval(r, '50k')
        t = r.get('training_time_min', 0)

        print(f"  {w:>8.1f}  "
              f"{e_mid.get('ppl', 0):>10.2f}  "
              f"{e_50k.get('ppl', 0):>10.2f}  "
              f"{e_50k.get('sub_train_rate', 0)*100:>9.0f}%  "
              f"{e_50k.get('sub_test_rate', 0)*100:>9.1f}%  "
              f"{e_50k.get('dead_bits', 0):>6d}  "
              f"{e_50k.get('semantic_gap', 0):>10.4f}  "
              f"{t:>8.0f}")

    print("  " + "-" * 90)

    # Best weight recommendation
    best_w = None
    best_sub_test = -1
    for w, r in results.items():
        e = get_eval(r, '50k')
        st = e.get('sub_test_rate', 0)
        if st > best_sub_test:
            best_sub_test = st
            best_w = w

    if best_w is not None:
        e = get_eval(results[best_w], '50k')
        print(f"\n  Best sub_test@50k: weight={best_w} "
              f"({best_sub_test*100:.1f}% test, PPL={e.get('ppl',0):.2f})")
    print("=" * 100)
    print()


# ============================================================
# Main
# ============================================================

def main():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("  Loading sweep results...")
    results = load_all_results()

    if not results:
        print("  ERROR: No results found. Run sub_weight_sweep.py first.")
        sys.exit(1)

    print(f"  Found results for weights: {sorted(results.keys())}")

    # Console summary
    print_summary(results)

    # Generate all plots
    print("  Generating plots...")
    plot_pareto(results, OUTPUT_DIR)
    plot_training_curves(results, OUTPUT_DIR)
    plot_comparison_bars(results, OUTPUT_DIR)
    plot_summary_table(results, OUTPUT_DIR)

    print()
    print(f"  All plots saved to: {OUTPUT_DIR}")
    print()


if __name__ == '__main__':
    main()

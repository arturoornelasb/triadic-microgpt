"""
Plot training loss curves for key TriadicGPT runs.
Extracts data from training_log.csv files and generates a publication-quality figure.
"""
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Key runs to plot
RUNS = {
    'danza_63bit_xl_v2': {'label': 'Run 15 (XL, 64-bit, MSE)', 'color': '#2196F3', 'ls': '-'},
    'danza_63bit_xl': {'label': 'Run 14 (XL, 64-bit, v1)', 'color': '#90CAF9', 'ls': '--'},
    'danza_bootstrap_xl': {'label': 'Bootstrap XL', 'color': '#4CAF50', 'ls': ':'},
    'danza_63bit_xl_v2_ifsq': {'label': 'iFSQ XL', 'color': '#FF9800', 'ls': '-'},
    'danza_gpt2medium_ternary_v2': {'label': 'GPT-2 355M ternary', 'color': '#F44336', 'ls': '--'},
    'gpt2_medium_infonce': {'label': 'GPT-2 InfoNCE', 'color': '#9C27B0', 'ls': '-.'},
}


def load_log(run_name):
    """Load training log CSV, return dict of arrays."""
    path = os.path.join(CKPT_DIR, run_name, 'training_log.csv')
    if not os.path.exists(path):
        return None
    data = {'step': [], 'loss': [], 'lang_loss': [], 'tri_loss': [],
            'bit_acc_train': [], 'bit_acc_test': [], 'sub_test': []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                if key in row:
                    try:
                        data[key].append(float(row[key]))
                    except (ValueError, TypeError):
                        data[key].append(float('nan'))
    return {k: np.array(v) for k, v in data.items() if len(v) > 0}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('TriadicGPT Training Dynamics', fontsize=14, fontweight='bold')

    loaded = {}
    for run_name, meta in RUNS.items():
        data = load_log(run_name)
        if data is not None and len(data.get('step', [])) > 1:
            loaded[run_name] = (data, meta)
            print(f"  Loaded {run_name}: {len(data['step'])} steps, "
                  f"final loss={data['loss'][-1]:.3f}")
        else:
            print(f"  SKIP {run_name}: no data or single point")

    # Panel A: Language loss
    ax = axes[0, 0]
    for run_name, (data, meta) in loaded.items():
        if 'lang_loss' in data and len(data['lang_loss']) > 0:
            ax.plot(data['step'], data['lang_loss'],
                    label=meta['label'], color=meta['color'], ls=meta['ls'], lw=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Language Loss')
    ax.set_title('(A) Language Loss')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    # Panel B: Triadic loss (for runs that have it)
    ax = axes[0, 1]
    for run_name, (data, meta) in loaded.items():
        if 'tri_loss' in data and len(data['tri_loss']) > 0 and np.nanmax(data['tri_loss']) > 0:
            ax.plot(data['step'], data['tri_loss'],
                    label=meta['label'], color=meta['color'], ls=meta['ls'], lw=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Triadic Loss')
    ax.set_title('(B) Triadic Alignment Loss')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    # Panel C: Bit accuracy
    ax = axes[1, 0]
    for run_name, (data, meta) in loaded.items():
        key = 'bit_acc_test' if 'bit_acc_test' in data and len(data['bit_acc_test']) > 0 else 'bit_acc_holdout'
        if key in data and len(data[key]) > 0:
            vals = data[key]
            if np.nanmax(vals) > 0:
                ax.plot(data['step'], vals,
                        label=meta['label'], color=meta['color'], ls=meta['ls'], lw=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Bit Accuracy (holdout)')
    ax.set_title('(C) Bit Accuracy')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)

    # Panel D: Subsumption test rate (for runs that have it)
    ax = axes[1, 1]
    has_sub = False
    for run_name, (data, meta) in loaded.items():
        if 'sub_test' in data and len(data['sub_test']) > 0 and np.nanmax(data['sub_test']) > 0:
            ax.plot(data['step'], data['sub_test'] * 100,
                    label=meta['label'], color=meta['color'], ls=meta['ls'], lw=1.5)
            has_sub = True
    if not has_sub:
        ax.text(0.5, 0.5, 'No subsumption data\nin selected runs',
                ha='center', va='center', transform=ax.transAxes, fontsize=11, color='gray')
    ax.set_xlabel('Step')
    ax.set_ylabel('Subsumption Rate (%)')
    ax.set_title('(D) Held-out Subsumption')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'training_curves.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved to {out_path}")

    # Also save PNG for quick viewing
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved PNG to {out_path.replace('.pdf', '.png')}")


if __name__ == '__main__':
    main()

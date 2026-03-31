#!/usr/bin/env python3
"""
Plot training loss curves for the paper (Sprint 2.4).

Reads training_log.csv from Run 15 and sparse finetune,
produces publication-quality loss curves with annotations.
"""

import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_log(path, subsample=50):
    """Load training log CSV, subsampled for plotting."""
    steps, losses, tri_losses = [], [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % subsample == 0:
                steps.append(int(row['step']))
                losses.append(float(row['loss']))
                tri_losses.append(float(row['tri_loss']))
    return np.array(steps), np.array(losses), np.array(tri_losses)


def smooth(y, window=20):
    """Moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


def main():
    # --- Load Run 15 ---
    run15_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'training_log.csv')
    if not os.path.exists(run15_path):
        print(f"Run 15 log not found: {run15_path}")
        sys.exit(1)

    steps, losses, tri_losses = load_log(run15_path, subsample=50)
    print(f"Run 15: {len(steps)} points, steps 1-{steps[-1]}")
    print(f"  Final loss: {losses[-1]:.3f}, triadic: {tri_losses[-1]:.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Language loss
    ax = axes[0]
    s_steps = steps[:len(smooth(losses))]
    ax.plot(s_steps, smooth(losses), 'b-', alpha=0.85, linewidth=1.2, label='Language loss')
    ax.set_xlabel('Training step', fontsize=11)
    ax.set_ylabel('Cross-entropy loss', fontsize=11)
    ax.set_title('(a) Language modeling loss', fontsize=12)
    ax.set_xlim(0, steps[-1])

    # Mark phases
    ax.axvline(x=5000, color='gray', linestyle=':', alpha=0.5)
    ax.text(5500, losses[0] * 0.85, 'warmup\nend', fontsize=8, color='gray')

    # Find elbow (where loss drops below 50% of initial)
    elbow_idx = np.argmax(losses < losses[0] * 0.3)
    if elbow_idx > 0:
        ax.axvline(x=steps[elbow_idx], color='orange', linestyle='--', alpha=0.5)
        ax.text(steps[elbow_idx] + 500, losses[0] * 0.5, f'crystallization\n~step {steps[elbow_idx]}',
                fontsize=8, color='orange')

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Triadic loss
    ax = axes[1]
    # triadic loss only becomes active after some steps
    tri_active = tri_losses > 0.001
    if tri_active.any():
        first_active = np.argmax(tri_active)
        s_tri = smooth(tri_losses[first_active:])
        s_steps_tri = steps[first_active:first_active + len(s_tri)]
        ax.plot(s_steps_tri, s_tri, 'r-', alpha=0.85, linewidth=1.2, label='Triadic loss')
        ax.set_xlabel('Training step', fontsize=11)
        ax.set_ylabel('Triadic loss', fontsize=11)
        ax.set_title('(b) Triadic projection loss', fontsize=12)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No triadic loss recorded', transform=ax.transAxes,
                ha='center', fontsize=12, color='gray')

    ax.grid(True, alpha=0.3)

    plt.suptitle('TriadicGPT Training Curves (Run 15, 40M params, 50K steps)', fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(PROJECT_ROOT, 'benchmarks', 'plots', 'training_curves_run15.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {out_path}")

    # Also save key statistics for the paper
    stats = {
        'total_steps': int(steps[-1]),
        'final_lang_loss': float(losses[-1]),
        'final_tri_loss': float(tri_losses[-1]),
        'initial_lang_loss': float(losses[0]),
        'loss_reduction': float(1 - losses[-1] / losses[0]),
        'crystallization_step': int(steps[elbow_idx]) if elbow_idx > 0 else None,
    }

    import json
    stats_path = os.path.join(PROJECT_ROOT, 'benchmarks', 'results', 'training_curve_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {stats_path}")
    print(f"  Loss reduction: {stats['loss_reduction']:.1%} ({stats['initial_lang_loss']:.2f} -> {stats['final_lang_loss']:.2f})")


if __name__ == '__main__':
    main()

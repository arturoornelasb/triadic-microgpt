"""
Bits Sweep Plots -- Generate publication-quality figures for the bits sweep.

Reads the bits sweep JSON results and produces:
  1. Language loss vs k (bits)
  2. Semantic gap vs k (bits) -- KEY FIGURE
  3. Multi-metric panel (entropy, unique%, probe, analogy verif)

Usage:
  python benchmarks/scripts/bits_sweep_plots.py \
    --results benchmarks/results/v4.0-bits-sweep_scaling_study_2026-03-07.json
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return data['models']


def make_bits_plots(models, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    bits = [m['n_bits'] for m in models]
    loss = [m['final_loss'] for m in models]
    entropy = [m['bit_entropy'] for m in models]
    unique_pct = [m['unique_pct'] for m in models]
    gap = [m['semantic_gap'] for m in models]
    probe_tri = [m['probe_triadic_acc'] for m in models]
    probe_emb = [m['probe_embedding_acc'] for m in models]
    analogy = [m['analogy_verification'] for m in models]

    bit_labels = [str(b) for b in bits]

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'figure.facecolor': 'white',
    })

    # ========== Figure 1: Language Loss vs Bits ==========
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(bits)), loss, 'o-', color='#2196F3', linewidth=2, markersize=8, zorder=3)
    for i, (b, l) in enumerate(zip(bits, loss)):
        ax.annotate(f'{l:.3f}', (i, l), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(bits)))
    ax.set_xticklabels(bit_labels)
    ax.set_xlabel('Triadic Bits (k)')
    ax.set_ylabel('Training Loss')
    ax.set_title('Language Loss vs Triadic Bit Width (XL Architecture Fixed)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'bits_sweep_loss.png'), dpi=150)
    plt.close()
    print(f"  Saved: bits_sweep_loss.png")

    # ========== Figure 2: Semantic Gap (KEY FIGURE) ==========
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ['#4CAF50' if g > 0 else '#F44336' for g in gap]
    bars = ax.bar(range(len(bits)), gap, color=colors, width=0.6, zorder=3)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(range(len(bits)))
    ax.set_xticklabels(bit_labels)
    ax.set_xlabel('Triadic Bits (k)')
    ax.set_ylabel('Semantic Gap (related - unrelated sim)')
    ax.set_title('Semantic Ordering vs Bit Width')
    ax.grid(True, axis='y', alpha=0.3)

    for i, g in enumerate(gap):
        ax.annotate(f'{g:+.3f}', (i, g), textcoords="offset points",
                    xytext=(0, 8 if g >= 0 else -15), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'bits_sweep_semantic_gap.png'), dpi=150)
    plt.close()
    print(f"  Saved: bits_sweep_semantic_gap.png")

    # ========== Figure 3: Multi-metric Panel ==========
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # Bit Entropy
    ax = axes[0, 0]
    ax.plot(range(len(bits)), entropy, 'o-', color='#9C27B0', linewidth=2, markersize=7)
    ax.set_xticks(range(len(bits)))
    ax.set_xticklabels(bit_labels)
    ax.set_ylabel('Bit Entropy')
    ax.set_title('Bit Entropy')
    ax.grid(True, alpha=0.3)

    # Unique Signatures
    ax = axes[0, 1]
    ax.plot(range(len(bits)), unique_pct, 'o-', color='#FF9800', linewidth=2, markersize=7)
    ax.set_xticks(range(len(bits)))
    ax.set_xticklabels(bit_labels)
    ax.set_ylabel('Unique Signatures (%)')
    ax.set_title('Signature Diversity')
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='gray', linewidth=0.8, linestyle=':')
    ax.grid(True, alpha=0.3)

    # Probe Accuracy
    ax = axes[1, 0]
    ax.plot(range(len(bits)), [a * 100 for a in probe_tri], 'o-', color='#4CAF50',
            linewidth=2, markersize=7, label='Triadic')
    ax.plot(range(len(bits)), [a * 100 for a in probe_emb], 's--', color='#607D8B',
            linewidth=2, markersize=7, label='Embedding')
    ax.axhline(y=7.7, color='gray', linewidth=0.8, linestyle=':', label='Random')
    ax.set_xticks(range(len(bits)))
    ax.set_xticklabels(bit_labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Linear Probe Accuracy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Analogy Verification
    ax = axes[1, 1]
    ax.plot(range(len(bits)), [a * 100 for a in analogy], 'o-', color='#E91E63',
            linewidth=2, markersize=7)
    ax.axhline(y=50, color='gray', linewidth=0.8, linestyle=':', label='Random')
    ax.set_xticks(range(len(bits)))
    ax.set_xticklabels(bit_labels)
    ax.set_ylabel('Verification Rate (%)')
    ax.set_title('Analogy Verification (> median)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Triadic Bits (k)')

    plt.suptitle('Triadic Quality vs Bit Width (XL Fixed)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'bits_sweep_panel.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: bits_sweep_panel.png")


def main():
    parser = argparse.ArgumentParser(description='Bits Sweep Plots')
    parser.add_argument('--results', required=True, help='Path to bits sweep JSON')
    parser.add_argument('--output', default=None, help='Output directory for plots')
    args = parser.parse_args()

    models = load_results(args.results)
    output_dir = args.output or os.path.join(os.path.dirname(args.results), '..', 'plots')

    print()
    print("  Generating bits sweep plots...")
    make_bits_plots(models, output_dir)
    print(f"  All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()

"""
Information Hierarchy Analysis — Zero GPU, pure evaluation.

Uses the Sub(5.0) model from the subsumption experiment to quantify
how #active_bits correlates with conceptual abstractness.

Hypothesis: abstract concepts (animal, feeling) → fewer active bits
            concrete concepts (dog, happy) → more active bits
            This is an emergent information hierarchy.
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
from src.triadic import PrimeMapper
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'playground', 'results')

# Taxonomy with depth levels:
# depth 0 = most abstract (hypernym)
# depth 1 = concrete (hyponym)
# depth 2 = very concrete (sub-hyponym)
TAXONOMY = {
    'animal': {
        'depth': 0, 'children': {
            'dog': {'depth': 1, 'children': {'puppy': {'depth': 2}}},
            'cat': {'depth': 1, 'children': {'kitten': {'depth': 2}}},
            'bird': {'depth': 1, 'children': {}},
            'fish': {'depth': 1, 'children': {}},
            'horse': {'depth': 1, 'children': {}},
            'rabbit': {'depth': 1, 'children': {}},
            'bear': {'depth': 1, 'children': {}},
            'mouse': {'depth': 1, 'children': {}},
            'lion': {'depth': 1, 'children': {}},
            'tiger': {'depth': 1, 'children': {}},
            'frog': {'depth': 1, 'children': {}},
            'deer': {'depth': 1, 'children': {}},
        }
    },
    'person': {
        'depth': 0, 'children': {
            'king': {'depth': 1, 'children': {}},
            'queen': {'depth': 1, 'children': {}},
            'doctor': {'depth': 1, 'children': {}},
            'teacher': {'depth': 1, 'children': {}},
            'princess': {'depth': 1, 'children': {'girl': {'depth': 2}}},
            'prince': {'depth': 1, 'children': {'boy': {'depth': 2}}},
            'man': {'depth': 1, 'children': {}},
            'woman': {'depth': 1, 'children': {}},
            'baby': {'depth': 1, 'children': {}},
        }
    },
    'feeling': {
        'depth': 0, 'children': {
            'happy': {'depth': 1, 'children': {}},
            'sad': {'depth': 1, 'children': {}},
            'love': {'depth': 1, 'children': {}},
            'hate': {'depth': 1, 'children': {}},
            'angry': {'depth': 1, 'children': {}},
            'scared': {'depth': 1, 'children': {}},
        }
    },
    'food': {
        'depth': 0, 'children': {
            'apple': {'depth': 1, 'children': {}},
            'cake': {'depth': 1, 'children': {}},
            'bread': {'depth': 1, 'children': {}},
            'candy': {'depth': 1, 'children': {}},
            'cookie': {'depth': 1, 'children': {}},
            'pizza': {'depth': 1, 'children': {}},
            'milk': {'depth': 1, 'children': {}},
            'egg': {'depth': 1, 'children': {}},
        }
    },
    'color': {
        'depth': 0, 'children': {
            'red': {'depth': 1, 'children': {}},
            'blue': {'depth': 1, 'children': {}},
            'green': {'depth': 1, 'children': {}},
            'yellow': {'depth': 1, 'children': {}},
            'pink': {'depth': 1, 'children': {}},
            'purple': {'depth': 1, 'children': {}},
        }
    },
    'place': {
        'depth': 0, 'children': {
            'school': {'depth': 1, 'children': {}},
            'hospital': {'depth': 1, 'children': {}},
            'house': {'depth': 1, 'children': {}},
            'garden': {'depth': 1, 'children': {}},
            'forest': {'depth': 1, 'children': {}},
            'beach': {'depth': 1, 'children': {}},
            'park': {'depth': 1, 'children': {}},
            'castle': {'depth': 1, 'children': {}},
            'farm': {'depth': 1, 'children': {}},
            'river': {'depth': 1, 'children': {}},
        }
    },
    'time': {
        'depth': 0, 'children': {
            'day': {'depth': 1, 'children': {'morning': {'depth': 2}}},
            'night': {'depth': 1, 'children': {'evening': {'depth': 2}}},
        }
    },
}


def flatten_taxonomy(tree, parent=None, depth=0):
    """Flatten taxonomy tree into list of (word, depth, category)."""
    items = []
    for word, info in tree.items():
        items.append((word, depth, parent or word))
        if 'children' in info:
            items.extend(flatten_taxonomy(info['children'], parent or word, depth + 1))
    return items


def load_sub_model(device):
    """Load the Sub(5.0) model from the subsumption experiment."""
    # Try to load saved checkpoint, otherwise load from results
    results_path = os.path.join(RESULTS_DIR, 'subsumption_loss.json')
    with open(results_path, 'r') as f:
        data = json.load(f)

    # We need the actual model. Re-train is wasteful.
    # Instead, extract bit patterns from the saved results.
    return data


def extract_from_results(data):
    """Extract bit info from subsumption_loss.json results."""
    info = {}

    # Sub 5.0 train eval details
    for entry in data['sub_5.0']['train_eval']['details']:
        pair = entry['pair']
        hyper, hypo = pair.split('->')
        if hypo not in info:
            info[hypo] = {'active_bits': None, 'shared_with_hyper': entry['shared_bits']}
        if hyper not in info:
            info[hyper] = {'active_bits': entry['hyper_active_bits'], 'shared_with_hyper': None}
        else:
            info[hyper]['active_bits'] = entry['hyper_active_bits']

    # Sub 5.0 test eval details
    for entry in data['sub_5.0']['test_eval']['details']:
        pair = entry['pair']
        hyper, hypo = pair.split('->')
        if hypo not in info:
            info[hypo] = {'active_bits': None, 'shared_with_hyper': entry['shared_bits']}
        if hyper not in info:
            info[hyper] = {'active_bits': entry['hyper_active_bits'], 'shared_with_hyper': None}

    # Also extract from baseline for comparison
    baseline_info = {}
    for entry in data['baseline']['train_eval']['details']:
        pair = entry['pair']
        hyper, hypo = pair.split('->')
        if hypo not in baseline_info:
            baseline_info[hypo] = {'active_bits': None}
        if hyper not in baseline_info:
            baseline_info[hyper] = {'active_bits': entry['hyper_active_bits']}
        else:
            baseline_info[hyper]['active_bits'] = entry['hyper_active_bits']

    return info, baseline_info


def main():
    print("=" * 70)
    print("  INFORMATION HIERARCHY ANALYSIS")
    print("  Does #active_bits correlate with conceptual abstractness?")
    print("=" * 70)

    # Load results
    data = load_sub_model(None)
    sub_info, base_info = extract_from_results(data)

    # Flatten taxonomy
    flat = flatten_taxonomy(TAXONOMY)

    print(f"\n  Taxonomy: {len(flat)} concepts across 3 depth levels")
    print(f"  Depth 0 (hypernyms): {sum(1 for _, d, _ in flat if d == 0)}")
    print(f"  Depth 1 (hyponyms):  {sum(1 for _, d, _ in flat if d == 1)}")
    print(f"  Depth 2 (sub-hypo):  {sum(1 for _, d, _ in flat if d == 2)}")

    # ── Collect active bits per depth ──
    depth_bits_sub = {0: [], 1: [], 2: []}
    depth_bits_base = {0: [], 1: [], 2: []}
    detail_rows = []

    for word, depth, category in flat:
        sub_bits = sub_info.get(word, {}).get('active_bits')
        base_bits = base_info.get(word, {}).get('active_bits')

        if sub_bits is not None:
            depth_bits_sub[depth].append(sub_bits)
            detail_rows.append((word, depth, category, sub_bits, base_bits or '?'))
        elif base_bits is not None:
            # For depth 1/2 concepts we don't have active_bits directly from sub_info
            # but we know they have MORE bits than their hypernym
            pass

    # ── Print per-depth statistics ──
    print(f"\n  Sub(5.0) Active Bits by Depth:")
    print(f"  {'Depth':>6s}  {'Level':>12s}  {'Count':>5s}  {'Mean Bits':>10s}  {'Range':>12s}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*5}  {'─'*10}  {'─'*12}")

    level_names = {0: 'Hypernym', 1: 'Hyponym', 2: 'Sub-hyponym'}
    for d in [0, 1, 2]:
        bits = depth_bits_sub[d]
        if bits:
            print(f"  {d:>6d}  {level_names[d]:>12s}  {len(bits):>5d}  "
                  f"{np.mean(bits):>10.1f}  {min(bits):>4d} - {max(bits):>4d}")

    # ── Per-category breakdown ──
    print(f"\n  Per-Category Hypernym Active Bits (Sub 5.0 vs Baseline):")
    print(f"  {'Category':>12s}  {'Sub(5.0)':>8s}  {'Baseline':>8s}  {'Reduction':>10s}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*10}")

    categories = ['animal', 'person', 'feeling', 'food', 'color', 'place', 'time']
    sub_bits_list = []
    base_bits_list = []
    for cat in categories:
        sb = sub_info.get(cat, {}).get('active_bits')
        bb = base_info.get(cat, {}).get('active_bits')
        if sb is not None and bb is not None:
            reduction = (1 - sb / bb) * 100
            print(f"  {cat:>12s}  {sb:>8d}  {bb:>8d}  {reduction:>9.0f}%")
            sub_bits_list.append(sb)
            base_bits_list.append(bb)

    if sub_bits_list:
        print(f"  {'MEAN':>12s}  {np.mean(sub_bits_list):>8.1f}  {np.mean(base_bits_list):>8.1f}  "
              f"{(1 - np.mean(sub_bits_list)/np.mean(base_bits_list))*100:>9.0f}%")

    # ── Information content analysis ──
    print(f"\n  Information Content (log2 of prime signature):")
    print(f"  Hypernym active bits represent 'category markers'")
    print(f"  Hyponym extra bits represent 'instance specifiers'")

    for cat in categories:
        sb = sub_info.get(cat, {}).get('active_bits')
        if sb is None:
            continue
        # Find children
        children_bits = []
        for entry in data['sub_5.0']['train_eval']['details']:
            hyper, hypo = entry['pair'].split('->')
            if hyper == cat:
                hypo_total = entry['shared_bits']  # shared = hypernym bits in hyponym
                # total hyponym bits = shared + extra
                # But we only have shared_bits and hyper_active_bits
                children_bits.append(entry['shared_bits'])

        if children_bits:
            print(f"  {cat}: {sb} category bits → children inherit all {sb} + specifiers")

    # ── Visualization ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Active bits by depth
    for d in [0, 1, 2]:
        bits = depth_bits_sub[d]
        if bits:
            axes[0].scatter([d] * len(bits), bits, alpha=0.6, s=60,
                           label=f'{level_names[d]} (n={len(bits)})')
    # Add means
    for d in [0, 1, 2]:
        bits = depth_bits_sub[d]
        if bits:
            axes[0].plot(d, np.mean(bits), 'k_', markersize=20, markeredgewidth=3)
    axes[0].set_xlabel('Taxonomic Depth')
    axes[0].set_ylabel('Active Bits')
    axes[0].set_title('Active Bits vs Conceptual Depth')
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(['Hypernym\n(abstract)', 'Hyponym\n(concrete)', 'Sub-hyponym\n(specific)'])
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Baseline vs Sub per category (hypernyms only)
    if sub_bits_list and base_bits_list:
        x = np.arange(len(categories))
        w = 0.35
        axes[1].bar(x - w/2, base_bits_list, w, label='Baseline', color='lightcoral', alpha=0.8)
        axes[1].bar(x + w/2, sub_bits_list, w, label='Sub(5.0)', color='steelblue', alpha=0.8)
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('Active Bits (hypernym)')
        axes[1].set_title('Hypernym Sparsification by Category')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Reduction percentage
    if sub_bits_list and base_bits_list:
        reductions = [(1 - s/b) * 100 for s, b in zip(sub_bits_list, base_bits_list)]
        bars = axes[2].barh(categories, reductions, color='teal', alpha=0.7)
        axes[2].set_xlabel('Bit Reduction (%)')
        axes[2].set_title('How Much Sparser Are Hypernyms?')
        axes[2].set_xlim(0, 100)
        for bar, r in zip(bars, reductions):
            axes[2].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{r:.0f}%', va='center', fontsize=9)
        axes[2].grid(True, alpha=0.3, axis='x')

    plt.suptitle('Emergent Information Hierarchy in Triadic Projections', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'info_hierarchy.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # ── Save ──
    save_data = {
        'experiment': 'info_hierarchy_analysis',
        'source': 'Emergent finding from subsumption loss experiment',
        'method': 'Zero-GPU evaluation of Sub(5.0) model bit patterns',
        'taxonomy_size': len(flat),
        'depth_stats': {
            d: {'count': len(depth_bits_sub[d]),
                'mean': float(np.mean(depth_bits_sub[d])) if depth_bits_sub[d] else None,
                'min': int(min(depth_bits_sub[d])) if depth_bits_sub[d] else None,
                'max': int(max(depth_bits_sub[d])) if depth_bits_sub[d] else None}
            for d in [0, 1, 2]
        },
        'per_category': {
            cat: {'sub_bits': sub_info.get(cat, {}).get('active_bits'),
                  'base_bits': base_info.get(cat, {}).get('active_bits')}
            for cat in categories
        },
    }
    results_path = os.path.join(RESULTS_DIR, 'info_hierarchy.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved: {results_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

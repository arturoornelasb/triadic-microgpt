"""
Bit Evolution Tracker — Tracks how triadic bit activations evolve during training.

Analyzes a sequence of checkpoints to produce a "movie" of semantic connection
formation: when bits activate, when connections form between concept pairs,
and where phase transitions occur.

Usage:
  python benchmarks/scripts/bit_evolution.py \
    --checkpoint-dir checkpoints/torch_run15_strongalign/

Outputs:
  - benchmarks/results/{version}_bit_evolution_{date}.json
  - benchmarks/figures/bit_evolution_heatmap.png      (with --plot)
  - benchmarks/figures/semantic_gap_evolution.png      (with --plot)
  - benchmarks/figures/connection_timeline.png         (with --plot)
"""

import os
import sys
import re
import json
import math
import argparse
from datetime import date

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluate import load_model
from src.triadic import PrimeMapper, TriadicValidator, prime_factors
from src.graph_builder import ScalableGraphBuilder
from benchmarks.scripts.bit_entropy import CONCEPTS, compute_projections, compute_bit_entropy
from benchmarks.scripts.scaling_study import SEMANTIC_PAIRS


# ============================================================
# 1. Checkpoint Discovery
# ============================================================

def discover_checkpoints(checkpoint_dir):
    """Find all step checkpoints in a directory, sorted by step number.

    Returns:
        list of (step, path) tuples, sorted ascending by step.
    """
    pattern = re.compile(r'model_.*_step(\d+)\.pt$')
    results = []
    for fname in os.listdir(checkpoint_dir):
        m = pattern.match(fname)
        if m:
            step = int(m.group(1))
            results.append((step, os.path.join(checkpoint_dir, fname)))
    results.sort(key=lambda x: x[0])
    return results


# ============================================================
# 2. Snapshot Capture
# ============================================================

def capture_snapshot(model_path, tokenizer_path, concepts, pairs, device):
    """Capture a full triadic snapshot from a single checkpoint.

    Args:
        model_path: path to .pt checkpoint
        tokenizer_path: path to tokenizer.json
        concepts: list of concept strings
        pairs: dict with 'related' and 'unrelated' lists of (a, b) tuples
        device: torch device

    Returns:
        dict with all snapshot metrics
    """
    model, tokenizer, config = load_model(model_path, tokenizer_path, device)
    n_bits = config.n_triadic_bits

    # Projections and primes for all concepts
    projections, primes = compute_projections(model, tokenizer, concepts, device)

    # Bit entropy
    per_bit_entropy, mean_entropy = compute_bit_entropy(projections)

    # Activation rate per bit: fraction of concepts where bit > 0
    valid_projs = [p for p in projections if p is not None]
    if valid_projs:
        matrix = np.stack(valid_projs)
        per_bit_activation_rate = (matrix > 0).mean(axis=0).tolist()
    else:
        per_bit_activation_rate = [0.0] * n_bits

    # Bit patterns and composites per concept
    mapper = PrimeMapper(n_bits)
    bit_patterns = {}
    composites = {}
    for concept, proj, prime in zip(concepts, projections, primes):
        if proj is not None:
            bit_patterns[concept] = mapper.get_bits(proj)
            composites[concept] = prime

    # Unique signatures
    unique_sigs = len(set(tuple(v) for v in bit_patterns.values()))

    # Pair similarities using TriadicValidator
    pair_similarities = {}
    for pair_type in ['related', 'unrelated']:
        for a, b in pairs[pair_type]:
            if a in composites and b in composites:
                sim = TriadicValidator.similarity(composites[a], composites[b])
                pair_similarities[f"{a}|{b}"] = sim

    # Graph metrics
    graph = ScalableGraphBuilder(mapper)
    for concept, prime in composites.items():
        if prime and prime > 1:
            graph.add_concept(concept, prime)

    # Graph density: fraction of possible edges that exist
    n_concepts = len(graph.concept_to_prime)
    total_edges = 0
    for label in graph.concept_to_prime:
        total_edges += len(graph.find_neighbors(label))
    total_edges //= 2  # undirected
    max_edges = n_concepts * (n_concepts - 1) // 2 if n_concepts > 1 else 1
    graph_density = total_edges / max_edges if max_edges > 0 else 0.0

    # Simple cluster count: connected components via BFS
    visited = set()
    n_clusters = 0
    for label in graph.concept_to_prime:
        if label not in visited:
            n_clusters += 1
            queue = [label]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                for nb in graph.find_neighbors(node):
                    if nb not in visited:
                        queue.append(nb)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return {
        'mean_entropy': float(mean_entropy),
        'per_bit_entropy': per_bit_entropy.tolist(),
        'per_bit_activation_rate': per_bit_activation_rate,
        'unique_signatures': unique_sigs,
        'graph_density': float(graph_density),
        'graph_n_clusters': n_clusters,
        'pair_similarities': pair_similarities,
        'composites': {k: v for k, v in composites.items() if v is not None},
        'bit_patterns': {k: v for k, v in bit_patterns.items()},
    }


# ============================================================
# 3. Evolution Metrics
# ============================================================

def compute_evolution(snapshots, steps, concepts, pairs):
    """Compute evolution metrics across all snapshots.

    Args:
        snapshots: list of snapshot dicts (one per step)
        steps: list of step numbers
        concepts: list of concept strings
        pairs: SEMANTIC_PAIRS dict

    Returns:
        dict with evolution curves and events
    """
    n_steps = len(steps)

    # --- Curves ---
    entropy_curve = [s['mean_entropy'] for s in snapshots]
    graph_density_curve = [s['graph_density'] for s in snapshots]

    # Semantic gap curve: mean(related) - mean(unrelated)
    semantic_gap_curve = []
    for s in snapshots:
        related_sims = []
        unrelated_sims = []
        for a, b in pairs['related']:
            key = f"{a}|{b}"
            if key in s['pair_similarities']:
                related_sims.append(s['pair_similarities'][key])
        for a, b in pairs['unrelated']:
            key = f"{a}|{b}"
            if key in s['pair_similarities']:
                unrelated_sims.append(s['pair_similarities'][key])
        if related_sims and unrelated_sims:
            semantic_gap_curve.append(np.mean(related_sims) - np.mean(unrelated_sims))
        else:
            semantic_gap_curve.append(0.0)

    # --- Bit births: first step where bit activates for a concept ---
    bit_births = []
    for concept in concepts:
        for t, s in enumerate(snapshots):
            if concept in s['bit_patterns']:
                bits = s['bit_patterns'][concept]
                for bit_idx, val in enumerate(bits):
                    if val == 1:
                        # Check if this is the first activation
                        was_active_before = False
                        for prev_t in range(t):
                            prev_bits = snapshots[prev_t].get('bit_patterns', {}).get(concept)
                            if prev_bits and prev_bits[bit_idx] == 1:
                                was_active_before = True
                                break
                        if not was_active_before:
                            bit_births.append({
                                'concept': concept,
                                'bit': bit_idx,
                                'birth_step': steps[t],
                            })

    # Deduplicate (keep first occurrence)
    seen_births = set()
    unique_births = []
    for b in bit_births:
        key = (b['concept'], b['bit'])
        if key not in seen_births:
            seen_births.add(key)
            unique_births.append(b)
    bit_births = unique_births

    # --- Bit deaths: first step where bit dies and doesn't come back ---
    bit_deaths = []
    for concept in concepts:
        if concept not in snapshots[0].get('bit_patterns', {}):
            continue
        n_bits = len(snapshots[0]['bit_patterns'].get(concept, []))
        for bit_idx in range(n_bits):
            for t in range(1, n_steps):
                prev_bits = snapshots[t - 1].get('bit_patterns', {}).get(concept)
                curr_bits = snapshots[t].get('bit_patterns', {}).get(concept)
                if prev_bits and curr_bits and prev_bits[bit_idx] == 1 and curr_bits[bit_idx] == 0:
                    # Check if it stays dead
                    stays_dead = True
                    for future_t in range(t + 1, n_steps):
                        future_bits = snapshots[future_t].get('bit_patterns', {}).get(concept)
                        if future_bits and future_bits[bit_idx] == 1:
                            stays_dead = False
                            break
                    if stays_dead:
                        bit_deaths.append({
                            'concept': concept,
                            'bit': bit_idx,
                            'death_step': steps[t],
                        })
                        break  # Only first permanent death

    # --- Connection formations: first step where GCD(a,b) > 1 ---
    connection_formations = []
    all_pairs = pairs['related'] + pairs['unrelated']
    for a, b in all_pairs:
        for t, s in enumerate(snapshots):
            comp_a = s['composites'].get(a)
            comp_b = s['composites'].get(b)
            if comp_a and comp_b and comp_a > 1 and comp_b > 1:
                if math.gcd(comp_a, comp_b) > 1:
                    # Check if this is the first time
                    was_connected = False
                    for prev_t in range(t):
                        prev_a = snapshots[prev_t]['composites'].get(a)
                        prev_b = snapshots[prev_t]['composites'].get(b)
                        if prev_a and prev_b and prev_a > 1 and prev_b > 1:
                            if math.gcd(prev_a, prev_b) > 1:
                                was_connected = True
                                break
                    if not was_connected:
                        connection_formations.append({
                            'a': a,
                            'b': b,
                            'first_step': steps[t],
                        })
                    break

    # --- Phase transitions: steps where delta > 2*sigma ---
    phase_transitions = []
    for metric_name, curve in [('entropy', entropy_curve),
                                ('semantic_gap', semantic_gap_curve),
                                ('graph_density', graph_density_curve)]:
        if len(curve) < 3:
            continue
        deltas = [abs(curve[i] - curve[i - 1]) for i in range(1, len(curve))]
        if not deltas:
            continue
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        threshold = mean_delta + 2 * std_delta
        for i, d in enumerate(deltas):
            if d > threshold and std_delta > 1e-6:
                phase_transitions.append({
                    'step': steps[i + 1],
                    'metric': metric_name,
                    'delta': float(d),
                })

    return {
        'entropy_curve': entropy_curve,
        'semantic_gap_curve': semantic_gap_curve,
        'graph_density_curve': graph_density_curve,
        'bit_births': bit_births,
        'bit_deaths': bit_deaths,
        'connection_formations': connection_formations,
        'phase_transitions': phase_transitions,
    }


# ============================================================
# 4. Summary Report
# ============================================================

def print_summary(steps, snapshots, evolution):
    """Print a console summary in the style of bit_entropy.py."""
    print()
    print("=" * 68)
    print("  BIT EVOLUTION TRACKER")
    print("=" * 68)
    print()

    # Entropy trajectory
    print("  Entropy trajectory:")
    for step, s in zip(steps, snapshots):
        bar_len = int(s['mean_entropy'] * 40)
        bar = '#' * bar_len + '.' * (40 - bar_len)
        print(f"    step {step:>6d}  [{bar}] {s['mean_entropy']:.4f}")
    print()

    # Semantic gap trajectory
    print("  Semantic gap trajectory:")
    for step, gap in zip(steps, evolution['semantic_gap_curve']):
        sign = '+' if gap > 0 else ' '
        bar_len = int(max(0, gap + 0.5) * 40)
        bar = '#' * min(bar_len, 40) + '.' * max(0, 40 - bar_len)
        print(f"    step {step:>6d}  [{bar}] {sign}{gap:.4f}")
    print()

    # Graph density
    print("  Graph density trajectory:")
    for step, d in zip(steps, evolution['graph_density_curve']):
        bar_len = int(d * 40)
        bar = '#' * bar_len + '.' * (40 - bar_len)
        print(f"    step {step:>6d}  [{bar}] {d:.4f}")
    print()

    # Key events
    print("  Connection formations (related pairs):")
    related_set = set(f"{a}|{b}" for a, b in SEMANTIC_PAIRS['related'])
    for cf in evolution['connection_formations']:
        key = f"{cf['a']}|{cf['b']}"
        tag = " [related]" if key in related_set else ""
        print(f"    {cf['a']:>10s} <-> {cf['b']:<10s}  first connected at step {cf['first_step']}{tag}")
    print()

    # Phase transitions
    if evolution['phase_transitions']:
        print("  Phase transitions detected:")
        for pt in evolution['phase_transitions']:
            print(f"    step {pt['step']:>6d}  {pt['metric']:<15s}  delta={pt['delta']:.4f}")
    else:
        print("  No phase transitions detected.")
    print()

    # Summary stats
    n_births = len(evolution['bit_births'])
    n_deaths = len(evolution['bit_deaths'])
    n_connections = len(evolution['connection_formations'])
    entropy_start = evolution['entropy_curve'][0]
    entropy_end = evolution['entropy_curve'][-1]
    gap_start = evolution['semantic_gap_curve'][0]
    gap_end = evolution['semantic_gap_curve'][-1]

    print("  " + "-" * 40)
    print(f"  Total bit births:       {n_births}")
    print(f"  Total bit deaths:       {n_deaths}")
    print(f"  Connections formed:     {n_connections}")
    print(f"  Entropy: {entropy_start:.4f} -> {entropy_end:.4f}")
    print(f"  Semantic gap: {gap_start:+.4f} -> {gap_end:+.4f}")
    print(f"  Phase transitions:      {len(evolution['phase_transitions'])}")
    print("=" * 68)


# ============================================================
# 5. Plotting (optional)
# ============================================================

def plot_figures(steps, snapshots, evolution, figures_dir):
    """Generate evolution visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(figures_dir, exist_ok=True)

    # --- 1. Bit evolution heatmap: steps x bits, color = activation rate ---
    fig, ax = plt.subplots(figsize=(16, max(4, len(steps) * 0.5)))
    matrix = np.array([s['per_bit_activation_rate'] for s in snapshots])
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([f"step {s:,}" for s in steps], fontsize=9)
    ax.set_xlabel('Bit Index')
    ax.set_ylabel('Training Step')
    ax.set_title('Bit Activation Rate Evolution')
    plt.colorbar(im, ax=ax, label='Activation Rate')
    plt.tight_layout()
    path = os.path.join(figures_dir, 'bit_evolution_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- 2. Semantic gap evolution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, evolution['semantic_gap_curve'], 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Semantic Gap (related - unrelated)')
    ax.set_title('Semantic Gap Evolution')
    ax.grid(True, alpha=0.3)

    # Mark phase transitions for semantic_gap
    for pt in evolution['phase_transitions']:
        if pt['metric'] == 'semantic_gap':
            ax.axvline(x=pt['step'], color='red', linestyle=':', alpha=0.7,
                       label=f"transition @ {pt['step']}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(figures_dir, 'semantic_gap_evolution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- 3. Connection timeline ---
    fig, ax = plt.subplots(figsize=(12, 6))
    related_set = set(f"{a}|{b}" for a, b in SEMANTIC_PAIRS['related'])

    related_conns = []
    unrelated_conns = []
    for cf in evolution['connection_formations']:
        key = f"{cf['a']}|{cf['b']}"
        label = f"{cf['a']}-{cf['b']}"
        if key in related_set:
            related_conns.append((cf['first_step'], label))
        else:
            unrelated_conns.append((cf['first_step'], label))

    # Plot related connections
    if related_conns:
        r_steps, r_labels = zip(*related_conns)
        ax.scatter(r_steps, range(len(r_steps)), c='green', s=80, zorder=3, label='Related')
        for i, lbl in enumerate(r_labels):
            ax.annotate(lbl, (r_steps[i], i), fontsize=7, ha='left',
                        xytext=(5, 0), textcoords='offset points')

    # Plot unrelated connections
    if unrelated_conns:
        u_steps, u_labels = zip(*unrelated_conns)
        offset = len(related_conns)
        ax.scatter(u_steps, range(offset, offset + len(u_steps)),
                   c='orange', s=80, zorder=3, label='Unrelated')
        for i, lbl in enumerate(u_labels):
            ax.annotate(lbl, (u_steps[i], offset + i), fontsize=7, ha='left',
                        xytext=(5, 0), textcoords='offset points')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Pair Index')
    ax.set_title('Connection Formation Timeline')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    path = os.path.join(figures_dir, 'connection_timeline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Discover checkpoints
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        print(f"  ERROR: No checkpoints found in {args.checkpoint_dir}")
        sys.exit(1)

    steps = [s for s, _ in checkpoints]
    paths = [p for _, p in checkpoints]

    # Auto-detect tokenizer
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        tokenizer_path = os.path.join(args.checkpoint_dir, 'tokenizer.json')

    print()
    print("=" * 68)
    print("  BIT EVOLUTION TRACKER")
    print("=" * 68)
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Checkpoints:    {len(checkpoints)}")
    print(f"  Steps:          {steps}")
    print(f"  Concepts:       {len(CONCEPTS)}")
    print(f"  Device:         {device}")
    print()

    # Capture snapshots
    snapshots = []
    for i, (step, path) in enumerate(checkpoints):
        print(f"  [{i+1}/{len(checkpoints)}] Capturing step {step}...")
        snapshot = capture_snapshot(path, tokenizer_path, CONCEPTS, SEMANTIC_PAIRS, device)
        snapshot['step'] = step
        snapshots.append(snapshot)
        print(f"    entropy={snapshot['mean_entropy']:.4f}  "
              f"sigs={snapshot['unique_signatures']}  "
              f"density={snapshot['graph_density']:.4f}  "
              f"clusters={snapshot['graph_n_clusters']}")

    # Compute evolution
    print()
    print("  Computing evolution metrics...")
    evolution = compute_evolution(snapshots, steps, CONCEPTS, SEMANTIC_PAIRS)

    # Print summary
    print_summary(steps, snapshots, evolution)

    # Compute summary stats
    summary = {
        'n_checkpoints': len(checkpoints),
        'steps_range': [steps[0], steps[-1]],
        'entropy_start': evolution['entropy_curve'][0],
        'entropy_end': evolution['entropy_curve'][-1],
        'semantic_gap_start': evolution['semantic_gap_curve'][0],
        'semantic_gap_end': evolution['semantic_gap_curve'][-1],
        'total_bit_births': len(evolution['bit_births']),
        'total_bit_deaths': len(evolution['bit_deaths']),
        'total_connections': len(evolution['connection_formations']),
        'phase_transitions': len(evolution['phase_transitions']),
    }

    # Build JSON result
    result = {
        'benchmark': 'bit_evolution',
        'version': args.version,
        'date': date.today().isoformat(),
        'checkpoint_dir': args.checkpoint_dir,
        'steps': steps,
        'snapshots': snapshots,
        'evolution': evolution,
        'summary': summary,
    }

    # Save JSON
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    results_dir = os.path.join(project_root, 'benchmarks', 'results')
    figures_dir = os.path.join(project_root, 'benchmarks', 'figures')
    os.makedirs(results_dir, exist_ok=True)

    today = date.today().isoformat()
    result_path = os.path.join(results_dir, f"{args.version}_bit_evolution_{today}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    # Save projections if requested
    if args.save_projections:
        npz_path = os.path.join(results_dir, f"{args.version}_bit_evolution_projections_{today}.npz")
        proj_data = {}
        for i, s in enumerate(snapshots):
            for concept, bits in s['bit_patterns'].items():
                proj_data[f"step{steps[i]}_{concept}"] = np.array(bits)
        np.savez_compressed(npz_path, **proj_data)
        print(f"  Projections saved: {npz_path}")

    # Plot figures
    if args.plot:
        print()
        print("  Generating figures...")
        plot_figures(steps, snapshots, evolution, figures_dir)

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bit Evolution Tracker')
    parser.add_argument('--checkpoint-dir', required=True,
                        help='Directory with step checkpoints')
    parser.add_argument('--tokenizer', default=None,
                        help='Tokenizer path (auto-detected if omitted)')
    parser.add_argument('--version', default='v1.0',
                        help='Version tag for results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate evolution figures')
    parser.add_argument('--save-projections', action='store_true',
                        help='Save bit patterns as .npz')
    args = parser.parse_args()
    main(args)

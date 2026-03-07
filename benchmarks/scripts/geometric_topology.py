"""
Geometric Concept Topology — Experimental metric inspired by UHRT.

Measures how concepts organize into geometric structures in prime-factor space:
  - 0-simplex (point):    Individual concept complexity
  - 1-simplex (line):     Pairwise relationships (shared prime factors)
  - 2-simplex (triangle): Three-way coherence clusters
  - 3-simplex (volume):   Semantic "bubbles" — clusters of related concepts

Also computes adapted UBS (Universal Binary Scale) metrics:
  - UBS(concept) = log2(Φ(x)) + H(bits)  [informational complexity]
  - Stability(A,B,C) = GCD coherence across concept triples
  - Bubble Entropy = Shannon entropy of prime factor distribution within a cluster

This is an EXPLORATORY experiment. Results may or may not be meaningful.
The hypothesis: semantically related concepts form dense simplicial complexes
in prime space, while unrelated concepts remain topologically disconnected.

Usage:
  python benchmarks/scripts/geometric_topology.py \
    --model checkpoints/torch/model_best.pt \
    --tokenizer checkpoints/torch/tokenizer.json
"""

import os
import sys
import json
import math
import argparse
from datetime import date
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluate import load_model
from src.triadic import PrimeMapper, TriadicValidator, prime_factors


# ============================================================
# Concept Vocabulary — organized into semantic domains
# ============================================================

SEMANTIC_DOMAINS = {
    "royalty": ["king", "queen", "prince", "princess", "crown", "throne"],
    "animals": ["dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep"],
    "family": ["mother", "father", "brother", "sister", "son", "daughter"],
    "emotions": ["happy", "sad", "angry", "afraid", "brave", "kind", "love", "hate"],
    "nature": ["sun", "moon", "star", "river", "mountain", "ocean", "tree", "flower"],
    "elements": ["fire", "water", "earth", "air", "rain", "snow", "cloud", "wind"],
    "body": ["hand", "head", "eye", "heart", "foot", "mouth", "ear", "nose"],
    "home": ["house", "door", "window", "table", "chair", "bed", "lamp", "room"],
    "food": ["bread", "milk", "apple", "cake", "egg", "soup", "rice", "meat"],
    "actions": ["run", "walk", "swim", "fly", "jump", "climb", "fall", "sleep"],
    "colors": ["red", "blue", "green", "white", "black", "yellow", "pink", "gold"],
    "professions": ["doctor", "teacher", "nurse", "farmer", "soldier", "artist"],
}


# ============================================================
# Metric Functions
# ============================================================

def compute_ubs(prime_value, bits_array):
    """
    Adapted Universal Binary Scale for a concept.

    UBS(x) = log2(Φ(x)) + H(bits)

    - log2(Φ(x)): informational "size" of the prime signature
    - H(bits): entropy of the bit pattern (how diverse is the encoding)

    Higher UBS = more complex concept (more prime factors, more diverse bits).
    """
    if prime_value <= 1:
        return 0.0

    log_term = math.log2(prime_value) if prime_value > 0 else 0.0

    # Bit entropy: treat positive activations as probabilities
    bits = np.array(bits_array)
    probs = (bits + 1.0) / 2.0  # tanh[-1,1] → [0,1]
    eps = 1e-10
    entropy = -np.mean(probs * np.log2(probs + eps) + (1 - probs) * np.log2(1 - probs + eps))

    return log_term + entropy


def compute_simplex_0(concepts, primes, projections):
    """
    0-simplex analysis: individual concept complexity.
    Returns UBS for each concept.
    """
    results = {}
    for concept, prime, proj in zip(concepts, primes, projections):
        if prime is None or proj is None:
            continue
        factors = prime_factors(prime)
        results[concept] = {
            "prime": prime,
            "n_factors": len(factors),
            "ubs": compute_ubs(prime, proj),
            "bit_entropy": float(-np.mean(
                ((proj + 1) / 2) * np.log2((proj + 1) / 2 + 1e-10) +
                (1 - (proj + 1) / 2) * np.log2(1 - (proj + 1) / 2 + 1e-10)
            )),
        }
    return results


def compute_simplex_1(concepts, primes):
    """
    1-simplex analysis: pairwise relationships.
    An edge exists if GCD(Φ(A), Φ(B)) > 1 (shared factors).
    Edge weight = Jaccard similarity of prime factors.
    """
    validator = TriadicValidator()
    edges = []

    valid = [(c, p) for c, p in zip(concepts, primes) if p is not None and p > 1]

    for (c1, p1), (c2, p2) in combinations(valid, 2):
        if p1 == p2:
            sim = 1.0
        else:
            sim = validator.similarity(p1, p2)

        if sim > 0:
            shared = math.gcd(p1, p2)
            edges.append({
                "a": c1, "b": c2,
                "similarity": sim,
                "shared_gcd": shared,
                "n_shared_factors": len(prime_factors(shared)),
                "subsumes_ab": p1 % p2 == 0,
                "subsumes_ba": p2 % p1 == 0,
            })

    return edges


def compute_simplex_2(concepts, primes, domain_map):
    """
    2-simplex analysis: triangles (3-way coherence).
    A triangle exists if all three pairwise GCDs share at least one common factor.
    Measures: do concepts within a domain form coherent triangles?
    """
    validator = TriadicValidator()
    triangles = []

    valid = {c: p for c, p in zip(concepts, primes) if p is not None and p > 1}

    for domain_name, domain_concepts in domain_map.items():
        domain_valid = [(c, valid[c]) for c in domain_concepts if c in valid]
        if len(domain_valid) < 3:
            continue

        for (c1, p1), (c2, p2), (c3, p3) in combinations(domain_valid, 3):
            gcd_12 = math.gcd(p1, p2)
            gcd_13 = math.gcd(p1, p3)
            gcd_23 = math.gcd(p2, p3)
            gcd_all = math.gcd(gcd_12, p3)  # GCD of all three

            # Triangle coherence: do all three share at least one factor?
            coherent = gcd_all > 1

            # Average pairwise similarity
            sim_12 = validator.similarity(p1, p2)
            sim_13 = validator.similarity(p1, p3)
            sim_23 = validator.similarity(p2, p3)
            avg_sim = (sim_12 + sim_13 + sim_23) / 3

            triangles.append({
                "domain": domain_name,
                "concepts": [c1, c2, c3],
                "coherent": coherent,
                "gcd_all": gcd_all,
                "avg_similarity": avg_sim,
                "n_shared_factors_all": len(prime_factors(gcd_all)),
            })

    return triangles


def compute_bubbles(concepts, primes, domain_map):
    """
    Semantic bubbles: full-domain cluster analysis.
    For each domain, compute:
    - Intra-domain density (avg similarity within domain)
    - Inter-domain separation (avg similarity to other domains)
    - Bubble entropy (Shannon entropy of prime factor distribution)
    - Bubble UBS (aggregate informational complexity)
    """
    validator = TriadicValidator()
    valid = {c: p for c, p in zip(concepts, primes) if p is not None and p > 1}

    bubbles = {}
    domain_means = {}

    for domain_name, domain_concepts in domain_map.items():
        domain_primes = [valid[c] for c in domain_concepts if c in valid]
        if len(domain_primes) < 2:
            continue

        # Intra-domain similarity
        intra_sims = []
        for p1, p2 in combinations(domain_primes, 2):
            intra_sims.append(validator.similarity(p1, p2))
        avg_intra = np.mean(intra_sims) if intra_sims else 0.0

        # All prime factors in this domain
        all_factors = []
        for p in domain_primes:
            all_factors.extend(prime_factors(p))
        factor_counts = defaultdict(int)
        for f in all_factors:
            factor_counts[f] += 1

        # Bubble entropy: distribution of prime factors
        total = sum(factor_counts.values())
        if total > 0:
            probs = [c / total for c in factor_counts.values()]
            bubble_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        else:
            bubble_entropy = 0.0

        # Bubble UBS: log2(LCM of all domain primes) + bubble entropy
        domain_lcm = domain_primes[0]
        for p in domain_primes[1:]:
            domain_lcm = (domain_lcm * p) // math.gcd(domain_lcm, p)
        bubble_ubs = math.log2(domain_lcm) + bubble_entropy if domain_lcm > 0 else 0.0

        # Domain centroid (GCD of all = shared backbone)
        domain_gcd = domain_primes[0]
        for p in domain_primes[1:]:
            domain_gcd = math.gcd(domain_gcd, p)

        bubbles[domain_name] = {
            "n_concepts": len(domain_primes),
            "avg_intra_similarity": float(avg_intra),
            "bubble_entropy": float(bubble_entropy),
            "bubble_ubs": float(bubble_ubs),
            "shared_backbone_gcd": domain_gcd,
            "n_shared_factors": len(prime_factors(domain_gcd)),
            "n_unique_factors": len(factor_counts),
        }
        domain_means[domain_name] = domain_primes

    # Inter-domain separation
    domain_names = list(bubbles.keys())
    for i, d1 in enumerate(domain_names):
        inter_sims = []
        for j, d2 in enumerate(domain_names):
            if i == j:
                continue
            for p1 in domain_means[d1]:
                for p2 in domain_means[d2]:
                    inter_sims.append(validator.similarity(p1, p2))
        bubbles[d1]["avg_inter_similarity"] = float(np.mean(inter_sims)) if inter_sims else 0.0
        # Separation ratio: intra/inter (higher = better separation)
        if bubbles[d1]["avg_inter_similarity"] > 0:
            bubbles[d1]["separation_ratio"] = bubbles[d1]["avg_intra_similarity"] / bubbles[d1]["avg_inter_similarity"]
        else:
            bubbles[d1]["separation_ratio"] = float('inf') if bubbles[d1]["avg_intra_similarity"] > 0 else 1.0

    return bubbles


# ============================================================
# Visualization
# ============================================================

def plot_bubble_comparison(bubbles, output_path):
    """Plot intra vs inter similarity per domain."""
    domains = sorted(bubbles.keys())
    intra = [bubbles[d]["avg_intra_similarity"] for d in domains]
    inter = [bubbles[d]["avg_inter_similarity"] for d in domains]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, intra, width, label='Intra-domain (cohesion)', color='steelblue')
    bars2 = ax.bar(x + width/2, inter, width, label='Inter-domain (separation)', color='salmon')

    ax.set_xlabel('Semantic Domain')
    ax.set_ylabel('Average Similarity')
    ax.set_title('Semantic Bubble Analysis: Cohesion vs Separation')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Bubble plot saved: {output_path}")


def plot_ubs_distribution(simplex_0, output_path):
    """Plot UBS distribution across concepts."""
    ubs_values = [v["ubs"] for v in simplex_0.values()]
    names = list(simplex_0.keys())

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(ubs_values)), sorted(ubs_values, reverse=True), color='teal', alpha=0.8)
    ax.set_xlabel('Concept (sorted by UBS)')
    ax.set_ylabel('UBS (bits)')
    ax.set_title('Concept Informational Complexity (UBS = log2(Φ) + H(bits))')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  UBS distribution saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 70)
    print("  GEOMETRIC CONCEPT TOPOLOGY — Experimental (UHRT-inspired)")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print()

    # Load model
    model, tokenizer, config = load_model(args.model, args.tokenizer, device)
    mapper = PrimeMapper(config.n_triadic_bits)
    print(f"  Config: {config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits")

    # Flatten all concepts
    all_concepts = []
    concept_to_domain = {}
    for domain, concepts in SEMANTIC_DOMAINS.items():
        for c in concepts:
            all_concepts.append(c)
            concept_to_domain[c] = domain

    print(f"  Concepts: {len(all_concepts)} across {len(SEMANTIC_DOMAINS)} domains")
    print()

    # Compute projections
    print("  [1/5] Computing triadic projections...")
    projections = []
    primes = []
    for concept in all_concepts:
        ids = tokenizer.encode(concept, add_special=False)
        if not ids:
            projections.append(None)
            primes.append(None)
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)
        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        projections.append(proj)
        primes.append(mapper.map(proj))

    # 0-simplex: Point analysis
    print("  [2/5] 0-simplex analysis (concept complexity)...")
    simplex_0 = compute_simplex_0(all_concepts, primes, projections)
    ubs_values = [v["ubs"] for v in simplex_0.values()]
    print(f"    Mean UBS: {np.mean(ubs_values):.2f} bits")
    print(f"    Std UBS:  {np.std(ubs_values):.2f} bits")
    print(f"    Range:    [{min(ubs_values):.2f}, {max(ubs_values):.2f}]")

    # 1-simplex: Edge analysis
    print("  [3/5] 1-simplex analysis (pairwise relationships)...")
    edges = compute_simplex_1(all_concepts, primes)
    connected = len([e for e in edges if e["similarity"] > 0])
    total_pairs = len(all_concepts) * (len(all_concepts) - 1) // 2
    print(f"    Connected pairs: {connected}/{total_pairs} ({connected/total_pairs:.1%})")
    if edges:
        sims = [e["similarity"] for e in edges]
        print(f"    Mean similarity: {np.mean(sims):.4f}")
        subsumptions = sum(1 for e in edges if e["subsumes_ab"] or e["subsumes_ba"])
        print(f"    Subsumption pairs: {subsumptions}")

    # 2-simplex: Triangle analysis
    print("  [4/5] 2-simplex analysis (triangular coherence)...")
    triangles = compute_simplex_2(all_concepts, primes, SEMANTIC_DOMAINS)
    if triangles:
        coherent = sum(1 for t in triangles if t["coherent"])
        total_tri = len(triangles)
        print(f"    Total triangles tested: {total_tri}")
        print(f"    Coherent (shared GCD > 1): {coherent}/{total_tri} ({coherent/total_tri:.1%})")
        avg_tri_sim = np.mean([t["avg_similarity"] for t in triangles])
        print(f"    Mean triangle similarity: {avg_tri_sim:.4f}")

        # Per-domain breakdown
        domain_coherence = defaultdict(list)
        for t in triangles:
            domain_coherence[t["domain"]].append(t["coherent"])
        print(f"    Per-domain coherence:")
        for d in sorted(domain_coherence.keys()):
            rate = np.mean(domain_coherence[d])
            print(f"      {d:>15s}: {rate:.0%}")

    # Bubble analysis
    print("  [5/5] Semantic bubble analysis (domain clusters)...")
    bubbles = compute_bubbles(all_concepts, primes, SEMANTIC_DOMAINS)
    print(f"\n    {'Domain':>15s} | {'Intra':>6s} | {'Inter':>6s} | {'Sep.':>6s} | {'Entropy':>8s} | {'UBS':>8s}")
    print(f"    {'─'*15} | {'─'*6} | {'─'*6} | {'─'*6} | {'─'*8} | {'─'*8}")
    for domain in sorted(bubbles.keys()):
        b = bubbles[domain]
        sep = f"{b['separation_ratio']:.2f}" if b['separation_ratio'] != float('inf') else "inf"
        print(f"    {domain:>15s} | {b['avg_intra_similarity']:>5.1%} | {b['avg_inter_similarity']:>5.1%} | {sep:>6s} | {b['bubble_entropy']:>8.3f} | {b['bubble_ubs']:>8.1f}")

    # Save results
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    results_dir = os.path.join(project_root, 'benchmarks', 'results')
    figures_dir = os.path.join(project_root, 'benchmarks', 'figures')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    version = args.version
    today = date.today().isoformat()

    result = {
        "benchmark": "geometric_topology",
        "version": version,
        "date": today,
        "model_checkpoint": args.model,
        "model_config": f"{config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits",
        "hypothesis": "Semantically related concepts form dense simplicial complexes in prime space",
        "metrics": {
            "simplex_0": {
                "mean_ubs": float(np.mean(ubs_values)),
                "std_ubs": float(np.std(ubs_values)),
                "unique_primes": len(set(p for p in primes if p is not None)),
                "total_concepts": len(all_concepts),
            },
            "simplex_1": {
                "connected_pairs": connected,
                "total_pairs": total_pairs,
                "connectivity": connected / total_pairs if total_pairs > 0 else 0,
                "mean_similarity": float(np.mean([e["similarity"] for e in edges])) if edges else 0,
                "subsumption_count": sum(1 for e in edges if e["subsumes_ab"] or e["subsumes_ba"]),
            },
            "simplex_2": {
                "total_triangles": len(triangles),
                "coherent_triangles": sum(1 for t in triangles if t["coherent"]),
                "coherence_rate": sum(1 for t in triangles if t["coherent"]) / max(len(triangles), 1),
            },
            "bubbles": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))} for k, v in bubbles.items()},
        },
        "interpretation": {
            "note": "This is an EXPLORATORY experiment. High separation_ratio (intra >> inter) validates the hypothesis. separation_ratio ~1.0 means domains are NOT differentiated in prime space.",
            "uhrt_origin": "Adapted from Unified Holographic Resonance Theory (UHRT) layered structure: point→line→triangle→volume→bubble",
        },
    }

    result_path = os.path.join(results_dir, f"{version}_geometric_topology_{today}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    # Plots
    plot_bubble_comparison(bubbles, os.path.join(figures_dir, "bubble_cohesion_separation.png"))
    plot_ubs_distribution(simplex_0, os.path.join(figures_dir, "ubs_distribution.png"))

    # Verdict
    print()
    print("=" * 70)
    if bubbles:
        mean_sep = np.mean([b["separation_ratio"] for b in bubbles.values()
                            if b["separation_ratio"] != float('inf')])
        print(f"  Mean Separation Ratio: {mean_sep:.2f}")
        if mean_sep > 1.5:
            print(f"  FINDING: Domains form distinct semantic bubbles in prime space!")
        elif mean_sep > 1.0:
            print(f"  FINDING: Weak domain separation detected. Some structure present.")
        else:
            print(f"  FINDING: No meaningful domain separation. Triadic collapse likely active.")
    coherence_rate = sum(1 for t in triangles if t["coherent"]) / max(len(triangles), 1)
    print(f"  Triangle Coherence: {coherence_rate:.0%} of intra-domain triples share factors")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Geometric Concept Topology (UHRT-inspired)')
    parser.add_argument('--model', required=True, help='Model checkpoint path')
    parser.add_argument('--tokenizer', default=None, help='Tokenizer path')
    parser.add_argument('--version', default='v1.1', help='Version tag')
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = os.path.join(os.path.dirname(args.model), 'tokenizer.json')

    main(args)

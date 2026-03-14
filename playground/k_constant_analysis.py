"""
P0 — K-Constant Analysis (from La Danza Cosmica, Cap. 25)

The Rule of Three: C4 = (a * C2 * C3) / (b * C1)
K = 1/(a*b) measures "truth" of a proportional relationship.

This script computes K for all learned analogy triples and checks
whether K correlates with semantic quality (similarity gap).

NO TRAINING REQUIRED — pure evaluation on Run 15 checkpoint.
"""

import os
import sys
import json
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
from src.triadic import PrimeMapper, TriadicValidator, prime_factors
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'model_L12_D512_B64_best.pt')
TOKENIZER = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'playground', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Analogy triples: (A, B, C, expected_D)
# A:B :: C:D  =>  K measures how well the model preserves this
# ============================================================

ANALOGY_TRIPLES = [
    # Classic word analogies
    ("king", "queen", "man", "woman"),
    ("king", "queen", "boy", "girl"),
    ("father", "mother", "brother", "sister"),
    ("father", "mother", "son", "daughter"),
    ("dog", "puppy", "cat", "kitten"),
    ("big", "small", "tall", "short"),
    ("hot", "cold", "day", "night"),
    ("happy", "sad", "love", "hate"),
    ("doctor", "hospital", "teacher", "school"),
    ("sun", "day", "moon", "night"),
    # Domain-specific (TinyStories)
    ("princess", "prince", "queen", "king"),
    ("forest", "tree", "ocean", "fish"),
    ("bird", "fly", "fish", "swim"),
    ("red", "blue", "green", "yellow"),
    ("old", "young", "big", "small"),
]

# Additional concept pairs for similarity analysis
DOMAIN_CONCEPTS = {
    'family': ["mother", "father", "sister", "brother", "son", "daughter", "baby"],
    'animals': ["dog", "cat", "bird", "fish", "bear", "rabbit", "horse"],
    'colors': ["red", "blue", "green", "yellow", "black", "white", "pink"],
    'emotions': ["happy", "sad", "angry", "scared", "love", "hate", "kind"],
    'royalty': ["king", "queen", "prince", "princess", "castle", "crown", "throne"],
    'nature': ["sun", "moon", "star", "tree", "flower", "river", "mountain"],
}


def load_model(device):
    """Load Run 15 model and tokenizer."""
    tokenizer = BPETokenizer.load(TOKENIZER)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'], block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'],
        n_head=cfg['n_head'], n_triadic_bits=cfg['n_triadic_bits'],
        dropout=0.0,
    )
    model = TriadicGPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer, config


def get_triadic_signature(model, tokenizer, word, device):
    """Get triadic projection and prime composite for a word."""
    ids = tokenizer.encode(word, add_special=False)
    if not ids:
        return None, None
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, triadic_proj, _ = model(x)
    proj = triadic_proj[0].mean(dim=0).cpu().numpy()
    return proj, proj


def compute_k_constant(phi_a, phi_b, phi_c, phi_d):
    """
    Compute K-constant for an analogy A:B :: C:D.

    The Rule of Three says: D = (a * B * C) / (b * A)
    K = 1/(a*b) measures truth of the relationship.

    In triadic space, we measure this via cosine similarities:
      sim(A,B) should correlate with sim(C,D)   [parallel relationship]
      sim(A,C) should correlate with sim(B,D)   [role relationship]

    K = sim(A,B) * sim(C,D) / (sim(A,C) * sim(B,D) + 1e-10)
    Perfect analogy => K = 1.0
    """
    def cosine(x, y):
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if nx < 1e-10 or ny < 1e-10:
            return 0.0
        return float(np.dot(x, y) / (nx * ny))

    sim_ab = cosine(phi_a, phi_b)
    sim_cd = cosine(phi_c, phi_d)
    sim_ac = cosine(phi_a, phi_c)
    sim_bd = cosine(phi_b, phi_d)

    # K measures parallelism of the analogy
    denom = abs(sim_ac * sim_bd) + 1e-10
    k = (sim_ab * sim_cd) / denom

    # Also compute the "offset vector" analogy quality
    # If A:B :: C:D, then B-A should be similar to D-C
    offset_ab = phi_b - phi_a
    offset_cd = phi_d - phi_c
    offset_sim = cosine(offset_ab, offset_cd)

    return {
        'K': float(k),
        'sim_AB': float(sim_ab),
        'sim_CD': float(sim_cd),
        'sim_AC': float(sim_ac),
        'sim_BD': float(sim_bd),
        'offset_similarity': float(offset_sim),
    }


def compute_algebraic_k(mapper, a_proj, b_proj, c_proj, d_proj):
    """Compute K using prime algebra (the book's method)."""
    phi_a = mapper.map(a_proj)
    phi_b = mapper.map(b_proj)
    phi_c = mapper.map(c_proj)
    phi_d = mapper.map(d_proj)

    # Algebraic analogy: A:B :: C:? => predicted_D
    predicted = TriadicValidator.analogy(phi_a, phi_b, phi_c)

    # Similarity between predicted and actual D
    sim_pred_actual = TriadicValidator.similarity(predicted, phi_d)

    # GCD-based K: how much structure is shared
    shared_ab = math.gcd(phi_a, phi_b)
    shared_cd = math.gcd(phi_c, phi_d)

    factors_shared_ab = len(prime_factors(shared_ab))
    factors_shared_cd = len(prime_factors(shared_cd))

    return {
        'phi_A': phi_a,
        'phi_B': phi_b,
        'phi_C': phi_c,
        'phi_D': phi_d,
        'predicted_D': predicted,
        'sim_predicted_actual': float(sim_pred_actual),
        'shared_factors_AB': factors_shared_ab,
        'shared_factors_CD': factors_shared_cd,
        'exact_match': predicted == phi_d,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 64)
    print("  K-CONSTANT ANALYSIS (La Danza Cosmica, Cap. 25)")
    print("=" * 64)
    print(f"  Device: {device}")

    # Load model
    print("\nLoading Run 15 model...")
    model, tokenizer, config = load_model(device)
    mapper = PrimeMapper(config.n_triadic_bits)
    print(f"  Config: {config.n_layer}L/{config.n_embd}D/{config.n_triadic_bits} bits")

    # ---- Part 1: Analogy K-Constants ----
    print("\n" + "=" * 64)
    print("  PART 1: K-Constants for Analogy Triples")
    print("=" * 64)

    analogy_results = []
    for a_word, b_word, c_word, d_word in ANALOGY_TRIPLES:
        projs = {}
        skip = False
        for label, word in [('A', a_word), ('B', b_word), ('C', c_word), ('D', d_word)]:
            proj, _ = get_triadic_signature(model, tokenizer, word, device)
            if proj is None:
                print(f"  SKIP: '{word}' not in vocabulary")
                skip = True
                break
            projs[label] = proj
        if skip:
            continue

        # Cosine-based K
        k_result = compute_k_constant(projs['A'], projs['B'], projs['C'], projs['D'])

        # Algebraic K
        alg_result = compute_algebraic_k(mapper, projs['A'], projs['B'], projs['C'], projs['D'])

        result = {
            'analogy': f"{a_word}:{b_word} :: {c_word}:{d_word}",
            **k_result,
            **alg_result,
        }
        analogy_results.append(result)

        print(f"\n  {a_word}:{b_word} :: {c_word}:{d_word}")
        print(f"    K = {k_result['K']:.4f}  |  offset_sim = {k_result['offset_similarity']:.4f}")
        print(f"    Algebraic: pred={alg_result['predicted_D']}, actual={alg_result['phi_D']}, "
              f"sim={alg_result['sim_predicted_actual']:.2%}")

    # ---- Part 2: Domain Separation via K ----
    print("\n" + "=" * 64)
    print("  PART 2: Intra-domain vs Inter-domain K Analysis")
    print("=" * 64)

    # Get all concept signatures
    concept_sigs = {}
    for domain, words in DOMAIN_CONCEPTS.items():
        for word in words:
            proj, _ = get_triadic_signature(model, tokenizer, word, device)
            if proj is not None:
                concept_sigs[word] = {'proj': proj, 'domain': domain}

    # Compute pairwise similarities within and across domains
    intra_sims = []
    inter_sims = []
    domain_stats = {}

    domains = list(DOMAIN_CONCEPTS.keys())
    for d in domains:
        domain_words = [w for w in DOMAIN_CONCEPTS[d] if w in concept_sigs]
        d_sims = []
        for i in range(len(domain_words)):
            for j in range(i + 1, len(domain_words)):
                w1, w2 = domain_words[i], domain_words[j]
                p1, p2 = concept_sigs[w1]['proj'], concept_sigs[w2]['proj']
                sim = float(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-10))
                intra_sims.append(sim)
                d_sims.append(sim)
        domain_stats[d] = {
            'n_words': len(domain_words),
            'mean_intra_sim': float(np.mean(d_sims)) if d_sims else 0,
            'std_intra_sim': float(np.std(d_sims)) if d_sims else 0,
        }

    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            words_i = [w for w in DOMAIN_CONCEPTS[domains[i]] if w in concept_sigs]
            words_j = [w for w in DOMAIN_CONCEPTS[domains[j]] if w in concept_sigs]
            for wi in words_i:
                for wj in words_j:
                    p1, p2 = concept_sigs[wi]['proj'], concept_sigs[wj]['proj']
                    sim = float(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-10))
                    inter_sims.append(sim)

    separation = np.mean(intra_sims) / (np.mean(inter_sims) + 1e-10)

    print(f"\n  Intra-domain sim: {np.mean(intra_sims):.4f} +/- {np.std(intra_sims):.4f}")
    print(f"  Inter-domain sim: {np.mean(inter_sims):.4f} +/- {np.std(inter_sims):.4f}")
    print(f"  Separation ratio: {separation:.4f}")

    print(f"\n  Per-domain:")
    for d, stats in domain_stats.items():
        print(f"    {d:>10s}: intra={stats['mean_intra_sim']:.4f} ({stats['n_words']} words)")

    # ---- Part 3: Plots ----
    print("\n  Generating plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: K values distribution
    k_values = [r['K'] for r in analogy_results]
    offset_sims = [r['offset_similarity'] for r in analogy_results]
    labels = [r['analogy'].split('::')[0].strip() for r in analogy_results]

    ax = axes[0]
    bars = ax.barh(range(len(k_values)), k_values, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('K-Constant')
    ax.set_title('Rule of Three: K per Analogy')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='K=1 (perfect)')
    ax.legend(fontsize=8)

    # Plot 2: Offset similarity (vector analogy quality)
    ax = axes[1]
    ax.barh(range(len(offset_sims)), offset_sims, color='coral', alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Offset Similarity (B-A vs D-C)')
    ax.set_title('Vector Analogy Quality')

    # Plot 3: Intra vs Inter domain
    ax = axes[2]
    ax.hist(intra_sims, bins=30, alpha=0.6, label=f'Intra ({np.mean(intra_sims):.3f})', color='green')
    ax.hist(inter_sims, bins=30, alpha=0.6, label=f'Inter ({np.mean(inter_sims):.3f})', color='red')
    ax.set_xlabel('Cosine Similarity (triadic space)')
    ax.set_ylabel('Count')
    ax.set_title(f'Domain Separation (ratio={separation:.3f})')
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'k_constant_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # ---- Save results ----
    results = {
        'experiment': 'k_constant_analysis',
        'source': 'La Danza Cosmica Cap. 25 — Rule of Three',
        'model': 'Run 15 v1.4-strongalign',
        'analogy_results': analogy_results,
        'domain_stats': domain_stats,
        'separation_ratio': float(separation),
        'mean_K': float(np.mean(k_values)),
        'mean_offset_sim': float(np.mean(offset_sims)),
        'mean_intra_sim': float(np.mean(intra_sims)),
        'mean_inter_sim': float(np.mean(inter_sims)),
    }

    results_path = os.path.join(RESULTS_DIR, 'k_constant_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    # ---- Summary ----
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  Mean K-constant:       {np.mean(k_values):.4f}")
    print(f"  Mean offset sim:       {np.mean(offset_sims):.4f}")
    print(f"  Algebraic exact match: {sum(1 for r in analogy_results if r['exact_match'])}/{len(analogy_results)}")
    print(f"  Mean pred-actual sim:  {np.mean([r['sim_predicted_actual'] for r in analogy_results]):.2%}")
    print(f"  Domain separation:     {separation:.4f}")
    print("=" * 64)


if __name__ == '__main__':
    main()

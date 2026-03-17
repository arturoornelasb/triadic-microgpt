"""
E6 — Meaningful Compression Benchmark.

The paper claims "8x compression (64 bits match 512D embeddings)" but prior
probes achieved ~8% accuracy (near random 7.7% for 13 categories), making the
claim weak. This script tests with richer tasks and more words per category
where both representations should perform well above random.

Tasks:
  A. Category Classification (Nearest Centroid, leave-one-out)
  B. Word Similarity Ranking (Spearman correlation, triadic vs embedding)
  C. Intra/Inter Category Separation Ratio
  D. k-NN Classification (k=3)

Zero GPU training — evaluation only on the Run 15 checkpoint.

Usage:
  python playground/compression_benchmark.py
  python playground/compression_benchmark.py --checkpoint path/to/model.pt
"""

import os
import sys
import json
import math
import random
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'playground', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Category definitions (TinyStories-friendly vocabulary) ──

CATEGORIES = {
    'animals': [
        'dog', 'cat', 'bird', 'fish', 'bear', 'horse', 'rabbit', 'frog',
        'mouse', 'duck', 'cow', 'pig', 'sheep', 'chicken', 'monkey',
        'elephant', 'lion', 'tiger', 'wolf', 'fox',
    ],
    'people': [
        'king', 'queen', 'princess', 'prince', 'mother', 'father',
        'brother', 'sister', 'baby', 'boy', 'girl', 'friend',
        'doctor', 'teacher', 'farmer',
    ],
    'emotions': [
        'happy', 'sad', 'angry', 'scared', 'brave', 'kind', 'nice',
        'mean', 'proud', 'shy', 'lonely', 'excited', 'worried',
        'tired', 'hungry',
    ],
    'colors': [
        'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'black',
        'white', 'brown', 'orange', 'golden',
    ],
    'nature': [
        'sun', 'moon', 'star', 'tree', 'flower', 'river', 'mountain',
        'rain', 'wind', 'snow', 'cloud', 'sky', 'sea', 'forest', 'garden',
    ],
    'food': [
        'cake', 'candy', 'cookie', 'apple', 'bread', 'cheese', 'milk',
        'water', 'juice', 'soup', 'pizza', 'ice',
    ],
    'body': [
        'hand', 'eye', 'head', 'heart', 'leg', 'arm', 'nose', 'ear',
        'mouth', 'face', 'hair', 'tooth',
    ],
    'home': [
        'house', 'door', 'window', 'bed', 'chair', 'table', 'room',
        'floor', 'wall', 'garden', 'kitchen', 'box', 'toy',
    ],
    'actions': [
        'run', 'walk', 'fly', 'swim', 'jump', 'play', 'sing', 'dance',
        'sleep', 'eat', 'drink', 'read', 'write', 'climb', 'hide',
    ],
}


# ── Helpers ──

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_similarity_matrix(X):
    """Pairwise cosine similarity matrix for rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_normed = X / norms
    return X_normed @ X_normed.T


def spearman_rank_correlation(x, y):
    """Compute Spearman rank correlation between two arrays."""
    n = len(x)
    if n < 3:
        return 0.0
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    d = rank_x - rank_y
    return float(1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1)))


# ── Feature extraction ──

def extract_features(model, tokenizer, device):
    """
    For each word in CATEGORIES, extract:
      - triadic projection (n_bits dim) via tanh output, mean over tokens
      - embedding (n_embd dim) via wte, mean over tokens

    Returns:
      words: list of valid words
      labels: list of category names (parallel to words)
      triadic_feats: (N, n_bits) array
      embed_feats: (N, n_embd) array
    """
    model.eval()
    words, labels = [], []
    triadic_list, embed_list = [], []

    with torch.no_grad():
        for category, word_list in CATEGORIES.items():
            for word in word_list:
                ids = tokenizer.encode(word, add_special=False)
                if not ids:
                    continue
                x = torch.tensor([ids], dtype=torch.long, device=device)
                _, triadic_proj, _ = model(x)
                embedding = model.wte(x)

                tri_vec = triadic_proj[0].mean(dim=0).cpu().numpy()
                emb_vec = embedding[0].mean(dim=0).cpu().numpy()

                words.append(word)
                labels.append(category)
                triadic_list.append(tri_vec)
                embed_list.append(emb_vec)

    triadic_feats = np.array(triadic_list)
    embed_feats = np.array(embed_list)
    return words, labels, triadic_feats, embed_feats


# ── Task A: Nearest Centroid Classification (leave-one-out) ──

def task_nearest_centroid(feats, labels):
    """
    Leave-one-out nearest centroid classifier.
    For each sample, compute category centroids excluding that sample,
    then predict by nearest centroid.
    """
    unique_cats = sorted(set(labels))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    y = np.array([cat_to_idx[l] for l in labels])
    n = len(y)
    n_cats = len(unique_cats)

    # Precompute category sums and counts
    cat_sums = np.zeros((n_cats, feats.shape[1]))
    cat_counts = np.zeros(n_cats)
    for i in range(n):
        cat_sums[y[i]] += feats[i]
        cat_counts[y[i]] += 1

    correct = 0
    for i in range(n):
        c = y[i]
        # Centroid of category c excluding sample i
        centroids = cat_sums.copy()
        counts = cat_counts.copy()
        centroids[c] -= feats[i]
        counts[c] -= 1

        # Compute distance to each centroid
        best_cat = -1
        best_sim = -2.0
        for ci in range(n_cats):
            if counts[ci] < 1:
                continue
            centroid = centroids[ci] / counts[ci]
            sim = cosine_similarity(feats[i], centroid)
            if sim > best_sim:
                best_sim = sim
                best_cat = ci

        if best_cat == c:
            correct += 1

    return correct / n


# ── Task B: Word Similarity Ranking (Spearman) ──

def task_similarity_ranking(triadic_feats, embed_feats, n_pairs=100, seed=42):
    """
    Sample random word pairs, compute cosine similarity in both spaces,
    then compute Spearman rank correlation.
    """
    n = len(triadic_feats)
    rng = random.Random(seed)
    pairs = set()
    while len(pairs) < min(n_pairs, n * (n - 1) // 2):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i != j:
            pairs.add((min(i, j), max(i, j)))

    pairs = list(pairs)
    tri_sims, emb_sims = [], []
    for i, j in pairs:
        tri_sims.append(cosine_similarity(triadic_feats[i], triadic_feats[j]))
        emb_sims.append(cosine_similarity(embed_feats[i], embed_feats[j]))

    rho = spearman_rank_correlation(np.array(tri_sims), np.array(emb_sims))
    return rho, len(pairs)


# ── Task C: Intra/Inter Category Separation ──

def task_separation_ratio(feats, labels):
    """
    For each category, compute mean intra-category similarity and
    mean inter-category similarity. Return per-category and overall ratios.
    """
    sim_matrix = cosine_similarity_matrix(feats)
    unique_cats = sorted(set(labels))
    cat_indices = {c: [] for c in unique_cats}
    for i, l in enumerate(labels):
        cat_indices[l].append(i)

    per_cat = {}
    all_intra, all_inter = [], []

    for cat in unique_cats:
        idx = cat_indices[cat]
        other_idx = [i for i in range(len(labels)) if labels[i] != cat]

        if len(idx) < 2 or len(other_idx) < 1:
            continue

        # Intra: mean similarity among members (exclude self-similarity)
        intra_sims = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                intra_sims.append(sim_matrix[idx[i], idx[j]])

        # Inter: mean similarity between members and non-members
        inter_sims = []
        for i in idx:
            for j in other_idx:
                inter_sims.append(sim_matrix[i, j])

        mean_intra = float(np.mean(intra_sims))
        mean_inter = float(np.mean(inter_sims))
        ratio = mean_intra / mean_inter if abs(mean_inter) > 1e-8 else 0.0

        per_cat[cat] = {
            'intra': mean_intra,
            'inter': mean_inter,
            'ratio': ratio,
        }
        all_intra.extend(intra_sims)
        all_inter.extend(inter_sims)

    overall_ratio = float(np.mean(all_intra)) / max(abs(float(np.mean(all_inter))), 1e-8)
    return per_cat, overall_ratio


# ── Task D: k-NN Classification ──

def task_knn(feats, labels, k=3):
    """
    k-NN classification with leave-one-out evaluation.
    For each sample, find k nearest neighbors (by cosine similarity)
    and predict by majority vote.
    """
    sim_matrix = cosine_similarity_matrix(feats)
    unique_cats = sorted(set(labels))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    y = np.array([cat_to_idx[l] for l in labels])
    n = len(y)

    correct = 0
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -2.0  # exclude self
        # Get k nearest neighbors
        nn_indices = np.argsort(sims)[-k:]
        # Majority vote
        votes = {}
        for j in nn_indices:
            c = y[j]
            votes[c] = votes.get(c, 0) + 1
        pred = max(votes, key=votes.get)
        if pred == y[i]:
            correct += 1

    return correct / n


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='E6: Meaningful Compression Benchmark')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to model checkpoint (default: Run 15)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve checkpoint paths
    if args.checkpoint:
        model_path = args.checkpoint
        tok_path = os.path.join(os.path.dirname(model_path), 'tokenizer.json')
    else:
        ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign')
        model_path = os.path.join(ckpt_dir, 'model_L12_D512_B64_best.pt')
        tok_path = os.path.join(ckpt_dir, 'tokenizer.json')

    print("=" * 72)
    print("  E6 — MEANINGFUL COMPRESSION BENCHMARK")
    print("  64-bit triadic projection vs 512-dim embedding")
    print("=" * 72)
    print(f"  Device:     {device}")
    print(f"  Checkpoint: {model_path}")
    print(f"  Tokenizer:  {tok_path}")

    # ── Load model ──
    print("\n[1/6] Loading model ...")
    tokenizer = BPETokenizer.load(tok_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'],
        block_size=cfg['block_size'],
        n_layer=cfg['n_layer'],
        n_embd=cfg['n_embd'],
        n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'],
        dropout=0.0,
    )
    model = TriadicGPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    n_bits = config.n_triadic_bits
    n_embd = config.n_embd
    compression_ratio = n_embd / n_bits

    print(f"  Config:     {config.n_layer}L/{n_embd}D/{config.n_head}H/{n_bits}bits")
    print(f"  Params:     {model.num_params():,}")
    print(f"  Compression: {n_embd}D -> {n_bits}D  ({compression_ratio:.1f}x)")

    # ── Extract features ──
    print("\n[2/6] Extracting features ...")
    words, labels, triadic_feats, embed_feats = extract_features(model, tokenizer, device)

    n_words = len(words)
    n_cats = len(set(labels))
    random_baseline = 1.0 / n_cats

    print(f"  Words:      {n_words} valid (out of {sum(len(v) for v in CATEGORIES.values())} defined)")
    print(f"  Categories: {n_cats}")
    print(f"  Triadic:    {triadic_feats.shape}")
    print(f"  Embedding:  {embed_feats.shape}")
    print(f"  Random baseline: {random_baseline:.1%} (1/{n_cats})")

    # Per-category counts
    from collections import Counter
    cat_counts = Counter(labels)
    for cat in sorted(cat_counts):
        print(f"    {cat:>10s}: {cat_counts[cat]:>3d} words")

    # ── Task A: Nearest Centroid ──
    print(f"\n[3/6] Task A: Nearest Centroid Classification (leave-one-out) ...")
    acc_centroid_tri = task_nearest_centroid(triadic_feats, labels)
    acc_centroid_emb = task_nearest_centroid(embed_feats, labels)
    print(f"  Triadic {n_bits}D:   {acc_centroid_tri:.1%}")
    print(f"  Embedding {n_embd}D: {acc_centroid_emb:.1%}")
    print(f"  Random baseline: {random_baseline:.1%}")

    # ── Task B: Similarity Ranking ──
    print(f"\n[4/6] Task B: Word Similarity Ranking (Spearman rho) ...")
    rho, n_pairs = task_similarity_ranking(triadic_feats, embed_feats, n_pairs=200)
    print(f"  Spearman rho:    {rho:.4f}  ({n_pairs} pairs)")
    print(f"  Interpretation:  {'strong' if rho > 0.7 else 'moderate' if rho > 0.4 else 'weak' if rho > 0.2 else 'negligible'} correlation")

    # ── Task C: Separation Ratio ──
    print(f"\n[5/6] Task C: Intra/Inter Category Separation ...")
    sep_tri, overall_tri = task_separation_ratio(triadic_feats, labels)
    sep_emb, overall_emb = task_separation_ratio(embed_feats, labels)

    print(f"  {'Category':>12s}  {'Tri ratio':>10s}  {'Emb ratio':>10s}  {'Tri intra':>10s}  {'Emb intra':>10s}")
    print(f"  {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    for cat in sorted(sep_tri.keys()):
        tr = sep_tri[cat]
        er = sep_emb.get(cat, {'ratio': 0, 'intra': 0})
        print(f"  {cat:>12s}  {tr['ratio']:>10.3f}  {er['ratio']:>10.3f}  "
              f"{tr['intra']:>10.4f}  {er['intra']:>10.4f}")
    print(f"  {'OVERALL':>12s}  {overall_tri:>10.3f}  {overall_emb:>10.3f}")

    # ── Task D: k-NN ──
    print(f"\n[6/6] Task D: k-NN Classification (k=3, leave-one-out) ...")
    acc_knn_tri = task_knn(triadic_feats, labels, k=3)
    acc_knn_emb = task_knn(embed_feats, labels, k=3)
    print(f"  Triadic {n_bits}D:   {acc_knn_tri:.1%}")
    print(f"  Embedding {n_embd}D: {acc_knn_emb:.1%}")
    print(f"  Random baseline: {random_baseline:.1%}")

    # ── Summary Table ──
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE")
    print("=" * 72)
    print(f"  {'Task':>40s}  {'Triadic':>10s}  {'Embedding':>10s}  {'Random':>10s}  {'Compress':>10s}")
    print(f"  {'─' * 40}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

    rows = [
        ("A. Centroid Classification (acc)", acc_centroid_tri, acc_centroid_emb, random_baseline, compression_ratio),
        ("B. Similarity Ranking (Spearman)", rho, 1.0, 0.0, compression_ratio),
        ("C. Separation Ratio (overall)", overall_tri, overall_emb, 1.0, compression_ratio),
        ("D. k-NN Classification (acc, k=3)", acc_knn_tri, acc_knn_emb, random_baseline, compression_ratio),
    ]

    for name, tri, emb, rand_val, cr in rows:
        print(f"  {name:>40s}  {tri:>10.4f}  {emb:>10.4f}  {rand_val:>10.4f}  {cr:>9.1f}x")

    # ── Verdict ──
    print(f"\n  VERDICT:")
    tri_above_random = (acc_centroid_tri > random_baseline * 1.5 and
                        acc_knn_tri > random_baseline * 1.5)
    emb_above_random = (acc_centroid_emb > random_baseline * 1.5 and
                        acc_knn_emb > random_baseline * 1.5)

    if tri_above_random:
        tri_vs_emb_centroid = acc_centroid_tri / max(acc_centroid_emb, 1e-8)
        tri_vs_emb_knn = acc_knn_tri / max(acc_knn_emb, 1e-8)
        avg_retention = (tri_vs_emb_centroid + tri_vs_emb_knn) / 2.0
        print(f"    Triadic {n_bits}-bit achieves {avg_retention:.0%} of {n_embd}D accuracy at {compression_ratio:.0f}x compression")
        if avg_retention > 0.8:
            print(f"    STRONG: 64 bits retain >80% of embedding quality")
        elif avg_retention > 0.5:
            print(f"    MODERATE: 64 bits retain >50% of embedding quality")
        else:
            print(f"    WEAK: 64 bits retain <50% of embedding quality")
    else:
        print(f"    FAIL: triadic representations near random ({acc_centroid_tri:.1%} centroid, {acc_knn_tri:.1%} kNN)")

    if rho > 0.4:
        print(f"    Similarity structure preserved (rho={rho:.3f})")
    else:
        print(f"    Similarity structure NOT preserved (rho={rho:.3f})")

    print("=" * 72)

    # ── Save results ──
    result = {
        'experiment': 'E6_compression_benchmark',
        'checkpoint': model_path,
        'model_config': f"{config.n_layer}L/{n_embd}D/{config.n_head}H/{n_bits}bits",
        'n_words': n_words,
        'n_categories': n_cats,
        'compression_ratio': compression_ratio,
        'random_baseline': random_baseline,
        'task_A_centroid': {
            'triadic_accuracy': acc_centroid_tri,
            'embedding_accuracy': acc_centroid_emb,
        },
        'task_B_similarity_ranking': {
            'spearman_rho': rho,
            'n_pairs': n_pairs,
        },
        'task_C_separation': {
            'triadic_overall': overall_tri,
            'embedding_overall': overall_emb,
            'triadic_per_category': sep_tri,
            'embedding_per_category': sep_emb,
        },
        'task_D_knn': {
            'triadic_accuracy': acc_knn_tri,
            'embedding_accuracy': acc_knn_emb,
            'k': 3,
        },
        'summary': {
            'triadic_above_random': tri_above_random,
            'embedding_above_random': emb_above_random,
            'centroid_retention': acc_centroid_tri / max(acc_centroid_emb, 1e-8),
            'knn_retention': acc_knn_tri / max(acc_knn_emb, 1e-8),
            'similarity_preserved': rho > 0.4,
        },
    }

    save_path = os.path.join(RESULTS_DIR, 'compression_benchmark.json')
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {save_path}")


if __name__ == '__main__':
    main()

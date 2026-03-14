"""
P2 — Concept Vocabulary Builder (from La Danza Cosmica, Cap. 34)

The book proposes replacing BPE (statistical) with a conceptual tokenizer
where each atomic concept is a prime number and composite concepts are
products of primes.

This script builds a concept vocabulary by:
1. Extracting token embeddings from Run 15's trained model
2. Clustering them into ~500 semantic groups (atomic concepts)
3. Assigning a unique prime to each cluster centroid
4. Evaluating whether the prime-based concept tokens preserve semantic structure

NO TRAINING REQUIRED — uses existing embeddings + clustering.
"""

import os
import sys
import json
import math
import numpy as np
import torch
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
from src.triadic import PrimeMapper, TriadicValidator, nth_prime, prime_factors
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'model_L12_D512_B64_best.pt')
TOKENIZER = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'playground', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Number of atomic concepts (clusters)
N_CONCEPTS = 256


def load_model(device):
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


def build_concept_vocabulary(model, tokenizer, device, n_concepts=N_CONCEPTS):
    """
    Build a concept vocabulary from the model's learned embeddings.

    Steps:
    1. Extract all token embeddings from wte (word token embedding)
    2. Filter out special tokens and very rare tokens
    3. Cluster into n_concepts groups using K-Means
    4. Assign a prime to each cluster
    5. Each BPE token gets a composite prime = product of its cluster's prime
       (for single-cluster membership) or product of top-k nearest cluster primes
    """
    print("\n  Step 1: Extracting embeddings...")
    with torch.no_grad():
        embeddings = model.wte.weight.cpu().numpy()  # (vocab_size, n_embd)

    vocab_size, n_embd = embeddings.shape
    print(f"  Vocabulary: {vocab_size} tokens, {n_embd}D embeddings")

    # Get token strings (skip specials)
    token_strings = {}
    special_ids = set(tokenizer.special_tokens.values())
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        try:
            text = tokenizer.decode([token_id], skip_special=True)
            if text.strip():
                token_strings[token_id] = text.strip()
        except Exception:
            pass

    valid_ids = sorted(token_strings.keys())
    valid_embeddings = embeddings[valid_ids]
    print(f"  Valid tokens: {len(valid_ids)}")

    # Normalize embeddings for clustering
    norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = valid_embeddings / norms

    # Step 2: Cluster into concepts
    print(f"\n  Step 2: K-Means clustering into {n_concepts} concepts...")
    kmeans = KMeans(n_clusters=n_concepts, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(normalized)
    centroids = kmeans.cluster_centers_

    # Silhouette score (quality metric)
    # Use a subsample for speed
    n_sample = min(5000, len(normalized))
    sample_idx = np.random.RandomState(42).choice(len(normalized), n_sample, replace=False)
    sil_score = silhouette_score(normalized[sample_idx], labels[sample_idx])
    print(f"  Silhouette score: {sil_score:.4f}")

    # Step 3: Analyze clusters
    print(f"\n  Step 3: Analyzing clusters...")
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append({
            'token_id': valid_ids[i],
            'text': token_strings[valid_ids[i]],
            'distance': float(np.linalg.norm(normalized[i] - centroids[label])),
        })

    # Sort by distance to centroid (closest = most representative)
    for cluster_id in clusters:
        clusters[cluster_id].sort(key=lambda x: x['distance'])

    # Step 4: Assign primes to concepts
    print(f"\n  Step 4: Assigning primes to {n_concepts} atomic concepts...")
    concept_vocab = {}
    for cluster_id in range(n_concepts):
        prime = nth_prime(cluster_id + 1)
        members = clusters[cluster_id]
        representative = members[0]['text'] if members else f'concept_{cluster_id}'
        concept_vocab[cluster_id] = {
            'prime': prime,
            'representative': representative,
            'size': len(members),
            'members_sample': [m['text'] for m in members[:10]],
        }

    # Step 5: Map tokens to composite primes
    print(f"\n  Step 5: Computing composite prime signatures...")
    # Each token's composite = prime of its primary cluster
    # For a richer representation, also consider soft membership (top-k clusters)
    token_to_prime = {}
    for i, label in enumerate(labels):
        token_id = valid_ids[i]
        text = token_strings[token_id]
        primary_prime = concept_vocab[label]['prime']

        # Soft membership: find top-3 closest clusters
        distances = np.linalg.norm(centroids - normalized[i], axis=1)
        top3_clusters = distances.argsort()[:3]
        top3_primes = [concept_vocab[c]['prime'] for c in top3_clusters]

        # Hard composite (just primary)
        hard_composite = primary_prime

        # Soft composite (product of top-3 cluster primes)
        soft_composite = 1
        for p in top3_primes:
            soft_composite *= p

        token_to_prime[token_id] = {
            'text': text,
            'primary_cluster': int(label),
            'primary_prime': primary_prime,
            'hard_composite': hard_composite,
            'top3_clusters': [int(c) for c in top3_clusters],
            'soft_composite': soft_composite,
        }

    return concept_vocab, token_to_prime, clusters, sil_score, centroids, normalized, labels


def evaluate_concept_tokenizer(concept_vocab, token_to_prime, tokenizer):
    """Evaluate whether concept-token structure preserves semantic relationships."""

    # Test: words that should share concepts
    test_groups = {
        'family': ['mother', 'father', 'sister', 'brother', 'son', 'daughter'],
        'animals': ['dog', 'cat', 'bird', 'fish', 'bear', 'rabbit'],
        'colors': ['red', 'blue', 'green', 'yellow', 'white', 'black'],
        'emotions': ['happy', 'sad', 'angry', 'scared', 'kind'],
        'royalty': ['king', 'queen', 'prince', 'princess'],
        'nature': ['sun', 'moon', 'tree', 'flower', 'river'],
    }

    print("\n  Evaluating semantic consistency of concept clusters...")

    group_results = {}
    for group_name, words in test_groups.items():
        # Get primary cluster for each word
        cluster_ids = []
        for word in words:
            ids = tokenizer.encode(word, add_special=False)
            if ids and ids[0] in token_to_prime:
                cluster_ids.append(token_to_prime[ids[0]]['primary_cluster'])

        if len(cluster_ids) < 2:
            continue

        # How many words share the same cluster?
        from collections import Counter
        cluster_counts = Counter(cluster_ids)
        most_common_cluster, most_common_count = cluster_counts.most_common(1)[0]
        consistency = most_common_count / len(cluster_ids)

        # Pairwise GCD analysis (shared prime factors)
        primes = []
        for word in words:
            ids = tokenizer.encode(word, add_special=False)
            if ids and ids[0] in token_to_prime:
                primes.append(token_to_prime[ids[0]]['soft_composite'])

        gcd_scores = []
        for i in range(len(primes)):
            for j in range(i + 1, len(primes)):
                shared = math.gcd(primes[i], primes[j])
                n_shared = len(prime_factors(shared))
                gcd_scores.append(n_shared)

        group_results[group_name] = {
            'n_words': len(cluster_ids),
            'n_unique_clusters': len(set(cluster_ids)),
            'consistency': float(consistency),
            'mean_shared_factors': float(np.mean(gcd_scores)) if gcd_scores else 0,
        }

        print(f"    {group_name:>10s}: {len(set(cluster_ids))} clusters for {len(cluster_ids)} words, "
              f"consistency={consistency:.0%}, mean_shared={np.mean(gcd_scores):.1f} factors")

    # Test: analogy via concept algebra
    print("\n  Testing concept-level analogies...")
    analogies = [
        ('king', 'queen', 'man', 'woman'),
        ('father', 'mother', 'brother', 'sister'),
        ('dog', 'puppy', 'cat', 'kitten'),
    ]

    analogy_results = []
    for a, b, c, d in analogies:
        ids_a = tokenizer.encode(a, add_special=False)
        ids_b = tokenizer.encode(b, add_special=False)
        ids_c = tokenizer.encode(c, add_special=False)
        ids_d = tokenizer.encode(d, add_special=False)

        if not all(ids and ids[0] in token_to_prime for ids in [ids_a, ids_b, ids_c, ids_d]):
            continue

        pa = token_to_prime[ids_a[0]]['soft_composite']
        pb = token_to_prime[ids_b[0]]['soft_composite']
        pc = token_to_prime[ids_c[0]]['soft_composite']
        pd = token_to_prime[ids_d[0]]['soft_composite']

        predicted = TriadicValidator.analogy(pa, pb, pc)
        sim = TriadicValidator.similarity(predicted, pd)

        analogy_results.append({
            'analogy': f'{a}:{b}::{c}:{d}',
            'predicted_prime': predicted,
            'actual_prime': pd,
            'similarity': float(sim),
        })
        print(f"    {a}:{b}::{c}:{d}  sim={sim:.2%}")

    return group_results, analogy_results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 64)
    print("  CONCEPT VOCABULARY BUILDER")
    print("  (La Danza Cosmica, Cap. 34: Conceptual Tokenizer)")
    print("=" * 64)
    print(f"  Device: {device}")
    print(f"  Target concepts: {N_CONCEPTS}")

    print("\nLoading Run 15 model...")
    model, tokenizer, config = load_model(device)

    # Build concept vocabulary
    concept_vocab, token_to_prime, clusters, sil_score, centroids, normalized, labels = \
        build_concept_vocabulary(model, tokenizer, device)

    # Evaluate
    group_results, analogy_results = evaluate_concept_tokenizer(concept_vocab, token_to_prime, tokenizer)

    # Print top concepts
    print("\n  Top 20 concepts (by cluster size):")
    sorted_concepts = sorted(concept_vocab.items(), key=lambda x: x[1]['size'], reverse=True)
    for cluster_id, info in sorted_concepts[:20]:
        print(f"    p={info['prime']:>5d}  size={info['size']:>4d}  "
              f"rep='{info['representative']}'  members={info['members_sample'][:5]}")

    # Visualize with t-SNE (subsample for speed)
    print("\n  Generating t-SNE visualization...")
    from sklearn.manifold import TSNE

    n_sample = min(3000, len(normalized))
    sample_idx = np.random.RandomState(42).choice(len(normalized), n_sample, replace=False)
    sample_emb = normalized[sample_idx]
    sample_labels = labels[sample_idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(sample_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: t-SNE colored by cluster
    ax = axes[0]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=sample_labels,
                         cmap='tab20', s=3, alpha=0.5)
    ax.set_title(f'Token Embeddings Clustered into {N_CONCEPTS} Concepts\n'
                 f'(Silhouette = {sil_score:.3f})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    # Plot 2: Cluster sizes distribution
    ax = axes[1]
    sizes = [info['size'] for info in concept_vocab.values()]
    ax.hist(sizes, bins=50, color='steelblue', alpha=0.8)
    ax.axvline(np.mean(sizes), color='red', linestyle='--', label=f'Mean={np.mean(sizes):.0f}')
    ax.set_xlabel('Tokens per Concept')
    ax.set_ylabel('Count')
    ax.set_title('Concept Cluster Size Distribution')
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'concept_tokenizer.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # Save concept vocabulary
    save_vocab = {}
    for cluster_id, info in concept_vocab.items():
        save_vocab[str(cluster_id)] = {
            'prime': info['prime'],
            'representative': info['representative'],
            'size': info['size'],
            'members_sample': info['members_sample'],
        }

    vocab_path = os.path.join(RESULTS_DIR, 'concept_vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(save_vocab, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {vocab_path}")

    # Save full results
    results = {
        'experiment': 'concept_tokenizer',
        'source': 'La Danza Cosmica Cap. 34 — Conceptual Tokenizer',
        'n_concepts': N_CONCEPTS,
        'silhouette_score': float(sil_score),
        'total_tokens': len(token_to_prime),
        'group_consistency': group_results,
        'analogies': analogy_results,
        'mean_cluster_size': float(np.mean(sizes)),
        'std_cluster_size': float(np.std(sizes)),
    }

    results_path = os.path.join(RESULTS_DIR, 'concept_tokenizer.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  Concepts: {N_CONCEPTS}")
    print(f"  Silhouette: {sil_score:.4f}")
    print(f"  Mean cluster size: {np.mean(sizes):.0f} tokens")
    mean_consistency = np.mean([r['consistency'] for r in group_results.values()])
    print(f"  Mean group consistency: {mean_consistency:.0%}")
    if analogy_results:
        mean_analogy_sim = np.mean([r['similarity'] for r in analogy_results])
        print(f"  Mean analogy similarity: {mean_analogy_sim:.2%}")
    print("=" * 64)


if __name__ == '__main__':
    main()

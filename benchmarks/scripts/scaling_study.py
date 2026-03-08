"""
Scaling Study — Measures triadic quality across model sizes.

For each model checkpoint, computes:
  - Language loss (final training loss from CSV)
  - Bit entropy (mean per-bit entropy of triadic projections)
  - Unique signatures (% of concepts with distinct prime composites)
  - Semantic ordering (related > unrelated similarity gap)
  - Probe accuracy (linear classifier on triadic bits vs embeddings)
  - Analogy verification (prime algebra > median rate)

Usage:
  python benchmarks/scripts/scaling_study.py \
    --models checkpoints/torch_run19_small/model_L4_D128_B16_best.pt \
             checkpoints/torch_run20_medium/model_L6_D256_B32_best.pt \
             checkpoints/torch_run21_large/model_L8_D384_B48_best.pt \
             checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt \
    --tokenizer checkpoints/torch_runXL/tokenizer.json
"""

import os
import sys
import json
import math
import argparse
from datetime import date
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluate import load_model
from src.triadic import PrimeMapper, TriadicValidator, prime_factors


# ============================================================
# Concept sets (shared with other benchmarks)
# ============================================================

SEMANTIC_PAIRS = {
    'related': [
        ("king", "queen"), ("dog", "cat"), ("happy", "sad"),
        ("mother", "father"), ("boy", "girl"), ("brother", "sister"),
        ("love", "hate"), ("sun", "moon"), ("fire", "water"),
        ("run", "walk"), ("big", "small"), ("doctor", "nurse"),
    ],
    'unrelated': [
        ("king", "dog"), ("queen", "fish"), ("happy", "table"),
        ("mother", "river"), ("boy", "cloud"), ("brother", "cake"),
        ("love", "chair"), ("sun", "pen"), ("fire", "girl"),
        ("run", "lamp"), ("big", "milk"), ("doctor", "star"),
    ],
}

CONCEPT_CATEGORIES = {
    "dog": "animal", "cat": "animal", "bird": "animal", "fish": "animal",
    "horse": "animal", "cow": "animal",
    "boy": "person", "girl": "person", "man": "person", "woman": "person",
    "king": "person", "queen": "person", "prince": "person", "princess": "person",
    "mother": "person", "father": "person", "brother": "person", "sister": "person",
    "friend": "person", "enemy": "person",
    "doctor": "profession", "nurse": "profession", "teacher": "profession",
    "student": "profession", "scientist": "profession",
    "happy": "feeling", "sad": "feeling", "angry": "feeling",
    "love": "feeling", "hate": "feeling", "fear": "feeling",
    "hope": "feeling", "joy": "feeling",
    "fire": "nature", "water": "nature", "earth": "nature", "air": "nature",
    "sun": "nature", "moon": "nature", "star": "nature", "cloud": "nature",
    "river": "nature", "mountain": "nature", "ocean": "nature",
    "food": "food", "bread": "food", "milk": "food", "apple": "food",
    "table": "artifact", "chair": "artifact", "book": "artifact",
    "pen": "artifact", "door": "artifact", "lamp": "artifact",
    "red": "attribute", "blue": "attribute", "big": "attribute",
    "small": "attribute", "fast": "attribute", "old": "attribute",
    "house": "place", "city": "place", "school": "place",
    "church": "place", "garden": "place",
    "morning": "time", "night": "time", "summer": "time", "winter": "time",
    "run": "action", "walk": "action", "swim": "action",
    "jump": "action", "climb": "action", "sleep": "action",
    "music": "communication", "dance": "communication", "song": "communication",
    "game": "communication", "story": "communication",
    "peace": "state", "war": "state",
    "cake": "food", "drink": "food",
}

ANALOGIES = [
    ("king", "queen", "man", "woman"),
    ("king", "queen", "boy", "girl"),
    ("king", "queen", "father", "mother"),
    ("man", "woman", "boy", "girl"),
    ("man", "woman", "father", "mother"),
    ("happy", "sad", "love", "hate"),
    ("happy", "sad", "peace", "war"),
    ("big", "small", "fast", "slow"),
    ("big", "small", "old", "young"),
    ("fire", "water", "sun", "moon"),
    ("morning", "night", "summer", "winter"),
    ("dog", "cat", "horse", "cow"),
    ("doctor", "hospital", "teacher", "school"),
]

VOCAB_POOL = list(set(
    list(CONCEPT_CATEGORIES.keys()) +
    ["slow", "young", "hospital", "castle", "window", "bed",
     "pig", "sheep", "flower", "tree", "car", "rain", "snow",
     "spring", "park", "village", "lawyer", "judge", "brave",
     "kind", "cruel", "afraid", "green", "fly", "fall",
     "dream", "magic", "pain", "forest"]
))


# ============================================================
# Core measurement functions
# ============================================================

def get_projections(model, tokenizer, concepts, device):
    """Get triadic projections for all concepts."""
    results = {}
    for concept in concepts:
        ids = tokenizer.encode(concept, add_special=False)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)
        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        results[concept] = proj
    return results


def measure_bit_entropy(projections):
    """Compute mean per-bit entropy from projections."""
    if not projections:
        return 0.0, 0
    matrix = np.array(list(projections.values()))
    bits = (matrix > 0).astype(float)
    p = bits.mean(axis=0)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return float(entropy.mean()), int(matrix.shape[1])


def measure_unique_signatures(projections):
    """Count unique prime signatures."""
    if not projections:
        return 0, 0
    bits_list = []
    for proj in projections.values():
        bits = tuple((proj > 0).astype(int))
        bits_list.append(bits)
    unique = len(set(bits_list))
    return unique, len(bits_list)


def measure_semantic_ordering(projections):
    """Compute semantic gap: mean related sim - mean unrelated sim."""
    def jaccard(a, b):
        ba = set(np.where(a > 0)[0])
        bb = set(np.where(b > 0)[0])
        if not ba and not bb:
            return 1.0
        union = ba | bb
        return len(ba & bb) / len(union) if union else 0.0

    related_sims = []
    for a, b in SEMANTIC_PAIRS['related']:
        if a in projections and b in projections:
            related_sims.append(jaccard(projections[a], projections[b]))

    unrelated_sims = []
    for a, b in SEMANTIC_PAIRS['unrelated']:
        if a in projections and b in projections:
            unrelated_sims.append(jaccard(projections[a], projections[b]))

    if not related_sims or not unrelated_sims:
        return 0.0, 0.0, 0.0

    mean_rel = np.mean(related_sims)
    mean_unrel = np.mean(unrelated_sims)
    gap = mean_rel - mean_unrel
    return float(mean_rel), float(mean_unrel), float(gap)


def measure_probe_accuracy(model, tokenizer, device):
    """Train linear probe on triadic bits, return accuracy."""
    concepts = list(CONCEPT_CATEGORIES.keys())
    triadic_features = []
    embedding_features = []
    labels = []

    for concept in concepts:
        ids = tokenizer.encode(concept, add_special=False)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)
            embed = model.wte(x)
        tri_feat = triadic_proj[0].mean(dim=0).cpu().numpy()
        emb_feat = embed[0].mean(dim=0).cpu().numpy()
        triadic_features.append(tri_feat)
        embedding_features.append(emb_feat)
        labels.append(CONCEPT_CATEGORIES[concept])

    if len(set(labels)) < 2:
        return 0.0, 0.0

    X_tri = np.array(triadic_features)
    X_emb = np.array(embedding_features)

    tri_acc = _train_probe_cv(X_tri, labels)
    emb_acc = _train_probe_cv(X_emb, labels)
    return tri_acc, emb_acc


def _train_probe_cv(X, y, n_splits=5):
    """Simple logistic regression with k-fold CV."""
    unique_labels = sorted(set(y))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_idx = np.array([label_to_idx[l] for l in y])
    n_classes = len(unique_labels)
    n_samples, n_features = X.shape

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n_samples // n_splits

    all_preds = np.zeros(n_samples, dtype=int)
    all_true = np.zeros(n_samples, dtype=int)

    for fold in range(n_splits):
        start = fold * fold_size
        end = start + fold_size if fold < n_splits - 1 else n_samples
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        X_train, X_test = X_norm[train_idx], X_norm[test_idx]
        y_train, y_test = y_idx[train_idx], y_idx[test_idx]

        W = np.zeros((n_features, n_classes))
        b = np.zeros(n_classes)

        for _ in range(200):
            logits = X_train @ W + b
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            targets = np.zeros_like(probs)
            targets[np.arange(len(y_train)), y_train] = 1.0
            grad = probs - targets
            dW = X_train.T @ grad / len(y_train) + 0.01 * W
            db = grad.mean(axis=0)
            W -= 0.1 * dW
            b -= 0.1 * db

        test_logits = X_test @ W + b
        all_preds[test_idx] = test_logits.argmax(axis=1)
        all_true[test_idx] = y_test

    return float((all_preds == all_true).mean())


def measure_analogy_verification(projections):
    """Compute analogy verification rate (correct answer > median similarity)."""
    mapper_bits = len(next(iter(projections.values())))
    mapper = PrimeMapper(mapper_bits)

    concept_primes = {}
    for concept, proj in projections.items():
        concept_primes[concept] = mapper.map(proj)

    vocab_primes = {w: concept_primes[w] for w in VOCAB_POOL if w in concept_primes}

    total = 0
    verification_correct = 0

    for a, b, c, d in ANALOGIES:
        if any(x not in concept_primes for x in [a, b, c, d]):
            continue
        if d not in vocab_primes:
            continue

        total += 1
        phi_a, phi_b, phi_c, phi_d = [concept_primes[x] for x in [a, b, c, d]]

        # Prime algebra: target = C * (B / gcd(A,B))
        shared_ab = math.gcd(phi_a, phi_b)
        transform = phi_b // shared_ab if shared_ab > 0 else phi_b
        target = (phi_c * transform) // math.gcd(phi_c, transform)

        # Jaccard similarity of prime factors
        def factor_sim(x, y):
            fx, fy = set(prime_factors(x)), set(prime_factors(y))
            if not fx and not fy:
                return 1.0
            union = fx | fy
            return len(fx & fy) / len(union) if union else 0.0

        exclude = {a, b, c}
        candidates = [(w, p) for w, p in vocab_primes.items() if w not in exclude]
        d_sim = factor_sim(target, phi_d)
        median_sim = np.median([factor_sim(target, p) for _, p in candidates])

        if d_sim > median_sim:
            verification_correct += 1

    return verification_correct, total


def read_final_loss(checkpoint_dir):
    """Read final training loss from CSV log."""
    csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
    if not os.path.exists(csv_path):
        return None
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    return float(rows[-1]['loss'])


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 68)
    print("  SCALING STUDY — Triadic Quality vs Model Size")
    print("=" * 68)
    print()

    all_concepts = set(CONCEPT_CATEGORIES.keys())
    for a, b in SEMANTIC_PAIRS['related'] + SEMANTIC_PAIRS['unrelated']:
        all_concepts.update([a, b])
    for a, b, c, d in ANALOGIES:
        all_concepts.update([a, b, c, d])
    all_concepts.update(VOCAB_POOL)

    results = []

    for model_path in args.models:
        print(f"  Loading: {model_path}")
        tokenizer_path = args.tokenizer or os.path.join(os.path.dirname(model_path), 'tokenizer.json')
        model, tokenizer, config = load_model(model_path, tokenizer_path, device)

        n_params = model.num_params()
        scale_tag = f"{config.n_layer}L/{config.n_embd}D/{config.n_triadic_bits}bits"
        checkpoint_dir = os.path.dirname(model_path)

        print(f"    Config: {scale_tag} | Params: {n_params:,}")

        # 1. Get projections
        projections = get_projections(model, tokenizer, all_concepts, device)
        print(f"    Encoded: {len(projections)} concepts")

        # 2. Bit entropy
        entropy, n_bits = measure_bit_entropy(projections)
        print(f"    Bit entropy: {entropy:.3f} ({n_bits} bits)")

        # 3. Unique signatures
        unique, total = measure_unique_signatures(projections)
        unique_pct = unique / total * 100 if total > 0 else 0
        print(f"    Unique sigs: {unique}/{total} ({unique_pct:.1f}%)")

        # 4. Semantic ordering
        mean_rel, mean_unrel, gap = measure_semantic_ordering(projections)
        print(f"    Semantic gap: {gap:+.3f} (rel={mean_rel:.3f}, unrel={mean_unrel:.3f})")

        # 5. Probe accuracy
        tri_acc, emb_acc = measure_probe_accuracy(model, tokenizer, device)
        print(f"    Probe: triadic={tri_acc:.1%}, embedding={emb_acc:.1%}")

        # 6. Analogy verification
        verif_correct, verif_total = measure_analogy_verification(projections)
        verif_rate = verif_correct / verif_total if verif_total > 0 else 0
        print(f"    Analogy verif: {verif_rate:.1%} ({verif_correct}/{verif_total})")

        # 7. Training loss
        final_loss = read_final_loss(checkpoint_dir)
        if final_loss:
            print(f"    Final loss: {final_loss:.4f}")

        results.append({
            'model_path': model_path,
            'scale': scale_tag,
            'n_params': n_params,
            'n_bits': n_bits,
            'final_loss': final_loss,
            'bit_entropy': entropy,
            'unique_signatures': unique,
            'unique_total': total,
            'unique_pct': unique_pct,
            'mean_related_sim': mean_rel,
            'mean_unrelated_sim': mean_unrel,
            'semantic_gap': gap,
            'probe_triadic_acc': tri_acc,
            'probe_embedding_acc': emb_acc,
            'analogy_verification': verif_rate,
            'analogy_correct': verif_correct,
            'analogy_total': verif_total,
        })

        # Free memory
        del model
        torch.cuda.empty_cache()
        print()

    # Summary table
    print("=" * 68)
    print("  SCALING SUMMARY")
    print("=" * 68)
    print(f"  {'Scale':<22} {'Params':>8} {'Loss':>7} {'Entropy':>8} {'Uniq%':>6} {'Gap':>7} {'Probe':>6} {'Verif':>6}")
    print("  " + "-" * 66)
    for r in results:
        print(f"  {r['scale']:<22} {r['n_params']:>8,} {r['final_loss'] or 0:>7.3f} {r['bit_entropy']:>8.3f} {r['unique_pct']:>5.1f}% {r['semantic_gap']:>+7.3f} {r['probe_triadic_acc']:>5.1%} {r['analogy_verification']:>5.1%}")
    print("=" * 68)

    # Save results
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    results_dir = os.path.join(project_root, 'benchmarks', 'results')
    os.makedirs(results_dir, exist_ok=True)

    today = date.today().isoformat()
    version = args.version

    output = {
        "benchmark": "scaling_study",
        "version": version,
        "date": today,
        "n_models": len(results),
        "models": results,
    }

    result_path = os.path.join(results_dir, f"{version}_scaling_study_{today}.json")
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scaling Study')
    parser.add_argument('--models', nargs='+', required=True, help='Model checkpoint paths (ordered by size)')
    parser.add_argument('--tokenizer', default=None, help='Shared tokenizer path')
    parser.add_argument('--version', default='v3.0-scaling')
    args = parser.parse_args()
    main(args)

"""
Interpretability Probe — Linear Classifier on Triadic Bits.

Trains a simple logistic regression on frozen triadic bit projections to predict
semantic categories (WordNet-inspired supersenses). Then compares to the same
probe trained on the model's hidden embeddings.

If triadic bits encode meaningful semantic information, the probe should achieve
non-trivial accuracy. If bits outperform or match embeddings, the triadic head
is an efficient semantic bottleneck.

Metrics:
  - Probe Accuracy (triadic bits): linear classifier on 64-bit projection
  - Probe Accuracy (embeddings): same classifier on 512-dim embeddings
  - Delta: triadic - embedding accuracy
  - Per-category F1

Usage:
  python benchmarks/scripts/interpretability_probe.py \
    --model checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt \
    --tokenizer checkpoints/torch_runXL/tokenizer.json \
    --version v1.4-strongalign
"""

import os
import sys
import json
import argparse
from datetime import date
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluate import load_model


# ============================================================
# Semantic categories (WordNet supersense-inspired)
# Each concept is assigned to exactly one primary category
# ============================================================

CONCEPT_CATEGORIES = {
    # noun.animal
    "dog": "animal", "cat": "animal", "bird": "animal", "fish": "animal",
    "horse": "animal", "cow": "animal", "pig": "animal", "sheep": "animal",
    # noun.person
    "boy": "person", "girl": "person", "man": "person", "woman": "person",
    "king": "person", "queen": "person", "prince": "person", "princess": "person",
    "mother": "person", "father": "person", "brother": "person", "sister": "person",
    "friend": "person", "enemy": "person",
    # noun.profession
    "doctor": "profession", "nurse": "profession", "teacher": "profession",
    "student": "profession", "lawyer": "profession", "judge": "profession",
    "scientist": "profession",
    # noun.feeling
    "happy": "feeling", "sad": "feeling", "angry": "feeling",
    "afraid": "feeling", "brave": "feeling", "kind": "feeling", "cruel": "feeling",
    "love": "feeling", "hate": "feeling", "fear": "feeling",
    "hope": "feeling", "joy": "feeling", "pain": "feeling",
    # noun.nature
    "fire": "nature", "water": "nature", "earth": "nature", "air": "nature",
    "sun": "nature", "moon": "nature", "star": "nature", "cloud": "nature",
    "river": "nature", "mountain": "nature", "ocean": "nature", "forest": "nature",
    "rain": "nature", "snow": "nature",
    # noun.food
    "food": "food", "drink": "food", "bread": "food", "milk": "food",
    "apple": "food", "cake": "food",
    # noun.artifact (furniture/objects)
    "table": "artifact", "chair": "artifact", "bed": "artifact", "lamp": "artifact",
    "book": "artifact", "pen": "artifact", "door": "artifact", "window": "artifact",
    # noun.attribute (colors/properties)
    "red": "attribute", "blue": "attribute", "green": "attribute",
    "big": "attribute", "small": "attribute", "fast": "attribute",
    "slow": "attribute", "old": "attribute", "young": "attribute",
    # noun.place
    "house": "place", "city": "place", "village": "place",
    "school": "place", "church": "place", "hospital": "place",
    "garden": "place", "park": "place",
    # noun.time
    "morning": "time", "night": "time", "summer": "time",
    "winter": "time", "spring": "time",
    # noun.act (actions)
    "run": "action", "walk": "action", "swim": "action", "fly": "action",
    "jump": "action", "climb": "action", "fall": "action", "sleep": "action",
    # noun.communication
    "music": "communication", "dance": "communication", "song": "communication",
    "game": "communication", "story": "communication", "dream": "communication",
    "magic": "communication",
    # noun.state
    "peace": "state", "war": "state",
}


def compute_features(model, tokenizer, concepts, device):
    """Extract triadic projections and embeddings for all concepts."""
    triadic_features = []
    embedding_features = []
    labels = []
    valid_concepts = []

    for concept in concepts:
        ids = tokenizer.encode(concept, add_special=False)
        if not ids:
            continue

        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)
            embed = model.wte(x)  # (1, T, n_embd)

        # Average across tokens
        tri_feat = triadic_proj[0].mean(dim=0).cpu().numpy()  # (n_bits,)
        emb_feat = embed[0].mean(dim=0).cpu().numpy()  # (n_embd,)

        triadic_features.append(tri_feat)
        embedding_features.append(emb_feat)
        labels.append(CONCEPT_CATEGORIES[concept])
        valid_concepts.append(concept)

    return (
        np.array(triadic_features),
        np.array(embedding_features),
        labels,
        valid_concepts,
    )


def train_probe(X, y, n_splits=5):
    """
    Train a linear probe using leave-one-out or k-fold cross-validation.
    Uses a simple softmax regression (no sklearn dependency).
    """
    from collections import Counter

    # Encode labels
    unique_labels = sorted(set(y))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_idx = np.array([label_to_idx[l] for l in y])
    n_classes = len(unique_labels)
    n_samples, n_features = X.shape

    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    # K-fold cross-validation
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

        # Train logistic regression via gradient descent
        W = np.zeros((n_features, n_classes))
        b = np.zeros(n_classes)
        lr = 0.1
        n_epochs = 200

        for epoch in range(n_epochs):
            # Forward
            logits = X_train @ W + b  # (N, C)
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # One-hot targets
            targets = np.zeros_like(probs)
            targets[np.arange(len(y_train)), y_train] = 1.0

            # Backward
            grad = probs - targets  # (N, C)
            dW = X_train.T @ grad / len(y_train) + 0.01 * W  # L2 reg
            db = grad.mean(axis=0)

            W -= lr * dW
            b -= lr * db

        # Predict
        test_logits = X_test @ W + b
        preds = test_logits.argmax(axis=1)
        all_preds[test_idx] = preds
        all_true[test_idx] = y_test

    # Compute metrics
    accuracy = (all_preds == all_true).mean()

    # Per-class F1
    per_class_f1 = {}
    for cls_idx, cls_name in enumerate(unique_labels):
        tp = ((all_preds == cls_idx) & (all_true == cls_idx)).sum()
        fp = ((all_preds == cls_idx) & (all_true != cls_idx)).sum()
        fn = ((all_preds != cls_idx) & (all_true == cls_idx)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1[cls_name] = {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'support': int((all_true == cls_idx).sum()),
        }

    macro_f1 = np.mean([v['f1'] for v in per_class_f1.values()])

    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'per_class': per_class_f1,
        'n_classes': n_classes,
        'n_samples': n_samples,
        'n_features': n_features,
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 68)
    print("  INTERPRETABILITY PROBE — Linear Classifier on Triadic Bits")
    print("=" * 68)
    print(f"  Model: {args.model}")
    print()

    model, tokenizer, config = load_model(args.model, args.tokenizer, device)
    print(f"  Config: {config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits")

    concepts = list(CONCEPT_CATEGORIES.keys())
    print(f"  Concepts: {len(concepts)}")
    print(f"  Categories: {len(set(CONCEPT_CATEGORIES.values()))}")
    print()

    # Extract features
    print("[1/3] Extracting features...")
    tri_features, emb_features, labels, valid_concepts = compute_features(
        model, tokenizer, concepts, device
    )
    print(f"  Triadic features: {tri_features.shape}")
    print(f"  Embedding features: {emb_features.shape}")
    print(f"  Valid concepts: {len(valid_concepts)}")

    # Category distribution
    from collections import Counter
    dist = Counter(labels)
    print(f"  Category distribution: {dict(sorted(dist.items(), key=lambda x: -x[1]))}")

    # Train probes
    print()
    print("[2/3] Training probe on TRIADIC BITS...")
    tri_result = train_probe(tri_features, labels)
    print(f"  Accuracy: {tri_result['accuracy']:.1%}")
    print(f"  Macro F1: {tri_result['macro_f1']:.3f}")

    print()
    print("[3/3] Training probe on EMBEDDINGS (baseline)...")
    emb_result = train_probe(emb_features, labels)
    print(f"  Accuracy: {emb_result['accuracy']:.1%}")
    print(f"  Macro F1: {emb_result['macro_f1']:.3f}")

    # Comparison
    delta_acc = tri_result['accuracy'] - emb_result['accuracy']
    delta_f1 = tri_result['macro_f1'] - emb_result['macro_f1']

    print()
    print("  Comparison:")
    print(f"    Triadic bits ({config.n_triadic_bits}D):  acc={tri_result['accuracy']:.1%}  F1={tri_result['macro_f1']:.3f}")
    print(f"    Embeddings ({config.n_embd}D):  acc={emb_result['accuracy']:.1%}  F1={emb_result['macro_f1']:.3f}")
    print(f"    Delta:                   acc={delta_acc:+.1%}  F1={delta_f1:+.3f}")

    # Per-category breakdown
    print()
    print("  Per-category F1 (triadic | embedding):")
    for cat in sorted(tri_result['per_class'].keys()):
        tri_f1 = tri_result['per_class'][cat]['f1']
        emb_f1 = emb_result['per_class'].get(cat, {}).get('f1', 0.0)
        sup = tri_result['per_class'][cat]['support']
        better = "<<<" if tri_f1 > emb_f1 + 0.05 else (">>>" if emb_f1 > tri_f1 + 0.05 else "   ")
        print(f"    {cat:>15}: {tri_f1:.2f} | {emb_f1:.2f}  (n={sup}) {better}")

    # Save results
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    results_dir = os.path.join(project_root, 'benchmarks', 'results')
    os.makedirs(results_dir, exist_ok=True)

    version = args.version
    today = date.today().isoformat()

    result = {
        "benchmark": "interpretability_probe",
        "version": version,
        "date": today,
        "model_checkpoint": args.model,
        "model_config": f"{config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits",
        "n_concepts": len(valid_concepts),
        "n_categories": tri_result['n_classes'],
        "metrics": {
            "triadic_accuracy": tri_result['accuracy'],
            "triadic_macro_f1": tri_result['macro_f1'],
            "triadic_n_features": tri_result['n_features'],
            "embedding_accuracy": emb_result['accuracy'],
            "embedding_macro_f1": emb_result['macro_f1'],
            "embedding_n_features": emb_result['n_features'],
            "delta_accuracy": delta_acc,
            "delta_f1": delta_f1,
        },
        "triadic_per_class": tri_result['per_class'],
        "embedding_per_class": emb_result['per_class'],
    }

    result_path = os.path.join(results_dir, f"{version}_interpretability_probe_{today}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {result_path}")

    # Verdict
    print()
    print("=" * 68)
    target_acc = 0.40
    print(f"  Triadic Probe:  {tri_result['accuracy']:.1%}  (target > {target_acc:.0%})  {'PASS' if tri_result['accuracy'] > target_acc else 'BELOW'}")
    print(f"  Embedding Probe: {emb_result['accuracy']:.1%}")
    print(f"  Delta:           {delta_acc:+.1%}  ({'triadic wins' if delta_acc > 0 else 'embedding wins'})")
    ratio = tri_result['accuracy'] / max(emb_result['accuracy'], 0.01)
    print(f"  Efficiency:      {config.n_triadic_bits} bits achieve {ratio:.0%} of {config.n_embd}-dim embedding accuracy")
    print("=" * 68)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpretability Probe')
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--version', default='v1.4-strongalign')
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = os.path.join(os.path.dirname(args.model), 'tokenizer.json')

    main(args)

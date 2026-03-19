"""
EXP-F4.4: Evaluacion D-A13 (GPT-2 Medium 355M)
================================================

INDISPENSABLE antes de publicar (~2 horas).

El entrenamiento completo (50K steps, 1.4GB checkpoint). Evaluacion formal pendiente.
Training log: sub holdout 100% (13/13), peak bit acc 89.4%, zeros colapsan a 0%.

CRITICAL:
  - Modelo:     GPT2MediumTernary (de gpt2_medium_ternary.py)
  - Checkpoint:  checkpoints/danza_gpt2medium_ternary/model_best.pt
  - Tokenizer:   GPT2Tokenizer de HuggingFace ('gpt2-medium')
  - max_tokens:  8 (consistente con entrenamiento via ids[:8])
  - NO usar:     GPT2TriadicModel (experiment10), max_tokens=4

Suite de evaluacion:
  1. Bit accuracy (train + holdout)
  2. Subsumption rate
  3. Ternary distribution {-1, 0, +1}
  4. Signature uniqueness
  5. Analogy verification (R3 on quads)
  6. Comparacion 40M vs 355M

Usage:
  cd C:\\Github\\triadic-microgpt
  python playground/audit_tests/test_d_a13_eval.py
"""

import os
import sys
import json
import math
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from common import (
    to_binary, to_ternary, hamming, cosine_sim, proj_to_prime,
    save_results, print_header, print_section, N_BITS,
    DA13_CKPT, DA13_CKPT_DIR,
)
from src.triadic import PrimeMapper, TriadicValidator


# ============================================================
# Model loading (D-A13 specific)
# ============================================================

def load_da13_model(device='cpu'):
    """Load D-A13 correctly: GPT2MediumTernary + GPT2Tokenizer."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    sys.path.insert(0, os.path.join(os.path.dirname(_THIS_DIR)))
    from gpt2_medium_ternary import GPT2MediumTernary

    print(f"  Loading GPT-2 Medium from HuggingFace...")
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model = GPT2MediumTernary(gpt2, n_triadic_bits=N_BITS, quantize_mode='fsq')

    print(f"  Loading checkpoint: {os.path.basename(DA13_CKPT)}")
    ckpt = torch.load(DA13_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params / 1e6:.1f}M params, {N_BITS} ternary trits")

    return model, tokenizer


@torch.no_grad()
def get_proj_da13(model, tokenizer, word, device='cpu'):
    """Extract projection from D-A13 with max_tokens=8."""
    ids = tokenizer.encode(word, add_special_tokens=False)[:8]
    if not ids:
        return None
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, proj, _ = model(x)
    return proj[0].mean(dim=0).cpu().numpy()


# ============================================================
# Tests
# ============================================================

def test_bit_accuracy(model, tokenizer, train_anchors, holdout_anchors, device):
    """Test 1: Bit accuracy on train and holdout anchors."""
    print_section("TEST 1: BIT ACCURACY")

    results = {}
    for split_name, anchors in [('train', train_anchors), ('holdout', holdout_anchors)]:
        correct_total = 0
        bits_total = 0
        per_word = []

        for word, data in anchors.items():
            proj = get_proj_da13(model, tokenizer, word, device)
            if proj is None:
                continue

            pred_bits = to_binary(proj)
            gold_bits = (data['target'] > 0).float().numpy().astype(np.int8)

            n_correct = int(np.sum(pred_bits == gold_bits))
            accuracy = n_correct / N_BITS
            correct_total += n_correct
            bits_total += N_BITS

            per_word.append({'word': word, 'accuracy': round(accuracy, 3)})

        mean_acc = correct_total / max(bits_total, 1)
        per_word.sort(key=lambda x: x['accuracy'])

        print(f"  {split_name}: {mean_acc:.1%} ({correct_total}/{bits_total} bits)")
        if per_word:
            print(f"    Worst 3: {[f\"{w['word']}={w['accuracy']:.0%}\" for w in per_word[:3]]}")
            print(f"    Best 3:  {[f\"{w['word']}={w['accuracy']:.0%}\" for w in per_word[-3:]]}")

        results[split_name] = {
            'mean_accuracy': round(mean_acc, 4),
            'n_concepts': len(per_word),
            'worst_3': per_word[:3],
        }

    return results


def test_subsumption(model, tokenizer, train_anchors, holdout_anchors, device, mapper):
    """Test 2: Subsumption rate."""
    print_section("TEST 2: SUBSUMPTION")

    from danza_bootstrap import HOLDOUT_INFO
    from danza_63bit import load_primitives

    prim_data = load_primitives()
    all_anchors = {**train_anchors, **holdout_anchors}

    # Build subsumption pairs from holdout info
    correct = 0
    total = 0

    for h_word, info in HOLDOUT_INFO.items():
        h_proj = get_proj_da13(model, tokenizer, h_word, device)
        if h_proj is None:
            continue

        h_prime = proj_to_prime(h_proj, mapper)

        for y_word in info.get('parents', []):
            y_proj = get_proj_da13(model, tokenizer, y_word, device)
            if y_proj is None:
                continue
            y_prime = proj_to_prime(y_proj, mapper)
            if h_prime > 1 and y_prime > 1:
                if h_prime % y_prime == 0:  # h subsumes y
                    correct += 1
                total += 1

    rate = correct / max(total, 1)
    print(f"  Holdout subsumption: {correct}/{total} = {rate:.1%}")

    return {'correct': correct, 'total': total, 'rate': round(rate, 4)}


def test_ternary_distribution(model, tokenizer, anchors, device):
    """Test 3: Distribution of ternary values {-1, 0, +1}."""
    print_section("TEST 3: TERNARY DISTRIBUTION")

    all_values = []
    for word in anchors:
        proj = get_proj_da13(model, tokenizer, word, device)
        if proj is not None:
            tern = to_ternary(proj)
            all_values.extend(tern.tolist())

    if not all_values:
        return {'error': 'no projections'}

    arr = np.array(all_values)
    neg = float(np.mean(arr == -1))
    zero = float(np.mean(arr == 0))
    pos = float(np.mean(arr == 1))

    print(f"  -1: {neg:.1%}  |  0: {zero:.1%}  |  +1: {pos:.1%}")

    if zero < 0.01:
        print(f"  WARNING: Zeros collapsed to {zero:.2%} — binary, not ternary!")
    elif zero < 0.05:
        print(f"  NOTE: Very few zeros ({zero:.1%}) — near-binary behavior")

    return {
        'neg': round(neg, 4), 'zero': round(zero, 4), 'pos': round(pos, 4),
        'collapsed_to_binary': zero < 0.01,
    }


def test_signature_uniqueness(model, tokenizer, anchors, device, mapper):
    """Test 4: Are all signatures unique?"""
    print_section("TEST 4: SIGNATURE UNIQUENESS")

    signatures = {}
    for word in anchors:
        proj = get_proj_da13(model, tokenizer, word, device)
        if proj is None:
            continue
        sig = tuple(to_binary(proj).tolist())
        if sig not in signatures:
            signatures[sig] = []
        signatures[sig].append(word)

    n_concepts = sum(len(v) for v in signatures.values())
    n_unique = len(signatures)
    duplicates = {k: v for k, v in signatures.items() if len(v) > 1}

    print(f"  Concepts: {n_concepts}")
    print(f"  Unique signatures: {n_unique}")
    print(f"  Duplicates: {len(duplicates)}")

    if duplicates:
        for sig, words in list(duplicates.items())[:5]:
            print(f"    Collision: {words}")

    return {
        'n_concepts': n_concepts,
        'n_unique': n_unique,
        'n_duplicates': len(duplicates),
        'uniqueness_rate': round(n_unique / max(n_concepts, 1), 4),
    }


def test_analogy(model, tokenizer, quads, device, mapper, validator):
    """Test 5: R3 analogy verification on bootstrap quads."""
    print_section("TEST 5: ANALOGY VERIFICATION (R3)")

    correct = 0
    total = 0

    for quad in quads:
        if len(quad) < 4:
            continue
        a, b, c, d = quad[:4]

        projs = {}
        skip = False
        for word in [a, b, c, d]:
            p = get_proj_da13(model, tokenizer, word, device)
            if p is None:
                skip = True
                break
            projs[word] = p

        if skip:
            continue

        # Compute primes
        primes = {w: proj_to_prime(p, mapper) for w, p in projs.items()}
        if any(v == 1 for v in primes.values()):
            continue

        # Predict D
        d_pred = validator.analogy(primes[a], primes[b], primes[c])
        if d_pred == primes[d]:
            correct += 1
        total += 1

    rate = correct / max(total, 1)
    print(f"  Analogy accuracy: {correct}/{total} = {rate:.1%}")

    return {'correct': correct, 'total': total, 'rate': round(rate, 4)}


# ============================================================
# MAIN
# ============================================================

def main():
    print_header("EXP-F4.4: D-A13 (GPT-2 MEDIUM 355M) EVALUATION")

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Verify checkpoint exists
    if not os.path.exists(DA13_CKPT):
        print(f"\n  ERROR: Checkpoint not found: {DA13_CKPT}")
        print(f"  D-A13 training may not have saved model_best.pt")
        # Try step checkpoint
        alt = os.path.join(DA13_CKPT_DIR, 'model_step50000.pt')
        if os.path.exists(alt):
            print(f"  Found alternative: {alt}")
            globals()['DA13_CKPT'] = alt
        else:
            print(f"  No checkpoints found. Exiting.")
            return

    model, tokenizer = load_da13_model(str(device))

    mapper = PrimeMapper(N_BITS)
    validator = TriadicValidator()

    # Load anchors
    from danza_63bit import load_primitives, load_anchors
    from danza_bootstrap import BOOTSTRAP_QUADS

    prim_data = load_primitives()
    all_anchors, _ = load_anchors(prim_data)

    # Split into train/holdout
    from danza_bootstrap import get_split
    train_words, holdout_words = get_split()
    train_anchors = {w: all_anchors[w] for w in train_words if w in all_anchors}
    holdout_anchors = {w: all_anchors[w] for w in holdout_words if w in all_anchors}
    print(f"  Train anchors: {len(train_anchors)}, Holdout: {len(holdout_anchors)}")

    # Run all tests
    bit_acc = test_bit_accuracy(model, tokenizer, train_anchors, holdout_anchors, str(device))
    sub_rate = test_subsumption(model, tokenizer, train_anchors, holdout_anchors, str(device), mapper)
    ternary = test_ternary_distribution(model, tokenizer, {**train_anchors, **holdout_anchors}, str(device))
    uniqueness = test_signature_uniqueness(model, tokenizer, {**train_anchors, **holdout_anchors}, str(device), mapper)
    analogy = test_analogy(model, tokenizer, BOOTSTRAP_QUADS, str(device), mapper, validator)

    # Comparison with Run 15 (40M)
    print_section("COMPARISON: 40M vs 355M")
    print(f"  {'Metric':<30} {'40M (Run 15)':>15} {'355M (D-A13)':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Bit accuracy (holdout)':<30} {'~89%':>15} {bit_acc.get('holdout', {}).get('mean_accuracy', 0):.1%}".rjust(15))
    print(f"  {'Subsumption (holdout)':<30} {'87.1%':>15} {sub_rate['rate']:.1%}".rjust(15))
    print(f"  {'Ternary zeros':<30} {'25.3%':>15} {ternary.get('zero', 0):.1%}".rjust(15))
    print(f"  {'Unique signatures':<30} {'100%':>15} {uniqueness.get('uniqueness_rate', 0):.1%}".rjust(15))

    # Summary
    print_section("SUMMARY")
    print(f"  Bit accuracy (holdout): {bit_acc.get('holdout', {}).get('mean_accuracy', 0):.1%}")
    print(f"  Subsumption: {sub_rate['rate']:.1%}")
    print(f"  Zeros collapsed: {'YES' if ternary.get('collapsed_to_binary') else 'NO'}")
    print(f"  Unique signatures: {uniqueness.get('uniqueness_rate', 0):.1%}")
    print(f"  Analogy (R3): {analogy['rate']:.1%}")

    save_results({
        'test': 'EXP-F4.4',
        'model': 'D-A13 (GPT-2 Medium 355M)',
        'checkpoint': DA13_CKPT,
        'n_bits': N_BITS,
        'bit_accuracy': bit_acc,
        'subsumption': sub_rate,
        'ternary_distribution': ternary,
        'signature_uniqueness': uniqueness,
        'analogy': analogy,
    }, 'f4_4_d_a13_eval.json')


if __name__ == '__main__':
    import torch
    main()

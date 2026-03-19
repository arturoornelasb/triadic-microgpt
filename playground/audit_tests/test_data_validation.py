"""
EXP-F0: Data Validation — Pre-flight check
============================================

MUST RUN BEFORE any other test.

Verifies the integrity of the data pipeline:
  1. Anchor gold targets: are they internally consistent?
  2. Model accuracy: how well does the model reproduce gold targets?
  3. Word coverage: which test words are in-domain vs OOD?
  4. Gold vs Model: do conclusions change when using gold targets directly?

This test determines whether failures in subsequent tests are:
  a) Real findings about the theory, or
  b) Artifacts of the model not having learned the targets perfectly

Usage:
  cd C:\\Github\\triadic-microgpt
  python playground/audit_tests/test_data_validation.py
"""

import os
import sys
import json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from common import (
    load_run15, get_projection, get_projections_batch,
    to_binary, hamming, cosine_sim, bits_shared, proj_to_prime,
    save_results, print_header, print_section, N_BITS,
)
from src.triadic import PrimeMapper, TriadicValidator


# ============================================================
# All words used across all tests
# ============================================================

ALL_TEST_WORDS = {
    # F2.1 indifference + false opposites
    'love', 'hate', 'indifference', 'passion', 'apathy', 'anger', 'joy',
    'sadness', 'boredom', 'excitement', 'obsession', 'devotion', 'contempt',
    'resentment', 'numbness', 'fear',
    'creative', 'logical', 'man', 'woman', 'socialism', 'capitalism',
    'hot', 'cold', 'order', 'chaos', 'light', 'darkness', 'presence', 'absence',
    # F2.2 aristotelian
    'tall', 'short', 'fast', 'slow', 'bright', 'dark', 'heavy', 'loud',
    'quiet', 'rich', 'poor', 'young', 'old', 'happy', 'sad',
    'alive', 'dead', 'true', 'false', 'present', 'absent', 'even', 'odd',
    'married', 'single', 'guilty', 'innocent', 'legal', 'illegal',
    'open', 'closed', 'visible', 'invisible', 'possible', 'impossible',
    'sight', 'blindness', 'sound', 'silence', 'knowledge', 'ignorance',
    'hope', 'despair', 'health', 'disease', 'wealth', 'poverty',
    'freedom', 'captivity', 'trust', 'distrust', 'courage', 'cowardice',
    'parent', 'child', 'teacher', 'student', 'buyer', 'seller',
    'doctor', 'patient', 'predator', 'prey', 'cause', 'effect',
    'question', 'answer', 'host', 'guest', 'employer', 'employee',
    'leader', 'follower',
    # F2.5 enantiodromia
    'tyranny', 'fanaticism', 'tolerance', 'perfectionism', 'acceptance',
    'recklessness', 'caution', 'boldness', 'greed', 'generosity',
    'pride', 'humility', 'confidence', 'rage', 'calm', 'irritation',
    'starvation', 'abundance', 'hunger', 'isolation', 'connection',
    'solitude', 'worship', 'admiration', 'interest',
    # F3.1 PF-Q6
    'pain', 'anxiety', 'pleasure', 'surprise', 'disgust', 'shame', 'guilt',
    'stone', 'water', 'number', 'triangle', 'gravity', 'velocity',
    'mass', 'distance', 'volume', 'density', 'carbon', 'iron',
    'oxygen', 'nitrogen', 'hydrogen', 'happiness',
}


def main():
    print_header("EXP-F0: DATA VALIDATION (pre-flight)")

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"  Loading Run 15...")
    model, tokenizer = load_run15(str(device))

    from danza_63bit import load_primitives, load_anchors
    prim_data = load_primitives()
    anchors, _ = load_anchors(prim_data)

    mapper = PrimeMapper(N_BITS)
    validator = TriadicValidator()

    # Load gold primes
    gold_path = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)),
                             'data', 'gold_primes_64.json')
    with open(gold_path) as f:
        gold_primes = json.load(f)
    gold_lower = {k.lower(): k for k in gold_primes}

    # ============================================================
    # 1. Word Coverage Classification
    # ============================================================
    print_section("1. WORD COVERAGE CLASSIFICATION")

    coverage = {'anchor': [], 'gold': [], 'ood': []}
    for w in sorted(ALL_TEST_WORDS):
        if w in anchors:
            coverage['anchor'].append(w)
        elif w in gold_primes or w.capitalize() in gold_primes or w in gold_lower:
            coverage['gold'].append(w)
        else:
            coverage['ood'].append(w)

    print(f"  Anchor (supervised gold target):  {len(coverage['anchor'])} words")
    print(f"  Gold primes (precomputed, no target): {len(coverage['gold'])} words")
    print(f"  Out-of-domain (no reference): {len(coverage['ood'])} words")
    print(f"  OOD words: {coverage['ood'][:15]}{'...' if len(coverage['ood']) > 15 else ''}")

    # ============================================================
    # 2. Model Accuracy on Anchors
    # ============================================================
    print_section("2. MODEL ACCURACY ON ANCHOR TARGETS")

    accuracies = {}
    for w in anchors:
        proj = get_projection(model, tokenizer, w, str(device), max_tokens=4)
        if proj is None:
            continue
        gold_bits = (anchors[w]['target'] > 0).float().numpy().astype(np.int8)
        pred_bits = to_binary(proj)
        acc = float(np.mean(gold_bits == pred_bits))
        accuracies[w] = acc

    mean_acc = np.mean(list(accuracies.values()))
    print(f"  Mean bit accuracy: {mean_acc:.1%}")
    print(f"  Anchors at 100%: {sum(1 for a in accuracies.values() if a >= 0.999)}/{len(accuracies)}")
    print(f"  Anchors >= 90%: {sum(1 for a in accuracies.values() if a >= 0.9)}/{len(accuracies)}")
    print(f"  Anchors < 80%: {sum(1 for a in accuracies.values() if a < 0.8)}/{len(accuracies)}")

    worst = sorted(accuracies.items(), key=lambda x: x[1])[:5]
    print(f"  Worst 5: {[(w, f'{a:.0%}') for w, a in worst]}")

    # ============================================================
    # 3. Gold vs Model: Critical Comparisons
    # ============================================================
    print_section("3. GOLD vs MODEL COMPARISON")
    print(f"  Does the theory hold in gold data even if model doesn't perfectly learn it?\n")

    # Key pairs to compare
    comparison_pairs = [
        ('love', 'hate', 'indifference', 'Indifference thesis'),
        ('hot', 'cold', 'warm', 'Contraries'),
        ('man', 'woman', 'child', 'Complements'),
        ('creative', 'logical', 'boring', 'Orthogonal (not opposite)'),
    ]

    gold_vs_model = []
    for w1, w2, w3, label in comparison_pairs:
        # Check all three are anchors
        all_anchor = all(w in anchors for w in [w1, w2, w3])

        if all_anchor:
            g1 = (anchors[w1]['target'] > 0).float().numpy().astype(np.int8)
            g2 = (anchors[w2]['target'] > 0).float().numpy().astype(np.int8)
            g3 = (anchors[w3]['target'] > 0).float().numpy().astype(np.int8)
            h12_gold = int(np.sum(g1 != g2))
            h13_gold = int(np.sum(g1 != g3))
        else:
            h12_gold = h13_gold = None

        p1 = get_projection(model, tokenizer, w1, str(device), max_tokens=4)
        p2 = get_projection(model, tokenizer, w2, str(device), max_tokens=4)
        p3 = get_projection(model, tokenizer, w3, str(device), max_tokens=4)

        if p1 is not None and p2 is not None and p3 is not None:
            h12_model = hamming(p1, p2)
            h13_model = hamming(p1, p3)
        else:
            h12_model = h13_model = None

        entry = {
            'label': label, 'w1': w1, 'w2': w2, 'w3': w3,
            'h12_gold': h12_gold, 'h13_gold': h13_gold,
            'h12_model': h12_model, 'h13_model': h13_model,
        }
        gold_vs_model.append(entry)

        g_str = f"H({w1},{w2})={h12_gold} H({w1},{w3})={h13_gold}" if h12_gold is not None else "N/A"
        m_str = f"H({w1},{w2})={h12_model} H({w1},{w3})={h13_model}" if h12_model is not None else "N/A"
        print(f"  {label}:")
        print(f"    GOLD:  {g_str}")
        print(f"    MODEL: {m_str}")
        if h12_gold is not None and h12_model is not None:
            agree = (h12_gold > h13_gold) == (h12_model > h13_model)
            print(f"    Agreement: {'YES' if agree else 'NO — model disagrees with gold!'}")
        print()

    # ============================================================
    # 4. INDIFFERENCE THESIS: Deep Analysis
    # ============================================================
    print_section("4. INDIFFERENCE THESIS: GOLD vs MODEL")

    g_love = (anchors['love']['target'] > 0).float().numpy().astype(np.int8)
    g_hate = (anchors['hate']['target'] > 0).float().numpy().astype(np.int8)
    g_indiff = (anchors['indifference']['target'] > 0).float().numpy().astype(np.int8)

    # Shared primitives analysis
    love_prims = set(np.where(g_love == 1)[0])
    hate_prims = set(np.where(g_hate == 1)[0])
    indiff_prims = set(np.where(g_indiff == 1)[0])

    shared_lh = love_prims & hate_prims
    shared_li = love_prims & indiff_prims
    only_love_vs_hate = love_prims - hate_prims
    only_love_vs_indiff = love_prims - indiff_prims

    print(f"  GOLD TARGET ANALYSIS:")
    print(f"    love has {len(love_prims)} primitives")
    print(f"    hate has {len(hate_prims)} primitives")
    print(f"    indifference has {len(indiff_prims)} primitives")
    print(f"")
    print(f"    love/hate share {len(shared_lh)} primitives ({len(shared_lh)/max(len(love_prims|hate_prims),1):.0%} Jaccard)")
    print(f"    love/indifference share {len(shared_li)} primitives ({len(shared_li)/max(len(love_prims|indiff_prims),1):.0%} Jaccard)")
    print(f"")
    print(f"    Hamming(love, hate) in GOLD = {int(np.sum(g_love != g_hate))}")
    print(f"    Hamming(love, indifference) in GOLD = {int(np.sum(g_love != g_indiff))}")

    # Named primitives for the difference
    only_love_names = [prim_data['bit_to_name'].get(i, f'bit{i}') for i in only_love_vs_hate]
    shared_lh_names = [prim_data['bit_to_name'].get(i, f'bit{i}') for i in sorted(shared_lh)]

    print(f"\n    Primitives love has but hate doesn't: {only_love_names}")
    print(f"    Primitives love/hate SHARE: {shared_lh_names[:10]}...")

    # What makes indifference different?
    indiff_unique = indiff_prims - love_prims - hate_prims
    indiff_unique_names = [prim_data['bit_to_name'].get(i, f'bit{i}') for i in sorted(indiff_unique)]
    print(f"    Primitives ONLY in indifference: {indiff_unique_names}")

    # GCD analysis with gold
    pr_love = proj_to_prime(g_love.astype(float), mapper)
    pr_hate = proj_to_prime(g_hate.astype(float), mapper)
    pr_indiff = proj_to_prime(g_indiff.astype(float), mapper)

    gcd_lh = validator.intersect(pr_love, pr_hate)
    gcd_li = validator.intersect(pr_love, pr_indiff)
    sim_lh = validator.similarity(pr_love, pr_hate)
    sim_li = validator.similarity(pr_love, pr_indiff)

    print(f"\n    GCD(love, hate) = {gcd_lh} (similarity: {sim_lh:.3f})")
    print(f"    GCD(love, indifference) = {gcd_li} (similarity: {sim_li:.3f})")

    gold_thesis = sim_lh > sim_li
    print(f"\n    GOLD CONFIRMS THESIS: {'YES' if gold_thesis else 'NO'}")
    print(f"    (love/hate are MORE similar = 'primos hermanos')")

    # ============================================================
    # 5. Reliability Classification
    # ============================================================
    print_section("5. TEST RELIABILITY CLASSIFICATION")

    print(f"  Tests should distinguish between:")
    print(f"    HIGH confidence: result uses only anchor words (gold targets)")
    print(f"    MEDIUM confidence: result uses gold_primes words (precomputed)")
    print(f"    LOW confidence: result uses OOD words (no reference)")
    print(f"")
    print(f"  Recommendation: Each test should report results on BOTH:")
    print(f"    1. Gold targets (ground truth about the THEORY)")
    print(f"    2. Model predictions (how well the MODEL captures the theory)")
    print(f"")
    print(f"  If GOLD passes but MODEL fails -> model needs more training")
    print(f"  If GOLD fails -> the theory itself has a problem")

    # ============================================================
    # 6. OOD Representation Quality
    # ============================================================
    print_section("6. OOD REPRESENTATION QUALITY")
    print(f"  Model trained on TinyStories (children's stories)")
    print(f"  OOD words get representations from GPT's pretrained BPE,")
    print(f"  but may not have meaningful triadic projections.\n")

    # Test: do OOD words produce diverse or collapsed signatures?
    ood_projs = get_projections_batch(model, tokenizer, coverage['ood'], str(device), max_tokens=4)
    anchor_projs = get_projections_batch(model, tokenizer, coverage['anchor'], str(device), max_tokens=4)

    def signature_stats(projs):
        if not projs:
            return {}
        active_counts = [int(to_binary(p).sum()) for p in projs.values()]
        sigs = [tuple(to_binary(p).tolist()) for p in projs.values()]
        unique = len(set(sigs))
        return {
            'n': len(projs),
            'mean_active': round(float(np.mean(active_counts)), 1),
            'std_active': round(float(np.std(active_counts)), 1),
            'min_active': min(active_counts),
            'max_active': max(active_counts),
            'unique_sigs': unique,
            'uniqueness': round(unique / len(projs), 3),
        }

    anchor_stats = signature_stats(anchor_projs)
    ood_stats = signature_stats(ood_projs)

    print(f"  {'Metric':<25} {'Anchors':>12} {'OOD':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    for key in ['n', 'mean_active', 'std_active', 'min_active', 'max_active', 'unique_sigs', 'uniqueness']:
        a = anchor_stats.get(key, 'N/A')
        o = ood_stats.get(key, 'N/A')
        print(f"  {key:<25} {str(a):>12} {str(o):>12}")

    # Save everything
    save_results({
        'test': 'EXP-F0-DATA-VALIDATION',
        'model_accuracy': {
            'mean': round(mean_acc, 4),
            'per_word': {w: round(a, 4) for w, a in accuracies.items()},
        },
        'coverage': {k: v for k, v in coverage.items()},
        'gold_vs_model': gold_vs_model,
        'indifference_thesis': {
            'gold_hamming_love_hate': int(np.sum(g_love != g_hate)),
            'gold_hamming_love_indiff': int(np.sum(g_love != g_indiff)),
            'gold_similarity_love_hate': round(sim_lh, 4),
            'gold_similarity_love_indiff': round(sim_li, 4),
            'gold_confirms_thesis': gold_thesis,
        },
        'anchor_stats': anchor_stats,
        'ood_stats': ood_stats,
    }, 'f0_data_validation.json')


if __name__ == '__main__':
    main()

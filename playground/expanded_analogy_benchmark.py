"""
E3 — Expanded Analogy Benchmark (50+ quadruples).

Evaluates Run 15 (v1.4-strongalign) on a substantially larger set of
analogies than the original 26-analogy benchmark (benchmarks/scripts/
analogy_benchmark.py), which reported 65.4% verification accuracy.

This experiment tests whether the triadic head's algebraic analogy
capabilities generalize across 12 semantic categories with 50+ tests,
measuring:
  - Verification rate (Jaccard similarity > 0.3)
  - Per-category and easy/hard breakdowns
  - Offset cosine: cos(b-a, d-c) in projection space
  - Top-1 retrieval: is d closest to the predicted vector?

Zero GPU training — loads the checkpoint and runs ~200 word inferences.

Usage:
  python playground/expanded_analogy_benchmark.py
  python playground/expanded_analogy_benchmark.py --checkpoint path/to/model.pt
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime

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


# ============================================================
# Analogy quadruples — 50+ organized by category
# Format: (a, b, c, d, category, difficulty)
#   "a is to b as c is to d"
#   difficulty: "easy" = same-domain, "hard" = cross-domain
# ============================================================

ANALOGIES = [
    # ---- Gender ----
    ("king", "queen", "man", "woman", "gender", "easy"),
    ("father", "mother", "son", "daughter", "gender", "easy"),
    ("prince", "princess", "boy", "girl", "gender", "easy"),
    ("brother", "sister", "uncle", "aunt", "gender", "easy"),
    ("husband", "wife", "groom", "bride", "gender", "easy"),

    # ---- Family ----
    ("father", "mother", "brother", "sister", "family", "easy"),
    ("father", "son", "mother", "daughter", "family", "easy"),
    ("parent", "child", "teacher", "student", "family", "hard"),

    # ---- Size ----
    ("big", "small", "tall", "short", "size", "easy"),
    ("large", "tiny", "wide", "narrow", "size", "easy"),
    ("giant", "dwarf", "mountain", "hill", "size", "hard"),

    # ---- Temperature ----
    ("hot", "cold", "summer", "winter", "temperature", "easy"),
    ("warm", "cool", "fire", "ice", "temperature", "easy"),
    ("hot", "cold", "day", "night", "temperature", "hard"),

    # ---- Emotion ----
    ("happy", "sad", "love", "hate", "emotion", "easy"),
    ("joy", "sorrow", "laugh", "cry", "emotion", "easy"),
    ("brave", "afraid", "strong", "weak", "emotion", "easy"),

    # ---- Animal ----
    ("dog", "puppy", "cat", "kitten", "animal", "easy"),
    ("bird", "nest", "fish", "pond", "animal", "hard"),
    ("horse", "foal", "cow", "calf", "animal", "easy"),
    ("dog", "bark", "cat", "meow", "animal", "hard"),

    # ---- Profession ----
    ("doctor", "hospital", "teacher", "school", "profession", "easy"),
    ("chef", "kitchen", "pilot", "airplane", "profession", "easy"),
    ("king", "castle", "judge", "court", "profession", "hard"),

    # ---- Geography ----
    ("sun", "day", "moon", "night", "geography", "easy"),
    ("mountain", "valley", "peak", "base", "geography", "easy"),
    ("river", "lake", "stream", "pond", "geography", "easy"),

    # ---- Color ----
    ("red", "blue", "green", "yellow", "color", "easy"),
    ("black", "white", "dark", "light", "color", "easy"),

    # ---- Action ----
    ("run", "walk", "fly", "swim", "action", "easy"),
    ("read", "book", "eat", "food", "action", "hard"),
    ("open", "close", "start", "stop", "action", "easy"),
    ("push", "pull", "give", "take", "action", "easy"),

    # ---- Degree ----
    ("good", "better", "bad", "worse", "degree", "easy"),
    ("fast", "faster", "slow", "slower", "degree", "easy"),

    # ---- Opposite ----
    ("up", "down", "left", "right", "opposite", "easy"),
    ("old", "young", "big", "small", "opposite", "easy"),
    ("hot", "cold", "fast", "slow", "opposite", "easy"),
    ("love", "hate", "friend", "enemy", "opposite", "easy"),

    # ---- Extra analogies for breadth ----
    ("man", "woman", "boy", "girl", "gender", "easy"),
    ("king", "queen", "prince", "princess", "gender", "easy"),
    ("happy", "sad", "peace", "war", "emotion", "hard"),
    ("morning", "night", "summer", "winter", "temperature", "hard"),
    ("dog", "cat", "horse", "cow", "animal", "easy"),
    ("big", "small", "fast", "slow", "size", "easy"),
    ("fire", "water", "sun", "moon", "geography", "hard"),
    ("friend", "enemy", "peace", "war", "opposite", "easy"),
    ("run", "walk", "swim", "float", "action", "easy"),
    ("doctor", "nurse", "king", "queen", "profession", "hard"),
    ("tree", "forest", "star", "sky", "geography", "hard"),
    ("bread", "food", "milk", "drink", "action", "hard"),
]

# Vocabulary pool for retrieval (all D answers + generous distractors).
# Intentionally broad to make top-1 retrieval challenging.
VOCAB_POOL = sorted(set([
    # People / gender
    "king", "queen", "man", "woman", "boy", "girl", "prince", "princess",
    "father", "mother", "son", "daughter", "brother", "sister", "uncle",
    "aunt", "husband", "wife", "groom", "bride", "parent", "child",
    # Animals
    "dog", "puppy", "cat", "kitten", "bird", "fish", "horse", "foal",
    "cow", "calf", "pig", "sheep", "bear", "rabbit", "lion",
    # Professions / places
    "doctor", "nurse", "teacher", "student", "chef", "pilot", "judge",
    "hospital", "school", "kitchen", "airplane", "castle", "court",
    # Nature / geography
    "sun", "moon", "star", "sky", "mountain", "valley", "peak", "base",
    "river", "lake", "stream", "pond", "forest", "tree", "hill",
    "nest", "ocean", "cloud", "rain", "snow",
    # Emotions / states
    "happy", "sad", "love", "hate", "joy", "sorrow", "brave", "afraid",
    "strong", "weak", "peace", "war", "hope", "fear", "angry", "kind",
    # Colors
    "red", "blue", "green", "yellow", "black", "white", "dark", "light",
    "pink", "purple",
    # Size / degree
    "big", "small", "tall", "short", "large", "tiny", "wide", "narrow",
    "giant", "dwarf", "good", "better", "bad", "worse", "fast", "faster",
    "slow", "slower", "old", "young",
    # Actions
    "run", "walk", "fly", "swim", "float", "read", "eat", "open", "close",
    "start", "stop", "push", "pull", "give", "take", "laugh", "cry",
    "jump", "fall", "climb", "sleep", "sing", "dance",
    # Temperature / elements
    "hot", "cold", "warm", "cool", "fire", "ice", "summer", "winter",
    "day", "night", "morning", "evening", "water",
    # Objects / misc
    "book", "food", "bread", "milk", "drink", "door", "window", "table",
    "chair", "bed", "pen", "lamp", "cake", "apple", "house", "car",
    "garden", "park", "magic", "dream", "story", "game", "music",
    "friend", "enemy",
    # Sounds
    "bark", "meow",
    # Family extra
    "up", "down", "left", "right",
]))


# ============================================================
# Model loading
# ============================================================

def load_model_and_tokenizer(ckpt_path, tok_path, device):
    """Load a TriadicGPT checkpoint and its tokenizer."""
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Tokenizer  : {tok_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'],
        block_size=cfg['block_size'],
        n_layer=cfg['n_layer'],
        n_embd=cfg['n_embd'],
        n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'],
        dropout=0.0,  # no dropout at eval
    )
    model = TriadicGPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    step = ckpt.get('step', '?')
    loss = ckpt.get('loss', '?')
    print(f"  Step={step}  Loss={loss}")
    print(f"  Config: {config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits")
    print(f"  Params: {model.num_params():,}")

    tokenizer = BPETokenizer.load(tok_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    return model, tokenizer, config


# ============================================================
# Projection helpers
# ============================================================

@torch.no_grad()
def get_word_projection(model, tokenizer, word, device):
    """
    Get the continuous triadic projection for a single word.

    Returns a numpy array of shape (n_bits,) in [-1, 1] (tanh range),
    or None if the word cannot be tokenized.
    """
    ids = tokenizer.encode(word, add_special=False)
    if not ids:
        return None
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, triadic_proj, _ = model(x)
    # Mean-pool over token positions
    proj = triadic_proj[0].mean(dim=0).cpu().numpy()
    return proj


def compute_all_projections(model, tokenizer, words, device):
    """
    Compute projections for all words. Returns dict word -> numpy array.
    Skips words that fail to tokenize.
    """
    projections = {}
    for word in words:
        if word in projections:
            continue
        proj = get_word_projection(model, tokenizer, word, device)
        if proj is not None:
            projections[word] = proj
    return projections


# ============================================================
# Metric functions
# ============================================================

def cosine_similarity(a, b):
    """Cosine similarity between two numpy vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        return 0.0
    return float(dot / norm)


def jaccard_prime_similarity(phi_a, phi_b):
    """Jaccard similarity of prime factor sets (delegates to TriadicValidator)."""
    return TriadicValidator.similarity(phi_a, phi_b)


def evaluate_single_analogy(a_proj, b_proj, c_proj, d_proj,
                            phi_a, phi_b, phi_c, phi_d,
                            mapper, all_projections, all_primes,
                            a_word, b_word, c_word, d_word,
                            verification_threshold=0.3):
    """
    Evaluate a single analogy (a:b::c:d).

    Returns a dict with all metrics for this analogy.
    """
    # --- Algebraic analogy via prime factorization ---
    predicted_prime = TriadicValidator.analogy(phi_a, phi_b, phi_c)
    algebraic_sim = jaccard_prime_similarity(predicted_prime, phi_d)
    verified = algebraic_sim > verification_threshold

    # --- Offset cosine in projection space ---
    # Does the direction b-a match d-c?
    offset_ab = b_proj - a_proj
    offset_cd = d_proj - c_proj
    offset_cosine = cosine_similarity(offset_ab, offset_cd)

    # --- Vector analogy: predicted = b - a + c ---
    predicted_vec = b_proj - a_proj + c_proj

    # --- Top-1 retrieval ---
    # Among all vocab words (excluding a, b, c), which is closest to
    # the algebraically predicted prime?
    exclude = {a_word, b_word, c_word}

    # Prime-algebra retrieval: rank by Jaccard similarity to predicted_prime
    best_prime_word = None
    best_prime_sim = -1.0
    for w, phi_w in all_primes.items():
        if w in exclude:
            continue
        s = jaccard_prime_similarity(predicted_prime, phi_w)
        if s > best_prime_sim:
            best_prime_sim = s
            best_prime_word = w

    prime_top1_correct = (best_prime_word == d_word)

    # Vector retrieval: rank by cosine to predicted_vec
    best_vec_word = None
    best_vec_sim = -1.0
    for w, proj_w in all_projections.items():
        if w in exclude:
            continue
        s = cosine_similarity(predicted_vec, proj_w)
        if s > best_vec_sim:
            best_vec_sim = s
            best_vec_word = w

    vec_top1_correct = (best_vec_word == d_word)

    # Direct cosine between predicted_vec and d
    vec_d_cosine = cosine_similarity(predicted_vec, d_proj)

    return {
        'algebraic_similarity': algebraic_sim,
        'verified': verified,
        'offset_cosine': offset_cosine,
        'vec_d_cosine': vec_d_cosine,
        'prime_top1': best_prime_word,
        'prime_top1_correct': prime_top1_correct,
        'vec_top1': best_vec_word,
        'vec_top1_correct': vec_top1_correct,
        'predicted_prime': str(predicted_prime),
        'phi_d': str(phi_d),
    }


# ============================================================
# Main evaluation
# ============================================================

def run_benchmark(model, tokenizer, config, device):
    """Run the full expanded analogy benchmark. Returns results dict."""
    n_bits = config.n_triadic_bits
    mapper = PrimeMapper(n_bits)

    # Collect all words that appear in analogies + vocab pool
    all_words = set(VOCAB_POOL)
    for a, b, c, d, _, _ in ANALOGIES:
        all_words.update([a, b, c, d])

    print(f"\n[1/3] Computing projections for {len(all_words)} words...")
    projections = compute_all_projections(model, tokenizer, all_words, device)
    encoded_count = len(projections)
    skipped = all_words - set(projections.keys())
    if skipped:
        print(f"  Skipped (could not tokenize): {sorted(skipped)}")
    print(f"  Encoded: {encoded_count}/{len(all_words)}")

    # Compute prime composites for all encoded words
    primes = {}
    for word, proj in projections.items():
        primes[word] = mapper.map(proj)

    # Evaluate analogies
    print(f"\n[2/3] Evaluating {len(ANALOGIES)} analogies...")
    results = []
    skipped_analogies = 0

    for a, b, c, d, category, difficulty in ANALOGIES:
        if any(w not in projections for w in [a, b, c, d]):
            missing = [w for w in [a, b, c, d] if w not in projections]
            skipped_analogies += 1
            results.append({
                'analogy': f"{a}:{b}::{c}:{d}",
                'category': category,
                'difficulty': difficulty,
                'skipped': True,
                'missing_words': missing,
            })
            continue

        metrics = evaluate_single_analogy(
            projections[a], projections[b],
            projections[c], projections[d],
            primes[a], primes[b], primes[c], primes[d],
            mapper, projections, primes,
            a, b, c, d,
        )

        results.append({
            'analogy': f"{a}:{b}::{c}:{d}",
            'category': category,
            'difficulty': difficulty,
            'skipped': False,
            **metrics,
        })

    # Aggregate
    evaluated = [r for r in results if not r['skipped']]
    n_eval = len(evaluated)

    if n_eval == 0:
        print("  ERROR: No analogies could be evaluated.")
        return None

    verification_pass = sum(1 for r in evaluated if r['verified'])
    prime_top1_correct = sum(1 for r in evaluated if r['prime_top1_correct'])
    vec_top1_correct = sum(1 for r in evaluated if r['vec_top1_correct'])

    offset_cosines = [r['offset_cosine'] for r in evaluated]
    alg_sims = [r['algebraic_similarity'] for r in evaluated]

    # Per-category breakdown
    categories = sorted(set(r['category'] for r in evaluated))
    per_category = {}
    for cat in categories:
        cat_results = [r for r in evaluated if r['category'] == cat]
        n_cat = len(cat_results)
        cat_verified = sum(1 for r in cat_results if r['verified'])
        cat_prime_top1 = sum(1 for r in cat_results if r['prime_top1_correct'])
        cat_vec_top1 = sum(1 for r in cat_results if r['vec_top1_correct'])
        cat_offset = [r['offset_cosine'] for r in cat_results]
        per_category[cat] = {
            'count': n_cat,
            'verification_rate': cat_verified / n_cat,
            'verification_pass': cat_verified,
            'prime_top1_accuracy': cat_prime_top1 / n_cat,
            'prime_top1_correct': cat_prime_top1,
            'vec_top1_accuracy': cat_vec_top1 / n_cat,
            'vec_top1_correct': cat_vec_top1,
            'mean_offset_cosine': float(np.mean(cat_offset)),
        }

    # Easy vs Hard breakdown
    easy_results = [r for r in evaluated if r['difficulty'] == 'easy']
    hard_results = [r for r in evaluated if r['difficulty'] == 'hard']

    def difficulty_stats(subset):
        n = len(subset)
        if n == 0:
            return {'count': 0, 'verification_rate': 0.0, 'prime_top1_accuracy': 0.0,
                    'vec_top1_accuracy': 0.0, 'mean_offset_cosine': 0.0}
        v = sum(1 for r in subset if r['verified'])
        pt = sum(1 for r in subset if r['prime_top1_correct'])
        vt = sum(1 for r in subset if r['vec_top1_correct'])
        oc = [r['offset_cosine'] for r in subset]
        return {
            'count': n,
            'verification_rate': v / n,
            'verification_pass': v,
            'prime_top1_accuracy': pt / n,
            'prime_top1_correct': pt,
            'vec_top1_accuracy': vt / n,
            'vec_top1_correct': vt,
            'mean_offset_cosine': float(np.mean(oc)),
        }

    easy_stats = difficulty_stats(easy_results)
    hard_stats = difficulty_stats(hard_results)

    summary = {
        'n_analogies_defined': len(ANALOGIES),
        'n_evaluated': n_eval,
        'n_skipped': skipped_analogies,
        'n_vocab_pool': len(VOCAB_POOL),
        'n_words_encoded': encoded_count,
        'overall': {
            'verification_rate': verification_pass / n_eval,
            'verification_pass': verification_pass,
            'prime_top1_accuracy': prime_top1_correct / n_eval,
            'prime_top1_correct': prime_top1_correct,
            'vec_top1_accuracy': vec_top1_correct / n_eval,
            'vec_top1_correct': vec_top1_correct,
            'mean_algebraic_similarity': float(np.mean(alg_sims)),
            'std_algebraic_similarity': float(np.std(alg_sims)),
            'mean_offset_cosine': float(np.mean(offset_cosines)),
            'std_offset_cosine': float(np.std(offset_cosines)),
        },
        'by_difficulty': {
            'easy': easy_stats,
            'hard': hard_stats,
        },
        'by_category': per_category,
    }

    return {
        'summary': summary,
        'details': results,
    }


# ============================================================
# Pretty-print report
# ============================================================

def print_report(data):
    """Print a formatted report to stdout."""
    s = data['summary']
    details = data['details']

    print()
    print("=" * 74)
    print("  E3 — EXPANDED ANALOGY BENCHMARK")
    print("=" * 74)

    print(f"\n  Analogies defined : {s['n_analogies_defined']}")
    print(f"  Evaluated         : {s['n_evaluated']}")
    print(f"  Skipped           : {s['n_skipped']}")
    print(f"  Vocab pool        : {s['n_vocab_pool']} words")
    print(f"  Words encoded     : {s['n_words_encoded']}")

    # Overall
    o = s['overall']
    print(f"\n  --- Overall Results ---")
    print(f"  Verification rate (sim > 0.3) : {o['verification_rate']:.1%}  "
          f"({o['verification_pass']}/{s['n_evaluated']})")
    print(f"  Prime top-1 retrieval         : {o['prime_top1_accuracy']:.1%}  "
          f"({o['prime_top1_correct']}/{s['n_evaluated']})")
    print(f"  Vector top-1 retrieval        : {o['vec_top1_accuracy']:.1%}  "
          f"({o['vec_top1_correct']}/{s['n_evaluated']})")
    print(f"  Mean algebraic similarity     : {o['mean_algebraic_similarity']:.4f} "
          f"(std {o['std_algebraic_similarity']:.4f})")
    print(f"  Mean offset cosine            : {o['mean_offset_cosine']:.4f} "
          f"(std {o['std_offset_cosine']:.4f})")

    # Easy vs Hard
    print(f"\n  --- Difficulty Breakdown ---")
    print(f"  {'':>6s}  {'Count':>5s}  {'Verif':>8s}  {'P-Top1':>8s}  {'V-Top1':>8s}  {'Offset':>8s}")
    print(f"  {'':>6s}  {'':>5s}  {'Rate':>8s}  {'Acc':>8s}  {'Acc':>8s}  {'Cos':>8s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for label, st in [('Easy', s['by_difficulty']['easy']),
                      ('Hard', s['by_difficulty']['hard'])]:
        if st['count'] == 0:
            continue
        print(f"  {label:>6s}  {st['count']:>5d}  {st['verification_rate']:>7.1%}"
              f"  {st['prime_top1_accuracy']:>7.1%}"
              f"  {st['vec_top1_accuracy']:>7.1%}"
              f"  {st['mean_offset_cosine']:>8.4f}")

    # Per-category
    print(f"\n  --- Per-Category Breakdown ---")
    print(f"  {'Category':<14s}  {'N':>3s}  {'Verif':>8s}  {'P-Top1':>8s}  {'V-Top1':>8s}  {'Offset':>8s}")
    print(f"  {'-'*14}  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for cat in sorted(s['by_category'].keys()):
        c = s['by_category'][cat]
        print(f"  {cat:<14s}  {c['count']:>3d}  {c['verification_rate']:>7.1%}"
              f"  {c['prime_top1_accuracy']:>7.1%}"
              f"  {c['vec_top1_accuracy']:>7.1%}"
              f"  {c['mean_offset_cosine']:>8.4f}")

    # Individual results table
    evaluated = [d for d in details if not d['skipped']]
    print(f"\n  --- Individual Results ---")
    print(f"  {'Analogy':<30s}  {'Cat':<12s} {'Diff':>4s}  "
          f"{'Verif':>5s}  {'AlgSim':>6s}  {'OffCos':>6s}  {'P-Top1':<12s}  {'V-Top1':<12s}")
    print(f"  {'-'*30}  {'-'*12} {'-'*4}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*12}")

    for r in evaluated:
        check = "PASS" if r['verified'] else "FAIL"
        p_mark = "OK" if r['prime_top1_correct'] else r['prime_top1']
        v_mark = "OK" if r['vec_top1_correct'] else r['vec_top1']
        diff = r['difficulty'][0].upper()
        print(f"  {r['analogy']:<30s}  {r['category']:<12s} {diff:>4s}  "
              f"{check:>5s}  {r['algebraic_similarity']:>6.3f}  {r['offset_cosine']:>6.3f}"
              f"  {p_mark:<12s}  {v_mark:<12s}")

    # Offset cosine distribution summary
    offsets = [r['offset_cosine'] for r in evaluated]
    pct_positive = sum(1 for o in offsets if o > 0) / len(offsets) if offsets else 0
    print(f"\n  Offset cosine distribution:")
    print(f"    min={min(offsets):.4f}  Q1={np.percentile(offsets, 25):.4f}  "
          f"median={np.median(offsets):.4f}  Q3={np.percentile(offsets, 75):.4f}  "
          f"max={max(offsets):.4f}")
    print(f"    Positive offsets: {pct_positive:.1%}")

    # Comparison note
    print(f"\n  --- Comparison ---")
    print(f"  Original benchmark (26 analogies):  65.4% verification")
    print(f"  This benchmark ({s['n_evaluated']} analogies):  "
          f"{o['verification_rate']:.1%} verification")

    print()
    print("=" * 74)


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='E3 - Expanded Analogy Benchmark (50+ quadruples)')
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint (default: Run 15 v1.4-strongalign)')
    parser.add_argument(
        '--tokenizer', type=str, default=None,
        help='Path to tokenizer JSON (default: same dir as checkpoint)')
    parser.add_argument(
        '--threshold', type=float, default=0.3,
        help='Verification threshold for Jaccard similarity (default: 0.3)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve paths
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(
            PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign',
            'model_L12_D512_B64_best.pt')

    if args.tokenizer:
        tok_path = args.tokenizer
    else:
        tok_path = os.path.join(os.path.dirname(ckpt_path), 'tokenizer.json')

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if not os.path.exists(tok_path):
        print(f"ERROR: Tokenizer not found: {tok_path}")
        sys.exit(1)

    # Load
    print()
    print("=" * 74)
    print("  Loading model...")
    print("=" * 74)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer, config = load_model_and_tokenizer(ckpt_path, tok_path, device)

    # Run benchmark
    data = run_benchmark(model, tokenizer, config, device)

    if data is None:
        sys.exit(1)

    # Print report
    print_report(data)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'expanded_analogy_benchmark.json')

    save_data = {
        'experiment': 'E3_expanded_analogy_benchmark',
        'date': datetime.now().isoformat(timespec='seconds'),
        'model_checkpoint': ckpt_path,
        'model_config': f"{config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits",
        'verification_threshold': args.threshold,
        **data,
    }

    # Convert any non-serializable types
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"  Results saved: {save_path}")


if __name__ == '__main__':
    main()

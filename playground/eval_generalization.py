"""
Generalization Evaluation for 49-Bit Concept GPT.

Tests whether the model correctly assigns primitives to:
  1. Tier 2 words (compound, 2-5 primitives) — used in subsumption pairs
     but NOT in supervised primitive targets during training.
  2. Completely unseen words — common TinyStories words absent from the
     seed lexicon entirely.

Usage:
  python playground/eval_generalization.py
  python playground/eval_generalization.py --checkpoint path/to/model.pt
  python playground/eval_generalization.py --top-k 5
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer
from conceptual_tokenizer.config import (
    PRIMITIVE_NAMES, PRIMITIVE_TO_CATEGORY, CATEGORY_NAMES, N_PRIMITIVES,
)
from conceptual_tokenizer.seed_lexicon import TIER_1, TIER_2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import ConceptTriadicGPT from training script
from playground.concept_gpt_49bit import ConceptTriadicGPT


# ============================================================
# Helpers
# ============================================================

def load_model_and_tokenizer(ckpt_path, tok_path, device):
    """Load checkpoint and tokenizer, return (model, tokenizer)."""
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Tokenizer:  {tok_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'],
        block_size=cfg['block_size'],
        n_layer=cfg['n_layer'],
        n_embd=cfg['n_embd'],
        n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'],
        dropout=cfg.get('dropout', 0.1),
    )

    activation = ckpt.get('args', {}).get('activation', 'tanh')
    model = ConceptTriadicGPT(config, activation=activation).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    step = ckpt.get('step', '?')
    loss = ckpt.get('loss', '?')
    print(f"  Step: {step}, Loss: {loss}")
    print(f"  Activation: {activation}")
    print(f"  Params: {model.num_params():,}")

    tokenizer = BPETokenizer.load(tok_path)
    print(f"  Vocab: {tokenizer.vocab_size}")

    return model, tokenizer


@torch.no_grad()
def get_projection(model, tokenizer, word, device, max_tok_len=4):
    """
    Get the 49-dim projection for a single word.

    Returns (proj_01, valid) where proj_01 is in [0, 1] range (49,)
    and valid indicates whether the word could be tokenized.
    """
    ids = tokenizer.encode(word, add_special=False)[:max_tok_len]
    if not ids:
        return None, False
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, proj, _ = model(x)
    # Mean-pool over token positions, map tanh [-1,1] -> [0,1]
    p = proj[0].mean(dim=0).cpu()
    p_01 = (p + 1) / 2
    return p_01, True


def top_k_primitives(proj_01, k=3):
    """Return list of (primitive_name, activation) for top-k activated bits."""
    topk = proj_01.topk(k)
    return [(PRIMITIVE_NAMES[i], v.item()) for v, i in zip(topk.values, topk.indices)]


# ============================================================
# Tier 2 Evaluation
# ============================================================

def evaluate_tier2(model, tokenizer, device, top_k=3):
    """
    Evaluate T2 compound words.

    For each T2 word, the expected primitives are those defined in the lexicon.
    We check:
      - top-1 hit: is ANY expected primitive the single highest activation?
      - top-K hit: is ANY expected primitive in the top-K activations?
      - any-hit rate: for words with 3+ expected primitives, what fraction
        of expected primitives land anywhere in top-K?

    Returns a results dict.
    """
    results = {
        'words': [],
        'top1_hits': 0,
        'topk_hits': 0,
        'total': 0,
        'any_hit_num': 0,  # sum of (found expected prims in topK)
        'any_hit_den': 0,  # sum of (total expected prims) for words with 3+
        'per_category': {},  # category -> {top1, topk, total}
    }

    for word, mapping in TIER_2.items():
        proj_01, valid = get_projection(model, tokenizer, word, device)
        if not valid:
            continue

        expected_prims = list(mapping.keys())
        expected_indices = set()
        for prim in expected_prims:
            if prim in PRIMITIVE_NAMES:
                expected_indices.add(PRIMITIVE_NAMES.index(prim))

        if not expected_indices:
            continue

        topk_vals = proj_01.topk(top_k)
        topk_indices = set(topk_vals.indices.tolist())
        top1_idx = proj_01.argmax().item()

        top1_hit = top1_idx in expected_indices
        topk_hit = bool(topk_indices & expected_indices)

        # Any-hit: how many expected prims land in top-K
        found_in_topk = len(topk_indices & expected_indices)

        results['total'] += 1
        if top1_hit:
            results['top1_hits'] += 1
        if topk_hit:
            results['topk_hits'] += 1

        # Any-hit only for words with 3+ expected primitives
        if len(expected_prims) >= 3:
            results['any_hit_num'] += found_in_topk
            results['any_hit_den'] += len(expected_indices)

        # Per-category: use the categories of expected primitives
        cats_seen = set()
        for prim in expected_prims:
            cat = PRIMITIVE_TO_CATEGORY.get(prim)
            if cat and cat not in cats_seen:
                cats_seen.add(cat)
                if cat not in results['per_category']:
                    results['per_category'][cat] = {'top1': 0, 'topk': 0, 'total': 0}
                results['per_category'][cat]['total'] += 1
                if top1_hit:
                    results['per_category'][cat]['top1'] += 1
                if topk_hit:
                    results['per_category'][cat]['topk'] += 1

        # Detailed per-word info
        top3_str = top_k_primitives(proj_01, k=3)
        results['words'].append({
            'word': word,
            'expected': expected_prims,
            'top1_hit': top1_hit,
            'topk_hit': topk_hit,
            'found_in_topk': found_in_topk,
            'top3': top3_str,
        })

    return results


# ============================================================
# Unseen Words Probe
# ============================================================

# Common TinyStories words that are NOT in the seed lexicon
UNSEEN_PROBE_WORDS = [
    # Animals
    "dog", "cat", "bird", "fish", "bunny", "bear", "horse", "frog",
    # Objects / Places
    "house", "car", "ball", "toy", "garden", "park", "bed", "table",
    # Actions
    "run", "eat", "sleep", "walk", "jump", "cry", "laugh", "talk",
    # Emotions / States
    "happy", "sad", "angry", "scared", "tired", "hungry", "little", "big",
    # Other
    "friend", "mommy", "daddy", "girl", "boy", "baby", "name", "day",
]


def probe_unseen(model, tokenizer, device):
    """
    Probe common TinyStories words not in the seed lexicon.

    Returns list of (word, top3_primitives, in_lexicon_flag).
    """
    all_lexicon = set(TIER_1.keys()) | set(TIER_2.keys())

    probes = []
    for word in UNSEEN_PROBE_WORDS:
        in_lexicon = word in all_lexicon
        proj_01, valid = get_projection(model, tokenizer, word, device)
        if not valid:
            probes.append((word, None, in_lexicon))
            continue
        top3 = top_k_primitives(proj_01, k=3)
        probes.append((word, top3, in_lexicon))

    return probes


# ============================================================
# Tier 1 Sanity Check (brief)
# ============================================================

def evaluate_tier1_brief(model, tokenizer, device):
    """Quick T1 accuracy check (top-1 and top-3) as a sanity baseline."""
    top1_ok = 0
    top3_ok = 0
    total = 0

    for word, mapping in TIER_1.items():
        prim_name = list(mapping.keys())[0]
        state = list(mapping.values())[0][0]
        if state != '+':
            continue

        proj_01, valid = get_projection(model, tokenizer, word, device)
        if not valid:
            continue

        expected_idx = PRIMITIVE_NAMES.index(prim_name)
        top1_idx = proj_01.argmax().item()
        top3_idx = set(proj_01.topk(3).indices.tolist())

        total += 1
        if top1_idx == expected_idx:
            top1_ok += 1
        if expected_idx in top3_idx:
            top3_ok += 1

    return top1_ok, top3_ok, total


# ============================================================
# Subcategory grouping for T2
# ============================================================

T2_GROUPS = {
    'Emotions':   ["love", "hate", "joy", "sadness", "anger", "fear", "hope",
                   "despair", "nostalgia", "grief", "pride", "shame",
                   "loneliness", "peace", "anxiety", "curiosity", "boredom",
                   "awe", "jealousy", "gratitude", "compassion", "excitement",
                   "calm", "rage", "melancholy"],
    'Nature':     ["sunrise", "sunset", "lightning", "thunder", "earthquake",
                   "rainbow", "volcano", "snow", "fog", "shadow", "night"],
    'People':     ["king", "queen", "child", "mother", "father", "warrior",
                   "teacher", "healer", "thief", "hero", "villain"],
    'Actions':    ["sing", "dance", "dream", "fight", "pray", "fly", "fall",
                   "sleep", "wake", "explore", "hide", "reveal", "transform",
                   "meditate", "write", "read", "teach", "learn"],
    'Abstract':   ["music", "beauty", "justice", "wisdom", "courage", "war",
                   "memory", "time", "space", "home", "journey", "treasure",
                   "gift", "sacrifice", "revenge", "forgiveness"],
    'Objects':    ["sword", "book", "mirror", "candle", "door", "bridge",
                   "crown", "tree", "star", "moon", "sun", "river", "forest",
                   "castle"],
}


def evaluate_tier2_by_group(t2_results):
    """Break down T2 results by semantic group."""
    word_lookup = {w['word']: w for w in t2_results['words']}
    group_stats = {}
    for group_name, words in T2_GROUPS.items():
        top1 = 0
        topk = 0
        total = 0
        for w in words:
            info = word_lookup.get(w)
            if info is None:
                continue
            total += 1
            if info['top1_hit']:
                top1 += 1
            if info['topk_hit']:
                topk += 1
        if total > 0:
            group_stats[group_name] = {'top1': top1, 'topk': topk, 'total': total}
    return group_stats


# ============================================================
# Pretty printing
# ============================================================

def print_header(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_t1_summary(top1, top3, total):
    print(f"\n  T1 Sanity Check (supervised training targets):")
    print(f"    Top-1 accuracy: {top1}/{total} ({top1/max(total,1):.1%})")
    print(f"    Top-3 accuracy: {top3}/{total} ({top3/max(total,1):.1%})")


def print_t2_summary(results, top_k):
    total = results['total']
    if total == 0:
        print("\n  No T2 words could be evaluated.")
        return

    top1_rate = results['top1_hits'] / total
    topk_rate = results['topk_hits'] / total
    any_hit_rate = (results['any_hit_num'] / results['any_hit_den']
                    if results['any_hit_den'] > 0 else 0.0)

    print(f"\n  Tier 2 Generalization ({total} compound words, top-K={top_k}):")
    print(f"  {'-'*60}")
    print(f"    Top-1 accuracy:    {results['top1_hits']:>3d}/{total}  ({top1_rate:.1%})")
    print(f"    Top-{top_k} accuracy:    {results['topk_hits']:>3d}/{total}  ({topk_rate:.1%})")
    print(f"    Any-hit rate (3+): {results['any_hit_num']:>3d}/{results['any_hit_den']}  "
          f"({any_hit_rate:.1%})  "
          f"[fraction of expected prims found in top-{top_k}]")


def print_t2_group_table(group_stats, top_k):
    print(f"\n  T2 by Semantic Group:")
    print(f"  {'Group':<12s}  {'Top-1':>10s}  {'Top-'+str(top_k):>10s}  {'Count':>6s}")
    print(f"  {'='*12}  {'='*10}  {'='*10}  {'='*6}")
    for group, s in group_stats.items():
        t = s['total']
        t1 = f"{s['top1']}/{t} ({s['top1']/t:.0%})"
        tk = f"{s['topk']}/{t} ({s['topk']/t:.0%})"
        print(f"  {group:<12s}  {t1:>10s}  {tk:>10s}  {t:>6d}")


def print_t2_details(results, top_k, max_show=20):
    """Show detailed per-word results for T2 (hits and misses)."""
    # Sort: misses first, then hits
    words = sorted(results['words'], key=lambda w: (w['topk_hit'], w['word']))

    print(f"\n  T2 Word Details (showing first {max_show}):")
    print(f"  {'Word':<14s}  {'Top-1':>5s}  {'Top-'+str(top_k):>5s}  "
          f"{'Model Top-3':<45s}  {'Expected'}")
    print(f"  {'='*14}  {'='*5}  {'='*5}  {'='*45}  {'='*30}")

    for i, w in enumerate(words):
        if i >= max_show:
            break
        t1 = "HIT" if w['top1_hit'] else " - "
        tk = "HIT" if w['topk_hit'] else " - "
        top3_str = ", ".join(f"{n}={v:.2f}" for n, v in w['top3'])
        exp_str = ", ".join(w['expected'][:4])
        if len(w['expected']) > 4:
            exp_str += "..."
        print(f"  {w['word']:<14s}  {t1:>5s}  {tk:>5s}  {top3_str:<45s}  {exp_str}")


def print_unseen_table(probes):
    truly_unseen = [(w, t, f) for w, t, f in probes if not f]
    lexicon_overlap = [(w, t, f) for w, t, f in probes if f]

    print(f"\n  Unseen Word Probes ({len(truly_unseen)} truly unseen, "
          f"{len(lexicon_overlap)} in lexicon):")

    if lexicon_overlap:
        print(f"\n  [Words that turned out to be in the lexicon — included for reference]")
        print(f"  {'Word':<12s}  {'Prim-1':<22s}  {'Prim-2':<22s}  {'Prim-3':<22s}")
        print(f"  {'='*12}  {'='*22}  {'='*22}  {'='*22}")
        for word, top3, _ in lexicon_overlap:
            if top3 is None:
                print(f"  {word:<12s}  (could not tokenize)")
                continue
            cols = [f"{n} ({v:.2f})" for n, v in top3]
            while len(cols) < 3:
                cols.append("")
            print(f"  {word:<12s}  {cols[0]:<22s}  {cols[1]:<22s}  {cols[2]:<22s}")

    print(f"\n  [Truly unseen words — qualitative probe]")
    print(f"  {'Word':<12s}  {'Prim-1':<22s}  {'Prim-2':<22s}  {'Prim-3':<22s}")
    print(f"  {'='*12}  {'='*22}  {'='*22}  {'='*22}")
    for word, top3, _ in truly_unseen:
        if top3 is None:
            print(f"  {word:<12s}  (could not tokenize)")
            continue
        cols = [f"{n} ({v:.2f})" for n, v in top3]
        while len(cols) < 3:
            cols.append("")
        print(f"  {word:<12s}  {cols[0]:<22s}  {cols[1]:<22s}  {cols[2]:<22s}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generalization evaluation for 49-Bit Concept GPT')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: XL step 50000)')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Path to tokenizer (default: same directory as checkpoint)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Top-K for hit evaluation (default: 3)')
    parser.add_argument('--show-details', type=int, default=30,
                        help='Number of T2 word details to show (default: 30)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve checkpoint path
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(
            PROJECT_ROOT, 'checkpoints', 'concept_gpt_49bit_xl',
            'model_L12_D512_B49_step50000.pt')

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

    # ── Load ──
    print_header("49-Bit Concept GPT — Generalization Evaluation")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model_and_tokenizer(ckpt_path, tok_path, device)

    # ── T1 Sanity Check ──
    print_header("PHASE 1: Tier 1 Sanity Check")
    t1_top1, t1_top3, t1_total = evaluate_tier1_brief(model, tokenizer, device)
    print_t1_summary(t1_top1, t1_top3, t1_total)

    # ── T2 Generalization ──
    print_header("PHASE 2: Tier 2 Generalization (Compound Words)")
    t2_results = evaluate_tier2(model, tokenizer, device, top_k=args.top_k)
    print_t2_summary(t2_results, args.top_k)

    group_stats = evaluate_tier2_by_group(t2_results)
    print_t2_group_table(group_stats, args.top_k)
    print_t2_details(t2_results, args.top_k, max_show=args.show_details)

    # ── Unseen Words ──
    print_header("PHASE 3: Unseen Word Probes")
    probes = probe_unseen(model, tokenizer, device)
    print_unseen_table(probes)

    # ── Summary ──
    print_header("SUMMARY")
    total_t2 = t2_results['total']
    any_den = t2_results['any_hit_den']
    print(f"  T1 (supervised):       top-1 {t1_top1}/{t1_total} "
          f"({t1_top1/max(t1_total,1):.1%}), "
          f"top-3 {t1_top3}/{t1_total} "
          f"({t1_top3/max(t1_total,1):.1%})")
    print(f"  T2 (generalization):   top-1 {t2_results['top1_hits']}/{total_t2} "
          f"({t2_results['top1_hits']/max(total_t2,1):.1%}), "
          f"top-{args.top_k} {t2_results['topk_hits']}/{total_t2} "
          f"({t2_results['topk_hits']/max(total_t2,1):.1%})")
    if any_den > 0:
        print(f"  T2 any-hit (3+ prims): {t2_results['any_hit_num']}/{any_den} "
              f"({t2_results['any_hit_num']/any_den:.1%})")

    # Generalization gap
    t1_rate = t1_top1 / max(t1_total, 1)
    t2_rate = t2_results['top1_hits'] / max(total_t2, 1)
    gap = t1_rate - t2_rate
    print(f"\n  Generalization gap (T1 top-1 - T2 top-1): {gap:+.1%}")
    if gap < 0.15:
        print(f"  --> Small gap: model generalizes well from T1 supervision to T2 compounds")
    elif gap < 0.35:
        print(f"  --> Moderate gap: partial generalization")
    else:
        print(f"  --> Large gap: limited generalization from T1 to T2")

    unseen_count = sum(1 for _, t, f in probes if t is not None and not f)
    print(f"\n  Unseen probes: {unseen_count} words evaluated (qualitative only)")
    print("=" * 72)


if __name__ == '__main__':
    main()

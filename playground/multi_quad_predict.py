"""
D-A12: Multi-Quad Algebraic Prediction.

Tests whether ENSEMBLING multiple analogy quads per holdout concept improves
algebraic prediction accuracy over the single-quad baselines from D-A5.

Uses the EXISTING D-A5 XL checkpoint — no retraining needed.

For each R3-reachable holdout concept, we define 3-5 quads using different
triadic axes (gender, temperature, valence, intensity, etc.), compute
D_predicted = C_proj + (B_proj - A_proj) in continuous tanh space, average
all predictions, then binarize.

Usage:
  python playground/multi_quad_predict.py
  python playground/multi_quad_predict.py --checkpoint checkpoints/danza_bootstrap_xl/
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
from collections import defaultdict

_PLAYGROUND = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

from danza_63bit import (
    load_primitives, load_anchors,
    DanzaTriadicGPT, ANCHOR_TRANSLATIONS,
    N_BITS,
)
from src.torch_transformer import TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer


# ============================================================
# Train / holdout split (must match D-A5 exactly)
# ============================================================

TRAIN_CONCEPTS = {
    'hombre', 'mujer', 'rey',
    'caliente', 'frío',
    'feliz', 'triste', 'amor',
    'abrir', 'cerrar', 'libre',
    'brillante', 'ruidoso', 'rápido', 'dulce', 'rico', 'orgulloso',
    'enseñar', 'sabio', 'creativo',
    'bueno', 'vivo',
    'sólido', 'oscuro',
}

HOLDOUT_INFO = {
    'reina':          ('R3',   'man:woman=king:queen'),
    'odio':           ('R3',   'happy:sad=love:hate'),
    'preso':          ('R3',   'open:close=free:prisoner'),
    'silencioso':     ('R3',   'hot:cold=loud:quiet'),
    'líquido':        ('R3',   'man:woman=solid:liquid'),
    'lógico':         ('R3',   'hot:cold=creative:logical'),
    'lento':          ('R3',   'bright:dark=fast:slow'),
    'pobre':          ('R3',   'bright:dark=rich:poor'),
    'amargo':         ('R3',   'bright:dark=sweet:bitter'),
    'humilde':        ('R3',   'bright:dark=proud:humble'),
    'malo':           ('R3',   'happy:sad=good:bad'),
    'muerto':         ('R3',   'happy:sad=alive:dead'),
    'aprender':       ('R3',   'open:close=teach:learn'),
    'ignorante':      ('R3',   'hot:cold=wise:ignorant'),
    'luna':           ('CTRL', 'no algebraic path'),
    'sol':            ('CTRL', 'no algebraic path'),
    'indiferencia':   ('CTRL', 'no algebraic path'),
    'gaseoso':        ('CTRL', 'no algebraic path'),
    'inmóvil':        ('CTRL', 'no algebraic path'),
    'oscuridad':      ('CTRL', 'no algebraic path'),
    'orden_concepto': ('CTRL', 'no algebraic path'),
    'caos_concepto':  ('CTRL', 'no algebraic path'),
    'apatía':         ('CTRL', 'no algebraic path'),
}


# ============================================================
# EXPANDED quads: 3-5 per R3 holdout concept
# ============================================================
# Format: (A, B, C, D_holdout) where A,B,C are TRAIN English words,
# D is a HOLDOUT English word.  D = C + (B - A).

EXPANDED_QUADS = [
    # --- reina (queen): gender axis and beyond ---
    # A,B,C must ALL be TRAIN; D is HOLDOUT.
    # TRAIN English: man, woman, king, happy, sad, love, good, hot, cold,
    #   bright, dark, fast, loud, sweet, rich, proud, open, close, free,
    #   teach, wise, creative, alive, solid
    ('man',    'woman',  'king',    'queen'),    # canonical gender
    ('hot',    'cold',   'king',    'queen'),    # temperature axis -> royalty
    ('bright', 'dark',   'king',    'queen'),    # mas->menos on royalty
    ('happy',  'sad',    'king',    'queen'),    # valence on royalty
    ('open',   'close',  'king',    'queen'),    # freedom on royalty

    # --- odio (hate): valence + separation axes ---
    ('happy',  'sad',    'love',    'hate'),     # canonical valence
    ('bright', 'dark',   'love',    'hate'),     # mas->menos on love
    ('open',   'close',  'love',    'hate'),     # freedom->control on union
    ('hot',    'cold',   'love',    'hate'),     # temperature on love
    ('man',    'woman',  'love',    'hate'),     # gender axis on love

    # --- preso (prisoner): freedom axis ---
    ('open',   'close',  'free',    'prisoner'), # canonical freedom
    ('happy',  'sad',    'free',    'prisoner'), # valence on freedom
    ('bright', 'dark',   'free',    'prisoner'), # mas->menos on freedom
    ('hot',    'cold',   'free',    'prisoner'), # temperature on freedom

    # --- silencioso (quiet): intensity + temperature ---
    ('hot',    'cold',   'loud',    'quiet'),    # canonical temperature
    ('bright', 'dark',   'loud',    'quiet'),    # canonical mas->menos
    ('happy',  'sad',    'loud',    'quiet'),    # valence on loudness
    ('open',   'close',  'loud',    'quiet'),    # freedom on loudness
    ('man',    'woman',  'loud',    'quiet'),    # gender axis on loudness

    # --- líquido (liquid): element transformation ---
    ('man',    'woman',  'solid',   'liquid'),   # canonical tierra->agua
    ('hot',    'cold',   'solid',   'liquid'),   # temperature on matter
    ('bright', 'dark',   'solid',   'liquid'),   # mas->menos on matter
    ('open',   'close',  'solid',   'liquid'),   # freedom axis on matter

    # --- lógico (logical): thought mode axis ---
    ('hot',    'cold',   'creative','logical'),  # canonical fuego->tierra
    ('bright', 'dark',   'creative','logical'),  # mas->menos on thought
    ('happy',  'sad',    'creative','logical'),  # valence on thought
    ('open',   'close',  'creative','logical'),  # freedom on thought

    # --- lento (slow): speed/intensity axes ---
    ('bright', 'dark',   'fast',    'slow'),     # canonical mas->menos
    ('hot',    'cold',   'fast',    'slow'),     # temperature on speed
    ('happy',  'sad',    'fast',    'slow'),     # valence on speed
    ('open',   'close',  'fast',    'slow'),     # freedom on speed
    ('man',    'woman',  'fast',    'slow'),     # gender axis on speed

    # --- pobre (poor): wealth/intensity axes ---
    ('bright', 'dark',   'rich',    'poor'),     # canonical mas->menos
    ('hot',    'cold',   'rich',    'poor'),     # temperature on wealth
    ('happy',  'sad',    'rich',    'poor'),     # valence on wealth
    ('open',   'close',  'rich',    'poor'),     # freedom on wealth
    ('man',    'woman',  'rich',    'poor'),     # gender axis on wealth

    # --- amargo (bitter): taste/intensity axes ---
    ('bright', 'dark',   'sweet',   'bitter'),   # canonical mas->menos
    ('hot',    'cold',   'sweet',   'bitter'),   # temperature on taste
    ('happy',  'sad',    'sweet',   'bitter'),   # valence on taste
    ('open',   'close',  'sweet',   'bitter'),   # freedom on taste

    # --- humilde (humble): pride/intensity axes ---
    ('bright', 'dark',   'proud',   'humble'),   # canonical mas->menos
    ('hot',    'cold',   'proud',   'humble'),   # temperature on pride
    ('happy',  'sad',    'proud',   'humble'),   # valence on pride
    ('man',    'woman',  'proud',   'humble'),   # gender axis on pride
    ('open',   'close',  'proud',   'humble'),   # freedom on pride

    # --- malo (bad/evil): moral/valence axes ---
    ('happy',  'sad',    'good',    'bad'),      # canonical valence
    ('bright', 'dark',   'good',    'bad'),      # mas->menos on morality
    ('hot',    'cold',   'good',    'bad'),      # temperature on morality
    ('open',   'close',  'good',    'bad'),      # freedom on morality
    ('man',    'woman',  'good',    'bad'),      # gender axis on morality

    # --- muerto (dead): vitality/valence axes ---
    ('happy',  'sad',    'alive',   'dead'),     # canonical valence
    ('bright', 'dark',   'alive',   'dead'),     # mas->menos on vitality
    ('hot',    'cold',   'alive',   'dead'),     # temperature on vitality
    ('open',   'close',  'alive',   'dead'),     # freedom on vitality

    # --- aprender (learn): action direction axes ---
    ('open',   'close',  'teach',   'learn'),    # canonical transmit->receive
    ('bright', 'dark',   'teach',   'learn'),    # mas->menos on knowledge
    ('hot',    'cold',   'teach',   'learn'),    # temperature on knowledge
    ('man',    'woman',  'teach',   'learn'),    # gender axis on knowledge

    # --- ignorante (ignorant): knowledge axes ---
    ('hot',    'cold',   'wise',    'ignorant'), # canonical structured->empty
    ('bright', 'dark',   'wise',    'ignorant'), # mas->menos on wisdom
    ('happy',  'sad',    'wise',    'ignorant'), # valence on wisdom
    ('open',   'close',  'wise',    'ignorant'), # freedom on wisdom
    ('man',    'woman',  'wise',    'ignorant'), # gender axis on wisdom
]


# ============================================================
# Helpers
# ============================================================

def get_split(all_anchors):
    """Split anchors into train/holdout dicts by Spanish concept."""
    train_anchors, holdout_anchors = {}, {}
    for eng_word, data in all_anchors.items():
        spanish = data['spanish']
        if spanish in TRAIN_CONCEPTS:
            train_anchors[eng_word] = data
        elif spanish in HOLDOUT_INFO:
            holdout_anchors[eng_word] = data
    return train_anchors, holdout_anchors


def load_checkpoint(ckpt_dir, device):
    """Load model + tokenizer from a D-A5 checkpoint directory."""
    import glob as glob_mod

    # Prefer latest step checkpoint
    step_ckpts = sorted(glob_mod.glob(os.path.join(ckpt_dir, 'model_step*.pt')))
    if step_ckpts:
        ckpt_path = step_ckpts[-1]
    else:
        ckpt_path = os.path.join(ckpt_dir, 'model_best.pt')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    cfg = ckpt['config']
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'], block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'],
        n_head=cfg['n_head'], n_triadic_bits=cfg['n_triadic_bits'],
    )
    model = DanzaTriadicGPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tok_path = os.path.join(ckpt_dir, 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)

    return model, tokenizer


# ============================================================
# Multi-Quad Prediction
# ============================================================

@torch.no_grad()
def run_multi_quad_predict(model, tokenizer, holdout_anchors, device):
    """Run expanded-quad algebraic prediction and return structured results."""

    def get_proj(word):
        """Get continuous tanh projection for a word."""
        ids = tokenizer.encode(word, add_special=False)[:4]
        if not ids:
            return None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, proj, _ = model(x)
        return proj[0].mean(dim=0)  # (N_BITS,)

    # Cache projections (many words reused across quads)
    proj_cache = {}
    def cached_proj(word):
        if word not in proj_cache:
            proj_cache[word] = get_proj(word)
        return proj_cache[word]

    # Group quads by holdout English word
    quads_by_holdout = defaultdict(list)
    for a, b, c, d in EXPANDED_QUADS:
        quads_by_holdout[d].append((a, b, c, d))

    # Map holdout English words to their Spanish concept
    eng_to_spanish = {}
    for eng, data in holdout_anchors.items():
        eng_to_spanish[eng] = data['spanish']

    results = {}  # spanish_concept -> result dict

    for spanish in sorted(HOLDOUT_INFO.keys()):
        rtype, desc = HOLDOUT_INFO[spanish]
        if rtype != 'R3':
            continue

        # Find primary English word for this concept
        eng_words = ANCHOR_TRANSLATIONS.get(spanish, [])
        if not eng_words:
            continue
        primary_eng = eng_words[0]
        if primary_eng not in holdout_anchors:
            continue

        gold_target = holdout_anchors[primary_eng]['target']
        gold_bits = (gold_target > 0).float().to(device)

        # Collect quads for ALL English translations of this concept
        all_quads = []
        for ew in eng_words:
            all_quads.extend(quads_by_holdout.get(ew, []))

        if not all_quads:
            continue

        # Compute per-quad predictions
        per_quad = []
        valid_preds = []

        for a, b, c, d in all_quads:
            pa, pb, pc = cached_proj(a), cached_proj(b), cached_proj(c)
            if any(p is None for p in [pa, pb, pc]):
                continue

            # D_predicted = C + (B - A)  in continuous tanh space
            predicted = pc + (pb - pa)
            pred_bits = (predicted > 0).float()
            acc = (pred_bits == gold_bits).float().mean().item()

            per_quad.append({
                'quad': f"{a}:{b}={c}:{d}",
                'accuracy': acc,
            })
            valid_preds.append(predicted)

        if not valid_preds:
            continue

        # Ensemble: average continuous predictions, then binarize
        ensemble_proj = torch.stack(valid_preds).mean(dim=0)
        ensemble_bits = (ensemble_proj > 0).float()
        ensemble_acc = (ensemble_bits == gold_bits).float().mean().item()

        # Per-quad stats
        quad_accs = [q['accuracy'] for q in per_quad]
        single_best = max(quad_accs)
        single_worst = min(quad_accs)
        single_mean = np.mean(quad_accs)

        # Bit-level detail: which bits does ensemble get right vs gold?
        match_mask = (ensemble_bits == gold_bits)
        n_correct = match_mask.sum().item()
        n_wrong = N_BITS - n_correct

        results[spanish] = {
            'english': primary_eng,
            'n_quads': len(valid_preds),
            'single_best': single_best,
            'single_worst': single_worst,
            'single_mean': single_mean,
            'ensemble_acc': ensemble_acc,
            'delta': ensemble_acc - single_best,
            'bits_correct': int(n_correct),
            'bits_wrong': int(n_wrong),
            'per_quad': per_quad,
        }

    return results


# ============================================================
# Display + CSV output
# ============================================================

def print_results(results):
    """Print formatted table and summary statistics."""
    trivial_baseline = 0.902  # from D-A5 analysis

    print(f"\n{'=' * 85}")
    print(f"  D-A12: MULTI-QUAD ALGEBRAIC PREDICTION")
    print(f"{'=' * 85}")

    # Per-concept table
    header = (f"  {'Concept':15s} {'Eng':10s} {'#Q':>3s} "
              f"{'Best1':>7s} {'Mean1':>7s} {'Ensem':>7s} "
              f"{'Delta':>7s} {'Bits':>7s}")
    print(f"\n{header}")
    print(f"  {'-'*15} {'-'*10} {'-'*3} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    single_bests, single_means, ensemble_accs = [], [], []

    for spanish in sorted(results.keys()):
        r = results[spanish]
        delta_str = f"{r['delta']:+.1%}"
        bits_str = f"{r['bits_correct']}/{N_BITS}"
        print(f"  {spanish:15s} {r['english']:10s} {r['n_quads']:3d} "
              f"{r['single_best']:7.1%} {r['single_mean']:7.1%} {r['ensemble_acc']:7.1%} "
              f"{delta_str:>7s} {bits_str:>7s}")

        single_bests.append(r['single_best'])
        single_means.append(r['single_mean'])
        ensemble_accs.append(r['ensemble_acc'])

        # Show per-quad breakdown
        for q in r['per_quad']:
            marker = '*' if q['accuracy'] == r['single_best'] else ' '
            print(f"    {marker} {q['quad']:40s} {q['accuracy']:.1%}")

    # Summary
    mean_best = np.mean(single_bests)
    mean_single = np.mean(single_means)
    mean_ensemble = np.mean(ensemble_accs)

    n_improved = sum(1 for r in results.values() if r['delta'] > 0)
    n_same = sum(1 for r in results.values() if r['delta'] == 0)
    n_worse = sum(1 for r in results.values() if r['delta'] < 0)

    print(f"\n{'=' * 85}")
    print(f"  SUMMARY ({len(results)} R3-reachable concepts)")
    print(f"{'=' * 85}")
    print(f"  Trivial baseline (prior):     {trivial_baseline:.1%}")
    print(f"  Mean single-quad (mean):      {mean_single:.1%}")
    print(f"  Mean single-quad (best):      {mean_best:.1%}")
    print(f"  Mean ENSEMBLE accuracy:       {mean_ensemble:.1%}")
    print(f"  Ensemble vs best-single:      {mean_ensemble - mean_best:+.1%}")
    print(f"  Ensemble vs trivial:          {mean_ensemble - trivial_baseline:+.1%}")
    print(f"  Concepts improved/same/worse: {n_improved}/{n_same}/{n_worse}")

    return {
        'trivial_baseline': trivial_baseline,
        'mean_single_mean': mean_single,
        'mean_single_best': mean_best,
        'mean_ensemble': mean_ensemble,
        'n_concepts': len(results),
        'n_improved': n_improved,
        'n_same': n_same,
        'n_worse': n_worse,
    }


def save_csv(results, summary, csv_path):
    """Save per-concept results + summary to CSV."""
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)

        # Per-concept rows
        w.writerow(['concept', 'english', 'n_quads', 'single_best',
                     'single_mean', 'ensemble_acc', 'delta',
                     'bits_correct', 'bits_wrong'])
        for spanish in sorted(results.keys()):
            r = results[spanish]
            w.writerow([spanish, r['english'], r['n_quads'],
                        f"{r['single_best']:.6f}",
                        f"{r['single_mean']:.6f}",
                        f"{r['ensemble_acc']:.6f}",
                        f"{r['delta']:.6f}",
                        r['bits_correct'], r['bits_wrong']])

        # Blank separator
        w.writerow([])

        # Summary
        w.writerow(['metric', 'value'])
        for k, v in summary.items():
            w.writerow([k, f"{v:.6f}" if isinstance(v, float) else v])

    print(f"\n  CSV saved: {csv_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='D-A12: Multi-Quad Algebraic Prediction')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(_PROJECT, 'checkpoints',
                                             'danza_bootstrap_xl'),
                        help='Checkpoint directory (default: danza_bootstrap_xl)')
    args = parser.parse_args()

    # --- GPU optimizations ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\n{'=' * 85}")
    print(f"  D-A12: Multi-Quad Algebraic Prediction")
    print(f"{'=' * 85}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Dtype: bfloat16 (inference)")
    print(f"  Expanded quads: {len(EXPANDED_QUADS)}")

    # --- Load data ---
    prim_data = load_primitives()
    all_anchors, skipped = load_anchors(prim_data)
    _, holdout_anchors = get_split(all_anchors)

    print(f"  Holdout anchors: {len(holdout_anchors)} English words")

    # Count quads per concept
    quads_per_concept = defaultdict(int)
    for _, _, _, d in EXPANDED_QUADS:
        for sp, eng_list in ANCHOR_TRANSLATIONS.items():
            if d in eng_list and sp in HOLDOUT_INFO:
                quads_per_concept[sp] += 1
                break
    for sp, nq in sorted(quads_per_concept.items()):
        print(f"    {sp:15s}: {nq} quads")

    # --- Load checkpoint ---
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    model, tokenizer = load_checkpoint(args.checkpoint, device)

    # Use bfloat16 for inference
    if device.type == 'cuda':
        model = model.to(dtype=torch.bfloat16)

    # --- Run prediction ---
    print(f"\n  Running multi-quad algebraic prediction...")
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
        results = run_multi_quad_predict(model, tokenizer, holdout_anchors, device)

    # --- Display + save ---
    summary = print_results(results)

    csv_path = os.path.join(args.checkpoint, 'multi_quad_results.csv')
    save_csv(results, summary, csv_path)

    # Also save JSON for programmatic use
    json_path = os.path.join(args.checkpoint, 'multi_quad_results.json')
    serializable = {}
    for sp, r in results.items():
        serializable[sp] = {k: v for k, v in r.items()}
    serializable['_summary'] = summary
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  JSON saved: {json_path}")


if __name__ == '__main__':
    main()

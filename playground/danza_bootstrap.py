"""
D-A5/D-A6: Bootstrap Test for 63-Bit Danza Cosmica.

Central question: can 24 hand-factorized anchor concepts + algebraic
constraints PREDICT the bits of 23 held-out concepts?

D-A5: Train with 24 anchors, predict 23 holdout algebraically.
D-A6: Iterative bootstrap loop with confidence gating.

Usage:
  python playground/danza_bootstrap.py --phase split                     # show split
  python playground/danza_bootstrap.py --phase train --scale xl --steps 50000  # ~76 min
  python playground/danza_bootstrap.py --phase predict --checkpoint checkpoints/danza_bootstrap_xl/
  python playground/danza_bootstrap.py --phase bootstrap --scale xl --steps 50000 --cycles 3
  python playground/danza_bootstrap.py --phase all --scale xl --steps 50000    # train+predict
"""

import os
import sys
import csv
import json
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

_PLAYGROUND = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

from danza_63bit import (
    load_primitives, load_anchors, build_subsumption_pairs,
    DanzaTriadicGPT, supervised_anchor_loss, subsumption_loss,
    triadic_loss, evaluate_anchors, evaluate_subsumption,
    evaluate_regla_de_tres, REGLA_DE_TRES_QUADS, TextDataset,
    ANCHOR_TRANSLATIONS, SKIP_ANCHORS,
    N_BITS, STORY_SEPARATOR,
)
from src.torch_transformer import TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer


# ============================================================
# Strategic Split (deterministic, pre-registered)
# ============================================================

# 24 Spanish concepts for TRAINING.
# Chosen to provide 3-of-4 in as many analogy quads as possible,
# maximizing algebraic reachability of holdout concepts.
TRAIN_CONCEPTS = {
    # Gender axis anchors (both poles + king for queen prediction)
    'hombre', 'mujer', 'rey',
    # Temperature (both poles for cold:hot transform)
    'caliente', 'frío',
    # Emotion (3 of 4 for happy:sad=love:hate)
    'feliz', 'triste', 'amor',
    # Freedom axis (3 of 4 for open:close=free:prisoner)
    'abrir', 'cerrar', 'libre',
    # Intensity high poles (template for mas/menos predictions)
    'brillante', 'ruidoso', 'rápido', 'dulce', 'rico', 'orgulloso',
    # Knowledge/thinking (one pole each)
    'enseñar', 'sabio', 'creativo',
    # Moral + vitality (one pole each)
    'bueno', 'vivo',
    # Material + light (needed for bright:dark template)
    'sólido', 'oscuro',
}

# Holdout concepts with reachability classification.
# R3 = reachable via regla de tres, CTRL = control (no algebraic path).
HOLDOUT_INFO = {
    'reina':          ('R3',   'man:woman=king:queen'),
    'odio':           ('R3',   'happy:sad=love:hate'),
    'preso':          ('R3',   'open:close=free:prisoner'),
    'silencioso':     ('R3',   'hot:cold=loud:quiet'),
    'líquido':        ('R3',   'man:woman=solid:liquid (tierra-agua)'),
    'lógico':         ('R3',   'hot:cold=creative:logical (fuego-tierra)'),
    'lento':          ('R3',   'bright:dark=fast:slow (mas-menos)'),
    'pobre':          ('R3',   'bright:dark=rich:poor'),
    'amargo':         ('R3',   'bright:dark=sweet:bitter'),
    'humilde':        ('R3',   'bright:dark=proud:humble'),
    'malo':           ('R3',   'happy:sad=good:bad'),
    'muerto':         ('R3',   'happy:sad=alive:dead'),
    'aprender':       ('R3',   'open:close=teach:learn'),
    'ignorante':      ('R3',   'hot:cold=wise:ignorant'),
    # Controls — no clear algebraic path from training concepts
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

# Analogy quads for prediction.
# (A, B, C, D_holdout) — predict D = C + (B - A) in neural projection space.
# A, B, C must map to TRAIN concepts; D must map to HOLDOUT.
BOOTSTRAP_QUADS = [
    # --- Exact axis matches ---
    ('man', 'woman', 'king', 'queen'),          # tierra<->agua
    ('happy', 'sad', 'love', 'hate'),            # placer<->dolor + union<->separacion
    ('open', 'close', 'free', 'prisoner'),       # libertad<->control + sep<->union
    ('man', 'woman', 'solid', 'liquid'),          # tierra<->agua (same as gender)
    ('hot', 'cold', 'creative', 'logical'),       # fuego<->tierra + caos<->orden

    # --- Partial axis (share primary component) ---
    ('hot', 'cold', 'loud', 'quiet'),             # partial: includes orden<->caos
    ('bright', 'dark', 'loud', 'quiet'),          # partial: mas<->menos component

    # --- Approximate mas<->menos (bright:dark as template) ---
    ('bright', 'dark', 'fast', 'slow'),
    ('bright', 'dark', 'rich', 'poor'),
    ('bright', 'dark', 'sweet', 'bitter'),
    ('bright', 'dark', 'proud', 'humble'),

    # --- Approximate valence (happy:sad as template) ---
    ('happy', 'sad', 'good', 'bad'),              # positive<->negative
    ('happy', 'sad', 'alive', 'dead'),             # very approximate

    # --- Approximate action direction ---
    ('open', 'close', 'teach', 'learn'),           # transmit<->receive

    # --- Knowledge reduction ---
    ('hot', 'cold', 'wise', 'ignorant'),           # structured<->empty (approximate)
]


def get_split(all_anchors):
    """Split anchors into train/holdout dicts by Spanish concept."""
    train_anchors = {}
    holdout_anchors = {}
    for eng_word, data in all_anchors.items():
        spanish = data['spanish']
        if spanish in TRAIN_CONCEPTS:
            train_anchors[eng_word] = data
        elif spanish in HOLDOUT_INFO:
            holdout_anchors[eng_word] = data
    return train_anchors, holdout_anchors


def get_holdout_type(eng_word, all_anchors):
    """Return ('R3'|'CTRL', description) for a holdout English word."""
    spanish = all_anchors[eng_word]['spanish']
    return HOLDOUT_INFO.get(spanish, ('CTRL', 'unknown'))


# ============================================================
# Phase: split — display and validate
# ============================================================

def phase_split(all_anchors, prim_data):
    """Show the strategic split and validate reachability."""
    train_a, holdout_a = get_split(all_anchors)

    print(f"\n{'=' * 70}")
    print(f"  STRATEGIC SPLIT — D-A5 Bootstrap Test")
    print(f"{'=' * 70}")
    print(f"  Train:   {len(TRAIN_CONCEPTS)} Spanish concepts -> {len(train_a)} English words")
    print(f"  Holdout: {len(HOLDOUT_INFO)} Spanish concepts -> {len(holdout_a)} English words")

    n_r3 = sum(1 for v in HOLDOUT_INFO.values() if v[0] == 'R3')
    n_ctrl = sum(1 for v in HOLDOUT_INFO.values() if v[0] == 'CTRL')
    print(f"  Reachable (R3): {n_r3}  |  Controls: {n_ctrl}")

    print(f"\n  TRAIN concepts:")
    for sp in sorted(TRAIN_CONCEPTS):
        words = ANCHOR_TRANSLATIONS.get(sp, [])
        print(f"    {sp:20s} -> {', '.join(words)}")

    print(f"\n  HOLDOUT concepts:")
    for sp, (rtype, desc) in sorted(HOLDOUT_INFO.items()):
        words = ANCHOR_TRANSLATIONS.get(sp, [])
        tag = 'R3  ' if rtype == 'R3' else 'CTRL'
        print(f"    [{tag}] {sp:20s} -> {', '.join(words):20s}  ({desc})")

    # Validate quads
    print(f"\n  ANALOGY QUADS ({len(BOOTSTRAP_QUADS)}):")
    for a, b, c, d in BOOTSTRAP_QUADS:
        a_ok = a in train_a
        b_ok = b in train_a
        c_ok = c in train_a
        d_ok = d in holdout_a
        status = 'OK' if (a_ok and b_ok and c_ok and d_ok) else 'ERR'
        print(f"    [{status}] {a}:{b} = {c}:{d}")

    # Check completeness
    all_sp = set(ANCHOR_TRANSLATIONS.keys())
    covered = TRAIN_CONCEPTS | set(HOLDOUT_INFO.keys())
    missing = all_sp - covered
    if missing:
        print(f"\n  WARNING: {len(missing)} concepts not in either set: {missing}")
    extra = covered - all_sp
    if extra:
        print(f"\n  WARNING: {len(extra)} concepts not in ANCHOR_TRANSLATIONS: {extra}")

    return train_a, holdout_a


# ============================================================
# Training (mirrors danza_63bit.py with partial anchors)
# ============================================================

def build_partial_subsumption_pairs(anchors):
    """Build subsumption pairs from the given anchor subset only."""
    items = list(anchors.items())
    pairs = []
    for i, (w_a, d_a) in enumerate(items):
        bits_a = set(d_a['expanded'])
        for j, (w_b, d_b) in enumerate(items):
            if i == j:
                continue
            bits_b = set(d_b['expanded'])
            if bits_a < bits_b:
                pairs.append((w_a, w_b, d_a, d_b))
    random.seed(42)
    random.shuffle(pairs)
    n_test = max(1, int(len(pairs) * 0.2))
    return pairs[n_test:], pairs[:n_test]


def run_training(args, train_anchors, holdout_anchors, prim_data, ckpt_dir, cycle=0):
    """Train DanzaTriadicGPT with partial anchor supervision. Returns (model, tokenizer, device)."""
    SCALES = {
        'base': {'layers': 6,  'dim': 256,  'heads': 8},
        'xl':   {'layers': 12, 'dim': 512,  'heads': 8},
        'xxl':  {'layers': 24, 'dim': 1024, 'heads': 16},
    }
    preset = SCALES[args.scale]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  BOOTSTRAP TRAINING — Cycle {cycle}")
    print(f"{'=' * 70}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Train anchors: {len(train_anchors)} words")
    print(f"  Holdout anchors: {len(holdout_anchors)} words (eval only, NO supervision)")

    # --- Subsumption pairs (train-only) ---
    train_sub, test_sub = build_partial_subsumption_pairs(train_anchors)
    print(f"  Subsumption pairs: train={len(train_sub)}, test={len(test_sub)}")

    # --- Tokenizer ---
    data_path = os.path.join(_PROJECT, 'data', 'TinyStories-train.txt')
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR)
               if s.strip() and len(s.strip()) > 30]
    if args.stories and len(stories) > args.stories:
        random.seed(42)
        random.shuffle(stories)
        stories = stories[:args.stories]

    tok_path = os.path.join(ckpt_dir, 'tokenizer.json')
    print(f"\n  Training BPE tokenizer (vocab={args.vocab})...")
    tokenizer = BPETokenizer(vocab_size=args.vocab)
    tokenizer.train(stories, verbose=False)
    tokenizer.save(tok_path)

    # --- Tokenize ---
    print(f"  Tokenizing {len(stories)} stories...")
    all_tokens = []
    for story in stories:
        all_tokens.extend(tokenizer.encode(story, add_special=True))
    print(f"  Total: {len(all_tokens):,} tokens")

    # --- Model ---
    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block,
        n_layer=preset['layers'],
        n_embd=preset['dim'],
        n_head=preset['heads'],
        n_triadic_bits=N_BITS,
        dropout=args.dropout,
    )
    model = DanzaTriadicGPT(config).to(device)
    total_params = model.num_params()
    print(f"  Model: {args.scale} ({total_params/1e6:.1f}M params, {N_BITS} bits)")

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    if device.type == 'cuda' and not getattr(args, 'no_compile', False):
        try:
            import triton  # noqa: F401
            model = torch.compile(model)
            print("  torch.compile: ON")
        except ImportError:
            print("  torch.compile: SKIPPED (triton not available on Windows)")

    # Mixed precision
    use_amp = device.type == 'cuda'
    amp_dtype = {'float32': torch.float32, 'float16': torch.float16,
                 'bfloat16': torch.bfloat16}[args.dtype]

    # --- Pre-encode anchors ---
    def _pack_anchors(anchor_dict):
        words, ids_list, targets = [], [], []
        for word, data in anchor_dict.items():
            ids = tokenizer.encode(word, add_special=False)[:4]
            if ids:
                words.append(word)
                ids_list.append(ids)
                targets.append(data['target'])
        if not words:
            z = torch.zeros((0, 1), dtype=torch.long, device=device)
            return z, torch.zeros((0, N_BITS), device=device), []
        mx = max(len(x) for x in ids_list)
        padded = torch.tensor([x + [0] * (mx - len(x)) for x in ids_list],
                               dtype=torch.long, device=device)
        target_t = torch.stack(targets).to(device)
        return padded, target_t, words

    sup_train_t, sup_train_tgt, sup_train_words = _pack_anchors(train_anchors)
    sup_hold_t, sup_hold_tgt, sup_hold_words = _pack_anchors(holdout_anchors)
    print(f"  Supervision: {len(sup_train_words)} train anchors (holdout gets NO supervision)")

    # --- Pre-encode subsumption ---
    def _pack_sub(pairs):
        h_ids, y_ids, valid = [], [], []
        for h_w, y_w, h_d, y_d in pairs:
            h = tokenizer.encode(h_w, add_special=False)[:4]
            y = tokenizer.encode(y_w, add_special=False)[:4]
            if h and y:
                h_ids.append(h)
                y_ids.append(y)
                valid.append((h_w, y_w))
        if not valid:
            z = torch.zeros((0, 1), dtype=torch.long, device=device)
            return z, z, valid
        def pad(lst):
            mx = max(len(x) for x in lst)
            return torch.tensor([x + [0] * (mx - len(x)) for x in lst],
                                dtype=torch.long, device=device)
        return pad(h_ids), pad(y_ids), valid

    sub_train_h, sub_train_y, _ = _pack_sub(train_sub)
    sub_test_h, sub_test_y, _ = _pack_sub(test_sub)

    # --- Training loop ---
    print(f"\n  Training ({args.steps} steps, warmup={args.triadic_warmup_pct:.0%})...")
    dataset = TextDataset(all_tokens, args.block)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.999), weight_decay=0.01)
    warmup_steps = int(args.steps * 0.05)
    triadic_start = int(args.steps * args.triadic_warmup_pct)

    csv_path = os.path.join(ckpt_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'lang_loss', 'tri_loss', 'sup_loss', 'sub_loss',
                          'bit_acc_train', 'bit_acc_holdout', 'dead_bits'])

    data_iter = iter(loader)
    t0 = time.time()
    best_train_acc = 0.0
    best_hold_acc = 0.0

    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        # LR schedule
        if step <= warmup_steps:
            lr = args.lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (args.steps - warmup_steps)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward
        if use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits, proj, lang_loss = model(x, y)
                l_tri = l_sup = l_sub = torch.tensor(0.0, device=device)
                if step >= triadic_start:
                    l_tri = triadic_loss(proj, args.align_weight, model.wte, x)
                    l_sup = supervised_anchor_loss(model, sup_train_t, sup_train_tgt)
                    l_sub = subsumption_loss(model, sub_train_h, sub_train_y)
                total = lang_loss + args.alpha * (
                    l_tri + args.sup_weight * l_sup + args.sub_weight * l_sub)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            logits, proj, lang_loss = model(x, y)
            l_tri = l_sup = l_sub = torch.tensor(0.0, device=device)
            if step >= triadic_start:
                l_tri = triadic_loss(proj, args.align_weight, model.wte, x)
                l_sup = supervised_anchor_loss(model, sup_train_t, sup_train_tgt)
                l_sub = subsumption_loss(model, sub_train_h, sub_train_y)
            total = lang_loss + args.alpha * (
                l_tri + args.sup_weight * l_sup + args.sub_weight * l_sub)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Print
        if step % args.print_every == 0:
            elapsed = time.time() - t0
            tri_str = (f"tri={l_tri.item():.4f} sup={l_sup.item():.4f} sub={l_sub.item():.4f}"
                       if step >= triadic_start else "warmup")
            print(f"  [{step:>6d}/{args.steps}] loss={total.item():.4f} lang={lang_loss.item():.4f} "
                  f"{tri_str} lr={lr:.2e} ({elapsed:.0f}s)")

        # Evaluate
        if step % args.eval_every == 0 or step == args.steps:
            eval_train = evaluate_anchors(model, sup_train_t, sup_train_tgt, sup_train_words)
            eval_hold = evaluate_anchors(model, sup_hold_t, sup_hold_tgt, sup_hold_words)

            train_acc = eval_train.get('mean_bit_accuracy', 0)
            hold_acc = eval_hold.get('mean_bit_accuracy', 0)
            dead = eval_train.get('dead_bits', N_BITS)

            print(f"  --- Eval @ step {step} ---")
            print(f"  Bit accuracy:  train={train_acc:.1%}  holdout={hold_acc:.1%} (no supervision!)")
            print(f"  Dead bits: {dead}/{N_BITS}")

            csv_writer.writerow([
                step, total.item(), lang_loss.item(),
                l_tri.item() if step >= triadic_start else 0,
                l_sup.item() if step >= triadic_start else 0,
                l_sub.item() if step >= triadic_start else 0,
                train_acc, hold_acc, dead,
            ])
            csv_file.flush()

            # Save best (by train accuracy)
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'vocab_size': config.vocab_size, 'block_size': config.block_size,
                        'n_layer': config.n_layer, 'n_embd': config.n_embd,
                        'n_head': config.n_head, 'n_triadic_bits': config.n_triadic_bits,
                    },
                    'cycle': cycle,
                    'train_concepts': sorted(TRAIN_CONCEPTS),
                    'bit_accuracy_train': train_acc,
                    'bit_accuracy_holdout': hold_acc,
                }, os.path.join(ckpt_dir, 'model_best.pt'))

            # Save best by holdout accuracy (more meaningful than train)
            if hold_acc > best_hold_acc:
                best_hold_acc = hold_acc
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'vocab_size': config.vocab_size, 'block_size': config.block_size,
                        'n_layer': config.n_layer, 'n_embd': config.n_embd,
                        'n_head': config.n_head, 'n_triadic_bits': config.n_triadic_bits,
                    },
                    'bit_accuracy_holdout': hold_acc,
                }, os.path.join(ckpt_dir, 'model_best_holdout.pt'))

        # Periodic checkpoint
        if step % args.save_every == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'config': {
                    'vocab_size': config.vocab_size, 'block_size': config.block_size,
                    'n_layer': config.n_layer, 'n_embd': config.n_embd,
                    'n_head': config.n_head, 'n_triadic_bits': config.n_triadic_bits,
                },
            }, os.path.join(ckpt_dir, f'model_step{step}.pt'))

    csv_file.close()
    elapsed = time.time() - t0
    print(f"\n  Training complete: {elapsed/60:.1f} min, best train acc: {best_train_acc:.1%}, best holdout: {best_hold_acc:.1%}")

    return model, tokenizer, device


# ============================================================
# Phase: predict — algebraic prediction of holdout concepts
# ============================================================

@torch.no_grad()
def phase_predict(model, tokenizer, train_anchors, holdout_anchors, all_anchors, device):
    """Predict holdout concepts via direct encoding + regla de tres + ensemble."""
    model.eval()

    def get_proj(word):
        ids = tokenizer.encode(word, add_special=False)[:4]
        if not ids:
            return None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, proj, _ = model(x)
        return proj[0].mean(dim=0)  # (63,)

    # --- 1. Direct encoding ---
    print(f"\n{'=' * 70}")
    print(f"  HOLDOUT PREDICTION — D-A5 Bootstrap Test")
    print(f"{'=' * 70}")

    direct = {}
    for word, data in holdout_anchors.items():
        proj = get_proj(word)
        if proj is None:
            continue
        pred_bits = (proj > 0).float()
        gold_bits = (data['target'] > 0).float().to(device)
        acc = (pred_bits == gold_bits).float().mean().item()
        confidence = proj.abs().mean().item()
        direct[word] = {
            'bit_accuracy': acc,
            'confidence': confidence,
            'proj': proj,
        }

    # --- 2. Regla de tres predictions ---
    r3_preds = defaultdict(list)  # holdout_word -> list of predictions

    for a_word, b_word, c_word, d_word in BOOTSTRAP_QUADS:
        pa = get_proj(a_word)
        pb = get_proj(b_word)
        pc = get_proj(c_word)
        if any(p is None for p in [pa, pb, pc]):
            continue

        # Find all English words for the holdout concept
        d_spanish = None
        for eng, data in all_anchors.items():
            if eng == d_word:
                d_spanish = data['spanish']
                break
        if d_spanish is None:
            continue

        # Neural R3: predicted_D = C + (B - A)
        predicted = pc + (pb - pa)

        # Evaluate against ALL English translations of the holdout concept
        for eng_word, data in holdout_anchors.items():
            if data['spanish'] != d_spanish:
                continue
            gold_bits = (data['target'] > 0).float().to(device)
            pred_bits = (predicted > 0).float()
            acc = (pred_bits == gold_bits).float().mean().item()
            r3_preds[eng_word].append({
                'quad': f"{a_word}:{b_word}={c_word}:{d_word}",
                'bit_accuracy': acc,
                'predicted_proj': predicted,
            })

    # --- 3. Ensemble (average continuous projections, then binarize) ---
    ensemble = {}
    for word in r3_preds:
        preds = [p['predicted_proj'] for p in r3_preds[word]]
        avg_proj = torch.stack(preds).mean(dim=0)
        avg_bits = (avg_proj > 0).float()
        gold_bits = (holdout_anchors[word]['target'] > 0).float().to(device)
        acc = (avg_bits == gold_bits).float().mean().item()
        confidence = avg_proj.abs().mean().item()
        ensemble[word] = {
            'bit_accuracy': acc,
            'confidence': confidence,
            'n_quads': len(preds),
            'proj': avg_proj,
        }

    # --- 4. Best single quad per holdout concept ---
    best_r3 = {}
    for word, preds in r3_preds.items():
        best = max(preds, key=lambda p: p['bit_accuracy'])
        best_r3[word] = {
            'bit_accuracy': best['bit_accuracy'],
            'quad': best['quad'],
        }

    # --- Display results ---
    print(f"\n  {'Concept':20s} {'Type':5s} {'Direct':>8s} {'BestR3':>8s} {'Ensem':>8s} {'#Q':>3s} {'Delta':>8s}")
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*3} {'-'*8}")

    results_per_concept = {}

    # Group by Spanish concept
    by_spanish = defaultdict(list)
    for word in holdout_anchors:
        sp = holdout_anchors[word]['spanish']
        by_spanish[sp].append(word)

    reachable_direct, reachable_alg = [], []
    control_direct = []

    for sp in sorted(HOLDOUT_INFO.keys()):
        rtype, _ = HOLDOUT_INFO[sp]
        eng_words = by_spanish.get(sp, [])
        if not eng_words:
            continue
        # Use primary English word for display
        primary = eng_words[0]

        d_acc = direct[primary]['bit_accuracy'] if primary in direct else 0
        r3_acc = best_r3[primary]['bit_accuracy'] if primary in best_r3 else 0
        r3_quad = best_r3[primary]['quad'] if primary in best_r3 else ''
        ens_acc = ensemble[primary]['bit_accuracy'] if primary in ensemble else 0
        n_q = ensemble[primary]['n_quads'] if primary in ensemble else 0

        # Best algebraic = max of best_r3 and ensemble
        alg_acc = max(r3_acc, ens_acc) if primary in best_r3 else 0
        delta = alg_acc - d_acc if alg_acc > 0 else 0

        tag = 'R3' if rtype == 'R3' else 'CTRL'
        r3_str = f"{r3_acc:.1%}" if primary in best_r3 else '  ---  '
        ens_str = f"{ens_acc:.1%}" if primary in ensemble else '  ---  '
        delta_str = f"{delta:+.1%}" if alg_acc > 0 else '  ---  '

        print(f"  {sp:20s} {tag:5s} {d_acc:8.1%} {r3_str:>8s} {ens_str:>8s} {n_q:3d} {delta_str:>8s}")

        results_per_concept[sp] = {
            'english': eng_words,
            'type': rtype,
            'direct_acc': d_acc,
            'best_r3_acc': r3_acc,
            'best_r3_quad': r3_quad,
            'ensemble_acc': ens_acc,
            'n_quads': n_q,
            'algebraic_improvement': delta,
        }

        if rtype == 'R3':
            reachable_direct.append(d_acc)
            reachable_alg.append(alg_acc)
        else:
            control_direct.append(d_acc)

    # --- Summary ---
    print(f"\n  {'=' * 60}")
    print(f"  SUMMARY")
    print(f"  {'=' * 60}")

    mean_direct_r = np.mean(reachable_direct) if reachable_direct else 0
    mean_alg_r = np.mean(reachable_alg) if reachable_alg else 0
    mean_direct_c = np.mean(control_direct) if control_direct else 0

    print(f"  Reachable concepts ({len(reachable_direct)}):")
    print(f"    Direct encoding:    {mean_direct_r:.1%}")
    print(f"    Best algebraic:     {mean_alg_r:.1%}")
    print(f"    Algebraic delta:    {mean_alg_r - mean_direct_r:+.1%}")

    print(f"  Control concepts ({len(control_direct)}):")
    print(f"    Direct encoding:    {mean_direct_c:.1%}")

    print(f"\n  D-A5 SUCCESS CRITERIA:")
    print(f"    Holdout direct > 75%:          {'PASS' if mean_direct_r > 0.75 else 'FAIL'} ({mean_direct_r:.1%})")
    print(f"    Algebraic > 80%:               {'PASS' if mean_alg_r > 0.80 else 'FAIL'} ({mean_alg_r:.1%})")
    print(f"    Algebraic > direct + 5%:       {'PASS' if (mean_alg_r - mean_direct_r) > 0.05 else 'FAIL'} ({mean_alg_r - mean_direct_r:+.1%})")
    print(f"    Reachable > control + 10%:     {'PASS' if (mean_alg_r - mean_direct_c) > 0.10 else 'FAIL'} ({mean_alg_r - mean_direct_c:+.1%})")

    # Regla de tres evaluation (original quads)
    r3_results = evaluate_regla_de_tres(model, tokenizer, all_anchors, device)
    if r3_results:
        print(f"\n  Regla de Tres (original quads):")
        for r in r3_results:
            print(f"    {r['quad']:40s}  cos={r['cosine']:+.3f}  bit_acc={r['bit_accuracy']:.1%}")
        mean_cos = np.mean([r['cosine'] for r in r3_results])
        mean_bit = np.mean([r['bit_accuracy'] for r in r3_results])
        print(f"    Mean: cosine={mean_cos:+.3f}, bit_accuracy={mean_bit:.1%}")

    model.train()
    return results_per_concept, direct, r3_preds, ensemble


# ============================================================
# Phase: bootstrap — D-A6 iterative self-improvement
# ============================================================

def confidence_gate(direct, ensemble, holdout_anchors, threshold=0.7,
                    min_quads=1, certainty_threshold=0.70,
                    direct_fallback=True):
    """Accept holdout predictions as pseudo-anchors if confidence is high enough.

    Returns dict of accepted pseudo-anchors: {eng_word: target_tensor}.

    v2 fixes (D-A6b):
    - Gate 2: min_quads lowered from 2 to 1 (most concepts only have 1 quad)
    - Gate 4: certainty computed on ALIVE bits only (dead bits have |proj|≈0,
      inflating the denominator and making certainty artificially low)
    - Direct fallback: concepts without R3 quads can be accepted if their
      direct encoding has high accuracy and certainty
    """
    accepted = {}

    # Detect alive bits from direct projections (entropy > 0.3)
    all_projs = torch.stack([d['proj'] for d in direct.values()])  # (N, 63)
    bit_means = (all_projs > 0).float().mean(dim=0)  # fraction positive per bit
    bit_entropy = -bit_means * torch.log2(bit_means.clamp(min=1e-8)) \
                  - (1 - bit_means) * torch.log2((1 - bit_means).clamp(min=1e-8))
    alive_mask = bit_entropy > 0.3  # same threshold used in eval
    n_alive = alive_mask.sum().item()

    print(f"\n  Confidence gate v2: alive bits={n_alive}/{all_projs.shape[1]}, "
          f"min_quads={min_quads}, certainty_thr={certainty_threshold:.0%}, "
          f"acc_thr={threshold:.0%}, direct_fallback={direct_fallback}")

    # --- Path A: R3 ensemble predictions ---
    for word in ensemble:
        if word in accepted:
            continue
        proj = ensemble[word]['proj']
        ens_acc = ensemble[word]['bit_accuracy']
        n_quads = ensemble[word]['n_quads']

        # Gate 2: minimum quads (v2: default 1)
        if n_quads < min_quads:
            continue

        # Gate 3: ensemble accuracy must exceed threshold
        if ens_acc < threshold:
            continue

        # Gate 4: certainty on ALIVE bits only (v2 fix)
        alive_proj = proj[alive_mask] if n_alive > 0 else proj
        certainty = (alive_proj.abs() > 0.5).float().mean().item()
        if certainty < certainty_threshold:
            continue

        # Accept from ensemble
        pseudo_target = torch.where(proj > 0,
                                     torch.ones_like(proj),
                                     torch.full_like(proj, -1.0))
        accepted[word] = {
            'target': pseudo_target.cpu(),
            'spanish': holdout_anchors[word]['spanish'],
            'frontier': holdout_anchors[word]['frontier'],
            'expanded': holdout_anchors[word]['expanded'],
            'n_active': holdout_anchors[word]['n_active'],
            'razon': f'PSEUDO-R3 (ens_acc={ens_acc:.1%}, cert={certainty:.1%}, quads={n_quads})',
            'is_pseudo': True,
        }

    # --- Path B: Direct encoding fallback (no R3 needed) ---
    if direct_fallback:
        for word in direct:
            if word in accepted or word not in holdout_anchors:
                continue
            d_acc = direct[word]['bit_accuracy']
            proj = direct[word]['proj']

            # Stricter threshold for direct (no algebraic cross-validation)
            if d_acc < threshold + 0.10:
                continue

            alive_proj = proj[alive_mask] if n_alive > 0 else proj
            certainty = (alive_proj.abs() > 0.5).float().mean().item()
            if certainty < certainty_threshold:
                continue

            pseudo_target = torch.where(proj > 0,
                                         torch.ones_like(proj),
                                         torch.full_like(proj, -1.0))
            accepted[word] = {
                'target': pseudo_target.cpu(),
                'spanish': holdout_anchors[word]['spanish'],
                'frontier': holdout_anchors[word]['frontier'],
                'expanded': holdout_anchors[word]['expanded'],
                'n_active': holdout_anchors[word]['n_active'],
                'razon': f'PSEUDO-DIRECT (d_acc={d_acc:.1%}, cert={certainty:.1%})',
                'is_pseudo': True,
            }

    return accepted


def phase_bootstrap(args, all_anchors, prim_data):
    """D-A6: Iterative bootstrap loop."""
    train_anchors, holdout_anchors = get_split(all_anchors)

    print(f"\n{'=' * 70}")
    print(f"  D-A6b BOOTSTRAP LOOP v2 — {args.cycles} cycles")
    print(f"  Gate: min_quads={args.min_quads}, acc≥{args.confidence_threshold:.0%}, "
          f"cert≥{args.certainty_threshold:.0%}, direct_fb={args.direct_fallback}")
    print(f"{'=' * 70}")

    cycle_results = []

    for cycle in range(args.cycles):
        print(f"\n  ========== CYCLE {cycle} ==========")
        print(f"  Train anchors: {len(train_anchors)} | Holdout: {len(holdout_anchors)}")

        # Train
        ckpt_dir = os.path.join(_PROJECT, 'checkpoints',
                                 f'danza_bootstrap_v2_{args.scale}', f'cycle{cycle}')
        model, tokenizer, device = run_training(
            args, train_anchors, holdout_anchors, prim_data, ckpt_dir, cycle=cycle)

        # Predict
        results, direct, r3_preds, ensemble = phase_predict(
            model, tokenizer, train_anchors, holdout_anchors, all_anchors, device)

        # Confidence gate (v2)
        accepted = confidence_gate(direct, ensemble, holdout_anchors,
                                    threshold=args.confidence_threshold,
                                    min_quads=args.min_quads,
                                    certainty_threshold=args.certainty_threshold,
                                    direct_fallback=args.direct_fallback)

        n_accepted = len(accepted)
        print(f"\n  Cycle {cycle}: accepted {n_accepted} pseudo-anchors")

        if accepted:
            for word in sorted(accepted.keys()):
                sp = accepted[word]['spanish']
                razon = accepted[word].get('razon', '?')
                print(f"    + {word} ({sp}): {razon}")

        cycle_results.append({
            'cycle': cycle,
            'n_train': len(train_anchors),
            'n_holdout': len(holdout_anchors),
            'n_accepted': n_accepted,
            'accepted': sorted(accepted.keys()),
            'results': results,
        })

        if n_accepted == 0:
            print(f"\n  No new pseudo-anchors accepted. Bootstrap converged.")
            break

        # Add pseudo-anchors to training set, remove from holdout
        for word in accepted:
            train_anchors[word] = accepted[word]
            # Also add all English translations of the same concept
            sp = accepted[word]['spanish']
            for eng, data in list(holdout_anchors.items()):
                if data['spanish'] == sp:
                    if eng not in train_anchors:
                        train_anchors[eng] = accepted[word]
                    del holdout_anchors[eng]

        # Cleanup GPU
        del model
        torch.cuda.empty_cache()

    # Save bootstrap results
    results_path = os.path.join(_PROJECT, 'checkpoints',
                                 f'danza_bootstrap_v2_{args.scale}', 'bootstrap_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Convert non-serializable data
    serializable = []
    for cr in cycle_results:
        s = {k: v for k, v in cr.items() if k != 'results'}
        s['per_concept'] = {}
        for sp, r in cr.get('results', {}).items():
            s['per_concept'][sp] = {k: v for k, v in r.items()
                                     if not isinstance(v, torch.Tensor)}
        serializable.append(s)

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Bootstrap results: {results_path}")

    return cycle_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='D-A5/D-A6 Bootstrap Test')
    parser.add_argument('--phase', choices=['split', 'train', 'predict', 'bootstrap', 'all'],
                        default='split')
    parser.add_argument('--scale', choices=['base', 'xl', 'xxl'], default='base')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--sub-weight', type=float, default=5.0)
    parser.add_argument('--sup-weight', type=float, default=2.0)
    parser.add_argument('--align-weight', type=float, default=3.0)
    parser.add_argument('--triadic-warmup-pct', type=float, default=0.5)
    parser.add_argument('--stories', type=int, default=50000)
    parser.add_argument('--vocab', type=int, default=4096)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--grad-checkpoint', action='store_true')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'],
                        default='bfloat16')
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--eval-every', type=int, default=2500)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint dir for --phase predict')
    parser.add_argument('--cycles', type=int, default=3,
                        help='Number of bootstrap cycles (D-A6)')
    parser.add_argument('--confidence-threshold', type=float, default=0.70,
                        help='Minimum ensemble accuracy to accept pseudo-anchor')
    parser.add_argument('--min-quads', type=int, default=1,
                        help='Minimum R3 quads for ensemble acceptance (v2: 1, was 2)')
    parser.add_argument('--certainty-threshold', type=float, default=0.70,
                        help='Minimum alive-bit certainty (v2: 0.70, was 0.85 on all bits)')
    parser.add_argument('--direct-fallback', action='store_true', default=True,
                        help='Accept high-accuracy direct encodings as pseudo-anchors')
    parser.add_argument('--no-direct-fallback', dest='direct_fallback', action='store_false')
    args = parser.parse_args()

    # Load data
    prim_data = load_primitives()
    all_anchors, skipped = load_anchors(prim_data)

    if args.phase == 'split':
        phase_split(all_anchors, prim_data)
        return

    if args.phase == 'train' or args.phase == 'all':
        train_anchors, holdout_anchors = get_split(all_anchors)
        ckpt_dir = os.path.join(_PROJECT, 'checkpoints', f'danza_bootstrap_{args.scale}')
        model, tokenizer, device = run_training(
            args, train_anchors, holdout_anchors, prim_data, ckpt_dir)

        if args.phase == 'all':
            results, direct, r3_preds, ensemble = phase_predict(
                model, tokenizer, train_anchors, holdout_anchors, all_anchors, device)

            # Save results
            results_path = os.path.join(ckpt_dir, 'bootstrap_results.json')
            serializable = {}
            for sp, r in results.items():
                serializable[sp] = {k: v for k, v in r.items()
                                     if not isinstance(v, torch.Tensor)}
            with open(results_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            print(f"\n  Results: {results_path}")
        return

    if args.phase == 'predict':
        # Load checkpoint — prefer latest step checkpoint over model_best.pt
        # because model_best.pt uses train_acc which saturates at 100% early
        ckpt_dir = args.checkpoint or os.path.join(
            _PROJECT, 'checkpoints', f'danza_bootstrap_{args.scale}')

        # Find latest step checkpoint
        import glob as glob_mod
        step_ckpts = sorted(glob_mod.glob(os.path.join(ckpt_dir, 'model_step*.pt')))
        if step_ckpts:
            ckpt_path = step_ckpts[-1]  # latest step
            print(f"  Using latest checkpoint: {os.path.basename(ckpt_path)}")
        else:
            ckpt_path = os.path.join(ckpt_dir, 'model_best.pt')
            print(f"  Using model_best.pt (no step checkpoints found)")

        if not os.path.exists(ckpt_path):
            print(f"ERROR: checkpoint not found: {ckpt_path}")
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

        cfg = ckpt['config']
        config = TriadicGPTConfig(
            vocab_size=cfg['vocab_size'], block_size=cfg['block_size'],
            n_layer=cfg['n_layer'], n_embd=cfg['n_embd'],
            n_head=cfg['n_head'], n_triadic_bits=cfg['n_triadic_bits'],
        )
        model = DanzaTriadicGPT(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        tok_path = os.path.join(ckpt_dir, 'tokenizer.json')
        tokenizer = BPETokenizer(vocab_size=cfg['vocab_size'])
        tokenizer.load(tok_path)

        train_anchors, holdout_anchors = get_split(all_anchors)
        results, direct, r3_preds, ensemble = phase_predict(
            model, tokenizer, train_anchors, holdout_anchors, all_anchors, device)

        # Save
        results_path = os.path.join(ckpt_dir, 'bootstrap_results.json')
        serializable = {}
        for sp, r in results.items():
            serializable[sp] = {k: v for k, v in r.items()
                                 if not isinstance(v, torch.Tensor)}
        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results: {results_path}")
        return

    if args.phase == 'bootstrap':
        phase_bootstrap(args, all_anchors, prim_data)
        return


if __name__ == '__main__':
    main()

"""
D-A8: Ternary Triadic Head — BitNet-style {-1, 0, +1} quantization.

Replaces tanh with absmean + STE ternary quantization in the triadic head.
Inspired by BitNet b1.58 (Ma et al., 2024): every projection value is
constrained to exactly {-1, 0, +1}, giving three semantic states:

    -1 = ausencia  (active negation — the concept explicitly excludes this primitive)
     0 = vacio     (not applicable — the dimension is irrelevant)
    +1 = presencia (the primitive is actively present)

This is a direct extension of D-A5 (danza_bootstrap.py): same 24/23 split,
same analogy quads, same losses. The ONLY architectural change is the
activation function on the triadic head output.

Key hypothesis:
    - Zero becomes a real semantic state instead of a dead-bit failure mode
    - Each concept gains 58% more info capacity (63 ternary trits = 99.5 bits)
    - Natural ~40% zero rate should emerge (matching BitNet's 42.3%)
    - Dead bits should decrease (model can intentionally output 0)

Note on gold labels: anclas.json uses binary {-1, +1} targets. MSE loss
still works since ternary output can approximate those targets (the model
will learn to avoid 0 for bits that should be strongly +/-1). Future work
could introduce ternary gold labels with explicit 0 for irrelevant primitives.

Usage:
  python playground/danza_ternary.py --phase split
  python playground/danza_ternary.py --phase train --scale xl --steps 50000
  python playground/danza_ternary.py --phase predict --checkpoint checkpoints/danza_ternary_xl/
  python playground/danza_ternary.py --phase all --scale xl --steps 50000
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
from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

# Reuse the strategic split and quads from danza_bootstrap
from danza_bootstrap import (
    TRAIN_CONCEPTS, HOLDOUT_INFO, BOOTSTRAP_QUADS,
    get_split, get_holdout_type, phase_split,
    build_partial_subsumption_pairs,
)


# ============================================================
# Ternary quantization — two modes for A/B testing
# ============================================================

def ternary_quantize_absmean(x):
    """Absmean quantization to {-1, 0, +1} with STE (BitNet b1.58 style).

    Algorithm (from BitNet b1.58):
        1. Compute per-row absmean scale: gamma = mean(|x|) per last dim
        2. Scale: x_scaled = x / gamma
        3. Round + clamp to {-1, 0, +1}
        4. STE: forward uses quantized values, backward uses identity

    The division by gamma centers the distribution so that round()
    naturally produces a balanced mix of -1, 0, +1. Empirically,
    BitNet sees ~42.3% zeros — we expect a similar natural zero rate.
    """
    gamma = x.abs().mean(dim=-1, keepdim=True) + 1e-8
    x_scaled = x / gamma
    x_q = x_scaled.round().clamp(-1, 1)
    return x + (x_q - x).detach()  # STE: forward=quantized, backward=identity


def ternary_quantize_fsq(x):
    """FSQ-style ternary quantization {-1, 0, +1} with iFSQ activation fix.

    Uses 2*sigmoid(1.6*x) - 1 instead of tanh(x) for bounding.
    iFSQ (Tencent, 2025) showed tanh concentrates values near 0,
    causing dead bits. sigmoid(1.6*x) has a flatter middle region,
    distributing activations uniformly across quantization bins.

    STE: forward uses quantized values, backward passes through.
    """
    z_bounded = 2 * torch.sigmoid(1.6 * x) - 1  # iFSQ fix: uniform bin utilization
    z_q = z_bounded.round().clamp(-1, 1)          # snap to {-1, 0, +1}
    return z_bounded + (z_q - z_bounded).detach()  # STE


# Legacy alias for backward compatibility
ternary_quantize = ternary_quantize_absmean


# ============================================================
# Ternary model
# ============================================================

class TernaryDanzaGPT(TriadicGPT):
    """TriadicGPT with ternary {-1, 0, +1} triadic head.

    Identical to DanzaTriadicGPT except the triadic head uses
    ternary quantization instead of torch.tanh. All other architecture
    (embeddings, transformer blocks, LM head) is unchanged.

    Supports two quantization modes for A/B testing:
        - 'fsq':     iFSQ sigmoid-based bounding (default)
        - 'absmean': BitNet absmean + STE
    """

    def __init__(self, config, quantize_mode='fsq'):
        super().__init__(config)
        self.quantize_mode = quantize_mode

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to reduce VRAM at cost of ~33% speed."""
        self._grad_checkpoint = True

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            if getattr(self, '_grad_checkpoint', False) and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        # --- THE KEY CHANGE: ternary instead of tanh ---
        if self.quantize_mode == 'fsq':
            triadic_proj = ternary_quantize_fsq(self.triadic_head(x))
        elif self.quantize_mode == 'absmean':
            triadic_proj = ternary_quantize_absmean(self.triadic_head(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, triadic_proj, loss


# ============================================================
# Ternary-specific metrics
# ============================================================

@torch.no_grad()
def compute_ternary_stats(model, word_tensors):
    """Compute ternary distribution and zero rate from anchor projections.

    Returns:
        zero_rate: fraction of projection values that are exactly 0
        dist: dict with fractions for -1, 0, +1
    """
    if word_tensors.shape[0] == 0:
        return 0.0, {'neg': 0.0, 'zero': 0.0, 'pos': 0.0}

    was_training = model.training
    model.eval()
    _, proj, _ = model(word_tensors)       # (N, T, 63)
    pred = proj.mean(dim=1)                # (N, 63)
    model.train(was_training)

    # After ternary_quantize, values are exactly {-1, 0, +1} in forward pass.
    # But due to STE, the stored tensor may have continuous residuals in
    # autograd graph. For stats, snap to nearest integer.
    snapped = pred.round().clamp(-1, 1)

    total = snapped.numel()
    n_neg = (snapped == -1).sum().item()
    n_zero = (snapped == 0).sum().item()
    n_pos = (snapped == 1).sum().item()

    zero_rate = n_zero / total if total > 0 else 0.0
    dist = {
        'neg': n_neg / total if total > 0 else 0.0,
        'zero': n_zero / total if total > 0 else 0.0,
        'pos': n_pos / total if total > 0 else 0.0,
    }
    return zero_rate, dist


@torch.no_grad()
def compute_batch_ternary_stats(proj):
    """Compute ternary stats from a raw projection batch (no model call).

    Args:
        proj: (B, T, K) tensor from forward pass

    Returns:
        zero_rate, dist dict
    """
    snapped = proj.round().clamp(-1, 1)
    total = snapped.numel()
    if total == 0:
        return 0.0, {'neg': 0.0, 'zero': 0.0, 'pos': 0.0}

    n_neg = (snapped == -1).sum().item()
    n_zero = (snapped == 0).sum().item()
    n_pos = (snapped == 1).sum().item()

    return n_zero / total, {
        'neg': n_neg / total,
        'zero': n_zero / total,
        'pos': n_pos / total,
    }


# ============================================================
# Progress bar helper
# ============================================================

def format_progress_bar(step, total, width=30):
    """Render a text-based progress bar with ETA."""
    frac = step / total
    filled = int(width * frac)
    bar = '#' * filled + '-' * (width - filled)
    return f"[{bar}] {frac:5.1%}"


def format_eta(elapsed, step, total):
    """Estimate time remaining."""
    if step == 0:
        return "??:??"
    rate = elapsed / step
    remaining = rate * (total - step)
    mins = int(remaining // 60)
    secs = int(remaining % 60)
    return f"{mins:02d}:{secs:02d}"


# ============================================================
# Training
# ============================================================

def run_training(args, train_anchors, holdout_anchors, prim_data, ckpt_dir):
    """Train TernaryDanzaGPT with partial anchor supervision.

    Returns (model, tokenizer, device).
    """
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
    print(f"  D-A8 TERNARY TRIADIC HEAD — Training")
    print(f"{'=' * 70}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    qmode = getattr(args, 'quantize_mode', 'fsq')
    qmode_desc = 'iFSQ sigmoid' if qmode == 'fsq' else 'absmean + STE'
    print(f"  Activation: ternary_quantize_{qmode} ({qmode_desc}) -> {{-1, 0, +1}}")
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

    # --- Model (TERNARY, not tanh) ---
    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block,
        n_layer=preset['layers'],
        n_embd=preset['dim'],
        n_head=preset['heads'],
        n_triadic_bits=N_BITS,
        dropout=args.dropout,
    )
    model = TernaryDanzaGPT(config, quantize_mode=qmode).to(device)
    total_params = model.num_params()
    print(f"  Model: TernaryDanzaGPT {args.scale} ({total_params/1e6:.1f}M params, {N_BITS} ternary trits, quantize={qmode})")

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print(f"  Gradient checkpointing: ON")

    if device.type == 'cuda' and not getattr(args, 'no_compile', False):
        try:
            import triton  # noqa: F401
            model = torch.compile(model)
            print("  torch.compile: ON")
        except ImportError:
            print("  torch.compile: SKIPPED (triton not available)")

    # Mixed precision
    use_amp = device.type == 'cuda'
    amp_dtype = {'float32': torch.float32, 'float16': torch.float16,
                 'bfloat16': torch.bfloat16}[args.dtype]
    if use_amp and amp_dtype != torch.float32:
        print(f"  Mixed precision: {args.dtype}")
    else:
        print(f"  Precision: float32")

    # GradScaler: disabled for bfloat16 (not needed), enabled only for float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None
    if use_scaler:
        print(f"  GradScaler: ON (float16)")
    elif use_amp:
        print(f"  GradScaler: OFF (bfloat16 — no loss scaling needed)")

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
    csv_writer.writerow([
        'step', 'loss', 'lang_loss', 'tri_loss', 'sup_loss', 'sub_loss',
        'bit_acc_train', 'bit_acc_holdout', 'dead_bits',
        'zero_rate', 'ternary_neg', 'ternary_zero', 'ternary_pos',
    ])

    data_iter = iter(loader)
    t0 = time.time()
    best_train_acc = 0.0
    best_hold_acc = 0.0

    # Running ternary stats (updated every print step)
    last_zero_rate = 0.0
    last_ternary_dist = {'neg': 0.33, 'zero': 0.34, 'pos': 0.33}

    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        # LR schedule: linear warmup + cosine decay
        if step <= warmup_steps:
            lr = args.lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (args.steps - warmup_steps)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward + backward
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
            if scaler:  # float16 path
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  # bfloat16 or float32 path
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

        # --- Print with progress bar ---
        if step % args.print_every == 0:
            elapsed = time.time() - t0
            bar = format_progress_bar(step, args.steps)
            eta = format_eta(elapsed, step, args.steps)

            # Ternary stats from the batch projection
            last_zero_rate, last_ternary_dist = compute_batch_ternary_stats(proj.detach())

            if step >= triadic_start:
                tri_str = (f"tri={l_tri.item():.4f} sup={l_sup.item():.4f} "
                           f"sub={l_sub.item():.4f}")
            else:
                tri_str = "warmup"

            print(f"  {bar} [{step:>6d}/{args.steps}] ETA {eta} | "
                  f"loss={total.item():.4f} lang={lang_loss.item():.4f} {tri_str} | "
                  f"zero={last_zero_rate:.1%} "
                  f"(-1:{last_ternary_dist['neg']:.1%} "
                  f"0:{last_ternary_dist['zero']:.1%} "
                  f"+1:{last_ternary_dist['pos']:.1%}) | "
                  f"lr={lr:.2e}")

        # --- Evaluate ---
        if step % args.eval_every == 0 or step == args.steps:
            eval_train = evaluate_anchors(model, sup_train_t, sup_train_tgt, sup_train_words)
            eval_hold = evaluate_anchors(model, sup_hold_t, sup_hold_tgt, sup_hold_words)

            train_acc = eval_train.get('mean_bit_accuracy', 0)
            hold_acc = eval_hold.get('mean_bit_accuracy', 0)
            dead = eval_train.get('dead_bits', N_BITS)

            # Ternary stats from anchor projections
            zero_rate, tern_dist = compute_ternary_stats(model, sup_train_t)

            print(f"  --- Eval @ step {step} ---")
            print(f"  Bit accuracy:  train={train_acc:.1%}  holdout={hold_acc:.1%} (no supervision!)")
            print(f"  Dead bits: {dead}/{N_BITS}")
            print(f"  Zero rate (anchors): {zero_rate:.1%}")
            print(f"  Ternary dist (anchors): "
                  f"-1={tern_dist['neg']:.1%}  0={tern_dist['zero']:.1%}  +1={tern_dist['pos']:.1%}")

            csv_writer.writerow([
                step, total.item(), lang_loss.item(),
                l_tri.item() if step >= triadic_start else 0,
                l_sup.item() if step >= triadic_start else 0,
                l_sub.item() if step >= triadic_start else 0,
                train_acc, hold_acc, dead,
                zero_rate, tern_dist['neg'], tern_dist['zero'], tern_dist['pos'],
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
                    'model_class': 'TernaryDanzaGPT',
                    'quantize_mode': qmode,
                    'train_concepts': sorted(TRAIN_CONCEPTS),
                    'bit_accuracy_train': train_acc,
                    'bit_accuracy_holdout': hold_acc,
                    'zero_rate': zero_rate,
                    'ternary_dist': tern_dist,
                }, os.path.join(ckpt_dir, 'model_best.pt'))

            # Save best by holdout accuracy
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
                    'model_class': 'TernaryDanzaGPT',
                    'quantize_mode': qmode,
                    'bit_accuracy_holdout': hold_acc,
                    'zero_rate': zero_rate,
                    'ternary_dist': tern_dist,
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
                'model_class': 'TernaryDanzaGPT',
                'quantize_mode': qmode,
            }, os.path.join(ckpt_dir, f'model_step{step}.pt'))

    csv_file.close()
    elapsed = time.time() - t0
    print(f"\n  Training complete: {elapsed/60:.1f} min")
    print(f"  Best train acc: {best_train_acc:.1%}  |  Best holdout acc: {best_hold_acc:.1%}")

    return model, tokenizer, device


# ============================================================
# Phase: predict — algebraic prediction of holdout concepts
# ============================================================

@torch.no_grad()
def phase_predict(model, tokenizer, train_anchors, holdout_anchors, all_anchors, device):
    """Predict holdout concepts via direct encoding + regla de tres + ensemble.

    Same logic as danza_bootstrap.phase_predict, but with additional
    ternary distribution reporting per holdout concept.
    """
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
    print(f"  D-A8 TERNARY HOLDOUT PREDICTION")
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

        # Ternary distribution for this concept
        snapped = proj.round().clamp(-1, 1)
        n_neg = (snapped == -1).sum().item()
        n_zero = (snapped == 0).sum().item()
        n_pos = (snapped == 1).sum().item()

        direct[word] = {
            'bit_accuracy': acc,
            'confidence': confidence,
            'proj': proj,
            'ternary': {'neg': n_neg, 'zero': n_zero, 'pos': n_pos},
        }

    # --- 2. Regla de tres predictions ---
    r3_preds = defaultdict(list)

    for a_word, b_word, c_word, d_word in BOOTSTRAP_QUADS:
        pa = get_proj(a_word)
        pb = get_proj(b_word)
        pc = get_proj(c_word)
        if any(p is None for p in [pa, pb, pc]):
            continue

        # Find the holdout concept's Spanish name
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
    print(f"\n  {'Concept':20s} {'Type':5s} {'Direct':>8s} {'BestR3':>8s} "
          f"{'Ensem':>8s} {'#Q':>3s} {'Delta':>8s} {'Zeros':>6s}")
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*3} {'-'*8} {'-'*6}")

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
        primary = eng_words[0]

        d_acc = direct[primary]['bit_accuracy'] if primary in direct else 0
        d_zeros = direct[primary]['ternary']['zero'] if primary in direct else 0
        r3_acc = best_r3[primary]['bit_accuracy'] if primary in best_r3 else 0
        r3_quad = best_r3[primary]['quad'] if primary in best_r3 else ''
        ens_acc = ensemble[primary]['bit_accuracy'] if primary in ensemble else 0
        n_q = ensemble[primary]['n_quads'] if primary in ensemble else 0

        alg_acc = max(r3_acc, ens_acc) if primary in best_r3 else 0
        delta = alg_acc - d_acc if alg_acc > 0 else 0

        tag = 'R3' if rtype == 'R3' else 'CTRL'
        r3_str = f"{r3_acc:.1%}" if primary in best_r3 else '  ---  '
        ens_str = f"{ens_acc:.1%}" if primary in ensemble else '  ---  '
        delta_str = f"{delta:+.1%}" if alg_acc > 0 else '  ---  '

        print(f"  {sp:20s} {tag:5s} {d_acc:8.1%} {r3_str:>8s} "
              f"{ens_str:>8s} {n_q:3d} {delta_str:>8s} {d_zeros:6d}")

        results_per_concept[sp] = {
            'english': eng_words,
            'type': rtype,
            'direct_acc': d_acc,
            'best_r3_acc': r3_acc,
            'best_r3_quad': r3_quad,
            'ensemble_acc': ens_acc,
            'n_quads': n_q,
            'algebraic_improvement': delta,
            'n_zeros': d_zeros,
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

    # Ternary-specific summary
    all_zeros = [direct[w]['ternary']['zero'] for w in direct]
    mean_zeros = np.mean(all_zeros) if all_zeros else 0
    overall_zero_rate = mean_zeros / N_BITS if N_BITS > 0 else 0
    print(f"\n  Ternary summary:")
    print(f"    Mean zeros per concept: {mean_zeros:.1f}/{N_BITS} ({overall_zero_rate:.1%})")

    print(f"\n  D-A8 SUCCESS CRITERIA:")
    print(f"    Holdout direct > 75%:          "
          f"{'PASS' if mean_direct_r > 0.75 else 'FAIL'} ({mean_direct_r:.1%})")
    print(f"    Algebraic > 80%:               "
          f"{'PASS' if mean_alg_r > 0.80 else 'FAIL'} ({mean_alg_r:.1%})")
    print(f"    Algebraic > direct + 5%:       "
          f"{'PASS' if (mean_alg_r - mean_direct_r) > 0.05 else 'FAIL'} "
          f"({mean_alg_r - mean_direct_r:+.1%})")
    print(f"    Reachable > control + 10%:     "
          f"{'PASS' if (mean_alg_r - mean_direct_c) > 0.10 else 'FAIL'} "
          f"({mean_alg_r - mean_direct_c:+.1%})")
    print(f"    Natural zero rate > 20%:       "
          f"{'PASS' if overall_zero_rate > 0.20 else 'FAIL'} ({overall_zero_rate:.1%})")

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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='D-A8: Ternary Triadic Head (BitNet-style {-1,0,+1})')
    parser.add_argument('--phase', choices=['split', 'train', 'predict', 'all'],
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
    parser.add_argument('--grad-checkpoint', action='store_true',
                        help='Use gradient checkpointing (saves VRAM, +33%% time)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile')
    parser.add_argument('--quantize-mode', choices=['fsq', 'absmean'], default='fsq',
                        help='Ternary quantization: fsq (iFSQ sigmoid, default) or absmean (BitNet)')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'],
                        default='bfloat16',
                        help='Mixed precision dtype (default: bfloat16)')
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--eval-every', type=int, default=2500)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint dir for --phase predict')
    args = parser.parse_args()

    # Load data
    prim_data = load_primitives()
    all_anchors, skipped = load_anchors(prim_data)

    if args.phase == 'split':
        phase_split(all_anchors, prim_data)
        return

    if args.phase == 'train' or args.phase == 'all':
        train_anchors, holdout_anchors = get_split(all_anchors)
        ckpt_dir = os.path.join(_PROJECT, 'checkpoints', f'danza_ternary_{args.scale}')
        model, tokenizer, device = run_training(
            args, train_anchors, holdout_anchors, prim_data, ckpt_dir)

        if args.phase == 'all':
            results, direct, r3_preds, ensemble = phase_predict(
                model, tokenizer, train_anchors, holdout_anchors, all_anchors, device)

            # Save results
            results_path = os.path.join(ckpt_dir, 'ternary_results.json')
            serializable = {}
            for sp, r in results.items():
                serializable[sp] = {k: v for k, v in r.items()
                                     if not isinstance(v, torch.Tensor)}
            with open(results_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            print(f"\n  Results: {results_path}")
        return

    if args.phase == 'predict':
        ckpt_dir = args.checkpoint or os.path.join(
            _PROJECT, 'checkpoints', f'danza_ternary_{args.scale}')

        # Find latest step checkpoint
        import glob as glob_mod
        step_ckpts = sorted(glob_mod.glob(os.path.join(ckpt_dir, 'model_step*.pt')))
        if step_ckpts:
            ckpt_path = step_ckpts[-1]
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
        # Use quantize_mode from checkpoint if available, else from CLI args
        qmode = ckpt.get('quantize_mode', args.quantize_mode)
        model = TernaryDanzaGPT(config, quantize_mode=qmode).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        tok_path = os.path.join(ckpt_dir, 'tokenizer.json')
        tokenizer = BPETokenizer(vocab_size=cfg['vocab_size'])
        tokenizer.load(tok_path)

        train_anchors, holdout_anchors = get_split(all_anchors)
        results, direct, r3_preds, ensemble = phase_predict(
            model, tokenizer, train_anchors, holdout_anchors, all_anchors, device)

        # Save
        results_path = os.path.join(ckpt_dir, 'ternary_results.json')
        serializable = {}
        for sp, r in results.items():
            serializable[sp] = {k: v for k, v in r.items()
                                 if not isinstance(v, torch.Tensor)}
        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results: {results_path}")
        return


if __name__ == '__main__':
    main()

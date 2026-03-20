"""
D-A19: GPT-2 Medium 355M — Fix Scale-Algebra Tradeoff.

D-A17 gets 97.7% bit accuracy but only 1.7% subsumption — algebraic structure
is destroyed at 355M scale. Root causes identified:

Bug #1: gpt2_triadic_loss() only had alignment — missing diversity, contrastive,
         and entropy regularizers that prevent collapse to ±1 (no zeros).
Bug #2: gpt2_subsumption_loss() used (x > 0).float() — kills gradients,
         making subsumption loss a no-op.

Fix: Port the FULL 4-component triadic_loss() and differentiable
subsumption_loss() from danza_63bit.py, plus a new sparsity target loss
to explicitly preserve ~42% zeros.

Goal: D-A14-level algebra (>90% sub) at 355M scale.

Usage:
    python playground/gpt2_355m_sparsity.py --steps 50000 --v2 --sparsity-weight 2.0
    python playground/gpt2_355m_sparsity.py --steps 50000 --v2 --resume
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

_PLAYGROUND = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

from danza_63bit import (
    load_primitives, load_anchors, load_all_anchors, build_subsumption_pairs,
    supervised_anchor_loss, evaluate_anchors, evaluate_subsumption,
    REGLA_DE_TRES_QUADS, TextDataset,
    ANCHOR_TRANSLATIONS, SKIP_ANCHORS,
    N_BITS, STORY_SEPARATOR,
)
from danza_bootstrap import (
    TRAIN_CONCEPTS, HOLDOUT_INFO, BOOTSTRAP_QUADS,
    get_split, get_holdout_type, phase_split,
    build_partial_subsumption_pairs,
)
from danza_ternary import (
    ternary_quantize_fsq, ternary_quantize_absmean,
    compute_ternary_stats, compute_batch_ternary_stats,
    format_progress_bar, format_eta,
)
from gpt2_medium_ternary import GPT2MediumTernary


# ============================================================
# Full 4-component triadic loss (ported from danza_63bit.py:493)
# ============================================================

def gpt2_full_triadic_loss(proj, align_weight, wte, input_ids):
    """Full 4-component triadic loss for GPT-2 backbone.

    Components (same as danza_63bit.triadic_loss):
      1. Diversity: bit_means → 0 (prevent all bits from being same sign)
      2. Contrastive: push sequences apart (different inputs → different projections)
      3. Entropy: maximize per-bit entropy (prevent dead bits)
      4. Alignment: MSE on cosine similarities (mirror embedding structure)

    D-A17's gpt2_triadic_loss() only had #4 (alignment). Missing #1-3
    allowed the 355M backbone to push all projections to ±1 (no zeros).
    """
    B, T, K = proj.shape

    # 1. Diversity: bit means → 0
    bit_means = proj.mean(dim=(0, 1))
    l_div = (bit_means ** 2).mean()

    # 2. Contrastive: push sequences apart
    seq_means = proj.mean(dim=1)  # (B, K)
    seq_norms = F.normalize(seq_means, dim=1)
    sim_matrix = seq_norms @ seq_norms.T
    mask = ~torch.eye(B, dtype=torch.bool, device=proj.device)
    l_ctr = (sim_matrix[mask] ** 2).mean() if mask.sum() > 0 else torch.tensor(0.0, device=proj.device)

    # 3. Entropy: maximize per-bit entropy
    q = (bit_means + 1) / 2  # map [-1,1] → [0,1]
    eps = 1e-7
    bit_ent = -(q * torch.log2(q + eps) + (1 - q) * torch.log2(1 - q + eps))
    l_ent = 1.0 - bit_ent.mean()

    # 4. Embedding alignment (MSE on cosine similarities)
    with torch.no_grad():
        emb = wte(input_ids)  # (B, T, D)
        emb_norm = F.normalize(emb, dim=-1)

    proj_norm = F.normalize(proj, dim=-1)

    n_pairs = min(64, T * T)
    idx_i = torch.randint(0, T, (n_pairs,), device=proj.device)
    idx_j = torch.randint(0, T, (n_pairs,), device=proj.device)

    sim_emb = (emb_norm[:, idx_i] * emb_norm[:, idx_j]).sum(dim=-1)
    sim_tri = (proj_norm[:, idx_i] * proj_norm[:, idx_j]).sum(dim=-1)

    l_align = F.mse_loss(sim_tri, sim_emb.detach())

    return l_div + l_ctr + l_ent + align_weight * l_align


# ============================================================
# Differentiable subsumption loss (ported from danza_63bit.py:472)
# ============================================================

def gpt2_subsumption_loss(model, h_tensors, y_tensors, n_sample=32):
    """Differentiable subsumption: relu(hyper_01 - hypo_01).mean().

    D-A17's version used (x > 0).float() which kills gradients.
    This uses (x + 1) / 2 to map [-1,+1] → [0,1] differentiably.
    """
    N = h_tensors.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=h_tensors.device)

    if N > n_sample:
        idx = torch.randperm(N, device=h_tensors.device)[:n_sample]
        h_batch, y_batch = h_tensors[idx], y_tensors[idx]
    else:
        h_batch, y_batch = h_tensors, y_tensors

    proj_h = model(h_batch)[1]
    proj_y = model(y_batch)[1]

    # Differentiable 0-1 mapping: (tanh_out + 1) / 2
    h_01 = (proj_h.mean(dim=1) + 1) / 2
    y_01 = (proj_y.mean(dim=1) + 1) / 2

    return F.relu(h_01 - y_01).mean()


# ============================================================
# NEW: Sparsity target loss — preserve ~42% zeros
# ============================================================

def sparsity_target_loss(raw_proj, target_rate=0.42, quantize_mode='fsq'):
    """Penalize deviation from target zero rate.

    At 355M scale, the backbone pushes all projections to ±1 (no zeros).
    This loss explicitly encourages ~42% of activations to be near zero,
    matching the natural ternary distribution seen at 40M scale (D-A14).

    Uses soft approximation of zero count for differentiability:
    sigmoid(10 * (0.5 - |activated|)) ≈ 1 when |activated| < 0.5 (near zero)
    """
    if quantize_mode == 'fsq':
        activated = 2 * torch.sigmoid(1.6 * raw_proj) - 1
    else:
        gamma = raw_proj.abs().mean(dim=-1, keepdim=True) + 1e-8
        activated = raw_proj / gamma

    # Soft zero rate (differentiable)
    soft_zero = torch.sigmoid(10.0 * (0.5 - activated.abs()))
    soft_zero_rate = soft_zero.mean()
    return (soft_zero_rate - target_rate) ** 2


# ============================================================
# GPT-2 Medium with raw projection access
# ============================================================

class GPT2MediumSparsity(GPT2MediumTernary):
    """GPT2MediumTernary subclass that also exposes raw pre-quantization projection.

    The parent class only returns quantized projections. We need the raw
    (pre-quantization) values for the sparsity target loss, which must
    operate on continuous activations to provide useful gradients.

    forward() returns the standard 3-tuple (logits, proj, loss) for
    compatibility with all eval functions. Use forward_with_raw() in
    the training loop to also get raw projections.
    """

    def forward_with_raw(self, input_ids, targets=None):
        """Forward pass returning (logits, quantized_proj, loss, raw_proj)."""
        outputs = self.gpt2.transformer(input_ids=input_ids)
        hidden_states = outputs.last_hidden_state  # (B, T, n_embd)

        # LM head
        logits = self.gpt2.lm_head(hidden_states)

        # Triadic head — get both raw and quantized
        raw = self.triadic_head(hidden_states)
        if self.quantize_mode == 'fsq':
            triadic_proj = ternary_quantize_fsq(raw)
        else:
            triadic_proj = ternary_quantize_absmean(raw)

        # Language loss (shifted)
        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, triadic_proj, loss, raw


# ============================================================
# GPT-2 specific supervised anchor loss
# ============================================================

def gpt2_supervised_anchor_loss(model, word_tensors, targets):
    """Compute supervised anchor loss for GPT-2 model."""
    if word_tensors.shape[0] == 0:
        return torch.tensor(0.0, device=word_tensors.device)

    result = model(word_tensors)
    proj = result[1]  # works for both 3-tuple and 4-tuple returns
    pred = proj.mean(dim=1)  # (N, 63)
    return F.mse_loss(pred, targets)


# ============================================================
# R3 evaluation for GPT-2 tokenizer
# ============================================================

@torch.no_grad()
def _eval_r3_gpt2(model, tokenizer, train_anchors, holdout_anchors, device):
    """Evaluate regla de tres analogies using GPT-2 tokenizer."""
    model.eval()
    all_anchors = {**train_anchors, **holdout_anchors}

    def get_proj(word):
        ids = tokenizer.encode(word, add_special_tokens=False)[:8]
        if not ids:
            return None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        proj = model(x)[1]
        return proj[0].mean(dim=0)  # (63,)

    results = []
    for a_word, b_word, c_word, d_word in REGLA_DE_TRES_QUADS:
        if not all(w in all_anchors for w in [a_word, b_word, c_word, d_word]):
            continue
        pa, pb, pc, pd = get_proj(a_word), get_proj(b_word), get_proj(c_word), get_proj(d_word)
        if any(p is None for p in [pa, pb, pc, pd]):
            continue
        predicted_d = pc + (pb - pa)
        cos = F.cosine_similarity(predicted_d.unsqueeze(0), pd.unsqueeze(0)).item()
        pred_bits = (predicted_d > 0).long()
        actual_bits = (pd > 0).long()
        bit_match = (pred_bits == actual_bits).float().mean().item()
        results.append({
            'quad': f"{a_word}:{b_word}={c_word}:{d_word}",
            'cosine': cos,
            'bit_accuracy': bit_match,
        })

    model.train()
    return results


# ============================================================
# Training
# ============================================================

def run_training(args, train_anchors, holdout_anchors, prim_data, ckpt_dir):
    """Train GPT-2 Medium 355M with full triadic losses + sparsity target."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import glob as _glob

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  D-A19: GPT-2 MEDIUM 355M — FIX SCALE-ALGEBRA TRADEOFF")
    print(f"{'=' * 70}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --- Load GPT-2 Medium ---
    print(f"\n  Loading GPT-2 Medium from HuggingFace...")
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2MediumSparsity(
        gpt2, n_triadic_bits=N_BITS, quantize_mode=args.quantize_mode,
    ).to(device)

    total_params = model.num_params()
    triadic_params = sum(p.numel() for p in model.triadic_head.parameters())
    print(f"  Model: GPT2MediumSparsity ({total_params/1e6:.1f}M total, "
          f"{triadic_params/1e3:.1f}K triadic head, {N_BITS} ternary trits)")
    print(f"  Quantize mode: {args.quantize_mode}")
    print(f"  Sparsity target: {args.sparsity_target:.0%} (weight={args.sparsity_weight})")
    print(f"  Triadic warmup: {args.triadic_warmup_pct:.0%} of steps")
    print(f"  Unfreeze: step {int(args.steps * args.unfreeze_pct)} "
          f"({args.unfreeze_pct:.0%}), last {args.unfreeze_layers} layers")

    # --- Resume from checkpoint ---
    resume_step = 0
    if args.resume:
        ckpt_files = sorted(
            _glob.glob(os.path.join(ckpt_dir, 'model_step*.pt')),
            key=lambda f: int(os.path.basename(f).replace('model_step', '').replace('.pt', '')))
        if ckpt_files:
            latest_ckpt = ckpt_files[-1]
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            resume_step = ckpt['step']
            print(f"  RESUMED from {os.path.basename(latest_ckpt)} (step {resume_step})")
        else:
            print(f"  --resume: no checkpoints found in {ckpt_dir}, starting fresh")

    # Phase 1: freeze backbone
    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Phase 1: backbone frozen, {trainable/1e3:.1f}K trainable params")

    # --- Subsumption pairs ---
    train_sub, test_sub = build_partial_subsumption_pairs(train_anchors)
    print(f"  Subsumption pairs: train={len(train_sub)}, test={len(test_sub)}")

    # --- Pre-encode anchors with GPT-2 tokenizer ---
    def _pack_anchors_gpt2(anchor_dict):
        words, ids_list, targets = [], [], []
        for word, data in anchor_dict.items():
            ids = tokenizer.encode(word, add_special_tokens=False)[:8]
            if ids:
                words.append(word)
                ids_list.append(ids)
                targets.append(data['target'])
        if not words:
            z = torch.zeros((0, 1), dtype=torch.long, device=device)
            return z, torch.zeros((0, N_BITS), device=device), []
        mx = max(len(x) for x in ids_list)
        padded = torch.tensor(
            [x + [tokenizer.eos_token_id] * (mx - len(x)) for x in ids_list],
            dtype=torch.long, device=device)
        target_t = torch.stack(targets).to(device)
        return padded, target_t, words

    sup_train_t, sup_train_tgt, sup_train_words = _pack_anchors_gpt2(train_anchors)
    sup_hold_t, sup_hold_tgt, sup_hold_words = _pack_anchors_gpt2(holdout_anchors)
    print(f"  Supervision: {len(sup_train_words)} train, {len(sup_hold_words)} holdout (eval only)")

    # --- Pre-encode subsumption pairs ---
    def _pack_sub_gpt2(pairs):
        h_ids, y_ids, valid = [], [], []
        for h_w, y_w, h_d, y_d in pairs:
            h = tokenizer.encode(h_w, add_special_tokens=False)[:8]
            y = tokenizer.encode(y_w, add_special_tokens=False)[:8]
            if h and y:
                h_ids.append(h)
                y_ids.append(y)
                valid.append((h_w, y_w))
        if not valid:
            z = torch.zeros((0, 1), dtype=torch.long, device=device)
            return z, z, valid
        def pad(lst):
            mx = max(len(x) for x in lst)
            return torch.tensor(
                [x + [tokenizer.eos_token_id] * (mx - len(x)) for x in lst],
                dtype=torch.long, device=device)
        return pad(h_ids), pad(y_ids), valid

    sub_train_h, sub_train_y, _ = _pack_sub_gpt2(train_sub)
    sub_test_h, sub_test_y, _ = _pack_sub_gpt2(test_sub)

    # --- Data: TinyStories tokenized with GPT-2 tokenizer ---
    data_path = os.path.join(_PROJECT, 'data', 'TinyStories-train.txt')
    print(f"\n  Loading TinyStories...")
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR)
               if s.strip() and len(s.strip()) > 30]
    if args.stories and len(stories) > args.stories:
        random.seed(42)
        random.shuffle(stories)
        stories = stories[:args.stories]

    print(f"  Tokenizing {len(stories)} stories with GPT-2 tokenizer...")
    all_tokens = []
    for story in stories:
        toks = tokenizer.encode(story, add_special_tokens=True)
        all_tokens.extend(toks)
    print(f"  Total: {len(all_tokens):,} tokens")

    # --- Mixed precision ---
    amp_dtype = {'float32': torch.float32, 'float16': torch.float16,
                 'bfloat16': torch.bfloat16}[args.dtype]
    use_scaler = (device.type == 'cuda' and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    print(f"  Mixed precision: {args.dtype}")

    # --- Training loop ---
    block_size = min(args.block, 1024)  # GPT-2 max is 1024
    dataset = TextDataset(all_tokens, block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, pin_memory=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    warmup_steps = int(args.steps * 0.05)
    triadic_start = int(args.steps * args.triadic_warmup_pct) if args.triadic_warmup_pct > 0 else 1
    unfreeze_step = int(args.steps * args.unfreeze_pct)

    csv_path = os.path.join(ckpt_dir, 'training_log.csv')
    if resume_step > 0:
        csv_file = open(csv_path, 'a', newline='')
    else:
        csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    if resume_step == 0:
        csv_writer.writerow([
            'step', 'loss', 'lang_loss', 'tri_loss', 'sup_loss', 'sub_loss',
            'sparsity_loss', 'bit_acc_train', 'bit_acc_holdout', 'dead_bits',
            'zero_rate', 'ternary_neg', 'ternary_zero', 'ternary_pos',
            'sub_train', 'sub_test',
        ])

    data_iter = iter(loader)
    t0 = time.time()
    best_hold_acc = 0.0
    best_sub_test = 0.0
    unfrozen = False

    # If resuming past unfreeze point, unfreeze immediately
    if resume_step >= unfreeze_step:
        model.unfreeze_last_n(args.unfreeze_layers)
        unfrozen = True
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
        print(f"  Resume: already past unfreeze point, Phase 2 active")
    if resume_step > 0:
        best_path = os.path.join(ckpt_dir, 'model_best.pt')
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
            best_hold_acc = best_ckpt.get('bit_accuracy_holdout', 0)
            best_sub_test = best_ckpt.get('sub_test', 0)
            print(f"  Resume: best holdout acc={best_hold_acc:.1%}, sub_test={best_sub_test:.1%}")
            del best_ckpt

    start_step = resume_step + 1
    remaining = args.steps - resume_step
    print(f"\n  Training ({remaining} steps remaining, {start_step}-{args.steps})...")
    print(f"  Triadic losses: active from step {triadic_start}")
    print(f"  Unfreeze: step {unfreeze_step} (last {args.unfreeze_layers} layers)")
    print(f"  Loss weights: alpha={args.alpha} sup={args.sup_weight} sub={args.sub_weight} "
          f"align={args.align_weight} sparsity={args.sparsity_weight}")

    for step in range(start_step, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        # Phase 2: unfreeze backbone layers
        if step == unfreeze_step and not unfrozen:
            model.unfreeze_last_n(args.unfreeze_layers)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
            unfrozen = True

        # LR schedule: linear warmup + cosine decay
        if step <= warmup_steps:
            lr = args.lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (args.steps - warmup_steps)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward + backward
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
            logits, proj, lang_loss, raw_proj = model.forward_with_raw(x, y)

            # Triadic losses (active from triadic_start)
            if step >= triadic_start:
                # Full 4-component triadic loss (FIX #1: was alignment-only)
                l_tri = gpt2_full_triadic_loss(proj, args.align_weight, model.wte, x)
                l_sup = gpt2_supervised_anchor_loss(model, sup_train_t, sup_train_tgt)
                # Differentiable subsumption (FIX #2: was non-differentiable)
                l_sub = gpt2_subsumption_loss(model, sub_train_h, sub_train_y)
                # Sparsity target (NEW: explicit zero preservation)
                l_sparse = sparsity_target_loss(
                    raw_proj, target_rate=args.sparsity_target,
                    quantize_mode=args.quantize_mode)

                triadic_total = args.alpha * (
                    l_tri
                    + args.sup_weight * l_sup
                    + args.sub_weight * l_sub
                    + args.sparsity_weight * l_sparse
                )
            else:
                l_tri = l_sup = l_sub = l_sparse = torch.tensor(0.0, device=device)
                triadic_total = torch.tensor(0.0, device=device)

            # Language loss only after unfreeze (before that, backbone is frozen)
            if unfrozen and lang_loss is not None:
                total = lang_loss + triadic_total
            else:
                total = triadic_total

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # --- Print ---
        if step % args.print_every == 0:
            elapsed = time.time() - t0
            bar = format_progress_bar(step, args.steps)
            eta = format_eta(elapsed, step - (start_step - 1), remaining)

            zero_rate, tern_dist = compute_batch_ternary_stats(proj.detach())

            phase = "P1-frozen" if not unfrozen else "P2-unfrozen"
            lang_str = f"lang={lang_loss.item():.4f}" if lang_loss is not None else "lang=N/A"

            print(f"  {bar} [{step:>6d}/{args.steps}] ETA {eta} | "
                  f"loss={total.item():.4f} {lang_str} "
                  f"tri={l_tri.item():.4f} sup={l_sup.item():.4f} "
                  f"sub={l_sub.item():.4f} spar={l_sparse.item():.4f} | "
                  f"zero={zero_rate:.1%} [{phase}] lr={lr:.2e}")

        # --- Evaluate ---
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                eval_train = evaluate_anchors(model, sup_train_t, sup_train_tgt, sup_train_words)
                eval_hold = evaluate_anchors(model, sup_hold_t, sup_hold_tgt, sup_hold_words)
            model.train()

            train_acc = eval_train.get('mean_bit_accuracy', 0)
            hold_acc = eval_hold.get('mean_bit_accuracy', 0)
            dead = eval_train.get('dead_bits', N_BITS)

            zero_rate, tern_dist = compute_ternary_stats(model, sup_train_t)

            print(f"  --- Eval @ step {step} ---")
            print(f"  Bit accuracy:  train={train_acc:.1%}  holdout={hold_acc:.1%}")
            print(f"  Dead bits: {dead}/{N_BITS}")
            print(f"  Ternary: -1={tern_dist['neg']:.1%}  0={tern_dist['zero']:.1%}  +1={tern_dist['pos']:.1%}")

            # Subsumption evaluation
            sub_train_rate, _ = evaluate_subsumption(
                model, sub_train_h, sub_train_y, len(train_sub))
            sub_test_rate, _ = evaluate_subsumption(
                model, sub_test_h, sub_test_y, len(test_sub))
            print(f"  Subsumption: train={sub_train_rate:.1%}  test={sub_test_rate:.1%}")

            # R3 evaluation at final step
            if step == args.steps:
                r3_results = _eval_r3_gpt2(model, tokenizer, train_anchors,
                                           holdout_anchors, device)
                if r3_results:
                    n_correct = sum(1 for r in r3_results if r['bit_accuracy'] > 0.8)
                    print(f"  Regla de Tres: {n_correct}/{len(r3_results)} "
                          f"({n_correct/len(r3_results):.1%} at >80% bit match)")

            csv_writer.writerow([
                step, total.item(),
                lang_loss.item() if lang_loss is not None else 0,
                l_tri.item(), l_sup.item(), l_sub.item(), l_sparse.item(),
                train_acc, hold_acc, dead,
                zero_rate, tern_dist['neg'], tern_dist['zero'], tern_dist['pos'],
                sub_train_rate, sub_test_rate,
            ])
            csv_file.flush()

            # Save best by subsumption test (primary metric for D-A19)
            if sub_test_rate > best_sub_test or (sub_test_rate == best_sub_test and hold_acc > best_hold_acc):
                best_sub_test = sub_test_rate
                best_hold_acc = max(best_hold_acc, hold_acc)
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'n_triadic_bits': N_BITS,
                    'quantize_mode': args.quantize_mode,
                    'backbone': 'gpt2-medium',
                    'bit_accuracy_train': train_acc,
                    'bit_accuracy_holdout': hold_acc,
                    'sub_train': sub_train_rate,
                    'sub_test': sub_test_rate,
                    'zero_rate': zero_rate,
                    'ternary_dist': tern_dist,
                    'dead_bits': dead,
                    'args': vars(args),
                }, os.path.join(ckpt_dir, 'model_best.pt'))
                print(f"  >> NEW BEST: sub_test={sub_test_rate:.1%} (hold_acc={hold_acc:.1%})")

        # Periodic checkpoint
        if step % args.save_every == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'n_triadic_bits': N_BITS,
                'quantize_mode': args.quantize_mode,
                'backbone': 'gpt2-medium',
            }, os.path.join(ckpt_dir, f'model_step{step}.pt'))

    csv_file.close()
    elapsed = time.time() - t0
    print(f"\n  Training complete: {elapsed/60:.1f} min")
    print(f"  Best: sub_test={best_sub_test:.1%}, hold_acc={best_hold_acc:.1%}")

    return model, tokenizer, device


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='D-A19: GPT-2 Medium 355M — Fix Scale-Algebra Tradeoff')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (16 for Medium on 16GB VRAM)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--sub-weight', type=float, default=5.0)
    parser.add_argument('--sup-weight', type=float, default=2.0)
    parser.add_argument('--align-weight', type=float, default=3.0)
    parser.add_argument('--sparsity-weight', type=float, default=2.0,
                        help='Weight for sparsity target loss (NEW in D-A19)')
    parser.add_argument('--sparsity-target', type=float, default=0.42,
                        help='Target zero rate (0.42 = D-A14 natural rate)')
    parser.add_argument('--triadic-warmup-pct', type=float, default=0.0,
                        help='Fraction of steps for language-only warmup (0 = active from start)')
    parser.add_argument('--unfreeze-pct', type=float, default=0.10,
                        help='Fraction of steps before unfreezing backbone (0.10 = step 5K)')
    parser.add_argument('--unfreeze-layers', type=int, default=4,
                        help='Number of GPT-2 layers to unfreeze in Phase 2')
    parser.add_argument('--stories', type=int, default=50000)
    parser.add_argument('--block', type=int, default=256,
                        help='Block size (max 1024 for GPT-2)')
    parser.add_argument('--quantize-mode', choices=['fsq', 'absmean'], default='fsq')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--v2', action='store_true',
                        help='Use expanded anchors (v1+v2 = 158 concepts)')
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=10000)
    parser.add_argument('--eval-every', type=int, default=2500)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in ckpt_dir')
    args = parser.parse_args()

    # Load primitives and anchors
    prim_data = load_primitives()
    if args.v2:
        all_anchors, skipped = load_all_anchors(prim_data)
        print(f"  Anchors: {len(all_anchors)} (v1+v2 merged)")
    else:
        all_anchors, skipped = load_anchors(prim_data)
        print(f"  Anchors: {len(all_anchors)} (v1 only)")

    # Split 80/20 for v2
    if args.v2:
        import random as _rng
        _rng.seed(42)
        words = list(all_anchors.keys())
        _rng.shuffle(words)
        n_test = max(1, int(len(words) * 0.2))
        holdout_words = set(words[:n_test])
        train_anchors = {w: all_anchors[w] for w in words if w not in holdout_words}
        holdout_anchors = {w: all_anchors[w] for w in holdout_words}
    else:
        train_anchors, holdout_anchors = get_split(all_anchors)

    suffix = '_v2' if args.v2 else ''
    ckpt_dir = os.path.join(_PROJECT, 'checkpoints', f'danza_gpt2_355m_sparsity{suffix}')
    model, tokenizer, device = run_training(
        args, train_anchors, holdout_anchors, prim_data, ckpt_dir)

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  D-A19 COMPLETE — Results in {ckpt_dir}")
    print(f"{'=' * 70}")
    print(f"\n  Key changes from D-A17:")
    print(f"  [FIX #1] Full 4-component triadic loss (was alignment-only)")
    print(f"  [FIX #2] Differentiable subsumption ((x+1)/2 not (x>0).float())")
    print(f"  [NEW]    Sparsity target loss (target={args.sparsity_target:.0%})")
    print(f"  [NEW]    Earlier unfreeze ({args.unfreeze_pct:.0%} not 50%)")
    print(f"  [NEW]    Triadic warmup={args.triadic_warmup_pct:.0%} (was 50%)")


if __name__ == '__main__':
    main()

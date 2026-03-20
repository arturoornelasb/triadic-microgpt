"""
D-A13: GPT-2 Medium + Ternary Triadic Head — Scaling Test.

Tests whether the ternary triadic head (proven on 40M from-scratch in D-A8)
works at scale with a pre-trained 355M backbone. Uses GPT-2 Medium from
HuggingFace with iFSQ ternary quantization.

Key hypothesis:
    Larger backbones produce richer hidden states, giving the triadic head
    more semantic signal to quantize. If D-A8 works at 40M, it should work
    better at 355M.

Architecture:
    GPT-2 Medium (355M, frozen then partially unfrozen)
      -> LM Head (GPT-2's original, weight-tied)
      -> Ternary Triadic Head (new, 63 trits, iFSQ activation)

Two-phase training:
    Phase 1 (0 to triadic_warmup): backbone frozen, language-only
    Phase 2 (triadic_warmup to end): unfreeze last 4 layers + ln_f,
            activate triadic losses (anchor supervision + subsumption)

Usage:
    python playground/gpt2_medium_ternary.py --steps 50000
    python playground/gpt2_medium_ternary.py --steps 50000 --unfreeze-layers 4
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
    supervised_anchor_loss, subsumption_loss,
    evaluate_anchors, evaluate_subsumption,
    evaluate_regla_de_tres, REGLA_DE_TRES_QUADS, TextDataset,
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


# ============================================================
# GPT-2 Medium + Ternary Triadic Head
# ============================================================

class GPT2MediumTernary(nn.Module):
    """Pre-trained GPT-2 Medium with iFSQ ternary triadic head.

    Same pattern as experiment10's GPT2TriadicModel but with:
    - ternary quantization (iFSQ) instead of tanh
    - Compatible interface with danza evaluation functions
    """

    def __init__(self, gpt2_model, n_triadic_bits=63, quantize_mode='fsq'):
        super().__init__()
        self.gpt2 = gpt2_model
        self.n_embd = gpt2_model.config.n_embd
        self.n_triadic_bits = n_triadic_bits
        self.block_size = gpt2_model.config.n_positions  # 1024
        self.quantize_mode = quantize_mode

        # config-like attribute for compatibility with evaluation functions
        self.config = type('Config', (), {
            'n_triadic_bits': n_triadic_bits,
            'n_embd': self.n_embd,
            'block_size': self.block_size,
        })()

        # Triadic projection head (new, random init)
        self.triadic_head = nn.Linear(self.n_embd, n_triadic_bits, bias=False)
        nn.init.normal_(self.triadic_head.weight, mean=0.0, std=0.02)

        # Expose wte for alignment loss compatibility
        self.wte = gpt2_model.transformer.wte

    def freeze_backbone(self):
        """Freeze all GPT-2 parameters. Only triadic head is trainable."""
        for param in self.gpt2.parameters():
            param.requires_grad = False
        for param in self.triadic_head.parameters():
            param.requires_grad = True

    def unfreeze_last_n(self, n=4):
        """Unfreeze the last N transformer blocks + final layer norm."""
        for param in self.gpt2.transformer.ln_f.parameters():
            param.requires_grad = True
        total = len(self.gpt2.transformer.h)
        for i in range(max(0, total - n), total):
            for param in self.gpt2.transformer.h[i].parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Unfroze last {n} layers + ln_f: {trainable/1e6:.1f}M trainable params")

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, targets=None):
        """Forward pass compatible with danza evaluation interface.

        Returns: (logits, triadic_proj, loss)
        """
        outputs = self.gpt2.transformer(input_ids=input_ids)
        hidden_states = outputs.last_hidden_state  # (B, T, n_embd)

        # LM head
        logits = self.gpt2.lm_head(hidden_states)

        # Ternary triadic head
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

        return logits, triadic_proj, loss


# ============================================================
# Alignment loss for GPT-2 embeddings
# ============================================================

def gpt2_triadic_loss(proj, align_weight, wte, input_ids):
    """Embedding alignment loss using GPT-2's own embeddings.

    Same logic as danza_63bit.triadic_loss but adapted for GPT-2:
    uses wte (GPT-2's token embedding table) for alignment.
    """
    B, T, K = proj.shape
    if T < 2:
        return torch.tensor(0.0, device=proj.device)

    # Embedding alignment (InfoNCE-lite via MSE on cosine similarities)
    embeds = wte(input_ids)  # (B, T, n_embd)

    # Sample pairs
    n_pairs = min(32, T)
    idx_a = torch.randint(0, T, (B, n_pairs), device=proj.device)
    idx_b = torch.randint(0, T, (B, n_pairs), device=proj.device)

    # Embedding similarities
    emb_a = torch.gather(embeds, 1, idx_a.unsqueeze(-1).expand(-1, -1, embeds.size(-1)))
    emb_b = torch.gather(embeds, 1, idx_b.unsqueeze(-1).expand(-1, -1, embeds.size(-1)))
    emb_sim = F.cosine_similarity(emb_a, emb_b, dim=-1)

    # Triadic similarities
    proj_a = torch.gather(proj, 1, idx_a.unsqueeze(-1).expand(-1, -1, K))
    proj_b = torch.gather(proj, 1, idx_b.unsqueeze(-1).expand(-1, -1, K))
    proj_sim = F.cosine_similarity(proj_a, proj_b, dim=-1)

    align_loss = F.mse_loss(proj_sim, emb_sim.detach())

    return align_weight * align_loss


# ============================================================
# GPT-2 specific supervised anchor loss
# ============================================================

def gpt2_supervised_anchor_loss(model, word_tensors, targets):
    """Compute supervised anchor loss for GPT-2 model."""
    if word_tensors.shape[0] == 0:
        return torch.tensor(0.0, device=word_tensors.device)

    _, proj, _ = model(word_tensors)
    pred = proj.mean(dim=1)  # (N, 63)
    return F.mse_loss(pred, targets)


def gpt2_subsumption_loss(model, h_tensors, y_tensors):
    """Compute subsumption loss for GPT-2 model."""
    if h_tensors.shape[0] == 0:
        return torch.tensor(0.0, device=h_tensors.device)

    _, proj_h, _ = model(h_tensors)
    _, proj_y, _ = model(y_tensors)
    h_bits = proj_h.mean(dim=1)  # (N, 63)
    y_bits = proj_y.mean(dim=1)

    # Subsumption: hypernym bits should be subset of hyponym bits
    # Penalize when hypernym is more active than hyponym
    h_active = (h_bits > 0).float()
    y_active = (y_bits > 0).float()
    violation = F.relu(h_active - y_active)  # 1 where h=1 but y=0
    return violation.mean()


# ============================================================
# Training
# ============================================================

def run_training(args, train_anchors, holdout_anchors, prim_data, ckpt_dir):
    """Train GPT-2 Medium + Ternary Triadic Head."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  D-A13: GPT-2 MEDIUM + TERNARY TRIADIC HEAD")
    print(f"{'=' * 70}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --- Load GPT-2 Medium ---
    print(f"\n  Loading GPT-2 Medium from HuggingFace...")
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2MediumTernary(
        gpt2, n_triadic_bits=N_BITS, quantize_mode=args.quantize_mode,
    ).to(device)

    total_params = model.num_params()
    triadic_params = sum(p.numel() for p in model.triadic_head.parameters())
    print(f"  Model: GPT2MediumTernary ({total_params/1e6:.1f}M total, "
          f"{triadic_params/1e3:.1f}K triadic head, {N_BITS} ternary trits)")

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
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16  # always bfloat16 for Blackwell
    print(f"  Mixed precision: bfloat16")

    # --- Training loop ---
    block_size = min(args.block, 1024)  # GPT-2 max is 1024
    dataset = TextDataset(all_tokens, block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, pin_memory=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    warmup_steps = int(args.steps * 0.05)
    # GPT-2 is already pretrained — no language warmup needed.
    # Triadic losses active from step 1. Backbone unfreezes later.
    triadic_start = 1
    unfreeze_step = int(args.steps * args.triadic_warmup_pct)

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
    best_hold_acc = 0.0
    unfrozen = False

    print(f"\n  Training ({args.steps} steps)...")
    print(f"  Phase 1: steps 1-{unfreeze_step} (backbone frozen, triadic head trains)")
    print(f"  Phase 2: steps {unfreeze_step}-{args.steps} (unfreeze last {args.unfreeze_layers} layers)")

    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        # Phase 2: unfreeze backbone layers
        if step == unfreeze_step and not unfrozen:
            model.unfreeze_last_n(args.unfreeze_layers)
            # Rebuild optimizer with new trainable params
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
            unfrozen = True

        # LR schedule
        if step <= warmup_steps:
            lr = args.lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (args.steps - warmup_steps)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward + backward
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            logits, proj, lang_loss = model(x, y)
            l_tri = gpt2_triadic_loss(proj, args.align_weight, model.wte, x)
            l_sup = gpt2_supervised_anchor_loss(model, sup_train_t, sup_train_tgt)
            l_sub = gpt2_subsumption_loss(model, sub_train_h, sub_train_y)
            triadic_total = args.alpha * (
                l_tri + args.sup_weight * l_sup + args.sub_weight * l_sub)
            # In Phase 1 (frozen backbone), lang_loss has no grad —
            # only include it after unfreezing
            if unfrozen and lang_loss is not None:
                total = lang_loss + triadic_total
            else:
                total = triadic_total

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # --- Print ---
        if step % args.print_every == 0:
            elapsed = time.time() - t0
            bar = format_progress_bar(step, args.steps)
            eta = format_eta(elapsed, step, args.steps)

            zero_rate, tern_dist = compute_batch_ternary_stats(proj.detach())

            tri_str = (f"tri={l_tri.item():.4f} sup={l_sup.item():.4f} "
                       f"sub={l_sub.item():.4f}")
            phase = "P1-frozen" if not unfrozen else "P2-unfrozen"
            lang_str = f"lang={lang_loss.item():.4f}" if lang_loss is not None else "lang=N/A"

            print(f"  {bar} [{step:>6d}/{args.steps}] ETA {eta} | "
                  f"loss={total.item():.4f} {lang_str} {tri_str} | "
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
            if True:  # triadic always active for GPT-2 transfer
                sub_train_rate, _ = evaluate_subsumption(
                    model, sub_train_h, sub_train_y, len(train_sub))
                sub_test_rate, _ = evaluate_subsumption(
                    model, sub_test_h, sub_test_y, len(test_sub))
                print(f"  Subsumption: train={sub_train_rate:.1%}  "
                      f"test={sub_test_rate:.1%}")
            else:
                sub_train_rate = 0
                sub_test_rate = 0

            csv_writer.writerow([
                step, total.item(), lang_loss.item(),
                l_tri.item() if step >= triadic_start else 0,
                l_sup.item() if step >= triadic_start else 0,
                l_sub.item() if step >= triadic_start else 0,
                train_acc, hold_acc, dead,
                zero_rate, tern_dist['neg'], tern_dist['zero'], tern_dist['pos'],
            ])
            csv_file.flush()

            # Save best by holdout accuracy
            if hold_acc > best_hold_acc:
                best_hold_acc = hold_acc
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
                }, os.path.join(ckpt_dir, 'model_best.pt'))

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
    print(f"  Best holdout acc: {best_hold_acc:.1%}")

    return model, tokenizer, device


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='D-A13: GPT-2 Medium + Ternary Triadic Head')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (16 for Medium on 16GB VRAM)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--sub-weight', type=float, default=5.0)
    parser.add_argument('--sup-weight', type=float, default=2.0)
    parser.add_argument('--align-weight', type=float, default=3.0)
    parser.add_argument('--triadic-warmup-pct', type=float, default=0.5,
                        help='Fraction of steps for language-only warmup')
    parser.add_argument('--unfreeze-layers', type=int, default=4,
                        help='Number of GPT-2 layers to unfreeze in Phase 2')
    parser.add_argument('--stories', type=int, default=50000)
    parser.add_argument('--block', type=int, default=256,
                        help='Block size (max 1024 for GPT-2)')
    parser.add_argument('--quantize-mode', choices=['fsq', 'absmean'], default='fsq')
    parser.add_argument('--v2', action='store_true',
                        help='Use expanded anchors (v1+v2 = 158 concepts)')
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=10000)
    parser.add_argument('--eval-every', type=int, default=2500)
    args = parser.parse_args()

    # Load primitives and anchors
    prim_data = load_primitives()
    if args.v2:
        all_anchors, skipped = load_all_anchors(prim_data)
        print(f"  Anchors: {len(all_anchors)} (v1+v2 merged)")
    else:
        all_anchors, skipped = load_anchors(prim_data)
        print(f"  Anchors: {len(all_anchors)} (v1 only)")

    # Split 80/20 for v2 (get_split uses TRAIN_CONCEPTS which is v1-only)
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
    ckpt_dir = os.path.join(_PROJECT, 'checkpoints', f'danza_gpt2medium_ternary{suffix}')
    model, tokenizer, device = run_training(
        args, train_anchors, holdout_anchors, prim_data, ckpt_dir)

    # Final evaluation summary
    print(f"\n{'=' * 70}")
    print(f"  D-A13 COMPLETE — Results in {ckpt_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

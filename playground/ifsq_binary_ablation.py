"""
D-A10: iFSQ Binary Ablation — isolate activation function vs ternary quantization.

Central question: does the iFSQ activation (2*sigmoid(1.6*x) - 1) alone fix dead
bits, WITHOUT switching to ternary {-1, 0, +1}?  If yes → the activation is the
key fix, not the quantization scheme.  If no → ternary quantization itself matters.

This is the SAME training pipeline as danza_bootstrap.py (D-A5) with ONE change:
  - D-A5: triadic_proj = tanh(triadic_head(x))          → binary {0,1} via threshold
  - D-A10: triadic_proj = 2*sigmoid(1.6*x) - 1          → binary {0,1} via threshold
  - D-A8: triadic_proj = ternary_quantize(triadic_head)  → ternary {-1,0,+1}

Comparison matrix:
  D-A5  (tanh + binary)   = baseline
  D-A10 (iFSQ + binary)   = activation ablation  ← THIS SCRIPT
  D-A8  (iFSQ + ternary)  = full ternary

Usage:
  python playground/ifsq_binary_ablation.py --phase all --scale xl --steps 50000
"""

import os
import sys
import csv
import json
import math
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
from danza_bootstrap import (
    TRAIN_CONCEPTS, HOLDOUT_INFO, BOOTSTRAP_QUADS,
    get_split, build_partial_subsumption_pairs, phase_split, phase_predict,
)
from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer


# ============================================================
# The ONE change: iFSQ activation instead of tanh
# ============================================================

class iFSQDanzaGPT(TriadicGPT):
    """DanzaTriadicGPT with iFSQ activation instead of tanh.

    The ONLY difference from DanzaTriadicGPT (danza_63bit.py):
      DanzaTriadicGPT:  triadic_proj = torch.tanh(self.triadic_head(x))
      iFSQDanzaGPT:     triadic_proj = 2 * torch.sigmoid(1.6 * self.triadic_head(x)) - 1

    iFSQ (Tencent, 2025) showed tanh concentrates activations near 0,
    causing dead bits. sigmoid(1.6*x) has a flatter middle region,
    distributing activations more uniformly across the [-1, +1] range.

    Output is STILL continuous [-1, +1], thresholded to binary {0, 1}
    at inference time via (proj > 0). NOT ternary.
    """

    def gradient_checkpointing_enable(self):
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

        # --- THE KEY CHANGE: iFSQ activation instead of tanh ---
        triadic_proj = 2 * torch.sigmoid(1.6 * self.triadic_head(x)) - 1

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, triadic_proj, loss


# ============================================================
# Training — identical to danza_bootstrap except model class
# ============================================================

def run_training(args, train_anchors, holdout_anchors, prim_data, ckpt_dir):
    """Train iFSQDanzaGPT with partial anchor supervision."""
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
    print(f"  D-A10 iFSQ BINARY ABLATION — Training")
    print(f"{'=' * 70}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Activation: 2*sigmoid(1.6*x) - 1 (iFSQ, NOT tanh)")
    print(f"  Output: binary {{0,1}} via threshold (NOT ternary)")
    print(f"  Train anchors: {len(train_anchors)} words")
    print(f"  Holdout anchors: {len(holdout_anchors)} words (eval only)")

    # --- Subsumption pairs ---
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

    # --- Model (iFSQ, NOT tanh) ---
    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block,
        n_layer=preset['layers'],
        n_embd=preset['dim'],
        n_head=preset['heads'],
        n_triadic_bits=N_BITS,
        dropout=args.dropout,
    )
    model = iFSQDanzaGPT(config).to(device)
    total_params = model.num_params()
    print(f"  Model: iFSQDanzaGPT {args.scale} ({total_params/1e6:.1f}M params, {N_BITS} bits)")

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
    print(f"  Supervision: {len(sup_train_words)} train anchors")

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
            eta_s = elapsed / step * (args.steps - step)
            eta = f"{int(eta_s // 3600)}h{int((eta_s % 3600) // 60):02d}m" if eta_s > 3600 else f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
            pct = step / args.steps
            bar = f"{'█' * int(pct * 20)}{'░' * (20 - int(pct * 20))}"
            tri_str = (f"tri={l_tri.item():.4f} sup={l_sup.item():.4f} sub={l_sub.item():.4f}"
                       if step >= triadic_start else "warmup")
            print(f"  {bar} [{step:>6d}/{args.steps}] ETA {eta} | "
                  f"loss={total.item():.4f} lang={lang_loss.item():.4f} "
                  f"{tri_str} lr={lr:.2e}")

        # Evaluate
        if step % args.eval_every == 0 or step == args.steps:
            eval_train = evaluate_anchors(model, sup_train_t, sup_train_tgt, sup_train_words)
            eval_hold = evaluate_anchors(model, sup_hold_t, sup_hold_tgt, sup_hold_words)

            train_acc = eval_train.get('mean_bit_accuracy', 0)
            hold_acc = eval_hold.get('mean_bit_accuracy', 0)
            dead = eval_train.get('dead_bits', N_BITS)

            print(f"  --- Eval @ step {step} ---")
            print(f"  Bit accuracy:  train={train_acc:.1%}  holdout={hold_acc:.1%}")
            print(f"  Dead bits: {dead}/{N_BITS}  (D-A5 baseline: 30)")

            csv_writer.writerow([
                step, total.item(), lang_loss.item(),
                l_tri.item() if step >= triadic_start else 0,
                l_sup.item() if step >= triadic_start else 0,
                l_sub.item() if step >= triadic_start else 0,
                train_acc, hold_acc, dead,
            ])
            csv_file.flush()

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
                    'activation': 'ifsq',
                    'bit_accuracy_train': train_acc,
                    'bit_accuracy_holdout': hold_acc,
                }, os.path.join(ckpt_dir, 'model_best.pt'))

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
                    'activation': 'ifsq',
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
                'activation': 'ifsq',
            }, os.path.join(ckpt_dir, f'model_step{step}.pt'))

    csv_file.close()
    elapsed = time.time() - t0
    print(f"\n  Training complete: {elapsed/60:.1f} min")
    print(f"  Best train acc: {best_train_acc:.1%}, best holdout: {best_hold_acc:.1%}")

    return model, tokenizer, device


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='D-A10: iFSQ Binary Ablation (sigmoid activation, binary output)')
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
    parser.add_argument('--grad-checkpoint', action='store_true')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'],
                        default='bfloat16')
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--eval-every', type=int, default=2500)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    # Load data
    prim_data = load_primitives()
    all_anchors, skipped = load_anchors(prim_data)

    if args.phase == 'split':
        phase_split(all_anchors, prim_data)
        return

    if args.phase == 'train' or args.phase == 'all':
        train_anchors, holdout_anchors = get_split(all_anchors)
        ckpt_dir = os.path.join(_PROJECT, 'checkpoints', f'danza_ifsq_binary_{args.scale}')
        model, tokenizer, device = run_training(
            args, train_anchors, holdout_anchors, prim_data, ckpt_dir)

        if args.phase == 'all':
            results, direct, r3_preds, ensemble = phase_predict(
                model, tokenizer, train_anchors, holdout_anchors, all_anchors, device)

            results_path = os.path.join(ckpt_dir, 'ifsq_binary_results.json')
            serializable = {}
            for sp, r in results.items():
                serializable[sp] = {k: v for k, v in r.items()
                                     if not isinstance(v, torch.Tensor)}
            with open(results_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            print(f"\n  Results: {results_path}")
        return

    if args.phase == 'predict':
        qmode = 'ifsq'
        ckpt_dir = args.checkpoint or os.path.join(
            _PROJECT, 'checkpoints', f'danza_ifsq_binary_{args.scale}')

        import glob as glob_mod
        step_ckpts = sorted(glob_mod.glob(os.path.join(ckpt_dir, 'model_step*.pt')))
        if step_ckpts:
            ckpt_path = step_ckpts[-1]
            print(f"  Using latest checkpoint: {os.path.basename(ckpt_path)}")
        else:
            ckpt_path = os.path.join(ckpt_dir, 'model_best.pt')
            print(f"  Using model_best.pt")

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
        model = iFSQDanzaGPT(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        tok_path = os.path.join(ckpt_dir, 'tokenizer.json')
        tokenizer = BPETokenizer.load(tok_path)

        train_anchors, holdout_anchors = get_split(all_anchors)
        results, direct, r3_preds, ensemble = phase_predict(
            model, tokenizer, train_anchors, holdout_anchors, all_anchors, device)

        results_path = os.path.join(ckpt_dir, 'ifsq_binary_results.json')
        serializable = {}
        for sp, r in results.items():
            serializable[sp] = {k: v for k, v in r.items()
                                 if not isinstance(v, torch.Tensor)}
        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results: {results_path}")


if __name__ == '__main__':
    main()

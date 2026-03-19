"""
D-A9: Hybrid Bits + Adversarial Disentanglement.

30 supervised bits (gold labels from anclas.json) + 33 free bits (contrastive only).
Adversarial discriminator prevents backbone from bypassing triadic head.

Based on CB-LLMs (Sun et al., ICLR 2025) adversarial disentanglement approach.

Loss = L_lang + alpha * (L_tri + sup_weight * L_sup + sub_weight * L_sub
                          + adv_weight * L_adv)

Usage:
  python playground/hybrid_adversarial.py --scale xl --steps 50000  # ~4.5h
  python playground/hybrid_adversarial.py --scale base --steps 1000  # smoke test
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
from torch.autograd import Function

_PLAYGROUND = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

from danza_63bit import (
    load_primitives, load_anchors, build_subsumption_pairs,
    DanzaTriadicGPT, subsumption_loss,
    triadic_loss, evaluate_anchors, evaluate_subsumption,
    evaluate_regla_de_tres, REGLA_DE_TRES_QUADS, TextDataset,
    ANCHOR_TRANSLATIONS, SKIP_ANCHORS, N_BITS, STORY_SEPARATOR,
)
from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer


N_SUPERVISED = 30
N_FREE = N_BITS - N_SUPERVISED  # 33


# ============================================================
# Gradient Reversal Layer (CB-LLM technique)
# ============================================================

class GradientReversal(Function):
    """Flip gradients during backward pass."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


# ============================================================
# Hybrid Triadic GPT — split supervised + free bits
# ============================================================

class HybridTriadicGPT(DanzaTriadicGPT):
    """DanzaTriadicGPT with split supervised/free bits and adversarial head."""

    def __init__(self, config, n_supervised=N_SUPERVISED, n_free=N_FREE):
        super().__init__(config)
        self.n_supervised = n_supervised
        self.n_free = n_free

        d = config.n_embd

        # Replace single triadic head with two separate projections
        del self.triadic_head
        self.sup_head = nn.Linear(d, n_supervised)
        self.free_head = nn.Linear(d, n_free)

        # Adversarial discriminator: predicts supervised bits from backbone hidden
        # Trained with gradient reversal so backbone learns to confound it
        self.adversary = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, n_supervised),
        )

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

        # Supervised + free projections
        sup_proj = torch.tanh(self.sup_head(x))     # (B, T, 30)
        free_proj = torch.tanh(self.free_head(x))    # (B, T, 33)
        triadic_proj = torch.cat([sup_proj, free_proj], dim=-1)  # (B, T, 63)

        # Adversary: predict supervised bits from backbone (with grad reversal)
        adv_logits = self.adversary(grad_reverse(x))  # (B, T, 30)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, triadic_proj, loss, sup_proj, adv_logits

    @property
    def triadic_head(self):
        """Compatibility: returns None (split into sup_head + free_head)."""
        return None


# ============================================================
# Loss functions
# ============================================================

def supervised_split_loss(model, word_tensors, target_vectors, n_sample=32):
    """MSE on supervised bits ONLY (first 30)."""
    N = word_tensors.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=word_tensors.device)

    if N > n_sample:
        idx = torch.randperm(N, device=word_tensors.device)[:n_sample]
        w_batch = word_tensors[idx]
        t_batch = target_vectors[idx]
    else:
        w_batch = word_tensors
        t_batch = target_vectors

    _, _, _, sup_proj, _ = model(w_batch)
    pred = sup_proj.mean(dim=1)  # (n, 30)

    # Only first 30 bits of target
    return F.mse_loss(pred, t_batch[:, :N_SUPERVISED])


def adversarial_loss(adv_logits, sup_proj):
    """Adversary tries to predict supervised bit signs from backbone hidden state.

    adv_logits: (B, T, 30) from adversary applied to grad_reverse(hidden)
    sup_proj:   (B, T, 30) ground truth supervised projections (detached)

    Returns BCE loss. The gradient reversal in the model handles the minimax.
    """
    targets = (sup_proj.detach() > 0).float()  # binary targets from current projections
    return F.binary_cross_entropy_with_logits(adv_logits, targets)


def hybrid_subsumption_loss(model, hyper_t, hypo_t, n_sample=32):
    """Subsumption on full 63 bits (supervised + free)."""
    N = hyper_t.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=hyper_t.device)

    if N > n_sample:
        idx = torch.randperm(N, device=hyper_t.device)[:n_sample]
        h_batch, y_batch = hyper_t[idx], hypo_t[idx]
    else:
        h_batch, y_batch = hyper_t, hypo_t

    _, h_proj, _, _, _ = model(h_batch)
    _, y_proj, _, _, _ = model(y_batch)

    h_01 = (h_proj.mean(dim=1) + 1) / 2
    y_01 = (y_proj.mean(dim=1) + 1) / 2

    return F.relu(h_01 - y_01).mean()


def hybrid_triadic_loss(proj, align_weight, wte, input_ids):
    """Standard triadic loss on full 63-bit concatenated projection."""
    return triadic_loss(proj, align_weight, wte, input_ids)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_hybrid(model, word_tensors, target_vectors, valid_words):
    """Evaluate supervised + free bits separately."""
    model.eval()
    N = word_tensors.shape[0]
    if N == 0:
        model.train()
        return {}

    _, _, _, sup_proj, adv_logits = model(word_tensors)
    _, full_proj, _, _, _ = model(word_tensors)
    pred_sup = sup_proj.mean(dim=1)     # (N, 30)
    pred_full = full_proj.mean(dim=1)   # (N, 63)

    # Supervised bit accuracy (first 30)
    sup_targets = target_vectors[:, :N_SUPERVISED]
    sup_bits_pred = (pred_sup > 0).float()
    sup_bits_gold = (sup_targets > 0).float()
    sup_acc = (sup_bits_pred == sup_bits_gold).float().mean().item()

    # Full bit accuracy
    full_bits_pred = (pred_full > 0).float()
    full_bits_gold = (target_vectors > 0).float()
    full_acc = (full_bits_pred == full_bits_gold).float().mean().item()

    # Free bits entropy (last 33)
    free_proj = pred_full[:, N_SUPERVISED:]  # (N, 33)
    free_means = free_proj.mean(dim=0)
    q = (free_means + 1) / 2
    eps = 1e-7
    free_ent = -(q * torch.log2(q + eps) + (1 - q) * torch.log2(1 - q + eps))
    free_dead = int((free_ent < 0.3).sum())

    # Adversary accuracy (should converge to ~50% = random)
    adv_pred = (adv_logits.mean(dim=1) > 0).float()  # (N, 30)
    adv_acc = (adv_pred == sup_bits_gold).float().mean().item()

    model.train()
    return {
        'sup_bit_accuracy': sup_acc,
        'full_bit_accuracy': full_acc,
        'free_dead_bits': free_dead,
        'free_mean_entropy': float(free_ent.mean()),
        'adversary_accuracy': adv_acc,
        'n_concepts': N,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='D-A9: Hybrid Bits + Adversarial')
    parser.add_argument('--scale', choices=['base', 'xl', 'xxl'], default='base')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--sub-weight', type=float, default=5.0)
    parser.add_argument('--sup-weight', type=float, default=2.0)
    parser.add_argument('--adv-weight', type=float, default=1.0)
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
    args = parser.parse_args()

    SCALES = {
        'base': {'layers': 6,  'dim': 256,  'heads': 8},
        'xl':   {'layers': 12, 'dim': 512,  'heads': 8},
        'xxl':  {'layers': 24, 'dim': 1024, 'heads': 16},
    }
    preset = SCALES[args.scale]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = os.path.join(_PROJECT, 'checkpoints', f'danza_hybrid_adv_{args.scale}')
    os.makedirs(ckpt_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("  D-A9: HYBRID BITS + ADVERSARIAL DISENTANGLEMENT")
    print("=" * 70)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    # --- Load data ---
    print(f"\n[1/6] Loading primitives and anchors...")
    prim_data = load_primitives()
    anchors, skipped = load_anchors(prim_data)
    print(f"  Anchors: {len(anchors)} words, {N_SUPERVISED} supervised + {N_FREE} free bits")

    print(f"\n[2/6] Building subsumption pairs...")
    train_sub, test_sub = build_subsumption_pairs(anchors, prim_data)
    print(f"  Subsumption: train={len(train_sub)}, test={len(test_sub)}")

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
    print(f"\n[3/6] Training BPE tokenizer (vocab={args.vocab})...")
    tokenizer = BPETokenizer(vocab_size=args.vocab)
    tokenizer.train(stories, verbose=False)
    tokenizer.save(tok_path)

    print(f"\n[4/6] Tokenizing {len(stories)} stories...")
    all_tokens = []
    for story in stories:
        all_tokens.extend(tokenizer.encode(story, add_special=True))
    print(f"  Total: {len(all_tokens):,} tokens")

    # --- Model ---
    print(f"\n[5/6] Initializing HybridTriadicGPT...")
    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block,
        n_layer=preset['layers'],
        n_embd=preset['dim'],
        n_head=preset['heads'],
        n_triadic_bits=N_BITS,
        dropout=args.dropout,
    )
    model = HybridTriadicGPT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Scale: {args.scale} ({preset['layers']}L/{preset['dim']}D/{preset['heads']}H)")
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Supervised: {N_SUPERVISED} bits | Free: {N_FREE} bits | Adversary: MLP")

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    if device.type == 'cuda' and not args.no_compile:
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
    print(f"  Precision: {args.dtype}")

    # --- Pre-encode anchors ---
    word_list, ids_list, target_list = [], [], []
    for word, data in anchors.items():
        ids = tokenizer.encode(word, add_special=False)[:4]
        if ids:
            word_list.append(word)
            ids_list.append(ids)
            target_list.append(data['target'])

    max_len = max(len(x) for x in ids_list)
    pad_id = 0
    word_tensors = torch.tensor(
        [ids + [pad_id] * (max_len - len(ids)) for ids in ids_list],
        dtype=torch.long, device=device
    )
    target_vectors = torch.stack(target_list).to(device)

    # Subsumption tensors
    if train_sub:
        hyper_t = torch.tensor(
            [[tokenizer.encode(h, add_special=False)[0]] for h, _, _, _ in train_sub],
            dtype=torch.long, device=device
        )
        hypo_t = torch.tensor(
            [[tokenizer.encode(y, add_special=False)[0]] for _, y, _, _ in train_sub],
            dtype=torch.long, device=device
        )
    else:
        z = torch.zeros((0, 1), dtype=torch.long, device=device)
        hyper_t, hypo_t = z, z

    if test_sub:
        test_hyper = torch.tensor(
            [[tokenizer.encode(h, add_special=False)[0]] for h, _, _, _ in test_sub],
            dtype=torch.long, device=device
        )
        test_hypo = torch.tensor(
            [[tokenizer.encode(y, add_special=False)[0]] for _, y, _, _ in test_sub],
            dtype=torch.long, device=device
        )
    else:
        z = torch.zeros((0, 1), dtype=torch.long, device=device)
        test_hyper, test_hypo = z, z

    # --- Training ---
    print(f"\n[6/6] Training ({args.steps} steps)...")
    print(f"  Alpha: {args.alpha}, Sup: {args.sup_weight}, Sub: {args.sub_weight}, "
          f"Adv: {args.adv_weight}, Align: {args.align_weight}")

    dataset = TextDataset(all_tokens, args.block)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    loader_iter = iter(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup_steps = int(args.steps * args.triadic_warmup_pct)

    log_path = os.path.join(ckpt_dir, 'training_log.csv')
    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['step', 'total_loss', 'lang_loss', 'tri_loss', 'sup_loss',
                         'adv_loss', 'sub_loss', 'ppl', 'lr'])

    start_time = time.time()
    best_sup_acc = 0.0

    for step in range(1, args.steps + 1):
        # LR schedule: warmup + cosine
        if step <= warmup_steps:
            lr_mult = step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr * lr_mult

        triadic_active = step > warmup_steps

        # Get batch
        try:
            x_batch, y_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x_batch, y_batch = next(loader_iter)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            logits, triadic_proj, lang_loss, sup_proj, adv_logits = model(x_batch, targets=y_batch)

            if triadic_active:
                l_tri = hybrid_triadic_loss(triadic_proj, args.align_weight,
                                            model.wte, x_batch)
                l_sup = supervised_split_loss(model, word_tensors, target_vectors)
                l_sub = hybrid_subsumption_loss(model, hyper_t, hypo_t)
                l_adv = adversarial_loss(adv_logits, sup_proj)

                total_loss = (lang_loss
                              + args.alpha * l_tri
                              + args.alpha * args.sup_weight * l_sup
                              + args.alpha * args.sub_weight * l_sub
                              + args.alpha * args.adv_weight * l_adv)
            else:
                l_tri = l_sup = l_sub = l_adv = torch.tensor(0.0)
                total_loss = lang_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % args.print_every == 0 or step == 1:
            elapsed = time.time() - start_time
            speed = step / elapsed
            eta = (args.steps - step) / speed
            ppl = math.exp(min(lang_loss.item(), 20))
            pct = step / args.steps
            bar = '#' * int(30 * pct) + '-' * (30 - int(30 * pct))

            print(f"  [{bar}] {step:>6}/{args.steps} | "
                  f"loss {total_loss.item():.4f} | lang {lang_loss.item():.4f} | "
                  f"tri {l_tri.item():.3f} | sup {l_sup.item():.3f} | "
                  f"adv {l_adv.item():.3f} | "
                  f"ppl {ppl:.1f} | {speed:.1f} it/s | ETA {eta:.0f}s")

            log_writer.writerow([step, total_loss.item(), lang_loss.item(),
                                 l_tri.item(), l_sup.item(), l_adv.item(),
                                 l_sub.item(), ppl, args.lr * lr_mult])

        # Evaluation
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_hybrid(model, word_tensors, target_vectors, word_list)
            sub_train, _ = evaluate_subsumption(model, hyper_t, hypo_t, len(train_sub))
            sub_test, _ = evaluate_subsumption(model, test_hyper, test_hypo, len(test_sub))

            print(f"\n  --- Eval @ step {step} ---")
            print(f"  Supervised bit acc: {metrics['sup_bit_accuracy']:.1%}")
            print(f"  Full bit acc:       {metrics['full_bit_accuracy']:.1%}")
            print(f"  Free dead bits:     {metrics['free_dead_bits']}/{N_FREE}")
            print(f"  Free entropy:       {metrics['free_mean_entropy']:.3f}")
            print(f"  Adversary acc:      {metrics['adversary_accuracy']:.1%} (target: ~50%)")
            print(f"  Subsumption train:  {sub_train:.1%}")
            print(f"  Subsumption test:   {sub_test:.1%}")
            print()

            if metrics['sup_bit_accuracy'] > best_sup_acc:
                best_sup_acc = metrics['sup_bit_accuracy']
                save_path = os.path.join(ckpt_dir, f'model_best.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': step,
                    'metrics': metrics,
                    'config': vars(config),
                }, save_path)

        # Checkpoint
        if step % args.save_every == 0 or step == args.steps:
            save_path = os.path.join(ckpt_dir, f'model_step{step}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step,
                'config': vars(config),
            }, save_path)

    log_file.close()
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  D-A9 COMPLETE — {elapsed/60:.1f} min")
    print(f"  Best supervised acc: {best_sup_acc:.1%}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Log: {log_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

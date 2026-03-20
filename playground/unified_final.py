"""
D-A18: Unified Final Model — iFSQ + Hybrid Bits + v2 Anchors.

Combines ALL proven optimal components from the Danza experiment line:
  - iFSQ activation: 2*sigmoid(1.6x)-1 (best LM loss, from D-A16)
  - Hybrid bits: 30 supervised + 33 free (fewer dead bits, from D-A9)
  - v2 anchors: 158 hand-factorized concepts (best accuracy, from D-A14)
  - Adversarial discriminator: prevents backbone bypass (from D-A9)
  - BitwiseValidator: O(1) evaluation (isomorphic to PrimeMapper)

Loss = L_lang + alpha * (L_tri + sup_weight * L_sup + sub_weight * L_sub
                          + adv_weight * L_adv)

Evidence:
  D-A14 (v2 tanh):      93.0% test, 98.3% sub, 26 dead bits
  D-A16 (iFSQ v2):      93.2% test, LM 0.924, ~20 dead bits
  D-A9  (hybrid v1):    69.3% test, 13 dead bits, 57/63 active

Target: >=90% test, >=95% sub, <10 dead bits, >50 active bits.

Usage:
  python playground/unified_final.py --scale xl --steps 50000 --dtype bfloat16
  python playground/unified_final.py --scale base --steps 1000  # smoke test
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
    load_primitives, load_all_anchors, build_subsumption_pairs,
    triadic_loss, supervised_anchor_loss, subsumption_loss,
    evaluate_anchors, evaluate_subsumption, evaluate_regla_de_tres,
    REGLA_DE_TRES_QUADS, TextDataset,
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
# Gradient Reversal Layer (from D-A9 / CB-LLMs)
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
# Unified Model: iFSQ + Hybrid Bits + Adversarial
# ============================================================

class UnifiedTriadicGPT(TriadicGPT):
    """Combines iFSQ activation with hybrid supervised/free bit split.

    Architecture:
        backbone → ln_f(x) → {
            lm_head(x)      → next-token logits
            sup_head(x)      → iFSQ → 30 supervised bits
            free_head(x)     → iFSQ → 33 free bits
            adversary(GR(x)) → predict sup bits (gradient reversed)
        }

    iFSQ: 2*sigmoid(1.6*x) - 1  (better gradient flow than tanh)
    """

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

        # iFSQ activation: 2*sigmoid(1.6*x) - 1
        sup_proj = 2 * torch.sigmoid(1.6 * self.sup_head(x)) - 1   # (B, T, 30)
        free_proj = 2 * torch.sigmoid(1.6 * self.free_head(x)) - 1  # (B, T, 33)
        triadic_proj = torch.cat([sup_proj, free_proj], dim=-1)       # (B, T, 63)

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
# Loss functions (adapted for hybrid split)
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

    The gradient reversal in the model handles the minimax.
    """
    targets = (sup_proj.detach() > 0).float()
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


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_unified(model, word_tensors, target_vectors, valid_words):
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

    # Full bit accuracy (all 63, gold for first 30 only)
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

    # Supervised dead bits
    sup_means = pred_sup.mean(dim=0)
    sq = (sup_means + 1) / 2
    sup_ent = -(sq * torch.log2(sq + eps) + (1 - sq) * torch.log2(1 - sq + eps))
    sup_dead = int((sup_ent < 0.3).sum())

    # Adversary accuracy (should converge to ~50% = random)
    adv_pred = (adv_logits.mean(dim=1) > 0).float()  # (N, 30)
    adv_acc = (adv_pred == sup_bits_gold).float().mean().item()

    # Per-concept accuracy (worst/best)
    per_word = []
    for i, word in enumerate(valid_words):
        pred_i = (pred_full[i] > 0).float()
        gold_i = (target_vectors[i] > 0).float()
        acc_i = (pred_i == gold_i).float().mean().item()
        per_word.append({'word': word, 'bit_accuracy': acc_i})
    per_word.sort(key=lambda x: x['bit_accuracy'])

    model.train()
    return {
        'sup_bit_accuracy': sup_acc,
        'full_bit_accuracy': full_acc,
        'sup_dead_bits': sup_dead,
        'free_dead_bits': free_dead,
        'total_dead_bits': sup_dead + free_dead,
        'free_mean_entropy': float(free_ent.mean()),
        'sup_mean_entropy': float(sup_ent.mean()),
        'adversary_accuracy': adv_acc,
        'n_concepts': N,
        'worst_5': per_word[:5],
        'best_5': per_word[-5:],
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='D-A18: Unified Final — iFSQ + Hybrid + v2 Anchors')
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
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    SCALES = {
        'base': {'layers': 6,  'dim': 256,  'heads': 8},
        'xl':   {'layers': 12, 'dim': 512,  'heads': 8},
        'xxl':  {'layers': 24, 'dim': 1024, 'heads': 16},
    }
    preset = SCALES[args.scale]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = os.path.join(_PROJECT, 'checkpoints', f'danza_unified_{args.scale}')
    os.makedirs(ckpt_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("  D-A18: UNIFIED FINAL — iFSQ + HYBRID BITS + v2 ANCHORS")
    print("=" * 70)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    # --- 1. Load data ---
    print(f"\n[1/6] Loading primitives and v2 anchors (158 concepts)...")
    prim_data = load_primitives()
    anchors, skipped = load_all_anchors(prim_data)
    print(f"  Anchors: {len(anchors)} words ({N_SUPERVISED} supervised + {N_FREE} free bits)")

    print(f"\n[2/6] Building subsumption pairs...")
    train_sub, test_sub = build_subsumption_pairs(anchors, prim_data)
    print(f"  Subsumption: train={len(train_sub)}, test={len(test_sub)}")

    # --- 2. Tokenizer ---
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

    print(f"  Tokenizing {len(stories)} stories...")
    all_tokens = []
    for story in stories:
        all_tokens.extend(tokenizer.encode(story, add_special=True))
    print(f"  Total: {len(all_tokens):,} tokens")

    # --- 3. Model ---
    print(f"\n[4/6] Initializing UnifiedTriadicGPT (iFSQ + hybrid {N_SUPERVISED}+{N_FREE})...")
    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block,
        n_layer=preset['layers'],
        n_embd=preset['dim'],
        n_head=preset['heads'],
        n_triadic_bits=N_BITS,
        dropout=args.dropout,
    )
    model = UnifiedTriadicGPT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Scale: {args.scale} ({preset['layers']}L/{preset['dim']}D/{preset['heads']}H)")
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Activation: iFSQ (2*sigmoid(1.6x)-1)")
    print(f"  Supervised: {N_SUPERVISED} bits | Free: {N_FREE} bits | Adversary: MLP")

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print(f"  Gradient checkpointing: ON")

    if device.type == 'cuda' and not args.no_compile:
        try:
            import triton  # noqa: F401
            model = torch.compile(model)
            print(f"  torch.compile: ON")
        except ImportError:
            print(f"  torch.compile: SKIPPED (triton not available)")

    # Mixed precision
    use_amp = device.type == 'cuda'
    amp_dtype = {'float32': torch.float32, 'float16': torch.float16,
                 'bfloat16': torch.bfloat16}[args.dtype]
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None
    if use_amp and amp_dtype != torch.float32:
        print(f"  Mixed precision: {args.dtype}")
        if amp_dtype == torch.bfloat16:
            print(f"  Blackwell Tensor Cores: bf16 (no loss scaling needed)")

    # VRAM estimate
    if device.type == 'cuda':
        bytes_per_param = 14 if amp_dtype != torch.float32 else 18
        model_vram = total_params * bytes_per_param / 1024**3
        act_per_item = preset['layers'] * args.block * 11 * preset['dim'] * 2 / 1024**3
        total_vram = model_vram + act_per_item * args.batch_size
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM estimate: {total_vram:.1f} GB / {gpu_mem:.1f} GB")

    # --- 4. Pre-encode anchors ---
    anchor_words = []
    anchor_ids_list = []
    anchor_targets = []
    for word, data in anchors.items():
        ids = tokenizer.encode(word, add_special=False)[:4]
        if ids:
            anchor_words.append(word)
            anchor_ids_list.append(ids)
            anchor_targets.append(data['target'])

    # 80/20 train/test split (deterministic)
    random.seed(42)
    indices = list(range(len(anchor_words)))
    random.shuffle(indices)
    n_test = max(1, int(len(indices) * 0.2))
    test_idx = set(indices[:n_test])
    train_idx = [i for i in indices if i not in test_idx]

    def _pack(idx_list):
        if not idx_list:
            z = torch.zeros((0, 1), dtype=torch.long, device=device)
            return z, torch.zeros((0, N_BITS), device=device), []
        words = [anchor_words[i] for i in idx_list]
        ids = [anchor_ids_list[i] for i in idx_list]
        tgts = [anchor_targets[i] for i in idx_list]
        mx = max(len(x) for x in ids)
        padded = torch.tensor([x + [0] * (mx - len(x)) for x in ids],
                               dtype=torch.long, device=device)
        target_t = torch.stack(tgts).to(device)
        return padded, target_t, words

    sup_train_t, sup_train_tgt, sup_train_words = _pack(train_idx)
    sup_test_t, sup_test_tgt, sup_test_words = _pack(list(test_idx))
    print(f"  v2 anchors: train={len(sup_train_words)}, test={len(sup_test_words)}")

    # Pre-encode subsumption pairs
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

    sub_train_h, sub_train_y, sub_train_valid = _pack_sub(train_sub)
    sub_test_h, sub_test_y, sub_test_valid = _pack_sub(test_sub)
    print(f"  Subsumption tensors: train={sub_train_h.shape[0]}, test={sub_test_h.shape[0]}")

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('step', 0)
        print(f"  Resumed at step {start_step}")

    # --- 5. Training ---
    print(f"\n[5/6] Training ({args.steps} steps, warmup={args.triadic_warmup_pct:.0%})...")
    print(f"  Alpha: {args.alpha}, Sup: {args.sup_weight}, Sub: {args.sub_weight}, "
          f"Adv: {args.adv_weight}, Align: {args.align_weight}")

    dataset = TextDataset(all_tokens, args.block)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, pin_memory=True)
    loader_iter = iter(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.999), weight_decay=0.01)
    warmup_steps = int(args.steps * 0.05)
    triadic_start = int(args.steps * args.triadic_warmup_pct)

    log_path = os.path.join(ckpt_dir, 'training_log.csv')
    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['step', 'total_loss', 'lang_loss', 'tri_loss', 'sup_loss',
                         'adv_loss', 'sub_loss', 'ppl', 'lr',
                         'sup_acc', 'full_acc', 'sup_dead', 'free_dead', 'adv_acc'])

    t0 = time.time()
    best_full_acc = 0.0

    for step in range(start_step + 1, args.steps + 1):
        # LR schedule: warmup + cosine
        if step <= warmup_steps:
            lr = args.lr * step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        triadic_active = step > triadic_start

        # Get batch
        try:
            x_batch, y_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x_batch, y_batch = next(loader_iter)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            logits, triadic_proj, lang_loss, sup_proj, adv_logits = model(
                x_batch, targets=y_batch)

            l_tri = torch.tensor(0.0, device=device)
            l_sup = torch.tensor(0.0, device=device)
            l_sub = torch.tensor(0.0, device=device)
            l_adv = torch.tensor(0.0, device=device)

            if triadic_active:
                l_tri = triadic_loss(triadic_proj, args.align_weight,
                                     model.wte, x_batch)
                l_sup = supervised_split_loss(model, sup_train_t, sup_train_tgt)
                l_sub = hybrid_subsumption_loss(model, sub_train_h, sub_train_y)
                l_adv = adversarial_loss(adv_logits, sup_proj)

            total_loss = (lang_loss
                          + args.alpha * l_tri
                          + args.alpha * args.sup_weight * l_sup
                          + args.alpha * args.sub_weight * l_sub
                          + args.alpha * args.adv_weight * l_adv)

        if scaler:  # float16
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:  # bfloat16 or float32
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Logging
        if step % args.print_every == 0 or step == 1:
            elapsed = time.time() - t0
            speed = (step - start_step) / max(elapsed, 1)
            eta = (args.steps - step) / max(speed, 0.01)
            ppl = math.exp(min(lang_loss.item(), 20))
            pct = step / args.steps
            bar = '#' * int(30 * pct) + '-' * (30 - int(30 * pct))

            if triadic_active:
                tri_str = (f"tri={l_tri.item():.3f} sup={l_sup.item():.3f} "
                           f"adv={l_adv.item():.3f} sub={l_sub.item():.3f}")
            else:
                tri_str = "warmup"

            print(f"  [{bar}] {step:>6}/{args.steps} | "
                  f"loss {total_loss.item():.4f} | lang {lang_loss.item():.4f} | "
                  f"{tri_str} | ppl {ppl:.1f} | {speed:.1f} it/s | ETA {eta:.0f}s")

        # Evaluation
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_unified(model, sup_test_t, sup_test_tgt, sup_test_words)
            train_metrics = evaluate_unified(model, sup_train_t, sup_train_tgt, sup_train_words)
            sub_train_rate, _ = evaluate_subsumption(model, sub_train_h, sub_train_y,
                                                      len(sub_train_valid))
            sub_test_rate, _ = evaluate_subsumption(model, sub_test_h, sub_test_y,
                                                     len(sub_test_valid))

            print(f"\n  --- Eval @ step {step} ---")
            print(f"  Supervised acc:  train={train_metrics.get('sup_bit_accuracy', 0):.1%}  "
                  f"test={metrics.get('sup_bit_accuracy', 0):.1%}")
            print(f"  Full bit acc:    train={train_metrics.get('full_bit_accuracy', 0):.1%}  "
                  f"test={metrics.get('full_bit_accuracy', 0):.1%}")
            print(f"  Dead bits:       sup={metrics.get('sup_dead_bits', 0)}/{N_SUPERVISED}  "
                  f"free={metrics.get('free_dead_bits', 0)}/{N_FREE}  "
                  f"total={metrics.get('total_dead_bits', 0)}/{N_BITS}")
            print(f"  Subsumption:     train={sub_train_rate:.1%}  test={sub_test_rate:.1%}")
            print(f"  Adversary acc:   {metrics.get('adversary_accuracy', 0):.1%} (target: ~50%)")

            if metrics.get('worst_5'):
                worst = [f"{w['word']}({w['bit_accuracy']:.0%})" for w in metrics['worst_5'][:3]]
                print(f"  Worst test:      {', '.join(worst)}")
            if metrics.get('best_5'):
                best = [f"{w['word']}({w['bit_accuracy']:.0%})" for w in metrics['best_5'][-3:]]
                print(f"  Best test:       {', '.join(best)}")
            print()

            log_writer.writerow([
                step, total_loss.item(), lang_loss.item(),
                l_tri.item() if triadic_active else 0,
                l_sup.item() if triadic_active else 0,
                l_adv.item() if triadic_active else 0,
                l_sub.item() if triadic_active else 0,
                math.exp(min(lang_loss.item(), 20)), lr,
                metrics.get('sup_bit_accuracy', 0),
                metrics.get('full_bit_accuracy', 0),
                metrics.get('sup_dead_bits', 0),
                metrics.get('free_dead_bits', 0),
                metrics.get('adversary_accuracy', 0),
            ])
            log_file.flush()

            # Save best (by full bit accuracy on test set)
            full_acc = metrics.get('full_bit_accuracy', 0)
            if full_acc > best_full_acc:
                best_full_acc = full_acc
                best_path = os.path.join(ckpt_dir, 'model_best.pt')
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'vocab_size': config.vocab_size,
                        'block_size': config.block_size,
                        'n_layer': config.n_layer,
                        'n_embd': config.n_embd,
                        'n_head': config.n_head,
                        'n_triadic_bits': config.n_triadic_bits,
                    },
                    'architecture': 'UnifiedTriadicGPT',
                    'activation': 'ifsq',
                    'n_supervised': N_SUPERVISED,
                    'n_free': N_FREE,
                    'full_bit_accuracy_test': full_acc,
                    'sup_bit_accuracy_test': metrics.get('sup_bit_accuracy', 0),
                    'sub_rate_test': sub_test_rate,
                    'dead_bits': metrics.get('total_dead_bits', 0),
                }, best_path)
                print(f"  ** New best: {full_acc:.1%} → saved {best_path}")

        # Periodic checkpoint
        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = os.path.join(ckpt_dir, f'model_step{step}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'config': {
                    'vocab_size': config.vocab_size,
                    'block_size': config.block_size,
                    'n_layer': config.n_layer,
                    'n_embd': config.n_embd,
                    'n_head': config.n_head,
                    'n_triadic_bits': config.n_triadic_bits,
                },
                'architecture': 'UnifiedTriadicGPT',
                'activation': 'ifsq',
                'n_supervised': N_SUPERVISED,
                'n_free': N_FREE,
            }, ckpt_path)

    log_file.close()

    # --- 6. Final evaluation ---
    print(f"\n[6/6] Final evaluation...")

    # Regla de tres
    r3_results = evaluate_regla_de_tres(model, tokenizer, anchors, device)
    print(f"\n  --- Regla de Tres ---")
    for r in r3_results:
        print(f"  {r['quad']:40s}  cos={r['cosine']:+.3f}  bit_acc={r['bit_accuracy']:.1%}")
    if r3_results:
        mean_cos = np.mean([r['cosine'] for r in r3_results])
        mean_bit = np.mean([r['bit_accuracy'] for r in r3_results])
        print(f"  Mean: cosine={mean_cos:+.3f}, bit_accuracy={mean_bit:.1%}")

    # Final metrics
    final_train = evaluate_unified(model, sup_train_t, sup_train_tgt, sup_train_words)
    final_test = evaluate_unified(model, sup_test_t, sup_test_tgt, sup_test_words)
    sub_final_train, _ = evaluate_subsumption(model, sub_train_h, sub_train_y,
                                               len(sub_train_valid))
    sub_final_test, _ = evaluate_subsumption(model, sub_test_h, sub_test_y,
                                              len(sub_test_valid))

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — D-A18 UNIFIED FINAL ({args.scale})")
    print(f"{'=' * 70}")
    print(f"  Training time:    {elapsed/60:.1f} min")
    print(f"  Supervised acc:   train={final_train.get('sup_bit_accuracy', 0):.1%}  "
          f"test={final_test.get('sup_bit_accuracy', 0):.1%}")
    print(f"  Full bit acc:     train={final_train.get('full_bit_accuracy', 0):.1%}  "
          f"test={final_test.get('full_bit_accuracy', 0):.1%}")
    print(f"  Best test acc:    {best_full_acc:.1%}")
    print(f"  Dead bits:        sup={final_test.get('sup_dead_bits', 0)}/{N_SUPERVISED}  "
          f"free={final_test.get('free_dead_bits', 0)}/{N_FREE}  "
          f"total={final_test.get('total_dead_bits', 0)}/{N_BITS}")
    print(f"  Subsumption:      train={sub_final_train:.1%}  test={sub_final_test:.1%}")
    print(f"  Adversary acc:    {final_test.get('adversary_accuracy', 0):.1%}")
    if r3_results:
        print(f"  Regla de tres:    cos={mean_cos:+.3f}, bits={mean_bit:.1%}")
    print(f"  Checkpoint:       {ckpt_dir}")
    print(f"{'=' * 70}")

    # Target check
    test_acc = final_test.get('full_bit_accuracy', 0)
    dead = final_test.get('total_dead_bits', 0)
    active = N_BITS - dead
    print(f"\n  TARGET CHECK:")
    print(f"    Test acc >=90%:     {test_acc:.1%}  {'PASS' if test_acc >= 0.90 else 'FAIL'}")
    print(f"    Sub test >=95%:     {sub_final_test:.1%}  {'PASS' if sub_final_test >= 0.95 else 'FAIL'}")
    print(f"    Dead bits <10:      {dead}  {'PASS' if dead < 10 else 'FAIL'}")
    print(f"    Active bits >50:    {active}  {'PASS' if active > 50 else 'FAIL'}")

    # Save results JSON
    results = {
        'experiment': 'D-A18_unified_final',
        'scale': args.scale,
        'architecture': 'UnifiedTriadicGPT',
        'activation': 'ifsq',
        'anchors': 'v2 (158)',
        'bits': f'{N_SUPERVISED} supervised + {N_FREE} free',
        'steps': args.steps,
        'training_time_min': round(elapsed / 60, 1),
        'sup_bit_accuracy_train': final_train.get('sup_bit_accuracy', 0),
        'sup_bit_accuracy_test': final_test.get('sup_bit_accuracy', 0),
        'full_bit_accuracy_train': final_train.get('full_bit_accuracy', 0),
        'full_bit_accuracy_test': final_test.get('full_bit_accuracy', 0),
        'best_test_accuracy': best_full_acc,
        'sup_dead_bits': final_test.get('sup_dead_bits', 0),
        'free_dead_bits': final_test.get('free_dead_bits', 0),
        'total_dead_bits': dead,
        'subsumption_train': sub_final_train,
        'subsumption_test': sub_final_test,
        'adversary_accuracy': final_test.get('adversary_accuracy', 0),
        'r3_mean_cosine': round(mean_cos, 3) if r3_results else None,
        'r3_mean_bit_accuracy': round(mean_bit, 3) if r3_results else None,
        'checkpoint_dir': ckpt_dir,
    }
    results_path = os.path.join(ckpt_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {results_path}")


if __name__ == '__main__':
    main()

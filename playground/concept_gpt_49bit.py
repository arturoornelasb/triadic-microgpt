"""
49-Bit Concept GPT — End-to-End Training with 7x7 Primitives.

Trains a TriadicGPT with 49 bits mapped to the Sistema 7x7 primitives,
using subsumption loss + supervised primitive targets from the seed lexicon.

Loss = L_lang + alpha * (L_triadic + sub_weight * L_sub + sup_weight * L_sup)

v2: Fixed all-ones collapse from v1.  Root cause: sigmoid+subsumption has a
trivial global minimum (all bits = 1.0 satisfies relu(h-y)=0 for all pairs).
Fix: (1) default to tanh (proven with subsumption in P6), (2) add supervised
primitive loss that anchors T1 words to their correct bits.

Usage:
  python playground/concept_gpt_49bit.py
  python playground/concept_gpt_49bit.py --scale xl --steps 50000
  python playground/concept_gpt_49bit.py --activation sigmoid  # experimental
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator
from conceptual_tokenizer.config import (
    PRIMITIVE_NAMES, PRIMITIVE_TO_PRIME, PRIMITIVE_TO_CATEGORY,
    CATEGORY_NAMES, N_PRIMITIVES,
)
from conceptual_tokenizer.seed_lexicon import TIER_1, TIER_2, get_full_lexicon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORY_SEPARATOR = '<' + '|endoftext|' + '>'


# ============================================================
# ConceptTriadicGPT: configurable activation (tanh or sigmoid+anneal)
# ============================================================

class ConceptTriadicGPT(TriadicGPT):
    """TriadicGPT with configurable triadic activation."""

    def __init__(self, config, activation='tanh'):
        super().__init__(config)
        self.activation = activation
        self.temperature = 1.0  # only used for sigmoid mode

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        raw = self.triadic_head(x)
        if self.activation == 'sigmoid':
            triadic_proj = torch.sigmoid(raw * self.temperature) * 2 - 1
        else:
            triadic_proj = torch.tanh(raw)  # default, proven with subsumption

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, triadic_proj, loss


# ============================================================
# Subsumption pairs from seed lexicon
# ============================================================

def build_subsumption_pairs(test_fraction=0.2):
    """
    Extract hypernym-hyponym pairs from the seed lexicon.

    For each Tier 2 word (compound, 2+ primitives), create a pair with
    the first Tier 1 word that shares each primitive. The Tier 1 word
    (1 primitive) is the hypernym — its bit should be active in the hyponym.

    Returns:
        train_pairs, test_pairs: [(hyper_word, hypo_word, hyper_map, hypo_map), ...]
    """
    # Reverse index: primitive -> first Tier 1 word
    prim_to_t1 = {}
    for word, mapping in TIER_1.items():
        for prim_name in mapping:
            if prim_name not in prim_to_t1:
                prim_to_t1[prim_name] = word

    all_pairs = []
    seen = set()
    for hypo_word, hypo_map in TIER_2.items():
        for prim_name in hypo_map:
            if prim_name in prim_to_t1:
                hyper_word = prim_to_t1[prim_name]
                key = (hyper_word, hypo_word)
                if key not in seen:
                    seen.add(key)
                    all_pairs.append((hyper_word, hypo_word, TIER_1[hyper_word], hypo_map))

    random.seed(42)
    random.shuffle(all_pairs)
    n_test = max(1, int(len(all_pairs) * test_fraction))
    return all_pairs[n_test:], all_pairs[:n_test]


def prepare_sub_tensors(pairs, tokenizer, device, max_tok_len=4):
    """Pre-encode subsumption pairs into padded tensors for efficient batched forward."""
    hyper_ids, hypo_ids = [], []
    valid = []
    for h_word, y_word, h_map, y_map in pairs:
        h = tokenizer.encode(h_word, add_special=False)[:max_tok_len]
        y = tokenizer.encode(y_word, add_special=False)[:max_tok_len]
        if h and y:
            hyper_ids.append(h)
            hypo_ids.append(y)
            valid.append((h_word, y_word, h_map, y_map))

    if not valid:
        z = torch.zeros((0, 1), dtype=torch.long, device=device)
        return z, z, valid

    def pad(ids_list):
        mx = max(len(x) for x in ids_list)
        return torch.tensor([x + [0] * (mx - len(x)) for x in ids_list],
                            dtype=torch.long, device=device)

    return pad(hyper_ids), pad(hypo_ids), valid


# ============================================================
# Subsumption loss (batched)
# ============================================================

def subsumption_loss_batch(model, hyper_t, hypo_t, n_sample=32):
    """
    Batched subsumption loss: relu(hyper_01 - hypo_01).mean()

    Samples n_sample pairs for efficiency.
    """
    N = hyper_t.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=hyper_t.device)

    if N > n_sample:
        idx = torch.randperm(N, device=hyper_t.device)[:n_sample]
        h_batch = hyper_t[idx]
        y_batch = hypo_t[idx]
    else:
        h_batch = hyper_t
        y_batch = hypo_t

    _, h_proj, _ = model(h_batch)  # (n, T_h, 49)
    _, y_proj, _ = model(y_batch)  # (n, T_y, 49)

    # Mean-pool over token positions
    h_p = h_proj.mean(dim=1)  # (n, 49)
    y_p = y_proj.mean(dim=1)

    # Map to [0, 1]
    h_01 = (h_p + 1) / 2
    y_01 = (y_p + 1) / 2

    # Penalize bits where hypernym > hyponym
    return F.relu(h_01 - y_01).mean()


# ============================================================
# Supervised primitive loss (anchors T1 words to correct bits)
# ============================================================

def prepare_supervised_tensors(tokenizer, device, max_tok_len=4, test_fraction=0.2):
    """
    Pre-encode T1 + T2 words and their target 49-bit vectors.

    Uses the full lexicon (Tier 1 + Tier 2). T1 words have 1 primitive,
    T2 words have 2-5 primitives. All 49 bits are supervised (active→target,
    inactive→0).

    Returns:
        train: (word_tensors, target_vectors, valid_words)
        test:  (word_tensors, target_vectors, valid_words)
    """
    full_lexicon = get_full_lexicon()  # T1 + T2 merged

    all_items = []
    for word, mapping in full_lexicon.items():
        ids = tokenizer.encode(word, add_special=False)[:max_tok_len]
        if not ids:
            continue

        target = torch.zeros(N_PRIMITIVES)
        for prim_name, (state, intensity) in mapping.items():
            idx = PRIMITIVE_NAMES.index(prim_name)
            if state == '+':
                target[idx] = intensity
            elif state == '-':
                target[idx] = -intensity
            elif state == '0':
                target[idx] = 0.0

        all_items.append((word, ids, target))

    # Split train/test
    random.seed(42)
    random.shuffle(all_items)
    n_test = max(1, int(len(all_items) * test_fraction))
    test_items = all_items[:n_test]
    train_items = all_items[n_test:]

    def _pack(items):
        if not items:
            z = torch.zeros((0, 1), dtype=torch.long, device=device)
            return z, torch.zeros((0, N_PRIMITIVES), device=device), []
        words = [it[0] for it in items]
        ids_list = [it[1] for it in items]
        targets = [it[2] for it in items]
        mx = max(len(x) for x in ids_list)
        padded = torch.tensor([x + [0] * (mx - len(x)) for x in ids_list],
                               dtype=torch.long, device=device)
        target_t = torch.stack(targets).to(device)
        return padded, target_t, words

    return _pack(train_items), _pack(test_items)


def supervised_loss_batch(model, word_t, target_t, n_sample=64):
    """
    MSE loss between model projections and target primitive vectors.

    Only penalizes bits that have non-zero targets (active primitives).
    Bits with target=0 (N/A) are left free for the model to decide.
    """
    N = word_t.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=word_t.device)

    if N > n_sample:
        idx = torch.randperm(N, device=word_t.device)[:n_sample]
        w_batch = word_t[idx]
        t_batch = target_t[idx]
    else:
        w_batch = word_t
        t_batch = target_t

    _, proj, _ = model(w_batch)      # (n, T, 49)
    pred = proj.mean(dim=1)          # (n, 49) mean-pool over tokens

    # Full supervision: active bits -> target, inactive bits -> 0
    # This grounds ALL 49 bits, preventing the "all high" failure mode
    return F.mse_loss(pred, t_batch)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_subsumption(model, hyper_t, hypo_t, valid_pairs):
    """Evaluate binary subsumption satisfaction."""
    model.eval()
    N = hyper_t.shape[0]
    if N == 0:
        return 0.0, 0.0, 0

    _, h_proj, _ = model(hyper_t)
    _, y_proj, _ = model(hypo_t)

    h_bits = ((h_proj.mean(dim=1) + 1) / 2 > 0.5).float()  # (N, 49)
    y_bits = ((y_proj.mean(dim=1) + 1) / 2 > 0.5).float()

    # Violation: hypernym bit ON but hyponym bit OFF
    violations = (h_bits * (1 - y_bits)).sum(dim=1)  # (N,)
    satisfied = (violations == 0).float().sum().item()
    avg_viol = violations.mean().item()

    model.train()
    return satisfied / N, avg_viol, N


@torch.no_grad()
def evaluate_primitive_quality(model, tokenizer, device):
    """Evaluate if each bit corresponds to the correct 7x7 primitive."""
    model.eval()

    # For each Tier 1 word (single primitive), check if the right bit is strongest
    correct = 0
    total = 0
    all_projs = []

    for word, mapping in TIER_1.items():
        prim_name = list(mapping.keys())[0]
        state = list(mapping.values())[0][0]
        if state != '+':
            continue

        ids = tokenizer.encode(word, add_special=False)
        if not ids:
            continue

        x = torch.tensor([ids[:4]], dtype=torch.long, device=device)
        _, proj, _ = model(x)
        p = proj[0].mean(dim=0).cpu()
        p_01 = (p + 1) / 2

        expected_idx = PRIMITIVE_NAMES.index(prim_name)
        if p_01.argmax().item() == expected_idx:
            correct += 1
        total += 1
        all_projs.append(p.numpy())

    primary_acc = correct / max(total, 1)

    # Dead bits & entropy
    if all_projs:
        arr = np.stack(all_projs)
        bit_means = (arr > 0).mean(axis=0)
        eps = 1e-7
        ent = -(bit_means * np.log2(bit_means + eps) +
                (1 - bit_means) * np.log2(1 - bit_means + eps))
        dead_bits = int((ent < 0.3).sum())
        mean_ent = float(ent.mean())
    else:
        dead_bits, mean_ent = 49, 0.0

    # Per-category accuracy
    cat_stats = {}
    for word, mapping in TIER_1.items():
        prim_name = list(mapping.keys())[0]
        state = list(mapping.values())[0][0]
        if state != '+':
            continue
        cat = PRIMITIVE_TO_CATEGORY[prim_name]
        ids = tokenizer.encode(word, add_special=False)
        if not ids:
            continue
        x = torch.tensor([ids[:4]], dtype=torch.long, device=device)
        _, proj, _ = model(x)
        p_01 = (proj[0].mean(dim=0).cpu() + 1) / 2
        expected_idx = PRIMITIVE_NAMES.index(prim_name)
        # Check if expected primitive is in top-3
        top3 = p_01.topk(3).indices.tolist()
        if cat not in cat_stats:
            cat_stats[cat] = {'top1': 0, 'top3': 0, 'total': 0}
        cat_stats[cat]['total'] += 1
        if expected_idx == p_01.argmax().item():
            cat_stats[cat]['top1'] += 1
        if expected_idx in top3:
            cat_stats[cat]['top3'] += 1

    model.train()
    return {
        'primary_accuracy': primary_acc,
        'primary_total': total,
        'dead_bits': dead_bits,
        'mean_entropy': mean_ent,
        'per_category': cat_stats,
    }


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:], dtype=torch.long))


@torch.no_grad()
def _eval_sup_accuracy(model, word_t, target_t, valid_words):
    """Top-1 accuracy: for each word, is ANY of its expected primitives the top-1 bit?"""
    model.eval()
    N = word_t.shape[0]
    if N == 0:
        return 0.0
    _, proj, _ = model(word_t)
    pred = proj.mean(dim=1)  # (N, 49)
    p_01 = (pred + 1) / 2
    correct = 0
    for i in range(N):
        top1_idx = p_01[i].argmax().item()
        # Check if top1 matches ANY non-zero target
        expected = (target_t[i] != 0).nonzero(as_tuple=True)[0].tolist()
        if top1_idx in expected:
            correct += 1
    model.train()
    return correct / N


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='49-Bit Concept GPT (7x7 End-to-End)')
    parser.add_argument('--scale', choices=['base', 'xl'], default='base')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.05, help='Triadic loss weight')
    parser.add_argument('--sub-weight', type=float, default=5.0, help='Subsumption loss multiplier')
    parser.add_argument('--sub-sample', type=int, default=32, help='Subsumption pairs per step')
    parser.add_argument('--sup-weight', type=float, default=2.0, help='Supervised primitive loss weight')
    parser.add_argument('--sup-sample', type=int, default=64, help='Supervised words per step')
    parser.add_argument('--entropy-weight', type=float, default=2.0)
    parser.add_argument('--align-weight', type=float, default=3.0)
    parser.add_argument('--align-mode', default='mse', choices=['mse', 'rank', 'infonce'])
    parser.add_argument('--activation', default='tanh', choices=['tanh', 'sigmoid'],
                        help='Triadic activation (tanh=proven, sigmoid=experimental)')
    parser.add_argument('--start-temp', type=float, default=1.0, help='Sigmoid start T (soft)')
    parser.add_argument('--end-temp', type=float, default=5.0, help='Sigmoid end T (hard)')
    parser.add_argument('--triadic-warmup-pct', type=float, default=0.5)
    parser.add_argument('--stories', type=int, default=50000)
    parser.add_argument('--vocab', type=int, default=4096)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=2500)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--no-distill', action='store_true', default=True)
    args = parser.parse_args()

    SCALES = {
        'base': {'layers': 6, 'dim': 256, 'heads': 8},
        'xl':   {'layers': 12, 'dim': 512, 'heads': 8},
    }
    preset = SCALES[args.scale]
    n_bits = N_PRIMITIVES  # 49

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', f'concept_gpt_49bit_{args.scale}')
    os.makedirs(ckpt_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("  49-BIT CONCEPT GPT  --  7x7 End-to-End Training")
    print("=" * 70)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Scale: {args.scale} ({preset['layers']}L/{preset['dim']}D/{preset['heads']}H)")
    print(f"  Bits: {n_bits} (7x7 primitives: {PRIMITIVE_NAMES[0]}..{PRIMITIVE_NAMES[-1]})")
    act_str = args.activation
    if args.activation == 'sigmoid':
        act_str += f" + anneal (T: {args.start_temp} -> {args.end_temp})"
    print(f"  Activation: {act_str}")
    print(f"  Subsumption: weight={args.sub_weight}, sample={args.sub_sample}/step")
    print(f"  Supervised:  weight={args.sup_weight}, sample={args.sup_sample}/step")

    # --- 1. Subsumption pairs ---
    print(f"\n[1/5] Building subsumption pairs from seed lexicon...")
    train_pairs, test_pairs = build_subsumption_pairs(test_fraction=0.2)
    print(f"  Total pairs: {len(train_pairs) + len(test_pairs)} "
          f"(train={len(train_pairs)}, test={len(test_pairs)})")
    for h, y, _, _ in train_pairs[:3]:
        print(f"    {h} subsumes {y}")
    for h, y, _, _ in test_pairs[:2]:
        print(f"    [TEST] {h} subsumes {y}")

    # --- 2. Tokenizer ---
    data_path = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 30]
    if args.stories and len(stories) > args.stories:
        random.seed(42)
        random.shuffle(stories)
        stories = stories[:args.stories]

    tok_path = os.path.join(ckpt_dir, 'tokenizer.json')
    if args.tokenizer and os.path.exists(args.tokenizer):
        print(f"\n[2/5] Loading tokenizer: {os.path.basename(args.tokenizer)}")
        tokenizer = BPETokenizer.load(args.tokenizer)
    else:
        print(f"\n[2/5] Training BPE tokenizer (vocab={args.vocab})...")
        tokenizer = BPETokenizer(vocab_size=args.vocab)
        tokenizer.train(stories, verbose=True)
        tokenizer.save(tok_path)
    print(f"  Vocab: {tokenizer.vocab_size}")

    # --- 3. Tokenize corpus ---
    print(f"\n[3/5] Tokenizing {len(stories)} stories...")
    all_tokens = []
    for i, story in enumerate(stories):
        ids = tokenizer.encode(story, add_special=True)
        all_tokens.extend(ids)
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(stories)} ({len(all_tokens):,} tokens)")
    print(f"  Total: {len(all_tokens):,} tokens")

    # --- 4. Model ---
    print(f"\n[4/5] Initializing ConceptTriadicGPT (49 bits, {args.activation})...")
    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block,
        n_layer=preset['layers'],
        n_embd=preset['dim'],
        n_head=preset['heads'],
        n_triadic_bits=n_bits,
        dropout=args.dropout,
    )
    model = ConceptTriadicGPT(config, activation=args.activation).to(device)
    total_params = model.num_params()
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Pre-encode subsumption pairs as tensors
    train_h_t, train_y_t, train_valid = prepare_sub_tensors(train_pairs, tokenizer, device)
    test_h_t, test_y_t, test_valid = prepare_sub_tensors(test_pairs, tokenizer, device)
    print(f"  Sub tensors: train={train_h_t.shape[0]}, test={test_h_t.shape[0]}")

    # Pre-encode supervised primitive targets (T1 + T2, with train/test split)
    (sup_word_t, sup_target_t, sup_valid), \
    (sup_test_word_t, sup_test_target_t, sup_test_valid) = \
        prepare_supervised_tensors(tokenizer, device)
    n_t1_train = sum(1 for w in sup_valid if w in TIER_1)
    n_t2_train = sum(1 for w in sup_valid if w in TIER_2)
    n_t1_test = sum(1 for w in sup_test_valid if w in TIER_1)
    n_t2_test = sum(1 for w in sup_test_valid if w in TIER_2)
    print(f"  Sup train: {sup_word_t.shape[0]} words (T1={n_t1_train}, T2={n_t2_train})")
    print(f"  Sup test:  {sup_test_word_t.shape[0]} words (T1={n_t1_test}, T2={n_t2_test})")

    # Verify PrimeMapper(49) matches 7x7 primes
    mapper = PrimeMapper(n_bits)
    expected_primes = [PRIMITIVE_TO_PRIME[name] for name in PRIMITIVE_NAMES]
    mapper_primes = mapper.primes[:n_bits]
    match = all(a == b for a, b in zip(mapper_primes, expected_primes))
    print(f"  PrimeMapper(49) matches 7x7 primes: {match}")
    if not match:
        print(f"    WARNING: mapper primes = {mapper_primes[:7]}...")
        print(f"    WARNING: 7x7 primes   = {expected_primes[:7]}...")

    # --- 5. Training ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=0.01, betas=(0.9, 0.95))
    dataset = TextDataset(all_tokens, args.block)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, drop_last=True, num_workers=0)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    triadic_warmup = int(args.steps * args.triadic_warmup_pct)

    print(f"\n[5/5] Training for {args.steps} steps...")
    print(f"  Triadic activation: step {triadic_warmup}")
    print(f"  Align: {args.align_mode} (weight={args.align_weight})")
    print(f"  Entropy: {args.entropy_weight}")
    print("-" * 70)

    # CSV log
    csv_path = os.path.join(ckpt_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'total_loss', 'lang_loss', 'tri_loss', 'sub_loss',
                          'sup_loss', 'temperature', 'lr', 'elapsed_s'])

    model.train()
    start_time = time.time()
    step = 0
    best_loss = float('inf')
    data_iter = iter(dataloader)

    while step < args.steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        # Temperature annealing: only anneal after triadic warmup
        if step < triadic_warmup:
            temperature = args.start_temp
        else:
            tri_progress = (step - triadic_warmup) / max(args.steps - triadic_warmup - 1, 1)
            temperature = args.start_temp + (args.end_temp - args.start_temp) * tri_progress
        model.temperature = temperature

        # LR: warmup + cosine
        warmup_steps = min(500, args.steps // 10)
        if step < warmup_steps:
            lr_t = args.lr * (step + 1) / warmup_steps
        else:
            p = (step - warmup_steps) / max(args.steps - warmup_steps, 1)
            lr_t = args.lr * max(0.1, 0.5 * (1.0 + math.cos(math.pi * p)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        # Forward
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)

            total_loss = lang_loss
            tri_loss_val = 0.0
            sub_loss_val = 0.0
            sup_loss_val = 0.0

            if step >= triadic_warmup:
                alpha_ramp = int(args.steps * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_ramp)
                current_alpha = args.alpha * alpha_factor

                # Standard triadic loss (diversity + contrastive + entropy + alignment)
                tri_loss = model.triadic_loss(
                    triadic_proj,
                    entropy_weight=args.entropy_weight,
                    input_ids=x,
                    align_weight=args.align_weight,
                    align_mode=args.align_mode,
                )
                total_loss = lang_loss + current_alpha * tri_loss
                tri_loss_val = tri_loss.item()

                # Subsumption loss (batched, sampled)
                if train_h_t.shape[0] > 0:
                    sub_loss = subsumption_loss_batch(
                        model, train_h_t, train_y_t, n_sample=args.sub_sample)
                    total_loss = total_loss + current_alpha * args.sub_weight * sub_loss
                    sub_loss_val = sub_loss.item()

                # Supervised primitive loss (anchors T1 words to correct bits)
                if sup_word_t.shape[0] > 0:
                    sup_loss = supervised_loss_batch(
                        model, sup_word_t, sup_target_t, n_sample=args.sup_sample)
                    total_loss = total_loss + current_alpha * args.sup_weight * sup_loss
                    sup_loss_val = sup_loss.item()

        # Backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # CSV
        elapsed = time.time() - start_time
        csv_writer.writerow([
            step + 1, f'{total_loss.item():.6f}', f'{lang_loss.item():.6f}',
            f'{tri_loss_val:.6f}', f'{sub_loss_val:.6f}', f'{sup_loss_val:.6f}',
            f'{temperature:.3f}', f'{lr_t:.8f}', f'{elapsed:.1f}',
        ])

        # Print
        if step % args.print_every == 0 or step == args.steps - 1:
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            remaining = (args.steps - step - 1) / sps if sps > 0 else 0
            pct = (step + 1) / args.steps * 100
            filled = int(30 * (step + 1) / args.steps)
            bar = '#' * filled + '-' * (30 - filled)
            eta = f"{remaining/60:.1f}m" if remaining >= 60 else f"{remaining:.0f}s"

            msg = f"  [{bar}] {pct:5.1f}%"
            msg += f" | step {step+1}/{args.steps}"
            msg += f" | loss {lang_loss.item():.4f}"
            if step >= triadic_warmup:
                msg += f" | tri {tri_loss_val:.4f} sub {sub_loss_val:.4f} sup {sup_loss_val:.4f}"
            msg += f" | {sps:.1f}s/s | ETA {eta}"
            print(msg)

        # Mid-training eval
        if (step + 1) % args.eval_every == 0 and step >= triadic_warmup:
            print(f"\n  --- Eval @ step {step+1} ---")
            tr_rate, tr_viol, tr_n = evaluate_subsumption(model, train_h_t, train_y_t, train_valid)
            te_rate, te_viol, te_n = evaluate_subsumption(model, test_h_t, test_y_t, test_valid)
            pq = evaluate_primitive_quality(model, tokenizer, device)

            # Supervised accuracy on held-out test set
            sup_test_acc = _eval_sup_accuracy(model, sup_test_word_t, sup_test_target_t,
                                              sup_test_valid)

            print(f"  Sub train: {tr_rate:.1%} ({int(tr_rate*tr_n)}/{tr_n}), "
                  f"avg violation: {tr_viol:.1f} bits")
            print(f"  Sub test:  {te_rate:.1%} ({int(te_rate*te_n)}/{te_n}), "
                  f"avg violation: {te_viol:.1f} bits")
            print(f"  Sup train acc: {pq['primary_accuracy']:.1%} "
                  f"({pq['primary_total']} words)")
            print(f"  Sup TEST acc:  {sup_test_acc:.1%} "
                  f"({len(sup_test_valid)} held-out words)")
            print(f"  Dead bits: {pq['dead_bits']}/49, entropy: {pq['mean_entropy']:.3f}")
            # Per-category summary
            for cat in CATEGORY_NAMES:
                cs = pq['per_category'].get(cat, {})
                t = cs.get('total', 0)
                if t > 0:
                    t1 = cs.get('top1', 0)
                    t3 = cs.get('top3', 0)
                    print(f"    {cat:>22s}: top1 {t1}/{t} ({t1/t:.0%}), "
                          f"top3 {t3}/{t} ({t3/t:.0%})")
            print()
            model.train()

        # Checkpoint
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            tag = f"L{preset['layers']}_D{preset['dim']}_B49"
            ckpt_path = os.path.join(ckpt_dir, f'model_{tag}_step{step+1}.pt')
            save_dict = {
                'model_state_dict': model.state_dict(),
                'config': vars(config),
                'step': step + 1,
                'loss': lang_loss.item(),
                'temperature': temperature,
                'args': vars(args),
                'primitive_names': PRIMITIVE_NAMES,
            }
            torch.save(save_dict, ckpt_path)

            if lang_loss.item() < best_loss:
                best_loss = lang_loss.item()
                best_path = os.path.join(ckpt_dir, f'model_{tag}_best.pt')
                torch.save(save_dict, best_path)

            tokenizer.save(os.path.join(ckpt_dir, 'tokenizer.json'))
            print(f"  >>> Saved: {ckpt_path}")

        step += 1

    csv_file.close()

    # ============================================================
    # Final evaluation
    # ============================================================
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("  FINAL EVALUATION")
    print("=" * 70)

    tr_rate, tr_viol, tr_n = evaluate_subsumption(model, train_h_t, train_y_t, train_valid)
    te_rate, te_viol, te_n = evaluate_subsumption(model, test_h_t, test_y_t, test_valid)
    pq = evaluate_primitive_quality(model, tokenizer, device)
    sup_train_acc = _eval_sup_accuracy(model, sup_word_t, sup_target_t, sup_valid)
    sup_test_acc = _eval_sup_accuracy(model, sup_test_word_t, sup_test_target_t, sup_test_valid)

    print(f"  Subsumption train: {tr_rate:.1%} ({int(tr_rate*tr_n)}/{tr_n})")
    print(f"  Subsumption test:  {te_rate:.1%} ({int(te_rate*te_n)}/{te_n})")
    print(f"  Sup train acc:     {sup_train_acc:.1%} ({len(sup_valid)} words, T1+T2)")
    print(f"  Sup TEST acc:      {sup_test_acc:.1%} ({len(sup_test_valid)} held-out words)")
    print(f"  Dead bits:         {pq['dead_bits']}/49")
    print(f"  Bit entropy:       {pq['mean_entropy']:.3f}")
    print(f"  Language loss:     {lang_loss.item():.4f}")
    print(f"  Time:              {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"  Speed:             {args.steps/elapsed:.1f} steps/s")

    # Per-category table
    print(f"\n  Per-category primitive accuracy:")
    print(f"  {'Category':>22s}  {'Top-1':>8s}  {'Top-3':>8s}  {'Count':>6s}")
    print(f"  {'='*22}  {'='*8}  {'='*8}  {'='*6}")
    for cat in CATEGORY_NAMES:
        cs = pq['per_category'].get(cat, {})
        t = cs.get('total', 0)
        if t > 0:
            print(f"  {cat:>22s}  {cs['top1']/t:>7.0%}  {cs['top3']/t:>7.0%}  {t:>6d}")

    # Sample generations
    model.eval()
    print(f"\n  Sample generations:")
    bos_id = tokenizer.special_tokens.get('<BOS>', 1)
    for i in range(3):
        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        output = model.generate(input_ids, max_new_tokens=40, temperature=0.7, top_k=50)
        text = tokenizer.decode(output[0].tolist(), skip_special=True)
        print(f"    {i+1}. {text[:80]}")

    # Primitive activations for key words
    print(f"\n  Primitive activations (top-3 per word):")
    probe_words = ["fire", "water", "love", "king", "sun", "death", "truth",
                   "music", "red", "mountain", "silence", "hero", "dream"]
    for word in probe_words:
        ids = tokenizer.encode(word, add_special=False)
        if not ids:
            continue
        x = torch.tensor([ids[:4]], dtype=torch.long, device=device)
        _, proj, _ = model(x)
        p_01 = (proj[0].mean(dim=0).cpu() + 1) / 2
        top3 = p_01.topk(3)
        prims = [f"{PRIMITIVE_NAMES[i]}={v:.2f}" for v, i in zip(top3.values, top3.indices)]
        # Expected from lexicon
        lexicon = get_full_lexicon()
        expected = list(lexicon.get(word, {}).keys())[:2]
        exp_str = f"(expect: {','.join(expected)})" if expected else ""
        print(f"    {word:>12s}: {', '.join(prims)}  {exp_str}")

    # Save results
    results = {
        'config': vars(args),
        'model_params': total_params,
        'training_time_s': elapsed,
        'final_lang_loss': lang_loss.item(),
        'subsumption_train': tr_rate,
        'subsumption_test': te_rate,
        'subsumption_train_n': tr_n,
        'subsumption_test_n': te_n,
        'primary_accuracy': pq['primary_accuracy'],
        'dead_bits': pq['dead_bits'],
        'mean_entropy': pq['mean_entropy'],
        'per_category': {k: v for k, v in pq['per_category'].items()},
        'sup_train_acc': sup_train_acc,
        'sup_test_acc': sup_test_acc,
        'sup_train_words': len(sup_valid),
        'sup_test_words': len(sup_test_valid),
    }
    results_path = os.path.join(PROJECT_ROOT, 'playground', 'results', 'concept_gpt_49bit.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results: {results_path}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Training log: {csv_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

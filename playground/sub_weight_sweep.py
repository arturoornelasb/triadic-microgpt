"""
E4 — Subsumption Loss Weight Sweep at XL Scale.

Tests sub_weight in {0.5, 1.0, 2.0, 5.0} to find the optimal PPL-subsumption
tradeoff.  Each weight trains a full XL model (12L/512D/8H/64bits, 40M params)
for 50K steps with evaluations at both 25K and 50K (P12 showed 25K may be
optimal due to overfitting in the second half).

Architecture: standard TriadicGPT (tanh activation, 64 bits).
Loss: L_lang + alpha * (L_triadic + sub_weight * L_sub)

Subsumption pairs follow the same sets used in P6/P12:
  Train (45 pairs): 7 hypernyms -> 45 hyponyms
  Test  (12 pairs): 4 hypernyms -> 12 held-out hyponyms

Usage:
  python playground/sub_weight_sweep.py --weight 1.0          # single weight
  python playground/sub_weight_sweep.py --all                  # all 4 weights (~5h)
  python playground/sub_weight_sweep.py --aggregate-only       # table from saved results
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


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORY_SEPARATOR = '<' + '|endoftext|' + '>'

SWEEP_WEIGHTS = [0.5, 1.0, 2.0, 5.0]

# Run 15 baseline (XL, tanh, no sub loss) for comparison
RUN15_BASELINE = {
    'loss': 0.946,
    'entropy': 0.749,
    'semantic_gap': 0.020,
    'dead_bits': 15,
    'ppl': 7.69,
    'subsumption_train': 0.0,
    'subsumption_test': 0.0,
}


# ============================================================
# Hypernym-Hyponym pairs (same as P6/P12)
# ============================================================

TRAIN_PAIRS = {
    "animal":  ["dog", "cat", "bird", "fish", "bear", "horse", "rabbit", "frog", "mouse"],
    "person":  ["king", "queen", "princess", "prince", "mother", "father", "brother", "sister"],
    "color":   ["red", "blue", "green", "yellow", "pink", "purple", "black", "white"],
    "feeling": ["happy", "sad", "angry", "scared", "brave"],
    "food":    ["cake", "candy", "cookie", "apple", "bread"],
    "place":   ["house", "castle", "school", "hospital", "garden"],
    "time":    ["morning", "night", "today", "tomorrow"],
}

TEST_PAIRS = {
    "animal": ["tiger", "lion", "elephant", "wolf"],
    "person": ["man", "woman", "baby"],
    "food":   ["pizza", "cheese", "milk"],
    "place":  ["church", "forest", "park"],  # castle removed (was in train)
}


# ============================================================
# Utilities
# ============================================================

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def progress_bar(current, total, width=30):
    pct = current / max(total, 1)
    filled = int(width * pct)
    return f"[{'#' * filled}{'-' * (width - filled)}] {pct:5.1%}"


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


# ============================================================
# Subsumption data preparation
# ============================================================

def prepare_subsumption_data(tokenizer, device, pairs_dict):
    """Pre-encode hypernym-hyponym pairs as token tensors."""
    sub_pairs = []
    skipped = []
    for hypernym, hyponyms in pairs_dict.items():
        hyper_ids = tokenizer.encode(hypernym, add_special=False)
        if not hyper_ids:
            skipped.append(hypernym)
            continue
        hyper_tensor = torch.tensor(hyper_ids, dtype=torch.long, device=device)
        for hyponym in hyponyms:
            hypo_ids = tokenizer.encode(hyponym, add_special=False)
            if not hypo_ids:
                skipped.append(hyponym)
                continue
            hypo_tensor = torch.tensor(hypo_ids, dtype=torch.long, device=device)
            sub_pairs.append({
                'hypernym': hypernym,
                'hyponym': hyponym,
                'hyper_ids': hyper_tensor,
                'hypo_ids': hypo_tensor,
            })
    if skipped:
        print(f"  Skipped (not in vocab): {skipped}")
    print(f"  Prepared {len(sub_pairs)} hypernym-hyponym pairs")
    return sub_pairs


# ============================================================
# Subsumption loss (sampled, batched)
# ============================================================

def compute_subsumption_loss(model, sub_pairs, device, n_sample=32):
    """
    Subsumption loss: relu(hyper_01 - hypo_01).mean()

    Samples n_sample pairs per step for efficiency.
    hyper_01/hypo_01 are (tanh(x) + 1) / 2 mapped to [0, 1].
    """
    if not sub_pairs:
        return torch.tensor(0.0, device=device)

    if len(sub_pairs) > n_sample:
        indices = random.sample(range(len(sub_pairs)), n_sample)
        sampled = [sub_pairs[i] for i in indices]
    else:
        sampled = sub_pairs

    losses = []
    for pair in sampled:
        hyper_x = pair['hyper_ids'].unsqueeze(0)
        hypo_x = pair['hypo_ids'].unsqueeze(0)

        _, proj_hyper, _ = model(hyper_x)
        _, proj_hypo, _ = model(hypo_x)

        # Mean-pool over token positions -> (n_bits,)
        h = proj_hyper[0].mean(dim=0)
        y = proj_hypo[0].mean(dim=0)

        # Map tanh [-1,1] to [0,1]
        h_01 = (h + 1) / 2
        y_01 = (y + 1) / 2

        # Penalize when hypernym bit > hyponym bit
        loss = F.relu(h_01 - y_01).mean()
        losses.append(loss)

    return torch.stack(losses).mean()


# ============================================================
# Evaluation: subsumption rate (algebraic)
# ============================================================

@torch.no_grad()
def evaluate_subsumption(model, tokenizer, device, pairs_dict, mapper, label=""):
    """Measure algebraic subsumption: Phi(hypo) % Phi(hyper) == 0."""
    model.eval()

    all_words = set()
    for hyper, hypos in pairs_dict.items():
        all_words.add(hyper)
        all_words.update(hypos)

    sigs = {}
    projs = {}
    for word in all_words:
        ids = tokenizer.encode(word, add_special=False)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, proj, _ = model(x)
        proj_np = proj[0].mean(dim=0).cpu().numpy()
        projs[word] = proj_np
        sigs[word] = mapper.map(proj_np)

    total_pairs = 0
    subsumes_count = 0
    bit_inheritance_scores = []

    for hypernym, hyponyms in pairs_dict.items():
        if hypernym not in sigs:
            continue
        hyper_bits = (projs[hypernym] > 0).astype(int)
        for hyponym in hyponyms:
            if hyponym not in sigs:
                continue
            hypo_bits = (projs[hyponym] > 0).astype(int)
            total_pairs += 1

            is_subsumes = TriadicValidator.subsumes(sigs[hyponym], sigs[hypernym])
            if is_subsumes:
                subsumes_count += 1

            hyper_active = hyper_bits.sum()
            if hyper_active > 0:
                inherited = (hyper_bits * hypo_bits).sum()
                inheritance_rate = inherited / hyper_active
            else:
                inheritance_rate = 1.0
            bit_inheritance_scores.append(float(inheritance_rate))

    subsumption_rate = subsumes_count / max(total_pairs, 1)
    mean_inheritance = float(np.mean(bit_inheritance_scores)) if bit_inheritance_scores else 0.0

    if label:
        print(f"    [{label}] Subsumption: {subsumes_count}/{total_pairs} "
              f"({subsumption_rate:.1%}), inheritance: {mean_inheritance:.1%}")

    model.train()
    return {
        'subsumption_rate': float(subsumption_rate),
        'mean_bit_inheritance': float(mean_inheritance),
        'total_pairs': total_pairs,
        'subsumes_count': subsumes_count,
    }


# ============================================================
# Evaluation: semantic gap, entropy, dead bits
# ============================================================

@torch.no_grad()
def evaluate_semantic(model, tokenizer, device, n_bits):
    """Compute semantic gap, dead bits, entropy, analogy verification."""
    model.eval()
    mapper = PrimeMapper(n_bits)

    concept_pairs = {
        'related': [
            ("king", "queen"), ("dog", "cat"), ("happy", "sad"),
            ("mother", "father"), ("sun", "moon"), ("hot", "cold"),
            ("love", "hate"), ("big", "small"), ("bird", "fish"),
            ("doctor", "hospital"), ("teacher", "school"),
            ("princess", "prince"), ("old", "young"),
        ],
        'unrelated': [
            ("king", "fish"), ("dog", "moon"), ("happy", "river"),
            ("mother", "blue"), ("sun", "cat"), ("hot", "queen"),
            ("bird", "school"), ("love", "tree"), ("big", "night"),
        ],
    }

    all_words = set()
    for group in concept_pairs.values():
        for w1, w2 in group:
            all_words.update([w1, w2])

    sigs = {}
    for word in all_words:
        ids = tokenizer.encode(word, add_special=False)
        if ids:
            x = torch.tensor([ids], dtype=torch.long, device=device)
            _, proj, _ = model(x)
            sigs[word] = proj[0].mean(dim=0).cpu().numpy()

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    related_sims = [cosine(sigs[w1], sigs[w2])
                    for w1, w2 in concept_pairs['related'] if w1 in sigs and w2 in sigs]
    random_sims = []
    words = list(sigs.keys())
    rng = random.Random(42)
    for _ in range(200):
        i, j = rng.sample(range(len(words)), 2)
        random_sims.append(cosine(sigs[words[i]], sigs[words[j]]))

    semantic_gap = float(np.mean(related_sims) - np.mean(random_sims))

    # Bit entropy and dead bits
    all_projs = np.stack(list(sigs.values()))
    bit_means = (all_projs > 0).mean(axis=0)
    eps = 1e-7
    bit_entropy = -(bit_means * np.log2(bit_means + eps) +
                    (1 - bit_means) * np.log2(1 - bit_means + eps))
    dead_bits = int((bit_entropy < 0.3).sum())
    mean_entropy = float(bit_entropy.mean())

    model.train()
    return {
        'semantic_gap': semantic_gap,
        'mean_bit_entropy': mean_entropy,
        'dead_bits': dead_bits,
        'active_bits': n_bits - dead_bits,
    }


# ============================================================
# Evaluation: perplexity
# ============================================================

@torch.no_grad()
def compute_perplexity(model, tokenizer, data_path, device, block_size, max_samples=200):
    """Compute PPL on held-out stories (last max_samples from corpus)."""
    model.eval()
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 50]
    val_stories = stories[-max_samples:]

    total_loss = 0.0
    total_tokens = 0
    for story in val_stories:
        ids = tokenizer.encode(story, add_special=True)
        if len(ids) < 3:
            continue
        ids = ids[:block_size + 1]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        _, _, loss = model(x, targets=y)
        total_loss += loss.item() * (len(ids) - 1)
        total_tokens += len(ids) - 1

    avg_loss = total_loss / max(total_tokens, 1)
    model.train()
    return math.exp(avg_loss), avg_loss


# ============================================================
# Full evaluation snapshot at a given step
# ============================================================

@torch.no_grad()
def full_eval(model, tokenizer, device, data_path, block_size, n_bits,
              train_pairs_dict, test_pairs_dict, mapper, step_label):
    """Run all evaluations and return a dict of metrics."""
    print(f"\n  --- Evaluation @ {step_label} ---")

    ppl, avg_loss = compute_perplexity(model, tokenizer, data_path, device, block_size)
    print(f"    PPL: {ppl:.2f} (avg loss: {avg_loss:.4f})")

    sem = evaluate_semantic(model, tokenizer, device, n_bits)
    print(f"    Semantic gap: {sem['semantic_gap']:+.4f}")
    print(f"    Dead bits: {sem['dead_bits']}/{n_bits}, entropy: {sem['mean_bit_entropy']:.3f}")

    sub_train = evaluate_subsumption(model, tokenizer, device, train_pairs_dict, mapper, "train")
    sub_test = evaluate_subsumption(model, tokenizer, device, test_pairs_dict, mapper, "test")

    return {
        'ppl': ppl,
        'avg_loss': avg_loss,
        'semantic_gap': sem['semantic_gap'],
        'mean_bit_entropy': sem['mean_bit_entropy'],
        'dead_bits': sem['dead_bits'],
        'active_bits': sem['active_bits'],
        'sub_train_rate': sub_train['subsumption_rate'],
        'sub_train_inheritance': sub_train['mean_bit_inheritance'],
        'sub_train_n': sub_train['total_pairs'],
        'sub_test_rate': sub_test['subsumption_rate'],
        'sub_test_inheritance': sub_test['mean_bit_inheritance'],
        'sub_test_n': sub_test['total_pairs'],
    }


# ============================================================
# Train one weight configuration
# ============================================================

def train_single_weight(sub_weight, args):
    """Train XL model with a given sub_weight, evaluate at 25K and 50K."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_bits = 64
    steps = args.steps
    block_size = args.block
    batch_size = args.batch_size

    # Output dirs — include warmup pct to separate 80% vs 50% runs
    warmup_tag = f'warmup{int(args.warmup_pct * 100)}'
    sweep_name = 'sub_weight_sweep' if args.warmup_pct == 0.80 else f'sub_weight_sweep_{warmup_tag}'
    weight_dir = os.path.join(PROJECT_ROOT, 'playground', 'results',
                              sweep_name, f'weight_{sub_weight}')
    ckpt_dir = os.path.join(weight_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print()
    print("=" * 70)
    print(f"  SUB WEIGHT SWEEP — weight={sub_weight}")
    print("=" * 70)
    print(f"  Device:     {device}")
    if device.type == 'cuda':
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Scale:      XL (12L/512D/8H/64bits)")
    print(f"  Steps:      {steps}")
    print(f"  Batch:      {batch_size}")
    print(f"  Block:      {block_size}")
    print(f"  Sub weight: {sub_weight}")
    print(f"  Alpha:      {args.alpha}")
    print(f"  Align:      {args.align_weight} (MSE)")
    print(f"  Entropy:    {args.entropy_weight}")
    print(f"  Warmup:     {args.warmup_pct:.0%}")
    print()

    # -- Paths --
    data_path = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
    tokenizer_path = os.path.join(PROJECT_ROOT, 'checkpoints',
                                  'torch_run15_strongalign', 'tokenizer.json')

    # -- Tokenizer --
    print("[1/6] Loading tokenizer (Run 15)...")
    tokenizer = BPETokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab: {vocab_size}")

    # -- Tokenize corpus --
    print()
    print("[2/6] Tokenizing corpus...")
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 30]
    random.seed(42)
    random.shuffle(stories)
    stories = stories[:50000]
    print(f"  Stories: {len(stories)}")

    all_tokens = []
    t0 = time.time()
    for i, story in enumerate(stories):
        ids = tokenizer.encode(story, add_special=True)
        all_tokens.extend(ids)
        if (i + 1) % 10000 == 0:
            print(f"  Encoded {i+1}/{len(stories)} ({len(all_tokens):,} tokens)")
    print(f"  Total: {len(all_tokens):,} tokens ({time.time()-t0:.1f}s)")

    # -- Model --
    print()
    print("[3/6] Initializing TriadicGPT (XL, tanh, 64 bits)...")
    config = TriadicGPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=12,
        n_embd=512,
        n_head=8,
        n_triadic_bits=n_bits,
        dropout=0.1,
    )
    model = TriadicGPT(config).to(device)
    total_params = model.num_params()
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # -- Subsumption pairs --
    print()
    print("[4/6] Preparing subsumption pairs...")
    print("  Training pairs:")
    train_sub_pairs = prepare_subsumption_data(tokenizer, device, TRAIN_PAIRS)
    print("  Test pairs:")
    test_sub_pairs = prepare_subsumption_data(tokenizer, device, TEST_PAIRS)

    mapper = PrimeMapper(n_bits)

    # -- Optimizer --
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01, betas=(0.9, 0.95))
    dataset = TextDataset(all_tokens, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=0)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    triadic_warmup = int(steps * args.warmup_pct)

    # Compute meaningful eval points based on warmup
    # mid_eval_step = halfway through active triadic training
    active_triadic_steps = steps - triadic_warmup
    mid_eval_step = triadic_warmup + active_triadic_steps // 2  # e.g. 37.5K for 50% warmup

    # -- CSV log --
    csv_path = os.path.join(weight_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'lang_loss', 'tri_loss', 'sub_loss',
                         'lr', 'entropy', 'dead_bits', 'elapsed_s'])

    # -- Training --
    print()
    print(f"[5/6] Training for {steps} steps (sub_weight={sub_weight})...")
    print(f"  Triadic activation at step {triadic_warmup}")
    print(f"  Alpha ramp: {triadic_warmup} -> {triadic_warmup + int(steps * 0.2)} (full weight)")
    print(f"  Subsumption loss every 5 steps")
    print(f"  Eval points: step {mid_eval_step} (mid, {mid_eval_step - triadic_warmup} active) + step {steps} (end, {steps - triadic_warmup} active)")
    print("-" * 70)

    model.train()
    data_iter = iter(dataloader)
    start_time = time.time()
    best_loss = float('inf')
    eval_25k = None
    eval_50k = None
    eval_mid = None  # midpoint of active triadic training

    for step in range(steps):
        # Get batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        # LR schedule: cosine with warmup
        warmup_steps = min(500, steps // 10)
        if step < warmup_steps:
            lr_t = args.lr * (step + 1) / warmup_steps
        else:
            prog = (step - warmup_steps) / max(steps - warmup_steps, 1)
            lr_t = args.lr * max(0.1, 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        # Forward
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            total_loss = lang_loss
            tri_loss_val = 0.0
            sub_loss_val = 0.0

            if step >= triadic_warmup:
                # Alpha ramp: linear from warmup to warmup + 20%
                alpha_warmup_steps = int(steps * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_warmup_steps)
                current_alpha = args.alpha * alpha_factor

                # Standard triadic loss (diversity + contrastive + entropy + alignment)
                tri_loss = model.triadic_loss(
                    triadic_proj,
                    entropy_weight=args.entropy_weight,
                    input_ids=x,
                    align_weight=args.align_weight,
                    align_mode='mse',
                )
                total_loss = lang_loss + current_alpha * tri_loss
                tri_loss_val = tri_loss.item()

                # Subsumption loss (every 5 steps to save compute)
                if step % 5 == 0:
                    sub_loss = compute_subsumption_loss(
                        model, train_sub_pairs, device, n_sample=32)
                    total_loss = total_loss + current_alpha * sub_weight * sub_loss
                    sub_loss_val = sub_loss.item()

        # Backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # CSV logging + console
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                flat = triadic_proj.reshape(-1, n_bits)
                bm = (flat > 0).float().mean(dim=0)
                eps = 1e-7
                ent = -(bm * (bm + eps).log2() + (1 - bm) * (1 - bm + eps).log2())
                mean_ent = ent.mean().item()
                dead = int((ent < 0.3).sum().item())

            elapsed = time.time() - start_time
            csv_writer.writerow([
                step + 1, f"{lang_loss.item():.4f}", f"{tri_loss_val:.4f}",
                f"{sub_loss_val:.6f}", f"{lr_t:.6f}",
                f"{mean_ent:.4f}", dead, f"{elapsed:.0f}",
            ])
            csv_file.flush()

            if step % 200 == 0 or step == steps - 1:
                steps_done = step + 1
                speed = steps_done / max(elapsed, 1)
                eta_s = (steps - steps_done) / max(speed, 0.01)
                bar = progress_bar(steps_done, steps)
                tri_phase = "ON " if step >= triadic_warmup else "off"

                print(f"  {bar}  step {step+1:>6d}/{steps}  "
                      f"loss={lang_loss.item():.3f}  tri[{tri_phase}]={tri_loss_val:.4f}  "
                      f"sub={sub_loss_val:.4f}  ent={mean_ent:.3f}  dead={dead}/64  "
                      f"ETA {format_time(eta_s)}  [{format_time(elapsed)}]")

        # Checkpoint + eval at midpoint of active triadic training
        if step + 1 == mid_eval_step:
            elapsed = time.time() - start_time
            ckpt_path = os.path.join(ckpt_dir, f'model_step{step+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'vocab_size': config.vocab_size, 'block_size': config.block_size,
                    'n_layer': config.n_layer, 'n_embd': config.n_embd,
                    'n_head': config.n_head, 'n_triadic_bits': config.n_triadic_bits,
                },
                'step': step + 1,
                'loss': lang_loss.item(),
                'sub_weight': sub_weight,
                'elapsed_s': elapsed,
            }, ckpt_path)
            active_steps_so_far = step + 1 - triadic_warmup
            print(f"  >>> Checkpoint saved: {ckpt_path}")
            print(f"  >>> Mid-eval: {active_steps_so_far} active triadic steps")

            eval_mid = full_eval(
                model, tokenizer, device, data_path, block_size, n_bits,
                TRAIN_PAIRS, TEST_PAIRS, mapper,
                f"step {step+1} (mid, {active_steps_so_far} active)")
            model.train()

        # Best checkpoint tracking
        if (step + 1) % 5000 == 0:
            if lang_loss.item() < best_loss:
                best_loss = lang_loss.item()

    # Final checkpoint
    elapsed = time.time() - start_time
    ckpt_path = os.path.join(ckpt_dir, f'model_step{steps}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': config.vocab_size, 'block_size': config.block_size,
            'n_layer': config.n_layer, 'n_embd': config.n_embd,
            'n_head': config.n_head, 'n_triadic_bits': config.n_triadic_bits,
        },
        'step': steps,
        'loss': lang_loss.item(),
        'sub_weight': sub_weight,
        'elapsed_s': elapsed,
    }, ckpt_path)
    print(f"  >>> Final checkpoint: {ckpt_path}")

    csv_file.close()

    # -- Final eval (50K) --
    print()
    print("[6/6] Final evaluation...")
    eval_50k = full_eval(
        model, tokenizer, device, data_path, block_size, n_bits,
        TRAIN_PAIRS, TEST_PAIRS, mapper, f"step {steps} (50K)")

    # -- Save results --
    elapsed = time.time() - start_time
    mid_label = f'step_{mid_eval_step}'
    results = {
        'experiment': 'sub_weight_sweep',
        'sub_weight': sub_weight,
        'config': '12L/512D/8H/64bits (XL, 40M params)',
        'steps': steps,
        'batch_size': batch_size,
        'block_size': block_size,
        'alpha': args.alpha,
        'entropy_weight': args.entropy_weight,
        'align_weight': args.align_weight,
        'warmup_pct': args.warmup_pct,
        'triadic_warmup_step': triadic_warmup,
        'mid_eval_step': mid_eval_step,
        'active_triadic_steps_at_mid': mid_eval_step - triadic_warmup,
        'active_triadic_steps_at_end': steps - triadic_warmup,
        'training_time_min': elapsed / 60,
        'total_params': total_params,
        'best_lang_loss': best_loss,
        'eval_mid': eval_mid,
        'eval_50k': eval_50k,
    }

    results_path = os.path.join(weight_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # -- Per-weight summary --
    active_mid = mid_eval_step - triadic_warmup
    active_end = steps - triadic_warmup
    print()
    print("=" * 70)
    print(f"  WEIGHT {sub_weight} — RESULTS (warmup={args.warmup_pct:.0%})")
    print("=" * 70)
    print(f"  {'Metric':>25s}  {'@mid('+str(active_mid)+'act)':>14s}  {'@end('+str(active_end)+'act)':>14s}  {'Run 15':>12s}")
    print(f"  {'='*25}  {'='*14}  {'='*14}  {'='*12}")

    if eval_mid and eval_50k:
        rows = [
            ('PPL', eval_mid['ppl'], eval_50k['ppl'], RUN15_BASELINE['ppl']),
            ('Semantic gap', eval_mid['semantic_gap'], eval_50k['semantic_gap'],
             RUN15_BASELINE['semantic_gap']),
            ('Dead bits', eval_mid['dead_bits'], eval_50k['dead_bits'],
             RUN15_BASELINE['dead_bits']),
            ('Entropy', eval_mid['mean_bit_entropy'], eval_50k['mean_bit_entropy'],
             RUN15_BASELINE['entropy']),
            ('Sub train', eval_mid['sub_train_rate'], eval_50k['sub_train_rate'],
             RUN15_BASELINE['subsumption_train']),
            ('Sub test', eval_mid['sub_test_rate'], eval_50k['sub_test_rate'],
             RUN15_BASELINE['subsumption_test']),
        ]
        for name, v_mid, v50, v_run15 in rows:
            print(f"  {name:>25s}  {v_mid:>14.4f}  {v50:>14.4f}  {v_run15:>12.4f}")

    print(f"\n  Training time: {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")
    print(f"  CSV log: {csv_path}")
    print(f"  Checkpoints: {ckpt_dir}")
    print("=" * 70)

    return results


# ============================================================
# Aggregate: comparison table from saved results
# ============================================================

def aggregate_results(warmup_pct=0.80):
    """Load results from all weights and print a comparison table."""
    warmup_tag = f'warmup{int(warmup_pct * 100)}'
    sweep_name = 'sub_weight_sweep' if warmup_pct == 0.80 else f'sub_weight_sweep_{warmup_tag}'
    sweep_dir = os.path.join(PROJECT_ROOT, 'playground', 'results', sweep_name)

    all_results = {}
    for w in SWEEP_WEIGHTS:
        result_path = os.path.join(sweep_dir, f'weight_{w}', 'results.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                all_results[w] = json.load(f)
        else:
            print(f"  WARNING: No results for weight={w} at {result_path}")

    if not all_results:
        print("  No results found. Run --all or individual --weight first.")
        return

    print()
    print("=" * 110)
    print(f"  E4: SUBSUMPTION WEIGHT SWEEP — COMPARISON TABLE (warmup={warmup_pct:.0%})")
    print("=" * 110)

    # Header
    cols = ['Weight', 'PPL@mid', 'PPL@end', 'SubTr@mid', 'SubTe@mid',
            'SubTr@end', 'SubTe@end', 'Gap@end', 'Dead@end', 'Time(min)']
    header = "  "
    for col in cols:
        header += f"{col:>14s}"
    print(header)
    print("  " + "-" * (14 * len(cols)))

    # Run 15 baseline row
    row = f"  {'Run15(ref)':>14s}"
    row += f"{RUN15_BASELINE['ppl']:>14.2f}"
    row += f"{RUN15_BASELINE['ppl']:>14.2f}"
    row += f"{RUN15_BASELINE['subsumption_train']:>14.1%}"
    row += f"{RUN15_BASELINE['subsumption_test']:>14.1%}"
    row += f"{RUN15_BASELINE['subsumption_train']:>14.1%}"
    row += f"{RUN15_BASELINE['subsumption_test']:>14.1%}"
    row += f"{RUN15_BASELINE['semantic_gap']:>14.4f}"
    row += f"{RUN15_BASELINE['dead_bits']:>14d}"
    row += f"{'N/A':>14s}"
    print(row)

    # Data rows — support both old (eval_25k) and new (eval_mid) formats
    for w in SWEEP_WEIGHTS:
        if w not in all_results:
            continue
        r = all_results[w]
        e_mid = r.get('eval_mid') or r.get('eval_25k', {})
        e50 = r.get('eval_50k', {})
        t_min = r.get('training_time_min', 0)

        row = f"  {w:>14.1f}"
        row += f"{e_mid.get('ppl', 0):>14.2f}"
        row += f"{e50.get('ppl', 0):>14.2f}"
        row += f"{e_mid.get('sub_train_rate', 0):>14.1%}"
        row += f"{e_mid.get('sub_test_rate', 0):>14.1%}"
        row += f"{e50.get('sub_train_rate', 0):>14.1%}"
        row += f"{e50.get('sub_test_rate', 0):>14.1%}"
        row += f"{e50.get('semantic_gap', 0):>14.4f}"
        row += f"{e50.get('dead_bits', 0):>14d}"
        row += f"{t_min:>14.1f}"
        print(row)

    print("  " + "-" * (14 * len(cols)))

    # Best weight analysis (use eval_mid if available, else eval_25k)
    best_w = None
    best_score = -float('inf')
    for w, r in all_results.items():
        e_mid = r.get('eval_mid') or r.get('eval_25k', {})
        # Score: prioritize high held-out subsumption with low PPL penalty
        sub_test = e_mid.get('sub_test_rate', 0)
        ppl = e_mid.get('ppl', 100)
        ppl_penalty = max(0, (ppl - RUN15_BASELINE['ppl']) / RUN15_BASELINE['ppl'])
        score = sub_test - ppl_penalty
        if score > best_score:
            best_score = score
            best_w = w

    if best_w is not None:
        e_mid = all_results[best_w].get('eval_mid') or all_results[best_w].get('eval_25k', {})
        print(f"\n  Recommended weight (best sub@mid with minimal PPL cost): {best_w}")
        print(f"    Sub test@mid: {e_mid.get('sub_test_rate', 0):.1%}, "
              f"PPL@mid: {e_mid.get('ppl', 0):.2f} "
              f"(Run 15: {RUN15_BASELINE['ppl']:.2f})")

    # Save aggregate
    agg_path = os.path.join(sweep_dir, 'aggregate.json')
    agg = {
        'experiment': 'sub_weight_sweep',
        'description': 'E4: Subsumption Loss Weight Sweep at XL Scale',
        'weights_tested': list(all_results.keys()),
        'recommended_weight': best_w,
        'run15_baseline': RUN15_BASELINE,
        'results': {str(w): r for w, r in all_results.items()},
    }
    with open(agg_path, 'w') as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\n  Aggregate saved: {agg_path}")
    print("=" * 110)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='E4: Subsumption Loss Weight Sweep at XL Scale')
    parser.add_argument('--weight', type=float, default=None,
                        help='Single sub_weight to train')
    parser.add_argument('--all', action='store_true',
                        help='Train all 4 weights sequentially')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only print comparison table from saved results')

    # Training config (XL defaults matching Run 15)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--entropy-weight', type=float, default=1.0)
    parser.add_argument('--align-weight', type=float, default=5.0)
    parser.add_argument('--warmup-pct', type=float, default=0.80,
                        help='Triadic warmup fraction (default: 80%%)')
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_results(args.warmup_pct)
        return

    if args.all:
        print()
        print("*" * 70)
        print("  E4: FULL SWEEP — sub_weight in", SWEEP_WEIGHTS)
        print(f"  Warmup: {args.warmup_pct:.0%}")
        print(f"  Estimated total time: ~{len(SWEEP_WEIGHTS) * 76} min ({len(SWEEP_WEIGHTS) * 76 / 60:.1f}h)")
        print("*" * 70)

        all_run_results = {}
        for i, w in enumerate(SWEEP_WEIGHTS):
            print(f"\n  >>> Starting weight {w} ({i+1}/{len(SWEEP_WEIGHTS)}) <<<")
            result = train_single_weight(w, args)
            all_run_results[w] = result

            # Clear CUDA cache between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final aggregate
        aggregate_results(args.warmup_pct)

    elif args.weight is not None:
        train_single_weight(args.weight, args)
        # Print aggregate if other results exist
        warmup_tag = f'warmup{int(args.warmup_pct * 100)}'
        sweep_name = 'sub_weight_sweep' if args.warmup_pct == 0.80 else f'sub_weight_sweep_{warmup_tag}'
        sweep_dir = os.path.join(PROJECT_ROOT, 'playground', 'results', sweep_name)
        existing = sum(1 for w in SWEEP_WEIGHTS
                       if os.path.exists(os.path.join(sweep_dir, f'weight_{w}', 'results.json')))
        if existing > 1:
            print("\n  Multiple weight results available, printing aggregate:")
            aggregate_results(args.warmup_pct)

    else:
        parser.print_help()
        print("\n  Examples:")
        print("    python playground/sub_weight_sweep.py --weight 1.0")
        print("    python playground/sub_weight_sweep.py --all")
        print("    python playground/sub_weight_sweep.py --aggregate-only")


if __name__ == '__main__':
    main()

"""
XL Subsumption Loss — Validacion a 40M params del breakthrough P6.

Entrena un TriadicGPT XL (12L/512D/8H/64bits, 40M params) con subsumption loss
y compara contra Run 15 (mismo scale, sin sub loss).

Run 15 metrics (baseline):
  Loss: 0.946 | PPL: 7.69 | Gap: +0.020 | Dead: ~15 | Subsumption: 0%

Base scale results (P6, Sub 5.0):
  Subsumption: 100% train, 91.7% held-out | Inheritance: 100%/97.9%
  Language loss IMPROVED: 1.810 -> 1.707

Hipotesis: Sub loss at XL will maintain language quality while achieving
held-out subsumption > 80%, resolving the paper's main limitation.

Uso:
  conda activate triadic-microgpt
  python playground/xl_subsumption.py [--steps 50000] [--sub-weight 5.0]
"""

import os
import sys
import csv
import time
import math
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
from src.triadic import PrimeMapper, TriadicValidator
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Run 15 baseline (XL, tanh, no sub loss)
RUN15_BASELINE = {
    'loss': 0.946,
    'entropy': 0.749,
    'semantic_gap': 0.020,
    'dead_bits': 15,
    'ppl': 7.69,
    'subsumption_train': 0.0,
    'subsumption_test': 0.0,
}

STORY_SEPARATOR = '<' + '|endoftext|' + '>'

# ── Hypernym-Hyponym pairs (same as P6) ──
HYPERNYM_PAIRS = {
    "animal": ["dog", "cat", "bird", "fish", "horse", "rabbit", "bear", "mouse", "lion"],
    "person": ["king", "queen", "doctor", "teacher", "princess", "prince", "boy", "girl"],
    "feeling": ["happy", "sad", "love", "hate", "angry", "scared"],
    "food": ["apple", "cake", "bread", "candy", "cookie"],
    "color": ["red", "blue", "green", "yellow", "pink", "purple"],
    "place": ["school", "hospital", "house", "garden", "forest", "beach", "park"],
    "time": ["day", "night", "morning", "evening"],
}

HELD_OUT_PAIRS = {
    "animal": ["tiger", "frog", "deer"],
    "person": ["man", "woman", "baby"],
    "food": ["pizza", "milk", "egg"],
    "place": ["castle", "farm", "river"],
}


# ============================================================
# Utilities
# ============================================================

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def progress_bar(current, total, width=30):
    pct = current / max(total, 1)
    filled = int(width * pct)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:6.1%}"


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
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


# ============================================================
# Subsumption loss + evaluation (from P6)
# ============================================================

def prepare_subsumption_data(tokenizer, device, pairs_dict):
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
                'hypernym': hypernym, 'hyponym': hyponym,
                'hyper_ids': hyper_tensor, 'hypo_ids': hypo_tensor,
            })
    if skipped:
        print(f"  Skipped (not in vocab): {skipped}")
    print(f"  Prepared {len(sub_pairs)} hypernym-hyponym pairs")
    return sub_pairs


def compute_subsumption_loss(model, sub_pairs, device):
    if not sub_pairs:
        return torch.tensor(0.0, device=device)
    losses = []
    for pair in sub_pairs:
        hyper_x = pair['hyper_ids'].unsqueeze(0)
        hypo_x = pair['hypo_ids'].unsqueeze(0)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            _, proj_hyper, _ = model(hyper_x)
            _, proj_hypo, _ = model(hypo_x)
        h = proj_hyper[0].mean(dim=0)
        y = proj_hypo[0].mean(dim=0)
        loss = F.relu(h - y).mean()
        losses.append(loss)
    return torch.stack(losses).mean()


def evaluate_subsumption(model, tokenizer, device, pairs_dict, mapper, label=""):
    model.eval()
    all_words = set()
    for hyper, hypos in pairs_dict.items():
        all_words.add(hyper)
        all_words.update(hypos)

    sigs = {}
    projs = {}
    with torch.no_grad():
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            _, proj, _ = model(x)
            proj_np = proj[0].mean(dim=0).cpu().numpy()
            projs[word] = proj_np
            sigs[word] = mapper.map(proj_np)

    results = []
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
            results.append({
                'pair': f'{hypernym}->{hyponym}',
                'subsumes': bool(is_subsumes),
                'bit_inheritance': float(inheritance_rate),
                'hyper_active_bits': int(hyper_active),
                'shared_bits': int((hyper_bits * hypo_bits).sum()),
            })

    subsumption_rate = subsumes_count / max(total_pairs, 1)
    mean_inheritance = np.mean(bit_inheritance_scores) if bit_inheritance_scores else 0.0

    if label:
        print(f"\n  [{label}] Subsumption Results:")
        print(f"    Algebraic subsumption: {subsumes_count}/{total_pairs} ({subsumption_rate:.1%})")
        print(f"    Mean bit inheritance:  {mean_inheritance:.1%}")
        for hypernym in pairs_dict:
            pair_results = [r for r in results if r['pair'].startswith(f'{hypernym}->')]
            if pair_results:
                h_inherit = np.mean([r['bit_inheritance'] for r in pair_results])
                h_sub = sum(r['subsumes'] for r in pair_results)
                h_bits = pair_results[0]['hyper_active_bits']
                print(f"      {hypernym:>10s}: inheritance={h_inherit:.0%}  "
                      f"subsumption={h_sub}/{len(pair_results)}  "
                      f"hyper_bits={h_bits}")

    return {
        'subsumption_rate': float(subsumption_rate),
        'mean_bit_inheritance': float(mean_inheritance),
        'total_pairs': total_pairs,
        'details': results,
    }


# ============================================================
# Semantic evaluation (gap, analogies, entropy)
# ============================================================

def evaluate_model(model, tokenizer, device, n_bits):
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

    analogy_triples = [
        ("king", "queen", "man", "woman"),
        ("father", "mother", "brother", "sister"),
        ("father", "mother", "son", "daughter"),
        ("dog", "puppy", "cat", "kitten"),
        ("big", "small", "tall", "short"),
        ("hot", "cold", "day", "night"),
        ("happy", "sad", "love", "hate"),
        ("princess", "prince", "queen", "king"),
        ("bird", "fly", "fish", "swim"),
        ("old", "young", "big", "small"),
        ("doctor", "hospital", "teacher", "school"),
        ("sun", "day", "moon", "night"),
        ("red", "blue", "green", "yellow"),
    ]

    all_words = set()
    for group in concept_pairs.values():
        for w1, w2 in group:
            all_words.update([w1, w2])
    for a, b, c, d in analogy_triples:
        all_words.update([a, b, c, d])

    sigs = {}
    with torch.no_grad():
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
    unrelated_sims = [cosine(sigs[w1], sigs[w2])
                      for w1, w2 in concept_pairs['unrelated'] if w1 in sigs and w2 in sigs]
    random_sims = []
    words = list(sigs.keys())
    for _ in range(200):
        i, j = random.sample(range(len(words)), 2)
        random_sims.append(cosine(sigs[words[i]], sigs[words[j]]))

    semantic_gap = np.mean(related_sims) - np.mean(random_sims)

    correct = 0
    total = 0
    for a, b, c, d in analogy_triples:
        if not all(w in sigs for w in [a, b, c, d]):
            continue
        phi_a, phi_b = mapper.map(sigs[a]), mapper.map(sigs[b])
        phi_c, phi_d = mapper.map(sigs[c]), mapper.map(sigs[d])
        predicted = TriadicValidator.analogy(phi_a, phi_b, phi_c)
        if TriadicValidator.similarity(predicted, phi_d) > 0.3:
            correct += 1
        total += 1

    all_projs = np.stack(list(sigs.values()))
    bit_means = (all_projs > 0).mean(axis=0)
    eps = 1e-7
    bit_entropy = -(bit_means * np.log2(bit_means + eps) +
                    (1 - bit_means) * np.log2(1 - bit_means + eps))
    dead_bits = int((bit_entropy < 0.3).sum())
    unique = len(set(mapper.map(p) for p in all_projs))

    return {
        'semantic_gap': float(semantic_gap),
        'related_vs_unrelated': float(np.mean(related_sims) - np.mean(unrelated_sims)),
        'mean_related_sim': float(np.mean(related_sims)),
        'mean_random_sim': float(np.mean(random_sims)),
        'analogy_verification': correct / max(total, 1),
        'mean_bit_entropy': float(bit_entropy.mean()),
        'dead_bits': dead_bits,
        'active_bits': n_bits - dead_bits,
        'unique_signatures': unique,
    }


def compute_perplexity(model, tokenizer, data_path, device, block_size, max_samples=200):
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 50]
    val_stories = stories[-max_samples:]
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
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
    return math.exp(avg_loss), avg_loss


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='XL Subsumption Loss Experiment')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--block', type=int, default=512)
    parser.add_argument('--sub-weight', type=float, default=5.0, help='Subsumption loss weight')
    parser.add_argument('--alpha', type=float, default=0.05, help='Triadic loss weight')
    parser.add_argument('--entropy-weight', type=float, default=1.0)
    parser.add_argument('--align-weight', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup-pct', type=float, default=0.25)
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume from')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 70)
    print("  XL SUBSUMPTION LOSS — P6 Breakthrough → Production Validation")
    print("=" * 70)
    print(f"  Device:     {device}")
    if device.type == 'cuda':
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Scale:      XL (12L/512D/8H/64bits)")
    print(f"  Steps:      {args.steps}")
    print(f"  Batch:      {args.batch_size}")
    print(f"  Block:      {args.block}")
    print(f"  Sub weight: {args.sub_weight}")
    print(f"  Alpha:      {args.alpha}")
    print(f"  Align:      {args.align_weight} (MSE)")
    print(f"  Entropy:    {args.entropy_weight}")
    print()
    print(f"  Run 15 (no sub loss) to beat:")
    print(f"    Loss={RUN15_BASELINE['loss']:.3f}  PPL={RUN15_BASELINE['ppl']:.2f}  "
          f"Gap=+{RUN15_BASELINE['semantic_gap']:.3f}  Dead={RUN15_BASELINE['dead_bits']}  "
          f"Sub=0%")
    print()
    print(f"  Base scale target (P6, Sub 5.0):")
    print(f"    Sub(train)=100%  Sub(test)=91.7%  Inheritance=100%/97.9%")
    print()

    # ── Paths ──
    data_path = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
    tokenizer_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'playground', 'checkpoints_xl_subsumption')
    results_dir = os.path.join(PROJECT_ROOT, 'playground', 'results')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ── Tokenizer ──
    print("[1/6] Loading tokenizer (Run 15's)...")
    tokenizer = BPETokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab: {vocab_size}")

    # ── Tokenize corpus ──
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

    # ── Model ──
    print()
    print("[3/6] Initializing TriadicGPT (XL, tanh)...")
    config = TriadicGPTConfig(
        vocab_size=vocab_size, block_size=args.block,
        n_layer=12, n_embd=512, n_head=8,
        n_triadic_bits=64, dropout=0.1,
    )
    model = TriadicGPT(config).to(device)
    total_params = model.num_params()
    print(f"  Parameters: {total_params:,}")

    # ── Subsumption pairs ──
    print()
    print("[4/6] Preparing subsumption pairs...")
    print("  Training pairs:")
    train_pairs = prepare_subsumption_data(tokenizer, device, HYPERNYM_PAIRS)
    print("  Held-out pairs:")
    test_pairs = prepare_subsumption_data(tokenizer, device, HELD_OUT_PAIRS)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    dataset = TextDataset(all_tokens, args.block)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    mapper = PrimeMapper(64)

    triadic_warmup = int(args.steps * args.warmup_pct)

    # ── Resume ──
    start_step = 0
    best_loss = float('inf')
    prior_elapsed = 0.0
    if args.resume:
        print()
        print(f"  Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("  Optimizer state restored")
        if 'scaler_state_dict' in ckpt and device.type == 'cuda':
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_step = ckpt.get('step', 0) + 1
        best_loss = ckpt.get('loss', float('inf'))
        prior_elapsed = ckpt.get('elapsed_s', 0.0)
        print(f"  Resuming at step {start_step} (loss={ckpt.get('loss', '?'):.4f})")

    # ── CSV log ──
    csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
    if args.resume and os.path.exists(csv_path):
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([])
    else:
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'loss', 'tri_loss', 'sub_loss', 'lr', 'entropy', 'dead_bits', 'elapsed_s'])

    # ── Training ──
    print()
    remaining = args.steps - start_step
    print(f"[5/6] Training for {remaining} steps ({start_step}/{args.steps} done)...")
    print(f"  Triadic activation at step {triadic_warmup}")
    print(f"  Subsumption loss every 5 steps (weight={args.sub_weight})")
    print("-" * 80)

    model.train()
    data_iter = iter(dataloader)
    start_time = time.time()

    for step in range(start_step, args.steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # LR schedule: cosine with warmup
        warmup_steps = min(500, args.steps // 10)
        if step < warmup_steps:
            lr_t = args.lr * (step + 1) / warmup_steps
        else:
            prog = (step - warmup_steps) / max(args.steps - warmup_steps, 1)
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
                alpha_warmup_steps = int(args.steps * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_warmup_steps)
                current_alpha = args.alpha * alpha_factor

                tri_loss = model.triadic_loss(
                    triadic_proj, entropy_weight=args.entropy_weight,
                    input_ids=x, align_weight=args.align_weight, align_mode='mse',
                )
                total_loss = lang_loss + current_alpha * tri_loss
                tri_loss_val = tri_loss.item()

                # Subsumption loss (every 5 steps to save compute)
                if step % 5 == 0:
                    sub_loss = compute_subsumption_loss(model, train_pairs, device)
                    total_loss = total_loss + current_alpha * args.sub_weight * sub_loss
                    sub_loss_val = sub_loss.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % 50 == 0 or step == args.steps - 1:
            with torch.no_grad():
                flat = triadic_proj.reshape(-1, 64)
                bm = (flat > 0).float().mean(dim=0)
                eps = 1e-7
                ent = -(bm * (bm + eps).log2() + (1 - bm) * (1 - bm + eps).log2())
                mean_ent = ent.mean().item()
                dead = int((ent < 0.3).sum().item())

            session_elapsed = time.time() - start_time
            total_elapsed = prior_elapsed + session_elapsed

            csv_writer.writerow([step, f"{lang_loss.item():.4f}", f"{tri_loss_val:.4f}",
                                 f"{sub_loss_val:.6f}", f"{lr_t:.6f}",
                                 f"{mean_ent:.4f}", dead, f"{total_elapsed:.0f}"])
            csv_file.flush()

            if step % 100 == 0:
                steps_done = step - start_step + 1
                steps_remaining = args.steps - step - 1
                speed = steps_done / max(session_elapsed, 1)
                eta_s = steps_remaining / max(speed, 0.01)

                bar = progress_bar(step + 1, args.steps)
                tri_phase = "ON " if step >= triadic_warmup else "off"

                print(f"  {bar}  step {step:>6d}/{args.steps}  "
                      f"loss={lang_loss.item():.3f}  tri[{tri_phase}]={tri_loss_val:.4f}  "
                      f"sub={sub_loss_val:.4f}  ent={mean_ent:.3f}  "
                      f"dead={dead}/64  "
                      f"ETA {format_time(eta_s)}  [{format_time(session_elapsed)}]")

        # Checkpoint every 5K steps
        if (step + 1) % 5000 == 0 or step == args.steps - 1:
            loss_val = lang_loss.item()
            session_elapsed = time.time() - start_time
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': {
                    'vocab_size': config.vocab_size, 'block_size': config.block_size,
                    'n_layer': config.n_layer, 'n_embd': config.n_embd,
                    'n_head': config.n_head, 'n_triadic_bits': config.n_triadic_bits,
                },
                'step': step,
                'loss': loss_val,
                'sub_weight': args.sub_weight,
                'elapsed_s': prior_elapsed + session_elapsed,
            }
            path = os.path.join(checkpoint_dir, f'model_step{step+1}.pt')
            torch.save(ckpt, path)
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(ckpt, os.path.join(checkpoint_dir, 'model_best.pt'))
                print(f"  >>> New best: {loss_val:.4f} (saved)")

        # Mid-training subsumption check at 25K
        if step + 1 == args.steps // 2:
            print(f"\n  --- Mid-training subsumption check (step {step+1}) ---")
            eval_mid_train = evaluate_subsumption(model, tokenizer, device, HYPERNYM_PAIRS, mapper, "MID-train")
            eval_mid_test = evaluate_subsumption(model, tokenizer, device, HELD_OUT_PAIRS, mapper, "MID-test")
            model.train()
            print(f"  --- Continuing training ---\n")

    csv_file.close()
    total_time = time.time() - start_time
    print()
    print(f"  Training complete: {total_time/60:.1f} min")

    # ── Evaluation ──
    print()
    print("[6/6] Evaluating...")

    # Perplexity
    ppl, avg_loss = compute_perplexity(model, tokenizer, data_path, device, args.block)
    print(f"  Perplexity: {ppl:.2f} (Run 15: {RUN15_BASELINE['ppl']:.2f})")

    # Semantic metrics
    sem = evaluate_model(model, tokenizer, device, 64)
    print(f"  Semantic gap:  {sem['semantic_gap']:+.4f} (Run 15: +{RUN15_BASELINE['semantic_gap']:.3f})")
    print(f"  Dead bits:     {sem['dead_bits']} (Run 15: {RUN15_BASELINE['dead_bits']})")
    print(f"  Bit entropy:   {sem['mean_bit_entropy']:.4f} (Run 15: {RUN15_BASELINE['entropy']:.3f})")
    print(f"  Analogy verif: {sem['analogy_verification']:.1%}")

    # Subsumption
    print()
    print("  SUBSUMPTION EVALUATION:")
    eval_train = evaluate_subsumption(model, tokenizer, device, HYPERNYM_PAIRS, mapper, "FINAL-train")
    eval_test = evaluate_subsumption(model, tokenizer, device, HELD_OUT_PAIRS, mapper, "FINAL-test")

    # ── Comparison ──
    print()
    print("=" * 70)
    print("  XL SUBSUMPTION vs RUN 15 (NO SUB)")
    print("=" * 70)
    print(f"  {'Metric':>25s}  {'XL+Sub':>15s}  {'Run 15':>15s}  {'Delta':>10s}")
    print(f"  {'─'*25}  {'─'*15}  {'─'*15}  {'─'*10}")

    comparisons = [
        ('Best loss', best_loss, RUN15_BASELINE['loss']),
        ('Perplexity', ppl, RUN15_BASELINE['ppl']),
        ('Semantic gap', sem['semantic_gap'], RUN15_BASELINE['semantic_gap']),
        ('Dead bits', sem['dead_bits'], RUN15_BASELINE['dead_bits']),
        ('Bit entropy', sem['mean_bit_entropy'], RUN15_BASELINE['entropy']),
        ('Sub (train)', eval_train['subsumption_rate'], RUN15_BASELINE['subsumption_train']),
        ('Sub (test)', eval_test['subsumption_rate'], RUN15_BASELINE['subsumption_test']),
        ('Inheritance (train)', eval_train['mean_bit_inheritance'], 0.0),
        ('Inheritance (test)', eval_test['mean_bit_inheritance'], 0.0),
    ]

    for name, v_new, v_old in comparisons:
        delta = v_new - v_old
        sign = '+' if delta > 0 else ''
        print(f"  {name:>25s}  {v_new:>15.4f}  {v_old:>15.4f}  {sign}{delta:>9.4f}")

    # ── Save ──
    results = {
        'experiment': 'xl_subsumption',
        'source': 'P6 subsumption breakthrough -> XL validation',
        'config': '12L/512D/8H/64bits (40M params)',
        'steps': args.steps,
        'sub_weight': args.sub_weight,
        'training_time_min': total_time / 60,
        'best_loss': best_loss,
        'perplexity': ppl,
        **sem,
        'subsumption_train': eval_train,
        'subsumption_test': eval_test,
        'run15_baseline': RUN15_BASELINE,
    }

    results_path = os.path.join(results_dir, 'xl_subsumption.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results: {results_path}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  CSV log: {csv_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

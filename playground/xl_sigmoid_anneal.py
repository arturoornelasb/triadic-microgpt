"""
XL Sigmoid+Anneal — Validacion a 40M params del mejor hallazgo del playground.

Entrena un TriadicGPT XL (12L/512D/8H/64bits, 40M params) con sigmoid+anneal
en el triadic head y compara contra Run 15 (tanh, misma escala).

Run 15 metrics (tanh baseline):
  Loss: 0.946 | Entropy: 0.749 | Semantic gap: +0.020 | Dead bits: ~15

Hipotesis: sigmoid+anneal elimina dead bits y mejora semantic gap.

Uso:
  conda activate triadic-microgpt
  python playground/xl_sigmoid_anneal.py [--steps 50000] [--batch-size 64]
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

from src.torch_transformer import TriadicGPT, TriadicGPTConfig, TransformerBlock
from src.triadic import PrimeMapper, TriadicValidator
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Run 15 baseline numbers (tanh, XL, 50K steps)
# ============================================================
RUN15_BASELINE = {
    'loss': 0.946,
    'entropy': 0.749,
    'semantic_gap': 0.020,
    'dead_bits': 15,
    'ppl': 7.69,
}


# ============================================================
# Sigmoid+Anneal Model
# ============================================================

class SigmoidAnnealGPT(TriadicGPT):
    """TriadicGPT with sigmoid activation + temperature annealing.

    Instead of tanh(Wx), uses sigmoid(temp * Wx) mapped to [-1, 1].
    Temperature starts at 1.0 (soft) and anneals to final_temp (hard).
    This is the "quantum superposition → collapse" strategy from
    La Danza Cosmica (Cap. 14-16).
    """

    def __init__(self, config, final_temp=10.0):
        super().__init__(config)
        self.final_temp = final_temp
        self.current_temp = 1.0  # updated externally each step

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

        # Sigmoid+anneal triadic head (maps to [-1, 1])
        raw = self.triadic_head(x)
        triadic_proj = 2.0 * torch.sigmoid(raw * self.current_temp) - 1.0

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, triadic_proj, loss


# ============================================================
# Dataset (same as torch_train.py)
# ============================================================

STORY_SEPARATOR = '<' + '|endoftext|' + '>'

class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ============================================================
# Evaluation (semantic gap + entropy + analogies)
# ============================================================

def evaluate_model(model, tokenizer, device, n_bits):
    """Full evaluation matching Run 15 benchmarks."""
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

    # Collect all signatures
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

    # Semantic gap
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

    # Analogy verification
    correct = 0
    total = 0
    for a, b, c, d in analogy_triples:
        if not all(w in sigs for w in [a, b, c, d]):
            continue
        phi_a = mapper.map(sigs[a])
        phi_b = mapper.map(sigs[b])
        phi_c = mapper.map(sigs[c])
        phi_d = mapper.map(sigs[d])
        predicted = TriadicValidator.analogy(phi_a, phi_b, phi_c)
        # Verification: is predicted closer to D than to median?
        sim = TriadicValidator.similarity(predicted, phi_d)
        if sim > 0.3:
            correct += 1
        total += 1

    # Bit entropy from all signatures
    all_projs = np.stack(list(sigs.values()))
    bit_means = (all_projs > 0).mean(axis=0)
    eps = 1e-7
    bit_entropy = -(bit_means * np.log2(bit_means + eps) +
                    (1 - bit_means) * np.log2(1 - bit_means + eps))
    dead_bits = int((bit_entropy < 0.3).sum())

    # Unique signatures
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
    """Compute perplexity on held-out data."""
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
            logits, _, loss = model(x, targets=y)
            total_loss += loss.item() * (len(ids) - 1)
            total_tokens += len(ids) - 1

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss), avg_loss


# ============================================================
# Main
# ============================================================

def format_time(seconds):
    """Format seconds as HH:MM:SS or MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def progress_bar(current, total, width=30):
    """Render a text progress bar."""
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {pct:6.1%}"


def main():
    parser = argparse.ArgumentParser(description='XL Sigmoid+Anneal Experiment')
    parser.add_argument('--steps', type=int, default=50000, help='Training steps (default: 50000, same as Run 15)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (Run 15 used 64)')
    parser.add_argument('--final-temp', type=float, default=10.0, help='Final annealing temperature')
    parser.add_argument('--block', type=int, default=512, help='Context/block size')
    parser.add_argument('--alpha', type=float, default=0.05, help='Triadic loss weight')
    parser.add_argument('--entropy-weight', type=float, default=1.0, help='Entropy reg weight')
    parser.add_argument('--align-weight', type=float, default=5.0, help='Embedding alignment weight')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--warmup-pct', type=float, default=0.25, help='Triadic warmup fraction')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 70)
    print("  XL SIGMOID+ANNEAL — Playground Finding → Production Validation")
    print("=" * 70)
    print(f"  Device:     {device}")
    if device.type == 'cuda':
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Scale:      XL (12L/512D/8H/64bits)")
    print(f"  Steps:      {args.steps}")
    print(f"  Batch:      {args.batch_size}")
    print(f"  Block:      {args.block}")
    print(f"  Annealing:  temp 1.0 → {args.final_temp}")
    print(f"  Alpha:      {args.alpha}")
    print(f"  Align:      {args.align_weight} (MSE)")
    print(f"  Entropy:    {args.entropy_weight}")
    print()
    print(f"  Run 15 (tanh baseline) to beat:")
    print(f"    Loss={RUN15_BASELINE['loss']:.3f}  PPL={RUN15_BASELINE['ppl']:.2f}  "
          f"Gap=+{RUN15_BASELINE['semantic_gap']:.3f}  Dead={RUN15_BASELINE['dead_bits']}")
    print()

    # Paths
    data_path = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
    tokenizer_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    temp_suffix = f"_temp{int(args.final_temp)}" if args.final_temp != 10.0 else ""
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'playground', f'checkpoints_xl_sigmoid_anneal{temp_suffix}')
    results_dir = os.path.join(PROJECT_ROOT, 'playground', 'results')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---- Tokenizer ----
    print("[1/5] Loading tokenizer (Run 15's)...")
    tokenizer = BPETokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab: {vocab_size}")

    # ---- Tokenize corpus ----
    print()
    print("[2/5] Tokenizing corpus...")
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
        if (i + 1) % 5000 == 0:
            print(f"  Encoded {i+1}/{len(stories)} ({len(all_tokens):,} tokens)")
    print(f"  Total: {len(all_tokens):,} tokens ({time.time()-t0:.1f}s)")

    # ---- Model ----
    print()
    print("[3/5] Initializing SigmoidAnnealGPT (XL)...")
    config = TriadicGPTConfig(
        vocab_size=vocab_size,
        block_size=args.block,
        n_layer=12,
        n_embd=512,
        n_head=8,
        n_triadic_bits=64,
        dropout=0.1,
    )
    model = SigmoidAnnealGPT(config, final_temp=args.final_temp).to(device)
    total_params = model.num_params()
    print(f"  Parameters: {total_params:,}")
    print(f"  Activation: sigmoid(temp * Wx) → [-1,1], temp: 1.0 → {args.final_temp}")

    # ---- Optimizer + DataLoader ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    dataset = TextDataset(all_tokens, args.block)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    amp_dtype = torch.bfloat16
    use_scaler = False  # bfloat16 doesn't need loss scaling
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    triadic_warmup = int(args.steps * args.warmup_pct)

    # ---- Resume from checkpoint ----
    start_step = 0
    best_loss = float('inf')
    prior_elapsed = 0.0
    if args.resume:
        print()
        print(f"[3b/5] Resuming from checkpoint: {args.resume}")
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
        print(f"  Steps remaining: {args.steps - start_step}")

    # ---- CSV log ----
    csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
    if args.resume and os.path.exists(csv_path):
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([])  # blank line to mark resume
    else:
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'loss', 'tri_loss', 'lr', 'temp', 'entropy', 'dead_bits', 'elapsed_s'])

    # ---- Training ----
    print()
    remaining = args.steps - start_step
    print(f"[4/5] Training {'(resumed) ' if args.resume else ''}for {remaining} steps "
          f"({start_step}/{args.steps} done)...")
    print(f"  Triadic activation at step {triadic_warmup}")
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

        # Temperature annealing: linear 1.0 → final_temp
        progress = step / args.steps
        model.current_temp = 1.0 + (args.final_temp - 1.0) * progress

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
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            total_loss = lang_loss
            tri_loss_val = 0.0

            if step >= triadic_warmup:
                alpha_warmup_steps = int(args.steps * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_warmup_steps)
                current_alpha = args.alpha * alpha_factor

                tri_loss = model.triadic_loss(
                    triadic_proj,
                    entropy_weight=args.entropy_weight,
                    input_ids=x,
                    align_weight=args.align_weight,
                    align_mode='mse',
                )
                total_loss = lang_loss + current_alpha * tri_loss
                tri_loss_val = tri_loss.item()

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
                                 f"{lr_t:.6f}", f"{model.current_temp:.2f}",
                                 f"{mean_ent:.4f}", dead, f"{total_elapsed:.0f}"])
            csv_file.flush()  # prevent data loss on crash/power outage

            if step % 100 == 0:
                # Calculate ETA
                steps_done_this_session = step - start_step + 1
                steps_remaining = args.steps - step - 1
                speed = steps_done_this_session / max(session_elapsed, 1)
                eta_s = steps_remaining / max(speed, 0.01)

                bar = progress_bar(step + 1, args.steps)
                tri_phase = "ON " if step >= triadic_warmup else "off"

                print(f"  {bar}  step {step:>6d}/{args.steps}  "
                      f"loss={lang_loss.item():.3f}  tri[{tri_phase}]={tri_loss_val:.4f}  "
                      f"temp={model.current_temp:.1f}  ent={mean_ent:.3f}  "
                      f"dead={dead}/64  "
                      f"ETA {format_time(eta_s)}  [{format_time(session_elapsed)}]")

        # Checkpoint
        if (step + 1) % 5000 == 0 or step == args.steps - 1:
            loss_val = lang_loss.item()
            session_elapsed = time.time() - start_time
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': {
                    'vocab_size': config.vocab_size,
                    'block_size': config.block_size,
                    'n_layer': config.n_layer,
                    'n_embd': config.n_embd,
                    'n_head': config.n_head,
                    'n_triadic_bits': config.n_triadic_bits,
                },
                'step': step,
                'loss': loss_val,
                'activation': 'sigmoid_anneal',
                'final_temp': args.final_temp,
                'elapsed_s': prior_elapsed + session_elapsed,
            }
            path = os.path.join(checkpoint_dir, f'model_step{step+1}.pt')
            torch.save(ckpt, path)

            if loss_val < best_loss:
                best_loss = loss_val
                best_path = os.path.join(checkpoint_dir, 'model_best.pt')
                torch.save(ckpt, best_path)
                print(f"  >>> New best: {loss_val:.4f} (saved)")

    csv_file.close()
    total_time = time.time() - start_time
    print()
    print(f"  Training complete: {total_time/60:.1f} min")

    # ---- Evaluation ----
    print()
    print("[5/5] Evaluating...")

    # Perplexity
    ppl, avg_loss = compute_perplexity(model, tokenizer, data_path, device, args.block)
    print(f"  Perplexity: {ppl:.2f} (Run 15: {RUN15_BASELINE['ppl']:.2f})")

    # Semantic metrics
    sem = evaluate_model(model, tokenizer, device, 64)
    print(f"  Semantic gap:  {sem['semantic_gap']:+.4f} (Run 15: +{RUN15_BASELINE['semantic_gap']:.3f})")
    print(f"  Dead bits:     {sem['dead_bits']} (Run 15: {RUN15_BASELINE['dead_bits']})")
    print(f"  Active bits:   {sem['active_bits']}")
    print(f"  Bit entropy:   {sem['mean_bit_entropy']:.4f} (Run 15: {RUN15_BASELINE['entropy']:.3f})")
    print(f"  Analogy verif: {sem['analogy_verification']:.1%}")
    print(f"  Unique sigs:   {sem['unique_signatures']}")

    # ---- Comparison ----
    print()
    print("=" * 70)
    print("  SIGMOID+ANNEAL vs RUN 15 (TANH)")
    print("=" * 70)
    print(f"  {'Metric':>25s}  {'Sigmoid+Anneal':>15s}  {'Run 15 (tanh)':>15s}  {'Delta':>10s}")
    print(f"  {'─'*25}  {'─'*15}  {'─'*15}  {'─'*10}")

    comparisons = [
        ('Best loss', best_loss, RUN15_BASELINE['loss']),
        ('Perplexity', ppl, RUN15_BASELINE['ppl']),
        ('Semantic gap', sem['semantic_gap'], RUN15_BASELINE['semantic_gap']),
        ('Dead bits', sem['dead_bits'], RUN15_BASELINE['dead_bits']),
        ('Bit entropy', sem['mean_bit_entropy'], RUN15_BASELINE['entropy']),
    ]

    for name, v_new, v_old in comparisons:
        delta = v_new - v_old
        sign = '+' if delta > 0 else ''
        print(f"  {name:>25s}  {v_new:>15.4f}  {v_old:>15.4f}  {sign}{delta:>9.4f}")

    # ---- Save ----
    results = {
        'experiment': 'xl_sigmoid_anneal',
        'source': 'playground soft_signatures → XL validation',
        'config': '12L/512D/8H/64bits (40M params)',
        'steps': args.steps,
        'final_temp': args.final_temp,
        'training_time_min': total_time / 60,
        'best_loss': best_loss,
        'perplexity': ppl,
        **sem,
        'run15_baseline': RUN15_BASELINE,
    }

    temp_tag = f"_temp{int(args.final_temp)}" if args.final_temp != 10.0 else ""
    results_path = os.path.join(results_dir, f'xl_sigmoid_anneal{temp_tag}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {results_path}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  CSV log: {csv_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

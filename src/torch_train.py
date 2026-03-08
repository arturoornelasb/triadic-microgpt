"""
GPU Training Loop - PyTorch-based training with CUDA acceleration.

Features:
  - Batch training with DataLoader
  - Mixed precision (torch.amp) for speed
  - Cosine LR schedule with warmup
  - Dual-objective: language loss + triadic alignment
  - Periodic checkpointing and sample generation
"""

import os
import sys
import csv
import time
import random
import argparse
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator


# ============================================================
# Dataset
# ============================================================

STORY_SEPARATOR = '<' + '|endoftext|' + '>'


class TextDataset(Dataset):
    """Chunked text dataset for language model training."""

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
# Data Loading
# ============================================================

def load_and_tokenize(data_path, tokenizer, max_stories=None, block_size=256):
    """Load corpus, tokenize, and return a flat array of token IDs."""
    print(f"  Loading: {data_path}")

    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    # Split into stories/documents
    if STORY_SEPARATOR in raw:
        stories = raw.split(STORY_SEPARATOR)
    else:
        stories = raw.split('\n')

    stories = [s.strip() for s in stories if s.strip() and len(s.strip()) > 30]

    if max_stories and len(stories) > max_stories:
        random.seed(42)
        random.shuffle(stories)
        stories = stories[:max_stories]

    print(f"  Documents: {len(stories)}")
    total_chars = sum(len(s) for s in stories)
    print(f"  Characters: {total_chars:,}")

    # Tokenize
    print("  Tokenizing...")
    t0 = time.time()
    all_tokens = []
    for story in stories:
        ids = tokenizer.encode(story, add_special=True)
        all_tokens.extend(ids)

    tok_time = time.time() - t0
    print(f"  Tokens: {len(all_tokens):,} ({tok_time:.1f}s)")
    print(f"  Compression: {total_chars / len(all_tokens):.1f} chars/token")

    return all_tokens


# ============================================================
# Training
# ============================================================

def train(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print()
    print("=" * 64)
    print("  TRIADIC MICROGPT - PyTorch GPU Training")
    print("=" * 64)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --- Paths ---
    data_path = args.data or os.path.join(project_root, 'data', 'TinyStories-train.txt')
    checkpoint_dir = args.checkpoint_dir or os.path.join(project_root, 'checkpoints', 'torch')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Step 1: Tokenizer ---
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    if STORY_SEPARATOR in raw:
        stories = raw.split(STORY_SEPARATOR)
    else:
        stories = raw.split('\n')
    stories = [s.strip() for s in stories if s.strip() and len(s.strip()) > 30]

    if args.stories and len(stories) > args.stories:
        random.seed(42)
        random.shuffle(stories)
        stories = stories[:args.stories]

    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')
    if args.tokenizer and os.path.exists(args.tokenizer):
        print("[1/4] Loading cached tokenizer...")
        tokenizer = BPETokenizer.load(args.tokenizer)
        print(f"  Loaded: {args.tokenizer} (vocab: {tokenizer.vocab_size})")
    else:
        print("[1/4] Training BPE tokenizer (this may take a while)...")
        tokenizer = BPETokenizer(vocab_size=args.vocab)
        tokenizer.train(stories, verbose=True)
        tokenizer.save(tokenizer_path)
    actual_vocab = tokenizer.vocab_size
    print(f"  Actual vocab: {actual_vocab}")

    # --- Step 2: Load or tokenize corpus ---
    print()
    if args.tokens and os.path.exists(args.tokens):
        print(f"[2/4] Loading cached tokens: {args.tokens}")
        all_tokens = np.load(args.tokens).tolist()
        print(f"  Loaded {len(all_tokens):,} tokens instantly")
    else:
        print("[2/4] Tokenizing corpus (use --tokens for cached .npy)...")
        all_tokens = []
        for i, story in enumerate(stories):
            ids = tokenizer.encode(story, add_special=True)
            all_tokens.extend(ids)
            if (i + 1) % 1000 == 0:
                print(f"  Encoded {i+1}/{len(stories)} stories ({len(all_tokens):,} tokens)")
        print(f"  Total tokens: {len(all_tokens):,}")

    # --- Step 3: Initialize model ---
    print()
    print("[3/4] Initializing model...")
    config = TriadicGPTConfig(
        vocab_size=actual_vocab,
        block_size=args.block,
        n_layer=args.layers,
        n_embd=args.dim,
        n_head=args.heads,
        n_triadic_bits=args.bits,
        dropout=args.dropout,
    )
    model = TriadicGPT(config).to(device)
    total_params = model.num_params()
    print(f"  Parameters: {total_params:,}")
    print(f"  Config: {args.layers}L / {args.dim}D / {args.heads}H / {args.bits} triadic bits")
    print(f"  Block size: {args.block}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # --- DataLoader ---
    dataset = TextDataset(all_tokens, args.block)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    # --- Mixed precision ---
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # --- Triadic components ---
    mapper = PrimeMapper(args.bits)
    validator = TriadicValidator()

    triadic_warmup = int(args.steps * args.triadic_warmup_pct)

    # --- Step 3.5: Gold Primes Distillation Loader ---
    gold_primes_path = os.path.join(project_root, 'data', f'gold_primes_{args.bits}.json')
    gold_sequences = []
    if not args.no_distill and os.path.exists(gold_primes_path):
        print("[4/5] Loading Gold Primes for Distillation...")
        with open(gold_primes_path, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
            
        for concept, data in gold_data.items():
            ids = tokenizer.encode(' ' + concept, add_special=False)
            ids_nospace = tokenizer.encode(concept, add_special=False)
            gold_sequences.append((ids, data['binary_signature']))
            if ids != ids_nospace:
                gold_sequences.append((ids_nospace, data['binary_signature']))
                
        print(f"  Mapped {len(gold_sequences)} token sequences for distillation.")
    else:
        print("[4/5] No gold_primes.json found. Skipping pure distillation.")

    # --- Step 4: Training loop ---
    print()
    print(f"[5/5] Training for {args.steps} steps...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Triadic activation: step {triadic_warmup}")
    print(f"  Entropy weight: {args.entropy_weight}")
    print(f"  Alignment weight: {args.align_weight}")
    print(f"  Distillation weight: {args.dist_weight}x alpha")
    print("-" * 64)

    model.train()
    start_time = time.time()
    step = 0
    best_loss = float('inf')
    data_iter = iter(dataloader)

    # CSV logging
    csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'tri_loss', 'dist_loss', 'lr', 'elapsed_s'])

    # Distillation Sequence caching
    if gold_sequences:
        print(f"  Initialized sequence matching for {len(gold_sequences)} distillation targets.")

    while step < args.steps:
        # Get batch (cycle through data)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        B, T = x.shape

        # Cosine learning rate with warmup
        warmup_steps = min(500, args.steps // 10)
        if step < warmup_steps:
            lr_t = args.lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(args.steps - warmup_steps, 1)
            lr_t = args.lr * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)

            # Triadic loss
            total_loss = lang_loss
            tri_loss_val = 0.0
            dist_loss_val = 0.0

            if step >= triadic_warmup:
                # Dynamic alpha scaling: linear warmup from warmup step to end of training
                # or a fixed window. Let's do linear from warmup to (warmup + 20% of steps)
                alpha_warmup_steps = int(args.steps * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_warmup_steps)
                current_alpha = args.alpha * alpha_factor
                
                tri_loss = model.triadic_loss(triadic_proj, entropy_weight=args.entropy_weight,
                                              input_ids=x, align_weight=args.align_weight)
                total_loss = lang_loss + current_alpha * tri_loss
                tri_loss_val = tri_loss.item()
                
                # Apply Knowledge Distillation
                if gold_sequences:
                    b_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
                    targets_proj = torch.zeros((B, T, args.bits), dtype=torch.float32, device=device)
                    
                    # Convert to CPU list for fast subsequence search
                    x_list = x.tolist()
                    match_found = False
                    
                    for b_idx in range(B):
                        seq = x_list[b_idx]
                        for target_ids, bits in gold_sequences:
                            n_tokens = len(target_ids)
                            # Slide over the sequence
                            for i in range(T - n_tokens + 1):
                                if seq[i:i+n_tokens] == target_ids:
                                    # Target is matched! Apply the prime bits to the LAST token of the concept
                                    b_mask[b_idx, i + n_tokens - 1] = True
                                    targets_proj[b_idx, i + n_tokens - 1] = torch.tensor(bits, dtype=torch.float32, device=device)
                                    match_found = True
                    
                    if match_found:
                        dist_loss = model.distillation_loss(triadic_proj, targets_proj, b_mask)
                        
                        # Distillation weight: reduced from 5x (which caused collapse) to 1x
                        dist_alpha = args.alpha * args.dist_weight * alpha_factor
                        total_loss = total_loss + dist_alpha * dist_loss
                        dist_loss_val = dist_loss.item()

        # Backward + step
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Log every step to CSV
        elapsed = time.time() - start_time
        csv_writer.writerow([step + 1, f'{lang_loss.item():.6f}', f'{tri_loss_val:.6f}', f'{dist_loss_val:.6f}', f'{lr_t:.8f}', f'{elapsed:.1f}'])

        # Print logging with progress bar and ETA
        if step % args.print_every == 0 or step == args.steps - 1:
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            remaining = (args.steps - step - 1) / sps if sps > 0 else 0
            pct = (step + 1) / args.steps * 100

            # Progress bar
            bar_len = 30
            filled = int(bar_len * (step + 1) / args.steps)
            bar = '#' * filled + '-' * (bar_len - filled)

            # Format ETA
            if remaining >= 60:
                eta_str = f"{remaining/60:.1f}m"
            else:
                eta_str = f"{remaining:.0f}s"

            msg = f"  [{bar}] {pct:5.1f}%"
            msg += f" | step {step+1}/{args.steps}"
            msg += f" | loss {lang_loss.item():.4f}"
            if step >= triadic_warmup:
                msg += f" | tri {tri_loss_val:.4f}"
                if gold_sequences:
                    msg += f" | dist {dist_loss_val:.4f}"
            msg += f" | {sps:.1f} stp/s"
            msg += f" | ETA {eta_str}"
            print(msg)

        # Checkpoint
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            model_tag = f"L{args.layers}_D{args.dim}_B{args.bits}"
            ckpt_path = os.path.join(checkpoint_dir, f'model_{model_tag}_step{step+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(config),
                'step': step + 1,
                'loss': lang_loss.item(),
            }, ckpt_path)

            if lang_loss.item() < best_loss:
                best_loss = lang_loss.item()
                best_path = os.path.join(checkpoint_dir, f'model_{model_tag}_best.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': vars(config),
                    'step': step + 1,
                    'loss': best_loss,
                }, best_path)

            print(f"  >>> Checkpoint saved: {ckpt_path}")

        step += 1

    # Close CSV log
    csv_file.close()
    print(f"  Training log: {csv_path}")

    # --- Final report ---
    elapsed = time.time() - start_time
    print()
    print("-" * 64)
    print(f"  Training complete!")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Final loss: {lang_loss.item():.4f}")
    print(f"  Speed: {args.steps/elapsed:.1f} steps/s")
    print(f"  Model: {total_params:,} params")

    # Generate samples
    model.eval()
    print()
    print("  Sample generation:")
    print("  " + "-" * 40)
    bos_id = tokenizer.special_tokens['<BOS>']
    for i in range(3):
        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        output = model.generate(input_ids, max_new_tokens=40, temperature=0.7, top_k=50)
        text = tokenizer.decode(output[0].tolist(), skip_special=True)
        print(f"  {i+1}. {text[:80]}")

    print()
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Tokenizer:   {tokenizer_path}")
    print("=" * 64)


import math

# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU Train Triadic MicroGPT (PyTorch)')
    parser.add_argument('--data', type=str, default=None, help='Training data path')
    parser.add_argument('--stories', type=int, default=50000, help='Max stories to use')
    parser.add_argument('--vocab', type=int, default=4096, help='BPE vocab size')
    parser.add_argument('--steps', type=int, default=50000, help='Training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--layers', type=int, default=6, help='Transformer layers')
    parser.add_argument('--dim', type=int, default=256, help='Embedding dim')
    parser.add_argument('--heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--bits', type=int, default=32, help='Triadic bits')
    parser.add_argument('--block', type=int, default=256, help='Block/context size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.05, help='Triadic loss weight')
    parser.add_argument('--entropy-weight', type=float, default=0.0, help='Entropy regularization weight (0=off, try 1.0-2.0)')
    parser.add_argument('--align-weight', type=float, default=0.0, help='Embedding alignment weight (0=off, try 1.0-3.0)')
    parser.add_argument('--dist-weight', type=float, default=1.0, help='Distillation multiplier on alpha (was 5.0, now default 1.0)')
    parser.add_argument('--triadic-warmup-pct', type=float, default=0.8, help='Warmup fraction')
    parser.add_argument('--print-every', type=int, default=50, help='Print frequency')
    parser.add_argument('--save-every', type=int, default=1000, help='Save frequency')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint dir')
    parser.add_argument('--tokenizer', type=str, default=None, help='Pre-trained tokenizer path (skip BPE training)')
    parser.add_argument('--tokens', type=str, default=None, help='Pre-tokenized .npy cache (skip encoding)')
    parser.add_argument('--no-distill', action='store_true', help='Skip gold primes distillation (faster training)')
    parser.add_argument('--scale', type=str, choices=['small', 'base', 'large', 'xl'], default='base', help='Model scale preset')
    parser.add_argument('--override-bits', type=int, default=None, help='Override triadic bits from scale preset (for bits sweep)')
    args = parser.parse_args()

    SCALE_PRESETS = {
        'small':  {'layers': 4,  'dim': 128, 'heads': 4, 'bits': 16},
        'base':   {'layers': 6,  'dim': 256, 'heads': 8, 'bits': 32},
        'large':  {'layers': 8,  'dim': 384, 'heads': 8, 'bits': 48},
        'xl':     {'layers': 12, 'dim': 512, 'heads': 8, 'bits': 64},
    }

    preset = SCALE_PRESETS[args.scale]
    args.layers = preset['layers']
    args.dim = preset['dim']
    args.heads = preset['heads']
    args.bits = preset['bits']

    if args.override_bits is not None:
        args.bits = args.override_bits

    train(args)

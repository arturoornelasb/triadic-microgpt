"""
PyTorch Fine-tune — Instruction tuning on conversational data (GPU).

Takes a pretrained PyTorch model and fine-tunes it on Alpaca-format
instruction data for chat capability.

Usage:
  python src/torch_finetune.py --model checkpoints/torch_run7/model_best.pt \
      --tokenizer checkpoints/torch_run7/tokenizer.json \
      --data data/alpaca_data_cleaned.json \
      --steps 2000 --checkpoint-dir checkpoints/finetuned_run7
"""

import os
import sys
import csv
import time
import json
import math
import random
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer


# ============================================================
# Alpaca Dataset
# ============================================================

def format_alpaca(item):
    """Format an Alpaca dataset item into (instruction, response)."""
    instruction = item.get('instruction', '').strip()
    inp = item.get('input', '').strip()
    response = item.get('output', '').strip()
    if inp:
        user_text = f"{instruction}\n\n{inp}"
    else:
        user_text = instruction
    return user_text, response


class InstructionDataset(Dataset):
    """Dataset of (instruction, response) pairs tokenized for training."""

    def __init__(self, examples, tokenizer, block_size):
        self.samples = []
        skipped = 0

        for user_msg, assistant_msg in examples:
            ids = tokenizer.encode_chat(user_msg, assistant_msg)
            if len(ids) > block_size + 1:
                skipped += 1
                continue
            # Pad to block_size + 1
            pad_id = tokenizer.special_tokens.get('<PAD>', 0)
            while len(ids) < block_size + 1:
                ids.append(pad_id)
            self.samples.append(ids[:block_size + 1])

        print(f"  Usable examples: {len(self.samples)}")
        if skipped:
            print(f"  Skipped (too long): {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


# ============================================================
# Fine-tuning
# ============================================================

def finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print()
    print("=" * 64)
    print("  TRIADIC MICROGPT — PyTorch Fine-Tuning (Chat)")
    print("=" * 64)
    print(f"  Device: {device}")
    print()

    checkpoint_dir = args.checkpoint_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'checkpoints', 'finetuned'
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Load tokenizer ---
    print("[1/4] Loading tokenizer...")
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"  Vocab: {tokenizer.vocab_size}")

    # --- Load pretrained model ---
    print()
    print("[2/4] Loading pretrained model...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'],
        block_size=cfg['block_size'],
        n_layer=cfg['n_layer'],
        n_embd=cfg['n_embd'],
        n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'],
        dropout=args.dropout,
    )
    model = TriadicGPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded: {args.model}")
    print(f"  Config: {config.n_layer}L / {config.n_embd}D / {config.n_head}H")
    print(f"  Params: {model.num_params():,}")

    # --- Load and tokenize instruction data ---
    print()
    print("[3/4] Loading instruction dataset...")
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    examples = [format_alpaca(item) for item in data]
    if args.max_examples and len(examples) > args.max_examples:
        random.seed(42)
        random.shuffle(examples)
        examples = examples[:args.max_examples]
    print(f"  Total examples: {len(examples)}")

    dataset = InstructionDataset(examples, tokenizer, config.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # --- Step 3.5: Gold Primes Distillation Loader ---
    gold_primes_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'data', 'gold_primes.json'
    )
    gold_dict_ids = {}
    if os.path.exists(gold_primes_path):
        print("[3.5/4] Loading Gold Primes for Distillation...")
        with open(gold_primes_path, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
            
        for concept, data in gold_data.items():
            ids = tokenizer.encode(' ' + concept, add_special=False)
            if len(ids) == 1:
                gold_dict_ids[ids[0]] = data['binary_signature']
            ids_nospace = tokenizer.encode(concept, add_special=False)
            if len(ids_nospace) == 1:
                gold_dict_ids[ids_nospace[0]] = data['binary_signature']
                
        print(f"  Mapped {len(gold_dict_ids)} single-token concepts for distillation.")
    else:
        print("[3.5/4] No gold_primes.json found. Skipping pure distillation.")

    # --- Fine-tune ---
    print()
    print(f"[4/4] Fine-tuning for {args.steps} steps...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print("-" * 64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # CSV logging
    csv_path = os.path.join(checkpoint_dir, 'finetune_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'tri_loss', 'dist_loss', 'lr', 'elapsed_s'])

    # Distillation Tensor caching
    if gold_dict_ids:
        distill_target_tensor = torch.zeros(tokenizer.vocab_size, config.n_triadic_bits, device=device)
        distill_mask_tensor = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
        for tok_id, bits in gold_dict_ids.items():
            distill_target_tensor[tok_id] = torch.tensor(bits, dtype=torch.float32, device=device)
            distill_mask_tensor[tok_id] = True
    else:
        distill_target_tensor = None
        distill_mask_tensor = None

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
        B, T = x.shape

        # Cosine LR with warmup
        warmup = min(100, args.steps // 10)
        if step < warmup:
            lr_t = args.lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / max(args.steps - warmup, 1)
            lr_t = args.lr * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        # Forward
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            tri_loss = model.triadic_loss(triadic_proj)
            total_loss = lang_loss + args.alpha * tri_loss
            
            dist_loss_val = 0.0
            
            # Apply Knowledge Distillation
            if distill_target_tensor is not None:
                flat_x = x.view(-1)
                b_mask = distill_mask_tensor[flat_x]
                
                if b_mask.any():
                    mask_bt = b_mask.view(B, T)
                    targets_proj = distill_target_tensor[x]
                    
                    dist_loss = model.distillation_loss(triadic_proj, targets_proj, mask_bt)
                    
                    # Apply heavy emphasis to distillation during fine-tuning
                    dist_alpha = args.alpha * 5.0
                    total_loss = total_loss + dist_alpha * dist_loss
                    dist_loss_val = dist_loss.item()

        # Backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        tri_loss_val = tri_loss.item()
        elapsed = time.time() - start_time

        # CSV log
        csv_writer.writerow([step + 1, f'{lang_loss.item():.6f}', f'{tri_loss_val:.6f}', f'{dist_loss_val:.6f}', f'{lr_t:.8f}', f'{elapsed:.1f}'])

        # Progress bar
        if step % args.print_every == 0 or step == args.steps - 1:
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            remaining = (args.steps - step - 1) / sps if sps > 0 else 0
            pct = (step + 1) / args.steps * 100
            bar_len = 30
            filled = int(bar_len * (step + 1) / args.steps)
            bar = '█' * filled + '░' * (bar_len - filled)
            eta = f"{remaining/60:.1f}m" if remaining >= 60 else f"{remaining:.0f}s"

            msg = f"  [{bar}] {pct:5.1f}%"
            msg += f" | step {step+1}/{args.steps}"
            msg += f" | loss {lang_loss.item():.4f}"
            msg += f" | tri {tri_loss_val:.4f}"
            if distill_target_tensor is not None:
                msg += f" | dist {dist_loss_val:.4f}"
            msg += f" | {sps:.1f} stp/s"
            msg += f" | ETA {eta}"
            print(msg)

        # Checkpoint
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            ckpt = {
                'model_state_dict': model.state_dict(),
                'config': vars(config),
                'step': step + 1,
                'loss': lang_loss.item(),
            }
            ckpt_path = os.path.join(checkpoint_dir, f'chat_step{step+1}.pt')
            torch.save(ckpt, ckpt_path)
            if lang_loss.item() < best_loss:
                best_loss = lang_loss.item()
                torch.save(ckpt, os.path.join(checkpoint_dir, 'chat_best.pt'))
            print(f"  >>> Saved: {ckpt_path}")

        step += 1

    csv_file.close()
    elapsed = time.time() - start_time

    # Save tokenizer copy
    tok_path = os.path.join(checkpoint_dir, 'tokenizer.json')
    tokenizer.save(tok_path)

    # Generate chat samples
    print()
    print("-" * 64)
    print(f"  Fine-tuning complete!")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Final loss: {lang_loss.item():.4f}")
    print()
    print("  Chat sample:")
    print("  " + "-" * 40)

    model.eval()
    test_prompts = [
        "What is the Sun?",
        "Tell me a story about a dog.",
        "Why is the sky blue?",
    ]
    for prompt in test_prompts:
        ids = tokenizer.encode_chat(prompt)
        # Remove the trailing EOS to let the model generate
        if ids[-1] == tokenizer.special_tokens.get('<EOS>', 2):
            ids = ids[:-1]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        output = model.generate(input_ids, max_new_tokens=60, temperature=0.7, top_k=50)
        text = tokenizer.decode(output[0].tolist(), skip_special=True)
        print(f"  Q: {prompt}")
        print(f"  A: {text[:120]}")
        print()

    print(f"  Checkpoints: {checkpoint_dir}")
    print("=" * 64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune Triadic MicroGPT for Chat')
    parser.add_argument('--model', type=str, required=True, help='Pretrained model .pt path')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer path')
    parser.add_argument('--data', type=str, required=True, help='Alpaca JSON dataset path')
    parser.add_argument('--steps', type=int, default=2000, help='Fine-tuning steps')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate (lower for fine-tune)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Triadic loss weight')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--max-examples', type=int, default=2000, help='Max instruction examples')
    parser.add_argument('--print-every', type=int, default=50, help='Print frequency')
    parser.add_argument('--save-every', type=int, default=500, help='Save frequency')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Output dir')
    args = parser.parse_args()

    finetune(args)

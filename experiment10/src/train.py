"""
Experiment 10 -- Two-Phase Training: GPT-2 + Triadic Head.

Phase 1: Freeze GPT-2 backbone, train only the triadic head.
         The alignment loss uses GPT-2's rich pre-trained wte embeddings
         as semantic teacher. Only ~49K trainable parameters.

Phase 2: Unfreeze last N transformer layers + ln_f for joint optimization.
         Language quality may improve on the fine-tuning corpus while
         triadic structure continues to refine. ~14M trainable parameters.

Usage:
  python experiment10/src/train.py
  python experiment10/src/train.py --phase1-steps 3000 --phase2-steps 8000
  python experiment10/src/train.py --model gpt2-medium --unfreeze-layers 3
"""

import os
import sys
import time
import math
import argparse
import csv

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Add project root to path for imports
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from model import GPT2TriadicModel


def load_gpt2(model_name, device):
    """Load pre-trained GPT-2 from HuggingFace."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"  Loading {model_name} from HuggingFace...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
    gpt2 = gpt2.to(device)

    print(f"  Loaded: {sum(p.numel() for p in gpt2.parameters()) / 1e6:.1f}M params")
    print(f"  Hidden dim: {gpt2.config.n_embd}, Layers: {gpt2.config.n_layer}, "
          f"Heads: {gpt2.config.n_head}")
    return gpt2, tokenizer


def load_data(tokenizer, data_path, seq_len, max_mb=300):
    """Load and tokenize TinyStories for GPT-2.

    Args:
        max_mb: Max megabytes of text to read (default 300MB ≈ 75M tokens,
                enough for 15K steps × 16 batch × 256 seq = 61M tokens).
    """
    print(f"  Loading data from {data_path} (max {max_mb}MB)...")
    max_chars = max_mb * 1024 * 1024
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read(max_chars)
    print(f"  Read {len(text) / 1e6:.1f}M characters")

    # Tokenize in chunks to avoid OOM
    chunk_size = 10 * 1024 * 1024  # 10MB text chunks
    print(f"  Tokenizing with GPT-2 tokenizer (vocab={tokenizer.vocab_size})...")
    all_tokens = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        all_tokens.extend(tokenizer.encode(chunk))
        if (i // chunk_size + 1) % 5 == 0:
            print(f"    Tokenized {(i + chunk_size) / 1e6:.0f}M chars ({len(all_tokens):,} tokens)...")

    # Free the text string
    del text

    tokens = torch.tensor(all_tokens, dtype=torch.long)
    del all_tokens
    print(f"  Total tokens: {len(tokens):,} ({len(tokens) // seq_len:,} chunks of {seq_len})")

    # Pre-chunk into sequences
    n_chunks = len(tokens) // seq_len
    tokens = tokens[:n_chunks * seq_len].view(n_chunks, seq_len)
    return tokens


def get_batch(data, batch_size, device):
    """Sample a random batch from pre-chunked data."""
    idx = torch.randint(0, data.size(0), (batch_size,))
    batch = data[idx].to(device)
    return batch


def train_phase(model, data, phase_name, steps, lr, batch_size, seq_len,
                alpha, entropy_weight, align_weight, warmup_pct,
                device, checkpoint_dir, log_writer, global_step_offset=0,
                backbone_frozen=False, align_mode='mse',
                amp_dtype=torch.bfloat16, use_scaler=False):
    """Run one training phase."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, betas=(0.9, 0.999), weight_decay=0.01
    )
    scaler = GradScaler('cuda', enabled=use_scaler)
    warmup_steps = int(steps * warmup_pct)

    trainable = model.num_params(trainable_only=True)
    total = model.num_params(trainable_only=False)
    print(f"\n  {phase_name}")
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")
    print(f"  Steps: {steps}, LR: {lr}, Warmup: {warmup_steps}")
    print(f"  Alpha: {alpha}, Entropy: {entropy_weight}, Align: {align_weight}")
    print()

    model.train()
    start_time = time.time()

    for step in range(1, steps + 1):
        # Learning rate schedule: linear warmup + cosine decay
        if step <= warmup_steps:
            lr_mult = step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        current_lr = lr * lr_mult
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        # Triadic warmup: activate triadic loss after 25% of steps
        # Skip warmup in Phase 1 (backbone frozen) — triadic head is the only trainable thing
        triadic_active = True if backbone_frozen else step > int(steps * 0.25)

        batch = get_batch(data, batch_size, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', dtype=amp_dtype):
            logits, triadic_proj, lang_loss = model(batch, labels=batch)

            if triadic_active:
                tri_loss = model.triadic_loss(
                    triadic_proj,
                    entropy_weight=entropy_weight,
                    input_ids=batch,
                    align_weight=align_weight,
                    align_mode=align_mode,
                )
                if backbone_frozen:
                    # Phase 1: only triadic head is trainable, lang_loss has no grad
                    total_loss = alpha * tri_loss
                else:
                    total_loss = lang_loss + alpha * tri_loss
            else:
                tri_loss = torch.tensor(0.0)
                total_loss = lang_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Logging
        global_step = global_step_offset + step
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta = (steps - step) / steps_per_sec

            ppl = math.exp(min(lang_loss.item(), 20))
            tri_val = tri_loss.item() if triadic_active else 0.0

            # Progress bar
            pct = step / steps
            bar_len = 30
            filled = int(bar_len * pct)
            bar = '#' * filled + '-' * (bar_len - filled)

            print(f"  [{bar}] {step:>6}/{steps} | "
                  f"loss {total_loss.item():.4f} | lang {lang_loss.item():.4f} | "
                  f"tri {tri_val:.4f} | ppl {ppl:.1f} | "
                  f"lr {current_lr:.2e} | {steps_per_sec:.1f} it/s | "
                  f"ETA {eta:.0f}s")

            if log_writer:
                log_writer.writerow([
                    global_step, total_loss.item(), lang_loss.item(),
                    tri_val, ppl, current_lr, phase_name
                ])

        # Checkpoint at end of phase
        if step == steps:
            save_path = os.path.join(checkpoint_dir, f'{phase_name.lower().replace(" ", "_")}_final.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_triadic_bits': model.n_triadic_bits,
                'n_embd': model.n_embd,
                'gpt2_model_name': args.model,
                'align_mode': align_mode,
                'step': global_step,
                'phase': phase_name,
                'lang_loss': lang_loss.item(),
            }, save_path)
            print(f"  Checkpoint saved: {save_path}")

    elapsed = time.time() - start_time
    print(f"\n  {phase_name} complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return global_step_offset + steps


def generate_samples(model, tokenizer, device, n_samples=3):
    """Generate text samples to verify language quality."""
    model.eval()
    prompts = ["Once upon a time", "The scientist discovered", "In a small village"]
    print("\n  Generation samples:")
    for prompt in prompts[:n_samples]:
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=60, temperature=0.7, top_k=50)
        text = tokenizer.decode(output[0].cpu().tolist())
        # Truncate at first newline for readability
        text = text.split('\n')[0][:200]
        print(f"    > {text}")
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 10: GPT-2 + Triadic Head')
    parser.add_argument('--model', default='gpt2', help='HuggingFace model name (gpt2, gpt2-medium)')
    parser.add_argument('--data', default='data/TinyStories-train.txt')
    parser.add_argument('--n-bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=256)

    # Phase 1: frozen backbone
    parser.add_argument('--phase1-steps', type=int, default=5000)
    parser.add_argument('--phase1-lr', type=float, default=1e-3)

    # Phase 2: partial unfreeze
    parser.add_argument('--phase2-steps', type=int, default=10000)
    parser.add_argument('--phase2-lr', type=float, default=3e-5)
    parser.add_argument('--unfreeze-layers', type=int, default=2)

    # Triadic hyperparameters (same as Run 15)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--entropy-weight', type=float, default=1.0)
    parser.add_argument('--align-weight', type=float, default=5.0)
    parser.add_argument('--warmup-pct', type=float, default=0.25)
    parser.add_argument('--align-mode', default='infonce', choices=['mse', 'rank', 'infonce'],
                        help='Alignment loss mode: mse (original), rank (margin ranking), infonce (contrastive)')
    parser.add_argument('--dtype', default='bfloat16', choices=['float32', 'float16', 'bfloat16'],
                        help='AMP precision (default: bfloat16 for Ampere+/Blackwell)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile')
    parser.add_argument('--grad-checkpoint', action='store_true',
                        help='Enable gradient checkpointing')

    parser.add_argument('--checkpoint-dir', default='experiment10/checkpoints')
    parser.add_argument('--skip-phase1', action='store_true', help='Skip Phase 1 (load from checkpoint)')
    parser.add_argument('--skip-phase2', action='store_true', help='Skip Phase 2')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = {'float32': torch.float32, 'float16': torch.float16,
                 'bfloat16': torch.bfloat16}[args.dtype]
    use_scaler = (device.type == 'cuda' and amp_dtype == torch.float16)
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print()
    print("=" * 80)
    print("  EXPERIMENT 10 - GPT-2 + Triadic Projection Head (Transfer)")
    print("=" * 80)
    print(f"  Base model:     {args.model}")
    print(f"  Triadic bits:   {args.n_bits}")
    print(f"  Phase 1:        {args.phase1_steps} steps (backbone frozen, LR={args.phase1_lr})")
    print(f"  Phase 2:        {args.phase2_steps} steps (unfreeze last {args.unfreeze_layers}, LR={args.phase2_lr})")
    print(f"  Triadic params: alpha={args.alpha}, entropy={args.entropy_weight}, align={args.align_weight}")
    print(f"  Align mode:     {args.align_mode}")
    print(f"  Device:         {device}")
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'
    print(f"  GPU:            {gpu_name}")
    print(f"  Precision:      {args.dtype} (amp_dtype={amp_dtype})")
    print(f"  Grad scaler:    {'ON' if use_scaler else 'OFF'}")
    compile_status = 'PENDING'  # updated after torch.compile attempt
    print(f"  torch.compile:  {compile_status}")
    print("=" * 80)

    # Load GPT-2
    gpt2, tokenizer = load_gpt2(args.model, device)

    # Wrap with triadic head
    model = GPT2TriadicModel(gpt2, n_triadic_bits=args.n_bits)
    model = model.to(device)
    print(f"  Total params: {model.num_params() / 1e6:.1f}M")

    # torch.compile with triton guard (Blackwell optimization)
    if device.type == 'cuda' and not args.no_compile:
        try:
            import triton  # noqa: F401
            model = torch.compile(model)
            print("  torch.compile: ON")
        except ImportError:
            print("  torch.compile: SKIPPED (triton not available on Windows)")

    # Load data
    data = load_data(tokenizer, args.data, args.seq_len)

    # CSV log
    log_path = os.path.join(args.checkpoint_dir, 'training_log.csv')
    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['step', 'total_loss', 'lang_loss', 'tri_loss', 'ppl', 'lr', 'phase'])

    global_step = 0

    # ===== Phase 1: Frozen backbone =====
    if not args.skip_phase1:
        model.freeze_backbone()
        global_step = train_phase(
            model, data,
            phase_name="Phase 1 (Frozen Backbone)",
            steps=args.phase1_steps,
            lr=args.phase1_lr,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            alpha=args.alpha,
            entropy_weight=args.entropy_weight,
            align_weight=args.align_weight,
            warmup_pct=args.warmup_pct,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            log_writer=log_writer,
            global_step_offset=0,
            backbone_frozen=True,
            align_mode=args.align_mode,
            amp_dtype=amp_dtype,
            use_scaler=use_scaler,
        )
        generate_samples(model, tokenizer, device)
    else:
        # Load Phase 1 checkpoint
        p1_path = os.path.join(args.checkpoint_dir, 'phase_1_(frozen_backbone)_final.pt')
        if os.path.exists(p1_path):
            print(f"  Loading Phase 1 checkpoint: {p1_path}")
            ckpt = torch.load(p1_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            global_step = ckpt.get('step', args.phase1_steps)
        else:
            print(f"  WARNING: No Phase 1 checkpoint found at {p1_path}")

    # ===== Phase 2: Partial unfreeze =====
    if not args.skip_phase2:
        model.unfreeze_last_n(args.unfreeze_layers)
        global_step = train_phase(
            model, data,
            phase_name="Phase 2 (Unfreeze Last Layers)",
            steps=args.phase2_steps,
            lr=args.phase2_lr,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            alpha=args.alpha,
            entropy_weight=args.entropy_weight,
            align_weight=args.align_weight,
            warmup_pct=args.warmup_pct,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            log_writer=log_writer,
            global_step_offset=global_step,
            align_mode=args.align_mode,
            amp_dtype=amp_dtype,
            use_scaler=use_scaler,
        )
        generate_samples(model, tokenizer, device)

    log_file.close()

    # ===== Final summary =====
    print()
    print("=" * 80)
    print("  EXPERIMENT 10 COMPLETE")
    print("=" * 80)
    print(f"  Checkpoints: {args.checkpoint_dir}/")
    print(f"  Training log: {log_path}")
    print()
    print("  Next: run evaluation with:")
    print(f"    python experiment10/src/evaluate.py --checkpoint {args.checkpoint_dir}/phase_2_(unfreeze_last_layers)_final.pt")
    print("=" * 80)

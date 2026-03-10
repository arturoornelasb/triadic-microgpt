"""
Train a triadic head on GPT-2 — complete example with validation.

Adds a 49K-parameter triadic projection head to GPT-2 and trains it
to produce interpretable prime-factor semantic signatures.

Usage:
    python examples/train_gpt2.py                    # random tokens (API demo)
    python examples/train_gpt2.py --data corpus.txt  # real training

After training:
  - validate() tells you if training worked (PASS/FAIL + random baseline)
  - explore() shows you how words relate to each other
  - Full report saved to --output-dir

TRAINING DURATION GUIDE:
  The number of training steps directly determines result quality.
  Short runs (< 10K steps) are useful for smoke-testing the pipeline
  but will NOT produce reliable semantic signatures.

  Recommended minimums:
    Smoke test:      5,000 steps   (verify pipeline works, ~5 min)
    Minimum viable:  20,000 steps  (basic semantic ordering, ~20 min)
    Good quality:    50,000 steps  (reliable word relationships, ~50 min)
    Production:      100,000+ steps (publish-ready signatures, ~2 hours)

  Times are approximate for GPT-2 (124M) on a single GPU.
  Larger models (LLaMA, Mistral) will need proportionally more steps.
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer
from triadic_head import TriadicWrapper

# ============================================================
# Config
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None, help='Path to text file for training')
parser.add_argument('--model', type=str, default='gpt2', help='HuggingFace model name')
parser.add_argument('--bits', type=int, default=64, help='Number of triadic bits')
parser.add_argument('--align-mode', type=str, default='infonce',
                    choices=['mse', 'rank', 'infonce'],
                    help='Alignment loss type (infonce for pre-trained, mse for from-scratch)')
parser.add_argument('--phase1-steps', type=int, default=2000, help='Steps with frozen backbone')
parser.add_argument('--phase2-steps', type=int, default=5000, help='Steps with unfrozen layers')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--seq-len', type=int, default=256)
parser.add_argument('--output-dir', type=str, default='results', help='Directory for all outputs')
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.05               # triadic loss weight (DO NOT exceed 0.10)
ENTROPY_WEIGHT = 1.0
ALIGN_WEIGHT = 5.0

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Tee: capture all print output to file AND console
class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()
        sys.stdout = self.stdout

log_path = os.path.join(args.output_dir, 'training_log.txt')
tee = Tee(log_path)
sys.stdout = tee

# ============================================================
# Setup
# ============================================================
total_steps = args.phase1_steps + args.phase2_steps

print(f"\n{'=' * 60}")
print(f"  TRIADIC HEAD TRAINING")
print(f"{'=' * 60}")
print(f"  Model:      {args.model}")
print(f"  Device:     {DEVICE}")
print(f"  Bits:       {args.bits}")
print(f"  Align mode: {args.align_mode}")
print(f"  Steps:      {total_steps:,} total ({args.phase1_steps:,} frozen + {args.phase2_steps:,} joint)")
print(f"  Output:     {os.path.abspath(args.output_dir)}")

if total_steps < 20_000:
    print(f"{'-' * 60}")
    print(f"  NOTE: {total_steps:,} steps is a quick smoke test.")
    print(f"  For reliable semantic signatures, use at least 20,000 steps.")
    print(f"  Example: --phase1-steps 5000 --phase2-steps 20000")
    if total_steps < 5_000:
        print(f"  WARNING: Very short run — results will be mostly noise.")

print(f"{'=' * 60}")

print(f"\nLoading {args.model}...")
model = TriadicWrapper(args.model, n_bits=args.bits, align_mode=args.align_mode, device=DEVICE)
print(f"  Backbone:      {model.num_params():,} params")
print(f"  Triadic head:  {model.triadic_params():,} params")

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# Training data
# ============================================================
print("\nPreparing training data...")

if args.data and os.path.exists(args.data):
    # Only read enough text for training (avoid loading huge files entirely)
    # We need: batch_size × seq_len × total_steps tokens, plus buffer
    needed_tokens = args.batch_size * args.seq_len * (args.phase1_steps + args.phase2_steps) * 2
    max_chars = needed_tokens * 5  # ~5 chars per token on average
    print(f"  Reading up to {max_chars // 1_000_000}M chars from {args.data}...")
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read(max_chars)
    print(f"  Tokenizing {len(text):,} chars...")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"  Loaded {len(tokens):,} tokens from {args.data}")
else:
    if args.data:
        print(f"  WARNING: {args.data} not found, using random tokens")
    tokens = list(range(1000)) * 100
    print("  Using random tokens (pass --data corpus.txt for real training)")


def make_batch():
    """Sample a random batch of token sequences."""
    ids = []
    for _ in range(args.batch_size):
        start = torch.randint(0, max(1, len(tokens) - args.seq_len), (1,)).item()
        seq = tokens[start:start + args.seq_len]
        if len(seq) < args.seq_len:
            seq = seq + [tokenizer.eos_token_id] * (args.seq_len - len(seq))
        ids.append(seq)
    return torch.tensor(ids, device=DEVICE)


# ============================================================
# Phase 1: Frozen backbone — train triadic head only
# ============================================================
print(f"\n{'=' * 60}")
print(f"  PHASE 1: Frozen backbone ({args.phase1_steps} steps)")
model.freeze_backbone()
print(f"  Trainable: {model.num_params(trainable_only=True):,} params (triadic head only)")
print(f"{'=' * 60}")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3, weight_decay=0.01,
)

train_log = []
for step in range(1, args.phase1_steps + 1):
    input_ids = make_batch()

    with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
        _, triadic_proj, _ = model(input_ids)
        loss = model.triadic_loss(
            triadic_proj, input_ids=input_ids,
            alpha=ALPHA, entropy_weight=ENTROPY_WEIGHT,
            align_weight=ALIGN_WEIGHT,
        )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    train_log.append({'phase': 1, 'step': step, 'tri_loss': loss.item()})
    if step % 200 == 0 or step == 1:
        print(f"  step {step:5d}/{args.phase1_steps} | tri_loss {loss.item():.4f}")


# ============================================================
# Phase 2: Unfreeze last layers — joint optimization
# ============================================================
print(f"\n{'=' * 60}")
print(f"  PHASE 2: Unfreeze last 2 layers ({args.phase2_steps} steps)")
model.unfreeze_last_n(2)
print(f"  Trainable: {model.num_params(trainable_only=True):,} params")
print(f"{'=' * 60}")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-5, weight_decay=0.01,
)

for step in range(1, args.phase2_steps + 1):
    input_ids = make_batch()

    with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
        logits, triadic_proj, lang_loss = model(input_ids, labels=input_ids)
        tri_loss = model.triadic_loss(
            triadic_proj, input_ids=input_ids,
            alpha=ALPHA, entropy_weight=ENTROPY_WEIGHT,
            align_weight=ALIGN_WEIGHT,
        )
        total_loss = lang_loss + tri_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    train_log.append({
        'phase': 2, 'step': step,
        'lang_loss': lang_loss.item(), 'tri_loss': tri_loss.item(),
    })
    if step % 500 == 0 or step == 1:
        print(f"  step {step:5d}/{args.phase2_steps} | lang {lang_loss.item():.3f} | tri {tri_loss.item():.4f}")


# ============================================================
# Save checkpoint
# ============================================================
ckpt_path = os.path.join(args.output_dir, 'triadic_head.pt')
torch.save({
    'triadic_head': model.triadic_head.state_dict(),
    'n_bits': args.bits,
    'n_embd': model.n_embd,
    'align_mode': args.align_mode,
    'backbone': args.model,
}, ckpt_path)
print(f"\nSaved triadic head to {ckpt_path}")


# ============================================================
# Validate — Did training work?
# ============================================================
model.eval()
report = model.validate(tokenizer=tokenizer, training_steps=total_steps)


# ============================================================
# Explore — Full audit of word relationships
# ============================================================
explore_result = model.explore(
    ['king', 'queen', 'prince', 'dog', 'cat', 'happy', 'sad', 'red', 'blue'],
    tokenizer=tokenizer,
    top_k=0,
    show_factors=True,
    threshold=0.80,
)


# ============================================================
# Save structured reports as JSON
# ============================================================
def _make_serializable(obj):
    """Convert non-serializable types for JSON output."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj

# Validation report
val_path = os.path.join(args.output_dir, 'validation.json')
with open(val_path, 'w') as f:
    json.dump(_make_serializable(report), f, indent=2)

# Explore report
exp_path = os.path.join(args.output_dir, 'explore.json')
with open(exp_path, 'w') as f:
    json.dump(_make_serializable(explore_result), f, indent=2)

# Training log
log_json_path = os.path.join(args.output_dir, 'training_log.json')
with open(log_json_path, 'w') as f:
    json.dump(train_log, f)


# ============================================================
# Final summary
# ============================================================
print(f"\n{'=' * 60}")
print(f"  OUTPUT FILES ({os.path.abspath(args.output_dir)}):")
print(f"    triadic_head.pt     — trained model checkpoint")
print(f"    validation.json     — PASS/FAIL checks + per-group breakdown")
print(f"    explore.json        — similarity matrix + factor index")
print(f"    training_log.json   — per-step loss values")
print(f"    training_log.txt    — full console output")
print(f"{'-' * 60}")
if report['overall_pass']:
    print("  Training SUCCESSFUL — your triadic head is ready to use!")
    print("  Next steps:")
    print("    model.config()                         # view/change settings")
    print("    model.config(align_mode='rank')         # switch alignment mode")
    print("    model.validate(word_groups={...})       # test your own domains")
    print("    model.explore(['your', 'words'])        # discover relationships")
    print("    model.explore([...], show_factors=True) # full factor audit")
    print("    model.explore([...], threshold=0.7)     # flag high-similarity pairs")
    print("    model.compare('word1', 'word2')         # deep-dive on one pair")
else:
    print("  Training needs more work — see FAIL details above.")
    print("  Try: more training steps, real text data, or adjust hyperparameters.")
print(f"{'=' * 60}\n")

# Close tee
tee.close()

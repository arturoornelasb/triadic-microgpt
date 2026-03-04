"""
Pretrain — Full pretraining pipeline for TinyStories or any text corpus.

Pipeline:
  1. Load and subset the corpus
  2. Train BPE tokenizer on the corpus
  3. Tokenize all documents
  4. Pretrain FastGPT with language loss
  5. Activate triadic loss after warmup
  6. Save checkpoint + tokenizer
"""

import os
import sys
import time
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fast_transformer import FastGPT, FastGPTConfig
from src.tensor_ops import AdamOptimizer, softmax_forward, cross_entropy_loss
from src.tokenizer import BPETokenizer, SPECIAL_TOKENS
from src.triadic import PrimeMapper, TriadicValidator


# ============================================================
# Data Preparation
# ============================================================

# TinyStories separator token
STORY_SEPARATOR = '<' + '|endoftext|' + '>'

def load_tinystories(filepath, max_stories=None):
    """Load TinyStories: stories separated by endoftext token."""
    print(f"  Loading: {filepath}")
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    stories = raw.split(STORY_SEPARATOR)
    stories = [s.strip() for s in stories if s.strip() and len(s.strip()) > 50]

    if max_stories and len(stories) > max_stories:
        random.shuffle(stories)
        stories = stories[:max_stories]

    print(f"  Loaded {len(stories)} stories")
    total_chars = sum(len(s) for s in stories)
    print(f"  Total: {total_chars:,} characters ({total_chars/1e6:.1f} MB)")
    return stories


def load_plain_text(filepath, max_docs=None):
    """Load plain text file: one document per line."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        docs = [line.strip() for line in f if line.strip() and len(line.strip()) > 10]
    if max_docs and len(docs) > max_docs:
        random.shuffle(docs)
        docs = docs[:max_docs]
    return docs


# ============================================================
# Pretraining
# ============================================================

def pretrain(
    data_path=None,
    max_stories=5000,
    vocab_size=2048,
    num_steps=2000,
    learning_rate=0.001,
    triadic_warmup_pct=0.8,
    triadic_alpha=0.05,
    n_layer=4,
    n_embd=128,
    n_head=4,
    n_triadic_bits=16,
    block_size=128,
    print_every=50,
    save_every=500,
    checkpoint_dir=None,
):
    """
    Full pretraining pipeline.

    Args:
        data_path: path to TinyStories-train.txt or any text file
        max_stories: max number of stories to use (None = all)
        vocab_size: BPE vocabulary size
        num_steps: total training steps
        learning_rate: peak learning rate
        triadic_warmup_pct: fraction of training before activating triadic loss
        triadic_alpha: triadic loss weight
        n_layer, n_embd, n_head, n_triadic_bits, block_size: model config
        print_every: print frequency
        save_every: checkpoint save frequency
        checkpoint_dir: directory for checkpoints
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'TinyStories-train.txt')
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- Banner ----
    print()
    print("=" * 64)
    print("  TRIADIC MICROGPT — Pretraining Pipeline")
    print("=" * 64)
    print()

    # ---- Step 1: Load corpus ----
    print("[1/5] Loading corpus...")
    random.seed(42)
    np.random.seed(42)

    if 'TinyStories' in data_path or 'tinystories' in data_path:
        stories = load_tinystories(data_path, max_stories=max_stories)
    else:
        stories = load_plain_text(data_path, max_docs=max_stories)
        print(f"  Loaded {len(stories)} documents")

    # ---- Step 2: Train BPE tokenizer ----
    print()
    print("[2/5] Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(stories, verbose=True)

    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    print(f"  Saved tokenizer: {tokenizer_path}")
    actual_vocab = tokenizer.vocab_size
    print(f"  Actual vocab size: {actual_vocab}")

    # ---- Step 3: Tokenize corpus ----
    print()
    print("[3/5] Tokenizing corpus...")
    t0 = time.time()
    all_tokens = []
    for story in stories:
        ids = tokenizer.encode(story, add_special=True)
        all_tokens.extend(ids)
    total_tokens = len(all_tokens)
    tok_time = time.time() - t0
    print(f"  {total_tokens:,} tokens ({tok_time:.1f}s)")
    print(f"  Compression ratio: {sum(len(s) for s in stories) / total_tokens:.1f} chars/token")

    # ---- Step 4: Initialize model ----
    print()
    print("[4/5] Initializing model...")
    config = FastGPTConfig(
        vocab_size=actual_vocab,
        block_size=block_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_triadic_bits=n_triadic_bits,
    )
    model = FastGPT(config)
    total_params = model.num_params()
    print(f"  Parameters: {total_params:,}")
    print(f"  Config: {n_layer}L / {n_embd}D / {n_head}H / {n_triadic_bits} triadic bits")
    print(f"  Block size: {block_size}")

    optimizer = AdamOptimizer(model.params(), lr=learning_rate, weight_decay=0.01)
    mapper = PrimeMapper(n_triadic_bits)
    validator = TriadicValidator()

    triadic_warmup = int(num_steps * triadic_warmup_pct)

    # ---- Step 5: Training loop ----
    print()
    print("[5/5] Training...")
    print(f"  Steps: {num_steps}")
    print(f"  Triadic activation: step {triadic_warmup}")
    print("-" * 64)

    start_time = time.time()
    loss_sum = 0.0
    tri_loss_sum = 0.0
    best_loss = float('inf')

    for step in range(num_steps):
        # Sample a random chunk from the corpus
        max_start = max(0, total_tokens - block_size - 1)
        start_idx = random.randint(0, max_start)
        chunk = all_tokens[start_idx:start_idx + block_size + 1]

        if len(chunk) < 3:
            continue

        T = len(chunk) - 1
        input_ids = chunk[:T]
        target_ids = np.array(chunk[1:T+1], dtype=np.int32)

        # Forward pass
        logits, hidden, caches = model.forward(input_ids)

        # Language loss
        probs = softmax_forward(logits)
        lang_loss, grad_logits = cross_entropy_loss(probs, target_ids)
        loss_sum += lang_loss

        # Triadic loss (after warmup)
        grad_hidden = None
        tri_loss_val = 0.0

        if step >= triadic_warmup and T >= 2:
            triadic_proj = model.project_to_triadic_np(hidden)
            tri_loss = 0.0
            grad_tri = np.zeros_like(hidden)

            for t in range(T - 1):
                pa = triadic_proj[t]
                pb = triadic_proj[t + 1]
                agreement = np.sum(pa * pb) / n_triadic_bits
                tri_loss += (1.0 - agreement)

                dtanh_a = 1 - pa * pa
                dtanh_b = 1 - pb * pb
                grad_pa = -pb * dtanh_a / n_triadic_bits
                grad_pb = -pa * dtanh_b / n_triadic_bits

                grad_tri[t] += grad_pa @ model.triadic_head.data
                grad_tri[t + 1] += grad_pb @ model.triadic_head.data
                model.triadic_head.grad += np.outer(grad_pa, hidden[t])
                model.triadic_head.grad += np.outer(grad_pb, hidden[t + 1])

            tri_loss /= (T - 1)
            grad_tri *= triadic_alpha / (T - 1)
            tri_loss_val = tri_loss
            tri_loss_sum += tri_loss
            grad_hidden = grad_tri

        # Backward + update
        optimizer.zero_grad()
        model.backward(grad_logits, grad_hidden, caches)

        # Cosine learning rate schedule
        progress = step / max(num_steps - 1, 1)
        cos_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        lr_t = learning_rate * max(cos_decay, 0.1)

        optimizer.step(lr_override=lr_t)

        # Logging
        if step % print_every == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            avg_loss = loss_sum / max(1, step + 1)
            sps = (step + 1) / elapsed if elapsed > 0 else 0

            msg = f"  step {step+1:6d}/{num_steps}"
            msg += f" | loss {lang_loss:.4f} (avg {avg_loss:.4f})"
            if step >= triadic_warmup:
                msg += f" | tri {tri_loss_val:.4f}"
            msg += f" | lr {lr_t:.6f}"
            msg += f" | {sps:.1f} stp/s"
            msg += f" | {elapsed:.0f}s"
            print(msg)

        # Save checkpoint
        if (step + 1) % save_every == 0 or step == num_steps - 1:
            ckpt_path = os.path.join(checkpoint_dir, f'model_step{step+1}')
            model.save_checkpoint(ckpt_path)
            avg_loss = loss_sum / (step + 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(checkpoint_dir, 'model_best')
                model.save_checkpoint(best_path)
            print(f"  >>> Checkpoint saved: {ckpt_path}.npz")

    # ---- Final report ----
    elapsed = time.time() - start_time
    print()
    print("-" * 64)
    print(f"  Pretraining complete!")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Final loss: {lang_loss:.4f}")
    print(f"  Speed: {num_steps/elapsed:.1f} steps/s")
    print(f"  Model: {total_params:,} params")
    print()

    # Generate a sample
    print("  Sample generation:")
    print("  " + "-" * 40)
    for _ in range(3):
        text = generate_sample(model, tokenizer, block_size, temperature=0.7)
        print(f"    {text[:80]}")
    print()
    print("=" * 64)

    return model, tokenizer, mapper, validator, config


def generate_sample(model, tokenizer, block_size, max_tokens=60, temperature=0.7):
    """Generate a text sample from the model."""
    bos_id = tokenizer.special_tokens['<BOS>']
    eos_id = tokenizer.special_tokens['<EOS>']

    input_ids = [bos_id]
    generated = []

    for _ in range(max_tokens):
        # Use last block_size tokens
        ctx = input_ids[-block_size:]
        logits, _, _ = model.forward(ctx)

        # Take logits for last position only
        last_logits = logits[-1]

        # Temperature scaling
        scaled = last_logits / temperature
        probs = softmax_forward(scaled.reshape(1, -1))[0]

        # Sample
        next_id = np.random.choice(len(probs), p=probs)

        if next_id == eos_id:
            break

        input_ids.append(next_id)
        generated.append(next_id)

    return tokenizer.decode(generated, skip_special=True)


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pretrain Triadic MicroGPT')
    parser.add_argument('--data', type=str, default=None, help='Training data path')
    parser.add_argument('--stories', type=int, default=5000, help='Max stories to use')
    parser.add_argument('--vocab', type=int, default=2048, help='BPE vocab size')
    parser.add_argument('--steps', type=int, default=2000, help='Training steps')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--layers', type=int, default=4, help='Transformer layers')
    parser.add_argument('--dim', type=int, default=128, help='Embedding dim')
    parser.add_argument('--heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--bits', type=int, default=16, help='Triadic bits')
    parser.add_argument('--block', type=int, default=128, help='Block/context size')
    parser.add_argument('--alpha', type=float, default=0.05, help='Triadic loss weight')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint dir')
    args = parser.parse_args()

    pretrain(
        data_path=args.data,
        max_stories=args.stories,
        vocab_size=args.vocab,
        num_steps=args.steps,
        learning_rate=args.lr,
        n_layer=args.layers,
        n_embd=args.dim,
        n_head=args.heads,
        n_triadic_bits=args.bits,
        block_size=args.block,
        triadic_alpha=args.alpha,
        checkpoint_dir=args.checkpoint_dir,
    )

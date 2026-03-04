"""
Fast Training Loop — NumPy-based training for scalable models.

Uses FastGPT (NumPy tensor ops) for ~100× faster training compared
to the scalar autograd version. Processes full sequences at once.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fast_transformer import FastGPT, FastGPTConfig
from src.tensor_ops import AdamOptimizer, softmax_forward, cross_entropy_loss
from src.triadic import PrimeMapper, TriadicValidator, prime_factors


# ============================================================
# Data Loading
# ============================================================

def load_corpus(filepath):
    """Load text corpus. Each line is a training document."""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def build_tokenizer(docs):
    """Character-level tokenizer (will be replaced with BPE in Phase 2)."""
    uchars = sorted(set(''.join(docs)))
    char_to_id = {ch: i for i, ch in enumerate(uchars)}
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    return uchars, char_to_id, vocab_size, BOS


def tokenize(doc, char_to_id, BOS):
    """Convert string to token ids wrapped with BOS."""
    return [BOS] + [char_to_id[ch] for ch in doc if ch in char_to_id] + [BOS]


# ============================================================
# Training
# ============================================================

def fast_train(
    data_path=None,
    num_steps=500,
    learning_rate=0.001,
    triadic_warmup=100,
    triadic_alpha=0.05,
    checkpoint_path=None,
    n_layer=2,
    n_embd=64,
    n_head=4,
    n_triadic_bits=8,
    block_size=64,
    print_every=10,
):
    """
    Train the FastGPT model with NumPy tensor ops.
    """
    # --- Resolve paths ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'concepts.txt')
    if checkpoint_path is None:
        checkpoint_path = os.path.join(project_root, 'model_fast')

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    # --- Load data ---
    print("=" * 60)
    print("  TRIADIC MICROGPT — Fast Training (NumPy)")
    print("=" * 60)
    print()

    docs = load_corpus(data_path)
    np.random.seed(42)
    np.random.shuffle(docs)
    print(f"  Corpus: {len(docs)} documents")

    uchars, char_to_id, vocab_size, BOS = build_tokenizer(docs)
    print(f"  Vocab:  {vocab_size} tokens ({len(uchars)} chars + BOS)")

    # --- Initialize model ---
    config = FastGPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_triadic_bits=n_triadic_bits,
    )
    model = FastGPT(config)
    total_params = model.num_params()
    print(f"  Model:  {total_params:,} parameters")
    print(f"  Config: {n_layer}L / {n_embd}D / {n_head}H / {n_triadic_bits} triadic bits")
    print()

    # --- Optimizer ---
    optimizer = AdamOptimizer(model.params(), lr=learning_rate)

    # --- Triadic components ---
    mapper = PrimeMapper(n_triadic_bits)
    validator = TriadicValidator()

    # --- Training loop ---
    print(f"  Training for {num_steps} steps (triadic warmup: {triadic_warmup})")
    print("-" * 60)

    start_time = time.time()
    concept_primes = {}
    loss_history = []

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = tokenize(doc, char_to_id, BOS)
        T = min(block_size, len(tokens) - 1)
        tokens = tokens[:T + 1]

        input_ids = tokens[:T]
        target_ids = np.array(tokens[1:T + 1], dtype=np.int32)

        # --- Forward pass ---
        logits, hidden, caches = model.forward(input_ids)

        # --- Language loss ---
        probs = softmax_forward(logits)
        lang_loss, grad_logits = cross_entropy_loss(probs, target_ids)

        # --- Triadic loss (after warmup) ---
        grad_hidden = None
        tri_loss_val = 0.0

        if step >= triadic_warmup and T >= 2:
            # Project hidden states to triadic space
            triadic_projections = model.project_to_triadic_np(hidden)

            # Simple triadic loss: encourage adjacent positions to agree
            # agreement = sum(proj[i] * proj[i+1]) should be high
            tri_loss = 0.0
            grad_tri = np.zeros_like(hidden)

            for t in range(T - 1):
                pa = triadic_projections[t]  # (n_bits,)
                pb = triadic_projections[t + 1]  # (n_bits,)
                agreement = np.sum(pa * pb) / n_triadic_bits
                tri_loss += (1.0 - agreement)

                # Gradient of triadic loss w.r.t. projections → hidden
                # d(1 - pa*pb)/d(pa) = -pb, scaled by dtanh/dhidden
                dtanh_a = 1 - pa * pa  # tanh derivative
                dtanh_b = 1 - pb * pb
                grad_pa = -pb * dtanh_a / n_triadic_bits
                grad_pb = -pa * dtanh_b / n_triadic_bits

                # Chain to hidden via triadic_head: proj = tanh(hidden @ W^T)
                grad_tri[t] += grad_pa @ model.triadic_head.data
                grad_tri[t + 1] += grad_pb @ model.triadic_head.data

                # Gradient to triadic_head weights
                model.triadic_head.grad += np.outer(grad_pa, hidden[t])
                model.triadic_head.grad += np.outer(grad_pb, hidden[t + 1])

            tri_loss /= (T - 1)
            grad_tri *= triadic_alpha / (T - 1)
            tri_loss_val = tri_loss
            grad_hidden = grad_tri

            # Track prime signatures
            if step % print_every == 0:
                words = doc.split()
                for i, word in enumerate(words[:min(3, T)]):
                    bits = triadic_projections[min(i, T - 1)]
                    prime = mapper.map(bits)
                    concept_primes[word] = prime

        # --- Backward pass ---
        optimizer.zero_grad()
        model.backward(grad_logits, grad_hidden, caches)

        # --- Learning rate schedule ---
        lr_t = learning_rate * (1 - step / num_steps)

        # --- Optimizer step ---
        optimizer.step(lr_override=lr_t)

        loss_history.append(lang_loss)

        # --- Logging ---
        if step % print_every == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            msg = f"  step {step+1:5d}/{num_steps:5d}"
            msg += f" | lang_loss {lang_loss:.4f}"
            if step >= triadic_warmup:
                msg += f" | tri_loss {tri_loss_val:.4f}"
            msg += f" | lr {lr_t:.6f}"
            msg += f" | {steps_per_sec:.1f} steps/s"
            msg += f" | {elapsed:.1f}s"
            print(msg)

    # --- Final report ---
    elapsed = time.time() - start_time
    print()
    print("-" * 60)
    print(f"  Training complete in {elapsed:.1f}s ({total_params:,} params)")
    print(f"  Final loss: {loss_history[-1]:.4f}")
    avg_speed = num_steps / elapsed
    print(f"  Average speed: {avg_speed:.1f} steps/second")
    print()

    # Show prime signatures
    if concept_primes:
        print("  Learned Prime Signatures:")
        print("  " + "-" * 45)
        for concept, prime in sorted(concept_primes.items()):
            factors = mapper.explain(prime)
            f_str = " × ".join(map(str, factors['factors'])) if factors['factors'] else "1"
            print(f"    {concept:15s} → Φ = {prime:10d}  [{f_str}]")

        # Relationships
        items = list(concept_primes.items())
        if len(items) >= 2:
            print()
            print("  Triadic Relationships:")
            print("  " + "-" * 45)
            for i in range(min(6, len(items))):
                for j in range(i + 1, min(i + 2, len(items))):
                    a_name, a_prime = items[i]
                    b_name, b_prime = items[j]
                    sim = validator.similarity(a_prime, b_prime)
                    gap = validator.explain_gap(a_prime, b_prime)
                    print(f"    {a_name} ↔ {b_name}: "
                          f"sim={sim:.0%}, "
                          f"shared={gap['shared_factors']}")

    # --- Save ---
    model.save_checkpoint(checkpoint_path)

    vocab_path = checkpoint_path + '.vocab'
    with open(vocab_path, 'w') as f:
        f.write(''.join(uchars))

    print(f"\n  Checkpoint: {checkpoint_path}.npz")
    print(f"  Vocabulary: {vocab_path}")
    print("=" * 60)

    return model, mapper, validator, config, uchars, char_to_id, vocab_size, BOS


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fast Train Triadic MicroGPT (NumPy)')
    parser.add_argument('--steps', type=int, default=500, help='Training steps')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--warmup', type=int, default=100, help='Triadic warmup steps')
    parser.add_argument('--alpha', type=float, default=0.05, help='Triadic loss weight')
    parser.add_argument('--layers', type=int, default=2, help='Transformer layers')
    parser.add_argument('--dim', type=int, default=64, help='Embedding dim')
    parser.add_argument('--heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--bits', type=int, default=8, help='Triadic bits')
    parser.add_argument('--block', type=int, default=64, help='Block size')
    parser.add_argument('--data', type=str, default=None, help='Data path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    args = parser.parse_args()

    fast_train(
        data_path=args.data,
        num_steps=args.steps,
        learning_rate=args.lr,
        triadic_warmup=args.warmup,
        triadic_alpha=args.alpha,
        n_layer=args.layers,
        n_embd=args.dim,
        n_head=args.heads,
        n_triadic_bits=args.bits,
        block_size=args.block,
        checkpoint_path=args.checkpoint,
    )

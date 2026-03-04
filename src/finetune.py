"""
Fine-tune — Instruction tuning on conversational data.

Pipeline for Phase 4:
  1. Load Alpaca-format instruction dataset
  2. Format into `<USER> ... <ASSISTANT> ...` using BPE Tokenizer
  3. Load pretrained FastGPT model
  4. Fine-tune using Language Loss (on Assistant tokens only)
     and Triadic Loss (on all tokens to maintain semantic alignment)
  5. Save fine-tuned conversational checkpoint
"""

import os
import sys
import time
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fast_transformer import FastGPT, FastGPTConfig
from src.tensor_ops import AdamOptimizer, softmax_forward, cross_entropy_loss
from src.tokenizer import BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator


# ============================================================
# Data Preparation
# ============================================================

def format_alpaca(item):
    """Format an Alpaca dataset item into (instruction, response) pairs."""
    instruction = item.get('instruction', '').strip()
    inp = item.get('input', '').strip()
    response = item.get('output', '').strip()

    if inp:
        user_text = f"{instruction}\n\n{inp}"
    else:
        user_text = instruction
        
    return user_text, response


def load_dataset(filepath, max_examples=None):
    """Load and format instruction dataset."""
    print(f"  Loading dataset: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    examples = [format_alpaca(item) for item in data]
    
    if max_examples and len(examples) > max_examples:
        random.shuffle(examples)
        examples = examples[:max_examples]
        
    print(f"  Loaded {len(examples)} instruction-response pairs")
    return examples


# ============================================================
# Fine-tuning
# ============================================================

def finetune(
    data_path,
    pretrained_ckpt,
    tokenizer_path,
    num_steps=1000,
    learning_rate=5e-5,  # Lower LR for fine-tuning
    triadic_alpha=0.01,  # Lower triadic alpha
    block_size=128,
    print_every=50,
    save_every=500,
    checkpoint_dir=None,
    max_examples=500,
):
    """
    Fine-tune a pretrained model on instructions.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(project_root, 'checkpoints', 'finetuned')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print()
    print("=" * 64)
    print("  TRIADIC MICROGPT — Instruction Tuning (Phase 4)")
    print("=" * 64)
    print()

    # ---- Step 1: Load Tokenizer ----
    print("[1/5] Loading tokenizer...")
    if not os.path.exists(tokenizer_path):
        print(f"  Error: Tokenizer not found at {tokenizer_path}")
        return
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"  Loaded BPE tokenizer (vocab: {tokenizer.vocab_size})")

    # ---- Step 2: Load Data ----
    print()
    print("[2/5] Preparing data...")
    examples = load_dataset(data_path, max_examples=max_examples)
    
    # Tokenize and create chunks
    # We want chunks that contain complete interactions up to block_size
    chunks = []
    skipped = 0
    t0 = time.time()
    
    for user_msg, asst_msg in examples:
        # Encode as: <BOS> <USER> msg <ASSISTANT> msg <EOS>
        ids = tokenizer.encode_chat(user_msg, asst_msg)
        
        if len(ids) > block_size + 1:
            # Skip examples that are too long for our context window
            skipped += 1
            continue
            
        # Find where the assistant starts
        asst_token = tokenizer.special_tokens['<ASSISTANT>']
        try:
            asst_idx = ids.index(asst_token)
            chunks.append((ids, asst_idx))
        except ValueError:
            continue
            
    tok_time = time.time() - t0
    print(f"  Encoded {len(chunks)} conversations ({tok_time:.1f}s)")
    if skipped > 0:
        print(f"  Skipped {skipped} examples (exceed block size {block_size})")

    # ---- Step 3: Load Model ----
    print()
    print("[3/5] Loading pretrained model...")
    # Read config from checkpoint name or use defaults matching Phase 3 target
    config = FastGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=4,
        n_embd=128,
        n_head=4,
        n_triadic_bits=16,
    )
    model = FastGPT(config)
    
    try:
        model.load_checkpoint(pretrained_ckpt)
        print(f"  Loaded weights from {pretrained_ckpt}")
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")
        return
        
    total_params = model.num_params()
    print(f"  Parameters: {total_params:,}")

    # For fine-tuning, optimizer learning rate is very small
    optimizer = AdamOptimizer(model.params(), lr=learning_rate, weight_decay=0.01)

    # ---- Step 4: Training Loop ----
    print()
    print("[4/5] Fine-tuning...")
    print(f"  Steps: {num_steps}")
    print(f"  Masking loss on USER tokens")
    print("-" * 64)

    start_time = time.time()
    loss_sum = 0.0
    
    for step in range(num_steps):
        # Sample a random conversation
        ids, asst_idx = random.choice(chunks)
        
        T = len(ids) - 1
        input_ids = ids[:T]
        target_ids = np.array(ids[1:T+1], dtype=np.int32)
        
        # Create loss mask — 0 for user prompt, 1 for assistant response
        # Assistant targets start after the <ASSISTANT> token
        loss_mask = np.zeros(T, dtype=np.float32)
        if asst_idx < T:
            loss_mask[asst_idx:] = 1.0
            
        # Avoid zero-loss batches
        if np.sum(loss_mask) == 0:
            continue

        # Forward pass
        logits, hidden, caches = model.forward(input_ids)

        # Custom cross entropy with mask
        probs = softmax_forward(logits)
        probs_clipped = np.clip(probs, 1e-8, 1.0)
        
        lang_loss = 0.0
        grad_logits = probs.copy()
        
        active_tokens = np.sum(loss_mask)
        
        for t in range(T):
            if loss_mask[t] > 0:
                lang_loss -= np.log(probs_clipped[t, target_ids[t]])
                grad_logits[t, target_ids[t]] -= 1.0
            else:
                grad_logits[t] = 0.0
                
        lang_loss /= max(1.0, active_tokens)
        grad_logits /= max(1.0, active_tokens)
        
        loss_sum += lang_loss

        # Triadic loss (keep concepts grounded during fine-tuning)
        grad_hidden = None
        tri_loss_val = 0.0

        if T >= 2:
            triadic_proj = model.project_to_triadic_np(hidden)
            tri_loss = 0.0
            grad_tri = np.zeros_like(hidden)

            for t in range(T - 1):
                pa = triadic_proj[t]
                pb = triadic_proj[t + 1]
                agreement = np.sum(pa * pb) / config.n_triadic_bits
                tri_loss += (1.0 - agreement)

                dtanh_a = 1 - pa * pa
                dtanh_b = 1 - pb * pb
                grad_pa = -pb * dtanh_a / config.n_triadic_bits
                grad_pb = -pa * dtanh_b / config.n_triadic_bits

                grad_tri[t] += grad_pa @ model.triadic_head.data
                grad_tri[t + 1] += grad_pb @ model.triadic_head.data
                model.triadic_head.grad += np.outer(grad_pa, hidden[t])
                model.triadic_head.grad += np.outer(grad_pb, hidden[t + 1])

            tri_loss /= (T - 1)
            grad_tri *= triadic_alpha / (T - 1)
            tri_loss_val = tri_loss
            grad_hidden = grad_tri

        # Backward + update
        optimizer.zero_grad()
        model.backward(grad_logits, grad_hidden, caches)

        # Constant LR with linear decay at the end
        if step > num_steps * 0.8:
            decay_steps = num_steps * 0.2
            current_decay_step = step - (num_steps * 0.8)
            lr_t = learning_rate * (1.0 - current_decay_step / decay_steps)
        else:
            lr_t = learning_rate

        optimizer.step(lr_override=max(lr_t, 1e-6))

        # Logging
        if step % print_every == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            avg_loss = loss_sum / max(1, step + 1)
            sps = (step + 1) / elapsed if elapsed > 0 else 0

            print(f"  step {step+1:5d}/{num_steps} | "
                  f"loss {lang_loss:.4f} (avg {avg_loss:.4f}) | "
                  f"tri {tri_loss_val:.4f} | "
                  f"{sps:.1f} stp/s | {elapsed:.0f}s")

        # Save checkpoint
        if (step + 1) % save_every == 0 or step == num_steps - 1:
            ckpt_path = os.path.join(checkpoint_dir, f'chat_model_step{step+1}')
            model.save_checkpoint(ckpt_path)

    # ---- Step 5: Final Report ----
    elapsed = time.time() - start_time
    print()
    print("-" * 64)
    print(f"  Fine-tuning complete!")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Final loss: {lang_loss:.4f}")
    
    # Save final model explicitly
    final_path = os.path.join(checkpoint_dir, 'chat_model_final')
    model.save_checkpoint(final_path)
    print(f"  Final model saved to: {final_path}.npz")
    print("=" * 64)

    return model, tokenizer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune Triadic MicroGPT for Chat')
    parser.add_argument('--data', type=str, required=True, help='Alpaca format JSON data')
    parser.add_argument('--model', type=str, required=True, help='Pretrained .npz model path')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer JSON path')
    parser.add_argument('--steps', type=int, default=1000, help='Fine-tuning steps')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--block', type=int, default=128, help='Context block size')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Output dir')
    args = parser.parse_args()

    finetune(
        data_path=args.data,
        pretrained_ckpt=args.model,
        tokenizer_path=args.tokenizer,
        num_steps=args.steps,
        learning_rate=args.lr,
        block_size=args.block,
        checkpoint_dir=args.checkpoint_dir,
    )

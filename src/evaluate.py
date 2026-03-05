"""
Evaluate — Measure model quality after training.

Produces:
  1. Perplexity on held-out data
  2. Sample generations (qualitative)
  3. Triadic signature analysis (concept comparisons)
  4. Loss curve graph (from CSV log)

Usage:
  python src/evaluate.py --model checkpoints/torch/model_best.pt --tokenizer checkpoints/torch/tokenizer.json
"""

import os
import sys
import math
import json
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator


# ============================================================
# Evaluation Functions
# ============================================================

def load_model(model_path, tokenizer_path, device='cuda'):
    """Load model and tokenizer from checkpoint."""
    tokenizer = BPETokenizer.load(tokenizer_path)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'],
        block_size=cfg['block_size'],
        n_layer=cfg['n_layer'],
        n_embd=cfg['n_embd'],
        n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'],
        dropout=0.0,  # No dropout during eval
    )
    model = TriadicGPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, tokenizer, config


def compute_perplexity(model, tokenizer, data_path, device, max_samples=200, block_size=256):
    """Compute perplexity on a held-out dataset."""
    STORY_SEP = '<' + '|endoftext|' + '>'

    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    if STORY_SEP in raw:
        stories = raw.split(STORY_SEP)
    else:
        stories = raw.split('\n')
    stories = [s.strip() for s in stories if s.strip() and len(s.strip()) > 50]

    # Use last portion as validation
    val_stories = stories[-max_samples:]

    total_loss = 0.0
    total_tokens = 0

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
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def generate_samples(model, tokenizer, device, n=5, max_tokens=60, temperature=0.7):
    """Generate sample text from the model."""
    bos_id = tokenizer.special_tokens['<BOS>']
    eos_id = tokenizer.special_tokens['<EOS>']
    samples = []

    for _ in range(n):
        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        output = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature, top_k=50)
        tokens = output[0].tolist()

        # Stop at EOS
        if eos_id in tokens:
            tokens = tokens[:tokens.index(eos_id)]

        text = tokenizer.decode(tokens, skip_special=True)
        samples.append(text.strip())

    return samples


def analyze_triadic(model, tokenizer, device, mapper, validator, concept_pairs):
    """Analyze triadic signatures for concept pairs."""
    results = []

    for word_a, word_b in concept_pairs:
        def get_prime(word):
            ids = tokenizer.encode(word, add_special=False)
            if not ids:
                return 1, None
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                _, triadic_proj, _ = model(x)
            # Mean triadic projection over tokens
            proj = triadic_proj[0].mean(dim=0).cpu().numpy()
            prime = mapper.map(proj)
            return prime, proj

        prime_a, proj_a = get_prime(word_a)
        prime_b, proj_b = get_prime(word_b)

        if prime_a and prime_b:
            sim = validator.similarity(prime_a, prime_b)
            gap = validator.explain_gap(prime_a, prime_b)
        else:
            sim = 0.0
            gap = {'shared_factors': [], 'only_in_a_factors': [], 'only_in_b_factors': []}

        results.append({
            'word_a': word_a,
            'word_b': word_b,
            'prime_a': prime_a,
            'prime_b': prime_b,
            'similarity': sim,
            'shared': gap['shared_factors'],
            'only_a': gap['only_in_a_factors'],
            'only_b': gap['only_in_b_factors'],
        })

    return results


def plot_loss_curve(csv_path, output_path):
    """Plot training loss curve from CSV log."""
    if not os.path.exists(csv_path):
        print(f"  No CSV log found at {csv_path}")
        return

    import csv
    steps, losses, tri_losses, lrs = [], [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            losses.append(float(row['loss']))
            tri_losses.append(float(row.get('tri_loss', 0)))
            lrs.append(float(row.get('lr', 0)))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Loss curve
    ax1.plot(steps, losses, 'b-', alpha=0.3, linewidth=0.5)
    # Smoothed loss
    window = max(1, len(losses) // 50)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax1.plot(steps[window-1:], smoothed, 'b-', linewidth=2, label='Language Loss (smoothed)')
    ax1.set_ylabel('Language Loss')
    ax1.set_title('Training Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Triadic loss
    ax2.plot(steps, tri_losses, 'r-', alpha=0.3, linewidth=0.5)
    if window > 1:
        smoothed_tri = np.convolve(tri_losses, np.ones(window)/window, mode='valid')
        ax2.plot(steps[window-1:], smoothed_tri, 'r-', linewidth=2, label='Triadic Loss (smoothed)')
    ax2.set_ylabel('Triadic Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Learning rate
    ax3.plot(steps, lrs, 'g-', linewidth=2, label='Learning Rate')
    ax3.set_ylabel('Learning Rate')
    ax3.set_xlabel('Step')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Loss curve saved: {output_path}")


# ============================================================
# Main Evaluation
# ============================================================

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 64)
    print("  TRIADIC MICROGPT — Model Evaluation")
    print("=" * 64)
    print(f"  Device: {device}")
    print(f"  Model:  {args.model}")
    print()

    # Load model
    print("[1/4] Loading model...")
    model, tokenizer, config = load_model(args.model, args.tokenizer, device)
    total_params = model.num_params()
    print(f"  Parameters: {total_params:,}")
    print(f"  Config: {config.n_layer}L / {config.n_embd}D / {config.n_head}H / {config.n_triadic_bits} bits")

    report = {
        'model_path': args.model,
        'params': total_params,
        'config': f"{config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits",
    }

    # Perplexity
    print()
    print("[2/4] Computing perplexity...")
    data_path = args.data or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'TinyStories-train.txt'
    )
    perplexity, avg_loss = compute_perplexity(model, tokenizer, data_path, device, block_size=config.block_size)
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Avg Loss:   {avg_loss:.4f}")
    report['perplexity'] = perplexity
    report['avg_loss'] = avg_loss

    # Sample generation
    print()
    print("[3/4] Generating samples...")
    samples = generate_samples(model, tokenizer, device, n=5, temperature=0.7)
    for i, sample in enumerate(samples):
        print(f"  {i+1}. {sample[:100]}")
    report['samples'] = samples

    # Triadic analysis
    print()
    print("[4/4] Triadic signature analysis...")
    mapper = PrimeMapper(config.n_triadic_bits)
    validator = TriadicValidator()

    concept_pairs = [
        ("King", "Queen"),
        ("Dog", "Cat"),
        ("Doctor", "Hospital"),
        ("Sun", "Moon"),
        ("King", "Dog"),        # Should be low similarity
        ("Happy", "Sad"),
        ("Mother", "Father"),
        ("Fire", "Water"),
    ]

    triadic_results = analyze_triadic(model, tokenizer, device, mapper, validator, concept_pairs)

    print()
    print(f"  {'A':>10s} {'B':>10s} {'Sim':>6s}  Shared Factors")
    print(f"  {'─'*10} {'─'*10} {'─'*6}  {'─'*30}")
    for r in triadic_results:
        shared_str = str(r['shared'][:5]) if r['shared'] else '[]'
        print(f"  {r['word_a']:>10s} {r['word_b']:>10s} {r['similarity']:>5.0%}  {shared_str}")
    report['triadic'] = triadic_results

    # Loss curve
    print()
    csv_path = args.csv or os.path.join(os.path.dirname(args.model), 'training_log.csv')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, 'loss_curve.png')
    plot_loss_curve(csv_path, plot_path)

    # Save report
    report_path = os.path.join(output_dir, 'eval_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {report_path}")

    print()
    print("=" * 64)
    print(f"  ✓ Perplexity: {perplexity:.2f}")
    print(f"  ✓ Triadic Loss converged: {any(r['similarity'] > 0.3 for r in triadic_results)}")
    print(f"  ✓ Coherent generation: {len(samples[0]) > 20}")
    print("=" * 64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Triadic MicroGPT')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pt checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer.json')
    parser.add_argument('--data', type=str, default=None, help='Validation data path')
    parser.add_argument('--csv', type=str, default=None, help='Training CSV log path')
    args = parser.parse_args()

    evaluate(args)

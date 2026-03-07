"""
Language Quality Benchmark — Measures generation quality with standard NLP metrics.

Metrics:
  - Perplexity (held-out validation set)
  - Distinct-n (lexical diversity)
  - Repetition rate (4-gram self-repetition)
  - MAUVE score (distribution similarity, optional — requires mauve-text)

Usage:
  python benchmarks/scripts/language_quality.py \
    --model checkpoints/torch/model_best.pt \
    --tokenizer checkpoints/torch/tokenizer.json \
    --data data/TinyStories-train.txt
"""

import os
import sys
import json
import math
import argparse
from datetime import date
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluate import load_model, compute_perplexity, generate_samples


def compute_distinct_n(texts, n_values=(1, 2, 3)):
    """Compute Distinct-n metric: ratio of unique n-grams to total n-grams."""
    results = {}
    for n in n_values:
        total_ngrams = 0
        unique_ngrams = set()
        for text in texts:
            words = text.lower().split()
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            total_ngrams += len(ngrams)
            unique_ngrams.update(ngrams)
        results[f"distinct_{n}"] = len(unique_ngrams) / max(total_ngrams, 1)
    return results


def compute_repetition_rate(texts, n=4):
    """Compute the percentage of texts containing at least one repeated n-gram."""
    repetitive = 0
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        if any(c > 1 for c in counts.values()):
            repetitive += 1
    return repetitive / max(len(texts), 1)


def compute_mauve_score(model_texts, reference_texts):
    """Compute MAUVE score (requires mauve-text package)."""
    try:
        import mauve
        result = mauve.compute_mauve(
            p_text=reference_texts,
            q_text=model_texts,
            max_text_length=256,
            verbose=False,
        )
        return result.mauve
    except ImportError:
        print("  [SKIP] mauve-text not installed. Run: pip install mauve-text")
        return None
    except Exception as e:
        print(f"  [SKIP] MAUVE computation failed: {e}")
        return None


def load_reference_stories(data_path, n=500, min_len=50):
    """Load reference stories from training data for MAUVE comparison."""
    STORY_SEP = '<' + '|endoftext|' + '>'
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    if STORY_SEP in raw:
        stories = raw.split(STORY_SEP)
    else:
        stories = raw.split('\n')

    stories = [s.strip() for s in stories if s.strip() and len(s.strip()) > min_len]
    # Use middle portion (not used in training or held-out eval)
    mid = len(stories) // 2
    return stories[mid:mid + n]


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 64)
    print("  LANGUAGE QUALITY BENCHMARK")
    print("=" * 64)
    print(f"  Model: {args.model}")
    print()

    model, tokenizer, config = load_model(args.model, args.tokenizer, device)
    print(f"  Config: {config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits")
    print(f"  Params: {model.num_params():,}")

    metrics = {}

    # 1. Perplexity
    print()
    print("[1/4] Computing perplexity...")
    ppl, avg_loss = compute_perplexity(model, tokenizer, args.data, device,
                                        max_samples=args.val_samples,
                                        block_size=config.block_size)
    metrics['perplexity'] = ppl
    metrics['avg_loss'] = avg_loss
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Avg Loss: {avg_loss:.4f}")

    # 2. Generate samples
    print()
    print(f"[2/4] Generating {args.num_generations} samples...")
    samples = generate_samples(model, tokenizer, device,
                                n=args.num_generations,
                                max_tokens=args.max_tokens,
                                temperature=0.7)

    # Show a few
    for i, s in enumerate(samples[:3]):
        print(f"  {i+1}. {s[:100]}...")

    # 3. Distinct-n
    print()
    print("[3/4] Computing lexical diversity...")
    distinct = compute_distinct_n(samples)
    metrics.update(distinct)
    for k, v in distinct.items():
        print(f"  {k}: {v:.4f}")

    # 4. Repetition rate
    rep_rate = compute_repetition_rate(samples)
    metrics['repetition_rate'] = rep_rate
    print(f"  Repetition rate (4-gram): {rep_rate:.1%}")

    # 5. MAUVE (optional)
    print()
    print("[4/4] Computing MAUVE score...")
    reference = load_reference_stories(args.data, n=min(500, args.num_generations))
    mauve_score = compute_mauve_score(samples[:len(reference)], reference)
    if mauve_score is not None:
        metrics['mauve'] = mauve_score
        print(f"  MAUVE: {mauve_score:.4f}")

    # Save results
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    results_dir = os.path.join(project_root, 'benchmarks', 'results')
    os.makedirs(results_dir, exist_ok=True)

    version = args.version
    today = date.today().isoformat()

    result = {
        "benchmark": "language_quality",
        "version": version,
        "date": today,
        "model_checkpoint": args.model,
        "model_config": f"{config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits",
        "num_generations": args.num_generations,
        "max_tokens": args.max_tokens,
        "val_samples": args.val_samples,
        "metrics": metrics,
        "sample_generations": samples[:10],
    }

    result_path = os.path.join(results_dir, f"{version}_language_quality_{today}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {result_path}")

    # Verdict
    print()
    print("=" * 64)
    print(f"  Perplexity:    {ppl:.2f} {'PASS' if ppl < 5.0 else 'CHECK'}")
    print(f"  Distinct-1:    {distinct['distinct_1']:.4f} {'PASS' if distinct['distinct_1'] > 0.5 else 'LOW'}")
    print(f"  Distinct-2:    {distinct['distinct_2']:.4f} {'PASS' if distinct['distinct_2'] > 0.7 else 'LOW'}")
    print(f"  Repetition:    {rep_rate:.1%} {'PASS' if rep_rate < 0.15 else 'HIGH'}")
    if mauve_score is not None:
        print(f"  MAUVE:         {mauve_score:.4f} {'PASS' if mauve_score > 0.7 else 'CHECK'}")
    print("=" * 64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language Quality Benchmark')
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--data', default='data/TinyStories-train.txt')
    parser.add_argument('--num-generations', type=int, default=500)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--val-samples', type=int, default=200)
    parser.add_argument('--version', default='v1.1')
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = os.path.join(os.path.dirname(args.model), 'tokenizer.json')

    main(args)

"""
Pre-tokenize — Encode a corpus and save as a .npy binary cache.

This eliminates the multi-hour tokenization bottleneck by doing it once
and saving the result. The training script can then load tokens instantly.

Usage:
  python src/pre_tokenize.py --data data/TinyStories-train.txt \
      --tokenizer checkpoints/torch/tokenizer.json \
      --stories 30000 --output data/tokens_30k.npy
"""

import os
import sys
import time
import random
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

STORY_SEPARATOR = '<' + '|endoftext|' + '>'


def pre_tokenize(args):
    print("=" * 60)
    print("  PRE-TOKENIZE — Building token cache")
    print("=" * 60)

    # Load tokenizer
    print(f"\n[1/3] Loading tokenizer: {args.tokenizer}")
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Load and split stories
    print(f"\n[2/3] Loading data: {args.data}")
    with open(args.data, 'r', encoding='utf-8', errors='ignore') as f:
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

    total_chars = sum(len(s) for s in stories)
    print(f"  Stories: {len(stories):,}")
    print(f"  Characters: {total_chars:,}")

    # Tokenize with progress
    print(f"\n[3/3] Tokenizing {len(stories):,} stories...")
    t0 = time.time()
    all_tokens = []
    for i, story in enumerate(stories):
        ids = tokenizer.encode(story, add_special=True)
        all_tokens.extend(ids)

        # Progress every 500 stories
        if (i + 1) % 500 == 0 or i == len(stories) - 1:
            elapsed = time.time() - t0
            sps = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(stories) - i - 1) / sps if sps > 0 else 0
            pct = (i + 1) / len(stories) * 100

            bar_len = 30
            filled = int(bar_len * (i + 1) / len(stories))
            bar = '█' * filled + '░' * (bar_len - filled)

            if remaining >= 60:
                eta = f"{remaining/60:.1f}m"
            else:
                eta = f"{remaining:.0f}s"

            print(f"  [{bar}] {pct:5.1f}% | {i+1}/{len(stories)} stories | {len(all_tokens):,} tokens | ETA {eta}")

    elapsed = time.time() - t0
    tokens_arr = np.array(all_tokens, dtype=np.int32)

    # Save
    output = args.output or os.path.join(os.path.dirname(args.data), f'tokens_{len(stories)//1000}k.npy')
    np.save(output, tokens_arr)

    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Total tokens: {len(all_tokens):,}")
    print(f"  Compression: {total_chars / len(all_tokens):.1f} chars/token")
    print(f"  Saved to: {output} ({tokens_arr.nbytes / 1e6:.1f} MB)")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-tokenize corpus')
    parser.add_argument('--data', type=str, required=True, help='Corpus path')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer path')
    parser.add_argument('--stories', type=int, default=None, help='Max stories')
    parser.add_argument('--output', type=str, default=None, help='Output .npy path')
    args = parser.parse_args()

    pre_tokenize(args)

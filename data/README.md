# Data Setup

## Required: TinyStories

The primary training corpus. ~1.8 GB, not included in the repo.

### Option A: HuggingFace datasets (recommended)

```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories')
with open('data/TinyStories-train.txt', 'w', encoding='utf-8') as f:
    for story in ds['train']:
        f.write(story['text'] + '\n')
print('Done — TinyStories-train.txt written')
"
```

### Option B: Direct download

Download from https://huggingface.co/datasets/roneneldan/TinyStories and extract `TinyStories-train.txt` into this directory.

## Optional: Pre-tokenized cache

Speeds up training startup by caching tokenized data as a `.npy` file. Not required — training will tokenize on the fly if missing.

```bash
python src/pre_tokenize.py --corpus data/TinyStories-train.txt --output data/tokens_30k.npy
```

## Already included

| File | Size | Description |
|------|------|-------------|
| `concepts.txt` | 1.7 KB | 31 semantic categories for debugging |
| `core_concepts.txt` | 83 KB | 10K word list for gold primes generation |
| `gold_primes_64.json` | 7.5 MB | WordNet gold primes (64-bit signatures) |
| `gold_primes_32.json` | 4.1 MB | WordNet gold primes (32-bit signatures) |
| `gold_primes.json` | 11 KB | Compact gold primes |
| `alpaca_data_cleaned.json` | 43 MB | Instruction fine-tuning pairs (52K) |
| `wikitext2_test_cache.json` | 1.2 MB | WikiText-2 eval cache (1,788 samples) |
| `lambada_test_cache.json` | 170 KB | LAMBADA eval cache (500 samples) |

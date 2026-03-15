"""
Cross-Dataset Evaluation — Validate Run 15 on WikiText-2 and LAMBADA.

Tests whether the triadic head generalizes beyond TinyStories.
Downloads datasets automatically on first run.

Usage:
  python playground/cross_dataset_eval.py
  python playground/cross_dataset_eval.py --checkpoint path/to/model.pt
"""

import os
import sys
import json
import math
import random
import argparse
import subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
from src.triadic import PrimeMapper, TriadicValidator
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
STORY_SEPARATOR = '<' + '|endoftext|' + '>'

# HuggingFace datasets server API (returns JSON, no file downloads needed)
HF_API = "https://datasets-server.huggingface.co/rows"
WIKITEXT2_PARAMS = "dataset=Salesforce/wikitext&config=wikitext-2-raw-v1&split=test"
LAMBADA_PARAMS = "dataset=EleutherAI/lambada_openai&config=default&split=test"


def _fetch_hf_rows(params, max_rows, batch=100):
    """Fetch rows from HuggingFace datasets server API via curl."""
    import time
    rows = []
    offset = 0
    retries = 0
    while offset < max_rows:
        length = min(batch, max_rows - offset)
        url = f"{HF_API}?{params}&offset={offset}&length={length}"
        result = subprocess.run(
            ["curl", "-L", "-s", "-w", "\n__HTTP__%{http_code}", url],
            timeout=30, capture_output=True, text=True,
        )
        output = result.stdout
        # Extract HTTP code from tail
        if "__HTTP__" in output:
            parts = output.rsplit("__HTTP__", 1)
            body, http_code = parts[0], parts[1].strip()
        else:
            body, http_code = output, "0"

        if http_code == "429" and retries < 3:
            retries += 1
            wait = 15 * retries
            print(f"    Rate limited, waiting {wait}s (retry {retries}/3) ...")
            time.sleep(wait)
            continue

        if http_code != "200":
            if offset == 0:
                raise RuntimeError(f"HF API failed: HTTP {http_code}")
            break

        retries = 0
        data = json.loads(body)
        batch_rows = data.get('rows', [])
        if not batch_rows:
            break
        rows.extend(batch_rows)
        offset += len(batch_rows)
        if len(batch_rows) < length:
            break
    return rows


def load_wikitext2(max_paragraphs=500):
    """Load WikiText-2 test from HuggingFace API."""
    cache = os.path.join(DATA_DIR, 'wikitext2_test_cache.json')
    if os.path.exists(cache):
        with open(cache, 'r', encoding='utf-8') as f:
            paragraphs = json.load(f)
        return paragraphs[:max_paragraphs]

    print(f"    Fetching WikiText-2 test from HuggingFace API ...")
    # WikiText-2 test has 4358 rows, many are empty/headers
    rows = _fetch_hf_rows(WIKITEXT2_PARAMS, 4400)
    paragraphs = []
    for r in rows:
        text = r['row'].get('text', '').strip()
        if not text or text.startswith('=') or len(text) < 50:
            continue
        paragraphs.append(text)

    with open(cache, 'w', encoding='utf-8') as f:
        json.dump(paragraphs, f)
    print(f"    Cached {len(paragraphs)} paragraphs to {cache}")
    return paragraphs[:max_paragraphs]


def load_lambada(max_samples=500):
    """Load LAMBADA test from HuggingFace API."""
    cache = os.path.join(DATA_DIR, 'lambada_test_cache.json')
    if os.path.exists(cache):
        with open(cache, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        return sentences[:max_samples]

    print(f"    Fetching LAMBADA test from HuggingFace API ...")
    rows = _fetch_hf_rows(LAMBADA_PARAMS, max_samples)
    sentences = []
    for r in rows:
        text = r['row'].get('text', '').strip()
        if text and len(text) > 20:
            sentences.append(text)

    with open(cache, 'w', encoding='utf-8') as f:
        json.dump(sentences, f)
    print(f"    Cached {len(sentences)} sentences to {cache}")
    return sentences[:max_samples]


def load_tinystories_val(max_samples=500):
    """Load TinyStories validation split (last N stories)."""
    path = os.path.join(DATA_DIR, 'TinyStories-train.txt')
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 50]
    return stories[-max_samples:]


def compute_ppl(model, tokenizer, texts, device, block_size, dataset_name=""):
    """Compute perplexity on a list of text passages."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    skipped = 0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, add_special=True)
            if len(ids) < 3:
                skipped += 1
                continue
            ids = ids[:block_size + 1]
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            _, _, loss = model(x, targets=y)
            n = len(ids) - 1
            total_loss += loss.item() * n
            total_tokens += n

    ppl = math.exp(total_loss / max(total_tokens, 1))
    if dataset_name:
        print(f"    {dataset_name}: PPL={ppl:.2f}  "
              f"(tokens={total_tokens:,}, passages={len(texts)-skipped}, skipped={skipped})")
    return ppl, total_tokens


def compute_triadic_metrics(model, tokenizer, device, n_bits):
    """Compute semantic gap and analogy verification."""
    mapper = PrimeMapper(n_bits)
    concept_pairs = {
        'related': [("king","queen"),("dog","cat"),("happy","sad"),("mother","father"),
                    ("sun","moon"),("hot","cold"),("love","hate"),("big","small"),("bird","fish"),
                    ("doctor","hospital"),("teacher","school"),("princess","prince"),("old","young")],
        'unrelated': [("king","fish"),("dog","moon"),("happy","river"),("mother","blue"),
                      ("sun","cat"),("hot","queen"),("bird","school"),("love","tree"),("big","night")],
    }
    analogy_triples = [
        ("king","queen","man","woman"),("father","mother","brother","sister"),
        ("father","mother","son","daughter"),("dog","puppy","cat","kitten"),
        ("big","small","tall","short"),("hot","cold","day","night"),
        ("happy","sad","love","hate"),("princess","prince","queen","king"),
        ("bird","fly","fish","swim"),("old","young","big","small"),
        ("doctor","hospital","teacher","school"),("sun","day","moon","night"),
        ("red","blue","green","yellow"),
    ]

    all_words = set()
    for group in concept_pairs.values():
        for w1, w2 in group:
            all_words.update([w1, w2])
    for a, b, c, d in analogy_triples:
        all_words.update([a, b, c, d])

    sigs = {}
    model.eval()
    with torch.no_grad():
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if ids:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                _, proj, _ = model(x)
                sigs[word] = proj[0].mean(dim=0).cpu().numpy()

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    related = [cosine(sigs[w1], sigs[w2]) for w1, w2 in concept_pairs['related'] if w1 in sigs and w2 in sigs]
    rand_sims = []
    words = list(sigs.keys())
    for _ in range(200):
        i, j = random.sample(range(len(words)), 2)
        rand_sims.append(cosine(sigs[words[i]], sigs[words[j]]))
    gap = np.mean(related) - np.mean(rand_sims)

    correct, total = 0, 0
    for a, b, c, d in analogy_triples:
        if not all(w in sigs for w in [a, b, c, d]):
            continue
        predicted = TriadicValidator.analogy(mapper.map(sigs[a]), mapper.map(sigs[b]), mapper.map(sigs[c]))
        if TriadicValidator.similarity(predicted, mapper.map(sigs[d])) > 0.3:
            correct += 1
        total += 1

    all_projs = np.stack(list(sigs.values()))
    bit_means = (all_projs > 0).mean(axis=0)
    eps = 1e-7
    bit_entropy = -(bit_means * np.log2(bit_means + eps) + (1 - bit_means) * np.log2(1 - bit_means + eps))
    dead_bits = int((bit_entropy < 0.3).sum())

    return {
        'semantic_gap': float(gap),
        'analogy_verification': correct / max(total, 1),
        'dead_bits': dead_bits,
        'mean_bit_entropy': float(bit_entropy.mean()),
        'unique_signatures': len(set(mapper.map(p) for p in all_projs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign',
                                             'model_L12_D512_B64_best.pt'))
    parser.add_argument('--max-samples', type=int, default=500)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"  CROSS-DATASET EVALUATION")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"  Max samples per dataset: {args.max_samples}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'], block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'], n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'], dropout=0.0,
    )
    model = TriadicGPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tok_path = os.path.join(os.path.dirname(args.checkpoint), 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)
    block_size = cfg['block_size']
    n_bits = cfg['n_triadic_bits']

    print(f"  Model: {cfg['n_layer']}L/{cfg['n_embd']}D/{n_bits}bits "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    print(f"  Block size: {block_size}, Vocab: {cfg['vocab_size']}")

    # --- Load datasets (fetches from HuggingFace API on first run, then caches) ---
    print(f"\n  --- Loading datasets ---")
    tinystories = load_tinystories_val(args.max_samples)
    wikitext = load_wikitext2(args.max_samples)
    lambada = load_lambada(args.max_samples)
    print(f"    TinyStories: {len(tinystories)} passages")
    print(f"    WikiText-2:  {len(wikitext)} paragraphs")
    print(f"    LAMBADA:     {len(lambada)} sentences")

    # --- Tokenization stats ---
    print(f"\n  --- Tokenization stats ---")
    for name, texts in [("TinyStories", tinystories), ("WikiText-2", wikitext), ("LAMBADA", lambada)]:
        sample = texts[:100]
        all_ids = [tokenizer.encode(t, add_special=False) for t in sample]
        total_chars = sum(len(t) for t in sample)
        total_toks = sum(len(ids) for ids in all_ids)
        chars_per_tok = total_chars / max(total_toks, 1)
        unk_id = 3  # <UNK>
        unk_count = sum(ids.count(unk_id) for ids in all_ids)
        unk_rate = unk_count / max(total_toks, 1)
        print(f"    {name:>12s}: {chars_per_tok:.1f} chars/tok, "
              f"UNK rate={unk_rate:.1%} ({unk_count}/{total_toks})")

    # --- Perplexity ---
    print(f"\n  --- Perplexity ---")
    ppl_ts, tok_ts = compute_ppl(model, tokenizer, tinystories, device, block_size, "TinyStories")
    ppl_wt, tok_wt = compute_ppl(model, tokenizer, wikitext, device, block_size, "WikiText-2")
    ppl_lb, tok_lb = compute_ppl(model, tokenizer, lambada, device, block_size, "LAMBADA")

    # --- Triadic metrics ---
    print(f"\n  --- Triadic Metrics (model-intrinsic, dataset-independent) ---")
    triadic = compute_triadic_metrics(model, tokenizer, device, n_bits)
    print(f"    Semantic gap:      {triadic['semantic_gap']:+.4f}")
    print(f"    Analogy verif:     {triadic['analogy_verification']:.1%}")
    print(f"    Dead bits:         {triadic['dead_bits']}")
    print(f"    Bit entropy:       {triadic['mean_bit_entropy']:.4f}")
    print(f"    Unique sigs:       {triadic['unique_signatures']}")

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Dataset':>15s}  {'PPL':>10s}  {'Tokens':>10s}  {'vs TinyStories':>15s}")
    print(f"  {'─'*15}  {'─'*10}  {'─'*10}  {'─'*15}")
    print(f"  {'TinyStories':>15s}  {ppl_ts:>10.2f}  {tok_ts:>10,}  {'baseline':>15s}")
    print(f"  {'WikiText-2':>15s}  {ppl_wt:>10.2f}  {tok_wt:>10,}  {f'+{(ppl_wt/ppl_ts - 1)*100:.0f}%':>15s}")
    print(f"  {'LAMBADA':>15s}  {ppl_lb:>10.2f}  {tok_lb:>10,}  {f'+{(ppl_lb/ppl_ts - 1)*100:.0f}%':>15s}")

    # --- Save results ---
    results = {
        'checkpoint': args.checkpoint,
        'perplexity': {'tinystories': ppl_ts, 'wikitext2': ppl_wt, 'lambada': ppl_lb},
        'tokens': {'tinystories': tok_ts, 'wikitext2': tok_wt, 'lambada': tok_lb},
        'triadic': triadic,
        'model_config': cfg,
    }
    out_path = os.path.join(PROJECT_ROOT, 'playground', 'results', 'cross_dataset_eval.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

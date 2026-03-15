"""
Evaluate a specific XL Subsumption checkpoint.
Loads model from checkpoint and runs full evaluation: PPL, semantic gap, subsumption.

Usage:
  python playground/eval_xl_sub_checkpoint.py --checkpoint playground/checkpoints_xl_subsumption/model_step25000.pt
"""

import os
import sys
import json
import math
import random
import argparse
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
STORY_SEPARATOR = '<' + '|endoftext|' + '>'

HYPERNYM_PAIRS = {
    "animal": ["dog", "cat", "bird", "fish", "horse", "rabbit", "bear", "mouse", "lion"],
    "person": ["king", "queen", "doctor", "teacher", "princess", "prince", "boy", "girl"],
    "feeling": ["happy", "sad", "love", "hate", "angry", "scared"],
    "food": ["apple", "cake", "bread", "candy", "cookie"],
    "color": ["red", "blue", "green", "yellow", "pink", "purple"],
    "place": ["school", "hospital", "house", "garden", "forest", "beach", "park"],
    "time": ["day", "night", "morning", "evening"],
}

HELD_OUT_PAIRS = {
    "animal": ["tiger", "frog", "deer"],
    "person": ["man", "woman", "baby"],
    "food": ["pizza", "milk", "egg"],
    "place": ["castle", "farm", "river"],
}


def evaluate_subsumption(model, tokenizer, device, pairs_dict, mapper, label=""):
    model.eval()
    all_words = set()
    for hyper, hypos in pairs_dict.items():
        all_words.add(hyper)
        all_words.update(hypos)

    sigs, projs = {}, {}
    with torch.no_grad():
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            _, proj, _ = model(x)
            proj_np = proj[0].mean(dim=0).cpu().numpy()
            projs[word] = proj_np
            sigs[word] = mapper.map(proj_np)

    results = []
    total, subsumes_count = 0, 0
    inheritance_scores = []

    for hypernym, hyponyms in pairs_dict.items():
        if hypernym not in sigs:
            continue
        hyper_bits = (projs[hypernym] > 0).astype(int)
        for hyponym in hyponyms:
            if hyponym not in sigs:
                continue
            hypo_bits = (projs[hyponym] > 0).astype(int)
            total += 1
            is_sub = TriadicValidator.subsumes(sigs[hyponym], sigs[hypernym])
            if is_sub:
                subsumes_count += 1
            hyper_active = hyper_bits.sum()
            inheritance = (hyper_bits * hypo_bits).sum() / max(hyper_active, 1) if hyper_active > 0 else 1.0
            inheritance_scores.append(float(inheritance))
            results.append({
                'pair': f'{hypernym}->{hyponym}', 'subsumes': bool(is_sub),
                'bit_inheritance': float(inheritance),
                'hyper_active_bits': int(hyper_active),
                'shared_bits': int((hyper_bits * hypo_bits).sum()),
            })

    sub_rate = subsumes_count / max(total, 1)
    mean_inh = np.mean(inheritance_scores) if inheritance_scores else 0.0

    if label:
        print(f"\n  [{label}] Subsumption Results:")
        print(f"    Algebraic subsumption: {subsumes_count}/{total} ({sub_rate:.1%})")
        print(f"    Mean bit inheritance:  {mean_inh:.1%}")
        for hypernym in pairs_dict:
            pr = [r for r in results if r['pair'].startswith(f'{hypernym}->')]
            if pr:
                h_inh = np.mean([r['bit_inheritance'] for r in pr])
                h_sub = sum(r['subsumes'] for r in pr)
                h_bits = pr[0]['hyper_active_bits']
                print(f"      {hypernym:>10s}: inheritance={h_inh:.0%}  "
                      f"subsumption={h_sub}/{len(pr)}  hyper_bits={h_bits}")

    return {'subsumption_rate': float(sub_rate), 'mean_bit_inheritance': float(mean_inh),
            'total_pairs': total, 'details': results}


def evaluate_model(model, tokenizer, device, n_bits):
    model.eval()
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
    with torch.no_grad():
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if ids:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                _, proj, _ = model(x)
                sigs[word] = proj[0].mean(dim=0).cpu().numpy()

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    related_sims = [cosine(sigs[w1], sigs[w2]) for w1, w2 in concept_pairs['related'] if w1 in sigs and w2 in sigs]
    random_sims = []
    words = list(sigs.keys())
    for _ in range(200):
        i, j = random.sample(range(len(words)), 2)
        random_sims.append(cosine(sigs[words[i]], sigs[words[j]]))
    semantic_gap = np.mean(related_sims) - np.mean(random_sims)

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
        'semantic_gap': float(semantic_gap), 'mean_bit_entropy': float(bit_entropy.mean()),
        'dead_bits': dead_bits, 'active_bits': n_bits - dead_bits,
        'analogy_verification': correct / max(total, 1),
        'unique_signatures': len(set(mapper.map(p) for p in all_projs)),
    }


def compute_perplexity(model, tokenizer, data_path, device, block_size, max_samples=200):
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 50]
    val_stories = stories[-max_samples:]
    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for story in val_stories:
            ids = tokenizer.encode(story, add_special=True)
            if len(ids) < 3:
                continue
            ids = ids[:block_size + 1]
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            _, _, loss = model(x, targets=y)
            total_loss += loss.item() * (len(ids) - 1)
            total_tokens += len(ids) - 1
    return math.exp(total_loss / max(total_tokens, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  EVALUATING CHECKPOINT: {os.path.basename(args.checkpoint)}")
    print(f"{'='*70}")
    print(f"  Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']
    step = ckpt.get('step', '?')
    print(f"  Step: {step}")
    print(f"  Train loss at checkpoint: {ckpt.get('loss', '?'):.4f}")

    # Load tokenizer
    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)

    # Build model
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'], block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'], n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'], dropout=0.0,
    )
    model = TriadicGPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    mapper = PrimeMapper(64)

    # PPL
    data_path = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
    ppl = compute_perplexity(model, tokenizer, data_path, device, cfg['block_size'])
    print(f"\n  Perplexity: {ppl:.2f} (Run 15: 7.69)")

    # Semantic
    sem = evaluate_model(model, tokenizer, device, 64)
    print(f"  Semantic gap:  {sem['semantic_gap']:+.4f} (Run 15: +0.020)")
    print(f"  Dead bits:     {sem['dead_bits']} (Run 15: 15)")
    print(f"  Bit entropy:   {sem['mean_bit_entropy']:.4f} (Run 15: 0.749)")
    print(f"  Analogy verif: {sem['analogy_verification']:.1%}")

    # Subsumption
    eval_train = evaluate_subsumption(model, tokenizer, device, HYPERNYM_PAIRS, mapper, "TRAIN")
    eval_test = evaluate_subsumption(model, tokenizer, device, HELD_OUT_PAIRS, mapper, "TEST (held-out)")

    # Comparison table
    print(f"\n{'='*70}")
    print(f"  CHECKPOINT step {step} vs RUN 15 vs FINAL (50K)")
    print(f"{'='*70}")
    print(f"  {'Metric':>25s}  {'Step {}'.format(step):>15s}  {'Run 15':>10s}  {'Final 50K':>10s}")
    print(f"  {'─'*25}  {'─'*15}  {'─'*10}  {'─'*10}")
    rows = [
        ('Perplexity', ppl, 7.69, 16.37),
        ('Semantic gap', sem['semantic_gap'], 0.020, 0.025),
        ('Dead bits', sem['dead_bits'], 15, 22),
        ('Bit entropy', sem['mean_bit_entropy'], 0.749, 0.491),
        ('Sub (train)', eval_train['subsumption_rate'], 0.0, 0.778),
        ('Sub (test)', eval_test['subsumption_rate'], 0.0, 0.667),
        ('Inheritance (train)', eval_train['mean_bit_inheritance'], 0.0, 1.0),
        ('Inheritance (test)', eval_test['mean_bit_inheritance'], 0.0, 0.958),
    ]
    for name, v_ckpt, v_r15, v_final in rows:
        print(f"  {name:>25s}  {v_ckpt:>15.4f}  {v_r15:>10.4f}  {v_final:>10.4f}")

    # Save
    results = {
        'checkpoint': args.checkpoint, 'step': step,
        'perplexity': ppl, **sem,
        'subsumption_train': eval_train, 'subsumption_test': eval_test,
    }
    out_path = os.path.join(PROJECT_ROOT, 'playground', 'results', f'xl_sub_step{step}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

"""
SimLex-999 Benchmark — External Anchor for Semantic Quality.

Evaluates triadic signatures against SimLex-999 human similarity judgments
(Hill et al., 2015). This is the CRITICAL external validation that breaks
the circular dependency in the triadic program:
  P2 defines ontology -> P4 trains with it -> P3 validates P4 against P2

SimLex-999 provides an independent human ground truth.

Kill criterion K1: if Spearman rho < 0.15, degrade claim to
"algebraic encoding with limited semantic grounding".

Metrics:
  - Spearman rho: rank correlation between Jaccard similarity of triadic
    signatures and SimLex-999 human judgments
  - Coverage: fraction of SimLex-999 pairs encodable by the model
  - WordNet subsumption accuracy on the covered pairs

Usage:
  python benchmarks/scripts/simlex_benchmark.py \
    --model checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt \
    --tokenizer checkpoints/torch_runXL/tokenizer.json \
    --version v1.4-strongalign

The script auto-downloads SimLex-999 if not present.
"""

import os
import sys
import json
import csv
import argparse
import urllib.request
from datetime import date

import numpy as np
from scipy import stats
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluate import load_model
from src.triadic import PrimeMapper, TriadicValidator, prime_factors


SIMLEX_URL = "https://fh295.github.io/SimLex-999.zip"
SIMLEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SIMLEX_FILE = os.path.join(SIMLEX_DIR, 'SimLex-999.txt')


def ensure_simlex():
    """Download SimLex-999 if not already present."""
    if os.path.exists(SIMLEX_FILE):
        return SIMLEX_FILE

    os.makedirs(SIMLEX_DIR, exist_ok=True)
    zip_path = os.path.join(SIMLEX_DIR, 'SimLex-999.zip')

    print(f"  Downloading SimLex-999...")
    urllib.request.urlretrieve(SIMLEX_URL, zip_path)

    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Extract the .txt file
        for name in z.namelist():
            if name.endswith('SimLex-999.txt'):
                z.extract(name, SIMLEX_DIR)
                extracted = os.path.join(SIMLEX_DIR, name)
                if extracted != SIMLEX_FILE:
                    os.rename(extracted, SIMLEX_FILE)
                break

    os.remove(zip_path)
    print(f"  SimLex-999 saved to: {SIMLEX_FILE}")
    return SIMLEX_FILE


def load_simlex(path):
    """Load SimLex-999 pairs and human similarity scores."""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            w1 = row['word1'].lower()
            w2 = row['word2'].lower()
            score = float(row['SimLex999'])
            pos = row.get('POS', '')
            pairs.append({
                'word1': w1,
                'word2': w2,
                'human_score': score,
                'pos': pos,
            })
    return pairs


def compute_projections(model, tokenizer, words, device):
    """Get triadic projections for all words."""
    mapper = PrimeMapper(model.config.n_triadic_bits)
    results = {}

    for word in words:
        if word in results:
            continue
        ids = tokenizer.encode(word, add_special=False)
        if not ids:
            continue

        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)

        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        prime = mapper.map(proj)
        bits = mapper.get_bits(proj)
        results[word] = {
            'projection': proj,
            'prime': prime,
            'bits': bits,
            'n_active': sum(bits),
        }

    return results


def jaccard_similarity(prime_a, prime_b):
    """Jaccard similarity of prime factor sets."""
    fa = set(prime_factors(prime_a))
    fb = set(prime_factors(prime_b))
    if not fa and not fb:
        return 1.0
    union = fa | fb
    return len(fa & fb) / len(union) if union else 0.0


def cosine_similarity(proj_a, proj_b):
    """Cosine similarity between raw projections."""
    norm_a = np.linalg.norm(proj_a)
    norm_b = np.linalg.norm(proj_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(proj_a, proj_b) / (norm_a * norm_b))


def evaluate_simlex(concept_data, simlex_pairs):
    """Evaluate against SimLex-999."""
    human_scores = []
    jaccard_scores = []
    cosine_scores = []
    covered_pairs = []

    for pair in simlex_pairs:
        w1, w2 = pair['word1'], pair['word2']
        if w1 not in concept_data or w2 not in concept_data:
            continue

        human_scores.append(pair['human_score'])
        j_sim = jaccard_similarity(
            concept_data[w1]['prime'], concept_data[w2]['prime'])
        c_sim = cosine_similarity(
            concept_data[w1]['projection'], concept_data[w2]['projection'])
        jaccard_scores.append(j_sim)
        cosine_scores.append(c_sim)
        covered_pairs.append({
            'word1': w1, 'word2': w2,
            'human': pair['human_score'],
            'jaccard': j_sim,
            'cosine': c_sim,
        })

    n_total = len(simlex_pairs)
    n_covered = len(human_scores)
    coverage = n_covered / n_total if n_total > 0 else 0.0

    # Spearman correlations
    if n_covered >= 3:
        rho_jaccard, p_jaccard = stats.spearmanr(human_scores, jaccard_scores)
        rho_cosine, p_cosine = stats.spearmanr(human_scores, cosine_scores)
    else:
        rho_jaccard, p_jaccard = 0.0, 1.0
        rho_cosine, p_cosine = 0.0, 1.0

    # By POS
    pos_results = {}
    for pos in ['N', 'V', 'A']:
        pos_human, pos_jaccard = [], []
        for pair, j in zip(covered_pairs, jaccard_scores):
            orig = next((p for p in simlex_pairs
                         if p['word1'] == pair['word1']
                         and p['word2'] == pair['word2']), None)
            if orig and orig.get('pos', '') == pos:
                pos_human.append(pair['human'])
                pos_jaccard.append(j)
        if len(pos_human) >= 3:
            r, p = stats.spearmanr(pos_human, pos_jaccard)
            pos_results[pos] = {'rho': float(r), 'p': float(p),
                                'n': len(pos_human)}

    return {
        'n_total': n_total,
        'n_covered': n_covered,
        'coverage': coverage,
        'rho_jaccard': float(rho_jaccard),
        'p_jaccard': float(p_jaccard),
        'rho_cosine': float(rho_cosine),
        'p_cosine': float(p_cosine),
        'by_pos': pos_results,
        'kill_k1': float(rho_jaccard) < 0.15,
        'sample_pairs': covered_pairs[:20],
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 68)
    print("  SIMLEX-999 BENCHMARK — External Semantic Anchor")
    print("=" * 68)
    print(f"  Model: {args.model}")
    print()

    # Load SimLex-999
    print("[1/3] Loading SimLex-999...")
    simlex_path = ensure_simlex()
    simlex_pairs = load_simlex(simlex_path)
    print(f"  Loaded {len(simlex_pairs)} pairs")

    # Load model
    print()
    print("[2/3] Loading model and computing projections...")
    model, tokenizer, config = load_model(args.model, args.tokenizer, device)
    print(f"  Config: {config.n_layer}L/{config.n_embd}D/"
          f"{config.n_head}H/{config.n_triadic_bits}bits")

    # All unique words from SimLex
    all_words = set()
    for p in simlex_pairs:
        all_words.add(p['word1'])
        all_words.add(p['word2'])
    print(f"  Unique words in SimLex: {len(all_words)}")

    concept_data = compute_projections(model, tokenizer, all_words, device)
    print(f"  Encoded: {len(concept_data)}/{len(all_words)} "
          f"({len(concept_data)/len(all_words):.0%} coverage)")

    # Evaluate
    print()
    print("[3/3] Computing correlations...")
    result = evaluate_simlex(concept_data, simlex_pairs)

    print()
    print(f"  Coverage:          {result['coverage']:.1%} "
          f"({result['n_covered']}/{result['n_total']})")
    print(f"  Spearman rho (Jaccard): {result['rho_jaccard']:.4f} "
          f"(p={result['p_jaccard']:.4e})")
    print(f"  Spearman rho (cosine):  {result['rho_cosine']:.4f} "
          f"(p={result['p_cosine']:.4e})")

    if result['by_pos']:
        print()
        print("  By POS:")
        for pos, r in result['by_pos'].items():
            label = {'N': 'Nouns', 'V': 'Verbs', 'A': 'Adjectives'}.get(pos, pos)
            print(f"    {label:>12}: rho={r['rho']:.4f} (n={r['n']})")

    # Sample pairs
    print()
    print("  Sample pairs (sorted by human score):")
    sorted_sample = sorted(result['sample_pairs'],
                           key=lambda x: x['human'], reverse=True)
    for p in sorted_sample[:8]:
        print(f"    {p['word1']:>10} ~ {p['word2']:<10} "
              f"human={p['human']:5.2f}  jaccard={p['jaccard']:.3f}  "
              f"cosine={p['cosine']:.3f}")

    # Save results
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    results_dir = os.path.join(project_root, 'benchmarks', 'results')
    os.makedirs(results_dir, exist_ok=True)

    version = args.version
    today = date.today().isoformat()

    save_data = {
        "benchmark": "simlex999",
        "version": version,
        "date": today,
        "model_checkpoint": args.model,
        "model_config": f"{config.n_layer}L/{config.n_embd}D/"
                        f"{config.n_head}H/{config.n_triadic_bits}bits",
        "metrics": {
            "coverage": result['coverage'],
            "n_covered": result['n_covered'],
            "n_total": result['n_total'],
            "rho_jaccard": result['rho_jaccard'],
            "p_jaccard": result['p_jaccard'],
            "rho_cosine": result['rho_cosine'],
            "p_cosine": result['p_cosine'],
            "by_pos": result['by_pos'],
        },
        "kill_k1": result['kill_k1'],
        "sample_pairs": result['sample_pairs'],
    }

    result_path = os.path.join(results_dir,
                               f"{version}_simlex999_{today}.json")
    with open(result_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved: {result_path}")

    # Verdict
    print()
    print("=" * 68)
    rho = result['rho_jaccard']
    print(f"  KILL CRITERION K1:")
    print(f"    Spearman rho = {rho:.4f}  (threshold: >= 0.15)")
    if rho >= 0.15:
        print(f"    PASS — algebraic semantics claim supported")
    else:
        print(f"    FAIL — degrade to 'algebraic encoding with "
              f"limited semantic grounding'")
    print(f"  CONTEXT:")
    print(f"    Random baseline:  rho ~ 0.00")
    print(f"    Word2Vec (2013):  rho ~ 0.44")
    print(f"    GloVe (2014):     rho ~ 0.37")
    print(f"    BERT (2019):      rho ~ 0.30-0.45")
    print(f"    Our model:        rho = {rho:.4f} "
          f"(on {result['n_covered']} covered pairs)")
    print("=" * 68)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimLex-999 Benchmark')
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--version', default='v1.4-strongalign')
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = os.path.join(os.path.dirname(args.model),
                                      'tokenizer.json')

    main(args)

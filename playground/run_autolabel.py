"""
Run AutoLabeler on D-A14 v2 (best model) to propose v3 primitives.

Loads model on CPU (safe while D-A18 trains on GPU), extracts bit codes
for all 158 anchors + 200 common English words, runs BitDiscovery + AutoLabeler,
exports proposed primitives for human review.

Usage:
    conda run -n triadic-microgpt python playground/run_autolabel.py
"""

import os
import sys
import json
import numpy as np
import torch

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from reptimeline.extractors.triadic import TriadicExtractor
from reptimeline.discovery import BitDiscovery
from reptimeline.autolabel import AutoLabeler
from reptimeline.core import ConceptSnapshot

# D-A14 v2 checkpoint (93% test, 98.3% sub — best model)
CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', 'danza_63bit_xl_v2')
CKPT_BEST = os.path.join(CKPT_DIR, 'model_best.pt')
TOK_PATH = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'reptimeline', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Common English words likely in TinyStories — broader than anchors
EXTRA_CONCEPTS = [
    # Emotions
    'angry', 'scared', 'brave', 'gentle', 'kind', 'mean', 'shy',
    'excited', 'worried', 'calm', 'nervous', 'cheerful', 'grumpy',
    # People
    'boy', 'girl', 'mother', 'father', 'baby', 'friend', 'teacher',
    'child', 'sister', 'brother', 'grandma', 'doctor', 'princess',
    # Animals
    'dog', 'cat', 'bird', 'fish', 'bear', 'rabbit', 'horse', 'fox',
    'lion', 'mouse', 'butterfly', 'elephant', 'tiger', 'wolf',
    # Nature
    'tree', 'flower', 'rain', 'snow', 'wind', 'river', 'mountain',
    'ocean', 'star', 'cloud', 'garden', 'forest', 'fire', 'water',
    'earth', 'sky', 'stone', 'ice', 'grass', 'sand',
    # Objects
    'house', 'door', 'window', 'book', 'ball', 'toy', 'cake', 'gift',
    'crown', 'sword', 'mirror', 'key', 'bridge', 'boat', 'tower',
    # Actions/States
    'run', 'jump', 'fly', 'sleep', 'eat', 'play', 'sing', 'dance',
    'fight', 'help', 'build', 'break', 'grow', 'fall', 'hide', 'find',
    # Abstract
    'truth', 'lie', 'hope', 'fear', 'dream', 'power', 'magic', 'secret',
    'peace', 'war', 'life', 'death', 'time', 'space', 'light', 'shadow',
    'strength', 'wisdom', 'beauty', 'courage', 'freedom', 'justice',
    # Qualities
    'big', 'small', 'old', 'new', 'young', 'tall', 'soft', 'hard',
    'warm', 'cool', 'wet', 'dry', 'clean', 'dirty', 'strong', 'weak',
    'empty', 'full', 'heavy', 'light', 'sharp', 'round', 'deep',
    # Colors
    'red', 'blue', 'green', 'white', 'black', 'gold', 'silver',
    # Time
    'morning', 'night', 'day', 'winter', 'summer', 'spring',
    # Social
    'king', 'queen', 'prince', 'village', 'castle', 'family', 'home',
]

# Anchor words from danza_63bit.py ANCHOR_TRANSLATIONS
ANCHOR_WORDS = [
    'cold', 'hot', 'love', 'hate', 'indifference', 'man', 'woman',
    'good', 'evil', 'wise', 'ignorant', 'creative', 'logical',
    'alive', 'dead', 'happy', 'sad', 'king', 'queen', 'sun', 'moon',
    'darkness', 'free', 'prisoner', 'fast', 'quick', 'slow', 'still',
    'frozen', 'rich', 'poor', 'proud', 'humble', 'sweet', 'bitter',
    'loud', 'noisy', 'quiet', 'silent', 'bright', 'shiny', 'dark',
    'solid', 'hard', 'liquid', 'gas', 'teach', 'learn', 'open',
    'close', 'order', 'chaos', 'apathy', 'bad',
]

# v2 anchor words (from anclas_v2.json 'en' field)
V2_ANCHOR_WORDS = [
    'socialism', 'capitalism', 'tyranny', 'democracy', 'fanaticism',
    'anarchy', 'utopia', 'dystopia', 'revolution', 'tradition',
    'nostalgia', 'hope', 'despair', 'ecstasy', 'melancholy',
    'rage', 'serenity', 'anxiety', 'gratitude', 'envy',
    'compassion', 'cruelty', 'forgiveness', 'revenge', 'loyalty',
    'betrayal', 'innocence', 'corruption', 'purity', 'contamination',
    'abundance', 'scarcity', 'generosity', 'greed', 'sacrifice',
    'selfishness', 'humility', 'arrogance', 'patience', 'impulsiveness',
    'curiosity', 'indifference', 'obsession', 'acceptance', 'denial',
    'rebellion', 'submission', 'independence', 'dependence', 'solitude',
    'community', 'silence', 'noise', 'harmony', 'discord',
    'creation', 'destruction', 'evolution', 'stagnation', 'transformation',
    'permanence', 'birth', 'death', 'growth', 'decay',
    'enlightenment', 'ignorance', 'meditation', 'distraction', 'focus',
    'confusion', 'clarity', 'doubt', 'faith', 'reason',
    'intuition', 'logic', 'imagination', 'reality', 'illusion',
    'truth', 'deception', 'honesty', 'hypocrisy', 'authenticity',
    'artifice', 'nature', 'technology', 'instinct', 'calculation',
    'spontaneity', 'routine', 'adventure', 'comfort', 'risk',
    'security', 'vulnerability', 'resistance', 'flow', 'control',
]

def main():
    print("=" * 60)
    print("  AUTOLABEL: D-A14 v2 → Propose v3 Primitives")
    print("=" * 60)

    # Merge all concepts (deduplicate)
    all_concepts = sorted(set(
        ANCHOR_WORDS + V2_ANCHOR_WORDS + EXTRA_CONCEPTS
    ))
    print(f"\n  Total concepts to analyze: {len(all_concepts)}")

    # Step 1: Extract bit codes using TriadicExtractor (CPU)
    print("\n  [1/5] Extracting bit codes from D-A14 v2 (CPU)...")
    extractor = TriadicExtractor(tokenizer_path=TOK_PATH, n_bits=63)
    snapshot = extractor.extract(CKPT_BEST, all_concepts, device='cpu')
    print(f"    Extracted {len(snapshot.codes)} / {len(all_concepts)} concepts")
    print(f"    Code dimension: {snapshot.code_dim}")

    # Step 2: Get wte embeddings for all concepts
    print("\n  [2/5] Extracting wte embeddings...")
    from src.evaluate import load_model
    model, tokenizer, config = load_model(CKPT_BEST, TOK_PATH, 'cpu')
    wte = model.wte.weight.detach().numpy()  # (vocab_size, d_model)

    embeddings = {}
    for concept in snapshot.codes.keys():
        ids = tokenizer.encode(concept, add_special=False)[:4]
        if ids:
            # Mean-pool token embeddings for multi-token concepts
            vecs = [wte[tid] for tid in ids if tid < wte.shape[0]]
            if vecs:
                embeddings[concept] = np.mean(vecs, axis=0)

    del model
    print(f"    Embeddings for {len(embeddings)} concepts (d={wte.shape[1]})")

    # Step 3: Run BitDiscovery
    print("\n  [3/5] Running BitDiscovery...")
    discovery = BitDiscovery(
        dead_threshold=0.02,
        dual_threshold=-0.3,
        dep_confidence=0.9,
        triadic_threshold=0.7,
        triadic_min_interaction=0.2,
    )
    report = discovery.discover(snapshot, top_k=15)
    discovery.print_report(report)

    # Step 4: Run AutoLabeler (both strategies)
    print("\n  [4/5] Running AutoLabeler...")
    labeler = AutoLabeler()

    print("\n  --- Strategy 1: Embedding Centroid ---")
    labels_emb = labeler.label_by_embedding(report, embeddings)
    labeler.print_labels(labels_emb)

    print("\n  --- Strategy 2: Contrastive ---")
    labels_con = labeler.label_by_contrast(report, embeddings)
    labeler.print_labels(labels_con)

    # Step 5: Export as primitives
    print("\n  [5/5] Exporting proposed v3 primitives...")

    # Use contrastive labels (generally better for distinguishing)
    out_path = os.path.join(OUTPUT_DIR, 'd_a14_v2_autolabel.json')
    labeler.export_as_primitives(labels_con, out_path)
    print(f"    Saved: {out_path}")

    # Also save a combined report with both strategies
    combined = {
        'model': 'D-A14 v2 tanh (93% test, 98.3% sub)',
        'checkpoint': CKPT_BEST,
        'n_concepts_analyzed': len(snapshot.codes),
        'n_active_bits': report.n_active_bits,
        'n_dead_bits': report.n_dead_bits,
        'n_duals': len(report.discovered_duals),
        'n_dependencies': len(report.discovered_deps),
        'n_triadic_interactions': len(report.discovered_triadic_deps),
        'labels_embedding': [
            {
                'bit': bl.bit_index,
                'label': bl.label,
                'confidence': round(bl.confidence, 3),
                'active_concepts': bl.active_concepts[:10],
                'inactive_concepts': bl.inactive_concepts[:5],
            }
            for bl in labels_emb if bl.label != "DEAD"
        ],
        'labels_contrastive': [
            {
                'bit': bl.bit_index,
                'label': bl.label,
                'confidence': round(bl.confidence, 3),
                'active_concepts': bl.active_concepts[:10],
                'inactive_concepts': bl.inactive_concepts[:5],
            }
            for bl in labels_con if bl.label != "DEAD"
        ],
        'bit_semantics': [
            {
                'bit': bs.bit_index,
                'activation_rate': round(bs.activation_rate, 3),
                'top_concepts': bs.top_concepts[:10],
                'anti_concepts': bs.anti_concepts[:5],
            }
            for bs in report.bit_semantics
        ],
        'discovered_duals': [
            {
                'bit_a': d.bit_a,
                'bit_b': d.bit_b,
                'anti_correlation': round(d.anti_correlation, 3),
            }
            for d in report.discovered_duals[:20]
        ],
        'discovered_triadic': [
            {
                'bit_i': t.bit_i,
                'bit_j': t.bit_j,
                'bit_r': t.bit_r,
                'p_r_ij': round(t.p_r_given_ij, 3),
                'p_r_i': round(t.p_r_given_i, 3),
                'p_r_j': round(t.p_r_given_j, 3),
                'strength': round(t.interaction_strength, 3),
                'support': t.support,
            }
            for t in report.discovered_triadic_deps[:30]
        ],
    }

    report_path = os.path.join(OUTPUT_DIR, 'd_a14_v2_autolabel_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"    Full report: {report_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Concepts analyzed:     {len(snapshot.codes)}")
    print(f"  Active bits:           {report.n_active_bits} / 63")
    print(f"  Dead bits:             {report.n_dead_bits}")
    print(f"  Discovered duals:      {len(report.discovered_duals)}")
    print(f"  Dependencies:          {len(report.discovered_deps)}")
    print(f"  Triadic interactions:  {len(report.discovered_triadic_deps)}")
    print(f"  Labels exported:       {out_path}")
    print(f"  Full report:           {report_path}")
    print()
    print("  Next: Human review → validate/reject labels → create anclas_v3.json")
    print("=" * 60)


if __name__ == '__main__':
    main()

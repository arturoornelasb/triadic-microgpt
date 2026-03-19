"""
Generate Gold-Standard Prime Signatures for Triadic Alignment

This script uses the deterministic Triadic-Neurosymbolic-Engine (neurosym) to generate
ground-truth prime factor signatures for a set of core vocabulary words.
The microgpt model will use these during contrastive training to align its native
triadic head with the true algebraic properties.
"""

import os
import sys
import json
import argparse
import sympy

# Add local Triadic Engine to path to avoid pip install issues
# Looks for Triadic-Neurosymbolic-Engine cloned alongside this repo
_ENGINE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            '..', 'Triadic-Neurosymbolic-Engine', 'src')
_ENGINE_PATH = os.environ.get('NEUROSYM_PATH', _ENGINE_PATH)
sys.path.insert(0, _ENGINE_PATH)

try:
    from neurosym import ContinuousEncoder, DiscreteMapper
except ImportError as e:
    print(f"ImportError: {e}")
    print("Warning: 'neurosym' or its dependencies are not installed in the environment.")
    ContinuousEncoder, DiscreteMapper = None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', type=str, default='data/core_concepts.txt', help='Input vocabulary list')
    parser.add_argument('--output', type=str, default='data/gold_primes.json', help='Output JSON mapping')
    parser.add_argument('--n_bits', type=int, default=48, help='Number of prime bits')
    parser.add_argument('--mode', type=str, choices=['pca', 'random', 'consensus', 'contrastive'], default='pca')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='Embedding model')
    args = parser.parse_args()

    if ContinuousEncoder is None:
        print("Cannot run without neurosym.")
        return

    # Load concepts
    try:
        with open(args.vocab, 'r', encoding='utf-8') as f:
            concepts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        # Fallback concepts for testing if file doesn't exist
        concepts = [
            "King", "Queen", "Man", "Woman", "Boy", "Girl", 
            "Castle", "House", "Dog", "Cat", "Sword", "Shield",
            "Apple", "Orange", "Fruit", "Water", "Fire", "Earth"
        ]
        print(f"Warning: {args.vocab} not found. Using a tiny fallback dictionary of {len(concepts)} words.")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Encoding {len(concepts)} concepts using {args.model}...")
    encoder = ContinuousEncoder(args.model)
    embeddings = encoder.encode(concepts)

    print(f"Projecting to {args.n_bits} prime bits using {args.mode} mode...")
    mapper = DiscreteMapper(n_bits=args.n_bits, projection=args.mode)
    
    # Generate the mapping
    prime_map = mapper.fit_transform(concepts, embeddings)
    
    # output_dict = {'Concept': {'prime_factor': int, 'binary_signature': list}}
    output_dict = {}

    print("\nSample Mappings:")
    for i, concept in enumerate(concepts):
        prime_val = prime_map[concept]
        bits = [1 if (prime_val % sympy.prime(j+1)) == 0 else 0 for j in range(args.n_bits)]
        
        if i < 10:
            print(f"  {concept:15s} -> {prime_val:30d} -> {bits[:5]}...")
            
        output_dict[concept] = {
            'prime_factor': prime_val,
            'binary_signature': bits
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2)

    print(f"Saved gold-standard prime signatures to {args.output}")

if __name__ == '__main__':
    main()

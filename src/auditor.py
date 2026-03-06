
import os
import sys
import torch
import math
import networkx as nx
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
from src.triadic import PrimeMapper
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

def build_semantic_graph(prime_map: Dict[str, int]):
    G = nx.Graph()
    concepts = list(prime_map.keys())
    G.add_nodes_from(concepts)
    
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            c1, c2 = concepts[i], concepts[j]
            if math.gcd(prime_map[c1], prime_map[c2]) > 1:
                G.add_edge(c1, c2)
    return G

def get_prime_map(model, concepts, tokenizer, mapper, device):
    model.eval()
    prime_map = {}
    with torch.no_grad():
        for concept in concepts:
            ids = torch.tensor([tokenizer.encode(concept)], device=device)
            _, triadic_proj, _ = model(ids)
            # Take mean over context (usually just 1-2 tokens for short words)
            proj = triadic_proj.mean(dim=1).squeeze().tolist()
            prime_map[concept] = mapper.map(proj)
    return prime_map

def run_audit(ckpt_a, ckpt_b, concept_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Auditing:\n  Model A: {ckpt_a}\n  Model B: {ckpt_b}")
    
    # Load Model A
    cp_a = torch.load(ckpt_a, map_location=device)
    config_a = TriadicGPTConfig(**cp_a['config'])
    model_a = TriadicGPT(config_a).to(device)
    model_a.load_state_dict(cp_a['model_state_dict'])
    
    # Load Model B
    cp_b = torch.load(ckpt_b, map_location=device)
    config_b = TriadicGPTConfig(**cp_b['config'])
    model_b = TriadicGPT(config_b).to(device)
    model_b.load_state_dict(cp_b['model_state_dict'])
    
    # Tokenizer
    tokenizer_path = os.path.join(os.path.dirname(ckpt_a), 'tokenizer.json')
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    mapper = PrimeMapper(config_a.n_triadic_bits)
    
    # 1. Encode
    print(f"Encoding {len(concept_list)} concepts...")
    map_a = get_prime_map(model_a, concept_list, tokenizer, mapper, device)
    map_b = get_prime_map(model_b, concept_list, tokenizer, mapper, device)
    
    # 2. Build Graphs
    print("Building semantic graphs...")
    graph_a = build_semantic_graph(map_a)
    graph_b = build_semantic_graph(map_b)
    
    # 3. Compare Topological Distances
    print("Calculating topological discrepancies...")
    paths_a = dict(nx.all_pairs_shortest_path_length(graph_a))
    paths_b = dict(nx.all_pairs_shortest_path_length(graph_b))
    
    discrepancies = []
    total_pairs = 0
    divergent_count = 0
    
    for i in range(len(concept_list)):
        for j in range(i + 1, len(concept_list)):
            total_pairs += 1
            w1, w2 = concept_list[i], concept_list[j]
            d_a = paths_a.get(w1, {}).get(w2, float('inf'))
            d_b = paths_b.get(w1, {}).get(w2, float('inf'))
            
            if d_a != d_b:
                divergent_count += 1
                discrepancies.append({
                    "Word 1": w1,
                    "Word 2": w2,
                    "Dist A": d_a if d_a != float('inf') else "INF",
                    "Dist B": d_b if d_b != float('inf') else "INF",
                    "Drift": abs(d_a - d_b) if (d_a != float('inf') and d_b != float('inf')) else "MAX"
                })
    
    print("\n" + "="*40)
    print("📊 TOPOLOGICAL AUDIT RESULTS")
    print("="*40)
    print(f"Total pairs: {total_pairs}")
    print(f"Divergent pairs: {divergent_count}")
    print(f"Drift Rate: {divergent_count/total_pairs:.1%}")
    
    if discrepancies:
        print("\nTop Drift Pairs (Sample):")
        print(f"{'Word 1':<12} | {'Word 2':<12} | {'Dist A':<6} | {'Dist B':<6} | {'Drift'}")
        print("-" * 55)
        for d in discrepancies[:20]:
            print(f"{d['Word 1']:<12} | {d['Word 2']:<12} | {str(d['Dist A']):<6} | {str(d['Dist B']):<6} | {d['Drift']}")
    
    return divergent_count / total_pairs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True)
    parser.add_argument("--ckpt-b", required=True)
    parser.add_argument("--concepts", default=None)
    args = parser.parse_args()
    
    if args.concepts and os.path.exists(args.concepts):
        with open(args.concepts, 'r') as f:
            concepts = [line.strip() for line in f if line.strip()]
    else:
        # Default interesting concept set
        concepts = [
            "king", "queen", "man", "woman", "prince", "princess", "royal",
            "dog", "cat", "animal", "pet", "wolf", "lion",
            "car", "bicycle", "truck", "engine", "wheel", "road",
            "house", "door", "window", "kitchen", "home",
            "happy", "sad", "angry", "love", "hate",
            "doctor", "nurse", "hospital", "medicine", "health",
            "sun", "moon", "star", "sky", "space"
        ]
        
    run_audit(args.ckpt_a, args.ckpt_b, concepts)

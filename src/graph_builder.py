
import os
import sys
import math
from collections import defaultdict
from typing import List, Dict, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ScalableGraphBuilder:
    """
    Optimized Graph Builder using an inverted prime factor index.
    Inspired by 'ScalableGraphBuilder' from the Triadic Neurosymbolic Engine.
    
    Rather than O(N^2) pairwise comparisons, this uses O(N*k) to find neighbors.
    """
    def __init__(self, mapper):
        self.mapper = mapper
        self.inverted_index = defaultdict(list)
        self.concept_to_prime = {}
        
    def add_concept(self, label: str, prime_val: int):
        """Add a concept and update the inverted index."""
        self.concept_to_prime[label] = prime_val
        
        # Extract individual prime factors (bits)
        explanation = self.mapper.explain(prime_val)
        for p in explanation['factors']:
            self.inverted_index[p].append(label)
            
    def find_neighbors(self, label: str) -> Set[str]:
        """
        Find all concepts that share at least one semantic feature with 'label'.
        Complexity: O(F * B) where F is factor count and B is avg bucket size.
        """
        if label not in self.concept_to_prime:
            return set()
            
        prime_val = self.concept_to_prime[label]
        explanation = self.mapper.explain(prime_val)
        
        neighbors = set()
        for p in explanation['factors']:
            neighbors.update(self.inverted_index[p])
            
        # Remove self
        if label in neighbors:
            neighbors.remove(label)
            
        return neighbors

    def get_shared_features(self, label1: str, label2: str) -> List[int]:
        """Get the specific prime factors shared between two concepts."""
        p1 = self.concept_to_prime.get(label1, 0)
        p2 = self.concept_to_prime.get(label2, 0)
        shared = math.gcd(p1, p2)
        return self.mapper.explain(shared)['factors'] if shared > 1 else []

if __name__ == "__main__":
    # Demo
    from src.triadic import PrimeMapper
    mapper = PrimeMapper(32)
    builder = ScalableGraphBuilder(mapper)
    
    # Mock some concepts
    concepts = {
        "king": 2*3*5,
        "queen": 2*7*5,
        "man": 3*11,
        "woman": 7*11,
        "dog": 13*17
    }
    
    for name, val in concepts.items():
        builder.add_concept(name, val)
        
    print(f"Neighbors of 'king': {builder.find_neighbors('king')}")
    print(f"Shared features (king, queen): {builder.get_shared_features('king', 'queen')}")

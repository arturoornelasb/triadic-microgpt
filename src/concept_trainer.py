"""
Semantic Alignment Trainer — Targeted fine-tuning for the Triadic Head.

This script frozen the main transformer weights and trains only the 
triadic_head on specific concept-pair relationships (subsumption/similarity)
using prime-factor supervision.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT
from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator

# ============================================================
# Concept Dataset
# ============================================================

CONCEPTS = {
    "king":   [2, 3, 5, 7],     # Person, Royal, Male, Power
    "queen":  [2, 3, 5, 11],    # Person, Royal, Female, Power
    "doctor": [2, 13, 17],      # Person, Medical, Science
    "hospital":[13, 17, 19],    # Medical, Science, Building
    "apple":  [23, 29, 31],     # Fruit, Food, Plant
    "fruit":  [23, 31],         # Fruit, Plant (Subsumes Apple)
    "man":    [2, 5],           # Person, Male
    "woman":  [2, 11],          # Person, Female
}

class AlignmentDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, concepts):
        self.tokenizer = tokenizer
        self.data = []
        for word, primes in concepts.items():
            ids = tokenizer.encode(word, add_special=False)
            # Create a target bitmask (0 or 1 for each of the 64 bits)
            target = torch.zeros(64)
            for p in primes:
                # Map prime to bit index (approximate for now)
                # In real scenario, we'd use PrimeMapper. wyjaśnij
                pass
            # For this simple alignment, we'll just use the first N primes
            # and map them to indices. 2 -> index 0, 3 -> index 1, etc.
            self.data.append((torch.tensor(ids), word, primes))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ============================================================
# Trainer
# ============================================================

def align(ckpt_path, lr=1e-3, steps=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model for alignment: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    config_dict = checkpoint['config']
    
    # Reconstruct config object
    from src.torch_transformer import TriadicGPTConfig
    config = TriadicGPTConfig(
        vocab_size=config_dict.get('vocab_size', 4096),
        block_size=config_dict.get('block_size', 256),
        n_layer=config_dict.get('n_layer', 6),
        n_embd=config_dict.get('n_embd', 256),
        n_head=config_dict.get('n_head', 8),
        n_triadic_bits=config_dict.get('n_triadic_bits', 32),
        dropout=config_dict.get('dropout', 0.1)
    )
    
    model = TriadicGPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer_path = os.path.join(os.path.dirname(ckpt_path), 'tokenizer.json')
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    # Freeze everything except triadic head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.triadic_head.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.Adam(model.triadic_head.parameters(), lr=lr)
    mapper = PrimeMapper(64)
    validator = TriadicValidator()

    print("Beginning semantic alignment...")
    model.train()
    
    for step in range(steps):
        total_loss = 0
        for word, primes in [("king", [2,3,5,7]), ("queen", [2,3,5,11]), ("apple", [23,29,31])]:
            # Simple word embedding forward
            ids = torch.tensor([tokenizer.encode(word)], device=device)
            _, projections, _ = model(ids)
            proj = projections[0, -1, :] # Last token projection
            
            # Target: bits corresponding to primes should be positive
            # Others should be negative. 
            target_bits = torch.full((64,), -1.0, device=device)
            for p in primes:
                idx = mapper.primes.index(p)
                target_bits[idx] = 1.0
            
            # MSE loss between tanh projections and target bits
            loss = F.mse_loss(proj, target_bits)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if step % 100 == 0:
            print(f"Step {step} | Loss: {total_loss:.4f}")

    # Save aligned head
    save_path = ckpt_path.replace(".pt", "_aligned.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': checkpoint['config'],
        'step': checkpoint['step'],
        'loss': checkpoint['loss'],
        'alignment_step': steps
    }, save_path)
    print(f"Aligned model saved: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    align(args.ckpt)

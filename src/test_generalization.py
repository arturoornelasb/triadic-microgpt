import os
import sys
import torch
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import load_model
from src.triadic import PrimeMapper, TriadicValidator

def test_generalization(ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, config = load_model(ckpt_path, "checkpoints/torch/tokenizer.json", device)
    
    mapper = PrimeMapper(config.n_triadic_bits)
    validator = TriadicValidator()

    gold_path = "data/gold_primes.json"
    seen_concepts = set()
    if os.path.exists(gold_path):
        with open(gold_path, 'r') as f:
            gold_data = json.load(f)
        seen_concepts = set(gold_data.keys())

    tests = [
        {"anchor": "King", "unseen": ["Duke", "Prince", "Emperor", "Baron", "Monarch"], "seen": ["Queen"]},
        {"anchor": "Dog", "unseen": ["Wolf", "Fox", "Puppy", "Hound", "Pet"], "seen": ["Cat"]},
        {"anchor": "House", "unseen": ["Mansion", "Cabin", "Hut", "Apartment", "Building"], "seen": ["Castle"]},
        {"anchor": "Water", "unseen": ["Ocean", "River", "Rain", "Drink", "Liquid"], "seen": ["Fire"]},
    ]

    print("\n" + "="*85)
    print("  STEP A: HELD-OUT SEMANTIC GENERALIZATION TEST (ZERO-SHOT)")
    print("="*85)

    def get_prime(word):
        ids = tokenizer.encode(word, add_special=False)
        if not ids:
            return 1, None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)
        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        return mapper.map(proj)

    for test in tests:
        anchor = test["anchor"]
        print(f"\n[ Anchor Category: {anchor} ] (Seen in Distillation? {'Yes' if anchor in seen_concepts else 'No'})")
        anchor_prime = get_prime(anchor)
        
        print(f"  {'Concept':<12s} | {'Seen':<5s} | {'Prime Value':<30s} | {'Sim%':<6s}")
        print("  " + "-"*65)
        
        all_words = test["seen"] + test["unseen"]
        for word in all_words:
            prime = get_prime(word)
            seen_str = "Yes" if word in seen_concepts else "No"
            
            sim = validator.similarity(prime, anchor_prime)
            
            print(f"  {word:<12s} | {seen_str:<5s} | {str(prime):<30s} | {sim:>6.1%}")

if __name__ == "__main__":
    ckpt = "checkpoints/torch/model_L12_D512_B64_step500.pt"
    if os.path.exists(ckpt):
        test_generalization(ckpt)
    else:
        print(f"Checkpoint not found: {ckpt}\nIf you used fine-tuning, you might want to point this to the chat checkpoint.")

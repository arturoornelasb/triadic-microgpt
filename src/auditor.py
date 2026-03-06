import os
import sys
import json
import random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import load_model
from src.triadic import PrimeMapper, TriadicValidator

def evaluate_relational_bias(ckpt_path, gold_path, num_pairs=5000):
    print("\n" + "="*85)
    print("  STEP C: QUANTITATIVE RELATIONAL BIAS AUDIT (EXP 8)")
    print("="*85)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, config = load_model(ckpt_path, "checkpoints/torch/tokenizer.json", device)
    
    mapper = PrimeMapper(config.n_triadic_bits)
    validator = TriadicValidator()

    if not os.path.exists(gold_path):
        print(f"Error: Could not find ground truth dataset at {gold_path}")
        return

    print(f"Loading Ground Truth from {gold_path}...")
    with open(gold_path, 'r') as f:
        gold_data = json.load(f)

    # Reconstruct true primes from binary signatures instead of loading from the json
    # Because the json has 'binary_signature', we need to map those bits to primes
    print("Reconstructing ground truth primes...")
    concepts = list(gold_data.keys())
    
    # We want a mix of True Positive and Negative pairs
    # Since random pairs have ~0% chance of subsumption, we'll find actual subsuming pairs
    # from the ground truth to use as Positives.
    
    true_primes = {}
    for concept in concepts:
        bits = gold_data[concept]['binary_signature']
        # Convert bits to bits array
        # Provide bits matching the n_bits format (1 for positive, 0 for 0)
        # But wait, True Prime conversion in PrimeMapper usually takes continuous vectors
        # If we have binary_signature (list of 0/1), we can create a mock projection:
        # 1 -> 1.0, 0 -> -1.0
        proj = np.array([1.0 if b == 1 else -1.0 for b in bits])
        true_prime = mapper.map(proj)
        true_primes[concept] = true_prime

    print(f"Loaded {len(true_primes)} ground truth primes.")
    
    # 1. Generate Positive Pairs (Ground Truth A % B == 0)
    print("Discovering ground truth logical relationships...")
    positive_pairs = []
    # To save time, just check a subset 
    subset_concepts = concepts[:2000]
    for i in range(len(subset_concepts)):
        for j in range(i+1, len(subset_concepts)):
            c1, c2 = subset_concepts[i], subset_concepts[j]
            p1, p2 = true_primes[c1], true_primes[c2]
            
            # Don't use pairs that are identical primes
            if p1 != p2:
                if p1 % p2 == 0:
                    positive_pairs.append((c1, c2))
                elif p2 % p1 == 0:
                    positive_pairs.append((c2, c1))
            
            if len(positive_pairs) >= num_pairs // 2:
                break
        if len(positive_pairs) >= num_pairs // 2:
            break

    # 2. Generate Negative Pairs (Ground Truth A % B != 0)
    negative_pairs = []
    while len(negative_pairs) < (num_pairs - len(positive_pairs)):
        c1, c2 = random.sample(concepts, 2)
        p1, p2 = true_primes[c1], true_primes[c2]
        if p1 % p2 != 0:
            negative_pairs.append((c1, c2))

    test_pairs = [(A, B, True) for A, B in positive_pairs] + [(A, B, False) for A, B in negative_pairs]
    random.shuffle(test_pairs)

    print(f"\nConstructed Bias Audit Dataset:")
    print(f"  Total Pairs: {len(test_pairs)}")
    print(f"  Positive Pair Ground Truth: {len(positive_pairs)}")
    print(f"  Negative Pair Ground Truth: {len(negative_pairs)}")
    print("\nAuditing Triadic-MicroGPT predictions...")

    def get_model_prime(word):
        ids = tokenizer.encode(word, add_special=False)
        if not ids:
            return 1
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)
        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        return mapper.map(proj)

    TP, FP, TN, FN = 0, 0, 0, 0
    
    for count, (c_A, c_B, gt_subsumes) in enumerate(test_pairs):
        pred_pA = get_model_prime(c_A)
        pred_pB = get_model_prime(c_B)
        
        pred_subsumes = (pred_pA % pred_pB == 0)
        
        if gt_subsumes and pred_subsumes:
            TP += 1
        elif not gt_subsumes and pred_subsumes:
            FP += 1
        elif not gt_subsumes and not pred_subsumes:
            TN += 1
        elif gt_subsumes and not pred_subsumes:
            FN += 1
            
        if (count + 1) % 500 == 0:
            print(f"  Processed {count + 1}/{len(test_pairs)} pairs...")

    print("\n" + "="*45)
    print("  EXPERIMENT 8: CONFUSION MATRIX")
    print("="*45)
    
    print(f"  True Positives (TP):  {TP:<5} | Expected Subsumption, Model agreed")
    print(f"  False Positives (FP): {FP:<5} | Vector Collision (Model hallucinated Subsumption)")
    print(f"  True Negatives (TN):  {TN:<5} | Correctly Isolated")
    print(f"  False Negatives (FN): {FN:<5} | Model broke semantic link")
    
    print("\n  Metrics:")
    precision = TP / max(1, (TP + FP))
    recall = TP / max(1, (TP + FN))
    fpr = FP / max(1, (FP + TN))
    accuracy = (TP + TN) / max(1, (TP + FP + TN + FN))
    
    print(f"  Subsumption Accuracy: {accuracy:>7.2%}")
    print(f"  Subsumption FPR:      {fpr:>7.2%} (Target: < 5%)")
    print(f"  Precision:            {precision:>7.2%}")
    print(f"  Recall:               {recall:>7.2%}")
    print("="*45)
    
    record_path = "reports/bias_audit_results.json"
    os.makedirs("reports", exist_ok=True)
    with open(record_path, "w") as f:
        json.dump({
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "accuracy": accuracy, "FPR": fpr, "precision": precision, "recall": recall
        }, f, indent=2)
    print(f"\nAudit results written to {record_path}")


if __name__ == "__main__":
    # We use the 64-bit XL model that we just trained
    ckpt = "checkpoints/torch/model_L12_D512_B64_step500.pt"
    gold = "data/gold_primes_64.json"
    if os.path.exists(ckpt):
        evaluate_relational_bias(ckpt, gold, num_pairs=2000)
    else:
        print(f"Checkpoint not found: {ckpt}")


import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chat import load_chat_model, generate_response

def evaluate_model(ckpt_path):
    model, tokenizer, mapper, validator, device = load_chat_model(ckpt_path)
    
    test_prompts = [
        "Once upon a time, there was a little",
        "The brave knight went to the",
        "Why is the sky blue?",
        "Tell me a story about a dragon and a",
        "Hello!"
    ]
    
    print("\n" + "="*60)
    print("🧪 AUTOMATED MODEL EVALUATION")
    print("="*60)
    
    results = []
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: '{prompt}'")
        try:
            response, h_prompt, h_resp = generate_response(
                model, tokenizer, prompt, device, max_tokens=50, temperature=0.7
            )
            
            # Triadic analysis
            with torch.no_grad():
                p_proj = torch.tanh(model.triadic_head(h_prompt)).squeeze().cpu().numpy()
                r_proj = torch.tanh(model.triadic_head(h_resp)).squeeze().cpu().numpy()
            
            prime_prompt = mapper.map(p_proj)
            prime_resp = mapper.map(r_proj)
            
            subsumes = validator.subsumes(prime_resp, prime_prompt)
            similarity = validator.similarity(prime_resp, prime_prompt)
            
            print(f"Response: {response}")
            print(f"Prime Prompt: {prime_prompt}")
            print(f"Prime Resp:   {prime_resp}")
            print(f"Alignment:    {similarity:.1%} (Subsumes: {subsumes})")
            
            results.append({
                "prompt": prompt,
                "response": response,
                "p_prime": prime_prompt,
                "r_prime": prime_resp,
                "similarity": similarity,
                "subsumes": subsumes
            })
            
        except Exception as e:
            print(f"Error testing prompt '{prompt}': {e}")
            
    # Check for failure modes
    print("\n" + "="*40)
    print("🔍 DIAGNOSTIC REPORT")
    print("="*40)
    
    # 1. Repetitive primes?
    primes = [r['r_prime'] for r in results]
    unique_primes = len(set(primes))
    if unique_primes == 1 and len(primes) > 1:
        print("🚩 BUG: Prime Collapse detected (all responses have same prime signature).")
    else:
        print(f"✅ Prime Diversity: {unique_primes}/{len(primes)} unique signatures.")
    
    # 2. Extremely large primes? (Potential overflow issues)
    max_prime = max(primes) if primes else 0
    if max_prime > 1e18:
        print(f"⚠️ WARNING: Large primes detected ({max_prime}). Might cause overflow in some environments.")
    else:
        print(f"✅ Prime Magnitudes: Max is {max_prime}, safe for 64-bit.")
        
    # 3. Coherence check
    avg_sim = np.mean([r['similarity'] for r in results])
    print(f"ℹ️ Average Semantic Similarity: {avg_sim:.1%}")
    if avg_sim < 0.1:
        print("⚠️ WARNING: Very low semantic alignment. Model might be generating gibberish or needs more training.")
    elif avg_sim > 0.9:
        print("⚠️ WARNING: Very high semantic similarity. Model might be parroting the prompt exactly.")

if __name__ == "__main__":
    ckpt = "checkpoints/torch_runXL/model_L12_D512_B64_best_pca_init.pt"
    if os.path.exists(ckpt):
        evaluate_model(ckpt)
    else:
        print(f"Checkpoint not found: {ckpt}")

"""
Chat Interface — Interactive terminal chat with Triadic Verification.

Phase 5: Load the fine-tuned conversational model and provide a UI
where the user can chat with the AI. After each response, it shows
a Triadic Verification panel explaining the prime factor relationship
between the user's prompt and the AI's answer.
"""

import os
import sys
import torch
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator

def load_chat_model(checkpoint_path):
    """Load the trained model and tokenizer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model to {device}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['config']
    
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
    model.eval()
    
    tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), 'tokenizer.json')
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    mapper = PrimeMapper(config.n_triadic_bits)
    validator = TriadicValidator()
    
    return model, tokenizer, mapper, validator, device

def generate_response(model, tokenizer, prompt, device, max_tokens=100, temperature=0.7):
    """Generate a response iteratively, stopping at <EOS>."""
    # Simple prompt formatting (no special chat template for now, just endoftext)
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    eos_id = tokenizer.special_tokens.get('<EOS>', tokenizer.special_tokens.get('<|endoftext|>', 0))
    generated = []
    
    # Capture prompt hidden state
    with torch.no_grad():
        # Get hidden state from ln_f for the prompt
        # We re-implement a partial forward pass here to capture internal states
        T = input_ids.shape[1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = model.wte(input_ids)
        pos_emb = model.wpe(pos)
        x = model.drop(tok_emb + pos_emb)
        for block in model.blocks:
            x = block(x)
        h_prompt_full = model.ln_f(x)
        h_prompt = h_prompt_full.mean(dim=1) # (1, n_embd)
    
    # Generation loop
    curr_ids = input_ids
    for _ in range(max_tokens):
        idx_cond = curr_ids[:, -model.config.block_size:]
        with torch.no_grad():
            logits, _, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        
        if next_id.item() == eos_id:
            break
            
        curr_ids = torch.cat([curr_ids, next_id], dim=1)
        generated.append(next_id.item())
        
    response_text = tokenizer.decode(generated, skip_special=True)
    
    # Get response hidden state
    if generated:
        resp_ids = torch.tensor([generated], device=device)
        with torch.no_grad():
            T_r = resp_ids.shape[1]
            pos_r = torch.arange(0, T_r, dtype=torch.long, device=device)
            t_emb = model.wte(resp_ids)
            p_emb = model.wpe(pos_r)
            xr = model.drop(t_emb + p_emb)
            for block in model.blocks:
                xr = block(xr)
            h_resp_full = model.ln_f(xr)
            h_resp = h_resp_full.mean(dim=1)
    else:
        h_resp = h_prompt
        
    return response_text.strip(), h_prompt, h_resp

def chat_loop(model, tokenizer, mapper, validator, device):
    """Main interactive chat loop."""
    print("\n" + "="*60)
    print("  TRIADIC AI XL — Explainable Neural Engine")
    print("="*60)
    print("  Type 'quit' or 'exit' to stop.")
    print("  Model will verify its grounding logic after each response.")
    print("-" * 60 + "\n")
    
    while True:
        try:
            prompt = input("\033[94mYou:\033[0m ")
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt.strip():
                continue
                
            print("\n\033[92mAI:\033[0m ", end="", flush=True)
            
            response, h_prompt, h_resp = generate_response(
                model, tokenizer, prompt, device, max_tokens=128, temperature=0.7
            )
            print(f"{response}\n")
            
            # Triadic Verification using the model's head
            with torch.no_grad():
                p_proj = torch.tanh(model.triadic_head(h_prompt)).squeeze().cpu().numpy()
                r_proj = torch.tanh(model.triadic_head(h_resp)).squeeze().cpu().numpy()
            
            prime_prompt = mapper.map(p_proj)
            prime_resp = mapper.map(r_proj)
            
            subsumes = validator.subsumes(prime_resp, prime_prompt)
            gap = validator.explain_gap(prime_resp, prime_prompt)
            similarity = validator.similarity(prime_resp, prime_prompt)
            
            # Display Verification Panel
            print("┌─ \033[93mTriadic Verification\033[0m ─────────────────────────┐")
            print(f"│ Question Φ = {prime_prompt:<20d}")
            print(f"│ Answer   Φ = {prime_resp:<20d}")
            
            sub_marker = "✅ Yes" if subsumes else "❌ No"
            print(f"│ Answer ⊇ Question (Logical Subsumption): {sub_marker:<10s} │")
            print(f"│ Semantic Affinity: {similarity:>7.1%}                   │")
            
            if gap['shared_factors']:
                shared = ", ".join(map(str, gap['shared_factors'][:5]))
                if len(gap['shared_factors']) > 5: shared += "..."
                print(f"│ Shared semantic primes: {shared:<25s} │")
                
            if not subsumes and gap['only_in_b_factors']:
                missing = ", ".join(map(str, gap['only_in_b_factors'][:5]))
                if len(gap['only_in_b_factors']) > 5: missing += "..."
                print(f"│ Missing features (Gap): {missing:<25s} │")
                
            print("└─────────────────────────────────────────────────┘\n")
            
        except EOFError:
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            traceback.print_exc()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to PyTorch .pt checkpoint')
    args = parser.parse_args()
    
    model, tokenizer, mapper, validator, device = load_chat_model(args.ckpt)
    chat_loop(model, tokenizer, mapper, validator, device)

"""
Chat Interface — Interactive terminal chat with Triadic Verification.

Phase 5: Load the fine-tuned conversational model and provide a UI
where the user can chat with the AI. After each response, it shows
a Triadic Verification panel explaining the prime factor relationship
between the user's prompt and the AI's answer.
"""

import os
import sys
import numpy as np
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fast_transformer import FastGPT, FastGPTConfig
from src.tensor_ops import softmax_forward
from src.tokenizer import BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator


def load_chat_model(checkpoint_path, tokenizer_path):
    """Load the trained model and tokenizer."""
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    # Configure model to match Phase 3/4 settings
    config = FastGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        n_layer=4,
        n_embd=128,
        n_head=4,
        n_triadic_bits=16,
    )
    
    print("Loading model weights...")
    model = FastGPT(config)
    try:
        model.load_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
        
    mapper = PrimeMapper(config.n_triadic_bits)
    validator = TriadicValidator()
    
    return model, tokenizer, mapper, validator, config


def generate_response(model, tokenizer, prompt, max_tokens=60, temperature=0.7):
    """Generate a response iteratively, stopping at <EOS>."""
    # Encode prompt in chat format
    input_ids = tokenizer.encode_chat(prompt)
    # Remove the trailing <EOS> so the model can generate the assistant response
    if input_ids[-1] == tokenizer.special_tokens['<EOS>']:
        input_ids = input_ids[:-1]
        
    eos_id = tokenizer.special_tokens['<EOS>']
    generated = []
    
    # We also want to capture the hidden state of the prompt
    # and the hidden state of the response for triadic verification
    prompt_hidden = None
    
    for step in range(max_tokens):
        # Truncate context to block_size
        ctx = input_ids[-model.config.block_size:]
        logits, hidden, _ = model.forward(ctx)
        
        # Save the prompt's hidden state (mean over prompt tokens)
        if step == 0:
            prompt_hidden = np.mean(hidden, axis=0)
            
        last_logits = logits[-1]
        
        # Temperature sampling
        scaled = last_logits / max(temperature, 1e-5)
        probs = softmax_forward(scaled.reshape(1, -1))[0]
        
        next_id = np.random.choice(len(probs), p=probs)
        
        if next_id == eos_id:
            break
            
        input_ids.append(next_id)
        generated.append(next_id)
        
    response_text = tokenizer.decode(generated, skip_special=True)
    
    # Get final hidden state for the response
    # Re-run forward with just the generated tokens to get clean response state
    if generated:
        _, resp_hidden_seq, _ = model.forward(generated)
        resp_hidden = np.mean(resp_hidden_seq, axis=0)
    else:
        resp_hidden = prompt_hidden
        
    return response_text.strip(), prompt_hidden, resp_hidden


def chat_loop(model, tokenizer, mapper, validator):
    """Main interactive chat loop."""
    print("\n" + "="*60)
    print("  TRIADIC AI — Explainable Conversational Model")
    print("="*60)
    print("  Type 'quit' or 'exit' to stop.")
    print("  The model will answer, then verify itself triadically.")
    print("-" * 60 + "\n")
    
    while True:
        try:
            prompt = input("\033[94mYou:\033[0m ")
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt.strip():
                continue
                
            print("\n\033[92mAI:\033[0m ", end="", flush=True)
            
            # Generate response
            response, h_prompt, h_resp = generate_response(
                model, tokenizer, prompt, max_tokens=80, temperature=0.7
            )
            print(f"{response}\n")
            
            # Triadic Verification
            # 1. Project to prime space
            p_proj = model.project_to_triadic_np(h_prompt)
            r_proj = model.project_to_triadic_np(h_resp)
            
            # 2. Map to composite primes
            prime_prompt = mapper.map(p_proj)
            prime_resp = mapper.map(r_proj)
            
            # 3. Algebraic verification
            subsumes = validator.subsumes(prime_resp, prime_prompt)
            gap = validator.explain_gap(prime_resp, prime_prompt)
            similarity = validator.similarity(prime_resp, prime_prompt)
            
            # 4. Display Verification Panel
            print("┌─ \033[93mTriadic Verification\033[0m ─────────────────────────┐")
            print(f"│ Question Φ = {prime_prompt:<20d}")
            print(f"│ Answer   Φ = {prime_resp:<20d}")
            
            sub_marker = "✅ Yes" if subsumes else "❌ No"
            print(f"│ Answer ⊇ Question: {sub_marker:<28s}")
            print(f"│ Similarity:        {similarity:>.1%}                        │")
            
            has_gap = bool(gap['only_in_a_factors'] or gap['only_in_b_factors'])
            if has_gap:
                shared = gap['shared_factors']
                extra = gap['only_in_a_factors']
                missing = gap['only_in_b_factors']
                print(f"│ Shared semantic factors: {str(shared)[:24]:<24s}")
                if missing:
                    print(f"│ Semantic gap (missed info): {str(missing)[:21]:<21s}")
                elif extra:
                    print(f"│ Added concepts (extra info): {str(extra)[:20]:<20s}")
            else:
                print("│ Exactly identical semantic space.                │")
                
            print("└─────────────────────────────────────────────────┘\n")
            
        except EOFError:
            print("\nExiting...")
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            traceback.print_exc()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Interactive Chat with Triadic Verification')
    parser.add_argument('--model', type=str, required=True, help='Path to fine-tuned .npz checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer.json')
    args = parser.parse_args()
    
    model, tokenizer, mapper, validator, config = load_chat_model(args.model, args.tokenizer)
    chat_loop(model, tokenizer, mapper, validator)

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fast_transformer import FastGPT, FastGPTConfig
from src.tensor_ops import softmax_forward
from src.tokenizer import BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator

def verify():
    checkpoint_path = 'checkpoints/model_best.npz'
    tokenizer_path = 'checkpoints/tokenizer.json'
    
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.load(tokenizer_path)
    
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
    model.load_checkpoint(checkpoint_path)
    
    mapper = PrimeMapper(config.n_triadic_bits)
    validator = TriadicValidator()
    
    print("\nGenerating Sample:")
    bos_id = tokenizer.special_tokens['<BOS>']
    input_ids = [bos_id]
    generated = []
    
    for _ in range(30):
        ctx = input_ids[-config.block_size:]
        logits, hidden, _ = model.forward(ctx)
        probs = softmax_forward(logits[-1].reshape(1, -1))[0]
        next_id = np.random.choice(len(probs), p=probs)
        if next_id == tokenizer.special_tokens['<EOS>']: break
        input_ids.append(next_id)
        generated.append(next_id)
        
    text = tokenizer.decode(generated, skip_special=True)
    print(f"  Generated: {text}")
    
    print("\nTriadic Analysis of 'King' vs 'Queen':")
    def get_prime(word):
        ids = tokenizer.encode(word, add_special=False)
        _, hidden, _ = model.forward(ids)
        # Mean hidden state over tokens
        h_mean = np.mean(hidden, axis=0)
        proj = model.project_to_triadic_np(h_mean)
        return mapper.map(proj)

    try:
        p_king = get_prime("King")
        p_queen = get_prime("Queen")
        sim = validator.similarity(p_king, p_queen)
        gap = validator.explain_gap(p_king, p_queen)
        
        print(f"  Φ(King)  = {p_king}")
        print(f"  Φ(Queen) = {p_queen}")
        print(f"  Similarity: {sim:.1%}")
        print(f"  Shared: {gap['shared_factors']}")
    except Exception as e:
        print(f"  Analysis error (maybe words not in vocab): {e}")

if __name__ == '__main__':
    verify()

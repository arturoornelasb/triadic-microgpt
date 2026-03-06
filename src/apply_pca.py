
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer
from src.torch_train import TextDataset

def apply_pca(ckpt_path, data_path, num_samples=2000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
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
    
    print(f"Loading data: {data_path}")
    # We only need a subset of tokens for PCA
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Read the first few MBs to get samples
        raw_text = f.read(1000000) 
    
    tokenizer_path = os.path.join(os.path.dirname(ckpt_path), 'tokenizer.json')
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    tokens = tokenizer.encode(raw_text)
    dataset = TextDataset(tokens, config.block_size)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print("Collecting hidden states for PCA...")
    hidden_states = []
    
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            # Forward pass up to ln_f
            # Replicating forward logic but stopping before heads
            B, T = x.shape
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
            tok_emb = model.wte(x)
            pos_emb = model.wpe(pos)
            h = model.drop(tok_emb + pos_emb)
            for block in model.blocks:
                h = block(h)
            h = model.ln_f(h) # (B, T, n_embd)
            
            # Use only a few tokens per sequence to avoid redundant sampling and memory overload
            # Middle token is usually well-contextualized
            hidden_states.append(h[:, T//2, :])
            
            if len(hidden_states) * 8 >= num_samples:
                break
                
    X = torch.cat(hidden_states, dim=0)
    print(f"Collected {X.shape[0]} hidden states.")
    
    # Apply PCA
    model.initialize_triadic_pca(X)
    
    # Save
    out_path = ckpt_path.replace('.pt', '_pca_init.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', 0),
        'pca_initialized': True
    }, out_path)
    
    print(f"PCA-initialized model saved to: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/TinyStories-train.txt")
    args = parser.parse_args()
    
    apply_pca(args.ckpt, args.data)

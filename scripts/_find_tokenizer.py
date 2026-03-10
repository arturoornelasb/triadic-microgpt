"""Prueba todos los tokenizers disponibles contra Run 15 para encontrar el compatible."""
import sys, os, math
sys.path.insert(0, '.')

import torch
from src.fast_tokenizer import FastBPETokenizer

MODEL_PATH = 'checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt'
DATA_PATH = 'data/TinyStories-train.txt'
BLOCK_SIZE = 256

# Todos los tokenizers disponibles
TOKENIZERS = [
    'checkpoints/torch/tokenizer.json',
    'checkpoints/dry_run_XL/tokenizer.json',
    'checkpoints/torch_runXL/tokenizer.json',
    'checkpoints/torch_run29_staged/tokenizer.json',
    'checkpoints/tokenizer.json',
    'checkpoints/torch_run15_strongalign/tokenizer.json',  # recién generado
]

print("Loading model...")
ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
from src.torch_transformer import TriadicGPT, TriadicGPTConfig
cfg = ckpt['config']
config = TriadicGPTConfig(**cfg)
model = TriadicGPT(config)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Sample text para evaluar
print("Loading sample text...")
raw = open(DATA_PATH, encoding='utf-8').read()
stories = raw.split('<|endoftext|>')
stories = [s.strip() for s in stories if s.strip() and len(s.strip()) > 30]
sample = ' '.join(stories[100:110])  # Use stories from middle (not shuffled set)

for tok_path in TOKENIZERS:
    if not os.path.exists(tok_path):
        print(f"  MISSING: {tok_path}")
        continue
    try:
        tok = FastBPETokenizer.load(tok_path)
        ids = tok.encode(sample)
        if len(ids) < 10:
            print(f"  {tok_path}: too few tokens ({len(ids)})")
            continue
        # Compute loss on first block
        ids_t = torch.tensor(ids[:BLOCK_SIZE+1], dtype=torch.long).unsqueeze(0)
        if ids_t.shape[1] < 2:
            continue
        x = ids_t[:, :-1]
        y = ids_t[:, 1:]
        with torch.no_grad():
            logits, _, _ = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
        ppl = math.exp(loss.item())
        print(f"  {tok_path}: loss={loss.item():.4f} PPL={ppl:.1f}  vocab={tok.vocab_size}")
    except Exception as e:
        print(f"  {tok_path}: ERROR — {e}")

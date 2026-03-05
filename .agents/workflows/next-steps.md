---
description: How to continue scaling and improving the Triadic MicroGPT model
---

# Scaling and Next Steps

## Current Best Model
- **Run 7**: 8L/384D, 16M params, loss 1.65, 50K TinyStories
- **Fine-tuned**: loss 0.78, basic chat capability

## Immediate Next Steps

### 1. Fix ByteLevel Decoding
The `FastBPETokenizer` uses ByteLevel encoding which adds `Ä` to decoded text. The decode method needs to properly handle byte-level decoding. Check `tokenizer.decode(ids, skip_special=True)`.

### 2. More Training Steps
The 16M model at loss 1.65 is still improving. Run 40K-50K steps to push below 1.0:
```powershell
python src/torch_train.py --tokenizer checkpoints/torch_run7/tokenizer.json `
  --stories 50000 --steps 50000 --layers 8 --dim 384 --heads 8 --bits 48 `
  --checkpoint-dir checkpoints/torch_run8
```

### 3. Scale to XL (if GPU memory allows)
```powershell
# ~45M params, estimated ~45 min on RTX 5060 Ti
python src/torch_train.py --stories 50000 --steps 30000 `
  --layers 12 --dim 512 --heads 8 --bits 64 `
  --checkpoint-dir checkpoints/torch_runXL
```

### 4. Improve Triadic Differentiation
The triadic head maps most single-word concepts to similar primes because it needs sentence context. Options:
- Train with concept-pair contrastive data
- Evaluate triadic on sentence-level, not word-level
- Add explicit concept vocabulary to training data

### 5. Production Chat Interface
Update `src/chat.py` to load PyTorch models and provide interactive REPL.

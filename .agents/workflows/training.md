---
description: How to train, evaluate, and fine-tune the Triadic MicroGPT model
---

# Triadic MicroGPT — Training Workflow

## Environment

```powershell
# Conda env with Python 3.10, PyTorch + CUDA, HuggingFace tokenizers
conda activate triadic-microgpt
```

**Key packages**: `torch` (2.12 nightly, CUDA 12.8), `tokenizers` (HuggingFace Rust), `numpy`, `matplotlib`

## Architecture Overview

```
Text → FastBPETokenizer → Token IDs → TriadicGPT → Two Heads:
  1. LM Head → next token prediction (cross-entropy)
  2. Triadic Head → tanh → bits → PrimeMapper → prime product (semantic fingerprint)
```

The model has **two losses** combined: `total = lang_loss + α × triadic_loss`

Triadic loss has 3 components (to prevent collapse):
- **Coherence**: adjacent tokens should agree (share primes)
- **Diversity**: bits should be ~50% active across batch
- **Contrastive**: different sequences should have different projections

## File Map

| File | Purpose |
|------|---------|
| `src/fast_tokenizer.py` | HuggingFace BPE tokenizer (Rust, 1000× faster) |
| `src/tokenizer.py` | Legacy Python BPE tokenizer (fallback) |
| `src/torch_transformer.py` | PyTorch model: `TriadicGPT` (nn.Module) |
| `src/torch_train.py` | GPU pretraining script |
| `src/torch_finetune.py` | GPU fine-tuning for chat |
| `src/evaluate.py` | Evaluation: perplexity, triadic, loss curves |
| `src/pre_tokenize.py` | Pre-encode corpus to `.npy` cache |
| `src/triadic.py` | PrimeMapper, TriadicValidator |
| `experiment_log.md` | All training runs with metrics |

## Step 1: Pre-train

// turbo
```powershell
# Full pipeline: train tokenizer + encode + GPU train (~15 min for 50K stories)
conda run -n triadic-microgpt python src/torch_train.py `
  --data data/TinyStories-train.txt `
  --stories 50000 `
  --vocab 4096 `
  --steps 20000 `
  --batch-size 32 `
  --lr 3e-4 `
  --layers 8 --dim 384 --heads 8 --bits 48 `
  --block 256 `
  --dropout 0.1 `
  --alpha 0.15 `
  --triadic-warmup-pct 0.3 `
  --print-every 200 --save-every 5000 `
  --checkpoint-dir checkpoints/torch_runN
```

**To skip tokenizer training** (saves minutes):
```powershell
--tokenizer checkpoints/torch_runN/tokenizer.json
```

**To skip encoding** (saves hours with old tokenizer):
```powershell
--tokens data/tokens_50k.npy
```

## Step 2: Evaluate

// turbo
```powershell
conda run -n triadic-microgpt python src/evaluate.py `
  --model checkpoints/torch_runN/model_best.pt `
  --tokenizer checkpoints/torch_runN/tokenizer.json `
  --data data/TinyStories-train.txt `
  --csv checkpoints/torch_runN/training_log.csv
```

This produces:
- **Perplexity** (lower = better, target < 5.0)
- **Sample generations** (qualitative check)
- **Triadic signature analysis** (concept pair similarity)
- **Loss curve graph** → `reports/loss_curve.png`
- **JSON report** → `reports/eval_report.json`

## Step 3: Document in experiment_log.md

After each run, add an entry to `experiment_log.md` with:
- Date, script, data, architecture, params
- Steps, final loss, perplexity, triadic loss
- Speed (stp/s), total time
- Sample generations
- Key observations and conclusion
- What to try next

## Step 4: Fine-tune for Chat

// turbo
```powershell
conda run -n triadic-microgpt python src/torch_finetune.py `
  --model checkpoints/torch_runN/model_best.pt `
  --tokenizer checkpoints/torch_runN/tokenizer.json `
  --data data/alpaca_data_cleaned.json `
  --steps 2000 `
  --batch-size 16 `
  --lr 5e-5 `
  --alpha 0.05 `
  --max-examples 2000 `
  --checkpoint-dir checkpoints/chat_runN
```

## Step 5: Run Tests

// turbo
```powershell
conda run -n triadic-microgpt python tests/test_all.py
```

All 37 tests should pass.

## Key Hyperparameters

| Param | Pretrain | Fine-tune | Notes |
|-------|----------|-----------|-------|
| `lr` | 3e-4 | 5e-5 | Lower for fine-tune to preserve pretrained weights |
| `alpha` | 0.15 | 0.05 | Triadic weight (too high → hurts language quality) |
| `triadic-warmup-pct` | 0.3 | 0 | Start triadic loss after 30% of pretrain steps |
| `dropout` | 0.1 | 0.1 | Regularization |
| `batch-size` | 32 | 16 | Smaller for fine-tune (less data) |

## Known Issues

1. **ByteLevel encoding**: FastBPETokenizer uses ByteLevel pre-tokenization which adds `Ä` prefix to tokens in decoded text. This is cosmetic.
2. **Triadic differentiation**: Single-word concepts may still map to similar primes — the head needs sentence-level context to differentiate.
3. **Tokenizer compatibility**: Old checkpoints (runs 1-6) use the Python BPE tokenizer (`src/tokenizer.py`). New checkpoints (run 7+) use HuggingFace tokenizer. They are NOT interchangeable.

## Scaling Guidelines

| Size | Layers | Dim | Heads | Params | GPU Time (20K steps) |
|------|--------|-----|-------|--------|---------------------|
| Small | 4 | 128 | 4 | ~1M | ~2 min |
| Medium | 6 | 256 | 8 | ~6M | ~4 min |
| Large | 8 | 384 | 8 | ~16M | ~15 min |
| XL | 12 | 512 | 8 | ~45M | ~45 min (est.) |

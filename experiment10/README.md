# Experiment 10: GPT-2 + Triadic Projection Head (Transfer)

## Hypothesis

TriadicGPT from-scratch achieves semantic gap +0.020 while Engine PCA achieves +0.136.
The gap is caused by embedding quality, not the triadic architecture.
If we add the triadic head to GPT-2 (pre-trained on 8M web pages), the richer
embeddings should produce much stronger triadic structure.

## Design

**Base model**: GPT-2 small (117M params, 12L/768D/12H, vocab 50257)
**Addition**: Triadic projection head W_tri in R^{64x768} (49K params)
**Training data**: TinyStories (same as from-scratch experiments)

### Two-Phase Training

| Phase | What trains | Params | LR | Steps |
|-------|-------------|--------|-----|-------|
| 1 (Frozen) | Triadic head only | 49K | 1e-3 | 5000 |
| 2 (Unfreeze) | Last 2 layers + ln_f + triadic head | ~14M | 3e-5 | 10000 |

Same triadic hyperparameters as Run 15: alpha=0.05, entropy=1.0, align=5.0.

## Setup

```bash
conda activate triadic-microgpt
pip install transformers
```

## Run

```bash
# Full two-phase training (~30 min on RTX 5060 Ti)
python experiment10/src/train.py

# Evaluate and compare with baselines
python experiment10/src/evaluate.py \
  --checkpoint experiment10/checkpoints/phase_2_(unfreeze_last_layers)_final.pt
```

## Expected Outcomes

| Scenario | Semantic Gap | Meaning |
|----------|-------------|---------|
| Success  | > +0.10 | Pre-trained embeddings work, architecture validated |
| Partial  | +0.03 to +0.10 | Improvement but gap not fully closed |
| Failure  | < +0.020 | Triadic loss formulation is the bottleneck |

## Files

```
experiment10/
  src/
    model.py       # GPT2TriadicModel (GPT-2 + triadic head wrapper)
    train.py       # Two-phase training script
    evaluate.py    # Benchmarks + comparison table
  results/         # Evaluation outputs
  checkpoints/     # Training checkpoints (gitignored)
```

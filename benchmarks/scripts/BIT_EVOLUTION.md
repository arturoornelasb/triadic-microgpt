# Bit Evolution Tracker

Tracks how triadic bit activations evolve across training checkpoints — a temporal analysis of semantic connection formation in the triadic head.

While standard benchmarks (`bit_entropy.py`, `scaling_study.py`) evaluate a single checkpoint, this benchmark produces a **longitudinal view**: when individual bits activate, when semantic connections form between concept pairs, and where phase transitions occur.

## What It Measures

### Per-Checkpoint Snapshot
- **Bit entropy** — per-bit and mean entropy across 87 concepts
- **Activation rate** — fraction of concepts activating each bit
- **Unique signatures** — number of distinct bit patterns
- **Pair similarities** — Jaccard similarity for 12 related + 12 unrelated concept pairs
- **Graph metrics** — density and cluster count of the semantic graph
- **Composites and bit patterns** — raw prime composites and binary vectors per concept

### Cross-Checkpoint Evolution
- **Bit births** — first step where a bit activates for a given concept
- **Bit deaths** — first step where a bit permanently deactivates
- **Connection formations** — first step where `GCD(a, b) > 1` for each concept pair
- **Semantic gap curve** — `mean(related_sim) - mean(unrelated_sim)` over time
- **Phase transitions** — steps where any metric jumps by more than 2 standard deviations

## Usage

```bash
# Basic run against a checkpoint directory
python benchmarks/scripts/bit_evolution.py \
  --checkpoint-dir checkpoints/torch_run15_strongalign/

# With figures and projection export
python benchmarks/scripts/bit_evolution.py \
  --checkpoint-dir checkpoints/torch_run15_strongalign/ \
  --version v1.0-run15 \
  --plot \
  --save-projections
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint-dir` | *required* | Directory containing `model_*_step*.pt` files |
| `--tokenizer` | auto-detected | Path to `tokenizer.json` (looks in checkpoint dir by default) |
| `--version` | `v1.0` | Version tag for the output filename |
| `--plot` | off | Generate PNG figures |
| `--save-projections` | off | Export bit patterns as `.npz` |

## Output

### JSON — `benchmarks/results/{version}_bit_evolution_{date}.json`

```json
{
  "benchmark": "bit_evolution",
  "steps": [5000, 10000, 15000, ...],
  "snapshots": [
    {
      "step": 5000,
      "mean_entropy": 0.42,
      "per_bit_entropy": [...],
      "per_bit_activation_rate": [...],
      "unique_signatures": 45,
      "graph_density": 0.85,
      "graph_n_clusters": 3,
      "pair_similarities": {"king|queen": 0.45, ...},
      "composites": {"king": 123456, ...},
      "bit_patterns": {"king": [1, 0, 1, ...], ...}
    }
  ],
  "evolution": {
    "entropy_curve": [...],
    "semantic_gap_curve": [...],
    "graph_density_curve": [...],
    "bit_births": [{"concept": "king", "bit": 7, "birth_step": 15000}],
    "bit_deaths": [{"concept": "fire", "bit": 3, "death_step": 40000}],
    "connection_formations": [{"a": "king", "b": "queen", "first_step": 10000}],
    "phase_transitions": [{"step": 25000, "metric": "semantic_gap", "delta": 0.08}]
  },
  "summary": { ... }
}
```

### Figures (with `--plot`)

| Figure | Description |
|--------|-------------|
| `bit_evolution_heatmap.png` | Steps (rows) x bits (columns), color = activation rate |
| `semantic_gap_evolution.png` | Temporal curve of the semantic gap with phase transition markers |
| `connection_timeline.png` | When each concept pair first shares a prime factor |

## Expected Results (Run 15, XL model)

Based on the 10 checkpoints at steps 5K–50K:

- **Entropy** should increase from ~0.3 to ~0.75 as the triadic head learns to use more bits
- **Semantic gap** should become positive around step 25K–30K, indicating the model has learned to assign higher similarity to related pairs than unrelated ones
- **Phase transitions** are expected near the midpoint of training, corresponding to the triadic warmup schedule completing

## Training Hook (Optional)

For future training runs, `src/evolution_hook.py` provides a lightweight snapshot function that can be called after each checkpoint save:

```python
from src.evolution_hook import save_triadic_snapshot

# After torch.save(...)
save_triadic_snapshot(model, tokenizer, config, checkpoint_dir, step, device)
```

This captures bit patterns and similarities for 12 representative concepts (~60ms overhead) and saves a JSON file alongside the checkpoint. The post-hoc `bit_evolution.py` script remains the primary analysis tool.

## Dependencies

Reuses existing infrastructure — no new dependencies:
- `src/evaluate.py` — `load_model()`
- `src/triadic.py` — `PrimeMapper`, `TriadicValidator`
- `src/graph_builder.py` — `ScalableGraphBuilder`
- `benchmarks/scripts/bit_entropy.py` — `CONCEPTS`, `compute_projections()`, `compute_bit_entropy()`
- `benchmarks/scripts/scaling_study.py` — `SEMANTIC_PAIRS`

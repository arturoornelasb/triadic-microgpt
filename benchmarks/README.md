# Benchmarks — Triadic MicroGPT

Industry-standard evaluation suite for measuring both language quality and triadic semantic quality.

## Directory Structure

```
benchmarks/
  scripts/          # Executable benchmark scripts
  results/          # JSON result files (versioned)
  figures/          # Generated plots and visualizations
  README.md         # This file
```

## Naming Convention

Results: `v{VERSION}_{BENCHMARK}_{YYYY-MM-DD}.json`
Figures: `{BENCHMARK}_{CHART_TYPE}.png`

## Quick Run

```bash
conda activate triadic-microgpt

# Run all benchmarks on a checkpoint
python benchmarks/scripts/run_all.py --model checkpoints/torch/model_best.pt

# Run individual benchmarks
python benchmarks/scripts/bit_entropy.py --model checkpoints/torch/model_best.pt
python benchmarks/scripts/taxonomic_consistency.py --model checkpoints/torch/model_best.pt
python benchmarks/scripts/language_quality.py --model checkpoints/torch/model_best.pt
```

## Benchmark Categories

| Category | Scripts | Measures |
|----------|---------|----------|
| **A. Language** | `language_quality.py` | Perplexity, MAUVE, Distinct-n, Repetition |
| **B. Triadic** | `bit_entropy.py`, `taxonomic_consistency.py`, `analogy.py`, `probe.py` | Entropy, diversity, subsumption, analogies |
| **C. Ablation** | `ablation_alpha.py`, `ablation_bits.py`, `ablation_scaling.py` | Pareto curves, scaling laws |
| **D. Comparison** | `compare_engine.py` | End-to-end vs post-hoc |

## Full Specification

See [.claude/benchmarks-spec.md](../.claude/benchmarks-spec.md) for detailed definitions, pass criteria, and methodology.

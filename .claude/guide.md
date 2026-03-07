# Agent Guide — Triadic MicroGPT

This document is the onboarding reference for any AI agent or human collaborator joining this project. Read CLAUDE.md (project root) first, then this guide.

## What This Project Does

We are building a **neurosymbolic language model** that integrates prime-factor semantic encoding directly into the transformer architecture. The model learns TWO things simultaneously:
1. Standard language modeling (predict next token)
2. Semantic prime signatures (map each concept to a composite prime integer)

The prime signatures enable algebraic verification: subsumption (A contains B?), composition (combine A+B), and gap analysis (what differs between A and B?) — operations impossible with cosine similarity.

## The Research Pipeline

```
1. Train model          →  src/torch_train.py
2. Evaluate             →  src/evaluate.py
3. Run bias audit       →  src/auditor.py
4. Run benchmarks       →  benchmarks/run_all.py (when created)
5. Document results     →  experiment_log.md
6. Save checkpoint      →  checkpoints/{run_name}/
7. Generate figures     →  reports/
```

## How to Run a Training Experiment

### Prerequisites
```bash
conda activate triadic-microgpt
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

### Standard Run
```bash
python src/torch_train.py \
  --scale xl \
  --steps 50000 \
  --stories 200000 \
  --checkpoint-dir checkpoints/torch_runN \
  --tokenizer checkpoints/torch/tokenizer.json
```

### After Training — ALWAYS Do These
1. Run `src/evaluate.py` and save the report
2. Run `src/auditor.py` if triadic changes were made
3. Add an entry to `experiment_log.md` with ALL metrics
4. Commit the results (not the checkpoint .pt files — they're in .gitignore)

## Key Files to Understand

| Priority | File | Why |
|----------|------|-----|
| 1 | `src/torch_transformer.py` | The model architecture — TriadicGPT class |
| 2 | `src/torch_train.py` | Training loop with dual losses and knowledge distillation |
| 3 | `src/triadic.py` | PrimeMapper and TriadicValidator — the math backbone |
| 4 | `src/evaluate.py` | How we measure quality |
| 5 | `src/auditor.py` | The relational bias audit (Experiment 8 from the paper) |
| 6 | `EVOLUTION_PLAN.md` | The research roadmap — what to work on and in what order |
| 7 | `experiment_log.md` | History of all runs — check before starting new work |

## Current Research Priority

**Phase 1: Solve Triadic Collapse** (see EVOLUTION_PLAN.md)

The triadic head maps ALL concepts to the same prime signature when given short inputs. This is the blocking problem. Any new work should focus on making the triadic projections actually differentiate between semantically distinct concepts.

## Experiment Documentation Protocol

Every training run MUST be documented in `experiment_log.md` with:

```markdown
## Run N: [Short Description]
| Key | Value |
|-----|-------|
| **Date** | YYYY-MM-DD |
| **Script** | `src/torch_train.py` |
| **Data** | description of training data |
| **Architecture** | NL / ND / NH / B bits |
| **Params** | X,XXX,XXX |
| **Steps** | N |
| **Final Loss** | X.XXXX |
| **Triadic Loss** | X.XXXX |
| **Time** | X min (GPU) |
| **Observations** | What changed, what improved, what failed |
```

## Benchmark Documentation Protocol

When running benchmarks, save results to `benchmarks/results/` as JSON:
```json
{
  "benchmark": "name",
  "version": "v2.1",
  "date": "YYYY-MM-DD",
  "model_checkpoint": "path/to/checkpoint.pt",
  "model_config": "12L/512D/8H/64bits",
  "metrics": { ... },
  "notes": "..."
}
```

## Don'ts
- DO NOT modify `src/autograd.py` or `src/transformer.py` — they are legacy educational code
- DO NOT train without documenting in experiment_log.md
- DO NOT use gold_primes for evaluation if the model was trained WITH distillation on them (circular evaluation)
- DO NOT push .pt checkpoint files to git (they are large; use Git LFS if needed)
- DO NOT change hyperparameter defaults without documenting the reason

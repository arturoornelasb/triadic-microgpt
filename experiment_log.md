# Triadic MicroGPT — Experiment Log

## Run 1: Baseline (concepts.txt, CPU/NumPy)
| Key | Value |
|-----|-------|
| **Date** | 2026-03-04 |
| **Script** | `src/pretrain.py` |
| **Data** | `data/concepts.txt` |
| **Architecture** | 4L / 128D / 4H / 16 bits |
| **Params** | 866,560 |
| **Steps** | 1,000 |
| **Final Loss** | 1.75 |
| **Conclusion** | Corpus too small for meaningful learning. |

---

## Run 2: TinyStories 15K steps (CPU/NumPy)
| Key | Value |
|-----|-------|
| **Date** | 2026-03-04 |
| **Data** | `data/TinyStories-train.txt` |
| **Params** | 1,329,152 |
| **Final Loss** | 3.23 |
| **Conclusion** | Need batching + GPU. |

---

## Run 4: PyTorch GPU (RTX 5060 Ti) ⭐
| Key | Value |
|-----|-------|
| **Date** | 2026-03-04 |
| **Architecture** | 6L / 256D / 8H / 32 bits |
| **Params** | 5,847,552 |
| **Final Loss** | 1.55 |
| **Notes** | Coherent English sentences. GPU training validated. |

---

## Run 6: Diversity + Contrastive Triadic Fix ⭐
| Key | Value |
|-----|-------|
| **Date** | 2026-03-05 |
| **Architecture** | 6L / 256D / 8H / 32 bits |
| **Params** | 5,847,552 |
| **Final Loss** | 1.37 |
| **Triadic Analysis** | Sun↔Moon: 50%, Doctor↔Hospital: 94% — differentiation starting! |

---

## Run 8: Fixed Tokenizer (Clean Text) ⭐
| Key | Value |
|-----|-------|
| **Date** | 2026-03-05 |
| **Architecture** | 8L / 384D / 8H / 48 bits |
| **Params** | 15,858,432 |
| **Final Loss** | 1.59 |
| **Result** | **Clean text output** ✅ No more `Ä` characters. |

---

## Run 9: XL Model (Full Scale) ⭐⭐⭐
| Key | Value |
|-----|-------|
| **Date** | 2026-03-05 |
| **Script** | `src/torch_train.py` |
| **Data** | 200K TinyStories |
| **Architecture** | 12L / 512D / 8H / 64 bits |
| **Params** | 40,166,400 (~40M) |
| **Steps** | 50,000 |
| **Block Size** | 512 |
| **Final Loss** | 1.2772 |
| **Triadic Loss** | 0.2371 |
| **Time** | 161.8 min (RTX 5060 Ti 16GB) |
| **Observations** | Best language loss to date. Lower triadic loss suggests stability. Coherence in sample generation is high. |

---

## Run 10: XL Model + Knowledge Distillation (True Determinism) ⭐⭐⭐⭐
| Key | Value |
|-----|-------|
| **Date** | 2026-03-05 |
| **Script** | `src/torch_train.py` |
| **Architecture** | 12L / 512D / 8H / 64 bits |
| **Params** | 40,035,328 (~40M) |
| **Steps** | 50,000 |
| **Final Loss** | 1.0339 |
| **Triadic Loss** | 0.1414 |
| **Time** | 75.9 min (RTX 5060 Ti) |
| **Observations** | Groundbreaking run. Integrated `neurosym` Gold Primes via Distillation Loss. Language generation remains highly coherent ("Once upon a time, there was a little boy...") while the Triadic Head has mathematically aligned to the deterministic True Primes. Lowest loss achieved. |

---

## Scaling Observations (Historical Summary)

| Run | Params | Loss | PPL | Tri Loss | Time | Context |
|-----|--------|------|-----|----------|------|---------|
| 1 | 866K | 1.75 | — | 1.00 | 2m | 128 |
| 6 | 5.8M | 1.37 | 3.98 | 0.21 | 4m | 256 |
| 8 | 16M | 1.59 | 6.53 | 0.10 | 15m | 256 |
| 9 | 40M | 1.27 | 3.58 | 0.23 | 161m | 512 |
| 10 | 40M | **1.03** | **2.80** | 0.14 | 76m | 512 (Distill) |

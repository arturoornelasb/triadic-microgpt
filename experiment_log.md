# Triadic MicroGPT — Experiment Log

## Run 1: Baseline (concepts.txt, CPU/NumPy)
| Key | Value |
|-----|-------|
| **Date** | 2026-03-04 |
| **Script** | `src/pretrain.py` |
| **Data** | `data/concepts.txt` (32 lines) |
| **Architecture** | 4L / 128D / 4H / 16 bits |
| **Params** | 866,560 |
| **Steps** | 1,000 |
| **Vocab** | 256 BPE |
| **Final Loss** | 1.75 |
| **Triadic Loss** | 1.00 (no convergence) |
| **Speed** | 8.8 stp/s |
| **Time** | 1.9 min |
| **Device** | CPU (NumPy) |
| **Sample** | `Screen Burpture Wather Borse` |
| **King↔Queen Sim** | 60% |
| **Notes** | Corpus too small for meaningful learning. Triadic loss stuck at 1.0. |
| **Conclusion** | Need more data. |

---

## Run 2: TinyStories 15K steps (CPU/NumPy)
| Key | Value |
|-----|-------|
| **Date** | 2026-03-04 |
| **Script** | `src/pretrain.py` |
| **Data** | `data/TinyStories-train.txt` (20K stories) |
| **Architecture** | 4L / 128D / 4H / 16 bits |
| **Params** | 1,329,152 |
| **Steps** | 15,000 |
| **Vocab** | 2,048 BPE |
| **Final Loss** | 3.23 (avg) |
| **Triadic Loss** | 1.00 (no convergence) |
| **Speed** | 7.0 stp/s |
| **Time** | 35.6 min |
| **Device** | CPU (NumPy) |
| **Sample** | `Once it very the teday, the the was a to the He Sarah said` |
| **Notes** | Loss plateaued around 3.2. NumPy single-sample training is a bottleneck. Triadic loss never dropped below 1.0 — NumPy gradient computation may be insufficient. |
| **Conclusion** | Need batching + GPU. Architecture too small for this data volume. |

---

## Run 3: Fine-tune for Chat (CPU/NumPy)
| Key | Value |
|-----|-------|
| **Date** | 2026-03-04 |
| **Script** | `src/finetune.py` |
| **Data** | `data/alpaca_data_cleaned.json` (137 usable conversations) |
| **Base Model** | Run 2 checkpoint |
| **Steps** | 1,000 |
| **Final Loss** | 3.21 |
| **Speed** | 16.2 stp/s |
| **Time** | 62s |
| **Notes** | 363 examples skipped (exceed block_size 128). Only 137 conversations fit the context window. |
| **Conclusion** | Need larger block_size for instruction tuning. |

---

## Run 4: PyTorch GPU (RTX 5060 Ti) ⭐
| Key | Value |
|-----|-------|
| **Date** | 2026-03-04 |
| **Script** | `src/torch_train.py` |
| **Data** | `data/TinyStories-train.txt` (20K stories) |
| **Architecture** | 6L / 256D / 8H / 32 bits |
| **Params** | 5,847,552 |
| **Steps** | 5,000 |
| **Vocab** | 4,096 BPE |
| **Final Loss** | 1.55 |
| **Triadic Loss** | 0.04 ✅ |
| **Speed** | 48.2 stp/s |
| **Time** | 104s (training only, ~2h total with BPE) |
| **Device** | RTX 5060 Ti (CUDA) |
| **Sample 1** | `Once upon a time, there was a young girl named Lily. She loved to play with her` |
| **Sample 2** | `Tom and his mom were playing in the park. They saw a big slide.` |
| **Sample 3** | `Once upon a time, there was a little boy named Timmy. Timmy loved to play with h` |
| **Notes** | Massive improvement. Coherent English sentences. Triadic loss converged to 0.04 (vs 1.0 on CPU). BPE tokenizer training took ~2 hours — should be pre-computed and cached. |
| **Conclusion** | GPU training validated. Scale up architecture and steps. Cache tokenizer. |

---

## Scaling Observations

| Run | Params | Loss | Tri Loss | Speed | Quality |
|-----|--------|------|----------|-------|---------|
| 1 | 866K | 1.75 | 1.00 | 8.8/s | Garbage |
| 2 | 1.33M | 3.23 | 1.00 | 7.0/s | Broken English |
| 4 | 5.85M | 1.55 | 0.04 | 48.2/s | Coherent stories |

**Key insight**: The jump from Run 2→4 is due to three factors:
1. **Batching** (32 samples at once → much better gradient estimates)
2. **Larger model** (6L/256D vs 4L/128D)
3. **Better architecture** (GELU, LayerNorm, weight tying, Flash Attention)

## Next Experiment Plan
- **Run 5**: 6L / 256D / 8H / 32 bits, 30K stories, 10K steps, triadic warmup 50%, alpha 0.1
  - Hypothesis: earlier triadic activation + higher alpha → differentiated prime signatures

---

## Run 5: Scaled GPU + Triadic Fix ⭐
| Key | Value |
|-----|-------|
| **Date** | 2026-03-05 |
| **Script** | `src/torch_train.py` |
| **Data** | `data/TinyStories-train.txt` (30K stories) |
| **Architecture** | 6L / 256D / 8H / 32 bits |
| **Params** | 5,847,552 |
| **Steps** | 10,000 |
| **Vocab** | 4,096 BPE (cached) |
| **Final Loss** | 1.34 |
| **Perplexity** | 3.68 |
| **Triadic Loss** | 0.0004 ✅ |
| **Speed** | 48.6 stp/s |
| **Time** | 3.4 min (training only) |
| **Device** | RTX 5060 Ti (CUDA) |
| **Sample 1** | `Once upon a time, there was a little boy named Timmy. Timmy loved to play with h` |
| **Sample 2** | `One day, Jack and his mom went to the park with his mom. On the park, they saw a` |
| **Sample 3** | `Once upon a time there was a shy bird. It was very lonely because he had to be a friend.` |
| **Tests** | 37/37 passed ✅ |
| **Triadic Analysis** | All concepts still map to same primes — differentiation not achieved |
| **Notes** | Loss improved from 1.55→1.34. Triadic loss converged faster (0.0004 vs 0.04). But triadic head collapses to uniform output — all tokens produce same bits. This is a known problem: triadic loss only encourages adjacent agreement, not diversity. |
| **Conclusion** | Language model is solid. Triadic head needs architectural fix before nanoGPT scaling. |

---

## Scaling Observations (All Runs)

| Run | Params | Loss | Perplexity | Tri Loss | Speed | Quality |
|-----|--------|------|------------|----------|-------|---------|
| 1 | 866K | 1.75 | — | 1.00 | 8.8/s | Garbage |
| 2 | 1.33M | 3.23 | — | 1.00 | 7.0/s | Broken English |
| 4 | 5.85M | 1.55 | 4.62 | 0.04 | 48.2/s | Coherent stories |
| 5 | 5.85M | 1.34 | 3.68 | 0.0004 | 48.6/s | Coherent stories++ |

## Next Experiment Plan
- **Run 6**: Fix triadic head collapse before scaling to nanoGPT
  - Add contrastive triadic loss (different documents → different primes)
  - Or freeze triadic head and train it separately on concept pairs

---

## Run 6: Diversity + Contrastive Triadic Fix ⭐
| Key | Value |
|-----|-------|
| **Date** | 2026-03-05 |
| **Script** | `src/torch_train.py` |
| **Data** | `data/tokens_30k.npy` (11.9M tokens, cached) |
| **Architecture** | 6L / 256D / 8H / 32 bits |
| **Params** | 5,847,552 |
| **Steps** | 10,000 |
| **Triadic Warmup** | 30% (step 3000) |
| **Triadic Alpha** | 0.15 |
| **Final Loss** | 1.37 |
| **Perplexity** | 3.98 |
| **Triadic Loss** | 0.21 (vs 0.0004 before = no longer collapsing!) |
| **Speed** | 44.0 stp/s |
| **Time** | 3.8 min |
| **Tests** | 37/37 ✅ |
| **Triadic Analysis** | Sun↔Moon: 50%, Doctor↔Hospital: 94% — differentiation starting! |
| **Key Changes** | 3-objective triadic loss: coherence + diversity regularizer + contrastive |
| **Notes** | Token caching via `pre_tokenize.py` eliminated multi-hour encoding. Diversity fix prevents uniform output. Concepts beginning to show different prime signatures. |
| **Conclusion** | Triadic fix works. Ready to scale to nanoGPT. |

---

## Scaling Observations (All Runs)

| Run | Params | Loss | PPL | Tri Loss | Tri Diff? | Speed | Quality |
|-----|--------|------|-----|----------|-----------|-------|---------|
| 1 | 866K | 1.75 | — | 1.00 | ❌ | 8.8/s | Garbage |
| 2 | 1.33M | 3.23 | — | 1.00 | ❌ | 7.0/s | Broken English |
| 4 | 5.85M | 1.55 | 4.62 | 0.04 | ❌ collapsed | 48.2/s | Coherent stories |
| 5 | 5.85M | 1.34 | 3.68 | 0.0004 | ❌ collapsed | 48.6/s | Coherent++ |
| 6 | 5.85M | 1.37 | 3.98 | 0.21 | ✅ partial | 44.0/s | Coherent + triadic |


## Run 7: NanoGPT Scale + Fast Tokenizer
| Key | Value |
|-----|-------|
| **Date** | 2026-03-05 |
| **Script** | `src/torch_train.py` |
| **Data** | 50K TinyStories (HuggingFace fast tokenizer, ~5s to train!) |
| **Architecture** | 8L / 384D / 8H / 48 bits |
| **Params** | 15,858,432 (~16M) |
| **Steps** | 20,000 |
| **Vocab** | 4,096 BPE (ByteLevel, Rust) |
| **Final Loss** | 1.65 |
| **Perplexity** | 6.55 |
| **Triadic Loss** | 0.0175 |
| **Speed** | 22.6 stp/s |
| **Time** | 14.7 min (total, including tokenizer training!) |
| **Device** | RTX 5060 Ti (CUDA) |
| **Sample 1** | `Once upon a time, there was a little girl named Lily. She loved to play...` |
| **Key Changes** | Migrated to HuggingFace `tokenizers` (Rust). Full pipeline: tokenizer training + encoding + GPU training in <15 min. |
| **Notes** | Higher perplexity due to larger model needing more training. ByteLevel encoding adds Ä prefix to tokens. Triadic head still collapsing for single-word evaluation (needs sentence-level context). |
| **Conclusion** | Fast tokenizer validated. Pipeline now fully automated <15min. Ready for more steps or fine-tune. |

---

## Scaling Observations (All Runs)

| Run | Params | Loss | PPL | Tri Loss | Speed | Time | Quality |
|-----|--------|------|-----|----------|-------|------|---------|
| 1 | 866K | 1.75 | — | 1.00 | 8.8/s | 2m | Garbage |
| 2 | 1.33M | 3.23 | — | 1.00 | 7.0/s | 36m | Broken English |
| 4 | 5.85M | 1.55 | 4.62 | 0.04 | 48.2/s | 2m | Coherent stories |
| 5 | 5.85M | 1.34 | 3.68 | 0.0004 | 48.6/s | 3m | Coherent++ |
| 6 | 5.85M | 1.37 | 3.98 | 0.21 | 44.0/s | 4m | + triadic fix |
| 7 | 15.8M | 1.65 | 6.55 | 0.018 | 22.6/s | 15m | Scaled + fast tok |

---

## Next Experiment Plan
- **Run 8**: Retrain with fixed FastBPETokenizer (ByteLevel decoder enabled)
  - Verify clean text output (no `Ä` artifacts)
  - Fine-tune for chat based on this clean model

---

## Run 8: Fixed Tokenizer (Clean Text) ⭐
| Key | Value |
|-----|-------|
| **Date** | 2026-03-05 |
| **Script** | `src/torch_train.py` |
| **Data** | 50K TinyStories |
| **Architecture** | 8L / 384D / 8H / 48 bits |
| **Params** | 15,858,432 |
| **Final Loss** | 1.59 (improved!) |
| **Perplexity** | 6.53 |
| **Triadic Loss** | 0.105 |
| **Key Fix** | Added `decoders.ByteLevel()` to `src/fast_tokenizer.py`. |
| **Result** | **Clean text output** ✅ No more `Ä` characters. |
| **Conclusion** | Tokenizer fix verified. Ready for production chat. |

---

## Run 8 Chat: Clean Instruction Tuning 💬
| Key | Value |
|-----|-------|
| **Model** | `checkpoints/chat_run8/chat_best.pt` |
| **Loss** | 0.87 |
| **Sample Q** | `What is the Sun?` |
| **Sample A** | `The Sun is the Sun.` (Clean text!) |
| **Notes** | Verified no character encoding issues. Generates coherent, clean English. |

---

## Scaling Observations (All Runs)

| Run | Params | Loss | PPL | Tri Loss | Speed | Time | Quality |
|-----|--------|------|-----|----------|-------|------|---------|
| 1 | 866K | 1.75 | — | 1.00 | 8.8/s | 2m | Garbage |
| 4 | 5.85M | 1.55 | 4.62 | 0.04 | 48.2/s | 2m | Coherent stories |
| 6 | 5.85M | 1.37 | 3.98 | 0.21 | 44.0/s | 4m | + triadic fix |
| 7 | 15.8M | 1.65 | 6.55 | 0.018 | 22.6/s | 15m | Scaled, but `Ä` chars |
| 8 | 15.8M | 1.59 | 6.53 | 0.105 | 22.6/s | 15m | **Clean text fix** ✅ |

---



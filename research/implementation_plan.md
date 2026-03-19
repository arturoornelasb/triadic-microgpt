# Implementation Plan — Final Model for Paper

> Date: 2026-03-19 | Status: PROPOSAL
> Goal: ONE paper, ONE model, ONE framework

---

## Phase 0: Consolidate What Exists (1-2 days) — NOW

No new code. Validate and document.

| Task | Files | Test | Done? |
|------|-------|------|-------|
| v2 training complete | `checkpoints/danza_63bit_xl_v2/` | 93% test, 98.3% sub | YES |
| reptimeline on v2 | `playground/audit_tests/analyze_v2.py` | 68 triadic, 48 active | YES |
| BitwiseValidator | `src/triadic.py`, `benchmarks/scripts/prime_vs_bitwise.py` | 1000/1000 equiv | YES |
| L1 bridge test | `playground/audit_tests/test_pf_bridge.py` | 3/4 PASS | YES |
| L3 blind primes | `playground/audit_tests/test_blind_primes.py` | PASS | YES |
| L2 D-A13 eval | `playground/audit_tests/eval_da13.py` | **PENDING — needs GPU** | NO |
| L11/L12 re-run with v2 | modify `test_indifference.py` | **PENDING** | NO |
| 7 book corrections | L4-L10 in book .tex | **PENDING** | NO |

**Exit criteria:** All Ls executed or documented why not. Book corrections listed.

---

## Phase 1: Bitwise as Default Backend (1 day)

Change validation layer only. Model and training unchanged.

| Task | File to modify | Change | Test |
|------|---------------|--------|------|
| 1.1 | `src/torch_train.py` | Use BitwiseValidator for eval subsumption/analogy | Results identical to prime |
| 1.2 | `src/evaluate.py` | Use BitwiseValidator for all algebraic checks | Results identical |
| 1.3 | `benchmarks/scripts/*.py` | Add `--backend bitwise` flag (default) | Benchmarks still pass |
| 1.4 | `ui/model_interface.py` | Use BitwiseValidator in desktop UI | UI functional |
| 1.5 | `tests/test_all.py` | Add tests: both backends produce same results | 37+ tests pass |

**What does NOT change:**
- `triadic_loss()` — differentiable, doesn't use primes or bits
- Model architecture — same TriadicGPT
- Training loop — same dual loss
- PrimeMapper stays in codebase for paper formalization

**Exit criteria:** `python benchmarks/scripts/prime_vs_bitwise.py` passes, all benchmarks produce identical results with bitwise backend.

---

## Phase 2: Discovery Loop Integration (3-5 days)

Close the loop: train -> discover -> re-anchor -> train.

| Task | Description | Priority |
|------|-------------|----------|
| 2.1 | Hook reptimeline discover at end of each bootstrap cycle | HIGH |
| 2.2 | Auto-generate new anchors from bit semantics | HIGH |
| 2.3 | Dead bit detection -> adaptive entropy regularization | MEDIUM |
| 2.4 | Use 3-way deps as composition loss signal | MEDIUM |
| 2.5 | AutoLabel -> LLM-generated names -> validate | LOW |

### 2.1 Discovery Hook

```python
# At end of each cycle in danza_bootstrap.py:
from reptimeline.discovery import BitDiscovery
from reptimeline.core import ConceptSnapshot

# After training, before next cycle:
codes = extract_all_projections(model, tokenizer, all_words)
snapshot = ConceptSnapshot(step=current_step, codes=codes)
discovery = BitDiscovery()
report = discovery.discover(snapshot)

# Use report to inform next cycle:
new_anchors = generate_anchors_from_discovery(report)
```

### 2.2 Anchor Generation from Discovery

When discovery finds:
- **Dual pair (bit_i <-> bit_j):** Create anchor pairs that activate one but not the other
- **3-way interaction (bit_i + bit_j -> bit_r):** Create composite anchors
- **Dead bit:** Increase entropy reg for that bit

### 2.4 Composition Loss from 3-Way Deps

```python
# If discovery found: bit_4 + bit_25 -> bit_33
# Then for concepts where bits 4 and 25 are ON, bit 33 should also be ON
# This becomes an additional loss term:
comp_loss = F.binary_cross_entropy(proj[:, 33], target_from_deps)
```

**Already works manually:**
- Bootstrap cycle: train -> eval -> re-anchor (manual)
- reptimeline: bit semantics + 3-way deps
- hybrid_adversarial: free bits learn unsupervised

**What's missing:** Closing the loop automatically.

**Exit criteria:** Script `train_unified.py` runs N cycles of train+discover automatically. Metrics improve or stay stable across cycles.

---

## Phase 3: iFSQ Activation (2-3 days)

Replace tanh with iFSQ as default activation.

| Task | Description | Test |
|------|-------------|------|
| 3.1 | Add iFSQ option to `torch_transformer.py` | Model builds |
| 3.2 | Train v2 equivalent with iFSQ | Compare: loss, accuracy, subsumption |
| 3.3 | Run reptimeline on iFSQ model | Compare triadic interactions |
| 3.4 | Measure sparsity distribution | Compare with ~42% prediction |

**Evidence:** D-A10 (iFSQ binary) achieved best LM loss (0.924) AND 87.1% subsumption. But it was trained with different anchors (54 vs 158). Need apples-to-apples comparison.

**Exit criteria:** iFSQ model with v2 anchors achieves >= 93% test AND <= 0.95 LM loss.

---

## Phase 4: Scale to 355M (1-2 weeks, GPU-dependent)

| Task | Description | Dependency |
|------|-------------|------------|
| 4.1 | Train 355M with bitwise backend | Phase 1 |
| 4.2 | Train 355M with discovery loop | Phase 2 |
| 4.3 | Run L2 (D-A13 formal eval) | GPU |
| 4.4 | Cross-corpus: WikiText2, LAMBADA | Phase 4.1 |
| 4.5 | Probe for categorical structure | Phase 4.2 |

**Open question:** At 355M, D-A13 showed zeros collapse to 0% (binary, not ternary). Does the ternary structure only hold at smaller scales? If so, document as scale-dependent finding.

**Exit criteria:** 355M model with subsumption >= 100% (matching D-A13) + discovery loop results.

---

## Tests Required Per Phase

### Phase 0 Tests
- [ ] L2: D-A13 eval (GPU)
- [ ] L11/L12: re-run with v2 checkpoint
- [ ] Verify all 80 unit tests pass

### Phase 1 Tests
- [ ] prime_vs_bitwise.py: 1000/1000 equivalence
- [ ] All 12 benchmarks produce identical results with bitwise backend
- [ ] UI functional with BitwiseValidator

### Phase 2 Tests
- [ ] Discovery loop runs N cycles without crash
- [ ] Metrics don't degrade across cycles
- [ ] Auto-generated anchors are semantically valid (manual inspection)
- [ ] Composition loss from 3-way deps improves subsumption

### Phase 3 Tests
- [ ] iFSQ model achieves >= 93% test accuracy
- [ ] iFSQ model has LM loss <= 0.95
- [ ] Sparsity distribution measured and compared to ~42% prediction
- [ ] reptimeline discovery on iFSQ model

### Phase 4 Tests
- [ ] L2 formal eval passes
- [ ] Cross-corpus generalization (WikiText2, LAMBADA)
- [ ] Categorical structure probing results
- [ ] Scale-dependent ternary analysis

---

## Paper Structure Mapping

| Paper Section | Phase | Evidence Source |
|--------------|-------|---------------|
| 3. Triadic Algebra | 0 | `src/triadic.py` (PrimeMapper + BitwiseValidator) |
| 3.3 Isomorphism proof | 1 | `benchmarks/scripts/prime_vs_bitwise.py` |
| 4. TriadicGPT Architecture | 0 | `src/torch_transformer.py` |
| 5. Discovery Loop | 2 | `reptimeline/`, `playground/audit_tests/analyze_v2.py` |
| 6.1 Scaling study | 0 | Runs 19-21, D-A13 |
| 6.2 Bits sweep | 0 | Runs 22-26 |
| 6.3 Ablation | 0 | Run 18, D-A15 (fail) |
| 6.4 Ternary convergence | 3 | D-A8/D-A10, convergence doc |
| 6.5 BitwiseValidator | 1 | prime_vs_bitwise benchmark |
| 6.6 Subsumption/analogy | 0 | D-A14 v2 results |
| 6.7 Domain separation | 0 | Experiment 11 |
| 6.8 NSM overlap | 0 | NSM mapping doc |
| 7. Discussion | all | Convergence analysis |

---

## What NOT to Do

- Do NOT move files (breaks imports)
- Do NOT create second paper (one topic)
- Do NOT publish triadic-head to PyPI yet
- Do NOT implement bitwise tokenizer (separate research)
- Do NOT over-engineer discovery loop (manual works, auto is bonus)
- Do NOT attempt ternary weights at 355M (evidence says they collapse)

---

## Timeline

```
Phase 0: NOW (1-2 days)     <- current
Phase 1: +2 days
Phase 2: +5 days
Phase 3: +7 days
Phase 4: +2-3 weeks (GPU)

Minimum for paper: Phase 0 + Phase 1 = 3 days
Full implementation: Phase 0-3 = ~10 days
With scaling: Phase 0-4 = ~3 weeks
```

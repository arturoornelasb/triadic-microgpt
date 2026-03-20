---
description: Current priorities and next steps for Triadic MicroGPT
---

# Next Steps (2026-03-19)

## Immediate (Blocked on D-A17 Training)

1. **D-A17 eval `--v2`** — run formal evaluation when training completes
   ```powershell
   conda run -n triadic-microgpt python playground/audit_tests/test_d_a13_eval.py --v2
   ```

2. **Update paper sections 5.4 and 6** with D-A17 scaling results (honest numbers)

3. **Train D-A18 (unified)** — iFSQ + hybrid 30+33 bits + v2 anchors + adversarial
   ```powershell
   conda run -n triadic-microgpt python playground/unified_final.py --scale xl --steps 50000 --dtype bfloat16
   ```

## Discovery Loop — Next Cycle

4. **Run reptimeline analysis** on D-A14 v2 (best model) to discover new bit semantics
5. **Human review** of discoveries → validate or reject
6. **Create anclas_v3.json** with expanded anchors (158 → 300+?)
7. **Retrain with v3 anchors** → measure improvement over 93%

## Paper Completion

8. **Finalize paper** with complete evidence chain:
   - From-scratch: Run 15 (PPL 7.69, gap +0.020)
   - Supervised 50 anchors: D-A5 (87%)
   - Supervised 158 anchors: D-A14 (93%, 98.3% sub)
   - Scaling 355M: D-A17 (TBD)
   - Unified best-of-all: D-A18 (TBD)
   - Discovery loop evidence: 50→158 = 87%→93%

## What's Done (No Action Needed)

- All 80 unit tests pass
- 12 benchmarks complete
- Validation E1-E7 complete
- BitwiseValidator migration (critical paths)
- torch_finetune.py bug fixes
- test_pf_bridge.py rewrite
- auditor.py / test_generalization.py tokenizer fix
- English translations
- EXPERIMENT_REFERENCE.md consolidated

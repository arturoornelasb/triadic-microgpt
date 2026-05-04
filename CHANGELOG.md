# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] - 2026-05-03

### triadic-head package: bumped to v0.1.1
- README BibTeX title corrected: "Emergent Algebraic Semantics" → "Learned Algebraic Encoding"
- README "InfoNCE closes 72% of gap" → "48% of gap" (audit correction)
- Added Zenodo DOI badge + `doi`/`publisher` fields in BibTeX entry
- Description: "zero language cost" → "negligible language cost (+1.7% PPL)"
- No source code changes (algebra.py, wrapper.py unchanged)

### Fixed (paper audit, commit dba2d54)
- **SimLex cosine claim corrected**: Run 15 cosine ρ = 0.046 (p = 0.144, NS), not 0.083/0.009 — that value belonged to sparse-v1 fine-tuned variant. Reframed paper as *algebraic encoding without semantic grounding* across abstract, discussion, conclusion.
- **`tab:ternary` D-A14 v2 lang loss**: 0.946 → 0.974 (training_log.csv)
- **`tab:ternary` D-A17 lang loss**: 2.73 → 3.10 (was copy-paste from D-A13)
- **`tab:ternary` "D-A5 (baseline)"**: → "Run 15 (baseline)" (correct attribution)
- **`tab:transfer` InfoNCE Unique Sigs**: 100% → 99.1% (110/111)
- **Sparse fine-tuning bit counts**: sparse-v4 20.5 → 16.4, sparse-v5 20.3 → 15.9
- **D-A13 holdout bit accuracy**: 89.4% → 88.0% (`f4_4_d_a13_eval.json`: 0.8797)
- **`tab:bits` k=64 probe**: 10.1% → 8.3% to match other rows of the sweep
- **README "Language cost: Zero"** → "Negligible (+1.7% PPL)" — consistent with paper

### Added
- **Paper figures**: `bits_sweep_loss.png` (visualizes U-shape claim) and `bits_sweep_panel.png` (4-metric comprehensive view) in Section 5.3
- **Methodological footnotes** on `tab:scaling` and `tab:pareto` clarifying the dual entropy methods (hard binarization vs continuous probabilities)
- **Reptimeline P2 duality companion DOI** in `.zenodo.json` (10.5281/zenodo.19375167)
- **12 inline `Table~\ref{tab:X}`** for navigation (was: only 3 of 17 tables referenced)

### Changed
- **Paper title alignment**: ZENODO_METADATA.md, .zenodo.json, CITATION.cff now match paper's "Learned Algebraic Encoding from Joint Training" (was "Emergent Algebraic Semantics")
- **Five principal findings** in Zenodo description (was eight) — matches paper post-7cfd3db reduction
- **`reptimeline/README.md`**: converted from full doc to snapshot pointer (591 → 68 lines); active dev moved to standalone repo
- **CITATION.cff version**: 0.1.0 → 0.3.0 (synced with .zenodo.json)
- **Keywords expanded** in .zenodo.json (9 → 16) and CITATION.cff (7 → 12) — added subsumption, analogy verification, compositional generalization, reptimeline, causal verification, etc.
- **Paper grew 27 → 30 pages** with new figures and audit-driven expansions

### Removed
- 5 orphan files in `paper/figures/`: scaling_loss.png, scaling_panel.png, bubble_cohesion_separation.png, ubs_distribution.png, training_curves.pdf (duplicate)
- 2 unused subfigure labels (`fig:entropy_heatmap`, `fig:entropy_dist`)
- ~230 GB of intermediate `step*.pt` checkpoints (Phase 1 disk cleanup; whitelisted 4 referenced step checkpoints)

### Research
- All 14 audit issues resolved; 0 unused refs, 0 LaTeX errors
- D-A19 (GPT-2 355M) restores algebra at scale: 76.9% subsumption (895 pairs), 100% analogy

## [0.2.0] - 2026-04-14
- Bumped `.zenodo.json` version metadata for republication

## [0.1.0] - 2026-03-24

### Added
- **triadic-head** published to PyPI (`pip install triadic-head`)
- OIDC Trusted Publishing via GitHub Actions
- CI workflow (tests on Python 3.10, 3.11, 3.12)
- Full badge set: CI, PyPI, Python, License, HuggingFace, DOI
- Zenodo integration for software archival
- Data download instructions (data/README.md)
- Community files: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- Issue and PR templates
- GitHub Discussions enabled

### Fixed
- Removed internal AI agent config files (.claude/, .agents/, CLAUDE.md) from repository
- Fixed repo structure in README (removed duplicates and gitignored directories)

### Research
- 11 experiments documented (60+ runs)
- Production model: Run 15 (v1.4-strongalign), 40M params
- Paper: 27 pages, all experiments included
- Two HuggingFace models: triadic-gpt-40m, triadic-gpt2-medium

## [0.9-beta] - 2026-03-20

### Added
- D-A19: GPT-2 355M with restored algebra
- D-A18: reptimeline integration
- Beta snapshot before public release refactor

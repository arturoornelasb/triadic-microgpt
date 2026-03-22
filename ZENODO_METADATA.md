# Zenodo Upload Metadata

## Title
End-to-End Prime Factorization in a Generative Language Model: Emergent Algebraic Semantics from Joint Training

## Authors
- **J. Arturo Ornelas Brand** — arturoornelas62@gmail.com

## Upload Type
Publication / Preprint

## Publication Date
2026-03-22

## DOI
(Auto-assigned by Zenodo)

## Description / Abstract
We present TriadicGPT, a 40M-parameter GPT language model augmented with a triadic projection head that produces discrete prime-factor signatures alongside standard next-token predictions. Unlike the post-hoc approach of the Triadic-Neurosymbolic-Engine, which projects frozen sentence embeddings into prime composites, TriadicGPT learns triadic representations end-to-end through a dual-objective training loss combining language modeling with a novel embedding alignment objective.

Across 29+ training runs and systematic ablation studies, we demonstrate eight principal findings: (1) the triadic head adds negligible cost to language quality (+1.7% perplexity); (2) semantic ordering emerges gradually with scale, crossing zero around 20M parameters; (3) a bits sweep reveals an optimal regime at k=32-64, shifted upward from the k=6-12 range for post-hoc projection; (4) attaching the triadic head to pre-trained GPT-2 with InfoNCE alignment closes 48% of the gap to the Engine's post-hoc PCA projection; (5) a differentiable subsumption loss recovers 100% held-out subsumption at k=64; (6) an iFSQ activation resolves the subsumption-language tradeoff entirely; (7) compositional analysis reveals that the bit space functions as a computational substrate with sub-linear error accumulation; and (8) a discovery loop expanding from 50 hand-labeled anchors to 158 improves holdout accuracy from 87% to 93%.

TriadicGPT achieves 98% analogy verification (50/51 analogies), 100% signature uniqueness, and reproducible semantic ordering (+0.038 +/- 0.005 gap, n=3, 95% CI positive) — all within a single forward pass.

## Keywords
- neurosymbolic AI
- prime factorization
- language model
- algebraic semantics
- discrete representations
- triadic projection
- subsumption
- analogy verification
- iFSQ
- ternary quantization
- compositional generalization
- information bottleneck
- embedding alignment
- GPT

## License
Business Source License 1.1 (BUSL-1.1)

- **Change Date**: 2030-03-22
- **Change License**: AGPL-3.0
- **Exception**: `triadic-head/` is MIT
- **Paper (PDF)**: CC-BY-4.0 (for academic sharing on Zenodo)
- **Commercial licensing**: support@fuaflow.com

## Language
English

## Related Identifiers
- **Is supplemented by**: https://github.com/arturoornelasb/triadic-microgpt (GitHub repository, source code)
- **Is new version of**: Triadic-Neurosymbolic-Engine (parent library, https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine)
- **References**: neurosym v0.2.0 on PyPI

## Subjects
- Artificial Intelligence
- Natural Language Processing
- Machine Learning
- Neurosymbolic AI
- Representation Learning

## Communities
(Optional — submit to relevant Zenodo communities if applicable)

## Notes
- Trained on TinyStories (50K stories) on a single NVIDIA RTX 5060 Ti (16 GB).
- All code and checkpoints available in the linked GitHub repository.
- Companion tool: reptimeline (training dynamics analysis for discrete representations).

## Files to Upload
1. `paper/triadic_microgpt.pdf` — The paper (compiled LaTeX)
2. `paper/triadic_microgpt.tex` — LaTeX source
3. `paper/figures/` — All figures referenced by the paper

## Citation (BibTeX)
```bibtex
@article{ornelas2026endtoend,
  title={End-to-End Prime Factorization in a Generative Language Model: Emergent Algebraic Semantics from Joint Training},
  author={Ornelas Brand, J. Arturo},
  year={2026},
  note={Preprint. Available at Zenodo.}
}
```

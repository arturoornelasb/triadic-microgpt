# Zenodo Upload Metadata

## Title
End-to-End Prime Factorization in a Generative Language Model: Learned Algebraic Encoding from Joint Training

## Authors
- **J. Arturo Ornelas Brand** — arturoornelas62@gmail.com

## Upload Type
Publication / Preprint

## Publication Date
2026-05-03

## DOI
(Auto-assigned by Zenodo)

## Description / Abstract
We present TriadicGPT, a 40M-parameter GPT language model augmented with a triadic projection head that produces discrete prime-factor signatures alongside standard next-token predictions. Unlike the post-hoc approach of the Triadic-Neurosymbolic-Engine, which projects frozen sentence embeddings into prime composites, TriadicGPT learns triadic representations end-to-end through a dual-objective training loss combining language modeling with a novel embedding alignment objective.

Across 29+ training runs and systematic ablation studies, we demonstrate five principal findings: (1) the triadic head adds negligible cost to language quality (+1.7% perplexity); (2) semantic ordering emerges gradually with scale, crossing zero around 20M parameters, with multi-seed validation confirming reproducibility (+0.038 +/- 0.005 gap, n=3, 95% CI positive); (3) attaching the triadic head to pre-trained GPT-2 with InfoNCE alignment closes 48% of the gap to the Engine's post-hoc PCA projection (+0.076 vs Engine +0.136); (4) a differentiable subsumption loss combined with an iFSQ activation resolves the subsumption-language tradeoff: language quality is preserved while achieving up to 87.1% held-out subsumption and 100% at k=64; and (5) compositional analysis reveals that the bit space supports multi-step algebraic operations: round-trip accuracy (98.1%) far exceeds the multiplicative prediction (81.9%), with sub-linear error accumulation in two-step transitive chains.

TriadicGPT achieves 98% analogy verification (50/51 analogies), 100% signature uniqueness, and 100% analogy verification across all algebraic operations — all within a single forward pass. External evaluation on SimLex-999 reveals that the system does not capture graded semantic similarity (Jaccard rho = -0.012, cosine rho = 0.046; neither significant), motivating the characterization as algebraic encoding rather than algebraic semantics. Sparse fine-tuning experiments reveal a fundamental sparsity-compositionality trade-off: dense codes preserve compositional structure while sparse codes enable subsumption detection, but no single configuration satisfies both. The companion tool reptimeline operationalizes temporal observability of discrete representations, with backend-agnostic validation on MNIST autoencoders (100% causal control via code swap) and Pythia-70M sparse autoencoders (16/16 features show selective causal effects).

## Keywords
- neurosymbolic AI
- prime factorization
- language model
- algebraic encoding
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
- reptimeline
- SimLex-999
- sparse autoencoder
- causal verification

## License
Business Source License 1.1 (BUSL-1.1)

- **Change Date**: 2030-03-22
- **Change License**: AGPL-3.0
- **Scope**: All code including `triadic-head/`
- **Paper (PDF)**: CC-BY-4.0 (for academic sharing on Zenodo)
- **Commercial licensing**: arturoornelas62@gmail.com

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
  title={End-to-End Prime Factorization in a Generative Language Model: Learned Algebraic Encoding from Joint Training},
  author={Ornelas Brand, J. Arturo},
  year={2026},
  note={Preprint. Available at Zenodo.}
}
```

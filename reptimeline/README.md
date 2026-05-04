# reptimeline (snapshot)

> **This folder is a frozen snapshot of reptimeline as it was used in the TriadicGPT paper.**
> **Active development is at [github.com/arturoornelasb/reptimeline](https://github.com/arturoornelasb/reptimeline) (latest release: v0.2.0).**
> The standalone package has features this snapshot does not — causal verification, interactive Plotly visualizations, JSON round-trip, MNIST/Pythia backends, CI, and packaging. Use the standalone for anything other than reproducing this paper.

---

**reptimeline** tracks how discrete representations evolve during neural network training — lifecycle events (births, deaths, connections), phase transitions, bottom-up ontology discovery (duals, dependencies, 3-way interactions), and theory reconciliation. Backend-agnostic: works with triadic bits, VQ-VAE, FSQ, sparse autoencoders, or any discrete bottleneck.

## Use the standalone

```bash
pip install git+https://github.com/arturoornelasb/reptimeline
```

- **Repo**: https://github.com/arturoornelasb/reptimeline
- **Code DOI**: [10.5281/zenodo.19208627](https://doi.org/10.5281/zenodo.19208627)
- **Paper DOI**: [10.5281/zenodo.19208672](https://doi.org/10.5281/zenodo.19208672)
- **License**: BUSL-1.1 (Change Date 2030-03-20, Change License AGPL-3.0)

## What's preserved in this folder

This snapshot is kept so the TriadicGPT paper remains reproducible from a single repo. It corresponds to the code state between commit `34e5f57` (initial reptimeline) and `de8fa47` (D-A19 analysis).

### Code

The package as it shipped with the paper: `core.py`, `tracker.py`, `discovery.py`, `autolabel.py`, `reconcile.py`, `extractors/{base,triadic}.py`, `overlays/primitive_overlay.py`, `viz/{swimlane,phase_dashboard,churn_heatmap,layer_emergence}.py`, `cli.py`, `tests/test_{smoke,discovery,reconcile}.py`.

### Saved results (referenced by the paper)

| Path | Model | Contents |
|---|---|---|
| `results/danza_63bit_xl/` | D-A14 v1 | `timeline.json` (8 ckpts: 2,500→50,000) + 4 plots — original POC |
| `results/d_a14_v2_discovery.json` + `d_a14_v2_plots/` | D-A14 v2 (production) | Discovery + 4 plots, 49 active / 14 dead bits |
| `results/d_a14_v2_autolabel.json`, `d_a14_v2_autolabel_report.json` | D-A14 v2 | AutoLabeler with embedding + contrastive strategies, 302 concepts |
| `results/d_a18_discovery.json` | D-A18 (iFSQ + hybrid 30+33) | 41 active / 22 dead, 5 duals, 11 triadic interactions |
| `results/d_a19_discovery.json` | D-A19 (GPT-2 355M sparsity_v2) | 47 active / 16 dead, 6 duals |

### Reproducing the snapshot tests

```bash
# From the triadic-microgpt root
python reptimeline/tests/test_smoke.py        # synthetic, always works
python reptimeline/tests/test_discovery.py    # synthetic, always works
python reptimeline/tests/test_reconcile.py    # requires step checkpoints in checkpoints/danza_63bit_xl/
```

`test_reconcile.py` skips when only `model_best.pt` is present — the original 8-step sequence is no longer in the repo. The numbers it produced (44.4% agreement, 35 critical mismatches) are preserved in the paper and in `results/danza_63bit_xl/timeline.json`.

### CLI as it shipped

```bash
python -m reptimeline --checkpoint-dir checkpoints/danza_63bit_xl_v2/ \
                      --primitives --overlay --plot \
                      --max-checkpoints 8 --output timeline.json
```

## Origin notes

The snapshot here predates the API changes in the standalone package:

- `TriadicExtractor` is part of the package here; in the standalone it lives in `examples/`.
- The CLI here hardcodes triadic-specific paths; the standalone uses `--snapshots <json>`.
- `PrimitiveOverlay` here auto-detects `playground/danza_data/primitivos.json`; the standalone requires an explicit path.
- No `to_dict`/`from_dict` serialization on `core.py` here.

If you want the up-to-date API, install the standalone. If you want to re-run paper figures, use this folder.

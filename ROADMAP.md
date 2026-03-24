# Roadmap — TriadicGPT Production & Commercialization

Status snapshot as of 2026-03-22. All items are derived from the actual repo state, not aspirational.

---

## Tecnico

### Listo (no requiere trabajo)

- [x] Modelo de produccion: Run 15 (40M, PPL 7.69, gap +0.020, 0% costo lenguaje)
- [x] Modelo supervisado: D-A14 v2 (93% test, 98.3% subsumption, 158 anchors)
- [x] Algebra a 355M restaurada: D-A19 (97.1% bits, 76.9% sub, 100% R3)
- [x] 37 unit tests pasando (autograd, transformer, triadic, integration)
- [x] 12 benchmark scripts ejecutados, 27 JSON de resultados
- [x] 8 audit tests formales con resultados (PF bridge, Aristotelian, enantiodromia, cherry-picking)
- [x] Paper compilado (27pp, 11 experimentos, 12 figuras)
- [x] Desktop UI funcional (PySide6, 7 tabs, 3 backends)
- [x] BitwiseValidator isomorfico a PrimeMapper (1000/1000 equiv, 5-78x mas rapido)
- [x] Bugs en torch_finetune.py corregidos (GradScaler, dtype, dist-weight)
- [x] Correcciones de paper integradas (+0.099 -> +0.076, 70% -> 48%, "zero cost" -> "+1.7%")
- [x] triadic-head v0.1.0 construido y validado (wheel + tar.gz, signal +8.5% sobre random)
- [x] reptimeline v0.1.0 committed al repo con tests, viz, y discovery
- [x] Guia de reproducibilidad completa (playground/REPRODUCIBILITY.md, ~46h GPU)
- [x] environment.yml + requirements.txt para instalacion

### Pendiente — Prioridad Alta

- [ ] **Publicar triadic-head en PyPI** — wheel listo en `triadic-head/dist/`, falta `twine upload`. Publicar DESPUES del paper para que la referencia exista.
- [ ] **Publicar reptimeline en PyPI** — codigo listo, falta revisar pyproject.toml y subir con twine. Depende de validacion con un segundo backend (no triadic) para confirmar generalidad.
- [ ] **CI/CD basico** — No hay GitHub Actions. Minimo: (1) `pytest tests/test_all.py` en push, (2) `pytest triadic-head/tests/` en push, (3) lint con ruff/flake8. Un workflow de ~30 lineas.
- [ ] **Limpiar archivos legacy de raiz** — `model_fast.npz` (431 KB), `model_fast.vocab` (48 B), `verify_training.py` (usa modulos obsoletos), `tokenizer.json` (13 KB, NO es el de Run 15). Estos confunden a usuarios nuevos.
- [ ] **Estrategia de checkpoints** — 45 dirs, ~235 GB via Git LFS. Solo 4-6 son utiles (Run 15, D-A14, D-A19, Concept 49-bit, chat_run8, Exp10 InfoNCE). El resto son artefactos experimentales. Opciones: (a) mover a un release de GitHub, (b) Zenodo dataset, (c) documentar cuales ignorar.

### Pendiente — Prioridad Media

- [ ] **torch.compile en Windows** — Requiere Triton (solo Linux). El guard `try: import triton` ya existe, pero training en Windows es ~10-30% mas lento. Opciones: (a) WSL2 doc, (b) Docker con CUDA, (c) aceptar el overhead.
- [ ] **Crear implementacion minima de referencia** — <500 lineas, solo torch_transformer.py + triadic.py + train loop. Para que usuarios entiendan la tecnica sin navegar 127 .py files. Pendiente en `pending_minimal_triadic.md`.
- [ ] **Hardcoded tokenizer paths** — Varios scripts asumen paths relativos a checkpoints/. Centralizar en un `config.py` o variable de entorno.
- [ ] **Conceptual tokenizer Phases 5-6** — Phase 4 (post-hoc) fallo. Phase 4b (end-to-end, 86.2%) funciono. Falta: Phase 5 (subsumption loss supervisado) y Phase 6 (self-supervised a 40M+). Experimental, no bloquea nada.
- [ ] **Relational prime chains** — Extension propuesta para anti-alucinacion algebraica O(1). Documentada en `research/relational_prime_chains.md`, codigo no iniciado. Phase A (post-hoc, 0 GPU, ~5 dias) es viable como siguiente paper.

### Pendiente — Prioridad Baja

- [ ] **Dead bits (~15/64)** — Entropy regularization mitiga pero no elimina. Aceptable para paper. iFSQ activation (D-A16) reduce a ~0 en modo supervisado, pero no resuelve el caso self-supervised.
- [ ] **R3 loss muerta a k=64** — 3 experimentos confirman colapso a 64/64 dead bits. A k=6-12 funciona pero destruye semantic gap. Documentado como limitacion, no como bug.
- [ ] **Cross-dataset generalization** — Run 15 entrenado solo en TinyStories. PPL en WikiText-2 y LAMBADA es alto (OOD esperado). Para produccion real, entrenar en corpus mas diverso.
- [ ] **HuggingFace model card** — Deliberadamente omitido (la tecnica importa, no los pesos). Reconsiderar si se busca adopcion academica.

---

## Comercial

### Activos Listos

- **triadic-head** (BUSL-1.1) — Paquete standalone para cualquier modelo HF. API: wrap, train, encode, compare, validate, explore. Soporta GPT-2, LLaMA, Mistral, Phi, Qwen, OPT, Falcon.
- **neurosym-client** (Proprietary, v0.1.0 en PyPI) — SDK Python para triadic-cloud API.
- **triadic-cloud** (Proprietary, repo privado) — FastAPI + Stripe ($29-299/mo), desplegado en Railway. Usa neurosym (post-hoc).
- **Paper** — 27 paginas compilado, listo para Zenodo. 11 experimentos, 29 runs, resultados reproducibles.
- **Desktop UI** — Aplicacion funcional para demos presenciales o screencasts.

### Pendiente — Monetizacion

- [ ] **Endpoint `/encode-e2e` en triadic-cloud** — Actualmente solo usa neurosym (post-hoc). Agregar backend MicroGPT para ofrecer encoding end-to-end como servicio. Requiere: (1) serializar Run 15 para inference, (2) endpoint FastAPI, (3) actualizar pricing tier.
- [ ] **Publicar paper en Zenodo** — PDF listo. Crear DOI. Sin arXiv (investigador independiente, sin afiliacion institucional). El DOI es necesario para que triadic-head tenga referencia citable.
- [ ] **Publicar triadic-head en PyPI** — Secuencia: paper DOI primero -> luego PyPI con DOI en metadata.
- [ ] **Documentacion publica** — No hay site de docs (readthedocs, GitHub Pages). El README es extenso (529 lineas) pero no reemplaza docs interactivas. Minimo: un tutorial "attach triadic head to GPT-2 in 20 lines".
- [ ] **Demo hosted** — No hay playground web. Opciones: (a) Gradio/Streamlit en HuggingFace Spaces (gratis), (b) pagina en fuaflow.com. La UI de escritorio no sirve como demo remota.
- [x] **Licencia** — BUSL-1.1 con modelo de consorcio. Individuos/academia/nonprofits gratis, empresas participan. Change Date 2030-03-22 → AGPL-3.0.

### Estrategia de Ecosistema (Definida)

| Componente | Licencia | Estado | Rol |
|------------|----------|--------|-----|
| triadic-head | BUSL-1.1 | Built, no publicado | Paquete standalone |
| triadic-microgpt | BUSL-1.1 | Completo | Referencia de investigacion + paper |
| triadic-cloud | Proprietary | Desplegado | Revenue |
| neurosym-client | Proprietary | v0.1.0 en PyPI | SDK para cloud API |
| Triadic Engine | BUSL-1.1 | v0.2.0 en PyPI | Libreria post-hoc (parent) |
| Paper | CC-BY-4.0 (Zenodo) | PDF listo | Credibilidad academica |

---

## Bloqueos

### Bloqueo 1: Paper sin DOI

**Problema**: El paper esta compilado (27pp) pero no tiene DOI ni esta publicado en ninguna plataforma. Sin DOI, triadic-head no puede referenciar el paper en su metadata de PyPI, y la tecnica no es citable.

**Causa raiz**: Investigador independiente sin afiliacion institucional. arXiv requiere endorsement.

**Opciones**:
1. **Zenodo** (mas rapido) — Subir PDF, obtener DOI en minutos. Sin peer review pero citable.
2. **OpenReview** — Submision abierta, peer review publico.
3. **arXiv endorsement** — Buscar endorser en cs.CL o cs.AI.

**Impacto**: Bloquea publicacion de triadic-head en PyPI (secuencia: DOI -> PyPI).

### Bloqueo 2: reptimeline necesita segundo backend

**Problema**: reptimeline esta disenado como backend-agnostic (ABC `RepresentationExtractor`), pero solo tiene un extractor implementado (`TriadicExtractor`). Para publicarlo como herramienta general, necesita al menos un segundo backend (VQ-VAE, SAE, o FSQ) que confirme que la abstraccion funciona.

**Causa raiz**: Prioridad fue el paper, no la generalidad del tooling.

**Opciones**:
1. Implementar `SAEExtractor` (~2 dias, SAE es el mas natural siguiente).
2. Publicar como triadic-only y generalizar despues.
3. Mantener en el repo sin publicar en PyPI.

**Impacto**: Bloquea publicacion de reptimeline como paquete standalone. No bloquea nada mas.

### Bloqueo 3: Tamanio del repositorio (~235 GB checkpoints)

**Problema**: Git LFS trackea `.pt` files, pero el repo tiene 45 directorios de checkpoints. Clonar el repo descarga ~235 GB. Esto hace el proyecto inaccesible para la mayoria de usuarios.

**Causa raiz**: Cada experimento guardo checkpoints intermedios. No hubo cleanup despues de la fase experimental.

**Opciones**:
1. **GitHub Releases** — Mover checkpoints a releases tagged (Run 15, D-A14, D-A19 como assets descargables).
2. **Zenodo dataset** — Upload como dataset citeable con DOI separado.
3. **HuggingFace Hub** — Subir solo los 4-6 modelos utiles.
4. **Purge + script** — Eliminar checkpoints del repo, agregar `scripts/download_checkpoints.py`.

**Impacto**: Bloquea adopcion publica del repo. No bloquea desarrollo local.

### No son bloqueos (resueltos o aceptados)

| Item | Estado | Razon |
|------|--------|-------|
| Bugs en torch_finetune.py | RESUELTO | Todos corregidos en commit 126eb7b |
| D-A17 algebra destruida a 355M | RESUELTO | D-A19 restaura algebra (bugs en loss, no limitacion de escala) |
| Coherence loss = collapse | RESUELTO | Permanentemente removida, documentada como anti-patron |
| Dead bits ~15/64 | ACEPTADO | Mitigado con entropy reg, no eliminable sin iFSQ supervisado |
| R3 muerta a k=64 | ACEPTADO | Documentado como limitacion inherente, funciona a k=6-12 |
| torch.compile en Windows | ACEPTADO | Guard existe, overhead 10-30% aceptable para desarrollo |
| Paper corrections | RESUELTO | 7 correcciones integradas en commits recientes |

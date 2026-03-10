# TriadicGPT Explorer — Desktop UI

Interactive desktop application for exploring, auditing, and conversing with TriadicGPT. Built with PySide6 (Qt for Python) and embedded matplotlib.

## Installation

```bash
conda activate triadic-microgpt
pip install PySide6>=6.4
```

## Launch

```bash
cd /path/to/triadic-microgpt
python ui/app.py
```

---

## Interface Overview

The application has a persistent **model panel** at the top and **7 tabs** below. Tabs 1–6 require a loaded model; Tab 7 (Benchmarks) works standalone.

```
┌────────────────────────────────────────────────────────────────────┐
│  TriadicGPT Explorer       Backend: [TriadicGPT nativo]            │
│  Checkpoint: [checkpoints/torch_run15_strongalign/...]  […] [Cargar]│
│  Tokenizer:  [checkpoints/torch/tokenizer.json]          […]       │
│  ████████████████████████████████████████████████ (loading bar)    │
├──────────┬──────────┬──────────┬──────────┬──────────┬────────┬───┤
│ Encoder  │ Comparar │ Explorar │ Analogía │ Validar  │  Chat  │ B │
├──────────┴──────────┴──────────┴──────────┴──────────┴────────┴───┤
│                          [tab content]                             │
└────────────────────────────────────────────────────────────────────┘
│ Modelo cargado: 12L/512D/8H/64bits | CUDA | 40.1M params           │
```

---

## Model Panel — Loading a Model

Located at the top of the window. **You must load a model before using any tab** (except Benchmarks). The app supports three backends:

### Backend 1: TriadicGPT Native (.pt)

This loads a from-scratch TriadicGPT model trained with `src/torch_train.py`.

| Field | Value |
|-------|-------|
| Backend | `TriadicGPT Native (.pt)` |
| Checkpoint | Path to `.pt` file (e.g. `checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt`) |
| Tokenizer | Path to `tokenizer.json` (e.g. `checkpoints/torch/tokenizer.json`) |
| Bits | Fixed from checkpoint config (not editable) |

**When to use:** You trained your own TriadicGPT from scratch and have both a `.pt` checkpoint and a `tokenizer.json`.

> **Note:** The tokenizer must match the training run. All runs ≥ Run 7 use `checkpoints/torch/tokenizer.json`. The Run 15 tokenizer was deleted — use the one from `checkpoints/torch/`.

### Backend 2: GPT-2 Transfer (Exp10)

This loads a pre-trained GPT-2 with a triadic projection head fine-tuned on top (Experiment 10).

| Field | Value |
|-------|-------|
| Backend | `GPT-2 Transfer (Exp10)` |
| Transfer .pt | Path to the transfer checkpoint (e.g. `experiment10/checkpoints_infonce/phase_2_(unfreeze_last_layers)_final.pt`) |
| Bits | Editable (default 64) |

**When to use:** You want the best triadic quality. The InfoNCE transfer model closes 72% of the gap to Engine PCA and has the strongest semantic signal.

> **Note:** This backend uses GPT-2's tokenizer (downloaded automatically from HuggingFace on first use). The text generation in the Chat tab will produce TinyStories-style output because GPT-2 was fine-tuned on that dataset.

### Backend 3: HuggingFace (TriadicWrapper)

This wraps any HuggingFace causal LM with a post-hoc triadic projection (no trained head — uses PCA/random projection).

| Field | Value |
|-------|-------|
| Backend | `HuggingFace (TriadicWrapper)` |
| HF model or .pt | HF model name (`gpt2`, `gpt2-medium`, etc.) or path to saved weights |
| Bits | Editable (8–128, default 64) |
| Align mode | `infonce` / `rank` / `mse` |

**When to use:** Quick exploration with any HuggingFace model. Lower triadic quality than a trained head but works with any model out of the box.

> **Note:** Chat is not available with this backend.

### Loading process

1. Select a backend from the dropdown
2. The checkpoint/tokenizer fields auto-fill with defaults (edit if needed)
3. Click **Load Model**
4. A progress bar appears while the model loads in a background thread
5. Status turns green: `✓ GPT-2 Transfer/64bits | CUDA | 124.5M params`
6. After loading, 280 probe words are indexed in the background for the Prime Inspector
7. Status bar shows: `Model ready -- 280 probe words indexed -- click any prime to inspect`

---

## Tab 1 — Encoder

**Purpose:** Visualize how any text is encoded into a prime signature.

**Workflow:**
1. Type a concept or phrase in the text box
2. Click **Encode →**
3. View the resulting bit vector, prime composite Φ, and projection bar chart

**Output panels:**
- **BIT VECTOR** — 64 colored squares. Green = active bit, dark = inactive. Hover shows `Bit i — prime pᵢ = N | val = X.XXX`
- **PRIME COMPOSITE** — Φ = 2 × 3 × 7 × 11 × ... (copy to clipboard with "Copiar Φ")
- **PROJECTION** — Bar chart of raw tanh activations per bit. Green bars = active (>0), red = inactive

---

## Tab 2 — Compare

**Purpose:** Compare two concepts algebraically.

**Workflow:**
1. Enter concept A and concept B
2. Click **Compare →**

**Output:**
- **Similarity %** — Jaccard similarity over prime factor sets, with visual bar
- **A ⊇ B / B ⊇ A** — Subsumption (divisibility check)
- **Factor chips** — Clickable prime number buttons organized by category:
  - **Shared** (green) — primes present in both A and B
  - **Only in A** (yellow) — primes unique to A
  - **Only in B** (peach) — primes unique to B
  - **Click any chip** to open the **Prime Inspector** (see below)
- **LCM composition** — A∪B = lcm(Φ_A, Φ_B)
- **Bit vectors** — Side-by-side comparison. Color encoding:
  - Green = shared active bit
  - Yellow = only A active
  - Peach = only B active
  - Dark = both inactive

### Prime Inspector (dialog)

When you click any prime chip in the Compare tab, a **Prime Inspector** dialog opens. This is the key tool for understanding what each prime "means" — since primes are learned features, their meaning is discovered empirically by looking at which words share them.

**Left panel — Words sharing this prime:**
- Shows all probe words whose encoding includes this prime factor
- Displays count: `N / 280 words share prime P`

**Right panel — Editable probe vocabulary:**
- Text area with one word per line (default: 280 words across 20+ semantic domains)
- Edit the list to add domain-specific words, remove irrelevant ones, or focus on a topic
- Click **Apply Vocabulary** to re-encode and refresh the word list

**How to interpret primes:**
- A prime shared by `king, queen, prince, emperor, lord` likely encodes "royalty" or "authority"
- A prime shared by `dog, cat, horse, bird, fish` likely encodes "animal"
- A prime shared by `red, blue, green, white, black` likely encodes "color"
- Some primes may encode more abstract features (e.g., "concrete noun" or "positive valence")
- The system does NOT claim to know what a prime means — it shows you the data and you interpret the pattern

---

## Tab 3 — Explore

**Purpose:** Compute a pairwise similarity matrix for N concepts.

**Workflow:**
1. Add words with the text field + **+ Agregar** (or press Enter)
2. Select and click **Eliminar** to remove one; **Limpiar** to reset
3. Click **Explore →**

**Default word set:** king, queen, dog, cat, doctor, hospital

**Output:**
- **Similarity heatmap** — N×N matrix, RdYlGn colormap (0=red, 1=green), values annotated
- **Ranked pairs table** — All unique pairs sorted by similarity (highest first), with count of shared prime factors

---

## Tab 4 — Analogy

**Purpose:** Solve and verify prime algebra analogies A:B::C:?

**Workflow:**
1. Fill in A, B, C (e.g. `king`, `queen`, `man`)
2. Click **Calcular →**

**Output:**
- **Transformación A→B** — Which prime factors were removed (red) and added (green) from A to B
- **TARGET Φ** — The computed target prime: `Φ_target = (Φ_C / gcd(Φ_A, Φ_C)) × (Φ_B / gcd(Φ_A, Φ_B))`
- **Results table** — Top-10 vocabulary matches ranked by similarity to Φ_target. ✓ = similarity ≥ median pool similarity
- **Verification** — Top-1 similarity vs median; shows VERIFICA / NO verifica

**Note:** The search pool uses the 280-word probe vocabulary (same words used by the Prime Inspector). If the vocabulary has been warmed in the background, results are near-instant since encodings are cached.

---

## Tab 5 — Validate

**Purpose:** Run the full semantic quality audit suite.

**Checks:**
| Check | Passes if |
|-------|-----------|
| **Diversity** | ≥ 75% of encoded concepts have unique prime signatures |
| **Active Bits** | Mean active bit fraction is between 15% and 85% |
| **Semantic Ordering** | Mean intra-group similarity > mean inter-group similarity (gap > 0) |

**Workflow:**
1. Edit word groups in the table (add groups with name + comma-separated words)
2. Click **Run Validation →**

**Default groups:** royalty, animals, medicine, technology

**Output:**
- Global **PASS / FAIL** banner
- Three colored check cards (green = pass, red = fail)
- Per-group table: intra similarity, inter similarity, gap, status

---

## Tab 6 — Chat

**Purpose:** Converse with TriadicGPT and inspect the triadic signature of each turn in real time.

> **Note:** Chat works with the **Native** and **GPT-2 Transfer** backends. HuggingFace mode will return an error. The Transfer backend generates TinyStories-style text (it was fine-tuned on that dataset).

**Layout (horizontal splitter):**

```
┌─────────────────────────────────┬──────────────────────┐
│  [conversation history]         │ ANÁLISIS TRIÁDICO     │
│                                 │ PREGUNTA             │
│  Usuario: What is a king?       │ Φ: 11688...          │
│  IA: A king is a ruler...       │ [bit vector]         │
│                                 │ RESPUESTA            │
│  [write message...] [Enviar]    │ Φ: 25221...          │
│  [Limpiar]                      │ [bit vector]         │
│                                 │ SIMILITUD: 51%       │
│                                 │ RESP ⊇ PREG: ✗ No   │
│                                 │ COMPARTIDOS: 2 3 7   │
│                                 │ RESP tiene de más:   │
│                                 │ 5 17 23              │
└─────────────────────────────────┴──────────────────────┘
```

The right panel updates after each AI response with the prime signature of the **last prompt** and **last response**, plus their algebraic relationship.

---

## Tab 7 — Benchmarks

**Purpose:** Browse all stored benchmark results in `benchmarks/results/`. **Does not require a loaded model.**

**Workflow:**
1. Select a JSON file from the list (filter by typing in the search box, click ↻ to refresh)
2. Results load automatically

**Supported benchmark types:**

| File pattern | Display |
|---|---|
| `*analogy*` | Top-1/Top-5/Verification accuracy cards + details table + bar chart |
| `*bit_entropy*` | Mean entropy + dead bits cards + per-bit bar chart (red = dead bit H<0.3) |
| `*language*` | PPL card + generated text samples |
| `*subsumption*` / others | Key metrics cards + generic table + bar chart |

**Results directory:** `benchmarks/results/` — 27 JSON files from all experimental runs.

---

## Architecture

```
ui/
├── app.py                  # QApplication entry point; loads style.qss
├── main_window.py          # QMainWindow: ModelPanel + QTabWidget(7) + QStatusBar
├── model_panel.py          # Top bar widget; emits model_loaded(ModelInterface)
├── model_interface.py      # Unified encode/compare/explore/analogy/validate/chat API
│
├── workers/
│   └── model_worker.py     # ModelLoadWorker (load .pt or HF) + TaskWorker (generic)
│
├── widgets/
│   ├── bit_vector_widget.py       # Custom QWidget: N colored squares (binary/gradient/compare)
│   ├── prime_display_widget.py    # Prime composite label + factors + copy button
│   ├── prime_inspector_dialog.py  # Click a prime → see all words sharing it + edit vocabulary
│   └── mpl_canvas.py              # FigureCanvasQTAgg wrapper with dark theme
│
├── tabs/
│   ├── encoder_tab.py      # Tab 1
│   ├── compare_tab.py      # Tab 2
│   ├── explore_tab.py      # Tab 3
│   ├── analogy_tab.py      # Tab 4
│   ├── validate_tab.py     # Tab 5
│   ├── chat_tab.py         # Tab 6
│   └── benchmarks_tab.py  # Tab 7 (no model needed)
│
└── resources/
    └── style.qss           # Dark theme (Catppuccin Mocha)
```

### Threading model

All model inference runs in `QThread` workers to keep the UI responsive:

```python
worker = TaskWorker(iface.encode, "king")
worker.result_ready.connect(on_result)   # fires in main thread via Qt signal
worker.error_occurred.connect(on_error)
worker.start()
```

`ModelLoadWorker` is a specialized worker for the slow initial model load (emits `progress` messages during loading).

### ModelInterface API

```python
# Core operations (all backends)
iface.encode(text)          -> {composite, bits, projection, n_active, factors}
iface.compare(a, b)         -> {similarity, shared_factors, only_a_factors, only_b_factors,
                                a_subsumes_b, b_subsumes_a, composition, enc_a, enc_b}
iface.explore([words])      -> {matrix, words, pairs, signatures}
iface.analogy(a, b, c)      -> {target_prime, transform_added, transform_removed,
                                matches, median_sim}
iface.validate({groups})    -> {checks, overall_pass, group_details, n_concepts,
                                unique_signatures}
iface.chat(prompt)          -> {response, prompt_prime, prompt_bits, response_prime,
                                response_bits, similarity, resp_subsumes_prompt,
                                shared_factors, resp_extra_factors}

# Probe vocabulary (for Prime Inspector)
iface.get_probe_words()     -> list[str]
iface.set_probe_words(ws)   -> None  (re-encodes new words)
iface.words_for_prime(p)    -> list[str]  (words sharing prime p)
iface.warm_vocab()          -> {n_words, n_probes}  (background pre-encoding)
```

All three backends (Native, Transfer, HuggingFace) expose the same API. Chat requires Native or Transfer.

---

## Customization

### Using a different model

Any `.pt` checkpoint trained with `src/torch_train.py` works. Select it via the **Browse (…)** button or type the path directly. The tokenizer must match the training run (all runs ≥ Run 7 use `checkpoints/torch/tokenizer.json`).

### Using a HuggingFace model

Switch Backend to **HuggingFace (TriadicWrapper)** and enter a HF model name (e.g. `gpt2`, `gpt2-medium`) or a path to a saved `.pt` weights file for the triadic head. Configure:
- **Bits (n_bits):** Number of triadic bits (8–128)
- **Align:** Alignment loss mode used during training (`infonce` recommended for pre-trained models)

### Editing the probe vocabulary

The probe vocabulary (280 default words) is used by the **Prime Inspector** and **Analogy** tabs. You can edit it at runtime:

1. Go to the **Compare** tab, compare any two words
2. Click any prime chip to open the **Prime Inspector**
3. Edit the word list in the right panel (one word per line)
4. Click **Apply Vocabulary** — all new words are encoded and the inspector refreshes

Changes persist for the session. Restarting the app resets to the default 280-word list.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: PySide6` | `pip install PySide6` |
| `Checkpoint not found` | Use the Browse (...) button to select the `.pt` file |
| `Tokenizer not found` | Use `checkpoints/torch/tokenizer.json` (works for all runs ≥ 7) |
| Chat returns error on HF backend | Chat requires Native or GPT-2 Transfer backend |
| Heatmap is blank | Add ≥ 2 words to the Explore list before clicking Explore |
| App launches but tabs are grey | Load a model first via the top bar |
| Analogy returns `AttributeError` | Make sure the model is loaded and probe vocab has warmed (check status bar) |
| Prime Inspector shows 0 words | Wait for "280 probe words indexed" in the status bar before clicking chips |
| GPT-2 Transfer gives children's stories | Expected — the backbone was fine-tuned on TinyStories. The triadic analysis (right panel) is what matters |

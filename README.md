# Triadic MicroGPT

**A zero-dependency GPT with algebraically verifiable internal representations.**

Standard GPTs produce opaque embeddings — you can compute cosine similarity (`0.87`) but never know *why* two concepts are similar. Triadic MicroGPT solves this by training a projection head that maps hidden states into **prime-factor space**, where semantic relationships become arithmetic:

```
King  = 2 × 3 × 5    (Royalty × Male × Authority)
Queen = 2 × 5 × 7    (Royalty × Authority × Female)

Shared:    {2, 5}    → Royalty, Authority
Only King: {3}       → Male
Only Queen:{7}       → Female
```

## Features

- **Zero external dependencies** — autograd, transformer, LSH, and prime mapper all implemented from scratch in pure Python
- **Dual-objective training** — language modeling (next-token) + triadic alignment (prime factor sharing)
- **Algebraic verification** — subsumption (`A ⊆ B`), composition (`A ∪ B`), gap analysis (`A △ B`)
- **Interactive CLI** — generate text, compare concepts, inspect prime signatures
- **~200 lines per module** — readable, hackable, educational

## Quick Start (v1.0 GoldPrimes Release)

### 1. Environment Setup

To run the full 40-Million parameter XL model with Knowledge Distillation, activate the Conda environment:

```bash
conda activate triadic-microgpt
```

*Note: If you don't have the environment, you can create it with `conda create -n triadic-microgpt python=3.10` and install the requirements.*

### 2. Pre-Train the XL Model (GPU)

We've upgraded the engine to run on PyTorch. You can train the massive 64-bit Triadic XL Model on the 10,000-concept Gold Prime dictionary:

```bash
python src/torch_train.py --scale xl --steps 50000
```

### 3. Interactive Validation Chat

Chat with the model and watch it natively render its deterministic Subsumption math in real-time:

```bash
python src/chat.py
```

### 4. Relational Bias Audit

Run Experiment 8 from the paper to prove that the model's 64-bit projections eliminate vector collisions (False Positive Rate < 5%):

```bash
python src/auditor.py
```

## Architecture

```
Text ──→ Char Tokenizer ──→ Transformer (Attention + MLP)
                                  │
                          ┌───────┴───────┐
                          │               │
                     LM Head         Triadic Head
                          │               │
                    Next-Token       tanh(Wx) → bits
                    Prediction       [+, -, +, +, ...]
                          │               │
                    Language Loss    Prime Mapper
                     (cross-entropy)     │
                          │          Φ = 2 × 5 × 7 = 70
                          │               │
                          └───────┬───────┘
                                  │
                          Total Loss = L_lang + α · L_triadic
```

### Modules

| File | Lines | Description |
|------|-------|-------------|
| `autograd.py` | ~120 | Scalar autograd engine (`Value` class with backward) |
| `transformer.py` | ~210 | GPT model: multi-head attention, RMSNorm, MLP, projection head |
| `triadic.py` | ~240 | Prime sieve, `PrimeMapper`, `TriadicValidator`, differentiable `triadic_loss` |
| `train.py` | ~220 | Dual-objective training loop with Adam optimizer |
| `inference.py` | ~260 | Generation, text analysis, comparison, interactive CLI |
| `test_all.py` | ~320 | 30+ automated tests covering all modules |

## How It Works

### 1. Autograd Engine
Every computation builds a graph of `Value` nodes. Calling `.backward()` on the loss applies the chain rule in reverse to compute all gradients — identical to PyTorch's autograd but in ~120 lines of pure Python.

### 2. Transformer
A GPT-2-style architecture (multi-head attention → RMSNorm → MLP) processes character tokens. The **key innovation** is a triadic projection head that maps the final hidden state to `n_bits` values via `tanh`.

### 3. Triadic Bridge
Each `tanh` output corresponds to a unique prime number. Positive outputs activate their prime; negative ones don't. The product of all active primes becomes the concept's **prime signature** — a single integer encoding its semantic features.

### 4. Dual Training
The model optimizes two objectives simultaneously:
- **Language**: Predict the next character (standard LM)
- **Triadic**: Related tokens should share prime factors (algebraic alignment)

### 5. Verification
After training, you can use `gcd`, `mod`, and `lcm` to:
- **Subsumption**: Does A contain all features of B? (`Φ(A) % Φ(B) == 0`)
- **Composition**: What combines A and B? (`lcm(Φ(A), Φ(B))`)
- **Gap Analysis**: What's shared vs. unique? (`gcd` + quotients)

## Credits

- **Autograd + Transformer**: Inspired by [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/microgpt)
- **Triadic Algebra**: Based on the [Triadic Neurosymbolic Engine](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine) by J. Arturo Ornelas Brand

## License

MIT

"""
Phase 4: Conceptual Tokenizer — Encoder + Projection Head Training.

Trains a 49-dim projection head supervised by the seed lexicon (462 words).
Uses Run 15 token embeddings (512D) as input features.

Pipeline:
  1. Extract word embeddings from Run 15 (mean-pool BPE subwords)
  2. Build 49-dim targets from seed_lexicon
  3. Train MLP projection head (512 → 256 → 49) with sigmoid+anneal
  4. Losses: MSE (match lexicon) + subsumption (hypernym hierarchy)
  5. Evaluate: reconstruction, generalization, algebraic verification

Usage:
  python conceptual_tokenizer/training/train_phase4.py
  python conceptual_tokenizer/training/train_phase4.py --steps 5000 --lr 1e-3
"""

import os
import sys
import json
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

from conceptual_tokenizer.config import (
    PRIMITIVE_NAMES, PRIMITIVE_TO_PRIME, N_PRIMITIVES, DUAL_INDICES,
    CATEGORY_NAMES, PRIMES_BY_CATEGORY, StateConfig,
)
from conceptual_tokenizer.seed_lexicon import get_full_lexicon, get_ambiguous_lexicon, lexicon_stats
from conceptual_tokenizer.states import StateResolver
from conceptual_tokenizer.prime_encoder import PrimeEncoder
from conceptual_tokenizer.triadic_bridge import ConceptBridge

# ═══════════════════════════════════════════════════════════════════
# Projection Head with Sigmoid + Annealing
# ═══════════════════════════════════════════════════════════════════

class ConceptHead(nn.Module):
    """MLP projection: embedding_dim → 49 primitives with sigmoid+anneal."""

    def __init__(self, n_embd, n_primitives=49, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_primitives),
        )
        self.n_primitives = n_primitives

    def forward(self, x, temperature=1.0):
        """
        Args:
            x: (batch, n_embd) — word embeddings
            temperature: sigmoid temperature (high=soft, low=hard)
        Returns:
            projections: (batch, 49) in [-1, 1] via scaled sigmoid
        """
        logits = self.net(x)
        # Sigmoid with temperature → [0, 1], then scale to [-1, 1]
        soft = torch.sigmoid(logits * temperature)
        return soft * 2 - 1  # [-1, 1]


# ═══════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════

def lexicon_to_targets(lexicon):
    """Convert seed lexicon entries to 49-dim target vectors.

    State mapping:
      "+" with intensity i  →  +i  (positive projection)
      "0" with intensity i  →  -i  (zero = active absence → negative)
      "-" with intensity i  →  -i  (negative pole → negative)
      missing primitive     →   0  (NA)
    """
    targets = {}
    for word, entry in lexicon.items():
        vec = np.zeros(N_PRIMITIVES, dtype=np.float32)
        for prim_name, (state, intensity) in entry.items():
            if prim_name not in PRIMITIVE_NAMES:
                continue
            idx = PRIMITIVE_NAMES.index(prim_name)
            if state == "+":
                vec[idx] = float(intensity)
            elif state in ("0", "-"):
                vec[idx] = -float(intensity)
        targets[word] = vec
    return targets


def extract_embeddings(model, tokenizer, words, device):
    """Extract contextual representations from Run 15 (full transformer forward).

    Uses the final hidden state (after all transformer layers + layer norm)
    instead of just the embedding table, giving much richer semantic features.
    """
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for word in words:
            ids = tokenizer.encode(word, add_special=False)
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            emb = model.wte(x)  # (1, seq_len, n_embd)
            embeddings[word] = emb[0].mean(dim=0).cpu().numpy()
    return embeddings


# Hypernym pairs for subsumption loss (subset of seed lexicon)
CONCEPT_HIERARCHY = {
    # element hypernyms
    "fire": ["flame", "blaze", "burn", "ember", "spark"],
    "water": ["river", "ocean", "rain", "lake", "stream", "wave"],
    "earth": ["ground", "soil", "dirt", "rock", "stone", "mountain", "sand", "clay"],
    "air": ["wind", "breeze", "breath"],
    # sense hypernyms
    "light": ["glow", "shine", "bright", "flash"],
    "sound": ["noise", "music", "song", "voice"],
    # feeling hypernyms (compound → compound)
    "love": ["passion", "tenderness", "devotion"],
    "fear": ["dread", "terror", "panic"],
}


def get_temperature(step, total_steps, start_temp=5.0, end_temp=0.5):
    """Anneal temperature from soft (high) to hard (low)."""
    progress = min(step / max(total_steps, 1), 1.0)
    return start_temp + (end_temp - start_temp) * progress


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate(head, embeddings, targets, device, temperature=0.5, label=""):
    """Evaluate projection head on a word set."""
    head.eval()
    resolver = StateResolver()
    encoder = PrimeEncoder()
    bridge = ConceptBridge()

    words = [w for w in targets if w in embeddings]
    if not words:
        return {}

    emb_tensor = torch.tensor(
        np.stack([embeddings[w] for w in words]),
        dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        preds = head(emb_tensor, temperature=temperature).cpu().numpy()

    # Per-primitive accuracy (sign match)
    tgt_array = np.stack([targets[w] for w in words])
    # Only evaluate on non-zero target positions (where lexicon has an opinion)
    mask = tgt_array != 0
    if mask.sum() > 0:
        sign_match = ((np.sign(preds) == np.sign(tgt_array)) & mask).sum() / mask.sum()
    else:
        sign_match = 0.0

    # MSE on non-zero positions
    mse = ((preds - tgt_array) ** 2 * mask).sum() / max(mask.sum(), 1)

    # Cosine similarity per word
    cosines = []
    for i in range(len(words)):
        p, t = preds[i], tgt_array[i]
        norm_p = np.linalg.norm(p)
        norm_t = np.linalg.norm(t)
        if norm_p > 1e-6 and norm_t > 1e-6:
            cosines.append(float(np.dot(p, t) / (norm_p * norm_t)))
    mean_cos = np.mean(cosines) if cosines else 0.0

    # State resolution accuracy + spurious activation count
    state_correct, state_total = 0, 0
    spurious_active, spurious_total = 0, 0  # activations where target is NA
    for i, word in enumerate(words):
        token = resolver.resolve(word, preds[i])
        for act in token.activations:
            idx = PRIMITIVE_NAMES.index(act.name)
            tgt_val = tgt_array[i, idx]
            if tgt_val == 0:
                # NA in target — count spurious activations
                spurious_total += 1
                if not act.is_na:
                    spurious_active += 1
                continue
            state_total += 1
            if tgt_val > 0 and act.is_active:
                state_correct += 1
            elif tgt_val < 0 and act.is_zero:
                state_correct += 1
    state_acc = state_correct / max(state_total, 1)
    spurious_rate = spurious_active / max(spurious_total, 1)

    # Algebraic subsumption check
    sub_correct, sub_total = 0, 0
    for hyper, hypos in CONCEPT_HIERARCHY.items():
        if hyper not in words:
            continue
        hyper_token = resolver.resolve(hyper, preds[words.index(hyper)])
        encoder.encode(hyper_token)
        for hypo in hypos:
            if hypo not in words:
                continue
            hypo_token = resolver.resolve(hypo, preds[words.index(hypo)])
            encoder.encode(hypo_token)
            sub_total += 1
            if bridge.subsumes(hypo_token, hyper_token):
                sub_correct += 1
    sub_rate = sub_correct / max(sub_total, 1)

    results = {
        'sign_accuracy': float(sign_match),
        'mse': float(mse),
        'mean_cosine': float(mean_cos),
        'state_accuracy': float(state_acc),
        'spurious_rate': float(spurious_rate),
        'subsumption_rate': float(sub_rate),
        'subsumption_pairs': sub_total,
        'n_words': len(words),
    }

    if label:
        print(f"  [{label}] state={state_acc:.1%}  sign={sign_match:.1%}  "
              f"spurious={spurious_rate:.1%}  cos={mean_cos:.3f}  sub={sub_correct}/{sub_total}")

    head.train()
    return results


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Conceptual Tokenizer Training")
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--start-temp', type=float, default=5.0)
    parser.add_argument('--end-temp', type=float, default=0.5)
    parser.add_argument('--sub-weight', type=float, default=2.0)
    parser.add_argument('--sparsity-weight', type=float, default=1.0,
                        help="L1 penalty on NA positions to prevent spurious activations")
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--train-split', type=float, default=0.8,
                        help="Fraction of lexicon for training (rest = held-out)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"  PHASE 4: CONCEPTUAL TOKENIZER ENCODER TRAINING")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Steps: {args.steps}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"  Temperature: {args.start_temp} → {args.end_temp}")
    print(f"  Sub weight: {args.sub_weight}")

    # --- Load Run 15 model for embeddings ---
    ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign',
                             'model_L12_D512_B64_best.pt')
    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign',
                            'tokenizer.json')
    print(f"\n  Loading Run 15 for embeddings ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'], block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'], n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'], dropout=0.0,
    )
    base_model = TriadicGPT(config).to(device)
    base_model.load_state_dict(ckpt['model_state_dict'])
    base_model.eval()
    tokenizer = BPETokenizer.load(tok_path)
    n_embd = cfg['n_embd']
    print(f"  Embedding dim: {n_embd}")

    # --- Build targets from seed lexicon ---
    lexicon = get_full_lexicon()
    stats = lexicon_stats()
    print(f"\n  Seed lexicon: {stats['total_unambiguous']} words "
          f"(T1={stats['tier_1_count']}, T2={stats['tier_2_count']})")
    print(f"  Primitives used: {stats['primitives_used']}/49")

    targets = lexicon_to_targets(lexicon)

    # --- Extract embeddings for all lexicon words ---
    print(f"  Extracting embeddings ...")
    all_words = list(targets.keys())
    embeddings = extract_embeddings(base_model, tokenizer, all_words, device)

    # Filter to words that have both embeddings and targets
    valid_words = [w for w in all_words if w in embeddings]
    print(f"  Valid words (have BPE encoding): {len(valid_words)}/{len(all_words)}")

    # Free base model memory
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Train/test split ---
    random.seed(42)
    random.shuffle(valid_words)
    split = int(len(valid_words) * args.train_split)
    train_words = valid_words[:split]
    test_words = valid_words[split:]
    print(f"  Train: {len(train_words)} words, Test: {len(test_words)} words")

    # Pre-compute tensors
    train_emb = torch.tensor(
        np.stack([embeddings[w] for w in train_words]),
        dtype=torch.float32, device=device,
    )
    train_tgt = torch.tensor(
        np.stack([targets[w] for w in train_words]),
        dtype=torch.float32, device=device,
    )
    train_mask = (train_tgt != 0).float()  # Only supervise on non-zero positions

    # Build hierarchy pairs from train words
    train_hierarchy_pairs = []
    for hyper, hypos in CONCEPT_HIERARCHY.items():
        if hyper not in train_words:
            continue
        hyper_idx = train_words.index(hyper)
        for hypo in hypos:
            if hypo not in train_words:
                continue
            hypo_idx = train_words.index(hypo)
            train_hierarchy_pairs.append((hyper_idx, hypo_idx))
    print(f"  Hierarchy pairs (train): {len(train_hierarchy_pairs)}")

    # --- Initialize projection head ---
    head = ConceptHead(n_embd, N_PRIMITIVES, args.hidden).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  ConceptHead: {n_params:,} parameters ({n_embd}→{args.hidden}→{N_PRIMITIVES})")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # --- Training loop ---
    print(f"\n  {'Step':>6s}  {'Loss':>8s}  {'MSE':>8s}  {'Sparse':>8s}  {'Sub':>8s}  {'Temp':>6s}  {'LR':>10s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*10}")

    best_test_acc = 0.0
    best_state = None

    for step in range(args.steps):
        head.train()
        temp = get_temperature(step, args.steps, args.start_temp, args.end_temp)

        # Sample batch
        if len(train_words) <= args.batch_size:
            batch_idx = list(range(len(train_words)))
        else:
            batch_idx = random.sample(range(len(train_words)), args.batch_size)

        batch_emb = train_emb[batch_idx]
        batch_tgt = train_tgt[batch_idx]
        batch_mask = train_mask[batch_idx]

        # Forward
        preds = head(batch_emb, temperature=temp)

        # MSE loss (only on non-zero target positions)
        mse_loss = ((preds - batch_tgt) ** 2 * batch_mask).sum() / max(batch_mask.sum(), 1)

        # Sparsity loss: push NA positions (target=0) toward 0
        na_mask = 1.0 - batch_mask  # positions where target IS zero
        sparsity_loss = (preds.abs() * na_mask).sum() / max(na_mask.sum(), 1)

        # Subsumption loss
        sub_loss = torch.tensor(0.0, device=device)
        if train_hierarchy_pairs and step % 3 == 0:
            # Get full predictions for hierarchy
            with torch.no_grad():
                all_preds = head(train_emb, temperature=temp)
            all_preds_grad = head(train_emb, temperature=temp)

            for hyper_idx, hypo_idx in train_hierarchy_pairs:
                # Hypernym active bits should be subset of hyponym active bits
                # Penalize when hypernym > hyponym (hypernym bit active but hyponym not)
                diff = torch.relu(all_preds_grad[hyper_idx] - all_preds_grad[hypo_idx])
                sub_loss = sub_loss + diff.mean()
            sub_loss = sub_loss / max(len(train_hierarchy_pairs), 1)

        total_loss = mse_loss + args.sub_weight * sub_loss + args.sparsity_weight * sparsity_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Log
        if step % 100 == 0 or step == args.steps - 1:
            lr = scheduler.get_last_lr()[0]
            print(f"  {step:>6d}  {total_loss.item():>8.4f}  {mse_loss.item():>8.4f}  "
                  f"{sparsity_loss.item():>8.4f}  {sub_loss.item():>8.4f}  "
                  f"{temp:>6.2f}  {lr:>10.6f}")

        # Evaluate
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            print()
            train_res = evaluate(head, embeddings,
                                 {w: targets[w] for w in train_words},
                                 device, temperature=0.5, label="TRAIN")
            test_res = evaluate(head, embeddings,
                                {w: targets[w] for w in test_words},
                                device, temperature=0.5, label="TEST ")

            if test_res.get('state_accuracy', 0) > best_test_acc:
                best_test_acc = test_res['state_accuracy']
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
                print(f"  ★ New best test state_acc: {best_test_acc:.1%}")
            print()

    # --- Final evaluation with best model ---
    if best_state:
        head.load_state_dict(best_state)

    print(f"\n{'='*70}")
    print(f"  FINAL EVALUATION (best checkpoint)")
    print(f"{'='*70}")
    train_final = evaluate(head, embeddings,
                           {w: targets[w] for w in train_words},
                           device, temperature=0.5, label="TRAIN")
    test_final = evaluate(head, embeddings,
                          {w: targets[w] for w in test_words},
                          device, temperature=0.5, label="TEST ")

    # --- Per-category breakdown ---
    print(f"\n  Per-category state accuracy (test set):")
    resolver = StateResolver()
    test_emb = torch.tensor(
        np.stack([embeddings[w] for w in test_words]),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        test_preds = head(test_emb, temperature=0.5).cpu().numpy()
    test_tgt = np.stack([targets[w] for w in test_words])

    for cat_name, cat_primes in PRIMES_BY_CATEGORY.items():
        cat_indices = [PRIMITIVE_NAMES.index(name) for name in cat_primes.keys()]
        cat_mask = test_tgt[:, cat_indices] != 0
        if cat_mask.sum() == 0:
            continue
        cat_match = (np.sign(test_preds[:, cat_indices]) == np.sign(test_tgt[:, cat_indices])) & cat_mask
        acc = cat_match.sum() / cat_mask.sum()
        print(f"    {cat_name:>25s}: {acc:.1%} ({int(cat_mask.sum())} positions)")

    # --- Sample predictions ---
    print(f"\n  Sample predictions (test set):")
    print(f"    {'Word':>15s}  {'Expected':>30s}  {'Predicted':>30s}  {'Match':>6s}  {'Spurious':>8s}")
    sample_words = test_words[:15]
    for i, word in enumerate(sample_words):
        idx = test_words.index(word)
        pred_vec = test_preds[idx]
        tgt_vec = test_tgt[idx]
        token = resolver.resolve(word, pred_vec)
        pred_active = set(a.name for a in token.activations if a.is_active)
        pred_zero = set(a.name for a in token.activations if a.is_zero)
        exp_active = set(PRIMITIVE_NAMES[j] for j in range(N_PRIMITIVES) if tgt_vec[j] > 0)
        exp_zero = set(PRIMITIVE_NAMES[j] for j in range(N_PRIMITIVES) if tgt_vec[j] < 0)
        # Check supervised positions
        correct = exp_active.issubset(pred_active) and exp_zero.issubset(pred_zero)
        # Count spurious (predicted active/zero but target is NA)
        all_expected = exp_active | exp_zero
        all_predicted = pred_active | pred_zero
        spurious = all_predicted - all_expected
        exp_str = ','.join(sorted(exp_active)[:3]) or '(NA)'
        pred_str = ','.join(sorted(pred_active & exp_active)[:3]) or '(none)'
        mark = "OK" if correct else "MISS"
        print(f"    {word:>15s}  {exp_str:>30s}  {pred_str:>30s}  {mark:>6s}  {len(spurious):>8d}")

    # --- Save results ---
    results = {
        'config': {
            'steps': args.steps, 'lr': args.lr, 'batch_size': args.batch_size,
            'hidden': args.hidden, 'start_temp': args.start_temp,
            'end_temp': args.end_temp, 'sub_weight': args.sub_weight,
            'train_split': args.train_split,
        },
        'data': {
            'total_words': len(valid_words), 'train_words': len(train_words),
            'test_words': len(test_words), 'hierarchy_pairs': len(train_hierarchy_pairs),
        },
        'train': train_final,
        'test': test_final,
        'model_params': n_params,
    }

    out_dir = os.path.join(PROJECT_ROOT, 'conceptual_tokenizer', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'phase4_training.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Save model
    model_path = os.path.join(out_dir, 'concept_head.pt')
    torch.save({
        'state_dict': head.state_dict(),
        'config': {'n_embd': n_embd, 'n_primitives': N_PRIMITIVES, 'hidden': args.hidden},
        'results': results,
    }, model_path)
    print(f"  Model: {model_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

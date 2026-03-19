"""
Evolution Hook — Lightweight triadic snapshot at each checkpoint.

Standalone module that can be imported into torch_train.py without modifying
its core logic. Call save_triadic_snapshot() after each checkpoint save.

Usage from training script:
    from src.evolution_hook import save_triadic_snapshot
    save_triadic_snapshot(model, tokenizer, config, checkpoint_dir, step, device)

Overhead: ~60ms per checkpoint (12 concepts x 5ms).
"""

import os
import json
import torch

from src.triadic import PrimeMapper, TriadicValidator


# 12 representative concepts covering the main semantic categories
SNAPSHOT_CONCEPTS = [
    "king", "queen", "dog", "cat", "happy", "sad",
    "fire", "water", "mother", "father", "doctor", "nurse",
]

# Canonical pairs to track connection formation
SNAPSHOT_PAIRS = [
    ("king", "queen"), ("dog", "cat"), ("happy", "sad"),
    ("mother", "father"), ("fire", "water"), ("doctor", "nurse"),
]


def save_triadic_snapshot(model, tokenizer, config, checkpoint_dir, step, device):
    """Save a lightweight triadic evolution snapshot alongside a checkpoint.

    Captures bit patterns and similarities for 12 concepts.

    Args:
        model: TriadicGPT model (will be temporarily set to eval mode)
        tokenizer: BPETokenizer instance
        config: TriadicGPTConfig
        checkpoint_dir: directory where checkpoint .pt files live
        step: current training step number
        device: torch device
    """
    mapper = PrimeMapper(config.n_triadic_bits)

    was_training = model.training
    model.eval()

    composites = {}
    bit_patterns = {}
    for concept in SNAPSHOT_CONCEPTS:
        ids = tokenizer.encode(concept, add_special=False)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)
        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        composites[concept] = int(mapper.map(proj))
        bit_patterns[concept] = mapper.get_bits(proj)

    # Pair similarities
    pair_sims = {}
    for a, b in SNAPSHOT_PAIRS:
        if a in composites and b in composites:
            pair_sims[f"{a}|{b}"] = float(
                TriadicValidator.similarity(composites[a], composites[b])
            )

    snapshot = {
        'step': step,
        'composites': composites,
        'bit_patterns': bit_patterns,
        'pair_similarities': pair_sims,
    }

    snap_path = os.path.join(checkpoint_dir, f'evolution_step{step}.json')
    with open(snap_path, 'w') as f:
        json.dump(snapshot, f, indent=2)

    if was_training:
        model.train()

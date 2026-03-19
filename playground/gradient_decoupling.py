"""
D-A14: Gradient Decoupling Analysis — Empirical Validation of Wang et al.

Trains DanzaTriadicGPT with gradient instrumentation to track:
  1. Per-bit gradient norms over training
  2. Gradient correlation matrix (off-diagonal → 0 = decoupling)
  3. Per-bit variance trajectory (lock-in detection)
  4. Lock-in time vs final entropy correlation

Validates Wang et al. (NeuS 2025) prediction that gradient flow over
odd activations (tanh) naturally decouples into per-bit optimization.

Usage:
  python playground/gradient_decoupling.py --scale xl --steps 50000        # full (~5h)
  python playground/gradient_decoupling.py --scale base --steps 5000       # fast test
  python playground/gradient_decoupling.py --analyze-only --results-dir ... # plots only
"""

import os
import sys
import csv
import json
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_PLAYGROUND = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

from danza_63bit import (
    load_primitives, load_anchors, build_subsumption_pairs,
    DanzaTriadicGPT, supervised_anchor_loss, subsumption_loss,
    triadic_loss, evaluate_anchors, evaluate_subsumption,
    evaluate_regla_de_tres, REGLA_DE_TRES_QUADS, TextDataset,
    ANCHOR_TRANSLATIONS, SKIP_ANCHORS, N_BITS, STORY_SEPARATOR,
)
from src.torch_transformer import TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer


# ============================================================
# Gradient Tracking
# ============================================================

class GradientTracker:
    """Tracks per-bit gradient statistics during training."""

    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.grad_norms = []       # (step, [n_bits])
        self.grad_corr = []        # (step, [n_bits, n_bits])
        self.bit_variance = []     # (step, [n_bits])
        self.bit_means = []        # (step, [n_bits]) — running mean of projections

    def track(self, model, step, last_proj=None):
        """Record gradient statistics from the triadic head.

        Args:
            model: DanzaTriadicGPT (must have .triadic_head.weight.grad)
            step: current training step
            last_proj: (B, T, K) tensor from last forward pass
        """
        head = model.triadic_head
        if head.weight.grad is None:
            return

        grad = head.weight.grad.detach()  # (K, D)

        # 1. Per-bit gradient norm
        norms = torch.norm(grad, dim=1).float().cpu().numpy()  # (K,)
        self.grad_norms.append((step, norms.copy()))

        # 2. Gradient correlation matrix
        grad_centered = grad - grad.mean(dim=1, keepdim=True)
        norms_for_corr = torch.norm(grad_centered, dim=1, keepdim=True).clamp(min=1e-8)
        grad_normed = grad_centered / norms_for_corr
        corr = (grad_normed @ grad_normed.T).float().cpu().numpy()  # (K, K)
        self.grad_corr.append((step, corr.copy()))

        # 3. Per-bit variance and mean of projections
        if last_proj is not None:
            proj_flat = last_proj.detach().reshape(-1, self.n_bits)  # (B*T, K)
            bit_var = proj_flat.var(dim=0).float().cpu().numpy()
            bit_mean = proj_flat.mean(dim=0).float().cpu().numpy()
            self.bit_variance.append((step, bit_var.copy()))
            self.bit_means.append((step, bit_mean.copy()))

    def analyze(self):
        """Post-training analysis: lock-in times, decoupling rate, etc."""
        results = {}

        if not self.grad_norms:
            return results

        steps = [s for s, _ in self.grad_norms]
        norms = np.array([n for _, n in self.grad_norms])  # (T, K)

        # --- Gradient decoupling: off-diagonal correlation over time ---
        if self.grad_corr:
            off_diag_means = []
            for step, corr in self.grad_corr:
                mask = ~np.eye(self.n_bits, dtype=bool)
                off_diag = np.abs(corr[mask]).mean()
                off_diag_means.append((step, off_diag))

            results['decoupling'] = {
                'start_correlation': off_diag_means[0][1],
                'end_correlation': off_diag_means[-1][1],
                'trajectory': [(int(s), float(v)) for s, v in off_diag_means],
            }

        # --- Lock-in times: when does each bit's gradient norm plateau? ---
        peak_norms = norms.max(axis=0)  # peak norm per bit
        lock_threshold = peak_norms * 0.05  # 5% of peak = "locked"

        lock_in_times = {}
        for bit_i in range(self.n_bits):
            # Find first step where norm drops below threshold and stays there
            locked = False
            for t_idx in range(len(steps) - 1, -1, -1):
                if norms[t_idx, bit_i] > lock_threshold[bit_i]:
                    if t_idx < len(steps) - 1:
                        lock_in_times[bit_i] = steps[t_idx + 1]
                    else:
                        lock_in_times[bit_i] = steps[-1]  # never locked
                    locked = True
                    break
            if not locked:
                lock_in_times[bit_i] = steps[0]  # locked from the start

        results['lock_in_times'] = {int(k): int(v) for k, v in lock_in_times.items()}

        # Distribution
        max_step = steps[-1]
        quarters = [max_step * i / 4 for i in range(1, 5)]
        dist = {'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0}
        for bit_i, t in lock_in_times.items():
            if t <= quarters[0]:
                dist['q1'] += 1
            elif t <= quarters[1]:
                dist['q2'] += 1
            elif t <= quarters[2]:
                dist['q3'] += 1
            else:
                dist['q4'] += 1
        results['lock_in_distribution'] = dist

        # --- Lock-in time vs final entropy correlation ---
        if self.bit_means:
            final_means = self.bit_means[-1][1]
            q = (final_means + 1) / 2
            eps = 1e-7
            q = np.clip(q, eps, 1 - eps)
            final_entropy = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))

            lock_times_array = np.array([lock_in_times[i] for i in range(self.n_bits)])

            # Pearson correlation
            if np.std(lock_times_array) > 0 and np.std(final_entropy) > 0:
                corr_val = np.corrcoef(lock_times_array, final_entropy)[0, 1]
            else:
                corr_val = 0.0

            results['locktime_entropy_correlation'] = float(corr_val)
            results['per_bit_entropy'] = final_entropy.tolist()

            # Dead vs alive analysis
            dead_mask = final_entropy < 0.3
            alive_mask = ~dead_mask
            if dead_mask.sum() > 0:
                results['dead_bits_mean_locktime'] = float(lock_times_array[dead_mask].mean())
            if alive_mask.sum() > 0:
                results['alive_bits_mean_locktime'] = float(lock_times_array[alive_mask].mean())
            results['n_dead'] = int(dead_mask.sum())
            results['n_alive'] = int(alive_mask.sum())

        # --- Per-bit gradient norm trajectories (for plotting) ---
        results['grad_norm_trajectories'] = {
            'steps': [int(s) for s in steps],
            'norms': norms.tolist(),  # (T, K) → list of lists
        }

        return results

    def save(self, path):
        """Save raw tracking data for later analysis."""
        data = {
            'grad_norms': [(int(s), n.tolist()) for s, n in self.grad_norms],
            'grad_corr': [(int(s), c.tolist()) for s, c in self.grad_corr],
            'bit_variance': [(int(s), v.tolist()) for s, v in self.bit_variance],
            'bit_means': [(int(s), m.tolist()) for s, m in self.bit_means],
        }
        with open(path, 'w') as f:
            json.dump(data, f)


# ============================================================
# Plotting
# ============================================================

def generate_plots(results, output_dir):
    """Generate analysis figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Gradient norm trajectories (example bits)
    traj = results.get('grad_norm_trajectories', {})
    if traj:
        steps = traj['steps']
        norms = np.array(traj['norms'])

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # Pick 8 representative bits: 4 with highest final entropy, 4 with lowest
        entropies = np.array(results.get('per_bit_entropy', [0.5] * N_BITS))
        sorted_bits = np.argsort(entropies)
        dead_bits = sorted_bits[:4]
        alive_bits = sorted_bits[-4:]

        for b in dead_bits:
            ax.plot(steps, norms[:, b], alpha=0.7, linestyle='--',
                    label=f'bit {b} (dead, H={entropies[b]:.2f})')
        for b in alive_bits:
            ax.plot(steps, norms[:, b], alpha=0.9, linewidth=2,
                    label=f'bit {b} (alive, H={entropies[b]:.2f})')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Per-Bit Gradient Norm Trajectories')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'gradient_norm_trajectories.png'), dpi=150)
        plt.close()

    # 2. Gradient correlation decay
    decoupling = results.get('decoupling', {})
    if decoupling.get('trajectory'):
        traj_data = decoupling['trajectory']
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot([s for s, _ in traj_data], [v for _, v in traj_data],
                'b-o', markersize=3)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Mean Off-Diagonal |Correlation|')
        ax.set_title('Gradient Decoupling Over Training')
        ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Decoupled threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'correlation_decay.png'), dpi=150)
        plt.close()

    # 3. Lock-in time vs entropy scatter
    per_bit_entropy = results.get('per_bit_entropy')
    lock_times = results.get('lock_in_times')
    if per_bit_entropy and lock_times:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ent = np.array(per_bit_entropy)
        lt = np.array([lock_times[str(i)] if str(i) in lock_times else lock_times.get(i, 0)
                        for i in range(N_BITS)])

        colors = ['red' if e < 0.3 else 'green' for e in ent]
        ax.scatter(lt, ent, c=colors, alpha=0.7, s=40)
        ax.set_xlabel('Lock-in Time (step)')
        ax.set_ylabel('Final Entropy')
        ax.set_title(f'Lock-in Time vs Entropy (r={results.get("locktime_entropy_correlation", 0):.2f})')
        ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Dead threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'locktime_vs_entropy.png'), dpi=150)
        plt.close()

    print(f"  Plots saved to {output_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='D-A14: Gradient Decoupling Analysis')
    parser.add_argument('--scale', choices=['base', 'xl', 'xxl'], default='base')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--sub-weight', type=float, default=5.0)
    parser.add_argument('--sup-weight', type=float, default=2.0)
    parser.add_argument('--align-weight', type=float, default=3.0)
    parser.add_argument('--triadic-warmup-pct', type=float, default=0.5)
    parser.add_argument('--tracking-interval', type=int, default=500,
                        help='Track gradients every N steps')
    parser.add_argument('--stories', type=int, default=50000)
    parser.add_argument('--vocab', type=int, default=4096)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--grad-checkpoint', action='store_true')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'],
                        default='bfloat16')
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=10000)
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--analyze-only', action='store_true',
                        help='Skip training, just analyze existing results')
    parser.add_argument('--results-dir', type=str, default=None)
    args = parser.parse_args()

    SCALES = {
        'base': {'layers': 6,  'dim': 256,  'heads': 8},
        'xl':   {'layers': 12, 'dim': 512,  'heads': 8},
        'xxl':  {'layers': 24, 'dim': 1024, 'heads': 16},
    }
    preset = SCALES[args.scale]
    ckpt_dir = os.path.join(_PROJECT, 'checkpoints', f'danza_grad_decoupling_{args.scale}')

    # --- Analyze-only mode ---
    if args.analyze_only:
        rdir = args.results_dir or ckpt_dir
        raw_path = os.path.join(rdir, 'gradient_tracking_raw.json')
        if not os.path.exists(raw_path):
            print(f"No tracking data found at {raw_path}")
            return

        print(f"Loading tracking data from {raw_path}...")
        tracker = GradientTracker(N_BITS)
        with open(raw_path) as f:
            raw = json.load(f)
        tracker.grad_norms = [(s, np.array(n)) for s, n in raw['grad_norms']]
        tracker.grad_corr = [(s, np.array(c)) for s, c in raw['grad_corr']]
        tracker.bit_variance = [(s, np.array(v)) for s, v in raw['bit_variance']]
        tracker.bit_means = [(s, np.array(m)) for s, m in raw['bit_means']]

        results = tracker.analyze()
        with open(os.path.join(rdir, 'decoupling_analysis.json'), 'w') as f:
            # Filter out large arrays for the summary
            summary = {k: v for k, v in results.items() if k != 'grad_norm_trajectories'}
            json.dump(summary, f, indent=2)

        generate_plots(results, os.path.join(rdir, 'figures'))
        print("Analysis complete.")
        return

    # --- Full training mode ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(ckpt_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("  D-A14: GRADIENT DECOUPLING ANALYSIS (Wang et al. Validation)")
    print("=" * 70)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    # --- Load data ---
    prim_data = load_primitives()
    anchors, _ = load_anchors(prim_data)
    train_sub, test_sub = build_subsumption_pairs(anchors, prim_data)

    data_path = os.path.join(_PROJECT, 'data', 'TinyStories-train.txt')
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR)
               if s.strip() and len(s.strip()) > 30]
    if args.stories and len(stories) > args.stories:
        random.seed(42)
        random.shuffle(stories)
        stories = stories[:args.stories]

    tokenizer = BPETokenizer(vocab_size=args.vocab)
    tokenizer.train(stories, verbose=False)
    tokenizer.save(os.path.join(ckpt_dir, 'tokenizer.json'))

    all_tokens = []
    for story in stories:
        all_tokens.extend(tokenizer.encode(story, add_special=True))
    print(f"  Tokens: {len(all_tokens):,}")

    # --- Model (standard DanzaTriadicGPT, NO torch.compile for grad tracking) ---
    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block,
        n_layer=preset['layers'],
        n_embd=preset['dim'],
        n_head=preset['heads'],
        n_triadic_bits=N_BITS,
        dropout=args.dropout,
    )
    model = DanzaTriadicGPT(config).to(device)
    total_params = model.num_params()
    print(f"  Model: {args.scale} ({total_params/1e6:.1f}M params)")
    print(f"  NOTE: torch.compile DISABLED (need raw gradients for tracking)")
    print(f"  Tracking interval: every {args.tracking_interval} steps")

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    use_amp = device.type == 'cuda'
    amp_dtype = {'float32': torch.float32, 'float16': torch.float16,
                 'bfloat16': torch.bfloat16}[args.dtype]
    print(f"  Precision: {args.dtype}")

    # --- Pre-encode anchors ---
    word_list, ids_list, target_list = [], [], []
    for word, data in anchors.items():
        ids = tokenizer.encode(word, add_special=False)[:4]
        if ids:
            word_list.append(word)
            ids_list.append(ids)
            target_list.append(data['target'])

    max_len = max(len(x) for x in ids_list)
    word_tensors = torch.tensor(
        [ids + [0] * (max_len - len(ids)) for ids in ids_list],
        dtype=torch.long, device=device
    )
    target_vectors = torch.stack(target_list).to(device)

    if train_sub:
        hyper_t = torch.tensor(
            [[tokenizer.encode(h, add_special=False)[0]] for h, _, _, _ in train_sub],
            dtype=torch.long, device=device
        )
        hypo_t = torch.tensor(
            [[tokenizer.encode(y, add_special=False)[0]] for _, y, _, _ in train_sub],
            dtype=torch.long, device=device
        )
    else:
        z = torch.zeros((0, 1), dtype=torch.long, device=device)
        hyper_t, hypo_t = z, z

    # --- Training with gradient tracking ---
    print(f"\n  Training ({args.steps} steps, tracking every {args.tracking_interval})...")

    tracker = GradientTracker(N_BITS)
    dataset = TextDataset(all_tokens, args.block)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    loader_iter = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup_steps = int(args.steps * args.triadic_warmup_pct)

    log_path = os.path.join(ckpt_dir, 'training_log.csv')
    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['step', 'total_loss', 'lang_loss', 'tri_loss', 'ppl', 'lr'])

    start_time = time.time()
    last_proj = None

    for step in range(1, args.steps + 1):
        # LR schedule
        if step <= warmup_steps:
            lr_mult = step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr * lr_mult

        triadic_active = step > warmup_steps

        try:
            x_batch, y_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x_batch, y_batch = next(loader_iter)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            logits, triadic_proj, lang_loss = model(x_batch, targets=y_batch)

            if triadic_active:
                l_tri = triadic_loss(triadic_proj, args.align_weight, model.wte, x_batch)
                l_sup = supervised_anchor_loss(model, word_tensors, target_vectors)
                l_sub = subsumption_loss(model, hyper_t, hypo_t)
                total_loss = (lang_loss
                              + args.alpha * l_tri
                              + args.alpha * args.sup_weight * l_sup
                              + args.alpha * args.sub_weight * l_sub)
            else:
                l_tri = torch.tensor(0.0)
                total_loss = lang_loss

        total_loss.backward()

        # --- GRADIENT TRACKING (after backward, before optimizer.step) ---
        if step % args.tracking_interval == 0:
            tracker.track(model, step, last_proj=triadic_proj)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        last_proj = triadic_proj.detach()

        # Logging
        if step % args.print_every == 0 or step == 1:
            elapsed = time.time() - start_time
            speed = step / elapsed
            eta = (args.steps - step) / speed
            ppl = math.exp(min(lang_loss.item(), 20))
            pct = step / args.steps
            bar = '#' * int(30 * pct) + '-' * (30 - int(30 * pct))

            tracked = len(tracker.grad_norms)
            print(f"  [{bar}] {step:>6}/{args.steps} | "
                  f"loss {total_loss.item():.4f} | lang {lang_loss.item():.4f} | "
                  f"tri {l_tri.item():.3f} | ppl {ppl:.1f} | "
                  f"{speed:.1f} it/s | ETA {eta:.0f}s | tracked {tracked}")

            log_writer.writerow([step, total_loss.item(), lang_loss.item(),
                                 l_tri.item(), ppl, args.lr * lr_mult])

        # Eval
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_anchors(model, word_tensors, target_vectors, word_list)
            print(f"\n  --- Eval @ {step} --- bit_acc={metrics.get('mean_bit_accuracy',0):.1%}, "
                  f"dead={metrics.get('dead_bits',0)}/{N_BITS}")

        # Checkpoint
        if step % args.save_every == 0 or step == args.steps:
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step,
                'config': vars(config),
            }, os.path.join(ckpt_dir, f'model_step{step}.pt'))

    log_file.close()
    elapsed = time.time() - start_time

    # --- Save raw tracking data ---
    raw_path = os.path.join(ckpt_dir, 'gradient_tracking_raw.json')
    print(f"\n  Saving raw gradient data ({len(tracker.grad_norms)} snapshots)...")
    tracker.save(raw_path)

    # --- Analysis ---
    print(f"  Running decoupling analysis...")
    results = tracker.analyze()

    summary_path = os.path.join(ckpt_dir, 'decoupling_analysis.json')
    summary = {k: v for k, v in results.items() if k != 'grad_norm_trajectories'}
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # --- Plots ---
    generate_plots(results, os.path.join(ckpt_dir, 'figures'))

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(f"  D-A14 GRADIENT DECOUPLING — COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")
    dec = results.get('decoupling', {})
    print(f"  Correlation start:  {dec.get('start_correlation', 'N/A'):.3f}")
    print(f"  Correlation end:    {dec.get('end_correlation', 'N/A'):.3f}")
    print(f"  Lock-in dist:       {results.get('lock_in_distribution', {})}")
    print(f"  Locktime↔entropy r: {results.get('locktime_entropy_correlation', 'N/A'):.3f}")
    print(f"  Dead bits:          {results.get('n_dead', '?')}")
    print(f"  Alive bits:         {results.get('n_alive', '?')}")
    if 'dead_bits_mean_locktime' in results:
        print(f"  Dead mean locktime: step {results['dead_bits_mean_locktime']:.0f}")
    if 'alive_bits_mean_locktime' in results:
        print(f"  Alive mean locktime:step {results['alive_bits_mean_locktime']:.0f}")
    print(f"\n  Checkpoints: {ckpt_dir}")
    print(f"  Raw data:    {raw_path}")
    print(f"  Analysis:    {summary_path}")
    print(f"  Figures:     {ckpt_dir}/figures/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

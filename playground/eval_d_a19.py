"""D-A19 Formal Evaluation — GPT-2 Medium 355M with fixed algebra."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'playground'))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

from playground.gpt2_355m_sparsity import GPT2MediumSparsity
from playground.danza_63bit import (
    load_primitives, load_all_anchors, N_BITS,
    REGLA_DE_TRES_QUADS,
)
from playground.danza_bootstrap import (
    build_partial_subsumption_pairs, phase_split,
)
from src.triadic import BitwiseMapper, BitwiseValidator

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'checkpoints', 'danza_gpt2_355m_sparsity_v2')
CKPT = os.path.join(CKPT_DIR, 'model_best.pt')


def load_model(device='cuda'):
    """Load GPT-2 Medium + triadic head from D-A19 checkpoint."""
    state = torch.load(CKPT, map_location=device, weights_only=False)
    qmode = state.get('quantize_mode', 'fsq')

    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2MediumSparsity(gpt2, n_triadic_bits=N_BITS, quantize_mode=qmode)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()

    return model, tokenizer, state


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, ckpt_state = load_model(device)

    prim_data = load_primitives()
    all_anchors, _ = load_all_anchors(prim_data)
    train_anchors, holdout_anchors = phase_split(all_anchors, prim_data)
    mapper = BitwiseMapper(N_BITS)
    validator = BitwiseValidator()

    print('=' * 70)
    print('  D-A19 FORMAL EVAL — GPT-2 Medium 355M (Fixed Algebra)')
    print('=' * 70)
    print(f'  Checkpoint: {os.path.basename(CKPT)}')
    print(f'  Saved at step: {ckpt_state.get("step", "?")}')
    print(f'  Anchors: {len(all_anchors)} ({len(train_anchors)} train, {len(holdout_anchors)} holdout)')
    print(f'  Bits: {N_BITS}')
    print(f'  Device: {device}')

    # Extract projections for all anchors
    projections = {}
    for word, data in all_anchors.items():
        ids = tokenizer.encode(word, add_special_tokens=False)[:8]
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            result = model(x)
        proj_vec = result[1][0].mean(dim=0).cpu().numpy()
        bits = mapper.get_bits(proj_vec)
        projections[word] = {
            'bits': bits,
            'target': data['target'].numpy().tolist(),
            'expanded': data['expanded'],
            'proj': proj_vec,
        }

    # [1] Bit accuracy (supervised + holdout)
    print('\n  [1] BIT ACCURACY')
    train_words = [w for w in train_anchors if w in projections]
    hold_words = [w for w in holdout_anchors if w in projections]
    all_words = [w for w in all_anchors if w in projections]

    for split, words in [('Train', train_words), ('Holdout', hold_words), ('All', all_words)]:
        correct = total = 0
        for w in words:
            bits = projections[w]['bits']
            target = projections[w]['target']
            for i in range(N_BITS):
                pred = 1 if bits[i] == 1 else -1
                if pred == target[i]:
                    correct += 1
                total += 1
        acc = correct / total if total else 0
        print(f'    {split}: {acc:.1%} ({correct}/{total})')

    # [2] Subsumption (BitwiseValidator)
    print('\n  [2] SUBSUMPTION (BitwiseValidator)')
    sub_pass = sub_total = 0
    sub_fail = []
    items = [(w, d) for w, d in all_anchors.items() if w in projections]
    for i, (wa, da) in enumerate(items):
        bits_a = set(da['expanded'])
        for j, (wb, db) in enumerate(items):
            if i == j:
                continue
            bits_b = set(db['expanded'])
            if bits_a < bits_b:
                pred_a = int(mapper.map(projections[wa]['proj'].tolist()))
                pred_b = int(mapper.map(projections[wb]['proj'].tolist()))
                sub_total += 1
                if (pred_a & pred_b) == pred_a:
                    sub_pass += 1
                else:
                    sub_fail.append(f'{wa}->{wb}')
    sub_rate = sub_pass / sub_total if sub_total else 0
    print(f'    Rate: {sub_rate:.1%} ({sub_pass}/{sub_total})')
    if sub_fail[:5]:
        print(f'    Failures (first 5): {sub_fail[:5]}')

    # [3] Ternary distribution
    print('\n  [3] TERNARY DISTRIBUTION')
    all_bits = []
    for data in projections.values():
        all_bits.extend(data['bits'])
    all_bits = np.array(all_bits)
    n_pos = (all_bits == 1).sum()
    n_neg = (all_bits == -1).sum()
    n_zero = (all_bits == 0).sum()
    total_b = len(all_bits)
    print(f'    +1: {n_pos/total_b:.1%}  0: {n_zero/total_b:.1%}  -1: {n_neg/total_b:.1%}')

    # [4] Dead bits
    print('\n  [4] DEAD BITS')
    bit_activity = np.zeros(N_BITS)
    for data in projections.values():
        for i, b in enumerate(data['bits']):
            if b != 0:
                bit_activity[i] += 1
    n_concepts = len(projections)
    dead = sum(1 for a in bit_activity if a / n_concepts < 0.02)
    always_on = sum(1 for a in bit_activity if a / n_concepts > 0.98)
    print(f'    Dead (<2%): {dead}')
    print(f'    Always-on (>98%): {always_on}')
    print(f'    Informative: {N_BITS - dead - always_on}')

    # [5] Signature uniqueness
    print('\n  [5] SIGNATURE UNIQUENESS')
    sigs = set(tuple(d['bits']) for d in projections.values())
    print(f'    Unique: {len(sigs)}/{len(projections)} ({len(sigs)/len(projections):.1%})')

    # [6] Regla de Tres (analogies)
    print('\n  [6] REGLA DE TRES (ANALOGIES)')

    def get_proj(word):
        ids = tokenizer.encode(word, add_special_tokens=False)[:8]
        if not ids:
            return None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            proj = model(x)[1]
        return proj[0].mean(dim=0)

    r3_results = []
    for a_w, b_w, c_w, d_w in REGLA_DE_TRES_QUADS:
        if not all(w in all_anchors for w in [a_w, b_w, c_w, d_w]):
            continue
        pa, pb, pc, pd = get_proj(a_w), get_proj(b_w), get_proj(c_w), get_proj(d_w)
        if any(p is None for p in [pa, pb, pc, pd]):
            continue
        predicted_d = pc + (pb - pa)
        cos = F.cosine_similarity(predicted_d.unsqueeze(0), pd.unsqueeze(0)).item()
        pred_bits = (predicted_d > 0).long()
        actual_bits = (pd > 0).long()
        bit_match = (pred_bits == actual_bits).float().mean().item()
        r3_results.append({
            'quad': f'{a_w}:{b_w}={c_w}:{d_w}',
            'cosine': cos,
            'bit_accuracy': bit_match,
        })

    r3_pass = sum(1 for r in r3_results if r['bit_accuracy'] > 0.80)
    r3_total = len(r3_results)
    print(f'    Pass (>80% bit match): {r3_pass}/{r3_total} ({r3_pass/r3_total:.1%})')
    for r in r3_results:
        status = 'PASS' if r['bit_accuracy'] > 0.80 else 'FAIL'
        print(f'    [{status}] {r["quad"]:40s} cos={r["cosine"]:.3f} bits={r["bit_accuracy"]:.1%}')

    # [7] Comparison table
    print(f'\n  [7] COMPARISON')
    zr = f'{n_zero/total_b:.1%}'
    print(f'  {"Metric":<25} {"D-A19 (355M fix)":>18} {"D-A17 (355M bug)":>18} {"D-A14 (40M)":>14}')
    print('  ' + '-' * 77)

    # Compute holdout bit accuracy for the table
    h_correct = h_total = 0
    for w in hold_words:
        bits = projections[w]['bits']
        target = projections[w]['target']
        for i in range(N_BITS):
            pred = 1 if bits[i] == 1 else -1
            if pred == target[i]:
                h_correct += 1
            h_total += 1
    hold_acc = h_correct / h_total if h_total else 0

    print(f'  {"Holdout bit accuracy":<25} {f"{hold_acc:.1%}":>18} {"97.7%":>18} {"93.0%":>14}')
    print(f'  {"Subsumption":<25} {f"{sub_rate:.1%}":>18} {"1.7%":>18} {"98.3%":>14}')
    print(f'  {"Dead bits":<25} {f"{dead}/63":>18} {"26/63":>18} {"26/63":>14}')
    print(f'  {"Always-on":<25} {f"{always_on}/63":>18} {"--":>18} {"--":>14}')
    print(f'  {"Zero rate":<25} {zr:>18} {"3.4%":>18} {"~42%":>14}')
    print(f'  {"R3 analogies":<25} {f"{r3_pass}/{r3_total}":>18} {"6.7%":>18} {"100%":>14}')
    print(f'  {"Unique sigs":<25} {f"{len(sigs)}/{len(projections)}":>18} {"--":>18} {"--":>14}')

    print('\n' + '=' * 70)
    if sub_rate >= 0.50 and hold_acc >= 0.88:
        print('  VERDICT: D-A19 PASS — algebra restored at 355M scale')
    elif sub_rate >= 0.50:
        print('  VERDICT: D-A19 PARTIAL PASS — subsumption fixed, accuracy borderline')
    else:
        print('  VERDICT: D-A19 FAIL — subsumption still broken')
    print('=' * 70)

    # Save results
    results = {
        'experiment': 'D-A19_formal_eval',
        'checkpoint': CKPT,
        'step': ckpt_state.get('step'),
        'n_anchors': len(all_anchors),
        'n_projected': len(projections),
        'holdout_bit_accuracy': hold_acc,
        'sub_rate': sub_rate,
        'sub_pass': sub_pass,
        'sub_total': sub_total,
        'sub_failures': sub_fail[:20],
        'zero_rate': float(n_zero / total_b),
        'dead_bits': dead,
        'always_on': always_on,
        'unique_sigs': len(sigs),
        'r3_pass': r3_pass,
        'r3_total': r3_total,
        'r3_details': r3_results,
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'audit_tests', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'f4_d_a19_eval.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'  Saved: {out}')


if __name__ == '__main__':
    main()

"""
Command-line interface for reptimeline.

Usage:
    python -m reptimeline --checkpoint-dir checkpoints/danza_63bit_xl_v2/ \\
                          --concepts king queen love hate fire water \\
                          --max-checkpoints 10

    python -m reptimeline --checkpoint-dir checkpoints/danza_63bit_xl_v2/ \\
                          --primitives \\
                          --max-checkpoints 10
"""

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from reptimeline.extractors.triadic import TriadicExtractor
from reptimeline.tracker import TimelineTracker


def _load_primitive_concepts():
    """Load the 63 primitive names (English) from primitivos.json + anclas."""
    prim_path = os.path.join(
        _PROJECT_ROOT, 'playground', 'danza_data', 'primitivos.json'
    )
    if not os.path.exists(prim_path):
        return None

    with open(prim_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # The primitive names are in Spanish — we need the English anchor names.
    # Load from anclas.json to get the English concept words for primitives.
    anclas_path = os.path.join(
        _PROJECT_ROOT, 'playground', 'danza_data', 'anclas.json'
    )
    if os.path.exists(anclas_path):
        with open(anclas_path, 'r', encoding='utf-8') as f:
            anclas = json.load(f)
        return list(anclas.keys())

    # Fallback: use primitive Spanish names
    return [p['nombre'] for p in data['primitivos']]


def main():
    parser = argparse.ArgumentParser(
        description='Representation Timeline — track discrete representation evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Track specific concepts
  python -m reptimeline --checkpoint-dir ckpts/ --concepts king queen love hate

  # Track all 63 primitive concepts (triadic-specific)
  python -m reptimeline --checkpoint-dir ckpts/ --primitives --overlay

  # Limit checkpoints for speed
  python -m reptimeline --checkpoint-dir ckpts/ --primitives --max-checkpoints 8
""",
    )

    parser.add_argument('--checkpoint-dir', required=True,
                        help='Directory containing model_*step*.pt checkpoints')
    parser.add_argument('--concepts', nargs='+',
                        help='Concepts to track (space-separated)')
    parser.add_argument('--primitives', action='store_true',
                        help='Track the 63 triadic primitive concepts')
    parser.add_argument('--overlay', action='store_true',
                        help='Run primitive overlay analysis (layer emergence, '
                             'dual coherence, deps)')
    parser.add_argument('--tokenizer', default=None,
                        help='Path to tokenizer.json (auto-detected if omitted)')
    parser.add_argument('--max-checkpoints', type=int, default=None,
                        help='Max number of checkpoints to process (evenly spaced)')
    parser.add_argument('--device', default='cpu',
                        help='Torch device (default: cpu)')
    parser.add_argument('--stability-window', type=int, default=3,
                        help='Consecutive stable snapshots to count as stabilized')
    parser.add_argument('--output', default=None,
                        help='Save Timeline JSON to this path')
    parser.add_argument('--n-bits', type=int, default=63,
                        help='Number of triadic bits (default: 63)')
    parser.add_argument('--max-tokens', type=int, default=4,
                        help='Max tokens per concept (4=custom BPE, 8=GPT-2)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--plot-dir', default=None,
                        help='Directory to save plots (default: checkpoint-dir/timeline_plots/)')

    args = parser.parse_args()

    # Determine concepts
    if args.primitives:
        concepts = _load_primitive_concepts()
        if concepts is None:
            print("ERROR: Could not load primitive concepts from primitivos.json")
            sys.exit(1)
        print(f"Tracking {len(concepts)} primitive concepts")
    elif args.concepts:
        concepts = args.concepts
    else:
        print("ERROR: Provide --concepts or --primitives")
        sys.exit(1)

    # Extract
    print(f"\nExtracting from: {args.checkpoint_dir}")
    print(f"Device: {args.device}")

    extractor = TriadicExtractor(
        tokenizer_path=args.tokenizer,
        n_bits=args.n_bits,
        max_tokens=args.max_tokens,
    )
    snapshots = extractor.extract_sequence(
        args.checkpoint_dir, concepts,
        device=args.device,
        max_checkpoints=args.max_checkpoints,
    )

    # Analyze
    print(f"\nAnalyzing {len(snapshots)} snapshots...")
    tracker = TimelineTracker(extractor, stability_window=args.stability_window)
    timeline = tracker.analyze(snapshots)
    timeline.print_summary()

    # Overlay
    report = None
    if args.overlay:
        from reptimeline.overlays import PrimitiveOverlay
        overlay = PrimitiveOverlay()
        report = overlay.analyze(timeline, concepts)
        overlay.print_report(report)

    # Plots
    if args.plot:
        plot_dir = args.plot_dir or os.path.join(args.checkpoint_dir, 'timeline_plots')
        os.makedirs(plot_dir, exist_ok=True)
        _generate_plots(timeline, report, concepts, plot_dir)
        print(f"\nPlots saved to {plot_dir}/")

    # Save
    if args.output:
        _save_timeline(timeline, args.output)
        print(f"\nTimeline saved to {args.output}")


def _generate_plots(timeline, report, concepts, plot_dir):
    """Generate all visualization plots."""
    from reptimeline.viz import (
        plot_swimlane, plot_phase_dashboard, plot_churn_heatmap, plot_layer_emergence,
    )

    print("\nGenerating plots...")

    plot_swimlane(
        timeline, concepts=concepts[:20],  # limit for readability
        save_path=os.path.join(plot_dir, 'swimlane.png'),
        show=False,
    )
    print("  swimlane.png")

    plot_phase_dashboard(
        timeline,
        save_path=os.path.join(plot_dir, 'phase_dashboard.png'),
        show=False,
    )
    print("  phase_dashboard.png")

    plot_churn_heatmap(
        timeline, concepts=concepts,
        save_path=os.path.join(plot_dir, 'churn_heatmap.png'),
        show=False,
    )
    print("  churn_heatmap.png")

    if report is not None:
        plot_layer_emergence(
            report,
            save_path=os.path.join(plot_dir, 'layer_emergence.png'),
            show=False,
        )
        print("  layer_emergence.png")


def _save_timeline(timeline, path):
    """Serialize Timeline to JSON."""
    data = {
        'steps': timeline.steps,
        'births': [
            {'step': e.step, 'concept': e.concept, 'code_index': e.code_index}
            for e in timeline.births
        ],
        'deaths': [
            {'step': e.step, 'concept': e.concept, 'code_index': e.code_index}
            for e in timeline.deaths
        ],
        'connections': [
            {'step': e.step, 'concept_a': e.concept_a,
             'concept_b': e.concept_b, 'shared_indices': e.shared_indices}
            for e in timeline.connections
        ],
        'phase_transitions': [
            {'step': pt.step, 'metric': pt.metric,
             'delta': pt.delta, 'direction': pt.direction}
            for pt in timeline.phase_transitions
        ],
        'curves': timeline.curves,
        'stability': {str(k): v for k, v in timeline.stability.items()},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()

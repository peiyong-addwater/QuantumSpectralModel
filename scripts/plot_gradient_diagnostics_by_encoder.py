#!/usr/bin/env python3
"""Plot merged gradient diagnostics into one output folder per encoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.naming import canonical_encoder_name  # noqa: E402
from scripts import plot_gradient_diagnostics as gradient_plots  # noqa: E402

DEFAULT_INPUT_DIR = gradient_plots.DEFAULT_INPUT_DIR
DEFAULT_OUTPUT_DIR = "results/figures/gradient_diagnostics_by_encoder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot", choices=gradient_plots.PLOT_CHOICES, default="all")
    gradient_plots.add_style_arguments(parser)
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=None,
        help="Optional encoder slugs to plot. Legacy aliases are canonicalized.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_styles:
        gradient_plots.print_available_styles()
        return
    records = gradient_plots.load_records(Path(args.input_dir))
    gradient_plots.use_style(args.style)

    encoders = select_encoders(records, args.encoders)
    written = plot_by_encoder(records, args.plot, Path(args.output_dir), encoders)
    if not written:
        raise ValueError(
            f"no figures were produced for --plot {args.plot!r} "
            f"and encoders {', '.join(encoders) or '<none>'}"
        )
    for path in written:
        print(path)


def select_encoders(
    records: list[gradient_plots.DiagnosticRecord],
    requested: list[str] | None,
) -> list[str]:
    """Return canonical encoder slugs to render, preserving a stable order."""

    if requested is None:
        return gradient_plots.ordered_encoders(records)
    selected: list[str] = []
    for encoder in requested:
        canonical = canonical_encoder_name(encoder)
        if canonical not in selected:
            selected.append(canonical)
    return selected


def plot_by_encoder(
    records: list[gradient_plots.DiagnosticRecord],
    plot: str,
    output_dir: Path,
    encoders: list[str],
) -> list[str]:
    """Render requested plot families after filtering records to each encoder."""

    written: list[str] = []
    for encoder in encoders:
        encoder_records = [record for record in records if record.encoder == encoder]
        if not encoder_records:
            continue
        encoder_dir = output_dir / encoder
        encoder_dir.mkdir(parents=True, exist_ok=True)
        written.extend(gradient_plots.plot_requested(encoder_records, plot, encoder_dir))
    return written


if __name__ == "__main__":
    main()

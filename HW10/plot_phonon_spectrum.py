import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_phonon_table(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a tab/space separated phonon table: first column is k-path, others are branches."""
    data = np.loadtxt(file_path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Input file format is invalid: expected at least 2 columns.")
    k_path = data[:, 0]
    branches = data[:, 1:]
    return k_path, branches


def detect_kpath_nodes(k_path: np.ndarray) -> list[int]:
    """Detect turning points in the sampled k-path using direction changes in dk."""
    if len(k_path) < 3:
        return [0, len(k_path) - 1]

    dk = np.diff(k_path)
    nodes = [0]
    for i in range(1, len(dk)):
        if np.sign(dk[i]) != np.sign(dk[i - 1]):
            nodes.append(i)
    nodes.append(len(k_path) - 1)

    # Deduplicate while preserving order.
    seen = set()
    unique_nodes = []
    for n in nodes:
        if n not in seen:
            unique_nodes.append(n)
            seen.add(n)
    return unique_nodes


def plot_phonon_spectrum(
    k_path: np.ndarray,
    branches: np.ndarray,
    out_path: Path,
    ylabel: str,
    title: str,
    k_labels: list[str] | None = None,
    k_label_positions: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot each branch; highlight imaginary frequencies (<0).
    n_branches = branches.shape[1]
    for i in range(n_branches):
        y = branches[:, i]
        ax.plot(k_path, y, color="#1f77b4", linewidth=0.85, alpha=0.9)

        neg_mask = y < 0
        if np.any(neg_mask):
            ax.plot(k_path[neg_mask], y[neg_mask], color="#d62728", linewidth=1.0)

    # Mark k-path segment boundaries and set high-symmetry labels only with trusted positions.
    nodes = detect_kpath_nodes(k_path)
    if k_labels and k_label_positions is not None and len(k_labels) == len(k_label_positions):
        for x in k_label_positions:
            ax.axvline(float(x), color="0.75", linewidth=0.8, linestyle="--")
        ax.set_xticks(k_label_positions)
        ax.set_xticklabels(k_labels)
    elif k_labels and len(k_labels) == len(nodes):
        tick_positions = np.array([k_path[idx] for idx in nodes], dtype=float)
        for x in tick_positions:
            ax.axvline(float(x), color="0.75", linewidth=0.8, linestyle="--")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(k_labels)
    else:
        for idx in nodes:
            ax.axvline(k_path[idx], color="0.75", linewidth=0.8, linestyle="--")

    ax.axhline(0.0, color="0.55", linewidth=0.9)
    ax.set_xlabel("High-symmetry k-path")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(float(np.min(k_path)), float(np.max(k_path)))
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot phonon dispersion from a text table.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a single phonon spectrum text file. If omitted, batch mode is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path for single-file mode.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="Phonon*.txt",
        help="Glob pattern for batch mode.",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="Frequency (THz)",
        help="Y-axis label (set unit manually if needed, e.g. THz or cm^-1).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Phonon Spectrum",
        help="Plot title used for all generated figures.",
    )
    parser.add_argument(
        "--klabels",
        type=str,
        default="",
        help="Comma-separated high-symmetry labels, e.g. Γ,X,H$_1$,C,H,Y,Γ,C.",
    )
    parser.add_argument(
        "--kpos",
        type=str,
        default="",
        help="Comma-separated k-path positions for labels (must match --klabels length).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_labels = [item.strip() for item in args.klabels.split(",") if item.strip()]
    k_label_positions = None
    if args.kpos.strip():
        k_label_positions = np.array(
            [float(item.strip()) for item in args.kpos.split(",") if item.strip()],
            dtype=float,
        )

    use_labels = bool(k_labels)
    if use_labels and k_label_positions is not None and len(k_labels) != len(k_label_positions):
        print("Warning: --klabels and --kpos length mismatch, labels are ignored.")
        k_labels = []
        k_label_positions = None

    if args.input is not None:
        input_files = [args.input]
    else:
        input_files = sorted(Path.cwd().glob(args.pattern))
        if not input_files:
            print(f"No files matched pattern: {args.pattern}")
            return

    for input_file in input_files:
        try:
            k_path, branches = load_phonon_table(input_file)
        except Exception as exc:
            print(f"Skip {input_file.name}: {exc}")
            continue

        if args.input is not None and args.output is not None:
            out_path = args.output
        else:
            out_path = input_file.with_suffix(".png")

        plot_title = args.title

        plot_phonon_spectrum(
            k_path,
            branches,
            out_path,
            args.ylabel,
            plot_title,
            k_labels=k_labels if k_labels else None,
            k_label_positions=k_label_positions,
        )

        if k_labels and k_label_positions is None:
            nodes = detect_kpath_nodes(k_path)
            if len(k_labels) != len(nodes):
                print(
                    f"Warning [{input_file.name}]: no explicit --kpos and auto-detected "
                    "nodes do not match label count; numeric x-axis is kept."
                )

        print(f"Saved figure to: {out_path}")
        print(f"Data points: {len(k_path)}, branches: {branches.shape[1]}")


if __name__ == "__main__":
    main()

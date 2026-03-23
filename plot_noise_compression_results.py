import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "noise_compression_results.csv"
OUTPUT_DIR = PROJECT_DIR / "plots"


LOSSY_ALGORITHMS = {
    "H264VideoCompressor",
    "H265VideoCompressor",
    "TemporalBinningCompressor",
}


def load_results(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    for row in rows:
        row["background_cps"] = float(row["background_cps"])
        row["dcr_cps"] = float(row["dcr_cps"])
        row["expected_total_independent_noise_hits_per_frame"] = float(
            row["expected_total_independent_noise_hits_per_frame"]
        )
        row["average_active_pixels_per_frame"] = float(row["average_active_pixels_per_frame"])
        row["average_sparsity"] = float(row["average_sparsity"])
        row["compression_ratio"] = float(row["compression_ratio"])
        row["compressed_bytes"] = int(row["compressed_bytes"])
        row["encode_seconds"] = float(row["encode_seconds"])
        row["decode_seconds"] = float(row["decode_seconds"])
        row["mismatch_ratio"] = float(row["mismatch_ratio"])
    return rows


def group_by_algorithm(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["algorithm_class"], []).append(row)

    for algorithm_rows in grouped.values():
        algorithm_rows.sort(key=lambda item: item["expected_total_independent_noise_hits_per_frame"])
    return grouped


def plot_metric(ax, grouped_rows, algorithm_names, metric_key, ylabel, title, use_logx=True, use_logy=False):
    for algorithm_name in algorithm_names:
        rows = grouped_rows[algorithm_name]
        x_values = [row["expected_total_independent_noise_hits_per_frame"] for row in rows]
        y_values = [row[metric_key] for row in rows]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=algorithm_name)

    if use_logx:
        ax.set_xscale("log", base=2)
    if use_logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Expected independent noise hits per frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def save_full_comparison_figure(grouped_rows, output_path: Path):
    algorithm_names = sorted(grouped_rows)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.28, wspace=0.2, right=0.8)

    plot_metric(
        axes[0, 0],
        grouped_rows,
        algorithm_names,
        "compression_ratio",
        "Compression ratio (x)",
        "Compression Ratio vs Noise",
        use_logx=True,
        use_logy=False,
    )
    plot_metric(
        axes[0, 1],
        grouped_rows,
        algorithm_names,
        "compressed_bytes",
        "Compressed size (bytes)",
        "Compressed Size vs Noise",
        use_logx=True,
        use_logy=True,
    )
    plot_metric(
        axes[1, 0],
        grouped_rows,
        algorithm_names,
        "encode_seconds",
        "Encode time (s)",
        "Encode Time vs Noise",
        use_logx=True,
        use_logy=False,
    )
    plot_metric(
        axes[1, 1],
        grouped_rows,
        algorithm_names,
        "mismatch_ratio",
        "Mismatch ratio",
        "Mismatch Ratio vs Noise",
        use_logx=True,
        use_logy=True,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=True, fontsize=9)
    fig.suptitle("SPAD Noise Compression Sweep", fontsize=16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_lossless_focus_figure(grouped_rows, output_path: Path):
    algorithm_names = [name for name in sorted(grouped_rows) if name not in LOSSY_ALGORITHMS]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    plt.subplots_adjust(wspace=0.22, right=0.8)

    plot_metric(
        axes[0],
        grouped_rows,
        algorithm_names,
        "compression_ratio",
        "Compression ratio (x)",
        "Lossless Compression Ratio vs Noise",
        use_logx=True,
        use_logy=False,
    )
    plot_metric(
        axes[1],
        grouped_rows,
        algorithm_names,
        "encode_seconds",
        "Encode time (s)",
        "Lossless Encode Time vs Noise",
        use_logx=True,
        use_logy=False,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=True, fontsize=9)
    fig.suptitle("SPAD Noise Sweep: Lossless Algorithms", fontsize=16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Result CSV not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_results(CSV_PATH)
    grouped_rows = group_by_algorithm(rows)

    full_figure_path = OUTPUT_DIR / "noise_compression_overview.png"
    lossless_figure_path = OUTPUT_DIR / "noise_compression_lossless.png"

    save_full_comparison_figure(grouped_rows, full_figure_path)
    save_lossless_focus_figure(grouped_rows, lossless_figure_path)

    print(f"Saved figure: {full_figure_path}")
    print(f"Saved figure: {lossless_figure_path}")


if __name__ == "__main__":
    main()
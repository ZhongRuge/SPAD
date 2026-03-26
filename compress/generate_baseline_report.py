import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CURRENT_DIR = Path(__file__).resolve().parent
DATA_GENERATE_DIR = CURRENT_DIR.parent / "data_generate"
if str(DATA_GENERATE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATE_DIR))

from experiment_output import dataset_name_from_paths
from experiment_output import find_latest_run_dir
from simulation_io import load_config
from simulation_io import resolve_compression_output_dir
from simulation_io import resolve_dataset_paths


LABELS = {
    "rle": "RLE",
    "delta_rle": "Delta+RLE",
    "delta_sparse": "Delta+Sparse",
    "delta_sparse_zlib": "Delta+Sparse+Zlib",
    "aer": "AER",
    "temporal_binning": "TemporalBinning",
}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _params_to_text(params):
    if not params:
        return "{}"
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def _lossless_rows(rows):
    return [row for row in rows if row["is_lossless_algorithm"]]


def _normalize(values, reverse=False):
    values = list(values)
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [1.0] * len(values)
    normalized = [(value - vmin) / (vmax - vmin) for value in values]
    if reverse:
        return [1.0 - value for value in normalized]
    return normalized


def _compute_balanced_choice(rows):
    lossless_rows = _lossless_rows(rows)
    cr_scores = _normalize([row["compression_ratio"] for row in lossless_rows], reverse=False)
    encode_scores = _normalize([row["encode_seconds"] for row in lossless_rows], reverse=True)
    decode_scores = _normalize([row["decode_seconds"] for row in lossless_rows], reverse=True)

    best_row = None
    best_score = None
    for row, cr_score, encode_score, decode_score in zip(lossless_rows, cr_scores, encode_scores, decode_scores):
        balance_score = (cr_score + encode_score + decode_score) / 3.0
        row["balance_score"] = balance_score
        if best_score is None or balance_score > best_score:
            best_score = balance_score
            best_row = row
    return best_row


def load_baseline_rows(run_dir):
    run_manifest = _load_json(run_dir / "run_manifest.json")
    rows = []
    for entry in run_manifest["algorithms"]:
        metrics = _load_json(entry["metrics_file"])
        manifest = _load_json(entry["manifest_file"])
        rows.append(
            {
                "algorithm_id": metrics["algorithm_id"],
                "label": LABELS.get(metrics["algorithm_id"], metrics["algorithm_id"]),
                "display_name": metrics["algorithm_display_name"],
                "params": metrics["algorithm_params"],
                "compression_ratio": metrics["compression_ratio"],
                "compressed_size_bytes": metrics["compressed_size_bytes"],
                "encode_seconds": metrics["encode_seconds"],
                "decode_seconds": metrics["decode_seconds"],
                "lossless_check_passed": metrics["lossless_check_passed"],
                "is_lossless_algorithm": metrics["is_lossless_algorithm"],
                "metrics_path": entry["metrics_file"],
                "manifest_path": entry["manifest_file"],
                "compressed_file": entry["compressed_file"],
                "variant_name": manifest["algorithm"]["variant_name"],
            }
        )
    return run_manifest, rows


def save_csv(rows, output_path):
    fieldnames = [
        "algorithm_id",
        "label",
        "display_name",
        "params",
        "compression_ratio",
        "compressed_size_bytes",
        "encode_seconds",
        "decode_seconds",
        "lossless_check_passed",
        "is_lossless_algorithm",
        "variant_name",
        "metrics_path",
        "manifest_path",
        "compressed_file",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["params"] = _params_to_text(csv_row["params"])
            writer.writerow(csv_row)


def plot_compression_ratio(rows, output_path):
    labels = [row["label"] for row in rows]
    values = [row["compression_ratio"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color="#2f5c8f")
    ax.set_title("Baseline Compression Ratio by Algorithm")
    ax.set_ylabel("Compression Ratio (x)")
    ax.set_xlabel("Algorithm")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}x", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_encode_decode_times(rows, output_path):
    labels = [row["label"] for row in rows]
    encode_values = [row["encode_seconds"] for row in rows]
    decode_values = [row["decode_seconds"] for row in rows]
    x = range(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x, encode_values, color="#d17b28")
    axes[0].set_title("Encode Time by Algorithm")
    axes[0].set_ylabel("Seconds")
    axes[0].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].set_axisbelow(True)

    axes[1].bar(x, decode_values, color="#3c9d5d")
    axes[1].set_title("Decode Time by Algorithm")
    axes[1].set_ylabel("Seconds")
    axes[1].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tradeoff(rows, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        color = "#c44e52" if row["is_lossless_algorithm"] else "#8172b2"
        marker = "o" if row["is_lossless_algorithm"] else "s"
        ax.scatter(row["decode_seconds"], row["compression_ratio"], s=110, color=color, marker=marker, alpha=0.85)
        ax.annotate(row["label"], (row["decode_seconds"], row["compression_ratio"]), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_title("Compression Ratio vs Decode Time")
    ax.set_xlabel("Decode Time (s)")
    ax.set_ylabel("Compression Ratio (x)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(run_manifest, rows, output_path, figure_paths):
    best_cr_all = max(rows, key=lambda row: row["compression_ratio"])
    best_cr_lossless = max(_lossless_rows(rows), key=lambda row: row["compression_ratio"])
    best_encode = min(rows, key=lambda row: row["encode_seconds"])
    best_decode = min(rows, key=lambda row: row["decode_seconds"])
    best_decode_lossless = min(_lossless_rows(rows), key=lambda row: row["decode_seconds"])
    most_balanced = _compute_balanced_choice(rows)

    suspicious_points = [
        "TemporalBinning is lossy, so its compression ratio is not directly comparable to the lossless algorithms.",
        "All current conclusions are based on a single dataset configuration and one latest run; more scenes/noise settings are still worth testing.",
        "Delta+RLE underperforms the other delta-based lossless baselines here, which is worth keeping in mind if future datasets become less sparse.",
    ]

    lines = [
        "# Baseline Compression Analysis",
        "",
        f"- Run ID: `{run_manifest['run_id']}`",
        f"- Dataset: `{run_manifest['dataset']['name']}`",
        f"- Data Path: `{run_manifest['dataset']['data_path']}`",
        f"- Meta Path: `{run_manifest['dataset']['meta_path']}`",
        "",
        "## Key Findings",
        "",
        f"- Best compression ratio overall: `{best_cr_all['label']}` at `{best_cr_all['compression_ratio']:.2f}x`.",
        f"- Best compression ratio among lossless algorithms: `{best_cr_lossless['label']}` at `{best_cr_lossless['compression_ratio']:.2f}x`.",
        f"- Fastest encode: `{best_encode['label']}` at `{best_encode['encode_seconds']:.4f}s`.",
        f"- Fastest decode overall: `{best_decode['label']}` at `{best_decode['decode_seconds']:.4f}s`.",
        f"- Fastest decode among lossless algorithms: `{best_decode_lossless['label']}` at `{best_decode_lossless['decode_seconds']:.4f}s`.",
        f"- Most balanced lossless baseline: `{most_balanced['label']}`.",
        "",
        "## Results Table",
        "",
        "| Algorithm | Params | CR | Size (bytes) | Encode (s) | Decode (s) | Lossless Check |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        lossless_text = "PASS" if row["lossless_check_passed"] is True else ("N/A" if row["lossless_check_passed"] is None else "FAIL")
        lines.append(
            f"| {row['label']} | `{_params_to_text(row['params'])}` | {row['compression_ratio']:.4f}x | {row['compressed_size_bytes']} | {row['encode_seconds']:.4f} | {row['decode_seconds']:.4f} | {lossless_text} |"
        )

    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
            f"- Compression ratio chart: `{figure_paths['compression_ratio']}`",
            f"- Encode/decode time chart: `{figure_paths['encode_decode_times']}`",
            f"- Tradeoff chart: `{figure_paths['tradeoff']}`",
            "",
            "## Follow-up Notes",
            "",
        ]
    )
    lines.extend([f"- {point}" for point in suspicious_points])
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    config = load_config(DATA_GENERATE_DIR / "config.yaml")
    dataset_paths = resolve_dataset_paths(config)
    dataset_name = dataset_name_from_paths(dataset_paths)
    run_dir = find_latest_run_dir(resolve_compression_output_dir(config), dataset_name)
    if run_dir is None:
        raise SystemExit("No baseline run found for current dataset.")

    run_manifest, rows = load_baseline_rows(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    compression_ratio_path = analysis_dir / "compression_ratio.png"
    encode_decode_times_path = analysis_dir / "encode_decode_times.png"
    tradeoff_path = analysis_dir / "compression_ratio_vs_decode_time.png"
    csv_path = analysis_dir / "baseline_results.csv"
    summary_path = analysis_dir / "baseline_summary.md"
    summary_json_path = analysis_dir / "baseline_summary.json"

    plot_compression_ratio(rows, compression_ratio_path)
    plot_encode_decode_times(rows, encode_decode_times_path)
    plot_tradeoff(rows, tradeoff_path)
    save_csv(rows, csv_path)

    write_summary(
        run_manifest,
        rows,
        summary_path,
        {
            "compression_ratio": compression_ratio_path.resolve(),
            "encode_decode_times": encode_decode_times_path.resolve(),
            "tradeoff": tradeoff_path.resolve(),
        },
    )

    summary_payload = {
        "run_id": run_manifest["run_id"],
        "dataset_name": run_manifest["dataset"]["name"],
        "analysis_dir": str(analysis_dir.resolve()),
        "figures": {
            "compression_ratio": str(compression_ratio_path.resolve()),
            "encode_decode_times": str(encode_decode_times_path.resolve()),
            "tradeoff": str(tradeoff_path.resolve()),
        },
        "table_csv": str(csv_path.resolve()),
        "summary_markdown": str(summary_path.resolve()),
        "rows": rows,
    }
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False)

    print(f"Analysis output directory: {analysis_dir.resolve()}")
    print(f"Compression ratio figure: {compression_ratio_path.resolve()}")
    print(f"Encode/decode time figure: {encode_decode_times_path.resolve()}")
    print(f"Tradeoff figure: {tradeoff_path.resolve()}")
    print(f"Summary markdown: {summary_path.resolve()}")
    print(f"Results CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()

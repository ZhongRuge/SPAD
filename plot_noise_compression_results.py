import csv
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "noise_compression_results.csv"
OUTPUT_DIR = PROJECT_DIR / "plots"


LOSSY_ALGORITHMS = {
    "H264VideoCompressor",
    "H265VideoCompressor",
    "TemporalBinningCompressor",
}

ALGORITHM_LABELS = {
    "AerCompressor": "AER 事件地址",
    "RleCompressor": "RLE 游程编码",
    "DeltaRleCompressor": "帧间差分 + RLE",
    "DeltaSparseCompressor": "帧间差分 + 稀疏坐标",
    "DeltaSparseVarintZlibCompressor": "差分稀疏流 + Varint + Zlib",
    "DeltaSparseZlibCompressor": "帧间差分 + 稀疏坐标 + Zlib",
    "PackBitsZlibCompressor": "位打包 + Zlib",
    "TemporalBinningCompressor": "时域累加 + Zlib",
    "GlobalEventStreamCompressor": "全局事件流 + Zlib",
}

REPRESENTATIVE_ALGORITHMS = [
    "RleCompressor",
    "DeltaSparseZlibCompressor",
    "DeltaSparseVarintZlibCompressor",
    "PackBitsZlibCompressor",
    "GlobalEventStreamCompressor",
    "TemporalBinningCompressor",
]


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


def _display_label(algorithm_name):
    return ALGORITHM_LABELS.get(algorithm_name, algorithm_name)


def _ordered_available_algorithms(grouped_rows, preferred_order):
    return [name for name in preferred_order if name in grouped_rows]


def _all_algorithms(grouped_rows):
    return sorted(grouped_rows, key=lambda name: _display_label(name))


def plot_metric(ax, grouped_rows, algorithm_names, metric_key, ylabel, title, use_logy=False):
    for algorithm_name in algorithm_names:
        rows = grouped_rows[algorithm_name]
        x_values = [row["expected_total_independent_noise_hits_per_frame"] for row in rows]
        y_values = [row[metric_key] for row in rows]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=_display_label(algorithm_name))

    if use_logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("每帧期望独立噪声命中数")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def save_full_comparison_figure(grouped_rows, output_path: Path):
    algorithm_names = _all_algorithms(grouped_rows)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    plt.subplots_adjust(hspace=0.32, wspace=0.24, right=0.79)

    plot_metric(
        axes[0, 0],
        grouped_rows,
        algorithm_names,
        "compression_ratio",
        "压缩率 (x)",
        "不同噪声下的压缩率",
        use_logy=False,
    )
    plot_metric(
        axes[0, 1],
        grouped_rows,
        algorithm_names,
        "compressed_bytes",
        "压缩后大小 (字节)",
        "不同噪声下的压缩后大小",
        use_logy=True,
    )
    plot_metric(
        axes[1, 0],
        grouped_rows,
        algorithm_names,
        "encode_seconds",
        "编码耗时 (秒)",
        "不同噪声下的编码耗时",
        use_logy=False,
    )
    plot_metric(
        axes[1, 1],
        grouped_rows,
        algorithm_names,
        "mismatch_ratio",
        "重建不一致比例",
        "不同噪声下的重建误差",
        use_logy=True,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=True, fontsize=9)
    fig.suptitle("SPAD 压缩算法噪声扫描总览", fontsize=16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_lossless_focus_figure(grouped_rows, output_path: Path):
    algorithm_names = [
        name for name in _all_algorithms(grouped_rows)
        if name not in LOSSY_ALGORITHMS
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
    plt.subplots_adjust(wspace=0.25, right=0.78)

    plot_metric(
        axes[0],
        grouped_rows,
        algorithm_names,
        "compression_ratio",
        "压缩率 (x)",
        "无损算法压缩率对比",
        use_logy=False,
    )
    plot_metric(
        axes[1],
        grouped_rows,
        algorithm_names,
        "encode_seconds",
        "编码耗时 (秒)",
        "无损算法编码耗时对比",
        use_logy=False,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=True, fontsize=9)
    fig.suptitle("SPAD 噪声扫描：无损算法对比", fontsize=16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_representative_figure(grouped_rows, output_path: Path):
    algorithm_names = _ordered_available_algorithms(grouped_rows, REPRESENTATIVE_ALGORITHMS)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.32, wspace=0.24, right=0.78)

    plot_metric(
        axes[0, 0],
        grouped_rows,
        algorithm_names,
        "compression_ratio",
        "压缩率 (x)",
        "精选算法：压缩率对比",
        use_logy=False,
    )
    plot_metric(
        axes[0, 1],
        grouped_rows,
        algorithm_names,
        "encode_seconds",
        "编码耗时 (秒)",
        "精选算法：编码耗时对比",
        use_logy=False,
    )
    plot_metric(
        axes[1, 0],
        grouped_rows,
        algorithm_names,
        "decode_seconds",
        "解码耗时 (秒)",
        "精选算法：解码耗时对比",
        use_logy=False,
    )
    plot_metric(
        axes[1, 1],
        grouped_rows,
        algorithm_names,
        "mismatch_ratio",
        "重建不一致比例",
        "精选算法：重建误差对比",
        use_logy=True,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=True, fontsize=9)
    fig.suptitle("SPAD 噪声扫描：精选代表算法", fontsize=16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"未找到结果 CSV 文件：{CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_results(CSV_PATH)
    grouped_rows = group_by_algorithm(rows)

    full_figure_path = OUTPUT_DIR / "噪声压缩总览.png"
    lossless_figure_path = OUTPUT_DIR / "噪声压缩_无损算法.png"
    representative_figure_path = OUTPUT_DIR / "噪声压缩_精选算法.png"

    save_full_comparison_figure(grouped_rows, full_figure_path)
    save_lossless_focus_figure(grouped_rows, lossless_figure_path)
    save_representative_figure(grouped_rows, representative_figure_path)

    print(f"已保存图像：{full_figure_path}")
    print(f"已保存图像：{lossless_figure_path}")
    print(f"已保存图像：{representative_figure_path}")


if __name__ == "__main__":
    main()

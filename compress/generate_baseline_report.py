import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


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
    "rle": "RLE 游程编码",
    "delta_rle": "帧间差分 + RLE",
    "delta_sparse": "帧间差分 + 稀疏坐标",
    "delta_sparse_varint_zlib": "差分稀疏流 + Varint + Zlib",
    "delta_sparse_zlib": "帧间差分 + 稀疏坐标 + Zlib",
    "packbits_zlib": "位打包 + Zlib",
    "global_event_stream": "全局事件流 + Zlib",
    "aer": "AER 事件地址表示",
    "temporal_binning": "时域累加 + Zlib",
}

TRADEOFF_CMAP = LinearSegmentedColormap.from_list(
    "performance_red_to_green",
    ["#2f9e44", "#f2c94c", "#d62828"],
)


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


def _compute_tradeoff_scores(rows):
    compression_scores = _normalize([row["compression_ratio"] for row in rows], reverse=False)
    decode_scores = _normalize([row["decode_seconds"] for row in rows], reverse=True)

    for row, compression_score, decode_score in zip(rows, compression_scores, decode_scores):
        row["tradeoff_score"] = (compression_score + decode_score) / 2.0


def _lossless_check_text(row):
    if row["lossless_check_passed"] is True:
        return "通过"
    if row["lossless_check_passed"] is None:
        return "不适用"
    return "失败"


def _cluster_rows(sorted_rows, y_threshold):
    clusters = []
    current_cluster = []
    for row in sorted_rows:
        if not current_cluster:
            current_cluster = [row]
            continue
        if abs(row["compression_ratio"] - current_cluster[-1]["compression_ratio"]) <= y_threshold:
            current_cluster.append(row)
        else:
            clusters.append(current_cluster)
            current_cluster = [row]
    if current_cluster:
        clusters.append(current_cluster)
    return clusters


def _build_label_offsets(rows, plot_x_min, plot_x_max, plot_y_min, plot_y_max):
    x_mid = (plot_x_min + plot_x_max) / 2.0
    y_range = plot_y_max - plot_y_min
    y_threshold = max(y_range * 0.06, 14.0)
    offsets = {}

    for side in ("left", "right"):
        side_rows = [
            row for row in rows
            if (row["decode_seconds"] < x_mid if side == "left" else row["decode_seconds"] >= x_mid)
        ]
        side_rows.sort(key=lambda row: row["compression_ratio"])

        for cluster in _cluster_rows(side_rows, y_threshold):
            count = len(cluster)
            spread_step = 8
            spread = [int((index - (count - 1) / 2.0) * spread_step) for index in range(count)]

            for row, cluster_offset in zip(cluster, spread):
                if side == "left":
                    x_offset = 8
                else:
                    x_offset = -82

                if row["compression_ratio"] >= plot_y_max - y_range * 0.12:
                    base_y_offset = -10
                elif row["compression_ratio"] <= plot_y_min + y_range * 0.12:
                    base_y_offset = 6
                else:
                    base_y_offset = 0

                offsets[row["algorithm_id"]] = (x_offset, base_y_offset + cluster_offset)

    return offsets


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
            csv_row = {field: row.get(field) for field in fieldnames}
            csv_row["params"] = _params_to_text(csv_row["params"])
            writer.writerow(csv_row)


def plot_compression_ratio(rows, output_path):
    sorted_rows = sorted(rows, key=lambda row: row["compression_ratio"], reverse=True)
    labels = [row["label"] for row in sorted_rows]
    values = [row["compression_ratio"] for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(11.5, 6))
    bars = ax.barh(labels, values, color="#2f5c8f")
    ax.set_title("各算法基线压缩率对比")
    ax.set_xlabel("压缩率 (x)")
    ax.set_ylabel("算法")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.1f}x", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_encode_decode_times(rows, output_path):
    labels = [row["label"] for row in rows]
    encode_values = [row["encode_seconds"] for row in rows]
    decode_values = [row["decode_seconds"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13.5, 6))
    encode_bars = ax.bar(x - width / 2, encode_values, width=width, color="#d17b28", label="编码")
    decode_bars = ax.bar(x + width / 2, decode_values, width=width, color="#2a9d8f", label="解码")

    ax.set_title("各算法编解码耗时成对对比")
    ax.set_ylabel("耗时 (秒)")
    ax.set_xlabel("算法")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    for bars in (encode_bars, decode_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tradeoff(rows, output_path):
    _compute_tradeoff_scores(rows)

    fig, ax = plt.subplots(figsize=(9.8, 6.6))

    x_min = min(row["decode_seconds"] for row in rows)
    x_max = max(row["decode_seconds"] for row in rows)
    y_min = min(row["compression_ratio"] for row in rows)
    y_max = max(row["compression_ratio"] for row in rows)
    x_padding = max((x_max - x_min) * 0.15, 0.08)
    y_padding = max((y_max - y_min) * 0.10, 10.0)
    plot_x_min = max(0.0, x_min - x_padding)
    plot_x_max = x_max + x_padding
    plot_y_min = max(0.0, y_min - y_padding)
    plot_y_max = y_max + y_padding

    gradient_size = 300
    x_values = np.linspace(0.0, 1.0, gradient_size)
    y_values = np.linspace(0.0, 1.0, gradient_size)
    xx, yy = np.meshgrid(x_values, y_values)
    background_score = (yy + (1.0 - xx)) / 2.0
    ax.imshow(
        background_score,
        extent=[plot_x_min, plot_x_max, plot_y_min, plot_y_max],
        origin="lower",
        cmap=TRADEOFF_CMAP,
        alpha=0.16,
        aspect="auto",
    )

    scores = [row["tradeoff_score"] for row in rows]
    colors = [TRADEOFF_CMAP(score) for score in scores]
    ax.scatter(
        [row["decode_seconds"] for row in rows],
        [row["compression_ratio"] for row in rows],
        s=130,
        c=colors,
        marker="o",
        edgecolors="white",
        linewidths=0.9,
        alpha=0.95,
    )

    offsets = _build_label_offsets(rows, plot_x_min, plot_x_max, plot_y_min, plot_y_max)
    for row in rows:
        x_offset, y_offset = offsets[row["algorithm_id"]]
        ax.annotate(
            row["label"],
            (row["decode_seconds"], row["compression_ratio"]),
            textcoords="offset points",
            xytext=(x_offset, y_offset),
            fontsize=9,
            annotation_clip=False,
        )

    ax.set_title("压缩率与解码耗时权衡")
    ax.set_xlabel("解码耗时 (秒，越小越好)")
    ax.set_ylabel("压缩率 (x，越大越好)")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(plot_y_min, plot_y_max)

    ax.text(
        0.015,
        0.98,
        "左上更优，右下更弱",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "boxstyle": "round,pad=0.25"},
    )

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

    follow_up_notes = [
        "TemporalBinning 属于有损算法，它的压缩率不能与无损算法直接横向比较。",
        "当前结论主要基于单一数据集配置与最近一次运行结果，后续仍建议扩展到更多场景和噪声水平。",
        "Delta+RLE 在当前稀疏 SPAD 数据上明显弱于其他差分型无损基线，后续是否保留可视具体研究目标决定。",
        "Delta+Sparse+Varint+Zlib 通过减少索引元数据开销提升压缩率，在空帧多、事件很稀疏时更有优势。",
        "GlobalEventStream 更适合整批数据全局稀疏的情况，而 PackBits+Zlib 更像通用、实现简单的强基线。",
    ]

    lines = [
        "# 基线压缩分析",
        "",
        f"- 运行编号：`{run_manifest['run_id']}`",
        f"- 数据集：`{run_manifest['dataset']['name']}`",
        f"- 数据路径：`{run_manifest['dataset']['data_path']}`",
        f"- 元数据路径：`{run_manifest['dataset']['meta_path']}`",
        "",
        "## 关键结论",
        "",
        f"- 整体压缩率最高：`{best_cr_all['label']}`，压缩率为 `{best_cr_all['compression_ratio']:.2f}x`。",
        f"- 无损算法中压缩率最高：`{best_cr_lossless['label']}`，压缩率为 `{best_cr_lossless['compression_ratio']:.2f}x`。",
        f"- 编码最快：`{best_encode['label']}`，耗时 `{best_encode['encode_seconds']:.4f}s`。",
        f"- 整体解码最快：`{best_decode['label']}`，耗时 `{best_decode['decode_seconds']:.4f}s`。",
        f"- 无损算法中解码最快：`{best_decode_lossless['label']}`，耗时 `{best_decode_lossless['decode_seconds']:.4f}s`。",
        f"- 综合最均衡的无损基线：`{most_balanced['label']}`。",
        "",
        "## 结果表",
        "",
        "| 算法 | 参数 | 压缩率 | 压缩后大小 (字节) | 编码耗时 (秒) | 解码耗时 (秒) | 无损校验 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['label']} | `{_params_to_text(row['params'])}` | {row['compression_ratio']:.4f}x | {row['compressed_size_bytes']} | {row['encode_seconds']:.4f} | {row['decode_seconds']:.4f} | {_lossless_check_text(row)} |"
        )

    lines.extend(
        [
            "",
            "## 生成图表",
            "",
            f"- 压缩率对比图：`{figure_paths['compression_ratio']}`",
            f"- 编解码耗时对比图：`{figure_paths['encode_decode_times']}`",
            f"- 压缩率与解码耗时权衡图：`{figure_paths['tradeoff']}`",
            "",
            "## 备注",
            "",
        ]
    )
    lines.extend([f"- {point}" for point in follow_up_notes])
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    config = load_config(DATA_GENERATE_DIR / "config.yaml")
    dataset_paths = resolve_dataset_paths(config)
    dataset_name = dataset_name_from_paths(dataset_paths)
    run_dir = find_latest_run_dir(resolve_compression_output_dir(config), dataset_name)
    if run_dir is None:
        raise SystemExit("当前数据集还没有可用的 baseline 运行结果。")

    run_manifest, rows = load_baseline_rows(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    compression_ratio_path = analysis_dir / "压缩率对比图.png"
    encode_decode_times_path = analysis_dir / "编解码耗时对比图.png"
    tradeoff_path = analysis_dir / "压缩率与解码耗时权衡图.png"
    csv_path = analysis_dir / "基线结果表.csv"
    summary_path = analysis_dir / "基线分析总结.md"
    summary_json_path = analysis_dir / "基线分析总结.json"

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

    print(f"分析输出目录：{analysis_dir.resolve()}")
    print(f"压缩率图：{compression_ratio_path.resolve()}")
    print(f"编解码耗时图：{encode_decode_times_path.resolve()}")
    print(f"权衡图：{tradeoff_path.resolve()}")
    print(f"总结文档：{summary_path.resolve()}")
    print(f"结果表：{csv_path.resolve()}")


if __name__ == "__main__":
    main()

"""SPAD 压缩项目 — 专业可视化报告生成器 v2。

生成 6 张高质量图表：
  1. 基线压缩率对比（有损/无损分色，无损排名突出）
  2. 编解码耗时 — 按总耗时排序的哑铃图
  3. 压缩效率-速度权衡散点 — 仅无损 + Pareto 前沿
  4. 多维综合雷达图 — 归一化多指标直觉对比
  5. 噪声扫描：压缩率随噪声变化趋势（分面板：无损 / 有损）
  6. 噪声扫描：热力图 — 算法×噪声级别 → 压缩率
"""

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import FancyArrowPatch

# ── 全局样式 ───────────────────────────────────────────────
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                        "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
})

# ── 配色方案（9 色，色盲友好） ─────────────────────────────
PALETTE = {
    "rle":                      "#4e79a7",
    "delta_rle":                "#a0cbe8",
    "delta_sparse":             "#59a14f",
    "delta_sparse_varint_zlib": "#8cd17d",
    "delta_sparse_zlib":        "#b6992d",
    "packbits_zlib":            "#f28e2b",
    "global_event_stream":      "#e15759",
    "aer":                      "#76b7b2",
    "temporal_binning":         "#b07aa1",
}

SHORT_LABELS = {
    "rle":                      "RLE",
    "delta_rle":                "Delta+RLE",
    "delta_sparse":             "Delta+Sparse",
    "delta_sparse_varint_zlib": "D+S+Varint+Zlib",
    "delta_sparse_zlib":        "Delta+Sparse+Zlib",
    "packbits_zlib":            "PackBits+Zlib",
    "global_event_stream":      "GES+Zlib",
    "aer":                      "AER",
    "temporal_binning":         "TemporalBin (有损)",
}

# 用于 noise sweep CSV (算法类名 → 标准 id)
CLASS_TO_ID = {
    "AerCompressor":                     "aer",
    "RleCompressor":                     "rle",
    "DeltaRleCompressor":                "delta_rle",
    "DeltaSparseCompressor":             "delta_sparse",
    "DeltaSparseVarintZlibCompressor":   "delta_sparse_varint_zlib",
    "DeltaSparseZlibCompressor":         "delta_sparse_zlib",
    "PackBitsZlibCompressor":            "packbits_zlib",
    "GlobalEventStreamCompressor":       "global_event_stream",
    "TemporalBinningCompressor":         "temporal_binning",
}

LOSSLESS_IDS = set(PALETTE) - {"temporal_binning"}

from spad.config import load_config, resolve_compression_output_dir, resolve_dataset_paths
from spad.evaluate import find_latest_run_dir, dataset_name_from_paths
from spad import PROJECT_ROOT


# ═══════════════════════════════════════════════════════════
#                  数据加载
# ═══════════════════════════════════════════════════════════

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline_rows(run_dir):
    manifest = _load_json(run_dir / "run_manifest.json")
    rows = []
    for entry in manifest["algorithms"]:
        m = _load_json(entry["metrics_file"])
        aid = m["algorithm_id"]
        rows.append({
            "id": aid,
            "label": SHORT_LABELS.get(aid, aid),
            "color": PALETTE.get(aid, "#999999"),
            "compression_ratio": m["compression_ratio"],
            "encode_seconds": m["encode_seconds"],
            "decode_seconds": m["decode_seconds"],
            "total_seconds": m["encode_seconds"] + m["decode_seconds"],
            "lossless": m["is_lossless_algorithm"],
            "compressed_size_bytes": m["compressed_size_bytes"],
        })
    return rows


def load_noise_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["noise_hits"] = float(r["expected_total_independent_noise_hits_per_frame"])
        r["compression_ratio"] = float(r["compression_ratio"])
        r["encode_seconds"] = float(r["encode_seconds"])
        r["decode_seconds"] = float(r["decode_seconds"])
        r["mismatch_ratio"] = float(r["mismatch_ratio"])
        r["compressed_bytes"] = int(r["compressed_bytes"])
        r["alg_id"] = CLASS_TO_ID.get(r["algorithm_class"], r["algorithm_class"])
    return rows


# ═══════════════════════════════════════════════════════════
#           图1: 压缩率对比（有损/无损分区）
# ═══════════════════════════════════════════════════════════

def plot_compression_ratio(rows, output_path):
    """水平条形图：无损算法按压缩率排序在上半区，有损算法单独标注在下方。"""
    lossless = sorted([r for r in rows if r["lossless"]], key=lambda r: r["compression_ratio"])
    lossy = sorted([r for r in rows if not r["lossless"]], key=lambda r: r["compression_ratio"])

    all_rows = lossless + lossy
    n_ll, n_ly = len(lossless), len(lossy)
    n_total = n_ll + n_ly

    fig, ax = plt.subplots(figsize=(10, max(4.5, n_total * 0.6 + 1)))

    # ── 无损 ──
    y_pos = np.arange(n_ll)
    bars_ll = ax.barh(y_pos, [r["compression_ratio"] for r in lossless],
                      color=[r["color"] for r in lossless], edgecolor="white", linewidth=0.5, height=0.68)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in lossless])

    for bar, r in zip(bars_ll, lossless):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{r['compression_ratio']:.1f}x", va="center", fontsize=9, fontweight="bold")

    # 最佳无损 — 金色星标，不用箭头避免重叠
    best_ll = lossless[-1]
    ax.scatter(best_ll["compression_ratio"] + 28, n_ll - 1, marker="*", s=200,
              color="#e8a838", edgecolors="#b8860b", linewidths=0.8, zorder=5)
    ax.text(best_ll["compression_ratio"] + 38, n_ll - 1, "无损最优",
            va="center", fontsize=9, fontweight="bold", color="#b8860b")

    # ── 有损（虚线分隔） ──
    if lossy:
        sep_y = n_ll - 0.5
        ax.axhline(sep_y, color="#bbb", linewidth=0.8, linestyle="--")
        y_lossy = np.arange(n_ll, n_total)
        bars_ly = ax.barh(y_lossy, [r["compression_ratio"] for r in lossy],
                          color=[r["color"] for r in lossy], edgecolor="white",
                          linewidth=0.5, height=0.68, hatch="///", alpha=0.65)
        ax.set_yticks(np.concatenate([y_pos, y_lossy]))
        labels = [r["label"] for r in lossless] + [r["label"] for r in lossy]
        ax.set_yticklabels(labels)
        for bar, r in zip(bars_ly, lossy):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{r['compression_ratio']:.1f}x (有损)", va="center", fontsize=9, color="#888")

    ax.set_xlabel("压缩率 (x)")
    ax.set_title("GES+Zlib 以 111x 领先无损压缩率 | 有损 TemporalBin 达 454x 但存在信息丢失")
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#           图2: 编解码耗时哑铃图
# ═══════════════════════════════════════════════════════════

def plot_encode_decode_dumbbell(rows, output_path):
    """哑铃图：每个算法一行，左端=编码，右端=解码，连线=总跨度。按总耗时排序。"""
    sorted_rows = sorted(rows, key=lambda r: r["total_seconds"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = np.arange(len(sorted_rows))

    for i, r in enumerate(sorted_rows):
        enc, dec = r["encode_seconds"], r["decode_seconds"]
        left, right = min(enc, dec), max(enc, dec)
        ax.plot([enc, dec], [i, i], color="#cccccc", linewidth=2.5, zorder=1)
        ax.scatter(enc, i, color="#d17b28", s=70, zorder=2, marker="o", edgecolors="white", linewidths=0.5)
        ax.scatter(dec, i, color="#2a9d8f", s=70, zorder=2, marker="s", edgecolors="white", linewidths=0.5)
        # 总耗时标注
        ax.text(right + 0.02, i, f"Σ {r['total_seconds']:.2f}s", va="center", fontsize=8, color="#555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in sorted_rows])
    ax.set_xlabel("耗时 (秒)")
    ax.set_title("TemporalBin 总耗时最低 (0.19s) | PackBits+Zlib 编码最慢 (0.99s)")

    # 图例
    ax.scatter([], [], color="#d17b28", s=50, marker="o", label="编码耗时")
    ax.scatter([], [], color="#2a9d8f", s=50, marker="s", label="解码耗时")
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#      图3: 压缩率-解码速度权衡（仅无损 + Pareto）
# ═══════════════════════════════════════════════════════════

def plot_tradeoff_pareto(rows, output_path):
    """散点图：X=解码耗时, Y=压缩率, 仅无损算法, 标注 Pareto 前沿。"""
    lossless = [r for r in rows if r["lossless"]]

    fig, ax = plt.subplots(figsize=(9, 6))

    # 绘制所有点
    for r in lossless:
        ax.scatter(r["decode_seconds"], r["compression_ratio"],
                   s=140, color=r["color"], edgecolors="white", linewidths=0.8,
                   zorder=3, alpha=0.95)
        ax.annotate(r["label"], (r["decode_seconds"], r["compression_ratio"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=9)

    # Pareto 前沿: 解码越快且压缩率越高越好
    # 按解码时间排序，找不被支配的点
    pareto = []
    for r in sorted(lossless, key=lambda r: r["decode_seconds"]):
        if not pareto or r["compression_ratio"] > pareto[-1]["compression_ratio"]:
            pareto.append(r)

    if len(pareto) >= 2:
        px = [r["decode_seconds"] for r in pareto]
        py = [r["compression_ratio"] for r in pareto]
        # 阶梯线连接 Pareto 点
        step_x, step_y = [px[0]], [py[0]]
        for i in range(1, len(px)):
            step_x.extend([px[i], px[i]])
            step_y.extend([py[i-1], py[i]])
        ax.plot(step_x, step_y, color="#e15759", linewidth=1.5, linestyle="--",
                alpha=0.6, zorder=2, label="Pareto 前沿")

    # 标注 Pareto 点
    for r in pareto:
        ax.scatter(r["decode_seconds"], r["compression_ratio"],
                   s=200, facecolors="none", edgecolors="#e15759", linewidths=2, zorder=4)

    ax.set_xlabel("解码耗时 (秒) →  越小越好")
    ax.set_ylabel("压缩率 (x) →  越大越好")
    ax.set_title("GES+Zlib 和 PackBits+Zlib 位于 Pareto 前沿 | 兼顾高压缩率与快解码")

    ax.text(0.02, 0.98, "← 左上更优", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, style="italic", color="#666")

    if pareto:
        ax.legend(loc="lower left", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#          图4: 雷达图 — 多维综合对比
# ═══════════════════════════════════════════════════════════

def _normalize_metric(values, higher_better=True):
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [1.0] * len(values)
    normed = [(v - vmin) / (vmax - vmin) for v in values]
    return normed if higher_better else [1 - n for n in normed]


def plot_radar(rows, output_path):
    """雷达图：对 Top-4 无损算法做分面对比（2×2）。"""
    lossless = [r for r in rows if r["lossless"]]
    if len(lossless) < 3:
        return

    metrics = ["压缩率", "编码速度", "解码速度", "体积效率"]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles.append(angles[0])

    cr_norm = _normalize_metric([r["compression_ratio"] for r in lossless], higher_better=True)
    enc_norm = _normalize_metric([r["encode_seconds"] for r in lossless], higher_better=False)
    dec_norm = _normalize_metric([r["decode_seconds"] for r in lossless], higher_better=False)
    size_norm = _normalize_metric([r["compressed_size_bytes"] for r in lossless], higher_better=False)

    # 选 Top-4（综合得分）
    scores = [(cr_norm[i] + enc_norm[i] + dec_norm[i] + size_norm[i]) / 4
              for i in range(len(lossless))]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), subplot_kw=dict(polar=True))

    for ax_idx, data_idx in enumerate(top_indices):
        ax = axes[ax_idx // 2][ax_idx % 2]
        r = lossless[data_idx]
        values = [cr_norm[data_idx], enc_norm[data_idx], dec_norm[data_idx], size_norm[data_idx]]
        values.append(values[0])

        # 灰色参考线（所有算法的平均）
        avg_vals = [np.mean(cr_norm), np.mean(enc_norm), np.mean(dec_norm), np.mean(size_norm)]
        avg_vals.append(avg_vals[0])
        ax.plot(angles, avg_vals, linewidth=1, color="#ccc", linestyle="--", label="平均")
        ax.fill(angles, avg_vals, color="#eee", alpha=0.3)

        ax.plot(angles, values, linewidth=2.5, color=r["color"], label=r["label"])
        ax.fill(angles, values, color=r["color"], alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="#aaa")
        ax.set_title(r["label"], fontsize=11, fontweight="bold", pad=12, color=r["color"])
        score_text = f"综合: {scores[data_idx]:.0%}"
        ax.text(0.5, -0.08, score_text, transform=ax.transAxes, ha="center", fontsize=9, color="#666")

    fig.suptitle("Top-4 无损算法多维性能画像 (灰色=平均基线)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#       图5: 噪声扫描 — 压缩率趋势折线图
# ═══════════════════════════════════════════════════════════

def plot_noise_trend(noise_rows, output_path):
    """3 子图：压缩率 / 编码耗时(不含PackBits) / PackBits单独面板。"""
    grouped = {}
    for r in noise_rows:
        grouped.setdefault(r["alg_id"], []).append(r)
    for v in grouped.values():
        v.sort(key=lambda r: r["noise_hits"])

    lossless_ids = [aid for aid in grouped if aid in LOSSLESS_IDS]
    lossless_ids.sort(key=lambda aid: SHORT_LABELS.get(aid, aid))

    # 拆分：PackBits 耗时单独
    lossless_no_pb = [aid for aid in lossless_ids if aid != "packbits_zlib"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    for aid in lossless_ids:
        data = grouped[aid]
        x = [r["noise_hits"] for r in data]
        y_cr = [r["compression_ratio"] for r in data]
        color = PALETTE.get(aid, "#999")
        label = SHORT_LABELS.get(aid, aid)
        ax1.plot(x, y_cr, marker="o", markersize=4, linewidth=1.8, color=color, label=label)

    ax1.set_xlabel("每帧噪声命中数")
    ax1.set_ylabel("压缩率 (x)")
    ax1.set_title("噪声↑ 时 GES+Zlib 仍保持最高")
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # 编码耗时（不含 PackBits）
    for aid in lossless_no_pb:
        data = grouped[aid]
        x = [r["noise_hits"] for r in data]
        y_enc = [r["encode_seconds"] for r in data]
        color = PALETTE.get(aid, "#999")
        label = SHORT_LABELS.get(aid, aid)
        ax2.plot(x, y_enc, marker="s", markersize=4, linewidth=1.8, color=color, label=label)

    ax2.set_xlabel("每帧噪声命中数")
    ax2.set_ylabel("编码耗时 (秒)")
    ax2.set_title("编码耗时（排除 PackBits 异常值）")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # PackBits 单独面板
    if "packbits_zlib" in grouped:
        data_pb = grouped["packbits_zlib"]
        x_pb = [r["noise_hits"] for r in data_pb]
        y_pb = [r["encode_seconds"] for r in data_pb]
        ax3.plot(x_pb, y_pb, marker="D", markersize=5, linewidth=2.2,
                 color=PALETTE["packbits_zlib"], label="PackBits+Zlib")
        ax3.fill_between(x_pb, y_pb, alpha=0.15, color=PALETTE["packbits_zlib"])
        ax3.set_xlabel("每帧噪声命中数")
        ax3.set_ylabel("编码耗时 (秒)")
        ax3.set_title("PackBits+Zlib 耗时随噪声指数增长")
        ax3.set_xscale("log", base=2)
        ax3.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax3.legend(loc="upper left", fontsize=8)

    # 共享图例（前两个面板）
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=8, frameon=True,
               ncol=4, bbox_to_anchor=(0.38, -0.04))

    fig.suptitle("SPAD 无损压缩算法噪声鲁棒性对比", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#       图6: 噪声扫描热力图 — 算法×噪声→压缩率
# ═══════════════════════════════════════════════════════════

def plot_noise_heatmap(noise_rows, output_path):
    """热力图：行=算法(按平均 CR 排序)，列=噪声级别，颜色=log(压缩率)。
    有损/无损算法用水平线分隔。"""
    from matplotlib.colors import LogNorm

    grouped = {}
    is_lossless_map = {}
    for r in noise_rows:
        grouped.setdefault(r["alg_id"], {})[r["noise_hits"]] = r["compression_ratio"]
        is_lossless_map[r["alg_id"]] = r["alg_id"] in LOSSLESS_IDS

    noise_levels = sorted({r["noise_hits"] for r in noise_rows})
    # 排序：无损按平均CR降序，有损在最后
    lossless_aids = sorted(
        [a for a in grouped if is_lossless_map.get(a, True)],
        key=lambda a: np.mean(list(grouped[a].values())), reverse=True)
    lossy_aids = sorted(
        [a for a in grouped if not is_lossless_map.get(a, True)],
        key=lambda a: np.mean(list(grouped[a].values())), reverse=True)
    alg_ids = lossless_aids + lossy_aids

    matrix = np.zeros((len(alg_ids), len(noise_levels)))
    for i, aid in enumerate(alg_ids):
        for j, nl in enumerate(noise_levels):
            matrix[i, j] = max(grouped[aid].get(nl, 1), 1)  # floor at 1 for log

    fig, ax = plt.subplots(figsize=(13, 5.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu",
                   norm=LogNorm(vmin=max(1, matrix.min()), vmax=matrix.max()))

    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f"{int(n)}" if n == int(n) else f"{n:.0f}" for n in noise_levels],
                       fontsize=9)
    ax.set_yticks(range(len(alg_ids)))
    ylabels = []
    for a in alg_ids:
        lbl = SHORT_LABELS.get(a, a)
        if not is_lossless_map.get(a, True):
            lbl += " [L]"
        ylabels.append(lbl)
    ax.set_yticklabels(ylabels, fontsize=9)

    # 分隔线
    if lossy_aids and lossless_aids:
        sep = len(lossless_aids) - 0.5
        ax.axhline(sep, color="white", linewidth=2.5)

    # 数值标注
    log_mid = np.exp((np.log(matrix.min() + 1) + np.log(matrix.max())) / 2)
    for i in range(len(alg_ids)):
        for j in range(len(noise_levels)):
            val = matrix[i, j]
            text_color = "white" if val > log_mid else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7,
                    color=text_color, fontweight="bold" if val > 50 else "normal")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="压缩率 (x) — 对数色标")
    ax.set_xlabel("每帧噪声命中数")
    ax.set_title("高噪声下所有算法压缩率普降 | GES+Zlib 在全噪声范围保持无损领先 ([L]=有损)")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#                      主函数
# ═══════════════════════════════════════════════════════════

def main():
    config = load_config()
    dp = resolve_dataset_paths(config)
    ds_name = dataset_name_from_paths(dp)
    out_dir = resolve_compression_output_dir(config)
    run_dir = find_latest_run_dir(out_dir, ds_name)

    output = PROJECT_ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)

    # ── 基线图表 ──
    if run_dir is not None:
        rows = load_baseline_rows(run_dir)
        plot_compression_ratio(rows, output / "1_compression_ratio.png")
        print("[OK] 1_compression_ratio.png")
        plot_encode_decode_dumbbell(rows, output / "2_encode_decode_dumbbell.png")
        print("[OK] 2_encode_decode_dumbbell.png")
        plot_tradeoff_pareto(rows, output / "3_tradeoff_pareto.png")
        print("[OK] 3_tradeoff_pareto.png")
        plot_radar(rows, output / "4_radar_multidim.png")
        print("[OK] 4_radar_multidim.png")
    else:
        print("[SKIP] No baseline run found")

    # -- noise sweep --
    noise_csv = PROJECT_ROOT / "noise_compression_results.csv"
    if noise_csv.exists():
        noise_rows = load_noise_csv(noise_csv)
        plot_noise_trend(noise_rows, output / "5_noise_trend.png")
        print("[OK] 5_noise_trend.png")
        plot_noise_heatmap(noise_rows, output / "6_noise_heatmap.png")
        print("[OK] 6_noise_heatmap.png")
    else:
        print(f"[SKIP] {noise_csv} not found")

    print(f"\n所有图表输出到: {output.resolve()}")


if __name__ == "__main__":
    main()

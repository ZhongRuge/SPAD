"""压缩结果查看器（交互式）。

对比原始帧与压缩后解码帧，显示差异热力图。
TemporalBinning 模式下展示 4 面板（原始/累计/存储/重建）。

用法：
  python scripts/view_compressed.py                   # 使用 config 中的默认算法
  python scripts/view_compressed.py --algorithm rle    # 指定算法
"""

import argparse
import json
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from spad.compress import build_compressor
from spad.config import (
    load_config,
    resolve_algorithm_params,
    resolve_compression_output_dir,
    resolve_dataset_paths,
    resolve_evaluator_config,
)
from spad.evaluate import (
    build_variant_name,
    dataset_name_from_paths,
    find_latest_run_dir,
)
from spad.io import SpadReader, stream_compressed_chunks


DISPLAY_NAMES = {
    "rle": "RLE", "delta_rle": "Delta+RLE", "delta_sparse": "Delta+Sparse",
    "delta_sparse_zlib": "Delta+Sparse+Zlib",
    "delta_sparse_varint_zlib": "D+S+Varint+Zlib",
    "packbits_zlib": "PackBits+Zlib", "global_event_stream": "GES+Zlib",
    "aer": "AER", "temporal_binning": "TemporalBinning",
}


def _metric_text(frame):
    return f"Sparsity: {np.mean(frame):.2%}" if frame.max() <= 1 else f"Mean: {np.mean(frame):.2f}"


# ── 标准模式：3 面板 (原始/解码/差异) ──────────────────────

def _load_paired(reader, compressor, compressed_path, batch_size):
    raw_all, dec_all = [], []
    raw_gen = reader.stream_batches(batch_size)
    chunk_gen = stream_compressed_chunks(compressed_path)
    for raw, (fc, chunk) in zip(raw_gen, chunk_gen):
        shape = (fc, reader.height, reader.width)
        dec_all.append(compressor.decode(chunk, shape))
        raw_all.append(raw)
    return np.concatenate(raw_all), np.concatenate(dec_all)


def _show_standard(reader, compressor, compressed_path, batch_size, name):
    raw, dec = _load_paired(reader, compressor, compressed_path, batch_size)
    diff = np.abs(dec.astype(np.int16) - raw.astype(np.int16)).astype(np.uint8)
    total = raw.shape[0]
    dec_vmax = 255 if dec.max() > 1 else 1
    diff_vmax = max(1, int(diff.max()))
    state = {"i": 0, "play": False}

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(bottom=0.18, top=0.82, wspace=0.08)
    ims = [
        axes[0].imshow(raw[0], cmap="gray", vmin=0, vmax=1),
        axes[1].imshow(dec[0], cmap="gray", vmin=0, vmax=dec_vmax),
        axes[2].imshow(diff[0], cmap="hot", vmin=0, vmax=diff_vmax),
    ]
    for ax in axes: ax.axis("off")
    axes[0].set_title("Raw"); axes[1].set_title("Decoded"); axes[2].set_title("Diff")

    def update():
        i = state["i"]
        ims[0].set_data(raw[i]); ims[1].set_data(dec[i]); ims[2].set_data(diff[i])
        axes[0].set_xlabel(_metric_text(raw[i]))
        axes[1].set_xlabel(_metric_text(dec[i]))
        dc = int(np.count_nonzero(diff[i]))
        axes[2].set_xlabel(f"Changed: {dc} ({dc/diff[i].size:.2%})")
        fig.suptitle(f"[{name}]  Frame {i}/{total-1}", fontsize=14)
        fig.canvas.draw_idle()

    def on_key(e):
        k = e.key
        if k == "right":    state["i"] = (state["i"] + 1) % total
        elif k == "left":   state["i"] = (state["i"] - 1) % total
        elif k == "down":   state["i"] = (state["i"] + 100) % total
        elif k == "up":     state["i"] = (state["i"] - 100) % total
        elif k == "pageup": state["i"] = (state["i"] + 1000) % total
        elif k == "pagedown": state["i"] = (state["i"] - 1000) % total
        elif k == " ":      state["play"] = not state["play"]
        else: return
        update()

    timer = fig.canvas.new_timer(interval=30)
    timer.add_callback(lambda: (update() if state["play"] else None,
                                 state.update(i=(state["i"]+10)%total) if state["play"] else None))
    timer.start()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.figtext(0.5, 0.04, "Left/Right: +/-1 | Up/Down: +/-100 | PgUp/Dn: +/-1000 | Space: Play",
                ha="center", fontsize=9, bbox=dict(facecolor="lightblue", alpha=0.5, pad=4))
    update()
    plt.show()


# ── TemporalBinning 模式：4 面板 ──────────────────────────

def _decode_bins(chunk, h, w):
    if len(chunk) < 2:
        return np.zeros((0, h, w), dtype=np.uint8)
    nb = int(np.frombuffer(chunk[:2], dtype=np.uint16)[0])
    raw = zlib.decompress(chunk[2:])
    return np.frombuffer(raw, dtype=np.uint8).reshape((nb, h, w))


def _show_temporal(reader, compressor, compressed_path, batch_size, name):
    raw_starts, raw_bins, stored_bins, approx_all, ranges = [], [], [], [], []
    raw_gen = reader.stream_batches(batch_size)
    chunk_gen = stream_compressed_chunks(compressed_path)
    offset = 0
    for raw, (fc, chunk) in zip(raw_gen, chunk_gen):
        sb = _decode_bins(chunk, reader.height, reader.width)
        ap = compressor.decode(chunk, (fc, reader.height, reader.width))
        idxs, rbs = [], []
        for bi in range(sb.shape[0]):
            s, e = bi * compressor.bin_size, min((bi+1)*compressor.bin_size, fc)
            if s >= fc: break
            idxs.append(s)
            rbs.append(np.sum(raw[s:e], axis=0, dtype=np.uint8))
            ranges.append((offset+s, offset+e-1))
        idxs = np.array(idxs, dtype=np.int64)
        raw_starts.append(raw[idxs])
        raw_bins.append(np.stack(rbs))
        stored_bins.append(sb)
        approx_all.append(ap[idxs])
        offset += fc

    rs = np.concatenate(raw_starts); rb = np.concatenate(raw_bins)
    sb = np.concatenate(stored_bins); ap = np.concatenate(approx_all)
    total = rb.shape[0]; bvmax = max(1, int(max(rb.max(), sb.max())))
    state = {"i": 0, "play": False}

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.18, top=0.82, wspace=0.08)
    ims = [
        axes[0].imshow(rs[0], cmap="gray", vmin=0, vmax=1),
        axes[1].imshow(rb[0], cmap="gray", vmin=0, vmax=bvmax),
        axes[2].imshow(sb[0], cmap="gray", vmin=0, vmax=bvmax),
        axes[3].imshow(ap[0], cmap="gray", vmin=0, vmax=1),
    ]
    for ax in axes: ax.axis("off")
    axes[0].set_title("Bin start"); axes[1].set_title("Raw accumulation")
    axes[2].set_title("Stored bin"); axes[3].set_title("Approx recon")

    def update():
        i = state["i"]
        ims[0].set_data(rs[i]); ims[1].set_data(rb[i]); ims[2].set_data(sb[i]); ims[3].set_data(ap[i])
        d = np.abs(sb[i].astype(np.int16)-rb[i].astype(np.int16))
        sm = int(np.count_nonzero((rb[i]>0).astype(np.uint8) != ap[i]))
        axes[0].set_xlabel(_metric_text(rs[i]))
        axes[1].set_xlabel(f"Hits: {int(rb[i].sum())} | Max: {int(rb[i].max())}")
        axes[2].set_xlabel(f"Diff sum: {int(d.sum())} | Max: {int(sb[i].max())}")
        axes[3].set_xlabel(f"Support diff: {sm} ({sm/ap[i].size:.2%})")
        fs, fe = ranges[i]
        fig.suptitle(f"[{name}]  Bin {i}/{total-1}  |  frames {fs}-{fe}", fontsize=14)
        fig.canvas.draw_idle()

    def on_key(e):
        k = e.key
        if k == "right":    state["i"] = (state["i"]+1) % total
        elif k == "left":   state["i"] = (state["i"]-1) % total
        elif k == "down":   state["i"] = (state["i"]+10) % total
        elif k == "up":     state["i"] = (state["i"]-10) % total
        elif k == "pageup": state["i"] = (state["i"]+50) % total
        elif k == "pagedown": state["i"] = (state["i"]-50) % total
        elif k == " ":      state["play"] = not state["play"]
        else: return
        update()

    timer = fig.canvas.new_timer(interval=150)
    timer.add_callback(lambda: (state.update(i=(state["i"]+1)%total) or update()) if state["play"] else None)
    timer.start()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.figtext(0.5, 0.04, "Left/Right: +/-1 bin | Up/Down: +/-10 | PgUp/Dn: +/-50 | Space: Play",
                ha="center", fontsize=9, bbox=dict(facecolor="lightblue", alpha=0.5, pad=4))
    update()
    plt.show()


# ── 入口 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="View compressed SPAD data vs original")
    parser.add_argument("--algorithm", default=None, help="Algorithm id (default: from config)")
    args = parser.parse_args()

    config = load_config()
    dp = resolve_dataset_paths(config)
    comp_dir = resolve_compression_output_dir(config)
    ev_cfg = resolve_evaluator_config(config)

    # 选算法
    if args.algorithm:
        alg_id = args.algorithm
    else:
        vis_cfg = config.get("visualization", {}).get("compressed", {})
        alg_id = vis_cfg.get("algorithm", "rle")

    params = resolve_algorithm_params(config, alg_id)
    compressor = build_compressor(alg_id, params)
    ds_name = dataset_name_from_paths(dp)
    variant = build_variant_name(alg_id, params)

    run_dir = find_latest_run_dir(comp_dir, ds_name)
    if run_dir is None:
        raise SystemExit(f"No compression runs for dataset: {ds_name}")

    compressed_path = str(run_dir / variant / "compressed.bin")
    if not Path(compressed_path).exists():
        raise SystemExit(f"Missing: {compressed_path}")

    reader = SpadReader(dp["meta_path"], dp["data_path"])
    name = DISPLAY_NAMES.get(alg_id, alg_id)
    batch_size = ev_cfg.get("batch_size", 1000)

    print(f"Algorithm: {name} | Dataset: {dp['data_path']}")
    if alg_id == "temporal_binning":
        _show_temporal(reader, compressor, compressed_path, batch_size, name)
    else:
        _show_standard(reader, compressor, compressed_path, batch_size, name)


if __name__ == "__main__":
    main()

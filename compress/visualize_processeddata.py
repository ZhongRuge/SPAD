import os
import sys
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithm_registry import build_compressor
from algorithm_registry import build_output_filename
from experiment_output import build_variant_name
from experiment_output import dataset_name_from_paths
from experiment_output import find_latest_run_dir
from io_manager import SpadIOManager


CURRENT_DIR = Path(__file__).resolve().parent
DATA_GENERATE_DIR = CURRENT_DIR.parent / "data_generate"
if str(DATA_GENERATE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATE_DIR))

from simulation_io import load_config
from simulation_io import resolve_algorithm_params
from simulation_io import resolve_compression_output_dir
from simulation_io import resolve_dataset_paths
from simulation_io import resolve_evaluator_config
from simulation_io import resolve_visualization_algorithm


ALGORITHM_DISPLAY_NAMES = {
    "rle": "RLE",
    "delta_rle": "Delta+RLE",
    "delta_sparse": "Delta+Sparse",
    "delta_sparse_zlib": "Delta+Sparse+Zlib",
    "delta_sparse_varint_zlib": "Delta+Sparse+Varint+Zlib",
    "packbits_zlib": "PackBits+Zlib",
    "global_event_stream": "GlobalEventStream",
    "aer": "AER",
    "temporal_binning": "TemporalBinning",
}


def _load_raw_and_decoded_frames(io_manager, compressor, compressed_path, batch_size):
    raw_batches = []
    decoded_batches = []

    raw_stream = io_manager.stream_batches(batch_size=batch_size)
    chunk_stream = io_manager.stream_compressed_chunks(compressed_path)

    for raw_batch, chunk_info in zip(raw_stream, chunk_stream):
        frame_count, compressed_chunk = chunk_info
        batch_shape = io_manager.get_batch_shape(frame_count)

        if raw_batch.shape != batch_shape:
            raise ValueError(
                f"Raw batch shape {raw_batch.shape} does not match compressed chunk shape {batch_shape}"
            )

        decoded_batch = compressor.decode(compressed_chunk, batch_shape)
        raw_batches.append(raw_batch)
        decoded_batches.append(decoded_batch)

    try:
        next(raw_stream)
        raise ValueError("Raw stream contains more batches than the compressed file")
    except StopIteration:
        pass

    try:
        next(chunk_stream)
        raise ValueError("Compressed file contains more chunks than the raw stream")
    except StopIteration:
        pass

    if not raw_batches:
        raise ValueError("No frames available for visualization")

    raw_video = np.concatenate(raw_batches, axis=0)
    decoded_video = np.concatenate(decoded_batches, axis=0)
    return raw_video, decoded_video


def _build_diff_frames(raw_video, decoded_video):
    if raw_video.shape != decoded_video.shape:
        raise ValueError(
            f"Raw and decoded video shapes do not match: {raw_video.shape} vs {decoded_video.shape}"
        )

    return np.abs(decoded_video.astype(np.int16) - raw_video.astype(np.int16)).astype(np.uint8)


def _frame_metric_text(frame):
    if frame.max() > 1:
        return f"Mean intensity: {np.mean(frame):.2f}"
    return f"Sparsity: {np.mean(frame):.2%}"


def _decode_temporal_binning_chunk_to_bins(compressed_chunk, height, width):
    if not compressed_chunk:
        return np.zeros((0, height, width), dtype=np.uint8)
    if len(compressed_chunk) < 2:
        raise ValueError("TemporalBinning chunk is missing the bin-count header")

    num_bins = int(np.frombuffer(compressed_chunk[:2], dtype=np.uint16)[0])
    raw_bytes = zlib.decompress(compressed_chunk[2:])
    expected_bytes = num_bins * height * width
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"TemporalBinning bin payload size mismatch: expected={expected_bytes}, actual={len(raw_bytes)}"
        )
    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape((num_bins, height, width))


def _load_temporal_binning_views(io_manager, compressor, compressed_path, batch_size):
    raw_start_frames = []
    raw_binned_frames = []
    stored_binned_frames = []
    approx_reconstructed_frames = []
    bin_frame_ranges = []

    raw_stream = io_manager.stream_batches(batch_size=batch_size)
    chunk_stream = io_manager.stream_compressed_chunks(compressed_path)
    global_frame_offset = 0

    for raw_batch, chunk_info in zip(raw_stream, chunk_stream):
        frame_count, compressed_chunk = chunk_info
        batch_shape = io_manager.get_batch_shape(frame_count)

        if raw_batch.shape != batch_shape:
            raise ValueError(
                f"Raw batch shape {raw_batch.shape} does not match compressed chunk shape {batch_shape}"
            )

        stored_bins = _decode_temporal_binning_chunk_to_bins(
            compressed_chunk,
            io_manager.height,
            io_manager.width,
        )
        approx_batch = compressor.decode(compressed_chunk, batch_shape)

        local_start_indices = []
        expected_raw_bins = []
        for bin_index in range(stored_bins.shape[0]):
            start = bin_index * compressor.bin_size
            end = min((bin_index + 1) * compressor.bin_size, frame_count)
            if start >= frame_count:
                break
            local_start_indices.append(start)
            expected_raw_bins.append(np.sum(raw_batch[start:end], axis=0, dtype=np.uint8))
            bin_frame_ranges.append((global_frame_offset + start, global_frame_offset + end - 1))

        local_start_indices = np.array(local_start_indices, dtype=np.int64)
        raw_start_frames.append(raw_batch[local_start_indices])
        raw_binned_frames.append(np.stack(expected_raw_bins, axis=0))
        stored_binned_frames.append(stored_bins)
        approx_reconstructed_frames.append(approx_batch[local_start_indices])

        global_frame_offset += frame_count

    try:
        next(raw_stream)
        raise ValueError("Raw stream contains more batches than the compressed file")
    except StopIteration:
        pass

    try:
        next(chunk_stream)
        raise ValueError("Compressed file contains more chunks than the raw stream")
    except StopIteration:
        pass

    if not raw_binned_frames:
        raise ValueError("No temporal bins available for visualization")

    return {
        "raw_start_video": np.concatenate(raw_start_frames, axis=0),
        "raw_binned_video": np.concatenate(raw_binned_frames, axis=0),
        "stored_binned_video": np.concatenate(stored_binned_frames, axis=0),
        "approx_video": np.concatenate(approx_reconstructed_frames, axis=0),
        "bin_frame_ranges": bin_frame_ranges,
    }


def _visualize_temporal_binning(io_manager, compressor, compressed_path, batch_size, display_name):
    print(f"Loading raw dataset: {io_manager.data_path}")
    print(f"Decoding compressed output: {compressed_path}")

    views = _load_temporal_binning_views(
        io_manager=io_manager,
        compressor=compressor,
        compressed_path=compressed_path,
        batch_size=batch_size,
    )

    raw_start_video = views["raw_start_video"]
    raw_binned_video = views["raw_binned_video"]
    stored_binned_video = views["stored_binned_video"]
    approx_video = views["approx_video"]
    bin_frame_ranges = views["bin_frame_ranges"]

    total_bins = raw_binned_video.shape[0]
    binned_vmax = max(1, int(max(raw_binned_video.max(), stored_binned_video.max())))

    print(
        f"Loaded temporal-binning visualization: "
        f"bins={total_bins}, raw_shape={raw_binned_video.shape}, stored_shape={stored_binned_video.shape}"
    )

    state = {
        "curr_idx": 0,
        "playing": False,
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.20, top=0.82, wspace=0.08)

    raw_start_im = axes[0].imshow(raw_start_video[0], cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    raw_binned_im = axes[1].imshow(raw_binned_video[0], cmap="gray", interpolation="nearest", vmin=0, vmax=binned_vmax)
    stored_binned_im = axes[2].imshow(stored_binned_video[0], cmap="gray", interpolation="nearest", vmin=0, vmax=binned_vmax)
    approx_im = axes[3].imshow(approx_video[0], cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    for ax in axes:
        ax.axis("off")

    axes[0].set_title("Bin start raw frame")
    axes[1].set_title("Raw temporal accumulation")
    axes[2].set_title("Stored binned frame")
    axes[3].set_title("Approx reconstruction")

    def update_display():
        idx = state["curr_idx"]
        frame_start, frame_end = bin_frame_ranges[idx]
        raw_start_frame = raw_start_video[idx]
        raw_binned_frame = raw_binned_video[idx]
        stored_binned_frame = stored_binned_video[idx]
        approx_frame = approx_video[idx]

        raw_start_im.set_data(raw_start_frame)
        raw_binned_im.set_data(raw_binned_frame)
        stored_binned_im.set_data(stored_binned_frame)
        approx_im.set_data(approx_frame)

        accumulation_diff = np.abs(
            stored_binned_frame.astype(np.int16) - raw_binned_frame.astype(np.int16)
        )
        support_diff = np.count_nonzero((raw_binned_frame > 0).astype(np.uint8) != approx_frame)

        axes[0].set_xlabel(_frame_metric_text(raw_start_frame))
        axes[1].set_xlabel(
            f"Total hits: {int(np.sum(raw_binned_frame, dtype=np.int64))} | Max: {int(raw_binned_frame.max())}"
        )
        axes[2].set_xlabel(
            f"Abs diff sum: {int(np.sum(accumulation_diff, dtype=np.int64))} | Max: {int(stored_binned_frame.max())}"
        )
        axes[3].set_xlabel(
            f"Support mismatch: {support_diff} ({support_diff / approx_frame.size:.2%})"
        )

        fig.suptitle(
            f"[{display_name}]  Bin: {idx} / {total_bins - 1}\n"
            f"Raw frame range: {frame_start} - {frame_end}",
            fontsize=15,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["curr_idx"] = (state["curr_idx"] + 1) % total_bins
        elif event.key == "left":
            state["curr_idx"] = (state["curr_idx"] - 1) % total_bins
        elif event.key == "down":
            state["curr_idx"] = (state["curr_idx"] + 10) % total_bins
        elif event.key == "up":
            state["curr_idx"] = (state["curr_idx"] - 10) % total_bins
        elif event.key == "pageup":
            state["curr_idx"] = (state["curr_idx"] + 50) % total_bins
        elif event.key == "pagedown":
            state["curr_idx"] = (state["curr_idx"] - 50) % total_bins
        elif event.key == " ":
            state["playing"] = not state["playing"]
            print("Auto-play:", "ON" if state["playing"] else "OFF")
        else:
            return

        update_display()

    def on_tick():
        if state["playing"]:
            state["curr_idx"] = (state["curr_idx"] + 1) % total_bins
            update_display()

    timer = fig.canvas.new_timer(interval=150)
    timer.add_callback(on_tick)
    timer.start()

    fig.canvas.mpl_connect("key_press_event", on_key)

    instructions = (
        "TemporalBinning view:\n"
        "Left/Right: +/-1 bin | Up/Down: +/-10 bins\n"
        "PgUp/PgDn: +/-50 bins | Space: Play/Pause\n"
        "Panels: raw start frame | raw bin sum | stored bin | approximate reconstruction"
    )
    plt.figtext(
        0.5,
        0.05,
        instructions,
        ha="center",
        fontsize=9,
        bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5},
    )

    update_display()
    print(f"Loaded {total_bins} temporal bins. Ready to visualize.")
    plt.show()


def visualize():
    config = load_config(DATA_GENERATE_DIR / "config.yaml")
    dataset_paths = resolve_dataset_paths(config)
    compression_output_dir = resolve_compression_output_dir(config)
    evaluator_config = resolve_evaluator_config(config)
    algorithm_id = resolve_visualization_algorithm(config)
    algorithm_params = resolve_algorithm_params(config, algorithm_id)
    compressor = build_compressor(algorithm_id, algorithm_params)
    dataset_name = dataset_name_from_paths(dataset_paths)
    variant_name = build_variant_name(algorithm_id, algorithm_params)

    meta_path = dataset_paths["meta_path"]
    data_path = dataset_paths["data_path"]
    latest_run_dir = find_latest_run_dir(compression_output_dir, dataset_name)
    if latest_run_dir is None:
        print(f"No compression runs found for dataset: {dataset_name}")
        return

    compressed_path = latest_run_dir / variant_name / build_output_filename(compressor)
    if not os.path.exists(compressed_path):
        print(f"Missing compressed file: {compressed_path}")
        return

    io_manager = SpadIOManager(meta_path, data_path)
    display_name = ALGORITHM_DISPLAY_NAMES.get(algorithm_id, algorithm_id)

    if algorithm_id == "temporal_binning":
        _visualize_temporal_binning(
            io_manager=io_manager,
            compressor=compressor,
            compressed_path=compressed_path,
            batch_size=evaluator_config["batch_size"],
            display_name=display_name,
        )
        return

    print(f"Loading raw dataset: {data_path}")
    print(f"Decoding compressed output: {compressed_path}")

    raw_video, decoded_video = _load_raw_and_decoded_frames(
        io_manager=io_manager,
        compressor=compressor,
        compressed_path=compressed_path,
        batch_size=evaluator_config["batch_size"],
    )
    diff_video = _build_diff_frames(raw_video, decoded_video)

    total_frames = raw_video.shape[0]
    decoded_is_grayscale = decoded_video.max() > 1
    raw_vmax = 1
    decoded_vmax = 255 if decoded_is_grayscale else 1
    diff_vmax = max(1, int(diff_video.max()))

    print(
        f"Loaded video for visualization: raw_shape={raw_video.shape}, "
        f"decoded_shape={decoded_video.shape}, diff_nonzero={int(np.count_nonzero(diff_video))}"
    )

    state = {
        "curr_idx": 0,
        "playing": False,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(bottom=0.18, top=0.82, wspace=0.08)

    raw_im = axes[0].imshow(raw_video[0], cmap="gray", interpolation="nearest", vmin=0, vmax=raw_vmax)
    decoded_im = axes[1].imshow(
        decoded_video[0],
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=decoded_vmax,
    )
    diff_im = axes[2].imshow(diff_video[0], cmap="hot", interpolation="nearest", vmin=0, vmax=diff_vmax)

    for ax in axes:
        ax.axis("off")

    axes[0].set_title("Raw data")
    axes[1].set_title("Decoded data")
    axes[2].set_title("Absolute difference")

    def update_display():
        idx = state["curr_idx"]
        raw_frame = raw_video[idx]
        decoded_frame = decoded_video[idx]
        diff_frame = diff_video[idx]

        raw_im.set_data(raw_frame)
        decoded_im.set_data(decoded_frame)
        diff_im.set_data(diff_frame)

        raw_text = _frame_metric_text(raw_frame)
        decoded_text = _frame_metric_text(decoded_frame)
        diff_count = int(np.count_nonzero(diff_frame))
        diff_ratio = diff_count / diff_frame.size

        axes[0].set_xlabel(raw_text)
        axes[1].set_xlabel(decoded_text)
        axes[2].set_xlabel(f"Changed pixels: {diff_count} ({diff_ratio:.2%})")

        fig.suptitle(
            f"[{display_name}]  Frame: {idx} / {total_frames - 1}\n"
            f"Raw vs decoded vs difference",
            fontsize=15,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["curr_idx"] = (state["curr_idx"] + 1) % total_frames
        elif event.key == "left":
            state["curr_idx"] = (state["curr_idx"] - 1) % total_frames
        elif event.key == "down":
            state["curr_idx"] = (state["curr_idx"] + 100) % total_frames
        elif event.key == "up":
            state["curr_idx"] = (state["curr_idx"] - 100) % total_frames
        elif event.key == "pageup":
            state["curr_idx"] = (state["curr_idx"] + 1000) % total_frames
        elif event.key == "pagedown":
            state["curr_idx"] = (state["curr_idx"] - 1000) % total_frames
        elif event.key == " ":
            state["playing"] = not state["playing"]
            print("Auto-play:", "ON" if state["playing"] else "OFF")
        else:
            return

        update_display()

    def on_tick():
        if state["playing"]:
            state["curr_idx"] = (state["curr_idx"] + 10) % total_frames
            update_display()

    timer = fig.canvas.new_timer(interval=30)
    timer.add_callback(on_tick)
    timer.start()

    fig.canvas.mpl_connect("key_press_event", on_key)

    instructions = (
        "Controls:\n"
        "Left/Right: +/-1 frame | Up/Down: +/-100 frames\n"
        "PgUp/PgDn: +/-1000 frames | Space: Play/Pause\n"
        "Diff panel: brighter means larger reconstruction error"
    )
    plt.figtext(
        0.5,
        0.05,
        instructions,
        ha="center",
        fontsize=9,
        bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5},
    )

    update_display()
    print(f"Loaded {total_frames} frames. Ready to visualize.")
    plt.show()


if __name__ == "__main__":
    visualize()

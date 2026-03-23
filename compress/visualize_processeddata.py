import os
import zlib

import matplotlib.pyplot as plt
import numpy as np

from io_manager import SpadIOManager
from algorithms import (
    AerCompressor,
    BlockSparseBitmapCompressor,
    DeltaGapZlibCompressor,
    DeltaRleCompressor,
    DeltaSparseCompressor,
    DeltaSparseZlibCompressor,
    FrameZeroSuppressionCompressor,
    GlobalEventStreamCompressor,
    H264VideoCompressor,
    H265VideoCompressor,
    MortonPackBitsZlibCompressor,
    PackBitsZlibCompressor,
    RleCompressor,
    RowSparseZlibCompressor,
    TemporalBinningCompressor,
)


SELECTED_ALGORITHM = "BlockSparseBitmapCompressor"


def build_algorithm_registry():
    return {
        "RleCompressor": lambda: RleCompressor(),
        "PackBitsZlibCompressor": lambda: PackBitsZlibCompressor(),
        "MortonPackBitsZlibCompressor": lambda: MortonPackBitsZlibCompressor(),
        "DeltaRleCompressor": lambda: DeltaRleCompressor(),
        "DeltaSparseCompressor": lambda: DeltaSparseCompressor(),
        "DeltaSparseZlibCompressor": lambda: DeltaSparseZlibCompressor(),
        "DeltaGapZlibCompressor": lambda: DeltaGapZlibCompressor(),
        "FrameZeroSuppressionCompressor": lambda: FrameZeroSuppressionCompressor(),
        "GlobalEventStreamCompressor": lambda: GlobalEventStreamCompressor(),
        "AerCompressor": lambda: AerCompressor(use_delta=False),
        "RowSparseZlibCompressor": lambda: RowSparseZlibCompressor(),
        "BlockSparseBitmapCompressor": lambda: BlockSparseBitmapCompressor(),
        "H264VideoCompressor": lambda: H264VideoCompressor(),
        "H265VideoCompressor": lambda: H265VideoCompressor(),
        "TemporalBinningCompressor": lambda: TemporalBinningCompressor(bin_size=255),
    }


def create_algorithm(name: str):
    registry = build_algorithm_registry()
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"未知算法 {name}，可选项: {available}")
    return registry[name]()


def get_visual_algorithm_name(algorithm) -> str:
    return algorithm.__class__.__name__


def _decode_temporal_binning_chunk(compressed_bytes: bytes, batch_shape: tuple, bin_size: int):
    if not compressed_bytes:
        return np.zeros((0, batch_shape[1], batch_shape[2]), dtype=np.uint8), []
    if len(compressed_bytes) < 2:
        raise ValueError("TemporalBinning 压缩流格式错误: 缺少 bin 数量头部")

    num_bins = int(np.frombuffer(compressed_bytes[:2], dtype=np.uint16)[0])
    raw_bytes = zlib.decompress(compressed_bytes[2:])
    _, height, width = batch_shape
    expected_bytes = num_bins * height * width
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"TemporalBinning 可视化尺寸不匹配: expected={expected_bytes}, actual={len(raw_bytes)}"
        )

    decoded_bins = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((num_bins, height, width))
    labels = []
    for bin_idx in range(num_bins):
        start_frame = bin_idx * bin_size
        end_frame = min((bin_idx + 1) * bin_size, batch_shape[0]) - 1
        labels.append((start_frame, end_frame))
    return decoded_bins, labels


def _build_temporal_binning_reference(raw_batch: np.ndarray, bin_size: int):
    reference_bins = []
    total_frames = raw_batch.shape[0]
    for start_frame in range(0, total_frames, bin_size):
        end_frame = min(start_frame + bin_size, total_frames)
        reference_bins.append(np.sum(raw_batch[start_frame:end_frame], axis=0, dtype=np.uint8))
    return np.array(reference_bins, dtype=np.uint8)


def load_visualization_payload(io_manager: SpadIOManager, compressed_path: str, algorithm):
    original_batches = []
    decoded_batches = []
    frame_labels = []
    global_frame_offset = 0

    raw_stream = io_manager.stream_batches(batch_size=1000)
    chunk_stream = io_manager.stream_compressed_chunks(compressed_path)

    for raw_batch, (frame_count, chunk) in zip(raw_stream, chunk_stream):
        batch_shape = io_manager.get_batch_shape(frame_count)

        if isinstance(algorithm, TemporalBinningCompressor):
            decoded_batch, local_labels = _decode_temporal_binning_chunk(
                chunk,
                batch_shape,
                algorithm.bin_size,
            )
            reference_batch = _build_temporal_binning_reference(raw_batch, algorithm.bin_size)
            original_batches.append(reference_batch)
            decoded_batches.append(decoded_batch)

            for start_frame, end_frame in local_labels:
                frame_labels.append(
                    f"Source Frames {global_frame_offset + start_frame}-{global_frame_offset + end_frame}"
                )
        else:
            decoded_batch = algorithm.decode(chunk, batch_shape)
            original_batches.append(raw_batch)
            decoded_batches.append(decoded_batch)
            for frame_idx in range(frame_count):
                frame_labels.append(str(global_frame_offset + frame_idx))

        global_frame_offset += frame_count

    original_matrix = np.concatenate(original_batches, axis=0)
    decoded_matrix = np.concatenate(decoded_batches, axis=0)
    return original_matrix, decoded_matrix, frame_labels


def build_difference_frame(original_frame: np.ndarray, decoded_frame: np.ndarray, is_binary: bool):
    if is_binary:
        return (original_frame != decoded_frame).astype(np.uint8)
    return np.abs(original_frame.astype(np.int16) - decoded_frame.astype(np.int16)).astype(np.uint8)


def build_metrics_text(original_frame: np.ndarray, decoded_frame: np.ndarray, difference_frame: np.ndarray, is_binary: bool):
    if is_binary:
        original_sparsity = np.mean(original_frame)
        decoded_sparsity = np.mean(decoded_frame)
        mismatch_ratio = np.mean(difference_frame)
        return (
            f"Orig Sparsity: {original_sparsity:.2%} | "
            f"Decoded Sparsity: {decoded_sparsity:.2%} | "
            f"Mismatch: {mismatch_ratio:.2%}"
        )

    original_mean = float(np.mean(original_frame))
    decoded_mean = float(np.mean(decoded_frame))
    abs_diff_mean = float(np.mean(difference_frame))
    return (
        f"Orig Mean: {original_mean:.2f} | "
        f"Decoded Mean: {decoded_mean:.2f} | "
        f"Abs Diff Mean: {abs_diff_mean:.2f}"
    )


def visualize():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(current_dir, "../data/spad_dataset.meta.json")
    data_path = os.path.join(current_dir, "../data/spad_dataset.bin")

    my_algorithm = create_algorithm(SELECTED_ALGORITHM)
    file_name = f"{my_algorithm.__class__.__name__}_compressed.bin"
    compressed_path = os.path.join(current_dir, f"../data/{file_name}")

    if not os.path.exists(compressed_path):
        print(f"找不到文件: {compressed_path}")
        return

    io_manager = SpadIOManager(meta_path, data_path)
    width = io_manager.width
    height = io_manager.height

    print(f"Decoding data with {my_algorithm.algorithm_name}")
    original_matrix, decoded_matrix, frame_labels = load_visualization_payload(
        io_manager,
        compressed_path,
        my_algorithm,
    )
    total_frames = decoded_matrix.shape[0]
    if total_frames == 0:
        print("Compressed file is empty. No frames available for visualization.")
        return

    is_binary = bool(np.max(original_matrix) <= 1 and np.max(decoded_matrix) <= 1)
    original_vmax = 1 if is_binary else max(1, int(np.max(original_matrix)))
    decoded_vmax = 1 if is_binary else max(1, int(np.max(decoded_matrix)))
    diff_global = np.abs(original_matrix.astype(np.int16) - decoded_matrix.astype(np.int16))
    diff_vmax = 1 if is_binary else max(1, int(np.max(diff_global)))

    print(f"Loaded data: {width}x{height}, {total_frames} visualization units")

    state = {
        "curr_idx": 0,
        "playing": False,
        "step": 1,
        "autoplay_step": 10,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    plt.subplots_adjust(bottom=0.18, top=0.86, wspace=0.08)

    original_im = axes[0].imshow(
        original_matrix[0],
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=original_vmax,
    )
    decoded_im = axes[1].imshow(
        decoded_matrix[0],
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=decoded_vmax,
    )
    initial_diff = build_difference_frame(original_matrix[0], decoded_matrix[0], is_binary)
    diff_im = axes[2].imshow(
        initial_diff,
        cmap="inferno",
        interpolation="nearest",
        vmin=0,
        vmax=diff_vmax,
    )

    axes[0].set_title("Original")
    axes[1].set_title("Decoded")
    axes[2].set_title("Difference")
    for axis in axes:
        axis.axis("off")

    def update_display():
        idx = state["curr_idx"]
        original_frame = original_matrix[idx]
        decoded_frame = decoded_matrix[idx]
        difference_frame = build_difference_frame(original_frame, decoded_frame, is_binary)

        original_im.set_data(original_frame)
        decoded_im.set_data(decoded_frame)
        diff_im.set_data(difference_frame)

        metric_text = build_metrics_text(original_frame, decoded_frame, difference_frame, is_binary)
        unit_label = "Bin" if isinstance(my_algorithm, TemporalBinningCompressor) else "Frame"
        algorithm_label = get_visual_algorithm_name(my_algorithm)
        fig.suptitle(
            f"[{algorithm_label}]  {unit_label}: {frame_labels[idx]} / {frame_labels[-1]}\n{metric_text}",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["curr_idx"] = (state["curr_idx"] + state["step"]) % total_frames
        elif event.key == "left":
            state["curr_idx"] = (state["curr_idx"] - state["step"]) % total_frames
        elif event.key == "down":
            state["curr_idx"] = (state["curr_idx"] + 100) % total_frames
        elif event.key == "up":
            state["curr_idx"] = (state["curr_idx"] - 100) % total_frames
        elif event.key == "pageup":
            state["curr_idx"] = (state["curr_idx"] + 1000) % total_frames
        elif event.key == "pagedown":
            state["curr_idx"] = (state["curr_idx"] - 1000) % total_frames
        elif event.key == "home":
            state["curr_idx"] = 0
        elif event.key == "end":
            state["curr_idx"] = total_frames - 1
        elif event.key == " ":
            state["playing"] = not state["playing"]
            print("Auto-play:", "ON" if state["playing"] else "OFF")
        elif event.key == "+":
            state["autoplay_step"] = min(state["autoplay_step"] * 2, 1000)
            print(f"Auto-play step: {state['autoplay_step']}")
        elif event.key == "-":
            state["autoplay_step"] = max(1, state["autoplay_step"] // 2)
            print(f"Auto-play step: {state['autoplay_step']}")
        else:
            return
        update_display()

    def on_tick():
        if state["playing"]:
            state["curr_idx"] = (state["curr_idx"] + state["autoplay_step"]) % total_frames
            update_display()

    timer = fig.canvas.new_timer(interval=30)
    timer.add_callback(on_tick)
    timer.start()
    fig.canvas.mpl_connect("key_press_event", on_key)

    instructions = (
        f"Algorithm: {SELECTED_ALGORITHM}\n"
        "Left/Right: ±1  |  Up/Down: ±100  |  PgUp/PgDn: ±1000\n"
        "Home/End: First/Last frame  |  Space: Play/Pause  |  +/-: Change auto-play step"
    )
    plt.figtext(
        0.5,
        0.04,
        instructions,
        ha="center",
        fontsize=9,
        bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5},
    )

    update_display()
    print(f"Loaded {total_frames} visualization units. Ready to display.")
    plt.show()


if __name__ == "__main__":
    visualize()
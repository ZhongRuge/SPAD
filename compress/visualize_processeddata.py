import os
import sys
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
from simulation_io import resolve_visualization_algorithm


def visualize():
    config = load_config(DATA_GENERATE_DIR / "config.yaml")
    dataset_paths = resolve_dataset_paths(config)
    compression_output_dir = resolve_compression_output_dir(config)
    algorithm_id = resolve_visualization_algorithm(config)
    algorithm_params = resolve_algorithm_params(config, algorithm_id)
    my_algorithm = build_compressor(algorithm_id, algorithm_params)
    dataset_name = dataset_name_from_paths(dataset_paths)
    variant_name = build_variant_name(algorithm_id, algorithm_params)

    meta_path = dataset_paths["meta_path"]
    data_path = dataset_paths["data_path"]
    latest_run_dir = find_latest_run_dir(compression_output_dir, dataset_name)
    if latest_run_dir is None:
        print(f"No compression runs found for dataset: {dataset_name}\n")
        return

    compressed_path = latest_run_dir / variant_name / build_output_filename(my_algorithm)

    if not os.path.exists(compressed_path):
        print(f"Missing compressed file: {compressed_path}\n")
        return

    io_manager = SpadIOManager(meta_path, data_path)
    width = io_manager.width
    height = io_manager.height

    print(f"Decoding algorithm output: {my_algorithm.algorithm_name}")

    video_matrix_list = []
    chunk_stream = io_manager.stream_compressed_chunks(compressed_path)

    for frame_count, chunk in chunk_stream:
        decoded_batch = my_algorithm.decode(chunk, (frame_count, height, width))
        video_matrix_list.append(decoded_batch)

    video_matrix = np.concatenate(video_matrix_list, axis=0)
    total_frames = video_matrix.shape[0]

    is_grayscale = video_matrix.max() > 1
    vmax = 255 if is_grayscale else 1

    print(f"Decoded video: {width}x{height}, total_frames={total_frames}")

    state = {
        "curr_idx": 0,
        "playing": False,
        "step": 1,
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.2)
    im = ax.imshow(video_matrix[0], cmap="gray", interpolation="nearest", vmin=0, vmax=vmax)
    ax.axis("off")

    def update_display():
        idx = state["curr_idx"]
        im.set_data(video_matrix[idx])

        if is_grayscale:
            metric = np.mean(video_matrix[idx])
            metric_str = f"Mean Intensity: {metric:.2f}"
        else:
            metric = np.mean(video_matrix[idx])
            metric_str = f"Sparsity: {metric:.2%}"

        ax.set_title(f"[{my_algorithm.algorithm_name}]\nFrame: {idx} / {total_frames - 1}\n{metric_str}")
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
        "Left/Right: +/-1 frame  |  Up/Down: +/-100 frames\n"
        "PgUp/PgDn: +/-1000 frames  |  Space: Play/Pause"
    )
    plt.figtext(
        0.5,
        0.05,
        instructions,
        ha="center",
        fontsize=9,
        bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5},
    )

    print(f"Loaded {total_frames} frames. Ready to visualize.")
    plt.show()


if __name__ == "__main__":
    visualize()

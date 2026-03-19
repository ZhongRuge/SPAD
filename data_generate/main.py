import os
import random
from contextlib import nullcontext

import numpy as np
from signal_generator import SignalGenerator
from noise_injector import NoiseInjector
from simulation_io import append_ground_truth
from simulation_io import build_metadata
from simulation_io import load_config
from simulation_io import resolve_output_paths
from simulation_io import resolve_total_frames
from simulation_io import scene_has_ground_truth
from simulation_io import write_metadata
from simulation_io import write_video_batch

def run_simulation():
    config = load_config()
    total_frames = resolve_total_frames(config)
    batch_size = int(config["simulation"]["batch_size"])
    seed = int(config["simulation"].get("random_seed", random.randint(0, 1_000_000)))
    save_as_bits = bool(config["io"].get("save_as_bits", True))
    paths = resolve_output_paths(config)

    seed_sequence = np.random.SeedSequence(seed)
    generator_seed, noise_seed = seed_sequence.spawn(2)
    generator_rng = np.random.default_rng(generator_seed)
    noise_rng = np.random.default_rng(noise_seed)

    generator = SignalGenerator(config, generator_rng)
    noise_injector = NoiseInjector(config, noise_rng)

    metadata = build_metadata(config, total_frames, seed, paths)
    write_metadata(paths["meta_path"], metadata)
    has_ground_truth = scene_has_ground_truth(config["scene"]["type"])

    print(
        f"开始生成: {metadata['width']}x{metadata['height']}, "
        f"共 {total_frames} 帧, save_as_bits={save_as_bits}, seed={seed}, "
        f"scene={config['scene']['type']}"
    )

    if not has_ground_truth and os.path.exists(paths["ground_truth_path"]):
        os.remove(paths["ground_truth_path"])

    gt_context = (
        open(paths["ground_truth_path"], "w", encoding="utf-8")
        if has_ground_truth
        else nullcontext(None)
    )

    ground_truth_records = 0

    with open(paths["data_path"], "wb") as data_file, gt_context as gt_file:
        for start_frame in range(0, total_frames, batch_size):
            # 按 batch 生成可以把内存占用限制在固定范围，便于后续扩大帧数测试。
            current_batch_size = min(batch_size, total_frames - start_frame)
            clean_batch, ground_truth_batch = generator.generate_batch(current_batch_size)
            noisy_batch = noise_injector.apply_noise(clean_batch)

            write_video_batch(data_file, noisy_batch, save_as_bits)
            if gt_file is not None and ground_truth_batch:
                append_ground_truth(gt_file, ground_truth_batch)
                ground_truth_records += len(ground_truth_batch)

            processed_frames = start_frame + current_batch_size
            print(f"已处理 {processed_frames} / {total_frames} 帧")

    file_size_mb = os.path.getsize(paths["data_path"]) / (1024 * 1024)
    print(
        f"\n生成完毕!\n"
        f"二进制文件: {paths['data_path']}\n"
        f"元数据文件: {paths['meta_path']}\n"
        f"文件大小: {file_size_mb:.2f} MB"
    )

    if has_ground_truth:
        print(f"真值文件: {paths['ground_truth_path']}\n真值条目数: {ground_truth_records}")
    else:
        print("当前场景不生成 ground truth 文件")

if __name__ == "__main__":
    run_simulation()
"""数据生成入口。

支持直接传入 config dict（用于 noise_sweep 等场景），
不再强制从磁盘读写配置。
"""

import os
import random
from contextlib import nullcontext

import numpy as np

from spad.config import (
    load_config,
    resolve_output_paths,
    resolve_runtime_batch_size,
    resolve_save_as_bits,
    resolve_total_frames,
    scene_has_ground_truth,
)
from spad.generate.noise import NoiseInjector
from spad.generate.signal import SignalGenerator
from spad.io import append_ground_truth, build_metadata, write_metadata, write_video_batch


def run_simulation(config=None, config_path=None):
    """运行 SPAD 数据仿真。

    Args:
        config: 已加载的配置字典（可选）。若为 None 则从 config_path 加载。
        config_path: 配置文件路径（可选）。
    """
    if config is None:
        config = load_config(config_path)

    total_frames = resolve_total_frames(config)
    batch_size = resolve_runtime_batch_size(config, "generate")
    seed = int(config["simulation"].get("random_seed", random.randint(0, 1_000_000)))
    save_as_bits = resolve_save_as_bits(config["io"])
    paths = resolve_output_paths(config)

    seed_seq = np.random.SeedSequence(seed)
    gen_seed, noise_seed = seed_seq.spawn(2)
    gen_rng = np.random.default_rng(gen_seed)
    noise_rng = np.random.default_rng(noise_seed)

    generator = SignalGenerator(config, gen_rng)
    noise_injector = NoiseInjector(config, noise_rng)

    metadata = build_metadata(config, total_frames, seed, paths)
    has_gt = scene_has_ground_truth(config["scene"]["type"])

    print(
        f"开始生成: {metadata['width']}x{metadata['height']}, "
        f"共 {total_frames} 帧, save_as_bits={save_as_bits}, seed={seed}, "
        f"scene={config['scene']['type']}"
    )

    if not has_gt and os.path.exists(paths["ground_truth_path"]):
        os.remove(paths["ground_truth_path"])
    if os.path.exists(paths["meta_path"]):
        os.remove(paths["meta_path"])

    gt_ctx = (
        open(paths["ground_truth_path"], "w", encoding="utf-8")
        if has_gt
        else nullcontext(None)
    )

    with open(paths["data_path"], "wb") as data_file, gt_ctx as gt_file:
        for start in range(0, total_frames, batch_size):
            n = min(batch_size, total_frames - start)
            clean, gt_batch = generator.generate_batch(n)
            noisy = noise_injector.apply_noise(clean)
            write_video_batch(data_file, noisy, save_as_bits)

            if gt_file is not None and gt_batch:
                append_ground_truth(gt_file, gt_batch)

            print(f"  已处理 {start + n} / {total_frames} 帧")

    write_metadata(paths["meta_path"], metadata)

    size_mb = os.path.getsize(paths["data_path"]) / (1024 * 1024)
    print(f"\n生成完毕! 文件: {paths['data_path']} ({size_mb:.2f} MB)")

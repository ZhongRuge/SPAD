import json
import math
import os
from pathlib import Path

import numpy as np
import yaml


SCENES_WITH_GROUND_TRUTH = {"moving_circle"}
REFERENCE_CROSSTALK_ORTHOGONAL = 1.1e-3 # 最近正交邻域的瞬时串扰率约为 1.1e-3
REFERENCE_CROSSTALK_DIAGONAL = 1.5e-4 # 最近对角邻域的瞬时串扰率约为 1.5e-4


def load_config(path="config.yaml"):
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path(__file__).resolve().parent / path_obj

    with open(path_obj, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["_config_dir"] = str(path_obj.parent)
    validate_config(config)
    return config


def validate_config(config):
    sensor = config["sensor"]
    simulation = config["simulation"]
    scene = config["scene"]
    noise = config.get("noise", {})
    io_config = config["io"]

    _require_positive_int(sensor["width"], "sensor.width")
    _require_positive_int(sensor["height"], "sensor.height")
    _require_positive_int(sensor["fps"], "sensor.fps")
    _require_probability(sensor.get("pde", 1.0), "sensor.pde")

    dead_time_ns = float(sensor.get("dead_time_ns", 0.0))
    if dead_time_ns < 0:
        raise ValueError("sensor.dead_time_ns 必须大于等于 0")

    _require_positive_int(simulation["batch_size"], "simulation.batch_size")
    if simulation.get("total_seconds") is None:
        raise KeyError("缺少 simulation.total_seconds")
    if float(simulation["total_seconds"]) <= 0:
        raise ValueError("simulation.total_seconds 必须大于 0")

    total_frame_count = float(simulation["total_seconds"]) * int(sensor["fps"])
    rounded_total_frames = round(total_frame_count)
    if not math.isclose(total_frame_count, rounded_total_frames, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "simulation.total_seconds * sensor.fps 必须对应整数帧数，"
            f"当前得到 {total_frame_count}"
        )

    if scene["type"] not in {"uniform_poisson", "moving_circle"}:
        raise ValueError("scene.type 仅支持 uniform_poisson 或 moving_circle")

    _require_non_negative(scene["background_cps"], "scene.background_cps")
    _require_non_negative(scene.get("signal_cps", 0.0), "scene.signal_cps")
    _require_non_negative(scene.get("target_radius", 0), "scene.target_radius")
    _require_non_negative(scene.get("velocity_pps", 0.0), "scene.velocity_pps")

    _require_non_negative(noise.get("dcr_cps", 0.0), "noise.dcr_cps")
    if "crosstalk_prob" in noise:
        _require_probability(noise.get("crosstalk_prob", 0.0), "noise.crosstalk_prob")
    if "crosstalk_orthogonal_prob" in noise:
        _require_probability(
            noise.get("crosstalk_orthogonal_prob", 0.0),
            "noise.crosstalk_orthogonal_prob",
        )
    if "crosstalk_diagonal_prob" in noise:
        _require_probability(
            noise.get("crosstalk_diagonal_prob", 0.0),
            "noise.crosstalk_diagonal_prob",
        )
    _require_probability(noise.get("afterpulsing_prob", 0.0), "noise.afterpulsing_prob")

    if not str(io_config["filename"]).strip():
        raise ValueError("io.filename 不能为空")

    resolve_save_as_bits(io_config)


def _require_positive_int(value, field_name):
    numeric_value = float(value)
    if not numeric_value.is_integer() or int(numeric_value) <= 0:
        raise ValueError(f"{field_name} 必须为正整数")


def _require_non_negative(value, field_name):
    if float(value) < 0:
        raise ValueError(f"{field_name} 必须大于等于 0")


def _require_probability(value, field_name):
    probability = float(value)
    if probability < 0 or probability > 1:
        raise ValueError(f"{field_name} 必须在 [0, 1] 区间内")


def resolve_save_as_bits(io_config):
    save_as_bits = io_config.get("save_as_bits", True)
    if isinstance(save_as_bits, bool):
        return save_as_bits

    raise ValueError("io.save_as_bits 必须为布尔值 true 或 false")


def resolve_crosstalk_probabilities(noise_config):
    orthogonal_prob = noise_config.get("crosstalk_orthogonal_prob")
    diagonal_prob = noise_config.get("crosstalk_diagonal_prob")

    if orthogonal_prob is not None or diagonal_prob is not None:
        return {
            "orthogonal_prob": float(orthogonal_prob or 0.0),
            "diagonal_prob": float(diagonal_prob or 0.0),
            "total_prob": 4.0 * float(orthogonal_prob or 0.0) + 4.0 * float(diagonal_prob or 0.0),
            "ratio_source": "explicit",
        }

    total_prob = float(noise_config.get("crosstalk_prob", 0.0))
    if total_prob <= 0:
        return {
            "orthogonal_prob": 0.0,
            "diagonal_prob": 0.0,
            "total_prob": 0.0,
            "ratio_source": "disabled",
        }

    # 参考公开测量结果: 最近正交邻域约 1.1e-3, 最近对角邻域约 1.5e-4。
    # 这里保留用户设定的总串扰强度，只按该比值把总概率分配到 8 个邻域。
    reference_ratio = REFERENCE_CROSSTALK_ORTHOGONAL / REFERENCE_CROSSTALK_DIAGONAL
    diagonal_prob = total_prob / (4.0 * (reference_ratio + 1.0))
    orthogonal_prob = reference_ratio * diagonal_prob
    return {
        "orthogonal_prob": orthogonal_prob,
        "diagonal_prob": diagonal_prob,
        "total_prob": total_prob,
        "ratio_source": "derived_from_total",
    }


def resolve_total_frames(config):
    simulation_config = config["simulation"]
    total_seconds = float(simulation_config["total_seconds"])
    fps = int(config["sensor"]["fps"])

    # 帧率固定时，所有数据规模统一由仿真时长决定，避免同时维护 seconds / frames 两套入口。
    return int(round(total_seconds * fps))


def resolve_output_paths(config):
    output_dir = Path(config["io"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(config["_config_dir"]) / output_dir

    filename = config["io"]["filename"]
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(str(output_dir), filename)
    base_name, _ = os.path.splitext(filename)
    meta_path = os.path.join(str(output_dir), f"{base_name}.meta.json")
    ground_truth_path = os.path.join(str(output_dir), f"{base_name}.ground_truth.jsonl")

    return {
        "data_path": data_path,
        "meta_path": meta_path,
        "ground_truth_path": ground_truth_path,
    }


def build_metadata(config, total_frames, seed, paths):
    save_as_bits = resolve_save_as_bits(config["io"])
    has_ground_truth = scene_has_ground_truth(config["scene"]["type"])
    crosstalk = resolve_crosstalk_probabilities(config.get("noise", {}))
    pixels_per_frame = int(config["sensor"]["width"]) * int(config["sensor"]["height"])
    return {
        "width": int(config["sensor"]["width"]),
        "height": int(config["sensor"]["height"]),
        "fps": int(config["sensor"]["fps"]),
        "total_seconds": float(config["simulation"]["total_seconds"]),
        "total_frames": int(total_frames),
        "storage_dtype": "uint8",
        "save_as_bits": save_as_bits,
        "storage_order": "frame-major: 统一按 frame-major 顺序写盘，便于后续压缩算法直接流式读取",
        "bit_packing": "per-frame numpy.packbits over flattened pixels" if save_as_bits else None,
        "seed": int(seed),
        "scene_type": config["scene"]["type"],
        "has_ground_truth": has_ground_truth,
        "sensor": {
            "pde": float(config["sensor"].get("pde", 1.0)),
            "dead_time_ns": float(config["sensor"].get("dead_time_ns", 0.0)),
        },
        "scene": {
            "background_cps": float(config["scene"]["background_cps"]),
            "signal_cps": float(config["scene"].get("signal_cps", 0.0)),
            "target_radius": int(config["scene"].get("target_radius", 0)),
            "velocity_pps": float(config["scene"].get("velocity_pps", 0.0)),
        },
        "noise": {
            "dcr_cps": float(config.get("noise", {}).get("dcr_cps", 0.0)),
            "crosstalk_prob": crosstalk["total_prob"],
            "crosstalk_orthogonal_prob": crosstalk["orthogonal_prob"],
            "crosstalk_diagonal_prob": crosstalk["diagonal_prob"],
            "crosstalk_ratio_source": crosstalk["ratio_source"],
            "afterpulsing_prob": float(config.get("noise", {}).get("afterpulsing_prob", 0.0)),
        },
        "pixels_per_frame": pixels_per_frame,
        "bytes_per_frame": math.ceil(pixels_per_frame / 8) if save_as_bits else pixels_per_frame,
        "ground_truth_filename": os.path.basename(paths["ground_truth_path"]) if has_ground_truth else None,
    }


def scene_has_ground_truth(scene_type):
    return scene_type in SCENES_WITH_GROUND_TRUTH


def write_metadata(meta_path, metadata):
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def write_video_batch(file_obj, frame_batch, save_as_bits):
    # 统一按 frame-major 顺序写盘，便于后续压缩算法直接流式读取。
    if frame_batch.dtype != np.uint8:
        frame_batch = frame_batch.astype(np.uint8)

    if save_as_bits:
        flattened_frames = frame_batch.reshape(frame_batch.shape[0], -1)
        payload = np.packbits(flattened_frames, axis=1).reshape(-1)
    else:
        payload = frame_batch.reshape(-1)

    file_obj.write(payload.tobytes())


def append_ground_truth(file_obj, ground_truth_batch):
    file_obj.writelines(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in ground_truth_batch
    )


def read_metadata(meta_path):
    with open(meta_path, "r", encoding="utf-8") as file:
        return json.load(file)


def expected_storage_bytes(metadata):
    pixels_per_frame = metadata["width"] * metadata["height"]
    if metadata.get("save_as_bits", True):
        return metadata["total_frames"] * math.ceil(pixels_per_frame / 8)
    return metadata["total_frames"] * pixels_per_frame


def load_video_matrix(data_path, metadata):
    raw_data = np.fromfile(data_path, dtype=np.uint8)
    expected_bytes = expected_storage_bytes(metadata)

    if raw_data.size != expected_bytes:
        raise ValueError(
            f"数据文件大小与元数据不一致: expected={expected_bytes} bytes, actual={raw_data.size} bytes"
        )

    total_frames = metadata["total_frames"]
    height = metadata["height"]
    width = metadata["width"]
    pixels_per_frame = width * height
    if metadata.get("save_as_bits", True):
        bytes_per_frame = math.ceil(pixels_per_frame / 8)
        framed_bytes = raw_data.reshape((total_frames, bytes_per_frame))
        decoded = np.unpackbits(framed_bytes, axis=1)[:, :pixels_per_frame]
    else:
        decoded = raw_data.reshape((total_frames, pixels_per_frame))

    return decoded.reshape((total_frames, height, width))
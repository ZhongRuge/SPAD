import json
import math
import os

import numpy as np
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_total_frames(config):
    simulation_config = config["simulation"]
    total_frames = simulation_config.get("total_frames")
    total_seconds = simulation_config.get("total_seconds")
    fps = int(config["sensor"]["fps"])

    if total_frames is not None and total_seconds is not None:
        derived_total_frames = int(total_seconds * fps)
        if int(total_frames) != derived_total_frames:
            raise ValueError("simulation.total_frames 与 simulation.total_seconds 不一致")
        return int(total_frames)

    if total_frames is not None:
        return int(total_frames)

    if total_seconds is not None:
        return int(total_seconds * fps)

    raise KeyError("缺少 simulation.total_frames 或 simulation.total_seconds")


def resolve_output_paths(config):
    output_dir = config["io"]["output_dir"]
    filename = config["io"]["filename"]
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(output_dir, filename)
    meta_path = data_path + ".meta.json"
    base_name, _ = os.path.splitext(filename)
    ground_truth_path = os.path.join(output_dir, f"{base_name}.ground_truth.jsonl")

    return {
        "data_path": data_path,
        "meta_path": meta_path,
        "ground_truth_path": ground_truth_path,
    }


def build_metadata(config, total_frames, seed, paths):
    return {
        "width": int(config["sensor"]["width"]),
        "height": int(config["sensor"]["height"]),
        "fps": int(config["sensor"]["fps"]),
        "total_frames": int(total_frames),
        "storage_dtype": "uint8",
        "save_as_bits": bool(config["io"].get("save_as_bits", True)),
        "seed": int(seed),
        "scene_type": config["scene"]["type"],
        "ground_truth_filename": os.path.basename(paths["ground_truth_path"]),
    }


def write_metadata(meta_path, metadata):
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def write_video_batch(file_obj, frame_batch, save_as_bits):
    if frame_batch.dtype != np.uint8:
        frame_batch = frame_batch.astype(np.uint8)

    if save_as_bits:
        payload = np.packbits(frame_batch.reshape(-1))
    else:
        payload = frame_batch.reshape(-1)

    file_obj.write(payload.tobytes())


def append_ground_truth(file_obj, ground_truth_batch):
    for record in ground_truth_batch:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_metadata(meta_path):
    with open(meta_path, "r", encoding="utf-8") as file:
        return json.load(file)


def expected_storage_bytes(metadata):
    total_pixels = metadata["width"] * metadata["height"] * metadata["total_frames"]
    if metadata.get("save_as_bits", True):
        return math.ceil(total_pixels / 8)
    return total_pixels


def load_video_matrix(data_path, metadata):
    raw_data = np.fromfile(data_path, dtype=np.uint8)
    expected_bytes = expected_storage_bytes(metadata)

    if raw_data.size != expected_bytes:
        raise ValueError(
            f"数据文件大小与元数据不一致: expected={expected_bytes} bytes, actual={raw_data.size} bytes"
        )

    total_pixels = metadata["width"] * metadata["height"] * metadata["total_frames"]
    if metadata.get("save_as_bits", True):
        decoded = np.unpackbits(raw_data)[:total_pixels]
    else:
        decoded = raw_data

    return decoded.reshape((metadata["total_frames"], metadata["height"], metadata["width"]))
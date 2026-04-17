"""SPAD 数据 I/O：流式读取、压缩写入、元数据管理。

合并了原 simulation_io.py 的 I/O 部分和 io_manager.py，
并做了以下性能优化：
  - SpadReader 在 stream_batches 期间保持文件句柄打开
  - CompressedWriter 上下文管理器，整次写入只开关一次文件
  - stream_compressed_chunks 同样使用上下文管理器
"""

import json
import math
import os
import struct
from pathlib import Path

import numpy as np


# ============== 元数据构建与读写 ==============

def build_metadata(config, total_frames, seed, paths):
    """根据配置构建元数据字典。"""
    from spad.config import resolve_save_as_bits, resolve_crosstalk_probabilities, scene_has_ground_truth

    save_as_bits = resolve_save_as_bits(config["io"])
    has_gt = scene_has_ground_truth(config["scene"]["type"])
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
        "storage_order": "frame-major: 统一按 frame-major 顺序写盘",
        "bit_packing": "per-frame numpy.packbits over flattened pixels" if save_as_bits else None,
        "seed": int(seed),
        "scene_type": config["scene"]["type"],
        "has_ground_truth": has_gt,
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
        "ground_truth_filename": os.path.basename(paths["ground_truth_path"]) if has_gt else None,
    }


def write_metadata(meta_path, metadata):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def read_metadata(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def expected_storage_bytes(metadata):
    pixels_per_frame = metadata["width"] * metadata["height"]
    if metadata.get("save_as_bits", True):
        return metadata["total_frames"] * math.ceil(pixels_per_frame / 8)
    return metadata["total_frames"] * pixels_per_frame


# ============== 原始数据写入 ==============

def write_video_batch(file_obj, frame_batch, save_as_bits):
    """将一个 batch 的帧写入文件（frame-major 顺序）。"""
    if frame_batch.dtype != np.uint8:
        frame_batch = frame_batch.astype(np.uint8)

    if save_as_bits:
        flat = frame_batch.reshape(frame_batch.shape[0], -1)
        payload = np.packbits(flat, axis=1).ravel()
    else:
        payload = frame_batch.ravel()

    file_obj.write(payload.tobytes())


def append_ground_truth(file_obj, ground_truth_batch):
    file_obj.writelines(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in ground_truth_batch
    )


def load_video_matrix(data_path, metadata):
    """一次性加载全部帧到内存，返回 (total_frames, H, W) uint8 矩阵。"""
    raw = np.fromfile(data_path, dtype=np.uint8)
    expected = expected_storage_bytes(metadata)
    if raw.size != expected:
        raise ValueError(f"数据文件大小不匹配: expected={expected}, actual={raw.size}")

    total_frames = metadata["total_frames"]
    height = metadata["height"]
    width = metadata["width"]
    pixels_per_frame = width * height

    if metadata.get("save_as_bits", True):
        bytes_per_frame = math.ceil(pixels_per_frame / 8)
        framed = raw.reshape((total_frames, bytes_per_frame))
        decoded = np.unpackbits(framed, axis=1)[:, :pixels_per_frame]
    else:
        decoded = raw.reshape((total_frames, pixels_per_frame))

    return decoded.reshape((total_frames, height, width))


# ============== SpadReader：流式读取原始数据 ==============

class SpadReader:
    """SPAD 原始数据流式读取器。

    合并了原 SpadIOManager 的读取功能，优化点:
    - stream_batches 内部只打开一次文件句柄
    - 预计算 bytes_per_frame 避免重复计算
    """

    def __init__(self, meta_path, data_path):
        self.meta_path = str(meta_path)
        self.data_path = str(data_path)

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"找不到元数据文件: {meta_path}")

        self.meta = read_metadata(self.meta_path)
        self.width = self.meta["width"]
        self.height = self.meta["height"]
        self.total_frames = self.meta["total_frames"]
        self.save_as_bits = self.meta.get("save_as_bits", True)
        self.pixels_per_frame = self.width * self.height

        if self.save_as_bits:
            self.bytes_per_frame = math.ceil(self.pixels_per_frame / 8)
        else:
            self.bytes_per_frame = self.pixels_per_frame

    def stream_batches(self, batch_size=1000):
        """生成器：按批次流式读取，yield (batch_size, H, W) uint8 矩阵。"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"找不到数据文件: {self.data_path}")

        with open(self.data_path, "rb") as f:
            for start in range(0, self.total_frames, batch_size):
                n = min(batch_size, self.total_frames - start)
                read_size = self.bytes_per_frame * n
                raw = f.read(read_size)

                if len(raw) != read_size:
                    raise ValueError(
                        f"数据文件被截断: 期望 {read_size} 字节，实际 {len(raw)} 字节"
                    )

                arr = np.frombuffer(raw, dtype=np.uint8)

                if self.save_as_bits:
                    framed = arr.reshape((n, self.bytes_per_frame))
                    pixels = np.unpackbits(framed, axis=1)[:, :self.pixels_per_frame]
                else:
                    pixels = arr.reshape((n, self.pixels_per_frame))

                yield pixels.reshape((n, self.height, self.width))

    def get_original_size_bytes(self):
        return os.path.getsize(self.data_path)

    def get_batch_shape(self, frame_count):
        return (int(frame_count), self.height, self.width)


# ============== 压缩数据写入/读取 ==============

_CHUNK_HEADER_FMT = "<II"
_CHUNK_HEADER_SIZE = struct.calcsize(_CHUNK_HEADER_FMT)


class CompressedWriter:
    """压缩数据写入器（上下文管理器）。

    整次评估只打开/关闭一次文件，消除原来 N 次 open("ab") 的开销。
    """

    def __init__(self, path):
        self.path = str(path)
        self._file = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._file = open(self.path, "wb")
        return self

    def __exit__(self, *exc):
        if self._file:
            self._file.close()
            self._file = None

    def write_chunk(self, compressed_bytes: bytes, frame_count: int):
        header = struct.pack(_CHUNK_HEADER_FMT, len(compressed_bytes), int(frame_count))
        self._file.write(header)
        self._file.write(compressed_bytes)


def stream_compressed_chunks(path):
    """生成器：从压缩文件流式读取 (frame_count, chunk_bytes)。"""
    with open(path, "rb") as f:
        while True:
            header = f.read(_CHUNK_HEADER_SIZE)
            if not header:
                break
            if len(header) != _CHUNK_HEADER_SIZE:
                raise ValueError("压缩文件头部不完整")

            chunk_size, frame_count = struct.unpack(_CHUNK_HEADER_FMT, header)
            data = f.read(chunk_size)
            if len(data) != chunk_size:
                raise ValueError(f"压缩文件被截断: 期望 {chunk_size} 字节，实际 {len(data)} 字节")

            yield frame_count, data

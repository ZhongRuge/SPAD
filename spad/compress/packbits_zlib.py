import zlib

import numpy as np

from spad.compress.base import BaseCompressor
from spad.compress.registry import register


@register("packbits_zlib")
class PackBitsZlibCompressor(BaseCompressor):
    """位打包 + Zlib（无损）。先 np.packbits 8x 缩减，再 zlib。"""

    def __init__(self, zlib_level=9):
        super().__init__()
        if not isinstance(zlib_level, int) or not (0 <= zlib_level <= 9):
            raise ValueError("zlib_level must be an integer in [0, 9]")
        self.zlib_level = zlib_level

    @property
    def algorithm_name(self) -> str:
        return f"PackBits+Zlib (level={self.zlib_level})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        if batch_pixels.size == 0:
            return b""
        flat = batch_pixels.reshape(batch_pixels.shape[0], -1)
        packed = np.packbits(flat, axis=1)
        return zlib.compress(packed.tobytes(), level=self.zlib_level)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)

        total_frames, height, width = batch_shape
        ppf = height * width
        bpf = int(np.ceil(ppf / 8.0))
        raw = zlib.decompress(compressed_bytes)
        expected = total_frames * bpf

        if len(raw) != expected:
            raise ValueError(f"PackBitsZlib byte count mismatch: expected={expected}, actual={len(raw)}")

        packed = np.frombuffer(raw, dtype=np.uint8).reshape((total_frames, bpf))
        unpacked = np.unpackbits(packed, axis=1)[:, :ppf]
        return unpacked.reshape(batch_shape)

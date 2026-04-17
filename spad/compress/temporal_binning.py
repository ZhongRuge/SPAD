import zlib

import numpy as np

from spad.compress._utils import UINT8_MAX
from spad.compress.base import BaseCompressor
from spad.compress.registry import register


@register("temporal_binning")
class TemporalBinningCompressor(BaseCompressor):
    """时域累加 + Zlib（有损）。将连续 bin_size 帧在时间维求和后压缩。"""

    def __init__(self, bin_size=255):
        super().__init__()
        if not isinstance(bin_size, int) or bin_size <= 0:
            raise ValueError("bin_size 必须为正整数")
        if bin_size > UINT8_MAX:
            raise ValueError("bin_size 不能超过 255，否则 uint8 会溢出")
        self.bin_size = bin_size

    @property
    def algorithm_name(self) -> str:
        return f"时域累加 (Binning={self.bin_size}) + Zlib"

    @property
    def is_lossless(self) -> bool:
        return False

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        T = batch_pixels.shape[0]
        num_bins = int(np.ceil(T / self.bin_size))
        binned = []

        for i in range(num_bins):
            start = i * self.bin_size
            end = min(start + self.bin_size, T)
            binned.append(np.sum(batch_pixels[start:end], axis=0, dtype=np.uint8))

        binned_arr = np.array(binned, dtype=np.uint8)
        compressed = zlib.compress(binned_arr.tobytes(), level=9)
        header = np.array([num_bins], dtype=np.uint16).tobytes()
        return header + compressed

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)
        if len(compressed_bytes) < 2:
            raise ValueError("TemporalBinning 格式错误：缺少 bin 数量头部")

        num_bins = int(np.frombuffer(compressed_bytes[:2], dtype=np.uint16)[0])
        raw = zlib.decompress(compressed_bytes[2:])
        _, Y, X = batch_shape
        expected = num_bins * Y * X

        if len(raw) != expected:
            raise ValueError(f"TemporalBinning 尺寸不匹配: expected={expected}, actual={len(raw)}")

        binned = np.frombuffer(raw, dtype=np.uint8).reshape((num_bins, Y, X))

        reconstructed = np.zeros(batch_shape, dtype=np.uint8)
        for i in range(num_bins):
            start = i * self.bin_size
            reconstructed[start] = (binned[i] > 0).astype(np.uint8)

        return reconstructed

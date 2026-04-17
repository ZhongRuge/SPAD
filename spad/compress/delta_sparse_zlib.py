import zlib

import numpy as np

from spad.compress.base import BaseCompressor
from spad.compress.delta_sparse import DeltaSparseCompressor
from spad.compress.registry import register


@register("delta_sparse_zlib")
class DeltaSparseZlibCompressor(BaseCompressor):
    """在 DeltaSparse 编码结果上叠加 zlib（无损）。"""

    def __init__(self):
        super().__init__()
        self._inner = DeltaSparseCompressor()

    @property
    def algorithm_name(self) -> str:
        return "帧间差分 + 稀疏坐标 + Zlib (Delta+Sparse+Zlib)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        raw = self._inner.encode(batch_pixels)
        return zlib.compress(raw, level=9)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)
        raw = zlib.decompress(compressed_bytes)
        return self._inner.decode(raw, batch_shape)

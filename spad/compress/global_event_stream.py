import zlib

import numpy as np

from spad.compress._utils import decode_uvarint, encode_uvarint
from spad.compress.base import BaseCompressor
from spad.compress.registry import register


@register("global_event_stream")
class GlobalEventStreamCompressor(BaseCompressor):
    """全局事件流 + Zlib（无损）。将整个 batch 的活跃像素编码为递增 gap 序列。"""

    def __init__(self, zlib_level=9):
        super().__init__()
        if not isinstance(zlib_level, int) or not (0 <= zlib_level <= 9):
            raise ValueError("zlib_level must be an integer in [0, 9]")
        self.zlib_level = zlib_level

    @property
    def algorithm_name(self) -> str:
        return f"GlobalEventStream+Zlib (level={self.zlib_level})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        active = np.flatnonzero(batch_pixels.ravel())
        if active.size == 0:
            return b""

        stream = bytearray()
        stream.extend(encode_uvarint(int(active.size)))

        prev = -1
        for idx in active:
            stream.extend(encode_uvarint(int(idx) - prev))
            prev = int(idx)

        return zlib.compress(bytes(stream), level=self.zlib_level)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        total = int(np.prod(batch_shape, dtype=np.int64))
        flat = np.zeros(total, dtype=np.uint8)
        if not compressed_bytes:
            return flat.reshape(batch_shape)

        raw = zlib.decompress(compressed_bytes)
        offset = 0
        count, offset = decode_uvarint(raw, offset)
        prev = -1

        for _ in range(count):
            gap, offset = decode_uvarint(raw, offset)
            if gap <= 0:
                raise ValueError("GlobalEventStream invalid event gap")
            idx = prev + gap
            if idx >= total:
                raise ValueError(f"GlobalEventStream index out of range: {idx}")
            flat[idx] = 1
            prev = idx

        if offset != len(raw):
            raise ValueError("GlobalEventStream has trailing undecoded bytes")

        return flat.reshape(batch_shape)

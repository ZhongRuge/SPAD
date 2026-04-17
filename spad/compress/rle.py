import numpy as np

from spad.compress._utils import rle_decode_flat, rle_encode_flat
from spad.compress.base import BaseCompressor
from spad.compress.registry import register


@register("rle")
class RleCompressor(BaseCompressor):
    """基础游程编码（无损）。把整个 batch 展平后对 0/1 序列做 RLE。"""

    @property
    def algorithm_name(self) -> str:
        return "RLE 游程编码 (Run-Length)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        return rle_encode_flat(batch_pixels.ravel())

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        expected = int(np.prod(batch_shape, dtype=np.int64))
        return rle_decode_flat(compressed_bytes, expected).reshape(batch_shape)

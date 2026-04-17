import numpy as np

from spad.compress._utils import (
    compute_xor_delta,
    reconstruct_from_xor_delta,
    rle_decode_flat,
    rle_encode_flat,
)
from spad.compress.base import BaseCompressor
from spad.compress.registry import register


@register("delta_rle")
class DeltaRleCompressor(BaseCompressor):
    """帧间 XOR 差分 + RLE（无损）。"""

    @property
    def algorithm_name(self) -> str:
        return "帧间差分 + RLE (Delta+RLE)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        delta = compute_xor_delta(batch_pixels)
        return rle_encode_flat(delta.ravel())

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        expected = int(np.prod(batch_shape, dtype=np.int64))
        delta = rle_decode_flat(compressed_bytes, expected).reshape(batch_shape)
        return reconstruct_from_xor_delta(delta)

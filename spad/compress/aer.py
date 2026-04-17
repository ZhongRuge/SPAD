import numpy as np

from spad.compress._utils import compute_xor_delta, reconstruct_from_xor_delta, validate_aer_capacity
from spad.compress.base import BaseCompressor
from spad.compress.registry import register


@register("aer")
class AerCompressor(BaseCompressor):
    """AER 事件地址表示（无损）。每个事件 = uint32 [16-bit T | 8-bit Y | 8-bit X]。"""

    def __init__(self, use_delta=False):
        super().__init__()
        self.use_delta = use_delta

    @property
    def algorithm_name(self) -> str:
        mode = "差分模式" if self.use_delta else "原始模式"
        return f"AER 事件地址表示 ({mode})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        validate_aer_capacity(batch_pixels.shape)

        data = compute_xor_delta(batch_pixels) if self.use_delta else batch_pixels

        t, y, x = np.nonzero(data)
        events = (t.astype(np.uint32) << 16) | (y.astype(np.uint32) << 8) | x.astype(np.uint32)
        return events.tobytes()

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        validate_aer_capacity(batch_shape)

        matrix = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return matrix
        if len(compressed_bytes) % 4 != 0:
            raise ValueError("AER 格式错误：字节数不是 uint32 的整数倍")

        events = np.frombuffer(compressed_bytes, dtype=np.uint32)
        t = (events >> 16) & 0xFFFF
        y = (events >> 8) & 0xFF
        x = events & 0xFF
        matrix[t, y, x] = 1

        if self.use_delta:
            matrix = reconstruct_from_xor_delta(matrix)

        return matrix

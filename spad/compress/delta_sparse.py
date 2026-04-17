import numpy as np

from spad.compress._utils import (
    compute_xor_delta,
    reconstruct_from_xor_delta,
    validate_sparse_index_capacity,
)
from spad.compress.base import BaseCompressor
from spad.compress.registry import register


@register("delta_sparse")
class DeltaSparseCompressor(BaseCompressor):
    """帧间差分 + 稀疏坐标编码（无损）。每帧只存差分中值为 1 的像素索引。"""

    @property
    def algorithm_name(self) -> str:
        return "帧间差分 + 稀疏坐标 (Delta+Sparse)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        validate_sparse_index_capacity(batch_pixels)
        delta = compute_xor_delta(batch_pixels)

        chunks = []
        for frame in delta:
            indices = np.flatnonzero(frame.ravel()).astype(np.uint16)
            count = np.array([len(indices)], dtype=np.uint16)
            chunks.append(count.tobytes())
            chunks.append(indices.tobytes())

        return b"".join(chunks)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        delta = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return delta

        offset = 0
        buf = compressed_bytes

        for i in range(batch_shape[0]):
            if offset + 2 > len(buf):
                raise ValueError("DeltaSparse 压缩流格式错误：缺少 count 字段")
            count = int(np.frombuffer(buf[offset:offset + 2], dtype=np.uint16)[0])
            offset += 2

            if count > 0:
                end = offset + count * 2
                if end > len(buf):
                    raise ValueError("DeltaSparse 压缩流格式错误：坐标数据长度不足")
                indices = np.frombuffer(buf[offset:end], dtype=np.uint16)
                offset = end
                delta[i].ravel()[indices] = 1

        if offset != len(buf):
            raise ValueError("DeltaSparse 压缩流格式错误：存在未消费的尾部字节")

        return reconstruct_from_xor_delta(delta)

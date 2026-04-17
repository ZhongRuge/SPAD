import zlib

import numpy as np

from spad.compress._utils import (
    compute_xor_delta,
    decode_uvarint,
    encode_uvarint,
    reconstruct_from_xor_delta,
)
from spad.compress.base import BaseCompressor
from spad.compress.registry import register


@register("delta_sparse_varint_zlib")
class DeltaSparseVarintZlibCompressor(BaseCompressor):
    """差分稀疏流：空帧抑制 + 索引 gap varint + zlib（无损）。"""

    @property
    def algorithm_name(self) -> str:
        return "Delta+Sparse+Varint+Zlib"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        delta = compute_xor_delta(batch_pixels)
        stream = bytearray()
        prev_fi = -1

        for fi, frame in enumerate(delta):
            indices = np.flatnonzero(frame.ravel())
            if indices.size == 0:
                continue

            stream.extend(encode_uvarint(fi - prev_fi))
            stream.extend(encode_uvarint(int(indices.size)))

            prev_idx = -1
            for idx in indices:
                stream.extend(encode_uvarint(int(idx) - prev_idx))
                prev_idx = int(idx)
            prev_fi = fi

        if not stream:
            return b""
        return zlib.compress(bytes(stream), level=9)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        delta = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return delta

        raw = zlib.decompress(compressed_bytes)
        total_frames, height, width = batch_shape
        ppf = height * width
        offset = 0
        prev_fi = -1

        while offset < len(raw):
            frame_gap, offset = decode_uvarint(raw, offset)
            event_count, offset = decode_uvarint(raw, offset)

            if frame_gap <= 0:
                raise ValueError("DeltaSparseVarintZlib invalid frame gap")
            if event_count <= 0:
                raise ValueError("DeltaSparseVarintZlib non-empty frame must have positive event count")

            fi = prev_fi + frame_gap
            if fi >= total_frames:
                raise ValueError(f"DeltaSparseVarintZlib frame index out of range: {fi}")
            if event_count > ppf:
                raise ValueError(f"DeltaSparseVarintZlib event count exceeds capacity: {event_count}")

            flat = delta[fi].ravel()
            prev_idx = -1
            for _ in range(event_count):
                gap, offset = decode_uvarint(raw, offset)
                if gap <= 0:
                    raise ValueError("DeltaSparseVarintZlib invalid pixel gap")
                pixel_idx = prev_idx + gap
                if pixel_idx >= ppf:
                    raise ValueError(f"DeltaSparseVarintZlib pixel index out of range: {pixel_idx}")
                flat[pixel_idx] = 1
                prev_idx = pixel_idx

            prev_fi = fi

        return reconstruct_from_xor_delta(delta)

import zlib

import numpy as np

from base_compressor import BaseCompressor


UINT16_MAX = np.iinfo(np.uint16).max
UINT8_MAX = np.iinfo(np.uint8).max


def _encode_uvarint(value: int) -> bytes:
    """将非负整数编码为 unsigned varint。"""
    if value < 0:
        raise ValueError(f"uvarint cannot encode negative value: {value}")

    encoded = bytearray()
    current_value = int(value)
    while current_value >= 0x80:
        encoded.append((current_value & 0x7F) | 0x80)
        current_value >>= 7
    encoded.append(current_value)
    return bytes(encoded)


def _decode_uvarint(buffer: bytes, offset: int) -> tuple:
    """从字节流中解码一个 unsigned varint，返回 (value, next_offset)。"""
    value = 0
    shift = 0
    current_offset = int(offset)

    while True:
        if current_offset >= len(buffer):
            raise ValueError("Unexpected end of buffer while decoding uvarint")

        byte_value = buffer[current_offset]
        current_offset += 1
        value |= (byte_value & 0x7F) << shift

        if (byte_value & 0x80) == 0:
            return value, current_offset

        shift += 7
        if shift > 63:
            raise ValueError("uvarint is too large to decode safely")


def _validate_sparse_index_capacity(batch_pixels: np.ndarray):
    """检查单帧像素总数是否能用 uint16 索引表示。"""
    pixels_per_frame = batch_pixels.shape[1] * batch_pixels.shape[2]
    if pixels_per_frame > UINT16_MAX:
        raise ValueError(
            "DeltaSparseCompressor 当前使用 uint16 存储像素索引，"
            f"单帧像素数 {pixels_per_frame} 超出上限 {UINT16_MAX}"
        )


def _validate_aer_capacity(shape: tuple):
    """检查 AER 编码的时间戳与坐标位宽是否足够。"""
    frames, height, width = shape
    if frames - 1 > UINT16_MAX:
        raise ValueError(
            "AerCompressor 当前使用 16-bit 时间戳，"
            f"batch 帧数 {frames} 超出可编码范围 {UINT16_MAX + 1}"
        )
    if height - 1 > UINT8_MAX or width - 1 > UINT8_MAX:
        raise ValueError(
            "AerCompressor 当前使用 8-bit X/Y 坐标，"
            f"输入尺寸 {height}x{width} 超出可编码范围 256x256"
        )


# ==========================================
# 算法 1：RLE（Run-Length Encoding）
# ==========================================
class RleCompressor(BaseCompressor):
    """
    基础游程编码（无损）。

    核心思路：
    1. 把整个 batch 展平成 0/1 一维序列。
    2. 只记录“连续相同值的长度”（run-length）以及首个值。

    适用场景：
    - 数据中长段连续 0 或连续 1 较多（尤其连续 0 很长）。

    局限：
    - 若 0/1 高频交替，run 数量会快速增多，压缩率可能变差。
    - 不利用帧间相关性，只对展平后的一维序列建模。
    """

    def __init__(self):
        super().__init__()

    @property
    def algorithm_name(self) -> str:
        return "RLE 游程编码 (Run-Length)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        """对展平后的 0/1 序列执行游程编码。"""
        flat = batch_pixels.ravel()
        if flat.size == 0:
            return b""

        # 找到数值发生变化的位置，用于切分各个 run。
        change_idx = np.where(flat[:-1] != flat[1:])[0] + 1
        change_idx = np.concatenate(([0], change_idx, [flat.size]))
        run_lengths = np.diff(change_idx)
        first_val = np.array([flat[0]], dtype=np.uint8)

        # run length 使用 uint32，避免超长 run 溢出。
        compressed_bytes = first_val.tobytes() + run_lengths.astype(np.uint32).tobytes()
        return compressed_bytes

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        """根据首值和 run-length 恢复展平序列，再 reshape 回原尺寸。"""
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)
        if (len(compressed_bytes) - 1) % 4 != 0:
            raise ValueError("RLE 压缩流格式错误：run-length 字节数不是 uint32 的整数倍")

        first_val = np.frombuffer(compressed_bytes[:1], dtype=np.uint8)[0]
        run_lengths = np.frombuffer(compressed_bytes[1:], dtype=np.uint32)

        # 根据 first_val 决定奇偶 run 哪一组为 1。
        values = np.zeros(len(run_lengths), dtype=np.uint8)
        if first_val == 1:
            values[0::2] = 1
        else:
            values[1::2] = 1

        flat_decoded = np.repeat(values, run_lengths)

        expected_pixels = int(np.prod(batch_shape, dtype=np.int64))
        if flat_decoded.size != expected_pixels:
            raise ValueError(
                f"RLE 解码像素数不匹配: expected={expected_pixels}, actual={flat_decoded.size}"
            )

        return flat_decoded.reshape(batch_shape)


# ==========================================
# 算法 2：帧间差分 + RLE
# ==========================================
class DeltaRleCompressor(BaseCompressor):
    """
    先做帧间 XOR 差分，再做 RLE（无损）。

    核心思路：
    1. 第一帧原样保存。
    2. 后续帧转为与前一帧的 XOR 变化图（只保留“变了没变”）。
    3. 对差分序列做 RLE 压缩。

    适用场景：
    - 邻近帧高度相似、变化区域小且连续。

    局限：
    - 如果帧间变化剧烈，差分后仍不稀疏，则 RLE 优势会减弱。
    """

    def __init__(self):
        super().__init__()

    @property
    def algorithm_name(self) -> str:
        return "帧间差分 + RLE (Delta+RLE)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        # 先计算帧间 XOR 差分。
        delta_pixels = np.zeros_like(batch_pixels)
        delta_pixels[0] = batch_pixels[0]
        delta_pixels[1:] = batch_pixels[1:] ^ batch_pixels[:-1]

        flat = delta_pixels.ravel()
        if flat.size == 0:
            return b""

        # 对差分后的 0/1 序列做 RLE。
        change_idx = np.where(flat[:-1] != flat[1:])[0] + 1
        change_idx = np.concatenate(([0], change_idx, [flat.size]))
        run_lengths = np.diff(change_idx)
        first_val = np.array([flat[0]], dtype=np.uint8)

        return first_val.tobytes() + run_lengths.astype(np.uint32).tobytes()

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)
        if (len(compressed_bytes) - 1) % 4 != 0:
            raise ValueError("DeltaRLE 压缩流格式错误：run-length 字节数不是 uint32 的整数倍")

        first_val = np.frombuffer(compressed_bytes[:1], dtype=np.uint8)[0]
        run_lengths = np.frombuffer(compressed_bytes[1:], dtype=np.uint32)

        values = np.zeros(len(run_lengths), dtype=np.uint8)
        if first_val == 1:
            values[0::2] = 1
        else:
            values[1::2] = 1

        flat_decoded = np.repeat(values, run_lengths)
        expected_pixels = int(np.prod(batch_shape, dtype=np.int64))
        if flat_decoded.size != expected_pixels:
            raise ValueError(
                f"DeltaRLE 解码像素数不匹配: expected={expected_pixels}, actual={flat_decoded.size}"
            )
        delta_pixels = flat_decoded.reshape(batch_shape)

        # 将差分帧做按时间累积 XOR，恢复原始序列。
        reconstructed = np.bitwise_xor.accumulate(delta_pixels, axis=0)
        return reconstructed


# ==========================================
# 算法 3：帧间差分 + 稀疏坐标
# ==========================================
class DeltaSparseCompressor(BaseCompressor):
    """
    记录差分帧中值为 1 的像素索引（无损）。

    核心思路：
    1. 先做帧间 XOR 差分。
    2. 每帧只保存“为 1 的位置索引”，跳过大量 0。
    3. 每帧格式为：`count(uint16) + indices(uint16[count])`。

    适用场景：
    - SPAD 这类极稀疏二值事件数据，单帧激活点很少。

    局限：
    - 单帧像素总数受 uint16 索引限制（<= 65535）。
    - 每帧都要写 count 头，在“空帧很多”的场景仍有元数据开销。
    """

    def __init__(self):
        super().__init__()

    @property
    def algorithm_name(self) -> str:
        return "帧间差分 + 稀疏坐标 (Delta+Sparse)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        _validate_sparse_index_capacity(batch_pixels)

        # 先计算帧间差分，突出变化事件。
        delta_pixels = np.zeros_like(batch_pixels)
        delta_pixels[0] = batch_pixels[0]
        delta_pixels[1:] = batch_pixels[1:] ^ batch_pixels[:-1]

        compressed_chunks = []

        for frame in delta_pixels:
            # 仅存储当前帧中值为 1 的索引。
            indices = np.where(frame.ravel() == 1)[0].astype(np.uint16)
            count = np.array([len(indices)], dtype=np.uint16)
            compressed_chunks.append(count.tobytes())
            compressed_chunks.append(indices.tobytes())

        return b"".join(compressed_chunks)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        delta_pixels = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return delta_pixels

        offset = 0
        total_frames = batch_shape[0]

        for i in range(total_frames):
            # 先读 count，再读 count 个 uint16 索引。
            if offset + 2 > len(compressed_bytes):
                raise ValueError("DeltaSparse 压缩流格式错误：缺少 count 字段")
            count = int(np.frombuffer(compressed_bytes[offset:offset + 2], dtype=np.uint16)[0])
            offset += 2

            if count > 0:
                bytes_to_read = count * 2
                if offset + bytes_to_read > len(compressed_bytes):
                    raise ValueError("DeltaSparse 压缩流格式错误：坐标数据长度不足")
                indices = np.frombuffer(
                    compressed_bytes[offset:offset + bytes_to_read],
                    dtype=np.uint16,
                )
                offset += bytes_to_read
                delta_pixels[i].ravel()[indices] = 1

        if offset != len(compressed_bytes):
            raise ValueError("DeltaSparse 压缩流格式错误：存在未消费的尾部字节")

        return np.bitwise_xor.accumulate(delta_pixels, axis=0)


# ==========================================
# 算法 4：帧间差分 + 稀疏坐标 + Zlib
# ==========================================
class DeltaSparseZlibCompressor(DeltaSparseCompressor):
    """
    在 DeltaSparse 输出字节流上再做 zlib（无损）。

    核心思路：
    - 前端用结构化稀疏编码减少冗余。
    - 后端用通用熵编码（zlib）进一步压缩。

    适用场景：
    - 数据稀疏且索引序列分布有可压缩模式时，通常优于纯 DeltaSparse。

    局限：
    - 相比不加 zlib 的版本，CPU 开销更高。
    """

    def __init__(self):
        super().__init__()

    @property
    def algorithm_name(self) -> str:
        return "帧间差分 + 稀疏坐标 + Zlib (Delta+Sparse+Zlib)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        raw_sparse_bytes = super().encode(batch_pixels)
        return zlib.compress(raw_sparse_bytes, level=9)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)

        raw_sparse_bytes = zlib.decompress(compressed_bytes)
        return super().decode(raw_sparse_bytes, batch_shape)


# ==========================================
# 算法 5：帧间差分 + 稀疏流 + Varint + Zlib
# ==========================================
class DeltaSparseVarintZlibCompressor(BaseCompressor):
    """
    差分稀疏流编码：空帧抑制 + 索引 gap varint + zlib（无损）。

    核心思路：
    1. 帧间 XOR。
    2. 跳过空帧，只记录非空帧与上一非空帧之间的 frame_gap。
    3. 帧内索引转为递增 gap，再用 varint 编码。
    4. 最后整体 zlib。

    适用场景：
    - 空帧多、激活点稀少、且激活索引整体递增 gap 较小的 SPAD 数据。

    局限：
    - 编解码逻辑更复杂，CPU 开销通常高于 DeltaSparseZlib。
    """

    def __init__(self):
        super().__init__()

    @property
    def algorithm_name(self) -> str:
        return "Delta+Sparse+Varint+Zlib"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        delta_pixels = np.zeros_like(batch_pixels)
        delta_pixels[0] = batch_pixels[0]
        delta_pixels[1:] = batch_pixels[1:] ^ batch_pixels[:-1]

        raw_stream = bytearray()
        previous_frame_index = -1

        for frame_index, frame in enumerate(delta_pixels):
            indices = np.flatnonzero(frame.ravel())
            if indices.size == 0:
                continue

            # 非空帧以 frame_gap 编码，避免给空帧写固定头部。
            frame_gap = frame_index - previous_frame_index
            raw_stream.extend(_encode_uvarint(frame_gap))
            raw_stream.extend(_encode_uvarint(int(indices.size)))

            # 帧内索引由绝对位置改为 gap，利于 varint 压缩。
            previous_index = -1
            for index in indices:
                gap = int(index) - previous_index
                raw_stream.extend(_encode_uvarint(gap))
                previous_index = int(index)

            previous_frame_index = frame_index

        if not raw_stream:
            return b""

        return zlib.compress(bytes(raw_stream), level=9)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        delta_pixels = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return delta_pixels

        raw_stream = zlib.decompress(compressed_bytes)
        total_frames, height, width = batch_shape
        pixels_per_frame = height * width
        offset = 0
        previous_frame_index = -1

        while offset < len(raw_stream):
            frame_gap, offset = _decode_uvarint(raw_stream, offset)
            event_count, offset = _decode_uvarint(raw_stream, offset)

            if frame_gap <= 0:
                raise ValueError("DeltaSparseVarintZlib invalid frame gap")
            if event_count <= 0:
                raise ValueError("DeltaSparseVarintZlib non-empty frame must have positive event count")

            frame_index = previous_frame_index + frame_gap
            if frame_index >= total_frames:
                raise ValueError(
                    f"DeltaSparseVarintZlib decoded frame index out of range: {frame_index}"
                )
            if event_count > pixels_per_frame:
                raise ValueError(
                    f"DeltaSparseVarintZlib event count exceeds frame capacity: {event_count}"
                )

            flat_frame = delta_pixels[frame_index].ravel()
            previous_index = -1
            for _ in range(event_count):
                gap, offset = _decode_uvarint(raw_stream, offset)
                if gap <= 0:
                    raise ValueError("DeltaSparseVarintZlib invalid pixel gap")

                pixel_index = previous_index + gap
                if pixel_index >= pixels_per_frame:
                    raise ValueError(
                        f"DeltaSparseVarintZlib decoded pixel index out of range: {pixel_index}"
                    )

                flat_frame[pixel_index] = 1
                previous_index = pixel_index

            previous_frame_index = frame_index

        return np.bitwise_xor.accumulate(delta_pixels, axis=0)


# ==========================================
# 算法 6：位打包 + Zlib
# ==========================================
class PackBitsZlibCompressor(BaseCompressor):
    """
    对原始 0/1 帧做位打包，再交给 zlib（无损）。

    核心思路：
    - `np.packbits` 把每 8 个像素压成 1 个字节，先拿到 8x 的结构性缩减。
    - 再用 zlib 压缩打包后的字节流。

    适用场景：
    - 想要一个实现简单、通用、且通常压缩率不错的无损基线。

    局限：
    - 不显式利用帧间时序关系；在超稀疏场景可能不如专门事件流模型。
    """

    def __init__(self, zlib_level=9):
        super().__init__()
        if not isinstance(zlib_level, int) or zlib_level < 0 or zlib_level > 9:
            raise ValueError("PackBitsZlibCompressor.zlib_level must be an integer in [0, 9]")
        self.zlib_level = zlib_level

    @property
    def algorithm_name(self) -> str:
        return f"PackBits+Zlib (level={self.zlib_level})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        if batch_pixels.size == 0:
            return b""

        # 先按帧展平，再逐帧 packbits。
        flattened_frames = batch_pixels.reshape(batch_pixels.shape[0], -1)
        packed_frames = np.packbits(flattened_frames, axis=1)
        return zlib.compress(packed_frames.tobytes(), level=self.zlib_level)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)

        total_frames, height, width = batch_shape
        pixels_per_frame = height * width
        bytes_per_frame = int(np.ceil(pixels_per_frame / 8.0))
        raw_bytes = zlib.decompress(compressed_bytes)
        expected_bytes = total_frames * bytes_per_frame

        if len(raw_bytes) != expected_bytes:
            raise ValueError(
                f"PackBitsZlib decoded byte count mismatch: expected={expected_bytes}, actual={len(raw_bytes)}"
            )

        packed_frames = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((total_frames, bytes_per_frame))
        unpacked_frames = np.unpackbits(packed_frames, axis=1)[:, :pixels_per_frame]
        return unpacked_frames.reshape(batch_shape)


# ==========================================
# 算法 7：全局事件流 + Zlib
# ==========================================
class GlobalEventStreamCompressor(BaseCompressor):
    """
    将整个 batch 看成一条全局事件流，只编码值为 1 的全局位置（无损）。

    核心思路：
    1. 直接对 batch 全展平后的序列取非零索引。
    2. 保存 event_count 与递增索引 gap 的 varint 编码。
    3. 末端 zlib。

    适用场景：
    - 全局极稀疏、激活事件总量远小于总像素数时，压缩比通常很强。

    局限：
    - 对随机访问不友好（解码某一帧往往需先解完整流）。
    - 随着数据变密，优势会明显下降。
    """

    def __init__(self, zlib_level=9):
        super().__init__()
        if not isinstance(zlib_level, int) or zlib_level < 0 or zlib_level > 9:
            raise ValueError("GlobalEventStreamCompressor.zlib_level must be an integer in [0, 9]")
        self.zlib_level = zlib_level

    @property
    def algorithm_name(self) -> str:
        return f"GlobalEventStream+Zlib (level={self.zlib_level})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        flat_pixels = batch_pixels.ravel()
        active_indices = np.flatnonzero(flat_pixels)
        if active_indices.size == 0:
            return b""

        raw_stream = bytearray()
        raw_stream.extend(_encode_uvarint(int(active_indices.size)))

        # 存储严格递增索引的 gap，而不是绝对索引。
        previous_index = -1
        for active_index in active_indices:
            gap = int(active_index) - previous_index
            raw_stream.extend(_encode_uvarint(gap))
            previous_index = int(active_index)

        return zlib.compress(bytes(raw_stream), level=self.zlib_level)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        total_pixels = int(np.prod(batch_shape, dtype=np.int64))
        flat_pixels = np.zeros(total_pixels, dtype=np.uint8)
        if not compressed_bytes:
            return flat_pixels.reshape(batch_shape)

        raw_stream = zlib.decompress(compressed_bytes)
        offset = 0
        event_count, offset = _decode_uvarint(raw_stream, offset)
        previous_index = -1

        for _ in range(event_count):
            gap, offset = _decode_uvarint(raw_stream, offset)
            if gap <= 0:
                raise ValueError("GlobalEventStream invalid event gap")

            active_index = previous_index + gap
            if active_index >= total_pixels:
                raise ValueError(
                    f"GlobalEventStream decoded active index out of range: {active_index}"
                )

            flat_pixels[active_index] = 1
            previous_index = active_index

        if offset != len(raw_stream):
            raise ValueError("GlobalEventStream has trailing undecoded bytes")

        return flat_pixels.reshape(batch_shape)


# ==========================================
# 算法 8：AER（Address-Event Representation）
# ==========================================
class AerCompressor(BaseCompressor):
    """
    AER 事件地址表示（无损）。

    编码格式：
    - 每个事件编码为 `uint32`：`[16-bit T | 8-bit Y | 8-bit X]`。

    `use_delta=False`（原始模式）：
    - 编码每一帧中的“亮点事件”。

    `use_delta=True`（差分模式）：
    - 先做帧间 XOR，再编码变化事件。

    适用场景：
    - 需要与事件相机/事件流表示对齐，或强调快速事件读写。

    局限：
    - 时间与坐标位宽固定（T 16-bit，X/Y 8-bit）。
    - 在极端稀疏场景下，压缩率未必优于 varint + zlib 类型方法。
    """

    def __init__(self, use_delta=False):
        super().__init__()
        # use_delta=False: 记录原始事件。
        # use_delta=True: 记录帧间变化事件。
        self.use_delta = use_delta

    @property
    def algorithm_name(self) -> str:
        mode = "差分模式" if self.use_delta else "原始模式"
        return f"AER 事件地址表示 ({mode})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        _validate_aer_capacity(batch_pixels.shape)

        # 根据模式决定是否先做帧间差分。
        if self.use_delta:
            data_to_encode = np.zeros_like(batch_pixels)
            data_to_encode[0] = batch_pixels[0]
            data_to_encode[1:] = batch_pixels[1:] ^ batch_pixels[:-1]
        else:
            data_to_encode = batch_pixels

        # nonzero 返回所有事件的时空坐标。
        t, y, x = np.nonzero(data_to_encode)
        t = t.astype(np.uint32)
        y = y.astype(np.uint32)
        x = x.astype(np.uint32)

        aer_events = (t << 16) | (y << 8) | x
        return aer_events.tobytes()

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        _validate_aer_capacity(batch_shape)

        matrix = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return matrix
        if len(compressed_bytes) % 4 != 0:
            raise ValueError("AER 压缩流格式错误：事件流字节数不是 uint32 的整数倍")

        aer_events = np.frombuffer(compressed_bytes, dtype=np.uint32)

        t = (aer_events >> 16) & 0xFFFF
        y = (aer_events >> 8) & 0xFF
        x = aer_events & 0xFF

        matrix[t, y, x] = 1

        # 差分模式下需做按时间累积 XOR 还原。
        if self.use_delta:
            matrix = np.bitwise_xor.accumulate(matrix, axis=0)

        return matrix


# ==========================================
# 算法 9：时域累加 + Zlib（有损）
# ==========================================
class TemporalBinningCompressor(BaseCompressor):
    """
    时域累加（binning）后再 zlib（有损）。

    核心思路：
    1. 把连续 `bin_size` 帧在时间维上求和，得到更少的灰度帧。
    2. 对累加结果做 zlib。

    适用场景：
    - 更关心统计强度/趋势，而非逐帧精确时序重建。
    - 需要显著提升压缩率并接受信息损失。

    局限：
    - 本质不可逆，`is_lossless=False`。
    - decode 仅做“形状安全的近似重建”，不能还原原始事件时序。
    """

    def __init__(self, bin_size=255):
        super().__init__()
        # 默认 bin_size=255，便于累加结果落在 uint8 范围内。
        if not isinstance(bin_size, int) or bin_size <= 0:
            raise ValueError("TemporalBinningCompressor.bin_size 必须为正整数")
        if bin_size > UINT8_MAX:
            raise ValueError("TemporalBinningCompressor.bin_size 不能超过 255，否则 uint8 会溢出")
        self.bin_size = bin_size

    @property
    def algorithm_name(self) -> str:
        return f"时域累加 (Binning={self.bin_size}) + Zlib"

    @property
    def is_lossless(self) -> bool:
        return False

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        T, Y, X = batch_pixels.shape

        num_bins = int(np.ceil(T / self.bin_size))
        binned_frames = []

        for i in range(num_bins):
            start = i * self.bin_size
            end = min((i + 1) * self.bin_size, T)
            chunk = batch_pixels[start:end]
            # 按时间累加，保持与现有实现一致（uint8 累加）。
            binned_frame = np.sum(chunk, axis=0, dtype=np.uint8)
            binned_frames.append(binned_frame)

        binned_array = np.array(binned_frames, dtype=np.uint8)
        compressed_bytes = zlib.compress(binned_array.tobytes(), level=9)
        shape_header = np.array([len(binned_frames)], dtype=np.uint16).tobytes()
        return shape_header + compressed_bytes

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        # 时域累加不可逆，这里仅做形状安全的近似重建。
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)
        if len(compressed_bytes) < 2:
            raise ValueError("TemporalBinning 压缩流格式错误：缺少 bin 数量头部")

        num_bins = int(np.frombuffer(compressed_bytes[:2], dtype=np.uint16)[0])
        zlib_data = compressed_bytes[2:]

        raw_bytes = zlib.decompress(zlib_data)
        _, Y, X = batch_shape
        expected_bytes = num_bins * Y * X
        if len(raw_bytes) != expected_bytes:
            raise ValueError(
                f"TemporalBinning 解码尺寸不匹配: expected={expected_bytes}, actual={len(raw_bytes)}"
            )
        binned_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((num_bins, Y, X))

        # 近似策略：某个 bin 内该像素累加值 > 0，则点亮该 bin 的首帧位置。
        reconstructed = np.zeros(batch_shape, dtype=np.uint8)
        for i in range(num_bins):
            start = i * self.bin_size
            reconstructed[start] = (binned_array[i] > 0).astype(np.uint8)

        return reconstructed

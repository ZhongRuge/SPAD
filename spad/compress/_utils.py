"""压缩算法共享工具：varint 编解码、帧间差分、容量验证。"""

import numpy as np

UINT16_MAX = np.iinfo(np.uint16).max
UINT8_MAX = np.iinfo(np.uint8).max


# --------------- varint 编解码 ---------------

def encode_uvarint(value: int) -> bytes:
    """将非负整数编码为 unsigned varint。"""
    if value < 0:
        raise ValueError(f"uvarint cannot encode negative value: {value}")
    buf = bytearray()
    v = int(value)
    while v >= 0x80:
        buf.append((v & 0x7F) | 0x80)
        v >>= 7
    buf.append(v)
    return bytes(buf)


def decode_uvarint(buffer: bytes, offset: int) -> tuple[int, int]:
    """解码一个 unsigned varint，返回 (value, next_offset)。"""
    value = 0
    shift = 0
    pos = int(offset)
    while True:
        if pos >= len(buffer):
            raise ValueError("Unexpected end of buffer while decoding uvarint")
        b = buffer[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("uvarint is too large to decode safely")


# --------------- 帧间差分 ---------------

def compute_xor_delta(batch_pixels: np.ndarray) -> np.ndarray:
    """帧间 XOR 差分：第一帧保留，后续帧为与前帧的 XOR。"""
    delta = np.empty_like(batch_pixels)
    delta[0] = batch_pixels[0]
    if batch_pixels.shape[0] > 1:
        np.bitwise_xor(batch_pixels[1:], batch_pixels[:-1], out=delta[1:])
    return delta


def reconstruct_from_xor_delta(delta_pixels: np.ndarray) -> np.ndarray:
    """从 XOR 差分帧累积还原原始序列。"""
    return np.bitwise_xor.accumulate(delta_pixels, axis=0)


# --------------- RLE 编解码 ---------------

def rle_encode_flat(flat: np.ndarray) -> bytes:
    """对扁平 0/1 序列做游程编码，返回 first_val(1B) + run_lengths(uint32[])。"""
    if flat.size == 0:
        return b""
    change_idx = np.where(flat[:-1] != flat[1:])[0] + 1
    change_idx = np.concatenate(([0], change_idx, [flat.size]))
    run_lengths = np.diff(change_idx)
    first_val = np.array([flat[0]], dtype=np.uint8)
    return first_val.tobytes() + run_lengths.astype(np.uint32).tobytes()


def rle_decode_flat(compressed_bytes: bytes, expected_pixels: int) -> np.ndarray:
    """从游程编码字节流恢复扁平 0/1 序列。"""
    if not compressed_bytes:
        return np.zeros(expected_pixels, dtype=np.uint8)
    if (len(compressed_bytes) - 1) % 4 != 0:
        raise ValueError("RLE 压缩流格式错误：run-length 字节数不是 uint32 的整数倍")

    first_val = compressed_bytes[0]
    run_lengths = np.frombuffer(compressed_bytes[1:], dtype=np.uint32)
    values = np.zeros(len(run_lengths), dtype=np.uint8)
    if first_val == 1:
        values[0::2] = 1
    else:
        values[1::2] = 1
    flat = np.repeat(values, run_lengths)
    if flat.size != expected_pixels:
        raise ValueError(f"RLE 解码像素数不匹配: expected={expected_pixels}, actual={flat.size}")
    return flat


# --------------- 容量验证 ---------------

def validate_sparse_index_capacity(batch_pixels: np.ndarray):
    """检查单帧像素总数是否能用 uint16 索引表示。"""
    pixels_per_frame = batch_pixels.shape[1] * batch_pixels.shape[2]
    if pixels_per_frame > UINT16_MAX:
        raise ValueError(
            f"DeltaSparse 使用 uint16 索引，"
            f"单帧像素数 {pixels_per_frame} 超出上限 {UINT16_MAX}"
        )


def validate_aer_capacity(shape: tuple):
    """检查 AER 编码的时间戳与坐标位宽是否足够。"""
    frames, height, width = shape
    if frames - 1 > UINT16_MAX:
        raise ValueError(
            f"AER 使用 16-bit 时间戳，batch 帧数 {frames} 超出范围 {UINT16_MAX + 1}"
        )
    if height - 1 > UINT8_MAX or width - 1 > UINT8_MAX:
        raise ValueError(
            f"AER 使用 8-bit 坐标，尺寸 {height}x{width} 超出范围 256x256"
        )

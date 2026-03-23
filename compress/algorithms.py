import numpy as np
from base_compressor import BaseCompressor
import os
import shutil
import subprocess
import tempfile
import zlib


UINT16_MAX = np.iinfo(np.uint16).max
UINT8_MAX = np.iinfo(np.uint8).max


def _validate_sparse_index_capacity(batch_pixels: np.ndarray):
    pixels_per_frame = batch_pixels.shape[1] * batch_pixels.shape[2]
    if pixels_per_frame > UINT16_MAX:
        raise ValueError(
            "DeltaSparseCompressor 当前使用 uint16 存储像素索引，"
            f"单帧像素数 {pixels_per_frame} 超出上限 {UINT16_MAX}"
        )


def _validate_aer_capacity(shape: tuple):
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
# 游程编码 (Run-Length Encoding, RLE)
# ==========================================
class RleCompressor(BaseCompressor):
    """
    基础游程编码算法。
    核心思想：记录连续相同数字的长度。
    例如:000001100 -> 记作 "5个0, 2个1, 2个0"
    """
    
    def __init__(self):
        super().__init__()
    
    @property
    def algorithm_name(self) -> str:
        return "RLE游程编码 (Run-Length)"
        
    def encode(self, batch_pixels: np.ndarray) -> bytes:
        """压缩过程：计算连续的 0 或 1 的长度"""
        # 将 3D 矩阵展平为一维直线，方便找连续的数字
        flat = batch_pixels.ravel()
        if flat.size == 0:
            return b""
            
        # 利用 numpy 快速找到数字发生变化的位置 (比如从 0 变 1)
        # flat[:-1] != flat[1:] 会对比相邻元素是否不同
        # change_idx 输出是位置变化的点比如[3, 5], 意为在位置为3、5的地方换了数字
        change_idx = np.where(flat[:-1] != flat[1:])[0] + 1
        
        # 在最前面加上起点 0，在最后面加上终点 flat.size
        # change_idx 此时变成[0, 3, 5, 9] 0是起点, 9是总长
        change_idx = np.concatenate(([0], change_idx, [flat.size]))
        
        # 计算每一段相同数字的长度 (后一个位置减去前一个位置)
        # run_lengths [3, 2, 4]意思是3个相同(0), 2个相同(1), 4个相同(0)
        run_lengths = np.diff(change_idx)
        
        # 记录第一个数字是 0 还是 1 (只需占用 1 个字节)
        first_val = np.array([flat[0]], dtype=np.uint8)
        
        # 将结果打包成纯字节流。
        # 这里用 uint32 (4字节) 存长度，防止某一段连续的 0 超过 65535 个装不下。
        compressed_bytes = first_val.tobytes() + run_lengths.astype(np.uint32).tobytes()
        
        return compressed_bytes

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        """解压过程：根据长度交替生成 0 和 1, 还原矩阵"""
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)
        if (len(compressed_bytes) - 1) % 4 != 0:
            raise ValueError("RLE 压缩流格式错误: run-length 字节数不是 uint32 的整数倍")
            
        # 拆解字节流：第 1 个字节是初始值，后面全是 4 字节的长度数据
        first_val = np.frombuffer(compressed_bytes[:1], dtype=np.uint8)[0]
        run_lengths = np.frombuffer(compressed_bytes[1:], dtype=np.uint32)
        
        # 生成交替的 0 和 1 (0, 1, 0, 1 或 1, 0, 1, 0)
        values = np.zeros(len(run_lengths), dtype=np.uint8)
        if first_val == 1:
            values[0::2] = 1  # 如果以 1 开头，偶数位置放 1
        else:
            values[1::2] = 1  # 如果以 0 开头，奇数位置放 1
            
        # 根据长度重复这些数字，并重塑回原来的 3D 形状
        flat_decoded = np.repeat(values, run_lengths)

        expected_pixels = int(np.prod(batch_shape, dtype=np.int64))
        if flat_decoded.size != expected_pixels:
            raise ValueError(
                f"RLE 解码像素数不匹配: expected={expected_pixels}, actual={flat_decoded.size}"
            )

        return flat_decoded.reshape(batch_shape)




# ==========================================
# 帧间差分 + 游程编码 (Temporal Delta + RLE)
# ==========================================
class DeltaRleCompressor(BaseCompressor):
    """
    帧间差分 + 游程编码。
    将当前帧与上一帧进行异或 (XOR) 得到极度稀疏的差分矩阵，然后再进行 RLE 压缩。
    """
    def __init__(self):
        super().__init__()

    @property
    def algorithm_name(self) -> str:
        return "帧间差分+RLE (Delta+RLE)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        # 计算帧间差分 (XOR)
        delta_pixels = np.zeros_like(batch_pixels)
        delta_pixels[0] = batch_pixels[0] 
        delta_pixels[1:] = batch_pixels[1:] ^ batch_pixels[:-1]

        # 对差分矩阵进行 RLE 压缩
        flat = delta_pixels.ravel()
        if flat.size == 0:
            return b""
            
        change_idx = np.where(flat[:-1] != flat[1:])[0] + 1
        change_idx = np.concatenate(([0], change_idx, [flat.size]))
        run_lengths = np.diff(change_idx)
        first_val = np.array([flat[0]], dtype=np.uint8)
        
        return first_val.tobytes() + run_lengths.astype(np.uint32).tobytes()

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)
        if (len(compressed_bytes) - 1) % 4 != 0:
            raise ValueError("DeltaRLE 压缩流格式错误: run-length 字节数不是 uint32 的整数倍")
            
        # RLE 解压出差分矩阵
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
        
        # 累积异或 (Cumulative XOR) 还原原始帧
        reconstructed = np.bitwise_xor.accumulate(delta_pixels, axis=0)
        return reconstructed
    


# ==========================================
# 帧间差分 + 稀疏坐标点记录 (Temporal Delta + Sparse Coordinate)
# ==========================================
class DeltaSparseCompressor(BaseCompressor):
    """
    帧间差分 + 稀疏坐标压缩。
    对差分后的矩阵，不再记录 0 的长度，而是直接记录发生翻转(为1)的像素的绝对索引。
    非常适合极其稀疏的 SPAD 噪点记录。
    """
    def __init__(self):
        super().__init__()

    @property
    def algorithm_name(self) -> str:
        return "帧间差分+点记录 (Delta+Sparse)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        _validate_sparse_index_capacity(batch_pixels)

        # 计算帧间差分 (XOR)
        delta_pixels = np.zeros_like(batch_pixels)
        delta_pixels[0] = batch_pixels[0]
        delta_pixels[1:] = batch_pixels[1:] ^ batch_pixels[:-1]

        compressed_chunks = []
        
        # 逐帧提取变化点的坐标
        for frame in delta_pixels:
            # 找到当前帧中所有为 1 的一维索引 (0 到 39999)
            # 强制转换为 uint16 极大地节省了空间 (每个坐标只占 2 字节)
            indices = np.where(frame.ravel() == 1)[0].astype(np.uint16)
            
            # 记录这一帧到底有多少个像素发生了变化 (也用 2 字节 uint16 存)
            count = np.array([len(indices)], dtype=np.uint16)
            
            # 将数量和具体的坐标拼接成二进制追加到列表
            compressed_chunks.append(count.tobytes())
            compressed_chunks.append(indices.tobytes())

        # 将所有的二进制块合并成一个完整的 bytes
        return b"".join(compressed_chunks)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        # 准备一个全黑的差分矩阵画布
        delta_pixels = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return delta_pixels

        offset = 0
        total_frames = batch_shape[0]
        
        # 逐帧还原差分矩阵
        for i in range(total_frames):
            # 先读 2 个字节，看看这帧有几个变化点
            if offset + 2 > len(compressed_bytes):
                raise ValueError("DeltaSparse 压缩流格式错误: 缺少 count 字段")
            count = int(np.frombuffer(compressed_bytes[offset:offset+2], dtype=np.uint16)[0])
            offset += 2
            
            if count > 0:
                # 根据数量，往后读取对应长度的坐标数据 (每个坐标 2 字节)
                bytes_to_read = count * 2
                if offset + bytes_to_read > len(compressed_bytes):
                    raise ValueError("DeltaSparse 压缩流格式错误: 坐标数据长度不足")
                indices = np.frombuffer(compressed_bytes[offset:offset+bytes_to_read], dtype=np.uint16)
                offset += bytes_to_read
                
                # 按照坐标，把画布上的对应像素点亮
                delta_pixels[i].ravel()[indices] = 1

        if offset != len(compressed_bytes):
            raise ValueError("DeltaSparse 压缩流格式错误: 存在未消费的尾部字节")
                
        # 累积异或还原原始帧
        reconstructed = np.bitwise_xor.accumulate(delta_pixels, axis=0)
        return reconstructed
    
# ==========================================
# 帧间差分 + 点记录 + 熵编码
# ==========================================
class DeltaSparseZlibCompressor(DeltaSparseCompressor):
    """
    在 Delta + Sparse 的基础上，加上终极武器：信息熵编码 (Zlib/DEFLATE)。
    利用父类的方法先提取极度稀疏的坐标，然后再用 zlib 榨干最后的统计冗余。
    """
    def __init__(self):
        super().__init__()

    @property
    def algorithm_name(self) -> str:
        return "帧间差分+点记录+Zlib (Delta+Sparse+Zlib)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        # 直接复用父类的方法，拿到纯粹的“坐标记录字节流”
        raw_sparse_bytes = super().encode(batch_pixels)
        
        # 使用 Zlib 进行熵压缩 (level=9 代表开启最高压缩率模式)
        compressed_bytes = zlib.compress(raw_sparse_bytes, level=9)
        return compressed_bytes

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)
            
        # 解开 zlib 的压缩包
        raw_sparse_bytes = zlib.decompress(compressed_bytes)
        
        # 把解开的坐标流交还给父类，让它去画图并还原矩阵
        return super().decode(raw_sparse_bytes, batch_shape)
    

# ==========================================
# 事件地址表示法 (Address Event Representation, AER)
# ==========================================
class AerCompressor(BaseCompressor):
    """
    真正的底层硬件 AER 协议仿真。
    将每个有效像素转换为一个独立的 32-bit 事件包：[ 16位时间 T | 8位 Y | 8位 X ]
    """
    def __init__(self, use_delta=False):
        super().__init__()
        # use_delta=False: 记录所有到来的光子 (标准 SPAD 模式)
        # use_delta=True:  只记录发生变化的像素 (标准 DVS 仿生视觉模式)
        self.use_delta = use_delta

    @property
    def algorithm_name(self) -> str:
        mode = "差分(DVS模式)" if self.use_delta else "原始(SPAD模式)"
        return f"事件地址表示法 AER ({mode})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        _validate_aer_capacity(batch_pixels.shape)

        # 判断是否要进行帧间差分提取变化点
        if self.use_delta:
            data_to_encode = np.zeros_like(batch_pixels)
            data_to_encode[0] = batch_pixels[0]
            data_to_encode[1:] = batch_pixels[1:] ^ batch_pixels[:-1]
        else:
            data_to_encode = batch_pixels

        # 找到所有为 1 的坐标 (返回 T, Y, X 的一维数组)
        t, y, x = np.nonzero(data_to_encode)
        
        # 强制转换为 32 位无符号整数，为位操作做准备
        t = t.astype(np.uint32)
        y = y.astype(np.uint32)
        x = x.astype(np.uint32)

        # 核心协议打包, 位移并融合为 32-bit 数据包
        # T 左移 16 位，Y 左移 8 位，X 不动。然后按位或 (|) 拼在一起。
        aer_events = (t << 16) | (y << 8) | x
        
        return aer_events.tobytes()

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        _validate_aer_capacity(batch_shape)

        matrix = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return matrix
        if len(compressed_bytes) % 4 != 0:
            raise ValueError("AER 压缩流格式错误: 事件流字节数不是 uint32 的整数倍")

        # 直接以 uint32 读取所有事件包
        aer_events = np.frombuffer(compressed_bytes, dtype=np.uint32)
        
        # 核心协议解包, 使用位掩码提取 T, Y, X
        # 0xFFFF 是 16 个 1 (提取低16位)；0xFF 是 8 个 1 (提取低8位)
        t = (aer_events >> 16) & 0xFFFF
        y = (aer_events >> 8) & 0xFF
        x = aer_events & 0xFF

        # 将事件点亮到矩阵中
        matrix[t, y, x] = 1

        # 如果是差分模式，需要累积还原
        if self.use_delta:
            matrix = np.bitwise_xor.accumulate(matrix, axis=0)

        return matrix
    

# ==========================================
# 算法 6: 时域累加 + 熵编码 (Temporal Binning + Zlib)
# ==========================================
class TemporalBinningCompressor(BaseCompressor):
    """
    注意：这是一种时间维度的“有损”压缩！

    片上直方图/时域累加架构。
    将连续的 N 帧二值图像，在时间轴上相加，输出 8-bit 灰度图像。
    然后再使用 Zlib 模拟常规的无损图像/视频内插压缩。
    """
    def __init__(self, bin_size=255):
        super().__init__()
        # 默认累加 255 帧，因为 255 刚好是 8-bit (uint8) 的最大值，完美契合字节边界
        if not isinstance(bin_size, int) or bin_size <= 0:
            raise ValueError("TemporalBinningCompressor.bin_size 必须为正整数")
        if bin_size > UINT8_MAX:
            raise ValueError("TemporalBinningCompressor.bin_size 不能超过 255，否则 uint8 会溢出")
        self.bin_size = bin_size

    @property
    def algorithm_name(self) -> str:
        return f"时域累加(Binning={self.bin_size})+Zlib"

    @property
    def is_lossless(self) -> bool:
        return False

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        T, Y, X = batch_pixels.shape
        
        # 计算当前 batch 能切成多少个完整的累加块
        num_bins = int(np.ceil(T / self.bin_size))
        binned_frames = []
        
        # 模拟芯片上的累加器 (Accumulator)
        for i in range(num_bins):
            start = i * self.bin_size
            end = min((i + 1) * self.bin_size, T)
            
            # 切出这几百帧，然后顺着时间轴 (axis=0) 全部加起来
            chunk = batch_pixels[start:end]
            binned_frame = np.sum(chunk, axis=0, dtype=np.uint8)
            binned_frames.append(binned_frame)
            
        # 合并成一块 8-bit 的灰度视频矩阵
        binned_array = np.array(binned_frames, dtype=np.uint8)
        
        # 调用 Zlib 榨干由于画面大面积黑色带来的空间冗余 (类似 PNG 压缩)
        compressed_bytes = zlib.compress(binned_array.tobytes(), level=9)
        
        # 写入 2 个字节的头部，记录到底压成了几帧灰度图，方便解压时使用
        shape_header = np.array([len(binned_frames)], dtype=np.uint16).tobytes()
        
        return shape_header + compressed_bytes

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        # 时域累加是不可逆的！这里只能做 伪还原
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)
        if len(compressed_bytes) < 2:
            raise ValueError("TemporalBinning 压缩流格式错误: 缺少 bin 数量头部")
            
        # 读取头部和压缩数据
        num_bins = int(np.frombuffer(compressed_bytes[:2], dtype=np.uint16)[0])
        zlib_data = compressed_bytes[2:]
        
        # 解压出 8-bit 灰度图
        raw_bytes = zlib.decompress(zlib_data)
        _, Y, X = batch_shape
        expected_bytes = num_bins * Y * X
        if len(raw_bytes) != expected_bytes:
            raise ValueError(
                f"TemporalBinning 解码尺寸不匹配: expected={expected_bytes}, actual={len(raw_bytes)}"
            )
        binned_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((num_bins, Y, X))
        
        # 伪还原：把 8-bit 灰度图的非零像素，粗暴地塞回这 255 帧里的“第一帧”
        # 这必然会导致与原数据完全不同，但能保持矩阵形状不崩溃
        reconstructed = np.zeros(batch_shape, dtype=np.uint8)
        for i in range(num_bins):
            start = i * self.bin_size
            # 只要这个像素累加值大于 0，我们就假装第一帧亮了
            reconstructed[start] = (binned_array[i] > 0).astype(np.uint8)
            
        return reconstructed


def _validate_frame_index_capacity(shape: tuple):
    frames = shape[0]
    if frames - 1 > UINT16_MAX:
        raise ValueError(
            "当前压缩流使用 uint16 存储帧索引，"
            f"batch 帧数 {frames} 超出可编码范围 {UINT16_MAX + 1}"
        )


def _validate_row_sparse_capacity(shape: tuple):
    _, height, width = shape
    if height - 1 > UINT8_MAX or width - 1 > UINT8_MAX:
        raise ValueError(
            "RowSparseZlibCompressor 当前使用 uint8 存储行列坐标，"
            f"输入尺寸 {height}x{width} 超出可编码范围 256x256"
        )


def _validate_zlib_level(level: int):
    if not isinstance(level, int) or not 0 <= level <= 9:
        raise ValueError("zlib level 必须是 0 到 9 的整数")


def _encode_varints(values: np.ndarray) -> bytes:
    encoded = bytearray()
    for value in np.asarray(values, dtype=np.uint32):
        remaining = int(value)
        while True:
            current_byte = remaining & 0x7F
            remaining >>= 7
            if remaining:
                encoded.append(current_byte | 0x80)
            else:
                encoded.append(current_byte)
                break
    return bytes(encoded)


def _decode_varints(buffer: bytes, offset: int, count: int):
    decoded = np.empty(count, dtype=np.uint32)
    for idx in range(count):
        value = 0
        shift = 0
        while True:
            if offset >= len(buffer):
                raise ValueError("Varint 解码失败: 数据提前结束")
            current_byte = buffer[offset]
            offset += 1
            value |= (current_byte & 0x7F) << shift
            if current_byte < 0x80:
                break
            shift += 7
            if shift > 28:
                raise ValueError("Varint 解码失败: 数值过大或格式损坏")
        decoded[idx] = value
    return decoded, offset


_MORTON_ORDER_CACHE = {}


def _part1by1(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32) & 0x0000FFFF
    values = (values | (values << 8)) & 0x00FF00FF
    values = (values | (values << 4)) & 0x0F0F0F0F
    values = (values | (values << 2)) & 0x33333333
    values = (values | (values << 1)) & 0x55555555
    return values


def _get_morton_permutation(height: int, width: int):
    cache_key = (height, width)
    if cache_key not in _MORTON_ORDER_CACHE:
        y_coords, x_coords = np.indices((height, width), dtype=np.uint32)
        morton_codes = (_part1by1(y_coords.ravel()) << 1) | _part1by1(x_coords.ravel())
        permutation = np.argsort(morton_codes, kind="stable")
        inverse_permutation = np.empty_like(permutation)
        inverse_permutation[permutation] = np.arange(permutation.size, dtype=permutation.dtype)
        _MORTON_ORDER_CACHE[cache_key] = (permutation, inverse_permutation)
    return _MORTON_ORDER_CACHE[cache_key]


def _resolve_ffmpeg_executable() -> str:
    ffmpeg_path = os.environ.get("SPAD_FFMPEG_PATH")
    if ffmpeg_path:
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path
        raise RuntimeError(
            f"SPAD_FFMPEG_PATH 已设置，但未找到 ffmpeg: {ffmpeg_path}"
        )

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    raise RuntimeError(
        "未找到 ffmpeg。请先把 ffmpeg 加入 PATH, 或设置环境变量 SPAD_FFMPEG_PATH。"
    )


def _run_ffmpeg(command):
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg 执行失败: {stderr.strip()}")


def _ensure_uint8_binary_frames(batch_pixels: np.ndarray) -> np.ndarray:
    return (np.asarray(batch_pixels, dtype=np.uint8) > 0).astype(np.uint8) * 255


# ==========================================
# 原始比特打包 + Zlib
# ==========================================
class PackBitsZlibCompressor(BaseCompressor):
    """
    最直接的无损基线。
    先把 0/1 像素按 bit 打包到字节，再做 Zlib 熵编码。
    当数据没有明显时空结构时，它通常是很好的参考下界。
    """
    def __init__(self, level=9):
        super().__init__()
        _validate_zlib_level(level)
        self.level = level

    @property
    def algorithm_name(self) -> str:
        return f"比特打包+Zlib (PackBits+Zlib, level={self.level})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        if batch_pixels.size == 0:
            return b""
        packed = np.packbits(batch_pixels.ravel())
        return zlib.compress(packed.tobytes(), level=self.level)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)

        expected_pixels = int(np.prod(batch_shape, dtype=np.int64))
        expected_bytes = (expected_pixels + 7) // 8
        packed_bytes = zlib.decompress(compressed_bytes)
        if len(packed_bytes) != expected_bytes:
            raise ValueError(
                f"PackBitsZlib 解码尺寸不匹配: expected={expected_bytes}, actual={len(packed_bytes)}"
            )

        unpacked = np.unpackbits(
            np.frombuffer(packed_bytes, dtype=np.uint8),
            count=expected_pixels,
        )
        return unpacked.astype(np.uint8).reshape(batch_shape)


# ==========================================
# 零帧抑制 + 稀疏点记录 (Frame Zero Suppression)
# ==========================================
class FrameZeroSuppressionCompressor(BaseCompressor):
    """
    只记录非空帧。
    对于 250,000 fps 下大部分帧全黑、只有少量帧出现事件的场景，这个方法通常优于逐帧固定开销的编码。
    """
    def __init__(self):
        super().__init__()

    @property
    def algorithm_name(self) -> str:
        return "零帧抑制+点记录 (FrameZeroSuppression)"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        _validate_sparse_index_capacity(batch_pixels)
        _validate_frame_index_capacity(batch_pixels.shape)

        total_frames = batch_pixels.shape[0]
        flat_frames = batch_pixels.reshape(total_frames, -1)
        non_empty_frames = np.flatnonzero(np.any(flat_frames, axis=1))

        compressed_chunks = [np.array([len(non_empty_frames)], dtype=np.uint16).tobytes()]
        for frame_idx in non_empty_frames:
            indices = np.flatnonzero(flat_frames[frame_idx]).astype(np.uint16)
            header = np.array([frame_idx, len(indices)], dtype=np.uint16)
            compressed_chunks.append(header.tobytes())
            compressed_chunks.append(indices.tobytes())

        return b"".join(compressed_chunks)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        matrix = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return matrix
        if len(compressed_bytes) < 2:
            raise ValueError("FrameZeroSuppression 压缩流格式错误: 缺少非空帧数量头部")

        offset = 0
        entry_count = int(np.frombuffer(compressed_bytes[offset:offset+2], dtype=np.uint16)[0])
        offset += 2

        for _ in range(entry_count):
            if offset + 4 > len(compressed_bytes):
                raise ValueError("FrameZeroSuppression 压缩流格式错误: 缺少帧头")
            frame_idx, count = np.frombuffer(compressed_bytes[offset:offset+4], dtype=np.uint16)
            offset += 4

            bytes_to_read = int(count) * 2
            if offset + bytes_to_read > len(compressed_bytes):
                raise ValueError("FrameZeroSuppression 压缩流格式错误: 坐标数据长度不足")
            indices = np.frombuffer(compressed_bytes[offset:offset+bytes_to_read], dtype=np.uint16)
            offset += bytes_to_read
            matrix[int(frame_idx)].ravel()[indices] = 1

        if offset != len(compressed_bytes):
            raise ValueError("FrameZeroSuppression 压缩流格式错误: 存在未消费的尾部字节")

        return matrix


# ==========================================
# 行稀疏坐标 + Zlib (Row-Sparse)
# ==========================================
class RowSparseZlibCompressor(BaseCompressor):
    """
    按行记录稀疏事件。
    如果光斑、边缘或条纹集中在少数几行，这种表示会比记录 16-bit 绝对索引更省。
    """
    def __init__(self, level=9):
        super().__init__()
        _validate_zlib_level(level)
        self.level = level

    @property
    def algorithm_name(self) -> str:
        return f"行稀疏坐标+Zlib (RowSparse+Zlib, level={self.level})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        _validate_row_sparse_capacity(batch_pixels.shape)

        compressed_chunks = []
        for frame in batch_pixels:
            non_empty_rows = np.flatnonzero(np.any(frame, axis=1))
            compressed_chunks.append(np.array([len(non_empty_rows)], dtype=np.uint8).tobytes())

            for row_idx in non_empty_rows:
                columns = np.flatnonzero(frame[row_idx]).astype(np.uint8)
                row_header = np.array([row_idx, len(columns)], dtype=np.uint8)
                compressed_chunks.append(row_header.tobytes())
                compressed_chunks.append(columns.tobytes())

        raw_bytes = b"".join(compressed_chunks)
        return zlib.compress(raw_bytes, level=self.level)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        _validate_row_sparse_capacity(batch_shape)

        matrix = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return matrix

        raw_bytes = zlib.decompress(compressed_bytes)
        offset = 0
        total_frames = batch_shape[0]

        for frame_idx in range(total_frames):
            if offset + 1 > len(raw_bytes):
                raise ValueError("RowSparseZlib 压缩流格式错误: 缺少行数量字段")
            row_count = int(np.frombuffer(raw_bytes[offset:offset+1], dtype=np.uint8)[0])
            offset += 1

            for _ in range(row_count):
                if offset + 2 > len(raw_bytes):
                    raise ValueError("RowSparseZlib 压缩流格式错误: 缺少行头")
                row_idx, column_count = np.frombuffer(raw_bytes[offset:offset+2], dtype=np.uint8)
                offset += 2

                bytes_to_read = int(column_count)
                if offset + bytes_to_read > len(raw_bytes):
                    raise ValueError("RowSparseZlib 压缩流格式错误: 列坐标数据长度不足")
                columns = np.frombuffer(raw_bytes[offset:offset+bytes_to_read], dtype=np.uint8)
                offset += bytes_to_read
                matrix[frame_idx, int(row_idx), columns] = 1

        if offset != len(raw_bytes):
            raise ValueError("RowSparseZlib 压缩流格式错误: 存在未消费的尾部字节")

        return matrix


# ==========================================
# 帧间差分 + 间隔编码 + Zlib (Gap Sparse)
# ==========================================
class DeltaGapZlibCompressor(BaseCompressor):
    """
    对差分后事件点的排序索引做 gap 编码，再用 Varint 和 Zlib 压缩。
    当事件在空间上成团出现、相邻索引间隔较小时，通常比直接存 16-bit 绝对索引更省。
    """
    def __init__(self, level=9):
        super().__init__()
        _validate_sparse_index_capacity(np.zeros((1, 200, 200), dtype=np.uint8))
        _validate_zlib_level(level)
        self.level = level

    @property
    def algorithm_name(self) -> str:
        return f"帧间差分+间隔编码+Zlib (DeltaGap+Zlib, level={self.level})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        _validate_sparse_index_capacity(batch_pixels)

        delta_pixels = np.zeros_like(batch_pixels)
        delta_pixels[0] = batch_pixels[0]
        delta_pixels[1:] = batch_pixels[1:] ^ batch_pixels[:-1]

        compressed_chunks = []
        for frame in delta_pixels:
            indices = np.flatnonzero(frame.ravel())
            compressed_chunks.append(np.array([len(indices)], dtype=np.uint16).tobytes())
            if indices.size == 0:
                continue

            compressed_chunks.append(np.array([indices[0]], dtype=np.uint16).tobytes())
            if indices.size > 1:
                gaps = np.diff(indices)
                compressed_chunks.append(_encode_varints(gaps))

        raw_bytes = b"".join(compressed_chunks)
        return zlib.compress(raw_bytes, level=self.level)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)

        raw_bytes = zlib.decompress(compressed_bytes)
        delta_pixels = np.zeros(batch_shape, dtype=np.uint8)
        offset = 0

        for frame_idx in range(batch_shape[0]):
            if offset + 2 > len(raw_bytes):
                raise ValueError("DeltaGapZlib 压缩流格式错误: 缺少 count 字段")
            count = int(np.frombuffer(raw_bytes[offset:offset+2], dtype=np.uint16)[0])
            offset += 2

            if count == 0:
                continue

            if offset + 2 > len(raw_bytes):
                raise ValueError("DeltaGapZlib 压缩流格式错误: 缺少首个索引字段")
            first_index = int(np.frombuffer(raw_bytes[offset:offset+2], dtype=np.uint16)[0])
            offset += 2

            indices = np.empty(count, dtype=np.uint32)
            indices[0] = first_index
            if count > 1:
                gaps, offset = _decode_varints(raw_bytes, offset, count - 1)
                indices[1:] = first_index + np.cumsum(gaps, dtype=np.uint32)

            delta_pixels[frame_idx].ravel()[indices.astype(np.intp)] = 1

        if offset != len(raw_bytes):
            raise ValueError("DeltaGapZlib 压缩流格式错误: 存在未消费的尾部字节")

        return np.bitwise_xor.accumulate(delta_pixels, axis=0)


# ==========================================
# 全局事件流 + 间隔编码 + Zlib (Global Event Stream)
# ==========================================
class GlobalEventStreamCompressor(BaseCompressor):
    """
    把整个 batch 视作一个全局 3D 稀疏事件流。
    直接记录所有 1 像素在时空展开后的一维位置，再做 gap 编码和 Zlib。
    当原始数据本身就极稀疏时，这通常是一个很强的无损参考方法。
    """
    def __init__(self, level=9):
        super().__init__()
        _validate_zlib_level(level)
        self.level = level

    @property
    def algorithm_name(self) -> str:
        return f"全局事件流+间隔编码+Zlib (GlobalEventStream, level={self.level})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        if batch_pixels.size == 0:
            return b""

        indices = np.flatnonzero(batch_pixels.ravel())
        header = np.array([indices.size], dtype=np.uint32).tobytes()
        if indices.size == 0:
            return header + zlib.compress(b"", level=self.level)

        payload = bytearray()
        payload.extend(np.array([indices[0]], dtype=np.uint32).tobytes())
        if indices.size > 1:
            payload.extend(_encode_varints(np.diff(indices)))

        return header + zlib.compress(bytes(payload), level=self.level)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        matrix = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return matrix
        if len(compressed_bytes) < 4:
            raise ValueError("GlobalEventStream 压缩流格式错误: 缺少事件数量头部")

        event_count = int(np.frombuffer(compressed_bytes[:4], dtype=np.uint32)[0])
        payload = zlib.decompress(compressed_bytes[4:])
        if event_count == 0:
            if payload:
                raise ValueError("GlobalEventStream 压缩流格式错误: 空事件流不应含负载")
            return matrix

        if len(payload) < 4:
            raise ValueError("GlobalEventStream 压缩流格式错误: 缺少首个全局索引")
        first_index = int(np.frombuffer(payload[:4], dtype=np.uint32)[0])
        indices = np.empty(event_count, dtype=np.uint32)
        indices[0] = first_index

        offset = 4
        if event_count > 1:
            gaps, offset = _decode_varints(payload, offset, event_count - 1)
            indices[1:] = first_index + np.cumsum(gaps, dtype=np.uint32)
        if offset != len(payload):
            raise ValueError("GlobalEventStream 压缩流格式错误: 存在未消费的尾部字节")

        matrix.ravel()[indices.astype(np.intp)] = 1
        return matrix


# ==========================================
# 块稀疏位图 + Zlib (Block Sparse Bitmap)
# ==========================================
class BlockSparseBitmapCompressor(BaseCompressor):
    """
    将每帧切成小块，只记录非空块及块内位图。
    当事件在局部区域聚集但块内密度高于纯稀疏点记录时，这类方法通常更合适。
    """
    def __init__(self, block_size=8, level=9):
        super().__init__()
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("BlockSparseBitmapCompressor.block_size 必须为正整数")
        _validate_zlib_level(level)
        self.block_size = block_size
        self.level = level

    @property
    def algorithm_name(self) -> str:
        return (
            f"块稀疏位图+Zlib (BlockSparseBitmap, block={self.block_size}, level={self.level})"
        )

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        height = batch_pixels.shape[1]
        width = batch_pixels.shape[2]
        blocks_y = (height + self.block_size - 1) // self.block_size
        blocks_x = (width + self.block_size - 1) // self.block_size

        compressed_chunks = []
        for frame in batch_pixels:
            frame_chunks = []
            non_empty_block_count = 0

            for block_y in range(blocks_y):
                y_start = block_y * self.block_size
                y_end = min(y_start + self.block_size, height)

                for block_x in range(blocks_x):
                    x_start = block_x * self.block_size
                    x_end = min(x_start + self.block_size, width)
                    block = frame[y_start:y_end, x_start:x_end]
                    if not np.any(block):
                        continue

                    non_empty_block_count += 1
                    block_id = block_y * blocks_x + block_x
                    packed_block = np.packbits(block.ravel())
                    frame_chunks.append(np.array([block_id], dtype=np.uint16).tobytes())
                    frame_chunks.append(packed_block.tobytes())

            compressed_chunks.append(np.array([non_empty_block_count], dtype=np.uint16).tobytes())
            compressed_chunks.extend(frame_chunks)

        raw_bytes = b"".join(compressed_chunks)
        return zlib.compress(raw_bytes, level=self.level)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        matrix = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return matrix

        height = batch_shape[1]
        width = batch_shape[2]
        blocks_x = (width + self.block_size - 1) // self.block_size
        raw_bytes = zlib.decompress(compressed_bytes)
        offset = 0

        for frame_idx in range(batch_shape[0]):
            if offset + 2 > len(raw_bytes):
                raise ValueError("BlockSparseBitmap 压缩流格式错误: 缺少块数量字段")
            block_count = int(np.frombuffer(raw_bytes[offset:offset+2], dtype=np.uint16)[0])
            offset += 2

            for _ in range(block_count):
                if offset + 2 > len(raw_bytes):
                    raise ValueError("BlockSparseBitmap 压缩流格式错误: 缺少块索引字段")
                block_id = int(np.frombuffer(raw_bytes[offset:offset+2], dtype=np.uint16)[0])
                offset += 2

                block_y = block_id // blocks_x
                block_x = block_id % blocks_x
                y_start = block_y * self.block_size
                x_start = block_x * self.block_size
                y_end = min(y_start + self.block_size, height)
                x_end = min(x_start + self.block_size, width)
                block_height = y_end - y_start
                block_width = x_end - x_start
                packed_size = (block_height * block_width + 7) // 8

                if offset + packed_size > len(raw_bytes):
                    raise ValueError("BlockSparseBitmap 压缩流格式错误: 块位图数据长度不足")
                packed_block = raw_bytes[offset:offset+packed_size]
                offset += packed_size

                unpacked = np.unpackbits(
                    np.frombuffer(packed_block, dtype=np.uint8),
                    count=block_height * block_width,
                ).astype(np.uint8)
                matrix[frame_idx, y_start:y_end, x_start:x_end] = unpacked.reshape(
                    block_height,
                    block_width,
                )

        if offset != len(raw_bytes):
            raise ValueError("BlockSparseBitmap 压缩流格式错误: 存在未消费的尾部字节")

        return matrix


# ==========================================
# Morton 空间重排 + 比特打包 + Zlib
# ==========================================
class MortonPackBitsZlibCompressor(BaseCompressor):
    """
    先按 Morton(Z-order) 重新排列空间像素，再比特打包并做 Zlib。
    它的目标是把空间上邻近的事件在一维流中也尽量放近，从而提高后端熵编码效率。
    """
    def __init__(self, level=9):
        super().__init__()
        _validate_zlib_level(level)
        self.level = level

    @property
    def algorithm_name(self) -> str:
        return f"Morton重排+比特打包+Zlib (MortonPackBits, level={self.level})"

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        if batch_pixels.size == 0:
            return b""

        _, height, width = batch_pixels.shape
        permutation, _ = _get_morton_permutation(height, width)
        reordered = batch_pixels.reshape(batch_pixels.shape[0], -1)[:, permutation]
        packed = np.packbits(reordered.ravel())
        return zlib.compress(packed.tobytes(), level=self.level)

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)

        total_frames, height, width = batch_shape
        expected_pixels = int(np.prod(batch_shape, dtype=np.int64))
        expected_bytes = (expected_pixels + 7) // 8
        packed_bytes = zlib.decompress(compressed_bytes)
        if len(packed_bytes) != expected_bytes:
            raise ValueError(
                f"MortonPackBits 解码尺寸不匹配: expected={expected_bytes}, actual={len(packed_bytes)}"
            )

        _, inverse_permutation = _get_morton_permutation(height, width)
        unpacked = np.unpackbits(
            np.frombuffer(packed_bytes, dtype=np.uint8),
            count=expected_pixels,
        ).reshape(total_frames, height * width)
        restored = unpacked[:, inverse_permutation]
        return restored.astype(np.uint8).reshape(batch_shape)


# ==========================================
# 视频编码器封装 (Experimental Video Codec)
# ==========================================
class _FfmpegVideoCompressor(BaseCompressor):
    """
    基于 ffmpeg 的实验性视频压缩封装。
    将 SPAD 二值帧当作灰度视频交给传统视频编码器，以评估跨帧预测在事件流上的效果。
    """
    codec_name = "ffmpeg"
    file_extension = ".mkv"

    def __init__(self, crf: int, preset: str):
        super().__init__()
        if not isinstance(crf, int) or crf < 0:
            raise ValueError("视频编码器的 crf 必须是非负整数")
        self.crf = crf
        self.preset = preset

    @property
    def is_lossless(self) -> bool:
        return False

    def _build_encode_command(self, ffmpeg_path: str, raw_input_path: str, output_path: str, width: int, height: int, fps: int):
        raise NotImplementedError()

    def encode(self, batch_pixels: np.ndarray) -> bytes:
        if batch_pixels.size == 0:
            return b""

        total_frames, height, width = batch_pixels.shape
        grayscale_frames = _ensure_uint8_binary_frames(batch_pixels)
        ffmpeg_path = _resolve_ffmpeg_executable()
        fps = max(total_frames, 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_input_path = os.path.join(temp_dir, "input.raw")
            output_path = os.path.join(temp_dir, f"output{self.file_extension}")

            with open(raw_input_path, "wb") as raw_file:
                raw_file.write(grayscale_frames.tobytes())

            command = self._build_encode_command(
                ffmpeg_path,
                raw_input_path,
                output_path,
                width,
                height,
                fps,
            )
            _run_ffmpeg(command)

            with open(output_path, "rb") as output_file:
                return output_file.read()

    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        if not compressed_bytes:
            return np.zeros(batch_shape, dtype=np.uint8)

        total_frames, height, width = batch_shape
        ffmpeg_path = _resolve_ffmpeg_executable()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, f"input{self.file_extension}")
            output_path = os.path.join(temp_dir, "decoded.raw")

            with open(input_path, "wb") as input_file:
                input_file.write(compressed_bytes)

            command = [
                ffmpeg_path,
                "-y",
                "-i",
                input_path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                output_path,
            ]
            _run_ffmpeg(command)

            with open(output_path, "rb") as output_file:
                raw_bytes = output_file.read()

        expected_bytes = total_frames * height * width
        if len(raw_bytes) != expected_bytes:
            raise ValueError(
                f"视频解码尺寸不匹配: expected={expected_bytes}, actual={len(raw_bytes)}"
            )

        decoded = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(batch_shape)
        return (decoded >= 128).astype(np.uint8)


# ==========================================
# H.264 视频压缩 (Experimental)
# ==========================================
class H264VideoCompressor(_FfmpegVideoCompressor):
    """
    实验性 H.264 压缩器。
    用传统视频编码器的帧内/帧间预测能力，测试 SPAD 二值视频能否从视频链路中受益。
    """
    codec_name = "libx264"
    file_extension = ".mp4"

    def __init__(self, crf=18, preset="medium"):
        super().__init__(crf=crf, preset=preset)

    @property
    def algorithm_name(self) -> str:
        return f"H.264视频压缩 (CRF={self.crf}, preset={self.preset})"

    def _build_encode_command(self, ffmpeg_path: str, raw_input_path: str, output_path: str, width: int, height: int, fps: int):
        return [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            raw_input_path,
            "-an",
            "-c:v",
            self.codec_name,
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]


# ==========================================
# H.265 视频压缩 (Experimental)
# ==========================================
class H265VideoCompressor(_FfmpegVideoCompressor):
    """
    实验性 H.265 压缩器。
    相比 H.264 更偏向高压缩率，适合在调研阶段测试更强时域预测是否值得。
    """
    codec_name = "libx265"
    file_extension = ".mp4"

    def __init__(self, crf=28, preset="medium"):
        super().__init__(crf=crf, preset=preset)

    @property
    def algorithm_name(self) -> str:
        return f"H.265视频压缩 (CRF={self.crf}, preset={self.preset})"

    def _build_encode_command(self, ffmpeg_path: str, raw_input_path: str, output_path: str, width: int, height: int, fps: int):
        return [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            raw_input_path,
            "-an",
            "-c:v",
            self.codec_name,
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
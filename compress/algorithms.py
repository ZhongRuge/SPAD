import numpy as np
from base_compressor import BaseCompressor

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
            
        # RLE 解压出差分矩阵
        first_val = np.frombuffer(compressed_bytes[:1], dtype=np.uint8)[0]
        run_lengths = np.frombuffer(compressed_bytes[1:], dtype=np.uint32)
        
        values = np.zeros(len(run_lengths), dtype=np.uint8)
        if first_val == 1:
            values[0::2] = 1
        else:
            values[1::2] = 1
            
        flat_decoded = np.repeat(values, run_lengths)
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
            count = int(np.frombuffer(compressed_bytes[offset:offset+2], dtype=np.uint16)[0])
            offset += 2
            
            if count > 0:
                # 根据数量，往后读取对应长度的坐标数据 (每个坐标 2 字节)
                bytes_to_read = count * 2
                indices = np.frombuffer(compressed_bytes[offset:offset+bytes_to_read], dtype=np.uint16)
                offset += bytes_to_read
                
                # 按照坐标，把画布上的对应像素点亮
                delta_pixels[i].ravel()[indices] = 1
                
        # 累积异或还原原始帧
        reconstructed = np.bitwise_xor.accumulate(delta_pixels, axis=0)
        return reconstructed
    

import zlib
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
        matrix = np.zeros(batch_shape, dtype=np.uint8)
        if not compressed_bytes:
            return matrix

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
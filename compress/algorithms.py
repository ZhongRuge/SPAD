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

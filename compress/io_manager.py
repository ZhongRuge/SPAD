import json
import math
import numpy as np
import os
import struct

class SpadIOManager:
    """
    SPAD 数据输入输出管理器
    负责流式读取、自动解包二进制数据，并提供写入压缩流的接口。
    """
    def __init__(self, meta_path, data_path):
        self.meta_path = meta_path
        self.data_path = data_path
        
        # 解析元数据
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"找不到元数据文件: {meta_path}")
            
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
            
        self.width = self.meta["width"]
        self.height = self.meta["height"]
        self.total_frames = self.meta["total_frames"]
        self.save_as_bits = self.meta.get("save_as_bits", True)
        self.chunk_header_format = "<II"
        self.chunk_header_size = struct.calcsize(self.chunk_header_format)
        
        # 计算单帧物理大小
        self.pixels_per_frame = self.width * self.height
        if self.save_as_bits:
            self.bytes_per_frame = math.ceil(self.pixels_per_frame / 8)
        else:
            self.bytes_per_frame = self.pixels_per_frame

    def stream_batches(self, batch_size=1000):
        """
        生成器：按批次读取数据，并自动还原为 0/1 矩阵。
        使用 batch_size 可以极大提升 Python 的处理速度，避免频繁的 I/O 调用。
        
        :yield: 形状为 (current_batch_size, height, width) 的 numpy 数组 (dtype=uint8)
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"找不到数据文件: {self.data_path}")

        with open(self.data_path, "rb") as f:
            for start_frame in range(0, self.total_frames, batch_size):
                current_batch_size = min(batch_size, self.total_frames - start_frame)
                read_bytes = self.bytes_per_frame * current_batch_size
                
                raw_bytes = f.read(read_bytes)
                if not raw_bytes:
                    break
                if len(raw_bytes) != read_bytes:
                    raise ValueError(
                        f"数据文件被截断: 期望读取 {read_bytes} 字节，实际只读到 {len(raw_bytes)} 字节"
                    )
                    
                # 将字节流转换为一维 numpy 数组
                raw_array = np.frombuffer(raw_bytes, dtype=np.uint8)
                
                # 如果是位打包存储，自动解包为 0 和 1
                if self.save_as_bits:
                    framed_bytes = raw_array.reshape((current_batch_size, self.bytes_per_frame))
                    batch_pixels = np.unpackbits(framed_bytes, axis=1)[:, :self.pixels_per_frame]
                else:
                    batch_pixels = raw_array.reshape((current_batch_size, self.pixels_per_frame))
                    
                # 重塑为三维张量 (帧数, 高, 宽)，直接交给算法层
                yield batch_pixels.reshape((current_batch_size, self.height, self.width))

    def init_compressed_file(self, output_path):
        """初始化（清空）压缩输出文件"""
        with open(output_path, "wb") as f:
            pass # 仅清空/创建文件

    def append_compressed_bytes(self, output_path, compressed_bytes):
        """将压缩后的纯字节流追加写入文件"""
        with open(output_path, "ab") as f:
            f.write(compressed_bytes)

    def get_original_size_bytes(self):
        """获取原始二进制文件的大小"""
        return os.path.getsize(self.data_path)
    
    

    def append_compressed_chunk(self, output_path, compressed_bytes, frame_count):
        """写入数据块时，记录块长度和该块对应的帧数。"""
        chunk_size = len(compressed_bytes)
        with open(output_path, "ab") as f:
            f.write(struct.pack(self.chunk_header_format, chunk_size, int(frame_count)))
            f.write(compressed_bytes)

    def stream_compressed_chunks(self, output_path):
        """从磁盘流式读取压缩数据，严格依靠头部中的块长度和帧数拆分。"""
        with open(output_path, "rb") as f:
            while True:
                header_bytes = f.read(self.chunk_header_size)
                if not header_bytes:
                    break
                if len(header_bytes) != self.chunk_header_size:
                    raise ValueError("压缩文件头部不完整，无法解析 chunk 信息")

                chunk_size, frame_count = struct.unpack(self.chunk_header_format, header_bytes)
                
                compressed_bytes = f.read(chunk_size)
                if len(compressed_bytes) != chunk_size:
                    raise ValueError(
                        f"压缩文件被截断: 期望读取 {chunk_size} 字节，实际只读到 {len(compressed_bytes)} 字节"
                    )

                yield frame_count, compressed_bytes

    def get_batch_shape(self, frame_count):
        return (int(frame_count), self.height, self.width)
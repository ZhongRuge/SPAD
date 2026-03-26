import time
import numpy as np
from io_manager import SpadIOManager
from base_compressor import BaseCompressor
import os

class CompressorEvaluator:
    """
    压缩算法评估器
    负责统筹 IO 管理器和压缩算法，执行自动化测试，并输出性能报告。
    """
    def __init__(
        self,
        io_manager: SpadIOManager,
        compressor: BaseCompressor,
        batch_size: int = 1000,
        verify_lossless: bool = True,
    ):
        self.io = io_manager
        self.compressor = compressor
        self.batch_size = int(batch_size)
        self.verify_lossless = bool(verify_lossless)

    def run_evaluation(self, output_path: str):
        print(f"\n========== 开始评估算法: {self.compressor.algorithm_name} ==========")
        self.io.init_compressed_file(output_path)

        total_encode_time = 0.0
        total_decode_time = 0.0
        total_original_bytes = self.io.get_original_size_bytes()
        is_lossless = self.compressor.is_lossless
        batch_size = self.batch_size

        # 按批次读取纯 0/1 的像素矩阵
        for batch_pixels in self.io.stream_batches(batch_size=batch_size):
            batch_shape = batch_pixels.shape

            # 测试压缩并计时
            start_time = time.time()
            compressed_bytes = self.compressor.encode(batch_pixels)
            encode_time = time.time() - start_time
            
            total_encode_time += encode_time
            
            # 写入磁盘
            self.io.append_compressed_chunk(output_path, compressed_bytes, batch_shape[0])

        total_compressed_bytes = os.path.getsize(output_path)

        raw_stream = self.io.stream_batches(batch_size=batch_size)
        
        # 使用流式生成器，从硬盘里把一个个压缩块(Chunk)抠出来
        chunk_stream = self.io.stream_compressed_chunks(output_path)

        for batch_idx, (raw_pixels, chunk_info) in enumerate(zip(raw_stream, chunk_stream)):
            frame_count, compressed_chunk = chunk_info
            batch_shape = self.io.get_batch_shape(frame_count)
            if raw_pixels.shape != batch_shape:
                raise ValueError(
                    f"原始数据批次形状 {raw_pixels.shape} 与压缩块头部记录的形状 {batch_shape} 不一致"
                )

            # 测试解压并计时
            start_time = time.time()
            decoded_pixels = self.compressor.decode(compressed_chunk, batch_shape)
            decode_time = time.time() - start_time
            total_decode_time += decode_time

            # 无损验证对比解压后的矩阵和原矩阵是否一样
            if self.verify_lossless and self.compressor.is_lossless and not np.array_equal(raw_pixels, decoded_pixels):
                is_lossless = False
                print(f"第 {batch_idx + 1} 批次解压数据与原始数据不匹配！")

        try:
            next(raw_stream)
            raise ValueError("压缩块数量少于原始数据批次数，评估结果不完整")
        except StopIteration:
            pass

        try:
            next(chunk_stream)
            raise ValueError("压缩块数量多于原始数据批次数，压缩文件结构异常")
        except StopIteration:
            pass

        # 计算并打印评估报告
        cr = total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0
        
        print("\n========== 评估报告 ==========")
        print(f"算法名称: {self.compressor.algorithm_name}")
        if self.compressor.is_lossless and self.verify_lossless:
            print(f"数据无损校验: {'通过 (Lossless)' if is_lossless else '失败 (Lossy)'}")
        elif self.compressor.is_lossless:
            print("Lossless verification: skipped by config (verify_lossless=false)")
        else:
            print("数据无损校验: 跳过，当前算法设计为有损压缩")
        print(f"基准数据体积: {total_original_bytes / 1024:.2f} KB")
        print(f"压缩后的体积: {total_compressed_bytes / 1024:.2f} KB")
        print(f"整体压缩比 (CR): {cr:.4f}x")
        print(f"总压缩耗时 (Encode): {total_encode_time:.4f} 秒")
        print(f"总解压耗时 (Decode): {total_decode_time:.4f} 秒")
        print("=========================================\n")

        lossless_check_passed = None
        if self.compressor.is_lossless and self.verify_lossless:
            lossless_check_passed = bool(is_lossless)

        return {
            "batch_size": batch_size,
            "verify_lossless": self.verify_lossless,
            "original_size_bytes": int(total_original_bytes),
            "compressed_size_bytes": int(total_compressed_bytes),
            "compression_ratio": float(cr),
            "encode_seconds": float(total_encode_time),
            "decode_seconds": float(total_decode_time),
            "is_lossless_algorithm": bool(self.compressor.is_lossless),
            "lossless_check_passed": lossless_check_passed,
        }

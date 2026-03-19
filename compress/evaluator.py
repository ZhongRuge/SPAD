import time
import numpy as np
from io_manager import SpadIOManager
from base_compressor import BaseCompressor

class CompressorEvaluator:
    """
    压缩算法评估器 (裁判员)
    负责统筹 IO 管理器和压缩算法，执行自动化测试，并输出性能报告。
    """
    def __init__(self, io_manager: SpadIOManager, compressor: BaseCompressor):
        self.io = io_manager
        self.compressor = compressor

    def run_evaluation(self, output_path: str):
        print(f"\n========== 开始评估算法: {self.compressor.algorithm_name} ==========")
        self.io.init_compressed_file(output_path)

        total_encode_time = 0.0
        total_decode_time = 0.0
        total_original_bytes = 0
        total_compressed_bytes = 0
        is_lossless = True

        # 按批次读取纯 0/1 的像素矩阵
        for batch_pixels in self.io.stream_batches(batch_size=1000):
            batch_shape = batch_pixels.shape
            
            # 记录理论上的原始体积
            # 因为 SPAD 数据本身是二值的，我们的基准线是 "1 bit 存 1 个像素"
            batch_pixels_count = batch_shape[0] * batch_shape[1] * batch_shape[2]
            original_bytes = batch_pixels_count // 8
            total_original_bytes += original_bytes

            # 测试压缩并计时
            start_time = time.time()
            compressed_bytes = self.compressor.encode(batch_pixels)
            encode_time = time.time() - start_time
            
            total_encode_time += encode_time
            total_compressed_bytes += len(compressed_bytes)
            
            # 写入磁盘
            self.io.append_compressed_bytes(output_path, compressed_bytes)

            # 测试解压并校验数据完整性
            start_time = time.time()
            decoded_pixels = self.compressor.decode(compressed_bytes, batch_shape)
            decode_time = time.time() - start_time
            total_decode_time += decode_time

            # 无损验证：对比解压后的矩阵和刚读进来的矩阵是否一模一样
            if not np.array_equal(batch_pixels, decoded_pixels):
                is_lossless = False
                print("解压后的数据与原始数据不一致！")
                break

        # 计算并打印评估报告
        cr = total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0
        
        print("\n========== 评估报告 ==========")
        print(f"算法名称: {self.compressor.algorithm_name}")
        print(f"数据无损校验: {'通过 (Lossless)' if is_lossless else '失败 (Lossy)'}")
        print(f"基准数据体积: {total_original_bytes / 1024:.2f} KB (基于 1-bit 紧凑存储)")
        print(f"压缩后的体积: {total_compressed_bytes / 1024:.2f} KB")
        print(f"整体压缩比 (CR): {cr:.4f}x")
        print(f"总压缩耗时 (Encode): {total_encode_time:.4f} 秒")
        print(f"总解压耗时 (Decode): {total_decode_time:.4f} 秒")
        print("=========================================\n")
"""压缩算法评估器（优化版）。

相比原版的关键改进：
  1. 单遍遍历：encode → write → decode → verify 在同一循环完成，不再二次读取原始数据
  2. 使用 CompressedWriter 上下文管理器，整次评估只开关一次文件
  3. 支持同时返回 mismatch_ratio（用于有损算法的精度评估）
"""

import os
import time

import numpy as np

from spad.compress.base import BaseCompressor
from spad.io import CompressedWriter, SpadReader


class CompressorEvaluator:
    """压缩算法评估器。"""

    def __init__(
        self,
        reader: SpadReader,
        compressor: BaseCompressor,
        batch_size: int = 1000,
        verify_lossless: bool = True,
    ):
        self.reader = reader
        self.compressor = compressor
        self.batch_size = int(batch_size)
        self.verify_lossless = bool(verify_lossless)

    def run_evaluation(self, output_path: str) -> dict:
        print(f"\n========== 开始评估算法: {self.compressor.algorithm_name} ==========")

        total_encode_time = 0.0
        total_decode_time = 0.0
        total_original_bytes = self.reader.get_original_size_bytes()
        is_lossless = self.compressor.is_lossless
        mismatch_pixels = 0
        total_pixels = 0

        # 单遍遍历：encode + write + decode + verify，不再二次读取原始数据
        with CompressedWriter(output_path) as writer:
            for batch_idx, batch_pixels in enumerate(self.reader.stream_batches(self.batch_size)):
                batch_shape = batch_pixels.shape

                # 编码
                t0 = time.perf_counter()
                compressed = self.compressor.encode(batch_pixels)
                total_encode_time += time.perf_counter() - t0

                # 写入压缩文件
                writer.write_chunk(compressed, batch_shape[0])

                # 解码
                t0 = time.perf_counter()
                decoded = self.compressor.decode(compressed, batch_shape)
                total_decode_time += time.perf_counter() - t0

                # 验证
                n_pixels = int(np.prod(batch_shape, dtype=np.int64))
                total_pixels += n_pixels

                if self.verify_lossless and self.compressor.is_lossless:
                    if not np.array_equal(batch_pixels, decoded):
                        is_lossless = False
                        mismatch_pixels += int(np.count_nonzero(decoded != batch_pixels))
                        print(f"  第 {batch_idx + 1} 批次解压数据与原始数据不匹配！")
                else:
                    mismatch_pixels += int(np.count_nonzero(decoded != batch_pixels))

        total_compressed_bytes = os.path.getsize(output_path)
        cr = total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0.0
        mismatch_ratio = mismatch_pixels / total_pixels if total_pixels else 0.0

        # 打印报告
        print(f"\n  算法: {self.compressor.algorithm_name}")
        if self.compressor.is_lossless and self.verify_lossless:
            print(f"  无损校验: {'通过' if is_lossless else '失败'}")
        elif self.compressor.is_lossless:
            print("  无损校验: 已跳过 (verify_lossless=false)")
        else:
            print("  有损算法，跳过无损校验")
        print(f"  原始: {total_original_bytes / 1024:.2f} KB → 压缩: {total_compressed_bytes / 1024:.2f} KB")
        print(f"  压缩比: {cr:.4f}x | 编码: {total_encode_time:.4f}s | 解码: {total_decode_time:.4f}s")
        print("=" * 50)

        lossless_check_passed = None
        if self.compressor.is_lossless and self.verify_lossless:
            lossless_check_passed = bool(is_lossless)

        return {
            "batch_size": self.batch_size,
            "verify_lossless": self.verify_lossless,
            "original_size_bytes": int(total_original_bytes),
            "compressed_size_bytes": int(total_compressed_bytes),
            "compression_ratio": float(cr),
            "encode_seconds": float(total_encode_time),
            "decode_seconds": float(total_decode_time),
            "is_lossless_algorithm": bool(self.compressor.is_lossless),
            "lossless_check_passed": lossless_check_passed,
            "mismatch_ratio": float(mismatch_ratio),
        }

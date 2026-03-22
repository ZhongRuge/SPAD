# 定义一个基础的类，规定所有具体的压缩算法必须实现哪些标准动作。比如，强制要求每个算法必须具备 encode()和 decode()这两个方法。
from abc import ABC, abstractmethod
import numpy as np

class BaseCompressor(ABC):
    """
    SPAD 压缩算法的抽象基类 (接口标准)。
    所有具体的压缩算法 (如 RLE, Huffman, 帧间差分等) 都必须继承此类，
    并实现其中的抽象方法。
    """

    def __init__(self, **kwargs):
        """
        初始化算法配置。
        子类可以重写此方法以接收特定的算法参数，
        例如块大小、量化步长或预测模式等。
        """
        self.config = kwargs

    @property
    def algorithm_name(self) -> str:
        """
        返回当前算法的名称，方便在输出日志或评估报告中展示。
        """
        return self.__class__.__name__

    @property
    def is_lossless(self) -> bool:
        """
        标记该算法是否设计为无损压缩。
        评估器据此决定是否执行逐像素一致性校验。
        """
        return True

    @abstractmethod
    def encode(self, batch_pixels: np.ndarray) -> bytes:
        """
        【核心压缩方法】
        将传入的 0/1 矩阵压缩为二进制字节流。
        
        :param batch_pixels: 形状为 (batch_size, height, width) 的 numpy 数组, dtype 为 uint8 (只有 0 和 1)
        :return: 压缩后的纯字节流 (bytes)
        """
        pass

    @abstractmethod
    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        """
        【核心解压方法】
        将压缩后的二进制字节流还原为原始的 0/1 矩阵。
        主要用于“无损验证”阶段，确保压缩没有丢失信息。
        
        :param compressed_bytes: 由 encode 方法生成的纯字节流
        :param batch_shape: 期望还原的矩阵形状，例如 (1000, 200, 200)
        :return: 还原后的 numpy 数组, dtype 必须为 uint8, 且只包含 0 和 1
        """
        pass
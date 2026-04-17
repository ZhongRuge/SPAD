from abc import ABC, abstractmethod

import numpy as np


class BaseCompressor(ABC):
    """SPAD 压缩算法抽象基类。"""

    def __init__(self, **kwargs):
        self.config = kwargs

    @property
    def algorithm_name(self) -> str:
        return self.__class__.__name__

    @property
    def is_lossless(self) -> bool:
        return True

    @abstractmethod
    def encode(self, batch_pixels: np.ndarray) -> bytes:
        """将 (batch, H, W) uint8 0/1 矩阵压缩为字节流。"""

    @abstractmethod
    def decode(self, compressed_bytes: bytes, batch_shape: tuple) -> np.ndarray:
        """将字节流还原为 (batch, H, W) uint8 0/1 矩阵。"""

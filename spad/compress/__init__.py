"""SPAD 压缩算法包。

导入本包时自动注册所有算法。新增算法只需:
  1. 在 spad/compress/ 下创建算法文件
  2. 用 @register("algorithm_id") 装饰压缩器类
  3. 在下方添加一行 import
"""

# 自动加载所有算法模块，触发 @register 装饰器注册
from spad.compress import (  # noqa: F401
    aer,
    delta_rle,
    delta_sparse,
    delta_sparse_varint_zlib,
    delta_sparse_zlib,
    global_event_stream,
    packbits_zlib,
    rle,
    temporal_binning,
)

# 公开 API
from spad.compress.base import BaseCompressor
from spad.compress.registry import build_compressor, list_algorithms

__all__ = ["BaseCompressor", "build_compressor", "list_algorithms"]

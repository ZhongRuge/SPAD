"""装饰器驱动的压缩算法注册表。

新增算法只需:
    1. 创建 spad/compress/my_algo.py
    2. 在类上标注 @register("my_algo")
    3. 在 spad/compress/__init__.py 加一行 from spad.compress import my_algo
"""

from __future__ import annotations

from typing import Any

from spad.compress.base import BaseCompressor

_REGISTRY: dict[str, type[BaseCompressor]] = {}


def register(algorithm_id: str):
    """类装饰器：将压缩算法注册到全局表中。"""
    normalized = algorithm_id.strip().lower()

    def decorator(cls: type[BaseCompressor]):
        if normalized in _REGISTRY:
            raise ValueError(
                f"Duplicate algorithm id '{normalized}': "
                f"{cls.__name__} vs {_REGISTRY[normalized].__name__}"
            )
        _REGISTRY[normalized] = cls
        return cls

    return decorator


def build_compressor(algorithm_id: str, params: dict[str, Any] | None = None) -> BaseCompressor:
    """根据算法 ID 和参数构造压缩器实例。"""
    normalized = algorithm_id.strip().lower()
    cls = _REGISTRY.get(normalized)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown algorithm '{algorithm_id}'. Available: {available}"
        )
    return cls(**(params or {}))


def list_algorithms() -> list[str]:
    """返回所有已注册的算法 ID。"""
    return sorted(_REGISTRY)

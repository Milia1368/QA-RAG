"""内存管理工具：MPS 缓存清理与分批处理辅助。"""

import gc
from typing import Generator, Iterable, List, TypeVar

T = TypeVar("T")


def clear_memory_cache(device: str = "mps") -> None:
    """释放 Python 对象并清空 PyTorch 设备缓存。"""
    gc.collect()
    if device != "mps":
        return
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def batched(items: Iterable[T], batch_size: int) -> Generator[List[T], None, None]:
    """将可迭代对象按固定大小分批 yield。"""
    batch: List[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

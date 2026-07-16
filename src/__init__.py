# -*- coding: utf-8 -*-
"""KL8 (快乐 8) 数据分析工具集包入口。"""

from importlib import import_module
from types import ModuleType

__all__ = ["analysis", "common", "config", "data_fetcher"]


def __getattr__(name: str) -> ModuleType:
    """按需加载子模块，避免轻量统计入口强制导入 PyTorch。"""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module

# -*- coding: utf-8 -*-
"""福彩3D、排列三、排列五数据分析工具集包入口。"""

from importlib import import_module
from types import ModuleType

__all__ = ["analysis", "lotteries"]


def __getattr__(name: str) -> ModuleType:
    """按需加载数字彩分析与玩法规则子模块。"""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module

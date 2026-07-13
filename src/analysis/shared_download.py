# -*- coding:utf-8 -*-
"""
下载辅助：集中管理数据下载，避免并发冲突。

提供 ensure_data_available(name: str, download_flag: int) 函数，
当 download_flag==1 时触发下载；否则跳过。
"""
from __future__ import annotations

from pathlib import Path
import sys


def ensure_data_available(name: str, download_flag: int = 1) -> None:
    """按需下载历史数据。

    - name: 彩种名称（如 "kl8"）
    - download_flag: 1 则下载，其他值跳过
    """
    if download_flag != 1:
        return
    # 兼容脚本直跑的相对/绝对导入
    try:
        from ..common import get_data_run  # type: ignore
    except Exception:
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.common import get_data_run  # type: ignore

    print("开始下载数据...")
    get_data_run(name=name, cq=0)
    print("数据下载完成")


__all__ = ["ensure_data_available"]

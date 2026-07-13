# -*- coding: utf-8 -*-
"""
快乐 8 历史数据高频号码统计示例。

步骤：
1. 读取仓库提供的 `data/kl8/data.csv`；
2. 统计号码出现频次；
3. 输出前 10 个高频号码。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import load_history  # noqa: E402


def analysis_example(top_n: int = 10) -> pd.Series:
    """
    计算历史开奖数据中出现频率最高的号码。

    参数：
        top_n: 返回的高频号码数量。
    返回：
        `pandas.Series`，索引为号码，值为出现次数。
    """

    df = load_history("kl8")
    number_columns = [col for col in df.columns if col.startswith("红球_")]
    flattened = pd.Series(df[number_columns].values.ravel()).astype(int)
    counts = flattened.value_counts().sort_values(ascending=False)
    return counts.head(top_n)


def main() -> None:
    print("=== 快乐 8 高频号码统计示例 ===")
    try:
        top_numbers = analysis_example()
    except FileNotFoundError as exc:
        data_hint = (PROJECT_ROOT / "data" / "kl8" / "data.csv").resolve()
        print(f"数据不存在：{exc}。请先执行 `make download-data` 或准备 {data_hint}")
        return

    for idx, (number, freq) in enumerate(top_numbers.items(), start=1):
        print(f"{idx:02d}. 号码 {number:02d} -> 出现 {freq} 次")
    print("=== 示例结束 ===")


if __name__ == "__main__":
    main()

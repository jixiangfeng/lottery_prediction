# -*- coding: utf-8 -*-
"""
快乐8历史数据下载脚本。

示例：
    python scripts/get_data.py --start 2024001 --end 2024350 --sequence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import get_data_run  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="选择爬取数据")
parser.add_argument('--cq', default=0, type=int, help="是否使用出球顺序，0：不使用（即按从小到大排序），1：使用")
args = parser.parse_args()

def main():
    """主函数"""
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        get_data_run(name=args.name, cq=args.cq)

if __name__ == "__main__":
    main()
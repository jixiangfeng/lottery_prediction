# -*- coding: utf-8 -*-
"""对账双色球官方 JSONL 证据并生成规范 CSV。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.analysis.ssq_history import reconcile_raw_jsonl


def main(argv: list[str] | None = None) -> int:
    """运行单官方源全量校验；数据缺失时安全不动作。"""

    parser = argparse.ArgumentParser(
        description="完整验证福彩官网双色球原始证据后生成LF规范CSV"
    )
    parser.add_argument(
        "--raw-jsonl",
        default="data/ssq/raw/history.jsonl",
        help="追加型原始证据路径",
    )
    parser.add_argument(
        "--output-csv",
        default="data/ssq/official_history.csv",
        help="规范历史CSV路径",
    )
    args = parser.parse_args(argv)
    raw_path = Path(args.raw_jsonl)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        print("双色球原始证据不存在或为空：不动作。")
        return 0
    try:
        draws = reconcile_raw_jsonl(raw_path, Path(args.output_csv))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"双色球对账失败：{error}", file=sys.stderr)
        print("建议：保留原始JSONL并修复证据来源；未生成新规范CSV。", file=sys.stderr)
        return 1
    print(f"双色球官方单源对账通过，已写入 {len(draws)} 期规范历史。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

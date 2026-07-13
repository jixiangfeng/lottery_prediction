# -*- coding: utf-8 -*-
"""生成快乐8逐期前推策略回测报告。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.daily_report import load_history_csv
from src.analysis.walk_forward_strategy_backtest import run_walk_forward_strategy_backtest, write_walk_forward_strategy_reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成快乐8逐期前推策略回测报告")
    parser.add_argument("--data", default=None, help="历史开奖 CSV 路径，默认 data/kl8/data.csv")
    parser.add_argument("--output-dir", default="reports", help="报告输出目录")
    parser.add_argument("--periods", default=300, type=int, help="回测最近多少期")
    parser.add_argument("--min-train-size", default=200, type=int, help="每个目标期至少使用多少历史期作为训练数据")
    parser.add_argument("--count", default=10, type=int, help="每期候选组数量")
    parser.add_argument("--group-size", default=10, type=int, help="每组号码数量")
    args = parser.parse_args(argv)

    history = load_history_csv(args.data)
    report = run_walk_forward_strategy_backtest(
        history,
        periods=args.periods,
        min_train_size=args.min_train_size,
        group_count=args.count,
        group_size=args.group_size,
    )
    markdown_path, json_path = write_walk_forward_strategy_reports(report, args.output_dir)
    print(markdown_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

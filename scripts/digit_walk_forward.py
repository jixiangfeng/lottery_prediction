# -*- coding: utf-8 -*-
"""数字彩严格逐期前推回测命令入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import load_digit_csv
from src.analysis.digit_walk_forward import (
    run_digit_walk_forward_backtest,
    write_digit_walk_forward_reports,
)
from src.lotteries import get_lottery_rule


def main(argv: list[str] | None = None) -> int:
    """运行数字彩严格逐期前推回测。"""

    parser = argparse.ArgumentParser(description="福彩3D/排列三/排列五严格逐期前推回测")
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3", "pl5"), help="彩票玩法")
    parser.add_argument("--csv", required=True, help="历史 CSV 路径")
    parser.add_argument("--output-dir", default="reports/evaluations", help="报告输出目录")
    parser.add_argument("--periods", type=int, default=300, help="限制最近目标期数")
    parser.add_argument("--min-train-size", type=int, default=100, help="每个目标期最少训练期数")
    parser.add_argument("--candidate-count", type=int, default=10, help="每期候选数量")
    parser.add_argument("--baseline-seed", type=int, default=20260715, help="随机基线复现种子")
    parser.add_argument("--baseline-runs", type=int, default=20, help="独立均匀随机基线运行次数")
    parser.add_argument("--nested-tuning", action="store_true", help="启用严格嵌套内层调参")
    parser.add_argument("--inner-validation-periods", type=int, default=10, help="内层尾部验证期数")
    parser.add_argument("--report-prefix", default="digit_walk_forward", help="报告文件名前缀")
    args = parser.parse_args(argv)

    rule = get_lottery_rule(args.lottery)
    history = load_digit_csv(args.csv, rule)
    report = run_digit_walk_forward_backtest(
        history,
        rule,
        periods=args.periods,
        min_train_size=args.min_train_size,
        candidate_count=args.candidate_count,
        baseline_seed=args.baseline_seed,
        baseline_runs=args.baseline_runs,
        nested_tuning=args.nested_tuning,
        inner_validation_periods=args.inner_validation_periods,
    )
    markdown_path, json_path = write_digit_walk_forward_reports(
        report,
        args.output_dir,
        stem_prefix=args.report_prefix,
    )
    print(f"Markdown 报告：{markdown_path}")
    print(f"JSON 报告：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

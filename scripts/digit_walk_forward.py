# -*- coding: utf-8 -*-
"""数字彩严格逐期前推回测命令入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_candidates import DigitCandidateConfig  # noqa: E402
from src.analysis.digit_data import load_digit_csv  # noqa: E402
from src.analysis.digit_walk_forward import (  # noqa: E402
    run_digit_walk_forward_backtest,
    write_digit_walk_forward_reports,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """运行数字彩严格逐期前推回测。"""

    parser = argparse.ArgumentParser(description="福彩3D/排列三/排列五严格逐期前推回测")
    parser.add_argument(
        "--lottery", required=True, choices=("fc3d", "pl3", "pl5"), help="彩票玩法"
    )
    parser.add_argument("--csv", required=True, help="历史 CSV 路径")
    parser.add_argument(
        "--output-dir", default="reports/evaluations", help="报告输出目录"
    )
    parser.add_argument("--periods", type=int, default=300, help="限制最近目标期数")
    parser.add_argument(
        "--min-train-size", type=int, default=100, help="每个目标期最少训练期数"
    )
    parser.add_argument("--candidate-count", type=int, default=10, help="每期候选数量")
    parser.add_argument(
        "--baseline-seed", type=int, default=20260715, help="随机基线复现种子"
    )
    parser.add_argument(
        "--baseline-runs", type=int, default=20, help="独立均匀随机基线运行次数"
    )
    parser.add_argument(
        "--nested-tuning", action="store_true", help="启用严格嵌套内层调参"
    )
    parser.add_argument(
        "--inner-validation-periods", type=int, default=10, help="内层尾部验证期数"
    )
    parser.add_argument(
        "--report-prefix", default="digit_walk_forward", help="报告文件名前缀"
    )
    parser.add_argument(
        "--advanced-models",
        action="store_true",
        help="启用蒙特卡洛与 sklearn 排序投票器",
    )
    parser.add_argument(
        "--monte-carlo-simulations",
        type=int,
        default=5000,
        help="每个外层目标期的模拟次数",
    )
    parser.add_argument(
        "--ml-training-periods",
        type=int,
        default=30,
        help="每个外层目标期的机器学习训练目标数",
    )
    parser.add_argument(
        "--ml-negative-samples",
        type=int,
        default=5,
        help="每个机器学习正样本的负样本数",
    )
    parser.add_argument(
        "--compare-windows",
        action="store_true",
        help="比较 30/50/100/300/全历史独立窗口",
    )
    parser.add_argument(
        "--constraint-mode",
        default="soft",
        choices=("off", "soft", "hard"),
        help="结构罕见度约束模式",
    )
    parser.add_argument(
        "--constraint-probability-floor",
        type=float,
        default=0.02,
        help="奇偶/大小/质合结构多窗口概率下限",
    )
    parser.add_argument(
        "--constraint-penalty-weight",
        type=float,
        default=0.05,
        help="soft 模式罕见结构惩罚权重",
    )
    args = parser.parse_args(argv)

    rule = get_lottery_rule(args.lottery)
    history = load_digit_csv(args.csv, rule)
    report = run_digit_walk_forward_backtest(
        history,
        rule,
        periods=args.periods,
        min_train_size=args.min_train_size,
        candidate_count=args.candidate_count,
        config=DigitCandidateConfig(
            constraint_mode=args.constraint_mode,
            constraint_probability_floor=args.constraint_probability_floor,
            constraint_penalty_weight=args.constraint_penalty_weight,
        ),
        baseline_seed=args.baseline_seed,
        baseline_runs=args.baseline_runs,
        nested_tuning=args.nested_tuning,
        inner_validation_periods=args.inner_validation_periods,
        advanced_models=args.advanced_models,
        monte_carlo_simulations=args.monte_carlo_simulations,
        ml_training_periods=args.ml_training_periods,
        ml_negative_samples=args.ml_negative_samples,
        compare_windows=args.compare_windows,
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

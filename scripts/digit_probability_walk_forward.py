# -*- coding: utf-8 -*-
"""三位彩概率 v2 严格前推开发评估入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_candidates import DigitCandidateConfig  # noqa: E402
from src.analysis.digit_data import load_digit_csv  # noqa: E402
from src.analysis.digit_probability import DigitProbabilityConfig  # noqa: E402
from src.analysis.digit_probability_walk_forward import (  # noqa: E402
    run_digit_probability_walk_forward,
    write_digit_probability_walk_forward_reports,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """运行概率 v2 固定校准后的严格前推开发评估。"""

    parser = argparse.ArgumentParser(
        description="福彩3D/排列三概率 v2 严格前推开发评估"
    )
    parser.add_argument(
        "--lottery", required=True, choices=("fc3d", "pl3"), help="彩票玩法"
    )
    parser.add_argument("--csv", required=True, help="历史 CSV 路径")
    parser.add_argument(
        "--output-dir", default="reports/evaluations", help="报告输出目录"
    )
    parser.add_argument("--periods", type=int, default=500, help="最近目标期数")
    parser.add_argument(
        "--min-train-size", type=int, default=100, help="最早目标期最少训练期数"
    )
    parser.add_argument("--candidate-count", type=int, default=10, help="每期候选数量")
    parser.add_argument(
        "--validation-periods", type=int, default=180, help="冻结前概率校准期数"
    )
    parser.add_argument(
        "--calibration-min-train-size",
        type=int,
        default=100,
        help="每个校准目标的最少历史期数",
    )
    parser.add_argument(
        "--minimum-validation-periods",
        type=int,
        default=90,
        help="允许启用学习概率的最少校准期数",
    )
    parser.add_argument(
        "--report-prefix", default="digit_probability_v2", help="报告文件名前缀"
    )
    args = parser.parse_args(argv)

    rule = get_lottery_rule(args.lottery)
    history = load_digit_csv(args.csv, rule)
    report = run_digit_probability_walk_forward(
        history,
        rule,
        periods=args.periods,
        min_train_size=args.min_train_size,
        candidate_count=args.candidate_count,
        candidate_config=DigitCandidateConfig(
            count=args.candidate_count,
            ranking_mode="ensemble",
            exclude_latest=False,
            constraint_mode="off",
        ),
        probability_config=DigitProbabilityConfig(
            validation_periods=args.validation_periods,
            min_train_size=args.calibration_min_train_size,
            minimum_validation_periods=args.minimum_validation_periods,
        ),
    )
    markdown_path, json_path = write_digit_probability_walk_forward_reports(
        report,
        args.output_dir,
        prefix=args.report_prefix,
    )
    print(f"Markdown 报告：{markdown_path}")
    print(f"JSON 报告：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""三位彩在线概率 v3 严格前推开发评估入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import load_digit_csv  # noqa: E402
from src.analysis.digit_probability_online import (  # noqa: E402
    DigitOnlineProbabilityConfig,
    run_digit_online_probability_walk_forward,
    write_digit_online_probability_reports,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """执行预测后反馈的在线概率500期开发评估。"""

    parser = argparse.ArgumentParser(
        description="福彩3D/排列三在线概率 v3 严格前推开发评估"
    )
    parser.add_argument(
        "--lottery", required=True, choices=("fc3d", "pl3"), help="彩票玩法"
    )
    parser.add_argument("--csv", required=True, help="历史 CSV 路径")
    parser.add_argument(
        "--output-dir", default="reports/evaluations", help="报告输出目录"
    )
    parser.add_argument("--periods", type=int, default=500, help="在线评估期数")
    parser.add_argument(
        "--min-train-size", type=int, default=100, help="开始在线预训练前的历史期数"
    )
    parser.add_argument("--candidate-count", type=int, default=10, help="每期候选数")
    parser.add_argument("--temperature", type=float, default=0.2, help="固定概率温度")
    parser.add_argument(
        "--uniform-prior-weight",
        type=float,
        default=0.5,
        help="均匀基线初始权重",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0, help="在线学习率")
    parser.add_argument(
        "--fixed-share", type=float, default=0.01, help="每期开奖后向初始权重收缩比例"
    )
    parser.add_argument(
        "--report-prefix",
        default="official_500_probability_v3_online",
        help="报告文件名前缀",
    )
    args = parser.parse_args(argv)

    rule = get_lottery_rule(args.lottery)
    history = load_digit_csv(args.csv, rule)
    config = DigitOnlineProbabilityConfig(
        min_train_size=args.min_train_size,
        temperature=args.temperature,
        uniform_prior_weight=args.uniform_prior_weight,
        learning_rate=args.learning_rate,
        fixed_share=args.fixed_share,
    )
    report = run_digit_online_probability_walk_forward(
        history,
        rule,
        periods=args.periods,
        candidate_count=args.candidate_count,
        online_config=config,
        progress_callback=lambda message: print(message, flush=True),
    )
    markdown_path, json_path = write_digit_online_probability_reports(
        report,
        args.output_dir,
        prefix=args.report_prefix,
    )
    print(f"Markdown 报告：{markdown_path}")
    print(f"JSON 报告：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""learned_ranker_v4逐期在线梯度开发入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import load_digit_development_csv  # noqa: E402
from src.analysis.digit_online_gradient import (  # noqa: E402
    OnlineGradientConfig,
    run_online_gradient_research,
    write_online_gradient_report,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="learned_ranker_v4在线梯度研究")
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-test-periods", type=int, default=500)
    parser.add_argument("--outer-periods", type=int, default=500)
    parser.add_argument("--calibration-interval", type=int, default=10)
    parser.add_argument("--search-lookback", type=int, default=300)
    parser.add_argument("--validation-lookback", type=int, default=100)
    parser.add_argument("--l2-penalty", type=float, default=0.001)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--weight-limit", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--direct-top-k", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rule = get_lottery_rule(args.lottery)
    development, total_periods = load_digit_development_csv(
        args.csv,
        rule,
        frozen_test_periods=args.frozen_test_periods,
    )
    report = run_online_gradient_research(
        development,
        rule,
        OnlineGradientConfig(
            development_end=total_periods - args.frozen_test_periods,
            outer_periods=args.outer_periods,
            calibration_interval=args.calibration_interval,
            search_lookback=args.search_lookback,
            validation_lookback=args.validation_lookback,
            l2_penalty=args.l2_penalty,
            gradient_clip=args.gradient_clip,
            weight_limit=args.weight_limit,
            temperature=args.temperature,
            direct_top_k=args.direct_top_k,
        ),
    )
    path = write_online_gradient_report(report, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

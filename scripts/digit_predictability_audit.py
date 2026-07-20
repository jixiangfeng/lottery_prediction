#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读开发区的三位彩可预测性审计入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import load_digit_development_csv  # noqa: E402
from src.analysis.digit_predictability_audit import (  # noqa: E402
    PredictabilityAuditConfig,
    run_predictability_audit,
    write_predictability_audit,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="learned_ranker_v4开发区可预测性审计")
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-test-periods", type=int, default=500)
    parser.add_argument("--min-train-size", type=int, default=150)
    parser.add_argument("--permutation-trials", type=int, default=499)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rule = get_lottery_rule(args.lottery)
    development, _ = load_digit_development_csv(
        args.csv,
        rule,
        frozen_test_periods=args.frozen_test_periods,
    )
    report = run_predictability_audit(
        development,
        rule,
        PredictabilityAuditConfig(
            min_train_size=args.min_train_size,
            permutation_trials=args.permutation_trials,
            block_size=args.block_size,
            seed=args.seed,
            fdr_alpha=args.fdr_alpha,
        ),
    )
    path = write_predictability_audit(report, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

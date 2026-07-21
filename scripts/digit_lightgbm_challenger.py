#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行LightGBM三位置多分类全部历史区块挑战评估。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import load_digit_csv  # noqa: E402
from src.analysis.digit_lightgbm_challenger import (  # noqa: E402
    LightGBMChallengeConfig,
    run_lightgbm_block_backtest,
    write_lightgbm_report,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LightGBM三位置多分类挑战模型")
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-periods", type=int)
    args = parser.parse_args(argv)
    rule = get_lottery_rule(args.lottery)
    history = load_digit_csv(args.csv, rule)
    if args.limit_periods:
        history = history.tail(args.limit_periods).reset_index(drop=True)
    config = LightGBMChallengeConfig()
    print(
        f"effective: topK={config.top_k} block={config.block_size} "
        f"innerValidation={config.inner_validation_periods} "
        f"models={len(config.parameter_grid)} shrinkages={config.shrinkages}",
        flush=True,
    )
    result = run_lightgbm_block_backtest(history, rule, config)
    destination = write_lightgbm_report(result, args.output)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

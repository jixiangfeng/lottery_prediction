#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行固定协议下全部连续历史500期区块回测。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import load_digit_csv  # noqa: E402
from src.analysis.digit_full_history_shadow import FullHistoryShadowConfig  # noqa: E402
from src.analysis.digit_historical_blocks import (  # noqa: E402
    run_historical_block_backtest,
    write_historical_block_report,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="全部历史连续500期区块回测")
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-size", type=int, default=500)
    args = parser.parse_args(argv)
    rule = get_lottery_rule(args.lottery)
    history = load_digit_csv(args.csv, rule)
    result = run_historical_block_backtest(
        history, rule, FullHistoryShadowConfig(), block_size=args.block_size
    )
    destination = write_historical_block_report(result, args.output)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

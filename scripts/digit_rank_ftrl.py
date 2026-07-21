#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行rank-aware FTRL v4.1全部历史区块评估。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import load_digit_csv  # noqa: E402
from src.analysis.digit_learned_features import FEATURE_NAMES  # noqa: E402
from src.analysis.digit_rank_ftrl import (  # noqa: E402
    FTRLConfig,
    run_rank_ftrl_blocks,
    write_rank_ftrl_report,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank-aware FTRL v4.1")
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-periods", type=int)
    parser.add_argument("--exclude-feature", choices=FEATURE_NAMES)
    args = parser.parse_args(argv)
    rule = get_lottery_rule(args.lottery)
    history = load_digit_csv(args.csv, rule)
    if args.limit_periods:
        history = history.tail(args.limit_periods).reset_index(drop=True)
    config = FTRLConfig()
    if args.exclude_feature:
        config = replace(config, excluded_features=(args.exclude_feature,))
    print(
        f"effective: topK={config.top_k} block={config.block_size} "
        f"loss={config.logloss_weight}/{config.rank_weight} "
        f"experts={config.expert_alphas} maxExpert={config.maximum_expert_weight}",
        flush=True,
    )
    result = run_rank_ftrl_blocks(history, rule, config)
    destination = write_rank_ftrl_report(result, args.output)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

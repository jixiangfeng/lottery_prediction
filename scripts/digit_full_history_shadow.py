#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用全部已开奖历史锁定稀疏v4影子状态。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import load_digit_csv  # noqa: E402
from src.analysis.digit_full_history_shadow import (  # noqa: E402
    train_full_history_shadow,
    write_locked_shadow_state,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="全历史连续预训练稀疏v4影子模型")
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rule = get_lottery_rule(args.lottery)
    history = load_digit_csv(args.csv, rule)
    result = train_full_history_shadow(history, rule)
    destination = write_locked_shadow_state(result, args.output)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

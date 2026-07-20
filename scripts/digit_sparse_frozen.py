#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稀疏v4一次性Frozen锁定和评估入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_sparse_frozen import (  # noqa: E402
    create_sparse_v4_lock,
    run_locked_sparse_v4_frozen,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="稀疏learned_ranker_v4一次性Frozen")
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock", help="锁定协议，不读取号码CSV")
    lock.add_argument("--lock", required=True)

    evaluate = subparsers.add_parser("evaluate", help="一次性读取并评估Frozen")
    evaluate.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    evaluate.add_argument("--csv", required=True)
    evaluate.add_argument("--lock", required=True)
    evaluate.add_argument("--marker", required=True)
    evaluate.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "lock":
        path, fingerprint = create_sparse_v4_lock(args.lock)
        print(f"{path}\n{fingerprint}")
        return 0
    rule = get_lottery_rule(args.lottery)
    run_locked_sparse_v4_frozen(
        args.csv,
        rule,
        lock_path=args.lock,
        marker_path=args.marker,
        output_path=args.output,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

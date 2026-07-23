#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快乐8选4安全今日入口。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.kl8_pick4_prediction import (  # noqa: E402
    build_pick4_prediction_boundary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="快乐8选4安全预测边界；默认空正式候选，--test输出等概率测试组合"
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--frozen-periods", type=int, default=500)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--date")
    parser.add_argument("--ticket-count", type=int, default=5)
    args = parser.parse_args(argv)
    if args.date and not args.test:
        parser.error("--date只能与--test同时使用")
    target_date = None
    test_ticket_count = 0
    if args.test:
        target_date = (
            args.date or datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        )
        test_ticket_count = args.ticket_count
    payload = build_pick4_prediction_boundary(
        args.csv,
        frozen_periods=args.frozen_periods,
        target_date=target_date,
        test_ticket_count=test_ticket_count,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

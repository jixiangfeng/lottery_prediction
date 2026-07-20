# -*- coding: utf-8 -*-
"""显式抓取福彩3D或排列三官方历史开奖。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_history_fetcher import fetch_digit_history  # noqa: E402
from src.analysis.digit_raw_evidence import (  # noqa: E402
    RawDigitRecord,
    append_raw_digit_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    """执行官方历史抓取并只追加原始JSONL证据。"""

    parser = argparse.ArgumentParser(
        description="从福彩官网或中彩网公开接口抓取三位彩历史"
    )
    parser.add_argument(
        "--lottery", required=True, choices=("fc3d", "pl3"), help="彩票玩法"
    )
    parser.add_argument(
        "--periods", type=int, default=0, help="抓取期数；0表示接口全部历史"
    )
    parser.add_argument(
        "--output-jsonl", help="原始JSONL，默认写入data/<玩法>/raw/history.jsonl"
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="单页失败重试次数")
    args = parser.parse_args(argv)

    output = Path(args.output_jsonl or f"data/{args.lottery}/raw/history.jsonl")
    resolved_output = (
        (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    )
    if not resolved_output.is_relative_to(ROOT.resolve()):
        parser.error("输出路径必须位于项目目录内")
    try:
        draws = fetch_digit_history(
            args.lottery,
            periods=args.periods,
            timeout=args.timeout,
            retries=args.retries,
        )
        fetched_at = datetime.now(timezone.utc).isoformat()
        records = [
            RawDigitRecord(
                lottery_code=args.lottery,
                issue=draw.issue,
                draw_date=draw.draw_date,
                digits=(draw.numbers[0], draw.numbers[1], draw.numbers[2]),
                source_name=urlparse(draw.source).hostname or draw.source,
                source_url=draw.source,
                fetched_at=fetched_at,
                raw={
                    "issue": draw.issue,
                    "digits": list(draw.numbers),
                    "drawDate": draw.draw_date,
                },
            )
            for draw in draws
        ]
        jsonl_path = append_raw_digit_jsonl(resolved_output, records)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"抓取失败：{error}", file=sys.stderr)
        print("建议：检查网络后重试，或继续使用人工提供的本地 CSV。", file=sys.stderr)
        return 1
    print(f"已追加 {len(records)} 条原始证据：{jsonl_path}")
    print("标准CSV未改动；请在多源对账后运行 reconcile_digit_jsonl.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

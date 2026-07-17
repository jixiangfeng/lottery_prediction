# -*- coding: utf-8 -*-
"""显式抓取福彩3D或排列三官方历史开奖。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import load_digit_csv  # noqa: E402
from src.analysis.digit_history_fetcher import (  # noqa: E402
    fetch_digit_history,
    write_digit_history_csv,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """执行官方历史抓取并在写入后复核 CSV。"""

    parser = argparse.ArgumentParser(
        description="从福彩官网或中彩网公开接口抓取三位彩历史"
    )
    parser.add_argument(
        "--lottery", required=True, choices=("fc3d", "pl3"), help="彩票玩法"
    )
    parser.add_argument("--periods", type=int, default=1000, help="抓取最近期数")
    parser.add_argument(
        "--output", help="输出 CSV，默认写入 data/<玩法>/official_history.csv"
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="单页失败重试次数")
    args = parser.parse_args(argv)

    output = Path(args.output or f"data/{args.lottery}/official_history.csv")
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
        csv_path = write_digit_history_csv(draws, resolved_output)
        normalized = load_digit_csv(csv_path, get_lottery_rule(args.lottery))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"抓取失败：{error}", file=sys.stderr)
        print("建议：检查网络后重试，或继续使用人工提供的本地 CSV。", file=sys.stderr)
        return 1
    print(f"已写入 {len(normalized)} 期：{csv_path}")
    print(f"期号范围：{normalized.iloc[-1]['期数']} - {normalized.iloc[0]['期数']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

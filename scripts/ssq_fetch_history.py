# -*- coding: utf-8 -*-
"""抓取福彩官网双色球历史并追加原始 JSONL 证据。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.analysis.ssq_history import append_raw_evidence, fetch_ssq_history


def main(argv: list[str] | None = None) -> int:
    """运行官方抓取入口；不会生成或覆盖规范 CSV。"""

    parser = argparse.ArgumentParser(
        description="仅从福彩官网固定双色球接口抓取历史并追加原始JSONL"
    )
    parser.add_argument(
        "--periods", type=int, default=0, help="抓取期数；0表示官网全部历史"
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/ssq/raw/history.jsonl",
        help="追加型原始证据路径",
    )
    args = parser.parse_args(argv)
    try:
        draws = fetch_ssq_history(args.periods)
        appended = append_raw_evidence(Path(args.output_jsonl), draws)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"双色球抓取失败：{error}", file=sys.stderr)
        print("建议：检查官方接口后重试；规范CSV不会被改写。", file=sys.stderr)
        return 1
    print(f"已校验并追加 {appended} 条双色球官方原始证据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

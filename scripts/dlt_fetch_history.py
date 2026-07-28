# -*- coding: utf-8 -*-
"""抓取体彩官网大乐透全部历史并原子保存原始 JSONL 证据。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.analysis.dlt_history import atomic_write_raw_evidence, fetch_dlt_history


def main(argv: list[str] | None = None) -> int:
    """抓取全部官方分页；任一页失败时不改写现有证据。"""

    parser = argparse.ArgumentParser(
        description="仅从体彩官网固定 gameNo=85 接口抓取大乐透全部历史"
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/dlt/raw/history.jsonl",
        help="原子写入的官方原始证据 JSONL 路径",
    )
    args = parser.parse_args(argv)
    try:
        draws = fetch_dlt_history()
        atomic_write_raw_evidence(Path(args.output_jsonl), draws)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"大乐透抓取失败：{error}", file=sys.stderr)
        print("任一分页异常均会失败关闭，现有证据文件不会被替换。", file=sys.stderr)
        return 1
    pages = len({draw.source_url for draw in draws})
    print(f"已校验 {pages} 页并原子写入 {len(draws)} 条大乐透官方原始证据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

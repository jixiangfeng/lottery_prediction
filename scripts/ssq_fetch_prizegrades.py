# -*- coding: utf-8 -*-
"""抓取并追加缓存福彩官网双色球逐期奖级奖金。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.analysis.ssq_prizegrades import (
    append_prizegrade_evidence,
    fetch_ssq_prizegrades,
    read_prizegrade_evidence,
    reconcile_prizegrade_evidence,
    write_prizegrades_json,
)


def main(argv: list[str] | None = None) -> int:
    """运行有限页官方采集；中断后以 ``--start-page`` 指定下一页续跑。"""

    parser = argparse.ArgumentParser(
        description="仅抓取福彩官网双色球 prizegrades，并追加带哈希的原始 JSONL 缓存"
    )
    parser.add_argument("--start-page", type=int, default=1, help="起始页（从 1 开始）")
    parser.add_argument("--pages", type=int, default=1, help="本次抓取页数（默认 1）")
    parser.add_argument(
        "--cache-jsonl",
        default="data/ssq/raw/prizegrades.jsonl",
        help="追加型原始奖级证据缓存",
    )
    parser.add_argument(
        "--snapshot-json",
        default=None,
        help="可选：对账全部缓存后原子写入的结构化快照路径",
    )
    args = parser.parse_args(argv)
    try:
        records = fetch_ssq_prizegrades(start_page=args.start_page, pages=args.pages)
        appended = append_prizegrade_evidence(Path(args.cache_jsonl), records)
        if args.snapshot_json:
            reconciled = reconcile_prizegrade_evidence(
                read_prizegrade_evidence(args.cache_jsonl)
            )
            write_prizegrades_json(Path(args.snapshot_json), reconciled)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"双色球奖级采集失败：{error}", file=sys.stderr)
        print("建议：检查福彩接口或缓存完整性；不会伪造缺失期数据。", file=sys.stderr)
        return 1
    print(
        f"已校验并追加 {appended} 条双色球官方奖级原始证据"
        f"（页 {args.start_page} 至 {args.start_page + args.pages - 1}）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

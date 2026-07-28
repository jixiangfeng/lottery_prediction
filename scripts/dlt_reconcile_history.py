# -*- coding: utf-8 -*-
"""对账大乐透官方 JSONL 并生成规范 CSV 与对账 JSON。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.analysis.dlt_history import reconcile_raw_jsonl


def main(argv: list[str] | None = None) -> int:
    """完整验证证据后才原子替换派生文件。"""

    parser = argparse.ArgumentParser(
        description="验证大乐透官方原始证据并生成 LF 规范 CSV 和对账 JSON"
    )
    parser.add_argument(
        "--raw-jsonl",
        default="data/dlt/raw/history.jsonl",
        help="官方原始证据 JSONL 路径",
    )
    parser.add_argument(
        "--output-csv",
        default="data/dlt/official_history.csv",
        help="LF 规范历史 CSV 路径",
    )
    parser.add_argument(
        "--output-report",
        default="data/dlt/reconciliation.json",
        help="稳定对账 JSON 路径",
    )
    args = parser.parse_args(argv)
    raw_path = Path(args.raw_jsonl)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        print("大乐透原始证据不存在或为空：不动作。")
        return 0
    try:
        draws, report = reconcile_raw_jsonl(
            raw_path, Path(args.output_csv), Path(args.output_report)
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"大乐透对账失败：{error}", file=sys.stderr)
        print("请保留原始证据并排查；不会生成伪造补行。", file=sys.stderr)
        return 1
    print(
        "大乐透官方单源对账通过："
        f"{len(draws)} 期，{report['first']} 至 {report['latest']}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

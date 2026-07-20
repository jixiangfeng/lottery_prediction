#!/usr/bin/env python3
"""将多源数字彩原始JSONL对账后生成标准化CSV。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_history_fetcher import write_digit_history_csv  # noqa: E402
from src.analysis.digit_raw_evidence import (  # noqa: E402
    build_reconciliation_report,
    read_raw_digit_jsonl,
    reconcile_raw_digit_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-jsonl", required=True, type=Path, nargs="+")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--conflict-report", type=Path)
    parser.add_argument("--minimum-sources", type=int, default=2)
    args = parser.parse_args()

    records = tuple(
        record for path in args.raw_jsonl for record in read_raw_digit_jsonl(path)
    )
    report = build_reconciliation_report(records, minimum_sources=args.minimum_sources)
    report_path = args.conflict_report or args.output_csv.with_suffix(
        ".reconciliation.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if report["conflicts"]:
        raise SystemExit(f"存在多来源冲突，CSV未修改：{report_path}")
    draws = reconcile_raw_digit_records(records, minimum_sources=args.minimum_sources)
    if not draws:
        raise SystemExit("对账后没有可写入的记录；检查来源数量和冲突清单")
    write_digit_history_csv(list(draws), args.output_csv)
    print(f"已写入 {len(draws)} 期：{args.output_csv}")
    print(f"对账报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

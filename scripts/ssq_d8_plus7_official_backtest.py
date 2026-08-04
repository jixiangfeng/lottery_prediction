# -*- coding: utf-8 -*-
"""运行 D8+7 严格前序官方逐票奖级结算回测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.ssq_d8_plus7_official_backtest import (
    evaluate_d8_plus7_official_backtest,
)
from src.analysis.ssq_history import load_official_history_csv
from src.analysis.ssq_prizegrades import SSQPrizeGrade, SSQPrizeGradeRecord


def _records(snapshot: Path) -> list[SSQPrizeGradeRecord]:
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError("官方奖级快照 records 非法")
    return [
        SSQPrizeGradeRecord(
            issue=str(row["issue"]),
            prizegrades=tuple(SSQPrizeGrade(**grade) for grade in row["prizegrades"]),
            source_url=str(row["sourceUrl"]),
            raw_hash=str(row["rawHash"]),
            raw={},
            fetched_at=str(row["fetchedAt"]),
        )
        for row in rows
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/ssq/official_history.csv")
    parser.add_argument("--prizegrades", default="data/ssq/prizegrades_snapshot.json")
    parser.add_argument(
        "--output",
        default="reports/retrospective/ssq_d8_plus7_official_backtest_v1.json",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate_d8_plus7_official_backtest(
            load_official_history_csv(args.csv), _records(Path(args.prizegrades))
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        print(f"D8+7官方逐票结算失败：{error}")
        return 1
    print("D8+7官方逐票结算完成，状态=uniform_abstain。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

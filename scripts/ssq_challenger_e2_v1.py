# -*- coding: utf-8 -*-
"""生成 E2 独立当前研究报告；不修改 incumbent ensemble。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from src.analysis.ssq_challenger_e2_v1 import build_current_report, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成SSQ Challenger E2独立当前研究报告"
    )
    parser.add_argument("--csv", default="data/ssq/official_history.csv")
    parser.add_argument(
        "--ensemble-report", default="reports/research/ssq_ensemble_v1.json"
    )
    parser.add_argument(
        "--selection-report",
        default="reports/retrospective/ssq_challenger_e2_selection_v1.json",
    )
    parser.add_argument(
        "--output", default="reports/research/ssq_challenger_e2_v1.json"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selection = cast(
        dict[str, object],
        json.loads(Path(args.selection_report).read_text(encoding="utf-8")),
    )
    report = build_current_report(args.csv, args.ensemble_report, selection)
    output = write_report(report, args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                "selectionStatus": report["selectionStatus"],
                "selectedCandidateId": report["selectedCandidateId"],
                "reportSha256": report["reportSha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

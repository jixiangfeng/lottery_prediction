# -*- coding: utf-8 -*-
"""运行冻结 E2 八候选选择，并同步生成安全当前报告。"""

from __future__ import annotations

import argparse
import json

from src.analysis.ssq_challenger_e2_selection_v1 import build_selection_report
from src.analysis.ssq_challenger_e2_v1 import build_current_report, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行SSQ Challenger E2固定有限选择")
    parser.add_argument("--csv", default="data/ssq/official_history.csv")
    parser.add_argument(
        "--ensemble-report", default="reports/research/ssq_ensemble_v1.json"
    )
    parser.add_argument(
        "--output",
        default="reports/retrospective/ssq_challenger_e2_selection_v1.json",
    )
    parser.add_argument(
        "--current-output", default="reports/research/ssq_challenger_e2_v1.json"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selection = build_selection_report(args.csv, args.ensemble_report)
    selection_path = write_report(selection, args.output)
    current = build_current_report(args.csv, args.ensemble_report, selection)
    current_path = write_report(current, args.current_output)
    print(
        json.dumps(
            {
                "selectionOutput": str(selection_path),
                "currentOutput": str(current_path),
                "selectionStatus": selection["selectionStatus"],
                "selectedCandidateId": selection["selectedCandidateId"],
                "selectionReportSha256": selection["reportSha256"],
                "currentReportSha256": current["reportSha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

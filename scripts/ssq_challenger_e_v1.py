# -*- coding: utf-8 -*-
"""生成双色球 Challenger E1 独立当前报告。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.analysis.ssq_calibrated_red_challenger_e_v1 import (
    build_current_report,
    write_report,
)

DEFAULT_CSV = Path("data/ssq/official_history.csv")
DEFAULT_ENSEMBLE_REPORT = Path("reports/research/ssq_ensemble_v1.json")
DEFAULT_OUTPUT = Path("reports/research/ssq_challenger_e_v1.json")


def build_parser() -> argparse.ArgumentParser:
    """构造仅含输入输出路径、无模型调参项的 CLI。"""

    parser = argparse.ArgumentParser(
        description="生成固定双色球 Challenger E1 当前研究报告"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--ensemble-report", type=Path, default=DEFAULT_ENSEMBLE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行独立 E1 当前报告生成。"""

    args = build_parser().parse_args(argv)
    report = build_current_report(args.csv, args.ensemble_report)
    path = write_report(report, args.output)
    target = report["currentTargetGroup"]
    print(f"已生成E1当前报告: {path}")
    print(f"当前目标组: 红球={target['red']} 蓝球={target['blue']}")
    print(f"reportSha256={report['reportSha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""运行双色球分散组合 v2 全历史严格前序 A/B/C 回溯。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.analysis.ssq_diversified_portfolio_v2_history import (
    evaluate_full_history,
    write_report,
)
from src.analysis.ssq_history import load_official_history_csv


def main(argv: list[str] | None = None) -> int:
    """仅允许指定输入输出路径，不开放算法、对照或统计参数。"""

    parser = argparse.ArgumentParser(
        description="双色球覆盖优先分散组合v2全历史严格前序A/B/C研究"
    )
    parser.add_argument(
        "--csv",
        default="data/ssq/official_history.csv",
        help="经官方证据对账的规范CSV",
    )
    parser.add_argument(
        "--output",
        default="reports/retrospective/ssq_diversified_portfolio_v2_full_history.json",
        help="回溯研究JSON路径",
    )
    args = parser.parse_args(argv)
    try:
        draws = load_official_history_csv(Path(args.csv))
        report = evaluate_full_history(draws)
        output = write_report(report, Path(args.output))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"双色球分散组合v2回溯失败：{error}", file=sys.stderr)
        print("建议：检查规范历史完整性；固定算法不接受CLI调参。", file=sys.stderr)
        return 1
    history = report["history"]
    print(f"双色球分散组合v2回溯完成：{output}；history={history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

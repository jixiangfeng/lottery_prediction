#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成双色球8红+1蓝研究影子v1全历史严格前序诊断。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.ssq_8red1blue_v1_history import (  # noqa: E402
    evaluate_full_history,
    write_report,
)
from src.analysis.ssq_history import load_official_history_csv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """运行固定协议；CLI只允许覆盖输入输出路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/ssq/official_history.csv")
    parser.add_argument(
        "--output",
        default="reports/retrospective/ssq_8red1blue_v1_full_history.json",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate_full_history(load_official_history_csv(args.csv))
        write_report(report, args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"D8全历史诊断失败：{error}", file=sys.stderr)
        print("建议：检查canonical历史；固定协议不接受调参。", file=sys.stderr)
        return 1
    history = report["history"]
    print(
        "D8全历史诊断完成："
        f"warmup={history['warmupPeriods']}，evaluated={history['evaluatedPeriods']}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

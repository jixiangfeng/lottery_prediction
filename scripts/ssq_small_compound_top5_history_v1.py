# -*- coding: utf-8 -*-
"""运行双色球 Top5 小复式全历史严格前序回溯。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.analysis.ssq_history import load_official_history_csv
from src.analysis.ssq_small_compound_top5_history_v1 import (
    evaluate_full_history,
    write_report,
)


def main(argv: list[str] | None = None) -> int:
    """只允许覆盖输入输出路径，不开放模型、对照或统计参数。"""

    parser = argparse.ArgumentParser(
        description="双色球5组7红1蓝、35注唯一票全历史严格前序回溯研究"
    )
    parser.add_argument(
        "--csv",
        default="data/ssq/official_history.csv",
        help="经官方证据对账的规范CSV",
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/retrospective/" "ssq_small_compound_top5_full_history_v1.json"
        ),
        help="回溯研究JSON路径",
    )
    args = parser.parse_args(argv)
    try:
        draws = load_official_history_csv(Path(args.csv))
        report = evaluate_full_history(draws)
        output = write_report(report, Path(args.output))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"双色球Top5小复式全历史回溯失败：{error}", file=sys.stderr)
        print(
            "建议：检查规范CSV完整性；本入口不允许调整模型或协议参数。", file=sys.stderr
        )
        return 1
    history = report["history"]
    model = report["model"]
    print(
        "双色球Top5小复式全历史回溯完成："
        f"periods={history['evaluatedPeriods']}，"
        f"blueHitRate={model['metrics']['sharedBlue']['hitRate']:.6f}，"
        f"report={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成双色球 Challenger E1 全历史严格前序独立诊断。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.ssq_challenger_e_v1_history import (  # noqa: E402
    evaluate_full_history,
    write_report,
)
from src.analysis.ssq_history import load_official_history_csv  # noqa: E402

DEFAULT_CSV = Path("data/ssq/official_history.csv")
DEFAULT_OUTPUT = Path("reports/retrospective/ssq_challenger_e_v1_full_history.json")


def build_parser() -> argparse.ArgumentParser:
    """构造只允许覆盖输入输出路径的固定协议 CLI。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行独立历史诊断；不接受任何模型或评估调参。"""

    args = build_parser().parse_args(argv)
    try:
        report = evaluate_full_history(load_official_history_csv(args.csv))
        path = write_report(report, args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"E1全历史诊断失败：{error}", file=sys.stderr)
        print(
            "建议：检查canonical历史；固定协议不接受调参。",
            file=sys.stderr,
        )
        return 1
    history = cast(dict[str, object], report["history"])
    differences = cast(dict[str, Mapping[str, float]], report["compoundDifferences"])
    print(
        "E1全历史诊断完成："
        f"warmup={history['warmupPeriods']}，"
        f"evaluated={history['evaluatedPeriods']}。"
    )
    print(
        "E-D8平均红8交集差="
        f"{differences['EMinusD8Mean']['red8Overlap']:.12f}；"
        "E-随机32平均红8交集差="
        f"{differences['EMinusC32IssueMean']['red8Overlap']:.12f}。"
    )
    print(f"已写入：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成数字彩统一模型证据总账。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_model_scoreboard import (  # noqa: E402
    build_model_scoreboard,
    render_model_scoreboard_markdown,
    write_model_scoreboard,
)

_DEFAULT_REPORTS = (
    "reports/development/behavioral_context_v3_fc3d_all_blocks_20260721.json",
    "reports/development/behavioral_context_v3_pl3_all_blocks_20260721.json",
    "reports/development/behavioral_context_v4_fc3d_all_blocks_20260721.json",
    "reports/development/behavioral_context_v4_pl3_all_blocks_20260721.json",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--behavior-report",
        action="append",
        dest="behavior_reports",
        help="行为全块报告；可重复传入，默认读取v3/v4两个彩种四份报告",
    )
    parser.add_argument(
        "--output-json",
        default="reports/development/model_scoreboard_20260721.json",
    )
    parser.add_argument("--output-markdown", default="docs/model_scoreboard.md")
    args = parser.parse_args(argv)

    report_paths = tuple(args.behavior_reports or _DEFAULT_REPORTS)
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths
    ]
    scoreboard = build_model_scoreboard(reports)
    json_path = write_model_scoreboard(scoreboard, args.output_json)
    markdown_path = Path(args.output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_model_scoreboard_markdown(scoreboard), encoding="utf-8"
    )
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

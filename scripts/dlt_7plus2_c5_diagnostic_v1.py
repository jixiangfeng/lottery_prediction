# -*- coding: utf-8 -*-
"""运行大乐透C5开发历史诊断；逻辑上绝不消费v1 Frozen行。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.analysis.dlt_7plus2_c5_diagnostic_v1 import summarize_c5_diagnostic
from src.analysis.dlt_7plus2_c5_hedge_v1 import walk_forward_c5_predictions

DEFAULT_CSV = Path("data/dlt/official_history.csv")
DEFAULT_OUTPUT = Path("reports/development/dlt_7plus2_c5_diagnostic_v1.json")
PROTOCOL_PATH = Path("docs/dlt_7plus2_c5_hedge_v1_protocol.md")
DEVELOPMENT_START = 600
DEVELOPMENT_STOP = 2401


@dataclass(frozen=True)
class DevelopmentDraw:
    issue: str
    date: str
    front: tuple[int, ...]
    back: tuple[int, ...]


def _parse_zone(text: str, *, size: int, count: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in text.split())
    if (
        len(values) != count
        or tuple(sorted(values)) != values
        or len(set(values)) != count
        or any(value < 1 or value > size for value in values)
    ):
        raise ValueError("大乐透开发前缀号码非法")
    return values


def _load_development_prefix(
    path: str | Path, *, limit: int = DEVELOPMENT_STOP
) -> tuple[DevelopmentDraw, ...]:
    """只调用reader的前``limit``行；后续Frozen/损坏尾行均不解析。"""

    draws: list[DevelopmentDraw] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index in range(limit):
            try:
                row = next(reader)
            except StopIteration as error:
                raise ValueError(f"大乐透CSV不足{limit}行开发前缀") from error
            if None in row or not row.get("issue") or not row.get("date"):
                raise ValueError(f"大乐透开发前缀第{index + 1}行结构非法")
            draws.append(
                DevelopmentDraw(
                    issue=row["issue"],
                    date=row["date"],
                    front=_parse_zone(row["front"], size=35, count=5),
                    back=_parse_zone(row["back"], size=12, count=2),
                )
            )
    return tuple(draws)


def _atomic_write_json(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(csv_path: Path, output_path: Path) -> dict[str, object]:
    draws = _load_development_prefix(csv_path)
    protocol_sha256 = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    predictions = walk_forward_c5_predictions(
        draws,
        start_index=DEVELOPMENT_START,
        stop_index=DEVELOPMENT_STOP,
    )
    report = summarize_c5_diagnostic(
        draws,
        predictions,
        start_index=DEVELOPMENT_START,
        stop_index=DEVELOPMENT_STOP,
        protocol_sha256=protocol_sha256,
    )
    _atomic_write_json(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(args.csv, args.output)
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "reportSha256": report["reportSha256"],
                "evidenceStatus": report["evidenceStatus"],
                "frozenRowsAccessed": report["boundary"]["frozenRowsAccessed"],  # type: ignore[index]
                "meanJointLogLossImprovement": metrics["meanJointLogLossImprovement"],
                "meanJointLogLossImprovementVsC4": metrics[
                    "meanJointLogLossImprovementVsC4"
                ],
                "meanUImprovementVsRandom512": metrics["meanUImprovementVsRandom512"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

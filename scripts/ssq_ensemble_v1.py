# -*- coding: utf-8 -*-
"""执行双色球固定基线集成 v1 严格前序评估。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence, cast

from src.analysis.ssq_ensemble_v1 import (
    evaluate_ssq_ensemble,
    validate_research_candidates,
)
from src.analysis.ssq_history import SSQDraw, load_official_history_csv


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _payload_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _bind_report_to_canonical(
    report: dict[str, object], draws: Sequence[SSQDraw]
) -> dict[str, object]:
    latest = draws[-1]
    data_payload = [
        {
            "issue": draw.issue,
            "date": draw.draw_date,
            "red": list(draw.red),
            "blue": draw.blue,
            "sourceUrl": draw.source_url,
            "rawHash": draw.raw_hash,
        }
        for draw in draws
    ]
    bound = dict(report)
    bound["latestIssue"] = latest.issue
    bound["dataSha256"] = _payload_sha256(data_payload)
    bound["reportSha256"] = _payload_sha256(bound)
    return bound


def main(argv: list[str] | None = None) -> int:
    """运行固定协议；只允许覆盖输入输出路径，不开放模型参数。"""

    parser = argparse.ArgumentParser(
        description="双色球ssq_ensemble_v1固定协议严格前序研究评估"
    )
    parser.add_argument(
        "--csv",
        default="data/ssq/official_history.csv",
        help="经官方证据对账的规范CSV",
    )
    parser.add_argument(
        "--output",
        default="reports/research/ssq_ensemble_v1.json",
        help="研究评估JSON路径",
    )
    args = parser.parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("双色球规范历史不存在或为空：不动作。")
        return 0
    try:
        draws = load_official_history_csv(csv_path)
        report = evaluate_ssq_ensemble(draws)
        validate_research_candidates(
            cast(list[dict[str, object]], report["researchCandidates"])
        )
        report = _bind_report_to_canonical(report, draws)
        _write_json(Path(args.output), report)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"双色球固定集成评估失败：{error}", file=sys.stderr)
        print("建议：先完成官方JSONL对账；模型参数不允许从CLI覆盖。", file=sys.stderr)
        return 1
    print(
        "双色球固定集成评估完成："
        f"decision={report['decision']}，hardGatePassed={report['hardGatePassed']}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

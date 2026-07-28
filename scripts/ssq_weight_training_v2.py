# -*- coding: utf-8 -*-
"""执行双色球静态约束权重训练 v2 严格挑战。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from src.analysis.ssq_history import load_official_history_csv
from src.analysis.ssq_weight_training_v2 import evaluate_ssq_weight_training


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main(argv: list[str] | None = None) -> int:
    """运行不可调参的 v2；CLI 仅允许输入输出路径。"""

    parser = argparse.ArgumentParser(
        description="双色球ssq_weight_training_v2一次性严格抗过拟合挑战"
    )
    parser.add_argument(
        "--csv",
        default="data/ssq/official_history.csv",
        help="经官方证据对账的规范CSV",
    )
    parser.add_argument(
        "--output",
        default="reports/research/ssq_weight_training_v2.json",
        help="研究挑战JSON路径",
    )
    args = parser.parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("双色球规范历史不存在或为空：不动作。")
        return 0
    try:
        report = evaluate_ssq_weight_training(
            load_official_history_csv(csv_path),
            csv_sha256=_sha256_file(csv_path),
        )
        _write_json(Path(args.output), report)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"双色球v2静态权重挑战失败：{error}", file=sys.stderr)
        print(
            "建议：确认官方CSV完整且按协议加载；CLI不允许覆盖任何训练参数。",
            file=sys.stderr,
        )
        return 1
    print(
        "双色球v2静态权重挑战完成："
        f"decision={report['decision']}，"
        f"validationOpened={report['validationOpened']}，"
        f"frozenTestOpened={report['frozenTestOpened']}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

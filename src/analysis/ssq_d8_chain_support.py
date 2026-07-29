# -*- coding: utf-8 -*-
"""D8前瞻链的内部签名、原子工件与边界重建支持。"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from src.analysis import ssq_ensemble_v1 as ensemble
from src.analysis.ssq_history import SSQDraw

SHANGHAI = ZoneInfo("Asia/Shanghai")
DRAW_WEEKDAYS = (1, 3, 6)  # 周二、周四、周日。
DRAW_CUTOFF = time(hour=21, minute=30)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: object) -> str:
    """返回稳定规范 JSON 的 SHA-256。"""

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _artifact_hmac(payload: Mapping[str, object], digest: str, key: bytes) -> str:
    if len(key) < 32:
        raise ValueError("HMAC密钥至少32字节")
    message = _canonical_bytes(payload) + b"\n" + digest.encode("ascii")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _artifact_document(payload: Mapping[str, object], key: bytes) -> dict[str, object]:
    unsigned = dict(payload)
    if "artifactSha256" in unsigned or "artifactHmacSha256" in unsigned:
        raise ValueError("payload不得预置签名字段")
    digest = payload_sha256(unsigned)
    return {
        **unsigned,
        "artifactSha256": digest,
        "artifactHmacSha256": _artifact_hmac(unsigned, digest, key),
    }


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_artifact(
    path: Path, payload: Mapping[str, object], key: bytes
) -> dict[str, object]:
    document = _artifact_document(payload, key)
    _write_exclusive(path, _canonical_bytes(document) + b"\n")
    return document


def _atomic_replace_artifact(
    path: Path, payload: Mapping[str, object], key: bytes
) -> dict[str, object]:
    document = _artifact_document(payload, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(document) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def load_and_verify_artifact(path: str | Path, hmac_key: bytes) -> dict[str, object]:
    """加载 JSON artifact，并验证自哈希与外部 HMAC。"""

    artifact_path = Path(path)
    try:
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取artifact：{artifact_path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"artifact必须是JSON对象：{artifact_path}")
    digest = document.get("artifactSha256")
    signature = document.get("artifactHmacSha256")
    if not isinstance(digest, str) or not isinstance(signature, str):
        raise ValueError(f"artifact缺少SHA/HMAC：{artifact_path}")
    payload = {
        key: value
        for key, value in document.items()
        if key not in {"artifactSha256", "artifactHmacSha256"}
    }
    expected_digest = payload_sha256(payload)
    if not hmac.compare_digest(digest, expected_digest):
        raise ValueError(f"artifact SHA校验失败：{artifact_path}")
    expected_signature = _artifact_hmac(payload, digest, hmac_key)
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError(f"artifact HMAC校验失败：{artifact_path}")
    return cast(dict[str, object], document)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(tz=SHANGHAI)
    if value.tzinfo is None:
        raise ValueError("now必须带时区")
    return value.astimezone(SHANGHAI)


def _draw_deadline(draw_date: str) -> datetime:
    day = date.fromisoformat(draw_date)
    return datetime.combine(day, DRAW_CUTOFF, tzinfo=SHANGHAI)


def _next_target(draws: Sequence[SSQDraw]) -> tuple[str, str]:
    if not draws:
        raise ValueError("canonical历史为空")
    latest = draws[-1]
    latest_date = date.fromisoformat(latest.draw_date)
    target_date = latest_date + timedelta(days=1)
    while target_date.weekday() not in DRAW_WEEKDAYS:
        target_date += timedelta(days=1)
    issue_year = int(latest.issue[:4])
    sequence = int(latest.issue[4:])
    target_issue = (
        f"{issue_year}{sequence + 1:03d}"
        if target_date.year == issue_year
        else f"{target_date.year}001"
    )
    return target_issue, target_date.isoformat()


def _draw_payload(draw: SSQDraw) -> dict[str, object]:
    return {
        "issue": draw.issue,
        "date": draw.draw_date,
        "red": list(draw.red),
        "blue": draw.blue,
        "sourceUrl": draw.source_url,
        "rawHash": draw.raw_hash,
    }


def canonical_data_sha256(draws: Sequence[SSQDraw]) -> str:
    """返回与现有 ssq_ensemble 报告一致的 canonical 数据摘要。"""

    return payload_sha256([_draw_payload(draw) for draw in draws])


def build_bound_ensemble_report(draws: Sequence[SSQDraw]) -> dict[str, object]:
    """重算固定 ensemble，并绑定 canonical 最新期与数据摘要。"""

    report = ensemble.evaluate_ssq_ensemble(draws)
    bound = dict(report)
    bound["latestIssue"] = draws[-1].issue
    bound["dataSha256"] = canonical_data_sha256(draws)
    bound["reportSha256"] = payload_sha256(bound)
    return bound

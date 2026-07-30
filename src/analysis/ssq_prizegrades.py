# -*- coding: utf-8 -*-
"""双色球官方逐期奖级奖金抓取、追加缓存与校验。

缓存保留福彩接口每一条原始记录，而不是推算/填补奖级数据。接口的第七项是
双色球宣传/空白占位项；仅当其金额和中奖注数均为空时忽略，奖级 1--6 必须完整。
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from src.analysis.ssq_history import (
    SSQ_HTTP_RETRIES,
    SSQ_HTTP_TIMEOUT_SECONDS,
    SSQ_MAX_PAGES,
    SSQ_PAGE_SIZE,
    build_ssq_source_url,
    validate_ssq_source_url,
)

_USER_AGENT = "Mozilla/5.0 (compatible; lottery-prediction/1.0; research-only)"


@dataclass(frozen=True)
class SSQPrizeGrade:
    """一个经校验的官方奖级：中奖注数和单注奖金均为非负整数（元）。"""

    grade: int
    winners: int
    amount: int


@dataclass(frozen=True)
class SSQPrizeGradeRecord:
    """一个双色球期号的六个官方奖级及其可审计原始记录。"""

    issue: str
    prizegrades: tuple[SSQPrizeGrade, ...]
    source_url: str
    raw_hash: str
    raw: Mapping[str, object]
    fetched_at: str | None = None


def build_ssq_prizegrade_source_url(page_no: int) -> str:
    """构造奖级查询的固定福彩官方分页 URL。"""

    return build_ssq_source_url(page_no)


def _nonnegative_integer(value: object, field: str) -> int:
    """解析官方纯数字金额，或“总额（含派奖明细）”中的前导总额。"""

    text = str(value).strip()
    if text.isdigit():
        return int(text)
    match = re.fullmatch(r"(\d+)（含(?:派奖|加奖)\d+）", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"双色球官方奖级{field}必须是非负整数")


def canonical_prizegrade_raw_hash(source_url: str, raw: Mapping[str, object]) -> str:
    """对官方来源 URL 和完整原始记录计算稳定 SHA-256。"""

    validate_ssq_source_url(source_url)
    serialized = json.dumps(
        {"sourceUrl": source_url, "raw": raw},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_prizegrade_record(
    raw: Mapping[str, object], source_url: str
) -> SSQPrizeGradeRecord:
    """校验一条官方 SSQ 记录的期号与奖级 1--6，不接受猜测或缺项。"""

    validate_ssq_source_url(source_url)
    if raw.get("name") != "双色球":
        raise ValueError("官方奖级记录玩法身份不是双色球")
    issue = str(raw.get("code", "")).strip()
    if not issue.isdigit() or not 5 <= len(issue) <= 12:
        raise ValueError(f"双色球官方奖级记录期号非法：{issue}")
    raw_grades = raw.get("prizegrades")
    if not isinstance(raw_grades, list):
        raise ValueError("双色球官方奖级 prizegrades 不是列表")

    grades: dict[int, SSQPrizeGrade] = {}
    for raw_grade in raw_grades:
        if not isinstance(raw_grade, dict):
            raise ValueError("双色球官方奖级项不是对象")
        grade_value = raw_grade.get("type")
        if isinstance(grade_value, bool) or not str(grade_value).strip().isdigit():
            raise ValueError("双色球官方奖级 type 非法")
        grade = int(str(grade_value).strip())
        if grade == 7:
            if (
                str(raw_grade.get("typenum", "")).strip()
                or str(raw_grade.get("typemoney", "")).strip()
            ):
                raise ValueError("双色球官方奖级1-6之外存在非空奖级")
            continue
        if grade not in range(1, 7) or grade in grades:
            raise ValueError("双色球官方奖级必须恰为奖级1-6且不重复")
        grades[grade] = SSQPrizeGrade(
            grade=grade,
            winners=_nonnegative_integer(raw_grade.get("typenum"), "中奖注数"),
            amount=_nonnegative_integer(raw_grade.get("typemoney"), "奖金金额"),
        )
    if set(grades) != set(range(1, 7)):
        raise ValueError("双色球官方奖级数量不足，必须完整包含奖级1-6")
    return SSQPrizeGradeRecord(
        issue=issue,
        prizegrades=tuple(grades[grade] for grade in range(1, 7)),
        source_url=source_url,
        raw_hash=canonical_prizegrade_raw_hash(source_url, raw),
        raw=dict(raw),
    )


def _default_request_json(url: str) -> Mapping[str, object]:
    validate_ssq_source_url(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Referer": "https://www.cwl.gov.cn/"},
        method="GET",
    )
    last_error: Exception | None = None
    for attempt in range(SSQ_HTTP_RETRIES + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=SSQ_HTTP_TIMEOUT_SECONDS
            ) as response:
                final_url = response.geturl()
                validate_ssq_source_url(final_url)
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("双色球官方奖级接口顶层 JSON 不是对象")
            return payload
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if attempt < SSQ_HTTP_RETRIES:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"双色球官方奖级接口请求失败：{last_error}") from last_error


def fetch_ssq_prizegrades(
    *,
    start_page: int = 1,
    pages: int = 1,
    request_json: Callable[[str], Mapping[str, object]] | None = None,
) -> list[SSQPrizeGradeRecord]:
    """抓取连续有限页；用 ``start_page/pages`` 可在中断后从下一页续跑。"""

    if not 1 <= start_page <= SSQ_MAX_PAGES or pages <= 0:
        raise ValueError("双色球奖级分页范围非法")
    if start_page + pages - 1 > SSQ_MAX_PAGES:
        raise ValueError("双色球奖级分页越过安全上限")
    request = request_json or _default_request_json
    records: dict[str, SSQPrizeGradeRecord] = {}
    for page_no in range(start_page, start_page + pages):
        source_url = build_ssq_prizegrade_source_url(page_no)
        payload = request(source_url)
        if payload.get("message") != "查询成功":
            raise RuntimeError(
                f"双色球官方奖级接口返回失败：{payload.get('message', '未知错误')}"
            )
        raw_results = payload.get("result")
        if not isinstance(raw_results, list) or len(raw_results) > SSQ_PAGE_SIZE:
            raise ValueError("双色球官方奖级接口 result 非法")
        for raw in raw_results:
            if not isinstance(raw, dict):
                raise ValueError("双色球官方奖级接口 result 包含非对象记录")
            record = validate_prizegrade_record(raw, source_url)
            previous = records.get(record.issue)
            if previous is not None and previous.raw_hash != record.raw_hash:
                raise ValueError(
                    f"双色球官方奖级同一期号在本批分页发生冲突：{record.issue}"
                )
            records[record.issue] = record
    return sorted(records.values(), key=lambda record: int(record.issue), reverse=True)


def _evidence_payload(
    record: SSQPrizeGradeRecord, fetched_at: str
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "lottery": "ssq",
        "kind": "prizegrades",
        "sourceUrl": record.source_url,
        "fetchedAt": fetched_at,
        "rawHash": record.raw_hash,
        "raw": record.raw,
    }


def append_prizegrade_evidence(
    path: str | Path,
    records: Sequence[SSQPrizeGradeRecord],
    *,
    fetched_at: str | None = None,
) -> int:
    """以锁与 fsync 追加原始 JSONL；不覆盖既有缓存字节。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = fetched_at or datetime.now(timezone.utc).isoformat()
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("双色球奖级抓取时间非法") from error
    lines = [
        json.dumps(
            _evidence_payload(record, timestamp),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for record in records
    ]
    with output.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell():
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    raise ValueError("既有双色球奖级 JSONL 末尾缺少 LF，拒绝追加")
            stream.seek(0, os.SEEK_END)
            for line in lines:
                stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return len(lines)


def read_prizegrade_evidence(path: str | Path) -> list[SSQPrizeGradeRecord]:
    """读取缓存并重新验证来源、抓取时间、原始哈希和六个奖级。"""

    records: list[SSQPrizeGradeRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"双色球奖级 JSONL 第 {line_number} 行缺少 LF")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"双色球奖级 JSONL 第 {line_number} 行不是合法 JSON"
                ) from error
            if not isinstance(payload, dict) or (
                payload.get("schemaVersion"),
                payload.get("lottery"),
                payload.get("kind"),
            ) != (1, "ssq", "prizegrades"):
                raise ValueError(f"双色球奖级 JSONL 第 {line_number} 行身份非法")
            source_url, fetched_at, raw_hash, raw = (
                payload.get("sourceUrl"),
                payload.get("fetchedAt"),
                payload.get("rawHash"),
                payload.get("raw"),
            )
            if (
                not isinstance(source_url, str)
                or not isinstance(fetched_at, str)
                or not isinstance(raw_hash, str)
                or not isinstance(raw, dict)
            ):
                raise ValueError(f"双色球奖级 JSONL 第 {line_number} 行字段类型非法")
            try:
                datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(
                    f"双色球奖级 JSONL 第 {line_number} 行抓取时间非法"
                ) from error
            record = validate_prizegrade_record(raw, source_url)
            if raw_hash != record.raw_hash:
                raise ValueError(f"双色球奖级 JSONL 第 {line_number} 行原始哈希不匹配")
            records.append(
                SSQPrizeGradeRecord(**{**record.__dict__, "fetched_at": fetched_at})
            )
    return records


def reconcile_prizegrade_evidence(
    records: Sequence[SSQPrizeGradeRecord],
) -> list[SSQPrizeGradeRecord]:
    """按抓取时间选择同一期最新可验证原始记录；同一时刻冲突直接拒绝。"""

    if not records:
        raise ValueError("双色球奖级原始证据为空")
    latest: dict[str, SSQPrizeGradeRecord] = {}
    for record in records:
        if record.fetched_at is None:
            raise ValueError("双色球奖级证据缺少抓取时间")
        previous = latest.get(record.issue)
        if previous is None or record.fetched_at > previous.fetched_at:  # type: ignore[operator]
            latest[record.issue] = record
        elif (
            record.fetched_at == previous.fetched_at
            and record.raw_hash != previous.raw_hash
        ):
            raise ValueError(f"双色球奖级同一期同一抓取时间发生冲突：{record.issue}")
    return sorted(latest.values(), key=lambda record: int(record.issue), reverse=True)


def write_prizegrades_json(
    path: str | Path, records: Sequence[SSQPrizeGradeRecord]
) -> None:
    """原子写入经缓存对账的结构化奖级快照，缺期不会生成任何占位数据。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "lottery": "ssq",
        "kind": "prizegrades",
        "records": [
            {
                "issue": record.issue,
                "sourceUrl": record.source_url,
                "fetchedAt": record.fetched_at,
                "rawHash": record.raw_hash,
                "prizegrades": [grade.__dict__ for grade in record.prizegrades],
            }
            for record in records
        ],
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "SSQPrizeGrade",
    "SSQPrizeGradeRecord",
    "append_prizegrade_evidence",
    "build_ssq_prizegrade_source_url",
    "canonical_prizegrade_raw_hash",
    "fetch_ssq_prizegrades",
    "read_prizegrade_evidence",
    "reconcile_prizegrade_evidence",
    "validate_prizegrade_record",
    "write_prizegrades_json",
]

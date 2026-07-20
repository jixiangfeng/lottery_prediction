# -*- coding: utf-8 -*-
"""数字彩原始JSONL证据层与多源对账。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from src.analysis.digit_history_fetcher import DigitHistoryDraw


@dataclass(frozen=True)
class RawDigitRecord:
    lottery_code: str
    issue: str
    draw_date: str
    digits: tuple[int, int, int]
    source_name: str
    source_url: str
    fetched_at: str
    raw: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.lottery_code not in {"fc3d", "pl3"}:
            raise ValueError("原始证据层只支持fc3d/pl3")
        if not self.issue.isdigit():
            raise ValueError("期号必须为纯数字")
        if len(self.digits) != 3 or any(
            value < 0 or value > 9 for value in self.digits
        ):
            raise ValueError("开奖号必须是三个0-9数字")
        if not self.source_name or not self.source_url or not self.fetched_at:
            raise ValueError("来源和抓取时间不能为空")

    @property
    def raw_sha256(self) -> str:
        payload = json.dumps(
            self.raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def dedup_key(self) -> tuple[str, str, str]:
        return self.lottery_code, self.issue, self.source_name

    @property
    def content_identity(self) -> tuple[object, ...]:
        return (
            self.lottery_code,
            self.issue,
            self.draw_date,
            self.digits,
            self.source_name,
            self.source_url,
            self.raw_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "lotteryCode": self.lottery_code,
            "issue": self.issue,
            "drawDate": self.draw_date,
            "digits": list(self.digits),
            "sourceName": self.source_name,
            "sourceUrl": self.source_url,
            "fetchedAt": self.fetched_at,
            "raw": dict(self.raw),
            "rawSha256": self.raw_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RawDigitRecord:
        raw = payload.get("raw")
        digits = payload.get("digits")
        if not isinstance(raw, Mapping) or not isinstance(digits, list):
            raise ValueError("JSONL原始记录缺少raw或digits")
        record = cls(
            lottery_code=str(payload.get("lotteryCode", "")),
            issue=str(payload.get("issue", "")),
            draw_date=str(payload.get("drawDate", "")),
            digits=tuple(int(value) for value in digits),  # type: ignore[arg-type]
            source_name=str(payload.get("sourceName", "")),
            source_url=str(payload.get("sourceUrl", "")),
            fetched_at=str(payload.get("fetchedAt", "")),
            raw=dict(raw),
        )
        expected_hash = payload.get("rawSha256")
        if expected_hash is not None and str(expected_hash) != record.raw_sha256:
            raise ValueError(f"原始记录语义哈希不匹配：{record.issue}")
        return record


class DigitHistoryProvider(Protocol):
    """多来源原始证据provider统一接口。"""

    @property
    def name(self) -> str: ...

    def load(self) -> tuple[RawDigitRecord, ...]: ...


@dataclass(frozen=True)
class JsonlDigitHistoryProvider:
    name: str
    path: Path

    def load(self) -> tuple[RawDigitRecord, ...]:
        records = read_raw_digit_jsonl(self.path)
        if any(record.source_name != self.name for record in records):
            raise ValueError(f"provider名称与JSONL来源不一致：{self.name}")
        return records


def read_raw_digit_jsonl(path: str | Path) -> tuple[RawDigitRecord, ...]:
    target = Path(path)
    if not target.exists():
        return ()
    records: list[RawDigitRecord] = []
    with target.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL第{line_number}行损坏") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL第{line_number}行不是对象")
            records.append(RawDigitRecord.from_dict(payload))
    return tuple(records)


def append_raw_digit_jsonl(path: str | Path, records: Iterable[RawDigitRecord]) -> Path:
    """按lottery+issue+source追加；同键内容冲突时阻断。"""

    target = Path(path)
    existing = {record.dedup_key: record for record in read_raw_digit_jsonl(target)}
    for record in records:
        prior = existing.get(record.dedup_key)
        if prior is not None and prior.content_identity != record.content_identity:
            raise ValueError(f"同来源同期开奖内容冲突：{record.dedup_key}")
        if prior is None:
            existing[record.dedup_key] = record
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in sorted(existing.values(), key=lambda item: item.dedup_key):
                stream.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def collect_provider_records(
    providers: Iterable[DigitHistoryProvider],
) -> tuple[RawDigitRecord, ...]:
    """按provider名称稳定顺序加载并合并原始证据。"""

    loaded: list[RawDigitRecord] = []
    names: set[str] = set()
    for provider in providers:
        if provider.name in names:
            raise ValueError(f"provider名称重复：{provider.name}")
        names.add(provider.name)
        loaded.extend(provider.load())
    return tuple(sorted(loaded, key=lambda item: item.dedup_key))


def build_reconciliation_report(
    records: Iterable[RawDigitRecord], *, minimum_sources: int = 2
) -> dict[str, Any]:
    """返回已接受、来源不足和冲突期次清单，不修改CSV。"""

    if minimum_sources <= 0:
        raise ValueError("minimum_sources必须大于零")
    grouped: dict[tuple[str, str], list[RawDigitRecord]] = {}
    for record in records:
        grouped.setdefault((record.lottery_code, record.issue), []).append(record)
    accepted: list[str] = []
    insufficient: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    for (lottery, issue), issue_records in sorted(grouped.items()):
        by_source = {record.source_name: record for record in issue_records}
        numbers = {record.digits for record in by_source.values()}
        dates = {record.draw_date for record in by_source.values() if record.draw_date}
        if len(numbers) > 1 or len(dates) > 1:
            conflicts.append(
                {
                    "lottery": lottery,
                    "issue": issue,
                    "sources": sorted(by_source),
                    "numbers": sorted("".join(map(str, value)) for value in numbers),
                    "dates": sorted(dates),
                }
            )
        elif len(by_source) < minimum_sources:
            insufficient.append(
                {
                    "lottery": lottery,
                    "issue": issue,
                    "sources": sorted(by_source),
                }
            )
        else:
            accepted.append(issue)
    return {
        "schemaVersion": 1,
        "minimumSources": minimum_sources,
        "acceptedIssues": accepted,
        "insufficientSourceIssues": insufficient,
        "conflicts": conflicts,
        "passed": not conflicts,
    }


def reconcile_raw_digit_records(
    records: Iterable[RawDigitRecord],
    *,
    minimum_sources: int = 2,
) -> tuple[DigitHistoryDraw, ...]:
    """多源号码一致才输出；来源不足的期次保留在证据层但不进入CSV。"""

    if minimum_sources <= 0:
        raise ValueError("minimum_sources必须大于零")
    grouped: dict[tuple[str, str], list[RawDigitRecord]] = {}
    for record in records:
        grouped.setdefault((record.lottery_code, record.issue), []).append(record)
    draws: list[DigitHistoryDraw] = []
    for (_, issue), issue_records in grouped.items():
        by_source = {record.source_name: record for record in issue_records}
        numbers = {record.digits for record in by_source.values()}
        if len(numbers) > 1:
            raise ValueError(f"多来源开奖号冲突：{issue}")
        if len(by_source) < minimum_sources:
            continue
        dates = {record.draw_date for record in by_source.values() if record.draw_date}
        if len(dates) > 1:
            raise ValueError(f"多来源开奖日期冲突：{issue}")
        digits = next(iter(numbers))
        draws.append(
            DigitHistoryDraw(
                issue=issue,
                numbers=digits,
                draw_date=next(iter(dates), ""),
                source="|".join(sorted(by_source)),
            )
        )
    return tuple(sorted(draws, key=lambda item: int(item.issue), reverse=True))


__all__ = [
    "DigitHistoryProvider",
    "JsonlDigitHistoryProvider",
    "RawDigitRecord",
    "append_raw_digit_jsonl",
    "build_reconciliation_report",
    "collect_provider_records",
    "read_raw_digit_jsonl",
    "reconcile_raw_digit_records",
]

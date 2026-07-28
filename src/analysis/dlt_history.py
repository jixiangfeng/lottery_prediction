# -*- coding: utf-8 -*-
"""大乐透官方历史抓取、原始证据保存及对账。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from src.lotteries.dlt import DLT_RULE

DLT_API_URL = "https://webapi.sporttery.cn/gateway/lottery/" "getHistoryPageListV1.qry"
DLT_SOURCE_HOST = "webapi.sporttery.cn"
DLT_SOURCE_PATH = "/gateway/lottery/getHistoryPageListV1.qry"
DLT_PAGE_SIZE = 30
DLT_MAX_PAGES = 1000
DLT_MAX_TOTAL = DLT_PAGE_SIZE * DLT_MAX_PAGES
DLT_HTTP_TIMEOUT_SECONDS = 20.0
DLT_HTTP_RETRIES = 3
DLT_CSV_FIELDS = (
    "issue",
    "date",
    "front",
    "back",
    "source_url",
    "raw_hash",
)
DLT_REQUIRED_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; lottery-prediction/1.0; research-only)",
    "Referer": "https://www.sporttery.cn/",
    "Origin": "https://www.sporttery.cn",
    "Accept": "application/json, text/plain, */*",
}
_QUERY_KEYS = frozenset(
    {"gameNo", "provinceId", "pageSize", "isVerify", "pageNo", "termLimits"}
)


@dataclass(frozen=True)
class DLTDraw:
    """一条已通过身份、日期和号码校验的大乐透开奖。"""

    issue: str
    draw_date: str
    front: tuple[int, ...]
    back: tuple[int, ...]
    source_url: str
    raw_hash: str
    raw: Mapping[str, object]


def build_dlt_source_url(page_no: int) -> str:
    """构造只有固定官方参数的大乐透分页 URL。"""

    if isinstance(page_no, bool) or not 1 <= page_no <= DLT_MAX_PAGES:
        raise ValueError("大乐透分页页码越过安全范围")
    query = urllib.parse.urlencode(
        {
            "gameNo": DLT_RULE.game_no,
            "provinceId": "0",
            "pageSize": DLT_PAGE_SIZE,
            "isVerify": "1",
            "pageNo": page_no,
            "termLimits": "0",
        }
    )
    return f"{DLT_API_URL}?{query}"


def validate_dlt_source_url(source_url: str) -> None:
    """拒绝非体彩官方 HTTPS 地址及任意查询参数漂移。"""

    parsed = urllib.parse.urlparse(source_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != DLT_SOURCE_HOST
        or parsed.path != DLT_SOURCE_PATH
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise ValueError("大乐透来源 URL 不在官方白名单")
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if frozenset(query) != _QUERY_KEYS or any(
        len(values) != 1 for values in query.values()
    ):
        raise ValueError("大乐透来源 URL 参数集合不符合固定协议")
    fixed = {
        "gameNo": DLT_RULE.game_no,
        "provinceId": "0",
        "pageSize": str(DLT_PAGE_SIZE),
        "isVerify": "1",
        "termLimits": "0",
    }
    for key, expected in fixed.items():
        if query[key][0] != expected:
            raise ValueError(f"大乐透来源 URL 固定参数非法：{key}")
    page_text = query["pageNo"][0]
    if not page_text.isdigit() or not 1 <= int(page_text) <= DLT_MAX_PAGES:
        raise ValueError("大乐透来源 URL 页码非法")


def canonical_raw_hash(source_url: str, raw: Mapping[str, object]) -> str:
    """计算绑定官方来源 URL 与完整原始记录的稳定 SHA-256。"""

    validate_dlt_source_url(source_url)
    serialized = json.dumps(
        {"sourceUrl": source_url, "raw": raw},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"大乐透官方接口 {field} 非法")
    text = str(value)
    if not text.isdigit():
        raise ValueError(f"大乐透官方接口 {field} 非法")
    return int(text)


def _normalized_date(value: object, *, today: date | None = None) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"大乐透官方记录日期非法：{value}")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError(f"大乐透官方记录日期非法：{value}") from error
    if parsed > (today or date.today()):
        raise ValueError(f"大乐透官方记录包含未来日期：{value}")
    return parsed.isoformat()


def validate_official_record(
    raw: Mapping[str, object], source_url: str, *, today: date | None = None
) -> DLTDraw:
    """严格校验一条体彩网大乐透原始记录。"""

    validate_dlt_source_url(source_url)
    if raw.get("lotteryGameNum") != DLT_RULE.game_no:
        raise ValueError("大乐透官方记录 gameNo 不是 85")
    if raw.get("lotteryGameName") != DLT_RULE.display_name:
        raise ValueError("大乐透官方记录名称不是超级大乐透")
    issue_value = raw.get("lotteryDrawNum")
    if not isinstance(issue_value, str):
        raise ValueError("大乐透官方记录期号非法")
    issue = issue_value.strip()
    if issue != issue_value or len(issue) != 5 or not issue.isdigit():
        raise ValueError(f"大乐透官方记录期号非法：{issue}")
    result = raw.get("lotteryDrawResult")
    if not isinstance(result, str) or result != result.strip():
        raise ValueError("大乐透官方开奖号码格式非法")
    parts = result.split(" ")
    if len(parts) != 7 or any(len(part) != 2 or not part.isdigit() for part in parts):
        raise ValueError("大乐透官方开奖号码必须为 7 个空格分隔两位数")
    front, back = DLT_RULE.validate_draw(
        tuple(int(part) for part in parts[:5]),
        tuple(int(part) for part in parts[5:]),
    )
    draw_date = _normalized_date(raw.get("lotteryDrawTime"), today=today)
    raw_copy = dict(raw)
    return DLTDraw(
        issue=issue,
        draw_date=draw_date,
        front=front,
        back=back,
        source_url=source_url,
        raw_hash=canonical_raw_hash(source_url, raw_copy),
        raw=raw_copy,
    )


def _default_request_json(url: str) -> Mapping[str, object]:
    validate_dlt_source_url(url)
    request = urllib.request.Request(url, headers=DLT_REQUIRED_HEADERS, method="GET")
    last_error: Exception | None = None
    for attempt in range(DLT_HTTP_RETRIES + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=DLT_HTTP_TIMEOUT_SECONDS
            ) as response:
                final_url = response.geturl()
                validate_dlt_source_url(final_url)
                content_type = response.headers.get_content_type()
                if content_type not in {"application/json", "text/json", "text/plain"}:
                    raise ValueError(
                        f"大乐透官方接口 Content-Type 非 JSON：{content_type}"
                    )
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("大乐透官方接口顶层 JSON 不是对象")
            return payload
        except ValueError:
            raise
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if attempt < DLT_HTTP_RETRIES:
                time.sleep(min(0.5 * (2**attempt), 4.0))
    raise RuntimeError(f"大乐透官方接口请求失败：{last_error}") from last_error


def _response_value(
    payload: Mapping[str, object], expected_page_no: int
) -> tuple[Mapping[str, object], int, int]:
    if (
        payload.get("success") is not True
        or payload.get("errorCode") != "0"
        or payload.get("errorMessage") != "处理成功"
    ):
        raise ValueError("大乐透官方接口响应状态或身份非法")
    value = payload.get("value")
    if not isinstance(value, dict):
        raise ValueError("大乐透官方接口 value 不是对象")
    page_no = _strict_int(value.get("pageNo"), "pageNo")
    page_size = _strict_int(value.get("pageSize"), "pageSize")
    pages = _strict_int(value.get("pages"), "pages")
    total = _strict_int(value.get("total"), "total")
    if page_no != expected_page_no:
        raise ValueError(
            f"大乐透官方接口 pageNo 不匹配：请求 {expected_page_no}，响应 {page_no}"
        )
    if page_size != DLT_PAGE_SIZE:
        raise ValueError("大乐透官方接口 pageSize 不符合固定协议")
    if total <= 0 or total > DLT_MAX_TOTAL:
        raise ValueError("大乐透官方接口 total 越过安全范围")
    expected_pages = math.ceil(total / DLT_PAGE_SIZE)
    if pages != expected_pages or not 1 <= pages <= DLT_MAX_PAGES:
        raise ValueError("大乐透官方接口 pages 与 total/pageSize 不一致")
    return value, pages, total


def _semantic(draw: DLTDraw) -> tuple[object, ...]:
    return (draw.draw_date, draw.front, draw.back)


def _validate_descending(draws: Sequence[DLTDraw]) -> None:
    for newer, older in zip(draws, draws[1:]):
        if int(newer.issue) <= int(older.issue):
            raise ValueError("大乐透官方分页期号不是严格降序")
        if newer.draw_date < older.draw_date:
            raise ValueError("大乐透官方分页日期不是降序")


def fetch_dlt_history(
    *, request_json: Callable[[str], Mapping[str, object]] | None = None
) -> list[DLTDraw]:
    """抓取且完整验证官方声明的全部分页，绝不补造缺失开奖。"""

    request = request_json or _default_request_json
    draws: list[DLTDraw] = []
    by_issue: dict[str, DLTDraw] = {}
    declared_pages: int | None = None
    declared_total: int | None = None
    expected_latest: tuple[str, str, tuple[int, ...], tuple[int, ...]] | None = None
    for page_no in range(1, DLT_MAX_PAGES + 1):
        source_url = build_dlt_source_url(page_no)
        payload = request(source_url)
        if not isinstance(payload, dict):
            raise ValueError("大乐透官方接口顶层 JSON 不是对象")
        value, pages, total = _response_value(payload, page_no)
        if declared_pages is None:
            declared_pages, declared_total = pages, total
            last_pool = value.get("lastPoolDraw")
            if not isinstance(last_pool, dict):
                raise ValueError("大乐透官方接口 lastPoolDraw 不是对象")
            latest = validate_official_record(last_pool, source_url)
            expected_latest = (
                latest.issue,
                latest.draw_date,
                latest.front,
                latest.back,
            )
        elif pages != declared_pages or total != declared_total:
            raise ValueError("大乐透官方接口分页 pages/total 不一致")
        raw_rows = value.get("list")
        if not isinstance(raw_rows, list):
            raise ValueError("大乐透官方接口 list 不是列表")
        expected_count = (
            DLT_PAGE_SIZE if page_no < pages else total - DLT_PAGE_SIZE * (pages - 1)
        )
        if len(raw_rows) != expected_count:
            raise ValueError(
                f"大乐透官方接口第 {page_no} 页记录数不匹配："
                f"应为 {expected_count}，实际 {len(raw_rows)}"
            )
        page_draws: list[DLTDraw] = []
        for raw in raw_rows:
            if not isinstance(raw, dict):
                raise ValueError("大乐透官方接口 list 包含非对象记录")
            draw = validate_official_record(raw, source_url)
            previous = by_issue.get(draw.issue)
            if previous is not None:
                if _semantic(previous) != _semantic(draw):
                    raise ValueError(f"大乐透官方数据同一期号发生冲突：{draw.issue}")
                raise ValueError(f"大乐透官方分页出现重复期号：{draw.issue}")
            by_issue[draw.issue] = draw
            page_draws.append(draw)
        _validate_descending(page_draws)
        draws.extend(page_draws)
        if page_no >= pages:
            break
    else:
        raise RuntimeError("大乐透官方接口分页超过安全上限")
    if declared_total is None or len(draws) != declared_total:
        raise ValueError("大乐透官方接口声明总数与实际记录数不一致")
    _validate_descending(draws)
    if not draws or expected_latest != (
        draws[0].issue,
        draws[0].draw_date,
        draws[0].front,
        draws[0].back,
    ):
        raise ValueError("大乐透 lastPoolDraw 与历史首条不一致")
    return draws


def evidence_payload(draw: DLTDraw, fetched_at: str) -> dict[str, object]:
    """生成保留完整官方记录及来源的 JSONL 证据对象。"""

    return {
        "schemaVersion": 1,
        "lottery": "dlt",
        "sourceUrl": draw.source_url,
        "fetchedAt": fetched_at,
        "rawHash": draw.raw_hash,
        "raw": draw.raw,
    }


def _atomic_write_bytes(path: str | Path, content: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_raw_evidence(
    path: str | Path,
    draws: Sequence[DLTDraw],
    *,
    fetched_at: str | None = None,
) -> None:
    """全部记录校验完成后，原子写入 LF 原始官方证据快照。"""

    if not draws:
        raise ValueError("大乐透原始证据不接受空历史")
    timestamp = fetched_at or datetime.now(timezone.utc).isoformat()
    parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed_timestamp.tzinfo is None:
        raise ValueError("大乐透抓取时间必须含时区")
    lines = [
        json.dumps(
            evidence_payload(draw, timestamp),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for draw in draws
    ]
    _atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def read_raw_evidence(path: str | Path) -> list[DLTDraw]:
    """逐行验证 JSONL 身份、来源、抓取时间和稳定原始哈希。"""

    evidence_path = Path(path)
    draws: list[DLTDraw] = []
    with evidence_path.open("rb") as stream:
        for line_number, binary_line in enumerate(stream, start=1):
            if not binary_line.endswith(b"\n") or b"\r" in binary_line:
                raise ValueError(f"大乐透 JSONL 第 {line_number} 行不是纯 LF")
            try:
                payload = json.loads(binary_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"大乐透 JSONL 第 {line_number} 行不是合法 UTF-8 JSON"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(f"大乐透 JSONL 第 {line_number} 行不是对象")
            if payload.get("schemaVersion") != 1 or payload.get("lottery") != "dlt":
                raise ValueError(f"大乐透 JSONL 第 {line_number} 行身份非法")
            source_url = payload.get("sourceUrl")
            fetched_at = payload.get("fetchedAt")
            raw_hash = payload.get("rawHash")
            raw = payload.get("raw")
            if not isinstance(source_url, str) or not isinstance(raw, dict):
                raise ValueError(f"大乐透 JSONL 第 {line_number} 行字段类型非法")
            if not isinstance(fetched_at, str):
                raise ValueError(f"大乐透 JSONL 第 {line_number} 行缺少抓取时间")
            try:
                parsed = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(
                    f"大乐透 JSONL 第 {line_number} 行抓取时间非法"
                ) from error
            if parsed.tzinfo is None:
                raise ValueError(f"大乐透 JSONL 第 {line_number} 行抓取时间无时区")
            expected_hash = canonical_raw_hash(source_url, raw)
            if raw_hash != expected_hash:
                raise ValueError(f"大乐透 JSONL 第 {line_number} 行原始哈希不匹配")
            draws.append(validate_official_record(raw, source_url))
    if not draws:
        raise ValueError("大乐透原始证据为空")
    return draws


def reconcile_evidence(draws: Iterable[DLTDraw]) -> list[DLTDraw]:
    """按期号语义去重，拒绝同一期号日期或号码冲突。"""

    by_issue: dict[str, DLTDraw] = {}
    for draw in draws:
        validate_dlt_source_url(draw.source_url)
        previous = by_issue.get(draw.issue)
        if previous is not None and _semantic(previous) != _semantic(draw):
            raise ValueError(f"大乐透官方数据同一期号发生冲突：{draw.issue}")
        if previous is None:
            by_issue[draw.issue] = draw
    if not by_issue:
        raise ValueError("大乐透原始证据为空")
    ordered = sorted(by_issue.values(), key=lambda draw: int(draw.issue))
    for older, newer in zip(ordered, ordered[1:]):
        if older.draw_date > newer.draw_date:
            raise ValueError("大乐透历史期号与日期 chronology 冲突")
    return ordered


def _csv_bytes(draws: Sequence[DLTDraw]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=DLT_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for draw in draws:
        writer.writerow(
            {
                "issue": draw.issue,
                "date": draw.draw_date,
                "front": DLT_RULE.format_zone(draw.front),
                "back": DLT_RULE.format_zone(draw.back),
                "source_url": draw.source_url,
                "raw_hash": draw.raw_hash,
            }
        )
    return stream.getvalue().encode("utf-8")


def _source_fingerprints(draws: Sequence[DLTDraw]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, DLTDraw]] = {}
    for draw in draws:
        grouped.setdefault(draw.source_url, {})[draw.issue] = draw
    fingerprints: list[dict[str, object]] = []
    for source_url in sorted(
        grouped,
        key=lambda url: int(
            urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["pageNo"][0]
        ),
    ):
        page_draws = sorted(
            grouped[source_url].values(), key=lambda item: int(item.issue)
        )
        material = json.dumps(
            {
                "sourceUrl": source_url,
                "rawHashes": [draw.raw_hash for draw in page_draws],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        page_no = int(
            urllib.parse.parse_qs(urllib.parse.urlparse(source_url).query)["pageNo"][0]
        )
        fingerprints.append(
            {
                "pageNo": page_no,
                "sourceUrl": source_url,
                "rowCount": len(page_draws),
                "sha256": hashlib.sha256(material).hexdigest(),
            }
        )
    return fingerprints


def reconcile_raw_jsonl(
    raw_path: str | Path,
    csv_path: str | Path,
    report_path: str | Path,
) -> tuple[list[DLTDraw], dict[str, object]]:
    """完整验证原始证据后原子生成 LF CSV 与稳定对账 JSON。"""

    raw_file = Path(raw_path)
    draws = reconcile_evidence(read_raw_evidence(raw_file))
    content = _csv_bytes(draws)
    report: dict[str, object] = {
        "schemaVersion": 1,
        "lottery": "dlt",
        "count": len(draws),
        "first": {"issue": draws[0].issue, "date": draws[0].draw_date},
        "latest": {"issue": draws[-1].issue, "date": draws[-1].draw_date},
        "dataSha256": hashlib.sha256(content).hexdigest(),
        "rawEvidenceSha256": hashlib.sha256(raw_file.read_bytes()).hexdigest(),
        "sourceFingerprints": _source_fingerprints(draws),
    }
    report_content = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(csv_path, content)
    _atomic_write_bytes(report_path, report_content)
    return draws, report


__all__ = [
    "DLT_API_URL",
    "DLT_CSV_FIELDS",
    "DLT_PAGE_SIZE",
    "DLT_REQUIRED_HEADERS",
    "DLTDraw",
    "atomic_write_raw_evidence",
    "build_dlt_source_url",
    "canonical_raw_hash",
    "fetch_dlt_history",
    "read_raw_evidence",
    "reconcile_evidence",
    "reconcile_raw_jsonl",
    "validate_dlt_source_url",
    "validate_official_record",
]

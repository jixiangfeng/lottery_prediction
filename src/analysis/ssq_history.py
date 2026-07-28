# -*- coding: utf-8 -*-
"""双色球官方历史证据抓取、校验、对账与加载。"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from src.lotteries.ssq import SSQ_RULE

SSQ_API_URL = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
SSQ_SOURCE_HOST = "www.cwl.gov.cn"
SSQ_SOURCE_PATH = "/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
SSQ_PAGE_SIZE = 100
SSQ_MAX_PAGES = 1000
SSQ_MAX_TOTAL = SSQ_PAGE_SIZE * SSQ_MAX_PAGES
SSQ_HTTP_TIMEOUT_SECONDS = 20.0
SSQ_HTTP_RETRIES = 2
SSQ_CSV_FIELDS = ("issue", "date", "red", "blue", "source_url", "raw_hash")
_QUERY_KEYS = frozenset(
    {
        "name",
        "pageNo",
        "pageSize",
        "issueCount",
        "issueStart",
        "issueEnd",
        "dayStart",
        "dayEnd",
        "week",
        "systemType",
    }
)
_USER_AGENT = "Mozilla/5.0 (compatible; lottery-prediction/1.0; research-only)"


@dataclass(frozen=True)
class SSQDraw:
    """一条经过完整身份、号码与日期校验的双色球开奖。"""

    issue: str
    draw_date: str
    red: tuple[int, ...]
    blue: int
    source_url: str
    raw_hash: str
    raw: Mapping[str, object]


def build_ssq_source_url(page_no: int) -> str:
    """构造只含预声明参数的官方双色球分页 URL。"""

    if page_no <= 0 or page_no > SSQ_MAX_PAGES:
        raise ValueError("双色球分页页码越过安全范围")
    query = urllib.parse.urlencode(
        {
            "name": "ssq",
            "pageNo": page_no,
            "pageSize": SSQ_PAGE_SIZE,
            "issueCount": "",
            "issueStart": "",
            "issueEnd": "",
            "dayStart": "",
            "dayEnd": "",
            "week": "",
            "systemType": "PC",
        }
    )
    return f"{SSQ_API_URL}?{query}"


def validate_ssq_source_url(source_url: str) -> None:
    """拒绝非福彩官网或参数集合被篡改的来源 URL。"""

    parsed = urllib.parse.urlparse(source_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != SSQ_SOURCE_HOST
        or parsed.path != SSQ_SOURCE_PATH
        or parsed.fragment
    ):
        raise ValueError("双色球来源 URL 不在官方白名单")
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if frozenset(query) != _QUERY_KEYS or any(
        len(values) != 1 for values in query.values()
    ):
        raise ValueError("双色球来源 URL 参数集合不符合固定协议")
    fixed = {
        "name": "ssq",
        "pageSize": str(SSQ_PAGE_SIZE),
        "issueCount": "",
        "issueStart": "",
        "issueEnd": "",
        "dayStart": "",
        "dayEnd": "",
        "week": "",
        "systemType": "PC",
    }
    for key, expected in fixed.items():
        if query[key][0] != expected:
            raise ValueError(f"双色球来源 URL 固定参数非法：{key}")
    page_text = query["pageNo"][0]
    if not page_text.isdigit() or not 1 <= int(page_text) <= SSQ_MAX_PAGES:
        raise ValueError("双色球来源 URL 页码非法")


def canonical_raw_hash(source_url: str, raw: Mapping[str, object]) -> str:
    """计算来源 URL 与官方原始记录的稳定 SHA-256。"""

    validate_ssq_source_url(source_url)
    serialized = json.dumps(
        {"sourceUrl": source_url, "raw": raw},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalized_date(value: object) -> str:
    text = str(value).strip()
    date_text = text[:10]
    try:
        parsed = datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError(f"双色球官方记录日期非法：{value}") from error
    if text and len(text) > 10 and text[10] not in {" ", "(", "（"}:
        raise ValueError(f"双色球官方记录日期尾部格式非法：{value}")
    return parsed.date().isoformat()


def validate_official_record(raw: Mapping[str, object], source_url: str) -> SSQDraw:
    """严格校验一条福彩官网双色球原始记录。"""

    validate_ssq_source_url(source_url)
    if raw.get("name") != "双色球":
        raise ValueError("官方记录玩法身份不是双色球")
    issue = str(raw.get("code", "")).strip()
    if not issue.isdigit() or len(issue) < 5 or len(issue) > 12:
        raise ValueError(f"双色球官方记录期号非法：{issue}")
    raw_red = str(raw.get("red", "")).strip().split(",")
    if len(raw_red) != SSQ_RULE.red_count or any(
        len(value) != 2 or not value.isdigit() for value in raw_red
    ):
        raise ValueError("双色球官方记录红球必须是六个逗号分隔两位数")
    raw_blue = str(raw.get("blue", "")).strip()
    if len(raw_blue) != 2 or not raw_blue.isdigit():
        raise ValueError("双色球官方记录蓝球必须是一个两位数")
    red, blue = SSQ_RULE.validate_draw(
        tuple(int(value) for value in raw_red), int(raw_blue)
    )
    return SSQDraw(
        issue=issue,
        draw_date=_normalized_date(raw.get("date", "")),
        red=red,
        blue=blue,
        source_url=source_url,
        raw_hash=canonical_raw_hash(source_url, raw),
        raw=dict(raw),
    )


def _merge_draw(draws: dict[str, SSQDraw], draw: SSQDraw) -> None:
    previous = draws.get(draw.issue)
    if previous is not None and (
        previous.draw_date != draw.draw_date
        or previous.red != draw.red
        or previous.blue != draw.blue
    ):
        raise ValueError(f"双色球官方数据同一期号发生冲突：{draw.issue}")
    if previous is None:
        draws[draw.issue] = draw


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
                raise ValueError("双色球官方接口顶层 JSON 不是对象")
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
    raise RuntimeError(f"双色球官方接口请求失败：{last_error}") from last_error


def fetch_ssq_history(
    periods: int = 0,
    *,
    request_json: Callable[[str], Mapping[str, object]] | None = None,
) -> list[SSQDraw]:
    """安全分页抓取官方历史；periods=0 表示接口全部历史。"""

    if periods < 0:
        raise ValueError("双色球抓取期数不得为负数")
    request = request_json or _default_request_json
    draws: dict[str, SSQDraw] = {}
    declared_total: int | None = None
    expected_pages: int | None = None
    for page_no in range(1, SSQ_MAX_PAGES + 1):
        source_url = build_ssq_source_url(page_no)
        payload = request(source_url)
        if payload.get("message") != "查询成功":
            raise RuntimeError(
                f"双色球官方接口返回失败：{payload.get('message', '未知错误')}"
            )
        raw_total = payload.get("total")
        if isinstance(raw_total, bool) or not str(raw_total).isdigit():
            raise ValueError("双色球官方接口 total 非法")
        total = int(str(raw_total))
        if total < 0 or total > SSQ_MAX_TOTAL:
            raise ValueError("双色球官方接口 total 越过安全范围")
        if declared_total is None:
            declared_total = total
            expected_pages = math.ceil(total / SSQ_PAGE_SIZE) if total else 0
        elif total != declared_total:
            raise ValueError("双色球官方接口分页 total 不一致")
        raw_results = payload.get("result")
        if not isinstance(raw_results, list):
            raise ValueError("双色球官方接口 result 不是列表")
        if len(raw_results) > SSQ_PAGE_SIZE:
            raise ValueError("双色球官方接口单页记录数超过固定 pageSize")
        before = len(draws)
        for raw in raw_results:
            if not isinstance(raw, dict):
                raise ValueError("双色球官方接口 result 包含非对象记录")
            _merge_draw(draws, validate_official_record(raw, source_url))
        if periods and len(draws) >= periods:
            break
        if expected_pages is None:
            raise RuntimeError("双色球官方接口未初始化分页总数")
        if expected_pages == 0 or page_no >= expected_pages:
            break
        if not raw_results or len(draws) == before:
            raise RuntimeError("双色球官方接口分页提前为空或没有新增期号")
    else:
        raise RuntimeError("双色球官方接口分页超过安全上限")
    if declared_total is None:
        raise RuntimeError("双色球官方接口未返回任何分页")
    if periods and len(draws) < periods:
        raise ValueError(
            f"双色球官方数据不足：请求 {periods} 期，仅获得 {len(draws)} 期"
        )
    if not periods and len(draws) != declared_total:
        raise ValueError(
            f"双色球官方数据总数不一致：声明 {declared_total}，去重后 {len(draws)}"
        )
    ordered = sorted(draws.values(), key=lambda draw: int(draw.issue), reverse=True)
    return ordered[:periods] if periods else ordered


def evidence_payload(draw: SSQDraw, fetched_at: str) -> dict[str, object]:
    """生成保留官方原始记录的追加型证据行。"""

    return {
        "schemaVersion": 1,
        "lottery": "ssq",
        "sourceUrl": draw.source_url,
        "fetchedAt": fetched_at,
        "rawHash": draw.raw_hash,
        "raw": draw.raw,
    }


def append_raw_evidence(
    path: str | Path,
    draws: Sequence[SSQDraw],
    *,
    fetched_at: str | None = None,
) -> int:
    """以文件锁和 fsync 方式追加 JSONL，绝不改写既有字节。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = fetched_at or datetime.now(timezone.utc).isoformat()
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    lines = [
        json.dumps(
            evidence_payload(draw, timestamp),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for draw in draws
    ]
    with output.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell():
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    raise ValueError("既有双色球 JSONL 末尾缺少 LF，拒绝追加")
            stream.seek(0, os.SEEK_END)
            for line in lines:
                stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return len(lines)


def read_raw_evidence(path: str | Path) -> list[SSQDraw]:
    """读取并验证所有证据行，包括来源、时间与原始哈希。"""

    evidence_path = Path(path)
    draws: list[SSQDraw] = []
    with evidence_path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"双色球 JSONL 第 {line_number} 行缺少 LF")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"双色球 JSONL 第 {line_number} 行不是合法 JSON"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(f"双色球 JSONL 第 {line_number} 行不是对象")
            if payload.get("schemaVersion") != 1 or payload.get("lottery") != "ssq":
                raise ValueError(f"双色球 JSONL 第 {line_number} 行身份非法")
            source_url = payload.get("sourceUrl")
            raw = payload.get("raw")
            fetched_at = payload.get("fetchedAt")
            raw_hash = payload.get("rawHash")
            if not isinstance(source_url, str) or not isinstance(raw, dict):
                raise ValueError(f"双色球 JSONL 第 {line_number} 行字段类型非法")
            if not isinstance(fetched_at, str):
                raise ValueError(f"双色球 JSONL 第 {line_number} 行缺少抓取时间")
            try:
                datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(
                    f"双色球 JSONL 第 {line_number} 行抓取时间非法"
                ) from error
            expected_hash = canonical_raw_hash(source_url, raw)
            if raw_hash != expected_hash:
                raise ValueError(f"双色球 JSONL 第 {line_number} 行原始哈希不匹配")
            draws.append(validate_official_record(raw, source_url))
    return draws


def reconcile_evidence(draws: Iterable[SSQDraw]) -> list[SSQDraw]:
    """单一官方源语义去重；任何同期开奖冲突都会整体失败。"""

    by_issue: dict[str, SSQDraw] = {}
    for draw in draws:
        validate_ssq_source_url(draw.source_url)
        _merge_draw(by_issue, draw)
    if not by_issue:
        raise ValueError("双色球原始证据为空")
    return sorted(by_issue.values(), key=lambda draw: int(draw.issue), reverse=True)


def write_official_history_csv(path: str | Path, draws: Sequence[SSQDraw]) -> None:
    """校验完成后原子写入 LF 规范 CSV。"""

    if not draws:
        raise ValueError("双色球规范 CSV 不接受空历史")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=SSQ_CSV_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for draw in draws:
                writer.writerow(
                    {
                        "issue": draw.issue,
                        "date": draw.draw_date,
                        "red": SSQ_RULE.format_red(draw.red),
                        "blue": SSQ_RULE.format_blue(draw.blue),
                        "source_url": draw.source_url,
                        "raw_hash": draw.raw_hash,
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def reconcile_raw_jsonl(raw_path: str | Path, csv_path: str | Path) -> list[SSQDraw]:
    """完整验证 JSONL 后才生成官方历史 CSV。"""

    reconciled = reconcile_evidence(read_raw_evidence(raw_path))
    write_official_history_csv(csv_path, reconciled)
    return reconciled


def load_official_history_csv(path: str | Path) -> list[SSQDraw]:
    """加载并再次校验规范 CSV，按期号升序返回。"""

    csv_path = Path(path)
    draws: dict[str, SSQDraw] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(SSQ_CSV_FIELDS):
            raise ValueError("双色球 CSV 表头不符合固定协议")
        for row_number, row in enumerate(reader, start=2):
            source_url = row["source_url"]
            validate_ssq_source_url(source_url)
            issue = row["issue"].strip()
            if not issue.isdigit():
                raise ValueError(f"双色球 CSV 第 {row_number} 行期号非法")
            red_text = row["red"].split(" ")
            if len(red_text) != 6 or any(
                len(value) != 2 or not value.isdigit() for value in red_text
            ):
                raise ValueError(f"双色球 CSV 第 {row_number} 行红球格式非法")
            blue_text = row["blue"].strip()
            if len(blue_text) != 2 or not blue_text.isdigit():
                raise ValueError(f"双色球 CSV 第 {row_number} 行蓝球格式非法")
            red, blue = SSQ_RULE.validate_draw(
                tuple(int(value) for value in red_text), int(blue_text)
            )
            draw = SSQDraw(
                issue=issue,
                draw_date=_normalized_date(row["date"]),
                red=red,
                blue=blue,
                source_url=source_url,
                raw_hash=row["raw_hash"].strip(),
                raw={},
            )
            if len(draw.raw_hash) != 64 or any(
                character not in "0123456789abcdef" for character in draw.raw_hash
            ):
                raise ValueError(f"双色球 CSV 第 {row_number} 行原始哈希非法")
            _merge_draw(draws, draw)
    return sorted(draws.values(), key=lambda draw: int(draw.issue))


__all__ = [
    "SSQ_API_URL",
    "SSQ_CSV_FIELDS",
    "SSQDraw",
    "append_raw_evidence",
    "build_ssq_source_url",
    "canonical_raw_hash",
    "fetch_ssq_history",
    "load_official_history_csv",
    "read_raw_evidence",
    "reconcile_raw_jsonl",
    "validate_official_record",
    "validate_ssq_source_url",
    "write_official_history_csv",
]

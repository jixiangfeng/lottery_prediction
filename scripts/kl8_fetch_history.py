#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仅从福彩官网白名单接口显式抓取快乐8历史。"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.kl8_pick5_probability_v1 import (  # noqa: E402
    _parse_date,
    _parse_issue,
    _parse_numbers,
)

KL8_API = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
OFFICIAL_HOSTS = frozenset({"www.cwl.gov.cn", "cwl.gov.cn"})


def _fetch_page(
    page: int,
    page_size: int,
    timeout: float,
    retries: int,
    *,
    day_start: str = "",
    day_end: str = "",
) -> tuple[list[dict[str, object]], int]:
    query = urllib.parse.urlencode(
        {
            "name": "kl8",
            "issueCount": "",
            "issueStart": "",
            "issueEnd": "",
            "dayStart": day_start,
            "dayEnd": day_end,
            "pageNo": page,
            "pageSize": page_size,
            "week": "",
            "systemType": "PC",
        }
    )
    url = f"{KL8_API}?{query}"
    if urllib.parse.urlparse(url).hostname not in OFFICIAL_HOSTS:
        raise ValueError("快乐8抓取地址不在福彩官网白名单")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "lottery-prediction/kl8-v1",
            "Referer": "https://www.cwl.gov.cn/",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                final_url = urllib.parse.urlparse(response.geturl())
                if final_url.scheme != "https":
                    raise ValueError("福彩接口最终响应URL必须使用HTTPS")
                if final_url.hostname not in OFFICIAL_HOSTS:
                    raise ValueError(
                        f"福彩接口重定向到非白名单域名：{final_url.hostname}"
                    )
                payload = json.loads(response.read().decode("utf-8"))
            rows = payload.get("result", [])
            if not isinstance(rows, list):
                raise ValueError("福彩接口result不是数组")
            parsed = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("福彩接口开奖记录格式错误")
                parsed.append(
                    {
                        "issue": _parse_issue(row.get("code")),
                        "date": _parse_date(
                            str(row.get("date", "")).split("(", 1)[0].strip()
                        ),
                        "numbers": _parse_numbers(row.get("red", "")),
                        "source": url,
                    }
                )
            advertised = payload.get("count")
            return parsed, int(advertised) if advertised is not None else len(parsed)
        except (
            OSError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as error:
            if isinstance(error, ValueError) and (
                "非白名单" in str(error) or "HTTPS" in str(error)
            ):
                raise
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"福彩快乐8接口请求失败：{last_error}") from last_error


def _fetch_archive_page(
    year: int, page: int, page_size: int, timeout: float, retries: int
) -> list[dict[str, object]]:
    """按自然年窗口读取一页官方历史；年度查询不依赖接口全局1000期上限。"""

    rows, _ = _fetch_page(
        page,
        page_size,
        timeout,
        retries,
        day_start=f"{year}-01-01",
        day_end=f"{year}-12-31",
    )
    return rows


def _merge_unique_row(
    rows: dict[str, dict[str, object]], row: dict[str, object]
) -> bool:
    issue = str(row["issue"])
    existing = rows.get(issue)
    if existing is not None:
        semantic_keys = ("issue", "date", "numbers")
        if any(existing[key] != row[key] for key in semantic_keys):
            raise ValueError(f"福彩接口同一期号内容冲突：{issue}")
        return False
    rows[issue] = row
    return True


def fetch_history(
    periods: int, timeout: float, retries: int
) -> list[dict[str, object]]:
    """分页抓取并按期号升序返回去重开奖记录。"""

    if periods < 0:
        raise ValueError("periods必须大于等于0")
    if timeout <= 0:
        raise ValueError("timeout必须大于0")
    if retries < 0:
        raise ValueError("retries必须大于等于0")
    page_size = min(1000, periods or 1000)
    page = 1
    rows: dict[str, dict[str, object]] = {}
    advertised_total: int | None = None
    target: int | None = None
    while target is None or len(rows) < target:
        batch, available = _fetch_page(page, page_size, timeout, retries)
        if available < 0:
            raise ValueError("福彩接口可用总期数不得为负")
        if advertised_total is None:
            advertised_total = available
            if periods > 0:
                if available < periods:
                    raise RuntimeError(
                        f"请求{periods}期，但福彩接口仅宣告{available}期"
                    )
                target = periods
            else:
                target = available
        elif available != advertised_total:
            raise RuntimeError(
                f"福彩接口分页宣告总期数变化：{advertised_total}->{available}"
            )
        before = len(rows)
        for row in batch:
            _merge_unique_row(rows, row)
        if len(rows) == before and len(rows) < target:
            expectation = f"请求{periods}期" if periods > 0 else f"接口宣告{target}期"
            raise RuntimeError(
                f"福彩接口第{page}页新增0期，未达到{expectation}的完整数量"
            )
        page += 1
    ordered = sorted(rows.values(), key=lambda row: int(str(row["issue"])))
    selected = ordered[-periods:] if periods else ordered
    if len(selected) != target:
        expectation = f"请求{periods}期" if periods else f"接口宣告{target}期"
        raise RuntimeError(f"抓取结果不完整：{expectation}，实际{len(selected)}期")
    return selected


def fetch_full_history(
    *,
    start_year: int = 2020,
    end_year: int | None = None,
    timeout: float,
    retries: int,
    page_size: int = 1000,
) -> list[dict[str, object]]:
    """按年度日期窗口抓取福彩官网可提供的全部快乐8历史。"""

    actual_end_year = end_year or datetime.now(timezone.utc).year
    if start_year < 2020 or actual_end_year < start_year:
        raise ValueError("全量历史年份范围无效")
    if timeout <= 0 or retries < 0 or not 1 <= page_size <= 1000:
        raise ValueError("全量历史timeout/retries/page_size无效")
    rows: dict[str, dict[str, object]] = {}
    for year in range(start_year, actual_end_year + 1):
        page = 1
        year_count = 0
        while True:
            batch = _fetch_archive_page(year, page, page_size, timeout, retries)
            if not batch:
                break
            before = len(rows)
            for row in batch:
                issue = str(row["issue"])
                row_date = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
                if not issue.startswith(str(year)) or row_date.year != year:
                    raise ValueError(f"福彩接口记录越过年度窗口：{issue}/{row_date}")
                _merge_unique_row(rows, row)
            added = len(rows) - before
            year_count += added
            if added == 0:
                raise RuntimeError(f"福彩接口{year}年第{page}页没有新增记录")
            page += 1
            if page > 1000:
                raise RuntimeError(f"福彩接口{year}年分页超过安全上限")
        if year_count == 0:
            raise RuntimeError(f"福彩接口{year}年度窗口没有返回快乐8记录")
    return sorted(rows.values(), key=lambda row: int(str(row["issue"])))


def _append_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell():
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise ValueError("既有非空JSONL末尾必须为换行，拒绝追加")
            handle.seek(0, os.SEEK_END)
            for row in rows:
                record = {**row, "lottery": "kl8", "fetchedAt": fetched_at}
                line = (
                    json.dumps(
                        record, ensure_ascii=False, sort_keys=True, allow_nan=False
                    )
                    + "\n"
                )
                handle.write(line.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_canonical_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["issue", "date", "numbers"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "issue": row["issue"],
                        "date": row["date"],
                        "numbers": " ".join(str(number) for number in row["numbers"]),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        content = temporary.read_bytes()
        if path.exists():
            if path.read_bytes() == content:
                if stat.S_IMODE(path.stat().st_mode) & 0o222:
                    raise ValueError(f"规范CSV已存在但不是只读文件：{path}")
                return
            raise FileExistsError(f"规范CSV已存在且内容不同，禁止覆盖：{path}")
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从固定福彩官网接口抓取快乐8历史")
    parser.add_argument(
        "--periods", type=int, default=0, help="抓取期数；0表示接口全部历史"
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="按2020年至今的年度日期窗口抓取官网全量历史",
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-jsonl", help="只追加原始JSONL证据")
    output.add_argument("--output-csv", help="只写一次规范CSV；不同内容拒绝覆盖")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        rows = (
            fetch_full_history(timeout=args.timeout, retries=args.retries)
            if args.full_history
            else fetch_history(args.periods, args.timeout, args.retries)
        )
        if args.output_jsonl:
            _append_jsonl(Path(args.output_jsonl), rows)
        else:
            _write_canonical_csv(Path(args.output_csv), rows)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"抓取失败：{error}", file=sys.stderr)
        print("建议：检查网络后重试；程序不会改写既有规范CSV。", file=sys.stderr)
        return 1
    print(f"已抓取并校验{len(rows)}期快乐8开奖；来源仅限cwl.gov.cn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

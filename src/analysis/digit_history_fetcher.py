# -*- coding: utf-8 -*-
"""从固定官方白名单接口显式抓取数字彩历史开奖。"""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FC3D_API_URL = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
PL3_API_URL = "https://jc.zhcw.com/port/client_json.php"
OFFICIAL_HOSTS = frozenset({"www.cwl.gov.cn", "jc.zhcw.com"})
_PAGE_SIZE = 1000
_USER_AGENT = "Mozilla/5.0 (compatible; lottery-prediction/1.0; +offline-research)"


@dataclass(frozen=True)
class DigitHistoryDraw:
    """一条经过基本格式校验的官方数字彩开奖。"""

    issue: str
    numbers: tuple[int, ...]
    draw_date: str
    source: str

    @property
    def number_text(self) -> str:
        return "".join(str(number) for number in self.numbers)


def _fetch_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    """访问固定白名单 URL，并以有限重试读取 JSON。"""

    host = urllib.parse.urlparse(url).hostname
    if host not in OFFICIAL_HOSTS:
        raise ValueError(f"数据源域名不在白名单：{host}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, **headers},
        method="GET",
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                final_host = urllib.parse.urlparse(response.geturl()).hostname
                if final_host not in OFFICIAL_HOSTS:
                    raise ValueError(f"官方接口重定向到非白名单域名：{final_host}")
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("官方接口返回的 JSON 顶层不是对象")
            return payload
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"官方接口请求失败：{last_error}") from last_error


def _validated_draw(
    *,
    issue: Any,
    raw_numbers: list[str],
    draw_date: Any,
    source: str,
) -> DigitHistoryDraw:
    issue_text = str(issue).strip()
    if not issue_text.isdigit():
        raise ValueError(f"官方数据包含非法期号：{issue}")
    digits = [str(value).strip() for value in raw_numbers]
    if len(digits) != 3 or any(
        len(value) != 1 or not value.isdigit() for value in digits
    ):
        raise ValueError(f"官方数据包含非法三位号码：{raw_numbers}")
    return DigitHistoryDraw(
        issue=issue_text,
        numbers=tuple(int(value) for value in digits),
        draw_date=str(draw_date).split("(", 1)[0].strip(),
        source=source,
    )


def _deduplicate_and_limit(
    draws: list[DigitHistoryDraw], periods: int
) -> list[DigitHistoryDraw]:
    by_issue: dict[str, DigitHistoryDraw] = {}
    for draw in draws:
        previous = by_issue.get(draw.issue)
        if previous is not None and previous.numbers != draw.numbers:
            raise ValueError(f"官方数据同一期号号码冲突：{draw.issue}")
        by_issue[draw.issue] = draw
    ordered = sorted(by_issue.values(), key=lambda item: int(item.issue), reverse=True)
    if len(ordered) < periods:
        raise ValueError(f"官方数据不足：请求 {periods} 期，只获得 {len(ordered)} 期")
    return ordered[:periods]


def _fetch_fc3d(periods: int, timeout: float, retries: int) -> list[DigitHistoryDraw]:
    draws: list[DigitHistoryDraw] = []
    total = periods
    page_no = 1
    while len(draws) < periods and (page_no - 1) * _PAGE_SIZE < total:
        query = urllib.parse.urlencode(
            {
                "name": "3d",
                "issueCount": "",
                "issueStart": "",
                "issueEnd": "",
                "dayStart": "",
                "dayEnd": "",
                "pageNo": page_no,
                "pageSize": _PAGE_SIZE,
                "week": "",
                "systemType": "PC",
            }
        )
        payload = _fetch_json(
            f"{FC3D_API_URL}?{query}",
            headers={"Referer": "https://www.cwl.gov.cn/"},
            timeout=timeout,
            retries=retries,
        )
        if int(payload.get("state", -1)) != 0:
            raise RuntimeError(
                f"福彩官网返回失败：{payload.get('message', '未知错误')}"
            )
        total = int(payload.get("total", 0))
        rows = payload.get("result", [])
        if not isinstance(rows, list) or not rows:
            break
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("福彩官网开奖记录格式错误")
            draws.append(
                _validated_draw(
                    issue=row.get("code"),
                    raw_numbers=str(row.get("red", "")).split(","),
                    draw_date=row.get("date", ""),
                    source=FC3D_API_URL,
                )
            )
        page_no += 1
    return _deduplicate_and_limit(draws, periods)


def _fetch_pl3(periods: int, timeout: float, retries: int) -> list[DigitHistoryDraw]:
    draws: list[DigitHistoryDraw] = []
    total = periods
    page_no = 1
    while len(draws) < periods and (page_no - 1) * _PAGE_SIZE < total:
        query = urllib.parse.urlencode(
            {
                "transactionType": "10001001",
                "lotteryId": "283",
                "issueCount": periods,
                "startIssue": "",
                "endIssue": "",
                "startDate": "",
                "endDate": "",
                "type": "0",
                "pageNum": page_no,
                "pageSize": _PAGE_SIZE,
            }
        )
        url = f"{PL3_API_URL}?{query}"
        payload: dict[str, Any] = {}
        for response_attempt in range(retries + 1):
            payload = _fetch_json(
                url,
                headers={"Referer": "https://www.zhcw.com/"},
                timeout=timeout,
                retries=retries,
            )
            if payload.get("resCode") == "000000":
                break
            if response_attempt < retries:
                time.sleep(min(2**response_attempt, 4))
        if payload.get("resCode") != "000000":
            raise RuntimeError(f"中彩网返回失败：{payload.get('message', '未知错误')}")
        total = int(payload.get("total", 0))
        rows = payload.get("data", [])
        if not isinstance(rows, list) or not rows:
            break
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("中彩网排列三开奖记录格式错误")
            draws.append(
                _validated_draw(
                    issue=row.get("issue"),
                    raw_numbers=str(row.get("frontWinningNum", "")).split(),
                    draw_date=row.get("openTime", ""),
                    source=PL3_API_URL,
                )
            )
        page_no += 1
    return _deduplicate_and_limit(draws, periods)


def fetch_digit_history(
    lottery: str,
    *,
    periods: int = 1000,
    timeout: float = 20.0,
    retries: int = 3,
) -> list[DigitHistoryDraw]:
    """从官方接口抓取福彩3D或排列三历史。

    示例：``fetch_digit_history("fc3d", periods=1000)``。抓取仅在调用本函数时
    发生，不会由日报或回测流程隐式触发。
    """

    if lottery not in {"fc3d", "pl3"}:
        raise ValueError("官方历史抓取目前只支持 fc3d 或 pl3")
    if periods <= 0 or timeout <= 0 or retries < 0:
        raise ValueError("periods、timeout 必须为正数，retries 不得为负数")
    if lottery == "fc3d":
        return _fetch_fc3d(periods, timeout, retries)
    return _fetch_pl3(periods, timeout, retries)


def write_digit_history_csv(
    draws: list[DigitHistoryDraw], output_path: str | Path
) -> Path:
    """以原子替换方式写入可被数字彩加载器读取的 UTF-8 CSV。"""

    if not draws:
        raise ValueError("至少需要一条开奖数据")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("期数", "开奖号码", "开奖日期", "数据来源")
            )
            writer.writeheader()
            for draw in draws:
                writer.writerow(
                    {
                        "期数": draw.issue,
                        "开奖号码": draw.number_text,
                        "开奖日期": draw.draw_date,
                        "数据来源": draw.source,
                    }
                )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


__all__ = [
    "DigitHistoryDraw",
    "fetch_digit_history",
    "write_digit_history_csv",
]

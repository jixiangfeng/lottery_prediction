# -*- coding: utf-8 -*-
"""
快乐 8 历史数据抓取与加载工具（精简版）。

相较于原仓库，该版本仅保留快乐 8（kl8）相关逻辑，负责：
1. 带重试的 HTTP 抓取；
2. HTML / 文本解析为 pandas.DataFrame；
3. 将数据保存到 `data/kl8/data.csv` 并生成下载元信息。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from .config import (
    ALLOWED_DOMAINS,
    DATA_FILE_NAME,
    LOTTERY_CONFIGS,
    NETWORK_CONFIG,
    PATHS,
    LotteryModelConfig,
    ensure_runtime_directories,
)


@dataclass
class DownloadResult:
    """记录一次快乐 8 历史数据下载的结果。"""

    code: str
    total_issues: int
    saved_path: str
    timestamp: str
    latest_issue: str = ""
    status: str = "ok"
    mode: str = "full"
    used_cache: bool = False
    updated: bool = True
    source: str = "cwl.gov.cn"
    message: str = ""


class LotteryHttpClient:
    """封装带重试和域名白名单校验的 HTTP 客户端。"""

    def __init__(
        self,
        timeout: float,
        retries: int,
        backoff_factor: float,
        user_agent: str,
    ) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "https://www.cwl.gov.cn/",
        }

    def get_text(self, url: str) -> str:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if all(allowed not in domain for allowed in ALLOWED_DOMAINS):
            raise ValueError(f"禁止访问域名：{domain}")
        response = self._session.get(url, headers=self._headers, timeout=self._timeout)
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text


def _build_history_url(config: LotteryModelConfig, start: Optional[int], end: Optional[int]) -> str:
    """构造快乐 8 官方历史开奖接口地址。

    500 彩票历史页面当前经常返回非标准 567 状态，因此默认改用中国福彩网公开接口。
    pageSize 设得足够大，下载后再按期号区间过滤。
    """

    query = {
        "name": config.code,
        "issueCount": "",
        "issueStart": start or "",
        "issueEnd": end or "",
        "dayStart": "",
        "dayEnd": "",
        "pageNo": 1,
        "pageSize": 5000,
        "week": "",
        "systemType": "PC",
    }
    return "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?" + urlencode(query)


def _parse_issue_list(config: LotteryModelConfig, html: str) -> pd.DataFrame:
    """解析快乐 8 历史页面，返回包含 20 个球位的 DataFrame。"""

    if html.lstrip().startswith("{"):
        return _parse_cwl_issue_list(config, html)

    soup = BeautifulSoup(html, "lxml")
    tbody = soup.find("tbody", attrs={"id": "tdata"})
    if not tbody:
        raise ValueError("未找到开奖号码数据表格 (id=tdata)")

    rows = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        issue = tds[0].get_text(strip=True)
        if not issue or not issue.isdigit():
            continue
        numbers = [
            td.get_text(strip=True)
            for td in tds
            if td.get_text(strip=True).isdigit()
        ]
        if len(numbers) < config.red.sequence_len:
            continue
        record = {"期数": issue}
        for idx, value in enumerate(numbers[: config.red.sequence_len]):
            record[f"红球_{idx + 1}"] = value
        rows.append(record)

    if not rows:
        raise ValueError("解析开奖号码失败，未获取到有效数据")
    df = pd.DataFrame(rows)
    df.sort_values("期数", ascending=False, inplace=True)
    return df.reset_index(drop=True)

def _parse_cwl_issue_list(config: LotteryModelConfig, text: str) -> pd.DataFrame:
    """解析中国福彩网快乐 8 JSON 接口响应。"""

    payload = json.loads(text)
    rows = []
    for item in payload.get("result", []):
        issue = str(item.get("code", "")).strip()
        red = str(item.get("red", "")).strip()
        if not issue or not red:
            continue
        numbers = [part.strip() for part in red.split(",") if part.strip()]
        if len(numbers) != config.red.sequence_len:
            continue
        record = {"期数": issue}
        for idx, value in enumerate(numbers[: config.red.sequence_len]):
            record[f"红球_{idx + 1}"] = f"{int(value):02d}"
        rows.append(record)
    if not rows:
        raise ValueError("解析中国福彩网快乐 8 数据失败，未获取到有效数据")
    df = pd.DataFrame(rows)
    df.sort_values("期数", ascending=False, inplace=True)
    return df.reset_index(drop=True)


def _parse_kl8_sequence(text: str) -> pd.DataFrame:
    """解析 917500 顺序文本为 DataFrame。"""

    rows = []
    for line in sorted(text.splitlines(), reverse=True):
        if not line or "," not in line:
            continue
        first_segment = line.split(",")[0]
        parts = [item for item in first_segment.split(" ") if item]
        if len(parts) < 21:
            continue
        issue = parts[0]
        record = {"期数": issue}
        for idx in range(1, 21):
            record[f"红球_{idx}"] = parts[idx]
        rows.append(record)
    if not rows:
        raise ValueError("快乐 8 出球顺序数据解析失败")
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


def get_current_issue(code: str, client: Optional[LotteryHttpClient] = None) -> str:
    """查询快乐 8 最新期号。"""

    cfg = LOTTERY_CONFIGS[code]
    client = client or LotteryHttpClient(
        timeout=NETWORK_CONFIG["timeout"],
        retries=NETWORK_CONFIG["retry_count"],
        backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
        user_agent=NETWORK_CONFIG["user_agent"],
    )

    url = _build_history_url(cfg, None, None)
    html = client.get_text(url)
    df = _parse_issue_list(cfg, html)
    value = str(df.iloc[0]["期数"])
    logger.info("【{}】最新期号：{}", cfg.name, value)
    return value


def download_history(
    code: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    use_sequence_order: bool = False,
    client: Optional[LotteryHttpClient] = None,
) -> DownloadResult:
    """下载快乐 8 历史数据并保存到 CSV。"""

    ensure_runtime_directories()
    cfg = LOTTERY_CONFIGS[code]
    client = client or LotteryHttpClient(
        timeout=NETWORK_CONFIG["timeout"],
        retries=NETWORK_CONFIG["retry_count"],
        backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
        user_agent=NETWORK_CONFIG["user_agent"],
    )

    if use_sequence_order:
        logger.info("下载快乐 8 出球顺序数据...")
        text = client.get_text("https://data.917500.cn/kl81000_cq_asc.txt")
        df = _parse_kl8_sequence(text)
    else:
        url = _build_history_url(cfg, start, end)
        logger.info("下载快乐 8 历史数据：{}", url)
        html = client.get_text(url)
        df = _parse_issue_list(cfg, html)

    save_dir = PATHS["data"] / cfg.code
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / DATA_FILE_NAME
    previous_latest = ""
    if output_path.exists():
        try:
            previous_df = pd.read_csv(output_path, encoding="utf-8")
            previous_latest = str(previous_df.iloc[0]["期数"]) if not previous_df.empty and "期数" in previous_df.columns else ""
        except Exception:
            previous_latest = ""
    latest_issue = str(df.iloc[0]["期数"])
    df.to_csv(output_path, index=False, encoding="utf-8")
    meta = DownloadResult(
        code=cfg.code,
        total_issues=len(df),
        saved_path=str(output_path),
        timestamp=datetime.now(timezone.utc).isoformat(),
        latest_issue=latest_issue,
        status="ok",
        mode="full" if previous_latest != latest_issue else "no_change",
        used_cache=False,
        updated=previous_latest != latest_issue,
        source="917500.cn" if use_sequence_order else "cwl.gov.cn",
    )
    (output_path.parent / "download_meta.json").write_text(
        json.dumps(meta.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.success("数据下载完成，共 {} 期，保存到 {}", meta.total_issues, output_path)
    return meta


def load_history(code: str) -> pd.DataFrame:
    """从本地 CSV 加载快乐 8 历史数据。"""

    cfg = LOTTERY_CONFIGS[code]
    path = PATHS["data"] / cfg.code / DATA_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"未找到 {cfg.name} 历史数据文件，请先执行下载：{path}")
    df = pd.read_csv(path, encoding="utf-8")
    if "期数" not in df.columns:
        raise ValueError(f"{path} 缺少【期数】字段，可能是损坏文件")
    return df


__all__ = [
    "DownloadResult",
    "LotteryHttpClient",
    "download_history",
    "get_current_issue",
    "load_history",
]

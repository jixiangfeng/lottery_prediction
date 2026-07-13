# -*- coding: utf-8 -*-
"""数据源状态报告。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_data_source_status(meta: dict[str, Any] | None) -> dict[str, Any]:
    """把下载元信息转换为前端友好的 dataSource 结构。"""

    meta = meta or {}
    return {
        "status": meta.get("status", "unknown"),
        "mode": meta.get("mode", "unknown"),
        "latestIssue": str(meta.get("latest_issue") or meta.get("latestIssue") or ""),
        "totalIssues": int(meta.get("total_issues") or meta.get("totalIssues") or 0),
        "usedCache": bool(meta.get("used_cache") if "used_cache" in meta else meta.get("usedCache", False)),
        "updated": bool(meta.get("updated", False)),
        "source": meta.get("source", "unknown"),
        "timestamp": meta.get("timestamp") or meta.get("lastDownloadAt"),
        "message": meta.get("message", ""),
    }


def read_download_meta(path: Path | str) -> dict[str, Any] | None:
    """读取 download_meta.json；不存在或损坏时返回 None。"""

    meta_path = Path(path)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_data_source_status(status: dict[str, Any], output_dir: Path | str) -> Path:
    """写入 reports/data_source.json。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "data_source.json"
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

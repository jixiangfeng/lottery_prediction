# -*- coding: utf-8 -*-

import json
from pathlib import Path

from src.analysis.data_source_status import build_data_source_status, write_data_source_status


def test_build_data_source_status_marks_cache_when_meta_says_cache():
    meta = {
        "code": "kl8",
        "total_issues": 2002,
        "latest_issue": "2026181",
        "status": "cache",
        "mode": "cache_fallback",
        "used_cache": True,
        "updated": False,
        "source": "local",
    }

    status = build_data_source_status(meta)

    assert status["status"] == "cache"
    assert status["usedCache"] is True
    assert status["latestIssue"] == "2026181"


def test_write_data_source_status_outputs_json(tmp_path):
    path = write_data_source_status(
        {
            "status": "ok",
            "mode": "full",
            "latestIssue": "2026181",
            "totalIssues": 2002,
            "usedCache": False,
            "updated": True,
            "source": "cwl.gov.cn",
        },
        tmp_path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["source"] == "cwl.gov.cn"

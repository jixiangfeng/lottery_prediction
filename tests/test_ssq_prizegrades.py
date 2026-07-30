# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from src.analysis.ssq_prizegrades import (
    append_prizegrade_evidence,
    build_ssq_prizegrade_source_url,
    fetch_ssq_prizegrades,
    read_prizegrade_evidence,
    reconcile_prizegrade_evidence,
    validate_prizegrade_record,
)


def _official_row(
    issue: int = 2026086, *, prizegrades: object | None = None
) -> dict[str, object]:
    return {
        "name": "双色球",
        "code": str(issue),
        "prizegrades": (
            prizegrades
            if prizegrades is not None
            else [
                {"type": grade, "typenum": str(grade * 10), "typemoney": str(grade)}
                for grade in range(1, 7)
            ]
        ),
    }


def test_validates_exactly_six_grades_and_normalizes_integer_amounts():
    record = validate_prizegrade_record(
        _official_row(), build_ssq_prizegrade_source_url(1)
    )

    assert record.issue == "2026086"
    assert [
        (grade.grade, grade.winners, grade.amount) for grade in record.prizegrades
    ] == [
        (1, 10, 1),
        (2, 20, 2),
        (3, 30, 3),
        (4, 40, 4),
        (5, 50, 5),
        (6, 60, 6),
    ]
    assert len(record.raw_hash) == 64


@pytest.mark.parametrize(
    ("prizegrades", "message"),
    [
        ([{"type": 1, "typenum": "1", "typemoney": "1"}], "奖级数量"),
        (
            [
                {"type": grade, "typenum": "1", "typemoney": "1"}
                for grade in (1, 2, 3, 4, 5, 5)
            ],
            "奖级1-6",
        ),
        (
            [
                {"type": grade, "typenum": "-1", "typemoney": "1"}
                for grade in range(1, 7)
            ],
            "中奖注数",
        ),
        (
            [
                {"type": grade, "typenum": "1", "typemoney": "1.5"}
                for grade in range(1, 7)
            ],
            "奖金金额",
        ),
    ],
)
def test_rejects_missing_duplicate_or_non_integer_prize_fields(prizegrades, message):
    with pytest.raises(ValueError, match=message):
        validate_prizegrade_record(
            _official_row(prizegrades=prizegrades),
            build_ssq_prizegrade_source_url(1),
        )


def test_fetches_limited_pages_and_keeps_only_six_official_grades():
    calls: list[str] = []

    def request_json(url: str):
        calls.append(url)
        page = int(parse_qs(urlparse(url).query)["pageNo"][0])
        return {
            "message": "查询成功",
            "total": 2,
            "result": [
                _official_row(
                    2026087 - page,
                    prizegrades=_official_row()["prizegrades"]
                    + [{"type": 7, "typenum": "", "typemoney": ""}],
                )
            ],
        }

    records = fetch_ssq_prizegrades(start_page=1, pages=2, request_json=request_json)

    assert [record.issue for record in records] == ["2026086", "2026085"]
    assert len(calls) == 2


def test_append_only_cache_detects_hash_tampering_and_reconciles_latest_fetch(
    tmp_path: Path,
):
    source_url = build_ssq_prizegrade_source_url(1)
    initial = validate_prizegrade_record(_official_row(), source_url)
    changed_raw = _official_row(
        prizegrades=[
            {"type": grade, "typenum": str(grade * 11), "typemoney": str(grade)}
            for grade in range(1, 7)
        ]
    )
    refreshed = validate_prizegrade_record(changed_raw, source_url)
    cache = tmp_path / "prizegrades.jsonl"

    append_prizegrade_evidence(cache, [initial], fetched_at="2026-07-29T00:00:00+00:00")
    initial_bytes = cache.read_bytes()
    append_prizegrade_evidence(
        cache, [refreshed], fetched_at="2026-07-29T01:00:00+00:00"
    )
    assert cache.read_bytes().startswith(initial_bytes)
    reconciled = reconcile_prizegrade_evidence(read_prizegrade_evidence(cache))
    assert reconciled[0].prizegrades[0].winners == 11

    payload = json.loads(cache.read_text(encoding="utf-8").splitlines()[0])
    payload["raw"]["prizegrades"][0]["typenum"] = "999"
    cache.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="原始哈希不匹配"):
        read_prizegrade_evidence(cache)

# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from src.analysis.ssq_history import (
    SSQ_PAGE_SIZE,
    append_raw_evidence,
    build_ssq_source_url,
    fetch_ssq_history,
    load_official_history_csv,
    read_raw_evidence,
    reconcile_raw_jsonl,
    validate_official_record,
    validate_ssq_source_url,
)


def _official_row(
    issue: int,
    *,
    red: str = "01,02,03,04,05,06",
    blue: str = "07",
    date: str = "2026-01-01(四)",
    name: str = "双色球",
) -> dict[str, object]:
    return {
        "code": str(issue),
        "date": date,
        "red": red,
        "blue": blue,
        "name": name,
    }


def test_official_pagination_deduplicates_identical_records_safely():
    first_page = [_official_row(2026000 + index) for index in range(1, 101)]
    second_page = [
        _official_row(2026101),
        _official_row(2026102),
        _official_row(2026100),
    ]
    calls: list[str] = []

    def request_json(url: str):
        calls.append(url)
        page = int(parse_qs(urlparse(url).query)["pageNo"][0])
        return {
            "message": "查询成功",
            "total": 102,
            "result": first_page if page == 1 else second_page,
        }

    draws = fetch_ssq_history(request_json=request_json)

    assert len(draws) == 102
    assert draws[0].issue == "2026102"
    assert draws[-1].issue == "2026001"
    assert len(calls) == 2
    assert all(
        parse_qs(urlparse(url).query)["pageSize"] == [str(SSQ_PAGE_SIZE)]
        for url in calls
    )


def test_official_pagination_rejects_conflicts_and_inconsistent_total():
    def conflict_request(url: str):
        page = int(parse_qs(urlparse(url).query)["pageNo"][0])
        return {
            "message": "查询成功",
            "total": 101,
            "result": (
                [_official_row(2026000 + index) for index in range(1, 101)]
                if page == 1
                else [_official_row(2026100, blue="08")]
            ),
        }

    with pytest.raises(ValueError, match="发生冲突"):
        fetch_ssq_history(request_json=conflict_request)

    def changing_total(url: str):
        page = int(parse_qs(urlparse(url).query)["pageNo"][0])
        return {
            "message": "查询成功",
            "total": 101 if page == 1 else 102,
            "result": (
                [_official_row(2026000 + index) for index in range(1, 101)]
                if page == 1
                else [_official_row(2026101)]
            ),
        }

    with pytest.raises(ValueError, match="total 不一致"):
        fetch_ssq_history(request_json=changing_total)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": "其他玩法"}, "身份不是双色球"),
        ({"red": "01,02,03,04,05,05"}, "不得重复"),
        ({"red": "01,02,03,04,05,34"}, "01-33"),
        ({"red": "01,02,03,04,06,05"}, "严格升序"),
        ({"blue": "17"}, "01-16"),
        ({"date": "2026-02-30"}, "日期非法"),
        ({"code": "20A6001"}, "期号非法"),
    ],
)
def test_official_record_identity_number_and_date_validation(changes, message):
    row = _official_row(2026001)
    row.update(changes)

    with pytest.raises(ValueError, match=message):
        validate_official_record(row, build_ssq_source_url(1))


def test_source_allowlist_rejects_host_and_query_tampering():
    valid = build_ssq_source_url(1)
    validate_ssq_source_url(valid)

    with pytest.raises(ValueError, match="官方白名单"):
        validate_ssq_source_url(valid.replace("www.cwl.gov.cn", "example.com"))
    with pytest.raises(ValueError, match="固定参数非法"):
        validate_ssq_source_url(valid.replace("name=ssq", "name=3d"))
    with pytest.raises(ValueError, match="参数集合"):
        validate_ssq_source_url(f"{valid}&extra=1")


def test_append_only_evidence_reconciles_to_lf_csv(tmp_path: Path):
    source_url = build_ssq_source_url(1)
    first = validate_official_record(_official_row(2026001), source_url)
    second = validate_official_record(
        _official_row(
            2026002,
            red="02,03,04,05,06,07",
            blue="08",
            date="2026-01-04(日)",
        ),
        source_url,
    )
    raw_path = tmp_path / "history.jsonl"
    csv_path = tmp_path / "official_history.csv"

    assert (
        append_raw_evidence(
            raw_path,
            [first, second],
            fetched_at="2026-07-24T00:00:00+00:00",
        )
        == 2
    )
    initial_bytes = raw_path.read_bytes()
    append_raw_evidence(raw_path, [first], fetched_at="2026-07-24T01:00:00+00:00")
    assert raw_path.read_bytes().startswith(initial_bytes)

    reconciled = reconcile_raw_jsonl(raw_path, csv_path)

    assert [draw.issue for draw in reconciled] == ["2026002", "2026001"]
    assert b"\r\n" not in csv_path.read_bytes()
    assert csv_path.read_bytes().endswith(b"\n")
    loaded = load_official_history_csv(csv_path)
    assert [draw.issue for draw in loaded] == ["2026001", "2026002"]
    assert loaded[-1].red == (2, 3, 4, 5, 6, 7)


def test_evidence_hash_tampering_is_detected(tmp_path: Path):
    draw = validate_official_record(_official_row(2026001), build_ssq_source_url(1))
    raw_path = tmp_path / "history.jsonl"
    append_raw_evidence(raw_path, [draw], fetched_at="2026-07-24T00:00:00+00:00")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    payload["raw"]["blue"] = "08"
    raw_path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="原始哈希不匹配"):
        read_raw_evidence(raw_path)

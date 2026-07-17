# -*- coding: utf-8 -*-

import urllib.parse

import pytest

from src.analysis import digit_history_fetcher
from src.analysis.digit_history_fetcher import (
    DigitHistoryDraw,
    fetch_digit_history,
    write_digit_history_csv,
)


def test_fetch_fc3d_paginates_and_normalizes_official_fields(monkeypatch):
    def fake_fetch(url, **kwargs):
        page = int(urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["pageNo"][0])
        rows = (
            [
                {"code": "2026003", "red": "0,2,3", "date": "2026-01-03(六)"},
                {"code": "2026002", "red": "9,8,7", "date": "2026-01-02(五)"},
            ]
            if page == 1
            else [{"code": "2026001", "red": "0,0,1", "date": "2026-01-01(四)"}]
        )
        return {"state": 0, "total": 3, "result": rows}

    monkeypatch.setattr(digit_history_fetcher, "_PAGE_SIZE", 2)
    monkeypatch.setattr(digit_history_fetcher, "_fetch_json", fake_fetch)

    draws = fetch_digit_history("fc3d", periods=3)

    assert [draw.issue for draw in draws] == ["2026003", "2026002", "2026001"]
    assert draws[0].number_text == "023"
    assert draws[0].draw_date == "2026-01-03"


def test_fetch_pl3_paginates_and_normalizes_official_fields(monkeypatch):
    def fake_fetch(url, **kwargs):
        page = int(
            urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["pageNum"][0]
        )
        rows = (
            [
                {
                    "issue": "26003",
                    "frontWinningNum": "0 8 6",
                    "openTime": "2026-01-03",
                },
                {
                    "issue": "26002",
                    "frontWinningNum": "6 6 5",
                    "openTime": "2026-01-02",
                },
            ]
            if page == 1
            else [
                {
                    "issue": "26001",
                    "frontWinningNum": "0 0 1",
                    "openTime": "2026-01-01",
                }
            ]
        )
        return {"resCode": "000000", "total": 3, "data": rows}

    monkeypatch.setattr(digit_history_fetcher, "_PAGE_SIZE", 2)
    monkeypatch.setattr(digit_history_fetcher, "_fetch_json", fake_fetch)

    draws = fetch_digit_history("pl3", periods=3)

    assert [draw.issue for draw in draws] == ["26003", "26002", "26001"]
    assert draws[0].numbers == (0, 8, 6)


def test_fetcher_rejects_non_whitelisted_source():
    with pytest.raises(ValueError, match="白名单"):
        digit_history_fetcher._fetch_json(
            "https://example.com/data",
            headers={},
            timeout=1,
            retries=0,
        )


def test_write_digit_history_csv_preserves_leading_zero(tmp_path):
    draws = [
        DigitHistoryDraw(
            issue="2026001",
            numbers=(0, 2, 3),
            draw_date="2026-01-01",
            source="https://www.cwl.gov.cn/",
        )
    ]

    output = write_digit_history_csv(draws, tmp_path / "history.csv")

    text = output.read_text(encoding="utf-8")
    assert "2026001,023,2026-01-01" in text


def test_fetcher_rejects_invalid_requests():
    with pytest.raises(ValueError, match="只支持"):
        fetch_digit_history("pl5", periods=10)
    with pytest.raises(ValueError, match="必须为正数"):
        fetch_digit_history("fc3d", periods=0)

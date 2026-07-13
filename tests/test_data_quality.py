# -*- coding: utf-8 -*-

import json

import pandas as pd

from src.analysis.data_quality import check_kl8_data_quality, write_data_quality_reports


def _valid_row(issue: str, start: int = 1) -> dict:
    row = {"期数": issue}
    for idx, number in enumerate(range(start, start + 20), start=1):
        row[f"红球_{idx}"] = number
    return row


def test_check_kl8_data_quality_passes_valid_data():
    df = pd.DataFrame([_valid_row("2026002", 2), _valid_row("2026001", 1)])

    result = check_kl8_data_quality(df)

    assert result.ok is True
    assert result.total_issues == 2
    assert result.latest_issue == "2026002"
    assert result.error_count == 0
    assert result.warnings == []


def test_check_kl8_data_quality_reports_invalid_numbers_duplicates_and_duplicate_issue():
    bad_row = _valid_row("2026002", 1)
    bad_row["红球_20"] = 81
    duplicate_number_row = _valid_row("2026001", 1)
    duplicate_number_row["红球_20"] = 1
    df = pd.DataFrame([bad_row, duplicate_number_row, _valid_row("2026001", 3)])

    result = check_kl8_data_quality(df)

    assert result.ok is False
    assert result.error_count >= 3
    assert any(item.code == "NUMBER_OUT_OF_RANGE" and item.issue == "2026002" for item in result.items)
    assert any(item.code == "DUPLICATE_NUMBER" and item.issue == "2026001" for item in result.items)
    assert any(item.code == "DUPLICATE_ISSUE" and item.issue == "2026001" for item in result.items)


def test_write_data_quality_reports_outputs_json_and_markdown(tmp_path):
    df = pd.DataFrame([_valid_row("2026001")])
    result = check_kl8_data_quality(df)

    json_path, md_path = write_data_quality_reports(result, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["latestIssue"] == "2026001"
    assert "数据质量正常" in md_path.read_text(encoding="utf-8")

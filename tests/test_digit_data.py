# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.digit_data import load_digit_csv, normalize_digit_dataframe
from src.lotteries import get_lottery_rule


def test_normalize_digit_dataframe_accepts_standard_columns():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "2026002", "百位": "1", "十位": "2", "个位": "3"},
            {"期数": "2026001", "百位": 9, "十位": 9, "个位": 9},
        ]
    )

    normalized = normalize_digit_dataframe(df, rule)

    assert list(normalized.columns) == ["期数", "百位", "十位", "个位"]
    assert normalized.iloc[0].to_dict() == {"期数": "2026002", "百位": 1, "十位": 2, "个位": 3}
    assert normalized.iloc[1]["个位"] == 9


def test_normalize_digit_dataframe_splits_number_column_for_fc3d():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"issue": "2026002", "number": "123"},
            {"issue": "2026001", "number": "009"},
        ]
    )

    normalized = normalize_digit_dataframe(df, rule)

    assert normalized.to_dict("records") == [
        {"期数": "2026002", "百位": 1, "十位": 2, "个位": 3},
        {"期数": "2026001", "百位": 0, "十位": 0, "个位": 9},
    ]


def test_normalize_digit_dataframe_splits_number_column_for_pl5():
    rule = get_lottery_rule("pl5")
    df = pd.DataFrame([{"期号": "2026001", "开奖号码": "01234"}])

    normalized = normalize_digit_dataframe(df, rule)

    assert normalized.iloc[0].to_dict() == {
        "期数": "2026001",
        "万位": 0,
        "千位": 1,
        "百位": 2,
        "十位": 3,
        "个位": 4,
    }


def test_load_digit_csv_reads_and_normalizes(tmp_path: Path):
    rule = get_lottery_rule("fc3d")
    csv_path = tmp_path / "fc3d.csv"
    csv_path.write_text("期号,开奖号码\n2026002,123\n2026001,009\n", encoding="utf-8")

    normalized = load_digit_csv(csv_path, rule)

    assert len(normalized) == 2
    assert normalized.iloc[0]["期数"] == "2026002"
    assert normalized.iloc[1]["个位"] == 9


def test_normalize_digit_dataframe_rejects_invalid_digit_range():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame([{"期数": "2026001", "百位": 1, "十位": 2, "个位": 10}])

    with pytest.raises(ValueError, match="范围"):
        normalize_digit_dataframe(df, rule)


def test_normalize_digit_dataframe_rejects_missing_issue():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame([{"number": "123"}])

    with pytest.raises(ValueError, match="期号"):
        normalize_digit_dataframe(df, rule)

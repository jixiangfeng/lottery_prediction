# -*- coding: utf-8 -*-

import pytest

from src.analysis.digit_daily_policy import select_daily_candidates


def test_daily_policy_excludes_latest_and_caps_but_keeps_one_triple():
    ranked = [
        "906",
        "111",
        "222",
        "123",
        "333",
        "456",
    ] + [
        value
        for value in (f"{number:03d}" for number in range(1000))
        if value not in {"906", "111", "222", "123", "333", "456"}
        and len(set(value)) > 1
    ]

    selected = select_daily_candidates(
        ranked, latest_exact="906", top_k=50, maximum_triples=1
    )

    assert len(selected) == 50
    assert len(set(selected)) == 50
    assert "906" not in selected
    assert "111" in selected
    assert "222" not in selected
    assert "333" not in selected
    assert sum(len(set(value)) == 1 for value in selected) == 1
    assert selected[:3] == ("111", "123", "456")


def test_daily_policy_rejects_insufficient_eligible_candidates():
    with pytest.raises(ValueError, match="不足50个"):
        select_daily_candidates(
            ["906", "111", "222"],
            latest_exact="906",
            top_k=50,
            maximum_triples=1,
        )

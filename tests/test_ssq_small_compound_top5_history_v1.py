# -*- coding: utf-8 -*-
"""双色球 Top5 小复式全历史严格前序回溯测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scripts import ssq_small_compound_top5_history_v1 as cli
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_reference_helpers import (
    _small_compound_7_red_1_blue_top5,
)
from src.analysis.ssq_small_compound_top5_history_v1 import (
    CONTROL_COUNT,
    EVENT_NAMES,
    SCHEMA_VERSION,
    build_matched_control,
    evaluate_full_history,
    write_report,
)
from src.lotteries.ssq import SSQ_RULE


def _draws(count: int) -> list[SSQDraw]:
    draws: list[SSQDraw] = []
    for index in range(count):
        red = tuple(sorted(((index + offset * 5) % 33) + 1 for offset in range(6)))
        normalized_red, blue = SSQ_RULE.validate_draw(red, index % 16 + 1)
        draws.append(
            SSQDraw(
                issue=str(2024001 + index),
                draw_date=f"2024-{index % 12 + 1:02d}-{index % 28 + 1:02d}",
                red=normalized_red,
                blue=blue,
                source_url="https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/",
                raw_hash=f"{index:064x}",
                raw={},
            )
        )
    return draws


def test_matched_control_is_deterministic_legal_unique_and_result_independent() -> None:
    first = build_matched_control("2026001", 0)
    second = build_matched_control("2026001", 0)
    other = build_matched_control("2026001", 1)

    assert first == second
    assert first != other
    assert len(first.red7_sets) == len(set(first.red7_sets)) == 5
    assert len(first.expanded_tickets) == len(set(first.expanded_tickets)) == 35
    assert 1 <= first.shared_blue <= 16
    for red7 in first.red7_sets:
        assert len(red7) == len(set(red7)) == 7
        assert tuple(sorted(red7)) == red7
    for red6 in first.expanded_tickets:
        SSQ_RULE.validate_draw(red6, first.shared_blue)


def test_full_history_report_has_every_issue_and_exact_current_construction() -> None:
    report = evaluate_full_history(_draws(123))
    history = cast(dict[str, object], report["history"])
    per_issue = cast(list[dict[str, object]], report["perIssue"])

    assert report["schemaVersion"] == SCHEMA_VERSION
    assert history["evaluatedPeriods"] == 3
    assert len(per_issue) == 3
    assert [item["issue"] for item in per_issue] == ["2024121", "2024122", "2024123"]
    for item in per_issue:
        audit = cast(list[dict[str, object]], item["red6AuditTop32"])
        expected = _small_compound_7_red_1_blue_top5(
            audit, f"{cast(int, item['sharedBlue']):02d}"
        )
        compounds = cast(list[dict[str, object]], expected["compounds"])
        expected_red = [
            [int(value) for value in cast(list[str], compound["red"])]
            for compound in compounds
        ]
        assert item["compoundRedSets"] == expected_red
        assert len(audit) == 32
        assert len(cast(list[int], item["compoundRedHitCounts"])) == 5
        assert (
            sum(
                cast(
                    dict[str, int], item["expanded35TicketRedHitDistribution"]
                ).values()
            )
            == 35
        )
        assert set(cast(dict[str, bool], item["patterns"])) == set(EVENT_NAMES)
        assert len(cast(dict[str, float], item["matchedControlMeans"])) == 16


def test_prefix_invariance_prevents_future_leakage() -> None:
    prefix = evaluate_full_history(_draws(122))
    extended = evaluate_full_history(_draws(123))

    prefix_issues = cast(list[dict[str, object]], prefix["perIssue"])
    extended_issues = cast(list[dict[str, object]], extended["perIssue"])
    assert extended_issues[: len(prefix_issues)] == prefix_issues


def test_report_metrics_controls_and_hash_are_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    first = evaluate_full_history(_draws(121))
    second = evaluate_full_history(list(reversed(_draws(121))))

    assert first == second
    matched = cast(dict[str, object], first["matchedControl"])
    assert matched["controlsPerIssue"] == CONTROL_COUNT
    assert matched["portfolioObservations"] == CONTROL_COUNT
    assert matched["noResultDependence"] is True
    assertions = cast(dict[str, bool], first["assertions"])
    assert assertions["strictPriorPrediction"] is True
    assert assertions["futureProbabilitiesStored"] is False
    tests = cast(
        dict[str, dict[str, object]],
        cast(dict[str, object], first["descriptivePairedTests"])["tests"],
    )
    assert set(tests) == {
        "averageCompoundRedHits",
        "averageExpandedTicketRedHits",
        "maximumTicketRedHits",
        "blueHit",
        *EVENT_NAMES,
    }
    output = write_report(first, tmp_path / "report.json")
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == first


def test_cli_rejects_model_or_control_overrides() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--controls", "1"])

    assert error.value.code == 2

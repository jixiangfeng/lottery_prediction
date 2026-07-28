# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from scripts import ssq_diversified_portfolio_v2_history as cli
from src.analysis.ssq_diversified_portfolio_v2_history import (
    CONTROL_COUNT,
    METRIC_NAMES,
    build_matched_control_c,
    evaluate_full_history,
    walk_forward_portfolio_fingerprints,
)
from src.analysis.ssq_history import SSQDraw
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
                source_url="fixture",
                raw_hash=f"{index:064x}",
                raw={},
            )
        )
    return draws


def test_matched_control_c_is_deterministic_legal_and_matched_cost() -> None:
    first = build_matched_control_c("2026086", 0)
    second = build_matched_control_c("2026086", 0)
    other = build_matched_control_c("2026086", 1)

    assert first == second
    assert first != other
    assert len(first.red7_groups) == len(set(first.red7_groups)) == 5
    assert len(first.blues) == len(set(first.blues)) == 5
    assert len(first.tickets) == len(set(first.tickets)) == 35
    for red6, blue in first.tickets:
        SSQ_RULE.validate_draw(red6, blue)


def test_prefix_invariance_and_no_future_leakage() -> None:
    prefix = _draws(124)
    extended = [*prefix, *_draws(3)]
    for index in range(len(prefix), len(extended)):
        draw = extended[index]
        extended[index] = SSQDraw(
            issue=str(2030000 + index),
            draw_date=draw.draw_date,
            red=tuple(reversed(draw.red)),
            blue=16,
            source_url=draw.source_url,
            raw_hash=draw.raw_hash,
            raw={},
        )

    assert walk_forward_portfolio_fingerprints(extended)[: len(prefix)] == (
        walk_forward_portfolio_fingerprints(prefix)
    )


def test_full_history_contains_a_b_c_deltas_tests_and_every_issue() -> None:
    report = evaluate_full_history(_draws(123))
    history = cast(dict[str, object], report["history"])
    per_issue = cast(list[dict[str, object]], report["perIssue"])

    assert history["warmupPeriods"] == 120
    assert history["evaluatedPeriods"] == 3
    assert len(per_issue) == 3
    assert [item["issue"] for item in per_issue] == ["2024121", "2024122", "2024123"]
    summary = cast(dict[str, dict[str, object]], report["summary"])
    assert set(summary) == {"A", "B", "C"}
    assert summary["A"]["portfolioObservations"] == 3
    assert summary["B"]["portfolioObservations"] == 3
    assert summary["C"]["portfolioObservations"] == 3 * CONTROL_COUNT
    assert set(cast(dict[str, object], report["deltas"])) == {"BMinusA", "BMinusC"}
    tests = cast(dict[str, dict[str, object]], report["descriptivePairedTests"])
    assert set(cast(dict[str, object], tests["BMinusA"])) == set(METRIC_NAMES)
    assert set(cast(dict[str, object], tests["BMinusC"])) == set(METRIC_NAMES)
    for item in per_issue:
        audit = cast(dict[str, object], cast(dict[str, object], item["B"])["audit"])
        assert audit["redUnionCount"] == 33
        assert audit["maximumRedExposure"] == 2
        assert cast(int, audit["maximumPairwiseIntersection"]) <= 3
        assert audit["distinctBlues"] == 5


def test_report_hash_binding_is_stable() -> None:
    first = evaluate_full_history(_draws(121))
    second = evaluate_full_history(list(reversed(_draws(121))))

    assert first == second
    claimed = first["reportSha256"]
    unsigned = dict(first)
    unsigned.pop("reportSha256")
    from src.analysis.ssq_diversified_portfolio_v2_history import _sha256_payload

    assert claimed == _sha256_payload(unsigned)


def test_cli_rejects_algorithm_tuning(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--controls", "1"])

    assert error.value.code == 2
    assert not (tmp_path / "report.json").exists()

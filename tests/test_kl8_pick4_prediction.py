# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.kl8_pick4_predict_today import main as pick4_main
from src.analysis.kl8_pick4_prediction import (
    build_pick4_prediction_boundary,
    generate_uniform_pick4_test_tickets,
    pick4_hit_pmf,
    validate_pick4_ticket,
)
from src.lotteries.kl8 import KL8_SUPPORTED_PICK_COUNTS


def _write_poisoned_csv(
    path: Path, development_periods: int = 20, frozen: int = 2
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["issue", "date", "numbers"])
        writer.writeheader()
        for index in range(development_periods):
            numbers = sorted(
                {((index * 7 + offset * 3) % 80) + 1 for offset in range(20)}
            )
            writer.writerow(
                {
                    "issue": str(2020001 + index),
                    "date": f"2020-11-{index + 1:02d}",
                    "numbers": " ".join(str(number) for number in numbers),
                }
            )
        for index in range(frozen):
            writer.writerow(
                {
                    "issue": str(2021001 + index),
                    "date": f"2021-01-{index + 1:02d}",
                    "numbers": "POISON_FROZEN",
                }
            )


def test_pick4_exact_hit_pmf_is_normalized_with_mean_one() -> None:
    pmf = pick4_hit_pmf()

    assert set(pmf) == {0, 1, 2, 3, 4}
    assert sum(pmf.values()) == pytest.approx(1.0)
    assert sum(
        hits * probability for hits, probability in pmf.items()
    ) == pytest.approx(1.0)
    assert pmf[4] == pytest.approx(0.003063392303898633)


def test_pick4_ticket_validation_rejects_duplicates_and_ranges() -> None:
    assert validate_pick4_ticket([1, 7, 40, 80]) == [1, 7, 40, 80]
    assert validate_pick4_ticket([80, 1, 40, 7]) == [1, 7, 40, 80]

    with pytest.raises(ValueError, match="4个唯一号码"):
        validate_pick4_ticket([1, 1, 2, 3])
    with pytest.raises(ValueError, match="1..80"):
        validate_pick4_ticket([0, 2, 3, 4])
    with pytest.raises(ValueError, match="整数"):
        validate_pick4_ticket([1, 2, 3, True])


def test_uniform_pick4_test_tickets_are_deterministic_and_valid() -> None:
    first = generate_uniform_pick4_test_tickets(
        target_date="2026-07-23", development_sha256="a" * 64, ticket_count=5
    )
    second = generate_uniform_pick4_test_tickets(
        target_date="2026-07-23", development_sha256="a" * 64, ticket_count=5
    )

    assert first == second
    assert len(first) == len({tuple(ticket) for ticket in first}) == 5
    assert all(validate_pick4_ticket(ticket) == ticket for ticket in first)


def test_pick4_boundary_never_parses_poisoned_frozen_numbers(tmp_path: Path) -> None:
    csv_path = tmp_path / "kl8.csv"
    _write_poisoned_csv(csv_path)

    safe = build_pick4_prediction_boundary(csv_path, frozen_periods=2)
    test = build_pick4_prediction_boundary(
        csv_path,
        frozen_periods=2,
        target_date="2026-07-23",
        test_ticket_count=5,
    )

    assert safe["play"] == "pick4"
    assert safe["formalRecommendation"] is None
    assert safe["userVisibleCandidates"] == []
    assert safe["testCandidates"] == []
    assert safe["frozenRead"] is False
    assert test["formalRecommendation"] is None
    assert test["userVisibleCandidates"] == []
    assert len(test["testCandidates"]) == 5
    assert test["testCandidateKind"] == "uniform_random_test_only"
    assert test["baseline"]["meanHitsPerTicket"] == pytest.approx(1.0)


def test_pick4_cli_requires_explicit_test_flag_for_test_numbers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "kl8.csv"
    _write_poisoned_csv(csv_path)

    assert pick4_main(["--csv", str(csv_path), "--frozen-periods", "2"]) == 0
    safe = json.loads(capsys.readouterr().out)
    assert safe["testCandidates"] == []

    assert (
        pick4_main(
            [
                "--csv",
                str(csv_path),
                "--frozen-periods",
                "2",
                "--test",
                "--date",
                "2026-07-23",
                "--ticket-count",
                "5",
            ]
        )
        == 0
    )
    test = json.loads(capsys.readouterr().out)
    assert len(test["testCandidates"]) == 5
    assert test["promotionPassed"] is False
    assert test["recommendationEnabled"] is False


def test_kl8_rule_declares_pick4_and_pick5_support() -> None:
    assert KL8_SUPPORTED_PICK_COUNTS == (4, 5)

# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

import pytest

import src.analysis.ssq_weight_training_v2 as training
from scripts.ssq_weight_training_v2 import main
from src.analysis.ssq_history import SSQDraw, build_ssq_source_url

LOW_RED = (1, 2, 3, 4, 5, 6)
HIGH_RED = (28, 29, 30, 31, 32, 33)


@pytest.fixture(autouse=True)
def _disable_descriptive_beam_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "generate_research_tickets", lambda *_args: [])


def _draw(index: int, red: tuple[int, ...], blue: int) -> SSQDraw:
    return SSQDraw(
        issue=str(2_000_000 + index),
        draw_date=(date(2020, 1, 1) + timedelta(days=index)).isoformat(),
        red=red,
        blue=blue,
        source_url=build_ssq_source_url(1),
        raw_hash=f"{index % 16:x}" * 64,
        raw={},
    )


def _regime_shift_draws(
    *,
    search_draws: int = 100,
    frozen_factory: Callable[[int], tuple[tuple[int, ...], int]] | None = None,
) -> list[SSQDraw]:
    frozen = frozen_factory or (lambda _index: (LOW_RED, 1))
    validation_start = training.EVALUATION_WARMUP_DRAWS + search_draws
    frozen_start = validation_start + training.VALIDATION_DRAWS
    total = frozen_start + training.FROZEN_TEST_DRAWS
    draws: list[SSQDraw] = []
    for index in range(total):
        if index < validation_start:
            red, blue = LOW_RED, 1
        elif index < frozen_start:
            red, blue = HIGH_RED, 16
        else:
            red, blue = frozen(index - frozen_start)
        draws.append(_draw(index, red, blue))
    return draws


def _no_eligible_draws(
    frozen_factory: Callable[[int], tuple[tuple[int, ...], int]] | None = None,
) -> list[SSQDraw]:
    frozen = frozen_factory or (lambda _index: (HIGH_RED, 16))
    total = (
        training.EVALUATION_WARMUP_DRAWS
        + 1
        + training.VALIDATION_DRAWS
        + training.FROZEN_TEST_DRAWS
    )
    frozen_start = total - training.FROZEN_TEST_DRAWS
    draws: list[SSQDraw] = []
    for index in range(total):
        if index < training.EVALUATION_WARMUP_DRAWS:
            red, blue = LOW_RED, 1
        elif index < frozen_start:
            red, blue = HIGH_RED, 16
        else:
            red, blue = frozen(index - frozen_start)
        draws.append(_draw(index, red, blue))
    return draws


def test_prior_outputs_do_not_use_current_or_future_draws() -> None:
    first = training.StaticExpertState()
    second = training.StaticExpertState()
    for index in range(training.EVALUATION_WARMUP_DRAWS):
        draw = _draw(index, LOW_RED, 1)
        first.update(draw)
        second.update(draw)

    low_observation = first.observe_prior(_draw(5000, LOW_RED, 1))
    high_observation = second.observe_prior(_draw(5000, HIGH_RED, 16))

    assert low_observation.red_experts == high_observation.red_experts
    assert low_observation.blue_experts == high_observation.blue_experts
    assert low_observation.pair_modifiers == high_observation.pair_modifiers


def test_search_selection_is_isolated_from_validation_and_frozen() -> None:
    first = training.evaluate_ssq_weight_training(_no_eligible_draws())
    second = training.evaluate_ssq_weight_training(
        _no_eligible_draws(
            lambda index: (LOW_RED, 1) if index % 2 == 0 else (HIGH_RED, 16)
        )
    )

    assert first["dataSha256"] != second["dataSha256"]
    assert first["searchCandidates"] == second["searchCandidates"]
    assert first["selectedWeights"] == second["selectedWeights"]
    assert first["validationOpened"] is False
    assert second["frozenTestOpened"] is False


def test_selection_is_deterministic_and_uses_fixed_grid() -> None:
    first = training.evaluate_ssq_weight_training(_regime_shift_draws())
    second = training.evaluate_ssq_weight_training(_regime_shift_draws())

    assert len(training.RED_WEIGHT_GRID) == 66
    assert len(training.BLUE_WEIGHT_GRID) == 11
    assert first["candidateCount"] == 726
    assert first["selectedWeights"] == second["selectedWeights"]
    assert first["searchCandidates"] == second["searchCandidates"]
    assert first["selectedWeights"] == {
        "red": [0.0, 0.0, 1.0],
        "blue": [0.0, 1.0],
        "pairModifier": 0.2,
    }


def test_no_eligible_candidate_keeps_validation_and_frozen_unopened() -> None:
    report = training.evaluate_ssq_weight_training(_no_eligible_draws())

    assert report["decision"] == "uniform_abstention_no_eligible_search_candidate"
    assert report["selection"] == "uniform_abstention"
    assert report["validationOpened"] is False
    assert report["frozenTestOpened"] is False
    assert report["validation"] == {"opened": False}
    assert report["frozenTest"] == {"opened": False}
    assert "未开封" in str(report["failureDisposition"])


def test_failed_validation_keeps_frozen_unopened_and_is_frozen_isolated() -> None:
    first = training.evaluate_ssq_weight_training(_regime_shift_draws())
    second = training.evaluate_ssq_weight_training(
        _regime_shift_draws(
            frozen_factory=lambda index: (
                (HIGH_RED, 16) if index % 2 == 0 else (LOW_RED, 1)
            )
        )
    )

    assert first["decision"] == "research_rejected_validation_failure"
    assert first["validationOpened"] is True
    assert first["validation"]["passed"] is False
    assert first["frozenTestOpened"] is False
    assert first["frozenTest"] == {"opened": False}
    assert first["selectedWeights"] == second["selectedWeights"]
    assert first["validation"] == second["validation"]


@pytest.mark.parametrize(
    "draws",
    [_no_eligible_draws(), _regime_shift_draws()],
)
def test_reports_never_emit_formal_recommendations(draws: list[SSQDraw]) -> None:
    report = training.evaluate_ssq_weight_training(draws)

    assert report["researchOnly"] is True
    assert report["recommendationEnabled"] is False
    assert report["productionActivation"] is False
    assert report["formalCandidates"] == []
    assert report["retryCount"] == 0
    assert report["postFrozenTuning"] is False


def test_cli_rejects_protocol_overrides() -> None:
    with pytest.raises(SystemExit):
        main(["--grid-step", "0.05"])

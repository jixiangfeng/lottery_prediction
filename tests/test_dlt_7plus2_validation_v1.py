from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import numpy as np

from src.analysis.dlt_7plus2_v1 import FROZEN_PROTOCOL_SHA256
from src.analysis.dlt_7plus2_validation_v1 import (
    VALIDATION_START,
    VALIDATION_STOP,
    builder_metrics,
    deterministic_random_controls,
    evaluate_gates,
    global_validation_cutoffs,
    holm_adjust,
    moving_block_bootstrap_p,
    run_validation,
    select_candidate,
)
from src.analysis.dlt_ridge_candidates_v1 import CANDIDATE_IDS, CandidateScores


@dataclass(frozen=True)
class Draw:
    front: tuple[int, ...]
    back: tuple[int, ...]
    issue: str


def cyclic_draw(index: int) -> Draw:
    return Draw(
        tuple(sorted((index * 5 + offset) % 35 + 1 for offset in range(5))),
        tuple(sorted(((index * 2) % 12 + 1, (index * 2 + 5) % 12 + 1))),
        f"I{index:04d}",
    )


def test_global_validation_partition_and_cutoffs_are_exact() -> None:
    assert (VALIDATION_START, VALIDATION_STOP) == (1901, 2401)
    assert global_validation_cutoffs() == tuple(range(1900, 2401, 25))
    assert len(global_validation_cutoffs()) == 21


def test_holm_is_step_down_monotone_and_maps_back() -> None:
    assert holm_adjust([0.04, 0.001, 0.02, 0.2]) == [0.08, 0.004, 0.06, 0.2]


def test_bootstrap_is_deterministic_and_one_sided() -> None:
    values = np.linspace(-0.05, 0.15, 500)
    first = moving_block_bootstrap_p(values, seed=123, replicates=2000, block_length=20)
    second = moving_block_bootstrap_p(
        values, seed=123, replicates=2000, block_length=20
    )
    assert first == second
    assert 0.0 < first < 0.05


def test_random_controls_are_exact_reproducible_and_same_cost() -> None:
    first = deterministic_random_controls(FROZEN_PROTOCOL_SHA256, "26001")
    second = deterministic_random_controls(FROZEN_PROTOCOL_SHA256, "26001")
    assert first == second
    assert len(first) == 512
    assert len(set(first)) == 512
    for front, back in first:
        assert len(front) == len(set(front)) == 7
        assert len(back) == len(set(back)) == 2
        assert all(1 <= value <= 35 for value in front)
        assert all(1 <= value <= 12 for value in back)
    assert first[0] == second[0]  # r=0 is retained separately by the report.


def test_builder_metrics_has_21_ticket_matrix_and_thresholds() -> None:
    metrics = builder_metrics((1, 2, 3, 4, 5, 6, 7), (1, 2), (1, 2, 3, 8, 9), (1, 3))
    assert metrics["HF"] == 3
    assert metrics["HB"] == 1
    assert metrics["U"] == 1.1
    assert sum(sum(row) for row in metrics["ticketHitMatrix"]) == 21
    assert metrics["bestTicket"] == {"frontHits": 3, "backHits": 1}
    assert metrics["thresholdIndicators"]["atLeast3Plus1"] is True
    assert metrics["thresholdIndicators"]["atLeast4Plus0"] is False
    assert metrics["frontFiveFullyCovered"] is False
    assert metrics["backTwoFullyHit"] is False
    assert metrics["exact5Plus2Exists"] is False


def _gate_input(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "meanJointLogLossImprovement": 0.1,
        "holmAdjustedP": 0.01,
        "meanFrontSetLogLoss": math.log(math.comb(35, 5)),
        "meanBackSetLogLoss": math.log(math.comb(12, 2)),
        "meanFrontMarginalBrier": 6 / 49,
        "meanBackMarginalBrier": 5 / 36,
        "jointBlockImprovements": [0.1, 0.1, 0.1, 0.1, -0.05],
        "meanUImprovementVsRandom512": 0.01,
        "uBlockImprovementsVsRandom512": [0.1, 0.1, 0.1, 0.1, -0.2],
    }
    result.update(updates)
    return result


def test_gates_are_exact_inclusive_except_strict_improvements() -> None:
    gates = evaluate_gates(_gate_input())
    assert all(gates.values())
    assert not evaluate_gates(_gate_input(meanJointLogLossImprovement=0.0))[
        "jointMeanPositive"
    ]
    assert not evaluate_gates(_gate_input(meanUImprovementVsRandom512=0.0))[
        "uMeanAboveRandom512"
    ]
    assert not evaluate_gates(
        _gate_input(jointBlockImprovements=[1, 1, 1, -0.01, -0.01])
    )["jointFourOfFiveNonnegative"]
    assert not evaluate_gates(
        _gate_input(jointBlockImprovements=[1, 1, 1, 1, -0.0500001])
    )["jointWorstBlockFloor"]


def _candidate(
    candidate_id: str, *, eligible: bool, ll: float, brier: float, u: float
) -> dict[str, object]:
    return {
        "candidateId": candidate_id,
        "eligible": eligible,
        "metrics": {
            "meanJointLogLossImprovement": ll,
            "combinedBrierImprovement": brier,
            "meanUImprovementVsRandom512": u,
        },
    }


def test_selection_rejects_none_and_uses_all_ties_deterministically() -> None:
    assert (
        select_candidate([_candidate("C1", eligible=False, ll=9, brier=9, u=9)]) is None
    )
    candidates = [
        _candidate("C3", eligible=True, ll=1, brier=2, u=3),
        _candidate("C2", eligible=True, ll=1, brier=2, u=3),
        _candidate("C1", eligible=True, ll=1, brier=1, u=9),
    ]
    assert select_candidate(candidates) == "C2"


class TrapDraws:
    def __init__(self) -> None:
        self.accessed: list[int] = []

    def __getitem__(self, index: int) -> Draw:
        if not isinstance(index, int):
            raise TypeError("validation must not slice/probe the sequence")
        if index >= VALIDATION_STOP:
            raise AssertionError("Frozen outcome accessed")
        self.accessed.append(index)
        return cyclic_draw(index)


def _constant_provider(_cache: object, cutoff: int) -> dict[str, CandidateScores]:
    score = CandidateScores((0.0,) * 35, (0.0,) * 12)
    return {candidate_id: score for candidate_id in CANDIDATE_IDS}


def _search_fixture() -> dict[str, object]:
    calibration = {
        "front": {"tau": 1.0, "epsilon": 0.0},
        "back": {"tau": 1.0, "epsilon": 0.0},
    }
    return {
        "schemaVersion": 1,
        "study": "dlt_7plus2_search_v1",
        "partitions": {"searchCount": 1301, "maximumConsumedIndex": 1900},
        "candidates": [
            {"candidateId": candidate_id, "selectedCalibration": calibration}
            for candidate_id in CANDIDATE_IDS
        ],
    }


def test_trap_proves_no_index_2401_or_later_and_no_frozen_scorer() -> None:
    draws = TrapDraws()
    report = run_validation(
        draws,
        search_report=_search_fixture(),
        search_report_bytes=b"fixture",
        score_provider=_constant_provider,
        bootstrap_replicates=100,
    )
    assert max(draws.accessed) == 2400
    assert report["partitions"]["maximumConsumedIndex"] == 2400
    assert report["frozenOpened"] is False
    assert report["frozenRowsAccessed"] == 0
    assert report["currentTargetGroup"] is None
    assert report["formalOutput"] == "uniform_abstain"
    assert report["formalGate"] is False
    import src.analysis.dlt_7plus2_validation_v1 as module

    assert "FrozenScorer" not in inspect.getsource(module)

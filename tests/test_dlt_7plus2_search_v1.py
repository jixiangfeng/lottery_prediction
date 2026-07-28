# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.analysis.dlt_7plus2_search_v1 import (
    EPSILON_GRID,
    SEARCH_START,
    SEARCH_STOP,
    TAU_GRID,
    calibration_grid,
    global_block_cutoffs,
    run_search_calibration,
    select_joint_grid_point,
    write_search_report,
)
from src.analysis.dlt_ridge_candidates_v1 import CANDIDATE_IDS, CandidateScores


@dataclass(frozen=True)
class Draw:
    front: tuple[int, ...]
    back: tuple[int, ...]
    issue: str = "fixture"


def cyclic_draw(index: int) -> Draw:
    return Draw(
        tuple(sorted((index * 5 + offset) % 35 + 1 for offset in range(5))),
        tuple(sorted(((index * 2) % 12 + 1, (index * 2 + 5) % 12 + 1))),
        issue=f"{index:05d}",
    )


def test_search_partition_and_global_cutoffs_are_exact() -> None:
    assert (SEARCH_START, SEARCH_STOP) == (600, 1901)
    assert global_block_cutoffs() == tuple(range(600, 1901, 25))
    assert global_block_cutoffs()[-1] == 1900


def test_calibration_grid_has_exact_twenty_points() -> None:
    assert TAU_GRID == (0.5, 0.75, 1.0, 1.5, 2.0)
    assert EPSILON_GRID == (0.0, 0.05, 0.1, 0.2)
    assert calibration_grid() == tuple(
        (tau, epsilon) for tau in TAU_GRID for epsilon in EPSILON_GRID
    )
    assert len(calibration_grid()) == 20


def test_joint_tie_rules_use_combined_values_then_penalties_and_lexicographic() -> None:
    # Equal joint LL: combined Brier wins even when neither zone is independently best.
    front = [
        {"tau": 0.5, "epsilon": 0.2, "meanLogLoss": 1.0, "meanBrier": 0.1},
        {"tau": 1.5, "epsilon": 0.0, "meanLogLoss": 1.0, "meanBrier": 0.2},
    ]
    back = [
        {"tau": 2.0, "epsilon": 0.0, "meanLogLoss": 2.0, "meanBrier": 0.3},
        {"tau": 1.0, "epsilon": 0.1, "meanLogLoss": 2.0, "meanBrier": 0.05},
    ]
    selected = select_joint_grid_point(front, back)
    assert selected["front"] == front[0]
    assert selected["back"] == back[1]

    # Full score tie: |tau-1| total, epsilon total, then parameter tuple.
    tied_front = [
        {"tau": 0.5, "epsilon": 0.0, "meanLogLoss": 1.0, "meanBrier": 0.1},
        {"tau": 1.5, "epsilon": 0.0, "meanLogLoss": 1.0, "meanBrier": 0.1},
    ]
    tied_back = [{"tau": 1.0, "epsilon": 0.0, "meanLogLoss": 2.0, "meanBrier": 0.2}]
    selected = select_joint_grid_point(tied_front, tied_back)
    assert selected["front"]["tau"] == 0.5


class TrapDraws:
    def __init__(self) -> None:
        self.accessed: list[int] = []

    def __getitem__(self, index: int) -> Draw:
        if not isinstance(index, int):
            raise AssertionError("Search path must not slice or inspect a suffix")
        if index >= SEARCH_STOP:
            raise AssertionError("Validation/Frozen was consumed")
        self.accessed.append(index)
        return cyclic_draw(index)


def _constant_score_provider(_draws: object):
    score = CandidateScores((0.0,) * 35, (0.0,) * 12)
    for candidate_id in CANDIDATE_IDS:
        yield candidate_id, tuple(
            (target, target - target % 25, score)
            for target in range(SEARCH_START, SEARCH_STOP)
        )


def test_search_consumes_only_indices_through_1900_via_trap_sequence() -> None:
    draws = TrapDraws()
    report = run_search_calibration(draws, score_provider=_constant_score_provider)
    assert min(draws.accessed) == SEARCH_START
    assert max(draws.accessed) == 1900
    assert report["partitions"]["searchCount"] == 1301
    assert report["selectionStatus"] == "search_calibration_only"


def test_search_report_is_byte_deterministic(tmp_path: Path) -> None:
    draws = TrapDraws()
    report = run_search_calibration(draws, score_provider=_constant_score_provider)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_search_report(report, first)
    write_search_report(report, second)
    assert first.read_bytes() == second.read_bytes()
    decoded = json.loads(first.read_text(encoding="utf-8"))
    assert len(decoded["candidates"]) == 4
    for candidate in decoded["candidates"]:
        assert len(candidate["frontCalibrationGrid"]) == 20
        assert len(candidate["backCalibrationGrid"]) == 20
        assert len(candidate["blockCutoffs"]) == 53

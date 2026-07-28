# -*- coding: utf-8 -*-
"""DLT 7+2 v1：严格只消费 Warmup+Search 的校准与审计报告。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from src.analysis.dlt_fixed_cardinality_v1 import FixedCardinalityDistribution
from src.analysis.dlt_ridge_candidates_v1 import (
    BACK_SIZE,
    BLOCK_SIZE,
    CANDIDATE_IDS,
    FRONT_SIZE,
    CandidateScores,
    DLTDrawLike,
    FeatureSnapshotCache,
    fit_candidate_scores_from_cache,
)

WARMUP_STOP = 600
SEARCH_START = 600
SEARCH_STOP = 1901
TAU_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
EPSILON_GRID = (0.0, 0.05, 0.1, 0.2)


@dataclass(frozen=True)
class SearchDraw:
    issue: str
    front: tuple[int, ...]
    back: tuple[int, ...]


def calibration_grid() -> tuple[tuple[float, float], ...]:
    return tuple((tau, epsilon) for tau in TAU_GRID for epsilon in EPSILON_GRID)


def global_block_cutoffs() -> tuple[int, ...]:
    return tuple(range(SEARCH_START, SEARCH_STOP, BLOCK_SIZE))


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _log_mixture(
    base_log_probability: float, uniform_log_probability: float, epsilon: float
) -> float:
    if epsilon == 0.0:
        return base_log_probability
    left = math.log1p(-epsilon) + base_log_probability
    right = math.log(epsilon) + uniform_log_probability
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _zone_grid_results(
    score_rows: Sequence[tuple[float, ...]],
    observed_rows: Sequence[tuple[int, ...]],
    *,
    size: int,
    count: int,
) -> list[dict[str, float]]:
    accumulators: dict[tuple[float, float], dict[str, list[float]]] = {
        point: {"logLoss": [], "brier": []} for point in calibration_grid()
    }
    uniform_log_probability = -math.log(math.comb(size, count))
    uniform_marginal = count / size
    for scores, observed in zip(score_rows, observed_rows, strict=True):
        observed_zero = tuple(value - 1 for value in observed)
        labels = [float(index in observed_zero) for index in range(size)]
        for tau in TAU_GRID:
            base = FixedCardinalityDistribution(scores, count, tau=tau, epsilon=0.0)
            base_log_probability = base.log_probability(observed_zero)
            for epsilon in EPSILON_GRID:
                mixed_log_probability = _log_mixture(
                    base_log_probability, uniform_log_probability, epsilon
                )
                marginals = tuple(
                    (1.0 - epsilon) * value + epsilon * uniform_marginal
                    for value in base.marginals
                )
                brier = (
                    math.fsum(
                        (probability - label) ** 2
                        for probability, label in zip(marginals, labels, strict=True)
                    )
                    / size
                )
                accumulator = accumulators[(tau, epsilon)]
                accumulator["logLoss"].append(-mixed_log_probability)
                accumulator["brier"].append(brier)
    results: list[dict[str, float]] = []
    for tau, epsilon in calibration_grid():
        values = accumulators[(tau, epsilon)]
        result = {
            "tau": tau,
            "epsilon": epsilon,
            "meanLogLoss": _mean(values["logLoss"]),
            "meanBrier": _mean(values["brier"]),
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise ArithmeticError("Search 校准网格产生非有限结果")
        results.append(result)
    return results


def select_joint_grid_point(
    front_results: Sequence[Mapping[str, float]],
    back_results: Sequence[Mapping[str, float]],
) -> dict[str, object]:
    """穷举前后区组合，严格复现联合协议的完整平局顺序。"""

    if not front_results or not back_results:
        raise ValueError("前后区校准结果不得为空")

    def key(pair: tuple[Mapping[str, float], Mapping[str, float]]) -> tuple[float, ...]:
        front, back = pair
        return (
            front["meanLogLoss"] + back["meanLogLoss"],
            front["meanBrier"] + back["meanBrier"],
            abs(front["tau"] - 1.0) + abs(back["tau"] - 1.0),
            front["epsilon"] + back["epsilon"],
            front["tau"],
            front["epsilon"],
            back["tau"],
            back["epsilon"],
        )

    front, back = min(
        ((front, back) for front in front_results for back in back_results), key=key
    )
    return {
        "front": dict(front),
        "back": dict(back),
        "meanJointLogLoss": front["meanLogLoss"] + back["meanLogLoss"],
        "combinedMarginalBrier": front["meanBrier"] + back["meanBrier"],
    }


def _default_score_provider(draws: Sequence[DLTDrawLike]):
    cache = FeatureSnapshotCache(draws, stop_index=SEARCH_STOP)
    by_candidate: dict[str, list[tuple[int, int, CandidateScores]]] = {
        candidate_id: [] for candidate_id in CANDIDATE_IDS
    }
    for cutoff in global_block_cutoffs():
        components = {
            candidate_id: fit_candidate_scores_from_cache(cache, cutoff, candidate_id)
            for candidate_id in CANDIDATE_IDS[:3]
        }
        components[CANDIDATE_IDS[3]] = CandidateScores(
            tuple(
                math.fsum(components[item].front[index] for item in CANDIDATE_IDS[:3])
                / 3.0
                for index in range(FRONT_SIZE)
            ),
            tuple(
                math.fsum(components[item].back[index] for item in CANDIDATE_IDS[:3])
                / 3.0
                for index in range(BACK_SIZE)
            ),
        )
        stop = min(cutoff + BLOCK_SIZE, SEARCH_STOP)
        for candidate_id, scores in components.items():
            by_candidate[candidate_id].extend(
                (target, cutoff, scores)
                for target in range(max(cutoff, SEARCH_START), stop)
            )
    for candidate_id in CANDIDATE_IDS:
        yield candidate_id, tuple(by_candidate[candidate_id])


def _score_fingerprint(rows: Sequence[tuple[int, int, CandidateScores]]) -> str:
    digest = hashlib.sha256()
    for target, cutoff, scores in rows:
        digest.update(struct.pack(">II", target, cutoff))
        for value in scores.front + scores.back:
            digest.update(struct.pack(">d", value))
    return digest.hexdigest()


def _probability_fingerprint(
    rows: Sequence[tuple[int, int, CandidateScores]], selected: Mapping[str, object]
) -> str:
    front_parameters = selected["front"]
    back_parameters = selected["back"]
    assert isinstance(front_parameters, Mapping) and isinstance(
        back_parameters, Mapping
    )
    digest = hashlib.sha256()
    for target, cutoff, scores in rows:
        digest.update(struct.pack(">II", target, cutoff))
        front = FixedCardinalityDistribution(
            scores.front,
            5,
            tau=float(front_parameters["tau"]),
            epsilon=float(front_parameters["epsilon"]),
        )
        back = FixedCardinalityDistribution(
            scores.back,
            2,
            tau=float(back_parameters["tau"]),
            epsilon=float(back_parameters["epsilon"]),
        )
        for value in front.marginals + back.marginals:
            digest.update(struct.pack(">d", value))
    return digest.hexdigest()


def run_search_calibration(
    draws: Sequence[DLTDrawLike],
    *,
    score_provider: (
        Callable[
            [Sequence[DLTDrawLike]],
            Iterable[tuple[str, Sequence[tuple[int, int, CandidateScores]]]],
        ]
        | None
    ) = None,
) -> dict[str, object]:
    """只读取零基 0..1900；绝不查询长度或读取 1901 及之后的行。"""

    provider = score_provider or _default_score_provider
    predictions = dict(provider(draws))
    observed_front = []
    observed_back = []
    issues = []
    for target in range(SEARCH_START, SEARCH_STOP):
        draw = draws[target]
        observed_front.append(tuple(draw.front))
        observed_back.append(tuple(draw.back))
        issues.append(str(getattr(draw, "issue", target)))
    candidates: list[dict[str, object]] = []
    for candidate_id in CANDIDATE_IDS:
        rows = tuple(predictions[candidate_id])
        if len(rows) != SEARCH_STOP - SEARCH_START:
            raise ValueError(f"{candidate_id} Search 预测数不等于 1301")
        expected_targets = tuple(range(SEARCH_START, SEARCH_STOP))
        if tuple(item[0] for item in rows) != expected_targets:
            raise ValueError(f"{candidate_id} Search 目标索引不连续")
        if tuple(sorted({item[1] for item in rows})) != global_block_cutoffs():
            raise ValueError(f"{candidate_id} 全局 25 期截止点不正确")
        front_grid = _zone_grid_results(
            tuple(item[2].front for item in rows), observed_front, size=35, count=5
        )
        back_grid = _zone_grid_results(
            tuple(item[2].back for item in rows), observed_back, size=12, count=2
        )
        selected = select_joint_grid_point(front_grid, back_grid)
        selected_front = selected["front"]
        selected_back = selected["back"]
        assert isinstance(selected_front, Mapping)
        assert isinstance(selected_back, Mapping)
        candidates.append(
            {
                "candidateId": candidate_id,
                "frontCalibrationGrid": front_grid,
                "backCalibrationGrid": back_grid,
                "selectedCalibration": selected,
                "properScores": {
                    "meanFrontSetLogLoss": selected_front["meanLogLoss"],
                    "meanBackSetLogLoss": selected_back["meanLogLoss"],
                    "meanJointLogLoss": selected["meanJointLogLoss"],
                    "meanFrontMarginalBrier": selected_front["meanBrier"],
                    "meanBackMarginalBrier": selected_back["meanBrier"],
                },
                "scoreFingerprintSha256": _score_fingerprint(rows),
                "probabilityFingerprintSha256": _probability_fingerprint(
                    rows, selected
                ),
                "blockCutoffs": list(global_block_cutoffs()),
            }
        )
    data_material = json.dumps(
        [
            [issues[index], list(observed_front[index]), list(observed_back[index])]
            for index in range(len(issues))
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "schemaVersion": 1,
        "study": "dlt_7plus2_search_v1",
        "selectionStatus": "search_calibration_only",
        "finalSelectionBuilt": False,
        "validationOrFrozenRead": False,
        "partitions": {
            "warmupIndices": [0, 600],
            "searchIndices": [600, 1901],
            "searchCount": 1301,
            "maximumConsumedIndex": 1900,
        },
        "calibrationGrid": {"tau": list(TAU_GRID), "epsilon": list(EPSILON_GRID)},
        "searchObservedSha256": hashlib.sha256(data_material).hexdigest(),
        "candidates": candidates,
    }


def write_search_report(report: Mapping[str, object], output_path: str | Path) -> None:
    content = (
        json.dumps(
            report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)


def read_search_prefix_csv(path: str | Path) -> tuple[SearchDraw, ...]:
    """从升序 CSV 精确读取前 1901 行，不探测下一行。"""

    result: list[SearchDraw] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for index in range(SEARCH_STOP):
            try:
                row = next(reader)
            except StopIteration as error:
                raise ValueError("DLT CSV 不足 1901 行") from error
            front = tuple(int(value) for value in row["front"].split())
            back = tuple(int(value) for value in row["back"].split())
            result.append(SearchDraw(row["issue"], front, back))
    return tuple(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="data/dlt/official_history.csv")
    parser.add_argument(
        "--output-report", default="reports/research/dlt_7plus2_search_v1.json"
    )
    args = parser.parse_args(argv)
    try:
        draws = read_search_prefix_csv(args.input_csv)
        report = run_search_calibration(draws)
        write_search_report(report, args.output_report)
    except (OSError, ValueError, ArithmeticError) as error:
        print(f"DLT Search 校准失败：{error}", file=sys.stderr)
        return 1
    print(f"DLT Search 校准报告已写入：{args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

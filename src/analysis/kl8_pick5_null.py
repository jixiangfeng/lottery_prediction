# -*- coding: utf-8 -*-
"""快乐8选5全流程均匀随机零假设模拟。"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, cast

import numpy as np
import pandas as pd

from src.analysis.kl8_pick5_probability_v1 import (
    EXPERT_NAMES,
    Kl8Pick5Config,
    _load_readonly_json,
    _segment_gate,
    _write_immutable_json,
    assert_canonical_formal_config,
    load_and_verify_kl8_report,
    payload_sha256,
    run_kl8_development,
)

NULL_SEED_BASE = 8_052_026
FORMAL_MIN_ITERATIONS = 5000
_GATE_INPUT_KEYS = {
    "deltaLogLossVsUniform",
    "deltaBrierVsUniform",
    "expectedPositiveDeviation",
    "exactPortfolioTotalHitsPValue",
    "meanHitsPerTicket",
    "meanPortfolioTotalHits",
    "blockBootstrap",
    "blockStability",
}
_BOOTSTRAP_KEYS = {"deltaLogLoss", "deltaBrier"}
_BOOTSTRAP_METRIC_KEYS = {"pValueMeanNonPositive"}
_BLOCK_STABILITY_KEYS = {
    "deltaLogLoss",
    "deltaBrier",
    "meanHitsPerTicket",
}


def _as_float(value: object) -> float:
    return float(cast(float, value))


def _json_number(value: object, field: str) -> int | float:
    if type(value) not in (int, float):
        raise ValueError(f"{field}必须为JSON数值")
    number = cast(int | float, value)
    if not math.isfinite(float(number)):
        raise ValueError(f"{field}必须为有限数")
    return number


def _bounded_json_number(
    value: object, field: str, minimum: float, maximum: float
) -> int | float:
    number = _json_number(value, field)
    if not minimum <= float(number) <= maximum:
        raise ValueError(f"{field}必须位于{minimum:g}..{maximum:g}")
    return number


def _canonical_gate_inputs(
    metrics: Mapping[str, object],
    config: Kl8Pick5Config,
    *,
    field: str,
    require_exact_keys: bool,
) -> dict[str, object]:
    """提取并验证生产门控唯一所需的规范JSON输入。"""

    if require_exact_keys and set(metrics) != _GATE_INPUT_KEYS:
        raise ValueError(f"{field}字段集合无效")
    try:
        bootstrap_source = metrics["blockBootstrap"]
        blocks_source = metrics["blockStability"]
    except KeyError as error:
        raise ValueError(f"{field}缺少门控输入") from error
    if not isinstance(bootstrap_source, Mapping):
        raise ValueError(f"{field}.blockBootstrap必须为对象")
    if require_exact_keys and set(bootstrap_source) != _BOOTSTRAP_KEYS:
        raise ValueError(f"{field}.blockBootstrap字段集合无效")

    bootstrap: dict[str, object] = {}
    for metric_name in ("deltaLogLoss", "deltaBrier"):
        raw_metric = bootstrap_source.get(metric_name)
        if not isinstance(raw_metric, Mapping):
            raise ValueError(f"{field}.blockBootstrap.{metric_name}必须为对象")
        if require_exact_keys and set(raw_metric) != _BOOTSTRAP_METRIC_KEYS:
            raise ValueError(f"{field}.blockBootstrap.{metric_name}字段集合无效")
        bootstrap[metric_name] = {
            "pValueMeanNonPositive": _bounded_json_number(
                raw_metric.get("pValueMeanNonPositive"),
                f"{field}.blockBootstrap.{metric_name}.pValueMeanNonPositive",
                0.0,
                1.0,
            )
        }

    if not isinstance(blocks_source, list):
        raise ValueError(f"{field}.blockStability必须为数组")
    if len(blocks_source) != config.stability_blocks:
        raise ValueError(f"{field}.blockStability必须恰好包含五块")
    blocks: list[dict[str, int | float]] = []
    for index, raw_block in enumerate(blocks_source):
        block_field = f"{field}.blockStability[{index}]"
        if not isinstance(raw_block, Mapping):
            raise ValueError(f"{block_field}必须为对象")
        if require_exact_keys and set(raw_block) != _BLOCK_STABILITY_KEYS:
            raise ValueError(f"{block_field}字段集合无效")
        blocks.append(
            {
                "deltaLogLoss": _bounded_json_number(
                    raw_block.get("deltaLogLoss"),
                    f"{block_field}.deltaLogLoss",
                    -20.0,
                    20.0,
                ),
                "deltaBrier": _bounded_json_number(
                    raw_block.get("deltaBrier"),
                    f"{block_field}.deltaBrier",
                    -1.0,
                    1.0,
                ),
                "meanHitsPerTicket": _bounded_json_number(
                    raw_block.get("meanHitsPerTicket"),
                    f"{block_field}.meanHitsPerTicket",
                    0.0,
                    5.0,
                ),
            }
        )

    return {
        "deltaLogLossVsUniform": _bounded_json_number(
            metrics.get("deltaLogLossVsUniform"),
            f"{field}.deltaLogLossVsUniform",
            -20.0,
            20.0,
        ),
        "deltaBrierVsUniform": _bounded_json_number(
            metrics.get("deltaBrierVsUniform"),
            f"{field}.deltaBrierVsUniform",
            -1.0,
            1.0,
        ),
        "expectedPositiveDeviation": _bounded_json_number(
            metrics.get("expectedPositiveDeviation"),
            f"{field}.expectedPositiveDeviation",
            0.0,
            60.0,
        ),
        "exactPortfolioTotalHitsPValue": _bounded_json_number(
            metrics.get("exactPortfolioTotalHitsPValue"),
            f"{field}.exactPortfolioTotalHitsPValue",
            0.0,
            1.0,
        ),
        "meanHitsPerTicket": _bounded_json_number(
            metrics.get("meanHitsPerTicket"),
            f"{field}.meanHitsPerTicket",
            0.0,
            5.0,
        ),
        "meanPortfolioTotalHits": _bounded_json_number(
            metrics.get("meanPortfolioTotalHits"),
            f"{field}.meanPortfolioTotalHits",
            0.0,
            25.0,
        ),
        "blockBootstrap": bootstrap,
        "blockStability": blocks,
    }


@dataclass(frozen=True)
class _NullTask:
    index: int
    seed: int
    config: Kl8Pick5Config
    history_periods: int
    frozen_periods_excluded: int


@dataclass(frozen=True)
class Kl8NullTrial:
    index: int
    seed: int
    search_passed: bool
    evaluation_passed: bool
    joint_passed: bool
    search_delta_logloss: float
    search_delta_brier: float
    search_mean_hits_per_ticket: float
    search_mean_portfolio_total_hits: float
    evaluation_delta_logloss: float
    evaluation_delta_brier: float
    evaluation_mean_hits_per_ticket: float
    evaluation_mean_portfolio_total_hits: float
    evaluation_portfolio_best_hit_at_least3_rate: float
    evaluation_portfolio_best_hit_at_least4_rate: float
    evaluation_portfolio_best_hit_exactly5_rate: float
    final_expert_weights: dict[str, float]
    search_gate_inputs: dict[str, object]
    evaluation_gate_inputs: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "index": self.index,
            "seed": self.seed,
            "searchPassed": self.search_passed,
            "evaluationPassed": self.evaluation_passed,
            "jointPassed": self.joint_passed,
            "searchDeltaLogLoss": self.search_delta_logloss,
            "searchDeltaBrier": self.search_delta_brier,
            "searchMeanHitsPerTicket": self.search_mean_hits_per_ticket,
            "searchMeanPortfolioTotalHits": self.search_mean_portfolio_total_hits,
            "evaluationDeltaLogLoss": self.evaluation_delta_logloss,
            "evaluationDeltaBrier": self.evaluation_delta_brier,
            "evaluationMeanHitsPerTicket": self.evaluation_mean_hits_per_ticket,
            "evaluationMeanPortfolioTotalHits": self.evaluation_mean_portfolio_total_hits,
            "evaluationPortfolioBestHitAtLeast3Rate": (
                self.evaluation_portfolio_best_hit_at_least3_rate
            ),
            "evaluationPortfolioBestHitAtLeast4Rate": (
                self.evaluation_portfolio_best_hit_at_least4_rate
            ),
            "evaluationPortfolioBestHitExactly5Rate": (
                self.evaluation_portfolio_best_hit_exactly5_rate
            ),
            "finalExpertWeights": self.final_expert_weights,
            "searchGateInputs": self.search_gate_inputs,
            "evaluationGateInputs": self.evaluation_gate_inputs,
        }
        payload["trialSha256"] = payload_sha256(payload)
        return payload


@dataclass(frozen=True)
class Kl8NullReport:
    reference_report_sha256: str
    config: Kl8Pick5Config
    history_periods: int
    frozen_periods_excluded: int
    iterations: int
    workers: int
    formal: bool
    trials: tuple[Kl8NullTrial, ...]
    new_completion_order: tuple[int, ...]
    checkpoint_used: bool
    observed_statistics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        trial_payloads = [trial.to_dict() for trial in self.trials]
        ordered_hash = payload_sha256(
            [trial["trialSha256"] for trial in trial_payloads]
        )
        joint_count = sum(trial.joint_passed for trial in self.trials)
        distributions = {
            expert: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
            }
            for expert in EXPERT_NAMES
            for values in (
                np.asarray(
                    [trial.final_expert_weights[expert] for trial in self.trials],
                    dtype=np.float64,
                ),
            )
        }
        statistic_values = {
            "evaluationDeltaLogLoss": [
                trial.evaluation_delta_logloss for trial in self.trials
            ],
            "evaluationDeltaBrier": [
                trial.evaluation_delta_brier for trial in self.trials
            ],
            "evaluationMeanHitsPerTicket": [
                trial.evaluation_mean_hits_per_ticket for trial in self.trials
            ],
            "evaluationMeanPortfolioTotalHits": [
                trial.evaluation_mean_portfolio_total_hits for trial in self.trials
            ],
            "evaluationPortfolioBestHitAtLeast3Rate": [
                trial.evaluation_portfolio_best_hit_at_least3_rate
                for trial in self.trials
            ],
            "evaluationPortfolioBestHitAtLeast4Rate": [
                trial.evaluation_portfolio_best_hit_at_least4_rate
                for trial in self.trials
            ],
            "evaluationPortfolioBestHitExactly5Rate": [
                trial.evaluation_portfolio_best_hit_exactly5_rate
                for trial in self.trials
            ],
        }
        empirical_p = {
            name: float(
                (1 + sum(value >= observed for value in statistic_values[name]))
                / (len(trial_payloads) + 1)
            )
            for name, observed in (
                (
                    "evaluationDeltaLogLoss",
                    self.observed_statistics["evaluationDeltaLogLoss"],
                ),
                (
                    "evaluationDeltaBrier",
                    self.observed_statistics["evaluationDeltaBrier"],
                ),
                (
                    "evaluationMeanHitsPerTicket",
                    self.observed_statistics["evaluationMeanHitsPerTicket"],
                ),
                (
                    "evaluationMeanPortfolioTotalHits",
                    self.observed_statistics["evaluationMeanPortfolioTotalHits"],
                ),
                (
                    "evaluationPortfolioBestHitAtLeast3Rate",
                    self.observed_statistics["evaluationPortfolioBestHitAtLeast3Rate"],
                ),
                (
                    "evaluationPortfolioBestHitAtLeast4Rate",
                    self.observed_statistics["evaluationPortfolioBestHitAtLeast4Rate"],
                ),
                (
                    "evaluationPortfolioBestHitExactly5Rate",
                    self.observed_statistics["evaluationPortfolioBestHitExactly5Rate"],
                ),
            )
        }
        payload: dict[str, object] = {
            "schemaVersion": "kl8_pick5_null_v1",
            "evidenceStatus": "formal_null" if self.formal else "smoke_only",
            "promotionPassed": False,
            "recommendationEnabled": False,
            "referenceReportSha256": self.reference_report_sha256,
            "config": asdict(self.config),
            "execution": {
                "iterations": self.iterations,
                "workers": self.workers,
                "historyPeriods": self.history_periods,
                "frozenPeriodsExcluded": self.frozen_periods_excluded,
                "fullPipelineReplayed": True,
                "checkpointUsed": self.checkpoint_used,
                "newCompletionOrder": list(self.new_completion_order),
            },
            "trials": trial_payloads,
            "orderedTrialSetSha256": ordered_hash,
            "summary": {
                "jointGatePassCount": joint_count,
                "empiricalFalsePositiveRate": joint_count / len(trial_payloads),
                "observedStatisticPValues": empirical_p,
                "finalExpertWeightDistributions": distributions,
                "nullSimulationPassed": bool(
                    self.formal
                    and self.iterations
                    >= max(FORMAL_MIN_ITERATIONS, self.config.required_null_iterations)
                    and joint_count / self.iterations <= self.config.alpha
                    and all(
                        value <= self.config.alpha for value in empirical_p.values()
                    )
                ),
            },
        }
        payload["reportSha256"] = payload_sha256(payload)
        return payload


def _random_history(periods: int, seed: int) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    start = pd.Timestamp("2000-01-01")
    return pd.DataFrame(
        {
            "issue": [str(900000000 + index) for index in range(periods)],
            "date": [
                (start + pd.Timedelta(days=index)).date().isoformat()
                for index in range(periods)
            ],
            "numbers": [
                sorted((generator.choice(80, size=20, replace=False) + 1).tolist())
                for _ in range(periods)
            ],
        }
    )


def _run_null_trial(task: _NullTask) -> Kl8NullTrial:
    report = run_kl8_development(
        _random_history(task.history_periods, task.seed),
        task.config,
        frozen_periods_excluded=task.frozen_periods_excluded,
    )
    search_metrics = cast(Mapping[str, object], report.search["metrics"])
    evaluation_metrics = cast(Mapping[str, object], report.evaluation["metrics"])
    search_gate = cast(Mapping[str, object], report.search["gate"])
    evaluation_gate = cast(Mapping[str, object], report.evaluation["gate"])
    search_gate_inputs = _canonical_gate_inputs(
        search_metrics,
        task.config,
        field="searchGateInputs",
        require_exact_keys=False,
    )
    evaluation_gate_inputs = _canonical_gate_inputs(
        evaluation_metrics,
        task.config,
        field="evaluationGateInputs",
        require_exact_keys=False,
    )
    search_passed = cast(bool, search_gate["passed"])
    evaluation_passed = cast(bool, evaluation_gate["passed"])
    if search_passed != bool(_segment_gate(search_gate_inputs, task.config)["passed"]):
        raise ValueError("Search生产门控结果与规范输入不一致")
    if evaluation_passed != bool(
        _segment_gate(evaluation_gate_inputs, task.config)["passed"]
    ):
        raise ValueError("Evaluation生产门控结果与规范输入不一致")
    return Kl8NullTrial(
        index=task.index,
        seed=task.seed,
        search_passed=search_passed,
        evaluation_passed=evaluation_passed,
        joint_passed=search_passed and evaluation_passed,
        search_delta_logloss=_as_float(search_metrics["deltaLogLossVsUniform"]),
        search_delta_brier=_as_float(search_metrics["deltaBrierVsUniform"]),
        search_mean_hits_per_ticket=_as_float(search_metrics["meanHitsPerTicket"]),
        search_mean_portfolio_total_hits=_as_float(
            search_metrics["meanPortfolioTotalHits"]
        ),
        evaluation_delta_logloss=_as_float(evaluation_metrics["deltaLogLossVsUniform"]),
        evaluation_delta_brier=_as_float(evaluation_metrics["deltaBrierVsUniform"]),
        evaluation_mean_hits_per_ticket=_as_float(
            evaluation_metrics["meanHitsPerTicket"]
        ),
        evaluation_mean_portfolio_total_hits=_as_float(
            evaluation_metrics["meanPortfolioTotalHits"]
        ),
        evaluation_portfolio_best_hit_at_least3_rate=_as_float(
            evaluation_metrics["portfolioBestHitAtLeast3Rate"]
        ),
        evaluation_portfolio_best_hit_at_least4_rate=_as_float(
            evaluation_metrics["portfolioBestHitAtLeast4Rate"]
        ),
        evaluation_portfolio_best_hit_exactly5_rate=_as_float(
            evaluation_metrics["portfolioBestHitExactly5Rate"]
        ),
        final_expert_weights=report.final_expert_weights,
        search_gate_inputs=search_gate_inputs,
        evaluation_gate_inputs=evaluation_gate_inputs,
    )


def _reference_identity(reference: Mapping[str, object]) -> str:
    expected = reference.get("reportSha256")
    unsigned = {key: value for key, value in reference.items() if key != "reportSha256"}
    if expected != payload_sha256(unsigned):
        raise ValueError("快乐8参考开发报告自哈希无效")
    return str(expected)


def build_formal_manifest(
    reference: Mapping[str, object],
    *,
    config: Kl8Pick5Config,
    history_periods: int,
    iterations: int,
    frozen_periods_excluded: int = 500,
) -> dict[str, object]:
    """在启动工作进程前验证正式迭代下限并构建身份清单。"""

    assert_canonical_formal_config(config)
    if frozen_periods_excluded != config.frozen_periods:
        raise ValueError("frozen_periods_excluded必须等于规范配置的500期")
    if iterations < max(FORMAL_MIN_ITERATIONS, config.required_null_iterations):
        raise ValueError("正式随机模拟至少5000次")
    return {
        "schemaVersion": "kl8_pick5_null_checkpoint_v1",
        "referenceReportSha256": _reference_identity(reference),
        "configSha256": payload_sha256(asdict(config)),
        "historyPeriods": history_periods,
        "frozenPeriodsExcluded": frozen_periods_excluded,
        "iterations": iterations,
        "seedBase": NULL_SEED_BASE,
    }


def _checkpoint_manifest(
    reference: Mapping[str, object],
    config: Kl8Pick5Config,
    history_periods: int,
    frozen_periods_excluded: int,
    iterations: int,
    formal: bool,
) -> dict[str, object]:
    if formal:
        manifest = build_formal_manifest(
            reference,
            config=config,
            history_periods=history_periods,
            iterations=iterations,
            frozen_periods_excluded=frozen_periods_excluded,
        )
    else:
        manifest = {
            "schemaVersion": "kl8_pick5_null_checkpoint_v1",
            "referenceReportSha256": _reference_identity(reference),
            "configSha256": payload_sha256(asdict(config)),
            "historyPeriods": history_periods,
            "frozenPeriodsExcluded": frozen_periods_excluded,
            "iterations": iterations,
            "seedBase": NULL_SEED_BASE,
        }
    manifest["formal"] = formal
    manifest["manifestSha256"] = payload_sha256(manifest)
    return manifest


def _load_trial(path: Path, task: _NullTask) -> Kl8NullTrial:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "index",
        "seed",
        "searchPassed",
        "evaluationPassed",
        "jointPassed",
        "searchDeltaLogLoss",
        "searchDeltaBrier",
        "searchMeanHitsPerTicket",
        "searchMeanPortfolioTotalHits",
        "evaluationDeltaLogLoss",
        "evaluationDeltaBrier",
        "evaluationMeanHitsPerTicket",
        "evaluationMeanPortfolioTotalHits",
        "evaluationPortfolioBestHitAtLeast3Rate",
        "evaluationPortfolioBestHitAtLeast4Rate",
        "evaluationPortfolioBestHitExactly5Rate",
        "finalExpertWeights",
        "searchGateInputs",
        "evaluationGateInputs",
        "trialSha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(f"检查点试验字段集合无效：{path}")
    trial_hash = payload.pop("trialSha256", None)
    try:
        valid_hash = trial_hash == payload_sha256(payload)
    except (TypeError, ValueError):
        valid_hash = False
    if type(payload["index"]) is not int:
        raise ValueError(f"检查点试验index必须为JSON整数：{path}")
    if type(payload["seed"]) is not int:
        raise ValueError(f"检查点试验seed必须为JSON整数：{path}")
    if (
        not valid_hash
        or payload.get("index") != task.index
        or payload.get("seed") != task.seed
    ):
        raise ValueError(f"检查点试验身份或哈希无效：{path}")
    if any(
        not isinstance(payload[name], bool)
        for name in ("searchPassed", "evaluationPassed", "jointPassed")
    ):
        raise ValueError(f"检查点试验门控字段必须为布尔值：{path}")
    search_passed = cast(bool, payload["searchPassed"])
    evaluation_passed = cast(bool, payload["evaluationPassed"])
    joint_passed = cast(bool, payload["jointPassed"])
    raw_search_gate_inputs = payload["searchGateInputs"]
    raw_evaluation_gate_inputs = payload["evaluationGateInputs"]
    if not isinstance(raw_search_gate_inputs, Mapping) or not isinstance(
        raw_evaluation_gate_inputs, Mapping
    ):
        raise ValueError(f"检查点试验门控输入必须为对象：{path}")
    search_gate_inputs = _canonical_gate_inputs(
        raw_search_gate_inputs,
        task.config,
        field="searchGateInputs",
        require_exact_keys=True,
    )
    evaluation_gate_inputs = _canonical_gate_inputs(
        raw_evaluation_gate_inputs,
        task.config,
        field="evaluationGateInputs",
        require_exact_keys=True,
    )
    derived_search_passed = bool(
        _segment_gate(search_gate_inputs, task.config)["passed"]
    )
    derived_evaluation_passed = bool(
        _segment_gate(evaluation_gate_inputs, task.config)["passed"]
    )
    if search_passed != derived_search_passed:
        raise ValueError(f"检查点试验Search门控派生结果无效：{path}")
    if evaluation_passed != derived_evaluation_passed:
        raise ValueError(f"检查点试验Evaluation门控派生结果无效：{path}")
    if joint_passed != (derived_search_passed and derived_evaluation_passed):
        raise ValueError(f"检查点试验联合门控派生结果无效：{path}")
    raw_weights_payload = payload["finalExpertWeights"]
    if not isinstance(raw_weights_payload, Mapping):
        raise ValueError(f"检查点试验专家权重必须为对象：{path}")
    raw_weights = dict(raw_weights_payload)
    if set(raw_weights) != set(EXPERT_NAMES):
        raise ValueError(f"检查点试验专家权重名称无效：{path}")
    if any(type(value) not in (int, float) for value in raw_weights.values()):
        raise ValueError(f"检查点试验专家权重必须为JSON数值：{path}")
    final_expert_weights = {
        str(key): float(cast(int | float, value)) for key, value in raw_weights.items()
    }
    values = np.asarray(list(final_expert_weights.values()), dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
        or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(f"检查点试验专家权重必须有限且和为1：{path}")
    delta_names = (
        "searchDeltaLogLoss",
        "searchDeltaBrier",
        "evaluationDeltaLogLoss",
        "evaluationDeltaBrier",
    )
    ticket_mean_names = (
        "searchMeanHitsPerTicket",
        "evaluationMeanHitsPerTicket",
    )
    portfolio_mean_names = (
        "searchMeanPortfolioTotalHits",
        "evaluationMeanPortfolioTotalHits",
    )
    rate_names = (
        "evaluationPortfolioBestHitAtLeast3Rate",
        "evaluationPortfolioBestHitAtLeast4Rate",
        "evaluationPortfolioBestHitExactly5Rate",
    )
    numeric_names = delta_names + ticket_mean_names + portfolio_mean_names + rate_names
    if any(type(payload[name]) not in (int, float) for name in numeric_names):
        raise ValueError(f"检查点试验统计量必须为JSON数值：{path}")
    statistics = {
        name: float(cast(int | float, payload[name])) for name in numeric_names
    }
    if not all(math.isfinite(value) for value in statistics.values()):
        raise ValueError(f"检查点试验统计量必须有限：{path}")
    if any(not 0.0 <= statistics[name] <= 5.0 for name in ticket_mean_names):
        raise ValueError(f"检查点试验每票均值必须位于0..5：{path}")
    if any(not 0.0 <= statistics[name] <= 25.0 for name in portfolio_mean_names):
        raise ValueError(f"检查点试验组合总命中均值必须位于0..25：{path}")
    if any(not 0.0 <= statistics[name] <= 1.0 for name in rate_names):
        raise ValueError(f"检查点试验统计量命中率必须位于0..1：{path}")
    gate_summary_pairs = (
        ("searchDeltaLogLoss", search_gate_inputs["deltaLogLossVsUniform"]),
        ("searchDeltaBrier", search_gate_inputs["deltaBrierVsUniform"]),
        ("searchMeanHitsPerTicket", search_gate_inputs["meanHitsPerTicket"]),
        (
            "searchMeanPortfolioTotalHits",
            search_gate_inputs["meanPortfolioTotalHits"],
        ),
        (
            "evaluationDeltaLogLoss",
            evaluation_gate_inputs["deltaLogLossVsUniform"],
        ),
        ("evaluationDeltaBrier", evaluation_gate_inputs["deltaBrierVsUniform"]),
        (
            "evaluationMeanHitsPerTicket",
            evaluation_gate_inputs["meanHitsPerTicket"],
        ),
        (
            "evaluationMeanPortfolioTotalHits",
            evaluation_gate_inputs["meanPortfolioTotalHits"],
        ),
    )
    if any(
        statistics[name] != float(cast(int | float, gate_value))
        for name, gate_value in gate_summary_pairs
    ):
        raise ValueError(f"检查点试验摘要统计与门控输入不一致：{path}")
    if not np.isclose(
        statistics["searchMeanPortfolioTotalHits"],
        statistics["searchMeanHitsPerTicket"] * 5.0,
        rtol=0.0,
        atol=1e-12,
    ) or not np.isclose(
        statistics["evaluationMeanPortfolioTotalHits"],
        statistics["evaluationMeanHitsPerTicket"] * 5.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"检查点试验每票均值与组合总命中均值不一致：{path}")
    if not (
        statistics["evaluationPortfolioBestHitAtLeast3Rate"]
        >= statistics["evaluationPortfolioBestHitAtLeast4Rate"]
        >= statistics["evaluationPortfolioBestHitExactly5Rate"]
    ):
        raise ValueError(f"检查点试验高命中率顺序不一致：{path}")
    return Kl8NullTrial(
        index=int(payload["index"]),
        seed=int(payload["seed"]),
        search_passed=search_passed,
        evaluation_passed=evaluation_passed,
        joint_passed=joint_passed,
        search_delta_logloss=float(payload["searchDeltaLogLoss"]),
        search_delta_brier=float(payload["searchDeltaBrier"]),
        search_mean_hits_per_ticket=statistics["searchMeanHitsPerTicket"],
        search_mean_portfolio_total_hits=statistics["searchMeanPortfolioTotalHits"],
        evaluation_delta_logloss=float(payload["evaluationDeltaLogLoss"]),
        evaluation_delta_brier=float(payload["evaluationDeltaBrier"]),
        evaluation_mean_hits_per_ticket=statistics["evaluationMeanHitsPerTicket"],
        evaluation_mean_portfolio_total_hits=statistics[
            "evaluationMeanPortfolioTotalHits"
        ],
        evaluation_portfolio_best_hit_at_least3_rate=statistics[
            "evaluationPortfolioBestHitAtLeast3Rate"
        ],
        evaluation_portfolio_best_hit_at_least4_rate=statistics[
            "evaluationPortfolioBestHitAtLeast4Rate"
        ],
        evaluation_portfolio_best_hit_exactly5_rate=statistics[
            "evaluationPortfolioBestHitExactly5Rate"
        ],
        final_expert_weights=final_expert_weights,
        search_gate_inputs=search_gate_inputs,
        evaluation_gate_inputs=evaluation_gate_inputs,
    )


def _prepare_checkpoint(directory: Path, manifest: dict[str, object]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise FileExistsError("检查点清单身份变化，禁止覆盖")
    else:
        _write_immutable_json(manifest, manifest_path, "快乐8 null检查点清单")
    trials = directory / "trials"
    trials.mkdir(parents=True, exist_ok=True)
    return trials


def _observed(reference: Mapping[str, object]) -> dict[str, float]:
    evaluation_section = cast(Mapping[str, object], reference["evaluation"])
    evaluation = cast(Mapping[str, object], evaluation_section["metrics"])
    return {
        "evaluationDeltaLogLoss": _as_float(evaluation["deltaLogLossVsUniform"]),
        "evaluationDeltaBrier": _as_float(evaluation["deltaBrierVsUniform"]),
        "evaluationMeanHitsPerTicket": _as_float(evaluation["meanHitsPerTicket"]),
        "evaluationMeanPortfolioTotalHits": _as_float(
            evaluation["meanPortfolioTotalHits"]
        ),
        "evaluationPortfolioBestHitAtLeast3Rate": _as_float(
            evaluation["portfolioBestHitAtLeast3Rate"]
        ),
        "evaluationPortfolioBestHitAtLeast4Rate": _as_float(
            evaluation["portfolioBestHitAtLeast4Rate"]
        ),
        "evaluationPortfolioBestHitExactly5Rate": _as_float(
            evaluation["portfolioBestHitExactly5Rate"]
        ),
    }


def _run_simulation(
    reference: Mapping[str, object],
    *,
    config: Kl8Pick5Config,
    history_periods: int,
    frozen_periods_excluded: int,
    iterations: int,
    workers: int,
    checkpoint_dir: str | Path,
    formal: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> Kl8NullReport:
    if iterations <= 0:
        raise ValueError("随机模拟次数必须为正整数")
    if formal and iterations < max(
        FORMAL_MIN_ITERATIONS, config.required_null_iterations
    ):
        raise ValueError("正式随机模拟至少5000次")
    if history_periods != config.required_periods:
        raise ValueError("随机模拟历史长度必须与锁定开发流程完全一致")
    if frozen_periods_excluded != config.frozen_periods:
        raise ValueError("frozen_periods_excluded必须等于配置的500期")
    reference_hash = _reference_identity(reference)
    manifest = _checkpoint_manifest(
        reference, config, history_periods, frozen_periods_excluded, iterations, formal
    )
    trial_directory = _prepare_checkpoint(Path(checkpoint_dir), manifest)
    tasks = [
        _NullTask(
            index,
            NULL_SEED_BASE + index,
            config,
            history_periods,
            frozen_periods_excluded,
        )
        for index in range(iterations)
    ]
    completed: dict[int, Kl8NullTrial] = {}
    completion_order: list[int] = []
    pending: list[_NullTask] = []
    for task in tasks:
        path = trial_directory / f"trial_{task.index:06d}.json"
        if path.exists():
            completed[task.index] = _load_trial(path, task)
        else:
            pending.append(task)
    processed = len(completed)
    if workers == 1:
        for task in pending:
            trial = _run_null_trial(task)
            _write_immutable_json(
                trial.to_dict(),
                trial_directory / f"trial_{task.index:06d}.json",
                "快乐8 null试验",
            )
            completed[task.index] = trial
            completion_order.append(task.index)
            processed += 1
            if progress_callback:
                progress_callback(processed, iterations)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        try:
            futures = {executor.submit(_run_null_trial, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    trial = future.result()
                except BaseException as error:
                    raise RuntimeError(
                        "快乐8 null进程执行失败，失败关闭且不回退线程"
                    ) from error
                _write_immutable_json(
                    trial.to_dict(),
                    trial_directory / f"trial_{task.index:06d}.json",
                    "快乐8 null试验",
                )
                completed[task.index] = trial
                completion_order.append(task.index)
                processed += 1
                if progress_callback:
                    progress_callback(processed, iterations)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    ordered = tuple(completed[index] for index in range(iterations))
    return Kl8NullReport(
        reference_report_sha256=reference_hash,
        config=config,
        history_periods=history_periods,
        frozen_periods_excluded=frozen_periods_excluded,
        iterations=iterations,
        workers=workers,
        formal=formal,
        trials=ordered,
        new_completion_order=tuple(completion_order),
        checkpoint_used=True,
        observed_statistics=_observed(reference),
    )


def run_kl8_null_smoke(
    reference: Mapping[str, object],
    *,
    config: Kl8Pick5Config,
    history_periods: int,
    frozen_periods_excluded: int,
    iterations: int,
    workers: int,
    checkpoint_dir: str | Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Kl8NullReport:
    """执行少量全流程smoke；永远不可晋级。"""

    return _run_simulation(
        reference,
        config=config,
        history_periods=history_periods,
        frozen_periods_excluded=frozen_periods_excluded,
        iterations=iterations,
        workers=workers,
        checkpoint_dir=checkpoint_dir,
        formal=False,
        progress_callback=progress_callback,
    )


def run_formal_kl8_null(
    protocol_path: str | Path,
    report_path: str | Path,
    locked_history: pd.DataFrame,
    *,
    config: Kl8Pick5Config,
    frozen_periods_excluded: int,
    frozen_boundary: Mapping[str, object],
    iterations: int,
    workers: int,
    checkpoint_dir: str | Path,
) -> Kl8NullReport:
    """从只读协议/报告和锁定DataFrame启动至少5000次正式模拟。"""

    assert_canonical_formal_config(config)
    if frozen_periods_excluded != config.frozen_periods:
        raise ValueError("frozen_periods_excluded必须等于规范配置的500期")
    if iterations < max(FORMAL_MIN_ITERATIONS, config.required_null_iterations):
        raise ValueError("正式随机模拟至少5000次")

    reference = load_and_verify_kl8_report(
        report_path,
        protocol_path,
        locked_history,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
        frozen_boundary=frozen_boundary,
    )
    protocol = _load_readonly_json(protocol_path, "快乐8开发协议")
    boundary = cast(Mapping[str, object], protocol["frozenBoundary"])
    development = cast(Mapping[str, object], protocol["developmentData"])
    if (
        boundary.get("periodsExcluded") != frozen_periods_excluded
        or boundary.get("firstIssue") != frozen_boundary.get("firstIssue")
        or boundary.get("lastIssue") != frozen_boundary.get("lastIssue")
        or boundary.get("numbersRead") is not False
        or development.get("periods") != len(locked_history)
    ):
        raise ValueError("正式null边界或历史身份与协议不一致")
    return _run_simulation(
        reference,
        config=config,
        history_periods=len(locked_history),
        frozen_periods_excluded=frozen_periods_excluded,
        iterations=iterations,
        workers=workers,
        checkpoint_dir=checkpoint_dir,
        formal=True,
        progress_callback=None,
    )


def write_kl8_null_report(report: Kl8NullReport, path: str | Path) -> Path:
    """不可覆盖写入自哈希null报告。"""

    return _write_immutable_json(report.to_dict(), path, "快乐8 null报告")


__all__ = [
    "FORMAL_MIN_ITERATIONS",
    "Kl8NullReport",
    "Kl8NullTrial",
    "build_formal_manifest",
    "run_formal_kl8_null",
    "run_kl8_null_smoke",
    "write_kl8_null_report",
]

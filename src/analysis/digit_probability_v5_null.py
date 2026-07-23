# -*- coding: utf-8 -*-
"""probability_v5全流程均匀随机模拟，支持确定性串行与进程并行。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.digit_probability_v5 import (
    ProbabilityV5DevelopmentConfig,
    _load_readonly_json,
    _protocol_identity,
    _write_immutable_content,
    load_and_verify_probability_v5_protocol,
    load_and_verify_probability_v5_report,
    run_probability_v5_development,
)
from src.lotteries import get_lottery_rule

_EMPIRICAL_SIGNIFICANCE_LEVEL = 0.01


def _payload_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}格式无效")
    return value


def _float_field(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"开发报告缺少数值字段：{key}")
    return float(value)


def _int_field(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"开发报告缺少整数字段：{key}")
    return value


def _float_value(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label}必须是数值")
    return float(value)


@dataclass(frozen=True)
class ProbabilityV5NullTrial:
    index: int
    seed: int
    selected_temperature: float
    mean_delta_log_loss: float
    mean_delta_brier: float
    policy_top_k_hits: int
    strict_statistical_gate_passed: bool
    final_expert_weights: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "index": self.index,
            "seed": self.seed,
            "selectedTemperature": self.selected_temperature,
            "meanDeltaLogLoss": self.mean_delta_log_loss,
            "meanDeltaBrier": self.mean_delta_brier,
            "policyTopKHits": self.policy_top_k_hits,
            "strictStatisticalGatePassed": self.strict_statistical_gate_passed,
            "finalExpertWeights": dict(self.final_expert_weights),
        }
        payload["trialSha256"] = _payload_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProbabilityV5NullTrial":
        document = dict(payload)
        claimed = document.pop("trialSha256", None)
        if claimed != _payload_sha256(document):
            raise ValueError("检查点trialSha256无效")
        weights = _mapping(payload.get("finalExpertWeights"), "检查点专家权重")
        gate = payload.get("strictStatisticalGatePassed")
        if not isinstance(gate, bool):
            raise ValueError("检查点统计闸门字段无效")
        return cls(
            index=_int_field(payload, "index"),
            seed=_int_field(payload, "seed"),
            selected_temperature=_float_field(payload, "selectedTemperature"),
            mean_delta_log_loss=_float_field(payload, "meanDeltaLogLoss"),
            mean_delta_brier=_float_field(payload, "meanDeltaBrier"),
            policy_top_k_hits=_int_field(payload, "policyTopKHits"),
            strict_statistical_gate_passed=gate,
            final_expert_weights=tuple(
                (str(name), _float_value(value, f"检查点专家权重{name}"))
                for name, value in weights.items()
            ),
        )


@dataclass(frozen=True)
class _NullTask:
    lottery: str
    config: ProbabilityV5DevelopmentConfig
    history_periods: int
    frozen_periods_excluded: int
    index: int


@dataclass(frozen=True)
class ProbabilityV5NullSimulationReport:
    lottery: str
    config: ProbabilityV5DevelopmentConfig
    protocol_identity: dict[str, object] | None
    reference_report_sha256: str
    reference_statistics: dict[str, object]
    formal: bool
    workers: int
    checkpoint_used: bool
    trials: tuple[ProbabilityV5NullTrial, ...]
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        trial_payloads = [trial.to_dict() for trial in self.trials]
        checkpoint_trial_set_sha256 = _payload_sha256(
            [payload["trialSha256"] for payload in trial_payloads]
        )
        payload: dict[str, object] = {
            "schemaVersion": "probability_v5_null_simulation_v1",
            "modelVersion": "probability_v5",
            "evaluationKind": "full_pipeline_uniform_null_simulation",
            "evidenceStatus": (
                "formal_development_null_simulation" if self.formal else "smoke_only"
            ),
            "lottery": self.lottery,
            "configSha256": _payload_sha256(self.config.to_dict()),
            "protocolIdentity": self.protocol_identity,
            "referenceReportSha256": self.reference_report_sha256,
            "execution": {
                "status": "complete" if self.formal else "smoke",
                "formal": self.formal,
                "workers": self.workers,
                "iterations": len(self.trials),
                "requiredIterations": self.config.required_null_simulations,
                "fullPipelineReplayed": True,
                "deterministicSeedPerTrial": True,
                "frozenRead": False,
                "checkpointRequiredForFormal": True,
                "checkpointUsed": self.checkpoint_used,
                "checkpointTrialSetSha256": checkpoint_trial_set_sha256,
            },
            "referenceStatistics": self.reference_statistics,
            "summary": self.summary,
            "trials": trial_payloads,
            "promotionPassed": False,
            "researchRankingEnabled": False,
            "recommendationEnabled": False,
            "promotionReasons": [
                "新的独立500期Validation尚未打开",
                *(
                    []
                    if self.summary["nullSimulationPassed"] is True
                    else ["全流程随机模拟闸门未通过或仅执行smoke"]
                ),
            ],
        }
        payload["reportSha256"] = _payload_sha256(payload)
        return payload


def _trial_seed(base_seed: int, index: int) -> int:
    sequence = np.random.SeedSequence([base_seed, index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _synthetic_uniform_history(periods: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 1000, size=periods, dtype=np.int16)
    return pd.DataFrame(
        {
            "期数": [str(index + 1) for index in range(periods)],
            "百位": values // 100,
            "十位": (values // 10) % 10,
            "个位": values % 10,
        }
    )


def _run_null_trial(task: _NullTask) -> ProbabilityV5NullTrial:
    seed = _trial_seed(task.config.random_seed, task.index)
    history = _synthetic_uniform_history(task.history_periods, seed)
    report = run_probability_v5_development(
        history,
        get_lottery_rule(task.lottery),
        task.config,
        frozen_periods_excluded=task.frozen_periods_excluded,
        include_period_details=False,
    ).to_dict()
    search = _mapping(report.get("search"), "随机模拟Search")
    evaluation = _mapping(report.get("evaluation"), "随机模拟Evaluation")
    weights = _mapping(
        evaluation.get("postFinalUpdateExpertWeights"), "最终更新后专家权重"
    )
    return ProbabilityV5NullTrial(
        index=task.index,
        seed=seed,
        selected_temperature=_float_field(report, "selectedTemperature"),
        mean_delta_log_loss=_float_field(evaluation, "meanDeltaLogLoss"),
        mean_delta_brier=_float_field(evaluation, "meanDeltaBrier"),
        policy_top_k_hits=_int_field(evaluation, "policyTopKHits"),
        strict_statistical_gate_passed=(
            search.get("strictStatisticalGatePassed") is True
            and evaluation.get("strictStatisticalGatePassed") is True
        ),
        final_expert_weights=tuple(
            (str(name), _float_value(value, f"专家权重{name}"))
            for name, value in weights.items()
        ),
    )


def _checkpoint_manifest(
    reference_report: Mapping[str, object],
    *,
    lottery: str,
    config: ProbabilityV5DevelopmentConfig,
    history_periods: int,
    frozen_periods_excluded: int,
    iterations: int,
    formal: bool,
    protocol_identity: Mapping[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schemaVersion": "probability_v5_null_checkpoint_v1",
        "modelVersion": "probability_v5",
        "lottery": lottery,
        "configSha256": _payload_sha256(config.to_dict()),
        "referenceReportSha256": reference_report["reportSha256"],
        "protocolIdentity": protocol_identity,
        "historyPeriods": history_periods,
        "frozenPeriodsExcluded": frozen_periods_excluded,
        "iterations": iterations,
        "formal": formal,
        "baseSeed": config.random_seed,
        "immutable": True,
    }
    payload["checkpointSha256"] = _payload_sha256(payload)
    return payload


def _write_immutable_json(
    payload: Mapping[str, object], path: Path, label: str
) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _write_immutable_content(path, content, label=label)


def _load_checkpoint_trials(
    checkpoint_dir: Path,
    manifest: Mapping[str, object],
    config: ProbabilityV5DevelopmentConfig,
    iterations: int,
) -> dict[int, ProbabilityV5NullTrial]:
    _write_immutable_json(manifest, checkpoint_dir / "manifest.json", "随机模拟检查点")
    checkpoint_sha256 = str(manifest["checkpointSha256"])
    trial_dir = checkpoint_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[int, ProbabilityV5NullTrial] = {}
    for path in sorted(trial_dir.glob("trial_*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"随机模拟检查点格式无效：{path}")
        if document.get("checkpointSha256") != checkpoint_sha256:
            raise RuntimeError(f"随机模拟检查点身份不匹配：{path}")
        trial = ProbabilityV5NullTrial.from_dict(
            _mapping(document.get("trial"), "随机模拟检查点trial")
        )
        if not 0 <= trial.index < iterations:
            raise RuntimeError(f"随机模拟检查点编号越界：{path}")
        if trial.seed != _trial_seed(config.random_seed, trial.index):
            raise RuntimeError(f"随机模拟检查点种子不匹配：{path}")
        if trial.index in loaded:
            raise RuntimeError(f"随机模拟检查点编号重复：{trial.index}")
        loaded[trial.index] = trial
    return loaded


def _write_checkpoint_trial(
    checkpoint_dir: Path,
    checkpoint_sha256: str,
    trial: ProbabilityV5NullTrial,
) -> None:
    payload: dict[str, object] = {
        "checkpointSha256": checkpoint_sha256,
        "trial": trial.to_dict(),
    }
    path = checkpoint_dir / "trials" / f"trial_{trial.index:05d}.json"
    _write_immutable_json(payload, path, "随机模拟试验检查点")


def _validate_reference_report(
    reference: Mapping[str, object],
    lottery: str,
    config: ProbabilityV5DevelopmentConfig,
    *,
    formal: bool,
    protocol_payload: Mapping[str, object] | None,
    frozen_periods_excluded: int,
    history_periods: int,
) -> tuple[Mapping[str, object], dict[str, object]]:
    reference_payload = dict(reference)
    document = dict(reference_payload)
    claimed_report_sha256 = document.pop("reportSha256", None)
    if claimed_report_sha256 != _payload_sha256(document):
        raise ValueError("开发报告reportSha256无效")
    if reference_payload.get("schemaVersion") != "probability_v5_development_v1":
        raise ValueError("开发报告schemaVersion无效")
    if reference_payload.get("modelVersion") != "probability_v5":
        raise ValueError("随机模拟必须绑定probability_v5开发报告")
    if reference_payload.get("evaluationKind") != "development_prequential_challenger":
        raise ValueError("开发报告evaluationKind无效")
    if reference_payload.get("lottery") != lottery:
        raise ValueError("随机模拟玩法与开发报告不一致")
    if reference_payload.get("configSha256") != _payload_sha256(config.to_dict()):
        raise ValueError("随机模拟配置与开发报告不一致")
    if reference_payload.get("config") != config.to_dict():
        raise ValueError("随机模拟完整配置与开发报告不一致")
    protocol = _mapping(reference_payload.get("protocol"), "开发报告protocol")
    if protocol.get("frozenRead") is not False:
        raise ValueError("随机模拟参考报告不得读取Frozen")
    if formal:
        if protocol_payload is None:
            raise ValueError("正式随机模拟必须绑定只读开发协议文件")
        identity = _protocol_identity(protocol_payload)
        development_data = _mapping(
            protocol_payload.get("developmentData"), "开发协议developmentData"
        )
        if reference_payload.get("dataSha256") != identity["dataSha256"]:
            raise ValueError("正式随机模拟开发数据身份不一致")
        if reference_payload.get("sourceFingerprint") != identity["sourceFingerprint"]:
            raise ValueError("正式随机模拟源码身份不一致")
        if reference_payload.get("protocolIdentity") != identity:
            raise ValueError("正式随机模拟协议身份不一致")
        if protocol.get("developmentProtocolRegistered") is not True:
            raise ValueError("正式随机模拟参考报告未登记开发协议")
        if protocol.get("developmentProtocolSha256") != identity["protocolSha256"]:
            raise ValueError("正式随机模拟协议与开发报告不一致")
        if frozen_periods_excluded != identity["frozenPeriodsExcluded"]:
            raise ValueError("正式随机模拟Frozen边界与开发协议不一致")
        if protocol.get("frozenPeriodsExcluded") != identity["frozenPeriodsExcluded"]:
            raise ValueError("开发报告Frozen边界与开发协议不一致")
        if history_periods != development_data.get("periods"):
            raise ValueError("正式随机模拟历史期数与开发协议不一致")
    search = _mapping(reference_payload.get("search"), "开发报告Search")
    evaluation = _mapping(reference_payload.get("evaluation"), "开发报告Evaluation")
    derived = _mapping(reference_payload.get("derivedGates"), "开发报告派生闸门")
    search_passed = search.get("strictStatisticalGatePassed") is True
    evaluation_passed = evaluation.get("strictStatisticalGatePassed") is True
    joint_passed = search_passed and evaluation_passed
    expected_derived = {
        "searchPassed": search_passed,
        "evaluationPassed": evaluation_passed,
        "searchAndEvaluationPassed": joint_passed,
    }
    if dict(derived) != expected_derived:
        raise ValueError("开发报告派生闸门不一致")
    if reference_payload.get("developmentSignalsPassed") is not joint_passed:
        raise ValueError("开发报告developmentSignalsPassed派生值不一致")
    statistics: dict[str, object] = {
        "developmentSignalsPassed": joint_passed,
        "meanDeltaLogLoss": _float_field(evaluation, "meanDeltaLogLoss"),
        "meanDeltaBrier": _float_field(evaluation, "meanDeltaBrier"),
        "policyTopKHits": _int_field(evaluation, "policyTopKHits"),
    }
    return evaluation, statistics


def _run_probability_v5_null_simulation(
    reference_report: Mapping[str, object],
    *,
    lottery: str,
    config: ProbabilityV5DevelopmentConfig,
    history_periods: int,
    frozen_periods_excluded: int,
    iterations: int,
    workers: int = 1,
    formal: bool = False,
    protocol_payload: Mapping[str, object] | None = None,
    checkpoint_dir: str | Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ProbabilityV5NullSimulationReport:
    """在均匀随机历史上完整重放专家、聚合、Calibration和联合闸门。"""

    if lottery not in {"fc3d", "pl3"}:
        raise ValueError("probability_v5随机模拟只支持fc3d/pl3")
    if iterations <= 0 or workers <= 0:
        raise ValueError("iterations和workers必须为正")
    if formal and iterations != 5000:
        raise ValueError("正式随机模拟必须执行5000次")
    if not formal and iterations >= config.required_null_simulations:
        raise ValueError("随机模拟smoke次数必须少于正式预注册次数")
    if formal and checkpoint_dir is None:
        raise ValueError("正式随机模拟必须提供检查点目录")
    required = config.warmup_history + config.required_prediction_periods
    if history_periods < required:
        raise ValueError(f"随机历史至少需要{required}期")
    reference_payload = dict(reference_report)
    _, reference_statistics = _validate_reference_report(
        reference_report,
        lottery,
        config,
        formal=formal,
        protocol_payload=protocol_payload,
        frozen_periods_excluded=frozen_periods_excluded,
        history_periods=history_periods,
    )
    completed: dict[int, ProbabilityV5NullTrial] = {}
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        manifest = _checkpoint_manifest(
            reference_payload,
            lottery=lottery,
            config=config,
            history_periods=history_periods,
            frozen_periods_excluded=frozen_periods_excluded,
            iterations=iterations,
            formal=formal,
            protocol_identity=(
                _protocol_identity(protocol_payload)
                if protocol_payload is not None
                else None
            ),
        )
        checkpoint_sha256 = str(manifest["checkpointSha256"])
        completed = _load_checkpoint_trials(
            checkpoint_path, manifest, config, iterations
        )
        if completed and progress_callback is not None:
            progress_callback(len(completed), iterations)
    tasks = tuple(
        _NullTask(
            lottery=lottery,
            config=config,
            history_periods=history_periods,
            frozen_periods_excluded=frozen_periods_excluded,
            index=index,
        )
        for index in range(iterations)
        if index not in completed
    )
    iterator: Iterable[ProbabilityV5NullTrial]
    executor: ProcessPoolExecutor | None
    if workers == 1:
        iterator = map(_run_null_trial, tasks)
        executor = None
    else:
        executor = None
        try:
            executor = ProcessPoolExecutor(max_workers=workers)
            futures = [executor.submit(_run_null_trial, task) for task in tasks]
            iterator = (future.result() for future in as_completed(futures))
        except Exception as exc:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError("随机模拟进程池执行失败，已关闭失败") from exc
    trials_by_index = dict(completed)
    try:
        for trial in iterator:
            if checkpoint_path is not None and checkpoint_sha256 is not None:
                _write_checkpoint_trial(checkpoint_path, checkpoint_sha256, trial)
            trials_by_index[trial.index] = trial
            if progress_callback is not None:
                progress_callback(len(trials_by_index), iterations)
    except Exception as exc:
        if executor is None:
            raise
        raise RuntimeError("随机模拟进程池执行失败，已关闭失败") from exc
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    trials = [trials_by_index[index] for index in range(iterations)]
    if checkpoint_path is not None and checkpoint_sha256 is not None:
        trial_hashes = [trial.to_dict()["trialSha256"] for trial in trials]
        checkpoint_trial_set: dict[str, object] = {
            "schemaVersion": "probability_v5_null_checkpoint_trial_set_v1",
            "checkpointSha256": checkpoint_sha256,
            "trialSha256s": trial_hashes,
            "trialSetSha256": _payload_sha256(trial_hashes),
            "complete": True,
        }
        _write_immutable_json(
            checkpoint_trial_set,
            checkpoint_path / "trial_set.json",
            "随机模拟检查点集合",
        )
    reference_log = _float_field(reference_statistics, "meanDeltaLogLoss")
    reference_brier = _float_field(reference_statistics, "meanDeltaBrier")
    reference_hits = _int_field(reference_statistics, "policyTopKHits")
    gate_passes = sum(trial.strict_statistical_gate_passed for trial in trials)

    def empirical_p_value(values: list[float], reference_value: float) -> float:
        return (1 + sum(value >= reference_value for value in values)) / (
            1 + len(values)
        )

    empirical_log = empirical_p_value(
        [trial.mean_delta_log_loss for trial in trials], reference_log
    )
    empirical_brier = empirical_p_value(
        [trial.mean_delta_brier for trial in trials], reference_brier
    )
    empirical_top_k = empirical_p_value(
        [float(trial.policy_top_k_hits) for trial in trials], float(reference_hits)
    )
    false_positive_rate = gate_passes / len(trials)
    null_passed = (
        formal
        and reference_statistics["developmentSignalsPassed"] is True
        and false_positive_rate < _EMPIRICAL_SIGNIFICANCE_LEVEL
        and empirical_log < _EMPIRICAL_SIGNIFICANCE_LEVEL
        and empirical_brier < _EMPIRICAL_SIGNIFICANCE_LEVEL
        and empirical_top_k < _EMPIRICAL_SIGNIFICANCE_LEVEL
    )
    expert_names = tuple(name for name, _ in trials[0].final_expert_weights)
    if any(
        tuple(name for name, _ in trial.final_expert_weights) != expert_names
        for trial in trials
    ):
        raise RuntimeError("随机模拟试验的专家权重字段不一致")
    expert_weight_distribution = {}
    for expert_name in expert_names:
        values = np.asarray(
            [dict(trial.final_expert_weights)[expert_name] for trial in trials],
            dtype=float,
        )
        expert_weight_distribution[expert_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "minimum": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.5)),
            "q75": float(np.quantile(values, 0.75)),
            "maximum": float(np.max(values)),
        }
    summary: dict[str, object] = {
        "strictGatePasses": gate_passes,
        "strictGateFalsePositiveRate": false_positive_rate,
        "empiricalPValueDeltaLogLoss": empirical_log,
        "empiricalPValueDeltaBrier": empirical_brier,
        "empiricalPValuePolicyTopK": empirical_top_k,
        "significanceLevel": _EMPIRICAL_SIGNIFICANCE_LEVEL,
        "expertWeightDistribution": expert_weight_distribution,
        "nullSimulationPassed": null_passed,
    }
    return ProbabilityV5NullSimulationReport(
        lottery=lottery,
        config=config,
        protocol_identity=(
            _protocol_identity(protocol_payload)
            if protocol_payload is not None
            else None
        ),
        reference_report_sha256=str(reference_payload["reportSha256"]),
        reference_statistics=reference_statistics,
        formal=formal,
        workers=workers,
        checkpoint_used=checkpoint_dir is not None,
        trials=tuple(trials),
        summary=summary,
    )


def run_probability_v5_null_simulation(
    reference_report: Mapping[str, object],
    *,
    lottery: str,
    config: ProbabilityV5DevelopmentConfig,
    history_periods: int,
    frozen_periods_excluded: int,
    iterations: int,
    workers: int = 1,
    checkpoint_dir: str | Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ProbabilityV5NullSimulationReport:
    """执行非正式smoke；该入口永远不绑定或声明已登记协议。"""

    return _run_probability_v5_null_simulation(
        reference_report,
        lottery=lottery,
        config=config,
        history_periods=history_periods,
        frozen_periods_excluded=frozen_periods_excluded,
        iterations=iterations,
        workers=workers,
        formal=False,
        protocol_payload=None,
        checkpoint_dir=checkpoint_dir,
        progress_callback=progress_callback,
    )


def _load_formal_inputs(
    protocol_path: str | Path,
    reference_report_path: str | Path,
    history: pd.DataFrame,
    *,
    lottery: str,
    config: ProbabilityV5DevelopmentConfig,
    frozen_periods_excluded: int,
) -> tuple[dict[str, object], dict[str, object]]:
    rule = get_lottery_rule(lottery)
    protocol_document = _load_readonly_json(protocol_path, "probability_v5开发协议")
    frozen_boundary = _mapping(
        protocol_document.get("frozenBoundary"), "开发协议frozenBoundary"
    )
    development_data = _mapping(
        protocol_document.get("developmentData"), "开发协议developmentData"
    )
    if frozen_periods_excluded != frozen_boundary.get("periodsExcluded"):
        raise ValueError("正式随机模拟Frozen边界与开发协议不一致")
    if len(history) != development_data.get("periods"):
        raise ValueError("正式随机模拟历史期数与开发协议不一致")
    protocol = load_and_verify_probability_v5_protocol(
        protocol_path,
        history,
        rule,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
    )
    reference = load_and_verify_probability_v5_report(
        reference_report_path,
        protocol_path,
        history,
        rule,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
    )
    _validate_reference_report(
        reference,
        lottery,
        config,
        formal=True,
        protocol_payload=protocol,
        frozen_periods_excluded=frozen_periods_excluded,
        history_periods=len(history),
    )
    return protocol, reference


def _formal_checkpoint_manifest(
    protocol_path: str | Path,
    reference_report_path: str | Path,
    history: pd.DataFrame,
    *,
    lottery: str,
    config: ProbabilityV5DevelopmentConfig,
    frozen_periods_excluded: int,
    iterations: int,
) -> dict[str, object]:
    if iterations != 5000:
        raise ValueError("正式随机模拟必须执行5000次")
    protocol, reference = _load_formal_inputs(
        protocol_path,
        reference_report_path,
        history,
        lottery=lottery,
        config=config,
        frozen_periods_excluded=frozen_periods_excluded,
    )
    return _checkpoint_manifest(
        reference,
        lottery=lottery,
        config=config,
        history_periods=len(history),
        frozen_periods_excluded=frozen_periods_excluded,
        iterations=iterations,
        formal=True,
        protocol_identity=_protocol_identity(protocol),
    )


def run_formal_probability_v5_null_simulation(
    protocol_path: str | Path,
    reference_report_path: str | Path,
    history: pd.DataFrame,
    *,
    lottery: str,
    config: ProbabilityV5DevelopmentConfig,
    frozen_periods_excluded: int,
    iterations: int,
    workers: int = 1,
    checkpoint_dir: str | Path | None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ProbabilityV5NullSimulationReport:
    """从只读协议/报告文件重算验证后执行正式5000次null。"""

    if iterations != 5000:
        raise ValueError("正式随机模拟必须执行5000次")
    if checkpoint_dir is None:
        raise ValueError("正式随机模拟必须提供检查点目录")
    protocol, reference = _load_formal_inputs(
        protocol_path,
        reference_report_path,
        history,
        lottery=lottery,
        config=config,
        frozen_periods_excluded=frozen_periods_excluded,
    )
    return _run_probability_v5_null_simulation(
        reference,
        lottery=lottery,
        config=config,
        history_periods=len(history),
        frozen_periods_excluded=frozen_periods_excluded,
        iterations=iterations,
        workers=workers,
        formal=True,
        protocol_payload=protocol,
        checkpoint_dir=checkpoint_dir,
        progress_callback=progress_callback,
    )


def write_probability_v5_null_report(
    report: ProbabilityV5NullSimulationReport, path: str | Path
) -> Path:
    """只写一次保存随机模拟报告。"""

    destination = Path(path)
    _write_immutable_json(report.to_dict(), destination, "probability_v5随机模拟报告")
    return destination


__all__ = [
    "ProbabilityV5NullSimulationReport",
    "ProbabilityV5NullTrial",
    "run_formal_probability_v5_null_simulation",
    "run_probability_v5_null_simulation",
    "write_probability_v5_null_report",
]

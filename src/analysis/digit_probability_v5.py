# -*- coding: utf-8 -*-
"""三位彩 probability_v5 开发模式：预注册专家、在线聚合与双口径评估。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.digit_daily_policy import select_daily_candidates
from src.analysis.digit_data import (
    canonical_digit_data_sha256,
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_learned_features import (
    FEATURE_NAMES,
    LearnedFeatureConfig,
    build_candidate_features,
    iter_rolling_history_states,
)
from src.analysis.digit_learned_ranker import rank_candidate_indices
from src.analysis.digit_online_gradient import (
    OnlineGradientCandidate,
    OnlineGradientConfig,
    _initial_weights,
    online_gradient_step,
)
from src.analysis.prediction_viability import evaluate_viability_metric
from src.lotteries.base import LotteryRule

_CANDIDATES = tuple(f"{value:03d}" for value in range(1000))
_CANDIDATE_DIGITS = np.asarray(
    [(value // 100, (value // 10) % 10, value % 10) for value in range(1000)],
    dtype=np.int16,
)
_EXPERT_NAMES = (
    "uniform",
    "ewma_position",
    "ewma_pairwise",
    "legacy_gradient",
)
_UNIFORM_LOG_LOSS = math.log(1000.0)
_UNIFORM_BRIER = 0.999


@dataclass(frozen=True)
class ProbabilityV5DevelopmentConfig:
    """开发模式的冻结配置；不包含正式Validation或生产激活开关。"""

    warmup_history: int = 150
    search_periods: int = 500
    calibration_periods: int = 250
    evaluation_periods: int = 500
    marginal_half_life: float = 80.0
    pairwise_half_life: float = 300.0
    legacy_learning_rate: float = 0.02
    initial_expert_weights: tuple[float, ...] = (0.50, 0.20, 0.15, 0.15)
    maximum_hedge_learning_rate: float = 0.05
    temperature_grid: tuple[float, ...] = (0.75, 1.0, 1.5, 2.0, 3.0)
    top_k: int = 50
    maximum_top50_triples: int = 1
    bootstrap_block_size: int = 10
    bootstrap_resamples: int = 999
    required_null_simulations: int = 5000
    random_seed: int = 20260722
    legacy_feature_config: LearnedFeatureConfig = LearnedFeatureConfig(
        windows=(20, 50, 150),
        half_life=50,
    )

    def __post_init__(self) -> None:
        if self.warmup_history < 20:
            raise ValueError("warmup_history至少为20")
        if (
            min(
                self.search_periods,
                self.calibration_periods,
                self.evaluation_periods,
            )
            <= 0
        ):
            raise ValueError("Search、Calibration和Evaluation期数必须为正")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.marginal_half_life, self.pairwise_half_life)
        ):
            raise ValueError("EWMA半衰期必须为正")
        if (
            not math.isfinite(self.legacy_learning_rate)
            or self.legacy_learning_rate < 0
        ):
            raise ValueError("legacy_learning_rate不得为负")
        if len(self.initial_expert_weights) != len(_EXPERT_NAMES):
            raise ValueError("初始专家权重数量错误")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.initial_expert_weights
        ):
            raise ValueError("初始专家权重必须为正")
        if not math.isclose(sum(self.initial_expert_weights), 1.0):
            raise ValueError("初始专家权重之和必须为1")
        if (
            not math.isfinite(self.maximum_hedge_learning_rate)
            or self.maximum_hedge_learning_rate <= 0
        ):
            raise ValueError("Hedge最大学习率必须为正")
        if not self.temperature_grid or any(
            not math.isfinite(value) or value <= 0 for value in self.temperature_grid
        ):
            raise ValueError("temperature_grid必须是非空正数数组")
        if len(set(self.temperature_grid)) != len(self.temperature_grid):
            raise ValueError("temperature_grid不得重复")
        if not 1 <= self.top_k < 1000:
            raise ValueError("top_k必须位于1..999")
        if self.maximum_top50_triples < 0:
            raise ValueError("maximum_top50_triples不得为负")
        if self.bootstrap_block_size <= 0 or self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap参数必须为正")
        if self.required_null_simulations != 5000:
            raise ValueError("required_null_simulations必须等于5000")
        if self.random_seed < 0:
            raise ValueError("random_seed不得为负")

    @property
    def required_prediction_periods(self) -> int:
        return self.search_periods + self.calibration_periods + self.evaluation_periods

    def to_dict(self) -> dict[str, object]:
        legacy_gradient_config = _legacy_online_config(self, development_end=1)
        return {
            "warmupHistory": self.warmup_history,
            "searchPeriods": self.search_periods,
            "calibrationPeriods": self.calibration_periods,
            "evaluationPeriods": self.evaluation_periods,
            "marginalHalfLife": self.marginal_half_life,
            "pairwiseHalfLife": self.pairwise_half_life,
            "legacyLearningRate": self.legacy_learning_rate,
            "expertNames": list(_EXPERT_NAMES),
            "initialExpertWeights": dict(
                zip(_EXPERT_NAMES, self.initial_expert_weights)
            ),
            "maximumHedgeLearningRate": self.maximum_hedge_learning_rate,
            "temperatureGrid": list(self.temperature_grid),
            "topK": self.top_k,
            "maximumTop50Triples": self.maximum_top50_triples,
            "bootstrapBlockSize": self.bootstrap_block_size,
            "bootstrapResamples": self.bootstrap_resamples,
            "requiredNullSimulations": self.required_null_simulations,
            "randomSeed": self.random_seed,
            "legacyFeatureConfig": {
                "windows": list(self.legacy_feature_config.windows),
                "halfLife": self.legacy_feature_config.half_life,
                "alpha": self.legacy_feature_config.alpha,
                "omissionCap": self.legacy_feature_config.omission_cap,
                "windowWeights": dict(self.legacy_feature_config.window_weights or ()),
            },
            "legacyGradientConfig": {
                "temperature": legacy_gradient_config.temperature,
                "l2Penalty": legacy_gradient_config.l2_penalty,
                "gradientClip": legacy_gradient_config.gradient_clip,
                "weightLimit": legacy_gradient_config.weight_limit,
                "featureNames": list(legacy_gradient_config.feature_names),
                "featureL2Multipliers": dict(
                    legacy_gradient_config.feature_l2_multipliers
                ),
                "zeroedFeatures": list(legacy_gradient_config.zeroed_features),
                "nonpositiveFeatures": list(
                    legacy_gradient_config.nonpositive_features
                ),
            },
        }


def probability_v5_smoke_config() -> ProbabilityV5DevelopmentConfig:
    """返回仅用于执行链测试的50期配置，不产生可晋级证据。"""

    return ProbabilityV5DevelopmentConfig(
        warmup_history=20,
        search_periods=10,
        calibration_periods=10,
        evaluation_periods=10,
        temperature_grid=(0.75, 1.0, 2.0),
        bootstrap_block_size=5,
        bootstrap_resamples=99,
    )


def prepare_probability_v5_development_history(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: ProbabilityV5DevelopmentConfig,
) -> pd.DataFrame:
    """固定取Frozen边界前最近一个完整协议窗口，消除Search前隐藏状态。"""

    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    required = config.warmup_history + config.required_prediction_periods
    if len(chronological) < required:
        raise ValueError(f"开发历史至少需要{required}期")
    return chronological.tail(required).reset_index(drop=True)


@dataclass
class _EWMAProbabilityState:
    marginal: np.ndarray
    pairwise: np.ndarray
    marginal_decay: float
    pairwise_decay: float

    @classmethod
    def create(
        cls, marginal_half_life: float, pairwise_half_life: float
    ) -> "_EWMAProbabilityState":
        return cls(
            marginal=np.full((3, 10), 0.1, dtype=float),
            pairwise=np.full((3, 10, 10), 0.01, dtype=float),
            marginal_decay=2.0 ** (-1.0 / marginal_half_life),
            pairwise_decay=2.0 ** (-1.0 / pairwise_half_life),
        )

    def update(self, digits: Sequence[int]) -> None:
        values = tuple(int(value) for value in digits)
        self.marginal *= self.marginal_decay
        for position, digit in enumerate(values):
            self.marginal[position, digit] += 1.0 - self.marginal_decay
        self.pairwise *= self.pairwise_decay
        for pair_index, (left, right) in enumerate(((0, 1), (0, 2), (1, 2))):
            self.pairwise[pair_index, values[left], values[right]] += (
                1.0 - self.pairwise_decay
            )

    def position_probabilities(self) -> np.ndarray:
        values = np.prod(
            self.marginal[np.arange(3)[:, None], _CANDIDATE_DIGITS.T], axis=0
        )
        return _normalize_probability(values)

    def pairwise_probabilities(self) -> np.ndarray:
        digits = _CANDIDATE_DIGITS
        values = (
            self.pairwise[0, digits[:, 0], digits[:, 1]]
            * self.pairwise[1, digits[:, 0], digits[:, 2]]
            * self.pairwise[2, digits[:, 1], digits[:, 2]]
        )
        return _normalize_probability(values)


@dataclass(frozen=True)
class _PrequentialRecord:
    target_index: int
    target_issue: str
    actual_index: int
    actual_text: str
    probabilities: np.ndarray
    expert_weights: tuple[float, ...]
    expert_weights_after: tuple[float, ...]
    expert_actual_probabilities: tuple[float, ...]
    raw_top_k: tuple[str, ...]
    policy_top_k: tuple[str, ...]
    raw_top_k_hit: bool
    policy_top_k_hit: bool
    raw_distribution_fingerprint: str


def _protocol_identity(payload: Mapping[str, object]) -> dict[str, object]:
    development_data = payload.get("developmentData")
    frozen_boundary = payload.get("frozenBoundary")
    if not isinstance(development_data, Mapping) or not isinstance(
        frozen_boundary, Mapping
    ):
        raise ValueError("probability_v5开发协议身份字段无效")
    identity: dict[str, object] = {
        "protocolSha256": payload["protocolSha256"],
        "lottery": payload["lottery"],
        "dataSha256": development_data["dataSha256"],
        "sourceFingerprint": payload["sourceFingerprint"],
        "configSha256": payload["configSha256"],
        "frozenPeriodsExcluded": frozen_boundary["periodsExcluded"],
    }
    identity["identitySha256"] = _payload_sha256(identity)
    return identity


@dataclass(frozen=True)
class ProbabilityV5DevelopmentReport:
    """开发区预序评估报告；永远不表示正式模型已激活。"""

    lottery: str
    frozen_periods_excluded: int
    data_sha256: str
    source_fingerprint: str
    config: ProbabilityV5DevelopmentConfig
    protocol_identity: dict[str, object] | None
    selected_temperature: float
    search: dict[str, object]
    calibration: dict[str, object]
    evaluation: dict[str, object]
    periods: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        search_passed = self.search.get("strictStatisticalGatePassed") is True
        evaluation_passed = self.evaluation.get("strictStatisticalGatePassed") is True
        development_signals_passed = search_passed and evaluation_passed
        protocol_identity = self.protocol_identity
        promotion_reasons = []
        if not development_signals_passed:
            promotion_reasons.append("开发区严格统计闸门未通过")
        promotion_reasons.extend(
            [
                "全流程随机模拟尚未执行",
                "新的独立500期Validation尚未打开",
            ]
        )
        payload: dict[str, object] = {
            "schemaVersion": "probability_v5_development_v1",
            "modelVersion": "probability_v5",
            "evaluationKind": "development_prequential_challenger",
            "evidenceStatus": "exploratory_reused_development",
            "lottery": self.lottery,
            "dataSha256": self.data_sha256,
            "sourceFingerprint": self.source_fingerprint,
            "configSha256": _payload_sha256(self.config.to_dict()),
            "config": self.config.to_dict(),
            "protocolIdentity": protocol_identity,
            "protocol": {
                "predictThenUpdate": True,
                "frozenRead": False,
                "frozenPeriodsExcluded": self.frozen_periods_excluded,
                "developmentProtocolRegistered": (protocol_identity is not None),
                "developmentProtocolSha256": (
                    protocol_identity["protocolSha256"]
                    if protocol_identity is not None
                    else None
                ),
                "rawProbabilityMetricsSeparatedFromPolicyTopK": True,
                "uniformMechanism": "permanent_expert_only",
                "secondaryUniformShrinkageEnabled": False,
                "legacyStateCompatibility": False,
                "validationOpened": False,
                "onlineStateUpdatesAllowed": True,
                "discretionaryParameterChangesAllowed": False,
            },
            "selectedTemperature": self.selected_temperature,
            "search": self.search,
            "calibration": self.calibration,
            "evaluation": self.evaluation,
            "derivedGates": {
                "searchPassed": search_passed,
                "evaluationPassed": evaluation_passed,
                "searchAndEvaluationPassed": development_signals_passed,
            },
            "fullPipelineNullSimulation": {
                "status": "not_run",
                "requiredIterations": self.config.required_null_simulations,
                "passed": False,
            },
            "developmentSignalsPassed": development_signals_passed,
            "promotionPassed": False,
            "promotionReasons": promotion_reasons,
            "researchRankingEnabled": False,
            "recommendationEnabled": False,
            "researchTop50": [],
            "formalRecommendation": None,
            "periods": list(self.periods),
        }
        payload["reportSha256"] = _payload_sha256(payload)
        return payload


def _normalize_probability(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (1000,) or not np.isfinite(probabilities).all():
        raise ValueError("专家必须输出1000个有限概率")
    if np.any(probabilities < 0):
        raise ValueError("专家概率不得为负")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("专家概率和必须为正")
    return probabilities / total


def _payload_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _source_fingerprint_paths() -> tuple[Path, ...]:
    """返回影响v5证据身份的全部源码与CLI路径。"""

    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    return (
        Path(__file__),
        directory / "digit_probability_v5_null.py",
        directory / "digit_data.py",
        directory / "digit_daily_policy.py",
        directory / "digit_learned_features.py",
        directory / "digit_learned_ranker.py",
        directory / "digit_online_gradient.py",
        directory / "prediction_viability.py",
        root / "scripts" / "digit_probability_v5.py",
        root / "scripts" / "digit_probability_v5_null.py",
    )


@lru_cache(maxsize=1)
def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    for path in _source_fingerprint_paths():
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return digest.hexdigest()


def build_probability_v5_protocol(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: ProbabilityV5DevelopmentConfig,
    *,
    frozen_periods_excluded: int,
) -> dict[str, object]:
    """构建确定性的v5开发协议；协议不包含运行结果。"""

    if rule.draw_count != 3 or rule.code not in {"fc3d", "pl3"}:
        raise ValueError("probability_v5只支持fc3d/pl3")
    if frozen_periods_excluded < 500:
        raise ValueError("probability_v5开发协议必须排除至少500期Frozen")
    chronological = prepare_probability_v5_development_history(history, rule, config)
    config_payload = config.to_dict()
    payload: dict[str, object] = {
        "schemaVersion": "probability_v5_development_protocol_v1",
        "modelVersion": "probability_v5",
        "immutable": True,
        "lottery": rule.code,
        "developmentData": {
            "periods": len(chronological),
            "firstIssue": str(chronological.iloc[0]["期数"]),
            "lastIssue": str(chronological.iloc[-1]["期数"]),
            "dataSha256": canonical_digit_data_sha256(chronological, rule),
        },
        "frozenBoundary": {
            "periodsExcluded": frozen_periods_excluded,
            "frozenRead": False,
        },
        "sourceFingerprint": _source_fingerprint(),
        "configSha256": _payload_sha256(config_payload),
        "config": config_payload,
        "evaluationPlan": {
            "searchPeriods": config.search_periods,
            "searchSelectionPerformed": False,
            "calibrationPeriods": config.calibration_periods,
            "evaluationPeriods": config.evaluation_periods,
            "policyTopK": config.top_k,
        },
        "nullSimulationPlan": {
            "requiredIterations": config.required_null_simulations,
            "baseSeed": config.random_seed,
            "fullPipelineReplayRequired": True,
        },
        "futureValidationPlan": {
            "status": "not_opened",
            "minimumUnseenPeriods": 500,
            "singleUse": True,
        },
        "legacyStateCompatibility": False,
        "recommendationEnabled": False,
    }
    payload["protocolSha256"] = _payload_sha256(payload)
    return payload


def _write_immutable_content(path: Path, content: str, *, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() == encoded and path.stat().st_mode & 0o222 == 0:
            return path
        if path.read_bytes() == encoded:
            raise PermissionError(f"{label}必须保持只读：{path}")
        raise FileExistsError(f"{label}已存在不同内容，禁止覆盖：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise FileExistsError(
                    f"{label}已存在不同内容，禁止覆盖：{path}"
                ) from None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _load_readonly_json(path: str | Path, label: str) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label}必须是普通只读文件：{source}")
    if source.stat().st_mode & 0o222:
        raise PermissionError(f"{label}必须是只读文件：{source}")
    loaded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label}格式无效")
    return loaded


def write_probability_v5_protocol(
    protocol: Mapping[str, object], path: str | Path
) -> Path:
    """只写一次保存开发协议，并校验协议内容指纹。"""

    document = dict(protocol)
    claimed = document.pop("protocolSha256", None)
    if claimed != _payload_sha256(document):
        raise ValueError("probability_v5开发协议内容指纹不匹配")
    document["protocolSha256"] = claimed
    content = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    return _write_immutable_content(Path(path), content, label="probability_v5开发协议")


def load_and_verify_probability_v5_protocol(
    path: str | Path,
    history: pd.DataFrame,
    rule: LotteryRule,
    config: ProbabilityV5DevelopmentConfig,
    *,
    frozen_periods_excluded: int,
) -> dict[str, object]:
    """加载协议并与当前源码、配置、开发数据和Frozen边界逐字段核对。"""

    loaded = _load_readonly_json(path, "probability_v5开发协议")
    development_data = loaded.get("developmentData")
    if not isinstance(development_data, Mapping):
        raise ValueError("probability_v5开发协议developmentData格式无效")
    if len(history) != development_data.get("periods"):
        raise ValueError("锁定开发DataFrame期数与probability_v5开发协议不一致")
    expected = build_probability_v5_protocol(
        history,
        rule,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
    )
    if loaded != expected:
        raise RuntimeError("probability_v5开发协议与当前源码、配置或数据不一致")
    return json.loads(json.dumps(expected, ensure_ascii=False))


def _adaptive_expert_weights(
    cumulative_excess_losses: np.ndarray,
    initial_weights: np.ndarray,
    cumulative_loss_range_squared: float,
    maximum_learning_rate: float,
) -> tuple[np.ndarray, float]:
    expert_count = len(initial_weights)
    adaptive_rate = math.sqrt(
        2.0 * math.log(expert_count) / (1.0 + cumulative_loss_range_squared)
    )
    learning_rate = min(maximum_learning_rate, adaptive_rate)
    logits = np.log(initial_weights) - learning_rate * cumulative_excess_losses
    logits -= float(logits.max())
    weights = np.exp(logits)
    weights /= float(weights.sum())
    return weights, learning_rate


def _legacy_online_config(
    config: ProbabilityV5DevelopmentConfig, development_end: int
) -> OnlineGradientConfig:
    return OnlineGradientConfig(
        development_end=development_end,
        outer_periods=1,
        warmup_history=config.warmup_history,
        learning_rates=(config.legacy_learning_rate,),
        shrinkages=(1.0,),
    )


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    log_values = np.log(np.maximum(np.asarray(probabilities, dtype=float), 1e-300))
    scaled = np.exp(log_values / temperature - float((log_values / temperature).max()))
    return scaled / float(scaled.sum())


def _probability_metrics(
    records: Sequence[_PrequentialRecord], temperature: float
) -> dict[str, float | int]:
    log_losses = []
    brier_scores = []
    for record in records:
        probabilities = _temperature_scale(record.probabilities, temperature)
        actual_probability = float(probabilities[record.actual_index])
        log_losses.append(-math.log(max(actual_probability, 1e-300)))
        brier_scores.append(
            float(np.dot(probabilities, probabilities) - 2 * actual_probability + 1)
        )
    return {
        "periods": len(records),
        "meanLogLoss": float(np.mean(log_losses)),
        "meanBrier": float(np.mean(brier_scores)),
        "meanDeltaLogLoss": _UNIFORM_LOG_LOSS - float(np.mean(log_losses)),
        "meanDeltaBrier": _UNIFORM_BRIER - float(np.mean(brier_scores)),
    }


def _block_bootstrap_lower_bound(
    values: Sequence[float],
    *,
    seed: int,
    block_size: int,
    resamples: int,
) -> float | None:
    array = np.asarray(values, dtype=float)
    if len(array) < block_size * 2:
        return None
    rng = np.random.default_rng(seed)
    blocks_per_sample = math.ceil(len(array) / block_size)
    starts = rng.integers(0, len(array), size=(resamples, blocks_per_sample))
    offsets = np.arange(block_size)
    sampled_indices = (starts[:, :, None] + offsets) % len(array)
    sampled = array[sampled_indices].reshape(resamples, -1)[:, : len(array)]
    return float(np.quantile(sampled.mean(axis=1), 0.01))


def _evaluation_payload(
    records: Sequence[_PrequentialRecord],
    temperature: float,
    config: ProbabilityV5DevelopmentConfig,
    *,
    include_period_details: bool = True,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    log_losses = []
    brier_scores = []
    period_payloads = []
    for record in records:
        probabilities = _temperature_scale(record.probabilities, temperature)
        actual_probability = float(probabilities[record.actual_index])
        log_loss = -math.log(max(actual_probability, 1e-300))
        brier = float(np.dot(probabilities, probabilities) - 2 * actual_probability + 1)
        log_losses.append(log_loss)
        brier_scores.append(brier)
        if include_period_details:
            calibrated_fingerprint = hashlib.sha256(
                probabilities.astype("<f8").tobytes()
            ).hexdigest()
            period_payloads.append(
                {
                    "targetIndex": record.target_index,
                    "targetIssue": record.target_issue,
                    "actual": record.actual_text,
                    "actualProbability": actual_probability,
                    "logLoss": log_loss,
                    "brier": brier,
                    "deltaLogLoss": _UNIFORM_LOG_LOSS - log_loss,
                    "deltaBrier": _UNIFORM_BRIER - brier,
                    "rawTopKHit": record.raw_top_k_hit,
                    "policyTopKHit": record.policy_top_k_hit,
                    "rawTopK": list(record.raw_top_k),
                    "policyTopK": list(record.policy_top_k),
                    "expertWeights": dict(zip(_EXPERT_NAMES, record.expert_weights)),
                    "expertWeightsAfter": dict(
                        zip(_EXPERT_NAMES, record.expert_weights_after)
                    ),
                    "expertActualProbabilities": dict(
                        zip(_EXPERT_NAMES, record.expert_actual_probabilities)
                    ),
                    "rawDistributionFingerprint": (record.raw_distribution_fingerprint),
                    "calibratedDistributionFingerprint": calibrated_fingerprint,
                }
            )
    viability = evaluate_viability_metric(
        "policy_direct",
        [record.policy_top_k_hit for record in records],
        [config.top_k / 1000.0] * len(records),
    )
    log_improvements = [_UNIFORM_LOG_LOSS - value for value in log_losses]
    brier_improvements = [_UNIFORM_BRIER - value for value in brier_scores]
    log_lower = _block_bootstrap_lower_bound(
        log_improvements,
        seed=config.random_seed,
        block_size=config.bootstrap_block_size,
        resamples=config.bootstrap_resamples,
    )
    brier_lower = _block_bootstrap_lower_bound(
        brier_improvements,
        seed=config.random_seed + 1,
        block_size=config.bootstrap_block_size,
        resamples=config.bootstrap_resamples,
    )
    blocks = []
    for block_index, indices in enumerate(
        np.array_split(np.arange(len(records)), min(3, len(records))), start=1
    ):
        if not len(indices):
            continue
        block_log = [log_losses[int(index)] for index in indices]
        block_brier = [brier_scores[int(index)] for index in indices]
        blocks.append(
            {
                "block": block_index,
                "periods": len(indices),
                "meanLogLoss": float(np.mean(block_log)),
                "meanBrier": float(np.mean(block_brier)),
                "policyTopKHits": sum(
                    records[int(index)].policy_top_k_hit for index in indices
                ),
            }
        )
    gate_passed = (
        viability.viable
        and log_lower is not None
        and log_lower > 0
        and brier_lower is not None
        and brier_lower > 0
    )
    payload: dict[str, object] = {
        **_probability_metrics(records, temperature),
        "uniformLogLoss": _UNIFORM_LOG_LOSS,
        "uniformBrier": _UNIFORM_BRIER,
        "rawTopKHits": sum(record.raw_top_k_hit for record in records),
        "policyTopKHits": sum(record.policy_top_k_hit for record in records),
        "topK": config.top_k,
        "policyTopKViability": viability.to_dict(),
        "logLossImprovementLowerBound99": log_lower,
        "brierImprovementLowerBound99": brier_lower,
        "strictStatisticalGatePassed": gate_passed,
        "timeBlocks": blocks,
    }
    return payload, tuple(period_payloads)


def _run_prequential(
    chronological: pd.DataFrame,
    rule: LotteryRule,
    config: ProbabilityV5DevelopmentConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
    *,
    include_candidate_lists: bool = True,
) -> tuple[_PrequentialRecord, ...]:
    values = chronological[list(rule.number_columns)].to_numpy(dtype=np.int16)
    ewma = _EWMAProbabilityState.create(
        config.marginal_half_life, config.pairwise_half_life
    )
    for row in values[: config.warmup_history]:
        ewma.update(row)

    online_config = _legacy_online_config(config, len(chronological))
    legacy_candidate = OnlineGradientCandidate(config.legacy_learning_rate, 1.0)
    legacy_weights = _initial_weights(online_config)
    initial_weights = np.asarray(config.initial_expert_weights, dtype=float)
    cumulative_excess_losses = np.zeros(len(_EXPERT_NAMES), dtype=float)
    cumulative_loss_range_squared = 0.0
    indices = tuple(range(config.warmup_history, len(chronological)))
    rolling_states = iter_rolling_history_states(
        chronological,
        rule,
        indices,
        config.legacy_feature_config,
    )
    records = []
    for processed, (target_index, history_state) in enumerate(
        zip(indices, rolling_states), start=1
    ):
        actual_digits = values[target_index]
        actual_text = "".join(str(int(value)) for value in actual_digits)
        actual_index = int(actual_text)
        matrix = build_candidate_features(history_state, rule)[
            list(FEATURE_NAMES)
        ].to_numpy(dtype=float)
        legacy_step = online_gradient_step(
            matrix,
            actual_index,
            legacy_weights,
            legacy_candidate,
            online_config,
        )
        expert_probabilities = (
            np.full(1000, 0.001, dtype=float),
            ewma.position_probabilities(),
            ewma.pairwise_probabilities(),
            legacy_step.final_probabilities,
        )
        expert_weights, _ = _adaptive_expert_weights(
            cumulative_excess_losses,
            initial_weights,
            cumulative_loss_range_squared,
            config.maximum_hedge_learning_rate,
        )
        mixture = np.average(
            np.vstack(expert_probabilities), axis=0, weights=expert_weights
        )
        mixture = _normalize_probability(mixture)
        order = rank_candidate_indices(mixture, _CANDIDATES)
        ranked = [_CANDIDATES[int(index)] for index in order]
        policy_top = select_daily_candidates(
            ranked,
            latest_exact="".join(str(int(value)) for value in values[target_index - 1]),
            top_k=config.top_k,
            maximum_triples=config.maximum_top50_triples,
        )
        expert_actual = np.asarray(
            [probabilities[actual_index] for probabilities in expert_probabilities],
            dtype=float,
        )
        losses = -np.log(np.maximum(expert_actual, 1e-300))
        next_cumulative_excess_losses = (
            cumulative_excess_losses + losses - _UNIFORM_LOG_LOSS
        )
        next_cumulative_loss_range_squared = cumulative_loss_range_squared + float(
            (losses.max() - losses.min()) ** 2
        )
        expert_weights_after, _ = _adaptive_expert_weights(
            next_cumulative_excess_losses,
            initial_weights,
            next_cumulative_loss_range_squared,
            config.maximum_hedge_learning_rate,
        )
        records.append(
            _PrequentialRecord(
                target_index=target_index,
                target_issue=str(chronological.iloc[target_index]["期数"]),
                actual_index=actual_index,
                actual_text=actual_text,
                probabilities=mixture.astype(np.float64, copy=True),
                expert_weights=tuple(float(value) for value in expert_weights),
                expert_weights_after=tuple(
                    float(value) for value in expert_weights_after
                ),
                expert_actual_probabilities=tuple(
                    float(value) for value in expert_actual
                ),
                raw_top_k=(
                    tuple(ranked[: config.top_k]) if include_candidate_lists else ()
                ),
                policy_top_k=policy_top if include_candidate_lists else (),
                raw_top_k_hit=actual_text in ranked[: config.top_k],
                policy_top_k_hit=actual_text in policy_top,
                raw_distribution_fingerprint=hashlib.sha256(
                    mixture.astype("<f8").tobytes()
                ).hexdigest(),
            )
        )
        cumulative_excess_losses = next_cumulative_excess_losses
        cumulative_loss_range_squared = next_cumulative_loss_range_squared
        legacy_weights = legacy_step.weights_after
        ewma.update(actual_digits)
        if progress_callback is not None and (
            processed % 500 == 0 or processed == len(indices)
        ):
            progress_callback(
                processed, len(indices), str(chronological.iloc[target_index]["期数"])
            )
    return tuple(records)


def _run_probability_v5_development(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: ProbabilityV5DevelopmentConfig = ProbabilityV5DevelopmentConfig(),
    *,
    frozen_periods_excluded: int,
    protocol_payload: Mapping[str, object] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    include_period_details: bool = True,
) -> ProbabilityV5DevelopmentReport:
    """运行不读取Frozen的v5开发评估；示例见 ``scripts/digit_probability_v5.py``。"""

    if rule.draw_count != 3 or rule.code not in {"fc3d", "pl3"}:
        raise ValueError("probability_v5只支持fc3d/pl3")
    if frozen_periods_excluded < 500:
        raise ValueError("probability_v5开发报告必须排除至少500期Frozen")
    chronological = prepare_probability_v5_development_history(history, rule, config)
    if protocol_payload is not None:
        expected_protocol = build_probability_v5_protocol(
            chronological,
            rule,
            config,
            frozen_periods_excluded=frozen_periods_excluded,
        )
        if dict(protocol_payload) != expected_protocol:
            raise RuntimeError("已验证probability_v5协议与当前核心输入身份不一致")
    records = _run_prequential(
        chronological,
        rule,
        config,
        progress_callback=progress_callback,
        include_candidate_lists=include_period_details,
    )
    evaluation_start = len(records) - config.evaluation_periods
    calibration_start = evaluation_start - config.calibration_periods
    search_start = calibration_start - config.search_periods
    search_records = records[search_start:calibration_start]
    calibration_records = records[calibration_start:evaluation_start]
    evaluation_records = records[evaluation_start:]
    calibration_candidates = [
        {
            "temperature": temperature,
            **_probability_metrics(calibration_records, temperature),
        }
        for temperature in config.temperature_grid
    ]
    selected = min(
        calibration_candidates,
        key=lambda item: (
            float(item["meanLogLoss"]),
            float(item["meanBrier"]),
            float(item["temperature"]),
        ),
    )
    temperature = float(selected["temperature"])
    evaluation, periods = _evaluation_payload(
        evaluation_records,
        temperature,
        config,
        include_period_details=include_period_details,
    )
    search_metrics, search_period_details = _evaluation_payload(
        search_records,
        1.0,
        config,
        include_period_details=include_period_details,
    )
    search: dict[str, object] = {
        **search_metrics,
        "periods": len(search_records),
        "selectionPerformed": False,
        "expertStructurePreRegistered": True,
        "periodDetails": list(search_period_details),
    }
    calibration: dict[str, object] = {
        "periods": len(calibration_records),
        "selectionMetricOrder": ["meanLogLoss", "meanBrier", "temperature"],
        "candidates": calibration_candidates,
        "selected": selected,
        "periodAudit": list(
            _evaluation_payload(
                calibration_records,
                temperature,
                config,
                include_period_details=include_period_details,
            )[1]
        ),
    }
    last_prediction_weights = records[-1].expert_weights
    post_final_update_weights = records[-1].expert_weights_after
    evaluation["lastPredictionExpertWeights"] = dict(
        zip(_EXPERT_NAMES, last_prediction_weights)
    )
    evaluation["postFinalUpdateExpertWeights"] = dict(
        zip(_EXPERT_NAMES, post_final_update_weights)
    )
    evaluation["expertWeightDistribution"] = {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "minimum": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.5)),
            "q75": float(np.quantile(values, 0.75)),
            "maximum": float(np.max(values)),
        }
        for name, values in (
            (
                expert_name,
                np.asarray(
                    [record.expert_weights[index] for record in evaluation_records]
                ),
            )
            for index, expert_name in enumerate(_EXPERT_NAMES)
        )
    }
    evaluation["meanUniformExpertWeight"] = float(
        np.mean([record.expert_weights[0] for record in evaluation_records])
    )
    return ProbabilityV5DevelopmentReport(
        lottery=rule.code,
        frozen_periods_excluded=frozen_periods_excluded,
        data_sha256=canonical_digit_data_sha256(chronological, rule),
        source_fingerprint=_source_fingerprint(),
        config=config,
        protocol_identity=(
            _protocol_identity(protocol_payload)
            if protocol_payload is not None
            else None
        ),
        selected_temperature=temperature,
        search=search,
        calibration=calibration,
        evaluation=evaluation,
        periods=periods,
    )


def run_probability_v5_development(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: ProbabilityV5DevelopmentConfig = ProbabilityV5DevelopmentConfig(),
    *,
    frozen_periods_excluded: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    include_period_details: bool = True,
) -> ProbabilityV5DevelopmentReport:
    """运行通用开发/smoke评估；该入口永远不声明协议已登记。"""

    return _run_probability_v5_development(
        history,
        rule,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
        protocol_payload=None,
        progress_callback=progress_callback,
        include_period_details=include_period_details,
    )


def run_registered_probability_v5_development(
    protocol_path: str | Path,
    history: pd.DataFrame,
    rule: LotteryRule,
    config: ProbabilityV5DevelopmentConfig = ProbabilityV5DevelopmentConfig(),
    *,
    frozen_periods_excluded: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    include_period_details: bool = True,
) -> ProbabilityV5DevelopmentReport:
    """从只读协议文件加载、完整重算并运行已登记开发评估。"""

    protocol = load_and_verify_probability_v5_protocol(
        protocol_path,
        history,
        rule,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
    )
    return _run_probability_v5_development(
        history,
        rule,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
        protocol_payload=protocol,
        progress_callback=progress_callback,
        include_period_details=include_period_details,
    )


def load_and_verify_probability_v5_report(
    path: str | Path,
    protocol_path: str | Path,
    history: pd.DataFrame,
    rule: LotteryRule,
    config: ProbabilityV5DevelopmentConfig,
    *,
    frozen_periods_excluded: int,
) -> dict[str, object]:
    """从锁定输入完整重算报告，拒绝只重写自哈希的伪造指标。"""

    loaded = _load_readonly_json(path, "probability_v5开发报告")
    claimed = loaded.get("reportSha256")
    unsigned = {key: value for key, value in loaded.items() if key != "reportSha256"}
    if claimed != _payload_sha256(unsigned):
        raise ValueError("probability_v5开发报告reportSha256无效")
    expected = run_registered_probability_v5_development(
        protocol_path,
        history,
        rule,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
        include_period_details=True,
    ).to_dict()
    if loaded != expected:
        raise RuntimeError("probability_v5开发报告与锁定输入重新计算结果不一致")
    return json.loads(json.dumps(expected, ensure_ascii=False))


def write_probability_v5_report(
    report: ProbabilityV5DevelopmentReport, path: str | Path
) -> Path:
    """原子写入开发报告；不会创建或更新任何模型状态。"""

    destination = Path(path)
    content = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
    return _write_immutable_content(
        destination, content, label="probability_v5开发报告"
    )


__all__ = [
    "ProbabilityV5DevelopmentConfig",
    "ProbabilityV5DevelopmentReport",
    "build_probability_v5_protocol",
    "load_and_verify_probability_v5_report",
    "load_and_verify_probability_v5_protocol",
    "prepare_probability_v5_development_history",
    "probability_v5_smoke_config",
    "run_probability_v5_development",
    "run_registered_probability_v5_development",
    "write_probability_v5_report",
    "write_probability_v5_protocol",
]

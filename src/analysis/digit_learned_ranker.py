# -*- coding: utf-8 -*-
"""三位彩固定、可解释线性评分算法 v4。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.analysis.digit_data import (
    canonical_digit_data_sha256,
    load_digit_csv,
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_learned_features import (
    FEATURE_NAMES,
    LearnedFeatureConfig,
    build_candidate_features,
    build_history_state,
)
from src.analysis.digit_statistics import classify_digit_shape
from src.analysis.digit_strategy_gate import StrategyStatus
from src.analysis.digit_strategy_registry import (
    StrategyRegistryUpdate,
    update_strategy_registry,
)
from src.lotteries.base import LotteryRule

DEFAULT_WEIGHTS = {
    "position_frequency": 1.0,
    "position_omission": 0.15,
    "pair_frequency": 0.45,
    "sum_distribution": 0.25,
    "span_distribution": 0.2,
    "recent_trend": 0.15,
    "position_trend": 0.05,
    "pair_trend": 0.05,
    "shape_transition": 0.0,
    "shape_recent_deviation": 0.0,
    "constraint_penalty": -0.25,
}


@dataclass(frozen=True)
class LearnedRankerParams:
    """固定公式参数，可稳定保存为 JSON 并计算指纹。"""

    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    temperature: float = 1.0
    uniform_shrinkage: float = 1.0
    direct_top_k: int = 10
    group_top_k: int = 10
    group_aggregation: str = "sum_prob"
    position_pool_size: int = 5
    group_digit_pool_size: int = 7
    random_seed: int = 20260717

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"存在未知特征权重：{sorted(unknown)}")
        if self.temperature <= 0:
            raise ValueError("softmax temperature 必须大于零")
        if not 0.0 <= self.uniform_shrinkage <= 1.0:
            raise ValueError("均匀收缩系数必须位于 0..1")
        if self.direct_top_k <= 0 or self.group_top_k <= 0:
            raise ValueError("TopK 必须为正整数")
        if self.direct_top_k > 1000:
            raise ValueError("直选 TopK 不得超过 1000")
        if self.group_top_k > 220:
            raise ValueError("组选 TopK 不得超过 220")
        if not 1 <= self.position_pool_size <= 10:
            raise ValueError("位置池大小必须在 1-10 之间")
        if not 1 <= self.group_digit_pool_size <= 10:
            raise ValueError("组选数字池大小必须在 1-10 之间")
        if self.group_aggregation not in {"sum_prob", "max_perm", "mean_top_perm"}:
            raise ValueError("组选聚合只支持 sum_prob、max_perm、mean_top_perm")
        if not math.isfinite(self.temperature):
            raise ValueError("softmax temperature 必须为有限数")
        if any(not math.isfinite(float(value)) for value in self.weights.values()):
            raise ValueError("特征权重必须为有限数")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["weights"] = dict(sorted(self.weights.items()))
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearnedRankerParams":
        return cls(
            weights={
                str(key): float(value)
                for key, value in dict(payload["weights"]).items()
            },
            temperature=float(payload.get("temperature", 1.0)),
            uniform_shrinkage=float(payload.get("uniform_shrinkage", 1.0)),
            direct_top_k=int(payload.get("direct_top_k", 10)),
            group_top_k=int(payload.get("group_top_k", 10)),
            group_aggregation=str(payload.get("group_aggregation", "sum_prob")),
            position_pool_size=int(payload.get("position_pool_size", 5)),
            group_digit_pool_size=int(payload.get("group_digit_pool_size", 7)),
            random_seed=int(payload.get("random_seed", 20260717)),
        )


@dataclass(frozen=True)
class LearnedDirectCandidate:
    text: str
    score: float
    probability: float
    contributions: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "probability": self.probability,
            "contributions": dict(self.contributions),
        }


@dataclass(frozen=True)
class LearnedGroupCandidate:
    group_key: str
    shape: str
    probability: float | None
    score: float | None
    aggregation: str
    permutations: int
    max_permutation_probability: float

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "group_key": self.group_key,
            "shape": self.shape,
            "aggregation": self.aggregation,
            "permutations": self.permutations,
            "max_permutation_probability": self.max_permutation_probability,
        }
        if self.probability is not None:
            payload["probability"] = self.probability
        if self.score is not None:
            payload["score"] = self.score
        return payload


def _group_aggregate_value(item: LearnedGroupCandidate) -> float:
    value = item.probability if item.probability is not None else item.score
    if value is None:
        raise ValueError("组选候选缺少 probability/score 聚合值")
    return float(value)


@dataclass(frozen=True)
class LearnedRankerPlan:
    rule_code: str
    params_fingerprint: str
    direct_candidates: tuple[LearnedDirectCandidate, ...]
    group_candidates: tuple[LearnedGroupCandidate, ...]
    position_pools: tuple[tuple[int, ...], ...]
    group_digit_pool: tuple[int, ...]
    distribution_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ruleCode": self.rule_code,
            "paramsFingerprint": self.params_fingerprint,
            "directCandidates": [item.to_dict() for item in self.direct_candidates],
            "groupCandidates": [item.to_dict() for item in self.group_candidates],
            "positionPools": [list(pool) for pool in self.position_pools],
            "groupDigitPool": list(self.group_digit_pool),
            "distributionFingerprint": self.distribution_fingerprint,
        }


def _stable_json(payload: object) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def payload_fingerprint(payload: Mapping[str, Any]) -> str:
    """计算 JSON 对象的稳定 SHA-256 指纹。"""

    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _feature_config_payload(config: LearnedFeatureConfig) -> dict[str, Any]:
    return {
        "windows": list(config.windows),
        "alpha": config.alpha,
        "halfLife": config.half_life,
        "omissionCap": config.omission_cap,
        "windowWeights": dict(config.window_weights or ()),
    }


def params_fingerprint(
    params: LearnedRankerParams,
    feature_config: LearnedFeatureConfig | None = None,
) -> str:
    """计算对评分参数及可选特征配置变化敏感的稳定 SHA-256 指纹。"""

    payload: dict[str, Any] = {"params": params.to_dict()}
    if feature_config is not None:
        payload["featureConfig"] = _feature_config_payload(feature_config)
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _feature_config_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> LearnedFeatureConfig | None:
    feature = dict((metadata or {}).get("featureConfig", {}))
    if not feature:
        return None
    return LearnedFeatureConfig(
        windows=tuple(feature.get("windows", LearnedFeatureConfig().windows)),
        alpha=float(feature.get("alpha", 2.0)),
        half_life=(
            float(feature["halfLife"]) if feature.get("halfLife") is not None else None
        ),
        omission_cap=int(feature.get("omissionCap", 50)),
        window_weights=feature.get("windowWeights"),
    )


def resolve_activation(
    *,
    common_passed: bool,
    direct_passed: bool,
    group_passed: bool,
    position_passed: bool,
) -> dict[str, Any]:
    """按公共闸门与直选、组选、位置池三个独立闸门解析状态。"""

    active_direct = bool(common_passed and direct_passed)
    active_group = bool(common_passed and group_passed)
    active_position = bool(common_passed and position_passed)
    return {
        "commonPassed": bool(common_passed),
        "directPassed": bool(direct_passed),
        "groupPassed": bool(group_passed),
        "positionPassed": bool(position_passed),
        "activeDirect": active_direct,
        "activeGroup": active_group,
        "activePosition": active_position,
        "overallPassed": bool(
            common_passed and direct_passed and group_passed and position_passed
        ),
        "overallSemantics": (
            "commonPassed && directPassed && groupPassed && positionPassed"
        ),
    }


def partition_plan_by_activation(
    plan: Mapping[str, Any], activation: Mapping[str, Any]
) -> dict[str, Any]:
    """把通过闸门的部分放入主推荐，其余仅保留为研究结果。"""

    active_direct = bool(activation.get("activeDirect", False))
    active_group = bool(activation.get("activeGroup", False))
    active_position = bool(activation.get("activePosition", False))
    direct = list(plan.get("directCandidates", []))
    groups = list(plan.get("groupCandidates", []))
    position_pools = list(plan.get("positionPools", []))
    group_pool = list(plan.get("groupDigitPool", []))
    output = dict(plan)
    output.update(
        {
            "activeDirect": active_direct,
            "activeGroup": active_group,
            "activePosition": active_position,
            "mainRecommendation": {
                "directCandidates": direct if active_direct else [],
                "groupCandidates": groups if active_group else [],
                "positionPools": position_pools if active_position else [],
                "groupDigitPool": group_pool if active_group else [],
            },
            "research": {
                "directCandidates": [] if active_direct else direct,
                "groupCandidates": [] if active_group else groups,
                "positionPools": [] if active_position else position_pools,
                "groupDigitPool": [] if active_group else group_pool,
            },
        }
    )
    return output


def save_params(
    params: LearnedRankerParams,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """原子性要求由调用侧目录保证；保存参数、指纹和可复现元数据。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_config = _feature_config_from_metadata(metadata) or LearnedFeatureConfig()
    payload = {
        "schemaVersion": 1,
        "params": params.to_dict(),
        "paramsFingerprint": params_fingerprint(params, feature_config),
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    payload["artifactFingerprint"] = payload_fingerprint(payload)
    _atomic_write(
        output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    )
    return output


def load_params_document(path: str | Path) -> dict[str, Any]:
    """加载并校验参数文件完整内容及模型指纹。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    params = LearnedRankerParams.from_dict(payload["params"])
    feature_config = (
        _feature_config_from_metadata(payload.get("metadata")) or LearnedFeatureConfig()
    )
    if payload.get("paramsFingerprint") != params_fingerprint(params, feature_config):
        raise ValueError("参数文件指纹校验失败")
    artifact_payload = dict(payload)
    artifact_fingerprint = artifact_payload.pop("artifactFingerprint", None)
    if artifact_fingerprint != payload_fingerprint(artifact_payload):
        raise ValueError("参数文件指纹校验失败")
    return payload


def load_params(path: str | Path) -> LearnedRankerParams:
    """加载并校验参数文件指纹。"""

    payload = load_params_document(path)
    params = LearnedRankerParams.from_dict(payload["params"])
    return params


def load_params_metadata(path: str | Path) -> dict[str, Any]:
    """加载已通过完整性校验的参数元数据。"""

    return dict(load_params_document(path).get("metadata", {}))


def load_params_artifact_fingerprint(path: str | Path) -> str:
    """返回已校验参数文件的语义内容指纹。"""

    return str(load_params_document(path)["artifactFingerprint"])


def score_candidates(features: pd.DataFrame, params: LearnedRankerParams) -> np.ndarray:
    """按固定线性公式计算全部候选分数。"""

    scores = np.zeros(len(features), dtype=float)
    for name in FEATURE_NAMES:
        scores += float(params.weights.get(name, 0.0)) * features[name].to_numpy(
            dtype=float
        )
    return scores


def probabilities_from_scores(
    scores: np.ndarray, *, temperature: float, uniform_shrinkage: float = 1.0
) -> np.ndarray:
    """稳定softmax并向均匀分布收缩；λ=0表示模型没有可用信号。"""

    values = np.asarray(scores, dtype=float) / float(temperature)
    if values.size == 0:
        raise ValueError("概率候选不能为空")
    if not 0.0 <= uniform_shrinkage <= 1.0:
        raise ValueError("均匀收缩系数必须位于 0..1")
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    model = exponentials / float(exponentials.sum())
    uniform = np.full(values.shape, 1.0 / values.size, dtype=float)
    return uniform_shrinkage * model + (1.0 - uniform_shrinkage) * uniform


def rank_candidate_indices(scores: np.ndarray, candidates: Sequence[str]) -> np.ndarray:
    """分数降序、候选文本升序的稳定排序。"""

    return np.asarray(
        sorted(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), candidates[index]),
        ),
        dtype=int,
    )


def aggregate_group_candidates(
    candidates: Sequence[str],
    probabilities: np.ndarray,
    *,
    aggregation: str = "sum_prob",
) -> tuple[LearnedGroupCandidate, ...]:
    """把直选排列概率严格聚合到无序组选集合。"""

    grouped: dict[str, list[tuple[str, float]]] = {}
    for text, probability in zip(candidates, probabilities):
        grouped.setdefault("".join(sorted(text)), []).append((text, float(probability)))
    output = []
    for key, entries in grouped.items():
        values = [value for _, value in sorted(entries)]
        ordered = sorted(values, reverse=True)
        if aggregation == "sum_prob":
            aggregate = math.fsum(values)
        elif aggregation == "max_perm":
            aggregate = ordered[0]
        elif aggregation == "mean_top_perm":
            aggregate = float(np.mean(ordered[: min(3, len(ordered))]))
        else:
            raise ValueError("未知组选聚合方式")
        output.append(
            LearnedGroupCandidate(
                group_key=key,
                shape=classify_digit_shape(tuple(int(value) for value in key)),
                probability=aggregate if aggregation == "sum_prob" else None,
                score=aggregate if aggregation != "sum_prob" else None,
                aggregation=aggregation,
                permutations=len(values),
                max_permutation_probability=max(values),
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda item: (
                -_group_aggregate_value(item),
                item.group_key,
            ),
        )
    )


def build_learned_ranker_plan(
    features: pd.DataFrame,
    params: LearnedRankerParams,
    rule: LotteryRule,
    feature_config: LearnedFeatureConfig | None = None,
) -> LearnedRankerPlan:
    """生成直选、组选、位置复式池和组选数字池。"""

    if rule.draw_count != 3 or rule.code not in {"fc3d", "pl3"}:
        raise ValueError("learned_ranker_v4 首版只支持 fc3d/pl3")
    features = (
        features.assign(candidate=features["candidate"].astype(str))
        .sort_values("candidate", kind="mergesort")
        .reset_index(drop=True)
    )
    candidates = features["candidate"].tolist()
    if not candidates:
        raise ValueError("候选集合不能为空")
    if len(set(candidates)) != len(candidates):
        raise ValueError("候选号码不得重复")
    scores = score_candidates(features, params)
    probabilities = probabilities_from_scores(
        scores,
        temperature=params.temperature,
        uniform_shrinkage=params.uniform_shrinkage,
    )
    order = rank_candidate_indices(probabilities, candidates)
    direct = []
    for index in order[: params.direct_top_k]:
        contributions = {
            name: float(params.weights.get(name, 0.0))
            * float(features.iloc[index][name])
            for name in FEATURE_NAMES
            if params.weights.get(name, 0.0)
        }
        direct.append(
            LearnedDirectCandidate(
                text=candidates[index],
                score=float(scores[index]),
                probability=float(probabilities[index]),
                contributions=dict(
                    sorted(
                        contributions.items(), key=lambda item: (-abs(item[1]), item[0])
                    )
                ),
            )
        )
    groups = aggregate_group_candidates(
        candidates, probabilities, aggregation=params.group_aggregation
    )
    position_pools = []
    for position in range(3):
        masses = {
            digit: math.fsum(
                float(probabilities[index])
                for index, text in enumerate(candidates)
                if int(text[position]) == digit
            )
            for digit in range(10)
        }
        position_pools.append(
            tuple(
                sorted(masses, key=lambda digit: (-masses[digit], digit))[
                    : params.position_pool_size
                ]
            )
        )
    digit_masses = {
        digit: math.fsum(
            _group_aggregate_value(item)
            for item in groups
            if str(digit) in item.group_key
        )
        for digit in range(10)
    }
    group_pool = tuple(
        sorted(digit_masses, key=lambda digit: (-digit_masses[digit], digit))[
            : params.group_digit_pool_size
        ]
    )
    canonical_probabilities = np.asarray(
        [
            probability
            for _, probability in sorted(
                zip(candidates, probabilities), key=lambda item: item[0]
            )
        ],
        dtype="<f8",
    )
    distribution_fingerprint = hashlib.sha256(
        canonical_probabilities.tobytes()
    ).hexdigest()
    return LearnedRankerPlan(
        rule_code=rule.code,
        params_fingerprint=params_fingerprint(params, feature_config),
        direct_candidates=tuple(direct),
        group_candidates=groups[: params.group_top_k],
        position_pools=tuple(position_pools),
        group_digit_pool=group_pool,
        distribution_fingerprint=distribution_fingerprint,
    )


def file_sha256(path: str | Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def learned_ranker_source_fingerprint() -> str:
    """计算 v4 核心源码的稳定组合指纹。"""

    directory = Path(__file__).resolve().parent
    lotteries_directory = directory.parent / "lotteries"
    paths = [
        directory / "digit_data.py",
        directory / "digit_statistics.py",
        directory / "digit_full_history_shadow.py",
        directory / "digit_historical_blocks.py",
        directory / "digit_learned_features.py",
        directory / "digit_lightgbm_challenger.py",
        directory / "digit_learned_ranker.py",
        directory / "digit_learned_ranker_adaptive.py",
        directory / "digit_learned_ranker_search.py",
        directory / "digit_learned_ranker_walk_forward.py",
        directory / "digit_online_gradient.py",
        directory / "digit_predictability_audit.py",
        directory / "digit_rank_ftrl.py",
        directory / "digit_sparse_frozen.py",
        lotteries_directory / "base.py",
        lotteries_directory / "fc3d.py",
        lotteries_directory / "pl3.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_feature_config_from_params(path: str | Path) -> LearnedFeatureConfig:
    """从参数元数据恢复特征配置；旧参数文件回退到设计默认值。"""

    payload = load_params_document(path)
    return (
        _feature_config_from_metadata(payload.get("metadata")) or LearnedFeatureConfig()
    )


def _atomic_write(path: Path, content: str, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() == encoded:
            return
        if immutable:
            raise FileExistsError(f"冻结快照已存在不同内容，禁止覆盖：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_immutable(path: Path, content: str, *, label: str) -> None:
    if path.exists() and path.read_bytes() != content.encode("utf-8"):
        raise FileExistsError(f"{label}已存在不同内容，禁止覆盖：{path}")


def _snapshot_generated_at(path: Path) -> str:
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("generatedAt")
            if value:
                return str(value)
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _validate_v4_snapshot(
    payload: Mapping[str, Any], history: pd.DataFrame, rule: LotteryRule
) -> tuple[pd.Series | None, str]:
    if payload.get("rankingMode") != "learned_ranker_v4":
        raise ValueError("快照不是 learned_ranker_v4")
    if payload.get("ruleCode") != rule.code:
        raise ValueError("v4 快照玩法不匹配")
    if payload.get("immutable") is not True:
        raise ValueError("v4 快照必须 immutable=true")
    fingerprint_payload = dict(payload)
    stored = fingerprint_payload.pop("snapshotFingerprint", None)
    if stored != payload_fingerprint(fingerprint_payload):
        raise ValueError("v4 快照 snapshotFingerprint 损坏")
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    source_issue = str(payload.get("sourceIssue", ""))
    if not source_issue.isdigit():
        raise ValueError("v4 快照 sourceIssue 非法")
    source_history = chronological[
        chronological["期数"].astype(int) <= int(source_issue)
    ]
    if canonical_digit_data_sha256(source_history, rule) != payload.get(
        "canonicalDataSha256"
    ):
        raise ValueError("v4 快照 canonicalDataSha256 与开奖历史不匹配")
    target_issue = payload.get("targetIssue")
    if target_issue is not None:
        matches = chronological[chronological["期数"].astype(str) == str(target_issue)]
    else:
        matches = chronological[
            chronological["期数"].astype(int) > int(source_issue)
        ].head(1)
    target = None if matches.empty else matches.iloc[0]
    dedup_key = payload_fingerprint(
        {
            "ruleCode": rule.code,
            "experimentId": payload.get("experimentId"),
            "paramsFingerprint": payload.get("paramsFingerprint"),
            "targetIssue": None if target is None else str(target["期数"]),
        }
    )
    return target, dedup_key


def process_learned_ranker_live_evaluations(
    history: pd.DataFrame,
    rule: LotteryRule,
    picks_dir: str | Path,
    evaluations_dir: str | Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """复盘 immutable v4 快照，并按实验与参数隔离累计汇总。"""

    picks = Path(picks_dir)
    output = Path(evaluations_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluations: list[dict[str, Any]] = []
    if picks.exists():
        for snapshot_path in sorted(
            picks.glob(f"{rule.code}_learned_ranker_v4_*.json")
        ):
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            target, dedup_key = _validate_v4_snapshot(payload, history, rule)
            if target is None:
                continue
            plan = dict(payload.get("candidates", payload.get("plan", {})))
            activation = dict(payload.get("activation", {}))
            actual_text = "".join(
                str(int(target[column])) for column in rule.number_columns
            )
            actual_group = "".join(sorted(actual_text))
            direct = [str(item["text"]) for item in plan.get("directCandidates", [])]
            groups = [
                str(item["group_key"]) for item in plan.get("groupCandidates", [])
            ]
            evaluation = {
                "schemaVersion": 1,
                "evaluationKind": "learned_ranker_v4_live",
                "ruleCode": rule.code,
                "experimentId": payload["experimentId"],
                "paramsFingerprint": payload["paramsFingerprint"],
                "sourceIssue": str(payload["sourceIssue"]),
                "targetIssue": str(target["期数"]),
                "actualText": actual_text,
                "activation": activation,
                "directHit": (
                    actual_text in direct if activation.get("activeDirect") else None
                ),
                "groupHit": (
                    actual_group in groups if activation.get("activeGroup") else None
                ),
                "dedupKey": dedup_key,
                "snapshotFingerprint": payload["snapshotFingerprint"],
            }
            evaluation["evaluationFingerprint"] = payload_fingerprint(evaluation)
            evaluation_path = output / f"{dedup_key}.evaluation.json"
            serialized = json.dumps(
                evaluation, ensure_ascii=False, indent=2, sort_keys=True
            )
            _preflight_immutable(evaluation_path, serialized, label="v4 实盘评估")
            _atomic_write(evaluation_path, serialized, immutable=True)
            evaluations.append(evaluation)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in evaluations:
        key = (str(item["experimentId"]), str(item["paramsFingerprint"]))
        grouped.setdefault(key, []).append(item)
    summary_paths = []
    for (experiment_id, fingerprint), items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: int(item["targetIssue"]))
        direct_values = [
            item["directHit"] for item in ordered if item["directHit"] is not None
        ]
        group_values = [
            item["groupHit"] for item in ordered if item["groupHit"] is not None
        ]
        summary = {
            "schemaVersion": 1,
            "summaryKind": "learned_ranker_v4_live",
            "ruleCode": rule.code,
            "experimentId": experiment_id,
            "paramsFingerprint": fingerprint,
            "periodCount": len(ordered),
            "directEvaluated": len(direct_values),
            "directHits": sum(bool(value) for value in direct_values),
            "groupEvaluated": len(group_values),
            "groupHits": sum(bool(value) for value in group_values),
            "dedupKeys": [item["dedupKey"] for item in ordered],
        }
        summary["summaryFingerprint"] = payload_fingerprint(summary)
        summary_path = output / (
            f"{rule.code}_{experiment_id}_{fingerprint[:16]}.summary.json"
        )
        _atomic_write(
            summary_path,
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        )
        summary_paths.append(summary_path)
    return evaluations, summary_paths


def _daily_markdown(payload: Mapping[str, Any]) -> str:
    plan = dict(payload["plan"])
    direct_heading = (
        "## 主推荐（直选）"
        if plan.get("activeDirect")
        else "## 研究分区（直选，未启用）"
    )
    group_heading = (
        "## 主推荐（组选）"
        if plan.get("activeGroup")
        else "## 研究分区（组选，未启用）"
    )
    position_heading = (
        "## 主推荐（位置池）"
        if plan.get("activePosition")
        else "## 研究分区（位置池，未启用）"
    )
    lines = [
        f"# {payload['displayName']} learned_ranker_v4 日报",
        "",
        f"- 状态：**{payload['mode']}**",
        f"- 最新历史期：`{payload['sourceIssue']}`",
        f"- 参数指纹：`{payload['paramsFingerprint']}`",
        f"- CSV SHA-256：`{payload['csvSha256']}`",
        f"- 源码指纹：`{payload['sourceFingerprint']}`",
        "",
        direct_heading,
        "",
        "| 排名 | 号码 | 分数 | 归一化概率 | 主要贡献 |",
        "|---:|---|---:|---:|---|",
    ]
    for index, item in enumerate(plan["directCandidates"], 1):
        contributions = sorted(
            dict(item["contributions"]).items(),
            key=lambda value: (-abs(float(value[1])), value[0]),
        )[:3]
        contribution_text = "，".join(
            f"{name}:{float(value):+.3f}" for name, value in contributions
        )
        lines.append(
            f"| {index} | `{item['text']}` | {float(item['score']):.6f} | "
            f"{float(item['probability']):.6%} | {contribution_text} |"
        )
    lines.extend(["", group_heading, ""])
    for index, item in enumerate(plan["groupCandidates"], 1):
        if item.get("aggregation") == "sum_prob":
            value_text = f"probability `{float(item['probability']):.6%}`"
        else:
            value_text = (
                f"score `{float(item['score']):.6f}`，aggregation "
                f"`{item['aggregation']}`"
            )
        lines.append(
            f"{index}. `{item['group_key']}`：{value_text}，排列数 `{item['permutations']}`"
        )
    recent_evaluation = dict(payload.get("recentEvaluation", {}))
    if recent_evaluation:
        lines.extend(["", "## 近期冻结评估摘要", ""])
        for window in ("50", "100", "300"):
            item = dict(recent_evaluation.get(window, {}))
            if not item:
                continue
            lines.append(
                f"- 最近 {window} 期（实际 `{item['periods']}` 期）："
                f"直选 `{item['directHits']}`，组选 `{item['groupHits']}`，"
                f"平均排名 `{float(item['meanRank']):.2f}`。"
            )
    position_pools = " / ".join(
        "".join(str(value) for value in pool) for pool in plan["positionPools"]
    )
    direct_count = int(np.prod([len(pool) for pool in plan["positionPools"]]))
    lines.extend(
        [
            "",
            position_heading,
            "",
            f"- 位置复式池：`{position_pools}`，共 `{direct_count}` 注，按每注 2 元为 `{direct_count * 2}` 元。",
            f"- 组选数字池：`{''.join(str(value) for value in plan['groupDigitPool'])}`。",
            "",
            "## 风险声明",
            "",
            "彩票开奖结果具有高度随机性。归一化概率仅用于候选排序，不代表真实开奖概率；本报告不宣称预测有效，不保证中奖或盈利，也不构成投注建议。",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_frozen_evaluation(
    path: Path,
    *,
    rule_code: str,
    model_fingerprint: str,
    params_artifact_fingerprint: str,
    source_fingerprint: str,
    canonical_data_sha256: str,
) -> tuple[
    dict[str, Any],
    str | None,
    dict[str, bool],
    dict[str, dict[str, float | int]],
]:
    validation = {
        "exists": path.exists(),
        "readable": False,
        "ruleMatched": False,
        "paramsMatched": False,
        "paramsArtifactMatched": False,
        "sourceMatched": False,
        "canonicalMatched": False,
        "fingerprintValid": False,
        "frozenTestMatched": False,
        "promoted": False,
    }
    if not path.exists():
        return (
            resolve_activation(
                common_passed=False,
                direct_passed=False,
                group_passed=False,
                position_passed=False,
            ),
            None,
            validation,
            {},
        )
    evaluation_fingerprint = file_sha256(path)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError("冻结评估根节点必须为 JSON 对象")
        evaluation = loaded
        fingerprint_payload = dict(evaluation)
        stored_evaluation_fingerprint = fingerprint_payload.pop(
            "reportFingerprint", None
        )
        fingerprint_valid = stored_evaluation_fingerprint == payload_fingerprint(
            fingerprint_payload
        )
        rule_matched = evaluation.get("ruleCode") == rule_code
        params_matched = evaluation.get("paramsFingerprint") == model_fingerprint
        params_artifact_matched = (
            evaluation.get("paramsArtifactFingerprint") == params_artifact_fingerprint
        )
        source_matched = evaluation.get("sourceFingerprint") == source_fingerprint
        canonical_matched = (
            evaluation.get("canonicalDataSha256") == canonical_data_sha256
        )
        split = dict(evaluation.get("split", {}))
        expected_indices = list(
            range(
                int(split.get("validationEnd", 0)),
                int(split.get("testEnd", 0)),
            )
        )
        periods = list(evaluation.get("periods", []))
        frozen_test_matched = (
            evaluation.get("evaluationKind") == "frozen_test"
            and evaluation.get("testSegmentUsedForSelection") is False
            and bool(expected_indices)
            and evaluation.get("testTargetIndices") == expected_indices
            and len(periods) == len(expected_indices)
        )
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return (
            resolve_activation(
                common_passed=False,
                direct_passed=False,
                group_passed=False,
                position_passed=False,
            ),
            evaluation_fingerprint,
            validation,
            {},
        )
    evidence_valid = all(
        (
            rule_matched,
            params_matched,
            params_artifact_matched,
            source_matched,
            canonical_matched,
            fingerprint_valid,
            frozen_test_matched,
        )
    )
    gate_payload = dict(evaluation.get("gate", {}))
    evaluation_activation = dict(gate_payload.get("activation", {}))
    if evaluation_activation:
        common_passed = bool(evaluation_activation.get("commonPassed", False))
        direct_passed = bool(evaluation_activation.get("directPassed", False))
        group_passed = bool(evaluation_activation.get("groupPassed", False))
        position_passed = bool(evaluation_activation.get("positionPassed", False))
    else:
        common_passed = direct_passed = group_passed = position_passed = False
    activation = resolve_activation(
        common_passed=bool(common_passed and evidence_valid),
        direct_passed=bool(direct_passed and evidence_valid),
        group_passed=bool(group_passed and evidence_valid),
        position_passed=bool(position_passed and evidence_valid),
    )
    validation = {
        "exists": True,
        "readable": True,
        "ruleMatched": rule_matched,
        "paramsMatched": params_matched,
        "paramsArtifactMatched": params_artifact_matched,
        "sourceMatched": source_matched,
        "canonicalMatched": canonical_matched,
        "fingerprintValid": fingerprint_valid,
        "frozenTestMatched": frozen_test_matched,
        "promoted": bool(activation["overallPassed"]),
    }
    recent_evaluation: dict[str, dict[str, float | int]] = {}
    if evidence_valid:
        for window in (50, 100, 300):
            selected = periods[-window:]
            ranks = [
                int(item.get("actual_rank", item.get("actualRank", 0)))
                for item in selected
            ]
            recent_evaluation[str(window)] = {
                "periods": len(selected),
                "directHits": sum(
                    bool(item.get("direct_hit", item.get("directHit", False)))
                    for item in selected
                ),
                "groupHits": sum(
                    bool(item.get("group_hit", item.get("groupHit", False)))
                    for item in selected
                ),
                "meanRank": float(np.mean(ranks)) if ranks else 0.0,
            }
    return activation, evaluation_fingerprint, validation, recent_evaluation


def generate_learned_ranker_daily(
    lottery: str,
    csv_path: str | Path,
    params_path: str | Path,
    *,
    output_dir: str | Path = "reports",
    evaluation_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """生成 v4 Markdown、JSON 和不可覆盖冻结快照。"""

    from src.lotteries import get_lottery_rule

    rule = get_lottery_rule(lottery)
    if rule.draw_count != 3 or rule.code not in {"fc3d", "pl3"}:
        raise ValueError("learned_ranker_v4 首版只支持 fc3d/pl3，pl5 请保持旧路径")
    history = load_digit_csv(csv_path, rule)
    if history.empty:
        raise ValueError("历史 CSV 不能为空")
    params = load_params(params_path)
    params_metadata = load_params_metadata(params_path)
    feature_config = load_feature_config_from_params(params_path)
    params_artifact_fingerprint = load_params_artifact_fingerprint(params_path)
    current_params_fingerprint = params_fingerprint(params, feature_config)
    state = build_history_state(history, rule, feature_config)
    features = build_candidate_features(state, rule)
    plan = build_learned_ranker_plan(features, params, rule, feature_config)
    report_dir = Path(output_dir)
    process_learned_ranker_live_evaluations(
        history,
        rule,
        report_dir / "picks" / "digit",
        report_dir / "evaluations" / "learned_ranker_v4_live",
    )
    effective_evaluation = (
        Path(evaluation_path)
        if evaluation_path is not None
        else report_dir / "evaluations" / f"learned_ranker_v4_{rule.code}.json"
    )
    source_fingerprint = learned_ranker_source_fingerprint()
    canonical_data_sha256 = canonical_digit_data_sha256(history, rule)
    frozen_data_canonical_sha256 = canonical_data_sha256
    split_payload = params_metadata.get("split")
    if isinstance(split_payload, Mapping) and split_payload.get("testEnd") is not None:
        test_end = int(split_payload["testEnd"])
        chronological = sort_digit_dataframe_by_issue(
            normalize_digit_dataframe(history, rule), ascending=True
        )
        if test_end <= 0 or test_end > len(chronological):
            frozen_data_canonical_sha256 = ""
        else:
            frozen_data_canonical_sha256 = canonical_digit_data_sha256(
                chronological.iloc[:test_end], rule
            )
    (
        activation,
        evaluation_fingerprint,
        evaluation_validation,
        recent_evaluation,
    ) = _validate_frozen_evaluation(
        effective_evaluation,
        rule_code=rule.code,
        model_fingerprint=current_params_fingerprint,
        params_artifact_fingerprint=params_artifact_fingerprint,
        source_fingerprint=source_fingerprint,
        canonical_data_sha256=frozen_data_canonical_sha256,
    )
    no_signal = params.uniform_shrinkage <= 0.0
    if no_signal:
        activation = resolve_activation(
            common_passed=False,
            direct_passed=False,
            group_passed=False,
            position_passed=False,
        )
        evaluation_validation = {
            **evaluation_validation,
            "valid": False,
            "abstained": True,
        }
    abstention_reason = "均匀收缩λ=0：模型无可确认信号" if no_signal else None
    gate_passed = bool(activation["overallPassed"])
    active_labels = [
        label
        for enabled, label in (
            (activation["activeDirect"], "直选"),
            (activation["activeGroup"], "组选"),
            (activation["activePosition"], "位置池"),
        )
        if enabled
    ]
    mode = (
        "本期无可确认预测信号，已放弃主推荐"
        if no_signal
        else (
            f"已启用：{'、'.join(active_labels)}；其余保持研究"
            if active_labels
            else "研究模式，不接入主推荐"
        )
    )
    experiment_id = str(
        params_metadata.get(
            "experimentId", f"learned_ranker_v4_{rule.code}_{params.random_seed}"
        )
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise ValueError("experimentId 只允许字母、数字、下划线和连字符")
    artifact_segment = f"_{experiment_id}_{current_params_fingerprint[:12]}"
    snapshot_path = (
        report_dir
        / "picks"
        / "digit"
        / (
            f"{rule.code}_learned_ranker_v4{artifact_segment}_"
            f"{state.history_end_issue}.json"
        )
    )
    payload = {
        "schemaVersion": 2,
        "modelVersion": "learned_ranker_v4",
        "rankingMode": "learned_ranker_v4",
        "experimentId": experiment_id,
        "ruleCode": rule.code,
        "displayName": rule.display_name,
        "sourceIssue": state.history_end_issue,
        "targetIssue": None,
        "nextIssueInterpretation": "当前不能可靠推导下一期号；后续数据中取 sourceIssue 之后按数值期号排序的第一期。",
        "generatedAt": _snapshot_generated_at(snapshot_path),
        "immutable": True,
        "historyPeriods": len(state.numbers),
        "mode": mode,
        "abstained": no_signal,
        "noSignal": no_signal,
        "abstentionReason": abstention_reason,
        "uniformShrinkage": params.uniform_shrinkage,
        "gatePassed": gate_passed,
        "directPassed": bool(activation["directPassed"]),
        "groupPassed": bool(activation["groupPassed"]),
        "positionPassed": bool(activation["positionPassed"]),
        "activation": activation,
        "csvSha256": file_sha256(csv_path),
        "canonicalDataSha256": canonical_data_sha256,
        "frozenDataCanonicalSha256": frozen_data_canonical_sha256,
        "sourceFingerprint": source_fingerprint,
        "paramsFingerprint": current_params_fingerprint,
        "paramsArtifactFingerprint": params_artifact_fingerprint,
        "evaluationFingerprint": evaluation_fingerprint,
        "evaluationValidation": evaluation_validation,
        "recentEvaluation": recent_evaluation,
        "featureConfig": {
            "windows": list(feature_config.windows),
            "alpha": feature_config.alpha,
            "halfLife": feature_config.half_life,
            "omissionCap": feature_config.omission_cap,
            "windowWeights": dict(feature_config.window_weights or ()),
        },
        "plan": partition_plan_by_activation(plan.to_dict(), activation),
        "disclaimer": "开奖结果具有随机性；不宣称预测有效，不保证中奖或盈利。",
    }
    registry_path = report_dir / "state" / "strategy_registry.json"
    strategy_statuses: dict[str, str] = {}
    for output_kind, activation_key in (
        ("direct", "activeDirect"),
        ("group", "activeGroup"),
        ("position", "activePosition"),
    ):
        requested_status = (
            StrategyStatus.ACTIVE
            if activation[activation_key]
            else StrategyStatus.RESEARCH
        )
        strategy_id = f"{experiment_id}:{rule.code}:{output_kind}"
        registry = update_strategy_registry(
            registry_path,
            StrategyRegistryUpdate(
                strategy_id=strategy_id,
                lottery=rule.code,
                output_kind=output_kind,
                requested_status=requested_status,
                reasons=(
                    ("独立冻结闸门通过",)
                    if requested_status is StrategyStatus.ACTIVE
                    else ("独立冻结闸门未通过或证据无效",)
                ),
                data_fingerprint=frozen_data_canonical_sha256,
                params_fingerprint=current_params_fingerprint,
                source_fingerprint=source_fingerprint,
                occurred_at=str(payload["generatedAt"]),
            ),
        )
        entries = dict(registry["entries"])
        strategy_statuses[output_kind] = str(dict(entries[strategy_id])["status"])
    payload["strategyRegistry"] = str(registry_path.relative_to(report_dir))
    payload["strategyStatuses"] = strategy_statuses
    payload["candidates"] = payload["plan"]
    payload["snapshotFingerprint"] = payload_fingerprint(payload)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    daily_dir = report_dir / "daily" / rule.code
    daily_stem = f"{rule.code}{artifact_segment}_daily_{state.history_end_issue}"
    markdown_path = daily_dir / f"{daily_stem}.md"
    json_path = daily_dir / f"{daily_stem}.json"
    markdown = _daily_markdown(payload)
    _preflight_immutable(snapshot_path, serialized, label="冻结快照")
    _preflight_immutable(markdown_path, markdown, label="v4 日报")
    _preflight_immutable(json_path, serialized, label="v4 日报")
    _atomic_write(markdown_path, markdown)
    _atomic_write(json_path, serialized)
    _atomic_write(snapshot_path, serialized, immutable=True)
    return markdown_path, json_path, snapshot_path

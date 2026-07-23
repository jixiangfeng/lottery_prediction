# -*- coding: utf-8 -*-
"""快乐8选5边际概率开发挑战器。"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import hypergeom

EXPERT_NAMES = (
    "uniform",
    "ewma20",
    "ewma80",
    "ewma300",
    "omissionShrink",
    "pairwiseAdjusted",
)
RESEARCH_ONLY_WORDING = "研究观察，未通过准入，不是正式推荐"
_POOL_COMBINATION_INDEXES = np.asarray(
    list(itertools.combinations(range(20), 5)), dtype=np.int16
)
_POOL_COMBINATION_INCIDENCE = np.zeros(
    (len(_POOL_COMBINATION_INDEXES), 20), dtype=np.int8
)
_POOL_COMBINATION_INCIDENCE[
    np.repeat(np.arange(len(_POOL_COMBINATION_INDEXES)), 5),
    _POOL_COMBINATION_INDEXES.reshape(-1),
] = 1
_POOL_COMBINATION_INCIDENCE.setflags(write=False)


@dataclass(frozen=True)
class Kl8Pick5Config:
    """快乐8选5首版预注册配置；默认值不得依据结果调整。"""

    epsilon: float = 1e-6
    omission_shrinkage: float = 0.9
    omission_scale: float = 80.0
    pair_half_life: float = 80.0
    pair_marginal_weight: float = 0.05
    combo_pair_weight: float = 0.02
    concentration_penalty: float = 0.05
    top_pool_size: int = 20
    output_combinations: int = 5
    calibration_temperatures: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    alpha: float = 0.05
    hedge_learning_rate: float = 0.05
    max_expert_loss: float = 10.0
    bootstrap_seed: int = 20260722
    bootstrap_resamples: int = 2000
    stability_blocks: int = 5
    minimum_mean_hits_per_ticket: float = 1.25
    warmup_periods: int = 300
    search_periods: int = 500
    calibration_periods: int = 250
    evaluation_periods: int = 500
    frozen_periods: int = 500
    required_null_iterations: int = 5000

    def __post_init__(self) -> None:
        """拒绝非有限、越界或破坏固定正式设计的配置。"""

        finite_fields = {
            "epsilon": self.epsilon,
            "omission_shrinkage": self.omission_shrinkage,
            "omission_scale": self.omission_scale,
            "pair_half_life": self.pair_half_life,
            "pair_marginal_weight": self.pair_marginal_weight,
            "combo_pair_weight": self.combo_pair_weight,
            "concentration_penalty": self.concentration_penalty,
            "alpha": self.alpha,
            "hedge_learning_rate": self.hedge_learning_rate,
            "max_expert_loss": self.max_expert_loss,
            "minimum_mean_hits_per_ticket": self.minimum_mean_hits_per_ticket,
        }
        for name, value in finite_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name}必须为有限数")
        if not 0.0 < self.epsilon < 0.25:
            raise ValueError("epsilon必须满足0<epsilon<0.25")
        if not 0.0 < self.alpha < 0.5:
            raise ValueError("alpha必须满足0<alpha<0.5")
        positive_fields = {
            "omission_scale": self.omission_scale,
            "pair_half_life": self.pair_half_life,
            "hedge_learning_rate": self.hedge_learning_rate,
            "max_expert_loss": self.max_expert_loss,
        }
        for name, value in positive_fields.items():
            if value <= 0.0:
                raise ValueError(f"{name}必须为正数")
        nonnegative_fields = {
            "omission_shrinkage": self.omission_shrinkage,
            "pair_marginal_weight": self.pair_marginal_weight,
            "combo_pair_weight": self.combo_pair_weight,
            "concentration_penalty": self.concentration_penalty,
            "minimum_mean_hits_per_ticket": self.minimum_mean_hits_per_ticket,
        }
        for name, value in nonnegative_fields.items():
            if value < 0.0:
                raise ValueError(f"{name}不得为负数")
        if self.omission_shrinkage > 1.0:
            raise ValueError("omission_shrinkage必须位于0..1")
        if self.minimum_mean_hits_per_ticket > 5.0:
            raise ValueError("minimum_mean_hits_per_ticket不得超过5")
        if type(self.top_pool_size) is not int or self.top_pool_size != 20:
            raise ValueError("top_pool_size必须恰好为20")
        if type(self.output_combinations) is not int or self.output_combinations != 5:
            raise ValueError("output_combinations必须恰好为5")
        if type(self.stability_blocks) is not int or self.stability_blocks != 5:
            raise ValueError("stability_blocks必须恰好为5")
        if type(self.frozen_periods) is not int or self.frozen_periods != 500:
            raise ValueError("frozen_periods必须恰好为500")
        if (
            type(self.required_null_iterations) is not int
            or self.required_null_iterations < 5000
        ):
            raise ValueError("required_null_iterations必须至少为5000")
        positive_integer_fields = {
            "bootstrap_resamples": self.bootstrap_resamples,
            "warmup_periods": self.warmup_periods,
            "search_periods": self.search_periods,
            "calibration_periods": self.calibration_periods,
            "evaluation_periods": self.evaluation_periods,
        }
        for name, value in positive_integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}必须为正整数")
        if (
            isinstance(self.bootstrap_seed, bool)
            or not isinstance(self.bootstrap_seed, int)
            or self.bootstrap_seed < 0
        ):
            raise ValueError("bootstrap_seed必须为非负整数")
        if not self.calibration_temperatures:
            raise ValueError("calibration_temperatures不得为空")
        if any(
            isinstance(temperature, bool)
            or not isinstance(temperature, Real)
            or not math.isfinite(float(temperature))
            or temperature <= 0.0
            for temperature in self.calibration_temperatures
        ):
            raise ValueError("calibration_temperatures必须全部为正有限数")

    @property
    def required_periods(self) -> int:
        """返回开发区所需总期数。"""

        return (
            self.warmup_periods
            + self.search_periods
            + self.calibration_periods
            + self.evaluation_periods
        )

    @classmethod
    def smoke(cls) -> "Kl8Pick5Config":
        """返回保持完整流程但缩短运行时间的冒烟配置。"""

        return cls(
            warmup_periods=20,
            search_periods=10,
            calibration_periods=10,
            evaluation_periods=10,
            bootstrap_resamples=40,
            stability_blocks=5,
        )


def payload_sha256(payload: object) -> str:
    """返回确定性 JSON SHA-256。"""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_compatible(payload: object) -> object:
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def assert_canonical_formal_config(config: Kl8Pick5Config) -> None:
    """要求正式路径逐字段等于唯一默认配置。"""

    if asdict(config) != asdict(Kl8Pick5Config()):
        raise ValueError("正式路径必须使用唯一规范正式配置")


def _as_float(value: object) -> float:
    return float(cast(float, value))


def _as_int(value: object) -> int:
    return int(cast(int, value))


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}必须是对象")
    return cast(Mapping[str, object], value)


def _normalize_frozen_boundary(
    frozen_boundary: Mapping[str, object], *, frozen_periods_excluded: int
) -> dict[str, object]:
    first_issue = _parse_issue(frozen_boundary.get("firstIssue"))
    last_issue = _parse_issue(frozen_boundary.get("lastIssue"))
    if int(first_issue) > int(last_issue):
        raise ValueError("Frozen边界首期不得晚于末期")
    if frozen_periods_excluded <= 0:
        raise ValueError("Frozen边界必须排除至少1期")
    return {
        "periodsExcluded": frozen_periods_excluded,
        "firstIssue": first_issue,
        "lastIssue": last_issue,
        "numbersRead": False,
    }


def _parse_issue(value: object) -> str:
    issue = str(value).strip()
    if not re.fullmatch(r"\d+", issue):
        raise ValueError(f"快乐8期号必须为纯数字，收到：{value}")
    return issue


def _parse_date(value: object) -> str:
    text = str(value).strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as error:
        raise ValueError(f"快乐8日期必须为YYYY-MM-DD，收到：{value}") from error


def _parse_numbers(value: object) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        numbers = [int(number) for number in value]
    else:
        numbers = [int(token) for token in re.findall(r"\d+", str(value))]
    if len(numbers) != 20:
        raise ValueError(f"快乐8每期必须恰好20个号码，收到{len(numbers)}个")
    if len(set(numbers)) != 20:
        raise ValueError("快乐8每期必须包含20个唯一号码")
    if any(number < 1 or number > 80 for number in numbers):
        raise ValueError("快乐8号码范围必须为1..80")
    return sorted(numbers)


def normalize_kl8_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """校验并标准化快乐8 DataFrame 为升序 issue/date/numbers。"""

    required = {"issue", "date", "numbers"}
    if not required.issubset(frame.columns):
        raise ValueError("快乐8数据必须包含issue、date、numbers列")
    rows = [
        {
            "issue": _parse_issue(row.issue),
            "date": _parse_date(row.date),
            "numbers": _parse_numbers(row.numbers),
        }
        for row in frame[["issue", "date", "numbers"]].itertuples(index=False)
    ]
    output = pd.DataFrame(rows, columns=["issue", "date", "numbers"])
    if output.empty:
        return output
    numeric_issues = output["issue"].map(int)
    if numeric_issues.duplicated().any():
        raise ValueError("快乐8数据包含重复期号")
    output = (
        output.assign(_issue=numeric_issues)
        .sort_values("_issue", kind="mergesort")
        .drop(columns="_issue")
        .reset_index(drop=True)
    )
    dates = pd.to_datetime(output["date"], format="%Y-%m-%d")
    if not dates.is_monotonic_increasing:
        raise ValueError("快乐8期号与日期顺序不一致")
    return output


def canonical_kl8_sha256(frame: pd.DataFrame) -> str:
    """返回按期号时间正序序列化的语义数据哈希。"""

    normalized = normalize_kl8_dataframe(frame)
    payload = [
        {"issue": row.issue, "date": row.date, "numbers": row.numbers}
        for row in normalized.itertuples(index=False)
    ]
    return payload_sha256(payload)


def load_kl8_development_csv(
    path: str | Path, *, frozen_periods: int = 500
) -> tuple[pd.DataFrame, dict[str, object]]:
    """两遍读取CSV；Frozen遍只读取issue/date，绝不解析号码字段。"""

    source = Path(path)
    metadata: list[tuple[int, str, str]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"issue", "date", "numbers"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("快乐8 CSV必须包含issue、date、numbers列")
        for row_index, row in enumerate(reader):
            metadata.append(
                (row_index, _parse_issue(row["issue"]), _parse_date(row["date"]))
            )
    if frozen_periods < 0 or len(metadata) <= frozen_periods:
        raise ValueError("快乐8历史不足以排除Frozen并保留开发数据")
    issues = [int(item[1]) for item in metadata]
    if len(set(issues)) != len(issues):
        raise ValueError("快乐8数据包含重复期号")
    ordered = sorted(metadata, key=lambda item: int(item[1]))
    if [item[2] for item in ordered] != sorted(item[2] for item in ordered):
        raise ValueError("快乐8期号与日期顺序不一致")
    development_indexes = (
        {item[0] for item in ordered[:-frozen_periods] if frozen_periods}
        if frozen_periods
        else {item[0] for item in ordered}
    )
    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index not in development_indexes:
                continue
            rows.append(
                {
                    "issue": row["issue"],
                    "date": row["date"],
                    "numbers": _parse_numbers(row["numbers"]),
                }
            )
    development = normalize_kl8_dataframe(pd.DataFrame(rows))
    frozen = ordered[-frozen_periods:] if frozen_periods else []
    return development, {
        "fullPeriods": len(metadata),
        "developmentPeriods": len(development),
        "frozenPeriods": frozen_periods,
        "frozenRead": False,
        "frozenBoundary": (
            {"firstIssue": frozen[0][1], "lastIssue": frozen[-1][1]} if frozen else None
        ),
    }


def normalize_sum20(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    """将80维有限分数裁剪并唯一缩放为期望正例数20。"""

    scores = np.asarray(values, dtype=np.float64)
    if scores.shape != (80,) or not np.isfinite(scores).all():
        raise ValueError("专家必须输出80个float64有限概率")
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    lower = 0.0
    upper = 80.0 / float(clipped.sum())
    for _ in range(80):
        scale = (lower + upper) / 2.0
        if scale == lower or scale == upper:
            break
        total = float(np.clip(clipped * scale, epsilon, 1.0 - epsilon).sum())
        if total < 20.0:
            lower = scale
        else:
            upper = scale
    result = np.clip(clipped * ((lower + upper) / 2.0), epsilon, 1.0 - epsilon).astype(
        np.float64
    )
    result *= 20.0 / float(result.sum())
    return result.astype(np.float64)


@dataclass
class _OnlineState:
    ewma20: np.ndarray
    ewma80: np.ndarray
    ewma300: np.ndarray
    gaps: np.ndarray
    pair: np.ndarray
    cumulative_losses: np.ndarray

    @classmethod
    def initial(cls) -> "_OnlineState":
        baseline = np.full(80, 0.25, dtype=np.float64)
        return cls(
            ewma20=baseline.copy(),
            ewma80=baseline.copy(),
            ewma300=baseline.copy(),
            gaps=np.zeros(80, dtype=np.float64),
            pair=np.full((80, 80), (20 / 80) * (19 / 79), dtype=np.float64),
            cumulative_losses=np.zeros(6, dtype=np.float64),
        )


def _expert_probabilities(state: _OnlineState, config: Kl8Pick5Config) -> np.ndarray:
    uniform = np.full(80, 0.25, dtype=np.float64)
    omission_raw = 1.0 - np.exp(-state.gaps / config.omission_scale)
    omission = config.omission_shrinkage * uniform + (
        1.0 - config.omission_shrinkage
    ) * normalize_sum20(omission_raw, epsilon=config.epsilon)
    marginals = np.clip(state.ewma80, config.epsilon, 1.0 - config.epsilon)
    expected_pair = np.outer(marginals, marginals)
    pair_lift = np.clip(
        state.pair / np.maximum(expected_pair, config.epsilon) - 1.0,
        -1.0,
        2.0,
    )
    context = np.asarray(
        sorted(range(80), key=lambda index: (-marginals[index], index))[:20],
        dtype=np.int64,
    )
    pair_adjustment = np.empty(80, dtype=np.float64)
    for index in range(80):
        peers = context[context != index]
        pair_adjustment[index] = float(pair_lift[index, peers].mean())
    pair_adjustment = np.clip(pair_adjustment, -1.0, 1.0)
    pair_adjusted = marginals * np.exp(config.pair_marginal_weight * pair_adjustment)
    outputs = np.vstack(
        [
            uniform,
            normalize_sum20(state.ewma20, epsilon=config.epsilon),
            normalize_sum20(state.ewma80, epsilon=config.epsilon),
            normalize_sum20(state.ewma300, epsilon=config.epsilon),
            normalize_sum20(omission, epsilon=config.epsilon),
            normalize_sum20(pair_adjusted, epsilon=config.epsilon),
        ]
    ).astype(np.float64)
    if outputs.shape != (6, 80) or not np.allclose(outputs.sum(axis=1), 20.0):
        raise AssertionError("六专家概率输出契约失败")
    return outputs


def _hedge_weights(state: _OnlineState, config: Kl8Pick5Config) -> np.ndarray:
    shifted = state.cumulative_losses - float(state.cumulative_losses.min())
    weights = np.exp(-config.hedge_learning_rate * shifted)
    return (weights / weights.sum()).astype(np.float64)


def _labels(numbers: Sequence[int]) -> np.ndarray:
    labels = np.zeros(80, dtype=np.float64)
    labels[np.asarray(numbers, dtype=int) - 1] = 1.0
    return labels


def _bernoulli_losses(
    probabilities: np.ndarray, labels: np.ndarray, config: Kl8Pick5Config
) -> np.ndarray:
    clipped = np.clip(probabilities, config.epsilon, 1.0 - config.epsilon)
    losses = -(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped)).mean(
        axis=1
    )
    return np.clip(losses, 0.0, config.max_expert_loss).astype(np.float64)


def _update_state(
    state: _OnlineState, labels: np.ndarray, losses: np.ndarray, config: Kl8Pick5Config
) -> None:
    for half_life, values in (
        (20.0, state.ewma20),
        (80.0, state.ewma80),
        (300.0, state.ewma300),
    ):
        rho = 2.0 ** (-1.0 / half_life)
        values *= rho
        values += (1.0 - rho) * labels
    state.gaps += 1.0
    state.gaps[labels == 1.0] = 0.0
    pair_rho = 2.0 ** (-1.0 / config.pair_half_life)
    cooccurrence = np.outer(labels, labels)
    state.pair *= pair_rho
    state.pair += (1.0 - pair_rho) * cooccurrence
    np.fill_diagonal(state.pair, 0.0)
    state.cumulative_losses += losses


def _temperature_scale(
    probabilities: np.ndarray, temperature: float, epsilon: float
) -> np.ndarray:
    scaled = expit(logit(np.clip(probabilities, epsilon, 1 - epsilon)) / temperature)
    return normalize_sum20(scaled, epsilon=epsilon)


def _pool_overlap_counts(chosen_index: int) -> np.ndarray:
    """返回Top20组合池中每个组合与指定组合的交集大小。"""

    if type(chosen_index) is not int or not 0 <= chosen_index < len(
        _POOL_COMBINATION_INDEXES
    ):
        raise ValueError("chosen_index超出Top20组合池")
    chosen_columns = _POOL_COMBINATION_INDEXES[chosen_index]
    return _POOL_COMBINATION_INCIDENCE[:, chosen_columns].sum(axis=1, dtype=np.int8)


def generate_top5_combinations(
    probabilities: np.ndarray, pair_scores: np.ndarray, config: Kl8Pick5Config
) -> list[list[int]]:
    """仅枚举边际Top20内组合，以确定性规则返回五个唯一选5组合。"""

    probs = normalize_sum20(probabilities, epsilon=config.epsilon)
    pool = sorted(range(80), key=lambda index: (-probs[index], index))[
        : config.top_pool_size
    ]
    combo_indexes = _POOL_COMBINATION_INDEXES
    combo_numbers = np.asarray(pool, dtype=np.int16)[combo_indexes]
    logits = logit(np.clip(probs, config.epsilon, 1 - config.epsilon))
    scores = logits[combo_numbers].sum(axis=1)
    for left, right in itertools.combinations(range(5), 2):
        scores += (
            config.combo_pair_weight
            * pair_scores[combo_numbers[:, left], combo_numbers[:, right]]
        )
    selected: list[np.ndarray] = []
    available = np.ones(len(combo_numbers), dtype=bool)
    adjusted = scores.copy()
    for _ in range(config.output_combinations):
        adjusted[~available] = -np.inf
        best_score = float(np.max(adjusted))
        tied = np.flatnonzero(np.isclose(adjusted, best_score, rtol=0.0, atol=1e-15))
        chosen_index = min(
            tied,
            key=lambda index: tuple(
                sorted(int(value) for value in combo_numbers[index])
            ),
        )
        chosen = np.sort(combo_numbers[chosen_index])
        selected.append(chosen)
        available[chosen_index] = False
        overlaps = _pool_overlap_counts(int(chosen_index))
        adjusted -= config.concentration_penalty * np.square(overlaps)
    return [[int(index) + 1 for index in combo] for combo in selected]


def _hypergeometric_tail(hits: int) -> float:
    return float(hypergeom.sf(hits - 1, 80, 20, 5))


@lru_cache(maxsize=None)
def _multiplicity_total_hits_pmf(
    population_size: int, draw_count: int, multiplicity_counts: tuple[int, ...]
) -> tuple[float, ...]:
    """按号码票面重数分组，精确计算不放回抽样总命中数 PMF。"""

    if population_size <= 0 or not 0 <= draw_count <= population_size:
        raise ValueError("总体与抽样数量无效")
    if any(count < 0 for count in multiplicity_counts):
        raise ValueError("重数分组计数不得为负")
    if sum(multiplicity_counts) != population_size:
        raise ValueError("重数分组计数之和必须等于总体大小")
    max_total = draw_count * (len(multiplicity_counts) - 1)
    ways = [[0] * (max_total + 1) for _ in range(draw_count + 1)]
    ways[0][0] = 1
    processed = 0
    for multiplicity, category_count in enumerate(multiplicity_counts):
        updated = [[0] * (max_total + 1) for _ in range(draw_count + 1)]
        for chosen_before in range(min(processed, draw_count) + 1):
            for total_before, prior_ways in enumerate(ways[chosen_before]):
                if prior_ways == 0:
                    continue
                maximum = min(category_count, draw_count - chosen_before)
                for chosen_here in range(maximum + 1):
                    updated[chosen_before + chosen_here][
                        total_before + multiplicity * chosen_here
                    ] += prior_ways * math.comb(category_count, chosen_here)
        ways = updated
        processed += category_count
    denominator = math.comb(population_size, draw_count)
    return tuple(value / denominator for value in ways[draw_count])


def _portfolio_multiplicity_signature(
    combinations: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    multiplicities = np.zeros(80, dtype=np.int8)
    for combination in combinations:
        if len(combination) != 5 or len(set(combination)) != 5:
            raise ValueError("每张快乐8选5票必须包含5个唯一号码")
        for number in combination:
            if number < 1 or number > 80:
                raise ValueError("快乐8选5号码必须位于1..80")
            multiplicities[number - 1] += 1
    if len(combinations) != 5:
        raise ValueError("完整投资组合必须恰好包含5张票")
    return tuple(int(np.count_nonzero(multiplicities == value)) for value in range(6))


def _exact_portfolio_total_hits_pvalue(
    total_hits: int, period_combinations: Sequence[Sequence[Sequence[int]]]
) -> float:
    distribution = np.asarray([1.0], dtype=np.float64)
    for combinations in period_combinations:
        signature = _portfolio_multiplicity_signature(combinations)
        period_pmf = np.asarray(
            _multiplicity_total_hits_pmf(80, 20, signature), dtype=np.float64
        )
        distribution = np.convolve(distribution, period_pmf)
    if total_hits < 0:
        return 1.0
    if total_hits >= len(distribution):
        return 0.0
    return float(np.clip(distribution[total_hits:].sum(), 0.0, 1.0))


def _period_metric(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    logloss = float(
        -(
            labels * np.log(probabilities) + (1.0 - labels) * np.log1p(-probabilities)
        ).mean()
    )
    brier = float(np.square(probabilities - labels).mean())
    return logloss, brier


def _segment_metrics(
    records: list[dict[str, object]], segment: str, config: Kl8Pick5Config
) -> dict[str, object]:
    subset = [record for record in records if record["segment"] == segment]
    deltas_logloss = np.asarray(
        [
            _as_float(record["uniformLogLoss"]) - _as_float(record["logLoss"])
            for record in subset
        ]
    )
    deltas_brier = np.asarray(
        [
            _as_float(record["uniformBrier"]) - _as_float(record["brier"])
            for record in subset
        ]
    )
    primary_hits = np.asarray(
        [_as_int(record["primaryHits"]) for record in subset], dtype=np.int64
    )
    combination_hits = np.asarray(
        [cast(Sequence[int], record["combinationHits"]) for record in subset],
        dtype=np.int64,
    )
    if combination_hits.shape != (len(subset), config.output_combinations):
        raise ValueError("逐期组合命中必须覆盖完整5票投资组合")
    ticket_hits = combination_hits.reshape(-1)
    portfolio_total_hits = np.asarray(
        [_as_int(record["portfolioTotalHits"]) for record in subset], dtype=np.int64
    )
    portfolio_best_hits = np.asarray(
        [_as_int(record["portfolioBestHits"]) for record in subset], dtype=np.int64
    )
    blocks = np.array_split(np.arange(len(subset)), config.stability_blocks)
    block_metrics = [
        {
            "deltaLogLoss": float(deltas_logloss[indexes].mean()),
            "deltaBrier": float(deltas_brier[indexes].mean()),
            "meanHitsPerTicket": float(combination_hits[indexes].mean()),
            "meanPortfolioTotalHits": float(portfolio_total_hits[indexes].mean()),
        }
        for indexes in blocks
        if len(indexes)
    ]
    bootstrap_logloss = _block_bootstrap(
        deltas_logloss, config, seed_offset=0 if segment == "Search" else 1
    )
    bootstrap_brier = _block_bootstrap(
        deltas_brier, config, seed_offset=2 if segment == "Search" else 3
    )
    return {
        "segment": segment,
        "periods": len(subset),
        "meanLogLoss": float(
            np.mean([_as_float(record["logLoss"]) for record in subset])
        ),
        "meanBrier": float(np.mean([_as_float(record["brier"]) for record in subset])),
        "deltaLogLossVsUniform": float(deltas_logloss.mean()),
        "deltaBrierVsUniform": float(deltas_brier.mean()),
        "expectedPositiveDeviation": float(
            np.mean(
                [
                    abs(_as_float(record["expectedPositives"]) - 20.0)
                    for record in subset
                ]
            )
        ),
        "primaryHitFrequenciesAuditOnly": {
            str(value): int(np.count_nonzero(primary_hits == value))
            for value in range(6)
        },
        "meanPrimaryHitsAuditOnly": float(primary_hits.mean()),
        "ticketHitFrequencies": {
            str(value): int(np.count_nonzero(ticket_hits == value))
            for value in range(6)
        },
        "portfolioTotalHitFrequencies": {
            str(value): int(np.count_nonzero(portfolio_total_hits == value))
            for value in range(26)
        },
        "portfolioBestHitFrequencies": {
            str(value): int(np.count_nonzero(portfolio_best_hits == value))
            for value in range(6)
        },
        "meanHitsPerTicket": float(ticket_hits.mean()),
        "meanPortfolioTotalHits": float(portfolio_total_hits.mean()),
        "matchedCostBaselineMeanPortfolioTotalHits": 6.25,
        "portfolioBestHitAtLeast3Rate": float(np.mean(portfolio_best_hits >= 3)),
        "portfolioBestHitAtLeast4Rate": float(np.mean(portfolio_best_hits >= 4)),
        "portfolioBestHitExactly5Rate": float(np.mean(portfolio_best_hits == 5)),
        "exactPortfolioTotalHitsPValue": _exact_portfolio_total_hits_pvalue(
            int(portfolio_total_hits.sum()),
            [
                cast(Sequence[Sequence[int]], record["researchCombinations"])
                for record in subset
            ],
        ),
        "singleTicketHypergeometricBaselineAuditOnly": {
            str(value): float(hypergeom.pmf(value, 80, 20, 5)) for value in range(6)
        },
        "blockBootstrap": {
            "deltaLogLoss": bootstrap_logloss,
            "deltaBrier": bootstrap_brier,
        },
        "blockStability": block_metrics,
    }


def _circular_block_bootstrap_means(
    values: np.ndarray, *, resamples: int, seed: int
) -> np.ndarray:
    """按旧RNG顺序向量化生成连续圆形块bootstrap均值。"""

    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or not len(sample) or not np.isfinite(sample).all():
        raise ValueError("bootstrap样本必须是一维非空有限数组")
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("bootstrap重采样次数必须为正整数")
    block_length = max(1, int(math.sqrt(len(sample))))
    blocks_per_resample = math.ceil(len(sample) / block_length)
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, len(sample), size=(resamples, blocks_per_resample))
    offsets = np.arange(block_length, dtype=np.int64)
    indexes = (starts[:, :, None] + offsets[None, None, :]) % len(sample)
    flattened = indexes.reshape(resamples, -1)[:, : len(sample)]
    return sample[flattened].mean(axis=1).astype(np.float64)


def _block_bootstrap(
    values: np.ndarray, config: Kl8Pick5Config, *, seed_offset: int
) -> dict[str, float | int]:
    """固定种子连续圆形块自助法；仅用于预注册单侧稳定性审计。"""

    sample = np.asarray(values, dtype=np.float64)
    block_length = max(1, int(math.sqrt(len(sample))))
    means = _circular_block_bootstrap_means(
        sample,
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed + seed_offset,
    )
    return {
        "resamples": config.bootstrap_resamples,
        "blockLength": block_length,
        "lowerOneSided95": float(np.quantile(means, config.alpha)),
        "pValueMeanNonPositive": float(
            (1 + np.count_nonzero(means <= 0.0)) / (config.bootstrap_resamples + 1)
        ),
    }


def _segment_gate(
    metrics: Mapping[str, object], config: Kl8Pick5Config
) -> dict[str, object]:
    blocks = cast(list[dict[str, object]], metrics["blockStability"])
    bootstrap = _as_mapping(metrics["blockBootstrap"], "blockBootstrap")
    bootstrap_logloss = _as_mapping(
        bootstrap["deltaLogLoss"], "blockBootstrap.deltaLogLoss"
    )
    bootstrap_brier = _as_mapping(bootstrap["deltaBrier"], "blockBootstrap.deltaBrier")
    marginal = (
        _as_float(metrics["deltaLogLossVsUniform"]) >= 0.0
        and _as_float(metrics["deltaBrierVsUniform"]) >= 0.0
        and _as_float(metrics["expectedPositiveDeviation"]) <= 1e-9
        and all(
            _as_float(block["deltaLogLoss"]) >= 0.0
            and _as_float(block["deltaBrier"]) >= 0.0
            for block in blocks
        )
        and _as_float(bootstrap_logloss["pValueMeanNonPositive"]) <= config.alpha
        and _as_float(bootstrap_brier["pValueMeanNonPositive"]) <= config.alpha
    )
    business = (
        _as_float(metrics["exactPortfolioTotalHitsPValue"]) <= config.alpha
        and _as_float(metrics["meanHitsPerTicket"])
        >= config.minimum_mean_hits_per_ticket
        and _as_float(metrics["meanPortfolioTotalHits"])
        >= config.minimum_mean_hits_per_ticket * config.output_combinations
        and all(
            _as_float(block["meanHitsPerTicket"]) >= config.minimum_mean_hits_per_ticket
            for block in blocks
        )
    )
    reasons = []
    if not marginal:
        reasons.append("边际概率质量或五块稳定性未超过Uniform")
    if not business:
        reasons.append("完整5票组合均值或五块每票均值低于匹配成本随机基线")
    return {
        "marginalGatePassed": marginal,
        "businessGatePassed": business,
        "passed": marginal and business,
        "reasons": reasons,
    }


@dataclass(frozen=True)
class Kl8DevelopmentReport:
    config: Kl8Pick5Config
    data_sha256: str
    source_fingerprint: str
    frozen_periods_excluded: int
    selected_temperature: float
    search: dict[str, object]
    calibration: dict[str, object]
    evaluation: dict[str, object]
    final_expert_weights: dict[str, float]
    periods: list[dict[str, object]]
    research_candidates: list[list[int]]
    audit_research_candidates: bool
    protocol_identity: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        search_gate = _as_mapping(self.search["gate"], "search.gate")
        evaluation_gate = _as_mapping(self.evaluation["gate"], "evaluation.gate")
        payload: dict[str, object] = {
            "schemaVersion": "kl8_pick5_development_v1",
            "evaluationKind": "development_prequential_challenger",
            "evidenceStatus": "exploratory_reused_development",
            "frozenRead": False,
            "frozenPeriodsExcluded": self.frozen_periods_excluded,
            "validationOpened": False,
            "promotionPassed": False,
            "recommendationEnabled": False,
            "formalRecommendation": None,
            "developmentSignalsPassed": bool(
                search_gate["passed"] and evaluation_gate["passed"]
            ),
            "userVisibleCandidates": [],
            "researchCandidates": (
                self.research_candidates if self.audit_research_candidates else []
            ),
            "researchCandidateNotice": (
                RESEARCH_ONLY_WORDING if self.audit_research_candidates else None
            ),
            "config": _json_compatible(asdict(self.config)),
            "dataSha256": self.data_sha256,
            "sourceFingerprint": self.source_fingerprint,
            "protocolIdentity": self.protocol_identity,
            "developmentProtocolRegistered": self.protocol_identity is not None,
            "selectedTemperature": self.selected_temperature,
            "search": self.search,
            "calibration": self.calibration,
            "evaluation": self.evaluation,
            "finalExpertWeights": self.final_expert_weights,
            "periods": self.periods,
        }
        payload["reportSha256"] = payload_sha256(payload)
        return payload


def _source_paths() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2]
    return (
        Path(__file__),
        Path(__file__).with_name("kl8_pick5_null.py"),
        root / "scripts" / "kl8_fetch_history.py",
        root / "scripts" / "kl8_pick5_development.py",
        root / "scripts" / "kl8_pick5_null.py",
        root / "scripts" / "kl8_pick5_predict_today.py",
    )


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    for path in _source_paths():
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return digest.hexdigest()


def _run_records(
    history: pd.DataFrame, config: Kl8Pick5Config
) -> tuple[list[dict[str, object]], _OnlineState, float, list[dict[str, float]]]:
    state = _OnlineState.initial()
    records: list[dict[str, object]] = []
    uniform = np.full(80, 0.25, dtype=np.float64)
    selected_temperature = 1.0
    calibration_scores: list[dict[str, float]] = []
    evaluation_start = (
        config.warmup_periods + config.search_periods + config.calibration_periods
    )
    for index, row in enumerate(history.itertuples(index=False)):
        experts = _expert_probabilities(state, config)
        weights_before = _hedge_weights(state, config)
        mixed = normalize_sum20(weights_before @ experts, epsilon=config.epsilon)
        if index == evaluation_start:
            selected_temperature, calibration_scores = _select_temperature(
                records, history, config
            )
        if index >= evaluation_start:
            mixed = _temperature_scale(mixed, selected_temperature, config.epsilon)
        combinations = generate_top5_combinations(mixed, state.pair, config)
        drawn_numbers = set(row.numbers)
        combination_hits = [
            len(set(combination) & drawn_numbers) for combination in combinations
        ]
        labels = _labels(row.numbers)
        losses = _bernoulli_losses(experts, labels, config)
        logloss, brier = _period_metric(mixed, labels)
        uniform_logloss, uniform_brier = _period_metric(uniform, labels)
        _update_state(state, labels, losses, config)
        weights_after = _hedge_weights(state, config)
        if index < config.warmup_periods:
            segment = "Warmup"
        elif index < config.warmup_periods + config.search_periods:
            segment = "Search"
        elif (
            index
            < config.warmup_periods + config.search_periods + config.calibration_periods
        ):
            segment = "Calibration"
        else:
            segment = "Evaluation"
        records.append(
            {
                "index": index,
                "issue": row.issue,
                "segment": segment,
                "mixedProbabilities": mixed.tolist(),
                "expertWeightsBefore": dict(zip(EXPERT_NAMES, weights_before.tolist())),
                "expertWeightsAfter": dict(zip(EXPERT_NAMES, weights_after.tolist())),
                "expertLosses": dict(zip(EXPERT_NAMES, losses.tolist())),
                "primaryCombination": combinations[0],
                "researchCombinations": combinations,
                "primaryHits": combination_hits[0],
                "combinationHits": combination_hits,
                "portfolioTotalHits": sum(combination_hits),
                "portfolioBestHits": max(combination_hits),
                "logLoss": logloss,
                "brier": brier,
                "uniformLogLoss": uniform_logloss,
                "uniformBrier": uniform_brier,
                "expectedPositives": float(mixed.sum()),
            }
        )
    return records, state, selected_temperature, calibration_scores


def _select_temperature(
    records: list[dict[str, object]], history: pd.DataFrame, config: Kl8Pick5Config
) -> tuple[float, list[dict[str, float]]]:
    calibration_records = [
        record for record in records if record["segment"] == "Calibration"
    ]
    scores = []
    for temperature in config.calibration_temperatures:
        losses = []
        for record in calibration_records:
            labels = _labels(history.iloc[_as_int(record["index"])]["numbers"])
            calibrated = _temperature_scale(
                np.asarray(
                    cast(Sequence[float], record["mixedProbabilities"]),
                    dtype=np.float64,
                ),
                temperature,
                config.epsilon,
            )
            losses.append(_period_metric(calibrated, labels)[0])
        scores.append(
            {"temperature": float(temperature), "meanLogLoss": float(np.mean(losses))}
        )
    selected = min(scores, key=lambda item: (item["meanLogLoss"], item["temperature"]))[
        "temperature"
    ]
    return selected, scores


def run_kl8_development(
    history: pd.DataFrame,
    config: Kl8Pick5Config | None = None,
    *,
    frozen_periods_excluded: int = 500,
    audit_research_candidates: bool = False,
) -> Kl8DevelopmentReport:
    """运行未登记的严格先预测后更新开发流程。"""

    return _run_kl8_development(
        history,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
        audit_research_candidates=audit_research_candidates,
        protocol_identity=None,
    )


def _run_kl8_development(
    history: pd.DataFrame,
    config: Kl8Pick5Config | None,
    *,
    frozen_periods_excluded: int,
    audit_research_candidates: bool,
    protocol_identity: dict[str, object] | None,
) -> Kl8DevelopmentReport:
    """内部运行器；协议身份只允许已验证登记路径注入。"""

    active = config or Kl8Pick5Config()
    if frozen_periods_excluded != active.frozen_periods:
        raise ValueError("frozen_periods_excluded必须等于配置的500期")
    normalized = normalize_kl8_dataframe(history)
    if len(normalized) != active.required_periods:
        raise ValueError(
            f"快乐8开发历史必须恰好{active.required_periods}期，收到{len(normalized)}期"
        )
    records, state, selected_temperature, calibration_scores = _run_records(
        normalized, active
    )
    search_metrics = _segment_metrics(records, "Search", active)
    evaluation_metrics = _segment_metrics(records, "Evaluation", active)
    search: dict[str, object] = {
        "metrics": search_metrics,
        "gate": _segment_gate(search_metrics, active),
    }
    evaluation: dict[str, object] = {
        "metrics": evaluation_metrics,
        "gate": _segment_gate(evaluation_metrics, active),
    }
    final_weights = dict(zip(EXPERT_NAMES, _hedge_weights(state, active).tolist()))
    final_experts = _expert_probabilities(state, active)
    next_probabilities = normalize_sum20(
        np.asarray(list(final_weights.values())) @ final_experts,
        epsilon=active.epsilon,
    )
    next_probabilities = _temperature_scale(
        next_probabilities, selected_temperature, active.epsilon
    )
    final_candidates = (
        generate_top5_combinations(next_probabilities, state.pair, active)
        if audit_research_candidates
        else []
    )
    report_records: list[dict[str, object]] = []
    for record in records:
        exported = dict(record)
        if not audit_research_candidates:
            exported.pop("researchCombinations")
        report_records.append(exported)
    return Kl8DevelopmentReport(
        config=active,
        data_sha256=canonical_kl8_sha256(normalized),
        source_fingerprint=source_fingerprint(),
        frozen_periods_excluded=frozen_periods_excluded,
        selected_temperature=selected_temperature,
        search=search,
        calibration={
            "periods": active.calibration_periods,
            "selectionOnly": "fixed_logit_temperature_grid",
            "scores": calibration_scores,
        },
        evaluation=evaluation,
        final_expert_weights=final_weights,
        periods=report_records,
        research_candidates=final_candidates,
        audit_research_candidates=audit_research_candidates,
        protocol_identity=protocol_identity,
    )


def build_kl8_protocol(
    history: pd.DataFrame,
    config: Kl8Pick5Config,
    *,
    frozen_periods_excluded: int,
    frozen_boundary: Mapping[str, object],
) -> dict[str, object]:
    assert_canonical_formal_config(config)
    if frozen_periods_excluded != config.frozen_periods:
        raise ValueError("frozen_periods_excluded必须等于规范配置的500期")
    normalized = normalize_kl8_dataframe(history)
    payload: dict[str, object] = {
        "schemaVersion": "kl8_pick5_protocol_v1",
        "lottery": "kl8",
        "config": _json_compatible(asdict(config)),
        "developmentData": {
            "periods": len(normalized),
            "sha256": canonical_kl8_sha256(normalized),
            "firstIssue": normalized.iloc[0]["issue"],
            "lastIssue": normalized.iloc[-1]["issue"],
        },
        "frozenBoundary": _normalize_frozen_boundary(
            frozen_boundary, frozen_periods_excluded=frozen_periods_excluded
        ),
        "sourceFingerprint": source_fingerprint(),
    }
    payload["protocolSha256"] = payload_sha256(payload)
    return payload


def _write_immutable_json(payload: object, destination: str | Path, label: str) -> Path:
    path = Path(destination)
    content = (
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            if stat.S_IMODE(path.stat().st_mode) & 0o222:
                raise ValueError(f"{label}已存在但不是只读文件：{path}")
            return path
        raise FileExistsError(f"{label}已存在，禁止覆盖：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _load_readonly_json(path: str | Path, label: str) -> dict[str, object]:
    source = Path(path)
    if stat.S_IMODE(source.stat().st_mode) & 0o222:
        raise ValueError(f"{label}必须为只读路径")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}必须是JSON对象")
    return payload


def write_kl8_protocol(protocol: Mapping[str, object], path: str | Path) -> Path:
    """不可覆盖写入开发协议。"""

    return _write_immutable_json(dict(protocol), path, "快乐8开发协议")


def write_kl8_report(report: Kl8DevelopmentReport, path: str | Path) -> Path:
    """不可覆盖写入自哈希开发报告。"""

    return _write_immutable_json(report.to_dict(), path, "快乐8开发报告")


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> None:
    expected = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    if expected != payload_sha256(unsigned):
        raise ValueError(f"{label}自哈希校验失败")


def load_and_verify_kl8_protocol(
    path: str | Path,
    history: pd.DataFrame,
    config: Kl8Pick5Config,
    *,
    frozen_periods_excluded: int,
    frozen_boundary: Mapping[str, object],
) -> dict[str, object]:
    assert_canonical_formal_config(config)
    if frozen_periods_excluded != config.frozen_periods:
        raise ValueError("frozen_periods_excluded必须等于规范配置的500期")
    payload = _load_readonly_json(path, "快乐8开发协议")
    _verify_self_hash(payload, "protocolSha256", "快乐8开发协议")
    expected = build_kl8_protocol(
        history,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
        frozen_boundary=frozen_boundary,
    )
    if payload != expected:
        raise ValueError("快乐8开发协议与当前源码/配置/数据确定性重算不一致")
    return payload


def run_registered_kl8_development(
    protocol_path: str | Path,
    history: pd.DataFrame,
    config: Kl8Pick5Config,
    *,
    frozen_periods_excluded: int,
    frozen_boundary: Mapping[str, object],
    audit_research_candidates: bool = False,
) -> Kl8DevelopmentReport:
    """从只读协议路径重建预期协议后运行已登记开发。"""

    assert_canonical_formal_config(config)
    if frozen_periods_excluded != config.frozen_periods:
        raise ValueError("frozen_periods_excluded必须等于规范配置的500期")
    protocol = load_and_verify_kl8_protocol(
        protocol_path,
        history,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
        frozen_boundary=frozen_boundary,
    )
    identity = {
        "protocolSha256": protocol["protocolSha256"],
        "path": str(Path(protocol_path).resolve()),
    }
    return _run_kl8_development(
        history,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
        audit_research_candidates=audit_research_candidates,
        protocol_identity=identity,
    )


def load_and_verify_kl8_report(
    report_path: str | Path,
    protocol_path: str | Path,
    history: pd.DataFrame,
    config: Kl8Pick5Config,
    *,
    frozen_periods_excluded: int,
    frozen_boundary: Mapping[str, object],
) -> dict[str, object]:
    """校验只读报告并完整确定性重算，拒绝“篡改后重哈希”。"""

    assert_canonical_formal_config(config)
    if frozen_periods_excluded != config.frozen_periods:
        raise ValueError("frozen_periods_excluded必须等于规范配置的500期")
    payload = _load_readonly_json(report_path, "快乐8开发报告")
    _verify_self_hash(payload, "reportSha256", "快乐8开发报告")
    protocol = load_and_verify_kl8_protocol(
        protocol_path,
        history,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
        frozen_boundary=frozen_boundary,
    )
    expected = _run_kl8_development(
        history,
        config,
        frozen_periods_excluded=frozen_periods_excluded,
        audit_research_candidates=bool(payload.get("researchCandidates")),
        protocol_identity={
            "protocolSha256": protocol["protocolSha256"],
            "path": str(Path(protocol_path).resolve()),
        },
    ).to_dict()
    if payload != expected:
        raise ValueError("快乐8开发报告与当前源码/配置/数据确定性重算不一致")
    return payload


__all__ = [
    "EXPERT_NAMES",
    "Kl8DevelopmentReport",
    "Kl8Pick5Config",
    "assert_canonical_formal_config",
    "build_kl8_protocol",
    "canonical_kl8_sha256",
    "generate_top5_combinations",
    "load_and_verify_kl8_protocol",
    "load_and_verify_kl8_report",
    "load_kl8_development_csv",
    "normalize_kl8_dataframe",
    "normalize_sum20",
    "payload_sha256",
    "run_kl8_development",
    "run_registered_kl8_development",
    "source_fingerprint",
    "write_kl8_protocol",
    "write_kl8_report",
]

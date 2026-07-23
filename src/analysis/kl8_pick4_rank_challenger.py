# -*- coding: utf-8 -*-
"""快乐8Pick4固定低复杂度排名挑战器。"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from src.analysis.kl8_feature_discovery_v2 import (
    build_prior_only_number_panel,
    write_kl8_feature_discovery_report,
)
from src.analysis.kl8_pick5_probability_v1 import (
    canonical_kl8_sha256,
    normalize_kl8_dataframe,
    normalize_sum20,
)

PICK4_RANK_FEATURES = (
    "frequency80",
    "frequency320",
    "frequency80Minus320",
    "omissionLog",
    "ewma80",
    "inPrevious",
)


@dataclass(frozen=True)
class Kl8Pick4RankConfig:
    """一次性Pick4排名挑战的固定低复杂度配置。"""

    initial_train: int = 300
    evaluation_periods: int = 1214
    refit_interval: int = 50
    stability_blocks: int = 5
    seed: int = 20260723
    n_estimators: int = 40
    learning_rate: float = 0.04
    num_leaves: int = 7
    max_depth: int = 3
    min_child_samples: int = 100
    reg_alpha: float = 0.2
    reg_lambda: float = 1.0
    lambdarank_truncation_level: int = 4
    label_gain: tuple[int, int] = (0, 1)
    probability_score_scale: float = 0.1
    probability_shrinkage: float = 0.1
    epsilon: float = 1e-6
    bootstrap_resamples: int = 2000
    bootstrap_block_length: int = 12
    n_jobs: int = 1

    def __post_init__(self) -> None:
        integers = {
            "initial_train": self.initial_train,
            "evaluation_periods": self.evaluation_periods,
            "refit_interval": self.refit_interval,
            "stability_blocks": self.stability_blocks,
            "seed": self.seed,
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "lambdarank_truncation_level": self.lambdarank_truncation_level,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_block_length": self.bootstrap_block_length,
            "n_jobs": self.n_jobs,
        }
        if any(type(value) is not int for value in integers.values()):
            raise ValueError("Pick4排名配置整数参数必须为整数")
        if self.initial_train <= 0 or self.evaluation_periods <= 0:
            raise ValueError("initial_train和evaluation_periods必须为正数")
        if self.refit_interval <= 0 or self.stability_blocks != 5:
            raise ValueError("refit_interval必须为正且stability_blocks固定为5")
        if self.n_estimators <= 0 or self.num_leaves < 2:
            raise ValueError("LightGBM模型规模无效")
        if self.max_depth == 0 or self.max_depth < -1:
            raise ValueError("max_depth必须为-1或正整数")
        if self.min_child_samples <= 0 or self.n_jobs == 0:
            raise ValueError("min_child_samples或n_jobs无效")
        if self.lambdarank_truncation_level != 4 or self.label_gain != (0, 1):
            raise ValueError("LambdaRank必须固定Top4截断和二元label_gain")
        if self.bootstrap_resamples < 100 or self.bootstrap_block_length <= 0:
            raise ValueError("bootstrap配置无效")
        floats = (
            self.learning_rate,
            self.reg_alpha,
            self.reg_lambda,
            self.probability_score_scale,
            self.probability_shrinkage,
            self.epsilon,
        )
        if not all(math.isfinite(value) for value in floats):
            raise ValueError("Pick4排名配置浮点参数必须有限")
        if self.learning_rate <= 0.0 or self.probability_score_scale <= 0.0:
            raise ValueError("learning_rate和probability_score_scale必须为正数")
        if not 0.0 <= self.probability_shrinkage <= 1.0:
            raise ValueError("probability_shrinkage必须位于0..1")
        if self.reg_alpha < 0.0 or self.reg_lambda < 0.0:
            raise ValueError("正则化参数不得为负数")
        if not 0.0 < self.epsilon < 0.25:
            raise ValueError("epsilon范围无效")

    @property
    def required_periods(self) -> int:
        return self.initial_train + self.evaluation_periods

    def validate_history_length(self, periods: int) -> None:
        if periods != self.required_periods:
            raise ValueError(
                f"Pick4排名挑战要求恰好{self.required_periods}期开发历史，实际{periods}期"
            )


def ranked_pick4_portfolio(scores: np.ndarray) -> list[list[int]]:
    """将Top20按排名层轮转为5张互不重叠的Pick4票。"""

    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (80,) or not np.isfinite(values).all():
        raise ValueError("Pick4排名分数必须为80维有限向量")
    ranking = np.lexsort((np.arange(80), -values))[:20] + 1
    tickets = [
        sorted(int(value) for value in ranking[offset::5]) for offset in range(5)
    ]
    tickets.sort()
    if len({number for ticket in tickets for number in ticket}) != 20:
        raise RuntimeError("Pick4组合包必须覆盖20个互不重复号码")
    return tickets


def build_pick4_rank_panel(development: pd.DataFrame) -> pd.DataFrame:
    """构建Pick4专用prior-only面板及唯一派生趋势列。"""

    panel = build_prior_only_number_panel(development)
    panel["frequency80Minus320"] = panel["frequency80"] - panel["frequency320"]
    return panel


def _validate_query_panel(panel: pd.DataFrame, periods: int) -> None:
    sizes = panel.groupby("periodIndex", sort=True).size()
    positives = panel.groupby("periodIndex", sort=True)["target"].sum()
    if (
        list(sizes.index) != list(range(periods))
        or not sizes.eq(80).all()
        or not positives.eq(20.0).all()
    ):
        raise ValueError("LambdaRank面板必须逐期连续且每组80行/20个正例")


def build_pick4_ranker(config: Kl8Pick4RankConfig) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        lambdarank_truncation_level=config.lambdarank_truncation_level,
        label_gain=list(config.label_gain),
        boosting_type="gbdt",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=config.seed,
        bagging_seed=config.seed,
        feature_fraction_seed=config.seed,
        data_random_seed=config.seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )


def audit_rank_probabilities(
    scores: np.ndarray, config: Kl8Pick4RankConfig
) -> np.ndarray:
    uniform = np.full(80, 0.25, dtype=np.float64)
    if config.probability_shrinkage == 0.0:
        return uniform
    centered = scores - float(np.mean(scores))
    standard_deviation = float(np.std(centered))
    if standard_deviation <= np.finfo(np.float64).eps:
        return uniform
    z_scores = centered / standard_deviation
    logits = np.clip(config.probability_score_scale * z_scores, -40.0, 40.0)
    raw = 1.0 / (1.0 + np.exp(-logits))
    ranked = normalize_sum20(raw, epsilon=config.epsilon)
    shrinkage = config.probability_shrinkage
    return shrinkage * ranked + (1.0 - shrinkage) * uniform


def _proper_scores(
    probabilities: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    log_loss = -float(
        np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))
    )
    brier = float(np.mean(np.square(clipped - target)))
    return log_loss, brier


def _sum_hypergeom_right_tail(
    *, observed_total: int, periods: int, selected_count: int
) -> float:
    one_period = hypergeom.pmf(np.arange(selected_count + 1), 80, 20, selected_count)
    distribution = np.polynomial.polynomial.polypow(one_period, periods)
    return float(distribution[observed_total:].sum())


def _block_bootstrap_nonpositive_pvalue(
    values: np.ndarray, config: Kl8Pick4RankConfig, *, seed_offset: int
) -> float:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or len(sample) == 0 or not np.isfinite(sample).all():
        raise ValueError("bootstrap样本必须为非空有限一维数组")
    block_length = min(config.bootstrap_block_length, len(sample))
    blocks = math.ceil(len(sample) / block_length)
    generator = np.random.default_rng(config.seed + seed_offset)
    starts = generator.integers(
        0, len(sample), size=(config.bootstrap_resamples, blocks)
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indexes = (starts[:, :, None] + offsets[None, None, :]) % len(sample)
    means = sample[
        indexes.reshape(config.bootstrap_resamples, -1)[:, : len(sample)]
    ].mean(axis=1)
    return float((np.count_nonzero(means <= 0.0) + 1) / (len(means) + 1))


def _holm_adjusted(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def _finite_number(record: Mapping[str, object], key: str) -> float:
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{key}必须为数值")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{key}必须为有限数")
    return output


def _integer_value(record: Mapping[str, object], key: str) -> int:
    value = record[key]
    if type(value) is not int:
        raise ValueError(f"{key}必须为整数")
    return int(value)


def _ticket_hit_vector(record: Mapping[str, object]) -> list[int]:
    value = record["ticketHits"]
    if (
        not isinstance(value, list)
        or len(value) != 5
        or any(type(item) is not int or not 0 <= item <= 4 for item in value)
    ):
        raise ValueError("ticketHits必须为5个0..4整数")
    return [int(item) for item in value]


def _summarize(records: Sequence[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("Pick4评估记录不得为空")
    periods = len(records)
    primary = np.asarray(
        [_integer_value(record, "primaryHits") for record in records], dtype=np.int64
    )
    total = np.asarray(
        [_integer_value(record, "portfolioTotalHits") for record in records],
        dtype=np.int64,
    )
    ticket_hits = np.asarray(
        [_ticket_hit_vector(record) for record in records], dtype=np.int64
    )
    log_loss = float(np.mean([_finite_number(record, "logLoss") for record in records]))
    brier = float(np.mean([_finite_number(record, "brier") for record in records]))
    uniform_log_loss = float(
        np.mean([_finite_number(record, "uniformLogLoss") for record in records])
    )
    uniform_brier = float(
        np.mean([_finite_number(record, "uniformBrier") for record in records])
    )
    return {
        "periods": periods,
        "primaryTotalHits": int(primary.sum()),
        "primaryMeanHits": float(primary.mean()),
        "primaryAtLeast1Rate": float(np.mean(primary >= 1)),
        "primaryAtLeast2Rate": float(np.mean(primary >= 2)),
        "primaryAtLeast3Rate": float(np.mean(primary >= 3)),
        "primaryExact4Rate": float(np.mean(primary == 4)),
        "primaryExactRandomRightTailPValue": _sum_hypergeom_right_tail(
            observed_total=int(primary.sum()), periods=periods, selected_count=4
        ),
        "portfolioTotalHits": int(total.sum()),
        "portfolioMeanTotalHits": float(total.mean()),
        "meanHitsPerTicket": float(ticket_hits.mean()),
        "bestTicketAtLeast2Rate": float(np.mean(ticket_hits.max(axis=1) >= 2)),
        "bestTicketAtLeast3Rate": float(np.mean(ticket_hits.max(axis=1) >= 3)),
        "bestTicketExact4Rate": float(np.mean(ticket_hits.max(axis=1) == 4)),
        "portfolioExactRandomRightTailPValue": _sum_hypergeom_right_tail(
            observed_total=int(total.sum()), periods=periods, selected_count=20
        ),
        "logLoss": log_loss,
        "brier": brier,
        "uniformLogLoss": uniform_log_loss,
        "uniformBrier": uniform_brier,
        "deltaLogLoss": uniform_log_loss - log_loss,
        "deltaBrier": uniform_brier - brier,
    }


def _development_gate(
    summary: Mapping[str, object], blocks: Sequence[Mapping[str, object]]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    holm_value = summary.get("holmAdjustedPValues")
    holm_passed = bool(
        isinstance(holm_value, Mapping)
        and set(holm_value)
        == {
            "primaryTop4",
            "portfolioTop20",
            "deltaLogLoss",
            "deltaBrier",
        }
        and all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) <= 0.05
            for value in holm_value.values()
        )
    )
    checks = {
        "holm_joint_gate_failed": holm_passed,
        "primary_mean_not_above_1": _finite_number(summary, "primaryMeanHits") > 1.0,
        "primary_random_p_above_0_05": _finite_number(
            summary, "primaryExactRandomRightTailPValue"
        )
        <= 0.05,
        "per_ticket_mean_not_above_1": _finite_number(summary, "meanHitsPerTicket")
        > 1.0,
        "portfolio_total_mean_not_above_5": _finite_number(
            summary, "portfolioMeanTotalHits"
        )
        > 5.0,
        "portfolio_random_p_above_0_05": _finite_number(
            summary, "portfolioExactRandomRightTailPValue"
        )
        <= 0.05,
        "logloss_not_improved": _finite_number(summary, "deltaLogLoss") > 0.0,
        "brier_not_improved": _finite_number(summary, "deltaBrier") > 0.0,
    }
    for reason, passed in checks.items():
        if not passed:
            reasons.append(reason)
    for index, block in enumerate(blocks, start=1):
        if _finite_number(block, "primaryMeanHits") < 1.0:
            reasons.append(f"block_{index}_primary_below_random")
        if _finite_number(block, "meanHitsPerTicket") < 1.0:
            reasons.append(f"block_{index}_per_ticket_below_random")
        if _finite_number(block, "deltaLogLoss") < 0.0:
            reasons.append(f"block_{index}_logloss_worse")
        if _finite_number(block, "deltaBrier") < 0.0:
            reasons.append(f"block_{index}_brier_worse")
    return not reasons, reasons


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__),
        root / "scripts" / "kl8_pick4_rank_challenger.py",
        root / "src" / "analysis" / "kl8_feature_discovery_v2.py",
        root / "src" / "analysis" / "kl8_pick5_probability_v1.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def run_pick4_rank_challenger(
    development: pd.DataFrame,
    config: Kl8Pick4RankConfig,
    *,
    frozen_periods_excluded: int,
    frozen_boundary: Mapping[str, object],
) -> dict[str, object]:
    """在开发区运行固定Pick4排名挑战；无论结果如何均不晋级。"""

    started = time.perf_counter()
    normalized = normalize_kl8_dataframe(development)
    config.validate_history_length(len(normalized))
    if frozen_periods_excluded <= 0:
        raise ValueError("Pick4挑战必须显式排除Frozen")
    panel = build_pick4_rank_panel(normalized)
    _validate_query_panel(panel, len(normalized))
    records: list[dict[str, object]] = []
    gain_totals = {feature: 0.0 for feature in PICK4_RANK_FEATURES}
    ranker: lgb.LGBMRanker | None = None
    train_count = 0
    last_training_query_count = 0
    uniform = np.full(80, 0.25, dtype=np.float64)
    start = config.initial_train
    end = config.required_periods
    for period_index in range(start, end):
        if ranker is None or (period_index - start) % config.refit_interval == 0:
            train_rows = panel["periodIndex"] < period_index
            last_training_query_count = period_index
            ranker = build_pick4_ranker(config)
            ranker.fit(
                panel.loc[train_rows, list(PICK4_RANK_FEATURES)],
                panel.loc[train_rows, "target"].astype(np.int32),
                group=[80] * period_index,
                eval_at=(4,),
            )
            gains = ranker.booster_.feature_importance(importance_type="gain")
            for feature, gain in zip(PICK4_RANK_FEATURES, gains, strict=True):
                gain_totals[feature] += float(gain)
            train_count += 1
        target_rows = panel["periodIndex"] == period_index
        target = panel.loc[target_rows, "target"].to_numpy(dtype=np.float64)
        if ranker is None:
            raise RuntimeError("Pick4排名器未拟合")
        scores = np.asarray(
            ranker.predict(panel.loc[target_rows, list(PICK4_RANK_FEATURES)]),
            dtype=np.float64,
        )
        ranking = np.lexsort((np.arange(80), -scores))
        primary = ranking[:4]
        tickets = ranked_pick4_portfolio(scores)
        ticket_indexes = np.asarray(tickets, dtype=np.int64) - 1
        ticket_hits = target[ticket_indexes].sum(axis=1).astype(np.int64)
        probabilities = audit_rank_probabilities(scores, config)
        log_loss, brier = _proper_scores(probabilities, target)
        uniform_log_loss, uniform_brier = _proper_scores(uniform, target)
        records.append(
            {
                "primaryHits": int(target[primary].sum()),
                "ticketHits": [int(value) for value in ticket_hits],
                "portfolioTotalHits": int(ticket_hits.sum()),
                "logLoss": log_loss,
                "brier": brier,
                "uniformLogLoss": uniform_log_loss,
                "uniformBrier": uniform_brier,
            }
        )
    summary = _summarize(records)
    log_loss_deltas = np.asarray(
        [
            _finite_number(record, "uniformLogLoss") - _finite_number(record, "logLoss")
            for record in records
        ],
        dtype=np.float64,
    )
    brier_deltas = np.asarray(
        [
            _finite_number(record, "uniformBrier") - _finite_number(record, "brier")
            for record in records
        ],
        dtype=np.float64,
    )
    log_loss_p = _block_bootstrap_nonpositive_pvalue(
        log_loss_deltas, config, seed_offset=101
    )
    brier_p = _block_bootstrap_nonpositive_pvalue(brier_deltas, config, seed_offset=102)
    raw_p_values = {
        "primaryTop4": _finite_number(summary, "primaryExactRandomRightTailPValue"),
        "portfolioTop20": _finite_number(
            summary, "portfolioExactRandomRightTailPValue"
        ),
        "deltaLogLoss": log_loss_p,
        "deltaBrier": brier_p,
    }
    summary.update(
        {
            "deltaLogLossBootstrapPValue": log_loss_p,
            "deltaBrierBootstrapPValue": brier_p,
            "holmAdjustedPValues": _holm_adjusted(raw_p_values),
        }
    )
    block_summaries = [
        {"block": index + 1, **_summarize([records[int(i)] for i in indexes])}
        for index, indexes in enumerate(
            np.array_split(np.arange(len(records)), config.stability_blocks)
        )
    ]
    gate_passed, gate_reasons = _development_gate(summary, block_summaries)
    if ranker is None:
        raise RuntimeError("Pick4排名器没有完成任何训练")
    trained_tree_count = int(ranker.booster_.num_trees())
    total_gain = sum(gain_totals.values())
    first = normalized.iloc[0]
    last = normalized.iloc[-1]
    report: dict[str, object] = {
        "schemaVersion": "kl8_pick4_rank_challenger_v2",
        "evidenceStatus": "exploratory_post_failure_reused_development",
        "frozenRead": False,
        "developmentGatePassed": gate_passed,
        "gateReasons": gate_reasons,
        "promotionPassed": False,
        "recommendationEnabled": False,
        "formalRecommendation": None,
        "userVisibleCandidates": [],
        "config": asdict(config),
        "features": list(PICK4_RANK_FEATURES),
        "strategy": "lambdarank_truncated_at_4_top20_round_robin_five_disjoint_pick4",
        "probabilityAudit": {
            "kind": "fixed_rank_score_mapping_risk_audit_not_calibration",
            "rankedFormula": "q=normalize_sum20(sigmoid(scoreScale*zscore(rankScore)))",
            "shrinkageFormula": "p=lambda*q+(1-lambda)*0.25",
            "lambda": config.probability_shrinkage,
            "scoreScale": config.probability_score_scale,
        },
        "baseline": {
            "meanHitsPerTicket": 1.0,
            "portfolioMeanTotalHits": 5.0,
            "primaryHitPmf": {
                str(k): float(hypergeom.pmf(k, 80, 20, 4)) for k in range(5)
            },
        },
        "evaluation": {
            **summary,
            "blocks": block_summaries,
            "trainCount": train_count,
            "lastTrainingQueryCount": last_training_query_count,
            "trainedTreeCount": trained_tree_count,
            "featureGainImportance": [
                {
                    "feature": feature,
                    "gain": gain_totals[feature],
                    "gainShare": (
                        gain_totals[feature] / total_gain if total_gain else 0.0
                    ),
                }
                for feature in sorted(
                    gain_totals, key=lambda item: (-gain_totals[item], item)
                )
            ],
        },
        "boundaries": {
            "development": {
                "periods": len(normalized),
                "firstIssue": str(first["issue"]),
                "lastIssue": str(last["issue"]),
                "firstDate": str(first["date"]),
                "lastDate": str(last["date"]),
            },
            "frozen": {
                "periodsExcluded": frozen_periods_excluded,
                "firstIssue": str(frozen_boundary["firstIssue"]),
                "lastIssue": str(frozen_boundary["lastIssue"]),
                "numbersRead": False,
            },
        },
        "dataSha256": canonical_kl8_sha256(normalized),
        "sourceFingerprint": _source_fingerprint(),
        "elapsedSeconds": time.perf_counter() - started,
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report


def write_pick4_rank_report(report: Mapping[str, object], path: str | Path) -> Path:
    """不可覆盖写入Pick4挑战报告。"""

    return write_kl8_feature_discovery_report(report, path)


__all__ = [
    "PICK4_RANK_FEATURES",
    "Kl8Pick4RankConfig",
    "audit_rank_probabilities",
    "build_pick4_rank_panel",
    "build_pick4_ranker",
    "ranked_pick4_portfolio",
    "run_pick4_rank_challenger",
    "write_pick4_rank_report",
]

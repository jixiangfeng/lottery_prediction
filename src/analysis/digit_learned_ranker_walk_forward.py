# -*- coding: utf-8 -*-
"""三位彩固定评分 v4 冻结参数逐期前推评估。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_evaluation import (
    CalibrationBin,
    evaluate_binary_calibration,
)
from src.analysis.digit_learned_features import (
    LearnedFeatureConfig,
    LearnedHistoryState,
    build_candidate_features,
    build_history_state,
    iter_rolling_history_states,
)
from src.analysis.digit_learned_ranker import (
    LearnedRankerParams,
    _atomic_write,
    _preflight_immutable,
    aggregate_group_candidates,
    params_fingerprint,
    payload_fingerprint,
    probabilities_from_scores,
    rank_candidate_indices,
    resolve_activation,
    score_candidates,
)
from src.analysis.digit_learned_ranker_search import LearnedSplit
from src.analysis.prediction_viability import poisson_binomial_right_tail
from src.lotteries.base import LotteryRule

DIRECT_CANDIDATE_BUDGETS = (10, 20, 50, 100, 250, 500, 700, 900, 990, 1000)


@dataclass(frozen=True)
class LearnedWalkForwardPeriod:
    target_index: int
    target_issue: str
    history_end_issue: str
    actual_text: str
    actual_rank: int
    actual_probability: float
    log_loss: float
    brier_score: float
    direct_hit: bool
    group_hit: bool
    group_random_probability: float
    group_rank: int = 221
    position_ranks: tuple[int, int, int] = (11, 11, 11)
    top_probability: float = 0.0
    top_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LearnedRankerReport:
    rule_code: str
    display_name: str
    params_fingerprint: str
    split: LearnedSplit
    test_target_indices: tuple[int, ...]
    periods: tuple[LearnedWalkForwardPeriod, ...]
    mean_log_loss: float
    uniform_log_loss: float
    mean_brier_score: float
    uniform_brier_score: float
    mean_rank: float
    mean_rank_percentile: float
    direct_hits: int
    direct_hit_rate: float
    direct_random_probability: float
    direct_p_value: float
    group_hits: int
    group_hit_rate: float
    group_random_hit_rate: float
    group_p_value: float
    block_mean_ranks: tuple[float, ...]
    stable_blocks: int
    gate_passed: bool
    gate_reasons: tuple[str, ...]
    expected_calibration_error: float = 0.0
    calibration: tuple[CalibrationBin, ...] = ()
    position_hits: int = 0
    position_hit_rate: float = 0.0
    position_random_probability: float = 0.0
    position_p_value: float = 1.0
    csv_sha256: str | None = None
    canonical_data_sha256: str | None = None
    source_fingerprint: str | None = None
    params_artifact_fingerprint: str | None = None
    test_segment_used_for_selection: bool = False

    def to_dict(self) -> dict[str, Any]:
        gate = build_gate_result(
            mean_rank=self.mean_rank,
            mean_log_loss=self.mean_log_loss,
            direct_p_value=self.direct_p_value,
            group_p_value=self.group_p_value,
            position_p_value=self.position_p_value,
            stable_blocks=self.stable_blocks,
        )
        payload = {
            "schemaVersion": 1,
            "evaluationKind": "frozen_test",
            "ruleCode": self.rule_code,
            "displayName": self.display_name,
            "paramsFingerprint": self.params_fingerprint,
            "paramsArtifactFingerprint": self.params_artifact_fingerprint,
            "csvSha256": self.csv_sha256,
            "canonicalDataSha256": self.canonical_data_sha256,
            "sourceFingerprint": self.source_fingerprint,
            "testSegmentUsedForSelection": self.test_segment_used_for_selection,
            "split": self.split.to_dict(),
            "testTargetIndices": list(self.test_target_indices),
            "periods": [item.to_dict() for item in self.periods],
            "metrics": {
                "meanLogLoss": self.mean_log_loss,
                "uniformLogLoss": self.uniform_log_loss,
                "meanBrierScore": self.mean_brier_score,
                "uniformBrierScore": self.uniform_brier_score,
                "meanRank": self.mean_rank,
                "meanRankPercentile": self.mean_rank_percentile,
                "directHits": self.direct_hits,
                "directHitRate": self.direct_hit_rate,
                "directRandomProbability": self.direct_random_probability,
                "directPValue": self.direct_p_value,
                "groupHits": self.group_hits,
                "groupHitRate": self.group_hit_rate,
                "groupRandomHitRate": self.group_random_hit_rate,
                "groupPValue": self.group_p_value,
                "positionHits": self.position_hits,
                "positionHitRate": self.position_hit_rate,
                "positionRandomProbability": self.position_random_probability,
                "positionPValue": self.position_p_value,
                "blockMeanRanks": list(self.block_mean_ranks),
                "stableBlocks": self.stable_blocks,
                "expectedCalibrationError": self.expected_calibration_error,
                "calibration": [item.to_dict() for item in self.calibration],
                "directCandidateBudgetCurve": build_candidate_budget_curve(
                    self.periods
                ),
                "groupBudgetCurve": build_group_budget_curve(self.periods),
                "positionPoolBudgetCurve": build_position_pool_budget_curve(
                    self.periods
                ),
            },
            "gate": gate,
            "gatePassed": self.gate_passed,
            "gateReasons": list(self.gate_reasons),
            "disclaimer": "开奖结果具有随机性；评估不构成预测有效、中奖或盈利承诺。",
        }
        payload["reportFingerprint"] = payload_fingerprint(payload)
        return payload


def _period(
    chronological: pd.DataFrame,
    index: int,
    rule: LotteryRule,
    params: LearnedRankerParams,
    feature_config: LearnedFeatureConfig,
    history_state: LearnedHistoryState | None = None,
) -> LearnedWalkForwardPeriod:
    target = chronological.iloc[index]
    target_issue = str(target["期数"])
    state = history_state or build_history_state(
        chronological.iloc[:index], rule, feature_config, target_issue=target_issue
    )
    features = build_candidate_features(state, rule)
    features = (
        features.assign(candidate=features["candidate"].astype(str))
        .sort_values("candidate", kind="mergesort")
        .reset_index(drop=True)
    )
    texts = features["candidate"].astype(str).tolist()
    scores = score_candidates(features, params)
    probabilities = probabilities_from_scores(
        scores,
        temperature=params.temperature,
        uniform_shrinkage=params.uniform_shrinkage,
    )
    order = rank_candidate_indices(probabilities, texts)
    actual_text = "".join(str(int(target[column])) for column in rule.number_columns)
    candidate_indices = {text: position for position, text in enumerate(texts)}
    try:
        actual_index = candidate_indices[actual_text]
    except KeyError as exc:
        raise ValueError(f"候选集合缺少真实号码：{actual_text}") from exc
    actual_rank = int(np.flatnonzero(order == actual_index)[0]) + 1
    actual_probability = float(probabilities[actual_index])
    brier = float(np.sum(probabilities**2) - 2 * actual_probability + 1.0)
    groups = aggregate_group_candidates(
        texts, probabilities, aggregation=params.group_aggregation
    )
    top_groups = groups[: params.group_top_k]
    actual_group = "".join(sorted(actual_text))
    group_rank = next(
        (
            position
            for position, item in enumerate(groups, start=1)
            if item.group_key == actual_group
        ),
        len(groups) + 1,
    )
    position_ranks = []
    for position in range(3):
        masses = {
            digit: math.fsum(
                float(probabilities[candidate_index])
                for candidate_index, text in enumerate(texts)
                if int(text[position]) == digit
            )
            for digit in range(10)
        }
        ordered_digits = sorted(masses, key=lambda digit: (-masses[digit], digit))
        position_ranks.append(ordered_digits.index(int(actual_text[position])) + 1)
    return LearnedWalkForwardPeriod(
        target_index=index,
        target_issue=target_issue,
        history_end_issue=str(state.history_end_issue),
        actual_text=actual_text,
        actual_rank=actual_rank,
        actual_probability=actual_probability,
        log_loss=-math.log(max(actual_probability, 1e-300)),
        brier_score=brier,
        direct_hit=actual_rank <= params.direct_top_k,
        group_hit=actual_group in {item.group_key for item in top_groups},
        group_random_probability=sum(item.permutations for item in top_groups) / 1000.0,
        group_rank=group_rank,
        position_ranks=(position_ranks[0], position_ranks[1], position_ranks[2]),
        top_probability=float(probabilities[int(order[0])]),
        top_hit=int(order[0]) == actual_index,
    )


def _gate_reasons(
    mean_rank: float,
    mean_log_loss: float,
    direct_p_value: float,
    group_p_value: float,
    position_p_value: float,
    stable_blocks: int,
) -> tuple[str, ...]:
    reasons = []
    if mean_rank >= 500.5:
        reasons.append("平均真实号排名未优于均匀随机中位排名")
    if mean_log_loss > math.log(1000):
        reasons.append("LogLoss 差于均匀分布")
    if direct_p_value >= 0.05:
        reasons.append("直选 TopK 单侧 p 值未小于 0.05")
    if group_p_value >= 0.05:
        reasons.append("组选 TopK 单侧 p 值未小于 0.05")
    if position_p_value >= 0.05:
        reasons.append("位置池 TopK 单侧 p 值未小于 0.05")
    if stable_blocks < 2:
        reasons.append("至少三个时间块中未达到两个稳定优势块")
    return tuple(reasons)


def build_gate_result(
    *,
    mean_rank: float,
    mean_log_loss: float,
    direct_p_value: float,
    group_p_value: float,
    position_p_value: float,
    stable_blocks: int,
) -> dict[str, Any]:
    """拆分公共概率/稳定性、直选命中和组选命中闸门。"""

    common_reasons = []
    if mean_rank >= 500.5:
        common_reasons.append("平均真实号排名未优于均匀随机中位排名")
    if mean_log_loss > math.log(1000):
        common_reasons.append("LogLoss 差于均匀分布")
    if stable_blocks < 2:
        common_reasons.append("至少三个时间块中未达到两个稳定优势块")
    direct_reasons = (
        ["直选 TopK 单侧 p 值未小于 0.05"] if direct_p_value >= 0.05 else []
    )
    group_reasons = ["组选 TopK 单侧 p 值未小于 0.05"] if group_p_value >= 0.05 else []
    position_reasons = (
        ["位置池 TopK 单侧 p 值未小于 0.05"] if position_p_value >= 0.05 else []
    )
    activation = resolve_activation(
        common_passed=not common_reasons,
        direct_passed=not direct_reasons,
        group_passed=not group_reasons,
        position_passed=not position_reasons,
    )
    return {
        "passed": activation["overallPassed"],
        "semantics": activation["overallSemantics"],
        "reasons": [
            *common_reasons,
            *direct_reasons,
            *group_reasons,
            *position_reasons,
        ],
        "common": {"passed": not common_reasons, "reasons": common_reasons},
        "direct": {"passed": not direct_reasons, "reasons": direct_reasons},
        "group": {"passed": not group_reasons, "reasons": group_reasons},
        "position": {"passed": not position_reasons, "reasons": position_reasons},
        "activation": activation,
    }


def build_candidate_budget_curve(
    periods: tuple[LearnedWalkForwardPeriod, ...],
) -> dict[str, dict[str, float | int]]:
    """在同一套排序上统计不同直选候选预算，不重复计算特征。"""
    total = len(periods)
    if not total:
        raise ValueError("候选预算曲线至少需要一个目标期")
    curve: dict[str, dict[str, float | int]] = {}
    for budget in DIRECT_CANDIDATE_BUDGETS:
        hits = sum(period.actual_rank <= budget for period in periods)
        rate = hits / total
        baseline = budget / 1000.0
        curve[str(budget)] = {
            "hits": hits,
            "periods": total,
            "hitRate": rate,
            "randomBaseline": baseline,
            "lift": rate / baseline if baseline else 0.0,
        }
    return curve


def build_group_budget_curve(
    periods: tuple[LearnedWalkForwardPeriod, ...],
) -> dict[str, dict[str, float | int]]:
    """统计不同组选 TopK 的命中曲线及固定组选空间随机基线。"""
    total = len(periods)
    curve: dict[str, dict[str, float | int]] = {}
    for budget in (10, 20, 50, 100, 150, 220):
        hits = sum(period.group_rank <= budget for period in periods)
        rate = hits / total
        baseline = budget / 220.0
        curve[str(budget)] = {
            "hits": hits,
            "periods": total,
            "hitRate": rate,
            "randomBaseline": baseline,
            "lift": rate / baseline if baseline else 0.0,
        }
    return curve


def build_position_pool_budget_curve(
    periods: tuple[LearnedWalkForwardPeriod, ...],
) -> dict[str, dict[str, float | int]]:
    """统计每位位置池大小的三位平均覆盖率及随机基线。"""
    total = len(periods) * 3
    curve: dict[str, dict[str, float | int]] = {}
    for budget in (3, 5, 7, 10):
        hits = sum(
            rank <= budget for period in periods for rank in period.position_ranks
        )
        rate = hits / total
        baseline = budget / 10.0
        curve[str(budget)] = {
            "hits": hits,
            "positions": total,
            "hitRate": rate,
            "randomBaseline": baseline,
            "lift": rate / baseline if baseline else 0.0,
        }
    return curve


def run_learned_ranker_walk_forward(
    history: pd.DataFrame,
    rule: LotteryRule,
    params: LearnedRankerParams,
    split: LearnedSplit,
    *,
    feature_config: LearnedFeatureConfig | None = None,
    csv_sha256: str | None = None,
    canonical_data_sha256: str | None = None,
    source_fingerprint: str | None = None,
    params_artifact_fingerprint: str | None = None,
    test_segment_used_for_selection: bool = False,
) -> LearnedRankerReport:
    """使用同一参数指纹逐期评估 frozen test，不执行任何调参。"""

    if rule.draw_count != 3 or rule.code not in {"fc3d", "pl3"}:
        raise ValueError("learned_ranker_v4 首版只支持 fc3d/pl3")
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    if split.test_end > len(chronological):
        raise ValueError("冻结测试终点超过历史长度")
    indices = tuple(range(split.validation_end, split.test_end))
    if not indices:
        raise ValueError("冻结测试段不能为空")
    effective_features = feature_config or LearnedFeatureConfig()
    states = iter_rolling_history_states(
        chronological, rule, indices, effective_features
    )
    periods = tuple(
        _period(
            chronological,
            index,
            rule,
            params,
            effective_features,
            history_state=state,
        )
        for index, state in zip(indices, states)
    )
    direct_hits = sum(item.direct_hit for item in periods)
    group_hits = sum(item.group_hit for item in periods)
    direct_probability = params.direct_top_k / 1000.0
    direct_p = float(binom.sf(direct_hits - 1, len(periods), direct_probability))
    group_probabilities = [item.group_random_probability for item in periods]
    group_p = float(poisson_binomial_right_tail(group_probabilities, group_hits))
    position_observations = len(periods) * 3
    position_hits = sum(
        rank <= params.position_pool_size
        for item in periods
        for rank in item.position_ranks
    )
    position_probability = params.position_pool_size / 10.0
    position_p = float(
        binom.sf(position_hits - 1, position_observations, position_probability)
    )
    blocks = [
        block
        for block in np.array_split(
            np.asarray([item.actual_rank for item in periods], dtype=float),
            min(3, len(periods)),
        )
        if len(block)
    ]
    block_means = tuple(float(block.mean()) for block in blocks)
    stable_blocks = sum(value < 500.5 for value in block_means)
    mean_rank = float(np.mean([item.actual_rank for item in periods]))
    mean_log_loss = float(np.mean([item.log_loss for item in periods]))
    reasons = _gate_reasons(
        mean_rank,
        mean_log_loss,
        direct_p,
        group_p,
        position_p,
        stable_blocks,
    )
    calibration, expected_calibration_error = evaluate_binary_calibration(
        [item.top_probability for item in periods],
        [item.top_hit for item in periods],
    )
    return LearnedRankerReport(
        rule_code=rule.code,
        display_name=rule.display_name,
        params_fingerprint=params_fingerprint(params, effective_features),
        split=split,
        test_target_indices=indices,
        periods=periods,
        mean_log_loss=mean_log_loss,
        uniform_log_loss=math.log(1000),
        mean_brier_score=float(np.mean([item.brier_score for item in periods])),
        uniform_brier_score=0.999,
        mean_rank=mean_rank,
        mean_rank_percentile=float(
            np.mean([(item.actual_rank - 1) / 999 for item in periods])
        ),
        direct_hits=direct_hits,
        direct_hit_rate=direct_hits / len(periods),
        direct_random_probability=direct_probability,
        direct_p_value=direct_p,
        group_hits=group_hits,
        group_hit_rate=group_hits / len(periods),
        group_random_hit_rate=float(np.mean(group_probabilities)),
        group_p_value=group_p,
        block_mean_ranks=block_means,
        stable_blocks=stable_blocks,
        gate_passed=not reasons,
        gate_reasons=reasons,
        expected_calibration_error=expected_calibration_error,
        calibration=calibration,
        position_hits=position_hits,
        position_hit_rate=position_hits / position_observations,
        position_random_probability=position_probability,
        position_p_value=position_p,
        csv_sha256=csv_sha256,
        canonical_data_sha256=canonical_data_sha256,
        source_fingerprint=source_fingerprint,
        params_artifact_fingerprint=params_artifact_fingerprint,
        test_segment_used_for_selection=test_segment_used_for_selection,
    )


def build_walk_forward_markdown(report: LearnedRankerReport) -> str:
    """生成包含闸门结论、分块稳定性和风险声明的 Markdown。"""

    mode = "通过冻结测试闸门" if report.gate_passed else "研究模式，不接入主推荐"
    lines = [
        f"# {report.display_name} learned_ranker_v4 冻结前推评估",
        "",
        f"- 结论：**{mode}**",
        f"- 参数指纹：`{report.params_fingerprint}`",
        f"- 冻结测试期数：`{len(report.periods)}`",
        f"- LogLoss：`{report.mean_log_loss:.6f}`（均匀 `{report.uniform_log_loss:.6f}`）",
        f"- Brier：`{report.mean_brier_score:.6f}`（均匀 `{report.uniform_brier_score:.6f}`）",
        f"- Top-1 ECE：`{report.expected_calibration_error:.6f}`",
        f"- 平均排名：`{report.mean_rank:.2f}`",
        f"- 直选 TopK：`{report.direct_hits}/{len(report.periods)}`，单侧 p=`{report.direct_p_value:.6f}`",
        f"- 组选 TopK：`{report.group_hits}/{len(report.periods)}`，"
        f"逐期精确随机基线 `{report.group_random_hit_rate:.4%}`，"
        f"单侧 p=`{report.group_p_value:.6f}`",
        f"- 位置池每位 TopK：`{report.position_hits}/{len(report.periods) * 3}`，"
        f"随机基线 `{report.position_random_probability:.4%}`，"
        f"单侧 p=`{report.position_p_value:.6f}`",
        f"- 分块平均排名：`{', '.join(f'{value:.2f}' for value in report.block_mean_ranks)}`",
        "",
        "## 闸门",
        "",
    ]
    lines.extend(
        [f"- {reason}" for reason in report.gate_reasons] or ["- 全部预设闸门通过。"]
    )
    lines.extend(
        [
            "",
            "## 风险声明",
            "",
            "彩票开奖结果具有高度随机性。本报告不宣称预测有效，不保证中奖或盈利，也不构成投注建议。",
            "",
        ]
    )
    return "\n".join(lines)


def write_walk_forward_report(
    report: LearnedRankerReport,
    output_dir: str | Path,
    *,
    prefix: str = "learned_ranker_v4",
) -> tuple[Path, Path]:
    """写出同源 Markdown 和 JSON 评估产物。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / f"{prefix}_{report.rule_code}.md"
    json_path = directory / f"{prefix}_{report.rule_code}.json"
    markdown = build_walk_forward_markdown(report)
    serialized = json.dumps(
        report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
    )
    version_suffix = payload_fingerprint(report.to_dict())[:12]
    for path, content in ((markdown_path, markdown), (json_path, serialized)):
        try:
            _preflight_immutable(path, content, label="冻结评估")
        except FileExistsError:
            versioned = path.with_name(f"{path.stem}.{version_suffix}{path.suffix}")
            _preflight_immutable(versioned, content, label="冻结评估")
            if path == markdown_path:
                markdown_path = versioned
            else:
                json_path = versioned
    _atomic_write(markdown_path, markdown, immutable=True)
    _atomic_write(json_path, serialized, immutable=True)
    return markdown_path, json_path

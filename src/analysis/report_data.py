# -*- coding: utf-8 -*-
"""快乐8日报结构化 JSON 数据。

该模块把日报内部对象转换成前端/Vue 友好的稳定 JSON 结构，避免后续 UI 从 Markdown 表格反解析。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def _candidate_group(group: Any, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "numbers": [int(number) for number in group.numbers],
        "score": float(group.score),
        "oddCount": int(group.odd_count),
        "evenCount": int(len(group.numbers) - group.odd_count),
        "bigCount": int(group.big_count),
        "smallCount": int(len(group.numbers) - group.big_count),
        "repeatLastCount": int(group.repeat_last_count),
        "zoneDistribution": [int(value) for value in group.zone_distribution],
    }


def _counter_dict(counter: Any) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(dict(counter).items(), key=lambda item: int(item[0]))}


def _backtest(summary: Any | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "drawCount": int(summary.draw_count),
        "groupCount": int(summary.group_count),
        "totalBets": int(summary.total_bets),
        "ticketPrice": int(summary.ticket_price),
        "totalCost": int(summary.total_cost),
        "totalPrize": int(summary.total_prize),
        "roi": float(summary.roi),
        "averageHit": float(summary.average_hit),
        "hitDistribution": _counter_dict(summary.hit_distribution),
        "maxMissStreak": int(summary.max_miss_streak),
    }


def _strategy_comparison(comparison: dict[str, Any] | None) -> dict[str, Any]:
    if not comparison:
        return {}
    return {
        strategy: {
            "strategy": result.strategy,
            "groups": [[int(number) for number in group] for group in result.groups],
            "summary": _backtest(result.summary),
        }
        for strategy, result in comparison.items()
    }


def _sliding(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    output = {}
    for strategy, result in summary.items():
        output[strategy] = {
            "strategy": result.strategy,
            "windowCount": int(result.window_count),
            "meanRoi": float(result.mean_roi),
            "bestRoi": float(result.best_roi),
            "worstRoi": float(result.worst_roi),
            "roiStd": float(result.roi_std),
            "meanHit": float(result.mean_hit),
            "meanHit5Plus": float(result.mean_hit5_plus),
            "windows": [
                {
                    "label": window.label,
                    "startIssue": window.start_issue,
                    "endIssue": window.end_issue,
                    "summary": _backtest(window.summary),
                }
                for window in result.windows
            ],
        }
    return output


def _parameter_search(results: Sequence[Any] | None) -> list[dict[str, Any]]:
    if not results:
        return []
    rows = []
    for rank, result in enumerate(results, 1):
        config = result.config
        rows.append(
            {
                "rank": rank,
                "name": config.name,
                "score": float(result.score),
                "weights": {
                    "hot": float(config.hot_weight),
                    "cold": float(config.cold_weight),
                    "omission": float(config.omission_weight),
                    "random": float(config.random_weight),
                    "repeatLast": float(getattr(config, "repeat_last_weight", 0.0)),
                },
                "maxRepeatLast": int(config.max_repeat_last),
                "groups": [[int(number) for number in group] for group in result.groups],
                "slidingSummary": _sliding({config.name: result.sliding_summary})[config.name],
            }
        )
    return rows


def build_report_data(
    *,
    stats: Any,
    groups: Sequence[Any],
    parameter_name: str,
    backtest_summary: Any | None = None,
    strategy_comparison: dict[str, Any] | None = None,
    sliding_window_summary: dict[str, Any] | None = None,
    parameter_search_results: Sequence[Any] | None = None,
    pick_snapshot_path: str | Path | None = None,
    html_path: str | Path | None = None,
    data_quality: Any | None = None,
    candidate_coverage: Any | None = None,
    candidate_portfolio_score: Any | None = None,
    candidate_batch_optimization: Any | None = None,
    walk_forward_validation: Any | None = None,
    strategy_mode: str | None = None,
    data_source: dict[str, Any] | None = None,
    live_parameter_weights: dict[str, Any] | None = None,
    walk_forward_parameter_weights: dict[str, Any] | None = None,
    review_feedback: Any | None = None,
    betting_plan: Any | None = None,
) -> dict[str, Any]:
    """构造 Vue/前端可直接消费的结构化日报数据。"""

    primary_window = min(stats.frequency_by_window)
    primary_frequency = stats.frequency_by_window[primary_window]
    return {
        "schemaVersion": 1,
        "lottery": "kl8",
        "play": "select10",
        "issue": str(stats.latest_issue),
        "parameterName": parameter_name,
        "strategyMode": strategy_mode or "auto",
        "latestNumbers": [int(number) for number in stats.latest_numbers],
        "frequencyWindow": int(primary_window),
        "hotNumbers": [int(number) for number in stats.hot_numbers[:20]],
        "coldNumbers": [int(number) for number in stats.cold_numbers[:20]],
        "currentOmission": {str(number): int(stats.current_omission[number]) for number in range(1, 81)},
        "frequency": {str(number): int(primary_frequency[number]) for number in range(1, 81)},
        "zoneDistribution": [int(value) for value in stats.zone_distribution],
        "tailDistribution": [int(value) for value in stats.tail_distribution],
        "candidateGroups": [_candidate_group(group, rank) for rank, group in enumerate(groups, 1)],
        "backtest": _backtest(backtest_summary),
        "strategyComparison": _strategy_comparison(strategy_comparison),
        "slidingWindow": _sliding(sliding_window_summary),
        "parameterSearch": _parameter_search(parameter_search_results),
        "candidateCoverage": candidate_coverage.to_dict() if candidate_coverage is not None else None,
        "candidatePortfolioScore": candidate_portfolio_score.to_dict() if candidate_portfolio_score is not None else None,
        "candidateBatchOptimization": candidate_batch_optimization.to_dict() if candidate_batch_optimization is not None else None,
        "walkForwardValidation": walk_forward_validation.to_dict() if walk_forward_validation is not None else None,
        "dataSource": data_source,
        "liveParameterWeights": live_parameter_weights,
        "walkForwardParameterWeights": walk_forward_parameter_weights,
        "reviewFeedback": review_feedback.to_dict() if review_feedback is not None else None,
        "bettingPlan": betting_plan.to_dict() if betting_plan is not None else None,
        "artifacts": {
            "pickSnapshot": str(pick_snapshot_path) if pick_snapshot_path else None,
            "html": str(html_path) if html_path else None,
        },
        "dataQuality": data_quality.to_dict() if data_quality is not None else None,
        "disclaimer": "仅做历史数据统计和娱乐参考，不保证中奖。",
    }


def write_report_data(
    *,
    stats: Any,
    groups: Sequence[Any],
    parameter_name: str,
    backtest_summary: Any | None = None,
    strategy_comparison: dict[str, Any] | None = None,
    sliding_window_summary: dict[str, Any] | None = None,
    parameter_search_results: Sequence[Any] | None = None,
    output_dir: Path | str,
    pick_snapshot_path: str | Path | None = None,
    html_path: str | Path | None = None,
    data_quality: Any | None = None,
    candidate_coverage: Any | None = None,
    candidate_portfolio_score: Any | None = None,
    candidate_batch_optimization: Any | None = None,
    walk_forward_validation: Any | None = None,
    strategy_mode: str | None = None,
    data_source: dict[str, Any] | None = None,
    live_parameter_weights: dict[str, Any] | None = None,
    walk_forward_parameter_weights: dict[str, Any] | None = None,
    review_feedback: Any | None = None,
    betting_plan: Any | None = None,
) -> Path:
    """写入 reports/data/kl8_daily_<期号>.json。"""

    data = build_report_data(
        stats=stats,
        groups=groups,
        parameter_name=parameter_name,
        backtest_summary=backtest_summary,
        strategy_comparison=strategy_comparison,
        sliding_window_summary=sliding_window_summary,
        parameter_search_results=parameter_search_results,
        pick_snapshot_path=pick_snapshot_path,
        html_path=html_path,
        data_quality=data_quality,
        candidate_coverage=candidate_coverage,
        candidate_portfolio_score=candidate_portfolio_score,
        candidate_batch_optimization=candidate_batch_optimization,
        walk_forward_validation=walk_forward_validation,
        strategy_mode=strategy_mode,
        data_source=data_source,
        live_parameter_weights=live_parameter_weights,
        walk_forward_parameter_weights=walk_forward_parameter_weights,
        review_feedback=review_feedback,
        betting_plan=betting_plan,
    )
    directory = Path(output_dir) / "data"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"kl8_daily_{stats.latest_issue}.json"
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output

# -*- coding: utf-8 -*-
"""三位彩固定评分 v4 参数搜索与冻结前推测试。"""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

import src.analysis.digit_learned_ranker_search as search_module
import src.analysis.digit_learned_ranker_walk_forward as walk_forward_module
from src.analysis.digit_learned_features import LearnedFeatureConfig
from src.analysis.digit_learned_ranker import LearnedRankerParams, params_fingerprint
from src.analysis.digit_learned_ranker_search import (
    LearnedSearchConfig,
    LearnedSplit,
    search_learned_ranker_params,
)
from src.analysis.digit_learned_ranker_walk_forward import (
    LearnedWalkForwardPeriod,
    build_candidate_budget_curve,
    build_group_budget_curve,
    build_position_pool_budget_curve,
    build_walk_forward_markdown,
    run_learned_ranker_walk_forward,
    write_walk_forward_report,
)
from src.lotteries import get_lottery_rule
from tests.test_digit_learned_features import _history


def test_search_and_validation_are_isolated_from_frozen_test_values():
    rule = get_lottery_rule("fc3d")
    original = _history(24)
    changed = original.copy()
    changed.loc[18:, ["百位", "十位", "个位"]] = [9, 9, 9]
    config = LearnedSearchConfig(
        split=LearnedSplit(search_end=12, validation_end=18, test_end=24),
        min_train_size=8,
        random_trials=2,
        local_trials=1,
        evaluation_stride=2,
        seed=7,
        feature_config=LearnedFeatureConfig(windows=(5, "all")),
    )

    first = search_learned_ranker_params(original, rule, config)
    second = search_learned_ranker_params(changed, rule, config)

    assert params_fingerprint(first.params) == params_fingerprint(second.params)
    assert first.validation_objective == second.validation_objective
    assert first.test_segment_used_for_selection is False
    assert max(first.selection_target_indices) < config.split.validation_end


def test_search_can_reproducibly_compare_feature_window_decay_and_omission_configs():
    rule = get_lottery_rule("pl3")
    feature_configs = (
        LearnedFeatureConfig(windows=(5, "all"), half_life=None, omission_cap=20),
        LearnedFeatureConfig(windows=(8, "all"), half_life=20, omission_cap=50),
    )
    config = LearnedSearchConfig(
        split=LearnedSplit(search_end=12, validation_end=18, test_end=20),
        min_train_size=8,
        random_trials=1,
        local_trials=0,
        evaluation_stride=3,
        seed=11,
        feature_config=feature_configs[0],
        feature_configs=feature_configs,
    )

    first = search_learned_ranker_params(_history(20), rule, config)
    second = search_learned_ranker_params(_history(20), rule, config)

    assert first.to_dict() == second.to_dict()
    assert first.feature_config in feature_configs
    assert {trial.feature_config for trial in first.trials} == set(feature_configs)


def test_validation_objective_does_not_choose_local_search_start(monkeypatch):
    rule = get_lottery_rule("fc3d")
    config = LearnedSearchConfig(
        split=LearnedSplit(search_end=12, validation_end=18, test_end=20),
        min_train_size=8,
        random_trials=1,
        local_trials=1,
        evaluation_stride=2,
        seed=3,
        feature_config=LearnedFeatureConfig(windows=(5, "all")),
    )
    random_params = replace(LearnedRankerParams(), temperature=0.5)
    local_bases = []

    monkeypatch.setattr(search_module, "_random_params", lambda rng: random_params)

    def fake_prepare(chronological, rule, indices, feature_config):
        return (search_module._PreparedTarget(pd.DataFrame(), str(indices[0])),)

    def fake_objective(targets, params):
        is_validation = targets[0].actual_text == str(config.split.search_end)
        return 2.0 if is_validation and params.temperature == 0.5 else 1.0

    def fake_local(base, rng):
        local_bases.append(base)
        return base

    monkeypatch.setattr(search_module, "_prepare_targets", fake_prepare)
    monkeypatch.setattr(search_module, "_objective", fake_objective)
    monkeypatch.setattr(search_module, "_local_params", fake_local)

    search_learned_ranker_params(_history(20), rule, config)

    assert local_bases == [LearnedRankerParams()]


def test_frozen_walk_forward_is_repeatable_and_reports_exact_group_baseline(tmp_path):
    rule = get_lottery_rule("pl3")
    history = _history(20)
    params = LearnedRankerParams(direct_top_k=10, group_top_k=10)
    split = LearnedSplit(search_end=10, validation_end=14, test_end=20)

    first = run_learned_ranker_walk_forward(
        history,
        rule,
        params,
        split,
        feature_config=LearnedFeatureConfig(windows=(5, "all")),
    )
    second = run_learned_ranker_walk_forward(
        history,
        rule,
        params,
        split,
        feature_config=LearnedFeatureConfig(windows=(5, "all")),
    )

    assert first.to_dict() == second.to_dict()
    assert first.params_fingerprint == params_fingerprint(
        params, LearnedFeatureConfig(windows=(5, "all"))
    )
    assert first.test_target_indices == tuple(range(14, 20))
    assert all(0 < item.group_random_probability <= 1 for item in first.periods)
    assert first.uniform_log_loss > 6.9
    assert "研究模式，不接入主推荐" in build_walk_forward_markdown(first)

    markdown_path, json_path = write_walk_forward_report(first, tmp_path, prefix="v4")
    assert markdown_path.exists() and json_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))[
        "paramsFingerprint"
    ] == params_fingerprint(params, LearnedFeatureConfig(windows=(5, "all")))


def test_frozen_walk_forward_rejects_pl5():
    rule = get_lottery_rule("pl5")
    history = pd.DataFrame(
        [{"期数": "1", "万位": 1, "千位": 2, "百位": 3, "十位": 4, "个位": 5}]
    )

    try:
        run_learned_ranker_walk_forward(
            history,
            rule,
            replace(LearnedRankerParams(), direct_top_k=5),
            LearnedSplit(search_end=0, validation_end=0, test_end=1),
        )
    except ValueError as exc:
        assert "只支持 fc3d/pl3" in str(exc)
    else:
        raise AssertionError("pl5 必须被 learned_ranker_v4 明确拒绝")


def test_walk_forward_result_is_independent_of_candidate_row_order(monkeypatch):
    rule = get_lottery_rule("fc3d")
    history = _history(20)
    params = LearnedRankerParams(direct_top_k=10, group_top_k=10)
    split = LearnedSplit(search_end=10, validation_end=14, test_end=20)
    feature_config = LearnedFeatureConfig(windows=(5, "all"))
    original_builder = walk_forward_module.build_candidate_features

    expected = run_learned_ranker_walk_forward(
        history, rule, params, split, feature_config=feature_config
    )

    def shuffled_builder(*args, **kwargs):
        return (
            original_builder(*args, **kwargs)
            .sample(frac=1.0, random_state=19)
            .reset_index(drop=True)
        )

    monkeypatch.setattr(
        walk_forward_module, "build_candidate_features", shuffled_builder
    )
    actual = run_learned_ranker_walk_forward(
        history, rule, params, split, feature_config=feature_config
    )

    assert actual.to_dict() == expected.to_dict()


def test_frozen_evaluation_artifacts_are_immutable(tmp_path):
    rule = get_lottery_rule("pl3")
    report = run_learned_ranker_walk_forward(
        _history(20),
        rule,
        LearnedRankerParams(),
        LearnedSplit(search_end=10, validation_end=14, test_end=20),
        feature_config=LearnedFeatureConfig(windows=(5, "all")),
    )
    _, json_path = write_walk_forward_report(report, tmp_path)
    original = json_path.read_bytes()
    changed = replace(report, gate_passed=not report.gate_passed)

    with pytest.raises(FileExistsError, match="冻结评估"):
        write_walk_forward_report(changed, tmp_path)

    assert json_path.read_bytes() == original


def test_candidate_budget_curve_reports_hit_rate_and_random_lift():
    periods = tuple(
        LearnedWalkForwardPeriod(
            target_index=index,
            target_issue=str(index),
            history_end_issue=str(index - 1),
            actual_text="000",
            actual_rank=rank,
            actual_probability=0.001,
            log_loss=6.9,
            brier_score=0.99,
            direct_hit=rank <= 10,
            group_hit=False,
            group_random_probability=0.1,
            group_rank=rank % 220 + 1,
            position_ranks=(rank % 10 + 1, rank % 10 + 1, rank % 10 + 1),
        )
        for index, rank in enumerate((1, 20, 500, 1000), start=1)
    )
    curve = build_candidate_budget_curve(periods)

    assert curve["10"] == {
        "hits": 1,
        "periods": 4,
        "hitRate": 0.25,
        "randomBaseline": 0.01,
        "lift": 25.0,
    }
    assert curve["1000"]["hitRate"] == 1.0
    assert curve["1000"]["randomBaseline"] == 1.0
    group_curve = build_group_budget_curve(periods)
    pool_curve = build_position_pool_budget_curve(periods)
    assert group_curve["220"]["hitRate"] == 1.0
    assert pool_curve["10"]["hitRate"] == 1.0
    assert pool_curve["10"]["randomBaseline"] == 1.0

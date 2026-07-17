# -*- coding: utf-8 -*-
"""三位彩固定评分 v4 排序测试。"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from src.analysis.digit_learned_features import (
    LearnedFeatureConfig,
    build_candidate_features,
    build_history_state,
)
from src.analysis.digit_learned_ranker import (
    LearnedRankerParams,
    aggregate_group_candidates,
    build_learned_ranker_plan,
    load_params,
    params_fingerprint,
    probabilities_from_scores,
    rank_candidate_indices,
    save_params,
)
from src.lotteries import get_lottery_rule
from tests.test_digit_learned_features import _history


def test_equal_scores_use_candidate_text_as_stable_tie_break():
    candidates = ["002", "000", "001"]
    order = rank_candidate_indices(np.zeros(3), candidates)
    assert order.tolist() == [1, 2, 0]


def test_probabilities_normalize_and_group_probability_sums_permutations():
    candidates = [f"{value:03d}" for value in range(1000)]
    probabilities = probabilities_from_scores(np.zeros(1000), temperature=1.0)
    groups = aggregate_group_candidates(candidates, probabilities)
    group = next(item for item in groups if item.group_key == "123")

    assert np.isclose(probabilities.sum(), 1.0)
    assert group.permutations == 6
    assert np.isclose(group.probability, 6 / 1000)
    assert np.isclose(sum(item.probability for item in groups), 1.0)


def test_parameter_fingerprint_is_stable_and_sensitive():
    params = LearnedRankerParams()
    assert params_fingerprint(params) == params_fingerprint(params)
    assert params_fingerprint(params) != params_fingerprint(
        replace(params, temperature=0.5)
    )
    assert params_fingerprint(
        params, LearnedFeatureConfig(windows=(10, "all"), half_life=20)
    ) != params_fingerprint(
        params, LearnedFeatureConfig(windows=(30, "all"), half_life=80)
    )


def test_plan_is_deterministic_and_contains_direct_group_and_pools():
    rule = get_lottery_rule("fc3d")
    params = LearnedRankerParams(direct_top_k=10, group_top_k=10)
    state = build_history_state(_history(), rule, LearnedFeatureConfig())
    features = build_candidate_features(state, rule)

    first = build_learned_ranker_plan(features, params, rule)
    second = build_learned_ranker_plan(features, params, rule)

    assert first.to_dict() == second.to_dict()
    assert len(first.direct_candidates) == 10
    assert len(first.group_candidates) == 10
    assert len(first.position_pools) == 3
    assert first.group_digit_pool


def test_plan_distribution_is_independent_of_candidate_row_order():
    rule = get_lottery_rule("fc3d")
    params = LearnedRankerParams(direct_top_k=10, group_top_k=10)
    state = build_history_state(_history(), rule, LearnedFeatureConfig())
    features = build_candidate_features(state, rule)
    shuffled = features.sample(frac=1.0, random_state=17).reset_index(drop=True)

    assert build_learned_ranker_plan(features, params, rule).to_dict() == (
        build_learned_ranker_plan(shuffled, params, rule).to_dict()
    )


def test_parameter_file_detects_metadata_tampering(tmp_path):
    path = tmp_path / "params.json"
    save_params(
        LearnedRankerParams(),
        path,
        metadata={
            "featureConfig": {
                "windows": [5, "all"],
                "alpha": 2.0,
                "halfLife": None,
                "omissionCap": 50,
            },
            "split": {"searchEnd": 10, "validationEnd": 15, "testEnd": 20},
            "testSegmentUsedForSelection": False,
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"]["split"]["validationEnd"] = 16
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="参数文件指纹校验失败"):
        load_params(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"direct_top_k": 1001},
        {"group_top_k": 221},
        {"position_pool_size": 11},
        {"group_digit_pool_size": 11},
    ],
)
def test_parameter_ranges_reject_impossible_candidate_counts(changes):
    with pytest.raises(ValueError):
        LearnedRankerParams(**changes)

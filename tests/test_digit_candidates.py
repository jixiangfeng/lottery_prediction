# -*- coding: utf-8 -*-

from dataclasses import replace

import pandas as pd
import pytest

from src.analysis.digit_candidates import (
    ENSEMBLE_MODEL_NAMES,
    DigitCandidateConfig,
    _effective_config,
    _enumerate_scored_candidates,
    generate_digit_betting_candidates,
    generate_digit_candidates,
    generate_uniform_digit_betting_candidates,
    generate_uniform_digit_candidates,
    score_digit_prefix,
)
from src.analysis.digit_statistics import analyze_digit_history, classify_digit_shape
from src.lotteries import get_lottery_rule


def test_generate_digit_candidates_for_fc3d_respects_filters_and_shape():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "2026005", "百位": 1, "十位": 2, "个位": 3},
            {"期数": "2026004", "百位": 1, "十位": 1, "个位": 2},
            {"期数": "2026003", "百位": 4, "十位": 5, "个位": 6},
            {"期数": "2026002", "百位": 7, "十位": 8, "个位": 9},
            {"期数": "2026001", "百位": 0, "十位": 0, "个位": 0},
        ]
    )
    stats = analyze_digit_history(df, rule)
    config = DigitCandidateConfig(
        count=8,
        sum_min=6,
        sum_max=18,
        span_min=1,
        span_max=8,
        allowed_shapes=("组六", "组三"),
    )

    result = generate_digit_candidates(stats, rule, config=config, seed=7)

    assert len(result.candidates) == 8
    assert result.rule_code == "fc3d"
    for candidate in result.candidates:
        assert 6 <= candidate.sum_value <= 18
        assert 1 <= candidate.span <= 8
        assert candidate.shape in {"组六", "组三"}
        assert candidate.shape == classify_digit_shape(candidate.numbers)
        assert len(candidate.text) == 3


def test_default_fc3d_candidates_exclude_latest_and_baozi():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "2026003", "百位": 8, "十位": 1, "个位": 8},
            {"期数": "2026002", "百位": 1, "十位": 1, "个位": 1},
            {"期数": "2026001", "百位": 2, "十位": 3, "个位": 4},
        ]
    )
    stats = analyze_digit_history(df, rule)

    result = generate_digit_candidates(
        stats, rule, config=DigitCandidateConfig(count=8), seed=11
    )

    assert all(candidate.text != "818" for candidate in result.candidates)
    assert all(candidate.shape != "豹子" for candidate in result.candidates)
    assert sum(1 for candidate in result.candidates if candidate.shape == "组三") <= 3
    assert (
        sum(1 for candidate in result.candidates[:5] if candidate.shape == "组三") <= 1
    )
    assert all(6 <= candidate.sum_value <= 21 for candidate in result.candidates)
    assert all(
        (candidate.span >= 1 if candidate.shape == "组三" else candidate.span >= 2)
        and candidate.span <= 9
        for candidate in result.candidates
    )


def test_default_fc3d_profile_uses_conservative_group_model_weights():
    rule = get_lottery_rule("fc3d")

    effective = _effective_config(rule, DigitCandidateConfig())

    assert effective.omission_weight == pytest.approx(0.03)
    assert effective.pair_weight == 0.0
    assert effective.shape_weight == 0.0
    assert effective.sum_weight == 0.0
    assert effective.span_weight == 0.0


def test_custom_fc3d_profile_keeps_explicit_model_weights():
    rule = get_lottery_rule("fc3d")
    config = DigitCandidateConfig(
        pair_weight=0.8,
        shape_weight=0.1,
        sum_weight=0.05,
        span_weight=0.02,
        omission_weight=0.2,
    )

    effective = _effective_config(rule, config)

    assert effective.pair_weight == pytest.approx(0.8)
    assert effective.shape_weight == pytest.approx(0.1)
    assert effective.sum_weight == pytest.approx(0.05)
    assert effective.span_weight == pytest.approx(0.02)
    assert effective.omission_weight == pytest.approx(0.2)


def test_generate_digit_candidates_for_pl5_keeps_leading_zero_text():
    rule = get_lottery_rule("pl5")
    df = pd.DataFrame(
        [
            {"期数": "2026003", "万位": 0, "千位": 1, "百位": 2, "十位": 3, "个位": 4},
            {"期数": "2026002", "万位": 9, "千位": 9, "百位": 8, "十位": 8, "个位": 7},
            {"期数": "2026001", "万位": 5, "千位": 6, "百位": 7, "十位": 8, "个位": 9},
        ]
    )
    stats = analyze_digit_history(df, rule)
    config = DigitCandidateConfig(
        count=6,
        sum_min=5,
        sum_max=35,
        allowed_shapes=("全不同", "二二一", "二一一一"),
        exclude_latest=False,
    )

    result = generate_digit_candidates(stats, rule, config=config, seed=3)

    assert len(result.candidates) == 6
    assert result.rule_code == "pl5"
    assert all(len(candidate.text) == 5 for candidate in result.candidates)
    assert any(candidate.text.startswith("0") for candidate in result.candidates)


def test_default_pl5_candidates_exclude_latest_and_heavy_repeats():
    rule = get_lottery_rule("pl5")
    df = pd.DataFrame(
        [
            {"期数": "2026003", "万位": 1, "千位": 2, "百位": 4, "十位": 4, "个位": 4},
            {"期数": "2026002", "万位": 9, "千位": 9, "百位": 8, "十位": 8, "个位": 7},
            {"期数": "2026001", "万位": 0, "千位": 1, "百位": 2, "十位": 3, "个位": 4},
        ]
    )
    stats = analyze_digit_history(df, rule)

    result = generate_digit_candidates(
        stats, rule, config=DigitCandidateConfig(count=8), seed=13
    )

    assert all(candidate.text != "12444" for candidate in result.candidates)
    assert all(
        candidate.shape in {"全不同", "二一一一", "二二一", "三一一", "三二"}
        for candidate in result.candidates
    )
    assert (
        sum(candidate.shape in {"三一一", "三二"} for candidate in result.candidates)
        <= 1
    )
    assert all(10 <= candidate.sum_value <= 35 for candidate in result.candidates)
    assert all(3 <= candidate.span <= 9 for candidate in result.candidates)


def test_digit_candidate_result_to_dict_is_report_friendly():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame([{"期数": "2026001", "百位": 1, "十位": 2, "个位": 3}])
    stats = analyze_digit_history(df, rule)

    result = generate_digit_candidates(
        stats, rule, config=DigitCandidateConfig(count=2), seed=1
    )
    payload = result.to_dict()

    assert payload["ruleCode"] == "fc3d"
    assert len(payload["candidates"]) == 2
    assert {"text", "numbers", "sum", "span", "shape", "score"} <= set(
        payload["candidates"][0]
    )
    assert payload["candidates"][0]["modelWeight"] > 0
    assert payload["candidates"][0]["compositeModelWeight"] > 0
    assert (
        payload["candidates"][0]["jointProbability"]
        == payload["candidates"][0]["compositeModelWeight"]
    )
    assert payload["candidates"][0]["jointProbabilityDeprecated"] is True
    assert "randomWeight" in payload["config"]["deprecatedCompatibilityFields"]


def test_three_digit_group3_allows_span_one_by_default():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "2026003", "百位": 1, "十位": 2, "个位": 3},
            {"期数": "2026002", "百位": 4, "十位": 5, "个位": 6},
            {"期数": "2026001", "百位": 7, "十位": 8, "个位": 9},
        ]
    )
    stats = analyze_digit_history(df, rule)
    config = DigitCandidateConfig(
        count=6, span_max=1, allowed_shapes=("组三",), exclude_latest=False
    )

    result = generate_digit_candidates(stats, rule, config=config, seed=1)

    assert len(result.candidates) == 6
    assert all(
        candidate.shape == "组三" and candidate.span == 1
        for candidate in result.candidates
    )


def test_default_pl5_keeps_defensive_triple_shapes_as_small_quota():
    rule = get_lottery_rule("pl5")
    df = pd.DataFrame(
        [
            {
                "期数": f"2026{index:03d}",
                "万位": index % 10,
                "千位": (index + 1) % 10,
                "百位": (index + 2) % 10,
                "十位": (index + 3) % 10,
                "个位": (index + 4) % 10,
            }
            for index in range(1, 31)
        ]
    )
    stats = analyze_digit_history(df, rule)

    result = generate_digit_candidates(
        stats, rule, config=DigitCandidateConfig(count=20), seed=3
    )

    defensive = [
        candidate
        for candidate in result.candidates
        if candidate.shape in {"三一一", "三二"}
    ]
    assert 1 <= len(defensive) <= 4
    assert (
        sum(
            candidate.shape in {"三一一", "三二"} for candidate in result.candidates[:5]
        )
        <= 1
    )
    assert len(defensive) < len(result.candidates) / 2


def test_default_pl5_candidates_cover_at_least_half_digits_per_position():
    rule = get_lottery_rule("pl5")
    df = pd.DataFrame(
        [
            {
                "期数": f"2026{index:03d}",
                "万位": index % 10,
                "千位": (index + 1) % 10,
                "百位": (index + 2) % 10,
                "十位": (index + 3) % 10,
                "个位": (index + 4) % 10,
            }
            for index in range(1, 51)
        ]
    )
    stats = analyze_digit_history(df, rule)

    result = generate_digit_candidates(
        stats, rule, config=DigitCandidateConfig(count=10)
    )

    for position in range(5):
        assert (
            len({candidate.numbers[position] for candidate in result.candidates}) >= 6
        )


def test_full_space_scoring_is_deterministic_and_not_limited_by_top_digits():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "2026003", "百位": 1, "十位": 2, "个位": 3},
            {"期数": "2026002", "百位": 1, "十位": 2, "个位": 4},
            {"期数": "2026001", "百位": 1, "十位": 5, "个位": 6},
        ]
    )
    stats = analyze_digit_history(df, rule)
    config = DigitCandidateConfig(
        count=8, top_digits_per_position=1, exclude_latest=False
    )

    first = generate_digit_candidates(stats, rule, config=config, seed=1)
    second = generate_digit_candidates(stats, rule, config=config, seed=999)

    assert len(first.candidates) == 8
    assert [candidate.text for candidate in first.candidates] == [
        candidate.text for candidate in second.candidates
    ]
    unique_by_position = [
        len({candidate.numbers[position] for candidate in first.candidates})
        for position in range(3)
    ]
    assert max(unique_by_position) >= 3
    assert len({candidate.text for candidate in first.candidates}) == 8


def test_candidate_generators_reject_filters_that_cannot_fill_requested_count():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame([{"期数": "1", "百位": 1, "十位": 2, "个位": 3}])
    stats = analyze_digit_history(df, rule)
    config = DigitCandidateConfig(count=10, allowed_shapes=("豹子",), span_min=1)

    with pytest.raises(ValueError, match="候选不足"):
        generate_digit_candidates(stats, rule, config=config)
    with pytest.raises(ValueError, match="候选不足"):
        generate_uniform_digit_candidates(stats, rule, config=config)


def test_joint_log_probability_ranks_observed_pair_above_marginal_decoy():
    rule = get_lottery_rule("fc3d")
    rows = []
    for index in range(80):
        first, second = (1, 2) if index % 2 == 0 else (3, 4)
        rows.append(
            {
                "期数": str(2026001 + index),
                "百位": first,
                "十位": second,
                "个位": 5 if index % 3 else 6,
            }
        )
    stats = analyze_digit_history(pd.DataFrame(rows), rule, frequency_windows=(20, 80))
    config = DigitCandidateConfig(
        count=20,
        allowed_shapes=("组六", "组三", "豹子"),
        sum_min=0,
        sum_max=27,
        span_min=0,
        span_max=9,
        frequency_windows=(20, 80),
        frequency_window_weights=(0.5, 0.5),
        marginal_weight=1.0,
        pair_weight=3.0,
        shape_weight=0.0,
        sum_weight=0.0,
        span_weight=0.0,
        omission_weight=0.0,
        exclude_latest=False,
        diversity_weight=0.0,
        score_floor=30.0,
    )

    result = generate_digit_candidates(stats, rule, config=config)
    scores = {candidate.text: candidate.score for candidate in result.candidates}

    assert scores["125"] > scores.get("145", float("-inf"))


def test_diversity_never_selects_below_configured_score_floor():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": str(2026001 + index), "百位": 1, "十位": 2, "个位": index % 10}
            for index in range(50)
        ]
    )
    stats = analyze_digit_history(df, rule)
    config = DigitCandidateConfig(
        count=12,
        score_floor=6.0,
        diversity_weight=1.0,
        exclude_latest=False,
        allowed_shapes=("组六",),
    )

    result = generate_digit_candidates(stats, rule, config=config)
    score_only = generate_digit_candidates(
        stats, rule, config=replace(config, diversity_weight=0.0)
    )
    scores = [candidate.score for candidate in result.candidates]

    assert min(scores) >= max(scores) - config.score_floor - 1e-9
    assert (
        min(scores)
        >= min(candidate.score for candidate in score_only.candidates) - 1e-9
    )


def test_default_score_floor_builds_a_strict_quality_pool():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 3,
                "十位": (index + 1) % 4,
                "个位": (index + 2) % 5,
            }
            for index in range(80)
        ]
    )
    stats = analyze_digit_history(df, rule)
    config = _effective_config(rule, DigitCandidateConfig())
    ranked = _enumerate_scored_candidates(stats, rule, config, None)

    assert config.score_floor == 2.0
    assert any(
        candidate.score < ranked[0].score - config.score_floor for candidate in ranked
    )


@pytest.mark.parametrize(
    ("allowed_shapes", "count", "expected_group6", "expected_group3"),
    [
        (None, 10, 8, 2),
        (None, 20, 16, 4),
        (("组六",), 10, 10, 0),
        (("组三",), 10, 0, 10),
    ],
)
def test_three_digit_statistical_and_random_candidates_share_exact_shape_budget(
    allowed_shapes, count, expected_group6, expected_group3
):
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 10,
                "十位": (index * 3 + 1) % 10,
                "个位": (index * 7 + 2) % 10,
            }
            for index in range(80)
        ]
    )
    stats = analyze_digit_history(df, rule)
    config = DigitCandidateConfig(count=count, allowed_shapes=allowed_shapes)

    statistical = generate_digit_betting_candidates(
        stats, rule, config=config, group_count=count
    )
    random_plan = generate_uniform_digit_betting_candidates(
        stats, rule, config=config, group_count=count, seed=7
    )

    for plan in (statistical, random_plan):
        assert len(plan.direct_candidates) == count
        assert len(plan.group_candidates) == count
        assert (
            sum(candidate.shape == "组六" for candidate in plan.direct_candidates)
            == expected_group6
        )
        assert (
            sum(candidate.shape == "组三" for candidate in plan.direct_candidates)
            == expected_group3
        )
        assert (
            sum(candidate.shape == "组六" for candidate in plan.group_candidates)
            == expected_group6
        )
        assert (
            sum(candidate.shape == "组三" for candidate in plan.group_candidates)
            == expected_group3
        )


@pytest.mark.parametrize("count", [10, 20])
def test_pl5_statistical_and_random_candidates_share_exact_defensive_budget(count):
    rule = get_lottery_rule("pl5")
    df = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "万位": index % 10,
                "千位": (index * 3 + 1) % 10,
                "百位": (index * 7 + 2) % 10,
                "十位": (index + 4) % 10,
                "个位": (index + 8) % 10,
            }
            for index in range(80)
        ]
    )
    stats = analyze_digit_history(df, rule)
    config = DigitCandidateConfig(count=count)

    statistical = generate_digit_candidates(stats, rule, config=config)
    random_result = generate_uniform_digit_candidates(
        stats, rule, config=config, seed=11
    )
    expected_defensive = min(3, int(count * 0.15))

    for result in (statistical, random_result):
        defensive = sum(
            candidate.shape in {"三一一", "三二"} for candidate in result.candidates
        )
        assert defensive == expected_defensive
        assert len(result.candidates) == count


def test_three_digit_betting_candidates_aggregate_unique_group_probability_mass():
    rule = get_lottery_rule("pl3")
    df = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 4,
                "十位": (index + 1) % 4,
                "个位": (index + 2) % 4,
            }
            for index in range(60)
        ]
    )
    stats = analyze_digit_history(df, rule)

    plan = generate_digit_betting_candidates(
        stats, rule, config=DigitCandidateConfig(count=20), group_count=10
    )

    assert plan.candidates == plan.direct_candidates
    assert len({candidate.group_key for candidate in plan.group_candidates}) == len(
        plan.group_candidates
    )
    assert all(
        candidate.shape in {"组六", "组三"} for candidate in plan.group_candidates
    )
    assert sum(candidate.shape == "组三" for candidate in plan.group_candidates) <= 2
    assert all(candidate.probability_mass > 0 for candidate in plan.group_candidates)
    group_payload = plan.group_candidates[0].to_dict()
    assert group_payload["compositeModelWeight"] == group_payload["probabilityMass"]
    assert group_payload["probabilityMassDeprecated"] is True

    pl5_rule = get_lottery_rule("pl5")
    pl5_df = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "万位": index % 10,
                "千位": (index + 1) % 10,
                "百位": (index + 2) % 10,
                "十位": (index + 3) % 10,
                "个位": (index + 4) % 10,
            }
            for index in range(30)
        ]
    )
    pl5_plan = generate_digit_betting_candidates(
        analyze_digit_history(pl5_df, pl5_rule),
        pl5_rule,
        config=DigitCandidateConfig(count=5),
    )
    assert pl5_plan.group_candidates == []


def test_pl3_and_pl5_share_identical_three_digit_prefix_score():
    rows = [
        {
            "期数": str(2026001 + index),
            "a": index % 10,
            "b": (index * 3 + 1) % 10,
            "c": (index * 7 + 2) % 10,
            "d": (index + 4) % 10,
            "e": (index + 8) % 10,
        }
        for index in range(80)
    ]
    pl3_rule = get_lottery_rule("pl3")
    pl5_rule = get_lottery_rule("pl5")
    pl3_df = pd.DataFrame(
        [
            {"期数": row["期数"], "百位": row["a"], "十位": row["b"], "个位": row["c"]}
            for row in rows
        ]
    )
    pl5_df = pd.DataFrame(
        [
            {
                "期数": row["期数"],
                "万位": row["a"],
                "千位": row["b"],
                "百位": row["c"],
                "十位": row["d"],
                "个位": row["e"],
            }
            for row in rows
        ]
    )
    config = DigitCandidateConfig(
        frequency_windows=(20, 50), frequency_window_weights=(0.6, 0.4)
    )
    pl3_stats = analyze_digit_history(
        pl3_df, pl3_rule, frequency_windows=config.frequency_windows
    )
    pl5_stats = analyze_digit_history(
        pl5_df, pl5_rule, frequency_windows=config.frequency_windows
    )

    candidates = ([1, 4, 9], [3, 0, 7], [8, 5, 2])
    pl3_scores = [
        score_digit_prefix(pl3_stats, numbers, config) for numbers in candidates
    ]
    pl5_scores = [
        score_digit_prefix(pl5_stats, numbers, config) for numbers in candidates
    ]

    assert pl3_scores == pytest.approx(pl5_scores)
    assert sorted(range(3), key=pl3_scores.__getitem__, reverse=True) == sorted(
        range(3), key=pl5_scores.__getitem__, reverse=True
    )


def test_candidate_config_validates_joint_weights_and_window_lengths():
    with pytest.raises(ValueError, match="窗口"):
        DigitCandidateConfig(
            frequency_windows=(20,), frequency_window_weights=(0.5, 0.5)
        )
    with pytest.raises(ValueError, match="权重"):
        DigitCandidateConfig(marginal_weight=-1.0)


def test_ensemble_ranking_exposes_independent_model_votes():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 4,
                "十位": (index * 3 + 1) % 7,
                "个位": (index * 7 + 2) % 10,
            }
            for index in range(80)
        ]
    )
    stats = analyze_digit_history(df, rule)

    result = generate_digit_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=10, ranking_mode="ensemble"),
    )
    payload = result.to_dict()

    assert payload["config"]["rankingMode"] == "ensemble"
    assert set(payload["config"]["ensembleModelWeights"]) == {
        "position",
        "pair",
        "shape",
        "sum",
        "span",
        "parity",
        "bigSmall",
        "primeComposite",
        "consecutive",
        "mirror",
        "sumTail",
        "latestDistance",
        "repeatLatest",
        "omission",
        "monteCarlo",
        "mlRanker",
    }
    for candidate in result.candidates:
        assert 0.0 <= candidate.ensemble_score <= 1.0
        assert len(candidate.model_rank_percentiles) == 16
        assert all(0.0 <= value <= 1.0 for value in candidate.model_rank_percentiles)
        serialized = candidate.to_dict()
        assert serialized["ensembleScore"] == candidate.ensemble_score
        assert set(serialized["modelRankPercentiles"]) == set(
            payload["config"]["ensembleModelWeights"]
        )
        assert 0 <= serialized["topDecileVotes"] <= 16
    assert set(result.model_candidates) == set(ENSEMBLE_MODEL_NAMES[:-2])
    assert all(len(values) == 10 for values in result.model_candidates.values())


def test_hard_structure_constraints_change_the_candidate_space():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": str(2026001 + index), "百位": 1, "十位": 2, "个位": 3}
            for index in range(60)
        ]
    )
    stats = analyze_digit_history(df, rule, frequency_windows=(30, 50))
    config = DigitCandidateConfig(
        count=5,
        exclude_latest=False,
        ranking_mode="ensemble",
        constraint_mode="hard",
        constraint_probability_floor=0.1,
        frequency_windows=(30, 50),
        frequency_window_weights=(0.6, 0.4),
    )

    result = generate_digit_candidates(stats, rule, config=config)

    assert all(candidate.constraint_penalty == 0.0 for candidate in result.candidates)
    assert all(
        sum(value % 2 for value in candidate.numbers) == 2
        for candidate in result.candidates
    )
    assert all(
        sum(value >= 5 for value in candidate.numbers) == 0
        for candidate in result.candidates
    )


def test_group_candidates_use_shape_specific_ranking_model():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 10,
                "十位": (index * 3 + 1) % 10,
                "个位": (index * 7 + 2) % 10,
            }
            for index in range(80)
        ]
    )

    plan = generate_digit_betting_candidates(
        analyze_digit_history(df, rule),
        rule,
        config=DigitCandidateConfig(count=10, ranking_mode="ensemble"),
        group_count=10,
    )

    assert all(
        candidate.ranking_model == "shape_specific_ensemble"
        for candidate in plan.group_candidates
    )
    assert all(
        len(candidate.model_rank_percentiles) == 16
        for candidate in plan.group_candidates
    )


def test_candidate_config_rejects_invalid_ensemble_configuration():
    with pytest.raises(ValueError, match="ranking_mode"):
        DigitCandidateConfig(ranking_mode="unknown")
    with pytest.raises(ValueError, match="集成模型权重"):
        DigitCandidateConfig(ensemble_model_weights=(1.0,))
    with pytest.raises(ValueError, match="集成模型权重"):
        DigitCandidateConfig(ensemble_model_weights=(1.0,) * 15 + (-1.0,))
    with pytest.raises(ValueError, match="constraint_mode"):
        DigitCandidateConfig(constraint_mode="unknown")


def test_uniform_three_digit_betting_plan_has_unique_group_keys_and_same_quota():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 10,
                "十位": (index + 3) % 10,
                "个位": (index + 6) % 10,
            }
            for index in range(30)
        ]
    )
    stats = analyze_digit_history(df, rule)

    plan = generate_uniform_digit_betting_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=10),
        group_count=10,
        seed=7,
    )

    assert len(plan.direct_candidates) == 10
    assert len(plan.group_candidates) == 10
    assert len({candidate.group_key for candidate in plan.group_candidates}) == 10
    assert sum(candidate.shape == "组三" for candidate in plan.group_candidates) <= 2
    assert all(
        candidate.shape in {"组六", "组三"} for candidate in plan.group_candidates
    )

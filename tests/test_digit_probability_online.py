# -*- coding: utf-8 -*-

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.digit_probability_online import (
    DigitOnlineProbabilityConfig,
    build_digit_online_probability_markdown,
    build_digit_online_probability_plan,
    run_digit_online_probability_walk_forward,
    update_online_weights,
    write_digit_online_probability_reports,
)
from src.lotteries import get_lottery_rule


def _history(periods: int = 25) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 10,
                "十位": (index * 3 + 1) % 10,
                "个位": (index * 7 + 2) % 10,
            }
            for index in range(periods)
        ]
    )


def test_online_weight_update_rewards_better_probability_and_stays_normalized():
    config = DigitOnlineProbabilityConfig(
        min_train_size=10,
        model_profiles=("position", "sum"),
        uniform_prior_weight=0.5,
        fixed_share=0.01,
    )
    updated = update_online_weights(
        config.prior_weights(),
        np.asarray([0.001, 0.01, 0.0001]),
        config,
    )

    assert float(updated.sum()) == pytest.approx(1.0)
    assert np.all(updated > 0)
    assert updated[1] > updated[0] > updated[2]


def test_online_walk_forward_predicts_before_feedback_and_tracks_weights(tmp_path):
    rule = get_lottery_rule("fc3d")
    original = _history()
    changed = original.copy()
    changed.loc[changed.index[-5:], ["百位", "十位", "个位"]] = [9, 9, 9]
    config = DigitOnlineProbabilityConfig(
        min_train_size=10,
        model_profiles=("position", "sum", "latestDistance"),
    )

    first = run_digit_online_probability_walk_forward(
        original,
        rule,
        periods=5,
        candidate_count=10,
        online_config=config,
    )
    second = run_digit_online_probability_walk_forward(
        changed,
        rule,
        periods=5,
        candidate_count=10,
        online_config=config,
    )

    assert first.pretraining_feedback_periods == 10
    assert len(first.issues) == 5
    assert first.issues[0].direct_candidates == second.issues[0].direct_candidates
    assert first.issues[0].weights_before == second.issues[0].weights_before
    assert first.issues[0].weights_after != second.issues[0].weights_after
    assert all(issue.train_end_issue < issue.issue for issue in first.issues)
    assert all(
        sum(issue.weights_before.values()) == pytest.approx(1.0)
        and sum(issue.weights_after.values()) == pytest.approx(1.0)
        for issue in first.issues
    )
    for previous, current in zip(first.issues, first.issues[1:]):
        assert current.weights_before == previous.weights_after

    markdown = build_digit_online_probability_markdown(first)
    markdown_path, json_path = write_digit_online_probability_reports(
        first, tmp_path, prefix="online"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "先预测、后开奖、再更新" in markdown
    assert markdown_path.exists()
    assert payload["experimentModel"] == "digit_probability_online_v3"
    assert payload["periods"] == 5
    assert payload["issues"][0]["weightsBefore"]
    assert payload["issues"][0]["weightsAfter"]
    assert payload["uniformLogLoss"] > 0


def test_online_daily_state_updates_only_appended_draw_and_matches_full_replay(
    tmp_path,
):
    rule = get_lottery_rule("fc3d")
    first_history = _history(20)
    appended_history = _history(21)
    config = DigitOnlineProbabilityConfig(
        min_train_size=10,
        model_profiles=("position", "sum", "latestDistance"),
    )
    state_path = tmp_path / "state.json"

    first = build_digit_online_probability_plan(
        first_history,
        rule,
        candidate_count=10,
        online_config=config,
        state_path=state_path,
    )
    incremental = build_digit_online_probability_plan(
        appended_history,
        rule,
        candidate_count=10,
        online_config=config,
        state_path=state_path,
    )
    rebuilt = build_digit_online_probability_plan(
        appended_history,
        rule,
        candidate_count=10,
        online_config=config,
        state_path=tmp_path / "rebuilt.json",
    )

    assert first.state_update.mode == "full_rebuild"
    assert first.state.feedback_periods == 10
    assert incremental.state_update.mode == "incremental"
    assert incremental.state_update.feedback_updates == 1
    assert incremental.state.processed_periods == 21
    assert incremental.state.weights == rebuilt.state.weights
    assert incremental.distribution.probabilities == rebuilt.distribution.probabilities
    assert [item.text for item in incremental.direct_candidates] == [
        item.text for item in rebuilt.direct_candidates
    ]
    assert incremental.distribution.probability_sum == pytest.approx(1.0)


def test_online_daily_state_rebuilds_after_historical_correction(tmp_path):
    rule = get_lottery_rule("fc3d")
    history = _history(20)
    config = DigitOnlineProbabilityConfig(
        min_train_size=10,
        model_profiles=("position", "sum"),
    )
    state_path = tmp_path / "state.json"
    build_digit_online_probability_plan(
        history, rule, online_config=config, state_path=state_path
    )
    corrected = history.copy()
    corrected.loc[corrected.index[5], ["百位", "十位", "个位"]] = [9, 9, 9]

    plan = build_digit_online_probability_plan(
        corrected, rule, online_config=config, state_path=state_path
    )

    assert plan.state_update.mode == "full_rebuild"
    assert plan.state_update.rebuild_reason == "history_changed"
    payload = plan.to_dict()
    assert payload["probabilityModel"] == "online_expert_mixture_v3"
    assert payload["onlineProbability"]["state"]["latestIssue"] == "2026020"
    assert payload["config"]["rankingMode"] == "online_probability"

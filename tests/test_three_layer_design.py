import numpy as np
import pandas as pd
import pytest

from src.analysis.digit_baselines import (
    build_baseline_suite,
    position_frequency_baseline,
    shape_transition_baseline,
    uniform_baseline,
)
from src.analysis.digit_evaluation import evaluate_probability_history
from src.analysis.digit_raw_evidence import (
    JsonlDigitHistoryProvider,
    RawDigitRecord,
    append_raw_digit_jsonl,
    build_reconciliation_report,
    collect_provider_records,
    read_raw_digit_jsonl,
    reconcile_raw_digit_records,
)
from src.analysis.digit_strategy_gate import (
    StrategyEvidence,
    StrategyStatus,
    decide_strategy_status,
)
from src.analysis.digit_strategy_registry import (
    StrategyRegistryUpdate,
    update_strategy_registry,
)
from src.analysis.digit_three_layer import (
    THREE_LAYER_SCHEMA_VERSION,
    three_layer_source_fingerprint,
)
from src.lotteries import get_lottery_rule


def _history(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "期数": [str(2026000 + index) for index in range(rows)],
            "百位": [index % 10 for index in range(rows)],
            "十位": [(index + 1) % 10 for index in range(rows)],
            "个位": [(index + 2) % 10 for index in range(rows)],
            "开奖日期": ["2026-01-01"] * rows,
            "数据来源": ["test"] * rows,
        }
    )


def test_baselines_are_valid_1000_candidate_distributions():
    rule = get_lottery_rule("fc3d")
    history = _history()
    for prediction in (
        uniform_baseline(),
        position_frequency_baseline(history, rule),
        shape_transition_baseline(history, rule),
    ):
        probabilities = np.asarray(prediction.probabilities)
        assert probabilities.shape == (1000,)
        assert np.isfinite(probabilities).all()
        assert probabilities.sum() == pytest.approx(1.0)


def test_fixed_baseline_suite_is_complete():
    suite = build_baseline_suite(_history(180), get_lottery_rule("pl3"))
    assert set(suite) == {
        "uniform",
        "shape_prior",
        "shape_transition_150",
        "sum_span_150",
        "position_20",
        "position_50",
        "position_150",
        "position_all",
    }
    assert all(len(item.probabilities) == 1000 for item in suite.values())


def test_uniform_evaluation_matches_theoretical_reference():
    probabilities = np.full((3, 1000), 0.001)
    result = evaluate_probability_history(probabilities, [0, 1, 2], top_ks=(10,))
    assert result.mean_log_loss == pytest.approx(np.log(1000))
    assert result.mean_brier_score == pytest.approx(0.999)
    assert result.mean_rank == pytest.approx(500.5)
    assert result.top_k_hit_rates[10] == 0.0


def test_raw_jsonl_deduplicates_and_reconcile_requires_two_sources(tmp_path):
    records = [
        RawDigitRecord(
            "fc3d",
            "2026001",
            "2026-01-01",
            (1, 2, 3),
            "a",
            "https://a",
            "now",
            {"x": 1},
        ),
        RawDigitRecord(
            "fc3d",
            "2026001",
            "2026-01-01",
            (1, 2, 3),
            "b",
            "https://b",
            "now",
            {"x": 2},
        ),
    ]
    path = append_raw_digit_jsonl(tmp_path / "raw.jsonl", records)
    repeated = RawDigitRecord(
        "fc3d",
        "2026001",
        "2026-01-01",
        (1, 2, 3),
        "a",
        "https://a",
        "later",
        {"x": 1},
    )
    append_raw_digit_jsonl(path, [repeated])
    assert len(read_raw_digit_jsonl(path)) == 2
    draws = reconcile_raw_digit_records(records)
    assert len(draws) == 1
    assert draws[0].number_text == "123"


def test_raw_reconcile_blocks_cross_source_conflict(tmp_path):
    records = [
        RawDigitRecord(
            "pl3", "26001", "2026-01-01", (1, 2, 3), "a", "https://a", "now", {}
        ),
        RawDigitRecord(
            "pl3", "26001", "2026-01-01", (1, 2, 4), "b", "https://b", "now", {}
        ),
    ]
    with pytest.raises(ValueError, match="冲突"):
        reconcile_raw_digit_records(records)


def test_jsonl_providers_merge_and_report_conflicts(tmp_path):
    first = RawDigitRecord(
        "fc3d", "2026001", "2026-01-01", (1, 2, 3), "a", "https://a", "now", {}
    )
    second = RawDigitRecord(
        "fc3d", "2026001", "2026-01-01", (1, 2, 3), "b", "https://b", "now", {}
    )
    first_path = append_raw_digit_jsonl(tmp_path / "a.jsonl", [first])
    second_path = append_raw_digit_jsonl(tmp_path / "b.jsonl", [second])
    merged = collect_provider_records(
        (
            JsonlDigitHistoryProvider("a", first_path),
            JsonlDigitHistoryProvider("b", second_path),
        )
    )
    report = build_reconciliation_report(merged)
    assert report["passed"] is True
    assert report["acceptedIssues"] == ["2026001"]

    conflict = RawDigitRecord(
        "fc3d", "2026001", "2026-01-01", (9, 9, 9), "b", "https://b", "now", {}
    )
    conflict_report = build_reconciliation_report((first, conflict))
    assert conflict_report["passed"] is False
    assert conflict_report["conflicts"][0]["issue"] == "2026001"


def test_strategy_registry_tracks_demotion_retirement_and_rollback(tmp_path):
    path = tmp_path / "registry.json"

    def update(status, occurred_at, rollback_to=None):
        return update_strategy_registry(
            path,
            StrategyRegistryUpdate(
                strategy_id="exp:fc3d:direct",
                lottery="fc3d",
                output_kind="direct",
                requested_status=status,
                reasons=("test",),
                data_fingerprint="d" * 64,
                params_fingerprint="p" * 64,
                source_fingerprint="s" * 64,
                occurred_at=occurred_at,
                rollback_to=rollback_to,
            ),
        )

    update(StrategyStatus.ACTIVE, "2026-01-01")
    demoted = update(StrategyStatus.RESEARCH, "2026-01-02", "previous-stable")
    retired = update(StrategyStatus.RESEARCH, "2026-01-03", "previous-stable")

    assert demoted["entries"]["exp:fc3d:direct"]["status"] == "demoted"
    assert retired["entries"]["exp:fc3d:direct"]["status"] == "retired"
    assert [item["to"] for item in retired["transitions"]] == [
        "active",
        "demoted",
        "retired",
    ]
    assert retired["transitions"][-1]["rollbackTo"] == "previous-stable"


def test_three_layer_source_fingerprint_is_stable_and_versioned():
    first = three_layer_source_fingerprint()
    second = three_layer_source_fingerprint()
    assert THREE_LAYER_SCHEMA_VERSION == 1
    assert first == second
    assert len(first) == 64


def test_strategy_gate_rejects_uncalibrated_evidence():
    evidence = StrategyEvidence(
        search_lift=1.2,
        validation_lift=1.1,
        stable_validation_blocks=2,
        validation_blocks=3,
        mean_log_loss=6.8,
        uniform_log_loss=6.9,
        mean_brier_score=0.8,
        uniform_brier_score=0.999,
        expected_calibration_error=0.2,
    )
    decision = decide_strategy_status(evidence)
    assert "ECE超过固定阈值" in decision.reasons


def test_strategy_gate_requires_frozen_before_active():
    evidence = StrategyEvidence(
        search_lift=1.2,
        validation_lift=1.1,
        stable_validation_blocks=2,
        validation_blocks=3,
        mean_log_loss=6.8,
        uniform_log_loss=6.9,
        mean_brier_score=0.8,
        uniform_brier_score=0.999,
    )
    decision = decide_strategy_status(evidence)
    assert decision.status is StrategyStatus.OBSERVATION
    assert decision.admitted is False


def test_strategy_gate_can_activate_only_after_frozen_passes():
    evidence = StrategyEvidence(
        search_lift=1.2,
        validation_lift=1.1,
        stable_validation_blocks=2,
        validation_blocks=3,
        mean_log_loss=6.8,
        uniform_log_loss=6.9,
        mean_brier_score=0.8,
        uniform_brier_score=0.999,
        frozen_test_evaluated=True,
        frozen_test_lift=1.05,
    )
    decision = decide_strategy_status(evidence)
    assert decision.status is StrategyStatus.ACTIVE
    assert decision.admitted is True

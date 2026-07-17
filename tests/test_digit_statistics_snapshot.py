# -*- coding: utf-8 -*-
"""数字彩持久化增量统计快照测试。"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.digit_statistics import analyze_digit_history
from src.analysis.digit_statistics_snapshot import analyze_digit_history_with_snapshot
from src.lotteries import get_lottery_rule


def _history(draw_count: int, size: int, *, start_issue: int = 2026001) -> pd.DataFrame:
    columns = (
        ["百位", "十位", "个位"]
        if draw_count == 3
        else ["万位", "千位", "百位", "十位", "个位"]
    )
    rows = []
    for index in range(size):
        row = {"期数": str(start_issue + index)}
        for position, column in enumerate(columns):
            row[column] = (index * (position + 2) + position) % 10
        rows.append(row)
    return pd.DataFrame(list(reversed(rows)))


def _assert_equivalent(
    actual,
    df: pd.DataFrame,
    lottery: str,
    windows: tuple[int, ...],
    prior: float = 2.0,
):
    expected = analyze_digit_history(
        df,
        get_lottery_rule(lottery),
        frequency_windows=windows,
        bayesian_prior_strength=prior,
    )
    assert actual.to_dict() == expected.to_dict()


def _write_newer_snapshot_while_holding_lock(
    path: str,
    lock_acquired_event,
    stale_identity_observed_event,
    result_queue,
) -> None:
    import src.analysis.digit_statistics_snapshot as snapshot_module

    original_process_lock = snapshot_module._process_lock

    @contextmanager
    def coordinated_process_lock(snapshot_path):
        with original_process_lock(snapshot_path):
            lock_acquired_event.set()
            if not stale_identity_observed_event.wait(timeout=5):
                raise TimeoutError("旧进程未在新快照写入前读取文件身份")
            yield

    snapshot_module._process_lock = coordinated_process_lock
    _, metadata = analyze_digit_history_with_snapshot(
        _history(3, 42),
        get_lottery_rule("fc3d"),
        path,
    )
    result_queue.put(("new", metadata.to_dict()))


def _read_stale_view_then_wait_for_newer_snapshot(
    path: str,
    lock_acquired_event,
    stale_identity_observed_event,
    result_queue,
    kwargs,
) -> None:
    import src.analysis.digit_statistics_snapshot as snapshot_module

    if not lock_acquired_event.wait(timeout=5):
        raise TimeoutError("新进程未先取得快照锁")
    original_file_identity = snapshot_module._file_identity
    identity_observed = False

    def observed_file_identity(snapshot_path):
        nonlocal identity_observed
        identity = original_file_identity(snapshot_path)
        if not identity_observed:
            identity_observed = True
            stale_identity_observed_event.set()
        return identity

    snapshot_module._file_identity = observed_file_identity
    _, metadata = analyze_digit_history_with_snapshot(
        _history(3, 41),
        get_lottery_rule("fc3d"),
        path,
        **kwargs,
    )
    result_queue.put(("stale", metadata.to_dict()))


def test_snapshot_first_full_build_then_cache_hit_only_processes_new_rows(
    tmp_path: Path,
):
    rule = get_lottery_rule("fc3d")
    df = _history(3, 40)
    path = tmp_path / "fc3d.json"

    first, first_meta = analyze_digit_history_with_snapshot(df, rule, path)
    second, second_meta = analyze_digit_history_with_snapshot(df, rule, path)

    _assert_equivalent(first, df, "fc3d", (30, 50, 100, 300))
    assert second.to_dict() == first.to_dict()
    assert first_meta.to_dict()["mode"] == "full_rebuild"
    assert first_meta.to_dict()["processedRows"] == 40
    assert first_meta.persisted is True
    assert first_meta.snapshot_written is True
    assert second_meta.to_dict() == {
        "mode": "cache_hit",
        "addedIssues": 0,
        "processedRows": 0,
        "rebuildReason": None,
        "requestedRebuildReason": None,
        "snapshotPath": str(path),
        "persisted": True,
        "snapshotWritten": False,
    }


@pytest.mark.parametrize("added", [1, 4])
def test_snapshot_incremental_append_is_deeply_equivalent(tmp_path: Path, added: int):
    rule = get_lottery_rule("fc3d")
    path = tmp_path / "fc3d.json"
    initial = _history(3, 35)
    analyze_digit_history_with_snapshot(initial, rule, path, frequency_windows=(10, 30))
    complete = _history(3, 35 + added)

    result, metadata = analyze_digit_history_with_snapshot(
        complete,
        rule,
        path,
        frequency_windows=(10, 30),
    )

    _assert_equivalent(result, complete, "fc3d", (10, 30))
    assert metadata.mode == "incremental"
    assert metadata.added_issues == added
    assert metadata.processed_rows == added
    assert metadata.persisted is True
    assert metadata.snapshot_written is True


def test_snapshot_window_eviction_omission_and_transition_boundaries(tmp_path: Path):
    rule = get_lottery_rule("fc3d")
    path = tmp_path / "fc3d.json"
    initial = pd.DataFrame(
        [
            {"期数": "4", "百位": 4, "十位": 4, "个位": 4},
            {"期数": "3", "百位": 3, "十位": 3, "个位": 3},
            {"期数": "2", "百位": 2, "十位": 2, "个位": 2},
            {"期数": "1", "百位": 1, "十位": 1, "个位": 1},
        ]
    )
    analyze_digit_history_with_snapshot(initial, rule, path, frequency_windows=(3,))
    complete = pd.concat(
        [
            pd.DataFrame([{"期数": "5", "百位": 5, "十位": 3, "个位": 3}]),
            initial,
        ],
        ignore_index=True,
    )

    result, metadata = analyze_digit_history_with_snapshot(
        complete, rule, path, frequency_windows=(3,)
    )

    _assert_equivalent(result, complete, "fc3d", (3,))
    assert result.position_frequency_windows[3]["百位"] == {3: 1, 4: 1, 5: 1}
    assert result.omission_windows[3]["百位"][2] == 3
    assert result.current_omission["百位"][1] == 4
    assert metadata.processed_rows == 1


def test_snapshot_supports_pl5_and_all_history_without_unbounded_recent_queue(
    tmp_path: Path,
):
    rule = get_lottery_rule("pl5")
    df = _history(5, 420)
    path = tmp_path / "pl5.json"
    windows = (10, 30, 100, len(df))

    result, _ = analyze_digit_history_with_snapshot(
        df,
        rule,
        path,
        frequency_windows=windows,
        all_history_window=True,
    )

    _assert_equivalent(result, df, "pl5", windows)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["recentRows"]) == 300
    assert str(len(df)) not in payload["windowStates"]


def test_all_history_dynamic_window_remains_incremental_when_total_changes(
    tmp_path: Path,
):
    rule = get_lottery_rule("fc3d")
    path = tmp_path / "fc3d.json"
    initial = _history(3, 35)
    analyze_digit_history_with_snapshot(
        initial,
        rule,
        path,
        frequency_windows=(10, 30, len(initial)),
        all_history_window=True,
    )
    complete = _history(3, 36)

    result, metadata = analyze_digit_history_with_snapshot(
        complete,
        rule,
        path,
        frequency_windows=(10, 30, len(complete)),
        all_history_window=True,
    )

    _assert_equivalent(result, complete, "fc3d", (10, 30, len(complete)))
    assert metadata.mode == "incremental"
    assert metadata.processed_rows == 1


@pytest.mark.parametrize(
    ("fixed_windows", "sizes"),
    [
        ((10, 30), (9, 10, 11)),
        ((10, 30), (29, 30, 31)),
        ((10, 30, 50, 100, 300), (299, 300, 301)),
        ((17,), (16, 17, 18)),
    ],
)
def test_all_history_window_collisions_keep_incremental_signature(
    tmp_path: Path,
    fixed_windows: tuple[int, ...],
    sizes: tuple[int, int, int],
):
    rule = get_lottery_rule("fc3d")
    path = tmp_path / "fc3d.json"

    for index, size in enumerate(sizes):
        windows = fixed_windows if size in fixed_windows else (*fixed_windows, size)
        result, metadata = analyze_digit_history_with_snapshot(
            _history(3, size),
            rule,
            path,
            frequency_windows=windows,
            all_history_window=True,
        )

        _assert_equivalent(result, _history(3, size), "fc3d", windows)
        if index:
            assert metadata.mode == "incremental"
            assert metadata.processed_rows == 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["windowSignature"] == [*fixed_windows, "allHistory"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("corrupt", "corrupt_json"),
        ("truncate", "history_truncated"),
        ("correct", "historical_prefix_changed"),
        ("non_append", "non_append_issue"),
        ("schema", "schema_version_mismatch"),
        ("engine", "engine_version_mismatch"),
        ("rule", "rule_signature_mismatch"),
        ("windows", "window_config_mismatch"),
        ("prior", "prior_strength_mismatch"),
    ],
)
def test_snapshot_invalidations_force_full_rebuild(
    tmp_path: Path, mutation: str, reason: str
):
    rule = get_lottery_rule("fc3d")
    path = tmp_path / "fc3d.json"
    df = _history(3, 20)
    analyze_digit_history_with_snapshot(df, rule, path, frequency_windows=(10, 30))
    current = df.copy()
    kwargs = {"frequency_windows": (10, 30), "bayesian_prior_strength": 2.0}

    if mutation == "corrupt":
        path.write_text("{broken", encoding="utf-8")
    elif mutation == "truncate":
        current = current.iloc[1:].copy()
    elif mutation == "correct":
        current.loc[current.index[-1], "百位"] = 9
    elif mutation == "non_append":
        inserted = pd.DataFrame([{"期数": "2026000", "百位": 1, "十位": 2, "个位": 3}])
        current = pd.concat([current, inserted], ignore_index=True)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "schema":
            payload["schemaVersion"] = -1
        elif mutation == "engine":
            payload["engineVersion"] = "old"
        elif mutation == "rule":
            payload["ruleSignature"] = "other"
        elif mutation == "windows":
            kwargs["frequency_windows"] = (10, 50)
        elif mutation == "prior":
            kwargs["bayesian_prior_strength"] = 3.0
        if mutation not in {"windows", "prior"}:
            path.write_text(json.dumps(payload), encoding="utf-8")

    result, metadata = analyze_digit_history_with_snapshot(
        current, rule, path, **kwargs
    )

    _assert_equivalent(
        result,
        current,
        "fc3d",
        kwargs["frequency_windows"],
        kwargs["bayesian_prior_strength"],
    )
    assert metadata.mode == "full_rebuild"
    assert metadata.rebuild_reason == reason
    assert metadata.processed_rows == len(current)


def test_snapshot_explicit_rebuild_forces_full_processing(tmp_path: Path):
    rule = get_lottery_rule("fc3d")
    df = _history(3, 12)
    path = tmp_path / "fc3d.json"
    analyze_digit_history_with_snapshot(df, rule, path)

    _, metadata = analyze_digit_history_with_snapshot(df, rule, path, rebuild=True)

    assert metadata.mode == "full_rebuild"
    assert metadata.rebuild_reason == "explicit_rebuild"
    assert metadata.processed_rows == len(df)


def test_incremental_path_does_not_call_full_history_analyzer(
    tmp_path: Path, monkeypatch
):
    import src.analysis.digit_statistics as statistics_module

    rule = get_lottery_rule("fc3d")
    path = tmp_path / "fc3d.json"
    analyze_digit_history_with_snapshot(_history(3, 20), rule, path)

    def fail_full_analyzer(*args, **kwargs):
        raise AssertionError("增量路径不得调用全量 analyze_digit_history")

    monkeypatch.setattr(statistics_module, "analyze_digit_history", fail_full_analyzer)
    _, metadata = analyze_digit_history_with_snapshot(_history(3, 21), rule, path)

    assert metadata.mode == "incremental"
    assert metadata.processed_rows == 1


def test_cache_hit_and_append_use_counter_probability_helper_without_expansion(
    tmp_path: Path, monkeypatch
):
    import src.analysis.digit_statistics_snapshot as snapshot_module

    rule = get_lottery_rule("fc3d")
    path = tmp_path / "fc3d.json"
    analyze_digit_history_with_snapshot(_history(3, 120), rule, path)
    calls: list[tuple[int, int, int]] = []
    original = snapshot_module._smoothed_probabilities_from_counter

    def record_counter(counter, sample_count, domain, prior_strength):
        calls.append((sum(counter.values()), sample_count, len(counter)))
        return original(counter, sample_count, domain, prior_strength)

    def fail_materialized_values(*args, **kwargs):
        raise AssertionError("快照概率不得将 Counter 展开为样本列表")

    monkeypatch.setattr(
        snapshot_module,
        "_smoothed_probabilities_from_counter",
        record_counter,
    )
    monkeypatch.setattr(
        snapshot_module,
        "_smoothed_probabilities",
        fail_materialized_values,
        raising=False,
    )

    cached, cache_metadata = analyze_digit_history_with_snapshot(
        _history(3, 120), rule, path
    )
    appended, append_metadata = analyze_digit_history_with_snapshot(
        _history(3, 121), rule, path
    )

    _assert_equivalent(cached, _history(3, 120), "fc3d", (30, 50, 100, 300))
    _assert_equivalent(appended, _history(3, 121), "fc3d", (30, 50, 100, 300))
    assert cache_metadata.mode == "cache_hit"
    assert append_metadata.mode == "incremental"
    assert calls
    assert all(
        counter_total == sample_count for counter_total, sample_count, _ in calls
    )
    assert max(distinct_count for _, _, distinct_count in calls) <= 100


@pytest.mark.parametrize("lottery", ["fc3d", "pl5"])
def test_empty_snapshot_result_matches_full_analysis(tmp_path: Path, lottery: str):
    rule = get_lottery_rule(lottery)
    empty = pd.DataFrame(columns=["期数", *rule.number_columns])

    result, _ = analyze_digit_history_with_snapshot(
        empty,
        rule,
        tmp_path / f"{lottery}.json",
    )

    assert result.current_omission == {}
    assert result.to_dict() == analyze_digit_history(empty, rule).to_dict()


def test_same_process_concurrent_snapshot_updates_are_atomic(tmp_path: Path):
    rule = get_lottery_rule("fc3d")
    path = tmp_path / "fc3d.json"
    df = _history(3, 50)

    def run_once():
        return analyze_digit_history_with_snapshot(df, rule, path)[1].mode

    with ThreadPoolExecutor(max_workers=2) as executor:
        modes = list(executor.map(lambda _: run_once(), range(2)))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(modes) == ["cache_hit", "full_rebuild"]
    assert payload["processedIssues"] == len(df)
    assert not list(tmp_path.glob(".fc3d.json.*.tmp"))


@pytest.mark.parametrize(
    ("stale_kwargs", "requested_rebuild_reason"),
    [
        ({}, None),
        (
            {
                "frequency_windows": (10, 20),
                "fixed_frequency_windows": (10, 20),
            },
            "window_config_mismatch",
        ),
        ({"bayesian_prior_strength": 3.0}, "prior_strength_mismatch"),
        ({"rebuild": True}, "explicit_rebuild"),
    ],
    ids=("same_config", "different_windows", "different_prior", "explicit_rebuild"),
)
def test_cross_process_stale_view_never_overwrites_newer_snapshot(
    tmp_path: Path, stale_kwargs, requested_rebuild_reason
):
    context = multiprocessing.get_context("fork")
    path = tmp_path / "fc3d.json"
    rule = get_lottery_rule("fc3d")
    analyze_digit_history_with_snapshot(_history(3, 40), rule, path)
    lock_acquired_event = context.Event()
    stale_identity_observed_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            name="new-view",
            target=_write_newer_snapshot_while_holding_lock,
            args=(
                str(path),
                lock_acquired_event,
                stale_identity_observed_event,
                result_queue,
            ),
        ),
        context.Process(
            name="old-view",
            target=_read_stale_view_then_wait_for_newer_snapshot,
            args=(
                str(path),
                lock_acquired_event,
                stale_identity_observed_event,
                result_queue,
                stale_kwargs,
            ),
        ),
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert not process.is_alive(), "跨进程快照测试发生死锁"
        assert process.exitcode == 0
    updates = dict(result_queue.get(timeout=2) for _ in processes)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["processedIssues"] == 42
    final, metadata = analyze_digit_history_with_snapshot(_history(3, 42), rule, path)
    _assert_equivalent(final, _history(3, 42), "fc3d", (30, 50, 100, 300))
    assert metadata.mode == "cache_hit"
    assert updates["new"]["mode"] == "incremental"
    assert updates["stale"] == {
        "mode": "stale_view",
        "addedIssues": 0,
        "processedRows": 41,
        "rebuildReason": "stale_view_not_persisted",
        "requestedRebuildReason": requested_rebuild_reason,
        "snapshotPath": str(path),
        "persisted": False,
        "snapshotWritten": False,
    }


def test_stale_window_mismatch_does_not_write_snapshot(tmp_path: Path, monkeypatch):
    import src.analysis.digit_statistics_snapshot as snapshot_module

    path = tmp_path / "fc3d.json"
    rule = get_lottery_rule("fc3d")
    analyze_digit_history_with_snapshot(_history(3, 42), rule, path)
    original_file_identity = snapshot_module._file_identity
    identity_calls = 0

    def changed_file_identity(snapshot_path):
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls == 1:
            return (-1, -1, -1)
        return original_file_identity(snapshot_path)

    def fail_atomic_write(*args, **kwargs):
        raise AssertionError("陈旧短视图不得写入快照")

    monkeypatch.setattr(snapshot_module, "_file_identity", changed_file_identity)
    monkeypatch.setattr(snapshot_module, "_atomic_write_json", fail_atomic_write)

    _, metadata = analyze_digit_history_with_snapshot(
        _history(3, 41),
        rule,
        path,
        frequency_windows=(10, 20),
        fixed_frequency_windows=(10, 20),
    )

    assert metadata.mode == "stale_view"
    assert metadata.processed_rows == 41
    assert metadata.rebuild_reason == "stale_view_not_persisted"
    assert metadata.persisted is False
    assert metadata.snapshot_written is False

# -*- coding: utf-8 -*-
"""数字彩持久化增量统计快照。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, cast

import pandas as pd

from src.analysis.digit_data import sort_digit_dataframe_by_issue
from src.analysis.digit_statistics import (
    DEFAULT_BAYESIAN_PRIOR_STRENGTH,
    DEFAULT_FREQUENCY_WINDOWS,
    DigitStatisticsResult,
    _big_small_label,
    _parity_label,
    _shape_domain,
    _smoothed_probabilities_from_counter,
    classify_digit_shape,
    digit_consecutive_count,
    digit_latest_distance,
    digit_mirror_count,
    digit_prime_composite_label,
    digit_repeat_latest_count,
    digit_sum_tail,
    get_digit_theoretical_probabilities,
)
from src.lotteries.base import LotteryRule, validate_numbers

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_ENGINE_VERSION = "digit-statistics-incremental-v1"
SNAPSHOT_BASE_WINDOWS = (10, 30, 50, 100, 300)
ALL_HISTORY_SENTINEL = "allHistory"

_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class DigitStatisticsUpdateMetadata:
    """数字彩统计快照更新诊断。"""

    mode: str
    added_issues: int
    processed_rows: int
    rebuild_reason: str | None
    snapshot_path: str
    persisted: bool = True
    snapshot_written: bool = False
    requested_rebuild_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "addedIssues": self.added_issues,
            "processedRows": self.processed_rows,
            "rebuildReason": self.rebuild_reason,
            "requestedRebuildReason": self.requested_rebuild_reason,
            "snapshotPath": self.snapshot_path,
            "persisted": self.persisted,
            "snapshotWritten": self.snapshot_written,
        }


def _path_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _normalized_windows(windows: Sequence[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(max(1, int(window)) for window in windows))


def _window_signature(
    fixed_windows: tuple[int, ...], all_history_window: bool
) -> tuple[int | str, ...]:
    if all_history_window:
        return (*fixed_windows, ALL_HISTORY_SENTINEL)
    return fixed_windows


def _snapshot_fixed_windows(snapshot: dict[str, Any]) -> tuple[int, ...]:
    signature = snapshot.get("windowSignature")
    if isinstance(signature, list):
        return tuple(
            int(window) for window in signature if window != ALL_HISTORY_SENTINEL
        )
    return _normalized_windows(snapshot.get("requestedWindows", ()))


def _is_snapshot_prefix(
    rows: Sequence[dict[str, Any]], snapshot: dict[str, Any]
) -> bool:
    processed = int(snapshot.get("processedIssues", 0))
    recent_rows = snapshot.get("recentRows", [])
    if not isinstance(recent_rows, list):
        return False
    recent_start = processed - len(recent_rows)
    if len(rows) < recent_start:
        return False
    comparable_count = len(rows) - recent_start
    return rows[recent_start:] == recent_rows[:comparable_count]


def _file_identity(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _rule_signature(rule: LotteryRule) -> str:
    payload = {
        "code": rule.code,
        "drawCount": rule.draw_count,
        "allowRepeated": rule.allow_repeated,
        "ballSpecs": [
            [spec.name, spec.min_number, spec.max_number] for spec in rule.ball_specs
        ],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_history(df: pd.DataFrame, rule: LotteryRule) -> list[dict[str, Any]]:
    missing = [
        column for column in ["期数", *rule.number_columns] if column not in df.columns
    ]
    if missing:
        raise ValueError(f"{rule.display_name}历史数据缺少字段：{', '.join(missing)}")
    chronological = sort_digit_dataframe_by_issue(df, ascending=True)
    rows: list[dict[str, Any]] = []
    for _, row in chronological.iterrows():
        numbers = validate_numbers(
            rule, [int(row[column]) for column in rule.number_columns]
        )
        rows.append({"issue": str(row["期数"]), "numbers": numbers})
    return rows


def _prefix_digest(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["issue"]).encode("utf-8"))
        digest.update(b":")
        digest.update("".join(str(number) for number in row["numbers"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _empty_feature_counts(rule: LotteryRule) -> dict[str, Any]:
    return {
        "rowCount": 0,
        "position": {column: {} for column in rule.number_columns},
        "pair": {
            f"{left}-{right}": {}
            for left in range(rule.draw_count)
            for right in range(left + 1, rule.draw_count)
        },
        "shape": {},
        "sum": {},
        "span": {},
        "parity": {},
        "bigSmall": {},
        "primeComposite": {},
        "consecutive": {},
        "mirror": {},
        "sumTail": {},
        "latestDistance": {},
        "repeatLatest": {},
        "prefix3Shape": {},
        "prefix3Sum": {},
        "prefix3Span": {},
    }


def _change(mapping: dict[str, int], key: Any, delta: int) -> None:
    text = str(key)
    value = int(mapping.get(text, 0)) + delta
    if value:
        mapping[text] = value
    else:
        mapping.pop(text, None)


def _apply_row(
    state: dict[str, Any], numbers: Sequence[int], rule: LotteryRule, delta: int
) -> None:
    state["rowCount"] = int(state["rowCount"]) + delta
    for column, number in zip(rule.number_columns, numbers):
        _change(state["position"][column], number, delta)
    for left in range(rule.draw_count):
        for right in range(left + 1, rule.draw_count):
            _change(
                state["pair"][f"{left}-{right}"],
                f"{numbers[left]},{numbers[right]}",
                delta,
            )
    prefix = list(numbers[:3])
    values = {
        "shape": classify_digit_shape(numbers),
        "sum": sum(numbers),
        "span": max(numbers) - min(numbers),
        "parity": _parity_label(numbers),
        "bigSmall": _big_small_label(numbers),
        "primeComposite": digit_prime_composite_label(numbers),
        "consecutive": digit_consecutive_count(numbers),
        "mirror": digit_mirror_count(numbers),
        "sumTail": digit_sum_tail(numbers),
        "prefix3Shape": classify_digit_shape(prefix),
        "prefix3Sum": sum(prefix),
        "prefix3Span": max(prefix) - min(prefix),
    }
    for feature, value in values.items():
        _change(state[feature], value, delta)


def _apply_transition(
    state: dict[str, Any], newer: Sequence[int], older: Sequence[int], delta: int
) -> None:
    _change(state["latestDistance"], digit_latest_distance(newer, older), delta)
    _change(state["repeatLatest"], digit_repeat_latest_count(newer, older), delta)


def _update_omission(
    omission: dict[str, dict[str, int]], numbers: Sequence[int], rule: LotteryRule
) -> None:
    for column, number, spec in zip(rule.number_columns, numbers, rule.ball_specs):
        values = omission[column]
        for digit in range(spec.min_number, spec.max_number + 1):
            values[str(digit)] = int(values.get(str(digit), 0)) + 1
        values[str(number)] = 0


def _new_snapshot(
    rule: LotteryRule,
    requested_windows: tuple[int, ...],
    maintained_windows: tuple[int, ...],
    prior_strength: float,
    all_history_window: bool,
) -> dict[str, Any]:
    return {
        "schemaVersion": SNAPSHOT_SCHEMA_VERSION,
        "engineVersion": SNAPSHOT_ENGINE_VERSION,
        "ruleSignature": _rule_signature(rule),
        "lotteryCode": rule.code,
        "requestedWindows": list(requested_windows),
        "windowSignature": list(
            _window_signature(requested_windows, all_history_window)
        ),
        "maintainedWindows": list(maintained_windows),
        "allHistoryWindow": all_history_window,
        "bayesianPriorStrength": prior_strength,
        "processedIssues": 0,
        "latestIssue": "",
        "latestNumbers": [],
        "processedPrefixDigest": _prefix_digest([]),
        "allHistory": _empty_feature_counts(rule),
        "windowStates": {
            str(window): _empty_feature_counts(rule) for window in maintained_windows
        },
        "recentRows": [],
        "currentOmission": {
            column: {
                str(digit): 0 for digit in range(spec.min_number, spec.max_number + 1)
            }
            for column, spec in zip(rule.number_columns, rule.ball_specs)
        },
    }


def _append_row(
    snapshot: dict[str, Any], row: dict[str, Any], rule: LotteryRule
) -> None:
    recent_rows = snapshot["recentRows"]
    previous_latest = recent_rows[-1]["numbers"] if recent_rows else None
    numbers = row["numbers"]
    _apply_row(snapshot["allHistory"], numbers, rule, 1)
    if previous_latest is not None:
        _apply_transition(snapshot["allHistory"], numbers, previous_latest, 1)

    recent_rows.append(row)
    for window_text, state in snapshot["windowStates"].items():
        window = int(window_text)
        _apply_row(state, numbers, rule, 1)
        if previous_latest is not None:
            _apply_transition(state, numbers, previous_latest, 1)
        if int(state["rowCount"]) > window:
            evicted = recent_rows[-(window + 1)]
            next_newer = recent_rows[-window]
            _apply_row(state, evicted["numbers"], rule, -1)
            _apply_transition(state, next_newer["numbers"], evicted["numbers"], -1)

    max_window = max(snapshot["maintainedWindows"], default=0)
    if max_window and len(recent_rows) > max_window:
        del recent_rows[: len(recent_rows) - max_window]
    _update_omission(snapshot["currentOmission"], numbers, rule)
    snapshot["processedIssues"] = int(snapshot["processedIssues"]) + 1
    snapshot["latestIssue"] = row["issue"]
    snapshot["latestNumbers"] = list(numbers)


def _counter(mapping: dict[str, int], *, pair: bool = False) -> Counter[Any]:
    if pair:
        return Counter(
            {
                tuple(int(part) for part in key.split(",")): int(value)
                for key, value in mapping.items()
            }
        )
    output: Counter[Any] = Counter()
    for key, value in mapping.items():
        normalized: Any = int(key) if key.lstrip("-").isdigit() else key
        output[normalized] = int(value)
    return output


def _probabilities(
    mapping: dict[str, int],
    sample_count: int,
    domain: Sequence[Any],
    prior: float,
) -> dict[Any, float]:
    counter = _counter(mapping)
    return _smoothed_probabilities_from_counter(counter, sample_count, domain, prior)


def _omission_for_rows(
    rows: Sequence[dict[str, Any]], rule: LotteryRule
) -> dict[str, dict[int, int]]:
    output: dict[str, dict[int, int]] = {}
    latest_first = list(reversed(rows))
    for position, (column, spec) in enumerate(
        zip(rule.number_columns, rule.ball_specs)
    ):
        output[column] = {}
        for digit in range(spec.min_number, spec.max_number + 1):
            miss = 0
            for row in latest_first:
                if row["numbers"][position] == digit:
                    break
                miss += 1
            output[column][digit] = miss
    return output


def _state_for_window(
    snapshot: dict[str, Any], window: int, all_history_windows: set[int]
) -> dict[str, Any]:
    if window in all_history_windows:
        return cast(dict[str, Any], snapshot["allHistory"])
    return cast(dict[str, Any], snapshot["windowStates"][str(window)])


def _result_from_snapshot(
    snapshot: dict[str, Any],
    rule: LotteryRule,
    windows: tuple[int, ...],
    prior: float,
    all_history_window: bool,
) -> DigitStatisticsResult:
    total = int(snapshot["processedIssues"])
    all_history_windows = {total} if all_history_window and total in windows else set()
    position_frequency_windows: dict[int, dict[str, Counter[int]]] = {}
    position_probabilities: dict[int, dict[str, dict[int, float]]] = {}
    pair_frequency_windows: dict[int, dict[str, Counter[tuple[int, int]]]] = {}
    pair_probabilities: dict[int, dict[str, dict[tuple[int, int], float]]] = {}
    feature_probabilities: dict[str, dict[int, dict[Any, float]]] = {
        key: {}
        for key in (
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
            "prefix3Shape",
            "prefix3Sum",
            "prefix3Span",
        )
    }
    omission_windows: dict[int, dict[str, dict[int, int]]] = {}
    domains: dict[str, tuple[Any, ...]] = {
        "shape": _shape_domain(rule.draw_count),
        "sum": tuple(range(rule.draw_count * 9 + 1)),
        "span": tuple(range(10)),
        "parity": tuple(
            f"奇{odd}偶{rule.draw_count - odd}" for odd in range(rule.draw_count + 1)
        ),
        "bigSmall": tuple(
            f"大{big}小{rule.draw_count - big}" for big in range(rule.draw_count + 1)
        ),
        "primeComposite": tuple(
            f"质{prime}合{composite}其他{rule.draw_count - prime - composite}"
            for prime in range(rule.draw_count + 1)
            for composite in range(rule.draw_count - prime + 1)
        ),
        "consecutive": tuple(range(rule.draw_count)),
        "mirror": tuple(range(rule.draw_count * (rule.draw_count - 1) // 2 + 1)),
        "sumTail": tuple(range(10)),
        "latestDistance": tuple(range(rule.draw_count * 9 + 1)),
        "repeatLatest": tuple(range(rule.draw_count + 1)),
        "prefix3Shape": _shape_domain(3),
        "prefix3Sum": tuple(range(28)),
        "prefix3Span": tuple(range(10)),
    }
    pair_domain = tuple((left, right) for left in range(10) for right in range(10))

    for window in windows:
        state = _state_for_window(snapshot, window, all_history_windows)
        row_count = int(state["rowCount"])
        position_frequency_windows[window] = {}
        position_probabilities[window] = {}
        for column, spec in zip(rule.number_columns, rule.ball_specs):
            counter = _counter(state["position"][column])
            position_frequency_windows[window][column] = counter
            digit_count = spec.max_number - spec.min_number + 1
            denominator = row_count + prior
            if denominator <= 0:
                position_probabilities[window][column] = {
                    digit: 1.0 / digit_count
                    for digit in range(spec.min_number, spec.max_number + 1)
                }
            else:
                position_probabilities[window][column] = {
                    digit: (counter.get(digit, 0) + prior / digit_count) / denominator
                    for digit in range(spec.min_number, spec.max_number + 1)
                }
        pair_frequency_windows[window] = {}
        pair_probabilities[window] = {}
        for key, mapping in state["pair"].items():
            counter = _counter(mapping, pair=True)
            pair_frequency_windows[window][key] = counter
            pair_probabilities[window][key] = _smoothed_probabilities_from_counter(
                counter,
                row_count,
                pair_domain,
                prior,
            )
        for feature, domain in domains.items():
            feature_sample_count = sum(int(count) for count in state[feature].values())
            feature_probabilities[feature][window] = _probabilities(
                state[feature],
                feature_sample_count,
                domain,
                prior,
            )

        if window in all_history_windows:
            omission_windows[window] = {
                column: {int(digit): int(value) for digit, value in values.items()}
                for column, values in snapshot["currentOmission"].items()
            }
        else:
            omission_windows[window] = _omission_for_rows(
                snapshot["recentRows"][-window:], rule
            )

    all_history = snapshot["allHistory"]
    current_omission = (
        {
            column: {int(digit): int(value) for digit, value in values.items()}
            for column, values in snapshot["currentOmission"].items()
        }
        if total
        else {}
    )
    return DigitStatisticsResult(
        code=rule.code,
        display_name=rule.display_name,
        draw_count=rule.draw_count,
        total_issues=total,
        position_frequency={
            column: _counter(values)
            for column, values in all_history["position"].items()
        },
        position_frequency_windows=position_frequency_windows,
        position_probabilities=position_probabilities,
        pair_frequency_windows=pair_frequency_windows,
        pair_probabilities=pair_probabilities,
        shape_probabilities=feature_probabilities["shape"],
        sum_probabilities=feature_probabilities["sum"],
        span_probabilities=feature_probabilities["span"],
        parity_probabilities=feature_probabilities["parity"],
        big_small_probabilities=feature_probabilities["bigSmall"],
        prime_composite_probabilities=feature_probabilities["primeComposite"],
        consecutive_probabilities=feature_probabilities["consecutive"],
        mirror_probabilities=feature_probabilities["mirror"],
        sum_tail_probabilities=feature_probabilities["sumTail"],
        latest_distance_probabilities=feature_probabilities["latestDistance"],
        repeat_latest_probabilities=feature_probabilities["repeatLatest"],
        omission_windows=omission_windows,
        prefix3_shape_probabilities=feature_probabilities["prefix3Shape"],
        prefix3_sum_probabilities=feature_probabilities["prefix3Sum"],
        prefix3_span_probabilities=feature_probabilities["prefix3Span"],
        current_omission=current_omission,
        sum_distribution=_counter(all_history["sum"]),
        span_distribution=_counter(all_history["span"]),
        shape_distribution=_counter(all_history["shape"]),
        parity_distribution=_counter(all_history["parity"]),
        big_small_distribution=_counter(all_history["bigSmall"]),
        theoretical_probabilities=get_digit_theoretical_probabilities(rule),
        latest_issue=str(snapshot["latestIssue"]),
        latest_numbers=[int(number) for number in snapshot["latestNumbers"]],
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_snapshot(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "snapshot_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None, "corrupt_json"
    if not isinstance(payload, dict):
        return None, "corrupt_json"
    return payload, None


def _compatibility_reason(
    snapshot: dict[str, Any],
    rule: LotteryRule,
    fixed_windows: tuple[int, ...],
    prior: float,
    all_history_window: bool,
) -> str | None:
    checks = (
        (
            snapshot.get("schemaVersion") == SNAPSHOT_SCHEMA_VERSION,
            "schema_version_mismatch",
        ),
        (
            snapshot.get("engineVersion") == SNAPSHOT_ENGINE_VERSION,
            "engine_version_mismatch",
        ),
        (
            snapshot.get("ruleSignature") == _rule_signature(rule),
            "rule_signature_mismatch",
        ),
        (
            tuple(
                snapshot.get(
                    "windowSignature",
                    _window_signature(
                        _normalized_windows(snapshot.get("requestedWindows", ())),
                        bool(snapshot.get("allHistoryWindow")),
                    ),
                )
            )
            == _window_signature(fixed_windows, all_history_window),
            "window_config_mismatch",
        ),
        (
            float(snapshot.get("bayesianPriorStrength", -1.0)) == prior,
            "prior_strength_mismatch",
        ),
        (
            bool(snapshot.get("allHistoryWindow")) == all_history_window,
            "window_config_mismatch",
        ),
    )
    return next((reason for valid, reason in checks if not valid), None)


def _build_snapshot_from_rows(
    rows: Sequence[dict[str, Any]],
    rule: LotteryRule,
    snapshot_windows: tuple[int, ...],
    maintained_windows: tuple[int, ...],
    prior: float,
    all_history_window: bool,
) -> dict[str, Any]:
    snapshot = _new_snapshot(
        rule,
        snapshot_windows,
        maintained_windows,
        prior,
        all_history_window,
    )
    for row in rows:
        _append_row(snapshot, row, rule)
    snapshot["processedPrefixDigest"] = _prefix_digest(rows)
    return snapshot


def analyze_digit_history_with_snapshot(
    df: pd.DataFrame,
    rule: LotteryRule,
    snapshot_path: str | Path,
    *,
    frequency_windows: Sequence[int] = DEFAULT_FREQUENCY_WINDOWS,
    fixed_frequency_windows: Sequence[int] | None = None,
    bayesian_prior_strength: float = DEFAULT_BAYESIAN_PRIOR_STRENGTH,
    all_history_window: bool = False,
    rebuild: bool = False,
) -> tuple[DigitStatisticsResult, DigitStatisticsUpdateMetadata]:
    """首次全量构建、后续仅追加更新数字彩统计快照。"""

    path = Path(snapshot_path)
    requested_windows = _normalized_windows(frequency_windows)
    prior = max(0.0, float(bayesian_prior_strength))
    rows = _canonical_history(df, rule)
    observed_snapshot_identity = _file_identity(path)

    with _path_lock(path):
        with _process_lock(path):
            snapshot, load_reason = _load_snapshot(path)
            snapshot_changed_while_waiting = (
                _file_identity(path) != observed_snapshot_identity
            )
            if fixed_frequency_windows is not None:
                fixed_windows = _normalized_windows(fixed_frequency_windows)
            elif not all_history_window:
                fixed_windows = requested_windows
            else:
                stored_fixed = (
                    _snapshot_fixed_windows(snapshot) if snapshot is not None else ()
                )
                expected_requested = (
                    stored_fixed
                    if len(rows) in stored_fixed
                    else (*stored_fixed, max(1, len(rows)))
                )
                if stored_fixed and requested_windows == expected_requested:
                    fixed_windows = stored_fixed
                else:
                    dynamic_window = (
                        len(rows) if len(rows) in requested_windows else None
                    )
                    fixed_windows = tuple(
                        window
                        for window in requested_windows
                        if window != dynamic_window
                    )
            snapshot_windows = (
                fixed_windows if all_history_window else requested_windows
            )
            maintained_windows = tuple(
                sorted(set(SNAPSHOT_BASE_WINDOWS).union(fixed_windows))
            )
            rebuild_reason = "explicit_rebuild" if rebuild else load_reason
            if snapshot is not None and not rebuild:
                rebuild_reason = _compatibility_reason(
                    snapshot, rule, snapshot_windows, prior, all_history_window
                )
            processed = (
                int(snapshot.get("processedIssues", 0)) if snapshot is not None else 0
            )
            stale_view = (
                snapshot is not None
                and len(rows) < processed
                and snapshot_changed_while_waiting
                and _is_snapshot_prefix(rows, snapshot)
            )
            if stale_view:
                stale_snapshot = _build_snapshot_from_rows(
                    rows,
                    rule,
                    snapshot_windows,
                    maintained_windows,
                    prior,
                    all_history_window,
                )
                return _result_from_snapshot(
                    stale_snapshot,
                    rule,
                    requested_windows,
                    prior,
                    all_history_window,
                ), DigitStatisticsUpdateMetadata(
                    mode="stale_view",
                    added_issues=0,
                    processed_rows=len(rows),
                    rebuild_reason="stale_view_not_persisted",
                    requested_rebuild_reason=rebuild_reason,
                    snapshot_path=str(path),
                    persisted=False,
                    snapshot_written=False,
                )
            if snapshot is not None and rebuild_reason is None:
                if len(rows) < processed:
                    rebuild_reason = "history_truncated"
                elif _prefix_digest(rows[:processed]) != snapshot.get(
                    "processedPrefixDigest"
                ):
                    current_latest = (
                        rows[processed - 1]
                        if processed and len(rows) >= processed
                        else None
                    )
                    if current_latest and current_latest["issue"] != snapshot.get(
                        "latestIssue"
                    ):
                        rebuild_reason = "non_append_issue"
                    else:
                        rebuild_reason = "historical_prefix_changed"

            if snapshot is None or rebuild_reason is not None:
                snapshot = _build_snapshot_from_rows(
                    rows,
                    rule,
                    snapshot_windows,
                    maintained_windows,
                    prior,
                    all_history_window,
                )
                _atomic_write_json(path, snapshot)
                result = _result_from_snapshot(
                    snapshot, rule, requested_windows, prior, all_history_window
                )
                metadata = DigitStatisticsUpdateMetadata(
                    mode="full_rebuild",
                    added_issues=len(rows),
                    processed_rows=len(rows),
                    rebuild_reason=rebuild_reason,
                    snapshot_path=str(path),
                    persisted=True,
                    snapshot_written=True,
                )
                return result, metadata

            added_rows = rows[processed:]
            for row in added_rows:
                _append_row(snapshot, row, rule)
            snapshot["processedPrefixDigest"] = _prefix_digest(rows)
            if added_rows:
                _atomic_write_json(path, snapshot)
            result = _result_from_snapshot(
                snapshot, rule, requested_windows, prior, all_history_window
            )
            metadata = DigitStatisticsUpdateMetadata(
                mode="incremental" if added_rows else "cache_hit",
                added_issues=len(added_rows),
                processed_rows=len(added_rows),
                rebuild_reason=None,
                snapshot_path=str(path),
                persisted=True,
                snapshot_written=bool(added_rows),
            )
            return result, metadata

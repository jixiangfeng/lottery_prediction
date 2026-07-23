# -*- coding: utf-8 -*-
"""快乐8选5最终严格契约回归测试。"""

from __future__ import annotations

import itertools
import json
import math
import stat
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts import kl8_fetch_history, kl8_pick5_development, kl8_pick5_null
from src.analysis import kl8_pick5_null as null_module
from src.analysis import kl8_pick5_probability_v1 as probability_module
from src.analysis.kl8_pick5_null import run_formal_kl8_null
from src.analysis.kl8_pick5_probability_v1 import (
    Kl8Pick5Config,
    _multiplicity_total_hits_pmf,
    assert_canonical_formal_config,
    load_and_verify_kl8_protocol,
    load_and_verify_kl8_report,
    payload_sha256,
    run_registered_kl8_development,
)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"alpha": 1.0}, "alpha"),
        ({"required_null_iterations": 1}, "required_null_iterations"),
        ({"frozen_periods": 1}, "frozen_periods"),
        ({"top_pool_size": 19}, "top_pool_size"),
        ({"top_pool_size": 21}, "top_pool_size"),
        ({"top_pool_size": 20.0}, "top_pool_size"),
        ({"required_null_iterations": 5000.5}, "required_null_iterations"),
        ({"calibration_temperatures": (1.0, 0.0)}, "calibration_temperatures"),
        ({"calibration_temperatures": (1.0, -1.0)}, "calibration_temperatures"),
        ({"calibration_temperatures": (1.0, math.nan)}, "calibration_temperatures"),
        ({"calibration_temperatures": (1.0, math.inf)}, "calibration_temperatures"),
    ],
)
def test_config_rejects_invalid_formal_domain(changes, message):
    with pytest.raises(ValueError, match=message):
        Kl8Pick5Config(**changes)


def test_canonical_formal_config_requires_exact_default():
    assert_canonical_formal_config(Kl8Pick5Config())
    with pytest.raises(ValueError, match="规范正式配置"):
        assert_canonical_formal_config(
            replace(Kl8Pick5Config(), concentration_penalty=0.0)
        )
    with pytest.raises(ValueError, match="规范正式配置"):
        assert_canonical_formal_config(Kl8Pick5Config.smoke())


def test_registered_and_formal_public_paths_reject_before_io_or_simulation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    noncanonical = replace(Kl8Pick5Config(), concentration_penalty=0.0)

    with pytest.raises(ValueError, match="规范正式配置"):
        probability_module.build_kl8_protocol(
            probability_module.pd.DataFrame(),
            noncanonical,
            frozen_periods_excluded=500,
            frozen_boundary={"firstIssue": "1", "lastIssue": "500"},
        )
    with pytest.raises(ValueError, match="规范正式配置"):
        load_and_verify_kl8_protocol(
            tmp_path / "missing-protocol.json",
            probability_module.pd.DataFrame(),
            noncanonical,
            frozen_periods_excluded=500,
            frozen_boundary={"firstIssue": "1", "lastIssue": "500"},
        )
    with pytest.raises(ValueError, match="规范正式配置"):
        load_and_verify_kl8_report(
            tmp_path / "missing-report.json",
            tmp_path / "missing-protocol.json",
            probability_module.pd.DataFrame(),
            noncanonical,
            frozen_periods_excluded=500,
            frozen_boundary={"firstIssue": "1", "lastIssue": "500"},
        )

    monkeypatch.setattr(
        probability_module,
        "load_and_verify_kl8_protocol",
        lambda *args, **kwargs: pytest.fail("不得读取协议"),
    )
    with pytest.raises(ValueError, match="规范正式配置"):
        run_registered_kl8_development(
            tmp_path / "missing-protocol.json",
            probability_module.pd.DataFrame(),
            noncanonical,
            frozen_periods_excluded=500,
            frozen_boundary={"firstIssue": "1", "lastIssue": "500"},
        )

    monkeypatch.setattr(
        null_module,
        "load_and_verify_kl8_report",
        lambda *args, **kwargs: pytest.fail("不得读取报告"),
    )
    monkeypatch.setattr(
        null_module,
        "_run_simulation",
        lambda *args, **kwargs: pytest.fail("不得启动模拟"),
    )
    with pytest.raises(ValueError, match="规范正式配置"):
        run_formal_kl8_null(
            tmp_path / "missing-protocol.json",
            tmp_path / "missing-report.json",
            probability_module.pd.DataFrame(),
            config=noncanonical,
            frozen_periods_excluded=500,
            frozen_boundary={"firstIssue": "1", "lastIssue": "500"},
            iterations=5000,
            workers=1,
            checkpoint_dir=tmp_path / "checkpoint",
        )
    with pytest.raises(ValueError, match="frozen_periods_excluded"):
        run_formal_kl8_null(
            tmp_path / "missing-protocol.json",
            tmp_path / "missing-report.json",
            probability_module.pd.DataFrame(),
            config=Kl8Pick5Config(),
            frozen_periods_excluded=1,
            frozen_boundary={"firstIssue": "1", "lastIssue": "500"},
            iterations=5000,
            workers=1,
            checkpoint_dir=tmp_path / "checkpoint",
        )


def test_non_smoke_clis_reject_noncanonical_before_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    noncanonical = replace(Kl8Pick5Config(), concentration_penalty=0.0)
    monkeypatch.setattr(kl8_pick5_development, "Kl8Pick5Config", lambda: noncanonical)
    monkeypatch.setattr(
        kl8_pick5_development,
        "load_kl8_development_csv",
        lambda *args, **kwargs: pytest.fail("登记前不得加载历史"),
    )
    with pytest.raises(ValueError, match="规范正式配置"):
        kl8_pick5_development.main(
            [
                "--csv",
                "missing.csv",
                "--protocol",
                "protocol.json",
                "--register-protocol",
            ]
        )

    monkeypatch.setattr(kl8_pick5_null, "Kl8Pick5Config", lambda: noncanonical)
    monkeypatch.setattr(
        kl8_pick5_null,
        "load_kl8_development_csv",
        lambda *args, **kwargs: pytest.fail("正式null前不得加载历史"),
    )
    with pytest.raises(ValueError, match="规范正式配置"):
        kl8_pick5_null.main(
            [
                "--csv",
                "missing.csv",
                "--output",
                "null.json",
                "--protocol",
                "protocol.json",
                "--reference-report",
                "report.json",
                "--checkpoint-dir",
                "checkpoint",
            ]
        )


def test_generalized_multiplicity_pmf_matches_bruteforce():
    multiplicities = (0, 0, 1, 2, 2, 3)
    counts = tuple(multiplicities.count(value) for value in range(4))
    pmf = _multiplicity_total_hits_pmf(6, 2, counts)
    brute_counts = np.zeros(7, dtype=np.float64)
    combinations = list(itertools.combinations(range(6), 2))
    for selected in combinations:
        brute_counts[sum(multiplicities[index] for index in selected)] += 1.0
    brute_counts /= len(combinations)
    assert np.asarray(pmf) == pytest.approx(brute_counts)
    assert sum(pmf) == pytest.approx(1.0)


def test_pool_overlap_counts_match_exact_set_intersections():
    combo_indexes = probability_module._POOL_COMBINATION_INDEXES
    for chosen_index in (0, 1, len(combo_indexes) // 2, len(combo_indexes) - 1):
        chosen = set(int(value) for value in combo_indexes[chosen_index])
        expected = np.asarray(
            [
                len(chosen & {int(value) for value in candidate})
                for candidate in combo_indexes
            ],
            dtype=np.int8,
        )
        actual = probability_module._pool_overlap_counts(chosen_index)
        assert actual.dtype == np.int8
        assert actual.shape == (len(combo_indexes),)
        assert np.array_equal(actual, expected)


def _slow_top5_reference(probabilities, pair_scores, config):
    probs = probability_module.normalize_sum20(probabilities, epsilon=config.epsilon)
    pool = sorted(range(80), key=lambda index: (-probs[index], index))[
        : config.top_pool_size
    ]
    combo_indexes = probability_module._POOL_COMBINATION_INDEXES
    combo_numbers = np.asarray(pool, dtype=np.int16)[combo_indexes]
    logits = probability_module.logit(
        np.clip(probs, config.epsilon, 1 - config.epsilon)
    )
    scores = logits[combo_numbers].sum(axis=1)
    for left, right in itertools.combinations(range(5), 2):
        scores += (
            config.combo_pair_weight
            * pair_scores[combo_numbers[:, left], combo_numbers[:, right]]
        )
    selected = []
    available = np.ones(len(combo_numbers), dtype=bool)
    for _ in range(config.output_combinations):
        adjusted = scores.copy()
        for previous in selected:
            overlaps = (
                (combo_numbers[:, :, None] == previous[None, None, :])
                .any(axis=2)
                .sum(axis=1)
            )
            adjusted -= config.concentration_penalty * np.square(overlaps)
        adjusted[~available] = -np.inf
        best_score = float(np.max(adjusted))
        tied = np.flatnonzero(np.isclose(adjusted, best_score, rtol=0.0, atol=1e-15))
        chosen_index = min(
            tied,
            key=lambda index: tuple(
                sorted(int(value) for value in combo_numbers[index])
            ),
        )
        selected.append(np.sort(combo_numbers[chosen_index]))
        available[chosen_index] = False
    return [[int(index) + 1 for index in combo] for combo in selected]


def test_fast_combination_selector_matches_slow_reference():
    generator = np.random.default_rng(20260723)
    for concentration_penalty in (0.0, 0.05, 0.2):
        config = replace(Kl8Pick5Config(), concentration_penalty=concentration_penalty)
        for _ in range(3):
            probabilities = generator.uniform(0.01, 0.8, size=80)
            pair_scores = generator.uniform(0.0, 1.0, size=(80, 80))
            pair_scores = (pair_scores + pair_scores.T) / 2.0
            expected = _slow_top5_reference(probabilities, pair_scores, config)
            actual = probability_module.generate_top5_combinations(
                probabilities, pair_scores, config
            )
            assert actual == expected


def test_vectorized_block_bootstrap_means_match_loop_reference():
    sample = np.random.default_rng(17).normal(size=500)
    resamples = 200
    seed = 20260723
    block_length = int(math.sqrt(len(sample)))
    generator = np.random.default_rng(seed)
    expected = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        collected = []
        while len(collected) < len(sample):
            start = int(generator.integers(0, len(sample)))
            collected.extend(
                float(sample[(start + offset) % len(sample)])
                for offset in range(block_length)
            )
        expected[index] = float(np.mean(collected[: len(sample)]))

    actual = probability_module._circular_block_bootstrap_means(
        sample, resamples=resamples, seed=seed
    )
    assert np.array_equal(actual, expected)


def test_payload_hash_and_immutable_json_reject_nonfinite(tmp_path: Path):
    with pytest.raises(ValueError):
        payload_sha256({"value": math.nan})
    with pytest.raises(ValueError):
        probability_module._write_immutable_json(
            {"value": math.inf}, tmp_path / "invalid.json", "测试JSON"
        )
    assert not (tmp_path / "invalid.json").exists()


def test_source_fingerprint_rereads_sources(monkeypatch: pytest.MonkeyPatch):
    original = Path.read_bytes
    reads = 0

    def counted(path: Path) -> bytes:
        nonlocal reads
        reads += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    probability_module.source_fingerprint()
    first_reads = reads
    probability_module.source_fingerprint()
    assert first_reads >= 6
    assert reads == first_reads * 2


def test_fetch_requires_exact_requested_or_advertised_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    rows = [
        {"issue": str(index), "date": "2026-01-01", "numbers": list(range(1, 21))}
        for index in range(1, 4)
    ]
    monkeypatch.setattr(kl8_fetch_history, "_fetch_page", lambda *args: (rows, 3))
    with pytest.raises(RuntimeError, match="请求5期"):
        kl8_fetch_history.fetch_history(5, 1.0, 0)

    responses = iter([(rows, 5), ([], 5)])
    monkeypatch.setattr(kl8_fetch_history, "_fetch_page", lambda *args: next(responses))
    with pytest.raises(RuntimeError, match="接口宣告5期"):
        kl8_fetch_history.fetch_history(0, 1.0, 0)


def test_canonical_csv_is_readonly_durable_and_rejects_writable_identical(
    tmp_path: Path,
):
    rows = [{"issue": "1", "date": "2026-01-01", "numbers": list(range(1, 21))}]
    destination = tmp_path / "history.csv"
    kl8_fetch_history._write_canonical_csv(destination, rows)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444

    destination.chmod(0o644)
    with pytest.raises(ValueError, match="不是只读"):
        kl8_fetch_history._write_canonical_csv(destination, rows)


def test_json_serialization_is_strict_standard_json():
    payload = {"finite": 1.0}
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload

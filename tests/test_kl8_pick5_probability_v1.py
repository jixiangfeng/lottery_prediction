# -*- coding: utf-8 -*-
"""快乐8选5开发挑战器契约测试。"""

from __future__ import annotations

import csv
import inspect
import json
import stat
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import kl8_fetch_history, kl8_pick5_predict_today
from scripts.kl8_pick5_predict_today import main as predict_today_main
from src.analysis import kl8_pick5_probability_v1 as probability_module
from src.analysis.kl8_pick5_probability_v1 import (
    Kl8Pick5Config,
    _hypergeometric_tail,
    _segment_gate,
    build_kl8_protocol,
    canonical_kl8_sha256,
    generate_top5_combinations,
    load_and_verify_kl8_report,
    load_kl8_development_csv,
    normalize_sum20,
    run_kl8_development,
    run_registered_kl8_development,
    write_kl8_protocol,
    write_kl8_report,
)
from src.lotteries import get_lottery_rule


def _history(periods: int) -> pd.DataFrame:
    rows = []
    for index in range(periods):
        start = (index * 7) % 80
        numbers = sorted(((start + offset * 3) % 80) + 1 for offset in range(20))
        rows.append(
            {
                "issue": str(202600000 + index),
                "date": (date(2000, 1, 1) + timedelta(days=index)).isoformat(),
                "numbers": numbers,
            }
        )
    return pd.DataFrame(rows)


def _write_csv(path: Path, development: int, frozen: int, *, poisoned: bool) -> None:
    history = _history(development + frozen)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["issue", "date", "numbers"])
        writer.writeheader()
        for index in range(development + frozen):
            numbers = (
                "BROKEN_FROZEN"
                if poisoned and index >= development
                else " ".join(str(number) for number in history.iloc[index]["numbers"])
            )
            writer.writerow(
                {
                    "issue": str(202600000 + index),
                    "date": history.iloc[index]["date"],
                    "numbers": numbers,
                }
            )


def _frozen_boundary(first_issue: str = "202699001", last_issue: str = "202699500"):
    return {"firstIssue": first_issue, "lastIssue": last_issue}


def test_registry_schema_hash_and_frozen_numbers_are_never_parsed(tmp_path: Path):
    rule = get_lottery_rule("kl8")
    assert rule.draw_count == 20
    assert rule.default_pick_count == 5

    csv_path = tmp_path / "kl8.csv"
    _write_csv(csv_path, 1550, 500, poisoned=True)
    development, metadata = load_kl8_development_csv(csv_path, frozen_periods=500)
    assert len(development) == 1550
    assert metadata["frozenRead"] is False
    assert metadata["frozenPeriods"] == 500
    assert metadata["frozenBoundary"] == {
        "firstIssue": str(202600000 + 1550),
        "lastIssue": str(202600000 + 2049),
    }
    assert canonical_kl8_sha256(development) == canonical_kl8_sha256(
        development.iloc[::-1].reset_index(drop=True)
    )

    invalid = _history(1)
    invalid.at[0, "numbers"] = [1] * 20
    with pytest.raises(ValueError, match="唯一"):
        canonical_kl8_sha256(invalid)


def test_probabilities_combinations_and_hypergeometric_contracts():
    probabilities = normalize_sum20(np.linspace(0.01, 0.6, 80), epsilon=1e-6)
    assert probabilities.dtype == np.float64
    assert probabilities.shape == (80,)
    assert probabilities.sum() == pytest.approx(20.0)
    assert np.all((probabilities >= 1e-6) & (probabilities <= 1 - 1e-6))

    pair = np.zeros((80, 80), dtype=np.float64)
    first = generate_top5_combinations(probabilities, pair, Kl8Pick5Config())
    second = generate_top5_combinations(probabilities, pair, Kl8Pick5Config())
    assert first == second
    assert len(first) == 5
    assert len({tuple(combo) for combo in first}) == 5
    assert all(len(combo) == len(set(combo)) == 5 for combo in first)
    assert all(combo == sorted(combo) for combo in first)
    assert _hypergeometric_tail(0) == pytest.approx(1.0)
    assert _hypergeometric_tail(5) == pytest.approx(15_504 / 24_040_016)


def test_pairwise_expert_uses_contextual_association_identity():
    config = Kl8Pick5Config()
    state = probability_module._OnlineState.initial()
    state.ewma80 = np.linspace(0.4, 0.1, 80, dtype=np.float64)
    marginals = state.ewma80.copy()
    state.pair = np.outer(marginals, marginals)

    context = np.arange(20)
    outside = np.arange(20, 80)
    increment = 0.01
    decrement = increment * len(context) / (len(outside) - 1)
    first = 30
    second = 31
    state.pair[first, context] += increment
    state.pair[context, first] += increment
    state.pair[first, outside[outside != first]] -= decrement
    state.pair[outside[outside != first], first] -= decrement
    state.pair[second, context] -= increment
    state.pair[context, second] -= increment
    state.pair[second, outside[outside != second]] += decrement
    state.pair[outside[outside != second], second] += decrement
    np.fill_diagonal(state.pair, 0.0)

    outputs = probability_module._expert_probabilities(state, config)
    assert outputs.dtype == np.float64
    assert outputs.shape == (6, 80)
    assert np.isfinite(outputs).all()
    assert np.allclose(outputs.sum(axis=1), 20.0)
    assert not np.allclose(outputs[5], outputs[2])
    assert outputs[5, first] > outputs[5, second]

    swapped = probability_module._OnlineState(
        ewma20=state.ewma20.copy(),
        ewma80=state.ewma80.copy(),
        ewma300=state.ewma300.copy(),
        gaps=state.gaps.copy(),
        pair=state.pair.copy(),
        cumulative_losses=state.cumulative_losses.copy(),
    )
    swapped.pair[[first, second], :] = swapped.pair[[second, first], :]
    swapped.pair[:, [first, second]] = swapped.pair[:, [second, first]]
    swapped_outputs = probability_module._expert_probabilities(swapped, config)
    assert swapped_outputs[5, first] < outputs[5, first]
    assert swapped_outputs[5, second] > outputs[5, second]


def test_concentration_penalty_reduces_selected_overlap():
    probabilities = np.full(80, 0.25, dtype=np.float64)
    pair = np.zeros((80, 80), dtype=np.float64)
    without_penalty = generate_top5_combinations(
        probabilities, pair, replace(Kl8Pick5Config(), concentration_penalty=0.0)
    )
    with_penalty = generate_top5_combinations(
        probabilities, pair, replace(Kl8Pick5Config(), concentration_penalty=0.05)
    )

    def total_overlap(combinations: list[list[int]]) -> int:
        return sum(
            len(set(left) & set(right))
            for left_index, left in enumerate(combinations)
            for right in combinations[left_index + 1 :]
        )

    assert total_overlap(with_penalty) < total_overlap(without_penalty)


def test_prequential_leakage_weights_and_candidate_boundaries():
    config = Kl8Pick5Config.smoke()
    history = _history(config.required_periods)
    original = run_kl8_development(history, config, audit_research_candidates=True)

    changed = history.copy(deep=True)
    target = config.warmup_periods + 3
    changed.at[target, "numbers"] = list(range(61, 81))
    future = target + 1
    changed.at[future, "numbers"] = list(range(1, 21))
    replay = run_kl8_development(changed, config, audit_research_candidates=True)

    assert (
        original.periods[target]["mixedProbabilities"]
        == replay.periods[target]["mixedProbabilities"]
    )
    assert (
        original.periods[target]["expertWeightsBefore"]
        == replay.periods[target]["expertWeightsBefore"]
    )
    assert (
        original.periods[target]["expertWeightsAfter"]
        != replay.periods[target]["expertWeightsAfter"]
    )
    assert original.periods[target - 1] == replay.periods[target - 1]
    payload = original.to_dict()
    assert payload["userVisibleCandidates"] == []
    assert payload["researchCandidates"]
    assert payload["formalRecommendation"] is None
    assert payload["evidenceStatus"] == "exploratory_reused_development"
    assert "protocol_identity" not in inspect.signature(run_kl8_development).parameters
    assert payload["protocolIdentity"] is None
    assert payload["developmentProtocolRegistered"] is False
    assert all(len(record["combinationHits"]) == 5 for record in original.periods)
    assert all(
        record["portfolioTotalHits"] == sum(record["combinationHits"])
        and record["portfolioBestHits"] == max(record["combinationHits"])
        for record in original.periods
    )
    for segment in (original.search, original.evaluation):
        metrics = segment["metrics"]
        assert metrics["matchedCostBaselineMeanPortfolioTotalHits"] == 6.25
        assert sum(metrics["ticketHitFrequencies"].values()) == metrics["periods"] * 5
        assert (
            sum(metrics["portfolioTotalHitFrequencies"].values()) == metrics["periods"]
        )
        assert (
            sum(metrics["portfolioBestHitFrequencies"].values()) == metrics["periods"]
        )
    with pytest.raises(TypeError):
        run_kl8_development(history, config, protocol_identity={"forged": True})


def test_calibration_and_evaluation_outcome_boundaries():
    config = Kl8Pick5Config.smoke()
    history = _history(config.required_periods)
    original = run_kl8_development(history, config)
    evaluation_start = (
        config.warmup_periods + config.search_periods + config.calibration_periods
    )

    evaluation_changed = history.copy(deep=True)
    for index in range(evaluation_start, config.required_periods):
        evaluation_changed.at[index, "numbers"] = list(range(61, 81))
    evaluation_replay = run_kl8_development(evaluation_changed, config)
    assert evaluation_replay.selected_temperature == original.selected_temperature
    assert (
        evaluation_replay.periods[evaluation_start]["mixedProbabilities"]
        == original.periods[evaluation_start]["mixedProbabilities"]
    )

    calibration_changed = history.copy(deep=True)
    calibration_start = config.warmup_periods + config.search_periods
    for index in range(calibration_start, evaluation_start):
        probabilities = original.periods[index]["mixedProbabilities"]
        calibration_changed.at[index, "numbers"] = [
            number + 1
            for number in np.argsort(np.asarray(probabilities, dtype=np.float64))[-20:]
        ]
    calibration_replay = run_kl8_development(calibration_changed, config)
    assert calibration_replay.selected_temperature != original.selected_temperature
    assert (
        calibration_replay.periods[:calibration_start]
        == original.periods[:calibration_start]
    )


def test_research_candidates_are_for_next_unseen_draw():
    config = Kl8Pick5Config.smoke()
    history = _history(config.required_periods)
    history.at[len(history) - 1, "numbers"] = list(range(61, 81))
    report = run_kl8_development(history, config, audit_research_candidates=True)

    from src.analysis import kl8_pick5_probability_v1 as module

    records, state, temperature, _ = module._run_records(history, config)
    experts = module._expert_probabilities(state, config)
    weights = module._hedge_weights(state, config)
    next_probabilities = module.normalize_sum20(
        weights @ experts, epsilon=config.epsilon
    )
    next_probabilities = module._temperature_scale(
        next_probabilities, temperature, config.epsilon
    )
    expected = module.generate_top5_combinations(next_probabilities, state.pair, config)

    assert report.research_candidates == expected
    assert report.research_candidates != records[-1]["researchCombinations"]
    assert report.to_dict()["userVisibleCandidates"] == []


def test_segment_gate_requires_preregistered_significance():
    config = Kl8Pick5Config.smoke()
    passing_blocks = [
        {
            "deltaLogLoss": 0.01,
            "deltaBrier": 0.01,
            "meanHitsPerTicket": 1.3,
            "meanPortfolioTotalHits": 6.5,
        }
        for _ in range(5)
    ]
    metrics = {
        "deltaLogLossVsUniform": 0.01,
        "deltaBrierVsUniform": 0.01,
        "expectedPositiveDeviation": 0.0,
        "meanHitsPerTicket": 1.3,
        "meanPortfolioTotalHits": 6.5,
        "exactPortfolioTotalHitsPValue": config.alpha,
        "blockStability": passing_blocks,
        "blockBootstrap": {
            "deltaLogLoss": {
                "lowerOneSided95": 0.01,
                "pValueMeanNonPositive": config.alpha + 0.01,
            },
            "deltaBrier": {
                "lowerOneSided95": 0.01,
                "pValueMeanNonPositive": config.alpha,
            },
        },
    }
    gate = _segment_gate(metrics, config)
    assert gate["marginalGatePassed"] is False
    assert gate["businessGatePassed"] is True
    metrics["blockBootstrap"]["deltaLogLoss"]["pValueMeanNonPositive"] = config.alpha
    metrics["exactPortfolioTotalHitsPValue"] = config.alpha + 0.01
    gate = _segment_gate(metrics, config)
    assert gate["marginalGatePassed"] is True
    assert gate["businessGatePassed"] is False


def test_joint_gate_requires_search_and_evaluation(monkeypatch: pytest.MonkeyPatch):
    from src.analysis import kl8_pick5_probability_v1 as module

    monkeypatch.setattr(
        module,
        "_segment_gate",
        lambda metrics, config: {
            "marginalGatePassed": metrics["segment"] == "Evaluation",
            "businessGatePassed": True,
            "passed": metrics["segment"] == "Evaluation",
            "reasons": [],
        },
    )
    report = run_kl8_development(
        _history(Kl8Pick5Config.smoke().required_periods), Kl8Pick5Config.smoke()
    )
    assert report.search["gate"]["passed"] is False
    assert report.evaluation["gate"]["passed"] is True
    assert report.to_dict()["developmentSignalsPassed"] is False


def test_protocol_report_readonly_recompute_and_tamper_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = Kl8Pick5Config()
    history = _history(config.required_periods)
    protocol_path = tmp_path / "protocol.json"
    report_path = tmp_path / "report.json"
    boundary = _frozen_boundary()
    protocol = build_kl8_protocol(
        history,
        config,
        frozen_periods_excluded=500,
        frozen_boundary=boundary,
    )
    assert protocol["frozenBoundary"] == {
        "periodsExcluded": 500,
        "firstIssue": boundary["firstIssue"],
        "lastIssue": boundary["lastIssue"],
        "numbersRead": False,
    }
    write_kl8_protocol(protocol, protocol_path)
    assert stat.S_IMODE(protocol_path.stat().st_mode) == 0o444
    smoke = Kl8Pick5Config.smoke()
    report = run_kl8_development(_history(smoke.required_periods), smoke)
    report = replace(
        report,
        config=config,
        data_sha256=canonical_kl8_sha256(history),
        frozen_periods_excluded=500,
        protocol_identity={
            "protocolSha256": protocol["protocolSha256"],
            "path": str(protocol_path.resolve()),
        },
    )
    monkeypatch.setattr(
        probability_module, "_run_kl8_development", lambda *args, **kwargs: report
    )
    write_kl8_report(report, report_path)
    report_path.chmod(0o444)
    loaded = load_and_verify_kl8_report(
        report_path,
        protocol_path,
        history,
        config,
        frozen_periods_excluded=500,
        frozen_boundary=boundary,
    )
    assert loaded["reportSha256"] == report.to_dict()["reportSha256"]

    tampered_path = tmp_path / "tampered.json"
    tampered = json.loads(json.dumps(report.to_dict()))
    tampered["evaluation"]["metrics"]["meanHitsPerTicket"] = 5.0
    unsigned = {key: value for key, value in tampered.items() if key != "reportSha256"}
    from src.analysis.kl8_pick5_probability_v1 import payload_sha256

    tampered["reportSha256"] = payload_sha256(unsigned)
    tampered_path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    tampered_path.chmod(0o444)
    with pytest.raises(ValueError, match="确定性重算"):
        load_and_verify_kl8_report(
            tampered_path,
            protocol_path,
            history,
            config,
            frozen_periods_excluded=500,
            frozen_boundary=boundary,
        )

    for changed_boundary in (
        _frozen_boundary(first_issue="202699002"),
        _frozen_boundary(last_issue="202699499"),
    ):
        with pytest.raises(ValueError, match="协议"):
            run_registered_kl8_development(
                protocol_path,
                history,
                config,
                frozen_periods_excluded=500,
                frozen_boundary=changed_boundary,
            )

    with pytest.raises(FileExistsError):
        write_kl8_report(replace(report, frozen_periods_excluded=499), report_path)

    report_path.chmod(0o644)
    with pytest.raises(ValueError, match="只读"):
        write_kl8_report(report, report_path)


def test_source_fingerprint_binds_all_six_execution_files():
    from src.analysis import kl8_pick5_probability_v1 as module

    root = Path(module.__file__).resolve().parents[2]
    paths = [str(path.relative_to(root)) for path in module._source_paths()]
    assert paths == [
        "src/analysis/kl8_pick5_probability_v1.py",
        "src/analysis/kl8_pick5_null.py",
        "scripts/kl8_fetch_history.py",
        "scripts/kl8_pick5_development.py",
        "scripts/kl8_pick5_null.py",
        "scripts/kl8_pick5_predict_today.py",
    ]


def test_official_fetch_rejects_redirect_and_predict_today_has_no_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    class RedirectedResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def geturl(self):
            return "https://example.com/redirected"

        def read(self):
            return b"{}"

    monkeypatch.setattr(
        kl8_fetch_history.urllib.request,
        "urlopen",
        lambda request, timeout: RedirectedResponse(),
    )
    with pytest.raises(ValueError, match="非白名单"):
        kl8_fetch_history._fetch_page(1, 10, 1.0, 0)

    csv_path = tmp_path / "history.csv"
    _write_csv(csv_path, 1550, 500, poisoned=True)
    assert predict_today_main(["--csv", str(csv_path), "--frozen-periods", "500"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["userVisibleCandidates"] == []
    assert payload["automaticFetch"] is False
    assert payload["stateOverwritten"] is False
    assert payload["researchTargetIssue"] is None

    class FakeReport:
        research_candidates = [[1, 2, 3, 4, 5]]

    monkeypatch.setattr(
        kl8_pick5_predict_today,
        "run_kl8_development",
        lambda *args, **kwargs: FakeReport(),
    )
    assert (
        predict_today_main(
            [
                "--csv",
                str(csv_path),
                "--frozen-periods",
                "500",
                "--audit-research-candidates",
            ]
        )
        == 0
    )
    audit = json.loads(capsys.readouterr().out)
    assert audit["developmentCutoffIssue"] == str(202600000 + 1549)
    assert audit["latestKnownIssue"] == str(202600000 + 2049)
    assert audit["researchTargetIssue"] == str(202600000 + 1550)
    assert audit["researchTargetKind"] == "locked_frozen_start_audit"
    assert "不是今日推荐" in audit["researchCandidateNotice"]


def test_fetch_rejects_https_downgrade_invalid_args_and_no_progress(
    monkeypatch: pytest.MonkeyPatch,
):
    class DowngradedResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def geturl(self):
            return (
                "http://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
            )

        def read(self):
            return b"{}"

    monkeypatch.setattr(
        kl8_fetch_history.urllib.request,
        "urlopen",
        lambda request, timeout: DowngradedResponse(),
    )
    with pytest.raises(ValueError, match="HTTPS"):
        kl8_fetch_history._fetch_page(1, 10, 1.0, 0)

    for arguments in ((-1, 1.0, 0), (1, 0.0, 0), (1, 1.0, -1)):
        with pytest.raises(ValueError, match="periods|timeout|retries"):
            kl8_fetch_history.fetch_history(*arguments)

    row = {
        "issue": "202600001",
        "date": "2026-01-01",
        "numbers": list(range(1, 21)),
        "source": "https://www.cwl.gov.cn/example",
    }
    monkeypatch.setattr(
        kl8_fetch_history,
        "_fetch_page",
        lambda page, page_size, timeout, retries: ([row], 2),
    )
    with pytest.raises(RuntimeError, match="新增0期"):
        kl8_fetch_history.fetch_history(0, 1.0, 0)


def test_full_history_fetches_annual_windows_and_rejects_conflicts(
    monkeypatch: pytest.MonkeyPatch,
):
    def row(issue: str, date: str, first: int):
        return {
            "issue": issue,
            "date": date,
            "numbers": list(range(first, first + 20)),
            "source": "https://www.cwl.gov.cn/example",
        }

    pages = {
        (2020, 1): [row("2020002", "2020-10-29", 2)],
        (2020, 2): [row("2020001", "2020-10-28", 1)],
        (2020, 3): [],
        (2021, 1): [row("2021001", "2021-01-01", 3)],
        (2021, 2): [],
    }
    monkeypatch.setattr(
        kl8_fetch_history,
        "_fetch_archive_page",
        lambda year, page, page_size, timeout, retries: pages[(year, page)],
    )
    fetched = kl8_fetch_history.fetch_full_history(
        start_year=2020, end_year=2021, timeout=1.0, retries=0, page_size=1
    )
    assert [item["issue"] for item in fetched] == [
        "2020001",
        "2020002",
        "2021001",
    ]

    conflicting = row("2020002", "2020-10-29", 4)
    pages[(2020, 2)] = [conflicting]
    with pytest.raises(ValueError, match="同一期号内容冲突"):
        kl8_fetch_history.fetch_full_history(
            start_year=2020, end_year=2021, timeout=1.0, retries=0, page_size=1
        )


def test_jsonl_append_rejects_broken_existing_tail(tmp_path: Path):
    output = tmp_path / "history.jsonl"
    output.write_bytes(b'{"issue":"202600001"}')
    rows = [
        {
            "issue": "202600002",
            "date": "2026-01-02",
            "numbers": list(range(1, 21)),
            "source": "https://www.cwl.gov.cn/example",
        }
    ]
    with pytest.raises(ValueError, match="换行"):
        kl8_fetch_history._append_jsonl(output, rows)
    assert output.read_bytes() == b'{"issue":"202600001"}'

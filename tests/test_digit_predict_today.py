# -*- coding: utf-8 -*-

from __future__ import annotations

import json

import pandas as pd

from scripts import digit_predict_today
from scripts.digit_predict_today import _build_prediction_result, _print_text
from src.analysis.digit_history_fetcher import DigitHistoryDraw
from src.analysis.digit_online_gradient import (
    OnlineGradientCandidate,
    OnlineGradientSelection,
)
from src.analysis.digit_prediction_narrative import (
    DeepSeekNarrativeConfig,
    load_deepseek_narrative_config,
    request_deepseek_prediction_narrative,
)


def _selection(*, abstained: bool = True) -> OnlineGradientSelection:
    return OnlineGradientSelection(
        block_start_index=100,
        candidate=OnlineGradientCandidate(0.01, 0.25),
        search_mean_log_loss=6.9,
        validation_mean_log_loss=6.91,
        validation_mean_brier=0.999,
        stable_blocks=1,
        abstained=abstained,
        reasons=("Validation LogLoss未优于均匀",) if abstained else (),
    )


def _inactive_result() -> dict[str, object]:
    return _build_prediction_result(
        lottery="fc3d",
        latest_history_issue="2026191",
        new_draws=[{"issue": "2026191", "number": "906"}],
        selection=_selection(),
        research_candidates=[
            {
                "rank": 1,
                "number": "923",
                "score": 0.2,
                "normalizedRankingWeight": 0.0011,
                "relativeToUniform": 1.1,
                "topContributions": [
                    {
                        "feature": "position_frequency",
                        "featureLabel": "位置频率",
                        "contribution": 0.2,
                    }
                ],
            }
        ],
        state_payload={
            "formalPredictionActivated": False,
            "evidenceStatus": "prospective_only",
            "prospectiveValidation": {
                "status": "collecting",
                "requiredPeriods": 500,
                "observedPeriods": 0,
            },
        },
        latest_exact="906",
    )


def test_merge_latest_draws_is_in_memory_and_prefers_fetched_issue(monkeypatch):
    base = pd.DataFrame(
        {
            "期数": ["2026190", "2026191"],
            "百位": [0, 9],
            "十位": [2, 0],
            "个位": [6, 5],
        }
    )
    original = base.copy(deep=True)
    monkeypatch.setattr(
        digit_predict_today,
        "fetch_digit_history",
        lambda *args, **kwargs: [
            DigitHistoryDraw(
                issue="2026191",
                numbers=(9, 0, 6),
                draw_date="2026-07-20",
                source="https://www.cwl.gov.cn/",
            ),
            DigitHistoryDraw(
                issue="2026192",
                numbers=(1, 2, 3),
                draw_date="2026-07-21",
                source="https://www.cwl.gov.cn/",
            ),
        ],
    )

    merged = digit_predict_today._merge_latest_draws(base, "fc3d", 5, 20.0, 3)

    pd.testing.assert_frame_equal(base, original)
    assert merged["期数"].tolist() == ["2026190", "2026191", "2026192"]
    issue = merged[merged["期数"] == "2026191"].iloc[0]
    assert [int(issue[column]) for column in ("百位", "十位", "个位")] == [9, 0, 6]


def test_inactive_prediction_hides_research_candidates_from_text(capsys):
    result = _inactive_result()

    _print_text(result)

    output = capsys.readouterr().out
    assert result["status"] == "abstained"
    assert result["userVisibleCandidates"] == []
    assert result["researchTop50"] == ["923"]
    assert result["prospectiveValidation"]["observedPeriods"] == 1
    assert "923" not in output
    assert "暂不提供正式推荐" in output
    assert "研究排序仅保留在 --json" in output


def test_research_preview_requires_explicit_flag(capsys):
    result = _inactive_result()

    _print_text(result, show_research=True)

    output = capsys.readouterr().out
    assert result["userVisibleCandidates"] == []
    assert "研究观察Top10（未通过准入，不是正式推荐）" in output
    assert "923" in output
    assert "位置频率 +0.200" in output


def test_uniform_abstention_does_not_create_fake_research_ranking(capsys):
    selection = OnlineGradientSelection(
        block_start_index=100,
        candidate=OnlineGradientCandidate(0.0, 0.0),
        search_mean_log_loss=float("nan"),
        validation_mean_log_loss=float("nan"),
        validation_mean_brier=0.999,
        stable_blocks=0,
        abstained=True,
        reasons=("Search选择λ=0",),
    )

    candidates = digit_predict_today._predict_from_learners(
        pd.DataFrame(), None, None, [], selection
    )

    assert candidates == []

    result = _build_prediction_result(
        lottery="fc3d",
        latest_history_issue="2026191",
        new_draws=[],
        selection=selection,
        research_candidates=candidates,
        state_payload={"formalPredictionActivated": False},
        latest_exact="906",
    )
    _print_text(result, show_research=True)
    output = capsys.readouterr().out

    assert result["researchTop50"] == []
    assert "当前没有可用号码排序" in result["narrative"]
    assert "λ=0表示没有可用号码排序" in output


def test_deepseek_config_loads_from_local_json(tmp_path):
    path = tmp_path / "ai.local.json"
    path.write_text(
        json.dumps(
            {
                "provider": "deepseek",
                "apiKey": "local-test-key",
                "model": "deepseek-v4-flash",
                "timeout": 12,
            }
        ),
        encoding="utf-8",
    )

    config = load_deepseek_narrative_config(path)

    assert config.api_key == "local-test-key"
    assert config.model == "deepseek-v4-flash"
    assert config.timeout == 12


def test_deepseek_narrative_receives_status_but_not_research_candidates(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return "https://api.deepseek.com/chat/completions"

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "当前信号不足，本期不提供候选。",
                            }
                        }
                    ]
                },
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(
        "src.analysis.digit_prediction_narrative.urllib.request.urlopen",
        fake_urlopen,
    )

    narrative = request_deepseek_prediction_narrative(
        _inactive_result(),
        DeepSeekNarrativeConfig(api_key="test-key", model="test-model", timeout=8),
    )

    request_body = captured["body"]
    context = json.loads(request_body["messages"][1]["content"])
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["timeout"] == 8
    assert request_body["model"] == "test-model"
    assert request_body["thinking"] == {"type": "disabled"}
    assert context["abstained"] is True
    assert context["userVisibleCandidateCount"] == 0
    assert "userVisibleCandidates" not in context
    assert "researchTop50" not in context
    assert "researchCandidates" not in context
    assert "923" not in json.dumps(context, ensure_ascii=False)
    assert narrative == "当前信号不足，本期不提供候选。"

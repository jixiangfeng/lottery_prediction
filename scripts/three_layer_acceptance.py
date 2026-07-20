#!/usr/bin/env python3
"""三层设计离线验收；不训练v4，也不读取Frozen目标结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_baselines import build_baseline_suite  # noqa: E402
from src.analysis.digit_data import (  # noqa: E402
    canonical_digit_data_sha256,
    load_digit_csv,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_evaluation import evaluate_probability_history  # noqa: E402
from src.analysis.digit_learned_ranker import (  # noqa: E402
    file_sha256,
    learned_ranker_source_fingerprint,
)
from src.analysis.digit_strategy_gate import (  # noqa: E402
    StrategyEvidence,
    decide_strategy_status,
)
from src.analysis.digit_three_layer import (  # noqa: E402
    THREE_LAYER_SCHEMA_VERSION,
    three_layer_source_fingerprint,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--frozen-periods", type=int, default=500)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rule = get_lottery_rule(args.lottery)
    history = sort_digit_dataframe_by_issue(
        load_digit_csv(args.csv, rule), ascending=True
    )
    if args.frozen_periods <= 0 or len(history) <= args.frozen_periods + 150:
        raise SystemExit("历史不足以建立150期特征窗口和Frozen边界")
    development_end = len(history) - args.frozen_periods
    target_indices = tuple(range(max(150, development_end - 200), development_end, 10))
    probability_rows: dict[str, list[tuple[float, ...]]] = {}
    actual_indices: list[int] = []
    for target_index in target_indices:
        prior = history.iloc[:target_index]
        target = history.iloc[target_index]
        actual_text = "".join(
            str(int(target[column])) for column in rule.number_columns
        )
        actual_indices.append(int(actual_text))
        predictions = build_baseline_suite(prior, rule)
        for name, prediction in predictions.items():
            probability_rows.setdefault(name, []).append(prediction.probabilities)

    evaluations = {
        name: evaluate_probability_history(rows, actual_indices)
        for name, rows in probability_rows.items()
    }
    baselines = {name: evaluation.to_dict() for name, evaluation in evaluations.items()}
    uniform = evaluations["uniform"]
    decision = decide_strategy_status(
        StrategyEvidence(
            search_lift=0.0,
            validation_lift=0.0,
            stable_validation_blocks=0,
            validation_blocks=3,
            mean_log_loss=uniform.mean_log_loss,
            uniform_log_loss=uniform.mean_log_loss,
            mean_brier_score=uniform.mean_brier_score,
            uniform_brier_score=uniform.mean_brier_score,
            data_fingerprint=canonical_digit_data_sha256(history, rule),
            source_fingerprint=learned_ranker_source_fingerprint(),
        )
    )
    payload = {
        "schemaVersion": THREE_LAYER_SCHEMA_VERSION,
        "threeLayerSourceFingerprint": three_layer_source_fingerprint(),
        "lottery": args.lottery,
        "layers": {
            "data": {
                "rows": len(history),
                "csvSha256": file_sha256(args.csv),
                "canonicalDataSha256": canonical_digit_data_sha256(history, rule),
                "developmentEndExclusive": development_end,
                "frozenPeriods": args.frozen_periods,
                "frozenRead": False,
            },
            "baseline": {
                "targetIndices": list(target_indices),
                "metrics": baselines,
            },
            "research": {
                "model": "learned_ranker_v4",
                "featureHistoryPeriods": 150,
                "selectionPolicy": "search_only_validation_confirmation",
                "sourceFingerprint": learned_ranker_source_fingerprint(),
            },
            "production": decision.to_dict(),
        },
        "accepted": True,
        "limitations": [
            "该验收不训练v4",
            "该验收不读取Frozen目标结果",
            "基线结果不构成预测有效或盈利承诺",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

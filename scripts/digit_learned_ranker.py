# -*- coding: utf-8 -*-
"""三位彩固定评分算法 v4 的训练、评估和日报 CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import (  # noqa: E402
    canonical_digit_data_sha256,
    load_digit_csv,
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_learned_features import LearnedFeatureConfig  # noqa: E402
from src.analysis.digit_learned_ranker import (  # noqa: E402
    file_sha256,
    generate_learned_ranker_daily,
    learned_ranker_source_fingerprint,
    load_feature_config_from_params,
    load_params,
    load_params_artifact_fingerprint,
    load_params_metadata,
    save_params,
)
from src.analysis.digit_learned_ranker_search import (  # noqa: E402
    LearnedSearchConfig,
    LearnedSplit,
    sample_feature_configs,
    search_learned_ranker_params,
)
from src.analysis.digit_learned_ranker_walk_forward import (  # noqa: E402
    run_learned_ranker_walk_forward,
    write_walk_forward_report,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def _feature_payload(config: LearnedFeatureConfig) -> dict[str, object]:
    return {
        "windows": list(config.windows),
        "alpha": config.alpha,
        "halfLife": config.half_life,
        "omissionCap": config.omission_cap,
        "windowWeights": dict(config.window_weights or ()),
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3", "pl5"))
    parser.add_argument("--csv", required=True, help="本地历史 CSV；命令不会联网")
    parser.add_argument("--output-dir", default="reports")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="三位彩固定评分算法 v4")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="严格 search/validation 参数搜索")
    _common(train)
    train.add_argument("--params")
    train.add_argument("--min-train-size", type=int, default=100)
    train.add_argument("--random-trials", type=int, default=24)
    train.add_argument("--local-trials", type=int, default=12)
    train.add_argument("--evaluation-stride", type=int, default=1)
    train.add_argument("--seed", type=int, default=20260717)
    train.add_argument(
        "--smoke", action="store_true", help="减少搜索次数和目标期，仅用于流水线冒烟"
    )
    evaluate = subparsers.add_parser("evaluate", help="冻结参数评估 test 段")
    _common(evaluate)
    evaluate.add_argument("--params", required=True)
    daily = subparsers.add_parser("daily", help="生成研究日报和冻结快照")
    _common(daily)
    daily.add_argument("--params", required=True)
    daily.add_argument("--evaluation")
    return parser


def _require_supported(lottery: str) -> None:
    if lottery not in {"fc3d", "pl3"}:
        raise ValueError("learned_ranker_v4 首版只支持 fc3d/pl3，pl5 请保持旧路径")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require_supported(args.lottery)
    rule = get_lottery_rule(args.lottery)
    history = load_digit_csv(args.csv, rule)
    output_dir = Path(args.output_dir)
    if args.command == "train":
        split = LearnedSplit.from_length(len(history))
        min_train = min(args.min_train_size, max(3, split.search_end - 1))
        random_trials = 2 if args.smoke else args.random_trials
        local_trials = 1 if args.smoke else args.local_trials
        stride = (
            max(args.evaluation_stride, max(1, split.validation_end // 5))
            if args.smoke
            else args.evaluation_stride
        )
        feature_config = LearnedFeatureConfig()
        feature_configs = sample_feature_configs(
            seed=args.seed,
            smoke=bool(args.smoke),
            limit=max(4, min(16, random_trials)),
        )
        config = LearnedSearchConfig(
            split=split,
            min_train_size=min_train,
            random_trials=random_trials,
            local_trials=local_trials,
            evaluation_stride=stride,
            seed=args.seed,
            feature_config=feature_config,
            feature_configs=feature_configs,
            smoke=bool(args.smoke),
        )
        result = search_learned_ranker_params(history, rule, config)
        params_path = (
            Path(args.params)
            if args.params
            else output_dir / "state" / "learned_ranker_v4" / f"{rule.code}_params.json"
        )
        metadata = {
            "ruleCode": rule.code,
            "csvSha256": file_sha256(args.csv),
            "canonicalDataSha256": canonical_digit_data_sha256(history, rule),
            "sourceFingerprint": learned_ranker_source_fingerprint(),
            "featureConfig": _feature_payload(result.feature_config),
            "split": split.to_dict(),
            "testSegmentUsedForSelection": False,
            "search": result.to_dict(),
            "smoke": bool(args.smoke),
        }
        save_params(result.params, params_path, metadata=metadata)
        search_path = (
            output_dir / "evaluations" / f"learned_ranker_v4_search_{rule.code}.json"
        )
        search_path.parent.mkdir(parents=True, exist_ok=True)
        search_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(params_path)
        return 0
    if args.command == "evaluate":
        params = load_params(args.params)
        metadata = load_params_metadata(args.params)
        split_payload = dict(metadata.get("split", {}))
        if not split_payload:
            raise ValueError("参数文件缺少冻结切分元数据，禁止重新推导 test 段")
        split = LearnedSplit(
            search_end=int(split_payload["searchEnd"]),
            validation_end=int(split_payload["validationEnd"]),
            test_end=int(split_payload["testEnd"]),
        )
        if metadata.get("testSegmentUsedForSelection") is not False:
            raise ValueError("参数文件未证明 frozen test 未参与选参")
        if metadata.get("ruleCode") not in {None, rule.code}:
            raise ValueError("参数文件玩法与 evaluate 玩法不匹配")
        chronological = sort_digit_dataframe_by_issue(
            normalize_digit_dataframe(history, rule), ascending=True
        )
        if split.test_end > len(chronological):
            raise ValueError("evaluate CSV 少于训练时冻结切分终点")
        frozen_history = chronological.iloc[: split.test_end]
        frozen_canonical_sha256 = canonical_digit_data_sha256(frozen_history, rule)
        if metadata.get("canonicalDataSha256") is not None:
            if metadata.get("canonicalDataSha256") != frozen_canonical_sha256:
                raise ValueError(
                    "evaluate CSV 冻结前缀与训练时 canonical 数据指纹不匹配"
                )
        elif metadata.get("csvSha256") != file_sha256(args.csv):
            raise ValueError("evaluate CSV 与训练时冻结 CSV 指纹不匹配")
        source_fingerprint = learned_ranker_source_fingerprint()
        if metadata.get("sourceFingerprint") != source_fingerprint:
            raise ValueError("evaluate 源码与训练时源码指纹不匹配")
        report = run_learned_ranker_walk_forward(
            history,
            rule,
            params,
            split,
            feature_config=load_feature_config_from_params(args.params),
            csv_sha256=(
                str(metadata["csvSha256"]) if metadata.get("csvSha256") else None
            ),
            canonical_data_sha256=frozen_canonical_sha256,
            source_fingerprint=source_fingerprint,
            params_artifact_fingerprint=load_params_artifact_fingerprint(args.params),
            test_segment_used_for_selection=False,
        )
        markdown_path, json_path = write_walk_forward_report(
            report, output_dir / "evaluations"
        )
        print(markdown_path)
        print(json_path)
        return 0
    markdown_path, json_path, snapshot_path = generate_learned_ranker_daily(
        args.lottery,
        args.csv,
        args.params,
        output_dir=output_dir,
        evaluation_path=args.evaluation,
    )
    print(markdown_path)
    print(json_path)
    print(snapshot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

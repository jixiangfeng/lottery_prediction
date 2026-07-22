# -*- coding: utf-8 -*-
"""三位彩固定评分算法 v4 的训练、评估和日报 CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import (  # noqa: E402
    canonical_digit_data_sha256,
    load_digit_csv,
    load_digit_development_csv,
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
from src.analysis.digit_learned_ranker_adaptive import (  # noqa: E402
    AdaptiveResearchConfig,
    run_adaptive_research,
    write_adaptive_report,
)
from src.analysis.digit_learned_ranker_search import (  # noqa: E402
    OBJECTIVE_PROFILES,
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

ALL_HIT_PROFILES = ("direct_hit_only", "group_hit_only", "pool_hit_only")


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
    train.add_argument("--min-train-size", type=int, default=150)
    train.add_argument("--random-trials", type=int, default=24)
    train.add_argument("--local-trials", type=int, default=12)
    train.add_argument("--evaluation-stride", type=int, default=1)
    train.add_argument(
        "--frozen-test-periods",
        type=int,
        help="显式冻结测试期数；例如 1000 期历史使用 500 期冻结测试",
    )
    train.add_argument(
        "--objective-profile",
        choices=(*OBJECTIVE_PROFILES, "all_hit_only"),
        default="research_calibrated",
        help="默认使用平滑proper scoring；只在Search选择，Validation确认，Frozen不参与",
    )
    train.add_argument("--direct-objective-top-k", type=int, default=50)
    train.add_argument("--group-objective-top-k", type=int, default=10)
    train.add_argument("--position-objective-pool-size", type=int, default=5)
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
    adaptive = subparsers.add_parser(
        "adaptive", help="开发区逐期预测、定期重选参数和无信号放弃模拟"
    )
    _common(adaptive)
    adaptive.add_argument("--frozen-test-periods", type=int, default=500)
    adaptive.add_argument("--outer-periods", type=int, default=500)
    adaptive.add_argument("--retrain-interval", type=int, default=10)
    adaptive.add_argument("--training-lookback", type=int, default=500)
    adaptive.add_argument("--inner-validation-periods", type=int, default=100)
    adaptive.add_argument("--inner-stride", type=int, default=10)
    adaptive.add_argument("--random-trials", type=int, default=4)
    adaptive.add_argument("--local-trials", type=int, default=2)
    adaptive.add_argument("--maximum-ece", type=float, default=0.05)
    adaptive.add_argument("--seed", type=int, default=20260717)
    adaptive.add_argument("--smoke", action="store_true")
    return parser


def _require_supported(lottery: str) -> None:
    if lottery not in {"fc3d", "pl3"}:
        raise ValueError("learned ranker 只支持 fc3d/pl3，当前不支持 pl5")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require_supported(args.lottery)
    rule = get_lottery_rule(args.lottery)
    output_dir = Path(args.output_dir)
    if args.command == "adaptive":
        history, total_periods = load_digit_development_csv(
            args.csv,
            rule,
            frozen_test_periods=args.frozen_test_periods,
        )
        split = LearnedSplit.from_length(
            total_periods, frozen_test_periods=args.frozen_test_periods
        )
        smoke = bool(args.smoke)
        adaptive_report = run_adaptive_research(
            history,
            rule,
            AdaptiveResearchConfig(
                development_end=split.validation_end,
                outer_periods=20 if smoke else args.outer_periods,
                retrain_interval=args.retrain_interval,
                training_lookback=200 if smoke else args.training_lookback,
                inner_validation_periods=(
                    30 if smoke else args.inner_validation_periods
                ),
                inner_stride=max(args.inner_stride, 10 if smoke else 1),
                min_train_size=150,
                random_trials=2 if smoke else args.random_trials,
                local_trials=0 if smoke else args.local_trials,
                seed=args.seed,
                maximum_ece=args.maximum_ece,
                checkpoint_dir=output_dir / "adaptive_checkpoints" / rule.code,
            ),
        )
        path = write_adaptive_report(
            adaptive_report,
            output_dir / "development" / f"learned_ranker_v4_adaptive_{rule.code}.json",
        )
        print(path)
        return 0
    history = load_digit_csv(args.csv, rule)
    if args.command == "train":
        split = LearnedSplit.from_length(
            len(history), frozen_test_periods=args.frozen_test_periods
        )
        min_train = min(args.min_train_size, max(3, split.search_end - 1))
        random_trials = 2 if args.smoke else args.random_trials
        local_trials = 1 if args.smoke else args.local_trials
        stride = (
            max(args.evaluation_stride, max(1, split.validation_end // 5))
            if args.smoke
            else args.evaluation_stride
        )
        feature_configs = sample_feature_configs(
            seed=args.seed,
            smoke=bool(args.smoke),
        )
        feature_config = feature_configs[0]
        config = LearnedSearchConfig(
            split=split,
            min_train_size=min_train,
            random_trials=random_trials,
            local_trials=local_trials,
            evaluation_stride=stride,
            seed=args.seed,
            feature_config=feature_config,
            feature_configs=feature_configs,
            objective_profile=(
                "direct_hit_only"
                if args.objective_profile == "all_hit_only"
                else args.objective_profile
            ),
            direct_objective_top_k=args.direct_objective_top_k,
            group_objective_top_k=args.group_objective_top_k,
            position_objective_pool_size=args.position_objective_pool_size,
            smoke=bool(args.smoke),
        )
        profiles = (
            ALL_HIT_PROFILES
            if args.objective_profile == "all_hit_only"
            else (args.objective_profile,)
        )
        if len(profiles) > 1 and args.params:
            raise ValueError("all_hit_only 会生成三套参数，不能指定单一 --params 路径")
        prepared_target_cache: dict[Any, Any] = {}
        all_profiles_passed = True
        for profile in profiles:
            params_path = (
                Path(args.params)
                if args.params
                else output_dir
                / "state"
                / "learned_ranker_v4"
                / (
                    f"{rule.code}_{profile}_params.json"
                    if len(profiles) > 1
                    else f"{rule.code}_params.json"
                )
            )
            incumbent_params = (
                load_params(params_path) if params_path.exists() else None
            )
            incumbent_feature_config = (
                load_feature_config_from_params(params_path)
                if params_path.exists()
                else None
            )
            profile_config = replace(
                config,
                objective_profile=profile,
                incumbent_params=incumbent_params,
                incumbent_feature_config=incumbent_feature_config,
                progress_checkpoint_path=output_dir
                / f"{args.lottery}_{profile}_search_progress.json",
                require_search_qualification=not bool(args.smoke),
                validation_lock_path=(
                    output_dir
                    / "state"
                    / "learned_ranker_v4"
                    / f"{rule.code}_{profile}_validation_once.marker.json"
                    if not args.smoke
                    else None
                ),
            )
            result = search_learned_ranker_params(
                history,
                rule,
                profile_config,
                prepared_target_cache=prepared_target_cache,
            )
            metadata = {
                "ruleCode": rule.code,
                "csvSha256": file_sha256(args.csv),
                "canonicalDataSha256": canonical_digit_data_sha256(history, rule),
                "sourceFingerprint": learned_ranker_source_fingerprint(),
                "featureConfig": _feature_payload(result.feature_config),
                "split": split.to_dict(),
                "testSegmentUsedForSelection": False,
                "objectiveProfile": profile,
                "searchPassed": result.search_passed,
                "searchReasons": list(result.search_reasons),
                "validationEvaluated": result.validation_evaluated,
                "validationPassed": result.validation_passed,
                "validationReasons": list(result.validation_reasons),
                "validationLockFingerprint": result.validation_lock_fingerprint,
                "search": result.to_dict(),
                "smoke": bool(args.smoke),
            }
            search_suffix = f"_{profile}" if len(profiles) > 1 else ""
            search_path = (
                output_dir
                / "evaluations"
                / f"learned_ranker_v4_search_{rule.code}{search_suffix}.json"
            )
            search_path.parent.mkdir(parents=True, exist_ok=True)
            search_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if result.search_passed and result.validation_passed:
                save_params(result.params, params_path, metadata=metadata)
                print(params_path)
            else:
                all_profiles_passed = False
                print(
                    f"参数未落盘：{profile} 未通过严格 Search/Validation，详见 {search_path}",
                    file=sys.stderr,
                )
        return 0 if all_profiles_passed else 2
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
        if metadata.get("smoke") is True:
            raise ValueError("smoke参数仅用于流水线检查，禁止进入Frozen")
        if metadata.get("validationPassed") is not True:
            reasons = metadata.get("validationReasons", [])
            detail = (
                "、".join(str(reason) for reason in reasons) or "缺少Validation确认"
            )
            raise ValueError(f"参数未通过Validation确认，禁止进入Frozen：{detail}")
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
        learned_report = run_learned_ranker_walk_forward(
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
            learned_report, output_dir / "evaluations"
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

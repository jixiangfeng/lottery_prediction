# -*- coding: utf-8 -*-
"""
频繁项集与关联规则挖掘工具
----------------------------
基于 FP-Growth 思想的轻量实现（针对小范围号码集合进行组合计数），
支持规则缓存与软/硬模式的组合过滤。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set

import json
import time
from itertools import combinations

try:
    from ..config import PATHS, RULE_MINER_CONFIG
except Exception:  # pragma: no cover - 兼容脚本直跑
    from pathlib import Path as _Path
    import sys

    PROJECT_ROOT = _Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import PATHS, RULE_MINER_CONFIG  # type: ignore


@dataclass(frozen=True)
class AssociationRule:
    """描述一条关联规则。"""

    antecedent: frozenset[int]
    consequent: int
    support: float
    confidence: float
    lift: float


@dataclass(frozen=True)
class RuleEvaluation:
    """组合规则评估的返回结构。"""

    accepted: bool
    penalty: float
    violated_rules: List[AssociationRule]


def _transactions_from_draws(draws: Sequence[Sequence[int]], limit: int | None = None) -> List[Set[int]]:
    """将开奖二维数组转换为交易集合。"""

    selected = draws if limit is None else draws[:limit]
    transactions: List[Set[int]] = []
    for row in selected:
        if len(row) <= 1:
            continue
        numbers = {int(v) for v in row[1:] if int(v) > 0}
        if numbers:
            transactions.append(numbers)
    return transactions


def _count_itemsets(transactions: List[Set[int]], max_itemset_size: int) -> Counter[tuple[int, ...]]:
    """统计指定规模内的频繁项出现次数。"""

    counter: Counter[tuple[int, ...]] = Counter()
    if not transactions:
        return counter
    for txn in transactions:
        ordered = sorted(txn)
        txn_len = len(ordered)
        upper = min(max_itemset_size, txn_len)
        for size in range(1, upper + 1):
            for combo in combinations(ordered, size):
                counter[combo] += 1
    return counter


def _generate_rules(
    support_counts: Counter[tuple[int, ...]],
    total_transactions: int,
    min_support: float,
    min_confidence: float,
) -> List[AssociationRule]:
    """根据频繁项集生成单元素后件的关联规则。"""

    rules: List[AssociationRule] = []
    if not support_counts or total_transactions == 0:
        return rules

    support_ratio = {
        itemset: count / total_transactions for itemset, count in support_counts.items()
    }

    for itemset, support in support_ratio.items():
        if len(itemset) < 2 or support < min_support:
            continue
        itemset_count = support_counts[itemset]
        for consequent in itemset:
            antecedent = tuple(sorted(v for v in itemset if v != consequent))
            antecedent_count = support_counts.get(antecedent)
            if not antecedent_count:
                continue
            confidence = itemset_count / antecedent_count
            if confidence < min_confidence:
                continue
            consequent_support = support_ratio.get((consequent,), 0.0)
            lift = confidence / consequent_support if consequent_support > 0 else 0.0
            rules.append(
                AssociationRule(
                    antecedent=frozenset(antecedent),
                    consequent=consequent,
                    support=support,
                    confidence=confidence,
                    lift=lift,
                )
            )
    return rules


def _build_cache_path(
    lottery_code: str,
    limit: int | None,
    min_support: float,
    min_confidence: float,
    max_itemset_size: int,
) -> Path:
    """按参数构造缓存文件路径。"""

    cache_dir = PATHS["data_cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    limit_label = f"{limit}" if limit is not None else "all"
    file_name = f"rules_{lottery_code}_{limit_label}_{min_support:.3f}_{min_confidence:.3f}_{max_itemset_size}.json"
    return cache_dir / file_name


def _load_cache(path: Path, ttl_seconds: int) -> List[AssociationRule] | None:
    """从缓存加载规则列表。"""

    if not path.exists():
        return None
    if ttl_seconds > 0:
        age = time.time() - path.stat().st_mtime
        if age > ttl_seconds:
            return None
    try:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        rules = [
            AssociationRule(
                antecedent=frozenset(rule["antecedent"]),
                consequent=int(rule["consequent"]),
                support=float(rule["support"]),
                confidence=float(rule["confidence"]),
                lift=float(rule["lift"]),
            )
            for rule in payload.get("rules", [])
        ]
        return rules
    except Exception:
        return None


def _save_cache(path: Path, rules: List[AssociationRule]) -> None:
    """将规则写入缓存文件。"""

    payload = {
        "rules": [
            {
                "antecedent": sorted(rule.antecedent),
                "consequent": rule.consequent,
                "support": rule.support,
                "confidence": rule.confidence,
                "lift": rule.lift,
            }
            for rule in rules
        ]
    }
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


class RuleBasedFilter:
    """用于组合过滤的规则执行器。"""

    def __init__(self, rules: List[AssociationRule], mode: str, penalty_weight: float):
        self._rules = rules
        self._mode = mode
        self._penalty_weight = max(penalty_weight, 0.0)

    def evaluate(self, numbers: Sequence[int]) -> RuleEvaluation:
        """判断组合是否命中规则；软模式给出附加惩罚。"""

        combo = set(numbers)
        violations: List[AssociationRule] = []
        for rule in self._rules:
            if rule.antecedent.issubset(combo) and rule.consequent not in combo:
                violations.append(rule)

        if not violations:
            return RuleEvaluation(True, 0.0, [])

        if self._mode == "hard":
            return RuleEvaluation(False, 0.0, violations)

        penalty_basis = max(rule.confidence for rule in violations)
        penalty = penalty_basis * self._penalty_weight
        return RuleEvaluation(True, penalty, violations)


def build_rule_filter(
    draws: Sequence[Sequence[int]],
    lottery_code: str,
    limit: int | None,
    mode: str,
    min_support: float | None = None,
    min_confidence: float | None = None,
    max_itemset_size: int | None = None,
    penalty_weight: float | None = None,
) -> RuleBasedFilter | None:
    """综合配置与缓存，返回规则过滤器实例。"""

    if mode not in {"hard", "soft"}:
        return None

    cfg_support = min_support if min_support is not None else RULE_MINER_CONFIG["min_support"]
    cfg_confidence = (
        min_confidence if min_confidence is not None else RULE_MINER_CONFIG["min_confidence"]
    )
    cfg_size = max_itemset_size if max_itemset_size is not None else RULE_MINER_CONFIG["max_itemset_size"]
    cfg_penalty = (
        penalty_weight if penalty_weight is not None else RULE_MINER_CONFIG["soft_penalty_weight"]
    )

    cache_path = _build_cache_path(
        lottery_code=lottery_code,
        limit=limit,
        min_support=cfg_support,
        min_confidence=cfg_confidence,
        max_itemset_size=cfg_size,
    )
    cached = _load_cache(cache_path, RULE_MINER_CONFIG["cache_ttl_seconds"])

    if cached is None:
        transactions = _transactions_from_draws(draws, limit)
        support_counts = Counter()
        if transactions:
            support_counts = _count_itemsets(transactions, cfg_size)
        rules = _generate_rules(
            support_counts=support_counts,
            total_transactions=len(transactions),
            min_support=cfg_support,
            min_confidence=cfg_confidence,
        )
        _save_cache(cache_path, rules)
    else:
        rules = cached

    if not rules:
        return None

    return RuleBasedFilter(rules=rules, mode=mode, penalty_weight=cfg_penalty)

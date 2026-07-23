# -*- coding: utf-8 -*-
"""快乐8选4安全预测边界与等概率测试组合。"""

from __future__ import annotations

import hashlib
import math
from datetime import date
from pathlib import Path
from typing import Mapping, cast

import numpy as np

from src.analysis.kl8_pick5_probability_v1 import (
    canonical_kl8_sha256,
    load_kl8_development_csv,
)

PICK4_COUNT = 4
PICK4_NUMBER_MIN = 1
PICK4_NUMBER_MAX = 80
PICK4_DRAW_COUNT = 20


def pick4_hit_pmf() -> dict[int, float]:
    """返回公平20-of-80开奖下一注Pick4命中0..4个的精确分布。"""

    denominator = math.comb(PICK4_NUMBER_MAX, PICK4_COUNT)
    return {
        hits: math.comb(PICK4_DRAW_COUNT, hits)
        * math.comb(PICK4_NUMBER_MAX - PICK4_DRAW_COUNT, PICK4_COUNT - hits)
        / denominator
        for hits in range(PICK4_COUNT + 1)
    }


def validate_pick4_ticket(numbers: object) -> list[int]:
    """严格校验并规范化一注快乐8选4。"""

    if not isinstance(numbers, (list, tuple)):
        raise ValueError("快乐8选4票必须是号码序列")
    if any(type(number) is not int for number in numbers):
        raise ValueError("快乐8选4号码必须为整数")
    normalized = sorted(cast(list[int], list(numbers)))
    if len(normalized) != PICK4_COUNT or len(set(normalized)) != PICK4_COUNT:
        raise ValueError("每张快乐8选4票必须包含4个唯一号码")
    if any(
        number < PICK4_NUMBER_MIN or number > PICK4_NUMBER_MAX for number in normalized
    ):
        raise ValueError("快乐8选4号码必须位于1..80")
    return normalized


def _validate_development_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("development_sha256必须为64位小写十六进制")
    return value


def _validate_target_date(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("target_date必须为ISO日期")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("target_date必须为ISO日期") from exc
    if parsed.isoformat() != value:
        raise ValueError("target_date必须为ISO日期")
    return value


def _uniform_test_seed(*, target_date: str, development_sha256: str) -> int:
    payload = f"kl8-pick4-uniform-test-v1|{target_date}|{development_sha256}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def generate_uniform_pick4_test_tickets(
    *, target_date: str, development_sha256: str, ticket_count: int
) -> list[list[int]]:
    """生成同日期/开发数据下可复现的等概率Pick4测试组合。"""

    target_date = _validate_target_date(target_date)
    development_sha256 = _validate_development_sha256(development_sha256)
    if type(ticket_count) is not int or not 1 <= ticket_count <= 100:
        raise ValueError("ticket_count必须为1..100整数")
    seed = _uniform_test_seed(
        target_date=target_date, development_sha256=development_sha256
    )
    generator = np.random.default_rng(seed)
    tickets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(tickets) < ticket_count:
        ticket = validate_pick4_ticket(
            [
                int(number)
                for number in generator.choice(
                    np.arange(PICK4_NUMBER_MIN, PICK4_NUMBER_MAX + 1),
                    size=PICK4_COUNT,
                    replace=False,
                )
            ]
        )
        identity = tuple(ticket)
        if identity not in seen:
            seen.add(identity)
            tickets.append(ticket)
    return tickets


def build_pick4_prediction_boundary(
    csv_path: str | Path,
    *,
    frozen_periods: int = 500,
    target_date: str | None = None,
    test_ticket_count: int = 0,
) -> dict[str, object]:
    """构建Pick4安全边界；Frozen只读期号/日期，测试号需显式请求。"""

    if type(test_ticket_count) is not int or not 0 <= test_ticket_count <= 100:
        raise ValueError("test_ticket_count必须为0..100整数")
    if (target_date is None) != (test_ticket_count == 0):
        raise ValueError("测试号码必须同时提供target_date和正ticket_count")
    development, metadata = load_kl8_development_csv(
        csv_path, frozen_periods=frozen_periods
    )
    development_sha256 = canonical_kl8_sha256(development)
    test_candidates: list[list[int]] = []
    seed: int | None = None
    if target_date is not None:
        target_date = _validate_target_date(target_date)
        test_candidates = generate_uniform_pick4_test_tickets(
            target_date=target_date,
            development_sha256=development_sha256,
            ticket_count=test_ticket_count,
        )
        seed = _uniform_test_seed(
            target_date=target_date, development_sha256=development_sha256
        )
    boundary_value = metadata["frozenBoundary"]
    boundary = (
        cast(Mapping[str, object], boundary_value)
        if boundary_value is not None
        else None
    )
    pmf = pick4_hit_pmf()
    return {
        "schemaVersion": "kl8_pick4_prediction_boundary_v1",
        "play": "pick4",
        "pickCount": PICK4_COUNT,
        "drawCount": PICK4_DRAW_COUNT,
        "historyPeriods": metadata["fullPeriods"],
        "developmentPeriods": len(development),
        "developmentCutoffIssue": str(development.iloc[-1]["issue"]),
        "latestKnownIssue": boundary["lastIssue"] if boundary else None,
        "developmentDataSha256": development_sha256,
        "automaticFetch": False,
        "stateOverwritten": False,
        "frozenRead": False,
        "validationOpened": False,
        "promotionPassed": False,
        "recommendationEnabled": False,
        "formalRecommendation": None,
        "userVisibleCandidates": [],
        "targetDate": target_date,
        "testCandidates": test_candidates,
        "testCandidateKind": "uniform_random_test_only" if test_candidates else None,
        "testCandidateSeed": seed,
        "testCandidateNotice": (
            "等概率娱乐测试组合；不是模型预测、正式推荐或收益保证"
            if test_candidates
            else None
        ),
        "baseline": {
            "meanHitsPerTicket": PICK4_COUNT * PICK4_DRAW_COUNT / PICK4_NUMBER_MAX,
            "hitPmf": {str(hits): probability for hits, probability in pmf.items()},
            "atLeast1Rate": 1.0 - pmf[0],
            "atLeast2Rate": sum(pmf[hits] for hits in range(2, PICK4_COUNT + 1)),
            "atLeast3Rate": sum(pmf[hits] for hits in range(3, PICK4_COUNT + 1)),
            "exact4Rate": pmf[4],
        },
    }


__all__ = [
    "PICK4_COUNT",
    "build_pick4_prediction_boundary",
    "generate_uniform_pick4_test_tickets",
    "pick4_hit_pmf",
    "validate_pick4_ticket",
]

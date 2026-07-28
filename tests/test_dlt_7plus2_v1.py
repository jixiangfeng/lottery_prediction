# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
from typing import cast

import pytest

from src.analysis.dlt_7plus2_v1 import (
    CANDIDATE_IDS,
    FIXED_COST_MULTIPLIER,
    FROZEN_PROTOCOL_SHA256,
    PROTOCOL,
    SPLIT_COUNTS,
    build_dlt_7plus2_v1,
    protocol_sha256,
    validate_dlt_7plus2_v1,
)


def test_protocol_is_frozen_to_four_candidates_and_exact_splits() -> None:
    assert SPLIT_COUNTS == {
        "warmup": 600,
        "search": 1301,
        "validation": 500,
        "frozen": 500,
    }
    assert sum(SPLIT_COUNTS.values()) == 2901
    assert CANDIDATE_IDS == (
        "C1_LONG_RIDGE",
        "C2_MULTISCALE_RIDGE",
        "C3_PAIR_GRAPH_RIDGE",
        "C4_EQUAL_LOGPOOL",
    )
    assert len(CANDIDATE_IDS) == 4
    assert cast(dict[str, object], PROTOCOL["splits"])["counts"] == SPLIT_COUNTS
    assert protocol_sha256() == FROZEN_PROTOCOL_SHA256
    assert len(FROZEN_PROTOCOL_SHA256) == 64


def test_builder_ties_use_number_ascending_and_expand_exactly_21() -> None:
    document = build_dlt_7plus2_v1([1.0 / 7.0] * 35, [1.0 / 6.0] * 12)

    assert document["frontRanking"] == list(range(1, 8))
    assert document["backRanking"] == [1, 2]
    assert document["front"] == list(range(1, 8))
    assert document["back"] == [1, 2]
    tickets = cast(list[dict[str, list[int]]], document["expandedTickets"])
    assert len(tickets) == 21
    assert (
        len({(tuple(ticket["front"]), tuple(ticket["back"])) for ticket in tickets})
        == 21
    )
    assert tickets[0] == {"front": [1, 2, 3, 4, 5], "back": [1, 2]}
    assert tickets[-1] == {"front": [3, 4, 5, 6, 7], "back": [1, 2]}
    assert document["fixedCostMultiplier"] == FIXED_COST_MULTIPLIER == 21
    assert document["researchOnly"] is True
    assert document["predictionClaim"] is False
    assert document["equalChanceNoEdge"] is True
    assert document["formalRecommendationStatus"] == "uniform_abstain"
    audit = cast(dict[str, object], document["audit"])
    assert audit["frontMarginalsSha256"] == (
        "d567ad46a1c4b19c23cb354e8ac325d419724131a3e65c18440dcdba272f95b8"
    )
    assert audit["backMarginalsSha256"] == (
        "460207b0c2761d1d06b3f393cfdfa2423a43af75ab5373b73a55ec6273b50493"
    )
    assert audit["selectionSha256"] == (
        "fe61ed8cf5d83b9176daa5a33a59514b326503bc2be86f06bb9704da8671b83a"
    )
    assert audit["expandedTicketsSha256"] == (
        "0fb72a419cd0a9fd10e1f4a54789dac31fde5682d6892ef251d55034c91c1a03"
    )


def test_builder_ranks_score_descending_then_number_ascending_deterministically() -> (
    None
):
    front = [0.3 / 28.0] * 35
    back = [0.02] * 12
    for number, score in {
        2: 0.8,
        3: 0.8,
        9: 0.9,
        10: 0.7,
        11: 0.6,
        12: 0.5,
        13: 0.4,
    }.items():
        front[number - 1] = score
    back[3] = 0.9
    back[7] = 0.9

    first = build_dlt_7plus2_v1(front, back)
    second = build_dlt_7plus2_v1(front, back)

    assert first == second
    assert first["frontRanking"] == [9, 2, 3, 10, 11, 12, 13]
    assert first["front"] == [2, 3, 9, 10, 11, 12, 13]
    assert first["backRanking"] == [4, 8]
    assert first["back"] == [4, 8]
    audit = cast(dict[str, object], first["audit"])
    assert len(cast(str, audit["frontMarginalsSha256"])) == 64
    assert len(cast(str, audit["backMarginalsSha256"])) == 64
    assert len(cast(str, audit["selectionSha256"])) == 64
    assert len(cast(str, audit["expandedTicketsSha256"])) == 64
    validate_dlt_7plus2_v1(first, front_marginals=front, back_marginals=back)


@pytest.mark.parametrize(
    "mutation",
    [
        "ticket",
        "ticket_hash",
        "selection_hash",
        "protocol_hash",
        "cost",
        "claim",
    ],
)
def test_validator_rejects_tampering(mutation: str) -> None:
    document = build_dlt_7plus2_v1([1.0 / 7.0] * 35, [1.0 / 6.0] * 12)
    tampered = copy.deepcopy(document)
    if mutation == "ticket":
        cast(list[dict[str, list[int]]], tampered["expandedTickets"])[0]["front"][
            -1
        ] = 6
    elif mutation == "ticket_hash":
        cast(dict[str, object], tampered["audit"])["expandedTicketsSha256"] = "0" * 64
    elif mutation == "selection_hash":
        cast(dict[str, object], tampered["audit"])["selectionSha256"] = "0" * 64
    elif mutation == "protocol_hash":
        tampered["protocolSha256"] = "0" * 64
    elif mutation == "cost":
        tampered["fixedCostMultiplier"] = 20
    else:
        tampered["equalChanceNoEdge"] = False

    with pytest.raises(ValueError):
        validate_dlt_7plus2_v1(tampered)


def test_validator_rejects_incomplete_expansion_and_wrong_probability_binding() -> None:
    front = [1.0 / 7.0] * 35
    back = [1.0 / 6.0] * 12
    document = build_dlt_7plus2_v1(front, back)
    incomplete = copy.deepcopy(document)
    cast(list[object], incomplete["expandedTickets"]).pop()

    with pytest.raises(ValueError):
        validate_dlt_7plus2_v1(incomplete)
    changed_front = list(front)
    changed_front[0] += 1e-6
    with pytest.raises(ValueError):
        validate_dlt_7plus2_v1(
            document, front_marginals=changed_front, back_marginals=back
        )


@pytest.mark.parametrize(
    "front,back",
    [
        ([0.1] * 34, [1.0 / 6.0] * 12),
        ([1.0 / 7.0] * 35, [0.1] * 11),
        ([float("nan")] + [1.0 / 7.0] * 34, [1.0 / 6.0] * 12),
        ([1.0 / 7.0] * 35, [float("inf")] + [1.0 / 6.0] * 11),
        ([-0.1] + [0.15] * 34, [1.0 / 6.0] * 12),
        ([1.0 / 7.0] * 35, [1.0] * 12),
    ],
)
def test_builder_rejects_invalid_marginals(
    front: list[float], back: list[float]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_dlt_7plus2_v1(front, back)

from src.analysis.ssq_d8_plus7_singles import build_d8_plus7_singles


def test_d8_plus7_is_deterministic_35_ticket_zero_b35_overlap_portfolio() -> None:
    red = [float(33 - value) for value in range(1, 34)]
    blue = [float(16 - value) for value in range(1, 17)]
    first = build_d8_plus7_singles(red, blue)
    second = build_d8_plus7_singles(red, blue)
    assert first == second
    assert first["audit"] == {"tickets": 35, "costYuan": 70, "b35Overlap": 0}
    assert len(first["expandedTickets"]) == 35
    assert len(first["supplementTickets"]) == 7
    assert (
        len(
            {
                (tuple(ticket["red"]), ticket["blue"])
                for ticket in first["expandedTickets"]
            }
        )
        == 35
    )

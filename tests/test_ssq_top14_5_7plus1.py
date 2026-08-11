from src.analysis.ssq_top14_5_7plus1 import build_top14_five_7plus1


def test_top14_five_tickets_are_unique_and_seven_red():
    p = [1 / (n + 1) for n in range(33)]
    b = [1 / (n + 1) for n in range(16)]
    x = build_top14_five_7plus1(p, b)
    assert len(x["tickets"]) == 5
    assert len({(tuple(t["red"]), t["blue"]) for t in x["tickets"]}) == 5
    assert all(len(t["red"]) == 7 for t in x["tickets"])
    assert x["costYuan"] == 70

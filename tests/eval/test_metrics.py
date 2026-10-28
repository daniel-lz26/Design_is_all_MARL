from __future__ import annotations

import pytest

from som.core.history import EpisodeLog, LegRecord, OrderRecord, ProcessRecord
from som.eval.compare import best_policy, mean_ci95, summarize, to_markdown
from som.eval.metrics import compute_metrics


def _log(delivered: list[float | None], terminated: bool, end: float) -> EpisodeLog:
    orders = [
        OrderRecord(f"order_{i}", "J", ["Turning"], 0.0, 100.0, d)
        for i, d in enumerate(delivered, 1)
    ]
    return EpisodeLog(
        "tiny",
        ["robot_0"],
        ["machine_turning"],
        orders,
        processes=[ProcessRecord("machine_turning", "order_1", 0.0, 50.0)],
        legs=[
            LegRecord("robot_0", "order_1", "pickup", 0.0, (0, 0), "station_SRC", 10.0),
            LegRecord("robot_0", "order_1", "dropoff", 10.0, (0, 0), "station_A", 40.0),
        ],
        end_time=end,
        terminated=terminated,
    )


def test_on_time_rate_counts_all_orders() -> None:
    m = compute_metrics(_log([90.0, 130.0, None, None], terminated=False, end=200.0))
    assert m.delivered == 2 and m.on_time == 1
    assert m.on_time_rate == pytest.approx(0.25)  # 1 of 4, not 1 of 2
    # tardiness: 0, 30, and undelivered ones measured to end_time: 100, 100
    assert m.mean_tardiness == pytest.approx(57.5)
    assert m.max_tardiness == pytest.approx(100.0)
    assert m.makespan == 200.0


def test_makespan_and_utilization_when_complete() -> None:
    m = compute_metrics(_log([80.0, 100.0], terminated=True, end=100.0))
    assert m.makespan == 100.0 and m.on_time == 2
    assert m.robot_empty_frac == pytest.approx(0.10)
    assert m.robot_loaded_frac == pytest.approx(0.30)
    assert m.machine_util == pytest.approx(0.50)


def test_mean_ci95() -> None:
    m, ci = mean_ci95([1.0, 2.0, 3.0])
    assert m == 2.0
    assert ci == pytest.approx(4.303 * 1.0 / 3**0.5)
    assert mean_ci95([5.0]) == (5.0, 0.0)


def test_summary_table_marks_best() -> None:
    rows = [
        {"policy": p, "on_time_rate": r, "mean_tardiness": t, "makespan": 100, "delivered": 10,
         "terminated": True}
        for p, r, t in [("A", 0.5, 10), ("A", 0.7, 12), ("B", 0.6, 5), ("B", 0.6, 5)]
    ]  # fmt: skip
    summary = summarize(rows)
    assert [s.policy for s in summary] == ["A", "B"]
    assert best_policy(summary).policy == "B"  # tie on 60% -> lower tardiness
    md = to_markdown(summary, "t", ["PLACEHOLDER inputs"])
    assert "| **B** |" in md and "> PLACEHOLDER inputs" in md

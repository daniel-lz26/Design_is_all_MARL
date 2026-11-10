"""Shared time estimates (seconds) used by observations, rules, the auction and planners.

All of them use nominal travel times on the open roadmap and the scenario's handling block,
and ignore queues, so they are cheap and identical everywhere they are used.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from som.cell_env.entities import RS, Box, Robot

if TYPE_CHECKING:
    from som.cell_env.des import CellSim


def remaining_proc(box: Box) -> float:
    return sum(box.spec.proc[box.step :])


def leg_overhead(sim: CellSim) -> float:
    """Dock + load + undock at the pickup, dock + unload + undock at the drop-off."""
    h = sim.h
    return 2 * (h.dock_time_s + h.undock_time_s) + h.arm_load_s + h.arm_unload_s


def remaining_work(sim: CellSim, box: Box, start: str) -> float:
    """Seconds to finish ``box`` (sitting in an out-buffer) if a robot at node ``start``
    takes it now: travel to it, then every remaining leg and process step."""
    stops = [sim.stations[box.where]]
    stops += [sim.stations[sim.machine_of[p]] for p in box.spec.route[box.step :]]
    stops.append(sim.stations[sim.sink])
    leg = leg_overhead(sim)
    t = sim.roadmap.time(start, stops[0].approach)
    t += sum(sim.roadmap.time(a.approach, b.approach) + leg for a, b in itertools.pairwise(stops))
    return t + remaining_proc(box)


def slack(sim: CellSim, box: Box, start: str) -> float:
    return box.spec.due - sim.now - remaining_work(sim, box, start)


def free_at(sim: CellSim, r: Robot) -> tuple[float, str]:
    """(seconds until the robot is free again, node where it will be). Down or absent robots
    return infinity."""
    if r.state in (RS.DOWN, RS.ABSENT):
        return float("inf"), r.node
    if r.task is None and r.box is None:
        return 0.0, sim.location(r.id)
    h = sim.h
    here = sim.location(r.id)
    box = sim.boxes[r.box or r.task or ""]
    dest = sim.next_station(box)
    t = 0.0
    if r.box is None:  # still has to fetch it
        pickup = sim.stations[box.where]
        t += sim.roadmap.time(here, pickup.approach) + h.dock_time_s + h.arm_load_s
        t += h.undock_time_s + sim.roadmap.time(pickup.approach, dest.approach)
    else:
        t += sim.roadmap.time(here, dest.approach)
    t += h.dock_time_s + h.arm_unload_s
    return t, dest.approach


def bid(sim: CellSim, r: Robot, box: Box) -> float:
    """Contract-net bid: time until free + travel from there to the pickup."""
    t_free, node = free_at(sim, r)
    return t_free + sim.roadmap.time(node, sim.stations[box.where].approach)

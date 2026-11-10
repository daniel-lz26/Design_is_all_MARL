"""check_invariants(sim): consistency checks for the cell, run after every event in debug mode."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from som.cell_env.entities import RS, Loc

if TYPE_CHECKING:
    from som.cell_env.des import CellSim

DOCKED = (RS.WAIT_ARM, RS.BEING_LOADED, RS.BEING_UNLOADED)


def check_invariants(sim: CellSim) -> list[str]:
    """Human-readable violations; empty when the cell is consistent."""
    p: list[str] = []

    # Every released box is in exactly one place, and its loc/where agree with that place.
    places: Counter[str] = Counter()
    for st in sim.stations.values():
        for bid in st.out_buf:
            places[bid] += 1
            b = sim.boxes[bid]
            if (b.loc, b.where) != (Loc.OUT, st.key):
                p.append(f"{bid} in {st.key} out-buffer but {b.loc.value}@{b.where}")
        for bid in st.in_buf:
            places[bid] += 1
            if sim.boxes[bid].loc is not Loc.IN:
                p.append(f"{bid} in {st.key} in-buffer but {sim.boxes[bid].loc.value}")
        for bid in (st.current, st.blocked_box):
            if bid is not None:
                places[bid] += 1
                if sim.boxes[bid].loc is not Loc.MACHINE:
                    p.append(f"{bid} in {st.key} machine but {sim.boxes[bid].loc.value}")
        if st.current is not None and st.blocked_box is not None:
            p.append(f"{st.key} runs a job while blocked")
    for r in sim.robots.values():
        if r.box is not None:
            places[r.box] += 1
            if (sim.boxes[r.box].loc, sim.boxes[r.box].where) != (Loc.ROBOT, r.id):
                p.append(f"{r.id} carries {r.box} which is {sim.boxes[r.box].loc.value}")
    for b in sim.boxes.values():
        if b.loc is Loc.DONE:
            places[b.id] += 1
        expected = 0 if b.loc is Loc.PENDING else 1
        if places[b.id] != expected:
            p.append(f"{b.id} ({b.loc.value}) is in {places[b.id]} places")
    if sim.delivered != sum(b.loc is Loc.DONE for b in sim.boxes.values()):
        p.append("delivered count is off")

    # Buffer caps, counting claimed boxes already heading for an in-buffer.
    inbound: Counter[str] = Counter()
    for r in sim.robots.values():
        bid = r.box or r.task
        if bid is not None:
            inbound[sim.next_station(sim.boxes[bid]).key] += 1
    for st in sim.stations.values():
        if st.in_cap is not None and len(st.in_buf) + st.inbound > st.in_cap:
            p.append(f"{st.key} in-buffer over capacity: {len(st.in_buf)} + {st.inbound} inbound")
        if st.out_cap is not None and len(st.out_buf) > st.out_cap:
            p.append(f"{st.key} out-buffer over capacity")
        if st.inbound != inbound[st.key]:
            p.append(f"{st.key} inbound {st.inbound} != {inbound[st.key]} robots heading there")

    # Claims: box.claimed_by <-> robot.task, at most one claim per robot.
    for b in sim.boxes.values():
        if b.claimed_by is not None and sim.robots[b.claimed_by].task != b.id:
            p.append(f"{b.id} claimed by {b.claimed_by} whose task is another box")
    for r in sim.robots.values():
        if r.task is not None and sim.boxes[r.task].claimed_by != r.id:
            p.append(f"{r.id} task {r.task} is not claimed by it")
        if r.box is not None and r.task != r.box:
            p.append(f"{r.id} carries {r.box} but its task is {r.task}")

    # Edges: never more robots than capacity.
    for e in sim.roadmap.edges.values():
        if len(e.occupants) > e.capacity:
            p.append(f"edge {e.a}-{e.b} holds {len(e.occupants)} > capacity {e.capacity}")

    # Docks: one robot per dock; arms: one handoff at a time, by a robot docked there.
    docked: Counter[str] = Counter()
    for r in sim.robots.values():
        if r.state in DOCKED:
            st = r.docked_at
            if st is None or sim.stations[st].dock_holder != r.id:
                p.append(f"{r.id} is {r.state.value} but does not hold a dock")
            else:
                docked[st] += 1
    for st in sim.stations.values():
        if docked[st.key] > 1:
            p.append(f"{st.key} has {docked[st.key]} robots docked")
        handling = [
            r
            for r in sim.robots.values()
            if r.state in (RS.BEING_LOADED, RS.BEING_UNLOADED) and r.docked_at == st.key
        ]
        if len(handling) > 1:
            p.append(f"{st.key} arm serves {len(handling)} robots at once")
        if handling and st.arm_busy != handling[0].id:
            p.append(f"{st.key} arm busy with {st.arm_busy}, robot {handling[0].id} is handled")

    # Down robots act on nothing; time never runs past the limit.
    for r in sim.robots.values():
        if r.state is RS.DOWN and (r.task is not None and r.box is None):
            p.append(f"{r.id} is DOWN but still holds a claim")
    if sim.now > sim.t_max + 1e-9:
        p.append("time ran past t_max")
    return p

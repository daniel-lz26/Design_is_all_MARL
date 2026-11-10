"""Roadmap graph: shortest-time paths with turn penalties, edge capacities and closures.

Travel time on an edge is ``length / v_cruise``. A turn of more than ``turn_threshold_deg`` at an
intermediate node costs ``turn_penalty_s``. Paths are found with A* over (node, previous node)
states so turn penalties are exact, and cached per (start, goal, closed edges). Dock and parking
nodes are leaves, so paths never pass through them on the way somewhere else.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from som.config.cell_schema import CellLayoutCfg

EdgeKey = frozenset[str]


@dataclass(slots=True)
class Edge:
    a: str
    b: str
    length: float
    capacity: int
    directed: bool
    occupants: list[str] = field(default_factory=list)  # robots on the edge now
    waiters: list[str] = field(default_factory=list)  # robots waiting at a node to enter
    closed: bool = False  # aisle blocked

    @property
    def key(self) -> EdgeKey:
        return frozenset((self.a, self.b))

    def allows(self, frm: str, to: str) -> bool:
        return not self.directed or (frm == self.a and to == self.b)


class Roadmap:
    def __init__(self, cfg: CellLayoutCfg) -> None:
        self.xy = {n: (float(x), float(y)) for n, (x, y) in cfg.nodes.items()}
        self.v = cfg.robot.v_cruise
        self.turn_s = cfg.robot.turn_penalty_s
        self.turn_cos = math.cos(math.radians(cfg.robot.turn_threshold_deg))
        self.edges: dict[EdgeKey, Edge] = {}
        self.adj: dict[str, list[str]] = {n: [] for n in self.xy}
        for e in cfg.edges:
            length = e.length_m if e.length_m is not None else self.dist(e.a, e.b)
            edge = Edge(e.a, e.b, length, e.capacity, e.directed)
            self.edges[edge.key] = edge
            self.adj[e.a].append(e.b)
            if not e.directed:
                self.adj[e.b].append(e.a)
        self.leaves = {s.dock for s in cfg.stations.values()} | set(cfg.parking)
        self._cache: dict[tuple[str, str, frozenset[EdgeKey]], tuple[float, list[str]]] = {}

    def dist(self, a: str, b: str) -> float:
        (x0, y0), (x1, y1) = self.xy[a], self.xy[b]
        return math.hypot(x1 - x0, y1 - y0)

    def edge(self, a: str, b: str) -> Edge:
        return self.edges[frozenset((a, b))]

    def edge_time(self, a: str, b: str) -> float:
        return self.edge(a, b).length / self.v

    def is_turn(self, prev: str, node: str, nxt: str) -> bool:
        (x0, y0), (x1, y1), (x2, y2) = self.xy[prev], self.xy[node], self.xy[nxt]
        u, w = (x1 - x0, y1 - y0), (x2 - x1, y2 - y1)
        nu, nw = math.hypot(*u), math.hypot(*w)
        if nu < 1e-9 or nw < 1e-9:
            return False
        return (u[0] * w[0] + u[1] * w[1]) / (nu * nw) < self.turn_cos

    def closed_edges(self) -> frozenset[EdgeKey]:
        return frozenset(k for k, e in self.edges.items() if e.closed)

    def path(
        self, start: str, goal: str, avoid: frozenset[EdgeKey] = frozenset()
    ) -> tuple[float, list[str]]:
        """(time, nodes after ``start``) of the fastest path, or (inf, []) if none."""
        key = (start, goal, avoid)
        hit = self._cache.get(key)
        if hit is not None:
            return hit[0], list(hit[1])
        result = self._astar(start, goal, avoid)
        self._cache[key] = result
        return result[0], list(result[1])

    def time(self, start: str, goal: str) -> float:
        """Nominal travel time with no closures (planning estimate for policies)."""
        return self.path(start, goal)[0]

    def _astar(self, start: str, goal: str, avoid: frozenset[EdgeKey]) -> tuple[float, list[str]]:
        if start == goal:
            return 0.0, []
        gx, gy = self.xy[goal]

        def h(n: str) -> float:
            x, y = self.xy[n]
            return math.hypot(gx - x, gy - y) / self.v

        State = tuple[str, str | None]
        best: dict[State, float] = {(start, None): 0.0}
        parent: dict[State, State] = {}
        heap: list[tuple[float, float, str, str | None]] = [(h(start), 0.0, start, None)]
        while heap:
            _, g, node, prev = heapq.heappop(heap)
            if g > best.get((node, prev), math.inf):
                continue
            if node == goal:
                nodes = [node]
                s: State = (node, prev)
                while s in parent:
                    s = parent[s]
                    nodes.append(s[0])
                nodes.reverse()
                return g, nodes[1:]
            for nxt in self.adj[node]:
                if nxt == prev:
                    continue
                if nxt in self.leaves and nxt != goal:
                    continue
                edge = self.edge(node, nxt)
                if edge.key in avoid or not edge.allows(node, nxt):
                    continue
                cost = g + edge.length / self.v
                if prev is not None and self.is_turn(prev, node, nxt):
                    cost += self.turn_s
                st = (nxt, node)
                if cost < best.get(st, math.inf):
                    best[st] = cost
                    parent[st] = (node, prev)
                    heapq.heappush(heap, (cost + h(nxt), cost, nxt, node))
        return math.inf, []

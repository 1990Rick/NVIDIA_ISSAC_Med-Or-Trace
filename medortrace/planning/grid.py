"""2D costmap utilities and an 8-connected A* planner."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from medortrace.common.geometry import OrientedBox


@dataclass
class GridSpec:
    origin: np.ndarray            # world xy of cell (0,0) lower-left corner
    res: float
    shape: tuple[int, int]        # (nx, ny)

    def world_to_cell(self, xy) -> np.ndarray:
        xy = np.atleast_2d(xy)
        c = np.floor((xy[:, :2] - self.origin) / self.res).astype(int)
        c[:, 0] = np.clip(c[:, 0], 0, self.shape[0] - 1)
        c[:, 1] = np.clip(c[:, 1], 0, self.shape[1] - 1)
        return c

    def cell_to_world(self, ij) -> np.ndarray:
        ij = np.atleast_2d(ij)
        return self.origin + (ij + 0.5) * self.res

    def centers(self) -> np.ndarray:
        ix, iy = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing="ij")
        return self.cell_to_world(np.stack([ix.ravel(), iy.ravel()], 1)).reshape(self.shape[0], self.shape[1], 2)

    def in_bounds(self, xy) -> np.ndarray:
        xy = np.atleast_2d(xy)
        c = np.floor((xy[:, :2] - self.origin) / self.res)
        return (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < self.shape[0]) & (c[:, 1] < self.shape[1])


def rasterize_boxes(grid: GridSpec, boxes: list[OrientedBox], inflate: float,
                    z_band: tuple[float, float] = (0.02, 1.9)) -> np.ndarray:
    """Boolean occupancy of boxes intersecting the robot's height band, inflated."""
    pts = grid.centers().reshape(-1, 2)
    occ = np.zeros(len(pts), dtype=bool)
    for b in boxes:
        if b.z_max < z_band[0] or b.z_min > z_band[1]:
            continue
        occ |= b.distance_xy(pts) <= inflate
    return occ.reshape(grid.shape)


_NEIGH = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]


def astar(cost: np.ndarray, lethal: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
          max_expand: int = 200000) -> list[tuple[int, int]] | None:
    """A* on a grid with additive per-cell cost (>=0) and a lethal mask."""
    nx, ny = cost.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if lethal[goal]:
        return None
    g = {start: 0.0}
    parent: dict = {start: None}
    h0 = np.hypot(goal[0] - start[0], goal[1] - start[1])
    openq = [(h0, 0.0, start)]
    closed = set()
    n = 0
    while openq and n < max_expand:
        _, gc, cur = heapq.heappop(openq)
        if cur in closed:
            continue
        closed.add(cur)
        n += 1
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for dx, dy, w in _NEIGH:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny) or nb in closed:
                continue
            # the start cell may be lethal (e.g. robot slightly inside inflation); allow leaving it
            if lethal[nb]:
                continue
            ng = gc + w * (1.0 + cost[nb])
            if ng < g.get(nb, np.inf):
                g[nb] = ng
                parent[nb] = cur
                heapq.heappush(openq, (ng + np.hypot(goal[0] - nb[0], goal[1] - nb[1]), ng, nb))
    return None


def nearest_free(lethal: np.ndarray, cell: tuple[int, int], max_r: int = 15) -> tuple[int, int] | None:
    i, j = int(cell[0]), int(cell[1])
    if not lethal[i, j]:
        return (i, j)
    nx, ny = lethal.shape
    for r in range(1, max_r + 1):
        best = None
        for di in range(-r, r + 1):
            for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and not lethal[a, b]:
                    d = di * di + dj * dj
                    if best is None or d < best[0]:
                        best = (d, (a, b))
        if best:
            return best[1]
    return None


def smooth_path(path_xy: np.ndarray, lethal: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Greedy line-of-sight shortcutting."""
    if len(path_xy) <= 2:
        return path_xy
    out = [path_xy[0]]
    i = 0
    while i < len(path_xy) - 1:
        j = len(path_xy) - 1
        while j > i + 1:
            seg = np.linspace(
                path_xy[i], path_xy[j], int(np.ceil(np.linalg.norm(path_xy[j] - path_xy[i]) / (grid.res * 0.5))) + 2
            )
            c = grid.world_to_cell(seg)
            if not lethal[c[:, 0], c[:, 1]].any():
                break
            j -= 1
        out.append(path_xy[j])
        i = j
    return np.array(out)


def dijkstra_field(cost: np.ndarray, lethal: np.ndarray, start: tuple[int, int]):
    """Single-source shortest-path field (metres-ish cost) and parent pointers."""
    nx, ny = cost.shape
    dist = np.full((nx, ny), np.inf)
    parent = -np.ones((nx, ny), dtype=np.int64)
    s = (int(start[0]), int(start[1]))
    dist[s] = 0.0
    pq = [(0.0, s)]
    while pq:
        d, cur = heapq.heappop(pq)
        if d > dist[cur]:
            continue
        for dx, dy, w in _NEIGH:
            a, b = cur[0] + dx, cur[1] + dy
            if not (0 <= a < nx and 0 <= b < ny) or lethal[a, b]:
                continue
            nd = d + w * (1.0 + cost[a, b])
            if nd < dist[a, b]:
                dist[a, b] = nd
                parent[a, b] = cur[0] * ny + cur[1]
                heapq.heappush(pq, (nd, (a, b)))
    return dist, parent


def extract_path(parent: np.ndarray, goal: tuple[int, int]) -> list[tuple[int, int]]:
    ny = parent.shape[1]
    path = [tuple(goal)]
    cur = parent[goal]
    n = 0
    while cur >= 0 and n < parent.size:
        c = (int(cur // ny), int(cur % ny))
        path.append(c)
        cur = parent[c]
        n += 1
    return path[::-1]

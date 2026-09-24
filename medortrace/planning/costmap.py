"""Navigation costmap built from the belief (never from ground truth).

Layers (all 2D over the occupancy grid):
  * lethal   - static occupancy above threshold (inflated by robot radius),
               sterile keep-out zones, room boundary;
  * soft     - proximity to obstacles, ambiguous (ghost-suspect) mass, voxel
               entropy (unknown space costs more than known free space);
  * edt      - Euclidean distance (m) to the nearest lethal-core cell, for MPC.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from medortrace.belief.occupancy import OccupancyBelief
from medortrace.planning.grid import GridSpec
from medortrace.world.scene import SterileZone


class Costmap:
    def __init__(self, occ: OccupancyBelief, zones: list[SterileZone], robot_radius: float,
                 occ_thresh: float = 0.65, unknown_cost: float = 0.6, ambiguous_cost: float = 2.0,
                 keepout_extra: float = 0.0, robot_height: float = 1.55):
        self.grid: GridSpec = occ.grid2d
        col = occ.column_occupancy(0.1, robot_height)
        core = col > occ_thresh
        # room boundary
        core[0, :] = core[-1, :] = core[:, 0] = core[:, -1] = True
        self.edt = distance_transform_edt(~core) * self.grid.res
        pts = self.grid.centers().reshape(-1, 2)
        keep = np.zeros(len(pts), dtype=bool)
        self.zone_dist = np.full(len(pts), np.inf)
        for z in zones:
            keep |= z.box.contains_xy(pts, margin=z.keepout_margin + keepout_extra)
            self.zone_dist = np.minimum(self.zone_dist, z.box.distance_xy(pts))
        self.keepout = keep.reshape(self.grid.shape)
        self.zone_dist = self.zone_dist.reshape(self.grid.shape)
        self.lethal = (self.edt < robot_radius + 0.05) | self.keepout
        unc = occ.uncertainty_field(0.1, robot_height)
        amb = np.clip(occ.ambiguous, 0, 1)
        self.soft = (unknown_cost * np.clip(unc, 0, 1) + ambiguous_cost * amb
                     + 2.0 * np.exp(-(self.edt - robot_radius) / 0.2).clip(0, 5))
        self.robot_radius = robot_radius

    def lookup(self, xy: np.ndarray, layer: str = "edt") -> np.ndarray:
        arr = getattr(self, layer)
        c = self.grid.world_to_cell(xy.reshape(-1, 2))
        out = arr[c[:, 0], c[:, 1]]
        inb = self.grid.in_bounds(xy.reshape(-1, 2))
        if layer == "edt":
            out = np.where(inb, out, 0.0)
        elif layer in ("lethal", "keepout"):
            out = np.where(inb, out, True)
        return out.reshape(xy.shape[:-1])

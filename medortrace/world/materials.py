"""Physically meaningful material library for the OR digital twin.

Each material carries four *consistent* views of the same surface:

1. **Visual** (``UsdPreviewSurface`` / OmniPBR parameters) for the RTX camera.
2. **Non-visual** (RTX lidar / radar) attributes following the Omniverse
   SimReady non-visual material schema (``omni:simready:nonvisual:*``) plus the
   explicit parameters the lite simulator uses (reflectance, specularity,
   transmissivity) so the two backends agree qualitatively.
3. **Acoustic** reflectivity / absorption coefficient at ~40 kHz (ultrasonic
   probe band).
4. **Physics** (PhysX material: static/dynamic friction, restitution, density).

Values are engineering estimates from public data sheets; they are *nuisance*
parameters and get perturbed by the Replicator/material randomizer within the
``perturb`` ranges below (see docs/sim_to_real_checklist.md for fidelity).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    # visual
    diffuse: tuple[float, float, float]
    metallic: float
    roughness: float
    opacity: float = 1.0
    mdl: str = "OmniPBR.mdl"
    # lidar (905 nm)
    lidar_reflectance: float = 0.5     # Lambertian albedo at 905 nm
    specularity: float = 0.0           # probability-weight of mirror-like return
    transmissivity: float = 0.0        # fraction of rays passing through
    # radar (77 GHz)
    radar_rcs_gain: float = 0.0        # dB offset relative to a matte dielectric
    radar_penetrable: bool = False     # fabric/thin plastic are largely transparent
    # acoustic (40 kHz)
    acoustic_reflectivity: float = 0.5
    # camera appearance cue
    glare: float = 0.0                 # specular highlight propensity under OR lights
    # physics
    static_friction: float = 0.6
    dynamic_friction: float = 0.5
    restitution: float = 0.1
    density: float = 1000.0            # kg/m^3
    # SimReady non-visual tokens (consumed by RTX lidar/radar material manager)
    nonvisual_base: str = "plastic"
    nonvisual_coating: str = "none"
    nonvisual_attributes: str = "none"
    perturb: dict = field(default_factory=dict)

    def perturbed(self, rng: np.random.Generator, scale: float = 1.0) -> "Material":
        """Return a nuisance-randomised copy (keeps material identity/causal class)."""
        kw = {}
        for key, (lo, hi) in self.perturb.items():
            base = getattr(self, key)
            delta = rng.uniform(lo, hi) * scale
            kw[key] = float(np.clip(base + delta, 0.0, 1.0 if key != "radar_rcs_gain" else 40.0))
        return replace(self, **kw) if kw else self


MATERIALS: dict[str, Material] = {
    "stainless_steel_brushed": Material(
        "stainless_steel_brushed", (0.62, 0.62, 0.64), 1.0, 0.35,
        mdl="OmniSurface/OmniSurfaceBase.mdl",
        lidar_reflectance=0.55, specularity=0.45, radar_rcs_gain=15.0,
        acoustic_reflectivity=0.95, glare=0.45, static_friction=0.45,
        dynamic_friction=0.35, restitution=0.2, density=8000.0,
        nonvisual_base="steel_stainless", nonvisual_coating="none", nonvisual_attributes="none",
        perturb={"roughness": (-0.15, 0.15), "specularity": (-0.15, 0.2), "glare": (-0.1, 0.2)},
    ),
    "instrument_steel_polished": Material(
        "instrument_steel_polished", (0.75, 0.76, 0.78), 1.0, 0.08,
        mdl="OmniSurface/OmniSurfaceBase.mdl",
        lidar_reflectance=0.4, specularity=0.85, radar_rcs_gain=12.0,
        acoustic_reflectivity=0.95, glare=0.9, static_friction=0.3,
        dynamic_friction=0.25, restitution=0.3, density=7900.0,
        nonvisual_base="steel_stainless", nonvisual_coating="clearcoat", nonvisual_attributes="none",
        perturb={"roughness": (-0.05, 0.1), "glare": (-0.2, 0.1)},
    ),
    "surgical_drape": Material(
        "surgical_drape", (0.18, 0.42, 0.55), 0.0, 0.9,
        lidar_reflectance=0.25, specularity=0.0, transmissivity=0.02,
        radar_rcs_gain=-12.0, radar_penetrable=True, acoustic_reflectivity=0.15,
        glare=0.02, static_friction=0.8, dynamic_friction=0.7, restitution=0.0, density=300.0,
        nonvisual_base="fabric", nonvisual_coating="none", nonvisual_attributes="none",
        perturb={"lidar_reflectance": (-0.08, 0.08), "roughness": (-0.1, 0.05)},
    ),
    "gown_fabric": Material(
        "gown_fabric", (0.35, 0.55, 0.6), 0.0, 0.9,
        lidar_reflectance=0.35, radar_rcs_gain=-3.0, acoustic_reflectivity=0.25,
        nonvisual_base="fabric", density=985.0,
        perturb={"lidar_reflectance": (-0.1, 0.1)},
    ),
    "plastic_hdpe": Material(
        "plastic_hdpe", (0.9, 0.9, 0.88), 0.0, 0.5,
        lidar_reflectance=0.7, specularity=0.05, radar_rcs_gain=-6.0, radar_penetrable=True,
        acoustic_reflectivity=0.6, glare=0.15, density=950.0,
        nonvisual_base="plastic", perturb={"roughness": (-0.2, 0.2)},
    ),
    "acrylic_clear": Material(
        "acrylic_clear", (0.95, 0.97, 0.98), 0.0, 0.05, opacity=0.15, mdl="OmniGlass.mdl",
        lidar_reflectance=0.08, specularity=0.3, transmissivity=0.85,
        radar_rcs_gain=-8.0, radar_penetrable=True, acoustic_reflectivity=0.7, glare=0.7,
        density=1180.0, nonvisual_base="glass", nonvisual_coating="none",
        perturb={"transmissivity": (-0.15, 0.1)},
    ),
    "painted_wall": Material(
        "painted_wall", (0.82, 0.86, 0.84), 0.0, 0.8,
        lidar_reflectance=0.8, radar_rcs_gain=0.0, acoustic_reflectivity=0.85,
        nonvisual_base="concrete", nonvisual_coating="paint", density=2400.0,
        perturb={"lidar_reflectance": (-0.1, 0.05)},
    ),
    "floor_vinyl": Material(
        "floor_vinyl", (0.55, 0.6, 0.58), 0.0, 0.35,
        lidar_reflectance=0.45, specularity=0.1, acoustic_reflectivity=0.6, glare=0.2,
        static_friction=0.8, dynamic_friction=0.7, nonvisual_base="rubber",
        perturb={"specularity": (-0.05, 0.15), "roughness": (-0.1, 0.1)},
    ),
    "monitor_glass": Material(
        "monitor_glass", (0.05, 0.05, 0.06), 0.2, 0.05, mdl="OmniGlass.mdl",
        lidar_reflectance=0.1, specularity=0.7, transmissivity=0.0, radar_rcs_gain=5.0,
        acoustic_reflectivity=0.9, glare=0.85, nonvisual_base="glass",
        nonvisual_coating="clearcoat", perturb={"specularity": (-0.1, 0.1)},
    ),
    "rubber_black": Material(
        "rubber_black", (0.04, 0.04, 0.04), 0.0, 0.8,
        lidar_reflectance=0.05, radar_rcs_gain=-4.0, acoustic_reflectivity=0.3,
        static_friction=1.0, dynamic_friction=0.9, nonvisual_base="rubber",
    ),
    "cotton_sponge": Material(
        "cotton_sponge", (0.95, 0.95, 0.95), 0.0, 1.0,
        lidar_reflectance=0.75, radar_rcs_gain=-20.0, radar_penetrable=True,
        acoustic_reflectivity=0.05, static_friction=0.9, dynamic_friction=0.8,
        density=100.0, nonvisual_base="fabric",
    ),
    "human": Material(
        "human", (0.4, 0.55, 0.6), 0.0, 0.85,
        lidar_reflectance=0.4, radar_rcs_gain=-1.0, acoustic_reflectivity=0.35,
        nonvisual_base="skin", density=985.0,
    ),
}


def get_material(name: str) -> Material:
    return MATERIALS[name]

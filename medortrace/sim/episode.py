"""Backend-independent episode construction.

``build_episode(cfg, seed)`` deterministically produces everything a backend
needs: the true scene, the surveyed scene (prior map), the workflow script,
the fault model and nuisance material table.  Both the lite backend and the
Isaac Sim backend call this function, so a seed means the same experiment in
either simulator.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from medortrace.common.config import config_hash
from medortrace.common.rng import RngStreams
from medortrace.sim.faults import FaultModel, apply_truth_modifications, build_prior_map, sample_faults
from medortrace.world.generator import generate_scene, perturbed_materials
from medortrace.world.materials import Material
from medortrace.world.scene import SceneObject, SceneSpec
from medortrace.world.workflow import WorkflowScript, generate_workflow


@dataclass
class Episode:
    cfg: dict
    seed: int
    streams: RngStreams
    spec: SceneSpec                 # true world
    survey_spec: SceneSpec          # world at survey time (before truth modifications)
    prior_map: list[SceneObject]    # robot's (possibly corrupted) static map
    workflow: WorkflowScript
    faults: FaultModel
    materials: dict[str, Material]
    cfg_hash: str


def build_episode(cfg: dict, seed: int) -> Episode:
    overrides = cfg.get("stream_overrides", {})
    streams = RngStreams(seed, overrides)
    spec = generate_scene(cfg, streams)
    survey = copy.deepcopy(spec)
    duration = float(cfg.get("episode", {}).get("duration_s", 180.0))
    fm = sample_faults(cfg, duration, streams["faults"], spec.hidden_cause)
    apply_truth_modifications(spec, fm, streams["faults"])
    # hidden-cause objects are never in the survey map
    survey.objects = [o for o in survey.objects if "hidden_cause" not in o.tags]
    prior = build_prior_map(spec, survey, fm, streams["faults"])
    wf = generate_workflow(spec, cfg, streams)
    mats = perturbed_materials(spec, streams)
    return Episode(cfg, seed, streams, spec, survey, prior, wf, fm, mats, config_hash(cfg))

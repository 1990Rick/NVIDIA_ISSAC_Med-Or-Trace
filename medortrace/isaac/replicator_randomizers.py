"""Domain randomisation that preserves causal labels + causal-label writer.

Two rules make the synthetic data usable for *counterfactual* learning:

1. **Nuisance only.**  Randomisers act on appearance and non-causal geometry
   (materials within their physical ranges, lighting, haze, clutter poses,
   drape colour, staff gown colour).  They never touch prims or attributes
   that encode the hidden cause (item placements under study, hidden-cause
   objects, the specular screen in CF-B, the moved cart in CF-D).
2. **Verified invariance.**  :class:`CausalLock` hashes every locked value
   before and after each randomisation step; a mismatch raises
   :class:`CausalViolation`, so a buggy randomiser cannot silently corrupt
   labels.

The USD-level randomisers use only ``pxr`` (deterministic, seeded from the
episode's ``materials``/``clutter``/``visibility`` RNG streams, unit-testable
outside Isaac Sim).  Frame capture uses Omniverse Replicator annotators and
the :class:`CausalLabelWriter` (``omni.replicator.core`` required).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade


class CausalViolation(RuntimeError):
    pass


LOCKED_TAGS = ("hidden_cause", "strong_specular", "moved_since_survey")


class CausalLock:
    """Collects prims/attributes that carry causal information and hashes them."""

    def __init__(self, stage: Usd.Stage):
        self.stage = stage
        self.prims: list[str] = []
        self.materials: list[str] = []
        hc = stage.GetRootLayer().customLayerData or {}
        self.factor = hc.get("medortrace:hidden_factor", "none")
        for p in stage.Traverse():
            tags = p.GetAttribute("medortrace:tags")
            tag_s = tags.Get() if tags and tags.IsValid() else ""
            if any(t in (tag_s or "") for t in LOCKED_TAGS):
                self.prims.append(str(p.GetPath()))
                mb = UsdShade.MaterialBindingAPI(p).GetDirectBinding().GetMaterialPath()
                if mb:
                    self.materials.append(str(mb))
            if p.GetAttribute("medortrace:item_id").IsValid():
                self.prims.append(str(p.GetPath()))   # item placements are causal labels
        self.prims = sorted(set(self.prims))
        self.materials = sorted(set(self.materials))

    def signature(self) -> str:
        h = hashlib.sha256()
        cache = UsdGeom.XformCache()
        for path in self.prims:
            p = self.stage.GetPrimAtPath(path)
            m = cache.GetLocalToWorldTransform(p)
            h.update(path.encode())
            h.update(np.round(np.array(m), 5).tobytes())
        for path in self.materials:
            p = self.stage.GetPrimAtPath(path)
            for a in sorted(p.GetAttributes(), key=lambda a: a.GetName()):
                if a.GetName().startswith(("medortrace:", "omni:simready")):
                    h.update(f"{a.GetName()}={a.Get()}".encode())
            for sh in p.GetChildren():
                for a in sorted(sh.GetAttributes(), key=lambda a: a.GetName()):
                    if a.GetName().startswith("inputs:"):
                        h.update(f"{sh.GetName()}.{a.GetName()}={a.Get()}".encode())
        h.update(json.dumps({k: str(v) for k, v in (self.stage.GetRootLayer().customLayerData or {}).items()},
                            sort_keys=True).encode())
        return h.hexdigest()

    def locked(self, path: str) -> bool:
        return path in self.prims or path in self.materials or any(path.startswith(m + "/") for m in self.materials)


class NuisanceRandomizer:
    """Deterministic, causal-lock-aware USD randomiser."""

    def __init__(self, stage: Usd.Stage, rng: np.random.Generator, lock: CausalLock | None = None,
                 strength: float = 1.0):
        self.stage = stage
        self.rng = rng
        self.lock = lock or CausalLock(stage)
        self.k = strength
        self._orig_xf: dict[str, Gf.Matrix4d] = {}

    # -- materials: roughness/metallic/colour within physical ranges -------------
    def materials(self) -> None:
        for mat in self.stage.GetPrimAtPath("/World/Looks").GetChildren():
            if self.lock.locked(str(mat.GetPath())):
                continue
            prev = UsdShade.Shader(self.stage.GetPrimAtPath(mat.GetPath().AppendChild("PreviewSurface")))
            mdl = UsdShade.Shader(self.stage.GetPrimAtPath(mat.GetPath().AppendChild("MDL")))
            r_in = prev.GetInput("roughness")
            if r_in:
                r = float(np.clip(r_in.Get() + self.rng.uniform(-0.12, 0.12) * self.k, 0.02, 1.0))
                r_in.Set(r)
                if mdl and mdl.GetInput("reflection_roughness_constant"):
                    mdl.GetInput("reflection_roughness_constant").Set(r)
            c_in = prev.GetInput("diffuseColor")
            if c_in:
                c = np.clip(np.array(c_in.Get()) * (1 + self.rng.uniform(-0.08, 0.08, 3) * self.k), 0, 1)
                c_in.Set(Gf.Vec3f(*c))
                if mdl and mdl.GetInput("diffuse_color_constant"):
                    mdl.GetInput("diffuse_color_constant").Set(Gf.Vec3f(*c))

    # -- lighting: surgical light intensity / colour temperature / ambient --------
    def lights(self) -> None:
        root = self.stage.GetPrimAtPath("/World/Lights")
        for p in root.GetChildren() if root else []:
            if p.IsA(UsdLux.DiskLight):
                lt = UsdLux.DiskLight(p)
                lt.GetIntensityAttr().Set(float(self.rng.uniform(4000, 16000)))
                lt.GetColorTemperatureAttr().Set(float(self.rng.uniform(3800, 5000)))
            elif p.IsA(UsdLux.RectLight):
                UsdLux.RectLight(p).GetIntensityAttr().Set(float(self.rng.uniform(300, 1500)))

    # -- clutter pose jitter (never the hidden-cause objects) ---------------------
    def clutter(self, max_shift: float = 0.2) -> None:
        root = self.stage.GetPrimAtPath("/World/Furniture")
        for p in root.GetChildren() if root else []:
            kind = p.GetAttribute("medortrace:kind")
            if not kind or kind.Get() != "clutter" or self.lock.locked(str(p.GetPath())):
                continue
            xf = UsdGeom.Xformable(p)
            ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
            t = ops.get("xformOp:translate")
            if t is None:
                continue
            key = str(p.GetPath())
            base = self._orig_xf.setdefault(key, Gf.Vec3d(t.Get()))
            d = self.rng.uniform(-max_shift, max_shift, 2) * self.k
            t.Set(Gf.Vec3d(base[0] + d[0], base[1] + d[1], base[2]))
            rz = ops.get("xformOp:rotateZ")
            if rz is not None:
                rz.Set(float(rz.Get() + self.rng.uniform(-20, 20) * self.k))

    # -- drape / gown colour (appearance only) -------------------------------------
    def fabrics(self) -> None:
        palette = [(0.18, 0.42, 0.55), (0.2, 0.5, 0.35), (0.15, 0.3, 0.6), (0.3, 0.55, 0.6)]
        for name in ("surgical_drape", "gown_fabric"):
            path = f"/World/Looks/{name}"
            if self.lock.locked(path):
                continue
            sh = UsdShade.Shader(self.stage.GetPrimAtPath(path + "/PreviewSurface"))
            if sh:
                sh.GetInput("diffuseColor").Set(Gf.Vec3f(*palette[int(self.rng.integers(0, len(palette)))]))

    def apply(self, which=("materials", "lights", "clutter", "fabrics")) -> dict:
        before = self.lock.signature()
        for w in which:
            getattr(self, w)()
        after = self.lock.signature()
        if before != after:
            raise CausalViolation("a nuisance randomiser modified a causally-locked prim/attribute")
        return {"causal_signature": after}

    @staticmethod
    def fog_settings(rng: np.random.Generator) -> dict:
        """RTX fog (haze) carb settings for the visibility nuisance factor."""
        return {"/rtx/fog/enabled": True, "/rtx/fog/fogColorIntensity": float(rng.uniform(0.0, 0.6)),
                "/rtx/fog/fogStartDist": float(rng.uniform(0.5, 3.0)), "/rtx/fog/fogEndDist": float(rng.uniform(8, 25))}


class CausalLabelWriter:
    """Replicator writer: annotator outputs + per-frame causal/truth labels.

    Output layout (per frame ``NNNNNN``)::

        rgb/NNNNNN.png  semantic/NNNNNN.npy  instance/NNNNNN.npy  depth/NNNNNN.npy
        bbox2d/NNNNNN.json   labels/NNNNNN.json   (scenario, hidden cause, item truth slots,
                                                   fault state, causal signature, camera pose)
    """

    def __init__(self, out_dir: str | Path, label_fn):
        self.out = Path(out_dir)
        self.label_fn = label_fn
        for d in ("rgb", "semantic", "instance", "depth", "bbox2d", "labels"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.i = 0

    def write(self, annotators: dict) -> None:  # pragma: no cover - requires Isaac Sim
        from PIL import Image
        n = f"{self.i:06d}"
        if "rgb" in annotators:
            Image.fromarray(np.asarray(annotators["rgb"].get_data())[..., :3]).save(self.out / "rgb" / f"{n}.png")
        if "semantic_segmentation" in annotators:
            d = annotators["semantic_segmentation"].get_data()
            np.save(self.out / "semantic" / f"{n}.npy", np.asarray(d["data"]))
            (self.out / "semantic" / f"{n}.json").write_text(json.dumps(d.get("info", {}), default=str))
        if "instance_id_segmentation_fast" in annotators:
            d = annotators["instance_id_segmentation_fast"].get_data()
            np.save(self.out / "instance" / f"{n}.npy", np.asarray(d["data"]))
        if "distance_to_image_plane" in annotators:
            np.save(self.out / "depth" / f"{n}.npy", np.asarray(annotators["distance_to_image_plane"].get_data(), np.float16))
        if "bounding_box_2d_tight" in annotators:
            d = annotators["bounding_box_2d_tight"].get_data()
            rows = [{k: (v.item() if hasattr(v, "item") else v) for k, v in zip(d["data"].dtype.names, r)} for r in d["data"]]
            (self.out / "bbox2d" / f"{n}.json").write_text(json.dumps({"boxes": rows, "info": d.get("info", {})}, default=str))
        (self.out / "labels" / f"{n}.json").write_text(json.dumps(self.label_fn(), default=str))
        self.i += 1

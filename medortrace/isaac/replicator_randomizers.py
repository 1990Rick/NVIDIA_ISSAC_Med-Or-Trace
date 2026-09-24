"""Domain randomisation that preserves causal labels + causal-label writer.

Three rules make the synthetic data usable for *counterfactual* learning:

1. **Nuisance only.**  Randomisers act on appearance and non-causal geometry
   (materials within their physical ranges, lighting, haze, clutter poses,
   drape colour, staff gown colour).  They never touch prims or attributes
   that encode the hidden cause: item placements, hidden-cause objects, the
   specular screen *and* the hamper whose mirror image is the CF-B ghost,
   the (possibly) moved cart of CF-D, staged rare-geometry / adversarial
   occluders, and the hidden-cause annotation.  Clutter is also kept out of a
   keep-clear disc around the CF-B aisle point, so jitter can never place a
   *real* obstacle where the specular-ghost arm must have none (furniture already standing in that
   disc is pose-locked instead).
2. **Verified invariance.**  :class:`CausalLock` records a reference hash of
   every locked value (world transforms, visibility, ``medortrace:*`` and
   semantic attributes, bound material shader inputs, non-visual tokens,
   layer metadata, keep-clear intruders).  ``NuisanceRandomizer.apply``
   verifies it before the first and after *every* randomiser, so a buggy
   randomiser - or any out-of-band edit between frames - raises
   :class:`CausalViolation` naming the changed prims.  Deliberate causal
   changes (placing items at their truth slots for a frame time) go through
   ``with lock.causal_edit(): ...``, which re-bases the reference.
3. **Matched pairs.**  Every random draw comes from a generator keyed on
   ``(seed, frame, prim path)`` rather than a shared stream, so a prim's
   nuisance does not depend on which other prims exist or are locked.  Both
   arms of a counterfactual pair (same scene seed, different hidden value)
   therefore receive identical nuisance for the same ``(seed, frame)``, and
   the lock set is arm-invariant (factor anchors lock the same objects in
   both arms).  :func:`nuisance_signature` makes this checkable.

The USD-level randomisers use only ``pxr`` (deterministic, unit-testable
outside Isaac Sim, see ``scripts/isaac/check_randomizers.py``).  RTX renders
the MDL shader, so every appearance change is mirrored onto both the
``PreviewSurface`` and the ``MDL`` shader authored by the scene builder.
Frame capture uses Omniverse Replicator annotators and
:class:`CausalLabelWriter`, which also accepts already-fetched annotator data
(numpy-only, no PIL needed: PNGs are encoded with ``zlib``).
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from medortrace.common.rng import stable_hash
from medortrace.usd.scene_builder import safe as _safe


class CausalViolation(RuntimeError):
    pass


LOCKED_TAGS = ("hidden_cause", "strong_specular", "moved_since_survey")   # pose + material locked
POSE_LOCKED_TAGS = ("rare", "adversarial")                                # staged conditions: pose locked
# objects that carry a factor's evidence in *both* arms of a pair (arm-invariant lock set)
FACTOR_ANCHORS = {
    "CF-B": ("steel_screen", "linen_hamper", "aisle_obstacle"),   # ghost = hamper mirrored in the screen
    "CF-D": ("cart_1",),                                          # the cart that may have moved since survey
}
ANNOTATION_PRIMS = ("/World/Annotations/HiddenCause",)
KEEP_CLEAR_RADIUS = 0.6
DEFAULT_RANDOMIZERS = ("materials", "lights", "clutter", "fabrics")


def _tags(p: Usd.Prim) -> set[str]:
    a = p.GetAttribute("medortrace:tags")
    s = a.Get() if a and a.IsValid() else ""
    return {t.strip() for t in (s or "").split(",") if t.strip()}


def _bound_material(p: Usd.Prim) -> str | None:
    mp = UsdShade.MaterialBindingAPI(p).GetDirectBinding().GetMaterialPath()
    return str(mp) if mp else None


def _hash_matrix(h, m) -> None:
    h.update((np.round(np.array(m, dtype=float), 5) + 0.0).tobytes())   # +0.0 folds -0.0 into 0.0


def _material_digest(stage: Usd.Stage, path: str) -> str:
    h = hashlib.sha256()
    p = stage.GetPrimAtPath(path)
    if not p.IsValid():
        return "<missing>"
    for a in sorted(p.GetAttributes(), key=lambda a: a.GetName()):
        if a.GetName().startswith(("medortrace:", "omni:simready")):
            h.update(f"{a.GetName()}={a.Get()}".encode())
    for sh in sorted(p.GetChildren(), key=lambda c: c.GetName()):
        for a in sorted(sh.GetAttributes(), key=lambda a: a.GetName()):
            if a.GetName().startswith("inputs:"):
                h.update(f"{sh.GetName()}.{a.GetName()}={a.Get()}".encode())
    return h.hexdigest()


def _keep_clear_zones(stage: Usd.Stage) -> list[tuple[np.ndarray, float]]:
    zones = []
    for path in ANNOTATION_PRIMS:
        p = stage.GetPrimAtPath(path)
        a = p.GetAttribute("medortrace:aisle_point") if p.IsValid() else None
        if a and a.IsValid() and a.Get():
            try:
                xy = np.asarray(json.loads(a.Get()), float)[:2]
                zones.append((xy, KEEP_CLEAR_RADIUS))
            except (ValueError, TypeError):
                pass
    return zones


class CausalLock:
    """Collects prims/attributes that carry causal information and hashes them."""

    def __init__(self, stage: Usd.Stage):
        self.stage = stage
        hc = stage.GetRootLayer().customLayerData or {}
        self.factor = str(hc.get("medortrace:hidden_factor", "none"))
        anchors = {_safe(a) for a in FACTOR_ANCHORS.get(self.factor, ())}
        prims: set[str] = set()
        mats: set[str] = set()
        for p in stage.Traverse():
            path = str(p.GetPath())
            tags = _tags(p)
            anchored = p.GetParent().IsValid() and str(p.GetParent().GetPath()) in ("/World/Furniture", "/World/Room") \
                and p.GetName() in anchors
            if tags & set(LOCKED_TAGS) or anchored:
                prims.add(path)
                m = _bound_material(p)
                if m:
                    mats.add(m)
            elif tags & set(POSE_LOCKED_TAGS):
                prims.add(path)
            if p.GetAttribute("medortrace:item_id").IsValid():
                prims.add(path)   # item placements are causal labels
        for path in ANNOTATION_PRIMS:
            if stage.GetPrimAtPath(path).IsValid():
                prims.add(path)
        self.keep_clear = _keep_clear_zones(stage)
        self._prim_set = set(prims)
        # furniture that already stands in a keep-clear disc is part of the evidence there (same in both
        # arms, shared clutter stream): pose-lock it so jitter can neither move it out nor rotate it
        self.keep_clear_locked = self.intruders()
        prims.update(self.keep_clear_locked)
        self.prims = sorted(prims)
        self.materials = sorted(mats)
        self._prim_set = set(self.prims)
        self._mat_set = set(self.materials)
        self.rebase()

    # ------------------------------------------------------------------
    def keep_clear_ok(self, xy) -> bool:
        xy = np.asarray(xy, float)[:2]
        return all(np.linalg.norm(xy - c) >= r for c, r in self.keep_clear)

    def intruders(self, cache: UsdGeom.XformCache | None = None) -> list[str]:
        """Unlocked furniture whose centre lies inside a keep-clear disc."""
        if not self.keep_clear:
            return []
        cache = cache or UsdGeom.XformCache()
        out = []
        root = self.stage.GetPrimAtPath("/World/Furniture")
        for p in root.GetChildren() if root.IsValid() else []:
            path = str(p.GetPath())
            if path in self._prim_set or not p.IsA(UsdGeom.Xformable):
                continue
            c = cache.GetLocalToWorldTransform(p).ExtractTranslation()
            if not self.keep_clear_ok((c[0], c[1])):
                out.append(path)
        return sorted(out)

    def digests(self) -> dict[str, str]:
        cache = UsdGeom.XformCache()
        out: dict[str, str] = {}
        for path in self.prims:
            p = self.stage.GetPrimAtPath(path)
            h = hashlib.sha256(path.encode())
            if not p.IsValid():
                h.update(b"<missing>")
            else:
                if p.IsA(UsdGeom.Xformable):
                    _hash_matrix(h, cache.GetLocalToWorldTransform(p))
                    h.update(str(UsdGeom.Imageable(p).ComputeVisibility()).encode())
                for a in sorted(p.GetAttributes(), key=lambda a: a.GetName()):
                    if a.GetName().startswith(("medortrace:", "semantic")):
                        h.update(f"{a.GetName()}={a.Get()}".encode())
                h.update(str(_bound_material(p)).encode())
            out[path] = h.hexdigest()
        for path in self.materials:
            out[path] = _material_digest(self.stage, path)
        # only our keys: Kit writes renderSettings/cameraSettings into the same dict (e.g. when fog changes)
        layer = {k: str(v) for k, v in (self.stage.GetRootLayer().customLayerData or {}).items()
                 if str(k).startswith("medortrace:")}
        out["<layer>"] = hashlib.sha256(json.dumps(layer, sort_keys=True).encode()).hexdigest()
        out["<keep_clear>"] = hashlib.sha256(json.dumps(self.intruders(cache)).encode()).hexdigest()
        return out

    @staticmethod
    def _combine(d: dict[str, str]) -> str:
        h = hashlib.sha256()
        for k in sorted(d):
            h.update(f"{k}={d[k]};".encode())
        return h.hexdigest()

    def signature(self) -> str:
        return self._combine(self.digests())

    def rebase(self) -> str:
        """Accept the current causal state as the new reference (after a deliberate causal edit)."""
        self._ref_digests = self.digests()
        self.reference = self._combine(self._ref_digests)
        return self.reference

    def verify(self, context: str = "") -> str:
        d = self.digests()
        sig = self._combine(d)
        if sig != self.reference:
            changed = sorted(k for k in set(d) | set(self._ref_digests) if d.get(k) != self._ref_digests.get(k))
            raise CausalViolation(f"{context or 'verify'}: causally-locked state changed: {changed}")
        return sig

    @contextlib.contextmanager
    def causal_edit(self):
        """``with lock.causal_edit(): <set item placements for time t>`` - verified before, re-based after."""
        self.verify("before causal edit")
        yield self
        self.rebase()

    def locked(self, path: str) -> bool:
        path = str(path)
        if path in self._prim_set or path in self._mat_set:
            return True
        return any(path.startswith(m + "/") for m in self.materials) or \
            any(path.startswith(p + "/") for p in self.prims)

    def describe(self) -> dict:
        return {"factor": self.factor, "prims": self.prims, "materials": self.materials,
                "keep_clear": [{"xy": c.tolist(), "radius": r} for c, r in self.keep_clear],
                "keep_clear_locked": self.keep_clear_locked,
                "reference": self.reference}


# ---------------------------------------------------------------------------
def _set_input(shader: UsdShade.Shader, name: str, value) -> None:
    if shader:
        i = shader.GetInput(name)
        if i:
            i.Set(value)


def _clutter_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    root = stage.GetPrimAtPath("/World/Furniture")
    out = []
    for p in sorted(root.GetChildren(), key=lambda c: c.GetName()) if root.IsValid() else []:
        k = p.GetAttribute("medortrace:kind")
        if k and k.IsValid() and k.Get() == "clutter":
            out.append(p)
    return out


def nuisance_signature(stage: Usd.Stage, lock: CausalLock) -> str:
    """Hash of the randomisable (unlocked) state: equal across the arms of a matched pair."""
    h = hashlib.sha256()
    looks = stage.GetPrimAtPath("/World/Looks")
    for m in sorted(looks.GetChildren(), key=lambda c: c.GetName()) if looks.IsValid() else []:
        if not lock.locked(str(m.GetPath())):
            h.update(f"{m.GetPath()}:{_material_digest(stage, str(m.GetPath()))}".encode())
    lights = stage.GetPrimAtPath("/World/Lights")
    for p in sorted(lights.GetChildren(), key=lambda c: c.GetName()) if lights.IsValid() else []:
        for a in sorted(p.GetAttributes(), key=lambda a: a.GetName()):
            if a.GetName().startswith("inputs:"):
                h.update(f"{p.GetPath()}.{a.GetName()}={a.Get()}".encode())
    for p in _clutter_prims(stage):
        if lock.locked(str(p.GetPath())):
            continue
        h.update(str(p.GetPath()).encode())
        _hash_matrix(h, UsdGeom.Xformable(p).GetLocalTransformation())
    return h.hexdigest()


class NuisanceRandomizer:
    """Deterministic, causal-lock-aware USD randomiser (per-prim keyed RNG, see module docstring)."""

    def __init__(self, stage: Usd.Stage, rng: np.random.Generator | None = None, lock: CausalLock | None = None,
                 strength: float = 1.0, seed: int | None = None):
        if seed is None:
            if rng is None:
                raise ValueError("NuisanceRandomizer needs an rng or an explicit seed")
            seed = int(rng.integers(0, 2**63 - 1))
        self.stage = stage
        self.seed = int(seed) & (2**63 - 1)
        self.lock = lock or CausalLock(stage)
        self.k = float(strength)
        self.frame = 0
        self._orig: dict[str, object] = {}

    def _rng(self, key: str) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, int(self.frame), stable_hash(key)]))

    def _base(self, key: str, getter):
        """Authored value captured on first touch: randomisation never accumulates across frames."""
        if key not in self._orig:
            self._orig[key] = getter()
        return self._orig[key]

    # -- materials: roughness / colour within physical ranges --------------------
    def materials(self) -> None:
        looks = self.stage.GetPrimAtPath("/World/Looks")
        for mat in sorted(looks.GetChildren(), key=lambda c: c.GetName()) if looks.IsValid() else []:
            path = str(mat.GetPath())
            g = self._rng("material:" + path)
            dr = float(g.uniform(-0.12, 0.12)) * self.k
            dc = g.uniform(-0.08, 0.08, 3) * self.k
            if self.lock.locked(path):
                continue
            prev = UsdShade.Shader(self.stage.GetPrimAtPath(path + "/PreviewSurface"))
            mdl = UsdShade.Shader(self.stage.GetPrimAtPath(path + "/MDL"))
            r_in = prev.GetInput("roughness") if prev else None
            if r_in and r_in.Get() is not None:
                r0 = float(self._base(path + ".roughness", lambda: float(r_in.Get())))
                r = float(np.clip(r0 + dr, 0.02, 1.0))
                r_in.Set(r)
                _set_input(mdl, "reflection_roughness_constant", r)
                _set_input(mdl, "frosting_roughness", r)
            c_in = prev.GetInput("diffuseColor") if prev else None
            if c_in and c_in.Get() is not None:
                c0 = np.asarray(self._base(path + ".diffuseColor", lambda: np.array(c_in.Get(), float)))
                c = Gf.Vec3f(*map(float, np.clip(c0 * (1 + dc), 0, 1)))
                c_in.Set(c)
                _set_input(mdl, "diffuse_color_constant", c)

    # -- lighting: surgical light intensity / colour temperature / ambient --------
    def lights(self) -> None:
        root = self.stage.GetPrimAtPath("/World/Lights")
        for p in sorted(root.GetChildren(), key=lambda c: c.GetName()) if root.IsValid() else []:
            path = str(p.GetPath())
            g = self._rng("light:" + path)
            s = float(np.exp(g.uniform(-0.4, 0.4) * self.k))
            ct = float(g.uniform(3800, 5000))
            if self.lock.locked(path):
                continue
            if p.IsA(UsdLux.DiskLight):
                lt = UsdLux.DiskLight(p)
                i0 = float(self._base(path + ".intensity", lambda: float(lt.GetIntensityAttr().Get() or 8000.0)))
                lt.GetIntensityAttr().Set(float(np.clip(i0 * s, 2000.0, 20000.0)))
                lt.CreateEnableColorTemperatureAttr(True)
                lt.CreateColorTemperatureAttr(ct)
            elif p.IsA(UsdLux.RectLight):
                lt = UsdLux.RectLight(p)
                i0 = float(self._base(path + ".intensity", lambda: float(lt.GetIntensityAttr().Get() or 600.0)))
                lt.GetIntensityAttr().Set(float(np.clip(i0 * s, 150.0, 2500.0)))

    # -- clutter pose jitter (never locked objects, never into keep-clear zones) --
    def clutter(self, max_shift: float = 0.2) -> None:
        for p in _clutter_prims(self.stage):
            path = str(p.GetPath())
            g = self._rng("clutter:" + path)
            d = g.uniform(-max_shift, max_shift, 2) * self.k
            drot = float(g.uniform(-20, 20)) * self.k
            if self.lock.locked(path):
                continue
            xf = UsdGeom.Xformable(p)
            ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
            t = ops.get("xformOp:translate")
            if t is None:
                continue
            base = self._base(path + ".translate", lambda: Gf.Vec3d(t.Get()))
            new = Gf.Vec3d(base[0] + d[0], base[1] + d[1], base[2])
            t.Set(new if self.lock.keep_clear_ok((new[0], new[1])) else Gf.Vec3d(base))
            rz = ops.get("xformOp:rotateZ")
            if rz is None:
                rz = xf.AddRotateZOp()
                rz.Set(0.0)
                order = [t, rz] + [op for op in xf.GetOrderedXformOps() if op.GetOpName() not in
                                   ("xformOp:translate", "xformOp:rotateZ")]
                xf.SetXformOpOrder(order)
            r0 = float(self._base(path + ".rotateZ", lambda: float(rz.Get() or 0.0)))
            rz.Set(r0 + drot)

    # -- drape / gown colour (appearance only) -------------------------------------
    def fabrics(self) -> None:
        palette = [(0.18, 0.42, 0.55), (0.2, 0.5, 0.35), (0.15, 0.3, 0.6), (0.3, 0.55, 0.6)]
        for name in ("surgical_drape", "gown_fabric"):
            path = f"/World/Looks/{_safe(name)}"
            g = self._rng("fabric:" + path)
            c = Gf.Vec3f(*palette[int(g.integers(0, len(palette)))])
            if self.lock.locked(path):
                continue
            prev = UsdShade.Shader(self.stage.GetPrimAtPath(path + "/PreviewSurface"))
            if prev and prev.GetInput("diffuseColor"):
                prev.GetInput("diffuseColor").Set(c)
                _set_input(UsdShade.Shader(self.stage.GetPrimAtPath(path + "/MDL")), "diffuse_color_constant", c)

    # ------------------------------------------------------------------
    def apply(self, which=DEFAULT_RANDOMIZERS, frame: int | None = None, haze: float | None = None) -> dict:
        """Randomise nuisance for ``frame`` (default: the next frame) with the causal lock enforced."""
        if frame is not None:
            self.frame = int(frame)
        self.lock.verify("before nuisance randomisation")
        for w in which:
            getattr(self, w)()
            self.lock.verify(f"nuisance randomiser {w!r}")
        info = {"causal_signature": self.lock.reference, "nuisance_seed": self.seed, "nuisance_frame": self.frame,
                "nuisance_signature": nuisance_signature(self.stage, self.lock), "randomizers": list(which),
                "fog": self.fog_settings(self._rng("fog"), haze)}
        self.frame += 1
        return info

    @staticmethod
    def fog_settings(rng: np.random.Generator, haze: float | None = None) -> dict:
        """RTX fog (haze) carb settings for the visibility nuisance factor (setting names: see README)."""
        if haze is None:
            dens = float(rng.uniform(0.0, 0.6))
        else:
            dens = float(np.clip(4.0 * haze * rng.uniform(0.7, 1.3), 0.0, 1.0))
        return {"/rtx/fog/enabled": True, "/rtx/fog/fogColorIntensity": dens,
                "/rtx/fog/fogStartDist": float(rng.uniform(0.5, 3.0)), "/rtx/fog/fogEndDist": float(rng.uniform(8, 25))}


# ---------------------------------------------------------------------------
def to_jsonable(o):
    """numpy / Gf / Sdf values -> JSON-serialisable Python objects (recursively)."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        if o.dtype.names:
            return [{n: to_jsonable(r[n]) for n in o.dtype.names} for r in np.atleast_1d(o)]
        return to_jsonable(o.tolist())
    if isinstance(o, np.generic):
        return to_jsonable(o.item())
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    if isinstance(o, bytes):
        return o.decode(errors="replace")
    if isinstance(o, Sdf.Path):
        return str(o)
    try:
        return [to_jsonable(v) for v in o]      # Gf vectors / matrices
    except TypeError:
        return str(o)


def write_png(path: str | Path, img: np.ndarray) -> None:
    """Minimal 8-bit RGB/RGBA/grey PNG encoder (zlib only)."""
    a = np.ascontiguousarray(np.asarray(img))
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = a[..., None]
    h, w, c = a.shape
    color = {1: 0, 3: 2, 4: 6}[c]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def _unwrap(d):
    """Annotator output -> (array, info)."""
    if isinstance(d, dict):
        return d.get("data"), d.get("info", {})
    return d, {}


class CausalLabelWriter:
    """Annotator outputs + per-frame causal/truth labels.

    Driven explicitly after ``rep.orchestrator.step()`` (so each frame's labels
    are computed from the same causal state that was rendered).  Output layout
    (per frame ``<frame_id>``, default ``NNNNNN``)::

        rgb/<id>.png  semantic/<id>.npy+.json  instance/<id>.npy+.json  depth/<id>.npy (float16)
        bbox2d/<id>.json  camera/<id>.json  labels/<id>.json
          labels: scenario, hidden cause, item truth slots at the frame time, fault state,
                  causal + nuisance signatures, camera pose, relative paths of the files above

    ``annotators`` maps annotator name -> Replicator annotator (``get_data()``) or already-fetched data.
    ``label_fn`` takes no argument or the frame id.
    """

    DIRS = {"rgb": "rgb", "semantic_segmentation": "semantic", "instance_id_segmentation_fast": "instance",
            "instance_segmentation_fast": "instance", "distance_to_image_plane": "depth",
            "bounding_box_2d_tight": "bbox2d", "camera_params": "camera"}

    def __init__(self, out_dir: str | Path, label_fn=None):
        self.out = Path(out_dir)
        self.label_fn = label_fn
        for d in set(self.DIRS.values()) | {"labels"}:
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.i = 0

    def _labels(self, frame_id: str) -> dict:
        if self.label_fn is None:
            return {}
        try:
            n_args = len(inspect.signature(self.label_fn).parameters)
        except (TypeError, ValueError):
            n_args = 0
        return dict(self.label_fn(frame_id) if n_args else self.label_fn())

    def write(self, annotators: dict, frame_id: str | None = None, labels: dict | None = None) -> dict:
        n = frame_id or f"{self.i:06d}"
        data = {k: (a.get_data() if hasattr(a, "get_data") else a) for k, a in annotators.items()}
        files: dict[str, str] = {}

        def rel(sub: str, ext: str) -> Path:
            files[sub] = f"{sub}/{n}{ext}"
            return self.out / sub / f"{n}{ext}"

        if data.get("rgb") is not None:
            arr, _ = _unwrap(data["rgb"])
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.size:
                write_png(rel("rgb", ".png"), arr[..., :3])
        for key in ("semantic_segmentation", "instance_id_segmentation_fast", "instance_segmentation_fast"):
            if data.get(key) is not None:
                arr, info = _unwrap(data[key])
                sub = self.DIRS[key]
                np.save(rel(sub, ".npy"), np.asarray(arr))
                (self.out / sub / f"{n}.json").write_text(json.dumps(to_jsonable(info)))
        if data.get("distance_to_image_plane") is not None:
            arr, _ = _unwrap(data["distance_to_image_plane"])
            np.save(rel("depth", ".npy"), np.asarray(arr, np.float32).astype(np.float16))
        if data.get("bounding_box_2d_tight") is not None:
            arr, info = _unwrap(data["bounding_box_2d_tight"])
            rel("bbox2d", ".json").write_text(json.dumps({"boxes": to_jsonable(arr if arr is not None else []),
                                                          "info": to_jsonable(info)}))
        if data.get("camera_params") is not None:
            rel("camera", ".json").write_text(json.dumps(to_jsonable(data["camera_params"])))
        lab = {**self._labels(n), **(labels or {})}
        lab.update({"frame_id": n, "files": dict(files)})
        files["labels"] = f"labels/{n}.json"
        (self.out / "labels" / f"{n}.json").write_text(json.dumps(to_jsonable(lab), indent=1))
        self.i += 1
        return {"frame_id": n, "files": files}

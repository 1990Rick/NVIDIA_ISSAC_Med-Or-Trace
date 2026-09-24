"""Time-synchronised assembly of per-sensor messages into a SensorBundle (no ROS imports).

On ROS 2 every sensor arrives on its own topic, at its own rate and with its own
latency, while :meth:`AutonomyStack.step` wants one
:class:`~medortrace.common.msgs.SensorBundle` per control tick - exactly what
the in-process simulator backends hand it.  ``SensorBundleAssembler`` closes
that gap.  Callbacks :meth:`push` converted messages as they arrive (stamping
``header.recv_stamp`` with the node clock); the control timer calls
:meth:`assemble` at the control rate.  Each channel has one of three delivery
semantics, chosen so the stack sees the same information it would get
in-process:

``consume-once``  lidar, camera, radar, acoustic, landmarks
    Measurements that are *integrated* as evidence (occupancy ray casting,
    item-belief log-likelihood updates, EKF landmark updates).  Delivering one
    twice would double-count evidence, so each message is delivered at most
    once.  If several arrived since the last tick, the newest wins (the
    others are counted as ``superseded``).
``sample-and-hold``  odom, contact, battery
    Rates / levels.  The latest sample is re-delivered on later ticks while it
    is younger than ``max_age_s`` (zero-order hold, like the in-process loop);
    older samples are withheld so the stack sees the dropout (the EKF inflates
    process noise without odometry, the supervisor sees missing data).
    Contact latches the strongest sample since the previous tick so a short
    bumper hit between two ticks is not lost.
``stream``  imu, workflow
    Every IMU sample received since the previous tick (bounded window), in
    stamp order; every workflow event exactly once, de-duplicated on
    ``event_id`` (transient-local topics replay history to late joiners).

Staleness is judged on the *receive* time (host / sim clock), never on the
sensor's own stamp: a skewed sensor clock is a fault the stack's
``TimeSyncMonitor`` must see (it compares ``stamp`` with ``recv_stamp``), not
something the middleware may silently hide.  Messages whose receive time is
later than the tick are kept for the next tick.  A clock that jumps backwards
(simulation reset, bag loop) resets all buffers.

Every delivered header gets a session-unique ``seq``: the stack names its
provenance evidence ``"<sensor>:<seq>"`` and the hash chain requires unique
node ids, while ROS 2 headers carry no sequence number.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from medortrace.common.msgs import SensorBundle, WorkflowEvent

CONSUME_ONCE = ("lidar", "camera", "radar", "acoustic", "landmarks")
SAMPLE_HOLD = ("odom", "contact", "battery")
STREAMS = ("imu", "workflow")
CHANNELS = CONSUME_ONCE + SAMPLE_HOLD + STREAMS

DEFAULT_MAX_AGE_S = {
    "lidar": 0.5,        # = TimeSyncMonitor.max_age_s: older scans would be rejected by the stack anyway
    "camera": 0.5,
    "radar": 0.3,
    "acoustic": 1.0,     # 2 Hz probe
    "landmarks": 0.5,
    "odom": 0.25,        # hold odometry for at most ~2 ticks, then report a dropout
    "contact": 0.25,
    "battery": 60.0,
    "imu": 0.5,
    "workflow": float("inf"),   # reports arrive late by nature; the stack reasons about event time
}


@dataclass
class AssemblerConfig:
    control_rate_hz: float = 10.0
    max_age_s: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MAX_AGE_S))
    max_imu_samples: int = 200
    dt_clamp: tuple[float, float] = (0.25, 5.0)    # dt limits as multiples of the nominal period
    reset_on_backwards_s: float = 0.5              # clock moved back by more than this -> reset

    @property
    def period(self) -> float:
        return 1.0 / self.control_rate_hz

    @staticmethod
    def from_dict(d: dict | None) -> "AssemblerConfig":
        d = dict(d or {})
        c = AssemblerConfig()
        c.control_rate_hz = float(d.get("control_rate_hz", c.control_rate_hz))
        c.max_age_s.update({k: float(v) for k, v in (d.get("max_age_s") or {}).items()})
        c.max_imu_samples = int(d.get("max_imu_samples", c.max_imu_samples))
        if "dt_clamp" in d:
            c.dt_clamp = (float(d["dt_clamp"][0]), float(d["dt_clamp"][1]))
        c.reset_on_backwards_s = float(d.get("reset_on_backwards_s", c.reset_on_backwards_s))
        unknown = set(c.max_age_s) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown assembler channels in max_age_s: {sorted(unknown)}")
        return c


@dataclass
class ChannelStats:
    received: int = 0
    delivered: int = 0
    stale: int = 0          # withheld / dropped because older than max_age_s
    superseded: int = 0     # consume-once message replaced by a newer one before a tick
    held: int = 0           # sample-and-hold re-deliveries
    duplicates: int = 0     # workflow events already seen
    last_recv: float | None = None
    rate_hz: float = 0.0    # EMA of the arrival rate

    def as_dict(self, now: float) -> dict:
        return {"received": self.received, "delivered": self.delivered, "stale": self.stale,
                "superseded": self.superseded, "held": self.held, "duplicates": self.duplicates,
                "age_s": None if self.last_recv is None else max(0.0, now - self.last_recv),
                "rate_hz": round(self.rate_hz, 3)}


@dataclass
class _Entry:
    recv: float
    msg: object
    delivered: bool = False


class SensorBundleAssembler:
    """Collects per-sensor messages and emits one SensorBundle per control tick."""

    def __init__(self, cfg: AssemblerConfig | None = None):
        self.cfg = cfg or AssemblerConfig()
        self.stats: dict[str, ChannelStats] = {c: ChannelStats() for c in CHANNELS}
        self.n_resets = 0
        self._seq = 0
        self.reset(clear_stats=False)

    # ------------------------------------------------------------------
    def reset(self, clear_stats: bool = True) -> None:
        """Drop all buffered data (clock jumped back, new episode)."""
        self._pending: dict[str, list[_Entry]] = {c: [] for c in CONSUME_ONCE}
        self._hold: dict[str, _Entry | None] = {c: None for c in SAMPLE_HOLD}
        self._contact_new: list[_Entry] = []
        self._imu: list[_Entry] = []
        self._workflow: list[_Entry] = []
        self._seen_events = set()
        self.last_tick: float | None = None
        if clear_stats:
            self.stats = {c: ChannelStats() for c in CHANNELS}

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # ------------------------------------------------------------------
    def push(self, channel: str, msg, recv_t: float | None = None) -> None:
        """Buffer one converted message.

        ``msg`` is the medortrace dataclass (``LidarScan``, ``CameraFrame``, ...,
        ``ImuSample``, ``WorkflowEvent``) or, for ``battery``, the energy in Wh.
        ``recv_t`` is the receive time on the stack clock; when omitted the
        message's ``header.recv_stamp`` is used (``WorkflowEvent``: ``t``).
        The caller's object is never mutated.
        """
        if channel not in CHANNELS:
            raise KeyError(f"unknown channel {channel!r}; expected one of {CHANNELS}")
        if channel == "battery":
            if recv_t is None:
                raise ValueError("battery samples need an explicit recv_t")
            entry = _Entry(float(recv_t), float(msg))
        elif channel == "workflow":
            ev: WorkflowEvent = copy.copy(msg)
            if not ev.event_id:
                ev.event_id = f"ros_{self._next_seq()}"
            entry = _Entry(float(ev.t if recv_t is None else recv_t), ev)
        else:
            m = copy.copy(msg)
            m.header = copy.copy(msg.header)
            if recv_t is not None:
                m.header.recv_stamp = float(recv_t)
            m.header.seq = self._next_seq()
            entry = _Entry(float(m.header.recv_stamp), m)
        st = self.stats[channel]
        if st.last_recv is not None and entry.recv > st.last_recv:
            inst = 1.0 / (entry.recv - st.last_recv)
            st.rate_hz = inst if st.rate_hz == 0.0 else 0.9 * st.rate_hz + 0.1 * inst
        st.received += 1
        st.last_recv = entry.recv if st.last_recv is None else max(st.last_recv, entry.recv)
        if channel in CONSUME_ONCE:
            self._pending[channel].append(entry)
        elif channel == "contact":
            self._contact_new.append(entry)
        elif channel in SAMPLE_HOLD:
            cur = self._hold[channel]
            if cur is None or entry.recv >= cur.recv:
                self._hold[channel] = entry
        elif channel == "imu":
            self._imu.append(entry)
            if len(self._imu) > 4 * self.cfg.max_imu_samples:
                del self._imu[: len(self._imu) - self.cfg.max_imu_samples]
        else:
            self._workflow.append(entry)

    # ------------------------------------------------------------------
    def _fresh(self, channel: str, recv: float, now: float) -> bool:
        return now - recv <= self.cfg.max_age_s.get(channel, float("inf")) + 1e-9

    def assemble(self, now: float) -> tuple[SensorBundle, float]:
        """Build the bundle for the control tick at ``now`` (stack clock) and its dt."""
        eps = 1e-9
        if self.last_tick is not None and now < self.last_tick - self.cfg.reset_on_backwards_s:
            self.n_resets += 1
            self.reset(clear_stats=False)
        T = self.cfg.period
        if self.last_tick is None:
            dt = T
        else:
            lo, hi = self.cfg.dt_clamp
            dt = min(max(now - self.last_tick, lo * T), hi * T)
        self.last_tick = now
        b = SensorBundle(t=float(now))
        # ---- consume-once: newest due message, if fresh ----------------------------------------------------------
        for ch in CONSUME_ONCE:
            q = self._pending[ch]
            due = [e for e in q if e.recv <= now + eps]
            if not due:
                continue
            self._pending[ch] = [e for e in q if e.recv > now + eps]
            newest = max(due, key=lambda e: e.recv)
            st = self.stats[ch]
            st.superseded += len(due) - 1
            if self._fresh(ch, newest.recv, now):
                setattr(b, ch, newest.msg)
                st.delivered += 1
            else:
                st.stale += 1
        # ---- sample-and-hold -----------------------------------------------------------------------------------
        new_contact = [e for e in self._contact_new if e.recv <= now + eps]
        self._contact_new = [e for e in self._contact_new if e.recv > now + eps]
        if new_contact:
            latest = max(new_contact, key=lambda e: e.recv)
            peak = max(new_contact, key=lambda e: (bool(e.msg.in_contact), float(e.msg.force_n), e.recv))
            chosen = peak if peak.msg.in_contact else latest
            cur = self._hold["contact"]
            if cur is None or latest.recv >= cur.recv:
                self._hold["contact"] = latest
            if self._fresh("contact", chosen.recv, now):
                b.contact = chosen.msg
                chosen.delivered = True
                self.stats["contact"].delivered += 1
            else:
                self.stats["contact"].stale += 1
        else:
            self._deliver_hold("contact", b, now, eps)
        self._deliver_hold("odom", b, now, eps)
        self._deliver_hold("battery", b, now, eps)
        # ---- IMU stream ------------------------------------------------------------------------------------------
        due = [e for e in self._imu if e.recv <= now + eps]
        self._imu = [e for e in self._imu if e.recv > now + eps]
        fresh = [e for e in due if self._fresh("imu", e.recv, now)]
        self.stats["imu"].stale += len(due) - len(fresh)
        fresh.sort(key=lambda e: e.msg.header.stamp)
        if len(fresh) > self.cfg.max_imu_samples:
            self.stats["imu"].stale += len(fresh) - self.cfg.max_imu_samples
            fresh = fresh[-self.cfg.max_imu_samples:]
        b.imu = [e.msg for e in fresh]
        self.stats["imu"].delivered += len(fresh)
        # ---- workflow stream (exactly once, event-time order) --------------------------------------------------
        due = [e for e in self._workflow if e.recv <= now + eps]
        self._workflow = [e for e in self._workflow if e.recv > now + eps]
        evs = []
        for e in sorted(due, key=lambda e: e.msg.t):
            if e.msg.event_id in self._seen_events:
                self.stats["workflow"].duplicates += 1
                continue
            self._seen_events.add(e.msg.event_id)
            evs.append(e.msg)
        b.workflow = evs
        self.stats["workflow"].delivered += len(evs)
        return b, float(dt)

    def _deliver_hold(self, ch: str, b: SensorBundle, now: float, eps: float) -> None:
        cur = self._hold[ch]
        if cur is None or cur.recv > now + eps:
            return
        st = self.stats[ch]
        if not self._fresh(ch, cur.recv, now):
            st.stale += 1
            return
        if cur.delivered:
            st.held += 1
        cur.delivered = True
        st.delivered += 1
        if ch == "battery":
            b.battery_wh = float(cur.msg)
        else:
            setattr(b, ch, cur.msg)

    # ------------------------------------------------------------------
    def ready(self, required: tuple[str, ...] = ("odom",)) -> bool:
        """True once every required channel has received at least one message."""
        return all(self.stats[c].received > 0 for c in required)

    def health(self, now: float) -> dict[str, dict]:
        return {c: s.as_dict(now) for c, s in self.stats.items()}


def explode_bundle(b: SensorBundle) -> list[tuple[str, object]]:
    """Inverse of :meth:`SensorBundleAssembler.assemble`: the per-channel messages of a bundle.

    Used by ``sim_bridge_node`` to publish a simulator bundle topic by topic and
    by the tests to replay simulator output through the assembler.
    """
    out: list[tuple[str, object]] = []
    for ch in CONSUME_ONCE:
        m = getattr(b, ch)
        if m is not None:
            out.append((ch, m))
    if b.odom is not None:
        out.append(("odom", b.odom))
    if b.contact is not None:
        out.append(("contact", b.contact))
    if b.battery_wh is not None:
        out.append(("battery", float(b.battery_wh)))
    out += [("imu", s) for s in b.imu]
    out += [("workflow", e) for e in b.workflow]
    return out

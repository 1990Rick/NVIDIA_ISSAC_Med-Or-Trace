"""OR workflow gateway: replays an OR log / voice transcript (JSONL) onto ``/medortrace/workflow/events``.

In the OR the robot learns about custody changes from the OR information system
(case log, count sheet) and from staff call-outs ("passing sponge two to the
surgeon").  This node is the adapter between such sources and the stack's
:class:`~medortrace.common.msgs.WorkflowEvent`: every accepted record becomes a
``medortrace_msgs/WorkflowEvent`` stamped with the time the event *happened*
and published when it was *reported* (``reported_t``, defaulting to ``t``), so
the stack sees the same report latency as in the field.  Raw transcript lines
are echoed on ``/medortrace/workflow/transcript`` for the audit record.

Accepted JSONL records (one JSON object per line; ``t`` in mission seconds, or
``stamp`` in absolute ROS seconds):

* native     ``{"t", "type", "item_id", "src", "dst", "reporter", "confidence", "event_id", "payload",
               "reported_t"}``
* dataset    ``{"kind": "workflow_log", "t", "type", "item", "src", "dst", "id"}`` - the
             ``events.jsonl`` written by ``medortrace.data.writer``; every other ``kind`` in that
             file (``truth_move``, ``verdict``, ...) is ignored: the robot never sees the truth;
* voice      ``{"kind": "voice" | "transcript", "t", "text", "speaker", "asr_confidence"}`` - parsed
             with a small keyword grammar and the vocabulary in ``config/voice_vocabulary.yaml``.

Record parsing is ROS-free (:func:`parse_record`, :func:`parse_transcript`).
``mode: follow`` tails a live file instead of replaying a finished one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from medortrace.common.msgs import WorkflowEvent
from medortrace.common.msgs import WorkflowEventType as WT

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "first": 1, "second": 2, "third": 3, "fourth": 4}
VERB_TYPES = {"count": WT.COUNT, "specimen_out": WT.SPECIMEN_OUT, "discard": WT.DISCARD, "open": WT.OPEN,
              "place": WT.PLACE, "handoff": WT.HANDOFF}
REPORTER_ROLES = {"circulator": "circulating_nurse"}


def default_vocabulary_path() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory("medortrace_ros")) / "config" / "voice_vocabulary.yaml"
        if p.is_file():
            return p
    except Exception:  # noqa: BLE001 - not in a ROS environment
        pass
    return Path(__file__).resolve().parents[1] / "config" / "voice_vocabulary.yaml"


def load_vocabulary(path: str | Path | None = None) -> dict:
    import yaml

    with open(path or default_vocabulary_path()) as f:
        return yaml.safe_load(f) or {}


# =====================================================================================================================
def _normalise(text: str) -> str:
    s = re.sub(r"[^a-z0-9_ ]+", " ", text.lower())
    s = " ".join(str(NUMBER_WORDS.get(w, w)) for w in s.split())
    return f" {s} "


def _longest(s: str, table: dict[str, str], after: int = 0) -> tuple[int, str, str] | None:
    """Earliest (then longest) phrase of ``table`` occurring in ``s`` at or after ``after``."""
    best = None
    for phrase in sorted(table, key=len, reverse=True):
        k = s.find(f" {phrase} ", after)
        if k >= 0 and (best is None or k < best[0]):
            best = (k, phrase, table[phrase])
    return best


def _place(s: str, vocab: dict, after: int = 0) -> tuple[int, str] | None:
    """Earliest slot mention after ``after``: a place phrase or a person (-> their hand)."""
    places = {**{p: v for p, v in vocab.get("places", {}).items()},
              **{r: f"hand:{v}" for r, v in vocab.get("roles", {}).items()}}
    hit = _longest(s, places, after)
    return None if hit is None else (hit[0], hit[2])


def parse_transcript(text: str, vocab: dict, speaker: str | None = None) -> dict | None:
    """Keyword grammar for OR call-outs -> dict(type, item_id, src, dst, payload) or None.

    ``"passing sponge two to the surgeon"`` -> handoff sponge_2 -> hand:surgeon;
    ``"clamp back on the mayo"`` -> place clamp_1 -> mayo:top;
    ``"sponge 3 into kick bucket 2"`` -> discard sponge_3 -> kick_bucket_2:inside;
    ``"starting the final count"`` -> count (payload phase final_count).
    """
    s = _normalise(text)
    typ = None
    for key, phrases in vocab.get("verbs", {}).items():
        if any(f" {p} " in s for p in phrases):
            typ = VERB_TYPES[key]
            break
    if typ == WT.COUNT:
        phase = next((ph for ph, words in vocab.get("count_phases", {}).items() if any(f" {w} " in s for w in words)),
                     "first_closing_count")
        return {"type": WT.COUNT, "item_id": None, "src": None, "dst": None, "payload": {"phase": phase}}
    it = _longest(s, vocab.get("items", {}))
    if it is None:
        return None
    k_item, phrase, tmpl = it
    m = re.match(r" (\d+) ", s[k_item + len(phrase) + 1:])
    item_id = tmpl.format(n=int(m.group(1)) if m else 1)
    src = dst = None
    mf = re.search(r" from ", s)
    if mf:
        hit = _place(s, vocab, mf.start())
        src = hit[1] if hit else None
    mt = None
    for prep in (" into ", " onto ", " to ", " on ", " in ", " back on "):
        k = s.find(prep, k_item)
        if k >= 0 and (mt is None or k < mt):
            mt = k
    if mt is not None:
        hit = _place(s, vocab, mt)
        dst = hit[1] if hit else None
    if dst is None:                                        # "bucket" without a preposition
        hit = _place(s, vocab, k_item + len(phrase))
        if hit and hit[1] != src:
            dst = hit[1]
    if typ is None:
        typ = WT.DISCARD if dst and ("bucket" in dst or "waste" in dst) else WT.HANDOFF if dst else None
    if typ is None:
        return None
    if typ == WT.SPECIMEN_OUT:
        item_id = item_id if item_id.startswith("specimen") else "specimen_1"
        src = src or "elsewhere"
        dst = dst or (f"hand:{speaker}" if speaker else "hand:surgeon")
    if typ == WT.HANDOFF and src is None and speaker and dst != f"hand:{speaker}":
        src = f"hand:{speaker}"                            # "passing X to the surgeon" said by the scrub nurse
    return {"type": typ, "item_id": item_id, "src": src, "dst": dst, "payload": {}}


@dataclass
class ParsedRecord:
    event: WorkflowEvent | None
    publish_t: float                  # mission time at which the report becomes available
    source: str                       # or_log | voice
    text: str | None = None           # raw transcript (voice records)


def parse_record(rec: dict, vocab: dict, index: int, t0: float = 0.0,
                 voice_confidence_scale: float = 0.85) -> ParsedRecord | None:
    """One JSONL record -> ParsedRecord (``None`` for records the robot must not / cannot use)."""
    kind = rec.get("kind")
    if "stamp" in rec and "t" not in rec:
        rec = {**rec, "t": float(rec["stamp"]) - t0}
    if "t" not in rec:
        return None
    t = float(rec["t"])
    reported = float(rec.get("reported_t", t))
    if kind in ("voice", "transcript"):
        text = str(rec.get("text", ""))
        speaker = rec.get("speaker")
        parsed = parse_transcript(text, vocab, speaker)
        if parsed is None:
            return ParsedRecord(None, reported, "voice", text)
        conf = float(rec.get("asr_confidence", 0.9)) * voice_confidence_scale
        reporter = REPORTER_ROLES.get(speaker, speaker) or "unknown"
        ev = WorkflowEvent(t, parsed["type"], parsed["item_id"], parsed["src"], parsed["dst"], reporter,
                           float(min(max(conf, 0.0), 1.0)), str(rec.get("id", f"voice_{index:04d}")),
                           {**parsed["payload"], "transcript": text})
        return ParsedRecord(ev, reported, "voice", text)
    if kind not in (None, "workflow_log", "or_log"):
        return None                                        # truth_move, verdict, safety, ...: never fed back
    try:
        typ = WT(rec["type"])
    except (KeyError, ValueError):
        return None
    ev = WorkflowEvent(t, typ, rec.get("item_id", rec.get("item")) or None, rec.get("src") or None,
                       rec.get("dst") or None, rec.get("reporter", "circulating_nurse"),
                       float(rec.get("confidence", 0.9)),
                       str(rec.get("event_id", rec.get("id", f"log_{index:04d}"))), dict(rec.get("payload") or {}))
    return ParsedRecord(ev, reported, "or_log")


def parse_jsonl(lines, vocab: dict, t0: float = 0.0, start_index: int = 0) -> list[ParsedRecord]:
    out = []
    for k, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        r = parse_record(rec, vocab, start_index + k, t0)
        if r is not None:
            out.append(r)
    return out


# =====================================================================================================================
class WorkflowGatewayNode:
    def __init__(self, node):
        from medortrace_msgs.msg import WorkflowEvent as RosWorkflowEvent
        from std_msgs.msg import String

        from medortrace_ros.qos import load_qos_config, qos_profile
        from medortrace_ros.topics import TOPICS

        self.node = node
        self.log = node.get_logger()

        def p(name, default):
            from medortrace_ros.node_utils import param
            return param(node, name, default)

        self.path = Path(str(p("log_file", "")))
        self.mode = str(p("mode", "replay"))                  # replay | follow
        self.shift = float(p("time_shift_s", 0.0))            # log time 0 == mission time shift
        self.origin = str(p("time_origin", "mission"))         # mission | start | zero
        self.vocab = load_vocabulary(str(p("vocabulary_file", "")) or None)
        qcfg = load_qos_config(str(p("qos_file", "")) or None)
        self.pub = node.create_publisher(RosWorkflowEvent, TOPICS["workflow"], qos_profile("workflow", qcfg))
        self.pub_text = node.create_publisher(String, TOPICS["workflow_transcript"],
                                              qos_profile("workflow_transcript", qcfg))
        self.t0: float | None = None
        if self.origin == "zero":
            self.t0 = 0.0
        elif self.origin == "mission":
            node.create_subscription(String, TOPICS["mission"], self._on_mission, qos_profile("mission", qcfg))
        self.queue: list[ParsedRecord] = []
        self._offset = 0
        self._n = 0
        self._published = 0
        self.timer = node.create_timer(0.05, self._on_timer)
        self.log.info(f"workflow gateway: {self.path} mode={self.mode} time_origin={self.origin}")

    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _on_mission(self, msg) -> None:
        if self.t0 is None:
            try:
                self.t0 = float(json.loads(msg.data).get("t0", self._now()))
            except (json.JSONDecodeError, TypeError, ValueError):
                self.t0 = self._now()

    def _read_new(self) -> None:
        if not self.path.is_file():
            return
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read()
        if not chunk:
            return
        if self.mode == "follow":
            complete, sep, _ = chunk.rpartition(b"\n")
            if not sep:
                return                                        # partial line: wait for the newline
            chunk = complete + sep
        self._offset += len(chunk)
        text = chunk.decode("utf-8", errors="replace")
        now_m = self._now() - (self.t0 or 0.0)
        recs = parse_jsonl(text.splitlines(), self.vocab, self.t0 or 0.0, self._n)
        self._n += len(text.splitlines())
        for r in recs:
            r.publish_t += self.shift
            if r.event is not None:
                r.event.t += self.shift
            if self.mode == "follow":                         # live source: the report is available now
                r.publish_t = max(r.publish_t, now_m)
        self.queue = sorted(self.queue + recs, key=lambda r: r.publish_t)

    def _on_timer(self) -> None:
        from std_msgs.msg import String

        from medortrace_ros import convert as cv

        now_ros = self._now()
        if now_ros <= 0.0:
            return                                            # sim time, no /clock yet
        if self.t0 is None:
            if self.origin == "mission":
                return                                        # wait for the mission origin
            self.t0 = now_ros
        if self.mode == "follow" or self._offset == 0:
            self._read_new()
        now = now_ros - self.t0
        while self.queue and self.queue[0].publish_t <= now:
            r = self.queue.pop(0)
            if r.text is not None:
                self.pub_text.publish(String(data=json.dumps({"t": r.publish_t, "text": r.text,
                                                              "parsed": r.event is not None})))
            if r.event is not None:
                self.pub.publish(cv.workflow_to_ros(r.event, self.t0, r.source))
                self._published += 1
                self.log.info(f"workflow {r.event.type.value} {r.event.item_id or ''} "
                              f"{r.event.src or '?'} -> {r.event.dst or '?'} (t={r.event.t:.1f}, {r.source})")


def main(args=None) -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = rclpy.create_node("medortrace_workflow_gateway")
    try:
        WorkflowGatewayNode(node)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

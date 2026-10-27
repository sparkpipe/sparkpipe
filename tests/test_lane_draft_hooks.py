#!/usr/bin/env python3
import copy
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane  # noqa: E402

failures = []
records = stage_lane.lane_records()["lanes"]
record_id = "a.x-k2"
lane = copy.deepcopy(records[record_id])
lane["taps"] = {"entries": "o1,o29,o57", "rank": 15, "stream": "10.10.250.2:7721"}
lane["draft_server"] = {"host": "127.0.0.1", "port": 7722, "target_model": record_id, "depth": 5, "wait_ms": 30, "min_confidence": 0.05}
lane["draftd"] = {"unit": "sparkpipe-draftd-axk2", "source": "/srv/workspace/sparkpipe/draftd-test", "drafter": "/srv/workspace/sparkpipe/drafter-weights/ax-k2-dspark",
                  "target_shard": "/srv/workspace/sparkpipe/drafter-weights/ax-k2-target-io/model.safetensors.index.json", "window": 128,
                  "context_bytes": 2147483648, "log": "/srv/workspace/sparkpipe/logs/draftd-axk2.log", "embed": "model.embed_tokens.weight",
                  "head": "lm_head.weight", "norm": "model.norm.weight", "delta": -1}

if stage_lane.draft_hooks(record_id, lane) != {}:
    failures.append("a draft server without a lifecycle stays outside the swap controller (always on)")

lane["draftd"]["lifecycle"] = "model"
hooks = stage_lane.draft_hooks(record_id, lane)
start, stop = hooks.get("draft_start", ""), hooks.get("draft_stop", "")
words = shlex.split(start)
settings = dict(word.split("=", 1) for word in words[1:] if "=" in word and not word.startswith("-"))
if words[:1] != ["env"] or words[-3:] != ["/srv/workspace/sparkpipe/draftd-test/hub_draftd.sh", "start", "/srv/workspace/sparkpipe/draftd-test"]:
    failures.append(f"draft_start runs the source's hub_draftd.sh start with the record's environment: {start}")
expected = {"DRAFTD_UNIT": "sparkpipe-draftd-axk2", "DRAFTD_TAP_PORT": "7721", "DRAFTD_DRAFT_PORT": "7722", "DRAFTD_WINDOW": "128", "DRAFTD_DELTA": "-1",
            "DRAFTD_EMBED": "model.embed_tokens.weight", "DRAFTD_HEAD": "lm_head.weight", "DRAFTD_NORM": "model.norm.weight"}
for key, value in expected.items():
    if settings.get(key) != value:
        failures.append(f"draft_start sets {key}={value}, not {settings.get(key)!r}")
if not stop.endswith("/srv/workspace/sparkpipe/draftd-test/hub_draftd.sh stop") or "DRAFTD_UNIT=sparkpipe-draftd-axk2" not in stop:
    failures.append(f"draft_stop stops the record's unit: {stop}")
entry = stage_lane.swap_entry(lane["model"], stage_lane.MODELS[lane["model"]], lane["lane"], hooks)
if entry.get("draft_start") != start or entry.get("draft_stop") != stop:
    failures.append("the swap entry carries the record's draft hooks")

lane["draftd"]["lifecycle"] = "always"
try:
    stage_lane.draft_hooks(record_id, lane)
    failures.append("an unknown draftd lifecycle is refused")
except SystemExit:
    pass

lane["draftd"]["lifecycle"] = "model"
lane["draftd"]["drafter"] = "/mnt/cold-raid6/models/ax-k2-dspark"
try:
    stage_lane.draft_hooks(record_id, lane)
    failures.append("a model-scoped draft server off the hub's local NVMe is refused")
except SystemExit:
    pass

for failure in failures:
    print("FAIL", failure)
if failures:
    raise SystemExit(1)
print("PASS lane draft hooks: a draftd record with lifecycle model gives the swap entry draft_start and draft_stop with the record's environment; no lifecycle keeps it always on; an unknown lifecycle or a path off local NVMe is refused")

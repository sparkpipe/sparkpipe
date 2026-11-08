#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import lane_config  # noqa: E402

NAMES = {"embed": "model.language_model.embed_tokens.weight", "head": "lm_head.weight", "norm": "model.language_model.norm.weight"}


def refused(lane, message):
    original = lane_config.record
    lane_config.record = lambda record_id: lane
    try:
        lane_config.draftd_environment("probe")
    except SystemExit as error:
        assert message in str(error), str(error)
        return
    finally:
        lane_config.record = original
    raise AssertionError(f"a draftd record rendered although {message}")


def record_names():
    source, environment = lane_config.draftd_environment("qwen3.8-27b")
    assert source.startswith("/srv/workspace/"), source
    for key, value in NAMES.items():
        assert environment[f"DRAFTD_TARGET_{key.upper()}"] == value, environment
    flags = lane_config.draftd_flags(environment)
    for key, value in NAMES.items():
        assert f"--{key} {value}" in flags, flags
    _, plain = lane_config.draftd_environment("kimi-k3.1-pro")
    assert not any(key.startswith("DRAFTD_TARGET_") and key != "DRAFTD_TARGET_SHARD" for key in plain), plain
    assert not any(flag.startswith(("--embed", "--head", "--norm")) for flag in lane_config.draftd_flags(plain))
    lane = json.loads((ROOT / "deployment/lanes.json").read_text())["lanes"]["qwen3.8-27b"]
    partial = dict(lane, draftd=dict(lane["draftd"], target_names={"embed": NAMES["embed"], "head": NAMES["head"]}))
    refused(partial, "target_names names the target shard's embed, head, norm tensors")
    empty = dict(lane, draftd=dict(lane["draftd"], target_names=dict(NAMES, norm="")))
    refused(empty, "each a non-empty string")
    shardless = dict(lane, draftd={key: value for key, value in lane["draftd"].items() if key != "target_shard"})
    refused(shardless, "the record has none")


def script_names():
    home = os.environ.get("HOME", "")
    if not home.startswith("/home/"):
        print("hub_draftd.sh names: skipped, the script serves paths under /srv/workspace or /home and HOME is " + home)
        return
    with tempfile.TemporaryDirectory(dir=home) as work:
        work = Path(work)
        source, bin_dir = work / "source", work / "bin"
        (source / "spec_offline").mkdir(parents=True)
        (source / "block_draftd.py").write_text("")
        (source / "spec_offline" / "block_drafter.py").write_text("")
        bin_dir.mkdir()
        record, log, ports = work / "systemd-run.args", work / "draftd.log", work / "listening"
        (bin_dir / "systemd-run").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {record}\nprintf '127.0.0.1:7731 \\n127.0.0.1:7732 \\n' > {ports}\necho DRAFTD-READY >> {log}\n")
        (bin_dir / "systemctl").write_text("#!/bin/sh\nexit 1\n")
        (bin_dir / "ss").write_text(f"#!/bin/sh\ncat {ports} 2>/dev/null\ntrue\n")
        for name in ("systemd-run", "systemctl", "ss"):
            (bin_dir / name).chmod(0o755)
        environment = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", DRAFTD_UNIT="probe-draftd", DRAFTD_LOG=str(log),
                           DRAFTD_DRAFTER=str(work / "drafter"), DRAFTD_TARGET_SHARD=str(work / "head.safetensors"), DRAFTD_TAP_HOST="127.0.0.1",
                           DRAFTD_TAP_PORT="7731", DRAFTD_DRAFT_PORT="7732", **{f"DRAFTD_TARGET_{key.upper()}": value for key, value in NAMES.items()})
        started = subprocess.run(["bash", str(ROOT / "tools/hub_draftd.sh"), "start", str(source)], env=environment, capture_output=True, text=True, timeout=60)
        assert started.returncode == 0, started.stdout + started.stderr
        command = record.read_text()
        for key, value in NAMES.items():
            assert f"--{key} '{value}'" in command, command
        record.unlink()
        ports.unlink()
        environment.pop("DRAFTD_TARGET_NORM")
        partial = subprocess.run(["bash", str(ROOT / "tools/hub_draftd.sh"), "start", str(source)], env=environment, capture_output=True, text=True, timeout=60)
        assert partial.returncode != 0 and "set all three and the shard, or none" in partial.stderr, partial.stdout + partial.stderr
        assert not record.exists(), "a partial set of target names still started draftd"


def main():
    record_names()
    script_names()
    print("PASS draftd target names render from the lane record into the drift flags and the unit's draftd flags, and a partial set is refused")


if __name__ == "__main__":
    main()

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


SOCKET = "import socket, sys; socket.socket(socket.AF_UNIX).bind(sys.argv[1])"


ENGINE_LANES = ("kimi-k3.1-pro", "qwen3.8-27b", "deepseek-v4.1-flash")


def record_server():
    roots, units = set(), set()
    for record_id in ENGINE_LANES:
        _, environment = lane_config.draftd_environment(record_id)
        assert environment["DRAFTD_SERVER"] == "engine" and environment["DRAFTD_ENGINE_LANE"] == record_id, environment
        assert lane_config.served_unit(environment) == environment["DRAFTD_ENGINE_UNIT"]
        assert lane_config.served_unit(dict(environment, DRAFTD_SERVER="block_draftd")) == environment["DRAFTD_UNIT"]
        assert environment["DRAFTD_WEIGHTD_UNIT"] == "sparkpipe-weightd-hub", environment
        roots.add(environment["DRAFTD_ENGINE_ROOT"])
        units.add(environment["DRAFTD_ENGINE_UNIT"])
    assert len(roots) == len(units) == 1, f"the engine lanes share one root and one draft server unit, not {sorted(roots)} and {sorted(units)}"
    _, plain = lane_config.draftd_environment("kimi-k3")
    assert "DRAFTD_SERVER" not in plain and lane_config.served_unit(plain) == plain["DRAFTD_UNIT"], plain
    lane = json.loads((ROOT / "deployment/lanes.json").read_text())["lanes"]["kimi-k3.1-pro"]
    refused(dict(lane, draftd=dict(lane["draftd"], server="vllm")), "draftd server is block_draftd or engine")
    refused(dict(lane, draftd={key: value for key, value in dict(lane["draftd"], server="engine").items() if key != "engine"}), "needs the record's draftd.engine")
    refused(dict(lane, draftd=dict(lane["draftd"], engine=dict(lane["draftd"]["engine"], root="/mnt/cold/engine"))), "is not on the hub's local NVMe")


def engine_configs():
    lanes = json.loads((ROOT / "deployment/lanes.json").read_text())["lanes"]
    swap = {spec["id"]: spec for spec in json.loads((ROOT / "deployment/hub/model_swap.json").read_text())["models"]}
    root = lanes[ENGINE_LANES[0]]["draftd"]["engine"]["root"]
    assert sorted(path.name for path in (ROOT / "deployment/hub").glob("draftd_engine*.json")) == ["draftd_engine.json"], "one engine config"
    config = json.loads((ROOT / "deployment/hub/draftd_engine.json").read_text())
    assert config["control_socket"].startswith(root + "/") and config["weightd_socket"].startswith(root + "/"), "the sockets live in the engine root"
    served = {lane["name"]: lane for lane in config["lanes"]}
    assert sorted(served) == sorted(lanes[record_id]["draftd"]["engine"]["lane"] for record_id in ENGINE_LANES), f"the engine config serves {sorted(served)}"
    for record_id in ENGINE_LANES:
        record = lanes[record_id]
        lane = served[record["draftd"]["engine"]["lane"]]
        tap_host, tap_port = record["taps"]["stream"].rsplit(":", 1)
        assert lane["draft_port"] == record["draft_server"]["port"] and lane["target_model"] == record["draft_server"]["target_model"], record_id
        assert lane["taps"]["host"] == tap_host and lane["taps"]["port"] == int(tap_port), record_id
        assert set(lane["taps"]["entries"].split(",")) <= set(record["taps"]["entries"].split(",")), record_id
        assert lane["taps"]["model_tag"] == swap[record_id]["kv_label"], f"{record_id}: the taps' model tag is the stage's KV label"
        assert lane["drafter"]["kind"] == "engine" and all(lane["drafter"][key].startswith(root + "/") for key in ("module", "pack")), record_id


def script_engine():
    home = os.environ.get("HOME", "")
    if not home.startswith("/home/"):
        print("hub_draftd.sh engine server: skipped, the script serves paths under /srv/workspace or /home and HOME is " + home)
        return
    with tempfile.TemporaryDirectory(dir=home) as work:
        work = Path(work)
        root, bin_dir, calls = work / "engine", work / "bin", work / "calls"
        (root / "bin").mkdir(parents=True)
        (root / "config").mkdir()
        bin_dir.mkdir()
        (root / "config" / "draftd.json").write_text(json.dumps({"weightd_socket": str(root / "run/weightd.sock"),
                                                                 "control_socket": str(root / "run/draftd.sock"), "lanes": []}))
        (root / "bin" / "sparkpipe_weightd").write_text("#!/bin/sh\n")
        (root / "bin" / "sparkpipe_draftd").write_text(f"#!/bin/sh\necho \"draftd $*\" >> {calls}\necho \"OK lane=$4\"\n")
        (bin_dir / "systemd-run").write_text(
            f"#!/bin/sh\necho \"systemd-run $*\" >> {calls}\n"
            f"case \"$*\" in *--unit=probe-weightd*) python3 -c \"{SOCKET}\" {root}/run/weightd.sock;; "
            f"*--unit=probe-engine*) python3 -c \"{SOCKET}\" {root}/run/draftd.sock;; esac\n")
        (bin_dir / "systemctl").write_text("#!/bin/sh\nexit 1\n")
        (bin_dir / "ss").write_text("#!/bin/sh\ntrue\n")
        for path in (root / "bin" / "sparkpipe_weightd", root / "bin" / "sparkpipe_draftd", bin_dir / "systemd-run", bin_dir / "systemctl", bin_dir / "ss"):
            path.chmod(0o755)
        environment = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", DRAFTD_SERVER="engine", DRAFTD_ENGINE_ROOT=str(root),
                           DRAFTD_ENGINE_LANE="probe-lane", DRAFTD_ENGINE_UNIT="probe-engine", DRAFTD_WEIGHTD_UNIT="probe-weightd",
                           DRAFTD_LOG=str(work / "draftd.log"), DRAFTD_TAP_PORT="7731", DRAFTD_DRAFT_PORT="7732")
        script = str(ROOT / "tools/hub_draftd.sh")
        started = subprocess.run(["bash", script, "start", str(work)], env=environment, capture_output=True, text=True, timeout=120)
        assert started.returncode == 0 and "OK lane=probe-lane" in started.stdout, started.stdout + started.stderr
        log = calls.read_text().splitlines()
        assert [line.split()[1] for line in log] == ["--user", "--user", "--control"], log
        assert "--unit=probe-weightd" in log[0] and "sparkpipe_weightd --socket" in log[0], log
        assert "--unit=probe-engine" in log[1] and "sparkpipe_draftd --config config/draftd.json" in log[1], log
        assert log[2] == f"draftd --control {root}/run/draftd.sock activate probe-lane", log
        stopped = subprocess.run(["bash", script, "stop"], env=environment, capture_output=True, text=True, timeout=60)
        assert stopped.returncode == 0 and calls.read_text().splitlines()[-1] == f"draftd --control {root}/run/draftd.sock deactivate probe-lane", stopped.stdout
        bogus = subprocess.run(["bash", script, "start", str(work)], env=dict(environment, DRAFTD_SERVER="vllm"), capture_output=True, text=True, timeout=60)
        assert bogus.returncode != 0 and "is block_draftd or engine" in bogus.stderr, bogus.stderr
        (root / "config" / "draftd.json").unlink()
        unstaged = subprocess.run(["bash", script, "start", str(work)], env=environment, capture_output=True, text=True, timeout=60)
        assert unstaged.returncode != 0 and "lacks config/draftd.json" in unstaged.stderr, unstaged.stderr


def main():
    record_names()
    record_server()
    engine_configs()
    script_names()
    script_engine()
    print("PASS draftd target names render from the lane record into the drift flags and the unit's draftd flags, and a partial set is refused; "
          "the K3.1 Pro, Qwen3.8-27B and DeepSeek V4.1 Flash records carry the server switch and share one engine root and unit, "
          "whose config serves each lane on its record's ports and taps, and hub_draftd.sh with DRAFTD_SERVER=engine starts the hub's "
          "weightd and the engine draft server once, activates the lane through the control socket, deactivates it on stop, and refuses an "
          "unknown server or an unstaged engine root")


if __name__ == "__main__":
    main()

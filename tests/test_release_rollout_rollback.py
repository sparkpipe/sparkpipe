"""Run release_rollout.sh rollback against a fake 16-node fleet: a missing backup refuses before any change, a complete one restores every lane."""
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release_rollout.sh"
HEX = "0123456789abcdef"
STAMP = "20261013T120000Z"
SHIMS = {
    "ssh": """#!/usr/bin/env bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host="$1"; shift
HOME="$FAKE_FLEET/$host" exec bash -c "$*"
""",
    "curl": "#!/usr/bin/env bash\nexit 0\n",
    "sleep": "#!/usr/bin/env bash\nexit 0\n",
}


def lanes():
    text = SCRIPT.read_text()
    models = re.search(r'^MODELS="([^"]*)"', text, re.M).group(1).split()
    extra = re.search(r'^EXTRA_LANES="([^"]*)"', text, re.M).group(1).split()
    return [re.search(r'^\s*' + re.escape(model) + r'\) echo "(\S+) \S+ \S+ (\S+) ', text, re.M).groups()
             for model in models], extra


def build(fleet, wd_bytes):
    model_lanes, extra = lanes()
    for index in range(16):
        home = fleet / f"spark{HEX[index]}"
        for lane in [lane for lane, _ in model_lanes] + extra:
            for name, content in (("root", "new"), (f"root.prev-{STAMP}", "old")):
                (home / lane / name / "bin").mkdir(parents=True, exist_ok=True)
                (home / lane / name / "bin" / "residentd").write_text(content)
                (home / lane / name / "kvsnapshot").mkdir(exist_ok=True)
            (home / lane / "root" / "kvsnapshot" / "page").write_text("kv")
        swap = home / "sparkdata" / "swap"
        swap.mkdir(parents=True, exist_ok=True)
        (swap / "weightd_warm").write_text("new")
        (swap / f"weightd_warm.prev-{STAMP}").write_text("old")
    hub = fleet / "rtx5090"
    for _, api in model_lanes:
        for name, content in ((api, "new"), (f"{api}.prev-{STAMP}", "old")):
            (hub / name / "bin").mkdir(parents=True, exist_ok=True)
            (hub / name / "bin" / "api").write_text(content)
    (hub / "release-staging" / "weightd-rollback").mkdir(parents=True)
    (hub / "release" / "core" / "bin").mkdir(parents=True)
    old_wd = hashlib.sha256(wd_bytes).hexdigest()[:16]
    (hub / "release-staging" / "weightd-rollback" / f"sparkpipe_weightd.{old_wd}").write_bytes(wd_bytes)
    return model_lanes, extra, old_wd


def snapshot(fleet):
    return {str(path.relative_to(fleet)): path.read_text() for path in sorted(fleet.rglob("*")) if path.is_file()}


def run(fleet, work, old_wd, state):
    shims = work / "shims"
    shims.mkdir(exist_ok=True)
    for name, body in SHIMS.items():
        (shims / name).write_text(body)
        (shims / name).chmod(0o755)
    firmware = work / "fw"
    firmware.mkdir(exist_ok=True)
    (firmware / "SOURCE_COMMIT").write_text("abcdef123\n")
    (firmware / "sparkpipe_weightd").write_text("new weightd")
    env = dict(os.environ, PATH=f"{shims}:{os.environ['PATH']}", FAKE_FLEET=str(fleet), FW=str(firmware),
               APIB="/nonexistent", STATE=str(state), OLD_WD=old_wd, ONLY="qwen3.8-max")
    return subprocess.run(["bash", str(SCRIPT), "rollback"], env=env, capture_output=True, text=True, timeout=600)


def check(condition, message, result=None):
    if not condition:
        detail = "" if result is None else f"\n--- stdout\n{result.stdout[-3000:]}\n--- stderr\n{result.stderr[-2000:]}"
        raise SystemExit(f"FAIL test_release_rollout_rollback: {message}{detail}")


def case(label, mutate, expect_refusal):
    with tempfile.TemporaryDirectory(prefix="release-rollback-") as directory:
        work = pathlib.Path(directory)
        fleet = work / "fleet"
        model_lanes, extra, old_wd = build(fleet, b"old weightd")
        state = work / "state"
        state.mkdir()
        (state / "utc").write_text(STAMP + "\n")
        mutate(fleet, state, model_lanes)
        before = snapshot(fleet)
        result = run(fleet, work, old_wd, state)
        after = snapshot(fleet)
        if expect_refusal:
            check(result.returncode != 0, f"{label}: rollback exited 0", result)
            check(before == after, f"{label}: rollback changed the fleet before refusing", result)
        else:
            check("ROLLBACK-REFUSED" not in result.stdout and "ROLLBACK-INCOMPLETE" not in result.stdout,
                  f"{label}: rollback refused a complete backup", result)
            for index in range(16):
                home = fleet / f"spark{HEX[index]}"
                for lane in [lane for lane, _ in model_lanes] + extra:
                    check((home / lane / "root" / "bin" / "residentd").read_text() == "old",
                          f"{label}: spark{HEX[index]} {lane} was not restored", result)
                for lane, _ in model_lanes:
                    check((home / lane / "root" / "kvsnapshot" / "page").read_text() == "kv",
                          f"{label}: spark{HEX[index]} {lane} lost its KV snapshot tier", result)
                check((home / "sparkdata" / "swap" / "weightd_warm").read_text() == "old",
                      f"{label}: spark{HEX[index]} weightd_warm was not restored", result)
            for _, api in model_lanes:
                check((fleet / "rtx5090" / api / "bin" / "api").read_text() == "old", f"{label}: hub {api} was not restored", result)
        print(f"ok {label}")


def drop_stamp(fleet, state, model_lanes):
    (state / "utc").unlink()


def drop_one_node_backup(fleet, state, model_lanes):
    shutil.rmtree(fleet / "spark7" / model_lanes[3][0] / f"root.prev-{STAMP}")


def drop_hub_backup(fleet, state, model_lanes):
    shutil.rmtree(fleet / "rtx5090" / f"{model_lanes[0][1]}.prev-{STAMP}")


def drop_warm_backup(fleet, state, model_lanes):
    (fleet / "sparkc" / "sparkdata" / "swap" / f"weightd_warm.prev-{STAMP}").unlink()


def keep_everything(fleet, state, model_lanes):
    pass


case("no backup stamp", drop_stamp, True)
case("one node missing one lane backup", drop_one_node_backup, True)
case("hub API backup missing", drop_hub_backup, True)
case("one node missing the weightd_warm backup", drop_warm_backup, True)
case("complete backup", keep_everything, False)
print("PASS test_release_rollout_rollback: rollback refuses an incomplete backup before changing anything and restores every lane, hub API and weightd_warm from a complete one")

"""Run release_rollout.sh stage, install and rollback against a fake 16-node fleet: every node must verify and install one identical set of files, an incomplete backup refuses before any change, and a rollback never activates models on an old weightd that is not ready."""
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
    "scp": """#!/usr/bin/env bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host="${2%%:*}"
[ "$host" = "${FAKE_SCP_FAIL:-}" ] && exit 1
mkdir -p "$FAKE_FLEET/$host" && cp "$1" "$FAKE_FLEET/$host/${2#*:}"
""",
    "curl": "#!/usr/bin/env bash\necho \"$*\" >> \"$FAKE_FLEET/curl.log\"\nexit 0\n",
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
    return {str(path.relative_to(fleet)): path.read_bytes() for path in sorted(fleet.rglob("*"))
            if path.is_file() and path.name != "curl.log"}


def run(step, fleet, work, old_wd, state, **extra_env):
    shims = work / "shims"
    shims.mkdir(exist_ok=True)
    for name, body in SHIMS.items():
        (shims / name).write_text(body)
        (shims / name).chmod(0o755)
    firmware = work / "fw"
    if not firmware.exists():
        firmware.mkdir()
        (firmware / "SOURCE_COMMIT").write_text("abcdef123\n")
        (firmware / "sparkpipe_weightd").write_text("new weightd")
        (firmware / "sparkpipe_model_residentd").write_text("residentd")
        (firmware / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256((firmware / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ("sparkpipe_weightd", "sparkpipe_model_residentd")))
    env = dict(os.environ, PATH=f"{shims}:{os.environ['PATH']}", FAKE_FLEET=str(fleet), FW=str(firmware),
               APIB="/nonexistent", STATE=str(state), OLD_WD=old_wd, ONLY="qwen3.8-max")
    env.update(extra_env)
    return subprocess.run(["bash", str(SCRIPT), step], env=env, capture_output=True, text=True, timeout=600)


def check(condition, message, result=None):
    if not condition:
        detail = "" if result is None else f"\n--- stdout\n{result.stdout[-3000:]}\n--- stderr\n{result.stderr[-2000:]}"
        raise SystemExit(f"FAIL test_release_rollout: {message}{detail}")


def fleet_case(label, body):
    with tempfile.TemporaryDirectory(prefix="release-rollout-") as directory:
        work = pathlib.Path(directory)
        fleet = work / "fleet"
        model_lanes, extra, old_wd = build(fleet, b"old weightd")
        state = work / "state"
        state.mkdir()
        (state / "utc").write_text(STAMP + "\n")
        body(work, fleet, state, model_lanes, extra, old_wd)
        print(f"ok {label}")


def refusal(mutate):
    def body(work, fleet, state, model_lanes, extra, old_wd):
        mutate(fleet, state, model_lanes)
        before = snapshot(fleet)
        result = run("rollback", fleet, work, old_wd, state)
        check(result.returncode != 0, "rollback exited 0", result)
        check(before == snapshot(fleet), "rollback changed the fleet before refusing", result)
    return body


def complete_rollback(work, fleet, state, model_lanes, extra, old_wd):
    result = run("rollback", fleet, work, old_wd, state)
    check("ROLLBACK-REFUSED" not in result.stdout and "ROLLBACK-INCOMPLETE" not in result.stdout,
          "rollback refused a complete backup", result)
    for index in range(16):
        home = fleet / f"spark{HEX[index]}"
        for lane in [lane for lane, _ in model_lanes] + extra:
            check((home / lane / "root" / "bin" / "residentd").read_text() == "old",
                  f"spark{HEX[index]} {lane} was not restored", result)
        for lane, _ in model_lanes:
            check((home / lane / "root" / "kvsnapshot" / "page").read_text() == "kv",
                  f"spark{HEX[index]} {lane} lost its KV snapshot tier", result)
        check((home / "sparkdata" / "swap" / "weightd_warm").read_text() == "old",
              f"spark{HEX[index]} weightd_warm was not restored", result)
    for _, api in model_lanes:
        check((fleet / "rtx5090" / api / "bin" / "api").read_text() == "old", f"hub {api} was not restored", result)
    calls = (fleet / "curl.log").read_text() if (fleet / "curl.log").exists() else ""
    check(result.returncode != 0, "rollback exited 0 although the old weightd never became ready", result)
    check("/activate" not in calls, "rollback activated a model on a weightd that is not ready", result)


def stage(fail_host):
    def body(work, fleet, state, model_lanes, extra, old_wd):
        extra_env = {"FAKE_SCP_FAIL": fail_host} if fail_host else {}
        result = run("stage", fleet, work, old_wd, state, **extra_env)
        verified = [index for index in range(16)
                    if (fleet / f"spark{HEX[index]}" / "rp-release-abcdef123" / "sparkpipe_model_residentd").exists()]
        if fail_host:
            check(result.returncode != 0, f"stage exited 0 with {fail_host} unstaged", result)
            check("STAGE-FAILED" in result.stdout, "stage did not report the failure", result)
            check(len(verified) == 15, f"stage reached {len(verified)} nodes, expected 15", result)
        else:
            check(result.returncode == 0, "stage failed on a healthy fleet", result)
            check(len(verified) == 16, f"stage reached {len(verified)} nodes", result)
    return body


def install(fail_host):
    def body(work, fleet, state, model_lanes, extra, old_wd):
        text = SCRIPT.read_text()
        libraries = sorted(set(re.findall(r'^\s*\S+\) echo "\S+ (\S+) ', text, re.M)) | {"libqwen38_27b_bf16_tp16_serving_adapter.so"})
        firmware = work / "fw"
        firmware.mkdir()
        (firmware / "SOURCE_COMMIT").write_text("abcdef123\n")
        for name in ["sparkpipe_weightd", "sparkpipe_model_residentd", "weightd_warm", "weightd_spine_budget.py",
                     "libhidden_transport_spark_host_rdma_verbs.so"] + libraries:
            (firmware / name).write_text(f"release {name}")
        build_directory = work / "api-build"
        build_directory.mkdir()
        for name in ["sparkpipe_model_api", "sparkpipe_tokenize_prompt"] + libraries:
            (build_directory / name).write_text(f"hub {name}")
        for index in range(16):
            host = f"spark{HEX[index]}"
            if host != fail_host:
                shutil.copytree(firmware, fleet / host / "rp-release-abcdef123")
            for lane in [lane for lane, _ in model_lanes] + extra:
                (fleet / host / lane / "root" / "lib").mkdir(parents=True, exist_ok=True)
        for _, api in model_lanes:
            (fleet / "rtx5090" / api / "runtime" / "lib").mkdir(parents=True, exist_ok=True)
        result = run("install", fleet, work, old_wd, state, APIB=str(build_directory))
        installed = [index for index in range(16)
                     if (fleet / f"spark{HEX[index]}" / model_lanes[0][0] / "root" / "bin" / "sparkpipe_model_residentd").exists()]
        if fail_host:
            check(result.returncode != 0, f"install exited 0 with {fail_host} missing the release", result)
            check("INSTALL-FAILED" in result.stdout, "install did not report the failure", result)
            check(len(installed) == 15, f"install reached {len(installed)} nodes, expected 15", result)
        else:
            check(result.returncode == 0, "install failed on a healthy fleet", result)
            check("INSTALL-FAILED" not in result.stdout, "install reported a failure on a healthy fleet", result)
            check(len(installed) == 16, f"install reached {len(installed)} nodes", result)
            for index in range(16):
                swap = fleet / f"spark{HEX[index]}" / "sparkdata" / "swap" / "weightd_warm"
                check(swap.read_text() == "release weightd_warm", f"spark{HEX[index]} shared weightd_warm was not installed", result)
            for _, api in model_lanes:
                check((fleet / "rtx5090" / api / "bin" / "sparkpipe_model_api").read_text() == "hub sparkpipe_model_api",
                      f"hub {api} was not installed", result)
    return body


fleet_case("stage verifies all 16 nodes", stage(""))
fleet_case("install puts one identical release on all 16 nodes and the hub", install(""))
fleet_case("install fails when one node has no staged release", install("spark9"))
fleet_case("stage fails when one node is not staged", stage("spark9"))
fleet_case("rollback refuses without a backup stamp", refusal(lambda fleet, state, lanes: (state / "utc").unlink()))
fleet_case("rollback refuses when one node misses one lane backup",
           refusal(lambda fleet, state, lanes: shutil.rmtree(fleet / "spark7" / lanes[3][0] / f"root.prev-{STAMP}")))
fleet_case("rollback refuses when a hub API backup is missing",
           refusal(lambda fleet, state, lanes: shutil.rmtree(fleet / "rtx5090" / f"{lanes[0][1]}.prev-{STAMP}")))
fleet_case("rollback refuses when one node misses the weightd_warm backup",
           refusal(lambda fleet, state, lanes: (fleet / "sparkc" / "sparkdata" / "swap" / f"weightd_warm.prev-{STAMP}").unlink()))
fleet_case("complete rollback restores everything and stops on a weightd that is not ready", complete_rollback)
print("PASS test_release_rollout: stage needs all 16 nodes verified; install needs one identical result from all 16; rollback refuses an incomplete backup before changing anything, restores every lane, hub API and weightd_warm from a complete one, and never activates on an old weightd that is not ready")

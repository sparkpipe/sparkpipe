"""Run release_rollout.sh stage, roots, backup, install and rollback against a fake 16-node fleet: every node must verify and install one identical set of files, a renumbered record's root is copied from the running release's root, hub backups go to the workspace volume and refuse when it is short of space, an incomplete backup refuses before any change, a rollback never activates models on an old weightd that is not ready, and the restart check fails unless the restarted API reaches every rank and answers as before."""
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release_rollout.sh"
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane  # noqa: E402
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
target="${2#*:}"
case "$target" in /*) ;; *) target="$FAKE_FLEET/$host/$target" ;; esac
mkdir -p "$(dirname "$target")" && cp "$1" "$target"
""",
    "curl": "#!/usr/bin/env bash\necho \"$*\" >> \"$FAKE_FLEET/curl.log\"\nexit 0\n",
    "sleep": "#!/usr/bin/env bash\nexit 0\n",
    "systemctl": "#!/usr/bin/env bash\nexit 0\n",
    "systemd-run": "#!/usr/bin/env bash\necho \"$*\" >> \"$FAKE_FLEET/systemd-run.log\"\nexit 0\n",
    "df": """#!/usr/bin/env bash
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
for path in "$@"; do case "$path" in -*) ;; *) echo "fake 999999999 1 ${FAKE_DF_FREE_KIB:-104857600} 1% $path" ;; esac; done
""",
}


def lanes():
    text = SCRIPT.read_text()
    models = re.search(r'^MODELS="([^"]*)"', text, re.M).group(1).split()
    rows = [subprocess.run([sys.executable, str(ROOT / "tools" / "lane_config.py"), "release-row", model], capture_output=True, text=True, check=True).stdout.split()
            for model in models]
    return [(row[0], row[3]) for row in rows], []


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
        (hub / api / "bin").mkdir(parents=True, exist_ok=True)
        (hub / api / "bin" / "api").write_text("new")
    work = fleet / "srv-workspace"
    (work / "swap-controller").mkdir(parents=True)
    (work / "swap-controller" / "token").write_text("s" * 40 + "\n")
    for _, api in model_lanes:
        (work / "backups" / f"{api}.prev-{STAMP}" / "bin").mkdir(parents=True, exist_ok=True)
        (work / "backups" / f"{api}.prev-{STAMP}" / "bin" / "api").write_text("old")
    (hub / "sparkpipe-swap").mkdir(parents=True)
    (hub / "sparkpipe-swap" / "model_swap.json").write_text("live controller\n")
    (hub / "sparkpipe-swap" / f"model_swap.json.prev-{STAMP}").write_text("previous controller\n")
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
               APIB="/nonexistent", STATE=str(state), OLD_WD=old_wd, ONLY="qwen3.8-max",
               HUB_WORK=str(fleet / "srv-workspace"))
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
    check((fleet / "rtx5090" / "sparkpipe-swap" / "model_swap.json").read_text() == "previous controller\n",
          "rollback did not restore the controller config", result)
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


def backup(free_kib):
    def body(work, fleet, state, model_lanes, extra, old_wd):
        (state / "utc").unlink()
        hub = fleet / "rtx5090"
        (hub / "sparkpipe-swap").mkdir(parents=True, exist_ok=True)
        (hub / "sparkpipe-swap" / "model_swap.json").write_text("{}")
        for index in range(16):
            weightd = fleet / f"spark{HEX[index]}" / "sparkdata" / "weightd"
            weightd.mkdir(parents=True, exist_ok=True)
            (weightd / "sparkpipe_weightd").write_text("node weightd")
        home_before = sorted(path.name for path in hub.iterdir())
        before = snapshot(fleet)
        result = run("backup", fleet, work, old_wd, state, FAKE_DF_FREE_KIB=str(free_kib))
        if free_kib < 20 * 1048576:
            check(result.returncode != 0, "backup ran with the hub short of space", result)
            check("HUB-SPACE-LOW" in result.stdout, "backup did not report the free space it found", result)
            check(before == snapshot(fleet), "backup changed the fleet before refusing for space", result)
            return
        check(result.returncode == 0, "backup failed on a healthy fleet", result)
        stamp = (state / "utc").read_text().strip()
        for _, api in model_lanes:
            copy = fleet / "srv-workspace" / "backups" / f"{api}.prev-{stamp}" / "bin" / "api"
            check(copy.exists() and copy.read_text() == "new", f"hub {api} was not backed up on the workspace volume", result)
        check(sorted(path.name for path in hub.iterdir()) == home_before, "backup wrote into the hub home", result)
    return body


def install(fail_host, process=None):
    def body(work, fleet, state, model_lanes, extra, old_wd):
        text = SCRIPT.read_text()
        libraries = sorted({model.get("firmware_adapter", model["adapter"]) for model in stage_lane.MODELS.values()})
        firmware = work / "fw"
        firmware.mkdir()
        (firmware / "SOURCE_COMMIT").write_text("abcdef123\n")
        for name in ["sparkpipe_weightd", "sparkpipe_model_residentd", "weightd_warm",
                     "libhidden_transport_spark_host_rdma_verbs.so"] + libraries:
            (firmware / name).write_text(f"release {name}")
        (firmware / "runtime_libs").mkdir()
        (firmware / "runtime_libs" / "libsparkpipe_device_cuda.so").write_text("release device library")
        build_directory = work / "api-build"
        build_directory.mkdir()
        for name in ["sparkpipe_model_api", "sparkpipe_tokenize_prompt"] + libraries:
            (build_directory / name).write_text(f"hub {name}")
        (build_directory / "runtime_libs").mkdir()
        (build_directory / "runtime_libs" / "libsparkpipe_device_cuda.so").write_text("hub device library")
        for index in range(16):
            host = f"spark{HEX[index]}"
            if host != fail_host:
                shutil.copytree(firmware, fleet / host / "rp-release-abcdef123")
            for lane in [lane for lane, _ in model_lanes] + extra:
                (fleet / host / lane / "root" / "lib").mkdir(parents=True, exist_ok=True)
        for _, api in model_lanes:
            (fleet / "rtx5090" / api / "runtime" / "lib").mkdir(parents=True, exist_ok=True)
        holder = None
        if process is not None:
            holder = subprocess.Popen(["bash", "-c", f'exec -a "{process}" sleep 300'])
        try:
            result = run("install", fleet, work, old_wd, state, APIB=str(build_directory))
        finally:
            if holder is not None:
                holder.kill()
                holder.wait()
        installed = [index for index in range(16)
                     if (fleet / f"spark{HEX[index]}" / model_lanes[0][0] / "root" / "bin" / "sparkpipe_model_residentd").exists()]
        if process is not None and process.startswith("./bin/"):
            check(result.returncode != 0, "install ran while a residentd was running", result)
            check("residentds running" in result.stdout, "install did not name the running residentd", result)
            check(installed == [], "install changed a node while a residentd was running", result)
        elif fail_host:
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


def records(busy_host):
    def body(work, fleet, state, model_lanes, extra, old_wd):
        shims = work / "push-shims"
        shims.mkdir()
        (shims / "python3").write_text(f"""#!/usr/bin/env bash
case "$*" in *lane_config.py\\ push\\ *) echo "push ${{@: -1}}" >> "$FAKE_FLEET/order.log"; exit 0 ;; esac
exec {sys.executable} "$@"
""")
        (shims / "python3").chmod(0o755)
        records_by_lane = {}
        for record, lane in json.loads((ROOT / "deployment" / "lanes.json").read_text())["lanes"].items():
            records_by_lane[lane["runtime_root"].format(host="HOST")[len("/home/HOST/"):-len("/root")]] = record
        named = {}
        for index in range(16):
            host = f"spark{HEX[index]}"
            for lane, _ in model_lanes:
                digest = hashlib.sha256(f"{host} {lane}".encode()).hexdigest()
                named[(host, records_by_lane[lane])] = digest
                packs = fleet / host / lane / "root" / "packs"
                packs.mkdir(parents=True, exist_ok=True)
                (packs / "pack.sha256").write_text(f"{digest}  rank{index}.spstage\n")
            warm = fleet / host / "sparkdata" / "swap" / "weightd_warm"
            warm.write_text(f"""#!/usr/bin/env bash
echo "reclaim {host} $(basename "$(dirname "$PWD")") $*" >> "$FAKE_FLEET/order.log"
[ "$1" = /tmp/spark_weightd.sock ] && [ "$2" = --reclaim-pack ] && [ $# = 3 ] || exit 2
exit {3 if host == busy_host else 0}
""")
            warm.chmod(0o755)
        result = run("records", fleet, work, old_wd, state, PATH=f"{shims}:{work / 'shims'}:{os.environ['PATH']}")
        order = (fleet / "order.log").read_text().splitlines() if (fleet / "order.log").exists() else []
        pushed = [line.split()[1] for line in order if line.startswith("push ")]
        if busy_host:
            check(result.returncode != 0 and "RECORDS-FAILED" in result.stdout, f"records went on although weightd on {busy_host} reported the lane's pack busy", result)
            check(pushed == [], f"records pushed {pushed} although the lane's current pack was not freed", result)
            return
        check(result.returncode == 0 and "RECORDS-FAILED" not in result.stdout, "records failed on a healthy fleet", result)
        lanes = sorted(json.loads((ROOT / "deployment" / "lanes.json").read_text())["lanes"])
        check(sorted(pushed) == lanes, f"records pushed {sorted(pushed)}, not every lane record", result)
        for record in pushed:
            position = order.index(f"push {record}")
            for index in range(16):
                host = f"spark{HEX[index]}"
                line = [entry for entry in order if entry.startswith(f"reclaim {host} ") and entry.endswith(f" {named[(host, record)]}")]
                check(len(line) == 1 and order.index(line[0]) < position,
                      f"{host} did not free the cold arena of the pack {record}'s lane named before its new record was pushed: {line}", result)
                check(line[0].split()[3:5] == ["/tmp/spark_weightd.sock", "--reclaim-pack"], f"{host} freed {record}'s pack with {line[0]}, not --reclaim-pack", result)
        check((fleet / "rtx5090" / "sparkpipe-swap" / "model_swap.json").read_text() == (ROOT / "deployment" / "hub" / "model_swap.json").read_text(),
              "records did not install the release's controller config", result)
        check("model_swap.py --config model_swap.json" in (fleet / "systemd-run.log").read_text(), "records did not restart the controller", result)
    return body


def restart(connected, compare_rc, capture_rc=0):
    def body(work, fleet, state, model_lanes, extra, old_wd):
        shims = work / "restart-shims"
        shims.mkdir()
        (shims / "systemctl").write_text('#!/usr/bin/env bash\necho "systemctl $*" >> "$FAKE_FLEET/order.log"\ncase "$*" in *is-active*) exit 1 ;; esac\nexit 0\n')
        (shims / "systemd-run").write_text('#!/usr/bin/env bash\necho "systemd-run $*" >> "$FAKE_FLEET/order.log"\nexit 0\n')
        (shims / "curl").write_text(f"""#!/usr/bin/env bash
echo "curl $*" >> "$FAKE_FLEET/order.log"
case "$*" in *8464/health*)
  if [ "$(grep -c 8464/health "$FAKE_FLEET/order.log")" -ge 3 ]; then echo '{{"ranks":16,"connected_ranks":{connected}}}'; else echo '{{"ranks":16,"connected_ranks":9}}'; fi ;;
esac
exit 0
""")
        (shims / "timeout").write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n')
        for name in ("systemctl", "systemd-run", "curl", "timeout"):
            (shims / name).chmod(0o755)
        (fleet / "rtx5090" / "sparkpipe-swap" / "model_swap.json").write_text((ROOT / "deployment" / "hub" / "model_swap.json").read_text())
        release = fleet / "srv-workspace" / "release-abcdef123"
        (release / "kit").mkdir(parents=True)
        (release / "qwen3.8-max").mkdir()
        (release / "qwen3.8-max" / "new.json").write_text("{}")
        (release / "kit" / "release_capture.py").write_text(
            'import os, sys\nopen(os.environ["FAKE_FLEET"] + "/order.log", "a").write("capture " + " ".join(sys.argv[1:]) + "\\n")\nopen(sys.argv[2], "w").write("{}")\nsys.exit(int(os.environ["FAKE_CAPTURE_RC"]))\n')
        (release / "kit" / "release_compare.py").write_text(
            'import os, sys\nopen(os.environ["FAKE_FLEET"] + "/order.log", "a").write("compare " + " ".join(sys.argv[1:]) + "\\n")\n'
            'print("EXACT 2/2" if os.environ["FAKE_COMPARE_RC"] == "0" else "EXACT 1/2")\nsys.exit(int(os.environ["FAKE_COMPARE_RC"]))\n')
        result = run("restart", fleet, work, old_wd, state, PATH=f"{shims}:{work / 'shims'}:{os.environ['PATH']}", FAKE_COMPARE_RC=str(compare_rc), FAKE_CAPTURE_RC=str(capture_rc))
        order = (fleet / "order.log").read_text().splitlines()

        def first(prefix):
            return next((index for index, line in enumerate(order) if line.startswith(prefix)), None)

        stop, start, health = first("systemctl --user stop kv-qwen38-max-api"), first("systemd-run --user --unit=kv-qwen38-max-api"), first("curl -s -m 3 http://127.0.0.1:8464/health")
        check(None not in (stop, start, health) and stop < start < health, f"restart did not stop the API, start it and then ask its health, in that order: {order}", result)
        captures = [line for line in order if line.startswith("capture ")]
        compares = [line for line in order if line.startswith("compare ")]
        if connected != 16:
            check(result.returncode != 0 and "API-RESTART-FAILED qwen3.8-max: its API did not reach every rank" in result.stdout,
                  "restart passed although the API never reached every rank", result)
            check(captures == [] and compares == [], f"restart captured from an API that had not reached every rank: {captures}", result)
            return
        check(sum(1 for line in order if "8464/health" in line) == 3, "restart did not wait for the health answer that shows every rank connected", result)
        check(captures == [f"capture 8464 qwen3.8-max/restart.json kit/prompt_source.c {fleet / 'srv-workspace'}/logs/qwen38_max-kv-api.log capital,story"],
              f"restart did not capture the two restart prompts from the restarted API: {captures}", result)
        if capture_rc != 0:
            check(result.returncode != 0 and "API-RESTART-FAILED qwen3.8-max: the capture after the restart failed" in result.stdout and compares == [],
                  "restart went on although the capture after the restart failed", result)
            return
        check(compares == ["compare qwen3.8-max/restart.json qwen3.8-max/new.json"] and first("capture ") < first("compare "),
              f"restart did not compare the restart capture with the capture taken before it: {compares}", result)
        if compare_rc != 0:
            check(result.returncode != 0 and "API-RESTART-FAILED qwen3.8-max: the restarted API's outputs differ" in result.stdout,
                  "restart passed although the outputs after the restart differ", result)
            return
        check(result.returncode == 0 and "api restart qwen3.8-max: outputs equal" in result.stdout and "API-RESTART-FAILED" not in result.stdout,
              "restart failed on an API that came back with equal outputs", result)
    return body


def roots(work, fleet, state, model_lanes, extra, old_wd):
    lane, previous = model_lanes[0][0], model_lanes[0][0].rsplit("/", 1)[0] + "/lane1"
    record = "qwen3.8-max"
    prev = work / "prev"
    shutil.copytree(ROOT / "tools", prev / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "deployment", prev / "deployment")
    records = json.loads((prev / "deployment" / "lanes.json").read_text())
    entry = records["lanes"][record]
    entry["lane"] = 1
    entry["runtime_root"] = entry["runtime_root"].replace(lane.rsplit("/", 1)[1] + "/root", "lane1/root")
    (prev / "deployment" / "lanes.json").write_text(json.dumps(records, indent=1) + "\n")
    for index in range(16):
        home = fleet / f"spark{HEX[index]}"
        shutil.rmtree(home / lane / "root")
        old = home / previous / "root"
        for name, text in (("bin/residentd", "previous"), ("kvsnapshot/page", "kv"), ("kvcache/block", "kv"), ("logs/residentd-1.log", "log")):
            (old / name).parent.mkdir(parents=True, exist_ok=True)
            (old / name).write_text(text)
    result = run("roots", fleet, work, old_wd, state)
    check(result.returncode != 0 and f"ROOTS-FAILED {record}" in result.stdout, "roots without PREV_TREE must name the record whose root is missing", result)
    check(not any((fleet / f"spark{HEX[index]}" / lane / "root").exists() for index in range(16)), "roots without PREV_TREE created a root", result)
    result = run("backup", fleet, work, old_wd, state)
    check(result.returncode != 0 and "run the roots step first" in result.stdout, "backup must refuse a record whose root is missing", result)
    check(not any((fleet / f"spark{HEX[index]}" / lane / f"root.prev-{STAMP}" / "bin" / "residentd").read_text() != "old" for index in range(16)),
          "backup changed a backup before refusing", result)
    result = run("roots", fleet, work, old_wd, state, PREV_TREE=str(prev))
    check(result.returncode == 0, "roots with PREV_TREE failed", result)
    for index in range(16):
        home = fleet / f"spark{HEX[index]}"
        new = home / lane / "root"
        check((new / "bin" / "residentd").read_text() == "previous", f"spark{HEX[index]} root was not copied from the previous root", result)
        check(all((new / name).is_dir() and not any((new / name).iterdir()) for name in ("kvsnapshot", "kvcache", "logs")),
              f"spark{HEX[index]} root kept KV files or logs", result)
        check((home / previous / "root" / "kvsnapshot" / "page").read_text() == "kv", f"spark{HEX[index]} previous root changed", result)


fleet_case("stage verifies all 16 nodes", stage(""))
fleet_case("backup puts the hub API backups on the workspace volume, not in the home", backup(104857600))
fleet_case("backup refuses before any change when the hub workspace is short of space", backup(1048576))
fleet_case("install puts one identical release on all 16 nodes and the hub", install(""))
fleet_case("install fails when one node has no staged release", install("spark9"))
fleet_case("install is not blocked by a process that only names the residentd binary",
           install("", "pgrep -f bin/sparkpipe_model_residentd"))
fleet_case("records frees each lane's current pack on all 16 nodes, then writes every lane's rendered files and installs and restarts the release's controller config", records(""))
fleet_case("records stops before any push when one node's weightd holds the lane's current pack", records("spark9"))
fleet_case("install refuses while a residentd runs", install("", "./bin/sparkpipe_model_residentd --deployment deployment.json --rank-index 6"))
fleet_case("stage fails when one node is not staged", stage("spark9"))
fleet_case("roots copies a renumbered record's root from the running release without KV files or logs, and backup refuses until it exists", roots)
fleet_case("rollback refuses without a backup stamp", refusal(lambda fleet, state, lanes: (state / "utc").unlink()))
fleet_case("rollback refuses when one node misses one lane backup",
           refusal(lambda fleet, state, lanes: shutil.rmtree(fleet / "spark7" / lanes[3][0] / f"root.prev-{STAMP}")))
fleet_case("rollback refuses when a hub API backup is missing",
           refusal(lambda fleet, state, lanes: shutil.rmtree(fleet / "srv-workspace" / "backups" / f"{lanes[0][1]}.prev-{STAMP}")))
fleet_case("rollback refuses when one node misses the weightd_warm backup",
           refusal(lambda fleet, state, lanes: (fleet / "sparkc" / "sparkdata" / "swap" / f"weightd_warm.prev-{STAMP}").unlink()))
fleet_case("complete rollback restores everything and stops on a weightd that is not ready", complete_rollback)
fleet_case("restart stops and starts a serving API, waits for every rank, and compares two prompts with the capture before it", restart(16, 0))
fleet_case("restart fails without capturing when the API does not reach every rank", restart(15, 0))
fleet_case("restart fails when the outputs after the restart differ", restart(16, 1))
fleet_case("restart fails when the capture after the restart fails", restart(16, 0, 3))
print("PASS test_release_rollout: stage copies the release to the hub once and needs all 16 nodes verified; roots copies a renumbered record\'s root from the running release without KV files or logs and backup refuses until every root exists; backup copies the hub API dirs to the workspace volume and refuses when it is short of space; install needs one identical result from all 16 and no running residentd (a command that only names the binary is not one); records frees the cold arena of each lane's current pack on all 16 nodes before it pushes that lane's record, stops before any push when a node reports it busy, and installs and restarts the release's controller config; rollback refuses an incomplete backup before changing anything, restores every lane, hub API, weightd_warm and the controller config from a complete one, and never activates on an old weightd that is not ready; restart stops and starts a serving API with the controller's commands, waits until its health shows every rank, captures two prompts and compares them with the capture before it, and fails on an API that does not come back, a failed capture or different outputs")

#!/usr/bin/env python3
"""Run scripts/gx_gate.sh against a fake 16-node fleet, a fake swap controller and a fake gate tool: each step runs with the cache change its plan names; a clear removes the lane's KV snapshots and backing files on all sixteen nodes and the hub API's prefix index, only after the controller stopped every engine, and touches no other lane; an evict removes nothing; a step that stops ends the block with its receipts copied and the fleet released; a block run again on its state does only the open steps, clears the cache before the first of them, and runs the park step again before a restore step; a finished state runs nothing; a step still running on the hub and a fleet with engines the controller did not start are refused before any change."""
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gx_gate.sh"
HEX = "0123456789abcdef"
MODEL = "qwen3.8-27b"
ORDER = ["chain", "park:8000", "restore:8000", "park:100000", "restore:100000", "cold:1"]
SHIMS = {
    "ssh": """#!/usr/bin/env bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host="$1"; shift
HOME="$FAKE_FLEET/$host" exec bash -c "$*"
""",
    "scp": """#!/usr/bin/env bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
resolve() { local host="${1%%:*}" path="${1#*:}"; case "$path" in /*) echo "$path" ;; *) echo "$FAKE_FLEET/$host/$path" ;; esac; }
case "$1" in
  *:*) cp "$(resolve "$1")" "$2" ;;
  *) target="$(resolve "$2")"; mkdir -p "$(dirname "$target")"; if [ "$(basename "$1")" = gx_gate.py ]; then cp "$FAKE_FLEET/fake_gx.py" "$target"; else cp "$1" "$target"; fi ;;
esac
""",
    "curl": """#!/usr/bin/env bash
cat > /dev/null
case "$*" in
  *8440/status*) printf '{"state": "idle", "active": %s, "error": null, "history": [{"seconds": 1.0}]}\\n' "$(cat "$FAKE_FLEET/active")" ;;
  *8440/activate*) model="$(printf '%s' "$*" | sed 's/.*"model":"\\([^"]*\\)".*/\\1/')"; printf '"%s"' "$model" > "$FAKE_FLEET/active"; echo "activate $model" >> "$FAKE_FLEET/events.log" ;;
  *8440/deactivate*) printf null > "$FAKE_FLEET/active"; echo deactivate >> "$FAKE_FLEET/events.log" ;;
  */v1/models*) echo '{"data": [{"max_model_len": 262144}]}' ;;
  *) exit 22 ;;
esac
""",
    "pgrep": """#!/usr/bin/env bash
[ "$(cat "$FAKE_FLEET/active")" != null ] || { [ -e "$FAKE_FLEET/stuck" ] && grep -q activate "$FAKE_FLEET/events.log" 2>/dev/null; } || [ -e "$FAKE_FLEET/foreign" ] || exit 1
echo 4242
""",
    "sleep": "#!/usr/bin/env bash\nexit 0\n",
    "setsid": "#!/usr/bin/env bash\nexec \"$@\"\n",
}
FAKE_GX = """#!/usr/bin/env python3
import glob
import json
import os
import sys

ORDER = (("chain", "none"), ("park:8000", "clear"), ("restore:8000", "evict"), ("park:100000", "none"), ("restore:100000", "evict"), ("cold:1", "clear"))
fleet = os.environ["FAKE_FLEET"]
arguments = sys.argv[1:]


def listing(state):
    return "\\n".join("%s %s %s 0/1" % (name, change, state.get(name, "open")) for name, change in ORDER)


def note(line):
    with open(fleet + "/events.log", "a") as log:
        log.write(line + "\\n")


if arguments[0] == "plan":
    if os.path.exists("state.json"):
        sys.exit(1)
    json.dump({}, open("state.json", "w"))
    note("plan " + " ".join(arguments[1:]))
    print(listing({}))
elif arguments[0] == "steps":
    print(listing(json.load(open("state.json"))))
elif arguments[0] == "run":
    lane = os.environ["FAKE_LANE"]
    snapshots = len(glob.glob(fleet + "/spark*/" + lane + "/root/kvsnapshot/*"))
    backing = len(glob.glob(fleet + "/spark*/" + lane + "/root/kvcache/*"))
    index = os.path.exists(fleet + "/rtx5090/" + os.environ["FAKE_API"] + "/runtime/prefix_index.spi")
    note("run %s active=%s snapshots=%d backing=%d index=%d api=%s log=%s rest=%s over=%s" % (arguments[1], open(fleet + "/active").read(), snapshots, backing, index,
         arguments[arguments.index("--api") + 1], arguments[arguments.index("--api-log") + 1], arguments[arguments.index("--rest-seconds") + 1], arguments[arguments.index("--rest-over-seconds") + 1]))
    for node in glob.glob(fleet + "/spark*/" + lane + "/root"):
        open(node + "/kvsnapshot/" + arguments[1].replace(":", "-"), "w").write("kv")
        open(node + "/kvcache/" + arguments[1].replace(":", "-"), "w").write("kv")
    open(fleet + "/rtx5090/" + os.environ["FAKE_API"] + "/runtime/prefix_index.spi", "w").write("index")
    stops = arguments[1] in os.environ.get("FAKE_GX_FAIL", "").split(",")
    state = json.load(open("state.json"))
    state[arguments[1]] = "stopped" if stops else "done"
    json.dump(state, open("state.json", "w"))
    print("ran " + arguments[1])
    sys.exit(1 if stops else 0)
else:
    state = json.load(open("state.json"))
    open(arguments[arguments.index("--output") + 1], "w").write('{"verdict": "PASS"}')
    print("verdict=PASS")
    sys.exit(0 if all(state.get(name) == "done" for name, _ in ORDER) else 2)
"""


def row(model):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "lane_config.py"), "release-row", model], capture_output=True, text=True, check=True).stdout.split()


def build(fleet):
    lane, api = row(MODEL)[0], row(MODEL)[3]
    other = row("kimi-k3")[0]
    for index in range(16):
        home = fleet / f"spark{HEX[index]}"
        for path in (lane, other):
            for name in ("kvsnapshot", "kvcache"):
                (home / path / "root" / name).mkdir(parents=True)
                (home / path / "root" / name / "old").write_text("kv")
            (home / path / "root" / "packs").mkdir()
            (home / path / "root" / "packs" / "pack.sha256").write_text("digest")
    hub = fleet / "rtx5090"
    (hub / api / "runtime").mkdir(parents=True)
    (hub / api / "runtime" / "prefix_index.spi").write_text("index")
    (hub / api / "runtime" / "other").write_text("keep")
    work = fleet / "srv-workspace"
    (work / "swap-controller").mkdir(parents=True)
    (work / "swap-controller" / "token").write_text("s" * 40 + "\n")
    (fleet / "active").write_text("null")
    (fleet / "fake_gx.py").write_text(FAKE_GX)
    bin_dir = fleet / "bin"
    bin_dir.mkdir()
    for name, body in SHIMS.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    return lane, api, other


def run(fleet, lane, api, **extra):
    out = fleet / "receipts"
    environment = dict(os.environ, PATH=f"{fleet / 'bin'}:{os.environ['PATH']}", FAKE_FLEET=str(fleet), FAKE_LANE=lane, FAKE_API=api, HUB_WORK=str(fleet / "srv-workspace"),
                       GX_RELEASE="testsha", GX_OUT=str(out), **extra)
    log = fleet / "events.log"
    before = len(log.read_text().splitlines()) if log.exists() else 0
    result = subprocess.run(["bash", str(SCRIPT), MODEL], capture_output=True, text=True, env=environment, timeout=300)
    return result, (log.read_text().splitlines() if log.exists() else [])[before:]


failures = []


def check(condition, message, result=None):
    if not condition:
        failures.append(message + (f"\n--- stdout\n{result.stdout[-1500:]}\n--- stderr\n{result.stderr[-800:]}" if result is not None else ""))


def runs(events):
    return {line.split()[1]: line for line in events if line.startswith("run ")}


def order(events):
    return [line.split()[0] + (":" + line.split()[1] if line.startswith("run ") else "") for line in events]


with tempfile.TemporaryDirectory() as directory:
    fleet = pathlib.Path(directory)
    lane, api, other = build(fleet)
    result, events = run(fleet, lane, api, GX_REST="45")
    seen = runs(events)
    check(result.returncode == 0 and f"GX-PASS {MODEL}" in result.stdout, "a passing gate does not exit 0 with GX-PASS", result)
    check([line.split()[1] for line in events if line.startswith("run ")] == ORDER, f"the steps did not run in plan order: {events}", result)
    check(any(line.startswith("plan ") and "--context-limit 262144" in line and "--release testsha" in line and f"--model {MODEL}" in line for line in events), f"the plan does not carry the API's context length, the release and the model: {events}")
    check("snapshots=16 backing=16 index=1" in seen.get("chain", ""), f"the chain step ran on a changed cache: {seen.get('chain')}")
    for step in ("park:8000", "cold:1"):
        check("snapshots=0 backing=0 index=0" in seen.get(step, ""), f"{step} follows a clear but found KV files or the prefix index: {seen.get(step)}")
    check("snapshots=16 backing=16 index=1" in seen.get("restore:8000", ""), f"an evict removed what park:8000 left: {seen.get('restore:8000')}")
    check("snapshots=32 backing=32 index=1" in seen.get("park:100000", ""), f"a step with no cache change found a changed cache: {seen.get('park:100000')}")
    check(all(f'active="{MODEL}"' in line for line in seen.values()), f"a step ran without the model active: {seen}")
    check(all(f"api=http://127.0.0.1:{row(MODEL)[4]}" in line and f"log={fleet / 'srv-workspace'}/logs/{api}.log" in line for line in seen.values()), f"a step did not get the model's API address and log: {seen}")
    check(all("rest=45 over=60" in line for line in seen.values()), f"GX_REST did not reach the steps: {seen}")
    sequence = order(events)
    check(sequence[sequence.index("run:chain") + 1:sequence.index("run:park:8000")] == ["deactivate", "activate"], f"a clear is not one stop and one start through the controller: {sequence}")
    check(sequence[sequence.index("run:park:8000") + 1:sequence.index("run:restore:8000")] == ["deactivate", "activate"], f"an evict is not one stop and one start through the controller: {sequence}")
    check(sequence[sequence.index("run:restore:8000") + 1:sequence.index("run:park:100000")] == [], "a step with no cache change restarted the model")
    check(events[-1] == "deactivate" and (fleet / "active").read_text() == "null", f"the fleet is not left with no model active: {events[-1:]}")
    check(all((fleet / f"spark{HEX[index]}" / other / "root" / name / "old").exists() for index in range(16) for name in ("kvsnapshot", "kvcache")), "a clear touched another lane's KV files")
    check(all((fleet / f"spark{HEX[index]}" / lane / "root" / "packs" / "pack.sha256").exists() for index in range(16)), "a clear touched the lane's pack digest")
    check((fleet / "rtx5090" / api / "runtime" / "other").exists(), "a clear removed more than the prefix index from the API's runtime directory")
    check(all((fleet / "receipts" / f"{MODEL}.{suffix}").exists() for suffix in ("json", "txt", "log")), "the receipt, the verdict table and the run log were not copied back", result)
    result, events = run(fleet, lane, api)
    check(result.returncode == 0 and f"GX-PASS {MODEL}" in result.stdout and events == ["deactivate"], f"a finished state ran steps or restarted the model: {events}", result)

with tempfile.TemporaryDirectory() as directory:
    fleet = pathlib.Path(directory)
    lane, api, other = build(fleet)
    result, events = run(fleet, lane, api, FAKE_GX_FAIL="restore:8000")
    check(result.returncode != 0 and f"GX-BLOCK-STOPPED {MODEL} restore:8000" in result.stdout and f"GX-INCOMPLETE {MODEL}" in result.stdout, "a stopped step is not reported with an incomplete gate", result)
    check(list(runs(events)) == ["chain", "park:8000", "restore:8000"], f"steps ran after a step stopped: {events}")
    check(events[-1] == "deactivate", "the fleet is not released after a stopped step")
    check((fleet / "receipts" / f"{MODEL}.json").exists(), "a stopped block's receipt was not copied back", result)
    result, events = run(fleet, lane, api)
    sequence = order(events)
    seen = runs(events)
    check(result.returncode == 0 and f"GX-PASS {MODEL}" in result.stdout, "the resumed block did not pass", result)
    check(sequence == ["deactivate", "activate", "run:park:8000", "deactivate", "activate", "run:restore:8000", "run:park:100000", "deactivate", "activate", "run:restore:100000",
                       "deactivate", "activate", "run:cold:1", "deactivate"], f"a block resumed at a restore step did not clear, park again, evict and go on: {sequence}", result)
    check("snapshots=0 backing=0 index=0" in seen.get("park:8000", ""), f"the park step run again did not start from a cleared cache: {seen.get('park:8000')}")
    check("snapshots=16 backing=16 index=1" in seen.get("restore:8000", ""), f"the eviction before the resumed restore step removed the parked chains: {seen.get('restore:8000')}")
    check("resumes" in result.stdout and "restore:8000=stopped" in result.stdout, "the resumed block does not say where it resumes", result)

with tempfile.TemporaryDirectory() as directory:
    fleet = pathlib.Path(directory)
    lane, api, other = build(fleet)
    result, events = run(fleet, lane, api, FAKE_GX_FAIL="chain")
    check(result.returncode != 0 and list(runs(events)) == ["chain"] and events[-1] == "deactivate", f"a stopped chain step did not end the block: {events}", result)
    result, events = run(fleet, lane, api)
    sequence = order(events)
    check(sequence[:3] == ["deactivate", "activate", "run:chain"] and "snapshots=0 backing=0 index=0" in runs(events).get("chain", ""), f"a block resumed at its chain step did not clear the cache first: {sequence}", result)
    check(result.returncode == 0 and [name for name in sequence if name.startswith("run:")] == ["run:" + name for name in ORDER], f"the resumed block did not run every step: {sequence}", result)

with tempfile.TemporaryDirectory() as directory:
    fleet = pathlib.Path(directory)
    lane, api, other = build(fleet)
    (fleet / "stuck").write_text("")
    result, events = run(fleet, lane, api)
    check("GX-CHANGE-FAILED" in result.stdout and "GX-BLOCK-STOPPED" in result.stdout and list(runs(events)) == ["chain"], f"a clear went ahead while an engine still ran: {events}", result)
    check(all((fleet / f"spark{HEX[index]}" / lane / "root" / "kvsnapshot" / "old").exists() for index in range(16)), "KV snapshots were removed while an engine still ran")
    check((fleet / "rtx5090" / api / "runtime" / "prefix_index.spi").exists(), "the prefix index was removed while an engine still ran")

with tempfile.TemporaryDirectory() as directory:
    fleet = pathlib.Path(directory)
    lane, api, other = build(fleet)
    (fleet / "foreign").write_text("")
    result, events = run(fleet, lane, api)
    check(result.returncode != 0 and "the gate needs the fleet alone" in result.stdout and not any(line.startswith(("run ", "activate", "plan")) for line in events), f"the gate started beside engines the controller did not start: {events}", result)

if os.path.exists("/proc/self/cmdline"):
    with tempfile.TemporaryDirectory() as directory:
        fleet = pathlib.Path(directory)
        lane, api, other = build(fleet)
        work = fleet / "srv-workspace" / "gx" / "testsha" / MODEL
        work.mkdir(parents=True)
        (work / "state.json").write_text("{}")
        running = subprocess.Popen(["sh", "-c", "sleep 120; : gx_gate.py run chain"])
        try:
            (work / "run.pid").write_text(str(running.pid))
            result, events = run(fleet, lane, api)
            check(result.returncode != 0 and "still runs on the hub" in result.stdout and not any(line.startswith(("run ", "activate", "plan")) for line in events), f"a second driver started beside a running step: {events}", result)
        finally:
            running.kill()
            running.wait()

if failures:
    print("FAIL test_gx_gate_driver:\n" + "\n".join(failures))
    sys.exit(1)
print("PASS test_gx_gate_driver: gx_gate.sh runs each step with the cache change its plan names; a clear is one stop and one start through the controller that removes the lane's KV snapshots, its backing files and the hub API's prefix index only once no engine runs, and leaves other lanes, the pack digest and the API's other files alone; an evict removes nothing; a stopped step ends the block with receipts copied and the fleet released; a block run again does only its open steps after a cache clear, with the park step run again before a restore step; a finished state runs nothing; a running step and engines the controller did not start are refused before any change")

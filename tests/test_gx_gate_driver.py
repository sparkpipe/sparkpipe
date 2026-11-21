#!/usr/bin/env python3
"""Run scripts/gx_gate.sh against a fake 16-node fleet, a fake swap controller and a fake gate tool: each step runs with the cache change its plan names; a clear removes the lane's KV snapshots and backing files on all sixteen nodes and the hub API's prefix index, only after the controller stopped every engine, and touches no other lane; an evict removes nothing; a failed chain step stops the model, a failed park step skips its restore step; the receipts come back; the fleet is left with no model active; a state that already exists and a fleet with engines the controller did not start are refused before any change."""
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gx_gate.sh"
HEX = "0123456789abcdef"
MODEL = "qwen3.8-27b"
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
import os
import sys

fleet = os.environ["FAKE_FLEET"]
arguments = sys.argv[1:]
if arguments[0] == "plan":
    if os.path.exists("state.json"):
        sys.exit(1)
    open("state.json", "w").write("{}")
    with open(fleet + "/events.log", "a") as log:
        log.write("plan " + " ".join(arguments[1:]) + "\\n")
    print("chain none open\\npark:8000 clear open\\nrestore:8000 evict open\\npark:100000 none open\\nrestore:100000 evict open\\ncold:1 clear open")
elif arguments[0] == "run":
    lane = os.environ["FAKE_LANE"]
    snapshots = len(glob.glob(fleet + "/spark*/" + lane + "/root/kvsnapshot/*"))
    backing = len(glob.glob(fleet + "/spark*/" + lane + "/root/kvcache/*"))
    index = os.path.exists(fleet + "/rtx5090/" + os.environ["FAKE_API"] + "/runtime/prefix_index.spi")
    with open(fleet + "/events.log", "a") as log:
        log.write("run %s active=%s snapshots=%d backing=%d index=%d api=%s log=%s\\n" % (arguments[1], open(fleet + "/active").read(), snapshots, backing, index,
                  arguments[arguments.index("--api") + 1], arguments[arguments.index("--api-log") + 1]))
    for node in glob.glob(fleet + "/spark*/" + lane + "/root"):
        open(node + "/kvsnapshot/" + arguments[1].replace(":", "-"), "w").write("kv")
        open(node + "/kvcache/" + arguments[1].replace(":", "-"), "w").write("kv")
    open(fleet + "/rtx5090/" + os.environ["FAKE_API"] + "/runtime/prefix_index.spi", "w").write("index")
    print("ran " + arguments[1])
    sys.exit(1 if arguments[1] in os.environ.get("FAKE_GX_FAIL", "").split(",") else 0)
else:
    open(arguments[arguments.index("--output") + 1], "w").write('{"verdict": "PASS"}')
    print("verdict=PASS")
    sys.exit(int(os.environ.get("FAKE_GX_VERDICT", "0")))
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
    result = subprocess.run(["bash", str(SCRIPT), MODEL], capture_output=True, text=True, env=environment, timeout=300)
    log = fleet / "events.log"
    return result, log.read_text().splitlines() if log.exists() else []


failures = []


def check(condition, message, result=None):
    if not condition:
        failures.append(message + (f"\n--- stdout\n{result.stdout[-1500:]}\n--- stderr\n{result.stderr[-800:]}" if result is not None else ""))


def runs(events):
    return {line.split()[1]: line for line in events if line.startswith("run ")}


with tempfile.TemporaryDirectory() as directory:
    fleet = pathlib.Path(directory)
    lane, api, other = build(fleet)
    result, events = run(fleet, lane, api)
    seen = runs(events)
    check(result.returncode == 0 and f"GX-PASS {MODEL}" in result.stdout, "a passing gate does not exit 0 with GX-PASS", result)
    check([line.split()[1] for line in events if line.startswith("run ")] == ["chain", "park:8000", "restore:8000", "park:100000", "restore:100000", "cold:1"], f"the steps did not run in plan order: {events}", result)
    check(any(line.startswith("plan ") and "--context-limit 262144" in line and "--release testsha" in line and f"--model {MODEL}" in line for line in events), f"the plan does not carry the API's context length, the release and the model: {events}")
    check("snapshots=16 backing=16 index=1" in seen.get("chain", ""), f"the chain step ran on a changed cache: {seen.get('chain')}")
    for step in ("park:8000", "cold:1"):
        check("snapshots=0 backing=0 index=0" in seen.get(step, ""), f"{step} follows a clear but found KV files or the prefix index: {seen.get(step)}")
    check("snapshots=16 backing=16 index=1" in seen.get("restore:8000", ""), f"an evict removed what park:8000 left: {seen.get('restore:8000')}")
    check("snapshots=32 backing=32 index=1" in seen.get("park:100000", ""), f"a step with no cache change found a changed cache: {seen.get('park:100000')}")
    check(all(f'active="{MODEL}"' in line for line in seen.values()), f"a step ran without the model active: {seen}")
    check(all(f"api=http://127.0.0.1:{row(MODEL)[4]}" in line and f"log={fleet / 'srv-workspace'}/logs/{api}.log" in line for line in seen.values()), f"a step did not get the model's API address and log: {seen}")
    order = [line.split()[0] + (":" + line.split()[1] if line.startswith("run ") else "") for line in events]
    before_park = order[order.index("run:chain") + 1:order.index("run:park:8000")]
    check(before_park == ["deactivate", "activate"], f"a clear is not one stop and one start through the controller: {before_park}")
    before_restore = order[order.index("run:park:8000") + 1:order.index("run:restore:8000")]
    check(before_restore == ["deactivate", "activate"], f"an evict is not one stop and one start through the controller: {before_restore}")
    check(order[order.index("run:restore:8000") + 1:order.index("run:park:100000")] == [], "a step with no cache change restarted the model")
    check(events[-1] == "deactivate" and (fleet / "active").read_text() == "null", f"the fleet is not left with no model active: {events[-1:]}")
    check(all((fleet / f"spark{HEX[index]}" / other / "root" / name / "old").exists() for index in range(16) for name in ("kvsnapshot", "kvcache")), "a clear touched another lane's KV files")
    check(all((fleet / f"spark{HEX[index]}" / lane / "root" / "packs" / "pack.sha256").exists() for index in range(16)), "a clear touched the lane's pack digest")
    check((fleet / "rtx5090" / api / "runtime" / "other").exists(), "a clear removed more than the prefix index from the API's runtime directory")
    check(all((fleet / "receipts" / f"{MODEL}.{suffix}").exists() for suffix in ("json", "txt", "log")), "the receipt, the verdict table and the run log were not copied back", result)
    result, events = run(fleet, lane, api)
    check(result.returncode != 0 and "holds a state" in result.stdout and not any(line.startswith(("run ", "activate")) for line in events[len(order):]), "a second run on an existing state was not refused before any change", result)

with tempfile.TemporaryDirectory() as directory:
    fleet = pathlib.Path(directory)
    lane, api, other = build(fleet)
    result, events = run(fleet, lane, api, FAKE_GX_FAIL="chain", FAKE_GX_VERDICT="2")
    check(result.returncode != 0 and f"GX-STEP-FAILED {MODEL} chain" in result.stdout and f"GX-INCOMPLETE {MODEL}" in result.stdout, "a failed chain step is not reported with an incomplete gate", result)
    check(list(runs(events)) == ["chain"], f"steps ran after the chain step failed: {events}")
    check(events[-1] == "deactivate", "the fleet is not released after a failed chain step")

with tempfile.TemporaryDirectory() as directory:
    fleet = pathlib.Path(directory)
    lane, api, other = build(fleet)
    result, events = run(fleet, lane, api, FAKE_GX_FAIL="park:8000", FAKE_GX_VERDICT="2")
    check(list(runs(events)) == ["chain", "park:8000", "park:100000", "restore:100000", "cold:1"], f"a failed park step did not skip only its restore step: {events}", result)
    check(f"GX-STEP-SKIPPED {MODEL} restore:8000" in result.stdout, "the skipped restore step is not named", result)

with tempfile.TemporaryDirectory() as directory:
    fleet = pathlib.Path(directory)
    lane, api, other = build(fleet)
    (fleet / "stuck").write_text("")
    result, events = run(fleet, lane, api, FAKE_GX_VERDICT="2")
    check("GX-CHANGE-FAILED" in result.stdout and list(runs(events)) == ["chain"], f"a clear went ahead while an engine still ran: {events}", result)
    check(all((fleet / f"spark{HEX[index]}" / lane / "root" / "kvsnapshot" / "old").exists() for index in range(16)), "KV snapshots were removed while an engine still ran")
    check((fleet / "rtx5090" / api / "runtime" / "prefix_index.spi").exists(), "the prefix index was removed while an engine still ran")

with tempfile.TemporaryDirectory() as directory:
    fleet = pathlib.Path(directory)
    lane, api, other = build(fleet)
    (fleet / "foreign").write_text("")
    result, events = run(fleet, lane, api)
    check(result.returncode != 0 and "the gate needs the fleet alone" in result.stdout and not any(line.startswith(("run ", "activate", "plan")) for line in events), f"the gate started beside engines the controller did not start: {events}", result)

if failures:
    print("FAIL test_gx_gate_driver:\n" + "\n".join(failures))
    sys.exit(1)
print("PASS test_gx_gate_driver: gx_gate.sh runs each step with the cache change its plan names; a clear is one stop and one start through the controller that removes the lane's KV snapshots, its backing files and the hub API's prefix index only once no engine runs, and leaves other lanes, the pack digest and the API's other files alone; an evict removes nothing; a failed chain step stops the model and a failed park step skips its restore step; receipts are copied back and the fleet is left with no model active; an existing state and engines the controller did not start are refused before any change")

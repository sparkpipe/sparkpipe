#!/usr/bin/env python3
"""release_rollout.sh gx hands the release's served models to scripts/gx_gate.sh with the release commit, a hub directory inside the release's work directory, a receipt directory inside the kit state, and qwen3.8-max as the model to leave active; ONLY limits the models; a gate that does not pass fails the step."""
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHA = "abcdef123"
SERVED = "kimi-k3.1-pro kimi-k3 glm-5.3 glm-5.3-flash qwen3.8-27b qwen3.8-max deepseek-v4.1-flash"
SHIMS = {
    "ssh": """#!/usr/bin/env bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host="$1"; shift
HOME="$FAKE_FLEET/$host" exec bash -c "$*"
""",
    "curl": "#!/usr/bin/env bash\ncat > /dev/null\necho '{\"state\": \"idle\", \"active\": \"qwen3.8-max\", \"error\": null}'\n",
}
STUB = """#!/usr/bin/env bash
echo "release=$GX_RELEASE dir=$GX_DIR out=$GX_OUT end=$GX_END models=$*" >> "$FAKE_FLEET/gx.log"
exit "${FAKE_GX_STATUS:-0}"
"""


def run(base, **extra):
    fleet = base / "fleet"
    environment = dict(os.environ, PATH=f"{base / 'bin'}:{os.environ['PATH']}", FAKE_FLEET=str(fleet), HUB_WORK=str(fleet / "srv-workspace"), FW=str(base / "fw"),
                       APIB="/nonexistent/build", OLD_WD="0" * 16, STATE=str(base / "state"), **extra)
    result = subprocess.run(["bash", str(base / "kit" / "scripts" / "release_rollout.sh"), "gx"], capture_output=True, text=True, env=environment, timeout=600)
    log = fleet / "gx.log"
    lines = log.read_text().splitlines() if log.exists() else []
    if log.exists():
        log.unlink()
    return result, lines


failures = []


def check(condition, message, result):
    if not condition:
        failures.append(f"{message}\n--- stdout\n{result.stdout[-800:]}\n--- stderr\n{result.stderr[-800:]}")


with tempfile.TemporaryDirectory() as directory:
    base = pathlib.Path(directory)
    (base / "kit" / "scripts").mkdir(parents=True)
    for name in ("release_rollout.sh", "rollover_lib.sh"):
        (base / "kit" / "scripts" / name).write_text((ROOT / "scripts" / name).read_text())
    (base / "kit" / "scripts" / "gx_gate.sh").write_text(STUB)
    (base / "kit" / "scripts" / "gx_gate.sh").chmod(0o755)
    for name in ("tools", "deployment"):
        (base / "kit" / name).symlink_to(ROOT / name)
    (base / "fw").mkdir()
    (base / "fw" / "SOURCE_COMMIT").write_text(SHA + "\n")
    (base / "fw" / "sparkpipe_weightd").write_text("weightd")
    (base / "bin").mkdir()
    for name, body in SHIMS.items():
        (base / "bin" / name).write_text(body)
        (base / "bin" / name).chmod(0o755)
    work = base / "fleet" / "srv-workspace"
    (work / "swap-controller").mkdir(parents=True)
    (work / "swap-controller" / "token").write_text("s" * 40 + "\n")
    (base / "fleet" / "rtx5090").mkdir()

    result, lines = run(base)
    check(result.returncode == 0, "the gx step failed although the gate passed", result)
    check(lines == [f"release={SHA} dir={work}/release-{SHA}/gx out={base / 'state'}/gx end=qwen3.8-max models={SERVED}"], f"the gate was not called once with the release's served models and directories: {lines}", result)
    result, lines = run(base, ONLY="qwen3.8-27b glm-5.3")
    check(result.returncode == 0 and len(lines) == 1 and lines[0].endswith("models=qwen3.8-27b glm-5.3"), f"ONLY did not limit the models: {lines}", result)
    result, lines = run(base, FAKE_GX_STATUS="1")
    check(result.returncode != 0 and "GX-FAILED" in result.stdout, "a gate that did not pass did not fail the step", result)

if failures:
    print("FAIL test_release_rollout_gx:\n" + "\n".join(failures))
    sys.exit(1)
print("PASS test_release_rollout_gx: release_rollout.sh gx runs scripts/gx_gate.sh for the release's served models with the release commit, a hub directory under the release's work directory, receipts under the kit state and qwen3.8-max left active; ONLY limits the models; a gate that does not pass fails the step")

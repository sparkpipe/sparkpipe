#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAKE_SSH = """#!/usr/bin/env bash
host="${@: -2:1}"
command="${@: -1}"
HOST="$host" exec bash -c "${command//\\/home\\//$FAKE_HOME/}"
"""
FAKE_SYSTEMCTL = """#!/usr/bin/env bash
case " $* " in
  *" is-active "*) [ -e "$FAKE_HOME/$HOST.stuck" ] ;;
  *) exit 0 ;;
esac
"""
LANE_SCRIPT = """#!/usr/bin/env bash
case "$1" in
  stop) ! [ -e "{flag}" ] ;;
  *) exit 2 ;;
esac
"""


def run_stop(home, stuck_hosts, failing_lane_hosts):
    bin_dir = home / "bin"
    for path in home.glob("*.stuck"):
        path.unlink()
    for host in stuck_hosts:
        (home / f"{host}.stuck").write_text("")
    for rank in range(16):
        host = f"spark{rank:x}"
        root = home / host / "stage-lanes" / "glm53_full" / "lane3" / "root"
        lane = root / "lane.sh"
        if host in failing_lane_hosts:
            root.mkdir(parents=True, exist_ok=True)
            flag = root / "stop-fails"
            flag.write_text("")
            lane.write_text(LANE_SCRIPT.format(flag=flag))
            lane.chmod(0o755)
        elif lane.exists():
            lane.unlink()
    environment = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FAKE_HOME=str(home),
                       STAGE_MODEL="glm53_full", STAGE_LANE="3", STAGE_FIRMWARE="/unused", STAGE_WEIGHTD_SOCKET="/unused",
                       STAGE_MEMORY_MAX="1G")
    result = subprocess.run(["bash", str(ROOT / "tools" / "stage_lane.sh"), "stop"], env=environment,
                            capture_output=True, text=True, timeout=120)
    return result.returncode, result.stderr


def main():
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        home = pathlib.Path(scratch)
        (home / "bin").mkdir()
        for name, body in (("ssh", FAKE_SSH), ("systemctl", FAKE_SYSTEMCTL)):
            path = home / "bin" / name
            path.write_text(body)
            path.chmod(0o755)
        code, stderr = run_stop(home, [], [])
        if code != 0:
            failures.append(f"a stop that leaves every unit inactive succeeds (exit {code}: {stderr.strip()[-200:]})")
        code, stderr = run_stop(home, ["spark7"], [])
        if code == 0 or "spark7" not in stderr:
            failures.append(f"a unit still active after stop on a host without lane.sh fails the stop and names the host (exit {code})")
        code, stderr = run_stop(home, [], ["sparkb"])
        if code == 0 or "sparkb" not in stderr:
            failures.append(f"a failing lane.sh stop fails the stop and names the host (exit {code})")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS stage_lane stop: a clean stop succeeds; a unit left active or a failing lane.sh stop fails the stop and names the host")


if __name__ == "__main__":
    main()

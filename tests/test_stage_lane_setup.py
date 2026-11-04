#!/usr/bin/env python3
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAKE_SSH = """#!/usr/bin/env bash
host="${@: -2:1}"
command="${@: -1}"
printf '%s %s\\n' "$host" "$command" >> "$FAKE_HOME/ssh.log"
case "$command" in
  "systemctl --user is-active "*) if [ -e "$FAKE_HOME/$host.state" ]; then cat "$FAKE_HOME/$host.state"; else echo inactive; exit 3; fi ;;
  *) exit 1 ;;
esac
"""
FAKE_SCP = "#!/usr/bin/env bash\nexit 1\n"


def run_setup(home, states):
    for path in home.glob("*.state"):
        path.unlink()
    for host, state in states.items():
        (home / f"{host}.state").write_text(state + "\n")
    log = home / "ssh.log"
    log.write_text("")
    environment = dict(os.environ, PATH=f"{home / 'bin'}:{os.environ['PATH']}", FAKE_HOME=str(home),
                       STAGE_MODEL="glm53_full", STAGE_LANE="3", STAGE_FIRMWARE=str(home / "firmware"), STAGE_WEIGHTD_SOCKET="/unused",
                       STAGE_MEMORY_MAX="1G")
    result = subprocess.run(["bash", str(ROOT / "tools" / "stage_lane.sh"), "setup"], env=environment,
                            capture_output=True, text=True, timeout=300)
    commands = [line.split(" ", 1)[1] for line in log.read_text().splitlines()]
    return result.returncode, result.stderr, commands


def main():
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        home = pathlib.Path(scratch)
        (home / "bin").mkdir()
        (home / "firmware").mkdir()
        for name, body in (("ssh", FAKE_SSH), ("scp", FAKE_SCP)):
            path = home / "bin" / name
            path.write_text(body)
            path.chmod(0o755)
        for states, why in (({"spark5": "active"}, "an active unit"), ({"sparkb": "activating"}, "a unit still starting"),
                            ({"spark0": "deactivating", "sparkf": "active"}, "units on two hosts")):
            code, stderr, commands = run_setup(home, states)
            touched = [command for command in commands if not command.startswith("systemctl --user is-active ")]
            if code == 0 or not all(host in stderr for host in states) or "stop the lane first" not in stderr:
                failures.append(f"setup with {why} must refuse and name every host (exit {code}: {stderr.strip()[-200:]})")
            if touched:
                failures.append(f"setup with {why} must not touch any node before refusing; it ran {touched[:2]}")
            if len(commands) != 16:
                failures.append(f"setup with {why} must check the unit on all 16 ranks before refusing; it checked {len(commands)}")
        code, stderr, commands = run_setup(home, {})
        if "is running on" in stderr or not any("-mindepth 1 -delete" in command for command in commands):
            failures.append(f"setup on a stopped lane must pass the check and go on to prepare the nodes (exit {code}: {stderr.strip()[-200:]})")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS stage_lane setup: an active, starting or stopping unit on any rank refuses setup before any node is touched and names the hosts; a stopped lane goes on to prepare the nodes")


if __name__ == "__main__":
    main()

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
  *"-mindepth 1 -delete"*|"scp -q -o BatchMode=yes "*) exit 0 ;;
  "systemctl --user is-active "*) if [ -e "$FAKE_HOME/$host.state" ]; then cat "$FAKE_HOME/$host.state"; else echo inactive; exit 3; fi ;;
  *) exit 1 ;;
esac
"""
FAKE_SCP = "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_HOME/scp.log\"\nexit 1\n"


def run_setup(home, states, firmware_host=""):
    for path in home.glob("*.state"):
        path.unlink()
    for host, state in states.items():
        (home / f"{host}.state").write_text(state + "\n")
    log = home / "ssh.log"
    log.write_text("")
    (home / "scp.log").write_text("")
    environment = dict(os.environ, PATH=f"{home / 'bin'}:{os.environ['PATH']}", FAKE_HOME=str(home),
                       STAGE_MODEL="glm53_full", STAGE_LANE="3", STAGE_FIRMWARE=str(home / "firmware"), STAGE_WEIGHTD_SOCKET="/unused",
                       STAGE_MEMORY_MAX="1G")
    if firmware_host:
        environment["STAGE_FIRMWARE_HOST"] = firmware_host
    result = subprocess.run(["bash", str(ROOT / "tools" / "stage_lane.sh"), "setup"], env=environment,
                            capture_output=True, text=True, timeout=300)
    commands = [line.split(" ", 1)[1] for line in log.read_text().splitlines()]
    if firmware_host:
        hosts = [line.split(" ", 1) for line in log.read_text().splitlines()]
        return result.returncode, result.stderr, hosts, (home / "scp.log").read_text().splitlines()
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
        firmware = str(home / "firmware")
        code, stderr, hosts, local = run_setup(home, {}, firmware_host="rtx5090")
        relayed = [command for host, command in hosts if host == "rtx5090"]
        wanted = {f"{firmware}/{name}" for name in ("sparkpipe_model_residentd", "weightd_warm", "runtime_libs/libsparkpipe_device_cuda.so",
                                                      "libhidden_transport_spark_host_rdma_verbs.so")}
        named = {word for command in relayed for word in command.split() if word.startswith(firmware)}
        if len(relayed) != 64 or not all(command.startswith("scp -q -o BatchMode=yes ") for command in relayed):
            failures.append(f"setup with STAGE_FIRMWARE_HOST must copy the four firmware sets of every rank through that host; it relayed {len(relayed)}")
        if not wanted <= named or len(named - wanted) != 1:
            failures.append(f"the relayed copies must name files in STAGE_FIRMWARE; they named {sorted(named)[:3]}")
        if any(firmware in line for line in local):
            failures.append("setup with STAGE_FIRMWARE_HOST must not copy firmware from this machine")
        if not all(any(f"spark{digit}:" in command for command in relayed) for digit in "0123456789abcdef"):
            failures.append("the relayed copies must reach every rank's host")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS stage_lane setup: an active, starting or stopping unit on any rank refuses setup before any node is touched and names the hosts; a stopped lane goes on to prepare the nodes; with STAGE_FIRMWARE_HOST every rank's firmware is copied through that host and none from this machine")


if __name__ == "__main__":
    main()

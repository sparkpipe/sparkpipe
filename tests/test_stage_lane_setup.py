#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane

MODEL = "glm53_full"
ADAPTER = stage_lane.MODELS[MODEL]["adapter"]
FAKE_SSH = """#!/usr/bin/env bash
host="${@: -2:1}"
command="${@: -1}"
printf '%s %s\\n' "$host" "$command" >> "$FAKE_HOME/ssh.log"
case "$command" in
  *"-mindepth 1 -delete"*|"scp -q -r -o BatchMode=yes "*) exit 0 ;;
  "test -d "*) [ ! -e "$FAKE_HOME/no-kernels" ] ;;
  "systemctl --user is-active "*) if [ -e "$FAKE_HOME/$host.state" ]; then cat "$FAKE_HOME/$host.state"; else echo inactive; exit 3; fi ;;
  *) exit 1 ;;
esac
"""
FAKE_SCP = "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_HOME/scp.log\"\nexit 1\n"


def run_setup(home, states, firmware, firmware_host=""):
    for path in home.glob("*.state"):
        path.unlink()
    for host, state in states.items():
        (home / f"{host}.state").write_text(state + "\n")
    log = home / "ssh.log"
    log.write_text("")
    (home / "scp.log").write_text("")
    environment = dict(os.environ, PATH=f"{home / 'bin'}:{os.environ['PATH']}", FAKE_HOME=str(home),
                       STAGE_MODEL=MODEL, STAGE_LANE="3", STAGE_FIRMWARE=str(firmware), STAGE_WEIGHTD_SOCKET="/unused",
                       STAGE_MEMORY_MAX="1G")
    if firmware_host:
        environment["STAGE_FIRMWARE_HOST"] = firmware_host
    result = subprocess.run(["bash", str(ROOT / "tools" / "stage_lane.sh"), "setup"], env=environment,
                            capture_output=True, text=True, timeout=300)
    hosts = [line.split(" ", 1) for line in log.read_text().splitlines()]
    return result.returncode, result.stderr, hosts, (home / "scp.log").read_text().splitlines()


def node_commands(hosts):
    return [(host, command) for host, command in hosts if host.startswith("spark") and not command.startswith("systemctl --user is-active ")]


def check_refusal(failures, why, code, stderr, hosts, local, missing, where):
    if code == 0 or missing not in stderr or where not in stderr:
        failures.append(f"setup {why} must refuse and name {missing} {where} (exit {code}: {stderr.strip()[-200:]})")
    if node_commands(hosts) or local:
        failures.append(f"setup {why} must refuse before any node is touched; it ran {(node_commands(hosts) + local)[:2]}")


def main():
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        home = pathlib.Path(scratch)
        (home / "bin").mkdir()
        firmware = home / "firmware"
        firmware.mkdir()
        for name, body in (("ssh", FAKE_SSH), ("scp", FAKE_SCP)):
            path = home / "bin" / name
            path.write_text(body)
            path.chmod(0o755)
        for states, why in (({"spark5": "active"}, "an active unit"), ({"sparkb": "activating"}, "a unit still starting"),
                            ({"spark0": "deactivating", "sparkf": "active"}, "units on two hosts")):
            code, stderr, hosts, _ = run_setup(home, states, firmware)
            commands = [command for _, command in hosts]
            touched = [command for command in commands if not command.startswith("systemctl --user is-active ")]
            if code == 0 or not all(host in stderr for host in states) or "stop the lane first" not in stderr:
                failures.append(f"setup with {why} must refuse and name every host (exit {code}: {stderr.strip()[-200:]})")
            if touched:
                failures.append(f"setup with {why} must not touch any node before refusing; it ran {touched[:2]}")
            if len(commands) != 16:
                failures.append(f"setup with {why} must check the unit on all 16 ranks before refusing; it checked {len(commands)}")
        kernels = firmware / f"{ADAPTER}.kernels"
        code, stderr, hosts, local = run_setup(home, {}, firmware)
        check_refusal(failures, "from a firmware directory without the adapter's kernels directory", code, stderr, hosts, local, str(kernels), "")
        kernels.mkdir()
        code, stderr, hosts, _ = run_setup(home, {}, firmware)
        if "is running on" in stderr or not any("-mindepth 1 -delete" in command for _, command in hosts):
            failures.append(f"setup on a stopped lane must pass the checks and go on to prepare the nodes (exit {code}: {stderr.strip()[-200:]})")
        remote = "/srv/firmware-on-the-firmware-host"
        (home / "no-kernels").write_text("")
        code, stderr, hosts, local = run_setup(home, {}, remote, firmware_host="rtx5090")
        check_refusal(failures, "when STAGE_FIRMWARE_HOST has no kernels directory for the adapter", code, stderr, hosts, local,
                      f"{remote}/{ADAPTER}.kernels", "on rtx5090")
        (home / "no-kernels").unlink()
        code, stderr, hosts, local = run_setup(home, {}, remote, firmware_host="rtx5090")
        through = [command for host, command in hosts if host == "rtx5090"]
        relayed = [command for command in through if command.startswith("scp -q -r -o BatchMode=yes ")]
        if through[:1] != [f"test -d {remote}/{ADAPTER}.kernels"] or len(through) != len(relayed) + 1:
            failures.append(f"setup with STAGE_FIRMWARE_HOST must look for the kernels directory on that host, once, before any copy; it ran {through[:2]}")
        if len(relayed) != 80:
            failures.append(f"setup with STAGE_FIRMWARE_HOST must copy the five firmware sets of every rank, the adapter's kernels directory among them, through that host; it relayed {len(relayed)}")
        wanted = {f"{remote}/{name}" for name in ("sparkpipe_model_residentd", "weightd_warm", ADAPTER, f"{ADAPTER}.kernels",
                                                  "runtime_libs/libsparkpipe_device_cuda.so", "libhidden_transport_spark_host_rdma_verbs.so")}
        named = {word for command in relayed for word in command.split() if word.startswith(remote)}
        if named != wanted:
            failures.append(f"the relayed copies must name exactly the firmware in STAGE_FIRMWARE; they missed {sorted(wanted - named)} and added {sorted(named - wanted)}")
        if any(remote in line for line in local):
            failures.append("setup with STAGE_FIRMWARE_HOST must not copy firmware from this machine")
        order = [(host, command) for host, command in hosts if host.startswith("spark") or command.startswith("scp ")]
        for digit in "0123456789abcdef":
            host = f"spark{digit}"
            lib = f"/home/{host}/stage-lanes/{MODEL}/lane3/root/lib"
            prepared = [index for index, (where, command) in enumerate(order) if where == host and f"rm -rf {lib}/{ADAPTER}.kernels " in command]
            adapter = [index for index, (_, command) in enumerate(order) if command.endswith(f" {remote}/{ADAPTER} {host}:{lib}/{ADAPTER}")]
            copied = [index for index, (_, command) in enumerate(order) if command.endswith(f" {remote}/{ADAPTER}.kernels {host}:{lib}/")]
            if len(adapter) != 1 or len(copied) != 1:
                failures.append(f"{host} must get the adapter once and its kernels directory once into {lib}/ through the firmware host; it got {len(adapter)} and {len(copied)}")
            elif len(prepared) != 1 or prepared[0] > copied[0]:
                failures.append(f"{host} must have its old {ADAPTER}.kernels removed before the new one is copied, so no stale cubin or manifest survives")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS stage_lane setup: an active, starting or stopping unit on any rank refuses setup before any node is touched and names the hosts; "
          "a firmware directory without the adapter's kernels directory refuses setup before any node is touched, on this machine and on STAGE_FIRMWARE_HOST; "
          "a stopped lane goes on to prepare the nodes; with STAGE_FIRMWARE_HOST every rank gets the adapter and its kernels directory, "
          "after its old kernels directory is removed, with the rest of its firmware, all through that host and none from this machine")


if __name__ == "__main__":
    main()

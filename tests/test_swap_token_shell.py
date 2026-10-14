#!/usr/bin/env python3
"""The release scripts reach the swap controller through rollover_lib.sh swap: on the hub it reads the controller token from the workspace and hands curl the Authorization header on stdin, so the token never appears in a command line; without the token file curl never runs and the call fails."""
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOKEN = "s" * 40
SSH = """#!/usr/bin/env bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host="$1"; shift
HOME="$FAKE_FLEET/$host" exec bash -c "$*"
"""
CURL = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_FLEET/curl-argv.log"
cat >> "$FAKE_FLEET/curl-stdin.log"
echo '{"state": "idle", "active": "m", "error": null}'
"""
DRIVER = """. "$ROOT/scripts/rollover_lib.sh"
controller
swap "-X POST localhost:8440/activate -d '{\\"model\\":\\"m\\"}'" > /dev/null && echo ACTIVATE-OK || echo ACTIVATE-FAILED
"""


def main():
    failures = []

    def check(condition, name):
        if not condition:
            failures.append(name)

    with tempfile.TemporaryDirectory() as directory:
        tmp = pathlib.Path(directory)
        shims = tmp / "shims"
        shims.mkdir()
        for name, body in (("ssh", SSH), ("curl", CURL)):
            (shims / name).write_text(body)
            (shims / name).chmod(0o755)
        work = tmp / "work"
        (work / "swap-controller").mkdir(parents=True)
        token = work / "swap-controller" / "token"
        token.write_text(TOKEN + "\n")
        token.chmod(0o600)
        (tmp / "driver.sh").write_text(DRIVER)
        env = dict(os.environ, PATH=f"{shims}:{os.environ['PATH']}", FAKE_FLEET=str(tmp), HUB_WORK=str(work), ROOT=str(ROOT))

        result = subprocess.run(["bash", str(tmp / "driver.sh")], env=env, capture_output=True, text=True, timeout=60)
        argv = (tmp / "curl-argv.log").read_text().splitlines()
        stdin = (tmp / "curl-stdin.log").read_text().splitlines()
        check(result.stdout.splitlines() == ["idle m None", "ACTIVATE-OK"], f"status and activate go through: {result.stdout!r} {result.stderr[-300:]!r}")
        check(argv == ["-sf -H @- localhost:8440/status", "-sf -H @- -X POST localhost:8440/activate -d {\"model\":\"m\"}"], f"curl gets the header from stdin: {argv}")
        check(stdin == ["Authorization: Bearer " + TOKEN] * 2, f"the header carries the workspace token: {stdin}")
        check(not any(TOKEN in line for line in argv), "the token never appears in curl's command line")

        token.unlink()
        result = subprocess.run(["bash", str(tmp / "driver.sh")], env=env, capture_output=True, text=True, timeout=60)
        check("ACTIVATE-FAILED" in result.stdout and "idle" not in result.stdout, f"without the token file the calls fail: {result.stdout!r}")
        check((tmp / "curl-argv.log").read_text().splitlines() == argv, "without the token file curl never runs")

    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS test_swap_token_shell: rollover_lib swap sends the controller token on curl's stdin, never in a command line, and fails before curl without the token file")


if __name__ == "__main__":
    main()

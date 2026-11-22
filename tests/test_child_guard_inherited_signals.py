"""The child guard test holds when its launcher ignores signals.

A shell starts a background job with SIGINT and SIGQUIT ignored, and nohup
ignores SIGHUP; a process inherits those dispositions. The guard installs its
cleanup handler only for signals left at their default, so a victim that
raises an ignored signal used to keep running and test_child_guard waited for
it forever. The victim now restores the default action of the signal it
raises. This test runs build/test_child_guard with SIGINT, SIGQUIT and SIGHUP
ignored, as such a launch leaves them, and requires it to pass within a bound.
"""
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "build" / "test_child_guard"
BOUND_SECONDS = 120


def ignore_launch_signals():
    for number in (signal.SIGINT, signal.SIGQUIT, signal.SIGHUP):
        signal.signal(number, signal.SIG_IGN)


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0 or not BINARY.is_file():
        print(built.stdout + built.stderr, end="")
        print("FAIL build/test_child_guard did not build")
        return 1
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen([str(BINARY)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
                                   preexec_fn=ignore_launch_signals, start_new_session=True)
        try:
            process.wait(timeout=BOUND_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            subprocess.run(["pkill", "-KILL", "-f", str(BINARY) + " --victim"], capture_output=True)
            log.seek(0)
            print(log.read(), end="")
            print("FAIL test_child_guard did not finish in %d s with SIGINT, SIGQUIT and SIGHUP ignored" % BOUND_SECONDS)
            return 1
        log.seek(0)
        output = log.read()
    print(output, end="")
    if process.returncode != 0 or "PASS test_child_guard" not in output:
        print("FAIL test_child_guard exited %d with SIGINT, SIGQUIT and SIGHUP ignored" % process.returncode)
        return 1
    print("PASS test_child_guard holds when its launcher ignores SIGINT, SIGQUIT and SIGHUP")
    return 0


if __name__ == "__main__":
    sys.exit(main())

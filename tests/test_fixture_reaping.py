import contextlib
import ctypes
import glob
import os
import pathlib
import shutil
import signal
import subprocess
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKER = "SPARK_TEST_FIXTURE_OWNER"
RESIDENTD = "sparkpipe_model_residentd"
PR_SET_CHILD_SUBREAPER = 36
SEEN_SECONDS = 120.0
SETTLE_SECONDS = 10.0
EXIT_SECONDS = 30.0
GONE_SECONDS = 5.0
POLL_SECONDS = 0.002
STEPLOOP = ("build/test_steploop_admission", "/tmp/sparkpipe-steploop-%d.json")
API_TEXT = ("build/test_model_api_text", "/tmp/sparkpipe-api-text-%d.json")
RUNNER_MAKEFILE = "run:\n\t@set -e; for test_binary in $(TESTS); do ./$$test_binary; done; echo runner-done\n"
MAKE_VARIABLES = ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES")
HAVE_PROC = os.path.isdir("/proc/self/task")
HAVE_BINARIES = all((ROOT / binary).is_file() for binary, _ in (STEPLOOP, API_TEXT))


def read_bytes(path):
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def stat_of(pid):
    raw = read_bytes("/proc/%d/stat" % pid)
    if raw is None:
        return None
    text = raw.decode("ascii", "replace")
    fields = text[text.rindex(")") + 2:].split()
    return {"state": fields[0], "ppid": int(fields[1]), "pgrp": int(fields[2]), "start": int(fields[19])}


def exe_of(pid):
    try:
        return os.readlink("/proc/%d/exe" % pid)
    except OSError:
        return ""


def cmdline_of(pid):
    raw = read_bytes("/proc/%d/cmdline" % pid)
    return [] if raw is None else [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def marker_of(pid):
    raw = read_bytes("/proc/%d/environ" % pid)
    for entry in (raw or b"").split(b"\0"):
        if entry.startswith(MARKER.encode() + b"="):
            return entry.split(b"=", 1)[1].decode("ascii", "replace")
    return ""


def all_pids():
    return [int(entry) for entry in os.listdir("/proc") if entry.isdigit()]


def children_of(pid):
    if not os.path.exists("/proc/self/task/%d/children" % os.getpid()):
        return {other for other in all_pids() if (stat_of(other) or {}).get("ppid") == pid}
    found = set()
    try:
        tasks = os.listdir("/proc/%d/task" % pid)
    except OSError:
        return found
    for task in tasks:
        raw = read_bytes("/proc/%d/task/%s/children" % (pid, task))
        found.update(int(child) for child in (raw or b"").split())
    return found


def descendants_of(pid):
    found, frontier = set(), [pid]
    while frontier:
        for child in children_of(frontier.pop()):
            if child not in found:
                found.add(child)
                frontier.append(child)
    return found


def alive(pid, start):
    stat = stat_of(pid)
    return stat is not None and stat["start"] == start and stat["state"] not in ("Z", "X")


def present(pid, start):
    stat = stat_of(pid)
    return stat is not None and stat["start"] == start


def is_residentd(pid):
    return exe_of(pid).endswith("/" + RESIDENTD)


def reap(pid):
    try:
        return os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return (pid, 0)


def fixtures_of(test_pid, deployment):
    owner = "%d." % test_pid
    found = []
    for pid in all_pids():
        stat = stat_of(pid)
        if pid == test_pid or stat is None or stat["state"] in ("Z", "X"):
            continue
        if deployment in cmdline_of(pid) or marker_of(pid).startswith(owner):
            found.append((pid, exe_of(pid), " ".join(cmdline_of(pid))))
    return found


@unittest.skipUnless(HAVE_PROC, "needs Linux /proc")
@unittest.skipUnless(HAVE_BINARIES, "build/test_steploop_admission and build/test_model_api_text not built on this host")
class FixtureReapingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fixture-reaping-")
        self.log_path = os.path.join(self.tmp.name, "test.log")
        self.started = []
        self.tests = []
        self.residents = []

    def tearDown(self):
        for pid, start in self.residents + self.tests:
            if present(pid, start):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for runner in self.started:
            if runner.poll() is None:
                runner.kill()
                runner.wait()
        deadline = time.monotonic() + GONE_SECONDS
        for pid, start in self.residents + self.tests:
            while present(pid, start) and time.monotonic() < deadline:
                reap(pid)
                time.sleep(0.01)
        for pid, _ in self.tests:
            for path in glob.glob("/tmp/sparkpipe-*-%d.*" % pid) + glob.glob("/tmp/sparkpipe-*-%d-*" % pid) + \
                    glob.glob(str(ROOT / "build" / ("*-%d.spi" % pid))):
                with contextlib.suppress(OSError):
                    os.unlink(path)
        self.tmp.cleanup()

    def log_tail(self):
        raw = read_bytes(self.log_path) or b""
        return raw[-4000:].decode("utf-8", "replace")

    def launch(self, binary, through_make):
        log = open(self.log_path, "wb")
        env = {key: value for key, value in os.environ.items() if key not in MAKE_VARIABLES}
        if through_make:
            makefile = os.path.join(self.tmp.name, "runner.mk")
            pathlib.Path(makefile).write_text(RUNNER_MAKEFILE)
            argv = [shutil.which("make"), "-s", "-f", makefile, "TESTS=" + binary, "run"]
        else:
            argv = ["./" + binary]
        runner = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        self.started.append(runner)
        return runner

    def find_test(self, runner, binary):
        target = os.path.realpath(ROOT / binary)
        deadline = time.monotonic() + SEEN_SECONDS
        while time.monotonic() < deadline:
            for pid in [runner.pid] + sorted(descendants_of(runner.pid)):
                if os.path.realpath(exe_of(pid)) == target:
                    stat = stat_of(pid)
                    if stat is not None:
                        self.tests.append((pid, stat["start"]))
                        return pid
            self.assertIsNone(runner.poll(), "the runner ended before %s started:\n%s" % (binary, self.log_tail()))
            time.sleep(POLL_SECONDS)
        self.fail("%s never started" % binary)

    def wait_state(self, pid, states):
        deadline = time.monotonic() + SETTLE_SECONDS
        while time.monotonic() < deadline:
            stat = stat_of(pid)
            if stat is not None and stat["state"] in states:
                return
            time.sleep(POLL_SECONDS)
        self.fail("pid %d never reached state %s" % (pid, states))

    def freeze_with_residents(self, test_pid, test_start):
        test_exe = exe_of(test_pid)
        deadline = time.monotonic() + SEEN_SECONDS
        while time.monotonic() < deadline:
            self.assertTrue(alive(test_pid, test_start), "the test ended before a residentd was seen:\n" + self.log_tail())
            if any(is_residentd(child) for child in children_of(test_pid)):
                os.kill(test_pid, signal.SIGSTOP)
                self.wait_state(test_pid, ("T",))
                settle = time.monotonic() + SETTLE_SECONDS
                while any(exe_of(child) == test_exe for child in children_of(test_pid)) and time.monotonic() < settle:
                    time.sleep(POLL_SECONDS)
                residents = sorted(child for child in children_of(test_pid) if is_residentd(child))
                if residents:
                    return residents
                os.kill(test_pid, signal.SIGCONT)
            time.sleep(POLL_SECONDS)
        self.fail("no residentd was seen within %.0f s" % SEEN_SECONDS)

    def check_residents(self, test_pid, residents, deployment):
        for pid in residents:
            stat = stat_of(pid)
            self.assertIsNotNone(stat, "residentd %d vanished while its test was frozen" % pid)
            self.residents.append((pid, stat["start"]))
            self.assertEqual(stat["pgrp"], pid, "residentd %d is not in its own process group" % pid)
            self.assertTrue(marker_of(pid).startswith("%d." % test_pid), "residentd %d lacks the owner marker" % pid)
            self.assertIn(deployment, cmdline_of(pid))

    def wait_test(self, test_pid):
        deadline = time.monotonic() + EXIT_SECONDS
        while time.monotonic() < deadline:
            pid, status = reap(test_pid)
            if pid == test_pid:
                return status
            time.sleep(0.01)
        self.fail("test %d did not exit within %.0f s:\n%s" % (test_pid, EXIT_SECONDS, self.log_tail()))

    def assert_no_fixture(self, test_pid, deployment, already_reaped):
        if already_reaped:
            left = [pid for pid, start in self.residents if present(pid, start)]
            self.assertEqual(left, [], "residentds not reaped by their test before it exited")
        deadline = time.monotonic() + GONE_SECONDS
        while time.monotonic() < deadline:
            for pid, start in self.residents:
                reap(pid)
            left = [pid for pid, start in self.residents if alive(pid, start)]
            found = fixtures_of(test_pid, deployment)
            if not left and not found:
                return
            time.sleep(0.05)
        self.fail("fixtures survived their test %d: residents=%s fixtures=%s" % (test_pid, left, found))

    def start_frozen(self, host, through_make):
        binary, pattern = host
        runner = self.launch(binary, through_make)
        test_pid = self.find_test(runner, binary)
        deployment = pattern % test_pid
        residents = self.freeze_with_residents(test_pid, self.tests[-1][1])
        self.check_residents(test_pid, residents, deployment)
        return runner, test_pid, deployment

    def signal_case(self, host, signal_number):
        runner, test_pid, deployment = self.start_frozen(host, False)
        os.kill(test_pid, signal_number)
        if signal_number != signal.SIGKILL:
            os.kill(test_pid, signal.SIGCONT)
        runner.wait(timeout=EXIT_SECONDS)
        self.assertEqual(runner.returncode, -signal_number, self.log_tail())
        self.assert_no_fixture(test_pid, deployment, signal_number != signal.SIGKILL)

    def make_case(self, host):
        runner, test_pid, deployment = self.start_frozen(host, True)
        runner.send_signal(signal.SIGTERM)
        runner.wait(timeout=EXIT_SECONDS)
        self.assertNotEqual(runner.returncode, 0)
        os.kill(test_pid, signal.SIGCONT)
        status = self.wait_test(test_pid)
        self.assertTrue(os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGTERM,
                        "the test outlived its runner: status=%#x\n%s" % (status, self.log_tail()))
        self.assert_no_fixture(test_pid, deployment, True)

    def fixture_crash_case(self, host):
        runner, test_pid, deployment = self.start_frozen(host, False)
        os.kill(self.residents[-1][0], signal.SIGKILL)
        os.kill(test_pid, signal.SIGCONT)
        runner.wait(timeout=EXIT_SECONDS)
        self.assertLess(runner.returncode, 0, "the test did not die by a signal with a dead rank:\n" + self.log_tail())
        self.assert_no_fixture(test_pid, deployment, True)

    def pass_case(self, host):
        binary, pattern = host
        runner = self.launch(binary, False)
        test_pid = self.find_test(runner, binary)
        runner.wait(timeout=600)
        self.assertEqual(runner.returncode, 0, self.log_tail())
        self.assert_no_fixture(test_pid, pattern % test_pid, True)


GUARD_THREAD = ROOT / "build/test_child_guard_thread"


@unittest.skipUnless(HAVE_PROC and GUARD_THREAD.is_file(), "needs /proc and build/test_child_guard_thread")
class ChildGuardThreadTest(unittest.TestCase):
    def test_signal_on_another_thread(self):
        test = subprocess.run([str(GUARD_THREAD)], capture_output=True, text=True, timeout=EXIT_SECONDS)
        fixture = int(test.stdout.split()[1])
        self.assertEqual(test.returncode, -signal.SIGTERM,
                         "a SIGTERM handled on a helper thread must end the test by SIGTERM, even when the main thread "
                         "fails on the fixture it kills: returncode=%d stderr=%s" % (test.returncode, test.stderr))
        self.assertIsNone(stat_of(fixture), "the fixture %d outlived its test" % fixture)


def add_cases():
    signals = {
        "sigabrt": signal.SIGABRT,
        "sigterm": signal.SIGTERM,
        "sigint": signal.SIGINT,
        "sighup": signal.SIGHUP,
        "sigpipe": signal.SIGPIPE,
        "sigkill": signal.SIGKILL,
    }
    for host_name, host in (("steploop", STEPLOOP), ("api_text", API_TEXT)):
        for signal_name, signal_number in signals.items():
            setattr(FixtureReapingTest, "test_%s_%s" % (host_name, signal_name),
                    lambda self, host=host, number=signal_number: self.signal_case(host, number))
        setattr(FixtureReapingTest, "test_%s_make_sigterm" % host_name, lambda self, host=host: self.make_case(host))
        setattr(FixtureReapingTest, "test_%s_pass" % host_name, lambda self, host=host: self.pass_case(host))
    setattr(FixtureReapingTest, "test_steploop_fixture_crash", lambda self: self.fixture_crash_case(STEPLOOP))


add_cases()

if __name__ == "__main__":
    unittest.main()

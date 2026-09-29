import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "stray_fixture_processes.py"
MARKER = "SPARK_TEST_FIXTURE_OWNER"


def stat_line(pid, state, ppid, start):
    rest = ["0"] * 17
    return "%d (x y) %s %d %s %s\n" % (pid, state, ppid, " ".join(rest), start)


class FakeProc:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        (self.root / "self").mkdir()
        (self.root / "uptime").write_text("1000.00 0.00\n")

    def add(self, pid, exe, cwd, argv, env=None, ppid=1, start="50000", state="S"):
        base = self.root / str(pid)
        base.mkdir()
        (base / "stat").write_text(stat_line(pid, state, ppid, start))
        (base / "environ").write_bytes(b"".join(("%s=%s" % kv).encode() + b"\0" for kv in (env or {}).items()))
        (base / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))
        os.symlink(exe, base / "exe")
        os.symlink(cwd, base / "cwd")


class StrayFixtureListerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="stray-proc-")
        self.proc = FakeProc(self.tmp.name)
        rd = "/home/sparkf/build-lane/build/sparkpipe_model_residentd"
        self.proc.add(100, "/home/sparkf/build-lane/build/test_model_api_text", "/home/sparkf/build-lane",
                      ["./build/test_model_api_text"], env={MARKER: "100.500"}, start="500")
        self.proc.add(101, rd, "/home/sparkf/build-lane", [rd, "--deployment", "/tmp/sparkpipe-api-text-100.json"],
                      env={MARKER: "100.500"}, ppid=100)
        self.proc.add(102, rd, "/home/sparkf/build-old (deleted)", [rd, "--deployment", "/tmp/sparkpipe-api-text-7.json"],
                      env={MARKER: "7.300"})
        self.proc.add(103, rd, "/home/sparkf/build-lane", [rd], env={MARKER: "100.499"})
        self.proc.add(104, rd, "/home/sparkf/build-gone (deleted)", [rd, "--deployment", "/tmp/sparkpipe-steploop-9.json"])
        self.proc.add(105, "/home/spark0/release/bin/sparkpipe_model_residentd", "/home/spark0/sparkdata/glm53flash.fp8.tp16",
                      ["/home/spark0/release/bin/sparkpipe_model_residentd", "--deployment", "model_resident.json"])
        self.proc.add(106, rd, "/home/sparkf/build-lane", [rd, "--deployment", "/tmp/sparkpipe-model-resident-100-0.json"], ppid=100)
        self.proc.add(107, rd, "/home/sparkf/build-lane", [rd], env={MARKER: "8.1"}, state="Z")
        self.proc.add(108, "/usr/bin/sleep", "/tmp (deleted)", ["sleep", "300"])
        self.proc.add(109, "/usr/bin/sleep", "/home/sparkf/other", ["sleep", "300"], env={MARKER: "8.1"})

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *extra):
        return subprocess.run([sys.executable, str(TOOL), "--proc", self.tmp.name, *extra],
                              capture_output=True, text=True, timeout=60)

    def listed(self, out, kind):
        return sorted(int(line.split()[1].split("=")[1]) for line in out.splitlines() if line.startswith(kind + " "))

    def test_lists_only_strays_and_exits_one(self):
        result = self.run_tool()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self.listed(result.stdout, "STRAY"), [102, 103, 104, 109])
        self.assertEqual(self.listed(result.stdout, "RUNNING"), [])
        self.assertIn("STRAY-FIXTURES n=4 running=2 stop them with: kill -KILL 102 103 104 109", result.stdout)

    def test_all_shows_fixtures_of_live_tests(self):
        result = self.run_tool("--all")
        self.assertEqual(self.listed(result.stdout, "RUNNING"), [101, 106])
        self.assertIn("owner 100.500 alive", result.stdout)
        self.assertIn("test parent 100 alive", result.stdout)

    def test_owner_pid_reuse_is_a_stray(self):
        result = self.run_tool()
        line = [l for l in result.stdout.splitlines() if l.startswith("STRAY pid=103 ")][0]
        self.assertIn("owner 100.499 gone", line)

    def test_production_engine_is_never_listed(self):
        result = self.run_tool("--all")
        shown = set(self.listed(result.stdout, "STRAY") + self.listed(result.stdout, "RUNNING"))
        self.assertEqual(shown & {100, 105, 107, 108}, set())

    def test_under_filters_by_exe_or_cwd(self):
        result = self.run_tool("--under", "/home/sparkf/build-gone")
        self.assertEqual(self.listed(result.stdout, "STRAY"), [104])
        result = self.run_tool("--under", "/home/sparkf/build-lan")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.listed(result.stdout, "STRAY"), [])

    def test_clean_host_exits_zero(self):
        for pid in (102, 103, 104, 109):
            (pathlib.Path(self.tmp.name) / str(pid) / "stat").write_text(stat_line(pid, "Z", 1, "1"))
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("STRAY-FIXTURES n=0 running=2", result.stdout)


@unittest.skipUnless(os.path.isdir("/proc/self"), "needs Linux /proc")
class StrayFixtureListerLiveTest(unittest.TestCase):
    def test_live_orphan_marker_is_listed(self):
        env = dict(os.environ)
        env[MARKER] = "999999999.1"
        child = subprocess.Popen(["sleep", "60"], env=env)
        try:
            deadline = time.time() + 5
            while time.time() < deadline:
                result = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True, timeout=60)
                if "STRAY pid=%d " % child.pid in result.stdout:
                    break
                time.sleep(0.1)
            self.assertIn("STRAY pid=%d " % child.pid, result.stdout)
            self.assertIn("owner 999999999.1 gone", result.stdout)
            self.assertEqual(result.returncode, 1)
        finally:
            child.send_signal(signal.SIGKILL)
            child.wait()


if __name__ == "__main__":
    unittest.main()

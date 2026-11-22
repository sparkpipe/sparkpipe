import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
AGENT = (ROOT / "tools/fleet_node_agent.sh").read_text()
HEAD = AGENT[:AGENT.index('\necho "$$" > "$PID_FILE"')] + "\n"
HELPER = (ROOT / "tools/hosted_image/files/usr/local/sbin/sparkpipe-compact").read_text()

SUDO = '''#!/usr/bin/env bash
[ "$1" = -n ] && shift
exec "$@"
'''

RESIDENTD = '''#!/usr/bin/env bash
case "${TEST_RESIDENTD:-ready}" in
ready) sleep 1; echo "model_residentd ready" >> residentd.log; exec sleep 60 ;;
fail) exit 3 ;;
hang) exec sleep 60 ;;
esac
'''


def write(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("flock") and shutil.which("setsid"),
                     "the agent's compaction window needs Linux, flock and setsid")
class CompactionWindowTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.state = self.base / "state"
        self.state.mkdir()
        (self.state / "window").write_text("")
        helper = HELPER
        for real, fake in (("/run/sparkpipe-compact", self.state), ("/proc/sys/vm/compaction_proactiveness", self.base / "proactiveness"),
                           ("/proc/sys/vm/min_free_kbytes", self.base / "reserve"), ("/proc/sys/vm/compact_memory", self.base / "compact")):
            helper = helper.replace(real, str(fake))
        helper = helper.replace("case \"${1:-}\" in", "echo \"$1\" >> \"" + str(self.base / "calls") + "\"\ncase \"${1:-}\" in", 1)
        write(self.base / "bin" / "sparkpipe-compact", helper, 0o755)
        write(self.base / "bin" / "sudo", SUDO, 0o755)
        (self.base / "proactiveness").write_text("20\n")
        (self.base / "reserve").write_text("45056\n")
        self.root = self.base / "root"
        write(self.root / "bin" / "sparkpipe_model_residentd", RESIDENTD, 0o755)

    def tearDown(self):
        subprocess.run(["pkill", "-f", str(self.root)], check=False)
        subprocess.run(["pkill", "-f", "compact-watch " + str(self.root)], check=False)
        shutil.rmtree(self.base, ignore_errors=True)

    def agent(self, script, helper=None, seconds=900, residentd="ready"):
        environment = dict(os.environ, PATH=f"{self.base / 'bin'}:{os.environ['PATH']}", TEST_RESIDENTD=residentd)
        program = "\n".join([
            "set -uo pipefail",
            f"COMPACT_HELPER={helper or self.base / 'bin' / 'sparkpipe-compact'}",
            f"COMPACT_STATE={self.state}",
            f"COMPACT_SECONDS={seconds}",
            "note_root() { echo \"NOTE $*\"; }",
            self.functions(),
            script,
        ])
        return subprocess.run(["bash", "-c", program], capture_output=True, text=True, env=environment, timeout=120)

    def functions(self):
        names = ("match_pids", "compact_watch", "compact_open", "compact_sweep")
        blocks = []
        for name in names:
            start = HEAD.index(f"\n{name}() {{\n") + 1
            end = HEAD.index("\n}\n", start) + 3
            blocks.append(HEAD[start:end])
        return "\n".join(blocks)

    def calls(self):
        path = self.base / "calls"
        return path.read_text().split() if path.exists() else []

    def wait_for(self, predicate, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.2)
        return predicate()

    def launch(self):
        return f'cd {self.root} && : > residentd.log && (nohup ./bin/sparkpipe_model_residentd > /dev/null 2>&1 < /dev/null &)'

    def test_the_window_opens_before_launch_and_closes_when_residentd_is_ready(self):
        result = self.agent(f'compact_open prod {self.root} || exit 9; echo OPENED $(cat {self.base}/proactiveness); {self.launch()}')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OPENED 90", result.stdout)
        self.assertTrue(self.wait_for(lambda: self.calls() == ["raise", "restore"], 10), self.calls())
        self.assertEqual((self.base / "proactiveness").read_text().strip(), "20")
        self.assertEqual((self.base / "reserve").read_text().strip(), "45056")
        self.assertIn("ready", (self.root / "compact.log").read_text())

    def test_a_residentd_that_dies_closes_the_window(self):
        result = self.agent(f'compact_open prod {self.root} || exit 9; {self.launch()}', residentd="fail")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.wait_for(lambda: self.calls() == ["raise", "restore"], 45), self.calls())
        self.assertEqual((self.base / "proactiveness").read_text().strip(), "20")
        self.assertIn("not running", (self.root / "compact.log").read_text())

    def test_a_residentd_that_never_gets_ready_closes_the_window_at_the_bound(self):
        result = self.agent(f'compact_open prod {self.root} || exit 9; {self.launch()}', seconds=3, residentd="hang")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.wait_for(lambda: self.calls() == ["raise", "restore"], 15), self.calls())
        self.assertIn("not ready after 3s", (self.root / "compact.log").read_text())

    def test_the_sweep_closes_a_window_whose_watcher_was_killed(self):
        result = self.agent(f'compact_open prod {self.root} || exit 9; {self.launch()}', residentd="hang")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.wait_for(lambda: self.calls() == ["raise"], 5), self.calls())
        subprocess.run(["pkill", "-KILL", "-f", "compact-watch " + str(self.root)], check=True)
        self.assertEqual((self.base / "proactiveness").read_text().strip(), "90")
        for _ in range(10):
            swept = self.agent("compact_sweep")
            self.assertEqual(swept.returncode, 0, swept.stdout + swept.stderr)
            if self.calls() == ["raise", "restore"]:
                break
            time.sleep(0.5)
        self.assertEqual(self.calls(), ["raise", "restore"])
        self.assertEqual((self.base / "proactiveness").read_text().strip(), "20")

    def test_the_sweep_leaves_an_open_window_alone(self):
        result = self.agent(f'compact_open prod {self.root} || exit 9; {self.launch()}', residentd="hang")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.wait_for(lambda: self.calls() == ["raise"], 5), self.calls())
        swept = self.agent("compact_sweep")
        self.assertEqual(swept.returncode, 0, swept.stdout + swept.stderr)
        self.assertEqual(self.calls(), ["raise"])
        self.assertEqual((self.base / "proactiveness").read_text().strip(), "90")

    def test_a_node_without_the_helper_opens_no_window(self):
        result = self.agent(f'compact_open prod {self.root} && echo SKIPPED', helper=self.base / "absent")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SKIPPED", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_a_raise_that_fails_refuses_the_start(self):
        write(self.base / "bin" / "sparkpipe-compact", "#!/usr/bin/env bash\necho \"$1\" >> " + str(self.base / "calls") + "\n[ \"$1\" = raise ] && exit 1\nexit 0\n", 0o755)
        result = self.agent(f'compact_open prod {self.root} && echo STARTED || echo REFUSED')
        self.assertIn("REFUSED", result.stdout)
        self.assertIn("NOTE prod failed-start the compaction window did not open", result.stdout)
        self.assertNotIn("STARTED", result.stdout)
        self.assertEqual(self.calls(), ["raise", "restore"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools/fleet_node_agent.sh").read_text()
NAMES = ("mesh_push", "mesh_fetch", "mesh_pull", "mesh_exchange_loop", "stop_mesh_exchange", "start_mesh_exchange")


def function(name):
    return name + "() {" + SOURCE.split("\n" + name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"


def constant(name):
    return next(line for line in SOURCE.splitlines() if line.startswith(name + "=")) + "\n"


HARNESS = r'''
set -uo pipefail
FLEET_HOSTS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
RANK=3
HOST=spark3.local
HUB=spec@hub
RELEASE_HTTP=http://hub:8802
HUBSSH=fake_ssh
NOW=${NOW:-1000}
date() { case "$1" in +%s) echo "$NOW";; *) echo 00:00:00;; esac; }
fake_ssh() {
    echo "SSH $2" >> "$WORK/calls"
    [ -z "${HUB_DOWN:-}" ] || return 255
    (cd "$WORK/hub" && bash -c "$2")
}
curl() {
    local z="" out="" url=""
    while [ $# -gt 0 ]; do
        case "$1" in
            -z) z=$2; shift 2 ;;
            -o) out=$2; shift 2 ;;
            --max-time) shift 2 ;;
            -*) shift ;;
            *) url=$1; shift ;;
        esac
    done
    echo "CURL ${url#http://hub:8802/} ${z:+conditional}" >> "$WORK/calls"
    [ -z "${HUB_DOWN:-}" ] || return 7
    local path="$WORK/hub/release/${url#http://hub:8802/}"
    [ -f "$path" ] || return 22
    [ -n "$z" ] && [ ! "$path" -nt "$z" ] && return 0
    if [ -n "$out" ]; then cp -p "$path" "$out"; else cat "$path"; fi
}
''' + constant("MESH_DIR").replace("/tmp/weightd-mesh", "$WORK/mesh") + constant("MESH_PIDFILE").replace("$HOME", "$WORK") + \
    constant("MESH_VERIFY_SECONDS") + constant("MESH_REFRESH_SECONDS") + "".join(function(name) for name in NAMES)


class FleetAgentMeshExchange(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        (self.work / "mesh").mkdir()
        (self.work / "hub").mkdir()
        self.hub_mesh = self.work / "hub/release/qpn"
        for index, host in enumerate("spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf".split()):
            if index != 3:
                self.publish(host, index, b"gen1-%d" % index)

    def tearDown(self):
        self.temp.cleanup()

    def publish(self, host, index, payload, mtime=None):
        path = self.hub_mesh / host / "mesh" / ("mesh-%x.rec" % index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        if mtime is not None:
            os.utime(path, ns=(mtime, mtime))

    def run_steps(self, steps, **env):
        script = HARNESS + "\nMESH_PUSHED=''\nMESH_VERIFIED=0\nMESH_REFRESHED=0\nMESH_HUB_DOWN=''\n" + steps
        environment = {"WORK": str(self.work), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        environment.update(env)
        (self.work / "calls").write_text("")
        result = subprocess.run(["bash", "-c", script], env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result, (self.work / "calls").read_text().splitlines()

    def test_own_record_is_published_once_by_rename_and_verified_against_the_hub(self):
        (self.work / "mesh/mesh-3.rec").write_bytes(b"own-a")
        result, calls = self.run_steps("mesh_push; mesh_push; NOW=1031; mesh_push; rm $WORK/hub/release/qpn/spark3/mesh/mesh-3.rec; NOW=1062; mesh_push")
        pushes = [c for c in calls if c.startswith("SSH")]
        self.assertEqual(len(pushes), 2, calls)
        self.assertIn("mv -f", pushes[0])
        self.assertEqual((self.hub_mesh / "spark3/mesh/mesh-3.rec").read_bytes(), b"own-a")
        self.assertEqual([p.name for p in (self.hub_mesh / "spark3/mesh").iterdir()], ["mesh-3.rec"])
        self.assertIn("differs from the live record", result.stdout)

    def test_a_new_own_record_is_published_immediately(self):
        (self.work / "mesh/mesh-3.rec").write_bytes(b"own-a")
        _, calls = self.run_steps("mesh_push; printf own-b > $WORK/mesh/mesh-3.rec; mesh_push")
        self.assertEqual(len([c for c in calls if c.startswith("SSH")]), 2, calls)
        self.assertEqual((self.hub_mesh / "spark3/mesh/mesh-3.rec").read_bytes(), b"own-b")

    def test_peer_records_are_fetched_then_only_changed_ones_are_replaced(self):
        _, calls = self.run_steps("mesh_pull")
        self.assertEqual(len(calls), 15, calls)
        self.assertTrue(all("conditional" not in c for c in calls), calls)
        self.assertFalse((self.work / "mesh/mesh-3.rec").exists())
        self.assertEqual((self.work / "mesh/mesh-a.rec").read_bytes(), b"gen1-10")
        self.publish("sparka", 10, b"gen2-10", mtime=time.time_ns() + 5 * 10**9)
        _, calls = self.run_steps("NOW=1001; MESH_REFRESHED=1000; mesh_pull")
        self.assertEqual(len(calls), 15, calls)
        self.assertTrue(all("conditional" in c for c in calls), calls)
        self.assertEqual((self.work / "mesh/mesh-a.rec").read_bytes(), b"gen2-10")
        self.assertEqual((self.work / "mesh/mesh-0.rec").read_bytes(), b"gen1-0")
        self.assertEqual(sorted(p.name for p in (self.work / "mesh").iterdir()),
                         sorted("mesh-%x.rec" % i for i in range(16) if i != 3))

    def test_a_full_refresh_ignores_timestamps(self):
        self.run_steps("mesh_pull")
        same_second = os.stat(self.work / "mesh/mesh-5.rec").st_mtime_ns
        self.publish("spark5", 5, b"gen2-5", mtime=same_second)
        self.run_steps("NOW=1001; MESH_REFRESHED=1000; mesh_pull")
        self.assertEqual((self.work / "mesh/mesh-5.rec").read_bytes(), b"gen1-5")
        _, calls = self.run_steps("NOW=1061; MESH_REFRESHED=1000; mesh_pull")
        self.assertTrue(all("conditional" not in c for c in calls), calls)
        self.assertEqual((self.work / "mesh/mesh-5.rec").read_bytes(), b"gen2-5")

    def test_an_unreachable_hub_keeps_records_and_is_reported_once(self):
        self.run_steps("mesh_pull")
        result, _ = self.run_steps("mesh_pull; mesh_pull; HUB_DOWN=; NOW=1007; mesh_pull", HUB_DOWN="1")
        self.assertEqual(result.stderr.count("unreachable"), 1, result.stderr)
        self.assertEqual(result.stdout.count("reachable again after 7s"), 1, result.stdout)
        self.assertEqual((self.work / "mesh/mesh-0.rec").read_bytes(), b"gen1-0")
        self.assertEqual(len(list((self.work / "mesh").iterdir())), 15)

    def test_a_missing_peer_record_leaves_the_old_copy(self):
        self.run_steps("mesh_pull")
        (self.hub_mesh / "spark7/mesh/mesh-7.rec").unlink()
        result, _ = self.run_steps("NOW=1061; mesh_pull")
        self.assertEqual((self.work / "mesh/mesh-7.rec").read_bytes(), b"gen1-7")
        self.assertNotIn("unreachable", result.stderr)

    def test_one_loop_runs_and_it_ends_with_the_agent(self):
        steps = r'''
date() { command date "$@"; }
start_mesh_exchange
first=$(cat "$MESH_PIDFILE")
start_mesh_exchange
second=$(cat "$MESH_PIDFILE")
sleep 0.5
kill -0 "$first" 2>/dev/null && echo FIRST-ALIVE
kill -0 "$second" 2>/dev/null && echo SECOND-ALIVE
stop_mesh_exchange
sleep 0.5
kill -0 "$second" 2>/dev/null && echo SECOND-STILL-ALIVE
[ -f "$MESH_PIDFILE" ] && echo PIDFILE-LEFT
( sleep 1 ) & agent=$!
mesh_exchange_loop "$agent" & loop=$!
wait "$agent"
for _ in 1 2 3 4 5 6; do kill -0 "$loop" 2>/dev/null || break; sleep 0.5; done
kill -0 "$loop" 2>/dev/null && echo ORPHAN-ALIVE
echo DONE
'''
        result, calls = self.run_steps(steps)
        self.assertIn("DONE", result.stdout)
        self.assertNotIn("FIRST-ALIVE", result.stdout)
        self.assertIn("SECOND-ALIVE", result.stdout)
        self.assertNotIn("SECOND-STILL-ALIVE", result.stdout)
        self.assertNotIn("PIDFILE-LEFT", result.stdout)
        self.assertNotIn("ORPHAN-ALIVE", result.stdout)
        self.assertTrue(any(c.startswith("CURL") for c in calls), calls)


if __name__ == "__main__":
    unittest.main()

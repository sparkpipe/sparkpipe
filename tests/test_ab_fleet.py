import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ab_fleet

SPEC_PATH = ROOT / "qualification" / "ab" / "fleet" / "glm53flash_w1.json"
AGENT = (ROOT / "tools" / "fleet_node_agent.sh").read_text()
HEAD = AGENT[:AGENT.index('\necho "$$" > "$PID_FILE"')] + "\n"
PRELUDE = r'''
hostname() { echo spark0; }
ssh() { return 0; }
scp() { return 0; }
ssh-keyscan() { return 0; }
curl() { return 1; }
source "$TEST_AGENT_HEAD" glm53flash.fp8.tp16 sparkf
mem_available_gib() { cat "$TEST_MEM"; }
'''


def bash4():
    if shutil.which("bash") is None:
        return False
    out = subprocess.run(["bash", "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True)
    return out.stdout.strip().isdigit() and int(out.stdout.strip()) >= 4


def spec():
    return json.loads(SPEC_PATH.read_text())


def memory(avail, window=None, lanes=(0,)):
    nodes = {}
    for host in ab_fleet.agent_fleet_hosts():
        nodes[host] = {"mem_available_gib": avail, "window_available_gib": avail if window is None else window,
                       "lanes_in_use": list(lanes)}
    return {"nodes": nodes}


class Spec(unittest.TestCase):
    def test_committed_spec_is_valid(self):
        s = ab_fleet.load_spec(SPEC_PATH)
        self.assertEqual(sorted(a["lane"] for a in s["arms"].values()), [1, 2, 13, 14, 15])

    def test_headroom_and_hosts_follow_the_agent(self):
        self.assertEqual(ab_fleet.agent_headroom_gib(), 20)
        self.assertEqual(ab_fleet.agent_fleet_hosts()[0], "spark0")
        with tempfile.TemporaryDirectory() as tmp:
            agent = Path(tmp) / "agent.sh"
            agent.write_text(AGENT.replace('FLEET_AGENT_HEADROOM_GIB:-20', 'FLEET_AGENT_HEADROOM_GIB:-25'))
            self.assertEqual(ab_fleet.agent_headroom_gib(agent), 25)
            agent.write_text("HEADROOM_GIB=20\n")
            with self.assertRaises(ab_fleet.FleetError):
                ab_fleet.agent_headroom_gib(agent)

    def refused(self, s, text):
        with self.assertRaises(ab_fleet.FleetError) as caught:
            ab_fleet.check_spec(s)
        self.assertIn(text, str(caught.exception))

    def test_lane_rules(self):
        s = spec()
        s["arms"]["F2"]["lane"] = 0
        self.refused(s, "not in the free lanes")
        s = spec()
        s["arms"]["F3"]["lane"] = 15
        self.refused(s, "also arm F2's lane")
        s = spec()
        s["free_lanes"].append(0)
        self.refused(s, "production lane 0 is listed as free")
        s = spec()
        s["free_lanes"].append(16)
        self.refused(s, "not a weightd mesh lane")

    def test_pool_rules(self):
        s = spec()
        s["arms"]["F1AA"]["pool_bytes"] = 34359738368
        self.refused(s, "share arena f1 with different pools")
        s = spec()
        s["arms"]["F0"]["pool_bytes"] = 42949672960
        self.refused(s, "differs from production's")

    def test_root_rules(self):
        s = spec()
        s["arms"]["F2"]["root"] = "glm53flash.fp8.tp16"
        self.refused(s, "production's root")
        s = spec()
        s["arms"]["F3"]["root"] = "qab-f2"
        self.refused(s, "also arm F2's root")
        s = spec()
        s["arms"]["F3"]["root"] = "../prod"
        self.refused(s, "not a fleet root name")

    def test_slot_rules(self):
        s = spec()
        s["slots"][0] = {"name": "A", "start": ["F1AA", "F1"]}
        self.refused(s, "before its owner is up")
        s = spec()
        s["slots"][0] = {"name": "A", "start": ["F1AA"], "optional": ["F1"]}
        self.refused(s, "before its owner is up")
        s = spec()
        s["arms"]["F1AA"]["port_group"] = 0
        self.refused(s, "share a port group")
        s = spec()
        s["common_env"]["SPARK_WEIGHTD_LANE"] = "3"
        self.refused(s, "per arm")
        s = spec()
        s["nodes"] = s["nodes"][:15]
        self.refused(s, "FLEET_HOSTS")


class Env(unittest.TestCase):
    def test_render(self):
        s = ab_fleet.load_spec(SPEC_PATH)
        env = ab_fleet.parse_env_file_text(ab_fleet.render_env(s, "F1"))
        self.assertEqual(env["AGENT_ROLE"], "dev")
        self.assertEqual(env["AGENT_SYNC"], "local")
        self.assertEqual(env["AGENT_MEMORY_NEED_GIB"], "50")
        self.assertEqual(env["SPARK_WEIGHTD_LANE"], "1")
        self.assertEqual(env["SPARK_WEIGHTD_EXPERT_POOL_BYTES"], "42949672960")
        self.assertEqual(ab_fleet.parse_env_file_text(ab_fleet.render_env(s, "F1AA"))["AGENT_MEMORY_NEED_GIB"], "12")
        self.assertEqual(ab_fleet.parse_env_file_text(ab_fleet.render_env(s, "F0"))["AGENT_MEMORY_NEED_GIB"], "12")

    def test_arm_env_keys_are_checked(self):
        s = spec()
        s["arms"]["F2"]["env"] = {"AGENT_ROLE": "production"}
        with self.assertRaises(ab_fleet.FleetError):
            ab_fleet.render_env(s, "F2")


class Placement(unittest.TestCase):
    def slot(self, report, name):
        return next(slot for slot in report["slots"] if slot["name"] == name)

    def test_shared_arenas_are_counted_once(self):
        s = ab_fleet.load_spec(SPEC_PATH)
        report = ab_fleet.place(s, memory(90.0), "live", 20)
        self.assertEqual(self.slot(report, "A")["sum_gib"], 50)
        self.assertEqual(self.slot(report, "A")["sum_with_optional_gib"], 62)
        self.assertEqual(self.slot(report, "B")["sum_gib"], 57)
        self.assertEqual(self.slot(report, "C")["sum_gib"], 12)
        self.assertEqual(self.slot(report, "C")["sum_with_optional_gib"], 45)

    def test_evicted_production_arena_is_paid_by_the_bridge(self):
        s = ab_fleet.load_spec(SPEC_PATH)
        report = ab_fleet.place(s, memory(90.0), "live", 20, evicted=["production"])
        self.assertEqual(self.slot(report, "C")["sum_gib"], 33)

    def test_boundary_is_the_agent_rule(self):
        s = ab_fleet.load_spec(SPEC_PATH)
        self.assertTrue(self.slot(ab_fleet.place(s, memory(82.0), "live", 20), "A")["optional_fits"])
        self.assertFalse(self.slot(ab_fleet.place(s, memory(81.9), "live", 20), "A")["optional_fits"])
        self.assertTrue(self.slot(ab_fleet.place(s, memory(70.0), "live", 20), "A")["fits"])
        self.assertFalse(self.slot(ab_fleet.place(s, memory(69.9), "live", 20), "A")["fits"])
        low = memory(90.0)
        low["nodes"]["sparkd"]["mem_available_gib"] = 78.0
        report = ab_fleet.place(s, low, "live", 20)
        self.assertTrue(self.slot(report, "A")["fits"])
        self.assertFalse(self.slot(report, "A")["optional_fits"])
        self.assertEqual(self.slot(report, "A")["worst_node"], "sparkd")
        self.assertTrue(self.slot(report, "B")["fits"])

    def test_a_without_the_aa_engine_places_f1_alone(self):
        s = ab_fleet.load_spec(SPEC_PATH)
        a = self.slot(ab_fleet.place(s, memory(75.0), "live", 20), "A")
        self.assertTrue(a["fits"])
        self.assertFalse(a["optional_fits"])
        self.assertEqual(a["nodes"]["spark0"]["margin_gib"], 5)
        a = self.slot(ab_fleet.place(s, memory(100.0, lanes=(0, 2)), "live", 20), "A")
        self.assertTrue(a["fits"])
        self.assertFalse(a["optional_fits"])
        a = self.slot(ab_fleet.place(s, memory(100.0, lanes=(0, 1)), "live", 20), "A")
        self.assertFalse(a["fits"])

    def test_optional_arms_are_fleet_wide(self):
        s = ab_fleet.load_spec(SPEC_PATH)
        mem = memory(70.0)
        mem["nodes"]["spark5"]["mem_available_gib"] = 60.0
        c = self.slot(ab_fleet.place(s, mem, "live", 20), "C")
        self.assertTrue(c["fits"])
        self.assertFalse(c["optional_fits"])

    def test_window_basis_and_missing_nodes(self):
        s = ab_fleet.load_spec(SPEC_PATH)
        report = ab_fleet.place(s, memory(50.0, window=90.0), "window", 20)
        self.assertTrue(all(slot["fits"] for slot in report["slots"]))
        report = ab_fleet.place(s, memory(50.0, window=90.0), "live", 20)
        self.assertFalse(self.slot(report, "A")["fits"])
        broken = memory(90.0)
        del broken["nodes"]["sparkf"]
        with self.assertRaises(ab_fleet.FleetError):
            ab_fleet.place(s, broken, "live", 20)
        broken = memory(90.0)
        broken["nodes"]["sparkf"] = {"error": "timeout"}
        with self.assertRaises(ab_fleet.FleetError):
            ab_fleet.place(s, broken, "live", 20)

    def test_live_lane_conflict_refuses_the_slot(self):
        s = ab_fleet.load_spec(SPEC_PATH)
        report = ab_fleet.place(s, memory(100.0, lanes=(0, 15)), "live", 20)
        self.assertFalse(self.slot(report, "B")["fits"])
        self.assertTrue(self.slot(report, "A")["fits"])

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Path(tmp) / "mem.json"
            mem.write_text(json.dumps(memory(100.0)))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ab_fleet.main(["place", str(SPEC_PATH), str(mem)]), 0)
            mem.write_text(json.dumps(memory(60.0)))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(ab_fleet.main(["place", str(SPEC_PATH), str(mem)]), 3)
            self.assertIn("slot A: DOES NOT FIT", out.getvalue())


class ArmRoot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ab-fleet-")
        self.home = Path(self.tmp.name).resolve()
        self.spec = ab_fleet.load_spec(SPEC_PATH)
        self.production = self.home / "sparkdata" / "glm53flash.fp8.tp16"
        (self.production / "kv_snapshot").mkdir(parents=True)
        (self.home / "kvcache" / "glm53flash.fp8.tp16").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def make_root(self, arm, snapshot="runs/r1/kv_snapshot", backing=None):
        name = self.spec["arms"][arm]["root"]
        root = self.home / "sparkdata" / name
        (root / "config").mkdir(parents=True)
        (root / "agent.env").write_text(ab_fleet.render_env(self.spec, arm))
        nodes = []
        for index, host in enumerate(self.spec["nodes"]):
            runtime_root = str(root) if index == 0 else f"/home/{host}/sparkdata/{name}"
            nodes.append({"rank_index": index, "runtime_root": runtime_root,
                          "kv_backing_directory": backing if (backing and index == 0) else runtime_root + "/kvcache"})
        (root / "model_resident.json").write_text(json.dumps({"nodes": nodes}))
        stage = {"stage_pack_path": "packs/rank0.sp"}
        if snapshot is not None:
            stage["kv_snapshot_directory"] = snapshot
        stage["score_dump_directory"] = "runs/r1/dump"
        (root / "config" / "stage_00.json").write_text(json.dumps(stage))
        return root

    def check(self, arm, root, fresh=False):
        return ab_fleet.root_check(self.spec, arm, str(root), str(self.home), fresh)

    def test_clean_root_passes(self):
        root = self.make_root("F2")
        errors, private = self.check("F2", root)
        self.assertEqual(errors, [])
        self.assertTrue(private["kv_snapshot_directory"].endswith("/qab-f2/runs/r1/kv_snapshot"))

    def test_snapshot_directory_in_production_is_refused(self):
        root = self.make_root("F0", snapshot="runs/snap")
        (root / "runs").mkdir()
        (root / "runs" / "snap").symlink_to(self.production / "kv_snapshot")
        errors, _ = self.check("F0", root)
        self.assertTrue(any("inside protected" in e and "kv_snapshot_directory" in e for e in errors), errors)
        self.assertTrue(any("outside the arm root" in e for e in errors), errors)

    def test_escaping_or_absolute_snapshot_directory_is_refused(self):
        root = self.make_root("F2", snapshot="../glm53flash.fp8.tp16/kv_snapshot")
        errors, _ = self.check("F2", root)
        self.assertTrue(any("not a normalized relative path" in e for e in errors), errors)
        shutil.rmtree(root)
        root = self.make_root("F2", snapshot=str(self.production / "kv_snapshot"))
        errors, _ = self.check("F2", root)
        self.assertTrue(any("must be relative" in e for e in errors), errors)

    def test_production_kv_backing_is_refused(self):
        root = self.make_root("F0", backing=str(self.home / "kvcache" / "glm53flash.fp8.tp16"))
        errors, _ = self.check("F0", root)
        self.assertTrue(any("kv_backing_directory" in e and "outside the arm root" in e for e in errors), errors)

    def test_root_aliasing_production_is_refused(self):
        name = self.spec["arms"]["F0"]["root"]
        (self.home / "sparkdata" / name).symlink_to(self.production)
        errors, _ = self.check("F0", self.home / "sparkdata" / name)
        self.assertTrue(any("overlaps protected" in e for e in errors), errors)

    def test_two_arms_sharing_a_snapshot_directory_are_refused(self):
        f2 = self.make_root("F2", snapshot="runs/snap")
        f3 = self.make_root("F3", snapshot="runs/snap")
        (f2 / "runs").mkdir()
        (f2 / "runs" / "snap").mkdir()
        (f3 / "runs").mkdir()
        (f3 / "runs" / "snap").symlink_to(f2 / "runs" / "snap")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = ab_fleet.main(["root-check", str(SPEC_PATH), f"F2={f2}", f"F3={f3}", "--home", str(self.home)])
        self.assertEqual(rc, 1)
        self.assertIn("REFUSED arm F3 kv_snapshot_directory", out.getvalue())
        e2, d2 = self.check("F2", f2)
        e3, d3 = self.check("F3", f3)
        overlaps = ab_fleet.roots_disjoint({"F2": d2, "F3": d3})
        self.assertTrue(any("overlaps arm F2" in e for e in overlaps + e3), overlaps + e3)

    def test_prefix_sibling_roots_are_not_confused(self):
        f1 = self.make_root("F1")
        f1aa = self.make_root("F1AA")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = ab_fleet.main(["root-check", str(SPEC_PATH), f"F1={f1}", f"F1AA={f1aa}", "--home", str(self.home)])
        self.assertEqual(rc, 0, out.getvalue())
        (f1aa / "elsewhere").mkdir()
        (f1 / "runs" / "r1").mkdir(parents=True)
        (f1 / "runs" / "r1" / "kv_snapshot").symlink_to(f1aa / "elsewhere")
        errors, _ = self.check("F1", f1)
        self.assertTrue(any("outside the arm root" in e for e in errors), errors)

    def test_remote_rank_kv_backing_outside_the_root_is_refused(self):
        root = self.make_root("F2")
        doc = json.loads((root / "model_resident.json").read_text())
        doc["nodes"][5]["kv_backing_directory"] = "/home/spark5/kvcache/glm53flash.fp8.tp16"
        (root / "model_resident.json").write_text(json.dumps(doc))
        errors, _ = self.check("F2", root)
        self.assertTrue(any("rank 5" in e and "kv_backing_directory" in e for e in errors), errors)

    def test_config_variants_are_checked(self):
        root = self.make_root("F2")
        (root / "config.probe").mkdir()
        (root / "config.probe" / "stage_00.json").write_text(json.dumps({"score_dump_directory": "../qab-f3/run/dump"}))
        errors, _ = self.check("F2", root)
        self.assertTrue(any("config.probe/stage_00.json" in e for e in errors), errors)

    def test_fresh_requires_empty_private_directories(self):
        root = self.make_root("F2")
        snap = root / "runs" / "r1" / "kv_snapshot"
        snap.mkdir(parents=True)
        self.assertEqual(self.check("F2", root, fresh=True)[0], [])
        (snap / "stale.kvs").write_text("x")
        errors, _ = self.check("F2", root, fresh=True)
        self.assertTrue(any("not empty" in e for e in errors), errors)

    def test_agent_env_drift_is_refused(self):
        root = self.make_root("F3")
        text = (root / "agent.env").read_text().replace("SPARK_WEIGHTD_LANE=13", "SPARK_WEIGHTD_LANE=0")
        (root / "agent.env").write_text(text)
        errors, _ = self.check("F3", root)
        self.assertTrue(any("SPARK_WEIGHTD_LANE" in e for e in errors), errors)


@unittest.skipUnless(bash4(), "the agent parser cross-check needs bash >= 4")
class AgentParser(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ab-fleet-agent-")
        self.home = Path(self.tmp.name).resolve()
        (self.home / "agent_head.sh").write_text(HEAD)
        self.mem = self.home / "mem"
        self.spec = ab_fleet.load_spec(SPEC_PATH)
        for arm in ("F1", "F1AA"):
            root = self.home / "sparkdata" / self.spec["arms"][arm]["root"]
            (root / "config").mkdir(parents=True)
            (root / "config" / "stage_00.json").write_text("{}\n")
            (root / "agent.env").write_text(ab_fleet.render_env(self.spec, arm))

    def tearDown(self):
        self.tmp.cleanup()

    def run_agent(self, body, avail):
        self.mem.write_text(f"{avail}\n")
        env = dict(os.environ, HOME=str(self.home), TEST_AGENT_HEAD=str(self.home / "agent_head.sh"), TEST_MEM=str(self.mem))
        env.pop("FLEET_AGENT_HEADROOM_GIB", None)
        env.pop("FLEET_AGENT_ROOTS_FILE", None)
        return subprocess.run(["bash", "-c", PRELUDE + body], env=env, capture_output=True, text=True,
                              cwd=str(self.home), timeout=60)

    def test_rendered_env_parses_as_a_dev_root(self):
        out = self.run_agent('root_config qab-f1 && echo "$RC_ROLE|$RC_SYNC|$RC_NEED_GIB|$RC_MEMORY_MAX|${RC_ENV[*]}"', 100)
        self.assertEqual(out.returncode, 0, out.stderr)
        role, sync, need, memory_max, env = out.stdout.strip().split("|")
        self.assertEqual((role, sync, need, memory_max), ("dev", "local", "50", "20G"))
        self.assertIn("SPARK_WEIGHTD_LANE=1", env.split())
        self.assertIn("SPARK_WEIGHTD_EXPERT_POOL_BYTES=42949672960", env.split())

    def test_agent_gate_agrees_with_placement(self):
        for avail, expected in ((82, True), (81, False)):
            body = ('check_root_gate qab-f1 || exit 11\n'
                    'echo $(( $(cat "$TEST_MEM") - RC_NEED_GIB )) > "$TEST_MEM"\n'
                    'check_root_gate qab-f1aa || exit 12\n')
            out = self.run_agent(body, avail)
            self.assertEqual(out.returncode == 0, expected, (avail, out.returncode, out.stderr))
            report = ab_fleet.place(self.spec, memory(float(avail)), "live")
            self.assertEqual(report["slots"][0]["optional_fits"], expected)


if __name__ == "__main__":
    unittest.main()

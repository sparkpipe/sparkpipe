#!/usr/bin/env python3
"""Behavioral gates for durable queue ownership and remote reconciliation."""
import argparse
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="sparkqueue-test-")
        self.addCleanup(self.tmp.cleanup)
        spec = importlib.util.spec_from_file_location("queue_test", ROOT / "tools/spark_queue.py")
        self.q = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"SPARK_QUEUE_STATE": self.tmp.name}):
            spec.loader.exec_module(self.q)
        self.units, self.calls, self.down = {}, [], set()
        self.actual_remote = self.q.remote
        self.q.remote = self.remote
        self.actual_observe = self.q.observe
        self.q.observe = self.observe
        self.census, self.persistent_units = {}, {}

    def assert_synced_source(self, repository, source_ref, expected, contents):
        run = self.q.subprocess.run
        copied = []
        def transfer(command, **kwargs):
            if command[0] != "rsync":
                return run(command, **kwargs)
            checkout = Path(command[-2])
            self.assertEqual((checkout / "marker").read_text(), contents)
            self.assertEqual(self.q.subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(), expected)
            self.assertEqual(self.q.subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip(), "HEAD")
            copied.append(str(checkout))
            return self.q.subprocess.CompletedProcess(command, 0)
        output = io.StringIO()
        with patch.object(self.q, "__file__", str(repository / "tools/spark_queue.py")), patch.object(self.q.subprocess, "run", side_effect=transfer), patch.object(self.q, "ssh", return_value=(0, "")) as remote, contextlib.redirect_stdout(output):
            self.q.cmd_sync(argparse.Namespace(id="pr-test", nodes="spark0", ref=source_ref))
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt["git_commit"], expected)
        self.assertEqual(receipt["source_ref"], source_ref)
        self.assertEqual(len(copied), 1)
        self.assertIn(expected, remote.call_args.args[1])
        self.assertIn("diff --quiet HEAD", remote.call_args.args[1])

    def test_sync_tests_unmerged_commits_with_exact_detached_source(self):
        repository = Path(self.tmp.name) / "repository"
        repository.mkdir()
        def git(*args):
            return self.q.subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()
        git("init", "-q", "-b", "main")
        git("config", "user.name", "Fixture")
        git("config", "user.email", "fixture@example.invalid")
        marker = repository / "marker"
        marker.write_text("base")
        git("add", "marker")
        git("commit", "-qm", "base")
        base = git("rev-parse", "HEAD")
        git("checkout", "-qb", "unmerged-pr")
        marker.write_text("pr")
        git("commit", "-qam", "PR")
        head = git("rev-parse", "HEAD")
        marker.write_text("uncommitted")
        for source_ref, expected, contents in (("HEAD", head, "pr"), (base, base, "base")):
            self.assert_synced_source(repository, source_ref, expected, contents)
        self.assertEqual(marker.read_text(), "uncommitted")

    def test_sync_preserves_remote_only_commit_from_shallow_local_source(self):
        upstream = Path(self.tmp.name) / "upstream"
        repository = Path(self.tmp.name) / "shallow-controller"
        upstream.mkdir()
        def git(path, *args):
            return self.q.subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
        git(upstream, "init", "-q", "-b", "main")
        git(upstream, "config", "user.name", "Fixture")
        git(upstream, "config", "user.email", "fixture@example.invalid")
        marker = upstream / "marker"
        marker.write_text("base")
        git(upstream, "add", "marker")
        git(upstream, "commit", "-qm", "base")
        base = git(upstream, "rev-parse", "HEAD")
        self.q.subprocess.run(["git", "clone", "--quiet", "--depth=1", "--single-branch",
                               upstream.as_uri(), str(repository)], check=True)
        git(upstream, "checkout", "-qb", "unmerged-pr")
        marker.write_text("fetched-pr")
        git(upstream, "commit", "-qam", "PR")
        head = git(upstream, "rev-parse", "HEAD")
        git(repository, "fetch", "--quiet", "--depth=1", "origin", "unmerged-pr")
        self.assertEqual(git(repository, "rev-parse", "--is-shallow-repository"), "true")
        self.assertEqual(git(repository, "rev-parse", "HEAD"), base)
        self.assertEqual(git(repository, "for-each-ref", "--format=%(objectname)", "refs/heads"), base)
        self.assertEqual(git(repository, "rev-parse", "FETCH_HEAD"), head)
        (repository / "marker").write_text("controller-dirty")
        self.assert_synced_source(repository, "FETCH_HEAD", head, "fetched-pr")
        git(repository, "update-ref", "refs/remotes/origin/unmerged-pr", head)
        self.assert_synced_source(repository, "refs/remotes/origin/unmerged-pr", head, "fetched-pr")
        self.assert_synced_source(repository, head, head, "fetched-pr")
        self.assertEqual(git(repository, "rev-parse", "HEAD"), base)
        self.assertEqual((repository / "marker").read_text(), "controller-dirty")

    def test_rdma_registration_uses_declared_finite_memory_budget(self):
        job = {"id": "rdma", "attempt": "test", "nodes": ["spark0"],
               "deadline": time.time() + 60, "cmd": "true", "memory_mib": 1536}
        with patch.object(self.q, "ssh", return_value=(0, "LoadState=loaded")) as ssh:
            self.actual_remote(job, "spark0", "launch")
        command = ssh.call_args.args[1]
        self.assertIn("--property=MemoryMax=1536M", command)
        self.assertIn("--property=LimitMEMLOCK=1536M", command)
        self.assertIn("--property=MemorySwapMax=0", command)
        self.assertNotIn("infinity", command)

    def remote(self, job, node, action):
        self.calls.append((job["id"], node, action))
        if node in self.down:
            return {"unknown": True}
        key = (job["attempt"], node)
        if action == "launch":
            self.units.setdefault(key, {"LoadState": "loaded", "ActiveState": "active", "SubState": "running", "ExecMainStatus": "0"})
        if action == "stop":
            self.units.pop(key, None)
        return self.units.get(key, {"LoadState": "not-found"})

    def observe(self, node, owners):
        report = {"total_mib": 122566, "available_mib": 114374, "gpu_names": ["NVIDIA GB10"],
                  "units": {}, "gpu_processes": [], "ports": []}
        for owner in owners:
            key = self.q.owner_key(owner)
            if key in self.persistent_units:
                report["units"][key] = dict(self.persistent_units[key])
            elif (owner.get("attempt"), node) in self.units:
                report["units"][key] = self.unit("sparkqueue-" + owner["attempt"], owner["memory_mib"] - owner.get("device_memory_mib", 0))
        report.update(self.census.get(node, {}))
        return report

    def unit(self, name="engine", memory=32768):
        return {"LoadState": "loaded", "ActiveState": "active", "InvocationID": name + "-epoch",
                "ControlGroup": "/system.slice/" + name + ".service", "MemoryMax": str(memory * self.q.MIB),
                "MemoryCurrent": str(4096 * self.q.MIB), "MemoryNonreclaimable": 4096 * self.q.MIB}

    def add_shared(self, name, nodes="spark0", memory=8192, *extra):
        self.add(name, nodes, "--resources", "gpu-shared", "--memory-mib", str(memory), "--device-memory-mib", str(memory // 2), *extra)

    def track(self, name="engine", memory=32768):
        self.persistent_units["system:" + name + ".service"] = self.unit(name, memory - 8192)
        self.cli("track", "--node", "spark0", "--unit", name + ".service", "--device-memory-mib", "8192")
        return self.state()["persistent"][-1]

    def cli(self, *args):
        with patch.object(sys, "argv", ["spark_queue.py", *args]):
            self.q.main()

    def add(self, name, nodes="spark0", *extra):
        self.cli("add", "--id", name, "--nodes", nodes, "--per-node",
                 "--cmd", "echo test", *extra)

    def state(self):
        with self.q.transaction() as state:
            return json.loads(json.dumps(state))

    def jobs(self):
        return {j["id"]: j for j in self.state()["jobs"]}

    def dispatch(self):
        self.q.cmd_dispatch(argparse.Namespace(ttl=3))

    def test_shared_jobs_run_concurrently_with_distinct_attempt_roots(self):
        self.add_shared("a", "spark0", 8192, "--ports", "24000:24999")
        self.add_shared("b", "spark0", 8192, "--ports", "25000:25999")
        self.dispatch()
        self.assertEqual({j["state"] for j in self.jobs().values()}, {"running"})
        self.assertNotEqual(self.jobs()["a"]["attempt"], self.jobs()["b"]["attempt"])
        with patch.object(self.q, "ssh", return_value=(0, "LoadState=loaded")) as ssh:
            self.actual_remote(self.jobs()["a"], "spark0", "launch")
        command = ssh.call_args.args[1]
        self.assertIn("SPARK_QUEUE_RUNTIME_ROOT=/tmp/sparkqueue-" + self.jobs()["a"]["attempt"], command)
        self.assertIn("SPARK_QUEUE_PORTS=24000:24999", command)
        self.assertIn("SPARK_QUEUE_MEMORY_MIB=8192", command)
        self.assertIn("SPARK_QUEUE_DEVICE_MEMORY_MIB=4096", command)
        self.assertIn("MemoryMax=4096M", command)
        self.assertIn("LimitMEMLOCK=8192M", command)

    def test_shared_jobs_require_explicit_budget_and_fresh_census(self):
        with self.assertRaises(SystemExit):
            self.add("a", "spark0", "--resources", "gpu-shared")
        self.add_shared("a")
        self.assertEqual(self.q.claim(3), [])
        self.assertIn("fresh resource census", self.jobs()["a"]["admission_error"])
        self.census["spark0"] = {"error": "node unreachable"}
        self.dispatch()
        self.assertEqual(self.jobs()["a"]["state"], "queued")
        self.assertIn("node unreachable", self.jobs()["a"]["admission_error"])

    def test_shared_unknown_gpu_process_blocks_only_affected_nodes(self):
        self.census["spark0"] = {"gpu_processes": [{"pid": 17, "used_mib": 2048, "control_group": "/unbounded"}]}
        self.add_shared("a")
        self.add_shared("b", "spark1")
        self.dispatch()
        self.assertIn("unaccounted GPU process 17", self.jobs()["a"]["admission_error"])
        self.assertEqual(self.jobs()["b"]["state"], "running")
        self.assertFalse(any(call[0] == "a" for call in self.calls))

    def test_shared_device_budget_is_explicit_and_separate_from_host_limit(self):
        for device in (None, "0", "-1", "8192", "8191"):
            extra = [] if device is None else ["--device-memory-mib", device]
            with self.assertRaises(SystemExit):
                self.add("a", "spark0", "--resources", "gpu-shared", "--memory-mib", "8192", *extra)
        self.assertEqual(self.state()["jobs"], [])

    def test_observed_cuda_usage_must_fit_each_accounted_owner_device_budget(self):
        self.add_shared("a")
        self.dispatch()
        group = "/system.slice/sparkqueue-" + self.jobs()["a"]["attempt"] + ".service"
        self.add_shared("b")
        self.census["spark0"] = {"gpu_processes": [
            {"pid": 17, "used_mib": 2048, "control_group": group},
            {"pid": 18, "used_mib": 2049, "control_group": group}]}
        self.dispatch()
        self.assertEqual(self.jobs()["b"]["state"], "queued")
        self.assertIn("exceeds declared device budget", self.jobs()["b"]["admission_error"])
        self.census["spark0"]["gpu_processes"][1]["used_mib"] = None
        self.dispatch()
        self.assertIn("memory is unavailable", self.jobs()["b"]["admission_error"])
        self.census["spark0"]["gpu_processes"][1]["used_mib"] = 2048
        self.dispatch()
        self.assertEqual(self.jobs()["b"]["state"], "running")

    def test_shared_memory_checks_actual_capacity_and_available_reserves(self):
        self.track(memory=32768)
        self.add_shared("a", "spark0", 16384)
        for values in ({"total_mib": 49152}, {"available_mib": 32768}):
            self.census["spark0"] = values
            self.dispatch()
            self.assertEqual(self.jobs()["a"]["state"], "queued")
            self.assertIn("insufficient memory", self.jobs()["a"]["admission_error"])
        self.census.clear()
        self.dispatch()
        self.assertEqual(self.jobs()["a"]["state"], "running")

    def test_available_memory_does_not_double_credit_reclaimable_owner_cache(self):
        self.track(memory=32768)
        self.persistent_units["system:engine.service"]["MemoryCurrent"] = str(32768 * self.q.MIB)
        self.census["spark0"] = {"available_mib": 40960}
        self.add_shared("a", "spark0", 8192)
        self.dispatch()
        self.assertEqual(self.jobs()["a"]["state"], "queued")
        self.assertIn("insufficient memory", self.jobs()["a"]["admission_error"])

    def test_persistent_finite_owner_shares_and_keeps_memory_reserved(self):
        owner = self.track(memory=65536)
        self.census["spark0"] = {"gpu_processes": [{"pid": 17, "used_mib": 2048, "control_group": owner["control_group"] + "/child"}]}
        self.add_shared("a", "spark0", 32768)
        self.add_shared("b", "spark0", 32768)
        self.dispatch()
        self.assertEqual(self.jobs()["a"]["state"], "running")
        self.assertEqual(self.jobs()["b"]["state"], "queued")
        self.cli("release", "--node", "spark0")
        self.assertEqual(self.state()["persistent"], [owner])
        with self.assertRaises(SystemExit):
            self.cli("untrack", "--id", owner["id"])
        self.assertEqual(self.state()["persistent"], [owner])

    def test_persistent_unknown_unbounded_and_restarted_units_fail_closed(self):
        owner = self.track()
        original = dict(self.persistent_units["system:engine.service"])
        self.add_shared("a")
        for changes, reason in [({"MemoryMax": "infinity"}, "finite memory bound"),
                                ({"InvocationID": "new-epoch"}, "identity changed"),
                                ({"MemoryMax": str(65536 * self.q.MIB)}, "budget/control group mismatch"),
                                ({"ActiveState": "inactive"}, "not verifiably active")]:
            self.persistent_units["system:engine.service"] = original | changes
            self.dispatch()
            self.assertEqual(self.jobs()["a"]["state"], "queued")
            self.assertIn(reason, self.jobs()["a"]["admission_error"])
            self.assertEqual(self.state()["persistent"], [owner])
        self.persistent_units["system:engine.service"] = original | {"MemoryMax": "infinity"}
        with self.assertRaises(SystemExit):
            self.cli("track", "--node", "spark0", "--device-memory-mib", "8192", "--unit", "engine.service")
        self.assertEqual(self.state()["persistent"], [owner])

    def test_persistent_identity_is_rechecked_before_remote_launch(self):
        self.track()
        self.add_shared("a")
        observations = self.q.resource_observations(self.state())
        job = self.q.claim(3, observations)[0]
        self.persistent_units["system:engine.service"]["InvocationID"] = "restarted-between-claim-and-launch"
        with patch.object(self.q, "ssh") as ssh:
            reply = self.actual_remote(job, "spark0", "launch")
        self.assertEqual(reply["LoadState"], "not-found")
        self.assertIn("changed before launch", reply["admission_error"])
        ssh.assert_not_called()

    def test_untrack_requires_stop_and_no_remaining_gpu_children(self):
        owner = self.track()
        self.persistent_units["system:engine.service"].update(LoadState="not-found", ActiveState="inactive")
        self.census["spark0"] = {"gpu_processes": [{"pid": 17, "used_mib": 2048, "control_group": owner["control_group"]}]}
        with self.assertRaises(SystemExit):
            self.cli("untrack", "--id", owner["id"])
        self.census.clear()
        self.cli("untrack", "--id", owner["id"])
        self.assertEqual(self.state()["persistent"], [])

    def test_queue_unit_cannot_be_double_counted_as_persistent(self):
        self.add_shared("a")
        self.dispatch()
        with self.assertRaises(SystemExit):
            self.cli("track", "--node", "spark0", "--device-memory-mib", "4096", "--unit", "sparkqueue-" + self.jobs()["a"]["attempt"] + ".service")
        self.assertEqual(self.state()["persistent"], [])

    def test_shared_accounts_for_live_queue_units_and_manual_unknown_budget(self):
        self.add_shared("a")
        self.dispatch()
        self.census["spark0"] = {"gpu_processes": [{"pid": 17, "used_mib": 2048, "control_group": "/system.slice/sparkqueue-" + self.jobs()["a"]["attempt"] + ".service"}]}
        self.add_shared("b")
        self.dispatch()
        self.assertEqual(self.jobs()["b"]["state"], "running")
        self.cli("reserve", "--node", "spark0", "--holder", "compiler", "--resources", "cpu")
        self.add_shared("c")
        self.dispatch()
        self.assertIn("lacks finite memory budget", self.jobs()["c"]["admission_error"])

    def test_shared_exclusive_waiter_and_port_collision_keep_claims(self):
        self.add_shared("a", "spark0", 8192, "--ports", "24000:24999")
        self.dispatch()
        self.add_shared("same-port", "spark0", 8192, "--ports", "24999:25999")
        self.add("exclusive", "spark0", "--resources", "exclusive", "--priority", "0")
        self.add_shared("late")
        self.dispatch()
        self.assertEqual(self.jobs()["same-port"]["state"], "queued")
        self.assertEqual(self.jobs()["late"]["state"], "queued")
        self.assertEqual(self.jobs()["exclusive"]["state"], "queued")

    def test_shared_listener_collision_and_unsupported_gpu_are_errors(self):
        self.add_shared("a", "spark0", 8192, "--ports", "24000:24999")
        self.census["spark0"] = {"ports": [24500]}
        self.dispatch()
        self.assertIn("already listening", self.jobs()["a"]["admission_error"])
        self.census["spark0"] = {"gpu_names": ["Unknown GPU"]}
        self.dispatch()
        self.assertIn("verified GB10", self.jobs()["a"]["admission_error"])

    def test_shared_stopping_owner_blocks_until_cleanup(self):
        self.add_shared("a")
        self.dispatch()
        self.cli("cancel", "--id", "a")
        self.down.add("spark0")
        self.add_shared("b")
        self.dispatch()
        self.assertEqual(self.jobs()["b"]["state"], "queued")
        self.down.clear()
        self.dispatch()
        self.assertEqual(self.jobs()["b"]["state"], "running")

    def test_preflight_is_readonly_and_reports_current_admission_reason(self):
        self.add("pending", "spark1")
        state_path = Path(self.tmp.name) / "state-v2.json"
        before = state_path.read_bytes()
        output = io.StringIO()
        self.census["spark0"] = {"gpu_processes": [{"pid": 17, "used_mib": 2048, "control_group": "/unknown"}]}
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as failure:
            self.cli("preflight", "--nodes", "spark0", "--memory-mib", "8192", "--device-memory-mib", "4096", "--ports", "24000:24999")
        self.assertEqual(failure.exception.code, 1)
        report = json.loads(output.getvalue())
        self.assertFalse(report["admissible"])
        self.assertIn("unaccounted GPU process 17", report["errors"]["spark0"])
        self.assertEqual(report["nodes"]["spark0"]["total_mib"], 122566)
        self.assertEqual(state_path.read_bytes(), before)
        self.assertFalse(self.calls)

    def test_persistent_port_reservations_precede_listener_creation(self):
        self.persistent_units["system:engine.service"] = self.unit()
        self.cli("track", "--node", "spark0", "--device-memory-mib", "8192", "--unit", "engine.service", "--ports", "24000:24999")
        owner = self.state()["persistent"][0]
        self.add_shared("a", "spark0", 8192, "--ports", "24500:25500")
        self.dispatch()
        self.assertEqual(self.jobs()["a"]["state"], "queued")
        self.persistent_units["system:engine.service"]["InvocationID"] = "restarted"
        with self.assertRaises(SystemExit):
            self.cli("track", "--node", "spark0", "--device-memory-mib", "8192", "--unit", "engine.service", "--ports", "24000:24999")
        self.assertEqual(self.state()["persistent"], [owner])

    def test_probe_does_not_hold_queue_transaction_during_ssh(self):
        self.add_shared("a")
        entered, release = threading.Event(), threading.Event()
        observe = self.q.observe
        def slow(*args):
            entered.set()
            self.assertTrue(release.wait(3))
            return observe(*args)
        self.q.observe = slow
        thread = threading.Thread(target=self.dispatch)
        thread.start()
        self.assertTrue(entered.wait(2))
        try:
            self.add("b", "spark1")
            self.assertIn("b", self.jobs())
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())

    def test_shared_invalid_ports_rejected(self):
        for ports in (["0:4"], ["65535:65536"], ["9:8"], ["20000"], ["24000:24999", "24500:25500"]):
            with self.assertRaises(SystemExit):
                self.add_shared("a", "spark0", 8192, *[value for port in ports for value in ("--ports", port)])
        self.assertEqual(self.state()["jobs"], [])

    def test_disjoint_jobs_claimed_in_one_pass(self):
        self.add("a")
        self.add("b", "spark1")
        self.dispatch()
        self.assertEqual([j["state"] for j in self.jobs().values()], ["running", "running"])

    def test_cpu_jobs_share_memory_budget_while_gpu_remains_exclusive(self):
        self.add("gpu")
        self.add("cpu", "spark0", "--resources", "cpu")
        self.add("cpu2", "spark0", "--resources", "cpu")
        self.add("gpu2")
        self.dispatch()
        self.assertEqual({k for k,v in self.jobs().items() if v["state"] == "running"}, {"cpu","cpu2","gpu"})

    def test_cpu_jobs_cannot_overcommit_declared_memory(self):
        self.add("cpu", "spark0", "--resources", "cpu", "--memory-mib", "80000")
        self.add("cpu2", "spark0", "--resources", "cpu", "--memory-mib", "80000")
        self.dispatch()
        self.assertEqual(self.jobs()["cpu"]["state"], "running")
        self.assertEqual(self.jobs()["cpu2"]["state"], "queued")

    def test_exclusive_waiter_drains_both_resource_classes(self):
        self.add("cpu", "spark0", "--resources", "cpu")
        self.dispatch()
        self.add("perf", "spark0", "--resources", "exclusive", "--priority", "0")
        self.add("gpu")
        self.dispatch()
        self.assertEqual(self.jobs()["gpu"]["state"], "queued")

    def test_declared_memory_budgets_are_added_across_resource_classes(self):
        self.add("gpu", "spark0", "--memory-mib", "80000")
        self.add("cpu", "spark0", "--resources", "cpu", "--memory-mib", "80000")
        self.dispatch()
        self.assertEqual(self.jobs()["gpu"]["state"], "running")
        self.assertEqual(self.jobs()["cpu"]["state"], "queued")

    def test_fenced_fleet_does_not_block_healthy_node(self):
        with self.q.transaction() as state:
            state["fences"]["sparkf"] = {"reason": "unreachable"}
        self.add("fleet", "spark0,sparkf", "--priority", "0")
        self.add("debug")
        self.dispatch()
        self.assertEqual(self.jobs()["debug"]["state"], "running")
        self.assertEqual(self.jobs()["fleet"]["state"], "queued")

    def test_cancel_retains_claim_until_stop_ack(self):
        self.add("a")
        self.dispatch()
        self.cli("cancel", "--id", "a")
        self.down.add("spark0")
        self.add("b")
        self.dispatch()
        self.assertEqual(self.jobs()["a"]["state"], "stopping")
        self.assertEqual(self.jobs()["b"]["state"], "queued")
        self.down.clear()
        self.dispatch()
        self.assertNotIn("a", self.jobs())
        self.assertEqual(self.jobs()["b"]["state"], "running")

    def test_partial_cleanup_releases_only_confirmed_peers(self):
        self.add("fleet", "spark0,sparkf")
        self.dispatch()
        self.cli("cancel", "--id", "fleet")
        self.down.add("sparkf")
        self.add("nextfleet", "spark0,sparkf", "--priority", "0")
        self.add("debug")
        self.dispatch()
        self.assertEqual(self.jobs()["fleet"]["released_nodes"], ["spark0"])
        self.assertEqual(self.jobs()["debug"]["state"], "running")
        self.assertEqual(self.jobs()["nextfleet"]["state"], "queued")

    def test_launch_ack_loss_is_reconciled_same_attempt(self):
        self.add("a")
        claimed = self.q.claim(3)
        self.remote(claimed[0], "spark0", "launch")  # controller dies before ACK
        self.dispatch()
        self.assertEqual(len(self.units), 1)
        self.assertEqual(self.jobs()["a"]["attempt"], claimed[0]["attempt"])

    def test_failed_dependency_does_not_run_child(self):
        self.add("a")
        self.cli("done", "--id", "a", "--exit", "1")
        self.add("b", "spark1", "--after", "a")
        self.add("c", "spark2", "--after", "b")
        self.add("independent", "spark1")
        self.dispatch()
        results = {j["id"]: j for j in self.state()["results"]}
        self.assertEqual(results["b"]["exit"], 125)
        self.assertEqual(results["b"]["failed_dependencies"], ["a"])
        self.assertEqual(results["c"]["exit"], 125)
        self.assertEqual(self.jobs()["independent"]["state"], "running")
        self.assertFalse(any(call[0] in {"b", "c"} for call in self.calls))

    def test_gate_and_note_children_receive_dependency_receipts(self):
        self.add("a")
        self.cli("done", "--id", "a", "--exit", "1")
        self.add("gate", "spark1", "--kind", "gate", "--after", "a")
        self.add("note", "spark1", "--kind", "note", "--after", "a")
        self.add("child", "spark1", "--after", "gate")
        self.dispatch()
        results = {j["id"]: j for j in self.state()["results"]}
        self.assertEqual(results["gate"]["exit"], 125)
        self.assertEqual(results["gate"]["failed_dependencies"], ["a"])
        self.assertNotIn("attempt", results["gate"])
        self.assertEqual(results["note"]["exit"], 125)
        self.assertEqual(results["child"]["exit"], 125)
        self.assertEqual(results["child"]["failed_dependencies"], ["gate"])
        self.assertEqual(self.state()["jobs"], [])
        self.assertFalse(self.calls)

    def test_readonly_commands_write_nothing(self):
        self.add("a")
        self.dispatch()
        path = Path(self.tmp.name) / "state-v2.json"
        before = path.read_bytes()
        self.cli("list")
        self.cli("list", "--all")
        self.cli("status", "--id", "a")
        self.cli("doctor")
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((Path(self.tmp.name) / "state-v2.tmp").exists())

    def test_reads_on_fresh_state_persist_nothing(self):
        self.cli("doctor")
        self.cli("list")
        names = {p.name for p in Path(self.tmp.name).iterdir()}
        self.assertNotIn("state-v2.json", names)
        self.assertNotIn("state-v2.tmp", names)

    def test_invalid_dependencies_rejected_before_submission(self):
        self.add("parent")
        for deps in ["missing", "child", "parent,parent", "parent,"]:
            with self.assertRaises(SystemExit):
                self.add("child", "spark1", "--after", deps)
            self.assertNotIn("child", self.jobs())

    def test_failed_parent_cleanup_precedes_descendant_completion(self):
        self.add("a")
        self.add("b", "spark1", "--after", "a")
        self.dispatch()
        self.cli("cancel", "--id", "a")
        self.down.add("spark0")
        self.dispatch()
        self.assertEqual(self.jobs()["a"]["state"], "stopping")
        self.assertEqual(self.jobs()["b"]["state"], "queued")
        self.down.clear()
        self.dispatch()
        self.assertNotIn("b", self.jobs())
        self.assertEqual(self.state()["results"][-1]["exit"], 125)
        self.assertFalse(any(call[0] == "b" for call in self.calls))

    def test_dependency_live_gate_preserves_unrelated_jobs(self):
        spec = importlib.util.spec_from_file_location("queue_gate_test", ROOT / "tools/spark_queue_live_gate.py")
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        gate.queue = self.q
        self.add("unrelated", "spark2")
        original = self.q.remote
        def complete(j, n, action):
            reply = original(j, n, action)
            if j["id"] != "unrelated" and action == "launch":
                reply.update(SubState="exited", ExecMainStatus="7" if j["cmd"] == "exit 7" else "0")
            return reply
        self.q.remote = complete
        output = Path(self.tmp.name) / "receipt.json"
        with patch.object(gate.time, "sleep"):
            gate.dependency_gate("spark1", output)
        receipt = json.loads(output.read_text())
        self.assertEqual(len(receipt["results"]), 3)
        self.assertEqual(receipt["remaining"], [])
        self.assertEqual(self.jobs()["unrelated"]["state"], "running")

    def test_success_requires_cleanup_and_then_unlocks_dependency(self):
        self.add("a")
        self.add("b", "spark1", "--after", "a")
        self.dispatch()
        for reply in self.units.values():
            reply["SubState"] = "exited"
        self.dispatch()
        self.assertEqual(self.jobs()["a"]["state"], "stopping")
        self.assertEqual(self.jobs()["b"]["state"], "queued")
        self.dispatch()
        self.assertEqual(self.jobs()["b"]["state"], "running")
        self.assertEqual(self.state()["results"][0]["exit"], 0)

    def test_submission_and_cancel_do_not_wait_for_remote_io(self):
        self.add("a")
        entered, release = threading.Event(), threading.Event()
        original = self.q.remote
        def slow(*args):
            entered.set()
            self.assertTrue(release.wait(3))
            return original(*args)
        self.q.remote = slow
        thread = threading.Thread(target=self.dispatch)
        thread.start()
        self.assertTrue(entered.wait(2))
        try:
            self.add("b", "spark1")
            self.cli("cancel", "--id", "a")
            self.assertEqual(self.jobs()["a"]["state"], "stopping")
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.jobs()["a"]["state"], "stopping")
        self.dispatch()
        self.assertNotIn("a", self.jobs())

    def test_second_dispatcher_does_not_duplicate_claim(self):
        self.add("a")
        jobs = self.q.claim(3)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(self.q.claim(3), [])

    def test_expired_job_stops_before_releasing(self):
        self.add("a")
        self.dispatch()
        with self.q.transaction() as state:
            state["jobs"][0]["deadline"] = time.time() - 1
        self.dispatch()
        self.assertEqual(self.state()["results"][0]["exit"], 124)
        self.assertEqual(self.units, {})

    def test_release_cannot_remove_running_ownership(self):
        self.add("a")
        self.dispatch()
        self.cli("release", "--node", "spark0")
        self.add("b")
        self.dispatch()
        self.assertEqual(self.jobs()["b"]["state"], "queued")

    def test_validation(self):
        for ttl in ["0", "-1", "nan", "inf", "16"]:
            with self.assertRaises(SystemExit):
                self.add("bad", "spark0", "--ttl-min", ttl)
        with self.assertRaises(SystemExit):
            self.cli("add", "--id", "bad", "--nodes", "spark0", "--cmd", " ")
        with self.assertRaises(SystemExit):
            self.cli("add", "--id", "bad", "--nodes", "spark0,spark1", "--cmd", "echo unsafe fanout")
        with self.assertRaises(SystemExit):
            self.add("bad;touch")
        self.assertEqual(self.state()["jobs"], [])

    def test_schedule_alias(self):
        self.add("a")
        self.cli("schedule")
        self.assertEqual(self.jobs()["a"]["ttl_minutes"], 3)

    def test_legacy_invalid_job_does_not_strand_other_work(self):
        self.add("bad")
        with self.q.transaction() as state:
            state["jobs"][0]["ttl_minutes"] = "nan"
        self.add("good")
        self.dispatch()
        self.assertEqual(self.jobs()["bad"]["state"], "invalid")
        self.assertEqual(self.jobs()["good"]["state"], "running")

    def test_legacy_running_jobs_fail_closed_and_receipts_survive(self):
        (Path(self.tmp.name) / "queue.jsonl").write_text(json.dumps({"id":"old","nodes":["spark0"],"state":"running"})+"\n")
        (Path(self.tmp.name) / "results.jsonl").write_text(json.dumps({"id":"done","exit":0})+"\n")
        state = self.state()
        self.assertEqual(state["jobs"][0]["state"], "legacy-review")
        self.assertIn("spark0", state["fences"])
        self.assertEqual(state["results"][0]["id"], "done")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Behavioral gates for durable queue ownership and remote reconciliation."""
import argparse
import importlib.util
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

    def test_disjoint_jobs_claimed_in_one_pass(self):
        self.add("a")
        self.add("b", "spark1")
        self.dispatch()
        self.assertEqual([j["state"] for j in self.jobs().values()], ["running", "running"])

    def test_cpu_and_gpu_each_retain_ownership(self):
        self.add("gpu")
        self.add("cpu", "spark0", "--resources", "cpu")
        self.add("cpu2", "spark0", "--resources", "cpu")
        self.add("gpu2")
        self.dispatch()
        self.assertEqual({k for k,v in self.jobs().items() if v["state"] == "running"}, {"cpu","gpu"})

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
        self.dispatch()
        self.assertEqual(self.jobs()["b"]["state"], "queued")

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

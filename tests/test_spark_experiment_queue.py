#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spark_experiment_queue", ROOT / "tools/spark_experiment_queue.py"
)
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class Arguments:
    def __init__(self, root, job, nodes, priority=0, blocked_on=None, resources=None):
        self.job = job
        self.biggulp = "model-dsv4-flash"
        self.source_task = "MOD-D4F-005"
        self.bite = job
        self.nodes = [",".join(nodes)]
        self.resources = resources or ["gpu"]
        self.role = "model-launcher"
        self.question = f"question for {job}"
        self.expected_value = "answers a production blocker"
        self.required_data = [
            "status code and wall seconds",
            "raw stdout and stderr paths",
            "decision unlocked by success or failure",
        ]
        self.spec = str(root / "spec.json")
        self.result = str(root / f"{job}.result.json")
        self.priority = priority
        self.blocked_on = blocked_on


class SparkExperimentQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "spec.json").write_text("{}\n")
        self.state = {"schema_version": 1, "jobs": []}
        self.pert = {"MOD-D4F-005"}

    def tearDown(self):
        self.temporary.cleanup()

    def submit(self, job, nodes, priority=0, blocked_on=None, now=10.0, resources=None):
        arguments = Arguments(self.root, job, nodes, priority, blocked_on, resources)
        return QUEUE.submit(self.state, arguments, self.pert, now)

    def test_disjoint_jobs_claim_concurrently(self):
        self.submit("d4f", ["spark4", "spark5", "spark6", "spark7"], 3)
        self.submit("q27", ["spark0", "spark1", "spark2", "spark3"], 2)
        self.submit("glm", ["spark8", "spark9", "sparka", "sparkb"], 1)
        self.assertEqual(QUEUE.claim(self.state, "a", "model-launcher", 20)["job_id"], "d4f")
        self.assertEqual(QUEUE.claim(self.state, "b", "model-launcher", 20)["job_id"], "q27")
        self.assertEqual(QUEUE.claim(self.state, "c", "model-launcher", 20)["job_id"], "glm")

    def test_overlap_blocks_only_overlapping_job(self):
        self.submit("first", ["spark0", "spark1"], 3)
        self.submit("overlap", ["spark1", "spark2"], 2)
        self.submit("free", ["spark3"], 1)
        QUEUE.claim(self.state, "a", "model-launcher", 20)
        self.assertEqual(QUEUE.claim(self.state, "b", "model-launcher", 20)["job_id"], "free")
        row = next(job for job in QUEUE.snapshot(self.state, 25)["jobs"] if job["job_id"] == "overlap")
        self.assertEqual(row["blocked_by_jobs"], ["first"])

    def test_blocked_prerequisite_does_not_reserve_nodes(self):
        self.submit("blocked", ["spark4"], 9, "fileadmin receipt")
        self.submit("ready", ["spark4"], 1)
        self.assertEqual(QUEUE.claim(self.state, "a", "model-launcher", 20)["job_id"], "ready")
        QUEUE.unblock(self.state, "blocked", 30)
        self.assertEqual(self.state["jobs"][0]["state"], "queued")

    def test_successful_queue_dependency_unblocks_child(self):
        self.submit("parent", ["spark4"])
        self.submit("child", ["spark4"], blocked_on="parent")
        self.state["jobs"][0]["state"] = "succeeded"
        QUEUE.resolve_dependencies(self.state, 30)
        self.assertEqual(self.state["jobs"][1]["state"], "queued")
        self.assertIsNone(self.state["jobs"][1]["blocked_on"])

    def test_failed_or_external_dependency_stays_blocked(self):
        self.submit("failed", ["spark4"])
        self.submit("failed-child", ["spark4"], blocked_on="failed")
        self.submit("external-child", ["spark5"], blocked_on="release-merged")
        self.state["jobs"][0]["state"] = "failed"
        QUEUE.resolve_dependencies(self.state, 30)
        self.assertEqual(self.state["jobs"][1]["state"], "blocked")
        self.assertEqual(self.state["jobs"][2]["state"], "blocked")

    def test_storage_transfer_can_overlap_gpu_on_same_nodes(self):
        self.submit("benchmark", ["spark4", "spark5"], resources=["gpu"])
        self.submit("copy", ["spark4", "spark5"], resources=["storage_io"])
        self.assertEqual(QUEUE.claim(self.state, "bench", "model-launcher", 20)["job_id"], "benchmark")
        self.state["jobs"][1]["role"] = "fileadmin"
        self.assertEqual(QUEUE.claim(self.state, "files", "fileadmin", 20)["job_id"], "copy")

    def test_service_activation_can_exclude_gpu_and_storage(self):
        self.submit("activate", ["spark4"], priority=3, resources=["gpu", "storage_io"])
        self.submit("benchmark", ["spark4"], priority=2, resources=["gpu"])
        self.submit("copy", ["spark4"], priority=1, resources=["storage_io"])
        QUEUE.claim(self.state, "activate", "model-launcher", 20)
        self.assertIsNone(QUEUE.claim(self.state, "bench", "model-launcher", 20))

    def test_completion_requires_owner_and_real_receipt(self):
        self.submit("job", ["spark4"])
        QUEUE.claim(self.state, "a", "model-launcher", 20)
        arguments = type("Done", (), {
            "job": "job", "executor": "a", "outcome": "succeeded",
            "receipt": str(self.root / "receipt.json"),
        })()
        with self.assertRaisesRegex(QUEUE.QueueError, "receipt does not exist"):
            QUEUE.complete(self.state, arguments, 30)
        (self.root / "receipt.json").write_text(json.dumps({"status": "PASS"}))
        self.assertEqual(QUEUE.complete(self.state, arguments, 30)["state"], "succeeded")

    def test_heartbeat_requires_owner_without_extending_lease(self):
        self.submit("job", ["spark4"])
        job = QUEUE.claim(self.state, "a", "model-launcher", 20)
        deadline = job["lease_deadline"]
        self.assertEqual(QUEUE.heartbeat(self.state, "a", "job", 30)["heartbeat_at"], 30)
        self.assertEqual(job["lease_deadline"], deadline)
        with self.assertRaisesRegex(QUEUE.QueueError, "does not own"):
            QUEUE.heartbeat(self.state, "b", "job", 40)

    def test_expired_job_requeues_without_blocking_others(self):
        self.submit("job", ["spark4"])
        QUEUE.claim(self.state, "a", "model-launcher", 20)
        QUEUE.expire(self.state, 20 + QUEUE.LEASE_SECONDS)
        self.assertEqual(self.state["jobs"][0]["state"], "queued")
        self.assertEqual(self.state["jobs"][0]["last_error"], "executor lease expired")


if __name__ == "__main__":
    unittest.main()

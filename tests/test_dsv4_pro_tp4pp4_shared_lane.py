#!/usr/bin/env python3
"""Contract tests for the dsv4_pro TP4xPP4 shared-lane wrapper (lane 5).

Covers the generated family deployment, the lane stage config, and the
wrapper's private-runtime prepare (dry-run: prepare must succeed and the
launch must then fail closed without the shared socket). Fail-closed
coverage: missing sidecar, missing runtime lib, wrong transport_host,
non-lane ports, unknown listener key.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "dsv4_pro_tp4pp4_gen_deployment.py"
WRAPPER = ROOT / "tools" / "devcycle" / "run-dsv4-pro-family-job.sh"
DEPLOYMENT = ROOT / "deployment" / "dsv4_pro_tp4pp4" / "model_resident.json"
STAGE = ROOT / "deployment" / "dsv4_pro_tp4pp4" / "config" / "dsv4_pro_tp4_pp4_stage.json"

LANE = 5
CONTROL = range(23080, 23096)
COLLECTIVE = range(53080, 53096)
TRANSPORT = range(64080, 64096)
RANKS = 16
ATTEMPT = "a" * 32


class DeploymentContract(unittest.TestCase):
    def test_deployment_is_current_generator_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(GENERATOR), "--output-dir", tmp],
                           check=True, capture_output=True)
            generated = json.loads((Path(tmp) / "model_resident.json").read_text())
            committed = json.loads(DEPLOYMENT.read_text())
            self.assertEqual(generated, committed)

    def test_rank_order_and_ports(self):
        document = json.loads(DEPLOYMENT.read_text())
        nodes = document["nodes"]
        self.assertEqual(len(nodes), RANKS)
        for rank, node in enumerate(nodes):
            self.assertEqual(node["rank_index"], rank)
            self.assertEqual(node["stage_index"], rank)
            self.assertEqual(node["control_endpoint"]["host"], f"spark{rank:x}")
            self.assertIn(node["control_endpoint"]["port"], CONTROL)
            self.assertEqual(node["control_endpoint"]["port"], 23080 + rank)
            self.assertEqual(node["transport_host"], f"spark{rank:x}-fabric")
            self.assertGreater(node["kv_backing_maximum_bytes"], 0)
            self.assertLessEqual(node["kv_backing_maximum_bytes"], 16 * 1024 ** 3)
            self.assertIn(f"/home/spark{rank:x}/sparkdata/dsv4_pro.tp4pp4",
                          node["runtime_root"])
        self.assertIn(document["transport"]["control_port_base"], TRANSPORT)
        self.assertEqual(document["weightd"]["socket_path"],
                         "/run/sparkpipe-weightd-shared/weightd.sock")

    def test_runtime_limits_match_validated_deployment(self):
        document = json.loads(DEPLOYMENT.read_text())
        self.assertEqual(document["runtime_limits"], {
            "max_inflight_submissions": 4,
            "max_active_sequences": 1024,
            "max_input_rows": 1024,
            "resident_sequence_capacity": 4096,
            "kv_logical_page_capacity": 1048576,
            "kv_physical_page_capacity": 16384,
        })
        self.assertEqual(document["eos_token_ids"], [])

    def test_stage_config_collective_block(self):
        config = json.loads(STAGE.read_text())
        collective = config["tp_collective"]
        self.assertEqual(collective["listen_port"], 53080)
        self.assertEqual(collective["peer_ports"], list(COLLECTIVE))
        for port in collective["peer_ports"]:
            self.assertLess(port, 65536)
        self.assertEqual(config["stage_pack_path"],
                         "packs/dsv4_pro_tp4_pp4_stage.spstage")
        self.assertEqual(config["cuda_graph_count_by_pp_stage"], [49, 46, 46, 46])


def make_source_root(base: Path) -> Path:
    """A minimal fake on-NVMe runtime root for prepare dry-runs."""
    source = base / "sparkdata" / "dsv4_pro.tp4pp4"
    (source / "config").mkdir(parents=True)
    (source / "packs").mkdir()
    (source / "lib").mkdir()
    shutil.copyfile(STAGE, source / "config" / "dsv4_pro_tp4_pp4_stage.json")
    pack = source / "packs" / "dsv4_pro_tp4_pp4_stage.spstage"
    pack.write_bytes(b"pack-bytes")
    pack_with_sha = Path(str(pack) + ".sha256")
    pack_with_sha.write_text("0" * 64 + "  " + pack.name + "\n")
    Path(str(pack) + ".experts").write_bytes(b"\x00" * 16)
    for name in ("libdsv4_pro_tp4_pp4_serving_adapter.so", "model_driver.so",
                 "libhidden_transport_spark_host_rdma_verbs.so"):
        (source / "lib" / name).write_bytes(b"so")
    return source


class WrapperPrepare(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsv4pro-wrapper-test-"))
        # The queue contract pins the runtime root to /tmp/sparkqueue-ATTEMPT.
        self.root = Path(f"/tmp/sparkqueue-{ATTEMPT}")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir()
        self.source = make_source_root(self.tmp)
        self.base_env = {
            "PATH": os.environ["PATH"],
            "SPARK_QUEUE_ATTEMPT": ATTEMPT,
            "SPARK_QUEUE_RUNTIME_ROOT": str(self.root),
            "SPARK_QUEUE_RANK": "0",
            "SPARK_QUEUE_SIZE": str(RANKS),
            "SPARK_QUEUE_PORTS": "23080:23095,53080:53095,64080:64095",
            "FAMILY_SKIP_HOST_CHECK": "1",
            "FAMILY_WEIGHTD_MODE": "shared-socket",
            "FAMILY_WEIGHTD_SOCKET": str(self.tmp / "absent-weightd.sock"),
            "FAMILY_DEPLOYMENT_SOURCE": str(DEPLOYMENT),
            "FAMILY_EXEC_PREFIX": str(self.tmp / "build"),
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.root, ignore_errors=True)

    def deploy_path(self) -> Path:
        return self.root / "deployment.json"

    def test_source_root_substitution(self):
        # The committed deployment points at the real on-NVMe roots; the
        # dry-run rewrites rank 0's node onto the fake source tree.
        document = json.loads(DEPLOYMENT.read_text())
        document["nodes"][0]["runtime_root"] = str(self.source)
        override = self.tmp / "deployment.override.json"
        override.write_text(json.dumps(document))
        env = dict(self.base_env,
                   FAMILY_DEPLOYMENT_SOURCE=str(override))
        result = subprocess.run(["bash", str(WRAPPER)], env=env,
                                capture_output=True, text=True)
        # Prepare must succeed; the launch then fails closed on the
        # missing shared socket (never started by hand).
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("shared weightd socket is not present", result.stderr)
        prepared = json.loads(self.deploy_path().read_text())
        node = prepared["nodes"][0]
        self.assertEqual(node["runtime_root"], str(self.root / "runtime"))
        self.assertEqual(node["kv_backing_directory"], str(self.root / "kv"))
        self.assertEqual(node["control_endpoint"]["port"], 23080)
        self.assertEqual(node["transport_host"], "spark0-fabric")
        runtime = self.root / "runtime"
        digests = list((runtime / "packs").glob("*.sha256"))
        self.assertEqual(len(digests), 1)
        self.assertTrue((runtime / "packs" / "dsv4_pro_tp4_pp4_stage.spstage.experts").is_file())
        for name in ("libdsv4_pro_tp4_pp4_serving_adapter.so", "model_driver.so",
                     "libhidden_transport_spark_host_rdma_verbs.so"):
            self.assertTrue((runtime / "lib" / name).exists())
        stage = json.loads((runtime / "config" / "dsv4_pro_tp4_pp4_stage.json").read_text())
        self.assertEqual(stage["tp_collective"]["listen_port"], 53080)
        self.assertEqual(stage["tp_collective"]["peer_ports"], list(COLLECTIVE))
        expected_id = int(ATTEMPT[:15], 16) + 1 + LANE
        self.assertEqual(stage["tp_collective"]["collective_identifier"], expected_id)

    def test_fail_closed_missing_sidecar(self):
        Path(str(self.source / "packs" / "dsv4_pro_tp4_pp4_stage.spstage") +
             ".experts").unlink()
        document = json.loads(DEPLOYMENT.read_text())
        document["nodes"][0]["runtime_root"] = str(self.source)
        override = self.tmp / "deployment.override.json"
        override.write_text(json.dumps(document))
        env = dict(self.base_env, FAMILY_DEPLOYMENT_SOURCE=str(override))
        result = subprocess.run(["bash", str(WRAPPER)], env=env,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing pack sidecar", result.stderr)

    def test_fail_closed_missing_lib(self):
        (self.source / "lib" / "model_driver.so").unlink()
        document = json.loads(DEPLOYMENT.read_text())
        document["nodes"][0]["runtime_root"] = str(self.source)
        override = self.tmp / "deployment.override.json"
        override.write_text(json.dumps(document))
        env = dict(self.base_env, FAMILY_DEPLOYMENT_SOURCE=str(override))
        result = subprocess.run(["bash", str(WRAPPER)], env=env,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime artifact missing", result.stderr)

    def test_fail_closed_unreserved_ports(self):
        document = json.loads(DEPLOYMENT.read_text())
        document["nodes"][0]["runtime_root"] = str(self.source)
        override = self.tmp / "deployment.override.json"
        override.write_text(json.dumps(document))
        env = dict(self.base_env, FAMILY_DEPLOYMENT_SOURCE=str(override),
                   SPARK_QUEUE_PORTS="23080:23095")
        result = subprocess.run(["bash", str(WRAPPER)], env=env,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not inside a queue-reserved range", result.stderr)

    def test_fail_closed_wrong_transport_host(self):
        document = json.loads(DEPLOYMENT.read_text())
        document["nodes"][0]["runtime_root"] = str(self.source)
        document["nodes"][0]["transport_host"] = "spark0"
        override = self.tmp / "deployment.override.json"
        override.write_text(json.dumps(document))
        env = dict(self.base_env, FAMILY_DEPLOYMENT_SOURCE=str(override))
        result = subprocess.run(["bash", str(WRAPPER)], env=env,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("transport_host", result.stderr)

    def test_fail_closed_unknown_listener_key(self):
        config = json.loads(STAGE.read_text())
        config["mystery_bridge_port"] = 12345
        (self.source / "config" / "dsv4_pro_tp4_pp4_stage.json").write_text(
            json.dumps(config))
        document = json.loads(DEPLOYMENT.read_text())
        document["nodes"][0]["runtime_root"] = str(self.source)
        override = self.tmp / "deployment.override.json"
        override.write_text(json.dumps(document))
        env = dict(self.base_env, FAMILY_DEPLOYMENT_SOURCE=str(override))
        result = subprocess.run(["bash", str(WRAPPER)], env=env,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unhandled listener configuration", result.stderr)


if __name__ == "__main__":
    unittest.main()

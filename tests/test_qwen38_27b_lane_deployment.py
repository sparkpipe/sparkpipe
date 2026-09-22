#!/usr/bin/env python3
"""Host-side checks for the qwen38_27b lane deployment stager.

The shared-socket family wrapper must fail closed without a GPU: verified
firmware, exactly one staged *.sha256 digest, listeners inside the reserved
lane blocks, an exact-member stage configuration, and a private deployment
tree under the queue runtime root.
"""
import hashlib
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import importlib.util

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qwen38_27b_lane_deployment",
    REPOSITORY_ROOT / "tools" / "qwen38_27b_lane_deployment.py")
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)

ATTEMPT = "0123456789abcdef0123456789abcdef"


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fresh_attempt(instance):
    shutil.rmtree(instance.queue_root, ignore_errors=True)
    instance.attempt = uuid.uuid4().hex
    instance.queue_root = Path("/tmp/sparkqueue-" + instance.attempt)
    instance.addCleanup(lambda: shutil.rmtree(instance.queue_root, ignore_errors=True))
    return instance.attempt


class FakeLane(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qwen38-lane-")
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.firmware = base / "firmware"
        self.pack_dir = base / "packs"
        self.attempt = uuid.uuid4().hex
        self.queue_root = Path("/tmp/sparkqueue-" + self.attempt)
        self.addCleanup(lambda: shutil.rmtree(self.queue_root, ignore_errors=True))
        self.pack_payload = b"qwen38-27b-nvfp4a16-tp4-rank-pack" * 1024
        self.pack_name = "tp4-rank02.q38sp"
        self.pack_dir.mkdir()
        (self.pack_dir / self.pack_name).write_bytes(self.pack_payload)
        (self.pack_dir / (self.pack_name + ".experts")).write_bytes(b"\x00" * 64)
        for relative in ("bin/sparkpipe_model_residentd",
                         "lib/model_serving_adapter.so",
                         "lib/hidden_transport.so",
                         "stages/stage_000/model_driver.so"):
            artifact = self.firmware / relative
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"firmware:" + relative.encode())
        (self.firmware / "SOURCE_COMMIT").write_text("1" * 40 + "\n")
        self.write_firmware_sums()
        self.env = {
            "SPARK_QUEUE_RUNTIME_ROOT": str(self.queue_root),
            "SPARK_QUEUE_RANK": "2",
            "SPARK_QUEUE_SIZE": "4",
            "SPARK_QUEUE_ATTEMPT": self.attempt,
            "QWEN38_27B_LANE_FIRMWARE_ROOT": str(self.firmware),
            "QWEN38_27B_LANE_SHARED_SOCKET": "/tmp/shared-weightd.sock",
            "QWEN38_27B_LANE_PACK_DIR": str(self.pack_dir),
            "QWEN38_27B_LANE_HOSTS": "spark0,spark1,spark2,spark3",
            "SPARK_QUEUE_PORTS": "23016:23031,53016:53031,64016:64031",
        }

    def tearDown(self):
        self.temporary.cleanup()

    def write_firmware_sums(self):
        lines = []
        for relative in ("bin/sparkpipe_model_residentd",
                         "lib/model_serving_adapter.so",
                         "lib/hidden_transport.so",
                         "stages/stage_000/model_driver.so"):
            payload = (self.firmware / relative).read_bytes()
            lines.append(digest_bytes(payload) + "  " + relative)
        (self.firmware / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    def stage(self, env=None):
        return lane.prepare(dict(self.env if env is None else env))


class LaneStagingTests(FakeLane):
    def test_stage_private_deployment(self):
        staged = self.stage()
        root = self.queue_root
        self.assertTrue((root / "deployment.json").is_file())
        self.assertTrue((root / "kv").is_dir())
        runtime = Path(staged["runtime_root"])
        self.assertEqual(str(runtime), str(root / "runtime"))
        deployment = json.loads((root / "deployment.json").read_text())
        self.assertEqual(deployment["weightd"]["socket_path"], "/tmp/shared-weightd.sock")
        self.assertEqual(deployment["schema_version"], 2)
        self.assertEqual(deployment["driver"]["program_name"], "resident_decode")
        self.assertEqual(len(deployment["nodes"]), 4)
        node = deployment["nodes"][2]
        self.assertEqual(node["control_endpoint"]["port"], 23018)
        self.assertEqual(node["transport_host"], "spark2")
        self.assertEqual(node["runtime_root"], str(runtime))
        self.assertEqual(node["adapter_configuration_path"], "config/stage.json")
        self.assertGreater(node["kv_backing_maximum_bytes"], 0)
        self.assertEqual(deployment["transport"]["control_port_base"], 64016)
        for relative in ("lib/model_serving_adapter.so", "lib/hidden_transport.so",
                         "stages/stage_000/model_driver.so"):
            link = runtime / relative
            self.assertTrue(link.is_symlink(), relative)
            self.assertEqual(link.resolve(), (self.firmware / relative).resolve())

    def test_exactly_one_digest_and_correct_value(self):
        staged = self.stage()
        packs = self.queue_root / "runtime" / "packs"
        digests = sorted(entry.name for entry in packs.iterdir()
                         if entry.name.endswith(".sha256"))
        self.assertEqual(digests, [self.pack_name + ".sha256"])
        self.assertEqual(staged["pack_sha256"], digest_bytes(self.pack_payload))
        written = (packs / digests[0]).read_text()
        self.assertEqual(written.strip(), digest_bytes(self.pack_payload))
        self.assertTrue((packs / self.pack_name).is_symlink())

    def test_stage_config_exact_members_and_ports(self):
        self.stage()
        config = json.loads(
            (self.queue_root / "runtime" / "config" / "stage.json").read_text())
        self.assertEqual(sorted(config), sorted(lane.STAGE_MEMBERS))
        self.assertEqual(config["tp_degree"], 4)
        self.assertEqual(config["tp_rank"], 2)
        self.assertEqual(config["stage_pack_path"], "packs/" + self.pack_name)
        self.assertEqual(config["tp_collective"]["listen_port"], 53018)
        self.assertEqual(config["tp_collective"]["peer_ports"],
                         [53016, 53017, 53018, 53019])
        sessions = config["tp_collective"]["session_ports"]
        assigned = [port for row in sessions for port in row if port]
        self.assertEqual(len(assigned), 12)
        self.assertEqual(len(set(assigned)), 12)
        self.assertTrue(all(53020 <= port <= 53031 for port in assigned))
        for row in range(4):
            self.assertEqual(sessions[row][row], 0)
        hc = config["tp_collective"]["session_ports_hc"]
        self.assertEqual(hc, [[0] * 4 for _ in range(4)])
        self.assertEqual(sorted(config["tp_collective"]),
                         sorted(lane.TP_COLLECTIVE_MEMBERS))
        identifier = config["tp_collective"]["collective_identifier"]
        self.assertEqual(identifier, int(self.attempt[:15], 16) + 1)

    def test_every_listener_inside_reserved_blocks(self):
        ports = lane.listener_ports(
            ["spark0", "spark1", "spark2", "spark3"],
            {"control": 23016, "collective": 53016, "transport": 64016})
        self.assertEqual(ports[0], 23016)
        self.assertEqual(ports[-1], 64019)
        lane.check_reserved(ports, [(23016, 23031), (53016, 53031), (64016, 64031)])
        with self.assertRaises(lane.LaneError):
            lane.check_reserved(ports, [(23016, 23031), (53016, 53031)])

    def test_experts_sidecar_staged(self):
        staged = self.stage()
        staged_experts = Path(staged["runtime_root"]) / "packs" / (self.pack_name + ".experts")
        self.assertTrue(staged_experts.is_symlink())

    def test_missing_experts_manifest_refused(self):
        (self.pack_dir / (self.pack_name + ".experts")).unlink()
        with self.assertRaises(lane.LaneError):
            self.stage()

    def test_trusted_source_digest_reused(self):
        sidecar_digest = "a" * 64
        (self.pack_dir / (self.pack_name + ".sha256")).write_text(sidecar_digest + "\n")
        env = dict(self.env)
        env["QWEN38_27B_LANE_TRUST_SOURCE_DIGEST"] = "1"
        staged = self.stage(env)
        self.assertEqual(staged["pack_sha256"], sidecar_digest)
        fresh_attempt(self)
        env["SPARK_QUEUE_ATTEMPT"] = self.attempt
        env["SPARK_QUEUE_RUNTIME_ROOT"] = str(self.queue_root)
        env["QWEN38_27B_LANE_TRUST_SOURCE_DIGEST"] = "0"
        staged = self.stage(env)
        self.assertEqual(staged["pack_sha256"], digest_bytes(self.pack_payload))


class LaneFailClosedTests(FakeLane):
    def test_missing_pack(self):
        (self.pack_dir / self.pack_name).unlink()
        with self.assertRaises(lane.LaneError):
            self.stage()

    def test_firmware_without_sums(self):
        (self.firmware / "SHA256SUMS").unlink()
        with self.assertRaises(lane.LaneError):
            self.stage()

    def test_firmware_digest_mismatch(self):
        target = self.firmware / "lib" / "hidden_transport.so"
        target.write_bytes(b"tampered")
        with self.assertRaises(lane.LaneError):
            self.stage()

    def test_firmware_missing_required_artifact_pin(self):
        sums = (self.firmware / "SHA256SUMS").read_text().splitlines()
        (self.firmware / "SHA256SUMS").write_text(
            "\n".join(line for line in sums if "hidden_transport" not in line) + "\n")
        with self.assertRaises(lane.LaneError):
            self.stage()

    def test_wrong_queue_namespace(self):
        env = dict(self.env)
        env["SPARK_QUEUE_RUNTIME_ROOT"] = str(Path(self.temporary.name) / "elsewhere")
        with self.assertRaises(lane.LaneError):
            self.stage(env)

    def test_reused_runtime_root_refused(self):
        self.stage()
        with self.assertRaises(lane.LaneError):
            self.stage()

    def test_size_mismatch(self):
        env = dict(self.env)
        env["SPARK_QUEUE_SIZE"] = "8"
        with self.assertRaises(lane.LaneError):
            self.stage(env)

    def test_mesh_rank_count_and_range(self):
        with self.assertRaises(lane.LaneError):
            lane.parse_mesh_ranks("0,1,2", 4)
        with self.assertRaises(lane.LaneError):
            lane.parse_mesh_ranks("0,1,2,16", 4)
        with self.assertRaises(lane.LaneError):
            lane.parse_mesh_ranks("0,1,2,2", 4)
        self.assertEqual(lane.parse_mesh_ranks("0,1,2,3", 4), [0, 1, 2, 3])

    def test_mesh_ranks_env_validated(self):
        env = dict(self.env)
        env["QWEN38_27B_LANE_MESH_RANKS"] = "0,1,2"
        with self.assertRaises(lane.LaneError):
            self.stage(env)

    def test_second_digest_rejected(self):
        packs = Path(self.temporary.name) / "staged-packs"
        lane.stage_pack(self.pack_dir / self.pack_name, packs, False)
        packs.joinpath("stray.sha256").write_text("b" * 64 + "\n")
        other = self.pack_dir / "tp4-rank03.q38sp"
        other.write_bytes(b"another rank pack")
        with self.assertRaises(lane.LaneError):
            lane.stage_pack(other, packs, False)

    def test_bad_attempt_and_relative_socket(self):
        env = dict(self.env)
        env["SPARK_QUEUE_ATTEMPT"] = "zz"
        with self.assertRaises(lane.LaneError):
            self.stage(env)
        env = dict(self.env)
        env["QWEN38_27B_LANE_SHARED_SOCKET"] = "weightd.sock"
        with self.assertRaises(lane.LaneError):
            self.stage(env)

    def test_sequence_position_cap(self):
        env = dict(self.env)
        env["QWEN38_27B_LANE_MAX_SEQUENCE_POSITIONS"] = "16384"
        with self.assertRaises(lane.LaneError):
            self.stage(env)


class LaneSessionGridTests(unittest.TestCase):
    def test_grid_is_compact_and_off_diagonal(self):
        grid = lane.session_grid(53016, 4)
        flat = [port for row in grid for port in row]
        self.assertEqual(flat[0:4], [0, 53020, 53021, 53022])
        self.assertEqual(flat[4:8], [53023, 0, 53024, 53025])
        self.assertEqual(flat[8:12], [53026, 53027, 0, 53028])
        self.assertEqual(flat[12:16], [53029, 53030, 53031, 0])
        self.assertEqual(sorted(set(flat)), [0] + list(range(53020, 53032)))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import copy
import importlib.util
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
spec = importlib.util.spec_from_file_location("inference_smoke", Path(__file__).resolve().parents[1] / "tools/inference_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeReceipts(unittest.TestCase):
    def setUp(self):
        self.batch = {"requests": [dict(request_id=11, sequence_id=21, output_token_budget=2),
                                   dict(request_id=12, sequence_id=22, output_token_budget=1)], "stop_token_ids": []}
        self.reference = dict(model_id="fixture", model_revision="r1", vocabulary_size=100,
                              eos_token_ids=[9], tokens={"11": [7, 8], "12": [3]})
        self.events = [dict(schema_version=1, event="ready", model_id="fixture", model_revision="r1")]
        for request_id, sequence_id, handle in ((11, 21, 101), (12, 22, 102)):
            self.events.append(dict(schema_version=1, event="accepted", status=0, request_id=request_id,
                                    sequence_id=sequence_id, request_handle=handle))
        for request_id, sequence_id, handle, index, token in ((11, 21, 101, 0, 7), (12, 22, 102, 0, 3), (11, 21, 101, 1, 8)):
            self.events.append(dict(schema_version=1, event="token", status=0, request_id=request_id,
                                    sequence_id=sequence_id, request_handle=handle, token_index=index,
                                    generated_token_count=index + 1, token_id=token, stop_token=False))
        for request_id, sequence_id, handle, count in ((12, 22, 102, 1), (11, 21, 101, 2)):
            self.events.append(dict(schema_version=1, event="completed", status=0, request_id=request_id,
                                    sequence_id=sequence_id, request_handle=handle, generated_token_count=count,
                                    stop_token=False))

    def verify(self, events=None):
        return smoke.verify_events(map(json.dumps, self.events if events is None else events), self.batch, self.reference)

    def test_interleaved_real_event_schema(self):
        self.assertEqual(self.verify(), 3)

    def test_identity_counts_and_tokens_are_independent_oracles(self):
        for index, key, value in ((0, "model_revision", "wrong"), (2, "request_handle", 101),
                                  (3, "request_id", 999), (3, "sequence_id", 22),
                                  (3, "request_handle", 102), (3, "status", 7),
                                  (3, "token_index", 1), (3, "generated_token_count", 8),
                                  (3, "token_id", 99), (3, "token_id", 100),
                                  (7, "generated_token_count", 1), (7, "event", "cancelled")):
            with self.subTest(index=index, key=key, value=value):
                events = copy.deepcopy(self.events)
                events[index][key] = value
                with self.assertRaises(ValueError):
                    self.verify(events)

    def test_missing_duplicate_and_reordered_terminal(self):
        for events in (self.events[:-1], self.events + self.events[-1:],
                       self.events[:3] + [self.events[-1]] + self.events[3:], []):
            with self.assertRaises(ValueError):
                self.verify(events)

    def test_early_eos_is_on_token_event_not_terminal_event(self):
        self.events[3].update(token_id=9, stop_token=True)
        del self.events[5]
        self.events[-1]["generated_token_count"] = 1
        self.reference["tokens"]["11"] = [9]
        self.assertEqual(self.verify(), 2)
        self.events[3]["stop_token"] = False
        with self.assertRaisesRegex(ValueError, "short completion"):
            self.verify()

    def test_full_namespace_includes_secondary_sessions_and_draft(self):
        config = {"draft_bridge_port": 7793, "tp_collective": {
            "listen_port": 63640, "peer_ports": [63640, 63641],
            "session_ports": [[0, 63001], [63002, 0]],
            "session_ports_hc": [[0, 64001], [64002, 0]]}}
        mapping = {str(port): 30000 + index for index, port in enumerate((7793, 63640, 63641, 63001, 63002, 64001, 64002))}
        result = smoke.private_config(config, mapping, [(30000, 30010)], {31000}, "a" * 32)
        self.assertEqual(result["draft_bridge_port"], 30000)
        self.assertEqual(result["tp_collective"]["session_ports_hc"], [[0, 30005], [30006, 0]])
        self.assertEqual(config["draft_bridge_port"], 7793)
        del mapping["64002"]
        with self.assertRaisesRegex(ValueError, "missing private mapping"):
            smoke.private_config(config, mapping, [(30000, 30010)], set(), "a" * 32)

    def test_unhandled_conflicting_and_unreserved_listener_rejected(self):
        for config, mapping, protected in (({"other_port": 7793}, {"7793": 30000}, set()),
                                           ({"listen_port": 7793}, {"7793": 30000}, {30000}),
                                           ({"listen_port": 7793}, {"7793": 40000}, set()),
                                           ({"peer_ports": [1, 2]}, {"1": 30000, "2": 30000}, set())):
            with self.assertRaises(ValueError):
                smoke.private_config(config, mapping, [(30000, 30010)], protected, "a" * 32)

    def test_concurrency_requires_overlapping_decode_not_only_live_processes(self):
        first = dict(valid=True, started_seconds=0, ttft_seconds=2, total_seconds=8)
        second = dict(valid=True, started_seconds=1, ttft_seconds=3, total_seconds=6)
        self.assertEqual(smoke.verify_concurrent([first, second])["overlap_seconds"], 3)
        second["started_seconds"] = 5
        with self.assertRaisesRegex(ValueError, "did not overlap"):
            smoke.verify_concurrent([first, second])
        second.update(started_seconds=0, valid=False)
        with self.assertRaisesRegex(ValueError, "failed"):
            smoke.verify_concurrent([first, second])

    def test_assigned_lane_requires_exact_single_rank_receipt(self):
        log = "GLM mesh lane mode=explicit requested=2 resolved=2 capacity=8 rank=5\n"
        smoke.verify_lane(log, 2, 5)
        for wrong in (log.replace("explicit", "automatic"), log.replace("resolved=2", "resolved=1"),
                      log.replace("rank=5", "rank=4"), log + log, ""):
            with self.assertRaises(ValueError):
                smoke.verify_lane(wrong, 2, 5)

    def test_observed_device_budget_and_ready_owner_are_required(self):
        limits = dict(weightd_device_bytes=2 << 20, weightd_overhead_bytes=1 << 20, model_device_bytes=4 << 20)
        receipt = {}
        children = [Mock(pid=10), Mock(pid=11), Mock(pid=12)]
        with patch.object(smoke.subprocess, "run", return_value=Mock(stdout="10, 3\n11, 4\n12, 2\n")):
            smoke.gpu_memory(children, limits, receipt, True)
        self.assertEqual(receipt["gpu_peak_bytes"], {"10": 3 << 20, "11": 4 << 20, "12": 2 << 20})
        for output in ("10, 4\n11, 4\n12, 2\n", "10, 3\n11, 5\n12, 2\n", "10, 3\n11, 4\n", "10, N/A\n"):
            with patch.object(smoke.subprocess, "run", return_value=Mock(stdout=output)), self.assertRaises(ValueError):
                smoke.gpu_memory(children, limits, {}, True)

    def test_shutdown_rejects_crash_and_timeout(self):
        for script, expected in (("raise SystemExit(3)", "FAIL"),
                                 ("import signal,time; signal.signal(signal.SIGTERM,lambda *x:exit(0)); print('ready',flush=True); time.sleep(60)", "PASS"),
                                 ("import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)", "FAIL")):
            child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, start_new_session=True)
            try:
                child.stdout.readline()
                if "SystemExit" in script:
                    child.wait()
                receipt = {"status": "PASS"}
                smoke.stop_owned([child], receipt, timeout=0.1)
                self.assertEqual(receipt["status"], expected)
                self.assertIsNotNone(child.poll())
            finally:
                child.stdout.close()
                if child.poll() is None:
                    child.kill()
                    child.wait()


class SmokePreparation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="inference-profile-")
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name)
        self.attempt = uuid.uuid4().hex
        self.root = Path("/tmp/sparkqueue-" + self.attempt)
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        config = {"stage_pack_path": "packs/model.pack", "tp_collective": {
            "backend_module_path": "lib/collective.so", "peer_hosts": ["spark0"],
            "peer_ports": [19000], "listen_port": 19000}}
        contents = {"lib/driver.so": "driver", "lib/adapter.so": "adapter", "lib/transport.so": "transport",
                    "lib/collective.so": "collective", "config/model.json": json.dumps(config),
                    "packs/model.pack": "fixture pack", "packs/model.pack.experts": "fixture manifest",
                    "packs/model.pack.sha256": hashlib.sha256(b"fixture pack").hexdigest()}
        for name, text in contents.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.deployment = {"schema_version": 2, "eos_token_ids": [9], "coordinator_rank_index": 0,
                           "driver": {"shared_object_path": "lib/driver.so", "program_name": "resident_decode"},
                           "adapter": {"shared_object_path": "lib/adapter.so"},
                           "transport": {"shared_object_path": "lib/transport.so", "control_port_base": 19000},
                           "nodes": [{"rank_index": 0, "runtime_root": str(self.source),
                                      "adapter_configuration_path": "config/model.json", "kv_backing_directory": "/old/cache",
                                      "kv_backing_maximum_bytes": 4096}]}
        self.deployment_path = self.source / "deployment.json"
        self.deployment_path.write_text(json.dumps(self.deployment))
        self.batch_path = self.source / "batch.json"
        self.batch_path.write_text('{"requests":[],"stop_token_ids":[]}')
        self.reference = dict(deployment_sha256=smoke.digest(self.deployment_path), eos_token_ids=[9],
                              environment={}, source_commit="a" * 40, batch_sha256=smoke.digest(self.batch_path),
                              executables={"build/" + name: "b" * 64 for name in
                                           ("sparkpipe_weightd", "sparkpipe_model_residentd", "sparkpipe_model_batch")},
                              ranks=[{"model_device_bytes": 1024,
                                      "assets": {name: smoke.digest(self.source / name) for name in contents if not name.endswith(".pack")}}])
        self.reference_path = self.source / "reference.json"
        self.spec = dict(hosts=["spark0"], port_base=30000, port_map={"19000": 30002}, environment={},
                         deployment=str(self.deployment_path), batch=str(self.batch_path), reference=str(self.reference_path),
                         budgets=dict(weightd_device_bytes=4096, model_device_bytes=1024, expert_pool_bytes=2048, spine_bytes=2048))
        self.environment = dict(SPARK_QUEUE_ATTEMPT=self.attempt, SPARK_QUEUE_RUNTIME_ROOT=str(self.root),
                                SPARK_QUEUE_RANK="0", SPARK_QUEUE_SIZE="1", SPARK_QUEUE_PORTS="30000:30010",
                                SPARK_QUEUE_DEVICE_MEMORY_MIB="1")

    def prepare(self):
        self.reference_path.write_text(json.dumps(self.reference))
        original = smoke.digest
        with patch.object(smoke, "digest", side_effect=lambda p: "b" * 64 if str(p).startswith("build/") else original(p)), \
             patch.object(smoke.subprocess, "check_output", return_value="a" * 40), \
             patch.object(smoke.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            return smoke.prepare(self.spec, self.environment)

    def test_private_runtime_preserves_pack_and_changes_all_owned_paths(self):
        root, rank, hosts, batch, reference, deployment = self.prepare()
        self.assertEqual(rank, 0)
        self.assertTrue((root / "runtime/packs/model.pack").is_symlink())
        self.assertFalse((root / "runtime/config/model.json").is_symlink())
        self.assertEqual(deployment["weightd"]["socket_path"], str(root / "weightd.sock"))
        self.assertEqual(deployment["nodes"][0]["kv_backing_directory"], str(root / "kv"))
        private = json.loads((root / "runtime/config/model.json").read_text())
        self.assertEqual(private["tp_collective"]["listen_port"], 30002)
        self.assertEqual(json.loads((self.source / "config/model.json").read_text())["tp_collective"]["listen_port"], 19000)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_changed_deployment_invalidates_memory_plan(self):
        self.deployment["runtime_limits"] = {"resident_sequence_capacity": 1000}
        self.deployment_path.write_text(json.dumps(self.deployment))
        with self.assertRaisesRegex(ValueError, "deployment differs"):
            self.prepare()

    def test_unpinned_collective_module_rejected(self):
        del self.reference["ranks"][0]["assets"]["lib/collective.so"]
        with self.assertRaisesRegex(ValueError, "collective module is not pinned"):
            self.prepare()

    def test_changed_asset_rejected(self):
        (self.source / "lib/driver.so").write_text("changed driver")
        with self.assertRaisesRegex(ValueError, "asset differs"):
            self.prepare()

    def test_memory_and_port_reservations_enforced(self):
        self.spec["budgets"]["model_device_bytes"] = 2 * 1024 * 1024
        with self.assertRaisesRegex(ValueError, "exceeds queue reservation"):
            self.prepare()
        self.spec["budgets"]["model_device_bytes"] = 1024
        self.environment["SPARK_QUEUE_PORTS"] = "30000:30001"
        with self.assertRaisesRegex(ValueError, "reserve every listener"):
            self.prepare()


class SharedSmokePreparation(SmokePreparation):
    def shared(self, count=4):
        self.deployment["runtime_limits"] = dict(max_inflight_submissions=1, max_active_sequences=1,
                                                max_input_rows=1, resident_sequence_capacity=1)
        self.deployment_path.write_text(json.dumps(self.deployment))
        self.reference["deployment_sha256"] = smoke.digest(self.deployment_path)
        self.batch_path.write_text('{"requests":[{"request_id":11}],"stop_token_ids":[]}')
        self.reference["batch_sha256"] = smoke.digest(self.batch_path)
        working = "packs/model.pack.wset"
        (self.source / working).write_bytes(bytes(8))
        self.reference["ranks"][0]["assets"][working] = smoke.digest(self.source / working)
        self.reference["executables"]["build/weightd_warm"] = "b" * 64
        self.spec["environment"] = {"SPARK_GLM5_NEXT_GRAPH_PATH": "1", "SPARK_GLM5_NEXT_PIN_EXPERTS": "1", "CUDA_MODULE_LOADING": "LAZY", "CUDA_MODULE_DATA_LOADING": "LAZY", "CUDA_DEVICE_MAX_CONNECTIONS": "32"}
        self.reference["environment"] = self.spec["environment"]
        self.spec["working_set"] = dict(mode="full", path=working)
        self.spec["replicas"] = [dict(lane=index, port_base=30000 + 100 * index,
                                      port_map={"19000": 30002 + 100 * index}) for index in range(count)]
        self.spec["budgets"]["weightd_overhead_bytes"] = 1024
        self.environment["SPARK_QUEUE_PORTS"] = "30000:31000"

    def test_two_three_four_residents_share_only_daemon_and_pinned_pack(self):
        for count in (2, 3, 4):
            with self.subTest(count=count):
                self.shared(count)
                root, *_ = self.prepare()
                identifiers = set()
                for index in range(count):
                    local = root if index == 0 else root / ("resident-" + str(index))
                    deployment = json.loads((local / "deployment.json").read_text())
                    config = json.loads((local / "runtime/config/model.json").read_text())
                    self.assertEqual(deployment["weightd"]["socket_path"], str(root / "weightd.sock"))
                    self.assertEqual(deployment["nodes"][0]["runtime_root"], str(local / "runtime"))
                    self.assertEqual(deployment["nodes"][0]["kv_backing_directory"], str(local / "kv"))
                    self.assertEqual(config["tp_collective"]["listen_port"], 30002 + index * 100)
                    identifiers.add(config["tp_collective"]["collective_identifier"])
                    working = local / "runtime/packs/model.pack.wset"
                    self.assertFalse(working.is_symlink())
                    working.write_bytes(b"recorded-runtime-demand")
                    self.assertEqual((self.source / "packs/model.pack.wset").read_bytes(), bytes(8))
                self.assertEqual(len(identifiers), count)
                shutil.rmtree(root)

    def test_runner_starts_one_daemon_explicit_lanes_and_real_wset_command(self):
        self.shared(2)
        batch = dict(requests=[dict(request_id=11, sequence_id=21, output_token_budget=2)], stop_token_ids=[])
        self.batch_path.write_text(json.dumps(batch))
        self.reference.update(model_id="fixture", model_revision="r1", vocabulary_size=100,
                              tokens={"11": [7, 8]}, batch_sha256=smoke.digest(self.batch_path))
        self.reference_path.write_text(json.dumps(self.reference))
        self.spec.update(timeout_seconds=5)
        calls, warm_calls = [], []

        def start(command, **kwargs):
            calls.append((command, kwargs["env"]))
            if "residentd" in command[0]:
                lane = kwargs["env"]["SPARK_WEIGHTD_LANE"]
                kwargs["stdout"].write((f"GLM mesh lane mode=explicit requested={lane} resolved={lane} capacity=8 rank=0\nmodel_residentd ready rank=0 \n").encode())
            else:
                kwargs["stdout"].write(b"spark_weightd ready fixture\n")
            kwargs["stdout"].flush()
            return Mock(pid=100 + len(calls), poll=lambda: None)

        def execute(command, **kwargs):
            if command[0] == "build/weightd_warm":
                warm_calls.append(command)
                kwargs["stdout"].write(b"WSET-WARM keys=1 elapsed_ms=1\n")
            return subprocess.CompletedProcess(command, 0)

        def measure(command, timeout, **kwargs):
            common = dict(schema_version=1, request_id=11, sequence_id=21, request_handle=1, status=0)
            events = [dict(schema_version=1, event="ready", model_id="fixture", model_revision="r1"),
                      dict(common, event="accepted")]
            for index, token in enumerate((7, 8)):
                events.append(dict(common, event="token", token_index=index, generated_token_count=index+1, token_id=token))
            events.append(dict(common, event="completed", generated_token_count=2))
            for event in events:
                kwargs["event_sink"].write((json.dumps(event) + "\n").encode())
            return dict(valid=True, errors=[], ttft_seconds=1, total_seconds=3)

        original = smoke.digest
        with patch.dict(smoke.os.environ, self.environment, clear=True), \
             patch.object(smoke, "digest", side_effect=lambda p: "b" * 64 if str(p).startswith("build/") else original(p)), \
             patch.object(smoke.subprocess, "check_output", return_value="a" * 40), \
             patch.object(smoke.subprocess, "run", side_effect=execute), \
             patch.object(smoke.subprocess, "Popen", side_effect=start), \
             patch.object(smoke, "gpu_memory"), patch.object(smoke, "measure", side_effect=measure), \
             patch.object(smoke, "stop_owned") as stopped:
            self.assertEqual(smoke.run(self.spec), 0)
        self.assertEqual([call[0][0] for call in calls], ["build/sparkpipe_weightd", "build/sparkpipe_model_residentd", "build/sparkpipe_model_residentd"])
        self.assertEqual([call[1]["SPARK_WEIGHTD_LANE"] for call in calls[1:]], ["0", "1"])
        self.assertEqual(len(warm_calls), 1)
        self.assertEqual(warm_calls[0][6], "--wset")
        self.assertEqual(warm_calls[0][7], str(self.root / "runtime/packs/model.pack.wset"))
        self.assertEqual(len(stopped.call_args.args[0]), 3)
        receipt = json.loads((self.root / "receipt.json").read_text())
        self.assertEqual(receipt["tokens"], 4)
        self.assertGreater(receipt["concurrent_decode"]["overlap_seconds"], 0)

    def test_shared_budget_counts_every_resident_and_daemon_overhead(self):
        self.shared()
        self.spec["budgets"]["model_device_bytes"] = 300000
        self.reference["ranks"][0]["model_device_bytes"] = 300000
        with self.assertRaisesRegex(ValueError, "exceeds queue reservation"):
            self.prepare()

    def test_duplicate_lane_listener_or_missing_reservation_rejected(self):
        for field, value, reason in (("lane", 0, "collective lane"), ("port_base", 30000, "listener"),
                                      ("port_map", {"19000": 30002}, "listener"),
                                      ("port_base", 40000, "reserve resident listeners")):
            self.shared()
            self.spec["replicas"][1][field] = value
            with self.assertRaisesRegex(ValueError, reason):
                self.prepare()
            shutil.rmtree(self.root, ignore_errors=True)

    def test_shared_cuda_knobs_must_match_qualified_baseline(self):
        self.shared()
        self.spec["environment"]["CUDA_MODULE_LOADING"] = "EAGER"
        with self.assertRaisesRegex(ValueError, "pinned CUDA"):
            self.prepare()

    def test_working_set_mode_cannot_hide_pin_all_cost(self):
        self.shared()
        self.spec["working_set"]["mode"] = "partial"
        with self.assertRaisesRegex(ValueError, "match explicit working set mode"):
            self.prepare()
        self.spec["environment"].update(SPARK_GLM5_NEXT_GRAPH_PATH="0", SPARK_GLM5_NEXT_PIN_EXPERTS="0")
        with self.assertRaisesRegex(ValueError, "must not premap full pack"):
            self.prepare()
        self.spec["working_set"]["mode"] = "full"
        self.spec["environment"].update(SPARK_GLM5_NEXT_GRAPH_PATH="1", SPARK_GLM5_NEXT_PIN_EXPERTS="1")
        self.spec["budgets"]["expert_pool_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "full pack budget"):
            self.prepare()

    def test_b1_cap_and_complete_pinned_working_set_are_required(self):
        self.shared()
        self.deployment["runtime_limits"]["max_input_rows"] = 2
        self.deployment_path.write_text(json.dumps(self.deployment))
        self.reference["deployment_sha256"] = smoke.digest(self.deployment_path)
        with self.assertRaisesRegex(ValueError, "requires B1"):
            self.prepare()
        self.shared()
        working = "packs/model.pack.wset"
        (self.source / working).write_bytes(bytes(7))
        self.reference["ranks"][0]["assets"][working] = smoke.digest(self.source / working)
        with self.assertRaisesRegex(ValueError, "complete key pairs"):
            self.prepare()
        del self.reference["ranks"][0]["assets"][working]
        with self.assertRaisesRegex(ValueError, "not pinned"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()

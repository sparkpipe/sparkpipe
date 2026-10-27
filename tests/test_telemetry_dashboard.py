import os
import sys

sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","tools","telemetry"))
import json
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import spark_telemetry_dashboard as dashboard


class SparkTelemetryDashboardTest(unittest.TestCase):
    def setUp(self):
        dashboard.reset_node_error_streaks()
        dashboard.MODEL_LAYER_PARTITIONS = None
        dashboard.MODEL_LAYER_PARTITIONS_JSON_OVERRIDE = "/nonexistent/ds4-dashboard-test-no-layer-partitions.json"
        dashboard.REPO_ROOT_OVERRIDE = ""

    def test_snapshot_excludes_untrusted_power_from_total(self):
        payload = {
            "updated_iso": "2026-05-26T00:00:00+00:00",
            "updated_unix": 1,
            "nodes": {
                "spark0": {"sample_count": 2, "last_gpu_util_pct": 96, "last_gpu_power_w": 0, "last_gpu_power_raw_w": 11, "last_gpu_power_limit_w": 0, "last_gpu_power_known": 0, "last_gpu_power_reason": "nvml-power-limit-unavailable"},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            snap = dashboard.build_snapshot(str(path))
        node = snap["nodes"][0]
        self.assertFalse(snap["power_known"])
        self.assertEqual(snap["total_gpu_power_w"],0)
        self.assertFalse(node["gpu_power_known"])
        self.assertEqual(node["gpu_power_w"],0)
        self.assertEqual(node["gpu_power_raw_w"],11)
        self.assertEqual(node["gpu_power_reason"],"nvml-power-limit-unavailable")

    def write_snapshot(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return(dashboard.build_snapshot(str(path)))

    def healthy_node(self, **fields):
        row = {"sample_count": 2, "last_sample_age_s": 2.0, "last_gpu_util_pct": 10, "last_gpu_temp_c": 40}
        row.update(fields)
        return(row)

    def test_fresh_summary_and_sample_are_not_stale(self):
        snap = self.write_snapshot({"updated_unix": int(time.time()), "nodes": {"spark0": self.healthy_node()}})
        self.assertFalse(snap["stale"])
        self.assertEqual(snap["nodes"][0]["state"],"idle")

    def test_old_summary_marks_every_node_stale(self):
        snap = self.write_snapshot({"updated_unix": int(time.time()) - int(dashboard.SUMMARY_STALE_AFTER_S) - 60, "nodes": {"spark0": self.healthy_node(last_vllm_requests_running=1)}})
        self.assertTrue(snap["stale"])
        self.assertGreater(snap["age_s"],dashboard.SUMMARY_STALE_AFTER_S)
        self.assertEqual(snap["nodes"][0]["state"],"warn")
        self.assertEqual(snap["nodes"][0]["state_label"],"stale")
        self.assertEqual(snap["active_nodes"],0)

    def test_old_node_sample_is_stale_in_a_fresh_summary(self):
        snap = self.write_snapshot({"updated_unix": int(time.time()), "nodes": {"spark0": self.healthy_node(last_sample_age_s=dashboard.SAMPLE_STALE_AFTER_S + 1.0, last_vllm_requests_running=1)}})
        self.assertFalse(snap["stale"])
        self.assertEqual(snap["nodes"][0]["state_label"],"stale")

    def test_latest_sample_error_is_shown(self):
        snap = self.write_snapshot({"updated_unix": int(time.time()), "nodes": {"spark0": self.healthy_node(last_sample_error="nvidia-smi timeout")}})
        self.assertEqual(snap["nodes"][0]["state"],"warn")
        self.assertEqual(snap["nodes"][0]["state_label"],"sample error")
        self.assertEqual(snap["nodes"][0]["last_sample_error"],"nvidia-smi timeout")

    def test_node_down_requires_three_distinct_error_snapshots(self):
        payload = {
            "updated_iso": "2026-05-26T00:00:00+00:00",
            "updated_unix": 1,
            "nodes": {"spark6": {"sample_count": 0, "error": "ssh timed out"}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            first = dashboard.build_snapshot(str(path))
            second_same_snapshot = dashboard.build_snapshot(str(path))
            payload["updated_unix"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            second = dashboard.build_snapshot(str(path))
            payload["updated_unix"] = 3
            path.write_text(json.dumps(payload), encoding="utf-8")
            third = dashboard.build_snapshot(str(path))
        self.assertEqual(first["nodes"][0]["state"], "warn")
        self.assertEqual(first["nodes"][0]["error_streak"], 1)
        self.assertEqual(second_same_snapshot["nodes"][0]["state"], "warn")
        self.assertEqual(second_same_snapshot["nodes"][0]["error_streak"], 1)
        self.assertEqual(second["nodes"][0]["state"], "warn")
        self.assertEqual(second["nodes"][0]["error_streak"], 2)
        self.assertEqual(third["nodes"][0]["state"], "down")
        self.assertEqual(third["nodes"][0]["error_streak"], 3)

    def test_snapshot_marks_global_ds4_api_queue_as_known(self):
        payload = {
            "updated_iso": "2026-05-26T00:00:00+00:00",
            "updated_unix": 1,
            "queue": {
                "local_queue_source": "ds4-api:http://10.20.0.10:8700",
                "local_queue_depth": 2,
                "local_queue_running": 2,
                "local_queue_queued": 1,
                "local_queue_ds_services": "dsv4_flash_pp8,qwen27_bf16_pp8",
                "local_queue_ds_service_count": 2,
                "local_queue_ds_model_count": 6,
                "local_queue_active_services": "dsv4_flash_pp8",
                "local_queue_active_service_count": 1,
                "local_queue_last_service": "dsv4_flash_pp8",
                "local_queue_kv_shards": 8,
                "local_queue_kv_by_node": "spark0:dsv4_flash_pp8",
                "local_queue_prompt_tok_s": 12.5,
                "local_queue_completion_tok_s": 44.25,
                "local_queue_total_tok_s": 56.75,
            },
            "nodes": {"spark0": {"sample_count": 1, "last_gpu_util_pct": 0, "last_vllm_metrics_up": 0}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            snap = dashboard.build_snapshot(str(path))
        self.assertFalse(snap["nodes"][0]["local_queue_known"])
        self.assertTrue(snap["nodes"][0]["kv_known"])
        self.assertEqual(snap["nodes"][0]["kv_label"],"api")
        self.assertEqual(snap["nodes"][0]["ds_service_id"],"dsv4_flash_pp8")
        self.assertEqual(snap["nodes"][0]["input_tok_s"],0.0)
        self.assertEqual(snap["nodes"][0]["output_tok_s"],0.0)
        self.assertEqual(snap["input_tok_s"],0.0)
        self.assertEqual(snap["output_tok_s"],0.0)
        self.assertEqual(snap["tok_s"],0.0)
        self.assertEqual(snap["vllm_running"],2.0)
        self.assertEqual(snap["vllm_waiting"],1.0)
        self.assertEqual(snap["queue_depth"],2.0)
        self.assertTrue(snap["ds_services_known"])
        self.assertEqual(snap["ds_services"],"dsv4_flash_pp8")
        self.assertEqual(snap["ds_service_count"],1.0)
        self.assertEqual(snap["ds_catalog_services"],"dsv4_flash_pp8,qwen27_bf16_pp8")
        self.assertEqual(snap["ds_catalog_service_count"],2.0)
        self.assertEqual(snap["ds_model_count"],6.0)
        self.assertEqual(snap["ds_last_service"],"dsv4_flash_pp8")
        self.assertEqual(snap["ds_kv_shards"],8.0)

    def test_snapshot_labels_pipeline_metrics_as_service_output(self):
        payload = {
            "updated_iso": "2026-05-26T00:00:00+00:00",
            "updated_unix": 1,
            "nodes": {
                "spark0": {
                    "sample_count": 1,
                    "last_gpu_util_pct": 0,
                    "last_vllm_metrics_up": 1,
                    "last_vllm_metrics_scope": "pipeline",
                    "last_vllm_pipeline_parallel_size": 8,
                    "last_vllm_pipeline_node_rank": 0,
                    "last_vllm_requests_running": 90,
                    "last_vllm_prompt_tokens_per_s": 1.5,
                    "last_vllm_generation_tokens_per_s": 300,
                    "last_vllm_tokens_per_s": 301.5,
                },
                "spark1": {"sample_count": 1, "last_gpu_util_pct": 80, "last_vllm_metrics_up": 0},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            snap = dashboard.build_snapshot(str(path))
        head = snap["nodes"][0]
        self.assertEqual(head["token_scope"],"pipeline")
        self.assertEqual(head["pipeline_parallel_size"],8)
        self.assertEqual(head["pipeline_node_rank"],0)
        self.assertEqual(head["output_tok_s"],300)
        self.assertEqual(snap["output_tok_s"],300)
        self.assertEqual(snap["tok_s"],301.5)
        self.assertEqual(snap["nodes"][1]["output_tok_s"],0)
        self.assertFalse(snap["nodes"][1]["vllm_metrics_up"])

    def test_snapshot_never_falls_back_to_equal_pipeline_shares(self):
        payload = {
            "updated_iso": "2026-05-26T00:00:00+00:00",
            "updated_unix": 1,
            "nodes": {
                "spark0": {
                    "sample_count": 1,
                    "last_vllm_metrics_up": 1,
                    "last_vllm_generation_tokens_per_s_by_model": "new-model-pp8:80",
                    "last_vllm_requests_running_by_model": "new-model-pp8:8",
                    "last_vllm_pipeline_stage_models": "new-model-pp8",
                    "last_vllm_pipeline_stage_pp_by_model": "new-model-pp8:8",
                    "last_vllm_pipeline_stage_rank_by_model": "new-model-pp8:0",
                },
                "spark1": {
                    "sample_count": 1,
                    "last_vllm_pipeline_stage_models": "new-model-pp8",
                    "last_vllm_pipeline_stage_pp_by_model": "new-model-pp8:8",
                    "last_vllm_pipeline_stage_rank_by_model": "new-model-pp8:1",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(payload),encoding="utf-8")
            dashboard.REPO_ROOT_OVERRIDE = str(Path(tmp) / "empty-repo")
            snap = dashboard.build_snapshot(str(path))
        self.assertEqual(snap["models"][0]["output_tok_s"],80)
        self.assertEqual(snap["output_tok_s"],80)
        self.assertEqual(snap["nodes"][0]["output_tok_s"],0)
        self.assertEqual(snap["nodes"][1]["output_tok_s"],0)
        self.assertNotIn("model_allocations",snap["nodes"][0])
        self.assertNotEqual(snap["nodes"][0]["output_tok_s"],10)

    def test_snapshot_uses_explicit_repo_root_for_installed_dashboard(self):
        payload = {
            "updated_iso": "2026-05-26T00:00:00+00:00",
            "updated_unix": 1,
            "nodes": {
                "spark0": {
                    "sample_count": 1,
                    "last_vllm_metrics_up": 1,
                    "last_vllm_generation_tokens_per_s_by_model": "example-pp8:38",
                    "last_vllm_requests_running_by_model": "example-pp8:38",
                    "last_vllm_pipeline_stage_models": "example-pp8",
                    "last_vllm_pipeline_stage_pp_by_model": "example-pp8:8",
                    "last_vllm_pipeline_stage_rank_by_model": "example-pp8:0",
                },
                "spark1": {
                    "sample_count": 1,
                    "last_vllm_pipeline_stage_models": "example-pp8",
                    "last_vllm_pipeline_stage_pp_by_model": "example-pp8:8",
                    "last_vllm_pipeline_stage_rank_by_model": "example-pp8:1",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            summary = Path(tmp) / "installed" / "summary.json"
            (root / "v2" / "profiles" / "production").mkdir(parents=True)
            (root / "v2" / "profiles" / "models").mkdir(parents=True)
            summary.parent.mkdir()
            (root / "v2" / "profiles" / "production" / "first3_resident_memory_budget.json").write_text(json.dumps({"layer_partitions":{"example_service":[4,5,6,5,5,5,4,4]}}),encoding="utf-8")
            (root / "v2" / "profiles" / "models" / "example.json").write_text(json.dumps({"routing":{"pipeline":{"served_model_name":"example-pp8","service_id":"example_service"}}}),encoding="utf-8")
            summary.write_text(json.dumps(payload),encoding="utf-8")
            dashboard.REPO_ROOT_OVERRIDE = str(root)
            snap = dashboard.build_snapshot(str(summary))
        self.assertEqual(snap["nodes"][0]["output_tok_s"],4)
        self.assertEqual(snap["nodes"][1]["output_tok_s"],5)

    def test_snapshot_uses_installed_layer_partition_map(self):
        payload = {
            "updated_iso": "2026-05-26T00:00:00+00:00",
            "updated_unix": 1,
            "nodes": {
                "spark0": {
                    "sample_count": 1,
                    "last_vllm_metrics_up": 1,
                    "last_vllm_generation_tokens_per_s_by_model": "example-pp8:38",
                    "last_vllm_requests_running_by_model": "example-pp8:38",
                    "last_vllm_pipeline_stage_models": "example-pp8",
                    "last_vllm_pipeline_stage_pp_by_model": "example-pp8:8",
                    "last_vllm_pipeline_stage_rank_by_model": "example-pp8:0",
                },
                "spark1": {
                    "sample_count": 1,
                    "last_vllm_pipeline_stage_models": "example-pp8",
                    "last_vllm_pipeline_stage_pp_by_model": "example-pp8:8",
                    "last_vllm_pipeline_stage_rank_by_model": "example-pp8:1",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            installed = Path(tmp) / "model_layer_partitions.json"
            summary.write_text(json.dumps(payload),encoding="utf-8")
            installed.write_text(json.dumps({"model_layer_partitions":{"example-pp8":[4,5,6,5,5,5,4,4]}}),encoding="utf-8")
            dashboard.REPO_ROOT_OVERRIDE = str(Path(tmp) / "repo-without-profiles")
            dashboard.MODEL_LAYER_PARTITIONS_JSON_OVERRIDE = str(installed)
            dashboard.MODEL_LAYER_PARTITIONS = None
            snap = dashboard.build_snapshot(str(summary))
        self.assertEqual(snap["nodes"][0]["token_scope"],"allocated")
        self.assertEqual(snap["nodes"][0]["output_tok_s"],4)
        self.assertEqual(snap["nodes"][1]["output_tok_s"],5)

    def test_snapshot_ignores_ds4_stage_payload_for_node_telemetry(self):
        payload = {
            "updated_iso": "2026-05-26T00:00:00+00:00",
            "updated_unix": int(time.time()),
            "queue": {
                "local_queue_source": "ds4-api:http://10.20.0.10:8700",
                "local_queue_depth": 0,
                "local_queue_running": 0,
                "local_queue_queued": 0,
                "local_queue_stage_service_by_node": "spark0:dsv4_flash_pp8",
                "local_queue_stage_sample_count_by_node": "spark0:8",
                "local_queue_stage_gpu_util_by_node": "spark0:96",
                "local_queue_stage_gpu_temp_by_node": "spark0:47",
                "local_queue_stage_gpu_power_by_node": "spark0:17.5",
                "local_queue_stage_gpu_power_raw_by_node": "spark0:17.5",
                "local_queue_stage_gpu_power_limit_by_node": "spark0:120",
                "local_queue_stage_gpu_power_known_by_node": "spark0:1",
                "local_queue_stage_gpu_power_source_by_node": "spark0:nvml.power.draw",
                "local_queue_stage_cpu_pct_by_node": "spark0:4",
                "local_queue_stage_mem_pct_by_node": "spark0:85",
                "local_queue_stage_prompt_tok_s_by_node": "spark0:12",
                "local_queue_stage_generation_tok_s_by_node": "spark0:7",
            },
            "nodes": {"spark0": {"sample_count": 1, "last_gpu_util_pct": 0, "last_gpu_temp_c": 41, "last_gpu_power_known": 0, "last_cpu_util_pct": 3, "last_mem_used_pct": 31, "last_vllm_metrics_up": 1, "last_vllm_requests_running": 0, "last_vllm_requests_waiting": 0, "last_vllm_prompt_tokens_per_s": 0, "last_vllm_generation_tokens_per_s": 0}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            snap = dashboard.build_snapshot(str(path))
        node = snap["nodes"][0]
        self.assertEqual(node["state"],"idle")
        self.assertEqual(node["state_label"],"idle")
        self.assertEqual(node["sample_count"],1)
        self.assertEqual(node["gpu_pct"],0)
        self.assertEqual(node["gpu_temp_c"],41)
        self.assertEqual(node["gpu_power_w"],0)
        self.assertFalse(node["gpu_power_known"])
        self.assertEqual(node["cpu_pct"],60)
        self.assertEqual(node["mem_pct"],31)
        self.assertEqual(node["input_tok_s"],0)
        self.assertEqual(node["output_tok_s"],0)
        self.assertEqual(node["tok_s"],0)
        self.assertEqual(node["ds_service_id"],"dsv4_flash_pp8")
        self.assertEqual(snap["active_nodes"],0)
        self.assertEqual(snap["busy_gpu_nodes"],0)
        self.assertEqual(snap["avg_gpu_pct"],0)
        self.assertEqual(snap["vllm_running"],0)
        self.assertEqual(snap["vllm_waiting"],0)

    def test_history_reads_last_csv_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spark2.csv"
            path.write_text(
                "unix_ts,iso_ts,node,gpu_util_pct,gpu_temp_c,gpu_power_w,cpu_util_pct,mem_used_pct,vllm_requests_running,vllm_requests_waiting,vllm_kv_cache_pct,vllm_tokens_per_s,vllm_prompt_tokens_per_s,vllm_generation_tokens_per_s,vllm_prompt_cache_hit_pct,vllm_external_prefix_cache_hit_pct\n"
                "1,2026-05-26T00:00:01+00:00,spark2,10,40,20,5,30,1,0,3,0.5,0.2,0.3,10,0\n"
                "2,2026-05-26T00:00:02+00:00,spark2,20,41,21,6,31,2,1,4,1.5,0.6,0.9,20,5\n"
                "3,2026-05-26T00:00:03+00:00,spark2,30,42,22,7,32,3,2,5,2.5,1.1,1.4,30,10\n",
                encoding="utf-8",
            )
            hist = dashboard.build_history(tmp, "spark2", 2)
        self.assertTrue(hist["ok"])
        self.assertEqual(hist["node"], "spark2")
        self.assertEqual(len(hist["points"]), 2)
        self.assertEqual(hist["points"][0]["iso_ts"], "2026-05-26T00:00:02+00:00")
        self.assertEqual(hist["points"][1]["gpu_pct"], 30)
        self.assertEqual(hist["points"][1]["tok_s"], 2.5)
        self.assertEqual(hist["points"][1]["input_tok_s"], 1.1)
        self.assertEqual(hist["points"][1]["output_tok_s"], 1.4)
        self.assertEqual(hist["points"][1]["cache_hit_pct"], 30)
        self.assertEqual(hist["points"][1]["cpu_pct"], 140)

    def test_history_rejects_invalid_node_name(self):
        hist = dashboard.build_history("/tmp", "../spark2", 2)
        self.assertFalse(hist["ok"])
        self.assertEqual(hist["error"], "invalid node")

    def test_history_skips_repeated_csv_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spark2.csv"
            path.write_text(
                "unix_ts,iso_ts,node,gpu_util_pct,gpu_temp_c,gpu_power_w,cpu_util_pct,mem_used_pct,vllm_requests_running,vllm_requests_waiting,vllm_kv_cache_pct,vllm_tokens_per_s,vllm_prompt_tokens_per_s,vllm_generation_tokens_per_s,vllm_prompt_cache_hit_pct,vllm_external_prefix_cache_hit_pct\n"
                "unix_ts,iso_ts,node,gpu_util_pct,gpu_temp_c,gpu_power_w,cpu_util_pct,mem_used_pct,vllm_requests_running,vllm_requests_waiting,vllm_kv_cache_pct,vllm_tokens_per_s,vllm_prompt_tokens_per_s,vllm_generation_tokens_per_s,vllm_prompt_cache_hit_pct,vllm_external_prefix_cache_hit_pct\n"
                "1,2026-05-26T00:00:01+00:00,spark2,10,40,20,5,30,1,0,3,0.5,0.2,0.3,10,0\n",
                encoding="utf-8",
            )
            hist = dashboard.build_history(tmp, "spark2", 10)
        self.assertEqual(len(hist["points"]), 1)
        self.assertEqual(hist["points"][0]["iso_ts"], "2026-05-26T00:00:01+00:00")

    def test_default_history_limit_is_one_hour(self):
        self.assertEqual(dashboard.history_limit(None), 720)
        self.assertEqual(dashboard.history_limit(""), 720)

    def test_http_summary_allocates_pipeline_tokens_by_explicit_repo_root(self):
        payload = {
            "updated_iso": "2026-05-26T00:00:00+00:00",
            "updated_unix": 1,
            "nodes": {
                "spark0": {
                    "sample_count": 1,
                    "last_vllm_metrics_up": 1,
                    "last_vllm_generation_tokens_per_s_by_model": "example-pp8:38",
                    "last_vllm_requests_running_by_model": "example-pp8:38",
                    "last_vllm_pipeline_stage_models": "example-pp8",
                    "last_vllm_pipeline_stage_pp_by_model": "example-pp8:8",
                    "last_vllm_pipeline_stage_rank_by_model": "example-pp8:0",
                },
                "spark1": {
                    "sample_count": 1,
                    "last_vllm_pipeline_stage_models": "example-pp8",
                    "last_vllm_pipeline_stage_pp_by_model": "example-pp8:8",
                    "last_vllm_pipeline_stage_rank_by_model": "example-pp8:1",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            summary = Path(tmp) / "summary.json"
            nodes_dir = Path(tmp) / "nodes"
            (root / "v2" / "profiles" / "production").mkdir(parents=True)
            (root / "v2" / "profiles" / "models").mkdir(parents=True)
            nodes_dir.mkdir()
            (root / "v2" / "profiles" / "production" / "first3_resident_memory_budget.json").write_text(json.dumps({"layer_partitions":{"example_service":[4,5,6,5,5,5,4,4]}}),encoding="utf-8")
            (root / "v2" / "profiles" / "models" / "example.json").write_text(json.dumps({"routing":{"pipeline":{"served_model_name":"example-pp8","service_id":"example_service"}}}),encoding="utf-8")
            summary.write_text(json.dumps(payload),encoding="utf-8")
            dashboard.REPO_ROOT_OVERRIDE = str(root)
            dashboard.MODEL_LAYER_PARTITIONS = None
            server = dashboard.ReusableThreadingHTTPServer(("127.0.0.1",0),dashboard.make_handler(str(summary),str(nodes_dir)))
            thread = threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                with urllib.request.urlopen("http://127.0.0.1:%d/api/summary" % port,timeout=3) as fp:
                    snap = json.loads(fp.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)
        self.assertEqual(snap["nodes"][0]["token_scope"],"allocated")
        self.assertEqual(snap["nodes"][0]["output_tok_s"],4)
        self.assertEqual(snap["nodes"][1]["output_tok_s"],5)
        self.assertEqual(snap["nodes"][0]["model_allocations"][0]["share_pct"],10.53)

    def test_dashboard_http_server_reuses_address_on_restart(self):
        self.assertTrue(dashboard.ReusableThreadingHTTPServer.allow_reuse_address)

    def test_dashboard_cpu_display_uses_twenty_core_range(self):
        self.assertEqual(dashboard.DISPLAY_CPU_PCT_MAX, 2000)
        self.assertEqual(dashboard.display_cpu_pct(40), 800)
        self.assertIn('"cpu_pct"', dashboard.DASHBOARD_HTML)
        self.assertIn("const CPU_PCT_MAX=2000", dashboard.DASHBOARD_HTML)
        self.assertIn("cpu_pct:CPU_PCT_MAX", dashboard.DASHBOARD_HTML)

    def test_dashboard_card_omits_power_and_marks_missing_vllm_na(self):
        self.assertNotIn('Pwr <b>', dashboard.DASHBOARD_HTML)
        self.assertNotIn('let pwr=', dashboard.DASHBOARD_HTML)
        self.assertIn('Svc <b>${n.ds_service_id||"n/a"}</b>', dashboard.DASHBOARD_HTML)
        self.assertIn('bar("KV",n.kv_pct,"kv",n.kv_known,n.kv_label)', dashboard.DASHBOARD_HTML)
        self.assertIn('function workKnown(n)', dashboard.DASHBOARD_HTML)
        self.assertIn('n.vllm_metrics_up||Number(n.local_q_depth)>0', dashboard.DASHBOARD_HTML)
        self.assertIn('workKnown(n)?rate(n[key])+unit:"n/a"', dashboard.DASHBOARD_HTML)
        self.assertIn('function tokenScope(n)', dashboard.DASHBOARD_HTML)
        self.assertIn('n.token_scope==="allocated"', dashboard.DASHBOARD_HTML)
        self.assertIn('Queue <b>${queueVal(n)}</b>', dashboard.DASHBOARD_HTML)
        self.assertIn('function modelHint(n)', dashboard.DASHBOARD_HTML)
        self.assertIn('m.output_tok_s', dashboard.DASHBOARD_HTML)

    def test_dashboard_summary_combines_tokens_and_omits_power(self):
        self.assertIn('grid-template-columns:repeat(6,minmax(110px,1fr))', dashboard.DASHBOARD_HTML)
        self.assertIn('metric("GPU Avg",d.gpu_known?pct(d.avg_gpu_pct):"n/a")', dashboard.DASHBOARD_HTML)
        self.assertIn('metric("Active Svc",d.ds_services_known?`${fmt(d.ds_service_count)} svc`:"n/a")', dashboard.DASHBOARD_HTML)
        self.assertIn('metric("Live In/Out",`${val(d.input_tok_s)} / ${val(d.output_tok_s)}`)', dashboard.DASHBOARD_HTML)
        self.assertIn('function renderModels(d)', dashboard.DASHBOARD_HTML)
        self.assertIn('class="model-table"', dashboard.DASHBOARD_HTML)
        self.assertIn('<th>Out/s</th>', dashboard.DASHBOARD_HTML)
        self.assertIn('<th>Total/s</th>', dashboard.DASHBOARD_HTML)
        self.assertIn('no active model token rate', dashboard.DASHBOARD_HTML)
        self.assertNotIn('metric("Total Power"', dashboard.DASHBOARD_HTML)
        self.assertNotIn('metric("In tok/s"', dashboard.DASHBOARD_HTML)
        self.assertNotIn('metric("Out tok/s"', dashboard.DASHBOARD_HTML)
        self.assertNotIn('metric("Cache hit"', dashboard.DASHBOARD_HTML)

    def test_dashboard_history_omits_power_series(self):
        self.assertNotIn('"power_w"', dashboard.DASHBOARD_HTML)
        self.assertNotIn('data.power_known', dashboard.DASHBOARD_HTML)
        self.assertNotIn('PWR', dashboard.DASHBOARD_HTML)


    def test_valid_node_names_cover_the_fleet_and_auxiliary_hosts(self):
        for node in dashboard.telemetry_common.SPARK_NODES + dashboard.telemetry_common.AUXILIARY_NODES:
            self.assertTrue(dashboard.valid_node_name(node), node)
        for node in ("lenovo10","hub","spark0;rm","../spark1"):
            self.assertFalse(dashboard.valid_node_name(node), node)

if __name__ == "__main__":
    unittest.main()

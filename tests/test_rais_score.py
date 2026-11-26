#!/usr/bin/env python3
from __future__ import annotations

import http.server
import json
import math
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import rais_bench
import rais_score


def sample_valid_record(model: str = "glm-5.3-flash", engine: str = "sparkpipe", scale: float = 1.0) -> dict:
    return {
        "rais_version": "1.0",
        "model": model,
        "engine": engine,
        "tier": "full",
        "hardware": "16x-gb10-1x100g",
        "metrics": {
            "b1_short_tok_s": 55.0 * scale,
            "b1_short_ctx": 256,
            "b1_long_tok_s": 46.0 * scale,
            "b1_long_ctx": 65536,
            "itl_p50_ms": 18.0,
            "itl_p99_ms": 20.0,
            "speculation": {
                "alpha_prose": 2.2,
                "alpha_code": 3.4,
                "alpha_edit": 4.0,
                "verify_v_b1": 1.25,
                "verify_b1_enabled": True,
                "verify_v_b8": 1.6,
                "verify_b8_enabled": True,
            },
            "batch": {
                "4": {"aggregate_tok_s": 130.0 * scale},
                "8": {"aggregate_tok_s": 175.0 * scale},
                "16": {"aggregate_tok_s": 227.0 * scale},
                "32": {"unsupported": True, "aggregate_tok_s": 0.0},
                "64": {"unsupported": True, "aggregate_tok_s": 0.0},
            },
            "b8_long_tok_s": 130.0 * scale,
            "b8_long_ctx": 32768,
            "prefill_8k_tokens": 8192,
            "prefill_8k_tok_s": 1100.0 * scale,
            "prefill_32k_tokens": 32768,
            "prefill_32k_tok_s": 1250.0 * scale,
            "prefill_128k_tokens": 130560,
            "prefill_128k_tok_s": 1350.0 * scale,
            "warm_8k_prefix_tokens": 8192,
            "warm_8k_new_tokens": 128,
            "warm_8k_ttft_ms": 200.0 / scale,
            "warm_100k_prefix_tokens": 100000,
            "warm_100k_new_tokens": 128,
            "warm_100k_ttft_ms": 420.0 / scale,
        },
        "gates": {
            "prose_exact": True,
            "code_exact": True,
            "prefix_exact": True,
            "errors": 0,
        },
    }


class TestRaisWeightsAndCatalog(unittest.TestCase):
    def test_submetric_weights_sum_to_one_and_five_pillars(self) -> None:
        self.assertEqual(rais_score.RAIS_VERSION, "1.1")
        self.assertEqual(len(rais_score.SUBMETRIC_SPECS), 15)
        total = sum(rais_score.SUBMETRIC_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=9)
        for pillar in ("P1", "P2", "P3", "P4"):
            p_sum = sum(w for _, p, _, w in rais_score.SUBMETRIC_SPECS if p == pillar)
            self.assertAlmostEqual(p_sum, 0.225, places=9)
        p5_sum = sum(w for _, p, _, w in rais_score.SUBMETRIC_SPECS if p == "P5")
        self.assertAlmostEqual(p5_sum, 0.10, places=9)
        self.assertAlmostEqual(
            rais_score.SUBMETRIC_WEIGHTS["p2b_verify_b1"] + rais_score.SUBMETRIC_WEIGHTS["p2b_verify_b8"],
            0.117,
            places=9,
        )
        self.assertAlmostEqual(rais_score.SUBMETRIC_WEIGHTS["p5a_cold_boot"], 0.05, places=9)
        self.assertAlmostEqual(rais_score.SUBMETRIC_WEIGHTS["p5b_memory_efficiency"], 0.03, places=9)
        self.assertAlmostEqual(rais_score.SUBMETRIC_WEIGHTS["p5c_batch_invariance"], 0.02, places=9)

    def test_all_hardware_profiles_present(self) -> None:
        expected = {
            "16x-gb10-1x100g",
            "16x-gb10-2x100g",
            "4x-gb10-1x100g",
            "4x-gb10-2x100g",
            "1x-gb10",
            "1x-m5-ultra",
            "8x-h200-nvlink",
        }
        self.assertEqual(set(rais_score.HARDWARE_PROFILES.keys()), expected)
        for hw in rais_score.HARDWARE_PROFILES.values():
            self.assertGreater(hw.nvme_bw_gb_s_per_rank, 0.0)
            self.assertGreater(hw.ram_gib_per_rank, 0.0)
            self.assertGreater(hw.os_reserve_gib_per_rank, 0.0)

    def test_all_deployment_lanes_and_fleet_weights(self) -> None:
        lanes = json.loads((ROOT / "deployment" / "lanes.json").read_text(encoding="utf-8"))["lanes"]
        self.assertEqual(len(lanes), 27)
        for lane_name, lane_cfg in lanes.items():
            self.assertIn(lane_name, rais_score.MODEL_SPECS)
            spec = rais_score.MODEL_SPECS[lane_name]
            self.assertEqual(spec.context_tokens, lane_cfg["context_tokens"])
            self.assertGreater(spec.hidden, 0)
            self.assertGreater(spec.layers, 0)
            self.assertGreater(spec.kv_bytes_per_token, 0)
            self.assertGreater(spec.active_params_b, 0.0)
            self.assertEqual(rais_score.resolve_model_name(lane_cfg["model"]), lane_name)

        canonical = [m for m in rais_score.MODEL_SPECS.values() if m.fleet_weight > 0.0]
        self.assertEqual(len(canonical), 19)
        self.assertAlmostEqual(sum(m.fleet_weight for m in canonical), 1.0, places=9)
        self.assertAlmostEqual(sum(m.fleet_weight for m in canonical if m.tier == 1), 0.55, places=9)
        self.assertAlmostEqual(sum(m.fleet_weight for m in canonical if m.tier == 2), 0.25, places=9)
        self.assertAlmostEqual(sum(m.fleet_weight for m in canonical if m.tier == 3), 0.20, places=9)


class TestRaisRooflineMath(unittest.TestCase):
    def test_overlap_bounds_and_single_rank_network_zero(self) -> None:
        for hw in rais_score.HARDWARE_PROFILES.values():
            for model in rais_score.MODEL_SPECS.values():
                rf = rais_score.compute_roofline(model, hw, batch=1, ctx=256)
                self.assertAlmostEqual(rf.t_floor_ms, max(rf.t_compute_ms, rf.t_memory_ms, rf.t_network_ms), places=3)
                self.assertAlmostEqual(rf.t_serial_ms, rf.t_compute_ms + rf.t_memory_ms + rf.t_network_ms, places=3)
                self.assertGreaterEqual(rf.overlap_gain_max, 1.0)
                self.assertLessEqual(rf.overlap_gain_max, 3.0001)
                if hw.ranks == 1:
                    self.assertEqual(rf.t_network_ms, 0.0)
                    self.assertEqual(rf.net_bytes_per_rank, 0.0)

    def test_physical_floors_match_gb10_targets(self) -> None:
        hw = rais_score.resolve_hardware("16x-gb10-1x100g")
        glm_full = rais_score.compute_roofline(rais_score.MODEL_SPECS["glm-5.3"], hw, batch=1, ctx=256)
        glm_flash = rais_score.compute_roofline(rais_score.MODEL_SPECS["glm-5.3-flash"], hw, batch=1, ctx=256)
        k3 = rais_score.compute_roofline(rais_score.MODEL_SPECS["kimi-k3"], hw, batch=1, ctx=256)
        q27_fp8 = rais_score.compute_roofline(rais_score.MODEL_SPECS["qwen3.8-27b-fp8"], hw, batch=1, ctx=256)
        self.assertEqual(glm_full.binding_bottleneck, "memory")
        self.assertTrue(100.0 <= glm_full.floor_tok_s <= 115.0)
        self.assertTrue(230.0 <= glm_flash.floor_tok_s <= 250.0)
        self.assertTrue(30.0 <= k3.floor_tok_s <= 33.5)
        self.assertTrue(150.0 <= q27_fp8.floor_tok_s <= 170.0)

    def test_dual_rail_halves_network_bound_prefill_floor(self) -> None:
        hw1 = rais_score.resolve_hardware("16x-gb10-1x100g")
        hw2 = rais_score.resolve_hardware("16x-gb10-2x100g")
        model = rais_score.MODEL_SPECS["glm-5.3-flash"]
        rf1 = rais_score.compute_roofline(model, hw1, batch=1, ctx=8192, rows_per_seq=8192, is_prefill=True)
        rf2 = rais_score.compute_roofline(model, hw2, batch=1, ctx=8192, rows_per_seq=8192, is_prefill=True)
        self.assertEqual(rf1.binding_bottleneck, "network")
        self.assertLess(rf2.t_network_ms, rf1.t_network_ms * 0.52)

    def test_long_context_increases_memory_bytes_and_moe_union(self) -> None:
        hw = rais_score.resolve_hardware("16x-gb10-1x100g")
        for name in ("glm-5.3-flash", "kimi-k3", "qwen3.8-27b"):
            model = rais_score.MODEL_SPECS[name]
            short_rf = rais_score.compute_roofline(model, hw, batch=1, ctx=256)
            long_rf = rais_score.compute_roofline(model, hw, batch=1, ctx=65536)
            self.assertGreater(long_rf.mem_bytes_total, short_rf.mem_bytes_total)

        self.assertAlmostEqual(rais_score.touched_expert_fraction(1, 256, 8), 8.0 / 256.0, places=9)
        self.assertGreater(rais_score.touched_expert_fraction(64, 256, 8), 0.85)
        self.assertAlmostEqual(rais_score.touched_expert_fraction(4096, 256, 8), 1.0, places=6)


class TestRaisScoringAndGates(unittest.TestCase):
    def test_flat_cap_penalty_on_unsupported_b32_b64(self) -> None:
        capped_rec = sample_valid_record()
        scaled_rec = sample_valid_record()
        scaled_rec["metrics"]["batch"]["32"] = {"aggregate_tok_s": 310.0}
        scaled_rec["metrics"]["batch"]["64"] = {"aggregate_tok_s": 420.0}

        res_capped = rais_score.score_model(capped_rec, "16x-gb10-1x100g")
        res_scaled = rais_score.score_model(scaled_rec, "16x-gb10-1x100g")

        p3b_cap = res_capped["submetrics"]["p3b_high_concurrency"]
        p3b_scl = res_scaled["submetrics"]["p3b_high_concurrency"]
        self.assertEqual(p3b_cap["bottleneck"], "capacity_cap")
        self.assertEqual(p3b_cap["points"]["B32"]["flat_capped_from_b"], 16)
        self.assertEqual(p3b_cap["points"]["B64"]["flat_capped_from_b"], 16)
        self.assertEqual(p3b_cap["points"]["B32"]["aggregate_tok_s"], 227.0)
        self.assertLess(p3b_cap["score"], p3b_scl["score"])
        self.assertLess(res_capped["raw_rais"], res_scaled["raw_rais"])

    def test_per_stream_sla_penalty_on_p3a(self) -> None:
        rec = sample_valid_record()
        rec["metrics"]["batch"]["16"] = {"aggregate_tok_s": 100.0}
        scored = rais_score.score_model(rec, "16x-gb10-1x100g")
        b16_info = scored["submetrics"]["p3a_interactive_batch"]["points"]["B16"]
        self.assertFalse(b16_info["sla_met"])
        self.assertLess(b16_info["score"], b16_info["raw_efficiency_pct"])

    def test_hard_binary_gates_invalidate_score(self) -> None:
        valid_res = rais_score.score_model(sample_valid_record(), "16x-gb10-1x100g")
        self.assertEqual(valid_res["status"], "VALID")
        self.assertEqual(valid_res["certified_rais"], valid_res["raw_rais"])

        for gate_patch, expected_tag in (
            ({"prose_exact": False}, "spec_exact_prose"),
            ({"code_exact": False}, "spec_exact_code"),
            ({"prefix_exact": False}, "prefix_exact_gx"),
            ({"errors": 2}, "errors=2"),
        ):
            rec = sample_valid_record()
            rec["gates"].update(gate_patch)
            res = rais_score.score_model(rec, "16x-gb10-1x100g")
            self.assertEqual(res["status"], "INVALID")
            self.assertIsNone(res["certified_rais"])
            self.assertIn(expected_tag, res["gates"]["failed_gates"])
            fleet = rais_score.score_fleet([rec], "16x-gb10-1x100g")
            self.assertEqual(fleet["fleet_status"], "INVALID")
            self.assertIsNone(fleet["fleet_rais_covered"])

    def test_speculation_scoring_and_base_roofline(self) -> None:
        rec = sample_valid_record()
        rec["metrics"]["speculation"] = {"alpha": 1.0}
        res_off = rais_score.score_model(rec, "16x-gb10-1x100g")
        self.assertEqual(res_off["pillars"]["P2"], 0.0)
        expected_r = round((res_off["pillars"]["P1"] + res_off["pillars"]["P3"] + res_off["pillars"]["P4"]) / 3.0, 2)
        self.assertAlmostEqual(res_off["base_roofline_r"], expected_r, places=2)

        rec_hi = sample_valid_record()
        rec_hi["metrics"]["speculation"] = {
            "alpha_prose": 4.0,
            "alpha_code": 4.0,
            "alpha_edit": 4.0,
            "verify_v_b1": 1.0,
            "verify_b1_enabled": True,
            "verify_v_b8": 2.0,
            "verify_b8_enabled": True,
        }
        res_hi = rais_score.score_model(rec_hi, "16x-gb10-1x100g")
        self.assertEqual(res_hi["submetrics"]["p2a_spec_acceptance"]["score"], 100.0)
        self.assertEqual(res_hi["submetrics"]["p2b_verify_b1"]["score"], 100.0)
        self.assertEqual(res_hi["submetrics"]["p2b_verify_b8"]["score"], 50.0)

    def test_cli_score_compare_and_import_hillclimb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            a_path = tmp_dir / "engine_a.jsonl"
            b_path = tmp_dir / "engine_b.jsonl"
            hc_path = tmp_dir / "hillclimb.jsonl"

            a_path.write_text(json.dumps(sample_valid_record("glm-5.3-flash", "vllm", 1.0)) + "\n", encoding="utf-8")
            b_path.write_text(json.dumps(sample_valid_record("glm-5.3-flash", "sparkpipe", 1.25)) + "\n", encoding="utf-8")
            hc_path.write_text(
                json.dumps({
                    "model": "kimi-k3",
                    "b1_nospec_tok_s": 16.2,
                    "b1_long_tok_s": 15.1,
                    "b1_long_ctx": 73000,
                    "b8_tok_s": 63.8,
                    "prefill_8k_tok_s": 590.0,
                }) + "\n",
                encoding="utf-8",
            )

            for fmt in ("markdown", "json", "html-data"):
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "tools" / "rais_score.py"), "score", str(b_path), "--hardware", "16x-gb10-1x100g", "--format", fmt],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if fmt in ("json", "html-data"):
                    parsed = json.loads(proc.stdout)
                    self.assertEqual(parsed["rais_version"], "1.1")
                else:
                    self.assertIn("ROI Priority Table", proc.stdout)

            cmp_proc = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "rais_score.py"), "compare", str(a_path), str(b_path), "--hardware", "16x-gb10-1x100g", "--format", "json"],
                capture_output=True,
                text=True,
                check=True,
            )
            cmp_data = json.loads(cmp_proc.stdout)
            self.assertGreater(cmp_data["delta_fleet_rais"], 0.0)
            self.assertGreater(cmp_data["models"][0]["delta_rais"], 0.0)

            imp_proc = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "rais_score.py"), "import-hillclimb", str(hc_path), "--score", "--format", "json"],
                capture_output=True,
                text=True,
                check=True,
            )
            imp_data = json.loads(imp_proc.stdout)
            self.assertEqual(imp_data["models"][0]["model"], "kimi-k3")
            self.assertGreater(imp_data["models"][0]["submetrics"]["p1a_short_b1"]["score"], 45.0)


class TestRaisBenchCli(unittest.TestCase):
    def test_parser_and_dry_run_tiers(self) -> None:
        parser = rais_bench.build_parser()
        args_fast = parser.parse_args([
            "--base-url", "http://127.0.0.1:8000",
            "--model", "glm-5.3-flash",
            "--tier", "fast",
            "--spec-toggle", "speculation",
            "--swap-url", "http://127.0.0.1:8001/swap",
            "--swap-token", "tok-test",
            "--dry-run",
        ])
        out_fast = rais_bench.run_benchmark(args_fast)
        self.assertTrue(out_fast["dry_run"])
        self.assertEqual(out_fast["plan"]["tier"], "fast")
        self.assertEqual(out_fast["endpoint"], "completions")
        self.assertTrue(out_fast["plan"]["run_b8_spec"])

        args_fast_nob8 = parser.parse_args([
            "--base-url", "http://127.0.0.1:8000",
            "--model", "glm-5.3-flash",
            "--tier", "fast",
            "--no-b8-spec",
            "--endpoint", "chat",
            "--dry-run",
        ])
        out_fast_nob8 = rais_bench.run_benchmark(args_fast_nob8)
        self.assertFalse(out_fast_nob8["plan"]["run_b8_spec"])
        self.assertEqual(out_fast_nob8["endpoint"], "chat")

        args_full = parser.parse_args([
            "--base-url", "http://127.0.0.1:8000",
            "--model", "kimi-k3.1-pro",
            "--tier", "full",
            "--b8-spec",
            "--endpoint", "auto",
            "--dry-run",
        ])
        out_full = rais_bench.run_benchmark(args_full)
        self.assertEqual(out_full["plan"]["tier"], "full")
        self.assertTrue(out_full["plan"]["run_b8_spec"])
        self.assertEqual(out_full["endpoint"], "completions")
        self.assertEqual(len(out_full["plan"]["prefill_long_targets"]), 2)
        self.assertEqual(len(out_full["plan"]["warm_targets"]), 2)

    def test_fast_tier_null_fields_score_cleanly(self) -> None:
        rec = sample_valid_record("deepseek-v4.1-flash", "sparkpipe", 1.0)
        rec["metrics"]["prefill_128k_tokens"] = None
        rec["metrics"]["prefill_128k_tok_s"] = None
        rec["metrics"]["warm_100k_prefix_tokens"] = None
        rec["metrics"]["warm_100k_ttft_ms"] = None
        rec["metrics"]["b1_long_ctx"] = None
        res = rais_score.score_model(rec, "16x-gb10-1x100g")
        self.assertEqual(res["status"], "VALID")
        self.assertGreater(res["certified_rais"], 0.0)

    def test_coalesced_sse_itl_and_engine_measurements(self) -> None:
        raw = {
            "started": 0.0,
            "http_status": 200,
            "token_times": [0.05] + [0.21] * 8 + [0.37] * 8,
            "event_times": [0.05, 0.21, 0.37],
            "event_token_counts": [1, 8, 8],
            "text": "hello world",
            "usage": {"prompt_tokens": 10, "completion_tokens": 17},
            "spec_stats": None,
            "error": None,
            "finish_reason": "length",
        }
        summary = rais_bench.summarize_stream(raw)
        self.assertAlmostEqual(summary["itl_p50_ms"], 20.0, places=2)
        self.assertAlmostEqual(summary["chunk_mean"], 17.0 / 3.0, places=2)

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "api.log"
            log_path.write_text(
                'noise line\n'
                '{"event":"engine_measurements","verify_lanes":12,"draft_tokens":60,"accepted_draft_tokens":42,"tree_frames":3,"tree_branch_frames":2,"tree_branch_accepted":1}\n',
                encoding="utf-8",
            )
            em = rais_bench.read_engine_measurements(str(log_path))
            self.assertIsNotNone(em)
            self.assertEqual(em["verify_lanes"], 12)
            self.assertEqual(em["accepted_draft_tokens"], 42)
            self.assertEqual(em["tree_branch_accepted"], 1)

    def test_stream_summary_and_capacity_detection(self) -> None:
        summary = rais_bench.summarize_stream({
            "http_status": 200,
            "error": None,
            "finish_reason": "length",
            "token_times": [0.10, 0.12, 0.14, 0.16, 0.21],
            "event_token_counts": [1, 2, 2],
            "text": "hello world",
            "usage": {"prompt_tokens": 128, "completion_tokens": 5},
            "spec_stats": None,
        })
        self.assertAlmostEqual(summary["ttft_ms"], 100.0, places=2)
        self.assertGreater(summary["decode_tok_s"], 30.0)
        self.assertGreater(summary["itl_p99_ms"], summary["itl_p50_ms"])
        self.assertTrue(rais_bench.is_capacity_rejection({"http_status": 429, "body": "busy"}))
        self.assertFalse(rais_bench.is_capacity_rejection({"http_status": 500, "body": "segfault"}))

    def test_edit_batch_prompt_and_chat_endpoint_fallback(self) -> None:
        p0 = rais_bench.format_batch_prompt(rais_bench.EDIT_PROMPT, 0, stream_prefix=True)
        p1 = rais_bench.format_batch_prompt(rais_bench.EDIT_PROMPT, 1, stream_prefix=True)
        self.assertNotEqual(p0, p1)
        self.assertTrue(p0.endswith("\nUpdated code:\n```c\n"))
        self.assertTrue(rais_bench.is_immediate_eos({
            "error": None,
            "http_status": 200,
            "token_times": [],
            "finish_reason": "stop",
        }))
        self.assertFalse(rais_bench.is_immediate_eos({
            "error": None,
            "http_status": 200,
            "token_times": [0.05],
            "finish_reason": "stop",
        }))

        seen_paths: list[str] = []
        seen_chat_has_messages: list[bool] = []

        class MockHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                seen_paths.append(self.path)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if self.path == "/v1/completions":
                    chunk = {"choices": [{"text": "", "finish_reason": "stop"}], "usage": {"prompt_tokens": 16, "completion_tokens": 0}}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                    return
                seen_chat_has_messages.append(self.path == "/v1/chat/completions" and "messages" in body)
                chunks = [
                    {"choices": [{"delta": {"reasoning_content": "plan "}}], "tokens": [1]},
                    {"choices": [{"delta": {"content": "typedef struct {"}}], "tokens": [2, 3]},
                    {"choices": [{"delta": {}, "finish_reason": "length"}], "usage": {"prompt_tokens": 24, "completion_tokens": 3}},
                ]
                for item in chunks:
                    self.wfile.write(f"data: {json.dumps(item)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            chat_res = rais_bench.stream_completion(url, {"model": "qwen3.8-flash", "prompt": "hi", "max_tokens": 4}, endpoint="chat")
            self.assertIsNone(chat_res["error"])
            self.assertEqual(chat_res["text"], "plan typedef struct {")
            self.assertEqual(len(chat_res["token_times"]), 3)
            self.assertTrue(seen_chat_has_messages and all(seen_chat_has_messages))

            comp_res = rais_bench.stream_completion(url, {"model": "qwen3.8-flash", "prompt": "hi", "max_tokens": 4}, endpoint="completions")
            self.assertTrue(rais_bench.is_immediate_eos(comp_res))

            parser = rais_bench.build_parser()
            args = parser.parse_args([
                "--base-url", url,
                "--model", "qwen3.8-flash",
                "--tier", "fast",
                "--max-batch", "4",
                "--spec-toggle", "none",
                "--no-b8-spec",
                "--endpoint", "auto",
            ])
            record = rais_bench.run_benchmark(args)
            self.assertEqual(record["endpoint"], "chat")
            self.assertEqual(record["endpoint_fallback"], "immediate_eos")
            self.assertEqual(record["gates"]["errors"], 0)
            self.assertIn("/v1/completions", seen_paths)
            self.assertIn("/v1/chat/completions", seen_paths)
        finally:
            server.shutdown()
            server.server_close()

    def test_effective_alpha_b8_sparse_and_full(self) -> None:
        self.assertAlmostEqual(rais_bench.compute_effective_alpha(16, 552, 768), 768.0 / 216.0, places=4)
        self.assertAlmostEqual(rais_bench.compute_effective_alpha(77, 558, 768), 768.0 / 210.0, places=4)
        self.assertAlmostEqual(rais_bench.compute_effective_alpha(96, 672, 768), 8.0, places=4)
        self.assertAlmostEqual(rais_bench.compute_effective_alpha(0, 0, 768), 1.0, places=4)
        self.assertAlmostEqual(rais_bench.compute_effective_alpha(16, 48, 0), 4.0, places=4)

    def test_synthetic_token_ids_and_tokenizer_bin_against_tokenizerless_server(self) -> None:
        seen_lengths: list[int] = []
        mutate_spec = {"enabled": False}

        class TokenizerlessHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if "prompt" in body or "prompt_token_ids" not in body:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":{"code":"tokenizer_unavailable"}}')
                    return
                ids = body["prompt_token_ids"]
                seen_lengths.append(len(ids))
                seed = sum((i + 1) * int(tok) for i, tok in enumerate(ids[:64])) % 10000
                out_toks = [2000 + ((seed + j * 13) % 5000) for j in range(4)]
                if mutate_spec["enabled"] and body.get("speculation") is True:
                    out_toks[-1] += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {"choices": [{"text": "", "finish_reason": None}], "tokens": out_toks[:2]},
                    {"choices": [{"text": "", "finish_reason": None}], "tokens": out_toks[2:]},
                    {"choices": [{"text": "", "finish_reason": "length"}], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out_toks)}, "tokens": []},
                ]
                for item in chunks:
                    self.wfile.write(f"data: {json.dumps(item)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TokenizerlessHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            parser = rais_bench.build_parser()
            args = parser.parse_args([
                "--base-url", url,
                "--model", "qwen3.8-27b",
                "--hardware", "1x-m5-ultra",
                "--tier", "fast",
                "--max-batch", "4",
                "--spec-toggle", "speculation",
                "--no-b8-spec",
                "--synthetic-token-ids",
            ])
            record = rais_bench.run_benchmark(args)
            self.assertEqual(record["prompt_mode"], "synthetic")
            self.assertEqual(record["gates"]["errors"], 0)
            self.assertTrue(record["gates"]["prose_exact"])
            self.assertTrue(record["gates"]["code_exact"])
            self.assertTrue(record["gates"]["prefix_exact"])
            self.assertIn(256, seen_lengths)
            self.assertIn(8192 + 128, seen_lengths)
            self.assertIn(32768, seen_lengths)
            scored = rais_score.score_model(record, "1x-m5-ultra")
            self.assertEqual(scored["status"], "VALID")

            mutate_spec["enabled"] = True
            bad_record = rais_bench.run_benchmark(args)
            self.assertFalse(bad_record["gates"]["prose_exact"])
            self.assertFalse(bad_record["gates"]["code_exact"])

            mutate_spec["enabled"] = False
            with tempfile.TemporaryDirectory() as tmp:
                tok_json = Path(tmp) / "tokenizer.json"
                tok_json.write_text("{}", encoding="utf-8")
                tok_bin = Path(tmp) / "fake_tokenize.py"
                tok_bin.write_text(
                    "#!/usr/bin/env python3\n"
                    "import sys\n"
                    "p = sys.argv[sys.argv.index('--prompt-file') + 1]\n"
                    "words = open(p, encoding='utf-8').read().split() or ['x']\n"
                    "for i, w in enumerate(words):\n"
                    "    print(1500 + (sum(ord(c) for c in w) + i) % 10000)\n",
                    encoding="utf-8",
                )
                tok_bin.chmod(0o755)
                args_tok = parser.parse_args([
                    "--base-url", url,
                    "--model", "qwen3.8-27b",
                    "--hardware", "1x-gb10",
                    "--tier", "fast",
                    "--max-batch", "4",
                    "--spec-toggle", "speculation",
                    "--no-b8-spec",
                    "--tokenizer-json", str(tok_json),
                    "--tokenizer-bin", str(tok_bin),
                ])
                tok_record = rais_bench.run_benchmark(args_tok)
                self.assertEqual(tok_record["prompt_mode"], "tokenizer")
                self.assertEqual(tok_record["gates"]["errors"], 0)
                self.assertTrue(tok_record["gates"]["prose_exact"])
                self.assertTrue(tok_record["gates"]["code_exact"])
                self.assertTrue(tok_record["gates"]["prefix_exact"])
        finally:
            server.shutdown()
            server.server_close()


class TestRaisDivisorReconstitution(unittest.TestCase):
    def test_reconstitution_preserves_index_continuity_and_chains(self) -> None:
        old_weights = {
            "kimi-k3": 0.4,
            "kimi-k3.1-pro": 0.6,
        }
        new_weights = {
            "kimi-k3.1-pro": 0.7,
            "deepseek-v4.1-flash": 0.3,
        }
        overlap = [
            sample_valid_record("kimi-k3", scale=1.3),
            sample_valid_record("kimi-k3.1-pro", scale=0.8),
            sample_valid_record("deepseek-v4.1-flash", scale=1.0),
        ]
        rec1 = rais_score.reconstitute_basket(
            overlap,
            "16x-gb10-1x100g",
            new_weights=new_weights,
            prev_weights=old_weights,
            prev_divisor=1.0,
            prev_basket_id="b0",
            new_basket_id="b1",
        )
        self.assertEqual(rec1["retired_models"], ["kimi-k3"])
        self.assertEqual(rec1["added_models"], ["deepseek-v4.1-flash"])
        self.assertIn("kimi-k3.1-pro", rec1["reweighted_models"])
        self.assertLess(rec1["continuity_gap"], 1e-6)
        self.assertAlmostEqual(rec1["index_before"], rec1["index_after"], places=4)

        rep_old = rais_score.score_fleet(overlap, "16x-gb10-1x100g", weights_override=old_weights, divisor=1.0, basket_id="b0")
        rep_new = rais_score.score_fleet(overlap, "16x-gb10-1x100g", weights_override=new_weights, divisor=rec1["new_divisor"], basket_id="b1")
        self.assertAlmostEqual(rep_old["fleet_rais_index"], rep_new["fleet_rais_index"], places=2)

        improved = [
            sample_valid_record("kimi-k3.1-pro", scale=1.0),
            sample_valid_record("deepseek-v4.1-flash", scale=1.2),
        ]
        rep_improved = rais_score.score_fleet(improved, "16x-gb10-1x100g", weights_override=new_weights, divisor=rec1["new_divisor"], basket_id="b1")
        self.assertGreater(rep_improved["fleet_rais_index"], rep_new["fleet_rais_index"])

        b2_weights = {
            "kimi-k3.1-pro": 0.5,
            "deepseek-v4.1-flash": 0.25,
            "glm-5.3": 0.25,
        }
        overlap2 = improved + [sample_valid_record("glm-5.3", scale=0.7)]
        rec2 = rais_score.reconstitute_basket(
            overlap2,
            "16x-gb10-1x100g",
            new_weights=b2_weights,
            prev_weights=new_weights,
            prev_divisor=rec1["new_divisor"],
            prev_basket_id="b1",
            new_basket_id="b2",
        )
        self.assertAlmostEqual(rec2["index_before"], rec2["index_after"], places=4)
        self.assertAlmostEqual(rec2["index_before"], rep_improved["fleet_rais_index"], places=1)

    def test_reconstitution_rejects_bad_weights_or_invalid_overlap(self) -> None:
        overlap = [sample_valid_record("kimi-k3"), sample_valid_record("kimi-k3.1-pro")]
        with self.assertRaises(ValueError):
            rais_score.reconstitute_basket(overlap, "16x-gb10-1x100g", new_weights={"kimi-k3.1-pro": 0.5}, prev_weights={"kimi-k3": 1.0})
        bad = sample_valid_record("kimi-k3.1-pro")
        bad["gates"]["prose_exact"] = False
        with self.assertRaises(ValueError):
            rais_score.reconstitute_basket([sample_valid_record("kimi-k3"), bad], "16x-gb10-1x100g", new_weights={"kimi-k3.1-pro": 1.0}, prev_weights={"kimi-k3": 1.0})

    def test_reconstitute_cli_and_score_with_weights_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            overlap_p = t / "overlap.jsonl"
            prev_w_p = t / "prev.json"
            new_w_p = t / "new.json"
            out_p = t / "reconstituted.json"
            records = [
                sample_valid_record("kimi-k3", scale=1.2),
                sample_valid_record("kimi-k3.1-pro", scale=0.9),
                sample_valid_record("glm-5.3-flash", scale=1.0),
            ]
            overlap_p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            prev_w_p.write_text(json.dumps({"basket_id": "b0", "divisor": 1.0, "weights": {"kimi-k3": 0.5, "kimi-k3.1-pro": 0.5}}), encoding="utf-8")
            new_w_p.write_text(json.dumps({"basket_id": "b1", "weights": {"kimi-k3.1-pro": 0.6, "glm-5.3-flash": 0.4}}), encoding="utf-8")
            rc = rais_score.main([
                "reconstitute", str(overlap_p),
                "--hardware", "16x-gb10-1x100g",
                "--prev-weights", str(prev_w_p),
                "--new-weights", str(new_w_p),
                "--output", str(out_p),
            ])
            self.assertEqual(rc, 0)
            res = json.loads(out_p.read_text(encoding="utf-8"))
            self.assertEqual(res["new_basket_id"], "b1")
            self.assertAlmostEqual(res["index_before"], res["index_after"], places=4)
            score_out = t / "scored.json"
            rc2 = rais_score.main([
                "score", str(overlap_p),
                "--hardware", "16x-gb10-1x100g",
                "--weights", str(out_p),
                "--divisor", str(res["new_divisor"]),
                "--basket-id", res["new_basket_id"],
                "--format", "json",
                "--output", str(score_out),
            ])
            self.assertEqual(rc2, 0)
            scored = json.loads(score_out.read_text(encoding="utf-8"))
            self.assertAlmostEqual(scored["fleet_rais_index"], res["index_after"], places=2)

    def test_latest_release_model_specs_and_reconstitution(self) -> None:
        hw = rais_score.resolve_hardware("16x-gb10-1x100g")
        for alias, canonical in (
            ("hy4_fp8", "hy4-fp8"),
            ("ling31_flash", "ling-3.1-flash"),
            ("ling32_air", "ling-3.2-air"),
            ("sage21_flash", "sage-s2.1-flash"),
            ("sage_s21_flash", "sage-s2.1-flash"),
        ):
            self.assertEqual(rais_score.resolve_model_name(alias), canonical)
            spec = rais_score.MODEL_SPECS[canonical]
            self.assertEqual(spec.fleet_weight, 0.0)
            rf = rais_score.compute_roofline(spec, hw, batch=1, ctx=256)
            self.assertGreater(rf.floor_tok_s, 10.0)
        overlap = [
            sample_valid_record("hy4-preview", scale=1.0),
            sample_valid_record("hy4-fp8", scale=1.05),
            sample_valid_record("ling-3.0-flash", scale=1.0),
            sample_valid_record("ling-3.1-flash", scale=0.85),
            sample_valid_record("ling-3.2-air", scale=1.15),
            sample_valid_record("glm-5.3-flash", scale=1.0),
            sample_valid_record("sage-s2.1-flash", scale=1.02),
        ]
        rec = rais_score.reconstitute_basket(
            overlap,
            "16x-gb10-1x100g",
            prev_weights={"hy4-preview": 0.4, "ling-3.0-flash": 0.3, "glm-5.3-flash": 0.3},
            new_weights={"hy4-fp8": 0.3, "ling-3.1-flash": 0.25, "ling-3.2-air": 0.2, "sage-s2.1-flash": 0.25},
        )
        self.assertAlmostEqual(rec["index_before"], rec["index_after"], places=4)

    def test_rais_history_rebuild_and_submetrics(self) -> None:
        import rais_history

        releases_dir = Path(__file__).resolve().parent.parent / "qualification" / "releases"
        snaps = rais_history.rebuild_history(releases_dir, hardware="16x-gb10-1x100g")
        self.assertGreaterEqual(len(snaps), 6)
        utcs = [s["utc"] for s in snaps]
        self.assertEqual(utcs, sorted(utcs))
        latest = snaps[-1]
        self.assertEqual(latest["rais_version"], "1.1")
        self.assertEqual(latest["fleet"]["status"], "VALID")
        self.assertEqual(latest["fleet"]["valid_models"], 19)
        self.assertEqual(len(latest["fleet"]["submetrics"]), 15)
        self.assertIn("P5", latest["fleet"]["pillars"])
        self.assertIn("kimi-k3.1-pro", latest["models"])
        self.assertEqual(len(latest["models"]["kimi-k3.1-pro"]["submetrics"]), 15)
        self.assertIn("P5", latest["models"]["kimi-k3.1-pro"]["pillars"])
        sanitized = rais_history.sanitize_site_snapshots(snaps)
        for snap in sanitized:
            self.assertNotIn("qualification/", snap["receipt"])
            self.assertIn("release ", snap["receipt"])


class TestRaisPillar5AndStackBadges(unittest.TestCase):
    def test_p5a_cold_boot_and_swap_floor(self) -> None:
        hw = rais_score.resolve_hardware("16x-gb10-1x100g")
        spec = rais_score.MODEL_SPECS["kimi-k3.1-pro"]
        expected_floor = max(1.0, (spec.total_weight_bytes / hw.ranks) / (hw.nvme_bw_gb_s_per_rank * 1e9))

        rec_none = sample_valid_record("kimi-k3.1-pro")
        res_none = rais_score.score_model(rec_none, "16x-gb10-1x100g")
        self.assertEqual(res_none["submetrics"]["p5a_cold_boot"]["score"], 0.0)
        self.assertAlmostEqual(res_none["submetrics"]["p5a_cold_boot"]["t_boot_floor_s"], expected_floor, places=2)

        rec_exact = sample_valid_record("kimi-k3.1-pro")
        rec_exact["metrics"]["cold_boot_s"] = expected_floor
        res_exact = rais_score.score_model(rec_exact, "16x-gb10-1x100g")
        self.assertEqual(res_exact["submetrics"]["p5a_cold_boot"]["score"], 100.0)

        rec_2x = sample_valid_record("kimi-k3.1-pro")
        rec_2x["metrics"]["swap_s"] = expected_floor * 2.0
        res_2x = rais_score.score_model(rec_2x, "16x-gb10-1x100g")
        self.assertAlmostEqual(res_2x["submetrics"]["p5a_cold_boot"]["score"], 50.0, places=1)

        rec_fast = sample_valid_record("kimi-k3.1-pro")
        rec_fast["metrics"]["cold_boot_s"] = expected_floor * 0.5
        res_fast = rais_score.score_model(rec_fast, "16x-gb10-1x100g")
        self.assertEqual(res_fast["submetrics"]["p5a_cold_boot"]["score"], 100.0)

    def test_p5b_usable_kv_memory_efficiency(self) -> None:
        hw = rais_score.resolve_hardware("1x-gb10")
        spec = rais_score.MODEL_SPECS["qwen3.8-27b"]
        weights_gib = spec.total_weight_bytes / float(1 << 30)
        expected_floor = max(1.0, hw.ram_gib_per_rank - weights_gib - hw.os_reserve_gib_per_rank)

        rec_none = sample_valid_record("qwen3.8-27b")
        res_none = rais_score.score_model(rec_none, "1x-gb10")
        self.assertEqual(res_none["submetrics"]["p5b_memory_efficiency"]["score"], 0.0)
        self.assertAlmostEqual(res_none["submetrics"]["p5b_memory_efficiency"]["m_free_floor_gib"], expected_floor, places=2)

        rec_half = sample_valid_record("qwen3.8-27b")
        rec_half["metrics"]["usable_kv_gib_per_rank"] = expected_floor * 0.5
        res_half = rais_score.score_model(rec_half, "1x-gb10")
        self.assertAlmostEqual(res_half["submetrics"]["p5b_memory_efficiency"]["score"], 50.0, places=1)

        rec_bytes = sample_valid_record("qwen3.8-27b")
        rec_bytes["metrics"]["kv_cache_bytes_per_rank"] = expected_floor * float(1 << 30)
        res_bytes = rais_score.score_model(rec_bytes, "1x-gb10")
        self.assertEqual(res_bytes["submetrics"]["p5b_memory_efficiency"]["score"], 100.0)

    def test_p5c_batch_invariance_and_p5_composite(self) -> None:
        rec_true = sample_valid_record("glm-5.3-flash")
        rec_true["metrics"]["batch_invariant_exact"] = True
        res_true = rais_score.score_model(rec_true, "16x-gb10-1x100g")
        self.assertEqual(res_true["submetrics"]["p5c_batch_invariance"]["score"], 100.0)
        self.assertEqual(res_true["pillars"]["P5"], 20.0)
        self.assertEqual(res_true["status"], "VALID")

        rec_false = sample_valid_record("glm-5.3-flash")
        rec_false["metrics"]["batch_invariant_exact"] = False
        res_false = rais_score.score_model(rec_false, "16x-gb10-1x100g")
        self.assertEqual(res_false["submetrics"]["p5c_batch_invariance"]["score"], 0.0)
        self.assertEqual(res_false["pillars"]["P5"], 0.0)
        self.assertEqual(res_false["status"], "VALID")
        self.assertIsNotNone(res_false["certified_rais"])
        self.assertAlmostEqual(res_true["raw_rais"] - res_false["raw_rais"], 2.0, places=2)
        self.assertEqual(res_true["base_roofline_r"], res_false["base_roofline_r"])

    def test_stack_badge_zero_score_impact_and_html_path_sanitization(self) -> None:
        rec_plain = sample_valid_record("qwen3.8-27b")
        rec_badge = sample_valid_record("qwen3.8-27b")
        rec_badge["stack_badge"] = "Zero-NVIDIA NVRM + SKL"
        s_plain = rais_score.score_model(rec_plain, "1x-gb10")
        s_badge = rais_score.score_model(rec_badge, "1x-gb10")
        self.assertEqual(s_plain["raw_rais"], s_badge["raw_rais"])
        self.assertEqual(s_plain["base_roofline_r"], s_badge["base_roofline_r"])
        self.assertEqual(s_plain["pillars"], s_badge["pillars"])
        self.assertEqual(s_badge["stack_badge"], "Zero-NVIDIA NVRM + SKL")

        fleet_jsonl = ROOT / "qualification" / "releases" / "dff7b5a96" / "rais-fleet-16x-gb10-1x100g.jsonl"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "rais_score.py"),
                "score",
                str(fleet_jsonl),
                "--hardware",
                "16x-gb10-1x100g",
                "--stack-badge",
                "CUDA 13.0",
                "--format",
                "html-data",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        for forbidden in ("qualification/", "tools/rais_", "docs/RAIS", "/srv/", "/home/", "/Users/"):
            self.assertNotIn(forbidden, proc.stdout)
        parsed = json.loads(proc.stdout)
        self.assertEqual(parsed["rais_version"], "1.1")
        self.assertEqual(parsed["fleet"]["receipt"], "release dff7b5a96 (16x-gb10-1x100g)")
        self.assertEqual(parsed["fleet"]["stack_badge"], "CUDA 13.0")
        self.assertEqual(len(parsed["submetrics_meta"]), 15)
        self.assertEqual(len(parsed["models"][0]["pillars"]), 5)

    def test_bench_p5_cli_and_live_batch_invariance_probe(self) -> None:
        state_lock = threading.Lock()
        in_flight = {"count": 0}
        perturb_batch = {"enabled": False}

        class BatchInvarianceHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                with state_lock:
                    in_flight["count"] += 1
                    concurrent_now = in_flight["count"]
                try:
                    import time
                    time.sleep(0.02)
                    with state_lock:
                        concurrent_now = max(concurrent_now, in_flight["count"])
                    prompt = body.get("prompt") or ""
                    out_text = f"det:{len(prompt)}:{prompt[:16]}"
                    if perturb_batch["enabled"] and concurrent_now > 1:
                        out_text += ":nondet"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    chunks = [
                        {"choices": [{"text": out_text[:10], "finish_reason": None}]},
                        {"choices": [{"text": out_text[10:], "finish_reason": "length"}], "usage": {"prompt_tokens": max(1, len(prompt) // 4), "completion_tokens": 2}},
                    ]
                    for item in chunks:
                        self.wfile.write(f"data: {json.dumps(item)}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                finally:
                    with state_lock:
                        in_flight["count"] -= 1

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), BatchInvarianceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            parser = rais_bench.build_parser()
            args = parser.parse_args([
                "--base-url", url,
                "--model", "qwen3.8-27b",
                "--hardware", "1x-gb10",
                "--tier", "fast",
                "--max-batch", "8",
                "--spec-toggle", "none",
                "--no-b8-spec",
                "--cold-boot-s", "10.8",
                "--usable-kv-gib", "52.0",
                "--stack-badge", "Zero-NVIDIA NVRM + SKL",
            ])
            rec_det = rais_bench.run_benchmark(args)
            self.assertEqual(rec_det["rais_version"], "1.1")
            self.assertEqual(rec_det["stack_badge"], "Zero-NVIDIA NVRM + SKL")
            self.assertEqual(rec_det["metrics"]["cold_boot_s"], 10.8)
            self.assertEqual(rec_det["metrics"]["usable_kv_gib_per_rank"], 52.0)
            self.assertTrue(rec_det["metrics"]["batch_invariant_exact"])
            scored_det = rais_score.score_model(rec_det, "1x-gb10")
            self.assertEqual(scored_det["status"], "VALID")
            self.assertEqual(scored_det["submetrics"]["p5c_batch_invariance"]["score"], 100.0)
            self.assertGreater(scored_det["pillars"]["P5"], 75.0)

            perturb_batch["enabled"] = True
            rec_nondet = rais_bench.run_benchmark(args)
            self.assertFalse(rec_nondet["metrics"]["batch_invariant_exact"])
            scored_nondet = rais_score.score_model(rec_nondet, "1x-gb10")
            self.assertEqual(scored_nondet["status"], "VALID")
            self.assertEqual(scored_nondet["submetrics"]["p5c_batch_invariance"]["score"], 0.0)
            self.assertAlmostEqual(scored_det["pillars"]["P5"] - scored_nondet["pillars"]["P5"], 20.0, places=2)
            rec_det_toggled = json.loads(json.dumps(rec_det))
            rec_det_toggled["metrics"]["batch_invariant_exact"] = False
            scored_det_toggled = rais_score.score_model(rec_det_toggled, "1x-gb10")
            self.assertAlmostEqual(scored_det["raw_rais"] - scored_det_toggled["raw_rais"], 2.0, places=2)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

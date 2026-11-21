#!/usr/bin/env python3
"""Gate GX against a fake engine with a prefix cache: an engine whose outputs depend only on the request passes every arm; one whose arithmetic follows its resume point fails reuse against cold; one that corrupts a restored chain fails restored against uninterrupted; a chain lost at eviction, a request from another client and a step not run each leave the verdict INCOMPLETE; a cold step on an uncleared cache stops and finishes after a clear; a step the engine dies under keeps its finished units and does only the rest when it is run again, and a restore that stopped has its park step redone only for the units still open; a request that fails on a healthy engine is recorded on its unit and the step goes on; a context the deployment cannot serve is skipped or fitted; each request's queue time, time to first token and decode rate, and each attempt's KV store and engine counters, come from the API log of the engine that is running."""
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gx_gate", ROOT / "tools/gx_gate.py")
gx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gx)


class FakeEngine:
    def __init__(self, follows_resume=False, loses_on_evict=False, corrupts_restored=False, log=None, dies_after=None, refuses=None):
        self.log = log
        self.follows_resume = follows_resume
        self.loses_on_evict = loses_on_evict
        self.corrupts_restored = corrupts_restored
        self.dies_after = dies_after
        self.refuses = refuses
        self.healthy = True
        self.chains = []
        self.restored = set()
        self.served = 0
        self.bodies = []
        self.lock = threading.Lock()

    def clear(self):
        self.chains = []
        self.restored = set()

    def evict(self):
        if self.loses_on_evict:
            self.chains = []
        self.restored = {tuple(chain) for chain in self.chains}

    def recover(self):
        self.healthy = True
        self.dies_after = None
        self.clear()

    def complete(self, body):
        prompt = body["prompt_token_ids"]
        with self.lock:
            if self.dies_after is not None and len(self.bodies) >= self.dies_after:
                self.healthy = False
            if not self.healthy:
                return 503, {"error": "engine down"}
            self.bodies.append({key: value for key, value in body.items() if key != "prompt_token_ids"})
            if self.refuses is not None and self.refuses(body):
                self.served += 1
                return 502, {"error": {"message": "the engine failed during the reply"}}
            cached, source = 0, None
            for chain in self.chains:
                common = 0
                while common < min(len(chain), len(prompt)) and chain[common] == prompt[common]:
                    common += 1
                common = min(common, len(prompt) - 1)
                if common > cached:
                    cached, source = common, tuple(chain)
            digest = hashlib.sha256(json.dumps(prompt).encode()).digest()
            tokens = [1000 + digest[index % 32] for index in range(body["max_tokens"])]
            shift = 0.0
            if self.follows_resume and cached:
                shift = 0.001
            if self.corrupts_restored and source in self.restored:
                tokens[-1] += 1
            self.chains.append(list(prompt) + tokens[:-1])
            self.served += 1
            if self.log:
                with open(self.log, "a") as log:
                    log.write("api " + json.dumps({"event": "request_measurements", "boot_pid": 7, "prompt_sha256": gx.prompt_digest(prompt), "accepted_ns": 1000000,
                                                   "first_dispatch_ns": 3000000, "stale_prefix_recomputes": 0, "adapter_id": "fake-adapter", "model_id": "fake", "model_revision": "r1",
                                                   "driver_program": "fake", "driver_artifact_sha256": "ab" * 32,
                                                   "tokens": [[token, 6000000 + index * 1000000] for index, token in enumerate(tokens)]}) + "\n")
                    for rank, pages in ((0, 3), (1, 4)):
                        log.write(json.dumps({"event": "kv_store_report", "boot_pid": 7, "rank": rank, "restore_pages": pages * self.served, "save_pages": 1, "restore_ns": 100 * (rank + 1)}) + "\n")
                    log.write(json.dumps({"event": "engine_measurements", "boot_pid": 7, "completed": self.served, "prefix_hits": 2}) + "\n")
        reply = {"object": "text_completion", "tokens": tokens, "finish_reason": "length",
                 "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(tokens), "prompt_tokens_details": {"cached_tokens": cached}}}
        if "logprobs" in body:
            reply["token_logprobs"] = [[[token, -0.5 - shift]] + [[token + 1 + rank, -2.0 - rank] for rank in range(body["logprobs"])] for token in tokens]
        return 200, reply


def serve(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if engine.healthy:
                self.answer(200, {"status": "ok", "ranks": 16, "connected_ranks": 16, "live_requests": 0, "served": engine.served})
            else:
                self.answer(503, {"status": "degraded", "ranks": 16, "connected_ranks": 15})

        def do_POST(self):
            self.answer(*engine.complete(json.loads(self.rfile.read(int(self.headers["Content-Length"])))))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def call(*arguments):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        status = gx.main(list(arguments))
    return status, output.getvalue()


class GxGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.state = str(Path(self.directory.name) / "state.json")
        self.receipt = str(Path(self.directory.name) / "receipt.json")
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            server.shutdown()
            server.server_close()
        self.directory.cleanup()

    def plan(self, *extra):
        return call("plan", "--state", self.state, "--model", "fake", "--release", "test", "--context-limit", "20000", "--contexts", "2048,4096",
                    "--declared-contexts", "1024,2048", "--drafted-contexts", "2048", "--tails", "1,40", "--max-tokens", "6", "--reply-tokens", "8", "--small-context", "2100", "--seed", "fixed", *extra)

    def start(self, engine):
        server = serve(engine)
        self.servers.append(server)
        status, listing = self.plan()
        self.assertEqual(status, 0, listing)
        return "http://127.0.0.1:%d" % server.server_address[1], [line.split()[:2] for line in listing.splitlines()]

    def step(self, api, name, *extra):
        return call("run", name, "--state", self.state, "--api", api, *extra)

    def verdict(self):
        status, table = call("verdict", "--state", self.state, "--output", self.receipt)
        return status, json.loads(Path(self.receipt).read_text()), table

    def run_gate(self, engine, skip=(), without_change=()):
        api, steps = self.start(engine)
        logged = ["--api-log", engine.log] if engine.log else []
        stopped = []
        for name, change in steps:
            if name in skip:
                continue
            if name not in without_change:
                getattr(engine, change, lambda: None)()
            status, output = self.step(api, name, *logged)
            if status != 0:
                stopped.append((name, output))
        status, receipt, table = self.verdict()
        return status, receipt, stopped, table

    def test_an_engine_that_follows_only_the_request_passes(self):
        engine = FakeEngine()
        status, receipt, stopped, table = self.run_gate(engine)
        self.assertEqual((status, receipt["verdict"], stopped), (0, "PASS", []), table)
        self.assertEqual(len(receipt["verdicts"]), 14)
        self.assertEqual({entry["case"] for entry in receipt["verdicts"]}, set(gx.CASES))
        self.assertEqual({(entry["nominal"], entry["group"]) for entry in receipt["verdicts"]}, {(1024, 2048), (2048, 2048), (4096, 4096)})
        for entry in receipt["verdicts"]:
            self.assertEqual((entry["reuse"], entry["restored"]), ("PASS", "PASS"), entry)
            self.assertEqual(entry["cached_tokens"]["cold"], 0)
            self.assertGreater(entry["cached_tokens"]["chain"], 0)
            self.assertGreater(entry["cached_tokens"]["restore"], 0)
        declared = [body for body in engine.bodies if "cache_points" in body]
        self.assertTrue(declared and all(body["cache_points"] in ([768], [1792]) and "logprobs" in body for body in declared))
        self.assertTrue(any("speculation" not in body for body in engine.bodies), "a drafted session's turns leave speculation at the engine's default")
        self.assertTrue(all(body.get("speculation") is False for body in engine.bodies if "logprobs" in body))
        chain = receipt["steps"]["chain"]["requests"]
        self.assertTrue(all(request["prefilled_tokens"] == request["prompt_tokens"] - request["cached_tokens"] and request["completion_tokens"] == request["max_tokens"] for request in chain))

    def test_the_steps_name_the_cache_change_each_needs(self):
        status, listing = self.plan()
        self.assertEqual(status, 0)
        self.assertEqual([line.split()[:3] for line in listing.splitlines()],
                         [["chain", "none", "open"], ["park:2048", "clear", "open"], ["restore:2048", "evict", "open"], ["park:4096", "none", "open"], ["restore:4096", "evict", "open"],
                          ["cold:1", "clear", "open"], ["cold:40", "clear", "open"]])
        self.assertEqual(listing.splitlines()[0].split()[3], "0/14")
        status, again = self.plan()
        self.assertEqual(status, 1)
        self.assertIn("planned once", again)

    def test_arithmetic_that_follows_the_resume_point_fails_reuse(self):
        status, receipt, stopped, table = self.run_gate(FakeEngine(follows_resume=True))
        self.assertEqual((status, receipt["verdict"]), (1, "FAIL"), table)
        entry = receipt["verdicts"][0]
        self.assertEqual(entry["reuse"], "FAIL")
        self.assertIsNone(entry["reuse_detail"]["first_token_difference"])
        self.assertEqual(entry["reuse_detail"]["first_logprob_difference"], 0)
        self.assertAlmostEqual(entry["reuse_detail"]["max_logprob_delta"], 0.001, places=6)
        self.assertEqual(entry["restored"], "PASS")

    def test_a_corrupted_restore_fails_restored_against_uninterrupted(self):
        status, receipt, stopped, table = self.run_gate(FakeEngine(corrupts_restored=True))
        self.assertEqual((status, receipt["verdict"]), (1, "FAIL"), table)
        entry = receipt["verdicts"][0]
        self.assertEqual((entry["reuse"], entry["restored"]), ("PASS", "FAIL"))
        self.assertEqual(entry["restored_detail"]["first_token_difference"], 5)

    def test_a_chain_lost_at_eviction_is_incomplete(self):
        status, receipt, stopped, table = self.run_gate(FakeEngine(loses_on_evict=True))
        self.assertEqual((status, receipt["verdict"]), (2, "INCOMPLETE"), table)
        self.assertEqual({entry["restored"] for entry in receipt["verdicts"] if entry["tail"] == 1}, {"LOST"})
        later = [entry for entry in receipt["verdicts"] if entry["tail"] == 40]
        self.assertEqual({entry["restored"] for entry in later}, {"PASS"})
        self.assertTrue(all(entry["recomputed_after_restore"] > 0 for entry in later if entry["case"] in gx.SESSION_CASES), later)

    def test_a_cold_step_on_an_uncleared_cache_stops_and_finishes_after_a_clear(self):
        engine = FakeEngine()
        status, receipt, stopped, table = self.run_gate(engine, without_change=("cold:1",))
        self.assertEqual((status, receipt["verdict"]), (2, "INCOMPLETE"), table)
        self.assertEqual([name for name, _ in stopped], ["cold:1"])
        self.assertIn("GX-STEP-STOPPED cold:1", stopped[0][1])
        self.assertIn("step cold:1 is stopped", table)
        self.assertIn("NOT-RUN", {entry["reuse"] for entry in receipt["verdicts"] if entry["tail"] == 1})
        self.assertEqual({entry["reuse"] for entry in receipt["verdicts"] if entry["tail"] == 40}, {"PASS"})
        engine.clear()
        api = "http://127.0.0.1:%d" % self.servers[0].server_address[1]
        self.assertEqual(self.step(api, "cold:1")[0], 0)
        status, receipt, table = self.verdict()
        self.assertEqual((status, receipt["verdict"]), (0, "PASS"), table)
        self.assertEqual(len(receipt["steps"]["cold:1"]["attempts"]), 2)

    def test_another_client_s_request_is_incomplete(self):
        engine = FakeEngine()
        original = engine.complete

        def intruding(body):
            if len(engine.bodies) == 3:
                engine.served += 1
            return original(body)
        engine.complete = intruding
        status, receipt, stopped, table = self.run_gate(engine)
        self.assertEqual((status, receipt["verdict"], receipt["alone"]), (2, "INCOMPLETE", False), table)
        self.assertEqual(receipt["steps"]["chain"]["attempts"][0]["foreign_requests"], 1)

    def test_a_step_not_run_is_incomplete(self):
        status, receipt, stopped, table = self.run_gate(FakeEngine(), skip=("park:4096", "restore:4096"))
        self.assertEqual((status, receipt["verdict"]), (2, "INCOMPLETE"), table)
        self.assertEqual({entry["restored"] for entry in receipt["verdicts"] if entry["nominal"] == 4096}, {"NOT-RUN"})
        self.assertIn("step park:4096 is open", table)

    def test_a_step_the_engine_dies_under_keeps_its_finished_units(self):
        engine = FakeEngine(dies_after=9)
        api, steps = self.start(engine)
        status, output = self.step(api, "chain")
        self.assertEqual(status, 1, output)
        self.assertIn("GX-STEP-STOPPED chain", output)
        state = json.loads(Path(self.state).read_text())
        kept = dict(state["steps"]["chain"]["results"])
        self.assertTrue(0 < len(kept) < 14, kept.keys())
        self.assertEqual(call("steps", "--state", self.state)[1].splitlines()[0].split()[2:], ["stopped", "%d/14" % len(kept)])
        engine.recover()
        status, output = self.step(api, "chain")
        self.assertEqual(status, 0, output)
        state = json.loads(Path(self.state).read_text())
        self.assertEqual(len(state["steps"]["chain"]["results"]), 14)
        for unit, result in kept.items():
            self.assertEqual(state["steps"]["chain"]["results"][unit], result)
        again = [request["label"] for request in state["steps"]["chain"]["requests"] if request["attempt"] == 2]
        self.assertEqual([label for label in again if label in kept], [])
        self.assertTrue(any(label.endswith("/warm") for label in again), "an open unit's prefix is warmed again on the cleared cache")
        for name, change in steps[1:]:
            getattr(engine, change, lambda: None)()
            self.assertEqual(self.step(api, name)[0], 0, name)
        status, receipt, table = self.verdict()
        self.assertEqual((status, receipt["verdict"]), (0, "PASS"), table)
        before = len(engine.bodies)
        self.assertTrue(self.step(api, "chain")[1].strip().endswith("chain is done"))
        self.assertEqual(len(engine.bodies), before)

    def test_an_attempt_the_engine_cut_short_reads_other_clients_from_its_log(self):
        log = str(Path(self.directory.name) / "api.log")
        for intruders, alone in ((0, True), (1, False)):
            Path(log).write_text(json.dumps({"event": "engine_measurements", "boot_pid": 7, "completed": 0}) + "\n")
            engine = FakeEngine(log=log, dies_after=9)
            original = engine.complete

            def intruding(body, engine=engine, original=original, intruders=intruders):
                if len(engine.bodies) == 6:
                    engine.served += intruders
                return original(body)
            engine.complete = intruding
            api, steps = self.start(engine)
            self.assertEqual(self.step(api, "chain", "--api-log", log)[0], 1)
            status, receipt, table = self.verdict()
            attempt = receipt["steps"]["chain"]["attempts"][0]
            self.assertNotIn("health_after", attempt)
            self.assertEqual((attempt["foreign_requests"], receipt["alone"]), (intruders, alone), table)
            Path(self.state).unlink()

    def test_a_restore_that_stopped_has_its_park_redone_for_the_open_units_only(self):
        engine = FakeEngine()
        api, steps = self.start(engine)
        self.assertEqual(self.step(api, "chain")[0], 0)
        engine.clear()
        self.assertEqual(self.step(api, "park:2048")[0], 0)
        engine.evict()
        engine.dies_after = len(engine.bodies) + 5
        status, output = self.step(api, "restore:2048")
        self.assertEqual(status, 1, output)
        state = json.loads(Path(self.state).read_text())
        restored = set(state["steps"]["restore:2048"]["results"])
        self.assertEqual(len(restored), 5)
        engine.recover()
        self.assertEqual(self.step(api, "park:2048")[0], 0)
        state = json.loads(Path(self.state).read_text())
        second = [request["label"] for request in state["steps"]["park:2048"]["requests"] if request["attempt"] == 2]
        replayed = {label.rsplit("/turn", 1)[0] for label in second if "/turn" in label}
        self.assertTrue(replayed and not replayed & restored, (replayed, restored))
        self.assertEqual([label for label in second if label.endswith("/warm")], ["declared/2048/warm"])
        engine.evict()
        self.assertEqual(self.step(api, "restore:2048")[0], 0)
        for name, change in steps[3:]:
            getattr(engine, change, lambda: None)()
            self.assertEqual(self.step(api, name)[0], 0, name)
        status, receipt, table = self.verdict()
        self.assertEqual((status, receipt["verdict"]), (0, "PASS"), table)

    def test_a_request_that_fails_on_a_healthy_engine_is_recorded_and_the_step_goes_on(self):
        target = {}
        engine = FakeEngine(refuses=lambda body: body.get("logprobs") is not None and len(body["prompt_token_ids"]) == target.get("length"))
        api, steps = self.start(engine)
        state = json.loads(Path(self.state).read_text())
        unit = next(unit for unit in state["units"] if unit["id"] == "prefix/4096/40")
        target["length"] = unit["context"] + unit["tail"]
        stopped = []
        for name, change in steps:
            getattr(engine, change, lambda: None)()
            if self.step(api, name)[0] != 0:
                stopped.append(name)
        self.assertEqual(stopped, [])
        status, receipt, table = self.verdict()
        self.assertEqual((status, receipt["verdict"], receipt["alone"]), (2, "INCOMPLETE", True), table)
        entry = next(entry for entry in receipt["verdicts"] if entry["id"] == "prefix/4096/40")
        self.assertEqual((entry["reuse"], entry["restored"]), ("ERROR", "ERROR"))
        self.assertIn("HTTP 502", entry["errors"]["chain"])
        self.assertIn("chain failed: HTTP 502", table)
        self.assertEqual({other["verdict"] for other in receipt["verdicts"] if other["id"] != "prefix/4096/40"}, {"PASS"})

    def test_a_prefix_that_does_not_warm_fails_its_units_only(self):
        target = {}
        engine = FakeEngine(refuses=lambda body: body["max_tokens"] == 1 and len(body["prompt_token_ids"]) == target.get("length"))
        api, steps = self.start(engine)
        state = json.loads(Path(self.state).read_text())
        target["length"] = next(unit for unit in state["units"] if unit["id"] == "declared/1024/1")["context"]
        self.assertEqual(self.step(api, "chain")[0], 0)
        state = json.loads(Path(self.state).read_text())
        errors = state["steps"]["chain"]["errors"]
        self.assertEqual(sorted(errors), ["declared/1024/1", "declared/1024/40"])
        self.assertTrue(all("did not warm" in message for message in errors.values()))
        self.assertEqual(len(state["steps"]["chain"]["results"]), 12)

    def test_a_long_request_is_followed_by_a_rest(self):
        engine = FakeEngine()
        api, steps = self.start(engine)
        rests = []
        original = gx.time.sleep
        gx.time.sleep = lambda seconds: rests.append(seconds)
        try:
            self.assertEqual(self.step(api, "chain", "--rest-seconds", "7", "--rest-over-seconds", "0")[0], 0)
        finally:
            gx.time.sleep = original
        self.assertEqual(rests.count(7.0), len(engine.bodies))

    def test_timing_and_counters_come_from_the_running_engine_s_log(self):
        log = str(Path(self.directory.name) / "api.log")
        with open(log, "w") as handle:
            handle.write(json.dumps({"event": "kv_store_report", "boot_pid": 5, "rank": 9, "restore_pages": 1000, "save_pages": 1000, "restore_ns": 50}) + "\n")
            handle.write("not json\n")
        status, receipt, stopped, table = self.run_gate(FakeEngine(log=log))
        self.assertEqual((status, receipt["verdict"], stopped), (0, "PASS", []), table)
        chain = receipt["steps"]["chain"]
        attempt = chain["attempts"][0]
        self.assertTrue(all((request["queue_ms"], request["first_token_ms"], request["stale_prefix_recomputes"]) == (2.0, 5.0, 0) for request in chain["requests"]), chain["requests"][:2])
        measured = [request for request in chain["requests"] if request["max_tokens"] == 6]
        self.assertTrue(measured and all((request["last_token_ms"], request["decode_tokens_per_second"]) == (10.0, 1000.0) for request in measured))
        self.assertTrue(all("decode_tokens_per_second" not in request for request in chain["requests"] if request["max_tokens"] == 1))
        self.assertEqual(attempt["engine"], {"adapter_id": "fake-adapter", "model_id": "fake", "model_revision": "r1", "driver_program": "fake", "driver_artifact_sha256": "ab" * 32})
        self.assertEqual((attempt["counters_before"]["boot_pid"], attempt["counters_before"]["ranks_reporting"]), (5, 1))
        self.assertEqual((attempt["counters_before"]["store"]["restore_pages"], attempt["counters_before"]["slowest_rank"]["restore_ns"]), (1000, 50))
        served = len(chain["requests"])
        self.assertEqual((attempt["counters_after"]["boot_pid"], attempt["counters_after"]["ranks_reporting"]), (7, 2))
        self.assertEqual((attempt["counters_after"]["store"]["restore_pages"], attempt["counters_after"]["store"]["save_pages"]), (7 * served, 2))
        self.assertEqual((attempt["counters_after"]["store"]["restore_ns"], attempt["counters_after"]["slowest_rank"]["restore_ns"]), (300, 200))
        self.assertEqual((attempt["counters_after"]["engine"]["completed"], attempt["counters_after"]["engine"]["prefix_hits"]), (served, 2))
        self.assertEqual(next(entry for entry in receipt["verdicts"] if entry["id"] == "prefix/2048/1")["first_token_ms"], {"chain": 5.0, "restore": 5.0, "cold": 5.0})

    def test_contexts_are_fitted_to_the_deployment(self):
        arguments = ["plan", "--state", self.state, "--model", "fake", "--release", "test", "--seed", "fixed"]
        status, listing = call(*arguments, "--context-limit", "32768")
        self.assertEqual(status, 0, listing)
        self.assertEqual([line.split()[0] for line in listing.splitlines()], ["chain", "park:100000", "restore:100000", "cold:1", "cold:40", "cold:500", "cold:5000"])
        state = json.loads(Path(self.state).read_text())
        self.assertEqual(sorted({(unit["nominal"], unit["context"]) for unit in state["units"] if unit["case"] == "prefix"}), [(8000, 8021), (100000, 16405)])
        self.assertEqual(sorted({(unit["nominal"], unit["context"]) for unit in state["units"] if unit["case"] == "declared"}), [(8000, 6913), (20000, 16129)])
        self.assertEqual({(entry["case"], entry["nominal"]) for entry in state["skipped"]}, {("prefix", 300000), ("session", 300000)})
        for unit in state["units"]:
            self.assertLessEqual(unit["context"] + 3 * 5000 + 2 * 96 + 32, 32768)
            if unit["case"] == "declared":
                self.assertEqual(((unit["context"] - 1) % 256, (unit["context"] - 1) % 1024), (0, 768))
            else:
                self.assertNotEqual(unit["context"] % 64, 0)
        Path(self.state).unlink()
        status, listing = call(*arguments, "--context-limit", "1048576")
        self.assertEqual(status, 0, listing)
        state = json.loads(Path(self.state).read_text())
        self.assertEqual(sorted({unit["context"] for unit in state["units"] if unit["case"] == "session"}), [8021, 100021, 300021])
        self.assertEqual([line.split()[0] for line in listing.splitlines() if line.startswith("park")], ["park:20000", "park:100000", "park:300000"])
        self.assertEqual(state["skipped"], [])
        self.assertEqual(state["options"]["reply_tokens"], 96)
        Path(self.state).unlink()
        status, listing = call(*arguments, "--context-limit", "16000")
        self.assertEqual(status, 1)
        self.assertIn("no room", listing)


if __name__ == "__main__":
    unittest.main()

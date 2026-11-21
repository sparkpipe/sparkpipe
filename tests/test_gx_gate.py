#!/usr/bin/env python3
"""Gate GX against a fake engine with a prefix cache: an engine whose outputs depend only on the request passes every arm; one whose arithmetic follows its resume point fails reuse against cold; one that corrupts a restored chain fails restored against uninterrupted; a chain lost at eviction, a cold step on an uncleared cache, a request from another client and a step not run each leave the verdict INCOMPLETE; a context the deployment cannot serve is skipped or fitted."""
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
    def __init__(self, follows_resume=False, loses_on_evict=False, corrupts_restored=False):
        self.follows_resume = follows_resume
        self.loses_on_evict = loses_on_evict
        self.corrupts_restored = corrupts_restored
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

    def complete(self, body):
        prompt = body["prompt_token_ids"]
        with self.lock:
            self.bodies.append({key: value for key, value in body.items() if key != "prompt_token_ids"})
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
        reply = {"object": "text_completion", "tokens": tokens, "finish_reason": "length",
                 "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(tokens), "prompt_tokens_details": {"cached_tokens": cached}}}
        if "logprobs" in body:
            reply["token_logprobs"] = [[[token, -0.5 - shift]] + [[token + 1 + rank, -2.0 - rank] for rank in range(body["logprobs"])] for token in tokens]
        return reply


def serve(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.answer({"status": "ok", "ranks": 16, "connected_ranks": 16, "live_requests": 0, "served": engine.served})

        def do_POST(self):
            self.answer(engine.complete(json.loads(self.rfile.read(int(self.headers["Content-Length"])))))

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

    def tearDown(self):
        self.directory.cleanup()

    def plan(self, *extra):
        return call("plan", "--state", self.state, "--model", "fake", "--release", "test", "--context-limit", "20000", "--contexts", "2048,4096",
                    "--declared-contexts", "1024,2048", "--drafted-contexts", "2048", "--tails", "1,40", "--max-tokens", "6", "--reply-tokens", "8", "--small-context", "2100", "--seed", "fixed", *extra)

    def run_gate(self, engine, skip=(), without_change=(), intrude=None):
        server = serve(engine)
        api = "http://127.0.0.1:%d" % server.server_address[1]
        try:
            status, listing = self.plan()
            self.assertEqual(status, 0, listing)
            failed = []
            for line in listing.splitlines():
                name, change, _ = line.split()
                if name in skip:
                    continue
                if name not in without_change:
                    getattr(engine, change, lambda: None)()
                if name == intrude:
                    engine.complete({"prompt_token_ids": [5, 6, 7], "max_tokens": 1})
                status, output = call("run", name, "--state", self.state, "--api", api)
                if status != 0:
                    failed.append((name, output))
            status, table = call("verdict", "--state", self.state, "--output", self.receipt)
            return status, json.loads(Path(self.receipt).read_text()), failed, table
        finally:
            server.shutdown()
            server.server_close()

    def test_an_engine_that_follows_only_the_request_passes(self):
        engine = FakeEngine()
        status, receipt, failed, table = self.run_gate(engine)
        self.assertEqual((status, receipt["verdict"], failed), (0, "PASS", []), table)
        self.assertEqual(len(receipt["verdicts"]), 14)
        self.assertEqual({(entry["nominal"], entry["group"]) for entry in receipt["verdicts"]}, {(1024, 2048), (2048, 2048), (4096, 4096)})
        self.assertEqual({entry["case"] for entry in receipt["verdicts"]}, set(gx.CASES))
        for entry in receipt["verdicts"]:
            self.assertEqual((entry["reuse"], entry["restored"]), ("PASS", "PASS"), entry)
            self.assertEqual(entry["cached_tokens"]["cold"], 0)
            self.assertGreater(entry["cached_tokens"]["chain"], 0)
            self.assertGreater(entry["cached_tokens"]["restore"], 0)
        declared = [body for body in engine.bodies if "cache_points" in body]
        self.assertTrue(declared and all(body["cache_points"] in ([768], [1792]) and "logprobs" in body for body in declared))
        self.assertTrue(any("speculation" not in body for body in engine.bodies), "a drafted session's turns leave speculation at the engine's default")
        self.assertTrue(all(body.get("speculation") is False for body in engine.bodies if "logprobs" in body))

    def test_the_steps_name_the_cache_change_each_needs(self):
        status, listing = self.plan()
        self.assertEqual(status, 0)
        self.assertEqual([line.split()[:2] for line in listing.splitlines()],
                         [["chain", "none"], ["park:2048", "clear"], ["restore:2048", "evict"], ["park:4096", "none"], ["restore:4096", "evict"], ["cold:1", "clear"], ["cold:40", "clear"]])
        status, again = self.plan()
        self.assertEqual(status, 1)
        self.assertIn("planned once", again)

    def test_arithmetic_that_follows_the_resume_point_fails_reuse(self):
        status, receipt, failed, table = self.run_gate(FakeEngine(follows_resume=True))
        self.assertEqual((status, receipt["verdict"]), (1, "FAIL"), table)
        entry = receipt["verdicts"][0]
        self.assertEqual(entry["reuse"], "FAIL")
        self.assertIsNone(entry["reuse_detail"]["first_token_difference"])
        self.assertEqual(entry["reuse_detail"]["first_logprob_difference"], 0)
        self.assertAlmostEqual(entry["reuse_detail"]["max_logprob_delta"], 0.001, places=6)
        self.assertEqual(entry["restored"], "PASS")

    def test_a_corrupted_restore_fails_restored_against_uninterrupted(self):
        status, receipt, failed, table = self.run_gate(FakeEngine(corrupts_restored=True))
        self.assertEqual((status, receipt["verdict"]), (1, "FAIL"), table)
        entry = receipt["verdicts"][0]
        self.assertEqual((entry["reuse"], entry["restored"]), ("PASS", "FAIL"))
        self.assertEqual(entry["restored_detail"]["first_token_difference"], 5)

    def test_a_chain_lost_at_eviction_is_incomplete(self):
        status, receipt, failed, table = self.run_gate(FakeEngine(loses_on_evict=True))
        self.assertEqual((status, receipt["verdict"]), (2, "INCOMPLETE"), table)
        self.assertEqual({entry["restored"] for entry in receipt["verdicts"] if entry["tail"] == 1}, {"LOST"})
        later = [entry for entry in receipt["verdicts"] if entry["tail"] == 40]
        self.assertEqual({entry["restored"] for entry in later}, {"PASS"})
        self.assertTrue(all(entry["recomputed_after_restore"] > 0 for entry in later if entry["case"] in gx.SESSION_CASES), later)

    def test_a_cold_step_on_an_uncleared_cache_stops_and_is_incomplete(self):
        status, receipt, failed, table = self.run_gate(FakeEngine(), without_change=("cold:1",))
        self.assertEqual((status, receipt["verdict"]), (2, "INCOMPLETE"), table)
        self.assertEqual([name for name, _ in failed], ["cold:1"])
        self.assertIn("GX-STEP-FAILED cold:1", failed[0][1])
        self.assertIn("NOT-RUN", {entry["reuse"] for entry in receipt["verdicts"] if entry["tail"] == 1})
        self.assertEqual({entry["reuse"] for entry in receipt["verdicts"] if entry["tail"] == 40}, {"PASS"})

    def test_another_client_s_request_is_incomplete(self):
        status, receipt, failed, table = self.run_gate(FakeEngine(), intrude="chain")
        self.assertEqual((status, receipt["verdict"], receipt["alone"]), (0, "PASS", True), table)
        engine = FakeEngine()
        original = engine.complete

        def intruding(body):
            if len(engine.bodies) == 3:
                engine.served += 1
            return original(body)
        engine.complete = intruding
        Path(self.state).unlink()
        status, receipt, failed, table = self.run_gate(engine)
        self.assertEqual((status, receipt["verdict"], receipt["alone"]), (2, "INCOMPLETE", False), table)

    def test_a_step_not_run_is_incomplete(self):
        status, receipt, failed, table = self.run_gate(FakeEngine(), skip=("park:4096", "restore:4096"))
        self.assertEqual((status, receipt["verdict"]), (2, "INCOMPLETE"), table)
        self.assertEqual({entry["restored"] for entry in receipt["verdicts"] if entry["nominal"] == 4096}, {"NOT-RUN"})

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
            self.assertLessEqual(unit["context"] + 3 * 5000 + 2 * 48 + 32, 32768)
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
        Path(self.state).unlink()
        status, listing = call(*arguments, "--context-limit", "16000")
        self.assertEqual(status, 1)
        self.assertIn("no room", listing)


if __name__ == "__main__":
    unittest.main()

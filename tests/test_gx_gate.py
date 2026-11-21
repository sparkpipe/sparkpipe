#!/usr/bin/env python3
"""Gate GX against a fake engine that resumes only at a wave end its request names (a cut, a stride state, a commit's end or one token before the prompt's end) and only where state was kept: an engine whose outputs depend only on the request passes every arm; a request resumed short of the cut its structure names is INCOMPLETE; one whose arithmetic follows its resume point fails reuse against cold; one that corrupts a restored chain fails restored against uninterrupted; a chain lost at eviction, a request from another client and a step not run each leave the verdict INCOMPLETE; a cold step on an uncleared cache stops and finishes after a clear; a step the engine dies under keeps its finished units and does only the rest when it is run again, and a restore that stopped has its park step redone only for the units still open; a request that fails on a healthy engine is recorded on its unit and the step goes on; a context the deployment cannot serve is skipped or fitted; each request's queue time, time to first token and decode rate, and each attempt's KV store and engine counters, come from the API log of the engine that is running."""
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gx_gate", ROOT / "tools/gx_gate.py")
gx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gx)
gx.RESTORE_SETTLE_SECONDS = 0.0


STRIDE = 1024


class FakeEngine:
    def __init__(self, follows_resume=False, loses_on_evict=False, corrupts_restored=False, log=None, dies_after=None, refuses=None, forgets_cuts=False, forgets_keeps=False,
                 follows_batch=False, watches_crowd=False):
        self.log = log
        self.follows_batch = follows_batch
        self.watches_crowd = watches_crowd
        self.present = []
        self.present_lock = threading.Lock()
        self.crowded = 0
        self.forgets_cuts = forgets_cuts
        self.forgets_keeps = forgets_keeps
        self.follows_resume = follows_resume
        self.loses_on_evict = loses_on_evict
        self.corrupts_restored = corrupts_restored
        self.dies_after = dies_after
        self.refuses = refuses
        self.healthy = True
        self.points = set()
        self.restored = set()
        self.served = 0
        self.bodies = []
        self.lock = threading.Lock()

    def clear(self):
        self.points = set()
        self.restored = set()

    def evict(self):
        if self.loses_on_evict:
            self.points = set()
        self.restored = set(self.points)

    def recover(self):
        self.healthy = True
        self.dies_after = None
        self.clear()

    def complete(self, body):
        prompt = body["prompt_token_ids"]
        mine = [False]
        with self.present_lock:
            for other in self.present:
                other[0] = mine[0] = True
            self.present.append(mine)
        deadline = time.monotonic() + (0.2 if self.watches_crowd and "logprobs" in body else 0.0)
        while not mine[0] and time.monotonic() < deadline:
            time.sleep(0.002)
        try:
            return self.answer(body, prompt, 2 if mine[0] else 1)
        finally:
            with self.present_lock:
                self.present.remove(mine)

    def answer(self, body, prompt, together):
        with self.lock:
            self.crowded = max(self.crowded, together)
            if self.dies_after is not None and len(self.bodies) >= self.dies_after:
                self.healthy = False
            if not self.healthy:
                return 503, {"error": "engine down"}
            self.bodies.append({key: value for key, value in body.items() if key != "prompt_token_ids"})
            if self.refuses is not None and self.refuses(body):
                self.served += 1
                return 502, {"error": {"message": "the engine failed during the reply"}}
            commit = body.get("commit_only") is True
            if commit and ("max_tokens" in body or "logprobs" in body):
                return 400, {"error": {"code": "commit_only_conflict"}}
            if (self.forgets_cuts and "cuts" in body) or (self.forgets_keeps and "keep" in body):
                body = {key: value for key, value in body.items() if key not in (("cuts",) if self.forgets_cuts else ()) + (("keep",) if self.forgets_keeps else ())}
            final = len(prompt) if commit else len(prompt) - 1
            strides = list(range(STRIDE, len(prompt), STRIDE))
            ends = sorted(set([cut for cut in body.get("cuts", []) if cut < len(prompt)] + strides + [final]))
            kept = set(strides) | set(body.get("keep", [])) | {final}
            cached, source = 0, None
            for end in ends:
                key = (hashlib.sha256(json.dumps(prompt[:end]).encode()).hexdigest(), tuple(cut for cut in ends if cut <= end))
                if end < len(prompt) and key in self.points:
                    cached, source = end, key
                if end in kept:
                    self.points.add(key)
            digest = hashlib.sha256(json.dumps(prompt).encode()).digest()
            tokens = [1000 + digest[index % 32] for index in range(0 if commit else body["max_tokens"])]
            shift = 0.0
            if self.follows_resume and cached:
                shift = 0.001
            if self.follows_batch and together > 1:
                shift += 0.002
            if self.corrupts_restored and source in self.restored and tokens:
                tokens[-1] += 1
            self.served += 1
            if self.log:
                with open(self.log, "a") as log:
                    log.write("api " + json.dumps({"event": "request_measurements", "boot_pid": 7, "request_id": 100000 + self.served, "prompt_sha256": gx.prompt_digest(prompt), "accepted_ns": 1000000,
                                                   "first_dispatch_ns": 3000000, "stale_prefix_recomputes": 0, "adapter_id": "fake-adapter", "model_id": "fake", "model_revision": "r1",
                                                   "driver_program": "fake", "driver_artifact_sha256": "ab" * 32,
                                                   "tokens": [[token, 6000000 + index * 1000000] for index, token in enumerate(tokens)]}) + "\n")
                    for rank, pages in ((0, 3), (1, 4)):
                        log.write(json.dumps({"event": "kv_store_report", "boot_pid": 7, "rank": rank, "restore_pages": pages * self.served, "save_pages": 1, "restore_ns": 100 * (rank + 1),
                                              "pool_physical_pages": 10, "pool_device_bytes": 10 << 20}) + "\n")
                    log.write(json.dumps({"event": "engine_measurements", "boot_pid": 7, "completed": self.served, "prefix_hits": 2}) + "\n")
        reply = {"object": "text_completion", "tokens": tokens, "finish_reason": "commit" if commit else "length",
                 "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(tokens), "prompt_tokens_details": {"cached_tokens": cached}}}
        if commit:
            reply["usage"]["published_tokens"] = len(prompt)
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

        def event(self, payload):
            self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
            self.wfile.flush()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            status, reply = engine.complete(body)
            if status != 200 or not body.get("stream"):
                self.answer(status, reply)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            tokens, lists, usage = reply["tokens"], reply.get("token_logprobs"), reply["usage"]
            for start in range(0, len(tokens), 3):
                event = {"object": "text_completion", "choices": [{"index": 0, "text": "", "finish_reason": None}], "tokens": tokens[start:start + 3]}
                if lists is not None:
                    event["token_logprobs"] = lists[start:start + 3]
                if start == 0:
                    event["usage"] = {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": len(event["tokens"]), "prompt_tokens_details": usage["prompt_tokens_details"]}
                self.event(event)
            self.event({"object": "text_completion", "choices": [{"index": 0, "text": "", "finish_reason": reply["finish_reason"]}], "usage": usage, "tokens": []})
            self.wfile.write(b"data: [DONE]\n\n")

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
                    "--kept-contexts", "1024,2048", "--drafted-contexts", "2048", "--tails", "1,40", "--max-tokens", "6", "--reply-tokens", "8", "--small-context", "2100", "--seed", "fixed",
                    "--served-tokens", "5", *extra)

    def start(self, engine, *plan):
        server = serve(engine)
        self.servers.append(server)
        status, listing = self.plan(*plan)
        self.assertEqual(status, 0, listing)
        return "http://127.0.0.1:%d" % server.server_address[1], [line.split()[:2] for line in listing.splitlines()]

    def step(self, api, name, *extra):
        return call("run", name, "--state", self.state, "--api", api, *extra)

    def verdict(self):
        status, table = call("verdict", "--state", self.state, "--output", self.receipt)
        return status, json.loads(Path(self.receipt).read_text()), table

    def run_gate(self, engine, skip=(), without_change=(), plan=()):
        api, steps = self.start(engine, *plan)
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
        for entry in receipt["verdicts"]:
            self.assertEqual((entry["cached_tokens"]["chain"], entry["cached_tokens"]["restore"]), (entry["resume_cut"], entry["resume_cut"]), entry)
        commits = [body for body in engine.bodies if body.get("commit_only")]
        self.assertTrue(commits and all("max_tokens" not in body and "logprobs" not in body and "cuts" not in body for body in commits), "a prefix is warmed by a commit, which generates nothing")
        kept = [body for body in engine.bodies if "keep" in body]
        self.assertTrue(kept and all(body["keep"] in ([769], [1793]) and body["cuts"] == body["keep"] for body in kept), "a kept unit asks for state at the cut where its shared prefix ends")
        self.assertTrue(all("cuts" in body for body in engine.bodies if not body.get("commit_only")), "every request that generates names its cuts")
        self.assertTrue(all(body["cuts"] == sorted(set(body["cuts"])) for body in engine.bodies if "cuts" in body))
        last = max((body for body in engine.bodies if "logprobs" in body), key=lambda body: len(body["cuts"]))
        self.assertEqual(len(last["cuts"]), 5, "a session's last prompt names where its opening ends, one token before each reply began and where each reply ends")
        self.assertTrue(any("speculation" not in body for body in engine.bodies), "a drafted session's turns leave speculation at the engine's default")
        self.assertTrue(all(body.get("speculation") is False for body in engine.bodies if "logprobs" in body))
        chain = receipt["steps"]["chain"]["requests"]
        self.assertTrue(all(request["prefilled_tokens"] == request["prompt_tokens"] - request["cached_tokens"] and request["completion_tokens"] == request["max_tokens"] for request in chain))
        self.assertTrue(all(body.get("stream") is True for body in engine.bodies), "every request is streamed, as the chat layer sends them")
        self.assertTrue(all(request["sent_ns"] <= request["end_ns"] and request["started_utc"] <= request["finished_utc"] and request["started_utc"].endswith("Z") and
                            (request["first_ns"] is None) == (request["max_tokens"] == 0) and request["rested_seconds"] == 0 for request in chain), chain[:2])
        served = receipt["steps"]["chain"]["served"]
        self.assertEqual(sorted(served), ["2048", "4096"], "each context's prefix units are followed by one reply as served")
        self.assertTrue(all(len(result["tokens"]) == 5 and result["cached_tokens"] == result["prompt_tokens"] - 1 and "logprobs" not in result for result in served.values()), served)
        timing = [body for body in engine.bodies if body.get("max_tokens") == 5]
        self.assertTrue(len(timing) == 2 and all("speculation" not in body and "logprobs" not in body and len(body["cuts"]) == 1 for body in timing), timing)
        self.assertEqual(receipt["block"], [])

    def test_a_request_resumed_short_of_its_cut_is_incomplete(self):
        status, receipt, stopped, table = self.run_gate(FakeEngine(forgets_keeps=True))
        self.assertEqual((status, receipt["verdict"], stopped), (2, "INCOMPLETE", []), table)
        reuse = {entry["id"]: entry for entry in receipt["verdicts"]}
        self.assertEqual({name: entry["reuse"] for name, entry in reuse.items() if entry["reuse"] != "PASS"}, {"kept/1024/1": "NOT-CACHED", "kept/2048/1": "SHORT"}, table)
        short = reuse["kept/2048/1"]
        self.assertTrue(short["cached_tokens"]["chain"] == 1024 < short["resume_cut"] == 1793 and short["reuse_detail"]["equal"], short)
        self.assertEqual(reuse["kept/2048/40"]["cached_tokens"]["chain"], 1793, "the one-token unit's own last cut is the shared cut, so the next unit resumes there")
        self.assertIn("SHORT", table)

    def test_requests_sent_together_are_compared_with_the_same_request_alone(self):
        for follows_batch, counts in ((False, {"equal": 4, "different": 0, "not_run": 0}), (True, {"equal": 0, "different": 4, "not_run": 0})):
            engine = FakeEngine(follows_batch=follows_batch, watches_crowd=True)
            status, receipt, stopped, table = self.run_gate(engine, plan=("--crowd", "2", "--cases", "prefix"))
            self.assertEqual((status, receipt["verdict"], stopped), (0, "PASS", []), table)
            self.assertEqual(receipt["concurrent"], counts, table)
            self.assertEqual(engine.crowded, 2, "the crowd step sends two requests at a time")
            crowd = receipt["steps"]["crowd"]
            self.assertEqual(sorted(crowd["results"]), ["prefix/2048/1", "prefix/2048/40", "prefix/4096/1", "prefix/4096/40"])
            self.assertEqual([request["label"] for request in crowd["requests"] if request["max_tokens"] == 0], ["prefix/2048/warm", "prefix/4096/warm"])
            together = [entry for entry in receipt["verdicts"] if "concurrent" in entry]
            self.assertTrue(all(entry["cached_tokens"]["crowd"] == entry["resume_cut"] for entry in together), together)
            self.assertIn("concurrent equal=%d different=%d not-run=0" % (counts["equal"], counts["different"]), table)
            if follows_batch:
                self.assertIn("sent together it differs from sent alone", table)
                self.assertAlmostEqual(together[0]["concurrent_detail"]["max_logprob_delta"], 0.002, places=6)
            Path(self.state).unlink()
        status, listing = self.plan("--crowd", "2")
        self.assertEqual([line.split()[:2] for line in listing.splitlines()][-1], ["crowd", "clear"])
        Path(self.state).unlink()
        with self.assertRaises(SystemExit):
            self.plan("--crowd", "1")

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
        self.assertEqual([label for label in second if label.endswith("/warm")], ["kept/2048/warm"])
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
        engine = FakeEngine(refuses=lambda body: body.get("max_tokens") == 1 and len(body["prompt_token_ids"]) == target.get("length"))
        api, steps = self.start(engine)
        state = json.loads(Path(self.state).read_text())
        target["length"] = next(unit for unit in state["units"] if unit["id"] == "kept/1024/1")["context"] + gx.KEPT_OTHER
        self.assertEqual(self.step(api, "chain")[0], 0)
        state = json.loads(Path(self.state).read_text())
        errors = state["steps"]["chain"]["errors"]
        self.assertEqual(sorted(errors), ["kept/1024/1", "kept/1024/40"])
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
        state = json.loads(Path(self.state).read_text())
        self.assertTrue(all(request["rested_seconds"] == 7.0 for request in state["steps"]["chain"]["requests"]))
        self.assertEqual(state["steps"]["chain"]["attempts"][0]["rest"], {"seconds": 7.0, "over_seconds": 0.0})

    def test_timing_and_counters_come_from_the_running_engine_s_log(self):
        log = str(Path(self.directory.name) / "api.log")
        with open(log, "w") as handle:
            handle.write(json.dumps({"event": "kv_store_report", "boot_pid": 5, "rank": 9, "restore_pages": 1000, "save_pages": 1000, "restore_ns": 50}) + "\n")
            handle.write("not json\n")
        status, receipt, stopped, table = self.run_gate(FakeEngine(log=log))
        self.assertEqual((status, receipt["verdict"], stopped), (0, "PASS", []), table)
        chain = receipt["steps"]["chain"]
        attempt = chain["attempts"][0]
        self.assertTrue(all((request["queue_ms"], request.get("first_token_ms", 5.0), request["stale_prefix_recomputes"]) == (2.0, 5.0, 0) for request in chain["requests"]), chain["requests"][:2])
        measured = [request for request in chain["requests"] if request["max_tokens"] == 6]
        self.assertTrue(measured and all((request["last_token_ms"], request["decode_tokens_per_second"]) == (10.0, 1000.0) for request in measured))
        self.assertTrue(all("decode_tokens_per_second" not in request for request in chain["requests"] if request["max_tokens"] <= 1))
        self.assertTrue(all("first_token_ms" not in request and request["finish_reason"] == "commit" and request["completion_tokens"] == 0 for request in chain["requests"] if request["max_tokens"] == 0))
        self.assertEqual(attempt["engine"], {"adapter_id": "fake-adapter", "model_id": "fake", "model_revision": "r1", "driver_program": "fake", "driver_artifact_sha256": "ab" * 32})
        self.assertEqual((attempt["counters_before"]["boot_pid"], attempt["counters_before"]["ranks_reporting"]), (5, 1))
        self.assertEqual((attempt["counters_before"]["store"]["restore_pages"], attempt["counters_before"]["slowest_rank"]["restore_ns"]), (1000, 50))
        served = len(chain["requests"])
        self.assertEqual((attempt["counters_after"]["boot_pid"], attempt["counters_after"]["ranks_reporting"]), (7, 2))
        self.assertEqual((attempt["counters_after"]["store"]["restore_pages"], attempt["counters_after"]["store"]["save_pages"]), (7 * served, 2))
        self.assertEqual((attempt["counters_after"]["store"]["restore_ns"], attempt["counters_after"]["slowest_rank"]["restore_ns"]), (300, 200))
        self.assertEqual((attempt["counters_after"]["engine"]["completed"], attempt["counters_after"]["engine"]["prefix_hits"]), (served, 2))
        self.assertEqual(next(entry for entry in receipt["verdicts"] if entry["id"] == "prefix/2048/1")["first_token_ms"], {"chain": 5.0, "restore": 5.0, "cold": 5.0})
        self.assertTrue(all(request["boot_pid"] == 7 and request["request_id"] > 100000 and request["accepted_ns"] == 1000000 and request["first_dispatch_ns"] == 3000000
                            for request in chain["requests"]), chain["requests"][:1])
        self.assertTrue(all((request["first_ready_ns"], request["last_ready_ns"]) == (6000000, 6000000 + 1000000 * (request["completion_tokens"] - 1))
                            for request in chain["requests"] if request["completion_tokens"]))
        self.assertEqual((attempt["rest"], attempt["log"]["path"]), ({"seconds": 0.0, "over_seconds": 60.0}, log))
        copy = Path(self.directory.name) / attempt["log"]["copy"]
        self.assertEqual(attempt["log"]["copy"], "chain.1.apilog")
        self.assertEqual(copy.read_bytes(), Path(log).read_bytes()[attempt["log"]["start"]:attempt["log"]["end"]])
        self.assertEqual(copy.read_text().count('"request_measurements"'), served)
        self.assertEqual(receipt["steps"]["restore:2048"]["attempts"][0]["log"]["copy"], "restore-2048.1.apilog")
        restored = receipt["steps"]["restore:2048"]["results"]
        self.assertTrue(all(result["restore"] == {"pages": 7, "rank_seconds_sum": 0.0, "slowest_rank_seconds": 0.0, "bytes": 7 << 20, "misses": 0} for result in restored.values()),
                        list(restored.values())[0].get("restore"))

    def test_notes_and_the_tool_s_own_checks(self):
        log = str(Path(self.directory.name) / "api.log")
        Path(log).write_text("")
        engine = FakeEngine(log=log)
        api, steps = self.start(engine, "--cases", "prefix", "--contexts", "2048", "--tails", "40")
        self.assertEqual([name for name, _ in steps], ["chain", "park:2048", "restore:2048", "cold:40"])
        self.assertEqual(call("note", "--state", self.state, "active_before=qwen3.8-max", "lane=stage-lanes/fake/lane0")[0], 0)
        self.assertEqual(call("note", "--state", self.state, "novalue")[0], 1)
        status, output = call("check", "--state", self.state)
        self.assertEqual(status, 1)
        self.assertIn("check: step chain is open", output)
        self.assertEqual(self.step(api, "chain", "--api-log", log)[0], 0)
        engine.clear()
        self.assertEqual(self.step(api, "cold:40", "--api-log", log)[0], 0)
        status, output = call("check", "--state", self.state)
        self.assertEqual((status, output.strip()), (0, "check: ok"))
        status, receipt, table = self.verdict()
        self.assertEqual([{key: value for key, value in note.items() if key != "at"} for note in receipt["block"]], [{"active_before": "qwen3.8-max", "lane": "stage-lanes/fake/lane0"}])
        self.assertEqual(receipt["verdict"], "INCOMPLETE", "a check of two steps is not a verdict")
        Path(self.state).unlink()
        api, steps = self.start(FakeEngine(), "--cases", "prefix", "--contexts", "2048", "--tails", "40")
        self.assertEqual(self.step(api, "chain")[0], 0)
        status, output = call("check", "--state", self.state, "--steps", "chain")
        self.assertEqual(status, 1)
        self.assertIn("has no record in the API log", output)
        self.assertIn("reuse is NOT-RUN", output)
        Path(self.state).unlink()
        for engine, expected, text in ((FakeEngine(forgets_keeps=True), 1, "check: kept/2048/1 reuse is SHORT"), (FakeEngine(follows_resume=True), 0, "check: finding: prefix/2048/40 differs between reuse and cold")):
            api, steps = self.start(engine, "--cases", "prefix,kept", "--contexts", "2048", "--kept-contexts", "2048", "--tails", "1,40")
            self.assertEqual(self.step(api, "chain")[0], 0)
            for tail in ("1", "40"):
                engine.clear()
                self.assertEqual(self.step(api, "cold:" + tail)[0], 0)
            status, output = call("check", "--state", self.state, "--no-log")
            self.assertEqual(status, expected, output)
            self.assertIn(text, output)
            Path(self.state).unlink()

    def test_contexts_are_fitted_to_the_deployment(self):
        arguments = ["plan", "--state", self.state, "--model", "fake", "--release", "test", "--seed", "fixed"]
        status, listing = call(*arguments, "--context-limit", "32768")
        self.assertEqual(status, 0, listing)
        self.assertEqual([line.split()[0] for line in listing.splitlines()], ["chain", "park:100000", "restore:100000", "cold:1", "cold:40", "cold:500", "cold:5000"])
        state = json.loads(Path(self.state).read_text())
        self.assertEqual(sorted({(unit["nominal"], unit["context"]) for unit in state["units"] if unit["case"] == "prefix"}), [(8000, 8021), (100000, 16405)])
        self.assertEqual(sorted({(unit["nominal"], unit["context"]) for unit in state["units"] if unit["case"] == "kept"}), [(8000, 6913), (20000, 16129)])
        self.assertEqual({(entry["case"], entry["nominal"]) for entry in state["skipped"]}, {("prefix", 300000), ("session", 300000)})
        for unit in state["units"]:
            self.assertLessEqual(unit["context"] + 3 * 5000 + 2 * 96 + 32, 32768)
            if unit["case"] == "kept":
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

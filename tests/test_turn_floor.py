#!/usr/bin/env python3
"""tools/turn_floor.py times an engine's turns by token ids and splits each first token's time into send, queue, restore, the prefill waves and the one-row wave of the last prompt token, using only the engine's own log. This test serves a stub engine that logs as release c2b714343 does (waves over every prompt token but the last, the last token as a wave of one row that yields the first token, then chains whose tokens share one stamp), keeps its own record of each of those times, runs every arm against it and checks that the split recovers them. The stub's sleeps set only a lower bound: a host that sleeps long changes what the stub records, never whether the split agrees with it."""
import hashlib
import http.server
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "turn_floor.py"
RECORDED = ROOT / "tests" / "fixtures" / "turn_floor_c2b714343"
RECORDED_LOG_BASE = 120651989
spec = importlib.util.spec_from_file_location("turn_floor", TOOL)
turn_floor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(turn_floor)

WAVE_ROWS = 512
WAVE_MS = 30.0
WAVE_GAP_MS = 2.0
STEP_MS = 8.0
CHAIN_TOKENS = 8
EARLIER_SAVES = 5
QUEUE_MS = 3.0
RESTORE_MS = 120.0
CONTEXT_LIMIT = 9000
MAX_CUTS = 16384
MAX_PROMPT_TOKENS = 1048576
BOOT = 4242
PART_SLACK_MS = 1.0
EARLY_MS = 1.0
CLOCK_APART_MS = 5.0
CLOCK_SHARED_MS = 0.5
ENGINE_PARTS = ("queue_ms", "restore_ms", "prefill_ms", "first_step_ms")

failures = []


def check(condition, what):
    if not condition:
        failures.append(what)


def near(value, expected, slack):
    return value is not None and abs(value - expected) <= slack


def rising(counts):
    return (isinstance(counts, list) and len(counts) <= MAX_CUTS and all(isinstance(count, int) and 0 < count <= MAX_PROMPT_TOKENS for count in counts) and
            all(left < right for left, right in zip(counts, counts[1:])))


class Stub:
    def __init__(self, log_path):
        self.log = open(log_path, "a")
        self.lock = threading.Lock()
        self.busy = threading.Lock()
        self.seen = []
        self.next_id = 100000
        self.restore_next = False
        self.store = {"save_pages": EARLIER_SAVES, "save_ns": EARLIER_SAVES * 1000000, "restore_pages": 0, "restore_ns": 0}
        self.requests = []
        self.truth = {}
        self.refuse_in = 0
        self.restored_ms = 0.0
        self.answers_state = True
        self.state_cuts = []
        self.unordered = 0

    def cached(self, ids):
        best = 0
        for earlier in self.seen:
            shared = 0
            for left, right in zip(earlier, ids):
                if left != right:
                    break
                shared += 1
            best = max(best, min(shared, len(ids) - 1))
        return best

    def report_store(self):
        self.write(json.dumps(dict({"event": "kv_store_report", "boot_pid": BOOT, "rank": 0, "residentd_pid": 7}, **self.store), separators=(",", ":")))

    def write(self, line):
        with self.lock:
            self.log.write(line + "\n")
            self.log.flush()

    @staticmethod
    def token(request_id, position):
        return 5000 + (request_id * 31 + position * 7) % 1000

    def step(self, request_id, kind, rows, steps, wall, context, gap):
        self.write(f"batch_step kind={kind} request={request_id} rows={rows} steps={steps} drafts=0 trunk=0 tree=0 kept={rows if kind == 'prefill' else steps} match=0 "
                   f"class=0 expect=0.00 predict_ms=0.00 service_ms={max(wall - 0.2, 0.0):.2f} wall_ms={wall:.2f} context={context} gap_ms={gap:.2f} draft_wait_ms=0.00")

    def serve(self, body, emit):
        ids, budget = body["prompt_token_ids"], body["max_tokens"]
        with self.lock:
            self.next_id += 1
            request_id = self.next_id
            cached = self.cached(ids)
            restore, self.restore_next = self.restore_next, False
        accepted = time.monotonic_ns()
        self.requests.append({"priority": body.get("priority"), "speculation": body.get("speculation"), "prompt": len(ids), "cached": cached,
                              "cuts": body.get("cuts") or [], "keep": body.get("keep") or []})
        with self.busy:
            time.sleep(QUEUE_MS / 1000)
            self.restored_ms = 0.0
            if restore:
                self.write(f"batch_rejected status=18 kind=1 submission=7 request={request_id}")
                time.sleep(RESTORE_MS / 1000)
                self.restored_ms = (time.monotonic_ns() - accepted) / 1e6
                self.store["restore_pages"] += cached // 64
                self.store["restore_ns"] += int(RESTORE_MS * 1e6)
                self.report_store()
            dispatched = time.monotonic_ns()
            rows_left, done, first = len(ids) - 1 - cached, cached, True
            while rows_left > 0:
                rows = min(rows_left, WAVE_ROWS)
                if not first:
                    time.sleep(WAVE_GAP_MS / 1000)
                began = time.monotonic_ns()
                time.sleep(WAVE_MS / 1000)
                wall = (time.monotonic_ns() - began) / 1e6
                done += rows
                self.step(request_id, "prefill", rows, 1, wall, done, 0.0 if first else WAVE_GAP_MS)
                rows_left -= rows
                first = False
            if not first:
                time.sleep(WAVE_GAP_MS / 1000)
            began = time.monotonic_ns()
            time.sleep(STEP_MS / 1000)
            tokens = [[self.token(request_id, 0), time.monotonic_ns()]]
            self.step(request_id, "prefill", 1, 1, (time.monotonic_ns() - began) / 1e6, len(ids), 0.0 if first else WAVE_GAP_MS)
            emit([tokens[0][0]], cached, None)
            while len(tokens) < budget:
                count = min(budget - len(tokens), CHAIN_TOKENS)
                chain_began = time.monotonic_ns()
                time.sleep(count * STEP_MS / 1000)
                ready = time.monotonic_ns()
                chain = [self.token(request_id, len(tokens) + index) for index in range(count)]
                tokens += [[token, ready] for token in chain]
                self.step(request_id, "chain", 1, count, (time.monotonic_ns() - chain_began) / 1e6, len(ids) + len(tokens), 0.1)
                emit(chain, cached, None)
        emit([], cached, "length")
        with self.lock:
            self.seen.append(list(ids) + [token for token, _ in tokens][:-1])
        self.store["save_pages"] += 1
        self.store["save_ns"] += 1000000
        self.report_store()
        wait = (dispatched - accepted) / 1e6
        spread = (tokens[-1][1] - tokens[0][1]) / 1e9
        self.truth.setdefault(turn_floor.prompt_sha256(ids), []).append({
            "restore_ms": wait if restore else 0.0, "queue_ms": 0.0 if restore else wait, "prefill_ms": (began - dispatched) / 1e6,
            "first_step_ms": (tokens[0][1] - began) / 1e6, "decode_tok_s": (len(tokens) - 1) / spread if spread > 0 else None})
        self.write(json.dumps({"event": "request_measurements", "boot_pid": BOOT, "request_id": request_id, "status": 0, "accepted_ns": accepted,
                               "first_dispatch_ns": dispatched, "stale_prefix_recomputes": 0, "prompt_tokens": len(ids), "cached_prompt_tokens": cached,
                               "prompt_sha256": turn_floor.prompt_sha256(ids), "tokens": tokens}, separators=(",", ":")))


def handler(stub):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *arguments):
            pass

        def reply(self, document, status=200):
            data = json.dumps(document).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply({"object": "list", "data": [{"id": "stub", "max_model_len": CONTEXT_LIMIT}]})
            else:
                self.reply({"status": "ok", "live_requests": 0})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not all(rising(body.get(name, [])) for name in ("cuts", "keep")):
                with stub.lock:
                    stub.unordered += 1
                self.reply({"error": {"message": "cuts and keep must be arrays of at most 16384 rising token counts", "type": "invalid_request_error",
                                      "code": "invalid_option"}}, 400)
                return
            if self.path == "/v1/cache/state":
                if not stub.answers_state:
                    self.send_error(404)
                    return
                stub.state_cuts.append(body.get("cuts") or [])
                held = stub.cached(body["prompt_token_ids"] + [0])
                self.reply({"object": "cache.state", "complete": True, "prompt_tokens": len(body["prompt_token_ids"]), "matched_tokens": held, "resume_tokens": held,
                            "probed_tokens": held, "ranks": 1, "ranks_reported": 1, "ranks_failed": 0, "held_tokens": held, "resident_tokens": held, "saved_tokens": 0,
                            "state_tokens": held, "read_tokens": held, "boot": str(BOOT)})
                return
            with stub.lock:
                stub.refuse_in -= 1
                refused = stub.refuse_in == 0
            if refused:
                data = b'{"error":{"message":"refused by the stub"}}'
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(tokens, cached, finish):
                event = {"object": "text_completion", "choices": [{"index": 0, "text": "", "finish_reason": finish}], "tokens": tokens,
                         "usage": {"prompt_tokens": len(body["prompt_token_ids"]), "prompt_tokens_details": {"cached_tokens": cached}}}
                if finish is not None:
                    event["cache"] = {"matched_tokens": cached + 3, "resumed_tokens": cached, "restore_ms": stub.restored_ms, "background_wait_ms": 0.0,
                                      "boot": str(BOOT), "artifact": "0123456789abcdef"}
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()

            stub.serve(body, emit)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return Handler


def run_tool(*arguments, fails=False):
    done = subprocess.run([sys.executable, str(TOOL), *arguments], capture_output=True, text=True, timeout=300)
    if (done.returncode != 0) != fails:
        raise SystemExit(f"FAIL test_turn_floor: turn_floor {' '.join(arguments[:2])} exited {done.returncode}: {done.stderr[-600:]}")
    return done.stdout


def clocks_apart_ms():
    child = subprocess.Popen([sys.executable, "-c", "import sys,time\nfor line in sys.stdin:\n    print(time.monotonic_ns(), flush=True)"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    apart = []
    for _ in range(5):
        before = time.monotonic_ns()
        child.stdin.write("now\n")
        child.stdin.flush()
        theirs = int(child.stdout.readline())
        after = time.monotonic_ns()
        apart.append(max(before - theirs, theirs - after, 0) / 1e6)
    child.stdin.close()
    child.wait(timeout=30)
    return min(apart)


def measured(digest, request_id, accepted, first_ready):
    return json.dumps({"event": "request_measurements", "boot_pid": BOOT, "request_id": request_id, "status": 0, "accepted_ns": accepted,
                       "first_dispatch_ns": accepted + 1000, "prompt_tokens": 3, "cached_prompt_tokens": 0, "prompt_sha256": digest,
                       "tokens": [[7, first_ready]]}, separators=(",", ":"))


def clock_guard(folder, sent_late):
    folder.mkdir()
    log_path = folder / "engine.log"
    digests = [turn_floor.prompt_sha256([1, 2, 3]), turn_floor.prompt_sha256([4, 5, 6])]
    log_path.write_text(measured(digests[0], 5, 2000000, 8000000) + "\n" + measured(digests[1], 6, 12000000, 18000000) + "\n")
    records = [{"kind": "mark", "what": "start", "api_log": str(log_path), "log_offset": 0, "model": "stub", "label": "guard"},
               {"kind": "turn", "prompt_sha256": digests[0], "sent_ns": 1000000, "first_ns": 9000000, "prompt_tokens": 3, "cached_tokens": 0},
               {"kind": "turn", "prompt_sha256": digests[1], "sent_ns": 13000000 if sent_late else 11000000, "first_ns": 19000000, "prompt_tokens": 3, "cached_tokens": 0}]
    (folder / "requests.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    rows, _, _, shared = turn_floor.join(folder)
    return rows, shared


check(turn_floor.prompt_sha256([1, 2, 3]) == hashlib.sha256(bytes([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0])).hexdigest(),
      "a prompt's digest is the SHA-256 of its token ids as little-endian 32-bit words, the engine's own digest")

rows, _, _, shared = turn_floor.join(RECORDED, RECORDED / "engine.log", RECORDED_LOG_BASE)
cold = rows[0]
check(len(rows) == 1 and shared and cold["request_id"] == 100002 and cold["prefill_waves"] == 10 and cold["prefill_rows"] == 19999 and cold["wave_rows_max"] == 2048 and
      cold["first_step_ms"] == 101.25 and cold["first_step_service_ms"] == 92.35 and round(cold["prefill_ms"]) == 31726 and round(cold["between_waves_ms"], 1) == 3.2 and
      round(cold["queue_ms"], 2) == 0.64 and cold["restore_ms"] == 0 and round(cold["send_ms"], 2) == 1.61 and round(cold["reply_ms"], 2) == 0.28 and
      round(cold["decode_tok_s"], 2) == 11.99 and cold["decode_step_kind"] == "chain" and cold["decode_step_tokens"] == 8,
      f"a request recorded on release c2b714343 splits as its log says: ten waves over 19,999 rows, then the last token's one-row wave of 101.25 ms: {cold}")

with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    rows, shared = clock_guard(folder / "together", False)
    check(shared and [row["send_ms"] for row in rows] == [1.0, 1.0] and [row["reply_ms"] for row in rows] == [1.0, 1.0] and all(row["queue_ms"] == 0.001 for row in rows),
          f"with one clock, send is the client's stamp to the engine's acceptance and reply is the engine's first token to the client's: {rows}")
    rows, shared = clock_guard(folder / "apart", True)
    check(not shared and all(row["send_ms"] is None and row["reply_ms"] is None for row in rows) and all(row["queue_ms"] == 0.001 for row in rows),
          f"a request the engine accepted before the client sent it shows two clocks: every send and reply time is left out and the engine's own parts stay: {rows}")
    log_path = folder / "engine.log"
    log_path.write_text("an earlier line\n")
    tokens_path = folder / "tokens.txt"
    tokens_path.write_text("\n".join(str(1000 + (index * 7919) % 20000) for index in range(60000)))
    stub = Stub(log_path)
    stub.report_store()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler(stub))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    output = folder / "run"
    common = ("run", "--endpoint", endpoint, "--tokens", str(tokens_path), "--output", str(output), "--label", "stub", "--api-log", str(log_path),
              "--contexts", "2000,4000", "--tails", "50,300", "--repeats", "2", "--spec-repeats", "1", "--once-from", "300", "--reply-tokens", "6",
              "--decode-tokens", "12", "--served-tokens", "10", "--wave-tokens", "6000", "--wave-lead", "0.02", "--probes-solo", "3")
    stub.refuse_in = 11
    log_start = log_path.stat().st_size
    run_tool(*common, "--arms", "turns", fails=True)
    state = json.loads((output / "session.json").read_text())
    check(state["levels"] == [2000] and state["level"] == {"context": 4000, "done": 3},
          f"a run that a refused request stops keeps the levels and the steps of the open level it finished: {state['levels']} {state.get('level')}")
    interrupted_history = len(state["history"])
    run_tool(*common, "--arms", "turns")
    state = json.loads((output / "session.json").read_text())
    check(state["levels"] == [2000, 4000], f"the turns arm finishes every context level that fits the engine: {state['levels']}")
    check(len(state["history"]) > 4000, "the session keeps its prompt and every reply as its history")
    run_tool(*common, "--arms", "truth", "--event", "after the turns")
    stub.answers_state = False
    run_tool(*common, "--arms", "truth", "--event", "an engine without the endpoint")
    stub.answers_state = True
    run_tool(*common, "--arms", "openings,wave")
    stub.restore_next = True
    run_tool(*common, "--arms", "resume", "--event", "swap")
    table = run_tool("parts", str(output))
    tables = run_tool("report", str(output))
    rows = [json.loads(line) for line in (output / "parts.jsonl").read_text().splitlines()]
    summary = json.loads((output / "summary.json").read_text())
    cut = json.loads(run_tool("excerpt", str(output), "--output", str(folder / "excerpt.log")))
    (folder / "excerpt.log").write_text((folder / "excerpt.log").read_text() + "an engine line of a later run\n")
    again = folder / "again"
    again.mkdir()
    (again / "requests.jsonl").write_text((output / "requests.jsonl").read_text())
    run_tool("parts", str(again), "--api-log", str(folder / "excerpt.log"), "--log-base", str(cut["log_base"]))
    recut = [json.loads(line) for line in (again / "parts.jsonl").read_text().splitlines()]
    excerpt_summary = json.loads((again / "summary.json").read_text())
    apart_ms = clocks_apart_ms()
    session_requests, session_questions = len(stub.requests), len(stub.state_cuts)
    short = folder / "short"
    run_tool("run", "--endpoint", endpoint, "--tokens", str(tokens_path), "--output", str(short), "--label", "stub", "--api-log", str(log_path), "--contexts", "500",
             "--tails", "1", "--repeats", "3", "--spec-repeats", "0", "--reply-tokens", "6", "--decode-tokens", "12", "--served-tokens", "10", "--arms", "turns,truth")
    short_turns = [json.loads(line) for line in (short / "requests.jsonl").read_text().splitlines()]
    short_requests, short_questions = stub.requests[session_requests:], stub.state_cuts[session_questions:]
    del stub.requests[session_requests:]
    server.shutdown()

check(summary["unmatched"] == 0, f"every request finds its own line in the engine log by prompt digest: {summary['unmatched']} unmatched")
sent = {}
for request in stub.requests:
    sent.setdefault(request["prompt"], []).append(request)
named, wrong, before_resume = [], [], None
for row in [row for row in rows if row.get("error") is None]:
    wanted = [cut for cut in named if cut < row["prompt_tokens"]]
    if row["kind"] in ("resume", "turn_after_resume") and before_resume is None:
        before_resume = list(named)
    if row["kind"] in ("grow", "turn", "turn_spec", "resume", "turn_after_resume", "decode", "decode_served"):
        if any(request["cuts"] != wanted or request["keep"] for request in sent[row["prompt_tokens"]]) or row["cut_count"] != len(wanted):
            wrong.append((row["kind"], row["prompt_tokens"], wanted[-2:], [request["cuts"][-2:] for request in sent[row["prompt_tokens"]]]))
    if row["kind"] in ("grow", "turn", "turn_spec", "resume", "turn_after_resume"):
        named += [row["prompt_tokens"] - 1, row["prompt_tokens"] + row["output_tokens"]]
last_turn = [row for row in rows if row["kind"] in ("grow", "turn", "turn_spec") and row.get("error") is None][-1]
check(not wrong and len(named) >= 24 and named == sorted(named),
      f"every session request names, for each earlier reply, one token before that turn's prompt ended and where its reply ended: {wrong[:3]}")
whole = last_turn["prompt_tokens"] + last_turn["output_tokens"]
check(before_resume[-2:] == [last_turn["prompt_tokens"] - 1, whole] and stub.state_cuts[0] == [cut for cut in before_resume if cut < whole] and
      stub.state_cuts[0][-1] == last_turn["prompt_tokens"] - 1,
      f"the question to the engine about the session names the cuts a request on the whole session would, ending one token before its last prompt ended: {stub.state_cuts[0][-4:]}")
opened = [request for request in stub.requests if request["prompt"] == CONTEXT_LIMIT - 6 - 64]
shared, brief = CONTEXT_LIMIT - 6 - 64 - 500, CONTEXT_LIMIT - 6 - 64 - int((CONTEXT_LIMIT - 6 - 64) * 0.03)
check([request["cuts"] for request in opened] == [[shared], [shared], [brief], [brief]] and all(request["keep"] == request["cuts"] for request in opened),
      f"an opening and the second request on it name where what they share ends, and ask for state there: {[(request['cuts'], request['keep']) for request in opened]}")
turned = [row for row in short_turns if row["kind"] in ("grow", "turn")]
check(stub.unordered == 0 and all(row.get("error") is None for row in short_turns) and [row["prompt_tokens"] for row in turned] == [500, 507, 514, 521] and
      [request["cuts"] for request in short_requests] == [[], [499, 506], [499, 506, 513], [499, 506, 513, 520], [499, 506, 513, 520], [499, 506, 513, 520]] and
      short_questions == [[499, 506, 513, 520]],
      f"with a one-token tail a reply ends one token before the next prompt does, and a request names that count once, so its cuts still rise: "
      f"{stub.unordered} refused, {[request['cuts'] for request in short_requests]} {short_questions}")
asked = summary["cache_states"]
check(len(asked) == 2 and asked[0]["event"] == "after the turns" and asked[0]["http_status"] == 200 and asked[0]["answer"]["ranks_failed"] == 0 and
      asked[0]["answer"]["held_tokens"] == asked[0]["history_tokens"] == asked[0]["answer"]["prompt_tokens"] and asked[0]["answer"]["resident_tokens"] == asked[0]["history_tokens"] and
      asked[1]["http_status"] == 404,
      f"the truth arm asks the engine what it holds of the session and keeps the answer, and an engine without the endpoint is recorded, not fatal: {asked}")
timed = ("queue_ms", "restore_ms", "prefill_ms", "first_step_ms", "prefill_waves", "prefill_rows", "decode_tok_s", "restore_polls", "request_id")
check(cut["log_start"] == log_start and cut["log_base"] < cut["log_start"] and cut["baseline_reports"] == 1 and cut["kept"] < cut["lines"] and len(recut) == len(rows) and
      all({name: one.get(name) for name in timed} == {name: other.get(name) for name in timed} for one, other in zip(rows, recut)) and
      excerpt_summary["kv_store"] == summary["kv_store"],
      f"an excerpt of the engine log, read with the byte it starts at, gives every request the same parts and the run the same store counters: {cut}")
refused = [row for row in rows if row.get("error") is not None]
check(len(refused) == 1 and refused[0]["kind"] == "turn" and refused[0]["context"] == 4000 and refused[0]["tail"] == 50 and "request_id" not in refused[0] and
      not refused[0].get("unmatched"), f"the refused request stays in the record with its error and takes no line of the engine's log: {refused}")
rows = [row for row in rows if row.get("error") is None]
taken = {}
for row in rows:
    recorded = stub.truth.get(row["prompt_sha256"], [])
    index = taken.get(row["prompt_sha256"], 0)
    taken[row["prompt_sha256"]] = index + 1
    row["truth"] = recorded[index] if index < len(recorded) else None
check(all(row["truth"] is not None for row in rows), "every request has the stub's own record of its times")
rows = [row for row in rows if row["truth"] is not None]
off = [(row["kind"], name, round(row.get(name) if row.get(name) is not None else -1.0, 2), round(row["truth"][name], 2)) for row in rows for name in ENGINE_PARTS
       if not near(row.get(name), row["truth"][name], PART_SLACK_MS)]
check(not off, f"queue, restore, prefill and the first decode step come out of the engine's log as the stub recorded them: {off[:6]}")
rates = [(row["kind"], row.get("decode_tok_s"), row["truth"]["decode_tok_s"]) for row in rows if row.get("decode_tok_s") is not None]
check(rates and all(recorded is not None and abs(found - recorded) <= 0.01 * recorded for _, found, recorded in rates),
      f"the decode rate comes from the engine's token times: {rates[:4]}")
turns = [row for row in rows if row["kind"] == "turn"]
check(sorted((row["context"], row["tail"]) for row in turns) == [(2000, 50), (2000, 50), (2000, 300), (4000, 50), (4000, 50), (4000, 300)],
      f"each level runs its short tail twice and its long tail once, and a resumed level repeats nothing it had finished: {[(row['context'], row['tail']) for row in turns]}")
again = [row for row in rows if row.get("first_of_run")]
check(len(again) == 1 and again[0]["kind"] == "turn" and again[0]["context"] == 4000 and again[0]["tail"] == 50 and again[0]["prompt_tokens"] == interrupted_history + 50,
      f"the resumed run sends the refused turn again from the saved history and marks it as the first of its run: {[(row['kind'], row['prompt_tokens']) for row in again]}")
counted = next(item for item in summary["summary"] if item["kind"] == "turn" and item["context"] == 4000 and item["tail"] == 50)
check(counted["requests"] == 1, f"a first request of a resumed run is left out of the level's figures: {counted['requests']}")
spec = [row for row in rows if row["kind"] == "turn_spec"]
check(all(row["speculation"] is False for row in turns) and len(spec) == 2 and all(row["speculation"] is None and row["tail"] == 50 for row in spec),
      "plain turns turn speculation off and the as-served turns, short tails only, leave it to the engine")
served = [row for row in rows if row["kind"] == "decode_served"]
check(len(served) == 2 and all(row["speculation"] is None and row["output_tokens"] == 10 and row["prefill_tokens"] == 1 for row in served) and
      "Decode as served, tok/s, by context." in tables, f"each level ends with a reply as served on the whole history: {[(row['context'], row['output_tokens']) for row in served]}")
small = [row for row in turns if row["tail"] == 50]
check(all(row["prefill_tokens"] == 51 for row in small), f"a turn prefills its new tail and the last reply token: {[row['prefill_tokens'] for row in small]}")
check(all(row["prefill_waves"] == 1 and row["prefill_ms"] >= WAVE_MS + WAVE_GAP_MS - EARLY_MS for row in small),
      f"a small tail costs one prefill wave: {[(row['prefill_waves'], round(row['prefill_ms'], 1)) for row in small]}")
check(all(row["first_step_ms"] >= STEP_MS - EARLY_MS and row["decode_step_kind"] == "chain" for row in small),
      f"the first token costs the one-row wave of the last prompt token, and the chains come after it: {[round(row['first_step_ms'], 1) for row in small]}")
check(all(row["queue_ms"] >= QUEUE_MS - EARLY_MS and row["restore_ms"] == 0 and row["restore_polls"] == 0 for row in small),
      f"queue is acceptance to the first admitted dispatch, and a resident prefix waits for no restore: {[(round(row['queue_ms'], 1), row['restore_ms']) for row in small]}")
if apart_ms >= CLOCK_APART_MS:
    check(summary["clock_shared"] is False and all(row["send_ms"] is None and row["reply_ms"] is None for row in rows) and "left out" in table and "left out" in tables,
          f"this interpreter's clock differs between processes by {apart_ms:.1f} ms, so the tool leaves send and reply times out and says so")
if apart_ms <= CLOCK_SHARED_MS:
    check(summary["clock_shared"] is True and all(row["send_ms"] >= 0 and row["reply_ms"] is not None and row["reply_ms"] >= 0 for row in rows),
          "send and reply times are measured against the engine's clock when the client shares it")
check(isinstance(summary["clock"], str) and summary["clock"] != "", f"the run records which clock the client read: {summary['clock']}")
grow = next(row for row in rows if row["kind"] == "grow" and row["context"] == 2000)
check(grow["cached_tokens"] == 0 and grow["prefill_waves"] == 4 and grow["prefill_ms"] >= 4 * WAVE_MS + 4 * WAVE_GAP_MS - EARLY_MS and
      near(grow["between_waves_ms"], grow["prefill_ms"] - grow["waves_ms"], 0.001) and grow["between_waves_ms"] >= 4 * WAVE_GAP_MS - EARLY_MS,
      f"a cold prompt's prefill is the sum of its waves and the gaps between them: {grow['prefill_waves']} waves, {round(grow['prefill_ms'], 1)} ms")
decode = [row for row in rows if row["kind"] == "decode"]
check(len(decode) == 2 and all(row["prefill_tokens"] == 1 and row["prefill_waves"] == 0 and row["output_tokens"] == 12 and row["first_step_ms"] >= STEP_MS - EARLY_MS and
                               row["prefill_ms"] < row["first_step_ms"] for row in decode),
      f"the decode sample resumes the whole history, so its first token costs the last token's wave and no other: {[(round(row['prefill_ms'], 2), round(row['first_step_ms'], 2)) for row in decode]}")
check(all(row["decode_tok_s"] is not None and row["decode_tok_s"] <= 1.01 * 1000 / STEP_MS for row in decode),
      f"the decode sample's rate is no faster than the stub's steps: {[row.get('decode_tok_s') for row in decode]}")
second = next(row for row in rows if row["kind"] == "subagent_second")
check(second["cached_tokens"] == second["shared_tokens"] and second["prefill_tokens"] == second["prompt_tokens"] - second["shared_tokens"],
      "a second subagent of one type shares its opening and prefills only its brief")
opening = next(row for row in rows if row["kind"] == "opening")
check(opening["cached_tokens"] == 0 and opening["prompt_tokens"] == CONTEXT_LIMIT - 6 - 64, "an opening larger than the engine's context is cut to fit and is cold")
behind = [row for row in rows if row["kind"] == "probe_behind"]
check(behind and all(row["priority"] == 2 for row in behind) and any(request["priority"] == 0 for request in stub.requests),
      "the wave arm sends its long prefill as background work and its probes as interactive")
check(any(row["queue_ms"] > 10 for row in behind), f"a probe that arrives during a background wave shows the wait as queue time: {[round(row['queue_ms'], 1) for row in behind]}")
resume = next(row for row in rows if row["kind"] == "resume")
check(resume["event"] == "swap" and resume["restore_ms"] >= RESTORE_MS + QUEUE_MS - EARLY_MS and resume["restore_polls"] == 1 and resume["queue_ms"] == 0 and
      resume["prefill_waves"] == 1 and resume["prefill_ms"] >= WAVE_MS + WAVE_GAP_MS - EARLY_MS,
      f"a request the ranks answered pending waited for a restore, apart from its prefill: {round(resume['restore_ms'], 1)} ms restore, {round(resume['prefill_ms'], 1)} ms prefill")
check(resume["store"]["restore_pages"] > 0 and resume["store"]["restore_ns"] == int(RESTORE_MS * 1e6), f"the resume's restored pages come from the engine's store reports: {resume['store']}")
check(resume["cache_matched_tokens"] == resume["cached_tokens"] + 3 and resume["cache_resumed_tokens"] == resume["cached_tokens"] and resume["cache_boot"] == str(BOOT) and
      near(resume["cache_restore_ms"], resume["restore_ms"], PART_SLACK_MS) and all(row["cache_restore_ms"] == 0 for row in small),
      f"what the engine says of itself in a reply's cache object is kept beside what the log says, so the two can be compared: {resume['cache_restore_ms']} and {resume['restore_ms']}")
entry = next(item for item in summary["summary"] if item["kind"] == "resume")
known = dict({name: resume["truth"][name] for name in ENGINE_PARTS}, **({"send_ms": resume["send_ms"]} if resume["send_ms"] is not None else {}))
check(entry["largest_part"] == max(known, key=known.get), f"the summary names the largest part of a resume: {entry['largest_part']} of {known}")
check(summary["kv_store"]["save_pages"] == len(rows), f"the run's saved pages are the change in the store counters over the run: {summary['kv_store']['save_pages']} of {len(rows)}")
check("| resume (swap) |" in table and "| turn tail 50 |" in table, "the table has a row for each kind of request")
cells = [turn_floor.floor_cell(item) for item in summary["summary"] if item["kind"] in ("turn", "decode")]
check(len(cells) == 6 and all("(" in text and text in tables for text in cells),
      f"the report shows each level's first token with the largest part the summary named: {cells}")
check("| stub | " in tables and "Plain decode, tok/s" in tables and "After a swap away and back" in tables and "Behind a wave" in tables,
      f"the report compares models by context, tail, cold event and wave wait: {tables[:300]}")

for failure in failures:
    print("FAIL test_turn_floor: " + failure)
if failures:
    sys.exit(1)
print("ok test_turn_floor")

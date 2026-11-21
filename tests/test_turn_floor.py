#!/usr/bin/env python3
"""tools/turn_floor.py times an engine's turns by token ids and splits each first token's time into send, queue, restore, prefill waves and the first decode step, using only the engine's own log. This test serves a stub engine that keeps its own record of each of those times, runs every arm against it and checks that the split recovers them. The stub's sleeps set only a lower bound: a host that sleeps long changes what the stub records, never whether the split agrees with it."""
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
spec = importlib.util.spec_from_file_location("turn_floor", TOOL)
turn_floor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(turn_floor)

WAVE_ROWS = 512
WAVE_MS = 30.0
WAVE_GAP_MS = 2.0
STEP_MS = 8.0
QUEUE_MS = 3.0
RESTORE_MS = 120.0
CONTEXT_LIMIT = 9000
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


class Stub:
    def __init__(self, log_path):
        self.log = open(log_path, "a")
        self.lock = threading.Lock()
        self.busy = threading.Lock()
        self.seen = []
        self.next_id = 100000
        self.restore_next = False
        self.store = {"save_pages": 0, "save_ns": 0, "restore_pages": 0, "restore_ns": 0}
        self.requests = []
        self.truth = {}

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

    def serve(self, body, emit):
        ids, budget = body["prompt_token_ids"], body["max_tokens"]
        with self.lock:
            self.next_id += 1
            request_id = self.next_id
            cached = self.cached(ids)
            restore, self.restore_next = self.restore_next, False
        accepted = time.monotonic_ns()
        self.requests.append({"priority": body.get("priority"), "speculation": body.get("speculation"), "prompt": len(ids), "cached": cached})
        with self.busy:
            time.sleep(QUEUE_MS / 1000)
            if restore:
                self.write(f"batch_rejected status=18 kind=1 submission=7 request={request_id}")
                time.sleep(RESTORE_MS / 1000)
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
                self.write(f"batch_step kind=prefill request={request_id} rows={rows} steps=1 drafts=0 trunk=0 tree=0 kept={rows} match=0 class=0 expect=0.00 "
                           f"predict_ms=0.00 service_ms={wall - 0.5:.2f} wall_ms={wall:.2f} context={done} gap_ms={0.0 if first else WAVE_GAP_MS:.2f} draft_wait_ms=0.00")
                rows_left -= rows
                first = False
            tokens, steps = [], min(budget, 4)
            gap_began = time.monotonic_ns()
            time.sleep(WAVE_GAP_MS / 1000 if not first else 0)
            began = time.monotonic_ns()
            gap = (began - gap_began) / 1e6 if not first else 0.0
            for step in range(budget):
                if step == steps:
                    wall = (time.monotonic_ns() - began) / 1e6
                    self.write(f"batch_step kind=chain request={request_id} rows=1 steps={steps} drafts=0 trunk=0 tree=0 kept={steps} match=0 class=0 expect=0.00 "
                               f"predict_ms=0.00 service_ms={wall - 0.2:.2f} wall_ms={wall:.2f} context={len(ids) + 1} gap_ms={gap:.2f} draft_wait_ms=0.00")
                time.sleep(STEP_MS / 1000)
                token = 5000 + (request_id * 31 + step * 7) % 1000
                tokens.append([token, time.monotonic_ns()])
                emit([token], cached, None)
            if budget <= steps:
                wall = (time.monotonic_ns() - began) / 1e6
                self.write(f"batch_step kind={'single' if steps == 1 else 'chain'} request={request_id} rows=1 steps={steps} drafts=0 trunk=0 tree=0 kept={steps} match=0 "
                           f"class=0 expect=0.00 predict_ms=0.00 service_ms={wall - 0.2:.2f} wall_ms={wall:.2f} context={len(ids) + 1} gap_ms={gap:.2f} draft_wait_ms=0.00")
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

        def reply(self, document):
            data = json.dumps(document).encode()
            self.send_response(200)
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
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(tokens, cached, finish):
                event = {"object": "text_completion", "choices": [{"index": 0, "text": "", "finish_reason": finish}], "tokens": tokens,
                         "usage": {"prompt_tokens": len(body["prompt_token_ids"]), "prompt_tokens_details": {"cached_tokens": cached}}}
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()

            stub.serve(body, emit)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return Handler


def run_tool(*arguments):
    done = subprocess.run([sys.executable, str(TOOL), *arguments], capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
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
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler(stub))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    output = folder / "run"
    common = ("run", "--endpoint", endpoint, "--tokens", str(tokens_path), "--output", str(output), "--label", "stub", "--api-log", str(log_path),
              "--contexts", "2000,4000", "--tails", "50,300", "--repeats", "2", "--spec-repeats", "1", "--reply-tokens", "6", "--decode-tokens", "12",
              "--wave-tokens", "6000", "--wave-lead", "0.02", "--probes-solo", "3")
    run_tool(*common, "--arms", "turns")
    state = json.loads((output / "session.json").read_text())
    check(state["levels"] == [2000, 4000], f"the turns arm finishes every context level that fits the engine: {state['levels']}")
    check(len(state["history"]) > 4000, "the session keeps its prompt and every reply as its history")
    run_tool(*common, "--arms", "openings,wave")
    stub.restore_next = True
    run_tool(*common, "--arms", "resume", "--event", "swap")
    table = run_tool("parts", str(output))
    tables = run_tool("report", str(output))
    rows = [json.loads(line) for line in (output / "parts.jsonl").read_text().splitlines()]
    summary = json.loads((output / "summary.json").read_text())
    apart_ms = clocks_apart_ms()
    server.shutdown()

check(summary["unmatched"] == 0, f"every request finds its own line in the engine log by prompt digest: {summary['unmatched']} unmatched")
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
check(len(turns) == 8, f"two levels, two tails and two repeats make eight plain turns: {len(turns)}")
check(all(row["speculation"] is False for row in turns) and all(row["speculation"] is None for row in rows if row["kind"] == "turn_spec"),
      "plain turns turn speculation off and the as-served turns leave it to the engine")
small = [row for row in turns if row["tail"] == 50]
check(all(row["prefill_tokens"] == 51 for row in small), f"a turn prefills its new tail and the last reply token: {[row['prefill_tokens'] for row in small]}")
check(all(row["prefill_waves"] == 1 and row["prefill_ms"] >= WAVE_MS + WAVE_GAP_MS - EARLY_MS for row in small),
      f"a small tail costs one prefill wave: {[(row['prefill_waves'], round(row['prefill_ms'], 1)) for row in small]}")
check(all(row["first_step_ms"] >= STEP_MS - EARLY_MS for row in small), f"the first decode step is timed from its dispatch: {[round(row['first_step_ms'], 1) for row in small]}")
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
check(len(decode) == 2 and all(row["prefill_tokens"] == 1 and row["prefill_waves"] == 0 and row["output_tokens"] == 12 for row in decode),
      "the decode sample resumes the whole history and prefills nothing")
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
entry = next(item for item in summary["summary"] if item["kind"] == "resume")
known = dict({name: resume["truth"][name] for name in ENGINE_PARTS}, **({"send_ms": resume["send_ms"]} if resume["send_ms"] is not None else {}))
check(entry["largest_part"] == max(known, key=known.get), f"the summary names the largest part of a resume: {entry['largest_part']} of {known}")
check(summary["kv_store"]["save_pages"] == len(rows), f"the run's saved pages are the change in the store counters over the run: {summary['kv_store']['save_pages']}")
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

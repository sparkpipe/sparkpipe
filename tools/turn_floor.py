#!/usr/bin/env python3
import argparse
import array
import hashlib
import http.client
import json
import random
import re
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path

USER_AGENT = "sparkpipe-turn-floor/1"
INTERACTIVE = 2
BACKGROUND = 0
OPENING_TOKENS = 46000
SUBAGENT_TOKENS = 37000
SUBAGENT_UNIQUE_SHARE = 0.03
SECOND_PROMPT_TOKENS = 500
PROBE_TOKENS = 2048
RESUME_TAIL_TOKENS = 300
SUBAGENT_BRIEF_TOKENS = 2048
SESSION_SLACK_TOKENS = 8192
BLOCK_MARGIN_TOKENS = 64
RESUME_REGION_TOKENS = 16 * RESUME_TAIL_TOKENS
RESUME_SETTLE_SECONDS = 2.5
STORE_LOOKBACK_BYTES = 16 << 20
STEP = re.compile(r"batch_step kind=(\w+) request=(\d+) rows=(\d+) steps=(\d+) .*?kept=(\d+) .*?service_ms=([0-9.]+) wall_ms=([0-9.]+) context=(\d+) gap_ms=([0-9.]+)")
REJECTED = re.compile(r"batch_rejected status=(\d+) kind=(\d+) submission=\d+ request=(\d+)")
STATUS_NOT_FOUND = 3
STATUS_BUSY = 15
STATUS_PENDING = 18
NOTE = re.compile(r"(batch_resume_cleared|batch_resume_superseded|batch_resume_unmarked|batch_shared_prefix_lost|batch_prefill_span_shrunk|batch_retry_deadline) request=(\d+)")
STORE_FIELDS = ("save_pages", "save_ns", "save_count", "restore_pages", "restore_ns", "restore_count", "restore_misses", "restore_failures", "pool_resident_evictions",
                "pool_evicted_entries", "spill_write_bytes", "spill_read_bytes", "write_bytes")
PART_NAMES = ("send_ms", "queue_ms", "restore_ms", "prefill_ms", "first_step_ms")
CLIENT_PARTS = ("send_ms", "reply_ms")
CACHE_FIELDS = ("matched_tokens", "resumed_tokens", "restore_ms", "background_wait_ms", "boot", "artifact")
SESSION_SAMPLES = ("turn", "turn_spec", "decode", "decode_served")
CLOCK_APART = "the client's clock is not the engine's (a send or reply time came out negative), so send and reply times are left out"


def now_ns():
    return time.monotonic_ns()


def rounded(value, digits=1):
    return None if value is None else round(value, digits)


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def load_tokens(path):
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return [int(value) for value in json.loads(text)]
    return [int(value) for value in text.split()]


def prompt_sha256(ids):
    words = array.array("I", ids)
    if words.itemsize != 4:
        raise SystemExit("turn_floor: this platform's unsigned int is not 4 bytes, so prompt digests would not match the engine's")
    if sys.byteorder != "little":
        words.byteswap()
    return hashlib.sha256(words.tobytes()).hexdigest()


class Source:
    def __init__(self, ids, salt, sizes):
        self.ids = ids
        self.rotation = salt % len(ids)
        self.regions = {}
        offset = 0
        for name, size in sizes:
            self.regions[name] = (offset, size)
            offset += size
        if offset > len(ids):
            raise SystemExit(f"turn_floor: the token file holds {len(ids)} tokens and this run's regions need {offset}")

    def piece(self, name, start, count):
        offset, size = self.regions[name]
        if start + count > size:
            raise SystemExit(f"turn_floor: the {name} region holds {size} tokens and the run asked for {start + count}")
        begin = (self.rotation + offset + start) % len(self.ids)
        if begin + count <= len(self.ids):
            return self.ids[begin:begin + count]
        return self.ids[begin:] + self.ids[:begin + count - len(self.ids)]


class Engine:
    def __init__(self, endpoint, timeout):
        self.url = urllib.parse.urlparse(endpoint.rstrip("/"))
        self.timeout = timeout

    def connection(self, timeout=None):
        return http.client.HTTPConnection(self.url.hostname, self.url.port, timeout=timeout or self.timeout)

    def get(self, path):
        connection = self.connection(10)
        try:
            connection.request("GET", path, headers={"User-Agent": USER_AGENT})
            response = connection.getresponse()
            return response.status, json.loads(response.read() or b"null")
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            return 0, {"error": f"{type(error).__name__}: {error}"}
        finally:
            connection.close()

    def cache_state(self, ids):
        connection = self.connection(60)
        try:
            connection.request("POST", "/v1/cache/state", json.dumps({"prompt_token_ids": ids}, separators=(",", ":")).encode(),
                               {"Content-Type": "application/json", "User-Agent": USER_AGENT})
            response = connection.getresponse()
            text = response.read().decode("utf-8", "replace")
            try:
                return response.status, json.loads(text)
            except json.JSONDecodeError:
                return response.status, {"body": text[:300]}
        except (OSError, http.client.HTTPException) as error:
            return 0, {"transport": f"{type(error).__name__}: {error}"}
        finally:
            connection.close()

    def context_limit(self):
        status, models = self.get("/v1/models")
        if status != 200:
            raise SystemExit(f"turn_floor: /v1/models answered {status}: {json.dumps(models)[:200]}")
        return int(models["data"][0]["max_model_len"]), models["data"][0]["id"]

    def complete(self, ids, max_tokens, priority=INTERACTIVE, speculation=None):
        body = {"prompt_token_ids": ids, "max_tokens": max_tokens, "stream": True, "temperature": 0, "priority": priority}
        if speculation is not None:
            body["speculation"] = speculation
        encode_started = now_ns()
        data = json.dumps(body, separators=(",", ":")).encode()
        record = {"prompt_tokens": len(ids), "prompt_sha256": prompt_sha256(ids), "max_tokens": max_tokens, "priority": priority, "speculation": speculation,
                  "body_bytes": len(data), "encode_ms": rounded((now_ns() - encode_started) / 1e6, 2), "tokens": [], "token_ns": [], "cached_tokens": None,
                  "finish": None, "error": None, "first_ns": None}
        connection = self.connection()
        record["sent_ns"] = now_ns()
        try:
            connection.request("POST", "/v1/completions", data, {"Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": USER_AGENT})
            response = connection.getresponse()
            if response.status != 200:
                record["error"] = {"http_status": response.status, "body": response.read().decode(errors="replace")[:400]}
                return record
            while True:
                raw = response.readline()
                if not raw:
                    record["error"] = record["error"] or {"stream": "ended before [DONE]"}
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                stamp = now_ns()
                if "error" in event:
                    record["error"] = event["error"]
                    continue
                tokens = event.get("tokens") or []
                if tokens and record["first_ns"] is None:
                    record["first_ns"] = stamp
                record["tokens"].extend(tokens)
                record["token_ns"].extend([stamp] * len(tokens))
                usage = event.get("usage")
                if usage:
                    record["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", record["cached_tokens"])
                if isinstance(event.get("cache"), dict):
                    record.update({"cache_" + name: event["cache"].get(name) for name in CACHE_FIELDS})
                finish = (event.get("choices") or [{}])[0].get("finish_reason")
                record["finish"] = finish or record["finish"]
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            record["error"] = {"transport": f"{type(error).__name__}: {error}"}
        finally:
            record["end_ns"] = now_ns()
            connection.close()
        return record


def shown(record):
    first, sent, end = record.get("first_ns"), record["sent_ns"], record["end_ns"]
    count = len(record["tokens"])
    stamps = record.get("token_ns") or []
    spread = (stamps[-1] - stamps[0]) / 1e9 if count > 1 else 0.0
    out = {key: value for key, value in record.items() if key not in ("tokens", "token_ns")}
    out.update(output_tokens=count, ttft_ms=rounded((first - sent) / 1e6) if first else None, total_ms=rounded((end - sent) / 1e6),
               stream_tok_s=rounded((count - 1) / spread, 2) if spread > 0 else None,
               prefill_tokens=record["prompt_tokens"] - record["cached_tokens"] if record["cached_tokens"] is not None else None)
    return out


class Run:
    def __init__(self, arguments):
        self.arguments = arguments
        self.output = Path(arguments.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.engine = Engine(arguments.endpoint, arguments.timeout)
        self.limit, self.model = self.engine.context_limit()
        self.margin = min(BLOCK_MARGIN_TOKENS, self.limit // 8)
        self.state_path = self.output / "session.json"
        self.contexts = [value for value in arguments.contexts
                         if value + self.growth(arguments) + max(arguments.decode_tokens, arguments.served_tokens) + self.margin < self.limit]
        self.resumed = self.state_path.exists()
        if self.resumed:
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"salt": arguments.salt if arguments.salt is not None else random.SystemRandom().randrange(1 << 30), "history": [], "taken": 0,
                          "resume_taken": 0, "levels": [], "sequence": 0, "wave_tokens": max(1, min(arguments.wave_tokens, self.limit - self.margin - 1)),
                          "session_tokens": arguments.session_tokens or min(self.limit, max(self.contexts + [arguments.decode_context or 0]) + self.growth(arguments))}
        room = self.limit - arguments.reply_tokens - self.margin
        self.opening_tokens, self.subagent_tokens, self.probe_tokens = min(OPENING_TOKENS, room), min(SUBAGENT_TOKENS, room), min(PROBE_TOKENS, room)
        self.source = Source(load_tokens(arguments.tokens), self.state["salt"],
                             (("session", self.state["session_tokens"] + SESSION_SLACK_TOKENS), ("opening", self.opening_tokens), ("second_prompt", SECOND_PROMPT_TOKENS),
                              ("subagent", self.subagent_tokens), ("subagent_brief", SUBAGENT_BRIEF_TOKENS), ("wave", self.state["wave_tokens"]),
                              ("probe", self.probe_tokens), ("resume", RESUME_REGION_TOKENS)))
        self.requests = open(self.output / "requests.jsonl", "a")
        self.log_path = Path(arguments.api_log) if arguments.api_log else None
        self.mark("start")

    @staticmethod
    def level_turns(arguments):
        turns = []
        for repeat in range(arguments.repeats + arguments.spec_repeats):
            for tail in arguments.tails:
                if repeat == 0 or not arguments.once_from or tail < arguments.once_from:
                    turns.append(("turn", tail, False) if repeat < arguments.repeats else ("turn_spec", tail, None))
        return turns

    @classmethod
    def growth(cls, arguments):
        return sum(tail + arguments.reply_tokens for _, tail, _ in cls.level_turns(arguments))

    def log_offset(self):
        try:
            return self.log_path.stat().st_size if self.log_path is not None else None
        except OSError:
            return None

    def mark(self, what, **extra):
        health = self.engine.get("/health")[1]
        self.write(dict(extra, kind="mark", what=what, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), monotonic_ns=now_ns(), log_offset=self.log_offset(),
                        model=self.model, context_limit=self.limit, label=self.arguments.label, salt=self.state["salt"], endpoint=self.arguments.endpoint,
                        api_log=str(self.log_path) if self.log_path else None, health=health, argv=sys.argv[1:],
                        clock=time.get_clock_info("monotonic").implementation,
                        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    def write(self, record):
        self.requests.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.requests.flush()

    def save(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, separators=(",", ":")))
        temporary.replace(self.state_path)

    def take(self, count):
        piece = self.source.piece("session", self.state["taken"], count)
        self.state["taken"] += count
        return piece

    def send(self, kind, ids, max_tokens, context=None, tail=None, priority=INTERACTIVE, speculation=None, **extra):
        self.state["sequence"] += 1
        if self.resumed and kind in SESSION_SAMPLES:
            extra["first_of_run"] = True
        self.resumed = False
        record = self.engine.complete(ids, max_tokens, priority, speculation)
        line = dict(shown(record), kind=kind, context=context, tail=tail, sequence=self.state["sequence"], label=self.arguments.label, model=self.model,
                    log_offset=self.log_offset(), **extra)
        self.write(line)
        print(json.dumps({key: line.get(key) for key in ("kind", "context", "tail", "prompt_tokens", "cached_tokens", "prefill_tokens", "output_tokens", "ttft_ms",
                                                         "total_ms", "stream_tok_s", "event", "error")}), flush=True)
        if record["error"] is not None:
            raise SystemExit(f"turn_floor: the {kind} request failed: {json.dumps(record['error'])[:300]}")
        return record

    def turn(self, kind, context, tail, speculation, done=None):
        history = self.state["history"]
        taken = self.state["taken"]
        new = self.take(tail)
        try:
            record = self.send(kind, history + new, self.arguments.reply_tokens, context, tail, speculation=speculation)
        except SystemExit:
            self.state["taken"] = taken
            raise
        self.state["history"] = history + new + record["tokens"]
        if done is not None:
            self.state["level"]["done"] = done
        self.save()

    def sample(self, kind, context, tokens, speculation, done):
        self.send(kind, list(self.state["history"]), tokens, context, 0, speculation=speculation)
        self.state["level"]["done"] = done
        self.save()

    def turns(self):
        arguments = self.arguments
        skipped = [value for value in arguments.contexts if value not in self.contexts]
        if skipped:
            print(json.dumps({"skipped": "levels", "contexts": skipped, "limit": self.limit, "growth": self.growth(arguments)}), flush=True)
        if not self.contexts:
            raise SystemExit(f"turn_floor: no context level of {arguments.contexts} fits the engine's {self.limit} positions beside {self.growth(arguments)} tokens of turns")
        for context in self.contexts:
            if context in self.state["levels"]:
                continue
            level = self.state.get("level")
            if not level or level["context"] != context:
                level = self.state["level"] = {"context": context, "done": 0}
            steps = [("grow", None, False)] + self.level_turns(arguments)
            steps += [("decode", arguments.decode_tokens, False)] if arguments.decode_tokens else []
            steps += [("decode_served", arguments.served_tokens, None)] if arguments.served_tokens else []
            for index, (kind, size, speculation) in enumerate(steps):
                if index < level["done"]:
                    continue
                if kind == "grow":
                    missing = context - len(self.state["history"])
                    if missing > 0:
                        self.turn("grow", context, missing, False, index + 1)
                    else:
                        level["done"] = index + 1
                        self.save()
                elif kind in ("decode", "decode_served"):
                    self.sample(kind, context, size, speculation, index + 1)
                else:
                    self.turn(kind, context, size, speculation, index + 1)
            self.state["levels"].append(context)
            self.state["level"] = None
            self.save()

    def decode(self):
        context = self.arguments.decode_context
        if context is None or context + self.arguments.decode_tokens + self.margin >= self.limit:
            print(json.dumps({"skipped": "decode", "context": context, "limit": self.limit}), flush=True)
            return
        missing = context - len(self.state["history"])
        if missing > 0:
            self.turn("grow", context, missing, False)
        self.send("decode", list(self.state["history"]), self.arguments.decode_tokens, context, 0, speculation=False)

    def openings(self):
        reply = self.arguments.reply_tokens
        opening = self.source.piece("opening", 0, self.opening_tokens)
        self.send("opening", opening, reply, len(opening), len(opening), speculation=False)
        shared = len(opening) - SECOND_PROMPT_TOKENS
        self.send("opening_second", opening[:shared] + self.source.piece("second_prompt", 0, SECOND_PROMPT_TOKENS), reply, len(opening), SECOND_PROMPT_TOKENS,
                  speculation=False, shared_tokens=shared)
        size = self.subagent_tokens
        unique = min(int(size * SUBAGENT_UNIQUE_SHARE), SUBAGENT_BRIEF_TOKENS)
        subagent = self.source.piece("subagent", 0, size)
        self.send("subagent", subagent, reply, size, size, speculation=False)
        self.send("subagent_second", subagent[:size - unique] + self.source.piece("subagent_brief", 0, unique), reply, size, unique, speculation=False,
                  shared_tokens=size - unique)

    def wave(self):
        arguments = self.arguments
        probe = self.source.piece("probe", 0, self.probe_tokens)
        self.send("probe_warm", probe, 1, len(probe), len(probe), speculation=False)
        solo = [self.send("probe_solo", probe, 1, len(probe), 0, speculation=False) for _ in range(arguments.probes_solo)]
        baseline = median([(record["first_ns"] - record["sent_ns"]) / 1e6 for record in solo])
        background = self.source.piece("wave", 0, self.state["wave_tokens"])
        done = threading.Event()
        holder = {}

        def run_background():
            holder["record"] = self.engine.complete(background, 1, BACKGROUND, False)
            done.set()

        thread = threading.Thread(target=run_background)
        thread.start()
        rng = random.Random(self.state["salt"])
        time.sleep(arguments.wave_lead)
        waits = []
        while not done.is_set():
            record = self.send("probe_behind", probe, 1, len(probe), 0, speculation=False, baseline_ms=rounded(baseline))
            waits.append((record["first_ns"] - record["sent_ns"]) / 1e6 - baseline)
            done.wait(rng.uniform(0.2, max(1.0, 1.5 * max(waits))))
        thread.join()
        record = holder["record"]
        self.state["sequence"] += 1
        self.write(dict(shown(record), kind="wave_background", context=len(background), tail=len(background), sequence=self.state["sequence"], label=arguments.label,
                        model=self.model, log_offset=self.log_offset()))
        if record["error"] is not None:
            raise SystemExit(f"turn_floor: the background request failed: {json.dumps(record['error'])[:300]}")
        print(json.dumps({"kind": "wave", "probes": len(waits), "baseline_ms": rounded(baseline), "wait_ms_median": rounded(median(waits)),
                          "wait_ms_max": rounded(max(waits, default=None)), "background_ms": shown(record)["total_ms"]}), flush=True)

    def resume(self):
        history = self.state["history"]
        if not history:
            raise SystemExit("turn_floor: there is no saved session to resume; run the turns arm first")
        new = self.source.piece("resume", self.state["resume_taken"], RESUME_TAIL_TOKENS)
        self.state["resume_taken"] += RESUME_TAIL_TOKENS
        record = self.send("resume", history + new, self.arguments.reply_tokens, len(history), RESUME_TAIL_TOKENS, speculation=False, event=self.arguments.event)
        self.state["history"] = history + new + record["tokens"]
        self.save()
        time.sleep(RESUME_SETTLE_SECONDS)
        self.turn("turn_after_resume", len(history), RESUME_TAIL_TOKENS, False)

    def truth(self):
        history = self.state["history"]
        if not history:
            raise SystemExit("turn_floor: there is no saved session to ask the engine about; run the turns arm first")
        status, answer = self.engine.cache_state(history)
        self.state["sequence"] += 1
        record = {"kind": "cache_state", "event": self.arguments.event, "sequence": self.state["sequence"], "label": self.arguments.label, "model": self.model,
                  "history_tokens": len(history), "http_status": status, "answer": answer, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "log_offset": self.log_offset()}
        self.write(record)
        self.save()
        print(json.dumps({key: record[key] for key in ("kind", "event", "history_tokens", "http_status", "answer")}), flush=True)

    def close(self):
        self.mark("end")
        self.requests.close()


def run(arguments):
    runner = Run(arguments)
    try:
        for arm in arguments.arms:
            getattr(runner, arm)()
    finally:
        runner.close()


def read_requests(folder):
    records = []
    with open(Path(folder) / "requests.jsonl") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


class LogSlice:
    def __init__(self, path, start):
        self.measurements = {}
        self.steps = {}
        self.notes = {}
        self.rejections = {}
        self.stores = []
        boot, order, first_boot = None, 0, None
        with open(path, "rb") as handle:
            back = max(0, start - STORE_LOOKBACK_BYTES)
            handle.seek(back)
            if back:
                handle.readline()
            while handle.tell() < start:
                raw = handle.readline()
                if not raw:
                    break
                if raw.startswith(b"{\"event\":\"kv_store_report\""):
                    try:
                        self.stores.append((0, json.loads(raw)))
                    except json.JSONDecodeError:
                        continue
            handle.seek(start)
            for raw in handle:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                order += 1
                if line.startswith("{\"event\":\"request_measurements\""):
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    boot = event.get("boot_pid")
                    first_boot = boot if first_boot is None else first_boot
                    event["order"] = order
                    self.measurements.setdefault(event.get("prompt_sha256"), []).append(event)
                elif line.startswith("{\"event\":\"kv_store_report\""):
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    boot = event.get("boot_pid")
                    first_boot = boot if first_boot is None else first_boot
                    self.stores.append((order, event))
                elif line.startswith("batch_step "):
                    match = STEP.match(line)
                    if match:
                        kind, request, rows, steps, kept, service, wall, context, gap = match.groups()
                        self.steps.setdefault((boot, int(request)), []).append({"kind": kind, "rows": int(rows), "steps": int(steps), "kept": int(kept),
                                                                                "service_ms": float(service), "wall_ms": float(wall), "context": int(context),
                                                                                "gap_ms": float(gap), "order": order})
                elif line.startswith("batch_rejected "):
                    match = REJECTED.match(line)
                    if match:
                        self.rejections.setdefault((boot, int(match.group(3))), []).append(int(match.group(1)))
                else:
                    match = NOTE.match(line)
                    if match:
                        self.notes.setdefault((boot, int(match.group(2))), []).append(match.group(1))
        self.lines = order
        if first_boot is not None:
            for table in (self.steps, self.rejections, self.notes):
                for key in [key for key in table if key[0] is None]:
                    table[(first_boot, key[1])] = table.pop(key) + table.get((first_boot, key[1]), [])

    def request_steps(self, measurement):
        found = self.steps.get((measurement.get("boot_pid"), measurement["request_id"]))
        if found is None:
            found = self.steps.get((None, measurement["request_id"]), [])
        return [step for step in found if step["order"] < measurement["order"]]

    def store_change(self, after_order, through_order):
        first, last = {}, {}
        for order, event in self.stores:
            if order > through_order:
                break
            key = (event.get("boot_pid"), event.get("residentd_pid"), event.get("rank"))
            if order <= after_order:
                first[key] = event
            last[key] = event
        ranks = [{field: event.get(field, 0) - first.get(key, {}).get(field, 0) for field in STORE_FIELDS} for key, event in last.items()]
        if not ranks:
            return None
        total = {field: sum(rank[field] for rank in ranks) for field in STORE_FIELDS}
        total.update(ranks=len(ranks), restore_ms_rank_max=max(rank["restore_ns"] for rank in ranks) / 1e6, save_ms_rank_max=max(rank["save_ns"] for rank in ranks) / 1e6,
                     restore_pages_rank_max=max(rank["restore_pages"] for rank in ranks), save_pages_rank_max=max(rank["save_pages"] for rank in ranks))
        return total


def split(request, measurement, steps, rejections):
    tokens = measurement.get("tokens") or []
    accepted, dispatched = measurement["accepted_ns"], measurement["first_dispatch_ns"]
    pending = sum(1 for status in rejections if status == STATUS_PENDING)
    parts = {"request_id": measurement["request_id"], "boot_pid": measurement.get("boot_pid"), "engine_cached_tokens": measurement.get("cached_prompt_tokens"),
             "stale_prefix_recomputes": measurement.get("stale_prefix_recomputes"), "send_ms": (accepted - request["sent_ns"]) / 1e6,
             "restore_polls": pending, "busy_rejections": sum(1 for status in rejections if status == STATUS_BUSY),
             "stale_rejections": sum(1 for status in rejections if status == STATUS_NOT_FOUND)}
    if not tokens or not dispatched:
        return parts
    wait = (dispatched - accepted) / 1e6
    parts["restore_ms"] = wait if pending else 0.0
    parts["queue_ms"] = wait - parts["restore_ms"]
    first_ready = tokens[0][1]
    parts["engine_ttft_ms"] = (first_ready - accepted) / 1e6
    if len(tokens) > 2 and tokens[-1][1] > first_ready:
        parts["decode_tok_s"] = (len(tokens) - 1) / ((tokens[-1][1] - first_ready) / 1e9)
    parts["reply_ms"] = (request["first_ns"] - first_ready) / 1e6 if request.get("first_ns") else None
    prefill = [step for step in steps if step["kind"] == "prefill"]
    decode = [step for step in steps if step["kind"] != "prefill"]
    last = prefill[-1] if prefill and prefill[-1]["rows"] == 1 else None
    waves = prefill[:-1] if last else prefill
    parts["first_step_ms"] = last["wall_ms"] if last else 0.0
    parts["first_step_service_ms"] = last["service_ms"] if last else None
    parts["prefill_ms"] = (first_ready - dispatched) / 1e6 - parts["first_step_ms"]
    parts["waves_ms"] = sum(step["wall_ms"] for step in waves)
    parts["between_waves_ms"] = parts["prefill_ms"] - parts["waves_ms"]
    parts.update(prefill_waves=len(waves), prefill_rows=sum(step["rows"] for step in waves), prefill_service_ms=sum(step["service_ms"] for step in waves),
                 wave_rows_max=max((step["rows"] for step in waves), default=0), wave_wall_ms_max=max((step["wall_ms"] for step in waves), default=None),
                 decode_step_kind=decode[0]["kind"] if decode else None, decode_step_tokens=decode[0]["steps"] if decode else None)
    return parts


def join(folder, log_path=None, log_base=0):
    records = read_requests(folder)
    marks = [record for record in records if record.get("kind") == "mark"]
    requests = [record for record in records if record.get("kind") not in ("mark", "cache_state")]
    log_path = log_path or next((mark["api_log"] for mark in marks if mark.get("api_log")), None)
    if log_path is None:
        raise SystemExit("turn_floor: the run recorded no engine log; pass --api-log")
    start = min((mark["log_offset"] for mark in marks if mark.get("log_offset") is not None), default=0)
    if start < log_base:
        raise SystemExit(f"turn_floor: the run's log starts at byte {start}, before the copied part of the log at byte {log_base}")
    log = LogSlice(log_path, start - log_base)
    sent = {}
    for position, request in enumerate(requests):
        sent.setdefault(request["prompt_sha256"], []).append(position)
    paired = {}
    for digest, positions in sent.items():
        candidates = log.measurements.get(digest, [])
        if len(candidates) < len(positions):
            positions = [position for position in positions if requests[position].get("error") is None]
            candidates = candidates[max(0, len(candidates) - len(positions)):]
        paired.update(zip(positions, candidates))
    joined = []
    for position, request in enumerate(requests):
        row = dict(request)
        measurement = paired.get(position)
        if measurement is not None:
            key = (measurement.get("boot_pid"), measurement["request_id"])
            row.update(split(request, measurement, log.request_steps(measurement), log.rejections.get(key, [])))
            row["notes"] = log.notes.get(key, [])
            row["order"] = measurement["order"]
        elif request.get("error") is None:
            row["unmatched"] = True
        joined.append(row)
    ordered = sorted((row["order"] for row in joined if "order" in row))
    for row in joined:
        if "order" in row:
            position = ordered.index(row["order"])
            before = ordered[position - 1] if position else 0
            through = ordered[position + 1] if position + 1 < len(ordered) else log.lines
            row["store"] = log.store_change(before, through)
    shared = not any(row.get(name) is not None and row[name] < 0 for row in joined for name in CLIENT_PARTS)
    if not shared:
        for row in joined:
            row.update({name: None for name in CLIENT_PARTS if name in row})
    return joined, marks, log.store_change(0, log.lines), shared


def cell(value, digits=0):
    if value is None:
        return "–"
    return f"{value:,.{digits}f}"


def summarize(rows):
    groups = {}
    for row in rows:
        if row.get("error") is None and not row.get("first_of_run") and row["kind"] in SESSION_SAMPLES + (
                "grow", "opening", "opening_second", "subagent", "subagent_second", "resume", "turn_after_resume", "probe_solo", "probe_behind", "wave_background"):
            key = (row["kind"], row.get("event"), row.get("context"), row.get("tail") if row["kind"] in ("turn", "turn_spec") else None)
            groups.setdefault(key, []).append(row)
    summary = []
    for (kind, event, context, tail), group in groups.items():
        entry = {"kind": kind, "event": event, "context": context, "tail": tail, "requests": len(group),
                 "prompt_tokens": median([row["prompt_tokens"] for row in group]), "cached_tokens": median([row["cached_tokens"] for row in group]),
                 "prefill_tokens": median([row["prefill_tokens"] for row in group]), "ttft_ms": median([row["ttft_ms"] for row in group]),
                 "ttft_ms_max": max((row["ttft_ms"] for row in group if row["ttft_ms"] is not None), default=None),
                 "stream_tok_s": median([row.get("stream_tok_s") for row in group]), "decode_tok_s": median([row.get("decode_tok_s") for row in group]),
                 "encode_ms": median([row.get("encode_ms") for row in group]), "reply_ms": median([row.get("reply_ms") for row in group]),
                 "prefill_waves": median([row.get("prefill_waves") for row in group]), "wave_rows_max": max((row.get("wave_rows_max") or 0 for row in group), default=0),
                 "wave_wall_ms_max": max((row.get("wave_wall_ms_max") or 0.0 for row in group), default=0.0),
                 "matched": sum(1 for row in group if not row.get("unmatched")), "notes": sorted({note for row in group for note in row.get("notes") or []})}
        for name in PART_NAMES + ("cache_restore_ms", "cache_background_wait_ms"):
            entry[name] = median([row.get(name) for row in group])
        named = {name: entry[name] for name in PART_NAMES if entry[name] is not None}
        entry["largest_part"] = max(named, key=named.get) if named else None
        stores = [row["store"] for row in group if row.get("store")]
        if stores:
            entry["restore_pages"] = sum(store["restore_pages"] for store in stores)
            entry["restore_ms_rank_max"] = max(store["restore_ms_rank_max"] for store in stores)
            entry["save_pages"] = sum(store["save_pages"] for store in stores)
            entry["save_ns"] = sum(store["save_ns"] for store in stores)
            entry["restore_ns"] = sum(store["restore_ns"] for store in stores)
            entry["ranks"] = max(store["ranks"] for store in stores)
        if kind == "probe_behind":
            waits = [row["ttft_ms"] - row["baseline_ms"] for row in group if row.get("baseline_ms") is not None and row["ttft_ms"] is not None]
            entry["wait_ms"] = {"median": median(waits), "max": max(waits, default=None), "p90": sorted(waits)[int(0.9 * (len(waits) - 1))] if waits else None}
        summary.append(entry)
    return summary


def markdown(summary):
    lines = ["| Request | Context | New tokens | First token, ms | Send | Queue | Restore | Prefill waves | First step | Largest part | Waves | Decode tok/s |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for entry in summary:
        name = entry["kind"] + (f" ({entry['event']})" if entry.get("event") else "") + (f" tail {entry['tail']}" if entry.get("tail") is not None else "")
        lines.append(f"| {name} | {cell(entry['prompt_tokens'])} | {cell(entry['prefill_tokens'])} | {cell(entry['ttft_ms'])} | {cell(entry['send_ms'])} | "
                     f"{cell(entry['queue_ms'], 1)} | {cell(entry['restore_ms'])} | {cell(entry['prefill_ms'])} | {cell(entry['first_step_ms'])} | "
                     f"{entry['largest_part'] or '–'} | {cell(entry['prefill_waves'])} | {cell(entry['decode_tok_s'] or entry['stream_tok_s'], 1)} |")
    return "\n".join(lines)


def entries(document, kind, event=None):
    return [entry for entry in document["summary"] if entry["kind"] == kind and (event is None or entry.get("event") == event)]


def seconds(value):
    return "–" if value is None else f"{value / 1000:,.2f}"


def floor_cell(entry):
    if entry is None or entry["ttft_ms"] is None:
        return "–"
    part = (entry.get("largest_part") or "").replace("_ms", "").replace("_", " ")
    return f"{seconds(entry['ttft_ms'])} ({part})" if part else seconds(entry["ttft_ms"])


def report_tables(documents):
    contexts = sorted({entry["context"] for document in documents for entry in entries(document, "turn")})
    tails = sorted({entry["tail"] for document in documents for entry in entries(document, "turn")})
    lines = []
    for tail in [0] + tails:
        title = "no new tokens (the decode sample)" if tail == 0 else f"{tail:,} new tokens"
        lines += ["", f"First token on a cached session, {title}: seconds, with the largest part.", "",
                  "| Model | " + " | ".join(f"{context:,}" for context in contexts) + " |", "| --- | " + " | ".join("---:" for _ in contexts) + " |"]
        for document in documents:
            found = {entry["context"]: entry for entry in (entries(document, "decode") if tail == 0 else entries(document, "turn")) if tail == 0 or entry["tail"] == tail}
            lines.append(f"| {document['model']} | " + " | ".join(floor_cell(found.get(context)) for context in contexts) + " |")
    lines += ["", "Parts of the first token with 50 new tokens, ms: send, queue, restore, prefill, first step.", "",
              "| Model | " + " | ".join(f"{context:,}" for context in contexts) + " |", "| --- | " + " | ".join("---" for _ in contexts) + " |"]
    for document in documents:
        found = {entry["context"]: entry for entry in entries(document, "turn") if entry["tail"] == min(tails, default=None)}
        lines.append(f"| {document['model']} | " + " | ".join("–" if context not in found else " / ".join(cell(found[context].get(name)) for name in PART_NAMES)
                                                                 for context in contexts) + " |")
    decode_contexts = sorted({entry["context"] for document in documents for entry in entries(document, "decode")})
    lines += ["", "Plain decode, tok/s, by context.", "", "| Model | " + " | ".join(f"{context:,}" for context in decode_contexts) + " |",
              "| --- | " + " | ".join("---:" for _ in decode_contexts) + " |"]
    for document in documents:
        found = {entry["context"]: entry for entry in entries(document, "decode")}
        lines.append(f"| {document['model']} | " + " | ".join(cell((found.get(context) or {}).get("decode_tok_s"), 1) for context in decode_contexts) + " |")
    served_contexts = sorted({entry["context"] for document in documents for entry in entries(document, "decode_served")})
    if served_contexts:
        lines += ["", "Decode as served, tok/s, by context.", "", "| Model | " + " | ".join(f"{context:,}" for context in served_contexts) + " |",
                  "| --- | " + " | ".join("---:" for _ in served_contexts) + " |"]
        for document in documents:
            found = {entry["context"]: entry for entry in entries(document, "decode_served")}
            lines.append(f"| {document['model']} | " + " | ".join(cell((found.get(context) or {}).get("decode_tok_s") or (found.get(context) or {}).get("stream_tok_s"), 1)
                                                                 for context in served_contexts) + " |")
    lines += ["", "Cold events: seconds to the first token, with the tokens prefilled.", "",
              "| Model | Opening | Second session, same opening | Subagent opening | Second subagent | After an engine restart | After a swap away and back |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for document in documents:
        cells = []
        for kind, event in (("opening", None), ("opening_second", None), ("subagent", None), ("subagent_second", None), ("resume", "restart"), ("resume", "swap")):
            found = entries(document, kind, event)
            cells.append("–" if not found else f"{seconds(found[0]['ttft_ms'])} ({cell(found[0]['prefill_tokens'])} of {cell(found[0]['prompt_tokens'])})")
        lines.append(f"| {document['model']} | " + " | ".join(cells) + " |")
    lines += ["", "Restore, save and the wait behind one background wave.", "",
              "| Model | Restore, s | Tokens restored | Restore tok/s | Rank restore, µs a page | Rank save, µs a page | Probe alone, ms | Behind a wave: median / max, ms | Background wave: rows, s |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for document in documents:
        resumes = [entry for entry in entries(document, "resume") if entry.get("restore_ms")]
        restore = max(resumes, key=lambda entry: entry["restore_ms"]) if resumes else None
        store = document.get("kv_store") or {}
        behind, alone, wave = entries(document, "probe_behind"), entries(document, "probe_solo"), entries(document, "wave_background")
        wait = (behind[0].get("wait_ms") or {}) if behind else {}
        lines.append(f"| {document['model']} | {seconds(restore['restore_ms']) if restore else '–'} | {cell(restore['cached_tokens']) if restore else '–'} | "
                     f"{cell(restore['cached_tokens'] / (restore['restore_ms'] / 1000)) if restore else '–'} | "
                     f"{cell(store['restore_ns'] / store['restore_pages'] / 1000) if store.get('restore_pages') else '–'} | "
                     f"{cell(store['save_ns'] / store['save_pages'] / 1000) if store.get('save_pages') else '–'} | {cell(alone[0]['ttft_ms']) if alone else '–'} | "
                     f"{cell(wait.get('median'))} / {cell(wait.get('max'))} | "
                     f"{cell(wave[0]['wave_rows_max']) if wave else '–'}, {seconds(wave[0]['wave_wall_ms_max']) if wave else '–'} |")
    apart = [document["model"] for document in documents if document.get("clock_shared") is False]
    if apart:
        lines += ["", f"For {', '.join(str(model) for model in apart)}, {CLOCK_APART}."]
    return "\n".join(lines).strip()


def report(arguments):
    documents = [json.loads((Path(folder) / "summary.json").read_text()) for folder in arguments.folders]
    print(report_tables(documents))


def parts(arguments):
    rows, marks, store, shared = join(arguments.folder, arguments.api_log, arguments.log_base)
    folder = Path(arguments.folder)
    with open(folder / "parts.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = summarize(rows)
    document = {"model": next((mark.get("model") for mark in marks), None), "label": next((mark.get("label") for mark in marks), None),
                "requests": len(rows), "unmatched": sum(1 for row in rows if row.get("unmatched")), "kv_store": store, "clock_shared": shared,
                "clock": next((mark.get("clock") for mark in marks if mark.get("clock")), None),
                "cache_states": [{key: record.get(key) for key in ("event", "sequence", "history_tokens", "http_status", "answer", "utc")}
                                 for record in read_requests(folder) if record.get("kind") == "cache_state"], "summary": summary}
    (folder / "summary.json").write_text(json.dumps(document, indent=1) + "\n")
    print(markdown(summary))
    if not shared:
        print("\n" + CLOCK_APART[0].upper() + CLOCK_APART[1:] + ".")


def excerpt(arguments):
    marks = [record for record in read_requests(arguments.folder) if record.get("kind") == "mark" and record.get("log_offset") is not None]
    log_path = arguments.api_log or next((mark["api_log"] for mark in marks if mark.get("api_log")), None)
    if log_path is None or not marks:
        raise SystemExit("turn_floor: the run recorded no engine log to take an excerpt of")
    start, end = min(mark["log_offset"] for mark in marks), max(mark["log_offset"] for mark in marks)
    before = {}
    with open(log_path, "rb") as handle:
        back = max(0, start - STORE_LOOKBACK_BYTES)
        handle.seek(back)
        if back:
            handle.readline()
        while handle.tell() < start:
            raw = handle.readline()
            if not raw:
                break
            if raw.startswith(b"{\"event\":\"kv_store_report\""):
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                before[(event.get("boot_pid"), event.get("residentd_pid"), event.get("rank"))] = raw.decode("utf-8", "replace")
        handle.seek(start)
        lines = handle.read(end - start).decode("utf-8", "replace").split("\n")
    baseline = "".join(before.values())
    kept, first, last = set(), {}, {}
    for index, line in enumerate(lines):
        if line.startswith("{\"event\":\"kv_store_report\""):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (event.get("boot_pid"), event.get("residentd_pid"), event.get("rank"))
            first.setdefault(key, index)
            last[key] = index
        elif line.startswith(("batch_step ", "batch_rejected ", "{\"event\":\"request_measurements\"")) or NOTE.match(line):
            kept.add(index)
    kept.update(first.values())
    kept.update(last.values())
    Path(arguments.output).write_text(baseline + "".join(lines[index] + "\n" for index in sorted(kept)))
    print(json.dumps({"excerpt": arguments.output, "log": str(log_path), "log_base": start - len(baseline.encode()), "log_start": start, "bytes": end - start,
                      "lines": len(lines), "kept": len(kept), "baseline_reports": len(before)}))


def numbers(text):
    return [int(value) for value in text.split(",") if value]


def build_parser():
    parser = argparse.ArgumentParser(description="Time one engine's turns by token ids and split each first token's time into its parts from the engine's log")
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--endpoint", required=True)
    runner.add_argument("--tokens", required=True)
    runner.add_argument("--output", required=True)
    runner.add_argument("--label", required=True)
    runner.add_argument("--api-log")
    runner.add_argument("--arms", type=lambda text: text.split(","), default=["turns"])
    runner.add_argument("--contexts", type=numbers, default=[20000, 50000, 135000, 300000])
    runner.add_argument("--tails", type=numbers, default=[50, 300, 2400])
    runner.add_argument("--repeats", type=int, default=3)
    runner.add_argument("--spec-repeats", type=int, default=1)
    runner.add_argument("--reply-tokens", type=int, default=16)
    runner.add_argument("--decode-tokens", type=int, default=96)
    runner.add_argument("--served-tokens", type=int, default=0)
    runner.add_argument("--once-from", type=int, default=0)
    runner.add_argument("--decode-context", type=int)
    runner.add_argument("--session-tokens", type=int)
    runner.add_argument("--wave-tokens", type=int, default=60000)
    runner.add_argument("--wave-lead", type=float, default=2.0)
    runner.add_argument("--probes-solo", type=int, default=5)
    runner.add_argument("--event")
    runner.add_argument("--salt", type=int)
    runner.add_argument("--timeout", type=float, default=3600.0)
    reporter = commands.add_parser("parts")
    reporter.add_argument("folder")
    reporter.add_argument("--api-log")
    reporter.add_argument("--log-base", type=int, default=0)
    cutter = commands.add_parser("excerpt")
    cutter.add_argument("folder")
    cutter.add_argument("--api-log")
    cutter.add_argument("--output", required=True)
    tables = commands.add_parser("report")
    tables.add_argument("folders", nargs="+")
    return parser


def main():
    arguments = build_parser().parse_args()
    if arguments.command == "run":
        unknown = [arm for arm in arguments.arms if arm not in ("turns", "decode", "openings", "wave", "resume", "truth")]
        if unknown:
            raise SystemExit(f"turn_floor: unknown arms {unknown}; the arms are turns, decode, openings, wave, resume and truth")
        if "resume" in arguments.arms and not arguments.event:
            raise SystemExit("turn_floor: the resume arm needs --event, what happened to the session since its last turn")
        run(arguments)
    elif arguments.command == "parts":
        parts(arguments)
    elif arguments.command == "excerpt":
        excerpt(arguments)
    else:
        report(arguments)


if __name__ == "__main__":
    main()

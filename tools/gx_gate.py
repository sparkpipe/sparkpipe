#!/usr/bin/env python3
"""Gate GX against a live SparkPipe C API: a request's tokens and logprobs
depend on the request, never on what the cache held.

The gate runs in steps, one cache lifetime at a time, and the operator changes
the cache between them. `steps` prints the order and the change each step
needs: `clear` (engine and API down, the lane's KV snapshots and the API's
prefix index removed, up again) or `evict` (engine and API down so the chains
go to NVMe and the device pool is dropped, up again).

  chain            a committed prefix is extended by a new tail, and three-turn
                   sessions run to their last prompt, with nothing interrupted
  park:GROUP       the same requests on a cleared cache, up to the eviction
  restore:GROUP    the rest of them on the restored chains
  cold:TAIL        every last prompt of that tail size, on a cleared cache
  crowd            planned with --crowd N: on a cleared cache each prefix is
                   committed alone, then its tails are sent N at a time

Every request names its cuts, the token counts where its structure ends a
prefill wave: where the shared prefix ends, one token before each reply
began and where each reply ends. An arm's last request is the same request
in every arm, cuts included.

`verdict` compares the arms: reuse against cold, and restored against
uninterrupted. A unit passes only when tokens and logprobs are equal, the
reusing run resumed at the cut its structure names and the cold run reports
no cached tokens. Anything not run, not cached, resumed short of that cut or
not cold leaves the verdict INCOMPLETE, never PASS.

The crowd step is measured, not judged: `verdict` reports how many requests
sent together equal the same request sent alone. Kernels are not yet
batch-invariant, so a difference there does not fail the gate.

A step that stops is run again after a cache clear and does only the units
it has not finished. A park step is always run whole for the units its
restore step still needs. A request that fails while the engine stays whole
is recorded on its unit, and the step goes on.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

KIND = "sparkpipe.gx-gate.v3"
PREFIX_CASES = ("prefix", "kept", "opening")
SESSION_CASES = ("session", "drafted", "committed")
CASES = PREFIX_CASES + SESSION_CASES
TOKEN_LOW = 1000
TOKEN_HIGH = 30000
BLOCK = 256
GRID = 1024
MID_BLOCK = 21
KEPT_OTHER = 8
SPREAD = 1.5
LOG_WAIT_SECONDS = 5.0
LIMIT = "limit"
CUT_RULE = {}
RESTORE_SETTLE_SECONDS = 1.0
LOG_TAIL_BYTES = 16 << 20
STORE_COUNTERS = ("save_pages", "save_ns", "save_failures", "restore_count", "restore_pages", "restore_ns", "restore_misses", "restore_failures", "store_files",
                  "store_used_bytes", "store_evictions", "pool_resident_pages", "pool_physical_pages", "pool_device_bytes", "pool_adopted_pages", "pool_evicted_entries",
                  "spill_write_bytes")
STORE_SLOWEST = ("save_ns", "restore_ns")
ENGINE_COUNTERS = ("completed", "prefix_hits", "prefix_misses", "prefix_hit_tokens", "stale_prefix_recomputes", "rejected_waves_busy", "rejected_waves_not_found",
                   "rejected_waves_capacity", "retry_deadlines", "verify_lanes", "draft_tokens", "accepted_draft_tokens")
LOAD_COUNTERS = ("rejected_waves_capacity", "retry_deadlines")
ENGINE_IDENTITY = ("adapter_id", "model_id", "model_revision", "driver_program", "driver_artifact_sha256")


class GateError(Exception):
    pass


class RequestError(GateError):
    def __init__(self, message, counted):
        super().__init__(message)
        self.counted = counted


class UnitError(GateError):
    pass


def token_run(seed, label, count):
    out = []
    index = 0
    while len(out) < count:
        digest = hashlib.sha256(("%s/%s/%d" % (seed, label, index)).encode()).digest()
        out.extend(TOKEN_LOW + int.from_bytes(digest[i:i + 4], "little") % (TOKEN_HIGH - TOKEN_LOW) for i in range(0, 32, 4))
        index += 1
    return out[:count]


def prompt_digest(tokens):
    return hashlib.sha256(struct.pack("<%dI" % len(tokens), *tokens)).hexdigest()


def chat_layer_cut(prefix_tokens):
    here = os.path.dirname(os.path.abspath(__file__))
    if "rule" not in CUT_RULE:
        source = next((path for path in (os.path.join(here, "prompt_cuts.py"), os.path.join(here, "serving", "prompt_cuts.py")) if os.path.exists(path)), None)
        if source is None:
            raise GateError("prompt_cuts.py is not beside gx_gate.py: the opening case takes its cuts from the chat layer's own rule")
        spec = importlib.util.spec_from_file_location("prompt_cuts", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        CUT_RULE["rule"] = module.structural_cut
    return CUT_RULE["rule"](prefix_tokens)


def context_tokens(case, nominal):
    if case == "kept":
        return nominal // GRID * GRID - BLOCK + 1
    return nominal + MID_BLOCK


def head_blocks(unit):
    tools = unit["context"] // 4
    return tools, tools + unit["context"] // 2


def head_cuts(unit):
    return sorted({cut for cut in (chat_layer_cut(end) for end in head_blocks(unit)) if cut > 0} | {unit["context"]})


def head_keeps(unit):
    return head_cuts(unit)[1:]


def build_plan(arguments):
    tails = sorted(set(arguments.tails))
    reserve = 3 * max(tails) + 2 * arguments.reply_tokens + max(arguments.max_tokens, arguments.served_tokens) + BLOCK
    room = (arguments.context_limit - reserve - MID_BLOCK) // GRID * GRID
    if room < 2 * GRID:
        raise GateError("a context of %d positions leaves no room for a prefix beside three %d-token tails" % (arguments.context_limit, max(tails)))
    units, skipped = [], []
    wanted = {"prefix": arguments.contexts, "session": arguments.contexts, "kept": arguments.kept_contexts, "drafted": arguments.drafted_contexts,
              "committed": arguments.committed_contexts, "opening": arguments.opening_contexts}
    for case in CASES:
        if case not in arguments.cases:
            continue
        kept = []
        for nominal in sorted({arguments.context_limit if value == LIMIT else value for value in wanted[case]}):
            fitted = min(nominal, room)
            if fitted < nominal and kept and fitted < SPREAD * kept[-1]:
                skipped.append({"case": case, "nominal": nominal, "reason": "the deployment serves %d positions; the largest prefix that fits, %d, is under %.1f times the %d already run" % (arguments.context_limit, fitted, SPREAD, kept[-1])})
                continue
            kept.append(fitted)
            for tail in tails:
                units.append({"id": "%s/%d/%d" % (case, nominal, tail), "case": case, "nominal": nominal, "context": context_tokens(case, fitted), "tail": tail})
    if not units:
        raise GateError("the plan holds no unit")
    for unit in units:
        if unit["case"] == "opening" and len(head_cuts(unit)) != 3:
            raise GateError("a head of %d tokens is too short for a cut after each of its first two blocks" % unit["context"])
    small = max([unit["nominal"] for unit in units if unit["context"] <= arguments.small_context], default=None)
    for unit in units:
        unit["group"] = small if unit["context"] <= arguments.small_context else unit["nominal"]
    return {"tails": tails, "units": units, "skipped": skipped}


def step_order(plan):
    order = [("chain", "none")]
    change = "clear"
    for group in sorted({unit["group"] for unit in plan["units"]}):
        order.append(("park:%d" % group, change))
        order.append(("restore:%d" % group, "evict"))
        change = "none"
    for tail in plan["tails"]:
        order.append(("cold:%d" % tail, "clear"))
    if (plan.get("options") or {}).get("crowd"):
        order.append(("crowd", "clear"))
    return order


def step_units(state, name):
    kind, _, value = name.partition(":")
    return [unit for unit in state["units"] if kind == "chain" or (kind == "crowd" and unit["case"] == "prefix") or (kind == "cold" and unit["tail"] == int(value)) or
            (kind in ("park", "restore") and unit["group"] == int(value))]


def step_status(state, name):
    record = state["steps"].get(name)
    if record is None:
        return "open"
    if "finished" in record:
        return "done"
    return "stopped" if "interrupted" in record else "partial"


class Prompts:
    def __init__(self, state):
        self.seed = state["seed"]
        self.cache = {}

    def run(self, label, count):
        key = (label, count)
        if key not in self.cache:
            self.cache = {held: tokens for held, tokens in self.cache.items() if len(tokens) < GRID}
            self.cache[key] = token_run(self.seed, label, count)
        return self.cache[key]

    def prefix(self, unit):
        return self.run("%s/%d/prefix" % (unit["case"], unit["nominal"]), unit["context"])

    def part(self, unit, name):
        return self.run("%s/%s" % (unit["id"], name), unit["tail"])

    def last(self, unit, replies):
        if unit["case"] == "opening":
            shared = head_blocks(unit)[1]
            return self.prefix(unit)[:shared] + self.run("%s/block" % unit["id"], unit["context"] - shared) + self.part(unit, "tail")
        if unit["case"] in PREFIX_CASES:
            return self.prefix(unit) + self.part(unit, "tail")
        return self.prefix(unit) + self.part(unit, "first") + replies[0] + self.part(unit, "second") + replies[1] + self.part(unit, "third")

    def other(self, unit):
        if unit["case"] == "opening":
            shared = head_blocks(unit)[1]
            return self.prefix(unit)[:shared] + self.run("%s/%d/other" % (unit["case"], unit["nominal"]), unit["context"] - shared + KEPT_OTHER)
        return self.prefix(unit) + self.run("%s/%d/other" % (unit["case"], unit["nominal"]), KEPT_OTHER)


def turn_cuts(unit, replies):
    if unit["case"] == "opening":
        return head_cuts(unit)
    cuts = [unit["context"]]
    end = unit["context"] + unit["tail"]
    for reply in replies:
        cuts += [end - 1, end + len(reply)]
        end += len(reply) + unit["tail"]
    return sorted(set(cuts))


def resume_cut(unit, replies):
    if unit["case"] == "opening":
        return chat_layer_cut(head_blocks(unit)[1])
    if unit["case"] in PREFIX_CASES:
        return unit["context"]
    if unit["case"] == "committed":
        return unit["context"] + 2 * unit["tail"] + len(replies[0]) + len(replies[1])
    return unit["context"] + 2 * unit["tail"] + len(replies[0]) - 1


class Api:
    def __init__(self, base, timeout, log_path):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.log_path = log_path

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=30) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise GateError("GET %s failed: %s" % (path, error))

    def health(self):
        health = self.get("/health")
        if health.get("status") != "ok" or health.get("connected_ranks") != health.get("ranks"):
            raise GateError("the API is not healthy on every rank: %s" % json.dumps(health))
        return health

    def log_size(self):
        if not self.log_path:
            return None
        try:
            return os.path.getsize(self.log_path)
        except OSError:
            return None

    def measurements(self, offset, digest):
        if offset is None:
            return {}
        deadline = time.monotonic() + LOG_WAIT_SECONDS
        while True:
            found = None
            try:
                with open(self.log_path, errors="replace") as log:
                    log.seek(offset)
                    for line in log:
                        if '"request_measurements"' in line and digest in line:
                            found = line
            except OSError:
                return {}
            if found is not None:
                try:
                    event = json.loads(found[found.index("{"):])
                except ValueError:
                    return {}
                out = {"stale_prefix_recomputes": event.get("stale_prefix_recomputes"), "queue_ms": round((event["first_dispatch_ns"] - event["accepted_ns"]) / 1e6, 3),
                       "request_id": event.get("request_id"), "boot_pid": event.get("boot_pid"), "accepted_ns": event["accepted_ns"], "first_dispatch_ns": event["first_dispatch_ns"],
                       "engine": {name: event.get(name) for name in ENGINE_IDENTITY}}
                ready = [entry[1] for entry in event.get("tokens") or []]
                if ready:
                    out["first_ready_ns"] = ready[0]
                    out["last_ready_ns"] = ready[-1]
                    out["first_token_ms"] = round((ready[0] - event["accepted_ns"]) / 1e6, 3)
                    out["last_token_ms"] = round((ready[-1] - event["accepted_ns"]) / 1e6, 3)
                if len(ready) > 1 and ready[-1] > ready[0]:
                    out["decode_tokens_per_second"] = round((len(ready) - 1) * 1e9 / (ready[-1] - ready[0]), 3)
                return out
            if time.monotonic() > deadline:
                return {}
            time.sleep(0.2)

    def counters(self):
        size = self.log_size()
        if size is None:
            return {}
        ranks, engine, boot = {}, None, None
        try:
            with open(self.log_path, errors="replace") as log:
                log.seek(max(0, size - LOG_TAIL_BYTES))
                for line in log:
                    if '"kv_store_report"' not in line and '"engine_measurements"' not in line:
                        continue
                    try:
                        event = json.loads(line[line.index("{"):])
                    except ValueError:
                        continue
                    if event.get("boot_pid") != boot:
                        ranks, engine, boot = {}, None, event.get("boot_pid")
                    if event.get("event") == "kv_store_report":
                        ranks[event.get("rank")] = event
                    else:
                        engine = event
        except OSError:
            return {}
        out = {"boot_pid": boot, "ranks_reporting": len(ranks), "store": {name: sum(report.get(name) or 0 for report in ranks.values()) for name in STORE_COUNTERS},
               "slowest_rank": {name: max([report.get(name) or 0 for report in ranks.values()], default=0) for name in STORE_SLOWEST}}
        if engine is not None:
            out["engine"] = {name: engine.get(name) for name in ENGINE_COUNTERS}
        return out

    def complete(self, prompt, max_tokens, options):
        commit = max_tokens == 0
        body = dict(options, prompt_token_ids=prompt, temperature=0, stream=True)
        body.update({"commit_only": True} if commit else {"max_tokens": max_tokens})
        request = urllib.request.Request(self.base + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        offset = self.log_size()
        tokens, logprobs, usage, reason, first_ns = [], [], {}, None, None
        started_utc, sent_ns = stamp_ms(), time.monotonic_ns()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                closed = False
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        closed = True
                        break
                    event = json.loads(data)
                    if "error" in event:
                        raise RequestError("the stream carried an error: %s" % json.dumps(event["error"])[:300], True)
                    if event.get("tokens"):
                        first_ns = first_ns if first_ns is not None else time.monotonic_ns()
                        tokens.extend(int(token) for token in event["tokens"])
                    logprobs.extend(event.get("token_logprobs") or [])
                    usage = event.get("usage") or usage
                    reason = ((event.get("choices") or [{}])[0]).get("finish_reason") or reason
                if not closed:
                    raise RequestError("the stream ended before [DONE]", True)
        except urllib.error.HTTPError as error:
            raise RequestError("HTTP %d: %s" % (error.code, error.read()[:300].decode(errors="replace")), error.code >= 500)
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise RequestError(str(error), False)
        end_ns = time.monotonic_ns()
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if not isinstance(cached, int) or usage.get("prompt_tokens") != len(prompt):
            raise RequestError("the reply carries no cached_tokens or another prompt length: %s" % json.dumps(usage)[:300], True)
        if commit and (tokens or reason != "commit" or usage.get("published_tokens") != len(prompt)):
            raise RequestError("the commit generated tokens or does not report its prompt as published: %s %s" % (reason, json.dumps(usage)[:300]), True)
        if not commit and (not tokens or reason is None):
            raise RequestError("the reply carries no tokens or no finish reason: %s" % json.dumps(usage)[:300], True)
        digest = prompt_digest(prompt)
        result = {"prompt_tokens": len(prompt), "prompt_sha256": digest, "cached_tokens": cached, "prefilled_tokens": len(prompt) - cached, "tokens": tokens,
                  "finish_reason": reason, "seconds": round((end_ns - sent_ns) / 1e9, 3), "started_utc": started_utc, "finished_utc": stamp_ms(),
                  "sent_ns": sent_ns, "first_ns": first_ns, "end_ns": end_ns}
        if "cuts" in options:
            result["cuts"] = options["cuts"]
        if "keep" in options:
            result["keep"] = options["keep"]
        if "logprobs" in options:
            result["logprobs"] = logprobs
        result.update(self.measurements(offset, digest))
        return result


class Step:
    def __init__(self, api, state, name, save, rest_seconds=0.0, rest_over_seconds=0.0, slices=None):
        self.api = api
        self.state = state
        self.name = name
        self.save = save
        self.rest_seconds = rest_seconds
        self.rest_over_seconds = rest_over_seconds
        self.slices = slices
        self.options = state["options"]
        self.prompts = Prompts(state)
        self.record = None
        self.attempt = None
        self.lock = threading.RLock()

    def units(self, cases, nominal=None, tail=None):
        return [unit for unit in self.state["units"] if unit["case"] in cases and nominal in (None, unit["nominal"]) and tail in (None, unit["tail"])]

    def nominals(self, group=None):
        return sorted({unit["nominal"] for unit in self.state["units"] if group in (None, unit["group"])})

    def settled(self, record):
        return set(record.get("results", {})) | set(record.get("errors", {}))

    def pending(self, units):
        return [unit for unit in units if unit["id"] not in self.settled(self.record)]

    def send(self, label, prompt, max_tokens, options, cold=False):
        try:
            result = self.api.complete(prompt, max_tokens, options)
        except RequestError as error:
            with self.lock:
                self.attempt["requests"] += 1 if error.counted else 0
                self.record["requests"].append({"label": label, "attempt": len(self.record["attempts"]), "prompt_tokens": len(prompt), "max_tokens": max_tokens, "error": str(error)})
                print("%s %s %s prompt=%d FAILED %s" % (stamp(), self.name, label, len(prompt), error), flush=True)
            self.api.health()
            raise UnitError(str(error))
        with self.lock:
            self.attempt["requests"] += 1
            self.attempt["completed"] += 1
            identity = result.pop("engine", None)
            if identity is not None:
                self.attempt.setdefault("engine", identity)
            rests = self.rest_seconds > 0 and result["seconds"] >= self.rest_over_seconds
            self.record["requests"].append(dict({key: value for key, value in result.items() if key not in ("tokens", "logprobs")}, label=label, attempt=len(self.record["attempts"]),
                                                max_tokens=max_tokens, completion_tokens=len(result["tokens"]), rested_seconds=self.rest_seconds if rests else 0))
            print("%s %s %s prompt=%d cached=%d out=%d %.1fs" % (stamp(), self.name, label, result["prompt_tokens"], result["cached_tokens"], len(result["tokens"]), result["seconds"]), flush=True)
        if cold and result["cached_tokens"] != 0:
            raise GateError("%s reports %d cached tokens where this step needs a cold run: clear the cache before the step" % (label, result["cached_tokens"]))
        if rests:
            time.sleep(self.rest_seconds)
        return result

    def settle(self, unit, action):
        try:
            action()
        except UnitError as error:
            with self.lock:
                self.record["errors"][unit["id"]] = str(error)
        with self.lock:
            self.save()

    def warmed(self, units):
        unit = units[0]
        label = "%s/%d/warm" % (unit["case"], unit["nominal"])
        try:
            if unit["case"] == "kept":
                self.send(label, self.prompts.other(unit), 1, {"speculation": False, "cuts": [unit["context"]], "keep": [unit["context"]]}, cold=True)
            elif unit["case"] == "opening":
                self.send(label, self.prompts.other(unit), 1, {"speculation": False, "cuts": head_cuts(unit), "keep": head_keeps(unit)}, cold=True)
            else:
                self.send(label, self.prompts.prefix(unit), 0, {}, cold=True)
        except UnitError as error:
            for unit in units:
                self.record["errors"][unit["id"]] = "its prefix did not warm: %s" % error
            self.save()
            return False
        self.save()
        return True

    def measured(self, unit, replies):
        options = {"logprobs": self.options["logprobs"], "speculation": False, "cuts": turn_cuts(unit, replies or [])}
        if unit["case"] == "kept":
            options["keep"] = [unit["context"]]
        if unit["case"] == "opening":
            options["keep"] = head_keeps(unit)
        return options

    def measure(self, unit, replies, cold=False):
        with self.lock:
            prompt = self.prompts.last(unit, replies)
        result = self.send(unit["id"], prompt, self.options["max_tokens"], self.measured(unit, replies), cold=cold)
        result["resume_cut"] = resume_cut(unit, replies)
        with self.lock:
            self.record["results"][unit["id"]] = result

    def turns(self, unit, first):
        options = {} if unit["case"] == "drafted" else {"speculation": False}
        prompt = self.prompts.prefix(unit) + self.prompts.part(unit, "first")
        replies = []
        for turn, name in enumerate(("second", "third")):
            cuts = turn_cuts(unit, replies)
            reply = self.send("%s/turn%d" % (unit["id"], turn + 1), prompt, self.options["reply_tokens"], dict(options, cuts=cuts), cold=first and turn == 0)
            if unit["case"] == "committed":
                self.send("%s/commit%d" % (unit["id"], turn + 1), prompt + reply["tokens"], 0, {"cuts": sorted(set(cuts + [len(prompt) - 1]))})
            replies.append(reply["tokens"])
            prompt = prompt + reply["tokens"] + self.prompts.part(unit, name)
        self.record["replies"][unit["id"]] = replies
        return replies

    def sessions(self, units, action):
        first = True
        for unit in units:
            self.settle(unit, lambda unit=unit, first=first: action(unit, first))
            first = False

    def chain(self, _):
        for nominal in self.nominals():
            for case in PREFIX_CASES:
                units = self.pending(self.units((case,), nominal))
                if units and self.warmed(units):
                    for unit in units:
                        self.settle(unit, lambda unit=unit: self.measure(unit, None))
            self.served(nominal)
            for case in SESSION_CASES:
                self.sessions(self.pending(self.units((case,), nominal)), lambda unit, first: self.measure(unit, self.turns(unit, first)))

    def served(self, nominal):
        units = [unit for unit in self.units(("prefix",), nominal) if unit["id"] in self.record["results"]]
        label = "prefix/%d/served" % nominal
        if not self.options.get("served_tokens") or not units or str(nominal) in self.record.setdefault("served", {}) or label in self.record["errors"]:
            return
        unit = min(units, key=lambda unit: (unit["tail"] < 40, unit["tail"]))
        try:
            self.record["served"][str(nominal)] = self.send(label, self.prompts.last(unit, None), self.options["served_tokens"], {"cuts": turn_cuts(unit, [])})
        except UnitError as error:
            self.record["errors"][label] = str(error)
        self.save()

    def park(self, group):
        restored = self.settled(self.state["steps"].get("restore:%d" % group) or {})
        self.record.update(results={}, replies={}, errors={})
        for nominal in self.nominals(group):
            for case in SESSION_CASES:
                self.sessions([unit for unit in self.units((case,), nominal) if unit["id"] not in restored], lambda unit, first: self.turns(unit, first))
        for nominal in self.nominals(group):
            for case in PREFIX_CASES:
                units = [unit for unit in self.units((case,), nominal) if unit["id"] not in restored]
                if units:
                    self.warmed(units)

    def restore(self, group):
        parked = self.state["steps"].get("park:%d" % group) or {}
        if "finished" not in parked:
            raise GateError("restore:%d needs a finished park:%d" % (group, group))
        for nominal in self.nominals(group):
            for unit in self.pending(self.units(CASES, nominal)):
                if unit["id"] in parked["errors"]:
                    self.record["errors"][unit["id"]] = "its park step failed: %s" % parked["errors"][unit["id"]]
                    continue
                before = self.api.counters()
                self.settle(unit, lambda unit=unit: self.measure(unit, parked["replies"].get(unit["id"])))
                self.restored(unit, before)
        self.save()

    def restored(self, unit, before):
        result = self.record["results"].get(unit["id"])
        if result is None or not before.get("store"):
            return
        time.sleep(RESTORE_SETTLE_SECONDS)
        after = self.api.counters()
        if after.get("boot_pid") != before.get("boot_pid") or not after.get("store"):
            return
        pages = after["store"]["restore_pages"] - before["store"]["restore_pages"]
        pool_pages = after["store"]["pool_physical_pages"]
        result["restore"] = {"pages": pages, "rank_seconds_sum": round((after["store"]["restore_ns"] - before["store"]["restore_ns"]) / 1e9, 3),
                             "slowest_rank_seconds": round((after["slowest_rank"]["restore_ns"] - before["slowest_rank"]["restore_ns"]) / 1e9, 3),
                             "bytes": pages * after["store"]["pool_device_bytes"] // pool_pages if pool_pages else None,
                             "misses": after["store"]["restore_misses"] - before["store"]["restore_misses"]}
        self.save()

    def cold(self, tail):
        chained = self.state["steps"].get("chain") or {}
        if "finished" not in chained:
            raise GateError("cold:%d needs a finished chain step: a session's last prompt holds its replies" % tail)
        for unit in self.pending(self.units(CASES, tail=tail)):
            if unit["case"] in SESSION_CASES and unit["id"] not in chained["replies"]:
                self.record["errors"][unit["id"]] = "its session did not finish in the chain step"
                continue
            self.settle(unit, lambda unit=unit: self.measure(unit, chained["replies"].get(unit["id"]), cold=True))
        self.save()

    def together(self, units):
        stopped = []

        def work(unit):
            try:
                self.settle(unit, lambda: self.measure(unit, None))
            except GateError as error:
                stopped.append(error)
        threads = [threading.Thread(target=work, args=(unit,)) for unit in units]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if stopped:
            raise stopped[0]

    def crowd(self, _):
        size = self.options["crowd"]
        for nominal in self.nominals():
            units = self.pending(self.units(("prefix",), nominal))
            if not units or not self.warmed(units):
                continue
            for start in range(0, len(units), size):
                self.together(units[start:start + size])

    def logged_foreign(self):
        before, after = self.attempt["counters_before"], self.attempt["counters_after"]
        counts = [(side.get("engine") or {}).get("completed") for side in (before, after)]
        if before.get("boot_pid") != after.get("boot_pid") or None in counts:
            return None
        return max(0, counts[1] - counts[0] - self.attempt["completed"])

    def run(self):
        kind, _, value = self.name.partition(":")
        known = {name for name, _ in step_order(self.state)}
        if self.name not in known:
            raise GateError("unknown step %s; this state has: %s" % (self.name, " ".join(sorted(known))))
        follower = "restore:%s" % value if kind == "park" else self.name
        if step_status(self.state, follower) == "done" and step_status(self.state, self.name) == "done":
            print("%s %s is done" % (stamp(), self.name), flush=True)
            return True
        before = self.api.health()
        if before.get("live_requests"):
            raise GateError("the API has %d live requests: the gate runs alone" % before["live_requests"])
        self.record = self.state["steps"].setdefault(self.name, {"requests": [], "results": {}, "replies": {}, "errors": {}, "attempts": []})
        self.record.pop("finished", None)
        self.record.pop("interrupted", None)
        self.attempt = {"started": stamp(), "requests": 0, "completed": 0, "health_before": before, "counters_before": self.api.counters(),
                        "rest": {"seconds": self.rest_seconds, "over_seconds": self.rest_over_seconds}, "log": {"path": self.api.log_path, "start": self.api.log_size()}}
        self.record["attempts"].append(self.attempt)
        self.save()
        try:
            getattr(self, kind)(int(value) if value else None)
        except GateError as error:
            self.record["interrupted"] = str(error)
        self.attempt["counters_after"] = self.api.counters()
        try:
            after = self.api.health()
            self.attempt["health_after"] = after
            self.attempt["foreign_requests"] = after.get("served", 0) - before.get("served", 0) - self.attempt["requests"]
        except GateError as error:
            self.record.setdefault("interrupted", str(error))
            self.attempt["foreign_requests"] = self.logged_foreign()
        self.attempt["finished"] = stamp()
        self.attempt["log"]["end"] = self.api.log_size()
        self.attempt["log"]["copy"] = self.slice()
        if "interrupted" not in self.record:
            self.record["finished"] = stamp()
        self.save()
        return "finished" in self.record

    def slice(self):
        log = self.attempt["log"]
        if self.slices is None or log["start"] is None or log["end"] is None or log["end"] < log["start"]:
            return None
        name = "%s.%d.apilog" % (self.name.replace(":", "-"), len(self.record["attempts"]))
        try:
            with open(log["path"], "rb") as source, open(os.path.join(self.slices, name), "wb") as copy:
                source.seek(log["start"])
                left = log["end"] - log["start"]
                while left > 0:
                    chunk = source.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    copy.write(chunk)
                    left -= len(chunk)
        except OSError:
            return None
        return name


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp_ms():
    now = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + ".%03dZ" % int(now % 1 * 1000)


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def logprob_delta(left, right, limit):
    worst = 0.0
    for a, b in list(zip(left, right))[:limit]:
        pairs = [{entry[0]: entry[1] for entry in side[1:]} for side in (a, b)]
        values = [(a[0][1], b[0][1])] + [(pairs[0][token], pairs[1][token]) for token in set(pairs[0]) & set(pairs[1])]
        for x, y in values:
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                worst = max(worst, abs(x - y))
    return worst


def compare(left, right):
    token = first_difference(left["tokens"], right["tokens"])
    logprobs = [side.get("logprobs") or [] for side in (left, right)]
    logprob = first_difference(logprobs[0], logprobs[1])
    limit = len(left["tokens"]) if token is None else token + 1
    return {"equal": token is None and logprob is None and bool(logprobs[0]), "first_token_difference": token, "first_logprob_difference": logprob,
            "max_logprob_delta": round(logprob_delta(logprobs[0], logprobs[1], limit), 9)}


def judge(state):
    steps = state["steps"]
    names = {"chain": lambda unit: "chain", "restore": lambda unit: "restore:%d" % unit["group"], "cold": lambda unit: "cold:%d" % unit["tail"]}
    verdicts = []
    for unit in state["units"]:
        records = {arm: steps.get(name(unit)) or {} for arm, name in names.items()}
        arms = {arm: record.get("results", {}).get(unit["id"]) for arm, record in records.items()}
        errors = {arm: record.get("errors", {}).get(unit["id"]) for arm, record in records.items() if record.get("errors", {}).get(unit["id"])}
        entry = dict(unit, resume_cut=next((result["resume_cut"] for result in arms.values() if result and "resume_cut" in result), None),
                     cached_tokens={arm: result["cached_tokens"] if result else None for arm, result in arms.items()},
                     seconds={arm: result["seconds"] if result else None for arm, result in arms.items()},
                     first_token_ms={arm: result.get("first_token_ms") if result else None for arm, result in arms.items()})
        if errors:
            entry["errors"] = errors
        chain, restore, cold = arms["chain"], arms["restore"], arms["cold"]
        if "chain" in errors or "cold" in errors:
            entry["reuse"] = "ERROR"
        elif chain is None or cold is None:
            entry["reuse"] = "NOT-RUN"
        elif cold["cached_tokens"] != 0 or cold["prompt_sha256"] != chain["prompt_sha256"]:
            entry["reuse"] = "NOT-COLD"
        elif chain["cached_tokens"] == 0:
            entry["reuse"] = "NOT-CACHED"
        else:
            entry["reuse_detail"] = compare(chain, cold)
            entry["reuse"] = "FAIL" if not entry["reuse_detail"]["equal"] else "SHORT" if chain["cached_tokens"] < chain.get("resume_cut", 1) else "PASS"
        if "chain" in errors or "restore" in errors:
            entry["restored"] = "ERROR"
        elif chain is None or restore is None:
            entry["restored"] = "NOT-RUN"
        elif restore["prompt_sha256"] != chain["prompt_sha256"]:
            entry["restored"] = "DIVERGED"
        elif restore["cached_tokens"] == 0:
            entry["restored"] = "LOST"
        else:
            entry["restored_detail"] = compare(restore, chain)
            entry["restored"] = "PASS" if entry["restored_detail"]["equal"] else "FAIL"
            entry["recomputed_after_restore"] = chain["cached_tokens"] - restore["cached_tokens"]
        if restore is not None and cold is not None and restore["prompt_sha256"] == cold["prompt_sha256"]:
            entry["restored_against_cold"] = compare(restore, cold)
        results = (entry["reuse"], entry["restored"])
        entry["verdict"] = "FAIL" if "FAIL" in results else "PASS" if results == ("PASS", "PASS") else "INCOMPLETE"
        if (state["options"].get("crowd") or 0) and unit["case"] == "prefix":
            crowd = (steps.get("crowd") or {}).get("results", {}).get(unit["id"])
            if crowd is None or chain is None or crowd["prompt_sha256"] != chain["prompt_sha256"]:
                entry["concurrent"] = "NOT-RUN"
            else:
                entry["concurrent_detail"] = compare(crowd, chain)
                entry["concurrent"] = "EQUAL" if entry["concurrent_detail"]["equal"] else "DIFFERENT"
                entry["cached_tokens"]["crowd"] = crowd["cached_tokens"]
        verdicts.append(entry)
    alone = all(not attempt.get("foreign_requests") for step in steps.values() for attempt in step.get("attempts", []))
    finished = all(step_status(state, name) == "done" for name, _ in step_order(state))
    every = [entry["verdict"] for entry in verdicts]
    verdict = "FAIL" if "FAIL" in every else "PASS" if alone and finished and all(value == "PASS" for value in every) else "INCOMPLETE"
    return verdicts, alone, verdict


def waited(state, names):
    counts = {}
    for name in names:
        counts[name] = dict.fromkeys(LOAD_COUNTERS, 0)
        for attempt in (state["steps"].get(name) or {}).get("attempts", []):
            before, after = attempt.get("counters_before") or {}, attempt.get("counters_after") or {}
            start = (before.get("engine") or {}) if before.get("boot_pid") == after.get("boot_pid") else {}
            for counter in LOAD_COUNTERS:
                counts[name][counter] += max(0, ((after.get("engine") or {}).get(counter) or 0) - (start.get(counter) or 0))
    return counts


def describe(detail):
    if detail is None:
        return "-"
    if detail["equal"]:
        return "equal"
    token = "tokens equal" if detail["first_token_difference"] is None else "token %d" % detail["first_token_difference"]
    return "%s, logprob %s, max delta %.6f" % (token, detail["first_logprob_difference"], detail["max_logprob_delta"])


def load(path):
    with open(path) as handle:
        state = json.load(handle)
    if state.get("kind") != KIND:
        raise GateError("%s is not a %s state" % (path, KIND))
    return state


def save(path, state):
    temporary = "%s.tmp-%d" % (path, os.getpid())
    with open(temporary, "w") as handle:
        json.dump(state, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def numbers(text):
    return [int(value) for value in text.split(",") if value]


def contexts(text):
    return [value if value == LIMIT else int(value) for value in text.split(",") if value]


def command_plan(arguments):
    if os.path.exists(arguments.state):
        raise GateError("%s exists: a state is planned once, because its prompts must be new to the cache" % arguments.state)
    plan = build_plan(arguments)
    state = dict(plan, kind=KIND, seed=arguments.seed, model=arguments.model, release=arguments.release, context_limit=arguments.context_limit,
                 options={"max_tokens": arguments.max_tokens, "reply_tokens": arguments.reply_tokens, "logprobs": arguments.logprobs, "crowd": arguments.crowd,
                          "served_tokens": arguments.served_tokens}, steps={}, block=[])
    save(arguments.state, state)
    return command_steps(arguments)


def command_steps(arguments):
    state = load(arguments.state)
    for name, change in step_order(state):
        units = step_units(state, name)
        settled = len([unit for unit in units if unit["id"] in set((state["steps"].get(name) or {}).get("results", {})) | set((state["steps"].get(name) or {}).get("errors", {}))])
        print("%s %s %s %d/%d" % (name, change, step_status(state, name), settled, len(units)))
    return 0


def command_run(arguments):
    state = load(arguments.state)
    step = Step(Api(arguments.api, arguments.timeout, arguments.api_log), state, arguments.step, lambda: save(arguments.state, state), arguments.rest_seconds, arguments.rest_over_seconds,
                os.path.dirname(os.path.abspath(arguments.state)) if arguments.api_log else None)
    passed = step.run()
    if not passed:
        print("GX-STEP-STOPPED %s: %s" % (arguments.step, step.record["interrupted"]), flush=True)
    return 0 if passed else 1


def command_note(arguments):
    state = load(arguments.state)
    note = {"at": stamp()}
    for pair in arguments.pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise GateError("a note is KEY=VALUE, not %r" % pair)
        note[key] = value
    state.setdefault("block", []).append(note)
    save(arguments.state, state)
    return 0


def command_check(arguments):
    state = load(arguments.state)
    verdicts, alone, _ = judge(state)
    wanted = arguments.steps or ["chain"] + ["cold:%d" % tail for tail in state["tails"]]
    problems, findings = [], []
    for name in wanted:
        if step_status(state, name) != "done":
            problems.append("step %s is %s" % (name, step_status(state, name)))
    if not alone:
        problems.append("the API served a request the gate did not send")
    requests = [request for name in wanted for request in (state["steps"].get(name) or {}).get("requests", [])]
    if any(unit["case"] == "prefix" for unit in state["units"]) and not any(request.get("max_tokens") == 0 and request.get("finish_reason") == "commit" for request in requests):
        problems.append("no commit was confirmed")
    for request in requests:
        if "error" in request:
            problems.append("%s failed: %s" % (request["label"], request["error"]))
        elif not arguments.no_log and ("queue_ms" not in request or (request["completion_tokens"] and "first_token_ms" not in request)):
            problems.append("%s has no record in the API log: check --api-log" % request["label"])
    for entry in verdicts:
        for arm in ("chain", "cold"):
            result = (state["steps"].get(arm if arm == "chain" else "cold:%d" % entry["tail"]) or {}).get("results", {}).get(entry["id"])
            if result is not None and (len(result.get("logprobs") or []) != len(result["tokens"]) or not all(isinstance(row, list) and row for row in result["logprobs"])):
                problems.append("%s %s carries %d logprob rows for %d tokens" % (entry["id"], arm, len(result.get("logprobs") or []), len(result["tokens"])))
        if entry["reuse"] == "FAIL":
            findings.append("%s differs between reuse and cold: %s" % (entry["id"], describe(entry["reuse_detail"])))
        elif entry["reuse"] != "PASS":
            problems.append("%s reuse is %s" % (entry["id"], entry["reuse"]))
    for name, counts in waited(state, wanted).items():
        if counts["retry_deadlines"]:
            problems.append("%d requests of step %s waited out the in-flight budget on a wave the ranks kept refusing" % (counts["retry_deadlines"], name))
        if counts["rejected_waves_capacity"]:
            findings.append("the ranks refused %d waves of step %s for capacity; each waits and goes out again with the same span" % (counts["rejected_waves_capacity"], name))
    for line in findings:
        print("check: finding: %s" % line)
    for line in problems:
        print("check: %s" % line)
    print("check: %s" % ("ok" if not problems else "%d problems" % len(problems)))
    return 1 if problems else 0


def command_verdict(arguments):
    state = load(arguments.state)
    verdicts, alone, verdict = judge(state)
    receipt = dict(state, verdicts=verdicts, alone=alone, verdict=verdict, load=waited(state, sorted(state["steps"])))
    together = [entry["concurrent"] for entry in verdicts if "concurrent" in entry]
    if together:
        receipt["concurrent"] = {name.lower().replace("-", "_"): together.count(name) for name in ("EQUAL", "DIFFERENT", "NOT-RUN")}
    if arguments.output:
        save(arguments.output, receipt)
    for entry in verdicts:
        print("%-22s context=%-7d cut=%-7s cached chain/restore/cold=%s/%s/%s reuse=%s (%s) restored=%s (%s)" % (
            entry["id"], entry["context"], entry["resume_cut"], entry["cached_tokens"]["chain"], entry["cached_tokens"]["restore"], entry["cached_tokens"]["cold"],
            entry["reuse"], describe(entry.get("reuse_detail")), entry["restored"], describe(entry.get("restored_detail"))))
        for arm, error in sorted(entry.get("errors", {}).items()):
            print("  %s failed: %s" % (arm, error))
        if entry.get("concurrent") == "DIFFERENT":
            print("  sent together it differs from sent alone: %s" % describe(entry["concurrent_detail"]))
    for skipped in state["skipped"]:
        print("skipped %s/%d: %s" % (skipped["case"], skipped["nominal"], skipped["reason"]))
    for name, _ in step_order(state):
        if step_status(state, name) != "done":
            print("step %s is %s%s" % (name, step_status(state, name), ": " + state["steps"][name]["interrupted"] if step_status(state, name) == "stopped" else ""))
    if together:
        print("concurrent equal=%d different=%d not-run=%d" % (together.count("EQUAL"), together.count("DIFFERENT"), together.count("NOT-RUN")))
    print("waves refused for capacity=%d requests that waited out the in-flight budget=%d" % tuple(sum(counts[counter] for counts in receipt["load"].values()) for counter in LOAD_COUNTERS))
    print("alone=%s" % ("yes" if alone else "no"))
    print("verdict=%s" % verdict)
    return {"PASS": 0, "INCOMPLETE": 2}.get(verdict, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="write a new state and print its steps")
    plan.add_argument("--state", required=True)
    plan.add_argument("--model", required=True)
    plan.add_argument("--release", required=True)
    plan.add_argument("--context-limit", type=int, required=True, help="positions the deployment serves")
    plan.add_argument("--contexts", type=contexts, default=[8000, 100000, 300000], help="prefix sizes in tokens; 'limit' is the deployment's context limit, fitted beside the tails")
    plan.add_argument("--kept-contexts", type=contexts, default=[8000, 20000])
    plan.add_argument("--drafted-contexts", type=contexts, default=[8000])
    plan.add_argument("--committed-contexts", type=contexts, default=[8000], help="sessions whose replies are committed at once, so the next turn resumes where the reply ends")
    plan.add_argument("--opening-contexts", type=contexts, default=[8000],
                      help="heads of three blocks whose cuts come from the chat layer's rule, with state at the second block's cut and the head's end; a second conversation shares the first two blocks")
    plan.add_argument("--tails", type=numbers, default=[1, 40, 500, 5000])
    plan.add_argument("--small-context", type=int, default=32768, help="prefixes up to this size share one park and restore step")
    plan.add_argument("--cases", type=lambda text: [name for name in text.split(",") if name], default=list(CASES))
    plan.add_argument("--max-tokens", type=int, default=32)
    plan.add_argument("--reply-tokens", type=int, default=96)
    plan.add_argument("--logprobs", type=int, default=4)
    plan.add_argument("--crowd", type=int, default=0, help="add a step that sends each committed prefix's tails this many at a time; 0 for none")
    plan.add_argument("--served-tokens", type=int, default=128, help="tokens of the timing-only reply after each context's prefix units, speculation at the engine default; 0 for none")
    plan.add_argument("--seed", default=str(time.time_ns()))
    steps = commands.add_parser("steps", help="print each step, the cache change before it, its state and its settled units")
    steps.add_argument("--state", required=True)
    run = commands.add_parser("run", help="run one step against the API, or the rest of a step that stopped")
    run.add_argument("step")
    run.add_argument("--state", required=True)
    run.add_argument("--api", required=True, help="C API base URL, e.g. http://127.0.0.1:8461")
    run.add_argument("--api-log", help="the API's log, for time to first token, decode rate and counters")
    run.add_argument("--timeout", type=float, default=7200)
    run.add_argument("--rest-seconds", type=float, default=0.0, help="idle time after a request that ran --rest-over-seconds or longer")
    run.add_argument("--rest-over-seconds", type=float, default=60.0)
    verdict = commands.add_parser("verdict", help="compare the arms and write the receipt")
    verdict.add_argument("--state", required=True)
    verdict.add_argument("--output")
    note = commands.add_parser("note", help="append facts about the block to the state: what was active, the lane, a cache change and how long it took")
    note.add_argument("--state", required=True)
    note.add_argument("pairs", nargs="+", metavar="KEY=VALUE")
    check = commands.add_parser("check", help="the tool's own checks on the steps run so far: exit 0 unless a step, a request or a record is not what the gate needs")
    check.add_argument("--state", required=True)
    check.add_argument("--steps", type=lambda text: [name for name in text.split(",") if name], default=None)
    check.add_argument("--no-log", action="store_true", help="do not require the API log's record of each request")
    arguments = parser.parse_args(argv)
    if arguments.command == "plan":
        unknown = sorted(set(arguments.cases) - set(CASES))
        if unknown or not arguments.cases:
            parser.error("unknown cases %s; known: %s" % (unknown, ",".join(CASES)))
        if not arguments.tails or min(arguments.tails) < 1 or arguments.max_tokens < 1 or arguments.reply_tokens < 1 or arguments.served_tokens < 0:
            parser.error("--tails, --max-tokens and --reply-tokens must be at least 1, and --served-tokens at least 0")
        if arguments.crowd == 1 or arguments.crowd < 0:
            parser.error("--crowd is 0 or at least 2")
    try:
        return {"plan": command_plan, "steps": command_steps, "run": command_run, "verdict": command_verdict, "note": command_note, "check": command_check}[arguments.command](arguments)
    except GateError as error:
        print("gx_gate: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

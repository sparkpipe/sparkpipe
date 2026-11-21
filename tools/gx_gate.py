#!/usr/bin/env python3
"""Gate GX against a live SparkPipe C API: a request's tokens and logprobs
depend on the request, never on what the cache held.

The gate runs in steps, one cache lifetime at a time, and the operator changes
the cache between them. `steps` prints the order and the change each step
needs: `clear` (engine and API down, the lane's KV snapshots and the API's
prefix index removed, up again) or `evict` (engine and API down so the chains
go to NVMe and the device pool is dropped, up again).

  chain            a warmed prefix is extended by a new tail, and three-turn
                   sessions run to their last prompt, with nothing interrupted
  park:CONTEXT     the same requests on a cleared cache, up to the eviction
  restore:CONTEXT  the rest of them on the restored chains
  cold:TAIL        every last prompt of that tail size, on a cleared cache

`verdict` compares the arms: reuse against cold, and restored against
uninterrupted. A unit passes only when tokens and logprobs are equal, the
reusing run reports cached tokens and the cold run reports none. Anything not
run, not cached or not cold leaves the verdict INCOMPLETE, never PASS.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

KIND = "sparkpipe.gx-gate.v1"
PREFIX_CASES = ("prefix", "declared")
SESSION_CASES = ("session", "drafted")
CASES = PREFIX_CASES + SESSION_CASES
TOKEN_LOW = 1000
TOKEN_HIGH = 30000
BLOCK = 256
GRID = 1024
MID_BLOCK = 21
SPREAD = 1.5
LOG_WAIT_SECONDS = 5.0
LOG_TAIL_BYTES = 16 << 20
STORE_COUNTERS = ("save_pages", "save_ns", "save_failures", "restore_count", "restore_pages", "restore_ns", "restore_misses", "restore_failures", "store_files",
                  "store_used_bytes", "store_evictions", "pool_resident_pages", "pool_adopted_pages", "pool_evicted_entries", "spill_write_bytes")
ENGINE_COUNTERS = ("completed", "prefix_hits", "prefix_misses", "prefix_hit_tokens", "stale_prefix_recomputes", "rejected_waves_busy", "rejected_waves_not_found",
                   "verify_lanes", "draft_tokens", "accepted_draft_tokens")


class GateError(Exception):
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


def context_tokens(case, nominal):
    if case == "declared":
        return nominal // GRID * GRID - BLOCK + 1
    return nominal + MID_BLOCK


def build_plan(arguments):
    tails = sorted(set(arguments.tails))
    reserve = 3 * max(tails) + 2 * arguments.reply_tokens + arguments.max_tokens + BLOCK
    room = (arguments.context_limit - reserve - MID_BLOCK) // GRID * GRID
    if room < 2 * GRID:
        raise GateError("a context of %d positions leaves no room for a prefix beside three %d-token tails" % (arguments.context_limit, max(tails)))
    units, skipped = [], []
    wanted = {"prefix": arguments.contexts, "session": arguments.contexts, "declared": arguments.declared_contexts, "drafted": arguments.drafted_contexts}
    for case in CASES:
        if case not in arguments.cases:
            continue
        kept = []
        for nominal in sorted(set(wanted[case])):
            fitted = min(nominal, room)
            if fitted < nominal and kept and fitted < SPREAD * kept[-1]:
                skipped.append({"case": case, "nominal": nominal, "reason": "the deployment serves %d positions; the largest prefix that fits, %d, is under %.1f times the %d already run" % (arguments.context_limit, fitted, SPREAD, kept[-1])})
                continue
            kept.append(fitted)
            for tail in tails:
                units.append({"id": "%s/%d/%d" % (case, nominal, tail), "case": case, "nominal": nominal, "context": context_tokens(case, fitted), "tail": tail})
    if not units:
        raise GateError("the plan holds no unit")
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
    return order


class Prompts:
    def __init__(self, state):
        self.seed = state["seed"]
        self.reply_tokens = state["options"]["reply_tokens"]
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
        if unit["case"] in PREFIX_CASES:
            return self.prefix(unit) + self.part(unit, "tail")
        return self.prefix(unit) + self.part(unit, "first") + replies[0] + self.part(unit, "second") + replies[1] + self.part(unit, "third")


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
                out = {"stale_prefix_recomputes": event.get("stale_prefix_recomputes"), "queue_ms": round((event["first_dispatch_ns"] - event["accepted_ns"]) / 1e6, 3)}
                if event.get("tokens"):
                    out["first_token_ms"] = round((event["tokens"][0][1] - event["accepted_ns"]) / 1e6, 3)
                    out["last_token_ms"] = round((event["tokens"][-1][1] - event["accepted_ns"]) / 1e6, 3)
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
        out = {"ranks_reporting": len(ranks), "store": {name: sum(report.get(name) or 0 for report in ranks.values()) for name in STORE_COUNTERS}}
        if engine is not None:
            out["engine"] = {name: engine.get(name) for name in ENGINE_COUNTERS}
        return out

    def complete(self, prompt, max_tokens, options):
        body = dict(options, prompt_token_ids=prompt, max_tokens=max_tokens, temperature=0)
        request = urllib.request.Request(self.base + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        offset = self.log_size()
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                reply = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise GateError("a completion of %d tokens failed with HTTP %d: %s" % (len(prompt), error.code, error.read()[:300].decode(errors="replace")))
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise GateError("a completion of %d tokens failed: %s" % (len(prompt), error))
        seconds = time.monotonic() - started
        usage = reply.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        tokens = reply.get("tokens")
        if not isinstance(tokens, list) or not tokens or not isinstance(cached, int) or usage.get("prompt_tokens") != len(prompt):
            raise GateError("a completion reply carries no tokens, no cached_tokens or another prompt length: %s" % json.dumps(reply)[:300])
        digest = prompt_digest(prompt)
        result = {"prompt_tokens": len(prompt), "prompt_sha256": digest, "cached_tokens": cached, "tokens": [int(token) for token in tokens],
                  "finish_reason": reply.get("finish_reason") or ((reply.get("choices") or [{}])[0]).get("finish_reason"), "seconds": round(seconds, 3)}
        if "logprobs" in options:
            result["logprobs"] = reply.get("token_logprobs")
        result.update(self.measurements(offset, digest))
        return result


class Step:
    def __init__(self, api, state, name):
        self.api = api
        self.state = state
        self.name = name
        self.options = state["options"]
        self.prompts = Prompts(state)
        self.record = {"started": stamp(), "requests": [], "results": {}, "replies": {}}

    def units(self, cases, nominal=None, tail=None):
        return [unit for unit in self.state["units"] if unit["case"] in cases and nominal in (None, unit["nominal"]) and tail in (None, unit["tail"])]

    def nominals(self, group=None):
        return sorted({unit["nominal"] for unit in self.state["units"] if group in (None, unit["group"])})

    def send(self, label, prompt, max_tokens, options, cold=False):
        result = self.api.complete(prompt, max_tokens, options)
        self.record["requests"].append(dict({key: value for key, value in result.items() if key not in ("tokens", "logprobs")}, label=label, max_tokens=max_tokens))
        print("%s %s %s prompt=%d cached=%d out=%d %.1fs" % (stamp(), self.name, label, result["prompt_tokens"], result["cached_tokens"], len(result["tokens"]), result["seconds"]), flush=True)
        if cold and result["cached_tokens"] != 0:
            raise GateError("%s reports %d cached tokens where this step needs a cold run: clear the cache, or plan a new state" % (label, result["cached_tokens"]))
        return result

    def warm(self, unit):
        self.send("%s/%d/warm" % (unit["case"], unit["nominal"]), self.prompts.prefix(unit), 1, {"speculation": False}, cold=True)

    def measured(self, unit):
        options = {"logprobs": self.options["logprobs"], "speculation": False}
        if unit["case"] == "declared":
            options["cache_points"] = [unit["context"] - 1]
        return options

    def measure(self, unit, prompt, cold=False):
        self.record["results"][unit["id"]] = self.send(unit["id"], prompt, self.options["max_tokens"], self.measured(unit), cold=cold)

    def turns(self, unit, first):
        options = {"speculation": False} if unit["case"] == "session" else {}
        prompt = self.prompts.prefix(unit) + self.prompts.part(unit, "first")
        replies = []
        for turn, name in enumerate(("second", "third")):
            reply = self.send("%s/turn%d" % (unit["id"], turn + 1), prompt, self.options["reply_tokens"], options, cold=first and turn == 0)
            replies.append(reply["tokens"])
            prompt = prompt + reply["tokens"] + self.prompts.part(unit, name)
        self.record["replies"][unit["id"]] = replies
        return prompt

    def chain(self):
        for nominal in self.nominals():
            for case in PREFIX_CASES:
                units = self.units((case,), nominal)
                if units:
                    self.warm(units[0])
                for unit in units:
                    self.measure(unit, self.prompts.last(unit, None))
            for case in SESSION_CASES:
                for index, unit in enumerate(self.units((case,), nominal)):
                    self.measure(unit, self.turns(unit, index == 0))

    def park(self, group):
        for nominal in self.nominals(group):
            for case in SESSION_CASES:
                for index, unit in enumerate(self.units((case,), nominal)):
                    self.turns(unit, index == 0)
        for nominal in self.nominals(group):
            for case in PREFIX_CASES:
                units = self.units((case,), nominal)
                if units:
                    self.warm(units[0])

    def restore(self, group):
        parked = self.state["steps"].get("park:%d" % group) or {}
        if parked.get("error") or "finished" not in parked:
            raise GateError("restore:%d needs a finished park:%d" % (group, group))
        for nominal in self.nominals(group):
            for unit in self.units(PREFIX_CASES, nominal):
                self.measure(unit, self.prompts.last(unit, None))
            for unit in self.units(SESSION_CASES, nominal):
                self.measure(unit, self.prompts.last(unit, parked["replies"][unit["id"]]))

    def cold(self, tail):
        chained = self.state["steps"].get("chain") or {}
        if chained.get("error") or "finished" not in chained:
            raise GateError("cold:%d needs a finished chain step: a session's last prompt holds its replies" % tail)
        for unit in self.units(CASES, tail=tail):
            self.measure(unit, self.prompts.last(unit, chained["replies"].get(unit["id"])), cold=True)

    def run(self):
        kind, _, value = self.name.partition(":")
        known = {name for name, _ in step_order(self.state)}
        if self.name not in known:
            raise GateError("unknown step %s; this state has: %s" % (self.name, " ".join(sorted(known))))
        before = self.api.health()
        self.record["health_before"] = before
        if before.get("live_requests"):
            raise GateError("the API has %d live requests: the gate runs alone" % before["live_requests"])
        self.record["counters_before"] = self.api.counters()
        try:
            if kind == "chain":
                self.chain()
            else:
                getattr(self, kind)(int(value))
        except GateError as error:
            self.record["error"] = str(error)
        try:
            after = self.api.health()
            self.record["health_after"] = after
            self.record["foreign_requests"] = after.get("served", 0) - before.get("served", 0) - len(self.record["requests"])
        except GateError as error:
            self.record.setdefault("error", str(error))
        self.record["counters_after"] = self.api.counters()
        self.record["finished"] = stamp()
        self.state["steps"][self.name] = self.record
        return "error" not in self.record


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    verdicts = []
    for unit in state["units"]:
        arms = {"chain": (steps.get("chain") or {}).get("results", {}).get(unit["id"]),
                "restore": (steps.get("restore:%d" % unit["group"]) or {}).get("results", {}).get(unit["id"]),
                "cold": (steps.get("cold:%d" % unit["tail"]) or {}).get("results", {}).get(unit["id"])}
        entry = dict(unit, cached_tokens={arm: result["cached_tokens"] if result else None for arm, result in arms.items()},
                     seconds={arm: result["seconds"] if result else None for arm, result in arms.items()})
        chain, restore, cold = arms["chain"], arms["restore"], arms["cold"]
        if chain is None or cold is None:
            entry["reuse"] = "NOT-RUN"
        elif cold["cached_tokens"] != 0 or cold["prompt_sha256"] != chain["prompt_sha256"]:
            entry["reuse"] = "NOT-COLD"
        elif chain["cached_tokens"] == 0:
            entry["reuse"] = "NOT-CACHED"
        else:
            entry["reuse_detail"] = compare(chain, cold)
            entry["reuse"] = "PASS" if entry["reuse_detail"]["equal"] else "FAIL"
        if chain is None or restore is None:
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
        verdicts.append(entry)
    alone = all(step.get("foreign_requests") == 0 and "error" not in step for step in steps.values())
    every = [entry["verdict"] for entry in verdicts]
    verdict = "FAIL" if "FAIL" in every else "PASS" if alone and all(value == "PASS" for value in every) else "INCOMPLETE"
    return verdicts, alone, verdict


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


def command_plan(arguments):
    if os.path.exists(arguments.state):
        raise GateError("%s exists: a state is planned once, because its prompts must be new to the cache" % arguments.state)
    plan = build_plan(arguments)
    state = dict(plan, kind=KIND, seed=arguments.seed, model=arguments.model, release=arguments.release, context_limit=arguments.context_limit,
                 options={"max_tokens": arguments.max_tokens, "reply_tokens": arguments.reply_tokens, "logprobs": arguments.logprobs}, steps={})
    save(arguments.state, state)
    return command_steps(arguments)


def command_steps(arguments):
    state = load(arguments.state)
    for name, change in step_order(state):
        done = state["steps"].get(name) or {}
        print("%s %s %s" % (name, change, "failed" if done.get("error") else "done" if "finished" in done else "open"))
    return 0


def command_run(arguments):
    state = load(arguments.state)
    step = Step(Api(arguments.api, arguments.timeout, arguments.api_log), state, arguments.step)
    passed = step.run()
    save(arguments.state, state)
    if not passed:
        print("GX-STEP-FAILED %s: %s" % (arguments.step, step.record["error"]), flush=True)
    return 0 if passed else 1


def command_verdict(arguments):
    state = load(arguments.state)
    verdicts, alone, verdict = judge(state)
    receipt = dict(state, verdicts=verdicts, alone=alone, verdict=verdict)
    if arguments.output:
        save(arguments.output, receipt)
    for entry in verdicts:
        print("%-22s context=%-7d cached chain/restore/cold=%s/%s/%s reuse=%s (%s) restored=%s (%s)" % (
            entry["id"], entry["context"], entry["cached_tokens"]["chain"], entry["cached_tokens"]["restore"], entry["cached_tokens"]["cold"],
            entry["reuse"], describe(entry.get("reuse_detail")), entry["restored"], describe(entry.get("restored_detail"))))
    for skipped in state["skipped"]:
        print("skipped %s/%d: %s" % (skipped["case"], skipped["nominal"], skipped["reason"]))
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
    plan.add_argument("--contexts", type=numbers, default=[8000, 100000, 300000])
    plan.add_argument("--declared-contexts", type=numbers, default=[8000, 20000])
    plan.add_argument("--drafted-contexts", type=numbers, default=[8000])
    plan.add_argument("--tails", type=numbers, default=[1, 40, 500, 5000])
    plan.add_argument("--small-context", type=int, default=32768, help="prefixes up to this size share one park and restore step")
    plan.add_argument("--cases", type=lambda text: [name for name in text.split(",") if name], default=list(CASES))
    plan.add_argument("--max-tokens", type=int, default=32)
    plan.add_argument("--reply-tokens", type=int, default=48)
    plan.add_argument("--logprobs", type=int, default=4)
    plan.add_argument("--seed", default=str(time.time_ns()))
    steps = commands.add_parser("steps", help="print the steps, the cache change before each, and what is done")
    steps.add_argument("--state", required=True)
    run = commands.add_parser("run", help="run one step against the API")
    run.add_argument("step")
    run.add_argument("--state", required=True)
    run.add_argument("--api", required=True, help="C API base URL, e.g. http://127.0.0.1:8461")
    run.add_argument("--api-log", help="the API's log, for time to first token and stale-prefix counts")
    run.add_argument("--timeout", type=float, default=7200)
    verdict = commands.add_parser("verdict", help="compare the arms and write the receipt")
    verdict.add_argument("--state", required=True)
    verdict.add_argument("--output")
    arguments = parser.parse_args(argv)
    if arguments.command == "plan":
        unknown = sorted(set(arguments.cases) - set(CASES))
        if unknown or not arguments.cases:
            parser.error("unknown cases %s; known: %s" % (unknown, ",".join(CASES)))
        if not arguments.tails or min(arguments.tails) < 1 or arguments.max_tokens < 1 or arguments.reply_tokens < 1:
            parser.error("--tails, --max-tokens and --reply-tokens must be at least 1")
    try:
        return {"plan": command_plan, "steps": command_steps, "run": command_run, "verdict": command_verdict}[arguments.command](arguments)
    except GateError as error:
        print("gx_gate: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import hashlib
import http.client
import json
import os
import random
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LCG_MOD = 2147483647
LCG_MUL = 48271


class E2EFail(Exception):
    pass


def require(condition, message, evidence=None):
    if not condition:
        raise E2EFail(message if evidence is None else f"{message}: {evidence}")


def lcg_tokens(length, seed):
    value = seed % LCG_MOD or 1
    out = []
    for _ in range(length):
        value = (value * LCG_MUL) % LCG_MOD
        out.append(1000 + value % 150000)
    return out


def prompt_sha(tokens):
    packed = struct.pack(f"<{len(tokens)}I", *tokens)
    return hashlib.sha256(packed).hexdigest()


def completion_path(chat):
    return "/v1/chat/completions" if chat else "/v1/completions"


class Api:
    def __init__(self, spec, timeout=600):
        self.host = spec["host"]
        self.port = spec["port"]
        self.chat = spec.get("chat", False)
        self.timeout = timeout

    def get(self, path):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=30)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, response.read().decode()
        finally:
            connection.close()

    def post(self, body):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request("POST", completion_path(self.chat),
                                json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read().decode()
        finally:
            connection.close()

    def open_stream(self, body):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request("POST", completion_path(self.chat),
                                json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response, connection
        except Exception:
            connection.close()
            raise

    def request(self, prompt_tokens, max_tokens, **extra):
        body = {"prompt_token_ids": prompt_tokens, "max_tokens": max_tokens,
                "temperature": 0.0}
        body.update(extra)
        return self.post(body)

    def stream(self, prompt_tokens, max_tokens, **extra):
        body = {"prompt_token_ids": prompt_tokens, "max_tokens": max_tokens,
                "temperature": 0.0, "stream": True}
        body.update(extra)
        return self.open_stream(body)

    def request_text(self, prompt, max_tokens, **extra):
        body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
        body.update(extra)
        return self.post(body)


class Stream:
    def __init__(self, response, connection):
        self.response = response
        self.connection = connection

    @property
    def status(self):
        return self.response.status

    def events(self):
        if self.response.status != 200:
            detail = self.response.read(400).decode(errors="replace")
            raise E2EFail(
                f"stream returned {self.response.status}: {detail}")
        while True:
            line = self.response.fp.readline()
            if not line:
                return
            text = line.decode(errors="replace").strip()
            if text.startswith("data: "):
                payload = text[6:]
                yield payload
                if payload == "[DONE]":
                    return

    def close(self):
        try:
            self.response.fp.read(65536)
        except Exception:
            pass
        try:
            self.connection.close()
        except Exception:
            pass


def stream_collect(stream):
    chunks = 0
    tokens = []
    finish = None
    done = False
    for event in stream.events():
        if event == "[DONE]":
            done = True
            continue
        data = json.loads(event)
        chunks += 1
        for choice in data.get("choices", []):
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
        for token in data.get("tokens", []):
            if isinstance(token, dict):
                token = token.get("id", token.get("token"))
            tokens.append(token)
    require(done, "stream ended without [DONE]")
    require(finish is not None, "no chunk carried finish_reason")
    return chunks, tokens, finish


def text_response(body):
    data = json.loads(body)
    for choice in data.get("choices", []):
        text = choice.get("text")
        if text is None:
            text = choice.get("message", {}).get("content")
        if text is None:
            text = choice.get("delta", {}).get("content")
        if text is not None:
            return text
    raise E2EFail(f"no text in response: {body[:200]}")


def response_tokens(body):
    data = json.loads(body)
    finish = None
    for choice in data.get("choices", []):
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    if isinstance(data.get("tokens"), list):
        return data["tokens"], finish or data.get("finish_reason")
    for choice in data.get("choices", []):
        if "token_ids" in choice:
            return choice["token_ids"], finish
        text = choice.get("text") or choice.get("delta", {}).get("content", "")
        return text, finish
    raise E2EFail(f"no choices in response: {body[:200]}")


class Evidence:
    def __init__(self, config, model):
        spec = config["models"][model]
        self.ssh_host = spec.get("ssh_host")
        self.api_log = spec.get("api_log")

    def available(self):
        return bool(self.api_log)

    def _read(self):
        if self.ssh_host:
            run = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", self.ssh_host,
                 "tail -n 4000 " + self.api_log],
                capture_output=True, text=True, timeout=60)
            return run.stdout
        with open(self.api_log, "r", errors="replace") as handle:
            return handle.read()

    def measurements(self, prompt_sha_value=None):
        out = []
        for line in self._read().split("\n"):
            if '"request_measurements"' not in line:
                continue
            start = line.find("{")
            if start < 0:
                continue
            try:
                row = json.loads(line[start:])
            except ValueError:
                continue
            if prompt_sha_value is None or row.get("prompt_sha256") == prompt_sha_value:
                out.append(row)
        return out


def measurement_for(evidence, tokens, deadline=30.0):
    sha = prompt_sha(tokens)
    cutoff = time.time() + deadline
    while time.time() < cutoff:
        rows = evidence.measurements(sha)
        if rows:
            return rows[-1]
        time.sleep(0.5)
    raise E2EFail(
        f"no request_measurements line for prompt sha256 {sha[:16]} within {deadline}s")


TESTS = {}
MODELS_CONFIG = {}


def test(tier, name):
    def register(function):
        TESTS[(tier, name)] = function
        return function
    return register


@test("A", "health_identity")
def t_health_identity(model, spec, api, evidence, receipt):
    status, body = api.get("/health")
    require(status == 200, f"/health returned {status}", body[:200])
    data = json.loads(body)
    require(data.get("status") == "ok", "engine degraded", body)
    require(data.get("tokenizer") is True, "tokenizer not ready")
    ranks = spec.get("ranks")
    if ranks:
        require(data.get("ranks") == ranks,
                f"ranks {data.get('ranks')} != {ranks}")
        require(data.get("connected_ranks") == ranks,
                f"connected {data.get('connected_ranks')} != {ranks}")
    status, body = api.get("/health/live")
    require(status == 200 and json.loads(body).get("ready") is True, body)
    status, body = api.get("/v1/models")
    require(status == 200 and json.loads(body).get("object") == "list", body[:200])
    receipt["health"] = data


@test("A", "completion_contract")
def t_completion_contract(model, spec, api, evidence, receipt):
    prompt = lcg_tokens(32, 11)
    status, body = api.request(prompt, 8)
    require(status == 200, f"completion returned {status}", body[:300])
    tokens, finish = response_tokens(body)
    require(finish == "length", f"finish_reason {finish} != length")
    require(len(tokens) == 8, f"{len(tokens)} tokens for max_tokens 8")
    status, body = api.request(prompt, 4, stop_token_ids=[tokens[0]])
    require(status == 200, body[:300])
    _, finish = response_tokens(body)
    require(finish == "stop", f"stop token ignored, finish {finish}")
    status, _ = api.get("/no/such/path")
    require(status == 404, f"unknown path returned {status}")
    connection = http.client.HTTPConnection(api.host, api.port, timeout=30)
    connection.request("POST", completion_path(api.chat), "{not json",
                       {"Content-Type": "application/json"})
    response = connection.getresponse()
    bad = response.status
    response.read()
    connection.close()
    require(bad in (400, 500), f"malformed body returned {bad}")
    status, body = api.get("/health")
    require(status == 200, "engine unhealthy after error surface", body[:200])
    receipt["max_tokens_honored"] = len(tokens)


@test("A", "greedy_determinism")
def t_greedy_determinism(model, spec, api, evidence, receipt):
    prompt = lcg_tokens(64, 23)
    _, first = api.request(prompt, 32)
    tokens_a, _ = response_tokens(first)
    _, second = api.request(prompt, 32)
    tokens_b, _ = response_tokens(second)
    require(tokens_a == tokens_b, "greedy decode not deterministic across repeats")
    receipt["tokens"] = tokens_a
    if evidence.available():
        row = measurement_for(evidence, prompt)
        require(row.get("temperature") == 0.0, "engine saw temperature != 0")
        require(row.get("finish_reason") == "length", row.get("finish_reason"))
        receipt["measurement"] = {k: row.get(k) for k in
                                  ("prompt_tokens", "cached_prompt_tokens",
                                   "stale_prefix_recomputes", "finish_reason")}


@test("A", "seeded_sampling")
def t_seeded_sampling(model, spec, api, evidence, receipt):
    prompt = lcg_tokens(48, 37)
    kwargs = {"temperature": 1.0, "seed": 4242, "top_p": 1.0, "top_k": 0}
    status, body = api.request(prompt, 24, **kwargs)
    require(status == 200, f"seeded sampling returned {status}", body[:300])
    tokens_a, _ = response_tokens(body)
    status, body = api.request(prompt, 24, **kwargs)
    require(status == 200, body[:300])
    tokens_b, _ = response_tokens(body)
    require(tokens_a == tokens_b,
            "same seed produced different streams",
            f"{tokens_a[:8]}... != {tokens_b[:8]}...")
    receipt["tokens"] = tokens_a


@test("A", "streaming_order")
def t_streaming_order(model, spec, api, evidence, receipt):
    prompt = lcg_tokens(48, 51)
    _, plain = api.request(prompt, 16)
    reference, finish_plain = response_tokens(plain)
    stream = None
    try:
        stream = Stream(*api.stream(prompt, 16))
        chunks, tokens, finish = stream_collect(stream)
    finally:
        if stream:
            stream.close()
    require(tokens == reference,
            "stream token order differs from non-stream",
            f"{tokens[:8]}... != {reference[:8]}...")
    require(finish == finish_plain, f"stream finish {finish} != {finish_plain}")
    require(chunks >= 2, "expected at least two stream chunks")
    receipt["chunks"] = chunks
    receipt["tokens"] = tokens


@test("A", "logprobs_shape")
def t_logprobs(model, spec, api, evidence, receipt):
    if not spec.get("logprobs_supported", True):
        raise E2EFail("config says logprobs unsupported")
    prompt = lcg_tokens(32, 61)
    status, body = api.request(prompt, 8, logprobs=1)
    require(status == 200, f"logprobs request returned {status}", body[:300])
    data = json.loads(body)
    scores = data.get("logprobs") or data.get("choices", [{}])[0].get("logprobs")
    require(scores is not None, "no logprobs in response", body[:300])
    require(len(scores) == 8, f"{len(scores)} logprobs for 8 tokens")
    for value in scores:
        entry = value.get("logprob") if isinstance(value, dict) else value
        require(isinstance(entry, (int, float)), f"logprob not numeric: {value}")
    receipt["logprobs_head"] = scores[:3]


@test("A", "prefix_share")
def t_prefix_share(model, spec, api, evidence, receipt):
    require(evidence.available(),
            "prefix_share needs api_log evidence; add api_log to the model config")
    base = lcg_tokens(256, 71)
    _, cold = api.request(base, 8)
    row = measurement_for(evidence, base)
    require(row.get("cached_prompt_tokens", 0) <= row.get("prompt_tokens", 0),
            "cold request reported a full cache hit", row)
    extended = base + lcg_tokens(32, 72)
    _, warm = api.request(extended, 8)
    warm_tokens, _ = response_tokens(warm)
    row = measurement_for(evidence, extended)
    require(row.get("cached_prompt_tokens", 0) == len(base),
            "prefix not shared on extension",
            f"cached {row.get('cached_prompt_tokens')} want {len(base)}")
    require(row.get("stale_prefix_recomputes") == 0,
            "stale prefix recompute on a live prefix", row)
    other = lcg_tokens(256, 999)
    api.request(other, 4)
    row = measurement_for(evidence, other)
    require(row.get("cached_prompt_tokens", 0) == 0,
            "unrelated prompt scored a cache hit", row)
    _, fresh = api.request(extended, 8)
    fresh_tokens, _ = response_tokens(fresh)
    require(warm_tokens == fresh_tokens,
            "cached-prefix decode differs from fresh decode")
    receipt["base_prompt_tokens"] = len(base)
    receipt["cached_on_extension"] = row.get("cached_prompt_tokens")


@test("A", "continuous_batching")
def t_continuous_batching(model, spec, api, evidence, receipt):
    prompts = [lcg_tokens(24 + 8 * i, 100 + i) for i in range(15)]
    references = []
    for prompt in prompts:
        _, plain = api.request(prompt, 24)
        tokens, _ = response_tokens(plain)
        references.append(tokens)
    started = time.time()
    results = [None] * len(prompts)

    def run(index):
        try:
            results[index] = api.request(prompts[index], 24)
        except Exception as error:
            results[index] = (0, f"transport: {error}")

    threads = []
    for index in range(len(prompts)):
        thread = threading.Thread(target=run, args=(index,))
        thread.start()
        threads.append(thread)
        time.sleep(0.2)
    for thread in threads:
        thread.join()
    wall = time.time() - started
    total = 0
    for index, (status, body) in enumerate(results):
        require(status == 200, f"batch request {index} returned {status}",
                str(body)[:200])
        tokens, _ = response_tokens(body)
        require(tokens == references[index],
                f"batched stream {index} diverged from its single reference")
        total += len(tokens)
    receipt["requests"] = len(prompts)
    receipt["wall_s"] = round(wall, 3)
    receipt["aggregate_tok_s"] = round(total / wall, 2)
    if evidence.available():
        for index, prompt in enumerate(prompts):
            row = measurement_for(evidence, prompt, deadline=60)
            require(row.get("status") == 0,
                    f"engine status {row.get('status')} for request {index}", row)


@test("A", "large_prefill_under_decode")
def t_large_prefill(model, spec, api, evidence, receipt):
    long_limit = spec.get("long_context", 131072)
    long_len = min(16384, long_limit - 512)
    bound = float(os.environ.get("E2E_PREFILL_GAP_BOUND_S", "60"))
    shorts = []
    for index in range(8):
        prompt = lcg_tokens(32, 200 + index)
        _, plain = api.request(prompt, 48)
        reference, _ = response_tokens(plain)
        shorts.append((prompt, reference))
    state = {"gaps": [], "streams": [None] * len(shorts), "errors": []}

    def watch(index, prompt):
        try:
            stream = Stream(*api.stream(prompt, 48))
            tokens = []
            last = time.time()
            for event in stream.events():
                if event == "[DONE]":
                    break
                if json.loads(event).get("tokens"):
                    now = time.time()
                    state["gaps"].append((index, round(now - last, 3)))
                    last = now
                    tokens = tokens + json.loads(event)["tokens"]
            state["streams"][index] = tokens
            stream.close()
        except Exception as error:
            state["errors"].append((index, str(error)))

    threads = []
    for index, (prompt, _) in enumerate(shorts):
        thread = threading.Thread(target=watch, args=(index, prompt))
        thread.start()
        threads.append(thread)
        time.sleep(0.05)
    time.sleep(1.0)
    long_prompt = lcg_tokens(long_len, 777)
    t0 = time.time()
    status, body = api.request(long_prompt, 16)
    require(status == 200, f"long prefill returned {status}", body[:300])
    ttft = time.time() - t0
    _, finish = response_tokens(body)
    require(finish == "length", f"long request finish {finish}")
    for thread in threads:
        thread.join()
    for index, tokens in enumerate(state["streams"]):
        require(tokens is not None,
                f"short stream {index} died during the large prefill",
                str(state["errors"]))
        require(tokens == shorts[index][1],
                f"short stream {index} diverged during the large prefill")
    worst = max((gap for _, gap in state["gaps"]), default=0.0)
    require(worst <= bound,
            f"decode stalled {worst}s during a {long_len}-token prefill (bound {bound}s)")
    receipt["long_prompt_tokens"] = long_len
    receipt["long_ttft_s"] = round(ttft, 3)
    receipt["worst_decode_gap_s"] = worst


@test("A", "cancel_disconnect")
def t_cancel_disconnect(model, spec, api, evidence, receipt):
    status, body = api.get("/health")
    baseline = json.loads(body).get("live_requests")
    prompt = lcg_tokens(64, 301)
    stream = Stream(*api.stream(prompt, 512))
    try:
        got_first = False
        for event in stream.events():
            if json.loads(event).get("tokens"):
                got_first = True
                break
        require(got_first, "no first token before abort")
    finally:
        stream.close()
    deadline = time.time() + 60
    live = None
    while time.time() < deadline:
        status, body = api.get("/health")
        require(status == 200, "engine unhealthy after client abort", body[:200])
        live = json.loads(body).get("live_requests")
        if baseline is None or (live is not None and live <= baseline):
            break
        time.sleep(1.0)
    if baseline is not None and live is not None:
        require(live <= baseline,
                f"live_requests {live} never returned to baseline {baseline}")
    prompt2 = lcg_tokens(32, 302)
    status, body = api.request(prompt2, 8)
    require(status == 200, "follow-up request failed after abort", body[:200])
    receipt["baseline_live_requests"] = baseline
    receipt["settled_live_requests"] = live


@test("A", "error_surfaces")
def t_error_surfaces(model, spec, api, evidence, receipt):
    limit = spec.get("long_context", 131072)
    huge = lcg_tokens(limit + 64, 401)
    status, body = api.request(huge, 4)
    require(status in (400, 413), f"oversized prompt returned {status}", body[:200])
    require("error" in json.loads(body), f"no error object: {body[:200]}")
    status, body = api.request(lcg_tokens(16, 402), 4, temperature=-1.0)
    require(status == 400, f"invalid temperature returned {status}", body[:200])
    status, body = api.get("/health")
    require(status == 200, "engine unhealthy after error surfaces")
    receipt["oversized_status"] = status


@test("A", "long_context_passkey")
def t_long_context_passkey(model, spec, api, evidence, receipt):
    template = spec.get("passkey_template")
    if not template:
        raise E2EFail("SKIPPED: no passkey_template in the model config")
    limit = spec.get("long_context", 131072)
    depth = min(32768, limit - 2048)
    code = random.Random(606).randrange(100000, 999999)
    filler = " ".join(f"row {i} holds no key" for i in range(depth))
    prompt = template.replace("{filler}", filler).replace("{code}", str(code))
    status, body = api.request_text(prompt, 16)
    require(status == 200, f"passkey request returned {status}", body[:300])
    receipt["depth_rows"] = depth
    receipt["code"] = code
    receipt["response_head"] = body[:400]


def rotation_run(config, args, timeout=900):
    rotation = config.get("rotation")
    require(rotation, "config has no rotation block")
    command = ["ssh", "-o", "BatchMode=yes", rotation["host"],
               f"{rotation['tool']} " + " ".join(args)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return run.returncode, run.stdout + run.stderr


def rotation_pause(config):
    rc, out = rotation_run(config, ["pause"])
    require(rc == 0, "rotation pause failed", out[-300:])


def rotation_resume(config):
    rotation_run(config, ["resume"])


def wait_healthy(api, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, body = api.get("/health")
            if status == 200 and json.loads(body).get("status") == "ok":
                return time.time()
        except Exception:
            pass
        time.sleep(1.0)
    raise E2EFail(f"model not healthy within {timeout}s")


@test("B", "model_toggle_under_minute")
def t_model_toggle(model, spec, api, evidence, receipt):
    rotation = config.get("rotation")
    if not rotation:
        raise E2EFail("SKIPPED: no rotation block in config")
    rotation_pause(config)
    try:
        rc, out = rotation_run(config, ["force", rotation["away_slot"]])
        require(rc == 0, "rotate-away failed", out[-500:])
        rc, out = rotation_run(config, ["force", rotation["slot"]])
        require(rc == 0, "rotate-back failed", out[-500:])
        t0 = time.time()
        ready = wait_healthy(api, 120)
        elapsed = ready - t0
        require(elapsed <= 60.0, f"promotion took {elapsed:.1f}s, gate is 60s")
        prompt = lcg_tokens(32, 601)
        status, body = api.request(prompt, 8)
        require(status == 200, "smoke completion failed after toggle", body[:200])
        receipt["health_ready_s"] = round(elapsed, 2)
    finally:
        rotation_resume(config)


@test("B", "kv_park_resume")
def t_kv_park_resume(model, spec, api, evidence, receipt):
    rotation = config.get("rotation")
    if not rotation:
        raise E2EFail("SKIPPED: no rotation block in config")
    require(evidence.available(), "kv_park_resume needs api_log evidence")
    prompt = lcg_tokens(512, 701)
    _, first = api.request(prompt, 32)
    pre_tokens, _ = response_tokens(first)
    rotation_pause(config)
    try:
        rc, out = rotation_run(config, ["force", rotation["away_slot"]])
        require(rc == 0, "rotate-away failed", out[-500:])
        time.sleep(10)
        rc, out = rotation_run(config, ["force", rotation["slot"]])
        require(rc == 0, "rotate-back failed", out[-500:])
        wait_healthy(api, 120)
        _, second = api.request(prompt, 32)
        post_tokens, _ = response_tokens(second)
        require(post_tokens == pre_tokens,
                "continuation after park/restore differs from the pre-park decode")
        row = measurement_for(evidence, prompt)
        require(row.get("cached_prompt_tokens", 0) > 0,
                "KV restore did not reuse the parked prefix",
                f"cached {row.get('cached_prompt_tokens')} of {len(prompt)}")
        require(row.get("stale_prefix_recomputes") == 0,
                "restored prefix needed recompute", row)
        receipt["cached_prompt_tokens"] = row.get("cached_prompt_tokens")
        receipt["prompt_tokens"] = len(prompt)
    finally:
        rotation_resume(config)


@test("C", "speculation_greedy_equivalence")
def t_spec_equivalence(model, spec, api, evidence, receipt):
    spec_cfg = config.get("speculation")
    if not spec_cfg:
        raise E2EFail("SKIPPED: no speculation block in config")
    api_on = Api(spec_cfg["on"])
    api_off = Api(spec_cfg["off"])
    for index in range(4):
        prompt = lcg_tokens(128, 801 + index)
        _, plain = api_off.request(prompt, 64)
        reference, _ = response_tokens(plain)
        _, drafted = api_on.request(prompt, 64)
        tokens, _ = response_tokens(drafted)
        require(tokens == reference,
                f"speculation changed greedy output on prompt {index}",
                f"{tokens[:8]}... != {reference[:8]}...")
    receipt["prompts"] = 4


@test("C", "speculation_payoff")
def t_spec_payoff(model, spec, api, evidence, receipt):
    spec_cfg = config.get("speculation")
    if not spec_cfg:
        raise E2EFail("SKIPPED: no speculation block in config")
    api_on = Api(spec_cfg["on"])
    api_off = Api(spec_cfg["off"])
    prompt = lcg_tokens(256, 851)

    def timed(target):
        t0 = time.time()
        status, body = target.request(prompt, 256)
        require(status == 200, f"payoff request returned {status}", body[:200])
        tokens, _ = response_tokens(body)
        return len(tokens) / (time.time() - t0), tokens

    rate_off, tokens_off = timed(api_off)
    rate_on, tokens_on = timed(api_on)
    require(tokens_on == tokens_off, "speculation stream diverged from greedy")
    receipt["no_spec_tok_s"] = round(rate_off, 2)
    receipt["spec_tok_s"] = round(rate_on, 2)
    receipt["ratio"] = round(rate_on / rate_off, 3)


@test("A", "context_boundaries")
def t_context_boundaries(model, spec, api, evidence, receipt):
    limit = spec.get("long_context", 131072)
    budget = 8
    fit = limit - budget
    prompt = lcg_tokens(fit, 901)
    status, body = api.request(prompt, budget)
    require(status == 200, f"exact-fit prompt ({fit}+{budget}) returned {status}",
            body[:300])
    tokens, finish = response_tokens(body)
    require(finish == "length" and len(tokens) == budget,
            f"exact-fit finish {finish} with {len(tokens)} tokens")
    over = lcg_tokens(fit + 1, 902)
    status, body = api.request(over, budget)
    require(status == 400, f"one-over-limit returned {status}", body[:200])
    data = json.loads(body)
    code = (data.get("error") or {}).get("code")
    require(code in ("context_length_exceeded", "invalid_request_error"),
            f"boundary error code {code}", body[:200])
    receipt["fit"] = fit
    receipt["over_rejected"] = code


@test("A", "edge_shapes")
def t_edge_shapes(model, spec, api, evidence, receipt):
    status, body = api.request(lcg_tokens(32, 911), 1)
    require(status == 200, f"max_tokens=1 returned {status}", body[:200])
    tokens, finish = response_tokens(body)
    require(len(tokens) == 1 and finish == "length",
            f"max_tokens=1 gave {len(tokens)} tokens, finish {finish}")
    status, body = api.request(lcg_tokens(1, 912), 4)
    require(status == 200, f"single-token prompt returned {status}", body[:200])
    tokens, _ = response_tokens(body)
    require(len(tokens) == 4, "single-token prompt mis-sized output")
    connection = http.client.HTTPConnection(api.host, api.port, timeout=30)
    both = {"prompt": "hello", "prompt_token_ids": [1, 2, 3],
            "max_tokens": 4, "temperature": 0.0}
    connection.request("POST", completion_path(api.chat),
                       json.dumps(both), {"Content-Type": "application/json"})
    response = connection.getresponse()
    code_both = response.status
    response.read()
    connection.close()
    require(code_both == 400, f"prompt+prompt_token_ids returned {code_both}")
    empty = {"prompt_token_ids": [], "max_tokens": 4, "temperature": 0.0}
    status, body = api.post(empty)
    require(status == 400, f"empty prompt returned {status}", body[:200])
    status, body = api.get("/health")
    require(status == 200, "engine unhealthy after edge shapes")


@test("A", "stop_token_edges")
def t_stop_token_edges(model, spec, api, evidence, receipt):
    prompt = lcg_tokens(32, 921)
    _, plain = api.request(prompt, 12)
    full, _ = response_tokens(plain)
    status, body = api.request(prompt, 12, stop_token_ids=[full[0]])
    require(status == 200, body[:200])
    tokens, finish = response_tokens(body)
    require(finish == "stop",
            f"stop-at-first finish {finish} with {len(tokens)} tokens")
    require(tokens in ([], [full[0]]),
            f"stop-at-first emitted unexpected tokens {tokens}")
    status, body = api.request(prompt, 12, stop_token_ids=[full[6]])
    require(status == 200, body[:200])
    tokens, finish = response_tokens(body)
    require(finish == "stop", f"mid-stop finish {finish}")
    require(tokens[:6] == full[:6] and len(tokens) in (6, 7),
            "early stop changed the tokens before the stop position",
            f"{tokens} vs {full}")
    status, body = api.request(prompt, 12, stop_token_ids=[999999])
    require(status == 200, body[:200])
    tokens, finish = response_tokens(body)
    require(finish == "length" and tokens == full,
            "never-hit stop changed the stream")


@test("A", "duplicate_concurrent")
def t_duplicate_concurrent(model, spec, api, evidence, receipt):
    prompt = lcg_tokens(128, 931)
    results = [None] * 4

    def run(index):
        try:
            results[index] = api.request(prompt, 24)
        except Exception as error:
            results[index] = (0, f"transport: {error}")

    threads = []
    for index in range(4):
        thread = threading.Thread(target=run, args=(index,))
        thread.start()
        threads.append(thread)
        time.sleep(0.05)
    for thread in threads:
        thread.join()
    streams = []
    for index, (status, body) in enumerate(results):
        require(status == 200, f"duplicate {index} returned {status}",
                str(body)[:200])
        tokens, _ = response_tokens(body)
        streams.append(tokens)
    for index in range(1, 4):
        require(streams[index] == streams[0],
                f"concurrent duplicate {index} diverged from duplicate 0")
    if evidence.available():
        rows = evidence.measurements(prompt_sha(prompt))
        ids = [row.get("request_id") for row in rows[-4:]]
        require(len(set(ids)) == 4,
                f"concurrent duplicates reused request ids: {ids}")
    receipt["identical_streams"] = 4


@test("A", "batch_invariance")
def t_batch_invariance(model, spec, api, evidence, receipt):
    probe = lcg_tokens(96, 941)
    _, solo = api.request(probe, 48)
    reference, _ = response_tokens(solo)
    results = [None] * 8
    noise = [lcg_tokens(24 + 16 * i, 950 + i) for i in range(8)]

    def run(index):
        try:
            results[index] = api.request(
                probe if index == 0 else noise[index], 48)
        except Exception as error:
            results[index] = (0, f"transport: {error}")

    threads = []
    for index in range(8):
        thread = threading.Thread(target=run, args=(index,))
        thread.start()
        threads.append(thread)
        time.sleep(0.05)
    for thread in threads:
        thread.join()
    status, body = results[0]
    require(status == 200, f"probe under load returned {status}",
            str(body)[:200])
    loaded, _ = response_tokens(body)
    require(loaded == reference,
            "greedy output changed under concurrent load (batch invariance)")
    _, again = api.request(probe, 48)
    after, _ = response_tokens(again)
    require(after == reference, "greedy output drifted after load")
    receipt["reference_tokens"] = len(reference)


@test("A", "cancel_storm")
def t_cancel_storm(model, spec, api, evidence, receipt):
    streams = []
    references = []
    survivors = [None] * 8
    prompts = [lcg_tokens(48, 960 + i) for i in range(16)]
    for index in range(16):
        _, plain = api.request(prompts[index], 24)
        tokens, _ = response_tokens(plain)
        references.append(tokens)

    def keep(index):
        try:
            stream = Stream(*api.stream(prompts[index], 24))
            _, tokens, _ = stream_collect(stream)
            survivors[index] = tokens
            stream.close()
        except Exception as error:
            survivors[index] = f"died: {error}"

    handles = []
    threads = []
    for index in range(16):
        if index < 8:
            thread = threading.Thread(target=keep, args=(index,))
            thread.start()
            threads.append(thread)
            time.sleep(0.05)
        else:
            stream = Stream(*api.stream(prompts[index], 512))
            handles.append(stream)
    got = 0
    for stream in handles:
        try:
            for event in stream.events():
                if json.loads(event).get("tokens"):
                    got += 1
                    break
        except Exception:
            pass
    for stream in handles:
        stream.close()
    require(got >= 1, "no cancelled stream produced a token before abort")
    for thread in threads:
        thread.join()
    for index in range(8):
        require(survivors[index] == references[index],
                f"survivor {index} diverged while 8 peers were cancelled: "
                f"{survivors[index]}")
    status, body = api.get("/health")
    require(status == 200, "engine unhealthy after cancel storm", body[:200])
    prompt = lcg_tokens(32, 979)
    status, body = api.request(prompt, 8)
    require(status == 200, "post-storm request failed", body[:200])
    receipt["cancelled"] = 8
    receipt["survivors"] = 8


@test("A", "error_recovery")
def t_error_recovery(model, spec, api, evidence, receipt):
    limit = spec.get("long_context", 131072)
    steps = []
    oversized = lcg_tokens(limit + 64, 981)
    status, _ = api.request(oversized, 4)
    steps.append(("oversized", status))
    prompt = lcg_tokens(24, 982)
    status, body = api.request(prompt, 6)
    require(status == 200, f"request after oversized error failed: {status}",
            body[:200])
    steps.append(("recovery_1", status))
    status, _ = api.request(lcg_tokens(16, 983), 4, temperature=-1.0)
    steps.append(("bad_temperature", status))
    status, body = api.request(prompt, 6)
    require(status == 200, "request after sampling error failed", body[:200])
    tokens, _ = response_tokens(body)
    require(len(tokens) == 6, "recovery completion mis-sized")
    steps.append(("recovery_2", status))
    connection = http.client.HTTPConnection(api.host, api.port, timeout=30)
    connection.request("POST", completion_path(api.chat), "{broken",
                       {"Content-Type": "application/json"})
    response = connection.getresponse()
    response.read()
    connection.close()
    steps.append(("malformed", response.status))
    status, body = api.request(prompt, 6)
    require(status == 200, "request after malformed body failed", body[:200])
    steps.append(("recovery_3", status))
    status, body = api.request(lcg_tokens(24, 984), 64, deadline_ms=1)
    steps.append(("deadline", status))
    if status == 200:
        if evidence.available():
            row = measurement_for(evidence, lcg_tokens(24, 984))
            require(row.get("deadline_expired") in (0, 1),
                    f"deadline_expired field {row.get('deadline_expired')}")
    status, body = api.request(prompt, 6)
    require(status == 200, "request after deadline test failed", body[:200])
    steps.append(("recovery_4", status))
    status, body = api.get("/health")
    require(status == 200, "engine unhealthy at end of error sequence")
    receipt["sequence"] = steps


@test("A", "churn_stability")
def t_churn_stability(model, spec, api, evidence, receipt):
    status, body = api.get("/health")
    baseline = json.loads(body).get("live_requests")
    rounds = 120
    lengths = []
    prompt_pool = [lcg_tokens(20 + 3 * i, 1000 + i) for i in range(24)]
    t0 = time.time()
    for round_index in range(rounds):
        prompt = prompt_pool[round_index % len(prompt_pool)]
        status, body = api.request(prompt, 6)
        require(status == 200,
                f"churn request {round_index} returned {status}",
                str(body)[:200])
        tokens, _ = response_tokens(body)
        require(len(tokens) == 6,
                f"churn request {round_index} emitted {len(tokens)} tokens")
        lengths.append(len(tokens))
    wall = time.time() - t0
    status, body = api.get("/health")
    data = json.loads(body)
    require(status == 200, "engine unhealthy after churn", body[:200])
    if baseline is not None and data.get("live_requests") is not None:
        require(data["live_requests"] <= baseline,
                f"live_requests leaked: {baseline} -> {data['live_requests']}")
    if evidence.available():
        ids = [row.get("request_id")
               for prompt in prompt_pool
               for row in evidence.measurements(prompt_sha(prompt))]
        recent = ids[-(rounds + 8):]
        require(len(recent) == len(set(recent)),
                "request ids repeated across churn (slot or id reuse)")
    receipt["requests"] = rounds
    receipt["wall_s"] = round(wall, 2)
    receipt["served"] = data.get("served")


@test("A", "utf8_stream_integrity")
def t_utf8_stream_integrity(model, spec, api, evidence, receipt):
    if not spec.get("text_supported"):
        raise E2EFail("SKIPPED: no text_supported in the model config")
    words = ["héllo", "日本語", "🎉", "wörld", "中文", "τthesis",
             "naïve", "🌍", "straße", "emoji"]
    prompt = " ".join(words * 3)
    status, body = api.request_text(prompt, 12)
    require(status == 200, f"text request returned {status}", body[:300])
    plain_text = text_response(body)
    body = {"prompt": prompt, "max_tokens": 12, "temperature": 0.0,
            "stream": True}
    stream = Stream(*api.open_stream(body))
    try:
        pieces = []
        for event in stream.events():
            if event == "[DONE]":
                break
            data = json.loads(event)
            for choice in data.get("choices", []):
                piece = choice.get("text")
                if piece is None:
                    piece = choice.get("delta", {}).get("content")
                if piece:
                    pieces.append(piece)
    finally:
        stream.close()
    streamed_text = "".join(pieces)
    require(streamed_text == plain_text,
            "streamed text differs from non-stream text",
            f"{streamed_text[:60]!r} != {plain_text[:60]!r}")
    try:
        streamed_text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise E2EFail(f"streamed text is not valid unicode: {error}")
    receipt["chars"] = len(streamed_text)


@test("A", "model_isolation")
def t_model_isolation(model, spec, api, evidence, receipt):
    other_name = spec.get("isolation_peer")
    if not other_name:
        raise E2EFail("SKIPPED: no isolation_peer in the model config")
    other_spec = MODELS_CONFIG.get(other_name)
    if not other_spec:
        raise E2EFail(f"SKIPPED: peer model {other_name} not in config")
    peer = Api(other_spec)
    prompt = lcg_tokens(64, 991)
    outcomes = [None] * 2

    def hit(index, target):
        try:
            outcomes[index] = target.request(prompt, 16)
        except Exception as error:
            outcomes[index] = (0, f"transport: {error}")

    threads = [threading.Thread(target=hit, args=(0, api)),
               threading.Thread(target=hit, args=(1, peer))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for index, (status, body) in enumerate(outcomes):
        require(status == 200, f"isolation request {index} returned {status}",
                str(body)[:200])
    if evidence.available():
        rows = evidence.measurements(prompt_sha(prompt))
        for row in rows[-2:]:
            require(row.get("adapter_id"),
                    "no adapter identity in evidence for isolation run")
    receipt["peer"] = other_name


@test("A", "evidence_integrity")
def t_evidence_integrity(model, spec, api, evidence, receipt):
    require(evidence.available(), "evidence_integrity needs api_log evidence")
    rows = evidence.measurements()
    require(len(rows) >= 4, f"only {len(rows)} measurement lines found")
    ids = [row.get("request_id") for row in rows]
    require(all(isinstance(i, int) and i > 0 for i in ids),
            f"non-numeric request ids: {ids[:6]}")
    require(len(ids) == len(set(ids)), "duplicate request ids in evidence")
    for row in rows:
        require("prompt_sha256" in row and len(row["prompt_sha256"]) == 64,
                f"malformed prompt sha: {row.get('prompt_sha256')}")
        require(row.get("prompt_tokens", 0) >= row.get("cached_prompt_tokens", 0),
                f"cached exceeds prompt: {row}")
        require(row.get("finish_reason") in ("length", "stop", "error"),
                f"unknown finish_reason {row.get('finish_reason')}")
    receipt["measurement_lines"] = len(rows)
    receipt["boot_pids"] = sorted({row.get("boot_pid") for row in rows})


class FakeApiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "served": 7, "tokenizer": True,
                             "ranks": 16, "connected_ranks": 16,
                             "missing_rank": -1, "degraded_rank": -1,
                             "live_requests": self.server.engine.live})
        elif self.path == "/health/live":
            self._json(200, {"status": "live", "ready": True})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "fake-model", "object": "model",
                 "owned_by": "sparkpipe", "served": 7}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            self._json(400, {"error": "bad request"})
            return
        if self.path not in ("/v1/completions", "/v1/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        tokens = body.get("prompt_token_ids")
        text = body.get("prompt")
        if text is not None and tokens is not None:
            self._json(400, {"error": {
                "message": "prompt and prompt_token_ids are exclusive"}})
            return
        if text is not None:
            words = text.split(" ")
            tokens = []
            for word in words:
                tokens.append(sum(word.encode()) % 90000 + 1000)
            body = dict(body)
            body["prompt_token_ids"] = tokens
            body["fake_words"] = words
        if not tokens:
            self._json(400, {"error": {
                "message": "prompt_token_ids required"}})
            return
        if body.get("temperature", 0.0) < 0:
            self._json(400, {"error": {"message": "bad temperature"}})
            return
        if len(tokens) + body.get("max_tokens", 16) > 65536:
            self._json(400, {"error": {
                "message": "the prompt exceeds this deployment's positions",
                "type": "invalid_request_error",
                "code": "context_length_exceeded"}})
            return
        out = self.server.engine.complete(body)
        if body.get("stream"):
            self._stream(out)
        else:
            self._json(200, out)

    def _json(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _stream(self, out):
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        pieces = out.get("fake_pieces") or ["x"] * len(out["tokens"])
        for token, piece in zip(out["tokens"], pieces):
            chunk = {"choices": [{"index": 0, "text": piece + " ",
                                  "finish_reason": None}],
                     "tokens": [token]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        final = {"choices": [{"index": 0, "text": "",
                              "finish_reason": out["finish_reason"]}]}
        self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


class FakeEngine:
    def __init__(self, log_path):
        self.log_path = log_path
        self.seen = []
        self.counter = 0
        self.live = 0

    WORDS = ["héllo", "日本語", "🎉", "wörld", "中文",
             "τthesis", "naïve", "🌍", "straße", "emoji"]

    def complete(self, body):
        tokens = body["prompt_token_ids"]
        max_tokens = body.get("max_tokens", 16)
        seed = body.get("seed", 0)
        temperature = body.get("temperature", 0.0)
        state = f"{tokens[:8]}:{len(tokens)}:{seed}:{temperature}".encode()
        value = int(hashlib.sha256(state).hexdigest()[:12], 16)
        stream = []
        stops = body.get("stop_token_ids", [])
        finish = "length"
        for _ in range(max_tokens):
            value = (value * LCG_MUL) % LCG_MOD
            token = 2000 + value % 100000
            if token in stops:
                finish = "stop"
                break
            stream.append(token)
        cached = 0
        for prior in self.seen:
            if len(prior) <= len(tokens) and tokens[:len(prior)] == prior:
                cached = max(cached, len(prior))
        self.seen.append(tokens)
        self.counter += 1
        row = {"event": "request_measurements", "boot_pid": os.getpid(),
               "request_id": self.counter, "status": 0, "engine_completed": 1,
               "stale_prefix_recomputes": 0, "prompt_tokens": len(tokens),
               "cached_prompt_tokens": cached,
               "prompt_sha256": prompt_sha(tokens), "adapter_id": "fake",
               "model_id": "fake-model", "model_revision": "fake",
               "session_fingerprint": 1, "deadline_expired": 0,
               "stream": 1 if body.get("stream") else 0,
               "temperature": temperature, "seed": seed,
               "finish_reason": finish, "tokens": []}
        with open(self.log_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        words = [self.WORDS[token % len(self.WORDS)] for token in stream]
        text_out = "".join(word + " " for word in words)
        result = {"choices": [{"index": 0, "text": text_out,
                               "finish_reason": finish}],
                  "tokens": stream, "finish_reason": finish,
                  "fake_pieces": words}
        if body.get("logprobs"):
            result["logprobs"] = [-0.5 - 0.01 * i for i in range(len(stream))]
        return result


def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "api.log")
        class FakeServer(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 128

        server = FakeServer(("127.0.0.1", 0), FakeApiHandler)
        server.engine = FakeEngine(log_path)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        config = {"models": {
            "fake": {
                "host": "127.0.0.1", "port": port, "chat": False,
                "api_log": log_path, "ranks": 16, "long_context": 65536,
                "logprobs_supported": True, "text_supported": True,
                "isolation_peer": "fake_peer"},
            "fake_peer": {
                "host": "127.0.0.1", "port": port, "chat": False,
                "api_log": log_path, "ranks": 16, "long_context": 65536}}}
        try:
            failures = run_selected(config, "fake", tiers=("A",),
                                    receipt_dir=None, quiet=False)
        finally:
            server.shutdown()
        return failures


def run_selected(config, model, tiers, receipt_dir, quiet):
    spec = config["models"][model]
    global MODELS_CONFIG
    MODELS_CONFIG.clear()
    MODELS_CONFIG.update(config["models"])
    api = Api(spec)
    evidence = Evidence(config, model)
    failures = 0
    skipped = 0
    for (tier, name), function in sorted(TESTS.items()):
        if tier not in tiers:
            continue
        receipt = {"test": name, "tier": tier, "model": model,
                   "utc": datetime.now(timezone.utc).isoformat()}
        try:
            function(model, spec, api, evidence, receipt)
            verdict = "PASS"
        except E2EFail as error:
            message = str(error)
            verdict = "SKIPPED" if message.startswith("SKIPPED") else "FAIL"
            receipt["failure"] = message
            if verdict == "FAIL":
                failures += 1
            else:
                skipped += 1
        except Exception as error:
            verdict = "FAIL"
            receipt["failure"] = f"{type(error).__name__}: {error}"
            failures += 1
        if receipt_dir:
            os.makedirs(receipt_dir, exist_ok=True)
            with open(os.path.join(receipt_dir, f"{name}.json"), "w") as handle:
                json.dump(receipt, handle, indent=1, sort_keys=True)
                handle.write("\n")
        if not quiet:
            suffix = "" if verdict == "PASS" else " " + receipt.get("failure", "")
            print(f"{verdict} {tier}/{name}{suffix}", flush=True)
    if not quiet:
        print(f"e2e complete: {failures} failed, {skipped} skipped", flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(
        description="SparkPipe end-to-end system tests; see tests/e2e/README.md")
    parser.add_argument("--config", default="tests/e2e/e2e_config.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tier", default="A")
    parser.add_argument("--only", default=None)
    parser.add_argument("--receipt-dir", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.list:
        for (tier, name) in sorted(TESTS):
            print(f"{tier} {name}")
        return 0
    if args.selftest:
        return min(selftest(), 125)
    with open(args.config) as handle:
        config = json.load(handle)
    model = args.model or sorted(config["models"])[0]
    tiers = tuple(args.tier.split(","))
    if args.only:
        for key in [k for k in TESTS if k[1] != args.only]:
            TESTS.pop(key)
    receipt_dir = args.receipt_dir or os.path.join(
        "qualification", "e2e",
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    return min(run_selected(config, model, tiers, receipt_dir, False), 125)


if __name__ == "__main__":
    sys.exit(main())

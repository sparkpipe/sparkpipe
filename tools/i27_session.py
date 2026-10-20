#!/usr/bin/env python3
"""I27 prefix-reuse session against a live SparkPipe C API.

Every case compares a run that reuses cached KV with an uninterrupted control
of the same prompt and requires identical tokens. A control is cold only when
the API reports cached_tokens 0 for it; the session evicts the cache by
flooding it with unrelated prompts until that holds. The restart and
write-back-fault cases need operator hooks; without one the case is not run
and the session verdict is INCOMPLETE, never PASS.
"""
import argparse
import hashlib
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

CASES = ("b1", "b16", "cow_mid_block", "abort_mid_prefill", "evict_recompute", "restart", "writeback_fault")


class SessionError(Exception):
    pass


class Api:
    def __init__(self, base, timeout):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt, max_tokens):
        body = json.dumps({"prompt_token_ids": prompt, "max_tokens": max_tokens, "temperature": 0}).encode()
        request = urllib.request.Request(self.base + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                reply = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise SessionError("completion failed with HTTP %d: %s" % (error.code, error.read()[:200]))
        tokens = reply.get("tokens")
        usage = reply.get("usage", {})
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
        if not isinstance(tokens, list) or not tokens or not isinstance(cached, int):
            raise SessionError("completion reply carries no tokens or cached_tokens")
        return [int(token) for token in tokens], cached

    def abort_after(self, prompt, max_tokens, seconds):
        host, _, port = self.base.split("//", 1)[1].partition(":")
        body = json.dumps({"prompt_token_ids": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
        connection = socket.create_connection((host, int(port or 80)), timeout=self.timeout)
        connection.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % (host.encode(), len(body)) + body)
        time.sleep(seconds)
        connection.close()

    def healthy(self):
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=self.timeout) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False


class Prompts:
    def __init__(self, seed, vocab_size, block_tokens):
        self.seed = seed
        self.vocab_size = vocab_size
        self.block_tokens = block_tokens
        self.counter = 0

    def tokens(self, label, count):
        out = []
        index = 0
        while len(out) < count:
            digest = hashlib.sha256(("%s/%s/%d" % (self.seed, label, index)).encode()).digest()
            out.extend(1000 + int.from_bytes(digest[i:i + 4], "little") % (self.vocab_size - 2000) for i in range(0, 32, 4))
            index += 1
        return out[:count]

    def fresh(self, label, count):
        self.counter += 1
        return self.tokens("%s/%d" % (label, self.counter), count)


class Session:
    def __init__(self, api, prompts, arguments):
        self.api = api
        self.prompts = prompts
        self.arguments = arguments
        self.cases = {}

    def record(self, name, passed, **details):
        self.cases[name] = dict(details, verdict="PASS" if passed else "FAIL")

    def evict(self):
        for _ in range(self.arguments.evict_prompts):
            self.api.complete(self.prompts.fresh("evict", self.arguments.evict_tokens), 1)

    def cold(self, prompt):
        tokens, cached = self.api.complete(prompt, self.arguments.max_tokens)
        if cached != 0:
            self.evict()
            tokens, cached = self.api.complete(prompt, self.arguments.max_tokens)
        if cached != 0:
            raise SessionError("the control still reports %d cached tokens after eviction; raise --evict-prompts" % cached)
        return tokens

    def case_b1(self):
        prompt = self.prompts.fresh("b1", self.arguments.prompt_tokens)
        control = self.cold(prompt)
        warm, cached = self.api.complete(prompt, self.arguments.max_tokens)
        self.record("b1", warm == control and cached > 0, cached_tokens=cached, identical=warm == control)

    def case_b16(self):
        shared = self.prompts.fresh("b16", self.arguments.prompt_tokens)
        prompts = [shared + self.prompts.fresh("b16-tail", self.prompts.block_tokens) for _ in range(self.arguments.concurrency)]
        controls = [self.cold(prompt) for prompt in prompts]
        results = [None] * len(prompts)
        errors = []

        def run(index):
            try:
                results[index] = self.api.complete(prompts[index], self.arguments.max_tokens)
            except SessionError as error:
                errors.append(str(error))
        threads = [threading.Thread(target=run, args=(index,)) for index in range(len(prompts))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise SessionError(errors[0])
        mismatched = [index for index, (tokens, _) in enumerate(results) if tokens != controls[index]]
        missed = [index for index, (_, cached) in enumerate(results) if cached == 0]
        self.record("b16", not mismatched and not missed, concurrency=len(prompts), mismatched=mismatched, missed=missed)

    def case_cow_mid_block(self):
        half = self.prompts.block_tokens // 2
        shared = self.prompts.fresh("cow", 2 * self.prompts.block_tokens + half)
        first = shared + self.prompts.fresh("cow-a", self.prompts.block_tokens)
        second = shared + self.prompts.fresh("cow-b", self.prompts.block_tokens)
        self.evict()
        self.api.complete(first, self.arguments.max_tokens)
        warm, cached = self.api.complete(second, self.arguments.max_tokens)
        self.evict()
        control = self.cold(second)
        self.record("cow_mid_block", warm == control and cached > 2 * self.prompts.block_tokens,
                    shared_tokens=len(shared), cached_tokens=cached, identical=warm == control)

    def case_abort_mid_prefill(self):
        prompt = self.prompts.fresh("abort", self.arguments.long_prompt_tokens)
        control = self.cold(prompt)
        self.evict()
        self.api.abort_after(prompt, self.arguments.max_tokens, self.arguments.abort_seconds)
        replay, cached = self.api.complete(prompt, self.arguments.max_tokens)
        self.record("abort_mid_prefill", replay == control and self.api.healthy(), cached_tokens=cached, identical=replay == control)

    def case_evict_recompute(self):
        prompt = self.prompts.fresh("evict-target", self.arguments.prompt_tokens)
        control = self.cold(prompt)
        warm, cached = self.api.complete(prompt, self.arguments.max_tokens)
        self.evict()
        recomputed, after = self.api.complete(prompt, self.arguments.max_tokens)
        self.record("evict_recompute", warm == control and recomputed == control and cached > 0 and after == 0,
                    warm_cached_tokens=cached, evicted_cached_tokens=after, identical=recomputed == control)

    def hook_case(self, name, command):
        if not command:
            self.cases[name] = {"verdict": "NOT-RUN", "reason": "needs --%s-command" % name.replace("_", "-")}
            return
        prompt = self.prompts.fresh(name, self.arguments.prompt_tokens)
        control = self.cold(prompt)
        completed = subprocess.run(command, shell=True, timeout=self.arguments.hook_timeout)
        deadline = time.monotonic() + self.arguments.hook_timeout
        while not self.api.healthy():
            if time.monotonic() > deadline:
                raise SessionError("the API did not return healthy after the %s hook" % name)
            time.sleep(2)
        replay, cached = self.api.complete(prompt, self.arguments.max_tokens)
        self.record(name, completed.returncode == 0 and replay == control, hook_status=completed.returncode,
                    cached_tokens=cached, identical=replay == control)

    def run(self, names):
        for name in names:
            try:
                if name == "restart":
                    self.hook_case(name, self.arguments.restart_command)
                elif name == "writeback_fault":
                    self.hook_case(name, self.arguments.writeback_fault_command)
                else:
                    getattr(self, "case_" + name)()
            except SessionError as error:
                self.cases[name] = {"verdict": "FAIL", "error": str(error)}
        verdicts = [self.cases[name]["verdict"] for name in names]
        if "FAIL" in verdicts:
            return "FAIL"
        return "PASS" if all(verdict == "PASS" for verdict in verdicts) else "INCOMPLETE"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True, help="C API base URL, e.g. http://hub:8446")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--block-tokens", type=int, required=True, help="KV block size of the deployment")
    parser.add_argument("--prompt-tokens", type=int, default=320)
    parser.add_argument("--long-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--evict-prompts", type=int, default=64)
    parser.add_argument("--evict-tokens", type=int, default=2048)
    parser.add_argument("--abort-seconds", type=float, default=0.2)
    parser.add_argument("--restart-command")
    parser.add_argument("--writeback-fault-command")
    parser.add_argument("--hook-timeout", type=float, default=900)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", default=str(time.time_ns()))
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    names = [name for name in arguments.cases.split(",") if name]
    unknown = sorted(set(names) - set(CASES))
    if unknown or not names:
        parser.error("unknown cases %s; known: %s" % (unknown, ",".join(CASES)))
    if arguments.block_tokens < 2 or arguments.vocab_size <= 4096:
        parser.error("--block-tokens must be at least 2 and --vocab-size above 4096")
    session = Session(Api(arguments.api, arguments.timeout), Prompts(arguments.seed, arguments.vocab_size, arguments.block_tokens), arguments)
    verdict = session.run(names)
    receipt = {"kind": "sparkpipe.i27-session.v1", "api": arguments.api, "seed": arguments.seed,
               "block_tokens": arguments.block_tokens, "cases": session.cases, "verdict": verdict}
    text = json.dumps(receipt, indent=1, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w") as handle:
            handle.write(text + "\n")
    print(text)
    print("verdict=%s" % verdict)
    return {"PASS": 0, "INCOMPLETE": 2}.get(verdict, 1)


if __name__ == "__main__":
    sys.exit(main())

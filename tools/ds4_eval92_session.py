#!/usr/bin/env python3
"""One ds4-eval 92 session for a model: activate, warm, batch with an in-flight interrupt and a fleet pause.

Runs on the hub next to the swap controller and the chat frontend:
  1. activate the model through the swap controller and wait until it serves
  2. warm: all 92 prompts with max_tokens 1, so the prefix cache holds them
  3. batch: all 92 at once; when --interrupt-at responses are saved, the client
     is killed, which disconnects every in-flight and queued request
  4. the engine must cancel them all: /health live_requests reaches 0
  5. pause: --pause-model is activated (this model's engines stop and its KV
     pool is released), then this model is activated again
  6. resume: the batch phase reruns with --resume, so only the unanswered cases
     are requested again; their cached prompt tokens show what the prefix cache
     kept across the interrupt and the pause
Without --max-tokens the completion budget is the model's served context
(context_tokens in the chat config) minus --prompt-reserve, at most
--max-tokens-cap. A batch that saves no new response for --stall-minutes is
stopped and recorded as a stall; cases that end in an error are requested
again up to --error-retries times, and every pass is in the timeline.
session.json and SESSION.md in --out hold the timeline and results.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def post_json(url: str, body: dict, timeout: float = 30) -> tuple[int, dict]:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


class Session:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        self.out.mkdir(parents=True, exist_ok=True)
        config = json.loads(Path(args.chat_config).expanduser().read_text())
        models = config["models"] if isinstance(config.get("models"), list) else list(config["models"].values())
        served = {model["id"]: model for model in models}
        if args.model not in served:
            raise SystemExit(f"{args.model} is not in {args.chat_config}")
        self.health_url = served[args.model]["engine"].rstrip("/") + "/health"
        if args.max_tokens is None:
            args.max_tokens = min(int(served[args.model]["context_tokens"]) - args.prompt_reserve, args.max_tokens_cap)
        self.timeline = []
        self.started = time.time()

    def mark(self, event: str, **fields) -> None:
        entry = {"event": event, "t_s": round(time.time() - self.started, 1),
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
        self.timeline.append(entry)
        print(json.dumps(entry), flush=True)
        self.write()

    def activate(self, model: str) -> float:
        started = time.time()
        status, payload = post_json(self.args.swap.rstrip("/") + "/activate", {"model": model})
        if status not in (200, 202):
            raise RuntimeError(f"activate {model}: HTTP {status} {payload}")
        while True:
            state = get_json(self.args.swap.rstrip("/") + "/status")
            if state.get("active") == model and state.get("state") == "idle":
                return round(time.time() - started, 1)
            if state.get("state") == "failed":
                raise RuntimeError(f"activate {model} failed: {state.get('error')}")
            if time.time() - started > self.args.activate_seconds:
                raise RuntimeError(f"activate {model} not done after {self.args.activate_seconds} s: {state}")
            time.sleep(3)

    def live_requests(self):
        try:
            return get_json(self.health_url).get("live_requests")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            return f"unreachable: {error!r}"

    def eval_command(self, phase: str, resume: bool) -> list:
        a = self.args
        command = [sys.executable, str(HERE / "ds4_eval92.py"), "--endpoint", a.endpoint, "--model", a.model,
                   "--out", str(self.out), "--max-tokens", str(a.max_tokens), "--temperature", str(a.temperature),
                   "--concurrency", str(a.concurrency), "--phases", phase, "--timeout", str(a.request_timeout)]
        for flag, value in (("--top-p", a.top_p), ("--top-k", a.top_k), ("--seed", a.seed), ("--reasoning-effort", a.reasoning_effort)):
            if value is not None:
                command += [flag, str(value)]
        if resume:
            command.append("--resume")
        return command

    def responses(self, phase: str) -> int:
        directory = self.out / phase / "responses"
        return len(list(directory.glob("*.json"))) if directory.exists() else 0

    def run_phase(self, phase: str, resume: bool = False, stop_at: int = 0) -> str:
        log = open(self.out / f"{phase}{'-resume' if resume else ''}.log", "a")
        process = subprocess.Popen(self.eval_command(phase, resume), stdout=log, stderr=subprocess.STDOUT)
        last_count, last_change = self.responses(phase), time.time()
        while process.poll() is None:
            time.sleep(2)
            count = self.responses(phase)
            if count != last_count:
                last_count, last_change = count, time.time()
            if stop_at and count >= stop_at and process.poll() is None:
                process.send_signal(signal.SIGKILL)
                process.wait()
                return "interrupted"
            if time.time() - last_change > self.args.stall_minutes * 60 and process.poll() is None:
                process.send_signal(signal.SIGKILL)
                process.wait()
                return "stalled"
        return "ok" if process.returncode == 0 else f"exit {process.returncode}"

    def summary(self, phase: str):
        path = self.out / phase / "summary.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return {key: value for key, value in data.items() if key != "results"}

    def write(self) -> None:
        a = self.args
        record = {"format": "sparkpipe-ds4-eval92-session-v1", "model": a.model, "label": a.label,
                  "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
                  "seed": a.seed, "reasoning_effort": a.reasoning_effort, "concurrency": a.concurrency,
                  "interrupt_at": a.interrupt_at, "pause_model": a.pause_model,
                  "timeline": self.timeline, "warm": self.summary("warm"), "batch": self.summary("batch")}
        (self.out / "session.json").write_text(json.dumps(record, indent=1) + "\n")
        batch = record["batch"] or {}
        lines = [f"# ds4-eval 92 session — {a.model} ({a.label})", "",
                 f"- reasoning_effort `{a.reasoning_effort or 'model default'}`, max_tokens `{a.max_tokens}`, temperature `{a.temperature}`,"
                 f" top_p `{a.top_p}`, top_k `{a.top_k}`, seed `{a.seed}`, concurrency `{a.concurrency}`",
                 f"- score **{batch.get('passed', '-')}/92**, errors {batch.get('errors', '-')}, finish {batch.get('finish_reasons', '-')}",
                 f"- by family {json.dumps(batch.get('by_family', {}))}", "", "| t s | event | detail |", "|---:|---|---|"]
        for entry in self.timeline:
            detail = {key: value for key, value in entry.items() if key not in ("event", "t_s", "utc")}
            lines.append(f"| {entry['t_s']} | {entry['event']} | {json.dumps(detail)} |")
        (self.out / "SESSION.md").write_text("\n".join(lines) + "\n")

    def run(self) -> int:
        a = self.args
        self.mark("activate", seconds=self.activate(a.model))
        result = self.run_phase("warm")
        warm = self.summary("warm") or {}
        self.mark("warm", result=result, cached_tokens=warm.get("cached_tokens"), prompt_tokens=warm.get("prompt_tokens"), wall_s=warm.get("wall_s"))
        if result != "ok":
            return 1
        if a.interrupt_at:
            result = self.run_phase("batch", stop_at=a.interrupt_at)
            self.mark("batch_part1", result=result, responses=self.responses("batch"), live_requests=self.live_requests())
            if result == "interrupted":
                started = time.time()
                live = self.live_requests()
                while live != 0 and time.time() - started < a.cancel_seconds:
                    time.sleep(2)
                    live = self.live_requests()
                self.mark("cancel_drain", seconds=round(time.time() - started, 1), live_requests=live)
                if a.pause_model:
                    self.mark("pause", model=a.pause_model, seconds=self.activate(a.pause_model))
                    self.mark("resume", model=a.model, seconds=self.activate(a.model))
            elif result != "ok":
                return 1
        for attempt in range(1 + a.error_retries):
            if self.responses("batch") == 92 and self.summary("batch") is not None and not self.summary("batch").get("errors"):
                break
            result = self.run_phase("batch", resume=self.responses("batch") > 0)
            batch = self.summary("batch") or {}
            self.mark("batch" if attempt == 0 else "batch_retry", result=result, passed=batch.get("passed"), errors=batch.get("errors"),
                      resumed=batch.get("resumed"), cached_tokens=batch.get("cached_tokens"), prompt_tokens=batch.get("prompt_tokens"),
                      completion_tokens=batch.get("completion_tokens"), wall_s=batch.get("wall_s"), completion_tok_s=batch.get("completion_tok_s"))
            if result == "stalled":
                return 1
        if (self.summary("batch") or {}).get("errors", 1):
            return 1
        self.mark("done", live_requests=self.live_requests())
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8433")
    parser.add_argument("--swap", default="http://127.0.0.1:8440")
    parser.add_argument("--chat-config", default="~/sparkpipe-chat/config.json")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-tokens-cap", type=int, default=65536)
    parser.add_argument("--prompt-reserve", type=int, default=1024)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--concurrency", type=int, default=92)
    parser.add_argument("--interrupt-at", type=int, default=0)
    parser.add_argument("--pause-model", default=None)
    parser.add_argument("--stall-minutes", type=float, default=90)
    parser.add_argument("--error-retries", type=int, default=2)
    parser.add_argument("--cancel-seconds", type=float, default=300)
    parser.add_argument("--activate-seconds", type=float, default=1200)
    parser.add_argument("--request-timeout", type=int, default=86400)
    args = parser.parse_args()
    if args.pause_model and not args.interrupt_at:
        raise SystemExit("--pause-model needs --interrupt-at")
    session = Session(args)
    try:
        code = session.run()
    except RuntimeError as error:
        session.mark("failed", error=str(error))
        code = 1
    session.write()
    return code


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rais_score

PROSE_PROMPT = (
    "Write a clear technical explanation of how distributed memory bandwidth, network collectives, "
    "and tensor-core compute interact during large language model inference across sixteen nodes. "
    "Start with single-stream decode and then analyze batch scaling.\n\nSection 1: "
)

CODE_PROMPT = (
    "Implement a deterministic ring buffer in C99 with fixed capacity, power-of-two masking, "
    "push, pop, peek, and count helpers. Include a small self-test main function.\n\n```c\n"
    "#include <stdint.h>\n#include <stdbool.h>\n\n"
)

EDIT_SOURCE = (
    "typedef struct {\n"
    "    uint32_t capacity;\n"
    "    uint32_t head;\n"
    "    uint32_t tail;\n"
    "    uint64_t dropped;\n"
    "    uint8_t *slots;\n"
    "} RingQueue;\n\n"
    "int ring_push(RingQueue *q, uint8_t value) {\n"
    "    uint32_t next = (q->head + 1u) % q->capacity;\n"
    "    if (next == q->tail) {\n"
    "        q->dropped += 1u;\n"
    "        return -1;\n"
    "    }\n"
    "    q->slots[q->head] = value;\n"
    "    q->head = next;\n"
    "    return 0;\n"
    "}\n\n"
    "int ring_pop(RingQueue *q, uint8_t *out) {\n"
    "    if (q->head == q->tail) {\n"
    "        return -1;\n"
    "    }\n"
    "    *out = q->slots[q->tail];\n"
    "    q->tail = (q->tail + 1u) % q->capacity;\n"
    "    return 0;\n"
    "}\n"
)

EDIT_PROMPT = (
    "Rewrite the C code below verbatim, changing only ring_push and ring_pop to track "
    "a high_watermark field in RingQueue whenever an item is pushed.\n\n"
    + EDIT_SOURCE
    + "\nUpdated code:\n```c\n"
)

FILLER_PARAGRAPH = (
    "The telemetry pipeline aggregates per-rank memory bandwidth counters, collective latency histograms, "
    "KV page residency metrics, and scheduler queue depths across every active inference lane. "
    "Each node reports monotonic timestamps, sequence admission counts, and thermal throttling flags "
    "at fixed intervals so regressions remain attributable to exact hardware boundaries. "
)

FOLLOWUP_DELTA_128 = (
    "\n\nFollow-up analysis request: summarize the three dominant hardware bottlenecks from the "
    "report above, list which workloads saturate memory bandwidth versus collective interconnect "
    "versus tensor compute, estimate the expected overlap gain if collectives hide behind GEMM tiles, "
    "and specify the next engineering change with the highest expected latency reduction per line "
    "of code changed. Keep the answer structured in four short numbered points with concrete ratios "
    "and exact bottleneck labels.\n\nAnswer:\n1. "
)

FOLLOWUP_TURN2 = (
    "\n\nConfirm the ordering above and include one concrete metric for each bottleneck.\n\n"
    "Verification:\n1. "
)


SYNTHETIC_VOCAB_MIN = 1000
SYNTHETIC_VOCAB_SPAN = 29000


def build_prompt_for_tokens(target_tokens: int, nonce: str) -> str:
    target_chars = max(64, int(target_tokens * 4.0))
    prefix = f"Session {nonce}. "
    chunks = [prefix]
    total = len(prefix)
    step = 0
    while total < target_chars:
        line = f"[Record {step:05d}] {FILLER_PARAGRAPH}\n"
        chunks.append(line)
        total += len(line)
        step += 1
    return "".join(chunks)


def synthetic_tokens_for_text(text: str, target_tokens: int | None = None) -> list[int]:
    pieces = re.findall(r"\w+|[^\w\s]", text) or ["token"]
    mapped: list[int] = []
    for piece in pieces:
        digest = hashlib.sha256(piece.encode("utf-8")).digest()
        val = int.from_bytes(digest[:4], "little")
        mapped.append(SYNTHETIC_VOCAB_MIN + (val % SYNTHETIC_VOCAB_SPAN))
    if target_tokens is None:
        return mapped
    out: list[int] = []
    while len(out) < target_tokens:
        out.extend(mapped)
    return out[:target_tokens]


def synthetic_tokens_for_length(target_tokens: int, nonce: str) -> list[int]:
    count = max(1, int(target_tokens))
    out: list[int] = []
    block = 0
    while len(out) < count:
        digest = hashlib.sha256(f"{nonce}:{block}".encode("utf-8")).digest()
        for offset in range(0, 32, 2):
            if len(out) >= count:
                break
            val = int.from_bytes(digest[offset:offset + 2], "little")
            out.append(SYNTHETIC_VOCAB_MIN + (val % SYNTHETIC_VOCAB_SPAN))
        block += 1
    return out


class PromptEncoder:
    def __init__(
        self,
        tokenizer_json: str | None = None,
        tokenizer_compiled: str | None = None,
        tokenizer_bin: str | None = None,
        synthetic_token_ids: bool = False,
    ) -> None:
        self.tokenizer_json = tokenizer_json
        self.tokenizer_compiled = tokenizer_compiled
        self.tokenizer_bin = tokenizer_bin
        self._hf_tokenizer: Any = None
        self._cache: dict[str, list[int]] = {}
        if tokenizer_json or tokenizer_compiled:
            self.mode = "tokenizer"
            self._init_tokenizer()
        elif synthetic_token_ids:
            self.mode = "synthetic"
        else:
            self.mode = "text"

    def _init_tokenizer(self) -> None:
        if self.tokenizer_bin:
            if not Path(self.tokenizer_bin).is_file():
                raise RuntimeError(f"tokenizer binary not found: {self.tokenizer_bin}")
            return
        if self.tokenizer_compiled:
            default_bin = SCRIPT_DIR.parent / "build" / "sparkpipe_tokenize_prompt"
            if default_bin.is_file():
                self.tokenizer_bin = str(default_bin)
                return
            raise RuntimeError("--tokenizer-compiled requires --tokenizer-bin pointing to sparkpipe_tokenize_prompt")
        try:
            import tokenizers

            self._hf_tokenizer = tokenizers.Tokenizer.from_file(self.tokenizer_json)
        except Exception as exc:
            default_bin = SCRIPT_DIR.parent / "build" / "sparkpipe_tokenize_prompt"
            if default_bin.is_file():
                self.tokenizer_bin = str(default_bin)
                return
            raise RuntimeError(
                f"unable to load --tokenizer-json without python tokenizers or --tokenizer-bin: {exc}"
            ) from exc

    def is_token_mode(self) -> bool:
        return self.mode != "text"

    def encode_text(self, text: str) -> list[int]:
        cached = self._cache.get(text)
        if cached is not None:
            return list(cached)
        if self.mode == "synthetic":
            ids = synthetic_tokens_for_text(text)
            self._cache[text] = ids
            return list(ids)
        if self._hf_tokenizer is not None:
            ids = [int(t) for t in self._hf_tokenizer.encode(text, add_special_tokens=False).ids]
        else:
            flag = "--tokenizer-compiled" if self.tokenizer_compiled else "--tokenizer-json"
            tok_path = self.tokenizer_compiled or self.tokenizer_json
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
                    handle.write(text)
                    tmp_path = handle.name
                proc = subprocess.run(
                    [str(self.tokenizer_bin), flag, str(tok_path), "--prompt-file", tmp_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                ids = [int(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdigit()]
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        if not ids:
            raise RuntimeError("tokenizer returned zero token IDs for prompt")
        self._cache[text] = ids
        return list(ids)

    def prompt_field_for_text(self, text: str, target_tokens: int | None = None) -> dict[str, Any]:
        if self.mode == "text":
            return {"prompt": text}
        if self.mode == "synthetic":
            return {"prompt_token_ids": synthetic_tokens_for_text(text, target_tokens=target_tokens)}
        return {"prompt_token_ids": self.encode_text(text)}

    def prompt_field_for_target(self, target_tokens: int, nonce: str) -> tuple[dict[str, Any], Any]:
        if self.mode == "text":
            prompt = build_prompt_for_tokens(target_tokens, nonce)
            return {"prompt": prompt}, prompt
        if self.mode == "synthetic":
            ids = synthetic_tokens_for_length(target_tokens, nonce)
            return {"prompt_token_ids": ids}, ids
        scale = 1.35
        while True:
            prompt = build_prompt_for_tokens(int(target_tokens * scale) + 64, nonce)
            ids = self.encode_text(prompt)
            if len(ids) >= target_tokens:
                trimmed = ids[:target_tokens]
                return {"prompt_token_ids": trimmed}, trimmed
            scale *= 1.5

    def extend_prompt(self, base_handle: Any, delta_text: str, delta_target_tokens: int = 128) -> tuple[dict[str, Any], Any]:
        if self.mode == "text":
            combined = str(base_handle) + delta_text
            return {"prompt": combined}, combined
        base_ids = list(base_handle) if isinstance(base_handle, list) else self.encode_text(str(base_handle))
        if self.mode == "synthetic":
            delta_ids = synthetic_tokens_for_text(delta_text, target_tokens=delta_target_tokens)
        else:
            delta_ids = self.encode_text(delta_text)
        combined_ids = base_ids + delta_ids
        return {"prompt_token_ids": combined_ids}, combined_ids

    def batch_prompt_field(self, base_handle: Any, idx: int, stream_prefix: bool = False) -> dict[str, Any]:
        if self.mode == "text":
            return {"prompt": format_batch_prompt(str(base_handle), idx, stream_prefix=stream_prefix)}
        if self.mode == "synthetic":
            base_ids = list(base_handle) if isinstance(base_handle, list) else synthetic_tokens_for_text(str(base_handle))
            tag = [
                SYNTHETIC_VOCAB_MIN + ((idx + 1) * 131 % SYNTHETIC_VOCAB_SPAN),
                SYNTHETIC_VOCAB_MIN + ((idx + 1) * 997 % SYNTHETIC_VOCAB_SPAN),
                SYNTHETIC_VOCAB_MIN + ((idx + 1) * 4093 % SYNTHETIC_VOCAB_SPAN),
            ]
            ids = (tag + base_ids) if stream_prefix else (base_ids + tag)
            return {"prompt_token_ids": ids}
        if isinstance(base_handle, str):
            return {"prompt_token_ids": self.encode_text(format_batch_prompt(base_handle, idx, stream_prefix=stream_prefix))}
        base_ids = list(base_handle)
        if stream_prefix:
            marker_ids = self.encode_text(f"Task {idx + 1}: ")
            return {"prompt_token_ids": marker_ids + base_ids}
        marker_ids = self.encode_text(f"\n\n[Stream {idx + 1}]{FOLLOWUP_DELTA_128}")
        return {"prompt_token_ids": base_ids + marker_ids}


def outputs_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_ids = a.get("token_ids") or []
    b_ids = b.get("token_ids") or []
    a_text = a.get("text") or ""
    b_text = b.get("text") or ""
    if a_ids or b_ids:
        if a_ids != b_ids:
            return False
        if a_text or b_text:
            return a_text == b_text
        return len(a_ids) > 0
    return a_text == b_text


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(pct / 100.0 * len(ordered))) - 1))
    return ordered[idx]


def make_connection(parsed: urllib.parse.ParseResult, timeout: float) -> http.client.HTTPConnection:
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=timeout)
    return http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)


def swap_model(swap_url: str, model: str, swap_token: str | None, timeout: float) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(swap_url)
    conn = make_connection(parsed, timeout)
    headers = {"Content-Type": "application/json"}
    if swap_token:
        headers["Authorization"] = f"Bearer {swap_token}"
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    started = time.monotonic()
    conn.request("POST", path, json.dumps({"model": model}), headers)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", "replace")
    conn.close()
    elapsed = round(time.monotonic() - started, 3)
    if resp.status >= 400:
        raise RuntimeError(f"model swap failed HTTP {resp.status}: {raw[:300]}")
    try:
        body = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        body = {"raw": raw[:200]}
    return {"http_status": resp.status, "elapsed_s": elapsed, "body": body}


def stream_completion(
    base_url: str,
    body: dict[str, Any],
    api_key: str | None = None,
    timeout: float = 900.0,
    endpoint: str = "completions",
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(base_url.rstrip("/"))
    suffix = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
    path = (parsed.path.rstrip("/") if parsed.path else "") + suffix
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = dict(body)
    if endpoint == "chat" and "prompt" in payload:
        prompt_text = payload.pop("prompt")
        payload["messages"] = [{"role": "user", "content": prompt_text}]
    payload["stream"] = True

    started = time.monotonic()
    token_times: list[float] = []
    token_ids: list[int] = []
    event_times: list[float] = []
    event_token_counts: list[int] = []
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    spec_stats: dict[str, Any] | None = None
    error: Any = None
    finish: str | None = None
    http_status = 0

    conn = make_connection(parsed, timeout)
    try:
        conn.request("POST", path, json.dumps(payload), headers)
        response = conn.getresponse()
        http_status = response.status
        if http_status != 200:
            raw_err = response.read().decode("utf-8", "replace")
            error = {"http_status": http_status, "body": raw_err[:400]}
            return {
                "started": started,
                "http_status": http_status,
                "token_times": token_times,
                "token_ids": token_ids,
                "event_times": event_times,
                "event_token_counts": event_token_counts,
                "text": "",
                "usage": None,
                "spec_stats": None,
                "error": error,
                "finish_reason": None,
            }
        buffer = b""
        while True:
            chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                doc = json.loads(data)
                if "error" in doc:
                    error = doc["error"]
                    continue
                now = time.monotonic()
                rel = now - started
                tokens_list = doc.get("tokens")
                choices = doc.get("choices") or []
                choice = choices[0] if choices else {}
                delta_text = choice.get("text")
                if not isinstance(delta_text, str):
                    delta = choice.get("delta") or {}
                    if isinstance(delta, dict):
                        content_piece = delta.get("content") or ""
                        reasoning_piece = delta.get("reasoning_content") or ""
                        delta_text = f"{reasoning_piece}{content_piece}"
                if isinstance(delta_text, str) and delta_text:
                    text_parts.append(delta_text)
                if isinstance(tokens_list, list) and tokens_list:
                    count = len(tokens_list)
                    for tok in tokens_list:
                        token_times.append(rel)
                        if isinstance(tok, int):
                            token_ids.append(tok)
                    event_times.append(rel)
                    event_token_counts.append(count)
                elif isinstance(delta_text, str) and delta_text:
                    token_times.append(rel)
                    event_times.append(rel)
                    event_token_counts.append(1)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                if isinstance(doc.get("usage"), dict):
                    usage = doc["usage"]
                if isinstance(doc.get("speculation"), dict):
                    spec_stats = doc["speculation"]
                elif isinstance(doc.get("sparkpipe"), dict) and isinstance(doc["sparkpipe"].get("speculation"), dict):
                    spec_stats = doc["sparkpipe"]["speculation"]
    except Exception as exc:
        error = {"exception": f"{type(exc).__name__}: {exc}"}
    finally:
        conn.close()

    return {
        "started": started,
        "http_status": http_status,
        "token_times": token_times,
        "token_ids": token_ids,
        "event_times": event_times,
        "event_token_counts": event_token_counts,
        "text": "".join(text_parts),
        "usage": usage,
        "spec_stats": spec_stats,
        "error": error,
        "finish_reason": finish,
    }


def read_engine_measurements(api_log_path: str | None) -> dict[str, int] | None:
    if not api_log_path:
        return None
    path = Path(api_log_path)
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > 131072:
                handle.seek(size - 131072)
            raw = handle.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in reversed(raw.splitlines()):
        if '"event":"engine_measurements"' not in line and '"event": "engine_measurements"' not in line:
            continue
        start = line.find("{")
        if start < 0:
            continue
        try:
            doc = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if doc.get("event") == "engine_measurements":
            return {
                "verify_lanes": int(doc.get("verify_lanes") or 0),
                "draft_tokens": int(doc.get("draft_tokens") or 0),
                "accepted_draft_tokens": int(doc.get("accepted_draft_tokens") or 0),
                "tree_frames": int(doc.get("tree_frames") or 0),
                "tree_branch_frames": int(doc.get("tree_branch_frames") or 0),
                "tree_branch_accepted": int(doc.get("tree_branch_accepted") or 0),
            }
    return None


def summarize_stream(result: dict[str, Any]) -> dict[str, Any]:
    times = result["token_times"]
    event_times = result.get("event_times") or []
    event_counts = result.get("event_token_counts") or []
    usage = result.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or len(times)
    prompt_tokens = usage.get("prompt_tokens")
    cached_tokens = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
    out: dict[str, Any] = {
        "error": result["error"],
        "http_status": result["http_status"],
        "finish_reason": result["finish_reason"],
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "text": result["text"],
        "token_ids": list(result.get("token_ids") or []),
    }
    if times:
        out["ttft_ms"] = round(times[0] * 1000.0, 2)
    if len(times) > 1 and times[-1] > times[0]:
        out["decode_tok_s"] = round((len(times) - 1) / (times[-1] - times[0]), 3)
        if len(event_times) > 1 and len(event_times) == len(event_counts):
            itls = [
                (event_times[i] - event_times[i - 1]) * 1000.0 / float(max(1, event_counts[i]))
                for i in range(1, len(event_times))
                if event_times[i] > event_times[i - 1]
            ]
        else:
            itls = [(times[i] - times[i - 1]) * 1000.0 for i in range(1, len(times)) if times[i] > times[i - 1]]
        if itls:
            out["itl_p50_ms"] = round(percentile(itls, 50.0) or 0.0, 3)
            out["itl_p99_ms"] = round(percentile(itls, 99.0) or 0.0, 3)
    if event_counts:
        out["chunk_mean"] = round(sum(event_counts) / float(len(event_counts)), 3)
    spec_stats = result.get("spec_stats")
    if isinstance(spec_stats, dict) and isinstance(spec_stats.get("alpha"), (int, float)):
        out["alpha"] = float(spec_stats["alpha"])
    elif isinstance(spec_stats, dict) and isinstance(spec_stats.get("accepted_per_round"), (int, float)):
        out["alpha"] = 1.0 + float(spec_stats["accepted_per_round"])
    elif event_counts:
        out["alpha_est"] = round(sum(event_counts) / float(len(event_counts)), 3)
    return out


def is_capacity_rejection(error: Any) -> bool:
    if not isinstance(error, dict):
        return False
    status = error.get("http_status")
    body = str(error.get("body", "")).lower()
    if status in (400, 422, 429, 503):
        return True
    for token in ("capacity", "max_sequences", "too many", "queue full", "busy", "limit"):
        if token in body:
            return True
    return False


def is_immediate_eos(result: dict[str, Any]) -> bool:
    return (
        result.get("error") is None
        and result.get("http_status") == 200
        and not result.get("token_times")
        and result.get("finish_reason") == "stop"
    )


def compute_effective_alpha(verify_lanes: int, accepted_drafts: int, total_output_tokens: int) -> float:
    if verify_lanes <= 0:
        return 1.0
    verify_alpha = 1.0 + accepted_drafts / float(verify_lanes)
    if total_output_tokens > accepted_drafts:
        total_steps = max(verify_lanes, total_output_tokens - accepted_drafts)
        return min(verify_alpha, total_output_tokens / float(total_steps))
    return verify_alpha


def format_batch_prompt(prompt_base: str, idx: int, stream_prefix: bool = False) -> str:
    if stream_prefix:
        return f"Task {idx + 1}: {prompt_base}"
    return f"{prompt_base}\n[Stream {idx + 1}] Technical report:\nThe primary "


def run_concurrent_batch(
    base_url: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    prompt_base: Any,
    spec_body: dict[str, Any],
    api_key: str | None,
    timeout: float,
    endpoint: str = "completions",
    stream_prefix: bool = False,
    encoder: PromptEncoder | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any] | None] = [None] * batch_size
    prompt_fields = [
        encoder.batch_prompt_field(prompt_base, i, stream_prefix=stream_prefix)
        if encoder is not None
        else {"prompt": format_batch_prompt(str(prompt_base), i, stream_prefix=stream_prefix)}
        for i in range(batch_size)
    ]

    def worker(idx: int) -> None:
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        body.update(prompt_fields[idx])
        body.update(spec_body)
        results[idx] = stream_completion(base_url, body, api_key=api_key, timeout=timeout, endpoint=endpoint)

    started = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(batch_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = time.monotonic() - started

    completed = [r for r in results if r is not None]
    errors = [r["error"] for r in completed if r["error"] is not None]
    if errors:
        cap_reject = all(is_capacity_rejection(e) for e in errors)
        return {
            "batch": batch_size,
            "unsupported": cap_reject,
            "failed": not cap_reject,
            "errors": errors,
            "aggregate_tok_s": 0.0,
        }

    decoding = [r for r in completed if r["token_times"]]
    if len(decoding) < batch_size:
        eos_empty = any(is_immediate_eos(r) for r in completed)
        return {
            "batch": batch_size,
            "unsupported": False,
            "failed": True,
            "immediate_eos": eos_empty,
            "errors": [{"reason": f"only {len(decoding)}/{batch_size} streams produced tokens"}],
            "aggregate_tok_s": 0.0,
        }

    window_begin = max(r["started"] + r["token_times"][0] for r in decoding)
    window_end = min(r["started"] + r["token_times"][-1] for r in decoding)
    if window_end > window_begin:
        inside = sum(1 for r in decoding for t in r["token_times"] if window_begin < r["started"] + t <= window_end)
        agg_tok_s = inside / (window_end - window_begin)
    else:
        total_tok = sum(len(r["token_times"]) for r in decoding)
        agg_tok_s = total_tok / max(0.001, wall_s)

    total_output_tokens = sum(int((r.get("usage") or {}).get("completion_tokens") or len(r["token_times"])) for r in decoding)
    return {
        "batch": batch_size,
        "unsupported": False,
        "failed": False,
        "errors": [],
        "wall_s": round(wall_s, 3),
        "total_output_tokens": total_output_tokens,
        "aggregate_tok_s": round(agg_tok_s, 2),
        "per_stream_tok_s": round(agg_tok_s / batch_size, 2),
    }


def tier_plan(tier: str, model_spec: rais_score.ModelSpec, max_batch: int) -> dict[str, Any]:
    ctx_cap = max(4096, model_spec.context_tokens - 512)
    if tier == "fast":
        return {
            "tier": "fast",
            "estimated_minutes": "3-4",
            "b1_short_tokens": 128,
            "b1_repeats": 2,
            "b1_long_ctx": min(32768, ctx_cap),
            "b1_long_tokens": 64,
            "spec_tokens": 128,
            "batch_sizes": [b for b in (4, 8, 16, 32, 64) if b <= max_batch],
            "batch_tokens": 96,
            "b8_long_ctx": min(32768, ctx_cap),
            "b8_long_tokens": 64,
            "prefill_8k_tokens": min(8192, ctx_cap),
            "prefill_long_targets": [min(32768, ctx_cap)],
            "warm_targets": [min(8192, ctx_cap)],
            "warm_new_tokens": 128,
            "gate_gx_tokens": 32,
            "run_b8_spec": True,
        }
    long_targets = [min(32768, ctx_cap)]
    if ctx_cap >= 98304:
        long_targets.append(min(131072, ctx_cap))
    warm_targets = [min(8192, ctx_cap)]
    if ctx_cap >= 98304:
        warm_targets.append(min(100000, ctx_cap))
    return {
        "tier": "full",
        "estimated_minutes": "15-25",
        "b1_short_tokens": 256,
        "b1_repeats": 3,
        "b1_long_ctx": min(65536, ctx_cap) if ctx_cap < 131072 else min(131072, ctx_cap),
        "b1_long_tokens": 128,
        "spec_tokens": 256,
        "batch_sizes": [b for b in (4, 8, 16, 32, 64) if b <= max_batch],
        "batch_tokens": 256,
        "b8_long_ctx": min(65536, ctx_cap) if ctx_cap >= 65536 else min(32768, ctx_cap),
        "b8_long_tokens": 128,
        "prefill_8k_tokens": min(8192, ctx_cap),
        "prefill_long_targets": long_targets,
        "warm_targets": warm_targets,
        "warm_new_tokens": 128,
        "gate_gx_tokens": 64,
        "run_b8_spec": True,
    }


def _spec_payload(spec_toggle: str, enabled: bool) -> dict[str, Any]:
    if not spec_toggle or spec_toggle == "none":
        return {}
    return {spec_toggle: enabled}


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    model_name = rais_score.resolve_model_name(args.model)
    model_spec = rais_score.MODEL_SPECS[model_name]
    plan = tier_plan(args.tier, model_spec, args.max_batch)
    if getattr(args, "b8_spec", None) is not None:
        plan["run_b8_spec"] = bool(args.b8_spec)
    encoder = PromptEncoder(
        tokenizer_json=getattr(args, "tokenizer_json", None),
        tokenizer_compiled=getattr(args, "tokenizer_compiled", None),
        tokenizer_bin=getattr(args, "tokenizer_bin", None),
        synthetic_token_ids=bool(getattr(args, "synthetic_token_ids", False)),
    )
    endpoint_mode = getattr(args, "endpoint", "completions") or "completions"
    active_endpoint = "completions" if encoder.is_token_mode() else ("chat" if endpoint_mode == "chat" else "completions")
    endpoint_fallback: str | None = None

    if args.dry_run:
        return {
            "dry_run": True,
            "model": model_name,
            "hardware": args.hardware,
            "endpoint": active_endpoint,
            "prompt_mode": encoder.mode,
            "plan": plan,
        }

    if args.swap_url:
        swap_model(args.swap_url, model_name, args.swap_token, args.timeout)

    def do_stream(body: dict[str, Any]) -> dict[str, Any]:
        nonlocal active_endpoint, endpoint_fallback
        raw = stream_completion(args.base_url, body, api_key=args.api_key, timeout=args.timeout, endpoint=active_endpoint)
        if not encoder.is_token_mode() and endpoint_mode == "auto" and active_endpoint == "completions" and is_immediate_eos(raw):
            active_endpoint = "chat"
            endpoint_fallback = "immediate_eos"
            raw = stream_completion(args.base_url, body, api_key=args.api_key, timeout=args.timeout, endpoint=active_endpoint)
        return raw

    def do_batch(batch_size: int, max_tokens: int, prompt_base: Any, spec_body: dict[str, Any], stream_prefix: bool = False) -> dict[str, Any]:
        nonlocal active_endpoint, endpoint_fallback
        res = run_concurrent_batch(
            args.base_url,
            model_name,
            batch_size,
            max_tokens,
            prompt_base,
            spec_body,
            args.api_key,
            args.timeout,
            endpoint=active_endpoint,
            stream_prefix=stream_prefix,
            encoder=encoder,
        )
        if not encoder.is_token_mode() and endpoint_mode == "auto" and active_endpoint == "completions" and res.get("immediate_eos"):
            active_endpoint = "chat"
            endpoint_fallback = "immediate_eos"
            res = run_concurrent_batch(
                args.base_url,
                model_name,
                batch_size,
                max_tokens,
                prompt_base,
                spec_body,
                args.api_key,
                args.timeout,
                endpoint=active_endpoint,
                stream_prefix=stream_prefix,
                encoder=encoder,
            )
        return res

    nospec_opt = _spec_payload(args.spec_toggle, False)
    spec_opt = _spec_payload(args.spec_toggle, True)
    errors_count = 0

    prose_field = encoder.prompt_field_for_text(PROSE_PROMPT, target_tokens=256 if encoder.mode == "synthetic" else None)
    code_field = encoder.prompt_field_for_text(CODE_PROMPT, target_tokens=256 if encoder.mode == "synthetic" else None)
    edit_field = encoder.prompt_field_for_text(EDIT_PROMPT, target_tokens=256 if encoder.mode == "synthetic" else None)

    b1_rates: list[float] = []
    p50_list: list[float] = []
    p99_list: list[float] = []
    nospec_prose_res: dict[str, Any] | None = None
    nospec_prose_chunk = 1.0

    for rep in range(plan["b1_repeats"]):
        body: dict[str, Any] = {
            "model": model_name,
            **prose_field,
            "max_tokens": plan["b1_short_tokens"],
            "temperature": 0,
        }
        body.update(nospec_opt)
        res = summarize_stream(do_stream(body))
        if res["error"] is not None:
            errors_count += 1
        else:
            if rep == 0:
                nospec_prose_res = res
                nospec_prose_chunk = float(res.get("chunk_mean") or 1.0)
            if res.get("decode_tok_s"):
                b1_rates.append(res["decode_tok_s"])
            if res.get("itl_p50_ms"):
                p50_list.append(res["itl_p50_ms"])
            if res.get("itl_p99_ms"):
                p99_list.append(res["itl_p99_ms"])

    b1_short_tok_s = round(statistics.median(b1_rates), 3) if b1_rates else None
    itl_p50_ms = round(statistics.median(p50_list), 3) if p50_list else None
    itl_p99_ms = round(statistics.median(p99_list), 3) if p99_list else None

    spec_metrics: dict[str, Any] = {}
    prose_exact: bool | None = True
    code_exact: bool | None = True

    if args.spec_toggle and args.spec_toggle != "none":
        domain_prompts = (
            ("prose", prose_field, nospec_prose_res, nospec_prose_chunk),
            ("code", code_field, None, 1.0),
            ("edit", edit_field, None, 1.0),
        )
        spec_tok_rates: list[float] = []
        nospec_tok_rates: list[float] = []
        alphas: dict[str, float] = {}
        total_verify_lanes = 0
        total_draft_tokens = 0
        total_accepted_tokens = 0
        total_tree_frames = 0
        total_tree_branch_accepted = 0
        log_used = False

        for domain, prompt_field, cached_nospec_res, cached_chunk_mean in domain_prompts:
            if cached_nospec_res is not None:
                base_res = cached_nospec_res
                base_rate = b1_short_tok_s
                base_chunk = cached_chunk_mean
            else:
                nb: dict[str, Any] = {
                    "model": model_name,
                    **prompt_field,
                    "max_tokens": plan["spec_tokens"],
                    "temperature": 0,
                }
                nb.update(nospec_opt)
                nr = summarize_stream(do_stream(nb))
                if nr["error"] is not None:
                    errors_count += 1
                    continue
                base_res = nr
                base_rate = nr.get("decode_tok_s")
                base_chunk = float(nr.get("chunk_mean") or 1.0)

            sb: dict[str, Any] = {
                "model": model_name,
                **prompt_field,
                "max_tokens": plan["spec_tokens"],
                "temperature": 0,
            }
            sb.update(spec_opt)
            em_before = read_engine_measurements(args.api_log)
            sr = summarize_stream(do_stream(sb))
            em_after = read_engine_measurements(args.api_log)
            if sr["error"] is not None:
                errors_count += 1
                continue

            exact = outputs_match(sr, base_res)
            if domain == "prose":
                prose_exact = exact
            else:
                code_exact = bool(code_exact and exact)

            dom_speedup = 1.0
            if sr.get("decode_tok_s") and base_rate:
                spec_tok_rates.append(sr["decode_tok_s"])
                nospec_tok_rates.append(base_rate)
                dom_speedup = sr["decode_tok_s"] / base_rate

            alpha_val: float | None = sr.get("alpha")
            if alpha_val is None and em_before is not None and em_after is not None:
                log_used = True
                d_lanes = max(0, em_after["verify_lanes"] - em_before["verify_lanes"])
                d_draft = max(0, em_after["draft_tokens"] - em_before["draft_tokens"])
                d_acc = max(0, em_after["accepted_draft_tokens"] - em_before["accepted_draft_tokens"])
                d_tf = max(0, em_after["tree_frames"] - em_before["tree_frames"])
                d_tba = max(0, em_after["tree_branch_accepted"] - em_before["tree_branch_accepted"])
                total_verify_lanes += d_lanes
                total_draft_tokens += d_draft
                total_accepted_tokens += d_acc
                total_tree_frames += d_tf
                total_tree_branch_accepted += d_tba
                if d_lanes > 0:
                    alpha_val = 1.0 + d_acc / float(d_lanes)
                else:
                    alpha_val = 1.0
            elif alpha_val is None:
                if dom_speedup <= 1.03:
                    alpha_val = 1.0
                elif base_chunk <= 1.15 and sr.get("alpha_est"):
                    alpha_val = float(sr["alpha_est"])
                else:
                    alpha_val = max(1.0, dom_speedup)
            if alpha_val is not None:
                alphas[domain] = round(float(alpha_val), 3)
                spec_metrics[f"alpha_{domain}"] = alphas[domain]

        if log_used:
            spec_metrics["verify_lanes"] = total_verify_lanes
            spec_metrics["draft_tokens"] = total_draft_tokens
            spec_metrics["accepted_draft_tokens"] = total_accepted_tokens
            spec_metrics["tree_frames"] = total_tree_frames
            spec_metrics["tree_branch_accepted"] = total_tree_branch_accepted

        if alphas:
            alpha_mean = sum(alphas.values()) / len(alphas)
            spec_metrics["alpha"] = round(alpha_mean, 3)
            spec_active = (total_verify_lanes > 0) if log_used else (alpha_mean > 1.03)
            spec_metrics["verify_b1_enabled"] = spec_active
            if spec_tok_rates and nospec_tok_rates:
                mean_spec = statistics.mean(spec_tok_rates)
                mean_nospec = statistics.mean(nospec_tok_rates)
                spec_metrics["b1_spec_tok_s"] = round(mean_spec, 3)
                speedup = mean_spec / mean_nospec if mean_nospec > 0 else 1.0
                if spec_active and speedup > 0 and alpha_mean > 1.001:
                    spec_metrics["verify_v_b1"] = round(alpha_mean / speedup, 4)

    batch_results: dict[str, Any] = {}
    for b in (4, 8, 16, 32, 64):
        if b > args.max_batch:
            batch_results[str(b)] = {"unsupported": True, "aggregate_tok_s": 0.0}
            continue
        res_b = do_batch(b, plan["batch_tokens"], PROSE_PROMPT, nospec_opt)
        if res_b.get("failed"):
            errors_count += len(res_b.get("errors") or [1])
        batch_results[str(b)] = res_b

    if plan["run_b8_spec"] and args.spec_toggle and args.spec_toggle != "none":
        b8_nospec = batch_results.get("8", {}).get("aggregate_tok_s") or 0.0
        if b8_nospec > 0.0:
            res_b8_edit_nospec = do_batch(8, plan["batch_tokens"], EDIT_PROMPT, nospec_opt, stream_prefix=True)
            b8_edit_base = res_b8_edit_nospec.get("aggregate_tok_s") or 0.0
            if not res_b8_edit_nospec.get("failed") and not res_b8_edit_nospec.get("unsupported") and b8_edit_base > 0.0:
                b8_nospec = b8_edit_base
                spec_metrics["b8_edit_nospec_tok_s"] = round(b8_edit_base, 2)
            em_b8_before = read_engine_measurements(args.api_log)
            res_b8_spec = do_batch(8, plan["batch_tokens"], EDIT_PROMPT, spec_opt, stream_prefix=True)
            em_b8_after = read_engine_measurements(args.api_log)
            if not res_b8_spec.get("failed") and not res_b8_spec.get("unsupported"):
                b8_spec_agg = res_b8_spec.get("aggregate_tok_s") or 0.0
                b8_total_gen = int(res_b8_spec.get("total_output_tokens") or 0)
                d_b8_lanes = 0
                d_b8_draft = 0
                d_b8_acc = 0
                if em_b8_before is not None and em_b8_after is not None:
                    d_b8_lanes = max(0, em_b8_after["verify_lanes"] - em_b8_before["verify_lanes"])
                    d_b8_draft = max(0, em_b8_after["draft_tokens"] - em_b8_before["draft_tokens"])
                    d_b8_acc = max(0, em_b8_after["accepted_draft_tokens"] - em_b8_before["accepted_draft_tokens"])
                    spec_metrics["b8_verify_lanes"] = d_b8_lanes
                    spec_metrics["b8_draft_tokens"] = d_b8_draft
                    spec_metrics["b8_accepted_draft_tokens"] = d_b8_acc
                if d_b8_lanes > 0:
                    a_b8 = round(compute_effective_alpha(d_b8_lanes, d_b8_acc, b8_total_gen), 3)
                    spec_metrics["verify_b8_enabled"] = True
                    spec_metrics["alpha_b8"] = a_b8
                    spec_metrics["b8_spec_tok_s"] = round(b8_spec_agg, 2)
                    spec_metrics["verify_v_b8"] = round(a_b8 / max(0.01, b8_spec_agg / b8_nospec), 4)
                elif b8_spec_agg > b8_nospec * 1.03 and spec_metrics.get("verify_b1_enabled"):
                    a_b8 = spec_metrics.get("alpha", 1.5)
                    spec_metrics["verify_b8_enabled"] = True
                    spec_metrics["alpha_b8"] = a_b8
                    spec_metrics["b8_spec_tok_s"] = round(b8_spec_agg, 2)
                    spec_metrics["verify_v_b8"] = round(a_b8 / (b8_spec_agg / b8_nospec), 4)

    nonce_8k = f"pf8k-{time.time_ns()}"
    _, prompt_8k_handle = encoder.prompt_field_for_target(plan["prefill_8k_tokens"], nonce_8k)
    cold_8k_field, cold_8k_handle = encoder.extend_prompt(prompt_8k_handle, FOLLOWUP_DELTA_128, delta_target_tokens=plan["warm_new_tokens"])
    cold_8k_res = summarize_stream(
        do_stream({"model": model_name, **cold_8k_field, "max_tokens": plan["gate_gx_tokens"], "temperature": 0, **nospec_opt})
    )
    prefill_8k_tokens = cold_8k_res.get("prompt_tokens") or plan["prefill_8k_tokens"]
    prefill_8k_tok_s = None
    if cold_8k_res["error"] is not None:
        errors_count += 1
    elif cold_8k_res.get("ttft_ms") and cold_8k_res["ttft_ms"] > 0:
        prefill_8k_tok_s = round(prefill_8k_tokens * 1000.0 / cold_8k_res["ttft_ms"], 2)

    warm_gx_res = summarize_stream(
        do_stream({"model": model_name, **cold_8k_field, "max_tokens": plan["gate_gx_tokens"], "temperature": 0, **nospec_opt})
    )
    prefix_exact = False
    if warm_gx_res["error"] is not None:
        errors_count += 1
    else:
        prefix_exact = outputs_match(warm_gx_res, cold_8k_res)

    warm_8k_field, _ = encoder.extend_prompt(
        cold_8k_handle,
        FOLLOWUP_TURN2,
        delta_target_tokens=plan["warm_new_tokens"],
    )
    warm_8k_res = summarize_stream(
        do_stream({"model": model_name, **warm_8k_field, "max_tokens": 16, "temperature": 0, **nospec_opt})
    )
    warm_8k_ttft_ms = None
    if warm_8k_res["error"] is not None:
        errors_count += 1
    else:
        warm_8k_ttft_ms = warm_8k_res.get("ttft_ms")

    prefill_32k_tok_s = None
    prefill_32k_tokens = None
    prefill_128k_tok_s = None
    prefill_128k_tokens = None
    b1_long_tok_s = None
    b1_long_ctx = plan["b1_long_ctx"]
    b8_long_tok_s = None
    b8_long_ctx = plan["b8_long_ctx"]
    warm_100k_ttft_ms = None
    warm_100k_prefix = None

    long_prompt_for_decode: Any = None
    for target_len in plan["prefill_long_targets"]:
        nonce_l = f"pfl-{target_len}-{time.time_ns()}"
        p_long_field, p_long_handle = encoder.prompt_field_for_target(target_len, nonce_l)
        out_toks = plan["b1_long_tokens"] if target_len == max(plan["prefill_long_targets"]) else 1
        r_long = summarize_stream(
            do_stream({"model": model_name, **p_long_field, "max_tokens": out_toks, "temperature": 0, **nospec_opt})
        )
        if r_long["error"] is not None:
            errors_count += 1
            continue
        pt = r_long.get("prompt_tokens") or target_len
        pf_rate = round(pt * 1000.0 / r_long["ttft_ms"], 2) if r_long.get("ttft_ms") else None
        if target_len <= 40000:
            prefill_32k_tokens = pt
            prefill_32k_tok_s = pf_rate
        else:
            prefill_128k_tokens = pt
            prefill_128k_tok_s = pf_rate
        if target_len == max(plan["prefill_long_targets"]):
            b1_long_ctx = pt
            b1_long_tok_s = r_long.get("decode_tok_s")
            long_prompt_for_decode = p_long_handle

    if len(plan["warm_targets"]) > 1 and long_prompt_for_decode is not None:
        w100_field, _ = encoder.extend_prompt(long_prompt_for_decode, FOLLOWUP_DELTA_128, delta_target_tokens=plan["warm_new_tokens"])
        r_w100 = summarize_stream(
            do_stream({"model": model_name, **w100_field, "max_tokens": 16, "temperature": 0, **nospec_opt})
        )
        if r_w100["error"] is not None:
            errors_count += 1
        else:
            warm_100k_ttft_ms = r_w100.get("ttft_ms")
            warm_100k_prefix = b1_long_ctx

    if long_prompt_for_decode is not None:
        if b8_long_ctx >= b1_long_ctx:
            b8_long_prompt = long_prompt_for_decode
        else:
            _, b8_long_prompt = encoder.prompt_field_for_target(b8_long_ctx, f"b8l-{time.time_ns()}")
        b8_long_res = do_batch(min(8, args.max_batch), plan["b8_long_tokens"], b8_long_prompt, nospec_opt)
        if b8_long_res.get("failed"):
            errors_count += len(b8_long_res.get("errors") or [1])
        elif not b8_long_res.get("unsupported"):
            b8_long_tok_s = b8_long_res.get("aggregate_tok_s")

    metrics_payload: dict[str, Any] = {
        "b1_short_tok_s": b1_short_tok_s,
        "b1_short_ctx": 256,
        "b1_long_tok_s": b1_long_tok_s,
        "b1_long_ctx": b1_long_ctx,
        "itl_p50_ms": itl_p50_ms,
        "itl_p99_ms": itl_p99_ms,
        "speculation": spec_metrics,
        "batch": batch_results,
        "b8_long_tok_s": b8_long_tok_s,
        "b8_long_ctx": b8_long_ctx,
        "prefill_8k_tokens": prefill_8k_tokens,
        "prefill_8k_tok_s": prefill_8k_tok_s,
        "prefill_32k_tokens": prefill_32k_tokens,
        "prefill_32k_tok_s": prefill_32k_tok_s,
        "prefill_128k_tokens": prefill_128k_tokens,
        "prefill_128k_tok_s": prefill_128k_tok_s,
        "warm_8k_prefix_tokens": prefill_8k_tokens,
        "warm_8k_new_tokens": plan["warm_new_tokens"],
        "warm_8k_ttft_ms": warm_8k_ttft_ms,
        "warm_100k_prefix_tokens": warm_100k_prefix,
        "warm_100k_new_tokens": plan["warm_new_tokens"],
        "warm_100k_ttft_ms": warm_100k_ttft_ms,
    }

    record: dict[str, Any] = {
        "rais_version": rais_score.RAIS_VERSION,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_name,
        "engine": args.engine,
        "tier": args.tier,
        "hardware": args.hardware,
        "base_url": args.base_url,
        "endpoint": active_endpoint,
        "prompt_mode": encoder.mode,
        "metrics": metrics_payload,
        "gates": {
            "prose_exact": prose_exact,
            "code_exact": code_exact,
            "prefix_exact": prefix_exact,
            "errors": errors_count,
        },
    }
    if endpoint_fallback is not None:
        record["endpoint_fallback"] = endpoint_fallback
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rais_bench", description="RAIS v1.0 OpenAI-compatible HTTP benchmark runner.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="Served model name (any of the 21 lanes or aliases).")
    parser.add_argument("--tier", default="fast", choices=("fast", "full"), help="Execution tier: fast (3-4m) or full (15-25m).")
    parser.add_argument("--hardware", default="16x-gb10-1x100g", choices=sorted(rais_score.HARDWARE_PROFILES))
    parser.add_argument("--engine", default="sparkpipe", help="Engine label recorded in JSONL output.")
    parser.add_argument("--spec-toggle", default="speculation", help="JSON field name toggling speculation, or 'none'.")
    parser.add_argument("--b8-spec", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable the B=8 batched speculation probe.")
    parser.add_argument("--endpoint", default="completions", choices=("completions", "chat", "auto"), help="Endpoint mode: completions, chat, or auto (fallback on immediate token-0 EOS).")
    parser.add_argument("--tokenizer-json", default=None, help="Optional HuggingFace tokenizer.json path to send prompt_token_ids instead of prompt text.")
    parser.add_argument("--tokenizer-compiled", default=None, help="Optional SparkPipe compiled tokenizer path (used with --tokenizer-bin) to send prompt_token_ids.")
    parser.add_argument("--tokenizer-bin", default=None, help="Optional sparkpipe_tokenize_prompt binary path for zero-dep real tokenization.")
    parser.add_argument("--synthetic-token-ids", action="store_true", help="Send deterministic vocab-safe non-EOS prompt_token_ids for tokenizerless endpoints.")
    parser.add_argument("--api-key", default=None, help="Optional Bearer token for /v1/completions.")
    parser.add_argument("--api-log", default=None, help="Optional engine API log path for exact verify/draft counters.")
    parser.add_argument("--swap-url", default=None, help="Optional model swap endpoint URL.")
    parser.add_argument("--swap-token", default=None, help="Optional Bearer token for --swap-url.")
    parser.add_argument("--max-batch", type=int, default=64, help="Maximum batch concurrency to attempt.")
    parser.add_argument("--timeout", type=float, default=900.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--output", default=None, help="Append RAIS JSONL record to this path.")
    parser.add_argument("--score", action="store_true", help="Print Markdown RAIS scorecard after benchmark finishes.")
    parser.add_argument("--dry-run", action="store_true", help="Emit planned tier workload without sending HTTP requests.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    record = run_benchmark(args)
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return 0
    line = json.dumps(record)
    if args.output:
        with open(args.output, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
    else:
        print(line)
    if args.score:
        report = rais_score.score_fleet([record], args.hardware)
        print(rais_score.render_markdown(report), file=sys.stderr if not args.output else sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())

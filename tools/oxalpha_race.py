#!/usr/bin/env python3
"""Local OpenAI-compatible request racer for Ox Alpha providers.

The proxy preserves one caller conversation while sending each individual model
turn to distinct provider failure domains.  It buffers upstream responses until
one is structurally complete, returns that winner, and cancels the remaining
requests.  Provider credentials are resolved from environment variables or
local credential files and are never accepted in the pool configuration.
"""

from __future__ import annotations

import argparse
import dataclasses
import http.client
import http.server
import itertools
import json
import math
import os
import queue
import re
import secrets
import shlex
import socket
import ssl
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Sequence


CONTEXT_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
SECRET_HEADER_RE = re.compile(
    r"(?:authorization|cookie|credential|api[-_]?key|access[-_]?key|private[-_]?key|"
    r"password|secret|token)",
    re.I,
)
SECRET_VALUE_RE = re.compile(
    r"(?:\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}|"
    r"\bsk-[A-Za-z0-9._~-]{8,})",
    re.I,
)
AUTHORIZATION_VALUE_RE = re.compile(
    r"(?i)\bauthorization\b['\"]?\s*[:=]\s*['\"]?"
    r"(?:[A-Za-z][A-Za-z0-9._~-]*\s+)?[^\s,;'\"}]{4,}"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:authorization|api[-_ ]?key|access[-_ ]?key|credential|password|secret|token)"
    r"\b\s*[:=]\s*[^\s,;]{4,}"
)
EventCallback = Callable[[str, dict[str, Any]], None]
EVENT_QUEUE_LIMIT = 4096


class PoolConfigurationError(ValueError):
    """The provider pool configuration is unsafe or malformed."""


class RaceFailure(RuntimeError):
    """No upstream returned a complete valid response."""


@dataclasses.dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    failure_domain: str
    base_url: str
    model: str
    auth: dict[str, Any]
    headers: dict[str, str]
    enabled: bool
    hedging_authorized: bool
    failure_domains: tuple[str, ...] = ()
    tool_choice: str | None = None
    stream: bool | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None

    @property
    def correlation_domains(self) -> tuple[str, ...]:
        return self.failure_domains or (self.failure_domain,)


@dataclasses.dataclass(frozen=True)
class PoolSettings:
    virtual_model: str
    redundancy: int
    first_byte_timeout_seconds: float
    idle_timeout_seconds: float
    request_timeout_seconds: float
    failure_cooldown_seconds: float
    max_request_bytes: int
    max_response_bytes: int
    providers: tuple[ProviderSpec, ...]


@dataclasses.dataclass
class ProviderHealth:
    calls: int = 0
    wins: int = 0
    failures: int = 0
    cancellations: int = 0
    in_flight: int = 0
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    last_status: int | None = None
    last_error: str | None = None
    last_latency_seconds: float | None = None
    last_first_byte_seconds: float | None = None
    last_started_at: float | None = None
    last_finished_at: float | None = None


@dataclasses.dataclass
class UpstreamResult:
    provider_id: str
    valid: bool
    status: int | None
    content_type: str
    body: bytes
    elapsed_seconds: float
    first_byte_seconds: float | None
    error: str | None = None
    canceled: bool = False
    completed_monotonic: float | None = None


def _number(value: Any, name: str, minimum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PoolConfigurationError(f"{name} must be a number >= {minimum}")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise PoolConfigurationError(f"{name} must be a finite number >= {minimum}")
    return number


def _integer(value: Any, name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise PoolConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _timeout(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number >= 0")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return timeout


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise PoolConfigurationError(f"invalid {name}: {value!r}")
    return value


def _validate_base_url(value: Any, provider_id: str) -> str:
    if not isinstance(value, str):
        raise PoolConfigurationError(f"{provider_id} base_url must be a string")
    parsed = urllib.parse.urlsplit(value)
    loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise PoolConfigurationError(f"{provider_id} base_url must use HTTPS or loopback HTTP")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise PoolConfigurationError(f"unsafe {provider_id} base_url")
    return value.rstrip("/")


def load_pool_settings(path: Path) -> PoolSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
        raise PoolConfigurationError("provider pool schema_version must be 1")
    virtual_model = _identifier(raw.get("virtual_model", "ox-alpha"), "virtual_model")
    redundancy = _integer(raw.get("redundancy", 2), "redundancy", 1)
    if redundancy > 8:
        raise PoolConfigurationError("redundancy must be <= 8")
    provider_values = raw.get("providers")
    if not isinstance(provider_values, list) or not provider_values:
        raise PoolConfigurationError("providers must be a non-empty array")
    providers = []
    provider_ids = set()
    for value in provider_values:
        if not isinstance(value, dict):
            raise PoolConfigurationError("each provider must be an object")
        provider_id = _identifier(value.get("id"), "provider id")
        if provider_id in provider_ids:
            raise PoolConfigurationError(f"duplicate provider id: {provider_id}")
        provider_ids.add(provider_id)
        raw_domains = value.get("failure_domains")
        if raw_domains is not None and "failure_domain" in value:
            raise PoolConfigurationError(
                f"{provider_id} must use failure_domains or failure_domain, not both"
            )
        if raw_domains is None:
            domains = (_identifier(value.get("failure_domain", provider_id), "failure_domain"),)
        else:
            if (
                not isinstance(raw_domains, list)
                or not raw_domains
                or len(raw_domains) > 8
                or not all(isinstance(item, str) for item in raw_domains)
            ):
                raise PoolConfigurationError(f"{provider_id} failure_domains must be a non-empty array")
            domains = tuple(_identifier(item, "failure domain") for item in raw_domains)
            if len(set(domains)) != len(domains):
                raise PoolConfigurationError(f"{provider_id} failure_domains contain duplicates")
        model = value.get("model")
        if not isinstance(model, str) or not model:
            raise PoolConfigurationError(f"{provider_id} model must be a non-empty string")
        tool_choice = value.get("tool_choice")
        if tool_choice not in {None, "auto"}:
            raise PoolConfigurationError(
                f"{provider_id} tool_choice must be auto when configured"
            )
        stream = value.get("stream")
        if stream is not None and not isinstance(stream, bool):
            raise PoolConfigurationError(f"{provider_id} stream must be boolean")
        max_output_tokens = value.get("max_output_tokens")
        if max_output_tokens is not None:
            max_output_tokens = _integer(
                max_output_tokens, f"{provider_id} max_output_tokens", 256
            )
        reasoning_effort = value.get("reasoning_effort")
        if reasoning_effort not in {None, "low", "high", "max"}:
            raise PoolConfigurationError(
                f"{provider_id} reasoning_effort must be low, high, or max"
            )
        auth = value.get("auth", {"kind": "none"})
        if not isinstance(auth, dict):
            raise PoolConfigurationError(f"{provider_id} auth must be an object")
        if auth.get("kind") not in {"none", "env", "file", "json"}:
            raise PoolConfigurationError(f"{provider_id} has unsupported auth kind")
        headers = value.get("headers", {})
        if not isinstance(headers, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in headers.items()
        ):
            raise PoolConfigurationError(f"{provider_id} headers must contain strings")
        invalid_headers = sorted(
            key
            for key, item in headers.items()
            if not HEADER_NAME_RE.fullmatch(key)
            or "\r" in item
            or "\n" in item
            or "\0" in item
        )
        if invalid_headers:
            raise PoolConfigurationError(
                f"{provider_id} has invalid static headers: {invalid_headers}"
            )
        forbidden = sorted(key for key in headers if SECRET_HEADER_RE.search(key))
        if forbidden:
            raise PoolConfigurationError(
                f"{provider_id} secret headers belong in auth, not configuration: {forbidden}"
            )
        secret_values = sorted(key for key, item in headers.items() if SECRET_VALUE_RE.search(item))
        if secret_values:
            raise PoolConfigurationError(
                f"{provider_id} static header values must not contain credentials: {secret_values}"
            )
        providers.append(
            ProviderSpec(
                provider_id=provider_id,
                failure_domain=domains[0],
                base_url=_validate_base_url(value.get("base_url"), provider_id),
                model=model,
                auth=dict(auth),
                headers=dict(headers),
                enabled=value.get("enabled", True) is True,
                hedging_authorized=value.get("hedging_authorized", False) is True,
                failure_domains=domains,
                tool_choice=tool_choice,
                stream=stream,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
            )
        )
    eligible = [item for item in providers if item.enabled and item.hedging_authorized]
    if not eligible:
        raise PoolConfigurationError("at least one enabled provider must authorize hedging")
    return PoolSettings(
        virtual_model=virtual_model,
        redundancy=redundancy,
        first_byte_timeout_seconds=_number(
            raw.get("first_byte_timeout_seconds", 30), "first_byte_timeout_seconds", 1
        ),
        idle_timeout_seconds=_number(raw.get("idle_timeout_seconds", 30), "idle_timeout_seconds", 1),
        request_timeout_seconds=_number(
            raw.get("request_timeout_seconds", 180), "request_timeout_seconds", 1
        ),
        failure_cooldown_seconds=_number(
            raw.get("failure_cooldown_seconds", 30), "failure_cooldown_seconds", 0
        ),
        max_request_bytes=_integer(
            raw.get("max_request_bytes", 64 << 20), "max_request_bytes", 1
        ),
        max_response_bytes=_integer(
            raw.get("max_response_bytes", 64 << 20), "max_response_bytes", 1
        ),
        providers=tuple(providers),
    )


def _secure_text_file(path: Path) -> str:
    stat_result = path.stat()
    if stat_result.st_mode & 0o077:
        raise PoolConfigurationError(f"credential file permissions are too broad: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise PoolConfigurationError(f"credential file is empty: {path}")
    return value


def _nested_value(value: Any, keys: Sequence[str]) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise PoolConfigurationError(f"credential JSON path is missing: {'.'.join(keys)}")
        current = current[key]
    return current


def _resolve_auth_header_and_secret(provider: ProviderSpec) -> tuple[str, str, str] | None:
    auth = provider.auth
    kind = auth.get("kind")
    if kind == "none":
        return None
    if kind == "env":
        name = auth.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]+", name):
            raise PoolConfigurationError(f"{provider.provider_id} has invalid auth env name")
        secret = os.environ.get(name, "").strip()
        if not secret:
            raise PoolConfigurationError(f"{provider.provider_id} credential is unavailable")
    elif kind == "file":
        value = auth.get("path")
        if not isinstance(value, str):
            raise PoolConfigurationError(f"{provider.provider_id} auth file path is missing")
        secret = _secure_text_file(Path(value).expanduser())
    elif kind == "json":
        value = auth.get("path")
        keys = auth.get("keys")
        if not isinstance(value, str) or not isinstance(keys, list) or not all(
            isinstance(key, str) and key for key in keys
        ):
            raise PoolConfigurationError(f"{provider.provider_id} JSON auth path is invalid")
        path = Path(value).expanduser()
        encoded = _secure_text_file(path)
        secret_value = _nested_value(json.loads(encoded), keys)
        if not isinstance(secret_value, str) or not secret_value.strip():
            raise PoolConfigurationError(f"{provider.provider_id} JSON credential is empty")
        secret = secret_value.strip()
    else:
        raise PoolConfigurationError(f"{provider.provider_id} has unsupported auth kind")
    if len(secret) < 4:
        raise PoolConfigurationError(f"{provider.provider_id} credential is too short")
    header = auth.get("header", "Authorization")
    prefix = auth.get("prefix", "Bearer ")
    if not isinstance(header, str) or not re.fullmatch(r"[A-Za-z0-9-]+", header):
        raise PoolConfigurationError(f"{provider.provider_id} auth header is invalid")
    if not isinstance(prefix, str) or any(ord(character) < 32 or ord(character) == 127 for character in prefix):
        raise PoolConfigurationError(f"{provider.provider_id} auth prefix is invalid")
    value = prefix + secret
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise PoolConfigurationError(f"{provider.provider_id} credential contains invalid characters")
    return header, value, secret


def resolve_auth_header(provider: ProviderSpec) -> tuple[str, str] | None:
    resolved = _resolve_auth_header_and_secret(provider)
    if resolved is None:
        return None
    return resolved[0], resolved[1]


def load_provider_env_file(settings: PoolSettings, path: Path) -> list[str]:
    """Load only provider credential variables from a private dotenv file."""
    required = {
        provider.auth.get("name")
        for provider in settings.providers
        if provider.auth.get("kind") == "env"
    }
    required = {
        name for name in required if isinstance(name, str) and re.fullmatch(r"[A-Z][A-Z0-9_]+", name)
    }
    if not path.exists():
        return []
    encoded = _secure_text_file(path)
    loaded = []
    for line_number, raw_line in enumerate(encoded.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]+", name):
            raise PoolConfigurationError(f"invalid dotenv name on line {line_number}")
        if name not in required or name in os.environ:
            continue
        try:
            values = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as error:
            raise PoolConfigurationError(f"invalid dotenv value on line {line_number}") from error
        if len(values) != 1 or not values[0]:
            raise PoolConfigurationError(f"dotenv value must be one non-empty string on line {line_number}")
        os.environ[name] = values[0]
        loaded.append(name)
    return sorted(loaded)


def _validate_tool_calls(value: Any) -> tuple[bool, str | None]:
    if not isinstance(value, list) or not value:
        return False, "tool_calls must be a non-empty array"
    identifiers = set()
    indices = set()
    for call in value:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            return False, "malformed tool call"
        call_index = call.get("index")
        if call_index is not None:
            if not isinstance(call_index, int) or isinstance(call_index, bool) or call_index < 0:
                return False, "invalid tool call index"
            if call_index in indices:
                return False, "duplicate tool call index"
            indices.add(call_index)
        if call.get("type") != "function":
            return False, "unsupported tool call type"
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in identifiers:
            return False, "invalid or duplicate tool call id"
        identifiers.add(call_id)
        function = call["function"]
        if not isinstance(function.get("name"), str) or not function["name"].strip():
            return False, "tool call is missing a function name"
        if "arguments" not in function:
            return False, "tool call is missing arguments"
        arguments = function["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return False, "tool call arguments are incomplete JSON"
        if not isinstance(arguments, dict):
            return False, "tool call arguments must decode to an object"
    return True, None


def _validate_complete_choice(choice: Any) -> tuple[bool, str | None]:
    if not isinstance(choice, dict):
        return False, "choice is not an object"
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        return False, "incomplete choice"
    if finish_reason not in {"stop", "tool_calls"}:
        return False, f"non-success finish reason: {finish_reason}"
    message = choice.get("message")
    if not isinstance(message, dict):
        return False, "choice message is not an object"
    content = message.get("content")
    calls = message.get("tool_calls")
    has_content = isinstance(content, str) and bool(content.strip())
    if calls is not None:
        valid, error = _validate_tool_calls(calls)
        if not valid:
            return False, error
    if finish_reason == "tool_calls" and calls is None:
        return False, "tool_calls finish reason has no tool calls"
    if finish_reason == "stop" and calls is not None:
        return False, "stop finish reason unexpectedly contains tool calls"
    if not has_content and calls is None:
        return False, "choice has neither non-empty content nor tool calls"
    return True, None


def _validate_stream_events(body: bytes) -> tuple[bool, str | None]:
    choices: dict[int, dict[str, Any]] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(b":"):
            continue
        if not line.startswith(b"data:"):
            return False, "malformed SSE line"
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False, "invalid SSE JSON"
        if not isinstance(value, dict) or value.get("error") is not None:
            return False, "upstream SSE error object"
        raw_choices = value.get("choices")
        if not isinstance(raw_choices, list):
            return False, "SSE choices is not an array"
        event_choice_indices = set()
        for raw_choice in raw_choices:
            if not isinstance(raw_choice, dict):
                return False, "malformed SSE choice"
            index = raw_choice.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                return False, "invalid SSE choice index"
            if index != 0:
                return False, "nonzero SSE choice index"
            if index in event_choice_indices:
                return False, "duplicate SSE choice index"
            event_choice_indices.add(index)
            aggregate = choices.setdefault(
                index,
                {"message": {"role": "assistant", "content": ""}, "finish_reason": None},
            )
            delta = raw_choice.get("delta", {})
            if not isinstance(delta, dict):
                return False, "malformed SSE delta"
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    return False, "SSE content delta is not text"
                aggregate["message"]["content"] += content
            raw_calls = delta.get("tool_calls")
            if raw_calls is not None:
                if not isinstance(raw_calls, list):
                    return False, "SSE tool_calls delta is not an array"
                call_map = aggregate.setdefault("call_map", {})
                for raw_call in raw_calls:
                    if not isinstance(raw_call, dict):
                        return False, "malformed SSE tool call delta"
                    call_index = raw_call.get("index")
                    if not isinstance(call_index, int) or isinstance(call_index, bool) or call_index < 0:
                        return False, "invalid SSE tool call index"
                    call = call_map.setdefault(
                        call_index,
                        {"id": None, "type": None, "function": {"name": "", "arguments": ""}},
                    )
                    if raw_call.get("id") is not None:
                        if not isinstance(raw_call["id"], str):
                            return False, "invalid SSE tool call id"
                        call["id"] = raw_call["id"]
                    if raw_call.get("type") is not None:
                        call["type"] = raw_call["type"]
                    function = raw_call.get("function", {})
                    if not isinstance(function, dict):
                        return False, "malformed SSE function delta"
                    for key in ("name", "arguments"):
                        fragment = function.get(key)
                        if fragment is not None:
                            if not isinstance(fragment, str):
                                return False, f"SSE function {key} delta is not text"
                            call["function"][key] += fragment
            finish_reason = raw_choice.get("finish_reason")
            if finish_reason is not None:
                if aggregate["finish_reason"] not in (None, finish_reason):
                    return False, "conflicting SSE finish reason"
                aggregate["finish_reason"] = finish_reason
    if not choices:
        return False, "incomplete SSE response"
    aggregate = choices.get(0)
    if aggregate is None:
        return False, "SSE response has no choice index 0"
    call_map = aggregate.pop("call_map", None)
    if call_map is not None:
        call_indices = sorted(call_map)
        if call_indices != list(range(len(call_indices))):
            return False, "SSE tool call indices are not contiguous"
        aggregate["message"]["tool_calls"] = [call_map[index] for index in call_indices]
    return _validate_complete_choice(aggregate)


def validate_chat_response(body: bytes, streamed: bool) -> tuple[bool, str | None]:
    if not body:
        return False, "empty response"
    if not streamed:
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False, "invalid JSON response"
        if not isinstance(value, dict) or value.get("error") is not None:
            return False, "upstream error object"
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices:
            return False, "missing choices"
        indexed_choices = {}
        for choice in choices:
            if not isinstance(choice, dict):
                return False, "malformed choice"
            index = choice.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                return False, "invalid choice index"
            if index != 0:
                return False, "nonzero choice index"
            if index in indexed_choices:
                return False, "duplicate choice index"
            indexed_choices[index] = choice
        choice_zero = indexed_choices.get(0)
        if choice_zero is None:
            return False, "response has no choice index 0"
        return _validate_complete_choice(choice_zero)
    return _validate_stream_events(body)


def _redact_error_text(value: str, sensitive_values: Sequence[str] = ()) -> str:
    redacted = value
    for sensitive in sensitive_values:
        if len(sensitive) >= 4:
            redacted = redacted.replace(sensitive, "[redacted]")
            if " " in sensitive:
                redacted = redacted.replace(sensitive.split(" ", 1)[1], "[redacted]")
    redacted = SECRET_VALUE_RE.sub("[redacted]", redacted)
    redacted = AUTHORIZATION_VALUE_RE.sub("[redacted]", redacted)
    return SECRET_ASSIGNMENT_RE.sub("[redacted]", redacted)


def safe_upstream_error(body: bytes, sensitive_values: Sequence[str] = ()) -> str:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(value, dict):
        return ""
    error = value.get("error", value)
    candidates = []
    if isinstance(error, dict):
        candidates.extend(error.get(key) for key in ("message", "detail", "type", "code"))
    elif isinstance(error, str):
        candidates.append(error)
    values = [item.strip() for item in candidates if isinstance(item, str) and item.strip()]
    if not values:
        return ""
    summary = " / ".join(values)
    return _redact_error_text(summary, sensitive_values)[:500]


def _provider_sensitive_values(provider: ProviderSpec) -> tuple[str, ...]:
    try:
        resolved = _resolve_auth_header_and_secret(provider)
    except Exception:
        return ()
    if resolved is None:
        return ()
    return resolved[1], resolved[2]


class ProviderRacePool:
    @staticmethod
    def _validate_settings(settings: PoolSettings) -> None:
        redundancy = _integer(settings.redundancy, "redundancy", 1)
        if redundancy > 8:
            raise PoolConfigurationError("redundancy must be <= 8")
        _integer(settings.max_request_bytes, "max_request_bytes", 1)
        _integer(settings.max_response_bytes, "max_response_bytes", 1)
        for name in (
            "first_byte_timeout_seconds",
            "idle_timeout_seconds",
            "request_timeout_seconds",
        ):
            if _number(getattr(settings, name), name, 0) == 0:
                raise PoolConfigurationError(f"{name} must be greater than 0")
        _number(
            settings.failure_cooldown_seconds,
            "failure_cooldown_seconds",
            0,
        )

    def __init__(
        self,
        settings: PoolSettings,
        *,
        event_callback: EventCallback | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self._validate_settings(settings)
        self.settings = settings
        self.event_callback = event_callback
        self.clock = clock
        self.monotonic = monotonic
        self._lock = threading.RLock()
        self._lifecycle_changed = threading.Condition(self._lock)
        self._closing = threading.Event()
        self._active_races: dict[str, threading.Event] = {}
        self._worker_threads: set[threading.Thread] = set()
        self._event_close_lock = threading.Lock()
        self._event_queue: queue.Queue[tuple[int, str, dict[str, Any]] | None] = queue.Queue(
            maxsize=EVENT_QUEUE_LIMIT
        )
        self._event_thread: threading.Thread | None = None
        self._event_closed = False
        self._event_stop_enqueued = False
        self._health = {provider.provider_id: ProviderHealth() for provider in settings.providers}
        self._last_winner: dict[str, str] = {}
        self._rotation: dict[str, int] = {}
        self._connections: dict[tuple[str, str], http.client.HTTPConnection] = {}
        self._requests_started = 0
        self._requests_won = 0
        self._requests_failed = 0
        self._event_callback_failures = 0
        self._last_event_callback_error: str | None = None
        self._events_enqueued = 0
        self._events_delivered = 0
        self._last_event_enqueued_at: float | None = None
        self._last_event_delivered_at: float | None = None
        if self.event_callback is not None:
            self._start_event_dispatcher()

    def _start_event_dispatcher(self) -> None:
        thread = threading.Thread(
            target=self._dispatch_events,
            name="oxalpha-event-dispatch",
            daemon=True,
        )
        self._event_thread = thread
        try:
            thread.start()
        except RuntimeError:
            self._event_thread = None
            self._event_callback_failures += 1
            self._last_event_callback_error = "event callback dispatcher failed to start"

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.event_callback is None:
            return
        with self._lock:
            if self._closing.is_set() or self._event_closed:
                self._event_callback_failures += 1
                self._last_event_callback_error = "event emitted after callback dispatcher closed"
                return
            if self._event_thread is None or not self._event_thread.is_alive():
                self._event_callback_failures += 1
                self._last_event_callback_error = "event callback dispatcher is unavailable"
                return
            sequence = self._events_enqueued + 1
            current = dict(payload)
            current["event_sequence"] = sequence
            try:
                self._event_queue.put_nowait((sequence, event, current))
            except queue.Full:
                self._event_callback_failures += 1
                self._last_event_callback_error = "event callback queue is full"
                return
            self._events_enqueued = sequence
            self._last_event_enqueued_at = self.clock()

    def _dispatch_events(self) -> None:
        while True:
            item = self._event_queue.get()
            try:
                if item is None:
                    return
                sequence, event, payload = item
                try:
                    assert self.event_callback is not None
                    self.event_callback(event, payload)
                except BaseException:
                    with self._lock:
                        self._event_callback_failures += 1
                        self._last_event_callback_error = "event callback failed"
                finally:
                    with self._lock:
                        self._events_delivered = max(self._events_delivered, sequence)
                        self._last_event_delivered_at = self.clock()
            finally:
                self._event_queue.task_done()

    def _flush_events_until(self, deadline: float) -> bool:
        with self._event_queue.all_tasks_done:
            while self._event_queue.unfinished_tasks:
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    return False
                self._event_queue.all_tasks_done.wait(remaining)
            return True

    def flush_events(self, timeout_seconds: float = 5.0) -> bool:
        deadline = self.monotonic() + _timeout(timeout_seconds, "timeout_seconds")
        return self._flush_events_until(deadline)

    def _acquire_until(self, lock: Any, deadline: float) -> bool:
        remaining = deadline - self.monotonic()
        if remaining <= 0:
            return lock.acquire(blocking=False)
        return lock.acquire(timeout=remaining)

    def _wait_for_lifecycle_until(self, deadline: float) -> bool:
        if not self._acquire_until(self._lock, deadline):
            return False
        try:
            while (
                self._active_races
                or any(thread.is_alive() for thread in self._worker_threads)
                or any(health.in_flight > 0 for health in self._health.values())
            ):
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    return False
                self._lifecycle_changed.wait(remaining)
            return True
        finally:
            self._lock.release()

    def close(self, timeout_seconds: float = 5.0) -> bool:
        timeout = _timeout(timeout_seconds, "timeout_seconds")
        deadline = self.monotonic() + timeout
        self._closing.set()
        if not self._acquire_until(self._event_close_lock, deadline):
            return False
        try:
            if not self._acquire_until(self._lock, deadline):
                return False
            try:
                self._event_closed = True
                thread = self._event_thread
                cancel_events = list(self._active_races.values())
                connections = list(self._connections.values())
            finally:
                self._lock.release()
            for cancel_event in cancel_events:
                cancel_event.set()
            for connection in connections:
                try:
                    connection.close()
                except Exception:
                    pass
            stop_ready = True
            if thread is not None and thread.is_alive() and not self._event_stop_enqueued:
                remaining = deadline - self.monotonic()
                try:
                    if remaining > 0:
                        self._event_queue.put(None, timeout=remaining)
                    else:
                        self._event_queue.put_nowait(None)
                except queue.Full:
                    stop_ready = False
                else:
                    self._event_stop_enqueued = True
            events_flushed = self._flush_events_until(deadline)
            lifecycle_settled = self._wait_for_lifecycle_until(deadline)
            if thread is threading.current_thread():
                return False
            if thread is not None:
                thread.join(timeout=max(0.0, deadline - self.monotonic()))
            dispatcher_settled = thread is None or not thread.is_alive()
            return stop_ready and events_flushed and lifecycle_settled and dispatcher_settled
        finally:
            self._event_close_lock.release()

    @staticmethod
    def _select_disjoint_group(
        providers: Sequence[ProviderSpec], occupied: set[str], limit: int
    ) -> tuple[ProviderSpec, ...]:
        maximum = min(max(0, limit), len(providers))
        for width in range(maximum, 0, -1):
            for group in itertools.combinations(providers, width):
                domains = set(occupied)
                for provider in group:
                    if not domains.isdisjoint(provider.correlation_domains):
                        break
                    domains.update(provider.correlation_domains)
                else:
                    return group
        return ()

    @staticmethod
    def _max_disjoint_width(providers: Sequence[ProviderSpec], limit: int) -> int:
        return len(ProviderRacePool._select_disjoint_group(providers, set(), limit))

    def snapshot(self) -> dict[str, Any]:
        now = self.clock()
        with self._lock:
            eligible = [
                provider
                for provider in self.settings.providers
                if provider.enabled
                and provider.hedging_authorized
                and self._credential_available(provider)
            ]
            healthy = [
                provider
                for provider in eligible
                if self._health[provider.provider_id].circuit_open_until <= now
            ]
            providers = []
            for spec in self.settings.providers:
                health = dataclasses.asdict(self._health[spec.provider_id])
                health["id"] = spec.provider_id
                health["failure_domain"] = spec.failure_domain
                health["failure_domains"] = list(spec.correlation_domains)
                health["enabled"] = spec.enabled
                health["hedging_authorized"] = spec.hedging_authorized
                health["circuit_open"] = health["circuit_open_until"] > now
                providers.append(health)
            return {
                "virtual_model": self.settings.virtual_model,
                "redundancy": self.settings.redundancy,
                "configured_redundancy": self.settings.redundancy,
                "effective_redundancy": self._max_disjoint_width(
                    healthy, self.settings.redundancy
                ),
                "eligible_provider_count": len(eligible),
                "healthy_provider_count": len(healthy),
                "requests_started": self._requests_started,
                "requests_won": self._requests_won,
                "requests_failed": self._requests_failed,
                "closing": self._closing.is_set(),
                "active_races": len(self._active_races),
                "active_provider_workers": sum(
                    thread.is_alive() for thread in self._worker_threads
                ),
                "event_callback_failures": self._event_callback_failures,
                "last_event_callback_error": self._last_event_callback_error,
                "event_callback_lag_events": self._events_enqueued - self._events_delivered,
                "event_queue_depth": self._event_queue.qsize(),
                "events_enqueued": self._events_enqueued,
                "events_delivered": self._events_delivered,
                "last_event_enqueued_at": self._last_event_enqueued_at,
                "last_event_delivered_at": self._last_event_delivered_at,
                "snapshot_generated_at": now,
                "providers": providers,
            }

    def wait_for_idle(self, timeout_seconds: float | None = None) -> bool:
        if timeout_seconds is None:
            timeout_seconds = self.settings.idle_timeout_seconds + 1.5
        deadline = self.monotonic() + _timeout(timeout_seconds, "timeout_seconds")
        while True:
            with self._lock:
                if (
                    all(health.in_flight == 0 for health in self._health.values())
                    and not self._active_races
                    and not any(thread.is_alive() for thread in self._worker_threads)
                ):
                    return True
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))

    def _credential_available(self, provider: ProviderSpec) -> bool:
        try:
            resolve_auth_header(provider)
            return True
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _candidate_order(self, context_key: str) -> list[ProviderSpec]:
        now = self.clock()
        with self._lock:
            eligible = [
                provider
                for provider in self.settings.providers
                if provider.enabled and provider.hedging_authorized and self._credential_available(provider)
            ]
            healthy = [
                provider
                for provider in eligible
                if self._health[provider.provider_id].circuit_open_until <= now
            ]
            preferred = self._last_winner.get(context_key)
            offset = self._rotation.get(context_key, 0)
            ordered = sorted(
                healthy,
                key=lambda item: (
                    0 if item.provider_id == preferred else 1,
                    self._health[item.provider_id].consecutive_failures,
                    self._health[item.provider_id].last_latency_seconds or 0.0,
                    item.provider_id,
                ),
            )
            if len(ordered) > 1:
                tail = [item for item in ordered if item.provider_id != preferred]
                if tail:
                    shift = offset % len(tail)
                    tail = tail[shift:] + tail[:shift]
                    ordered = ([item for item in ordered if item.provider_id == preferred] + tail)
            self._rotation[context_key] = offset + 1
            return ordered

    def _start_call(self, provider: ProviderSpec) -> bool:
        with self._lock:
            health = self._health[provider.provider_id]
            if health.circuit_open_until > self.clock():
                return False
            health.calls += 1
            health.in_flight += 1
            health.last_started_at = self.clock()
            return True

    def _rollback_start_call(self, provider: ProviderSpec) -> None:
        with self._lock:
            health = self._health[provider.provider_id]
            health.calls = max(0, health.calls - 1)
            health.in_flight = max(0, health.in_flight - 1)

    def _finish_call(self, result: UpstreamResult, won: bool) -> None:
        with self._lock:
            health = self._health[result.provider_id]
            health.in_flight = max(0, health.in_flight - 1)
            health.last_finished_at = self.clock()
            health.last_status = result.status
            health.last_latency_seconds = result.elapsed_seconds
            health.last_first_byte_seconds = result.first_byte_seconds
            if won:
                health.wins += 1
                health.consecutive_failures = 0
                health.circuit_open_until = 0.0
                health.last_error = None
            elif result.canceled:
                health.cancellations += 1
            elif not result.valid:
                health.failures += 1
                health.consecutive_failures += 1
                health.last_error = (result.error or "invalid response")[:500]
                health.circuit_open_until = max(
                    health.circuit_open_until,
                    self.clock() + self.settings.failure_cooldown_seconds,
                )

    def _register_connection(
        self, request_id: str, provider_id: str, connection: http.client.HTTPConnection
    ) -> None:
        with self._lock:
            self._connections[(request_id, provider_id)] = connection

    def _unregister_connection(self, request_id: str, provider_id: str) -> None:
        with self._lock:
            self._connections.pop((request_id, provider_id), None)

    def _cancel_connections(self, request_id: str, winner: str | None = None) -> None:
        with self._lock:
            connections = [
                connection
                for (current_request, provider_id), connection in self._connections.items()
                if current_request == request_id and provider_id != winner
            ]
        for connection in connections:
            try:
                connection.close()
            except OSError:
                pass

    def _call_provider(
        self,
        request_id: str,
        provider: ProviderSpec,
        body: dict[str, Any],
        cancel_event: threading.Event,
    ) -> UpstreamResult:
        started = self.monotonic()
        first_byte = None
        status = None
        response_body = b""
        content_type = "application/json"
        connection: http.client.HTTPConnection | None = None
        sensitive_values: list[str] = []
        try:
            if cancel_event.is_set():
                raise InterruptedError("canceled before upstream request")
            parsed = urllib.parse.urlsplit(provider.base_url)
            if parsed.scheme == "https":
                connection = http.client.HTTPSConnection(
                    parsed.hostname,
                    parsed.port or 443,
                    timeout=self.settings.first_byte_timeout_seconds,
                    context=ssl.create_default_context(),
                )
            else:
                connection = http.client.HTTPConnection(
                    parsed.hostname,
                    parsed.port or 80,
                    timeout=self.settings.first_byte_timeout_seconds,
                )
            self._register_connection(request_id, provider.provider_id, connection)
            if cancel_event.is_set():
                raise InterruptedError("canceled before upstream request")
            request = dict(body)
            request["model"] = provider.model
            if provider.tool_choice is not None and "tools" in request:
                request["tool_choice"] = provider.tool_choice
            if provider.stream is not None:
                request["stream"] = provider.stream
                if not provider.stream:
                    request.pop("stream_options", None)
            if provider.max_output_tokens is not None:
                requested_tokens = request.get("max_tokens")
                if isinstance(requested_tokens, int) and not isinstance(requested_tokens, bool):
                    request["max_tokens"] = min(
                        requested_tokens, provider.max_output_tokens
                    )
            if provider.reasoning_effort is not None:
                request["reasoning_effort"] = provider.reasoning_effort
            encoded = json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            if len(encoded) > self.settings.max_request_bytes:
                raise ValueError("upstream request exceeds size limit")
            path = (parsed.path.rstrip("/") + "/chat/completions") or "/chat/completions"
            headers = {
                "Accept": "text/event-stream" if request.get("stream") is True else "application/json",
                "Content-Type": "application/json",
                "User-Agent": "sparkpipe-oxalpha-race/1",
                **provider.headers,
            }
            auth_header = _resolve_auth_header_and_secret(provider)
            if auth_header is not None:
                headers[auth_header[0]] = auth_header[1]
                sensitive_values.extend((auth_header[1], auth_header[2]))
            connection.request("POST", path, body=encoded, headers=headers)
            response = connection.getresponse()
            if cancel_event.is_set():
                raise InterruptedError("canceled after another provider won")
            first_byte = self.monotonic() - started
            status = response.status
            content_type = response.getheader("Content-Type", content_type)
            if connection.sock is not None:
                connection.sock.settimeout(self.settings.idle_timeout_seconds)
            if status < 200 or status >= 300:
                response_body = response.read(min(65536, self.settings.max_response_bytes) + 1)
                if len(response_body) > self.settings.max_response_bytes:
                    raise ValueError("upstream error response exceeds size limit")
                detail = safe_upstream_error(
                    response_body,
                    sensitive_values,
                )
                return UpstreamResult(
                    provider.provider_id,
                    False,
                    status,
                    content_type,
                    response_body,
                    self.monotonic() - started,
                    first_byte,
                    error=f"HTTP {status}" + (f": {detail}" if detail else ""),
                )
            if request.get("stream") is True:
                chunks = []
                response_size = 0
                while True:
                    if cancel_event.is_set():
                        raise InterruptedError("canceled after another provider won")
                    if self.monotonic() - started >= self.settings.request_timeout_seconds:
                        raise TimeoutError("request deadline exceeded")
                    remaining = self.settings.max_response_bytes - response_size
                    line = response.readline(remaining + 1)
                    if not line:
                        break
                    chunks.append(line)
                    response_size += len(line)
                    if response_size > self.settings.max_response_bytes:
                        raise ValueError("upstream stream exceeds size limit")
                response_body = b"".join(chunks)
                streamed = True
            else:
                response_body = response.read(self.settings.max_response_bytes + 1)
                if len(response_body) > self.settings.max_response_bytes:
                    raise ValueError("upstream response exceeds size limit")
                streamed = False
            valid, error = validate_chat_response(response_body, streamed)
            return UpstreamResult(
                provider.provider_id,
                valid,
                status,
                content_type,
                response_body,
                self.monotonic() - started,
                first_byte,
                error=error,
            )
        except InterruptedError as error:
            return UpstreamResult(
                provider.provider_id,
                False,
                status,
                content_type,
                b"",
                self.monotonic() - started,
                first_byte,
                error=str(error),
                canceled=True,
            )
        except Exception as error:
            canceled = cancel_event.is_set()
            rendered = _redact_error_text(
                f"{type(error).__name__}: {error}",
                sensitive_values,
            )[:500]
            return UpstreamResult(
                provider.provider_id,
                False,
                status,
                content_type,
                response_body,
                self.monotonic() - started,
                first_byte,
                error=("canceled" if canceled else rendered),
                canceled=canceled,
            )
        finally:
            self._unregister_connection(request_id, provider.provider_id)
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass

    def race(self, body: dict[str, Any], context_key: str) -> tuple[str, UpstreamResult]:
        if self._closing.is_set():
            raise RaceFailure("provider race pool is closing")
        if not CONTEXT_KEY_RE.fullmatch(context_key):
            raise RaceFailure("invalid context key")
        try:
            encoded = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise RaceFailure(f"request is not JSON serializable: {error}") from error
        if len(encoded) > self.settings.max_request_bytes:
            raise RaceFailure("request exceeds configured size limit")
        candidates = self._candidate_order(context_key)
        if not candidates:
            now = self.clock()
            with self._lock:
                cooling = [
                    self._health[provider.provider_id].circuit_open_until - now
                    for provider in self.settings.providers
                    if provider.enabled
                    and provider.hedging_authorized
                    and self._credential_available(provider)
                    and self._health[provider.provider_id].circuit_open_until > now
                ]
            if cooling:
                raise RaceFailure(
                    f"all authorized providers are cooling down; retry after {min(cooling):.3f}s"
                )
            raise RaceFailure("no authorized provider credential is available")
        request_id = secrets.token_hex(8)
        cancel_event = threading.Event()
        with self._lock:
            if self._closing.is_set():
                raise RaceFailure("provider race pool is closing")
            self._requests_started += 1
            self._active_races[request_id] = cancel_event
        try:
            return self._race_admitted(
                body,
                context_key,
                candidates,
                request_id,
                cancel_event,
            )
        finally:
            with self._lock:
                self._active_races.pop(request_id, None)
                self._lifecycle_changed.notify_all()

    def _race_admitted(
        self,
        body: dict[str, Any],
        context_key: str,
        candidates: Sequence[ProviderSpec],
        request_id: str,
        cancel_event: threading.Event,
    ) -> tuple[str, UpstreamResult]:
        results: queue.Queue[UpstreamResult] = queue.Queue()
        settlement_lock = threading.Lock()
        self_settle: dict[str, bool] = {}
        pending = list(candidates)
        active_providers: dict[str, set[str]] = {}
        launched_providers: list[str] = []
        active = 0
        failures = []
        deadline = self.monotonic() + self.settings.request_timeout_seconds

        def launch(
            provider: ProviderSpec,
            notifications: list[tuple[str, dict[str, Any]]],
        ) -> bool:
            nonlocal active
            if not self._start_call(provider):
                return False
            active += 1
            active_providers[provider.provider_id] = set(provider.correlation_domains)

            def deregister_worker() -> None:
                with self._lock:
                    self._worker_threads.discard(threading.current_thread())
                    self._lifecycle_changed.notify_all()

            def publish_result(result: UpstreamResult) -> None:
                with self._lock:
                    self._worker_threads.discard(threading.current_thread())
                    results.put_nowait(result)
                    self._lifecycle_changed.notify_all()

            def run_worker() -> None:
                worker_started = self.monotonic()
                call_started = self.monotonic()
                if call_started >= deadline:
                    completed_monotonic = call_started
                    result = UpstreamResult(
                        provider.provider_id,
                        False,
                        None,
                        "application/json",
                        b"",
                        max(0.0, call_started - worker_started),
                        None,
                        error="request deadline exceeded",
                    )
                elif cancel_event.is_set():
                    completed_monotonic = call_started
                    result = UpstreamResult(
                        provider.provider_id,
                        False,
                        None,
                        "application/json",
                        b"",
                        max(0.0, call_started - worker_started),
                        None,
                        error="canceled",
                        canceled=True,
                    )
                else:
                    try:
                        result = self._call_provider(request_id, provider, body, cancel_event)
                        completed_monotonic = self.monotonic()
                        if not isinstance(result, UpstreamResult):
                            raise TypeError("provider worker returned an invalid result")
                        if result.provider_id != provider.provider_id:
                            raise ValueError("provider worker returned a mismatched provider id")
                        if cancel_event.is_set():
                            result = dataclasses.replace(
                                result,
                                valid=False,
                                canceled=True,
                                error="canceled",
                            )
                    except BaseException as error:
                        completed_monotonic = self.monotonic()
                        canceled = cancel_event.is_set()
                        detail = _redact_error_text(
                            f"worker exception: {type(error).__name__}: {error}",
                            _provider_sensitive_values(provider),
                        )[:500]
                        result = UpstreamResult(
                            provider.provider_id,
                            False,
                            None,
                            "application/json",
                            b"",
                            self.monotonic() - worker_started,
                            None,
                            error="canceled" if canceled else detail,
                            canceled=canceled,
                        )
                result = dataclasses.replace(
                    result,
                    completed_monotonic=completed_monotonic,
                )
                with settlement_lock:
                    deadline_failure = self_settle.get(provider.provider_id)
                    if deadline_failure is None:
                        publish_result(result)
                        return
                settle(result, deadline_failure)

            def worker() -> None:
                try:
                    run_worker()
                finally:
                    deregister_worker()

            thread = threading.Thread(
                target=worker,
                name=f"oxalpha-{request_id}-{provider.provider_id}",
                daemon=True,
            )
            aborted = False
            try:
                with self._lock:
                    if (
                        self._closing.is_set()
                        or cancel_event.is_set()
                        or self.monotonic() >= deadline
                    ):
                        aborted = True
                    else:
                        self._worker_threads.add(thread)
                        try:
                            thread.start()
                        except BaseException:
                            self._worker_threads.discard(thread)
                            self._lifecycle_changed.notify_all()
                            raise
            except BaseException as error:
                active -= 1
                active_providers.pop(provider.provider_id, None)
                self._rollback_start_call(provider)
                detail = _redact_error_text(
                    f"worker start failed: {error}",
                    _provider_sensitive_values(provider),
                )[:500]
                failures.append({"provider": provider.provider_id, "error": detail, "status": None})
                notifications.append(
                    (
                        "provider_launch_failed",
                        {
                            "request_id": request_id,
                            "context_key": context_key,
                            "provider": provider.provider_id,
                            "error": detail,
                        },
                    )
                )
                return False
            if aborted:
                active -= 1
                active_providers.pop(provider.provider_id, None)
                self._rollback_start_call(provider)
                return False
            launched_providers.append(provider.provider_id)
            notifications.append(
                (
                    "provider_started",
                    {
                        "request_id": request_id,
                        "context_key": context_key,
                        "provider": provider.provider_id,
                        "failure_domains": list(provider.correlation_domains),
                    },
                )
            )
            return True

        def fill() -> None:
            if self.monotonic() >= deadline:
                return
            notifications: list[tuple[str, dict[str, Any]]] = []
            while (
                active < self.settings.redundancy
                and pending
                and not cancel_event.is_set()
                and not self._closing.is_set()
                and self.monotonic() < deadline
            ):
                now = self.clock()
                with self._lock:
                    healthy_pending = [
                        provider
                        for provider in pending
                        if self._health[provider.provider_id].circuit_open_until <= now
                    ]
                available = self.settings.redundancy - active
                occupied = set().union(*active_providers.values()) if active_providers else set()
                selected = self._select_disjoint_group(healthy_pending, occupied, available)
                if not selected:
                    break
                for provider in selected:
                    if self.monotonic() >= deadline:
                        break
                    pending.remove(provider)
                    launch(provider, notifications)
            for event, payload in notifications:
                self._emit(event, payload)

        def settle(result: UpstreamResult, deadline_failure: bool) -> None:
            if deadline_failure:
                result = dataclasses.replace(
                    result,
                    valid=False,
                    canceled=False,
                    error="request deadline exceeded",
                )
            self._finish_call(result, False)
            self._emit(
                "provider_settled",
                {
                    "request_id": request_id,
                    "context_key": context_key,
                    "provider": result.provider_id,
                    "canceled": result.canceled,
                    "valid_loser": result.valid,
                },
            )

        def handoff_settlement(provider_ids: set[str], deadline_failure: bool) -> None:
            queued = []
            deferred = []
            with settlement_lock:
                for provider_id in provider_ids:
                    self_settle[provider_id] = deadline_failure
                while True:
                    try:
                        result = results.get_nowait()
                    except queue.Empty:
                        break
                    if result.provider_id in provider_ids:
                        queued.append(result)
                    else:
                        deferred.append(result)
                for result in deferred:
                    results.put(result)
            for result in queued:
                settle(result, deadline_failure)

        fill()
        self._emit(
            "race_started",
            {
                "request_id": request_id,
                "context_key": context_key,
                "providers": list(launched_providers),
                "failure_domains": sorted(
                    set().union(*active_providers.values()) if active_providers else set()
                ),
            },
        )
        while active > 0:
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                break
            try:
                result = results.get(timeout=remaining)
            except queue.Empty:
                break
            # Publication and worker deregistration share this lock.  Taking it
            # after get prevents result processing from outracing deregistration.
            with self._lock:
                pass
            active -= 1
            active_providers.pop(result.provider_id, None)
            if cancel_event.is_set():
                result = dataclasses.replace(
                    result,
                    valid=False,
                    canceled=True,
                    error="canceled",
                )
            if result.valid:
                completed_monotonic = result.completed_monotonic
                if completed_monotonic is None or not math.isfinite(completed_monotonic):
                    result = dataclasses.replace(
                        result,
                        valid=False,
                        canceled=False,
                        error="provider result has no trusted completion timestamp",
                    )
                elif completed_monotonic >= deadline:
                    result = dataclasses.replace(
                        result,
                        valid=False,
                        canceled=False,
                        error="request deadline exceeded",
                    )
            if result.valid:
                self._finish_call(result, True)
                with self._lock:
                    self._last_winner[context_key] = result.provider_id
                    self._requests_won += 1
                cancel_event.set()
                self._cancel_connections(request_id, result.provider_id)
                self._emit(
                    "race_won",
                    {
                        "request_id": request_id,
                        "context_key": context_key,
                        "provider": result.provider_id,
                        "latency_seconds": result.elapsed_seconds,
                        "first_byte_seconds": result.first_byte_seconds,
                    },
                )
                if active:
                    handoff_settlement(set(active_providers), False)
                return request_id, result
            self._finish_call(result, False)
            failures.append({"provider": result.provider_id, "error": result.error, "status": result.status})
            self._emit(
                "provider_failed",
                {
                    "request_id": request_id,
                    "context_key": context_key,
                    "provider": result.provider_id,
                    "error": result.error,
                    "status": result.status,
                },
            )
            if self.monotonic() < deadline:
                fill()
        cancel_event.set()
        self._cancel_connections(request_id)
        unsettled = set(active_providers) if active else set()
        closing_failure = self._closing.is_set()
        if unsettled and not closing_failure:
            with self._lock:
                cooldown_until = self.clock() + self.settings.failure_cooldown_seconds
                for provider_id in unsettled:
                    health = self._health[provider_id]
                    health.circuit_open_until = max(
                        health.circuit_open_until,
                        cooldown_until,
                    )
                    health.last_error = "request deadline exceeded"
        with self._lock:
            self._requests_failed += 1
        self._emit(
            "race_failed",
            {"request_id": request_id, "context_key": context_key, "failures": failures},
        )
        if unsettled:
            handoff_settlement(unsettled, not closing_failure)
        summary = "; ".join(f"{item['provider']}: {item['error']}" for item in failures)
        if closing_failure:
            raise RaceFailure(summary or "provider race pool is closing")
        raise RaceFailure(summary or "all providers exceeded the request deadline")


class RaceProxyServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        pool: ProviderRacePool,
        token: str,
    ):
        self.pool = pool
        self.token = token
        super().__init__(address, RaceProxyHandler)


class RaceProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: RaceProxyServer

    def _write_json(self, status: int, value: Any) -> None:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _authorized(self) -> bool:
        expected = "Bearer " + self.server.token
        return secrets.compare_digest(self.headers.get("Authorization", ""), expected)

    def _read_request_body(self, length: int) -> bytes:
        deadline = time.monotonic() + self.server.pool.settings.first_byte_timeout_seconds
        remaining = length
        chunks = []
        reader = getattr(self.rfile, "read1", self.rfile.read)
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise TimeoutError("request body deadline exceeded")
            self.connection.settimeout(max(0.001, timeout))
            chunk = reader(min(65536, remaining))
            if not chunk:
                raise ValueError("request body ended before Content-Length")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/healthz":
            self._write_json(200, {"status": "ok"})
            return
        if not self._authorized():
            self._write_json(401, {"error": {"message": "unauthorized", "type": "authentication_error"}})
            return
        if parsed.path == "/v1/models":
            model = self.server.pool.settings.virtual_model
            self._write_json(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
        elif parsed.path == "/v1/race/status":
            self._write_json(200, self.server.pool.snapshot())
        else:
            self._write_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/v1/chat/completions":
            self._write_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        if not self._authorized():
            self._write_json(401, {"error": {"message": "unauthorized", "type": "authentication_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length < 0 or length > self.server.pool.settings.max_request_bytes:
            self._write_json(413, {"error": {"message": "invalid request size", "type": "invalid_request_error"}})
            return
        try:
            encoded = self._read_request_body(length)
        except (TimeoutError, socket.timeout):
            self.close_connection = True
            self._write_json(408, {"error": {"message": "request body timeout", "type": "timeout_error"}})
            return
        except ValueError:
            self.close_connection = True
            self._write_json(
                400,
                {"error": {"message": "incomplete request body", "type": "invalid_request_error"}},
            )
            return
        except OSError:
            return
        try:
            body = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
            return
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            self._write_json(400, {"error": {"message": "messages are required", "type": "invalid_request_error"}})
            return
        context_key = self.headers.get("X-Oxalpha-Context-Key", "default")
        try:
            request_id, winner = self.server.pool.race(body, context_key)
        except RaceFailure as error:
            self._write_json(503, {"error": {"message": str(error), "type": "provider_unavailable"}})
            return
        self.send_response(200)
        self.send_header("Content-Type", winner.content_type)
        self.send_header("Content-Length", str(len(winner.body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Oxalpha-Provider", winner.provider_id)
        self.send_header("X-Oxalpha-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(winner.body)

    def log_message(self, format_string: str, *arguments: Any) -> None:
        return


def start_proxy(
    pool: ProviderRacePool,
    host: str,
    port: int,
    token: str,
) -> tuple[RaceProxyServer, threading.Thread]:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise PoolConfigurationError("the bootstrap proxy must bind to loopback")
    server = RaceProxyServer((host, port), pool, token)
    thread = threading.Thread(target=server.serve_forever, name="oxalpha-race-proxy", daemon=True)
    thread.start()
    return server, thread


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--env-file", type=Path, default=Path("~/.env"))
    serve = subparsers.add_parser("serve")
    serve.add_argument("--config", type=Path, required=True)
    serve.add_argument("--env-file", type=Path, default=Path("~/.env"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8788)
    serve.add_argument("--token-env", default="OXALPHA_RACE_PROXY_TOKEN")
    probe = subparsers.add_parser("probe")
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--env-file", type=Path, default=Path("~/.env"))
    probe.add_argument("--provider", action="append", default=[])
    probe.add_argument("--include-disabled", action="store_true")
    probe.add_argument("--require-tool-call", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    settings = load_pool_settings(arguments.config)
    load_provider_env_file(settings, arguments.env_file.expanduser())
    for provider in settings.providers:
        if provider.enabled and provider.hedging_authorized:
            resolve_auth_header(provider)
    if arguments.command == "validate":
        print(
            f"valid provider pool: {len(settings.providers)} providers, "
            f"redundancy={settings.redundancy}"
        )
        return 0
    if arguments.command == "probe":
        selected = set(arguments.provider)
        providers = [
            provider
            for provider in settings.providers
            if ((provider.enabled and provider.hedging_authorized) or arguments.include_disabled)
            and (not selected or provider.provider_id in selected)
        ]
        missing = selected - {provider.provider_id for provider in providers}
        if missing:
            raise SystemExit(
                f"providers are unavailable for this probe selection: {sorted(missing)}"
            )
        failures = 0
        for provider in providers:
            probe_provider = dataclasses.replace(provider, enabled=True, hedging_authorized=True)
            single = dataclasses.replace(settings, redundancy=1, providers=(probe_provider,))
            pool = ProviderRacePool(single)
            request: dict[str, Any] = {
                "model": settings.virtual_model,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "temperature": 0,
                "max_tokens": 2048 if arguments.require_tool_call else 128,
                "stream": False,
            }
            if arguments.require_tool_call:
                request["messages"] = [
                    {"role": "user", "content": "Call report_ready with ready=true."}
                ]
                request["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": "report_ready",
                            "description": "Report probe readiness.",
                            "parameters": {
                                "type": "object",
                                "properties": {"ready": {"type": "boolean"}},
                                "required": ["ready"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ]
                request["tool_choice"] = {
                    "type": "function",
                    "function": {"name": "report_ready"},
                }
            try:
                _, result = pool.race(request, f"probe-{provider.provider_id}")
                response = json.loads(result.body)
                message = response["choices"][0]["message"]
                tool_ok = True
                if arguments.require_tool_call:
                    calls = message.get("tool_calls")
                    tool_ok = (
                        isinstance(calls, list)
                        and len(calls) == 1
                        and calls[0].get("function", {}).get("name") == "report_ready"
                    )
                if not tool_ok:
                    raise RaceFailure("valid completion did not contain the required tool call")
                print(
                    f"{provider.provider_id}: OK status={result.status} "
                    f"latency={result.elapsed_seconds:.3f}s "
                    f"tool_call={tool_ok if arguments.require_tool_call else 'not-tested'}"
                )
            except (RaceFailure, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                failures += 1
                print(f"{provider.provider_id}: FAILED {type(error).__name__}: {error}")
        return 1 if failures else 0
    token = os.environ.get(arguments.token_env, "")
    if not token:
        raise SystemExit(f"missing local proxy token in {arguments.token_env}")
    pool = ProviderRacePool(settings)
    server, thread = start_proxy(pool, arguments.host, arguments.port, token)
    print(f"Ox Alpha race proxy: http://{arguments.host}:{server.server_port}/v1", flush=True)
    try:
        thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Host-only fault tests for the Ox Alpha provider racing proxy."""

import http.client
import http.server
import importlib.util
import dataclasses
import json
import os
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("oxalpha_race", ROOT / "tools" / "oxalpha_race.py")
assert SPEC is not None and SPEC.loader is not None
RACE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RACE
SPEC.loader.exec_module(RACE)


def completion(name, content="ok"):
    return {
        "id": f"chatcmpl-{name}",
        "object": "chat.completion",
        "model": name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def tool_call(name="edit", arguments=None, *, call_id="call-edit"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments if arguments is not None else {}),
        },
    }


class Upstream:
    def __init__(self, name, *, delay=0.0, status=200, valid=True, stream=False, payload=None):
        self.name = name
        self.delay = delay
        self.status = status
        self.valid = valid
        self.stream = stream
        self.payload = payload
        self.calls = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                owner.calls.append(request)
                time.sleep(owner.delay)
                if owner.payload is not None:
                    payload = owner.payload
                    content_type = "text/event-stream" if owner.stream else "application/json"
                elif owner.stream:
                    if owner.valid:
                        chunks = [
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": owner.name},
                                        "finish_reason": None,
                                    }
                                ]
                            },
                            {
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                                ]
                            },
                        ]
                        payload = b"".join(
                            b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks
                        ) + b"data: [DONE]\n\n"
                        content_type = "text/event-stream"
                    else:
                        payload = b'data: {"choices":[]}\n\n'
                        content_type = "text/event-stream"
                else:
                    payload = json.dumps(
                        completion(owner.name) if owner.valid else {"choices": []}
                    ).encode()
                    content_type = "application/json"
                try:
                    self.send_response(owner.status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format_string, *arguments):
                return

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def provider(provider_id, upstream, *, domain=None, domains=None):
    return RACE.ProviderSpec(
        provider_id=provider_id,
        failure_domain=domain or provider_id,
        base_url=upstream.base_url,
        model=f"model-{provider_id}",
        auth={"kind": "none"},
        headers={},
        enabled=True,
        hedging_authorized=True,
        failure_domains=tuple(domains or ()),
    )


def settings(
    providers,
    *,
    redundancy=2,
    timeout=2.0,
    max_request_bytes=1 << 20,
    max_response_bytes=1 << 20,
):
    return RACE.PoolSettings(
        virtual_model="ox-alpha",
        redundancy=redundancy,
        first_byte_timeout_seconds=timeout,
        idle_timeout_seconds=timeout,
        request_timeout_seconds=timeout,
        failure_cooldown_seconds=0.1,
        max_request_bytes=max_request_bytes,
        max_response_bytes=max_response_bytes,
        providers=tuple(providers),
    )


class ValidationTests(unittest.TestCase):
    def test_complete_nonstream_and_stream_responses(self):
        body = json.dumps(completion("one")).encode()
        self.assertEqual(RACE.validate_chat_response(body, False), (True, None))
        stream = (
            b'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        self.assertEqual(RACE.validate_chat_response(stream, True), (True, None))
        self.assertFalse(RACE.validate_chat_response(b'{"choices":[]}', False)[0])
        self.assertFalse(RACE.validate_chat_response(stream.replace(b"[DONE]", b""), True)[0])

    def test_malformed_tool_calls_and_sse_cannot_validate(self):
        malformed = completion("bad", content=None)
        malformed["choices"][0]["finish_reason"] = "tool_calls"
        malformed["choices"][0]["message"]["tool_calls"] = [
            {
                "id": "call-edit",
                "type": "function",
                "function": {"name": "edit", "arguments": "{"},
            }
        ]
        self.assertFalse(
            RACE.validate_chat_response(json.dumps(malformed).encode(), False)[0]
        )
        malformed_stream = (
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
            b'"id":"call-edit","type":"function",'
            b'"function":{"name":"edit","arguments":"{"}}]},"finish_reason":"tool_calls"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        self.assertFalse(RACE.validate_chat_response(malformed_stream, True)[0])

    def test_only_choice_zero_can_make_a_response_win(self):
        value = completion("mixed")
        valid = value["choices"][0]
        value["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": None,
            },
            {**valid, "index": 1},
        ]
        self.assertFalse(RACE.validate_chat_response(json.dumps(value).encode(), False)[0])
        missing_calls = completion("missing", content=None)
        missing_calls["choices"][0]["finish_reason"] = "tool_calls"
        self.assertFalse(
            RACE.validate_chat_response(json.dumps(missing_calls).encode(), False)[0]
        )

        later_only = completion("later-only")
        later_only["choices"][0]["index"] = 1
        self.assertFalse(
            RACE.validate_chat_response(json.dumps(later_only).encode(), False)[0]
        )

        stream = (
            b'data: {"choices":[{"index":1,"delta":{"content":"later"},'
            b'"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        self.assertFalse(RACE.validate_chat_response(stream, True)[0])

    def test_mixed_and_duplicate_choice_zero_sets_are_rejected(self):
        valid_zero = completion("zero")["choices"][0]
        valid_one = {**completion("one")["choices"][0], "index": 1}
        mixed = {"choices": [valid_zero, valid_one]}
        duplicate_zero = {"choices": [valid_zero, dict(valid_zero)]}
        self.assertFalse(
            RACE.validate_chat_response(json.dumps(mixed).encode(), False)[0]
        )
        self.assertFalse(
            RACE.validate_chat_response(json.dumps(duplicate_zero).encode(), False)[0]
        )

        mixed_event = {
            "choices": [
                {"index": 0, "delta": {"content": "zero"}, "finish_reason": "stop"},
                {"index": 1, "delta": {"content": "one"}, "finish_reason": "stop"},
            ]
        }
        duplicate_event = {
            "choices": [
                {"index": 0, "delta": {"content": "a"}, "finish_reason": None},
                {"index": 0, "delta": {"content": "b"}, "finish_reason": "stop"},
            ]
        }
        for event in (mixed_event, duplicate_event):
            with self.subTest(event=event):
                stream = (
                    b"data: "
                    + json.dumps(event).encode()
                    + b"\n\ndata: [DONE]\n\n"
                )
                self.assertFalse(RACE.validate_chat_response(stream, True)[0])

    def test_finish_reason_and_tool_call_contract_is_strict(self):
        for finish_reason in ("length", "content_filter", "network_error", "", None, True):
            with self.subTest(finish_reason=finish_reason):
                value = completion("terminal")
                value["choices"][0]["finish_reason"] = finish_reason
                self.assertFalse(
                    RACE.validate_chat_response(json.dumps(value).encode(), False)[0]
                )

        stop_with_calls = completion("stop-with-calls")
        stop_with_calls["choices"][0]["message"]["tool_calls"] = [tool_call()]
        self.assertFalse(
            RACE.validate_chat_response(json.dumps(stop_with_calls).encode(), False)[0]
        )

        valid_calls = completion("valid-calls", content=None)
        valid_calls["choices"][0]["finish_reason"] = "tool_calls"
        valid_calls["choices"][0]["message"]["tool_calls"] = [tool_call()]
        self.assertTrue(
            RACE.validate_chat_response(json.dumps(valid_calls).encode(), False)[0]
        )

        invalid_calls = [
            [],
            [{"type": "function", "function": {"name": "edit", "arguments": "{}"}}],
            [{"id": "call-edit", "function": {"name": "edit", "arguments": "{}"}}],
            [{"id": "call-edit", "type": "function", "function": {"name": "edit"}}],
        ]
        for calls in invalid_calls:
            with self.subTest(calls=calls):
                value = completion("invalid-calls", content=None)
                value["choices"][0]["finish_reason"] = "tool_calls"
                value["choices"][0]["message"]["tool_calls"] = calls
                self.assertFalse(
                    RACE.validate_chat_response(json.dumps(value).encode(), False)[0]
                )

    def test_boolean_fractional_and_missing_indices_are_rejected(self):
        for index in (True, 0.5, None):
            with self.subTest(kind="json-choice", index=index):
                value = completion("bad-choice-index")
                if index is None:
                    value["choices"][0].pop("index")
                else:
                    value["choices"][0]["index"] = index
                self.assertFalse(
                    RACE.validate_chat_response(json.dumps(value).encode(), False)[0]
                )

            with self.subTest(kind="sse-choice", index=index):
                choice = {
                    "delta": {"content": "ok"},
                    "finish_reason": "stop",
                }
                if index is not None:
                    choice["index"] = index
                stream = (
                    b"data: "
                    + json.dumps({"choices": [choice]}).encode()
                    + b"\n\ndata: [DONE]\n\n"
                )
                self.assertFalse(RACE.validate_chat_response(stream, True)[0])

        for index in (True, 0.5):
            with self.subTest(kind="json-tool", index=index):
                value = completion("bad-tool-index", content=None)
                value["choices"][0]["finish_reason"] = "tool_calls"
                call = tool_call()
                call["index"] = index
                value["choices"][0]["message"]["tool_calls"] = [call]
                self.assertFalse(
                    RACE.validate_chat_response(json.dumps(value).encode(), False)[0]
                )

            with self.subTest(kind="sse-tool", index=index):
                stream = (
                    b"data: "
                    + json.dumps(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": [{**tool_call(), "index": index}]},
                                    "finish_reason": "tool_calls",
                                }
                            ]
                        }
                    ).encode()
                    + b"\n\ndata: [DONE]\n\n"
                )
                self.assertFalse(RACE.validate_chat_response(stream, True)[0])

    def test_unknown_finish_reason_cannot_win(self):
        value = completion("vendor-error")
        value["choices"][0]["finish_reason"] = "network_error"
        self.assertFalse(RACE.validate_chat_response(json.dumps(value).encode(), False)[0])

    def test_config_rejects_secret_headers_and_unapproved_hedging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "pool.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "providers": [
                            {
                                "id": "bad",
                                "base_url": "https://example.invalid/v1",
                                "model": "ox",
                                "hedging_authorized": True,
                                "headers": {"Authorization": "secret"},
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(RACE.PoolConfigurationError, "secret headers"):
                RACE.load_pool_settings(path)
            value = json.loads(path.read_text())
            value["providers"][0]["headers"] = {"api-key": "literal-secret"}
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RACE.PoolConfigurationError, "secret headers"):
                RACE.load_pool_settings(path)
            value = json.loads(path.read_text())
            value["providers"][0].pop("headers")
            value["providers"][0]["hedging_authorized"] = False
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RACE.PoolConfigurationError, "authorize hedging"):
                RACE.load_pool_settings(path)

    def test_integer_configuration_rejects_booleans_and_fractions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pool.json"
            base = {
                "schema_version": 1,
                "redundancy": 1,
                "max_request_bytes": 1024,
                "max_response_bytes": 2048,
                "providers": [
                    {
                        "id": "provider-a",
                        "base_url": "https://example.invalid/v1",
                        "model": "ox",
                        "hedging_authorized": True,
                    }
                ],
            }
            for name, value in (
                ("schema_version", True),
                ("schema_version", 1.0),
                ("redundancy", True),
                ("redundancy", 1.5),
                ("max_request_bytes", True),
                ("max_request_bytes", 1024.5),
                ("max_response_bytes", True),
                ("max_response_bytes", 2048.5),
            ):
                with self.subTest(name=name, value=value):
                    current = dict(base)
                    current[name] = value
                    path.write_text(json.dumps(current), encoding="utf-8")
                    with self.assertRaises(RACE.PoolConfigurationError):
                        RACE.load_pool_settings(path)

    def test_nonfinite_numeric_configuration_and_timeouts_are_rejected(self):
        numeric_fields = (
            "schema_version",
            "redundancy",
            "first_byte_timeout_seconds",
            "idle_timeout_seconds",
            "request_timeout_seconds",
            "failure_cooldown_seconds",
            "max_request_bytes",
            "max_response_bytes",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pool.json"
            base = {
                "schema_version": 1,
                "redundancy": 1,
                "first_byte_timeout_seconds": 1,
                "idle_timeout_seconds": 1,
                "request_timeout_seconds": 1,
                "failure_cooldown_seconds": 0,
                "max_request_bytes": 1024,
                "max_response_bytes": 2048,
                "providers": [
                    {
                        "id": "provider-a",
                        "base_url": "https://example.invalid/v1",
                        "model": "ox",
                        "hedging_authorized": True,
                    }
                ],
            }
            for name in numeric_fields:
                for value in (float("nan"), float("inf"), float("-inf")):
                    with self.subTest(source="file", name=name, value=value):
                        current = dict(base)
                        current[name] = value
                        path.write_text(json.dumps(current), encoding="utf-8")
                        with self.assertRaises(RACE.PoolConfigurationError):
                            RACE.load_pool_settings(path)

        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        configured = settings([provider("finite", upstream)], redundancy=1)
        runtime_fields = numeric_fields[1:]
        for name in runtime_fields:
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(source="runtime", name=name, value=value):
                    with self.assertRaises(RACE.PoolConfigurationError):
                        RACE.ProviderRacePool(dataclasses.replace(configured, **{name: value}))

        pool = RACE.ProviderRacePool(configured)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(source="flush", value=value):
                with self.assertRaises(ValueError):
                    pool.flush_events(value)
            with self.subTest(source="idle", value=value):
                with self.assertRaises(ValueError):
                    pool.wait_for_idle(value)
            with self.subTest(source="close", value=value):
                with self.assertRaises(ValueError):
                    pool.close(value)

    def test_error_details_redact_bearer_jwt_and_known_values(self):
        body = json.dumps(
            {
                "error": {
                    "message": (
                        "Authorization: Bearer ordinary-secret-token-12345 "
                        "eyJabcdefghijk.abcdefghijkl.abcdefghijkl"
                    )
                }
            }
        ).encode()
        rendered = RACE.safe_upstream_error(body, ("Bearer ordinary-secret-token-12345",))
        self.assertNotIn("ordinary-secret", rendered)
        self.assertNotIn("eyJabcdefghijk", rendered)
        self.assertIn("[redacted]", rendered)

    def test_generic_authorization_values_are_redacted_before_assignments(self):
        marker = "unconfigured-bearer-secret-12345"
        rendered = RACE._redact_error_text(
            f"upstream rejected Authorization: Bearer {marker}; retry disabled"
        )
        self.assertNotIn(marker, rendered)
        self.assertNotIn("Bearer", rendered)
        self.assertIn("[redacted]", rendered)

    def test_json_credential_requires_private_permissions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "auth.json"
            path.write_text(json.dumps({"provider": {"key": "value"}}))
            path.chmod(0o600)
            spec = RACE.ProviderSpec(
                "test",
                "test",
                "https://example.invalid/v1",
                "ox",
                {"kind": "json", "path": str(path), "keys": ["provider", "key"]},
                {},
                True,
                True,
            )
            self.assertEqual(RACE.resolve_auth_header(spec), ("Authorization", "Bearer value"))
            path.chmod(0o644)
            with self.assertRaisesRegex(RACE.PoolConfigurationError, "permissions"):
                RACE.resolve_auth_header(spec)

    def test_credentials_reject_controls_and_transport_errors_are_redacted(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        spec = provider("credential", upstream)
        spec = dataclasses.replace(
            spec,
            auth={"kind": "env", "name": "OXALPHA_TEST_CREDENTIAL"},
        )
        previous = os.environ.get("OXALPHA_TEST_CREDENTIAL")
        try:
            os.environ["OXALPHA_TEST_CREDENTIAL"] = "secret\nmarker"
            with self.assertRaisesRegex(RACE.PoolConfigurationError, "invalid characters"):
                RACE.resolve_auth_header(spec)
            marker = "transport-secret-marker-12345"
            os.environ["OXALPHA_TEST_CREDENTIAL"] = marker
            pool = RACE.ProviderRacePool(settings([spec], redundancy=1))

            def leak(_connection, _method, _path, body=None, headers=None):
                raise ValueError(f"bad header {headers['Authorization']}")

            with mock.patch.object(RACE.http.client.HTTPConnection, "request", leak):
                result = pool._call_provider(
                    "request",
                    spec,
                    {"messages": [{"role": "user", "content": "work"}]},
                    threading.Event(),
                )
            self.assertNotIn(marker, result.error)
            self.assertIn("[redacted]", result.error)

            prefixed = dataclasses.replace(
                spec,
                auth={
                    "kind": "env",
                    "name": "OXALPHA_TEST_CREDENTIAL",
                    "prefix": "Token-",
                },
            )

            def leak_raw(_connection, _method, _path, body=None, headers=None):
                self.assertEqual(headers["Authorization"], "Token-" + marker)
                raise ValueError(f"raw credential echoed: {marker}")

            with mock.patch.object(RACE.http.client.HTTPConnection, "request", leak_raw):
                result = pool._call_provider(
                    "request-prefixed",
                    prefixed,
                    {"messages": [{"role": "user", "content": "work"}]},
                    threading.Event(),
                )
            self.assertNotIn(marker, result.error)

            def leak_runtime(_connection, _method, _path, body=None, headers=None):
                raise RuntimeError(f"runtime transport echoed {marker}")

            with mock.patch.object(RACE.http.client.HTTPConnection, "request", leak_runtime):
                result = pool._call_provider(
                    "request-runtime",
                    prefixed,
                    {"messages": [{"role": "user", "content": "work"}]},
                    threading.Event(),
                )
            self.assertNotIn(marker, result.error)
            self.assertIn("[redacted]", result.error)
            upstream_error = json.dumps({"error": {"message": f"echo {marker}"}}).encode()
            rendered = RACE.safe_upstream_error(upstream_error, ("Token-" + marker, marker))
            self.assertNotIn(marker, rendered)
        finally:
            if previous is None:
                os.environ.pop("OXALPHA_TEST_CREDENTIAL", None)
            else:
                os.environ["OXALPHA_TEST_CREDENTIAL"] = previous

    def test_dotenv_loads_only_declared_provider_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
            settings_value = settings([provider("aiml", upstream)], redundancy=1)
            declared = dataclasses.replace(
                settings_value,
                providers=(
                    dataclasses.replace(
                        settings_value.providers[0],
                        auth={"kind": "env", "name": "AIMLAPI_KEY"},
                    ),
                ),
            )
            path = root / ".env"
            path.write_text("AIMLAPI_KEY='test-value'\nUNRELATED_SECRET=ignored\n")
            path.chmod(0o600)
            previous = os.environ.pop("AIMLAPI_KEY", None)
            os.environ.pop("UNRELATED_SECRET", None)
            try:
                self.assertEqual(RACE.load_provider_env_file(declared, path), ["AIMLAPI_KEY"])
                self.assertEqual(os.environ["AIMLAPI_KEY"], "test-value")
                self.assertNotIn("UNRELATED_SECRET", os.environ)
            finally:
                os.environ.pop("AIMLAPI_KEY", None)
                if previous is not None:
                    os.environ["AIMLAPI_KEY"] = previous
            path.chmod(0o644)
            with self.assertRaisesRegex(RACE.PoolConfigurationError, "permissions"):
                RACE.load_provider_env_file(declared, path)


class RacingTests(unittest.TestCase):
    def setUp(self):
        self.upstreams = []

    def tearDown(self):
        for upstream in self.upstreams:
            upstream.close()

    def upstream(self, *arguments, **keywords):
        value = Upstream(*arguments, **keywords)
        self.upstreams.append(value)
        return value

    def test_invalid_fast_response_refills_and_first_valid_wins(self):
        invalid = self.upstream("invalid", delay=0.01, valid=False)
        slow = self.upstream("slow", delay=0.3)
        refill = self.upstream("refill", delay=0.02)
        pool = RACE.ProviderRacePool(
            settings(
                [
                    provider("a-invalid", invalid),
                    provider("b-slow", slow),
                    provider("c-refill", refill),
                ]
            )
        )
        request_id, result = pool.race(
            {"model": "ox-alpha", "messages": [{"role": "user", "content": "work"}]},
            "pair-1",
        )
        self.assertRegex(request_id, r"^[0-9a-f]{16}$")
        self.assertEqual(result.provider_id, "c-refill")
        self.assertEqual(invalid.calls[0]["model"], "model-a-invalid")
        self.assertEqual(refill.calls[0]["messages"][0]["content"], "work")
        time.sleep(0.1)
        snapshot = pool.snapshot()
        health = {item["id"]: item for item in snapshot["providers"]}
        self.assertEqual(health["a-invalid"]["failures"], 1)
        self.assertEqual(health["c-refill"]["wins"], 1)

    def test_malformed_fast_tool_response_does_not_cancel_valid_leg(self):
        malformed = completion("bad", content=None)
        malformed["choices"][0]["finish_reason"] = "tool_calls"
        malformed["choices"][0]["message"]["tool_calls"] = [
            {"type": "function", "function": {"name": "edit", "arguments": "{"}}
        ]
        bad = self.upstream("bad", delay=0.01, payload=json.dumps(malformed).encode())
        good = self.upstream("good", delay=0.05)
        pool = RACE.ProviderRacePool(
            settings([provider("bad", bad), provider("good", good)])
        )
        _, result = pool.race(
            {"messages": [{"role": "user", "content": "work"}]}, "pair-malformed"
        )
        self.assertEqual(result.provider_id, "good")
        self.assertEqual(pool.snapshot()["providers"][0]["failures"], 1)

    def test_redundancy_uses_distinct_failure_domains(self):
        one = self.upstream("one", delay=0.2)
        same_domain = self.upstream("same", delay=0.01)
        other = self.upstream("other", delay=0.02)
        pool = RACE.ProviderRacePool(
            settings(
                [
                    provider("a-one", one, domain="gateway-a"),
                    provider("b-same", same_domain, domain="gateway-a"),
                    provider("c-other", other, domain="gateway-c"),
                ]
            )
        )
        _, result = pool.race(
            {"messages": [{"role": "user", "content": "work"}]}, "pair-2"
        )
        self.assertEqual(result.provider_id, "c-other")
        self.assertEqual(len(same_domain.calls), 0)

    def test_launch_selection_matches_reported_maximum_disjoint_width(self):
        broad = self.upstream("broad", delay=0.2)
        left = self.upstream("left", delay=0.02)
        right = self.upstream("right", delay=0.03)
        pool = RACE.ProviderRacePool(
            settings(
                [
                    provider("a-broad", broad, domains=("x", "y")),
                    provider("b-left", left, domains=("x",)),
                    provider("c-right", right, domains=("y",)),
                ]
            )
        )
        self.assertEqual(pool.snapshot()["effective_redundancy"], 2)
        pool.race({"messages": [{"role": "user", "content": "work"}]}, "domain-trap")
        self.assertEqual(len(broad.calls), 0)
        self.assertEqual(len(left.calls), 1)
        self.assertEqual(len(right.calls), 1)

    def test_correlated_gateway_is_deferred_then_used_as_sequential_fallback(self):
        primary = self.upstream("primary", delay=0.01, valid=False)
        correlated = self.upstream("correlated", delay=0.01)
        independent = self.upstream("independent", delay=0.3)
        pool = RACE.ProviderRacePool(
            settings(
                [
                    provider(
                        "a-primary",
                        primary,
                        domains=("gateway-primary", "upstream-shared"),
                    ),
                    provider(
                        "b-correlated",
                        correlated,
                        domains=("gateway-correlated", "upstream-shared"),
                    ),
                    provider(
                        "c-independent",
                        independent,
                        domains=("gateway-independent",),
                    ),
                ]
            )
        )
        _, result = pool.race(
            {"messages": [{"role": "user", "content": "work"}]}, "pair-correlated"
        )
        self.assertEqual(result.provider_id, "b-correlated")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(correlated.calls), 1)
        self.assertEqual(len(independent.calls), 1)

    def test_oversized_response_cannot_win(self):
        oversized = self.upstream("x" * 1000)
        pool = RACE.ProviderRacePool(
            settings([provider("oversized", oversized)], redundancy=1, max_response_bytes=100)
        )
        with self.assertRaisesRegex(RACE.RaceFailure, "size limit"):
            pool.race({"messages": [{"role": "user", "content": "work"}]}, "pair-size")
        health = pool.snapshot()["providers"][0]
        self.assertEqual(health["failures"], 1)

    def test_direct_request_limit_is_enforced_before_network(self):
        upstream = self.upstream("unused")
        pool = RACE.ProviderRacePool(
            settings(
                [provider("limited", upstream)],
                redundancy=1,
                max_request_bytes=64,
            )
        )
        with self.assertRaisesRegex(RACE.RaceFailure, "request exceeds"):
            pool.race(
                {"messages": [{"role": "user", "content": "x" * 4096}]},
                "pair-request-size",
            )
        self.assertEqual(upstream.calls, [])

    def test_stream_line_is_bounded_before_newline(self):
        oversized = self.upstream("x" * 1000, stream=True)
        pool = RACE.ProviderRacePool(
            settings(
                [provider("oversized-stream", oversized)],
                redundancy=1,
                max_response_bytes=100,
            )
        )
        with self.assertRaisesRegex(RACE.RaceFailure, "size limit"):
            pool.race(
                {"stream": True, "messages": [{"role": "user", "content": "work"}]},
                "pair-stream-size",
            )

    def test_provider_cooldown_is_not_bypassed(self):
        failing = self.upstream("failing", valid=False)
        configured = dataclasses.replace(
            settings([provider("failing", failing)], redundancy=1),
            failure_cooldown_seconds=60.0,
        )
        pool = RACE.ProviderRacePool(configured)
        with self.assertRaises(RACE.RaceFailure):
            pool.race({"messages": [{"role": "user", "content": "one"}]}, "cooling")
        with self.assertRaisesRegex(RACE.RaceFailure, "cooling down"):
            pool.race({"messages": [{"role": "user", "content": "two"}]}, "cooling")
        self.assertEqual(len(failing.calls), 1)
        self.assertEqual(pool.snapshot()["effective_redundancy"], 0)

    def test_stale_candidate_is_rechecked_before_launch(self):
        upstream = self.upstream("must-not-run")
        spec = provider("cooling", upstream)
        pool = RACE.ProviderRacePool(settings([spec], redundancy=1))
        with pool._lock:
            pool._health[spec.provider_id].circuit_open_until = time.time() + 60.0
        pool._candidate_order = lambda _context_key: [spec]
        with self.assertRaises(RACE.RaceFailure):
            pool.race(
                {"messages": [{"role": "user", "content": "stale"}]},
                "stale-candidate",
            )
        self.assertEqual(upstream.calls, [])
        health = pool.snapshot()["providers"][0]
        self.assertEqual(health["calls"], 0)
        self.assertEqual(health["in_flight"], 0)

    def test_worker_target_rechecks_deadline_before_provider_call(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        spec = provider("delayed-worker", upstream)
        configured = dataclasses.replace(
            settings([spec], redundancy=1, timeout=0.05),
            failure_cooldown_seconds=60.0,
        )
        pool = RACE.ProviderRacePool(configured)
        calls = []
        original_thread = RACE.threading.Thread

        def delayed_thread(*arguments, **keywords):
            target = keywords.get("target")
            name = keywords.get("name", "")
            if target is not None and name.startswith("oxalpha-"):
                def delayed_target():
                    time.sleep(0.07)
                    target()

                keywords["target"] = delayed_target
            return original_thread(*arguments, **keywords)

        def should_not_call(_request_id, current, _body, _cancel_event):
            calls.append(current.provider_id)
            return RACE.UpstreamResult(
                current.provider_id,
                True,
                200,
                "application/json",
                json.dumps(completion("unexpected")).encode(),
                0.01,
                0.01,
            )

        pool._call_provider = should_not_call
        started = time.monotonic()
        with mock.patch.object(RACE.threading, "Thread", side_effect=delayed_thread):
            with self.assertRaisesRegex(RACE.RaceFailure, "deadline"):
                pool.race(
                    {"messages": [{"role": "user", "content": "work"}]},
                    "worker-target-deadline",
                )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.2)
        self.assertTrue(pool.wait_for_idle(1.0))
        snapshot = pool.snapshot()
        health = snapshot["providers"][0]
        self.assertEqual(calls, [])
        self.assertEqual(snapshot["requests_started"], 1)
        self.assertEqual(snapshot["requests_failed"], 1)
        self.assertEqual(snapshot["requests_won"], 0)
        self.assertEqual(snapshot["active_provider_workers"], 0)
        self.assertEqual(health["calls"], 1)
        self.assertEqual(health["wins"], 0)
        self.assertEqual(health["failures"], 1)
        self.assertEqual(health["cancellations"], 0)
        self.assertEqual(health["in_flight"], 0)
        self.assertEqual(health["last_error"], "request deadline exceeded")
        self.assertTrue(health["circuit_open"])

    def test_deadline_opens_cooldown_before_async_worker_settles(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        spec = provider("stuck", upstream)
        configured = dataclasses.replace(
            settings([spec], redundancy=1, timeout=0.05),
            failure_cooldown_seconds=60.0,
        )
        pool = RACE.ProviderRacePool(configured)
        calls = []

        def delayed(_request_id, current, _body, cancel_event):
            calls.append(current.provider_id)
            time.sleep(0.2)
            return RACE.UpstreamResult(
                current.provider_id,
                False,
                None,
                "application/json",
                b"",
                0.2,
                None,
                error="canceled",
                canceled=cancel_event.is_set(),
            )

        pool._call_provider = delayed
        with self.assertRaises(RACE.RaceFailure):
            pool.race({"messages": [{"role": "user", "content": "one"}]}, "deadline")
        with self.assertRaisesRegex(RACE.RaceFailure, "cooling down"):
            pool.race({"messages": [{"role": "user", "content": "two"}]}, "deadline")
        self.assertEqual(calls, ["stuck"])
        self.assertTrue(pool.wait_for_idle(1.0))

    def test_worker_exception_settles_as_one_failure(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        spec = provider("crash", upstream)
        pool = RACE.ProviderRacePool(settings([spec], redundancy=1, timeout=0.2))

        def crash(_request_id, _current, _body, _cancel_event):
            raise RuntimeError("worker exploded")

        pool._call_provider = crash
        with self.assertRaisesRegex(RACE.RaceFailure, "worker exception.*worker exploded"):
            pool.race(
                {"messages": [{"role": "user", "content": "work"}]},
                "worker-exception",
            )
        self.assertTrue(pool.wait_for_idle(1.0))
        snapshot = pool.snapshot()
        health = snapshot["providers"][0]
        self.assertEqual(snapshot["requests_started"], 1)
        self.assertEqual(snapshot["requests_failed"], 1)
        self.assertEqual(snapshot["requests_won"], 0)
        self.assertEqual(health["calls"], 1)
        self.assertEqual(health["failures"], 1)
        self.assertEqual(health["cancellations"], 0)
        self.assertEqual(health["in_flight"], 0)

    def test_loser_exception_after_cancellation_settles_once(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        fast = provider("a-fast", upstream, domain="fast")
        crashing = provider("b-crashing", upstream, domain="crashing")
        slow_started = threading.Event()
        pool = RACE.ProviderRacePool(settings([fast, crashing], timeout=0.5))

        def scripted(_request_id, current, _body, cancel_event):
            if current.provider_id == "a-fast":
                slow_started.wait(1.0)
                return RACE.UpstreamResult(
                    current.provider_id,
                    True,
                    200,
                    "application/json",
                    json.dumps(completion("fast")).encode(),
                    0.01,
                    0.01,
                )
            slow_started.set()
            cancel_event.wait(1.0)
            raise RuntimeError("loser cleanup exploded")

        pool._call_provider = scripted
        _, result = pool.race(
            {"messages": [{"role": "user", "content": "work"}]},
            "canceled-exception",
        )
        self.assertEqual(result.provider_id, "a-fast")
        self.assertTrue(pool.wait_for_idle(1.0))
        snapshot = pool.snapshot()
        health = {item["id"]: item for item in snapshot["providers"]}
        self.assertEqual(snapshot["requests_won"], 1)
        self.assertEqual(snapshot["requests_failed"], 0)
        self.assertEqual(health["b-crashing"]["calls"], 1)
        self.assertEqual(health["b-crashing"]["cancellations"], 1)
        self.assertEqual(health["b-crashing"]["failures"], 0)
        self.assertEqual(health["b-crashing"]["in_flight"], 0)

    def test_valid_result_after_deadline_settles_as_one_deadline_failure(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        spec = provider("late-valid", upstream)
        configured = dataclasses.replace(
            settings([spec], redundancy=1, timeout=0.03),
            failure_cooldown_seconds=60.0,
        )
        pool = RACE.ProviderRacePool(configured)

        def late_valid(_request_id, current, _body, _cancel_event):
            time.sleep(0.1)
            return RACE.UpstreamResult(
                current.provider_id,
                True,
                200,
                "application/json",
                json.dumps(completion("late")).encode(),
                0.1,
                0.1,
            )

        pool._call_provider = late_valid
        with self.assertRaisesRegex(RACE.RaceFailure, "deadline"):
            pool.race(
                {"messages": [{"role": "user", "content": "work"}]},
                "late-valid",
            )
        self.assertTrue(pool.wait_for_idle(1.0))
        snapshot = pool.snapshot()
        health = snapshot["providers"][0]
        self.assertEqual(snapshot["requests_failed"], 1)
        self.assertEqual(snapshot["requests_won"], 0)
        self.assertEqual(health["calls"], 1)
        self.assertEqual(health["wins"], 0)
        self.assertEqual(health["failures"], 1)
        self.assertEqual(health["cancellations"], 0)
        self.assertEqual(health["in_flight"], 0)
        self.assertEqual(health["last_error"], "request deadline exceeded")
        self.assertTrue(health["circuit_open"])

    def test_completion_exactly_at_deadline_is_expired(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        spec = provider("equal-deadline", upstream)
        configured = dataclasses.replace(
            settings([spec], redundancy=1, timeout=0.05),
            failure_cooldown_seconds=60.0,
        )
        clock_lock = threading.Lock()
        current_time = [100.0]
        deadline_time = current_time[0] + configured.request_timeout_seconds
        allow_completion = threading.Event()
        original_get = RACE.queue.Queue.get

        def controlled_monotonic():
            with clock_lock:
                return current_time[0]

        pool = RACE.ProviderRacePool(configured, monotonic=controlled_monotonic)

        def complete_at_deadline(_request_id, current, _body, _cancel_event):
            self.assertTrue(allow_completion.wait(1.0))
            with clock_lock:
                current_time[0] = deadline_time
            return RACE.UpstreamResult(
                current.provider_id,
                True,
                200,
                "application/json",
                json.dumps(completion("equal-deadline")).encode(),
                0.05,
                0.05,
            )

        def release_on_queue_wait(current, block=True, timeout=None):
            if current.maxsize == 0 and timeout is not None:
                allow_completion.set()
            return original_get(current, block=block, timeout=timeout)

        pool._call_provider = complete_at_deadline
        with mock.patch.object(RACE.queue.Queue, "get", release_on_queue_wait):
            with self.assertRaisesRegex(RACE.RaceFailure, "deadline"):
                pool.race(
                    {"messages": [{"role": "user", "content": "work"}]},
                    "equal-deadline",
                )
        self.assertTrue(pool.wait_for_idle(1.0))
        snapshot = pool.snapshot()
        health = snapshot["providers"][0]
        self.assertEqual(snapshot["requests_started"], 1)
        self.assertEqual(snapshot["requests_failed"], 1)
        self.assertEqual(snapshot["requests_won"], 0)
        self.assertEqual(health["calls"], 1)
        self.assertEqual(health["wins"], 0)
        self.assertEqual(health["failures"], 1)
        self.assertEqual(health["cancellations"], 0)
        self.assertEqual(health["in_flight"], 0)
        self.assertEqual(health["last_error"], "request deadline exceeded")
        self.assertTrue(health["circuit_open"])

    def test_delayed_queue_wake_cannot_accept_post_deadline_completion(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        spec = provider("late-queue", upstream)
        configured = dataclasses.replace(
            settings([spec], redundancy=1, timeout=0.05),
            failure_cooldown_seconds=60.0,
        )
        pool = RACE.ProviderRacePool(configured)
        original_get = RACE.queue.Queue.get

        def complete_at_sixty_ms(_request_id, current, _body, _cancel_event):
            time.sleep(0.06)
            return RACE.UpstreamResult(
                current.provider_id,
                True,
                200,
                "application/json",
                json.dumps(completion("late-queue")).encode(),
                0.06,
                0.06,
            )

        def delayed_queue_get(current, block=True, timeout=None):
            if current.maxsize == 0 and timeout is not None:
                time.sleep(0.08)
                return original_get(current, block=False)
            return original_get(current, block=block, timeout=timeout)

        pool._call_provider = complete_at_sixty_ms
        started = time.monotonic()
        with mock.patch.object(RACE.queue.Queue, "get", delayed_queue_get):
            with self.assertRaisesRegex(RACE.RaceFailure, "deadline"):
                pool.race(
                    {"messages": [{"role": "user", "content": "work"}]},
                    "post-deadline-queue-wake",
                )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.25)
        self.assertTrue(pool.wait_for_idle(1.0))
        snapshot = pool.snapshot()
        health = snapshot["providers"][0]
        self.assertEqual(snapshot["requests_started"], 1)
        self.assertEqual(snapshot["requests_failed"], 1)
        self.assertEqual(snapshot["requests_won"], 0)
        self.assertEqual(health["calls"], 1)
        self.assertEqual(health["wins"], 0)
        self.assertEqual(health["failures"], 1)
        self.assertEqual(health["in_flight"], 0)
        self.assertEqual(health["last_error"], "request deadline exceeded")
        self.assertTrue(health["circuit_open"])

    def test_late_invalid_completion_cannot_launch_fallback(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        primary = provider("a-invalid-late", upstream, domain="invalid-late")
        fallback = provider("b-fallback", upstream, domain="fallback")
        configured = dataclasses.replace(
            settings([primary, fallback], redundancy=1, timeout=0.05),
            failure_cooldown_seconds=60.0,
        )
        pool = RACE.ProviderRacePool(configured)
        original_get = RACE.queue.Queue.get
        calls = []

        def scripted(_request_id, current, _body, _cancel_event):
            calls.append(current.provider_id)
            if current.provider_id == "a-invalid-late":
                time.sleep(0.06)
                return RACE.UpstreamResult(
                    current.provider_id,
                    False,
                    200,
                    "application/json",
                    b'{"choices":[]}',
                    0.06,
                    0.06,
                    error="structurally invalid response",
                )
            return RACE.UpstreamResult(
                current.provider_id,
                True,
                200,
                "application/json",
                json.dumps(completion("fallback")).encode(),
                0.01,
                0.01,
            )

        def delayed_queue_get(current, block=True, timeout=None):
            if current.maxsize == 0 and timeout is not None:
                time.sleep(0.09)
                return original_get(current, block=False)
            return original_get(current, block=block, timeout=timeout)

        pool._call_provider = scripted
        started = time.monotonic()
        with mock.patch.object(RACE.queue.Queue, "get", delayed_queue_get):
            with self.assertRaisesRegex(RACE.RaceFailure, "structurally invalid"):
                pool.race(
                    {"messages": [{"role": "user", "content": "work"}]},
                    "late-invalid-no-refill",
                )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.25)
        self.assertTrue(pool.wait_for_idle(1.0))
        snapshot = pool.snapshot()
        health = {item["id"]: item for item in snapshot["providers"]}
        self.assertEqual(calls, ["a-invalid-late"])
        self.assertEqual(snapshot["requests_started"], 1)
        self.assertEqual(snapshot["requests_failed"], 1)
        self.assertEqual(snapshot["requests_won"], 0)
        self.assertEqual(health["a-invalid-late"]["calls"], 1)
        self.assertEqual(health["a-invalid-late"]["wins"], 0)
        self.assertEqual(health["a-invalid-late"]["failures"], 1)
        self.assertEqual(health["a-invalid-late"]["in_flight"], 0)
        self.assertTrue(health["a-invalid-late"]["circuit_open"])
        self.assertEqual(health["b-fallback"]["calls"], 0)
        self.assertEqual(health["b-fallback"]["wins"], 0)
        self.assertEqual(health["b-fallback"]["failures"], 0)
        self.assertEqual(health["b-fallback"]["in_flight"], 0)

    def test_in_flight_remains_truthful_until_loser_worker_settles(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        fast = provider("a-fast", upstream, domain="fast")
        slow = provider("b-slow", upstream, domain="slow")
        pool = RACE.ProviderRacePool(settings([fast, slow], timeout=0.05))
        release = threading.Event()

        def scripted(_request_id, current, _body, cancel_event):
            if current.provider_id == "a-fast":
                return RACE.UpstreamResult(
                    current.provider_id,
                    True,
                    200,
                    "application/json",
                    json.dumps(completion("fast")).encode(),
                    0.01,
                    0.01,
                )
            release.wait(1.0)
            return RACE.UpstreamResult(
                current.provider_id,
                False,
                None,
                "application/json",
                b"",
                0.2,
                None,
                error="canceled",
                canceled=cancel_event.is_set(),
            )

        pool._call_provider = scripted
        pool.race({"messages": [{"role": "user", "content": "work"}]}, "truthful")
        health = {item["id"]: item for item in pool.snapshot()["providers"]}
        self.assertEqual(health["b-slow"]["in_flight"], 1)
        release.set()
        self.assertTrue(pool.wait_for_idle(1.0))
        health = {item["id"]: item for item in pool.snapshot()["providers"]}
        self.assertEqual(health["b-slow"]["in_flight"], 0)
        self.assertEqual(health["b-slow"]["cancellations"], 1)
        self.assertEqual(health["b-slow"]["failures"], 0)

    def test_result_consumption_waits_for_worker_deregistration_stress(self):
        original_put = RACE.queue.Queue.put
        for iteration in range(25):
            with self.subTest(iteration=iteration):
                upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
                fast = provider("a-fast-publish", upstream, domain="fast-publish")
                slow = provider("b-slow-publish", upstream, domain="slow-publish")
                pool = RACE.ProviderRacePool(settings([fast, slow], timeout=0.5))
                slow_started = threading.Event()
                release_slow = threading.Event()
                published = threading.Event()
                release_publication = threading.Event()
                race_done = threading.Event()
                outcomes = []
                errors = []

                def scripted(_request_id, current, _body, cancel_event):
                    if current.provider_id == "a-fast-publish":
                        self.assertTrue(slow_started.wait(1.0))
                        return RACE.UpstreamResult(
                            current.provider_id,
                            True,
                            200,
                            "application/json",
                            json.dumps(completion("fast-publish")).encode(),
                            0.01,
                            0.01,
                        )
                    slow_started.set()
                    self.assertTrue(release_slow.wait(1.0))
                    return RACE.UpstreamResult(
                        current.provider_id,
                        False,
                        None,
                        "application/json",
                        b"",
                        0.2,
                        None,
                        error="canceled",
                        canceled=cancel_event.is_set(),
                    )

                def pause_after_publication(current, item, block=True, timeout=None):
                    value = original_put(current, item, block=block, timeout=timeout)
                    if (
                        current.maxsize == 0
                        and isinstance(item, RACE.UpstreamResult)
                        and item.provider_id == "a-fast-publish"
                    ):
                        published.set()
                        release_publication.wait(1.0)
                    return value

                def run_race():
                    try:
                        outcomes.append(
                            pool.race(
                                {"messages": [{"role": "user", "content": "work"}]},
                                f"publication-order-{iteration}",
                            )
                        )
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        race_done.set()

                pool._call_provider = scripted
                coordinator = threading.Thread(target=run_race, daemon=True)
                with mock.patch.object(RACE.queue.Queue, "put", pause_after_publication):
                    coordinator.start()
                    try:
                        self.assertTrue(published.wait(1.0))
                        self.assertFalse(race_done.wait(0.02))
                        release_publication.set()
                        self.assertTrue(race_done.wait(1.0))
                        coordinator.join(timeout=1.0)
                        self.assertFalse(coordinator.is_alive())
                        self.assertEqual(errors, [])
                        self.assertEqual(len(outcomes), 1)
                        self.assertEqual(outcomes[0][1].provider_id, "a-fast-publish")
                        snapshot = pool.snapshot()
                        health = {item["id"]: item for item in snapshot["providers"]}
                        self.assertEqual(snapshot["active_provider_workers"], 1)
                        self.assertEqual(health["a-fast-publish"]["in_flight"], 0)
                        self.assertEqual(health["b-slow-publish"]["in_flight"], 1)
                    finally:
                        release_publication.set()
                        release_slow.set()
                        coordinator.join(timeout=1.0)
                self.assertTrue(pool.wait_for_idle(1.0))
                snapshot = pool.snapshot()
                health = {item["id"]: item for item in snapshot["providers"]}
                self.assertEqual(snapshot["requests_started"], 1)
                self.assertEqual(snapshot["requests_won"], 1)
                self.assertEqual(snapshot["requests_failed"], 0)
                self.assertEqual(snapshot["active_provider_workers"], 0)
                self.assertEqual(health["a-fast-publish"]["wins"], 1)
                self.assertEqual(health["a-fast-publish"]["in_flight"], 0)
                self.assertEqual(health["b-slow-publish"]["cancellations"], 1)
                self.assertEqual(health["b-slow-publish"]["failures"], 0)
                self.assertEqual(health["b-slow-publish"]["in_flight"], 0)

    def test_close_tracks_noncooperative_loser_and_rejects_new_races(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        fast = provider("a-fast-close", upstream, domain="fast-close")
        stuck = provider("b-stuck-close", upstream, domain="stuck-close")
        stuck_started = threading.Event()
        release_stuck = threading.Event()
        pool = RACE.ProviderRacePool(settings([fast, stuck], timeout=1.0))

        def scripted(_request_id, current, _body, cancel_event):
            if current.provider_id == "a-fast-close":
                stuck_started.wait(1.0)
                return RACE.UpstreamResult(
                    current.provider_id,
                    True,
                    200,
                    "application/json",
                    json.dumps(completion("fast-close")).encode(),
                    0.01,
                    0.01,
                )
            stuck_started.set()
            release_stuck.wait(2.0)
            return RACE.UpstreamResult(
                current.provider_id,
                False,
                None,
                "application/json",
                b"",
                0.2,
                None,
                error="canceled",
                canceled=cancel_event.is_set(),
            )

        pool._call_provider = scripted
        _, result = pool.race(
            {"messages": [{"role": "user", "content": "work"}]},
            "close-worker-tracking",
        )
        self.assertEqual(result.provider_id, "a-fast-close")
        snapshot = pool.snapshot()
        self.assertEqual(snapshot["active_races"], 0)
        self.assertEqual(snapshot["active_provider_workers"], 1)
        self.assertEqual(
            {item["id"]: item for item in snapshot["providers"]}["b-stuck-close"]["in_flight"],
            1,
        )

        started = time.monotonic()
        self.assertFalse(pool.close(0.05))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.2)
        snapshot = pool.snapshot()
        self.assertTrue(snapshot["closing"])
        self.assertEqual(snapshot["active_provider_workers"], 1)
        with self.assertRaisesRegex(RACE.RaceFailure, "pool is closing"):
            pool.race(
                {"messages": [{"role": "user", "content": "new"}]},
                "rejected-after-close",
            )

        release_stuck.set()
        self.assertTrue(pool.close(1.0))
        snapshot = pool.snapshot()
        self.assertEqual(snapshot["active_races"], 0)
        self.assertEqual(snapshot["active_provider_workers"], 0)
        self.assertEqual(
            {item["id"]: item for item in snapshot["providers"]}["b-stuck-close"]["in_flight"],
            0,
        )

    def test_dashboard_callback_failure_cannot_abort_inference(self):
        upstream = self.upstream("healthy")

        def broken_callback(_event, _payload):
            raise RuntimeError("dashboard storage unavailable")

        pool = RACE.ProviderRacePool(
            settings([provider("healthy", upstream)], redundancy=1),
            event_callback=broken_callback,
        )
        _, result = pool.race(
            {"messages": [{"role": "user", "content": "work"}]}, "callback"
        )
        self.assertEqual(result.provider_id, "healthy")
        self.assertTrue(pool.flush_events(1.0))
        snapshot = pool.snapshot()
        self.assertGreater(snapshot["event_callback_failures"], 0)
        self.assertEqual(snapshot["last_event_callback_error"], "event callback failed")
        self.assertNotIn("dashboard storage unavailable", snapshot["last_event_callback_error"])
        pool.close()

    def test_callback_exception_cannot_expose_any_provider_credential(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        names = ("OXALPHA_CALLBACK_SECRET_A", "OXALPHA_CALLBACK_SECRET_B")
        markers = ("opaque-callback-marker-a", "opaque-callback-marker-b")
        previous = {name: os.environ.get(name) for name in names}
        providers = []
        pool = None
        try:
            for index, (name, marker) in enumerate(zip(names, markers), start=1):
                os.environ[name] = marker
                providers.append(
                    dataclasses.replace(
                        provider(f"callback-{index}", upstream),
                        auth={"kind": "env", "name": name, "prefix": "Opaque "},
                    )
                )

            def leaking_callback(_event, _payload):
                raise RuntimeError(f"callback leaked {markers[0]} and {markers[1]}")

            pool = RACE.ProviderRacePool(
                settings(providers, redundancy=2),
                event_callback=leaking_callback,
            )
            pool._emit("credential-probe", {})
            self.assertTrue(pool.flush_events(1.0))
            snapshot = pool.snapshot()
            rendered = json.dumps(snapshot)
            self.assertEqual(snapshot["last_event_callback_error"], "event callback failed")
            self.assertNotIn("callback leaked", rendered)
            for marker in markers:
                self.assertNotIn(marker, rendered)
        finally:
            if pool is not None:
                pool.close(1.0)
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_slow_dashboard_callback_is_not_on_inference_path(self):
        upstream = self.upstream("healthy")

        def slow_callback(_event, _payload):
            time.sleep(0.2)

        pool = RACE.ProviderRacePool(
            settings([provider("healthy", upstream)], redundancy=1, timeout=0.1),
            event_callback=slow_callback,
        )
        started = time.monotonic()
        _, result = pool.race(
            {"messages": [{"role": "user", "content": "work"}]}, "slow-callback"
        )
        elapsed = time.monotonic() - started
        self.assertEqual(result.provider_id, "healthy")
        self.assertLess(elapsed, 0.15)
        self.assertTrue(pool.flush_events(2.0))
        pool.close()

    def test_callback_dispatch_starts_after_initial_provider_launches(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        first = provider("a-first", upstream, domain="first")
        second = provider("b-second", upstream, domain="second")
        callback_entered = threading.Event()
        release_callback = threading.Event()
        release_workers = threading.Event()
        both_workers_started = threading.Event()
        started = set()
        started_lock = threading.Lock()
        pool_holder = {}
        outcome = {}

        def blocked_callback(_event, _payload):
            pool = pool_holder["pool"]
            with pool._lock:
                callback_entered.set()
                release_callback.wait(1.0)

        pool = RACE.ProviderRacePool(
            settings([first, second], timeout=0.5),
            event_callback=blocked_callback,
        )
        pool_holder["pool"] = pool

        def scripted(_request_id, current, _body, cancel_event):
            with started_lock:
                started.add(current.provider_id)
                if len(started) == 2:
                    both_workers_started.set()
            release_workers.wait(1.0)
            valid = current.provider_id == "a-first"
            return RACE.UpstreamResult(
                current.provider_id,
                valid,
                200 if valid else None,
                "application/json",
                json.dumps(completion("first")).encode() if valid else b"",
                0.01,
                0.01 if valid else None,
                error=None if valid else "canceled",
                canceled=not valid and cancel_event.is_set(),
            )

        def run_race():
            try:
                outcome["result"] = pool.race(
                    {"messages": [{"role": "user", "content": "work"}]},
                    "callback-launch-order",
                )[1]
            except BaseException as error:
                outcome["error"] = error

        pool._call_provider = scripted
        thread = threading.Thread(target=run_race, daemon=True)
        thread.start()
        try:
            self.assertTrue(callback_entered.wait(1.0))
            self.assertTrue(both_workers_started.is_set())
            release_workers.set()
            release_callback.set()
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())
            self.assertNotIn("error", outcome)
            self.assertEqual(outcome["result"].provider_id, "a-first")
        finally:
            release_workers.set()
            release_callback.set()
            thread.join(timeout=1.0)
            self.assertTrue(pool.wait_for_idle(1.0))
            self.assertTrue(pool.close(1.0))

    def test_event_queue_is_bounded_and_nonblocking(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        entered = threading.Event()
        release = threading.Event()

        def blocked_callback(_event, _payload):
            entered.set()
            release.wait(1.0)

        pool = RACE.ProviderRacePool(
            settings([provider("bounded", upstream)], redundancy=1),
            event_callback=blocked_callback,
        )
        pool._emit("blocked", {})
        self.assertTrue(entered.wait(1.0))
        started = time.monotonic()
        for index in range(RACE.EVENT_QUEUE_LIMIT + 32):
            pool._emit("queued", {"index": index})
        elapsed = time.monotonic() - started
        snapshot = pool.snapshot()
        self.assertLess(elapsed, 1.0)
        self.assertLessEqual(snapshot["event_queue_depth"], RACE.EVENT_QUEUE_LIMIT)
        self.assertGreater(snapshot["event_callback_failures"], 0)
        release.set()
        self.assertTrue(pool.flush_events(2.0))
        self.assertTrue(pool.close(1.0))

    def test_close_timeout_retry_and_repeated_close_settle_once(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        entered = threading.Event()
        release = threading.Event()
        delivered = []

        def blocked_callback(event, _payload):
            entered.set()
            release.wait(1.0)
            delivered.append(event)

        pool = RACE.ProviderRacePool(
            settings([provider("closer", upstream)], redundancy=1),
            event_callback=blocked_callback,
        )
        pool._emit("one", {})
        self.assertTrue(entered.wait(1.0))
        self.assertFalse(pool.close(0.01))
        release.set()
        self.assertTrue(pool.close(1.0))
        self.assertTrue(pool.close(0.0))
        self.assertTrue(pool.flush_events(0.0))
        snapshot = pool.snapshot()
        self.assertEqual(delivered, ["one"])
        self.assertEqual(snapshot["events_enqueued"], 1)
        self.assertEqual(snapshot["events_delivered"], 1)
        self.assertEqual(snapshot["event_callback_lag_events"], 0)
        self.assertEqual(snapshot["event_queue_depth"], 0)

    def test_concurrent_close_lock_wait_uses_the_same_deadline(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        entered = threading.Event()
        release = threading.Event()
        first_result = {}

        def blocked_callback(_event, _payload):
            entered.set()
            release.wait(1.0)

        pool = RACE.ProviderRacePool(
            settings([provider("concurrent-close", upstream)], redundancy=1),
            event_callback=blocked_callback,
        )
        pool._emit("blocked", {})
        self.assertTrue(entered.wait(1.0))

        def first_close():
            first_result["value"] = pool.close(1.0)

        thread = threading.Thread(target=first_close, daemon=True)
        thread.start()
        lock_deadline = time.monotonic() + 1.0
        while not pool._event_close_lock.locked() and time.monotonic() < lock_deadline:
            time.sleep(0.001)
        self.assertTrue(pool._event_close_lock.locked())
        started = time.monotonic()
        self.assertFalse(pool.close(0.05))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.2)
        release.set()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(first_result["value"])
        self.assertTrue(pool.close(0.0))

    def test_callback_lag_is_visible_and_winner_precedes_settlement(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        fast = provider("a-fast", upstream, domain="fast")
        slow = provider("b-slow", upstream, domain="slow")
        entered = threading.Event()
        release = threading.Event()
        events = []

        def blocked_callback(event, payload):
            entered.set()
            release.wait(1.0)
            events.append((event, payload["event_sequence"]))

        pool = RACE.ProviderRacePool(
            settings([fast, slow], timeout=0.2),
            event_callback=blocked_callback,
        )

        def scripted(_request_id, current, _body, cancel_event):
            if current.provider_id == "a-fast":
                return RACE.UpstreamResult(
                    current.provider_id,
                    True,
                    200,
                    "application/json",
                    json.dumps(completion("fast")).encode(),
                    0.01,
                    0.01,
                )
            time.sleep(0.05)
            return RACE.UpstreamResult(
                current.provider_id,
                False,
                None,
                "application/json",
                b"",
                0.05,
                None,
                error="canceled",
                canceled=cancel_event.is_set(),
            )

        pool._call_provider = scripted
        pool.race({"messages": [{"role": "user", "content": "work"}]}, "event-order")
        self.assertTrue(entered.wait(1.0))
        snapshot = pool.snapshot()
        self.assertGreater(snapshot["event_callback_lag_events"], 0)
        self.assertIsNotNone(snapshot["last_event_enqueued_at"])
        release.set()
        self.assertTrue(pool.wait_for_idle(1.0))
        self.assertTrue(pool.flush_events(1.0))
        names = [event for event, _sequence in events]
        self.assertLess(names.index("race_won"), names.index("provider_settled"))
        self.assertEqual([sequence for _event, sequence in events], sorted(sequence for _event, sequence in events))
        pool.close()

    def test_worker_start_failure_rolls_back_accounting(self):
        upstream = types.SimpleNamespace(base_url="http://127.0.0.1:1/v1")
        spec = provider("unstarted", upstream)
        pool = RACE.ProviderRacePool(settings([spec], redundancy=1))
        with mock.patch.object(
            RACE.threading.Thread,
            "start",
            side_effect=RuntimeError("cannot start new thread"),
        ):
            with self.assertRaisesRegex(RACE.RaceFailure, "worker start failed"):
                pool.race(
                    {"messages": [{"role": "user", "content": "work"}]},
                    "thread-exhaustion",
                )
        snapshot = pool.snapshot()
        self.assertEqual(snapshot["requests_started"], 1)
        self.assertEqual(snapshot["requests_failed"], 1)
        self.assertEqual(snapshot["requests_won"], 0)
        self.assertEqual(snapshot["providers"][0]["calls"], 0)
        self.assertEqual(snapshot["providers"][0]["failures"], 0)
        self.assertEqual(snapshot["providers"][0]["cancellations"], 0)
        self.assertEqual(snapshot["providers"][0]["in_flight"], 0)

    def test_streaming_proxy_returns_complete_winner_and_status(self):
        first = self.upstream("first", delay=0.01, stream=True)
        second = self.upstream("second", delay=0.2, stream=True)
        pool = RACE.ProviderRacePool(
            settings([provider("first", first), provider("second", second)])
        )
        token = "local-test-token"
        server, thread = RACE.start_proxy(pool, "127.0.0.1", 0, token)
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            body = json.dumps(
                {
                    "model": "ox-alpha",
                    "stream": True,
                    "messages": [{"role": "user", "content": "work"}],
                }
            )
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "X-Oxalpha-Context-Key": "pair-3",
                },
            )
            response = connection.getresponse()
            payload = response.read()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("X-Oxalpha-Provider"), "first")
            self.assertIn(b"data: [DONE]", payload)
            connection.close()
            status_connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=2
            )
            status_connection.request(
                "GET", "/v1/race/status", headers={"Authorization": f"Bearer {token}"}
            )
            status_response = status_connection.getresponse()
            snapshot = json.loads(status_response.read())
            self.assertEqual(snapshot["requests_won"], 1)
            self.assertEqual(snapshot["providers"][0]["wins"], 1)
            time.sleep(0.3)
            settled = pool.snapshot()
            self.assertEqual(sum(item["in_flight"] for item in settled["providers"]), 0)
            self.assertGreaterEqual(settled["providers"][1]["cancellations"], 1)
            status_connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_proxy_request_body_has_a_deadline(self):
        upstream = self.upstream("unused")
        pool = RACE.ProviderRacePool(
            settings([provider("unused", upstream)], redundancy=1, timeout=0.05)
        )
        server, thread = RACE.start_proxy(pool, "127.0.0.1", 0, "local-token")
        client = socket.create_connection(("127.0.0.1", server.server_port), timeout=1)
        try:
            request = (
                "POST /v1/chat/completions HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Authorization: Bearer local-token\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: 100\r\n\r\n"
                "{"
            ).encode()
            client.sendall(request)
            response = client.recv(4096)
            self.assertIn(b" 408 ", response)
            self.assertEqual(upstream.calls, [])
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_proxy_request_body_trickle_cannot_extend_deadline(self):
        upstream = self.upstream("unused-trickle")
        pool = RACE.ProviderRacePool(
            settings([provider("unused-trickle", upstream)], redundancy=1, timeout=0.08)
        )
        server, thread = RACE.start_proxy(pool, "127.0.0.1", 0, "local-token")
        client = socket.create_connection(("127.0.0.1", server.server_port), timeout=1)
        stop = threading.Event()
        sender = None
        try:
            request = (
                "POST /v1/chat/completions HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Authorization: Bearer local-token\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: 64\r\n\r\n"
            ).encode()
            client.sendall(request)

            def trickle():
                for _index in range(64):
                    if stop.wait(0.04):
                        return
                    try:
                        client.sendall(b"x")
                    except OSError:
                        return

            sender = threading.Thread(target=trickle, daemon=True)
            started = time.monotonic()
            sender.start()
            response = client.recv(4096)
            elapsed = time.monotonic() - started
            self.assertIn(b" 408 ", response)
            self.assertLess(elapsed, 0.25)
            self.assertEqual(upstream.calls, [])
        finally:
            stop.set()
            if sender is not None:
                sender.join(timeout=1)
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

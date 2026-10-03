import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import aiohttp
import jinja2
from aiohttp import web
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

ROLE_ALIASES = {"developer": "system"}
UNSUPPORTED_FIELDS = ("n", "logprobs", "top_logprobs", "logit_bias", "best_of", "echo", "suffix")


class RequestError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def strftime_now(format_string):
    return time.strftime(format_string)


def error_body(code, message):
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


def error_response(error):
    body = error_body(error.code, error.message)
    if error.status >= 500:
        body["error"]["type"] = "server_error"
    headers = {"Retry-After": "5"} if error.status == 503 else None
    return web.json_response(body, status=error.status, headers=headers)


def longest_marker_prefix(text, markers):
    longest = 0
    for marker in markers:
        for length in range(min(len(marker) - 1, len(text)), longest, -1):
            if marker.startswith(text[-length:]):
                longest = length
                break
    return longest


class Model:
    def __init__(self, spec, base):
        directory = (base / spec["model_dir"]).resolve()
        self.id = spec["id"]
        self.context_tokens = int(spec["context_tokens"])
        self.engine = spec["engine"].rstrip("/")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        environment.filters["tojson"] = tojson
        environment.globals["raise_exception"] = raise_exception
        environment.globals["strftime_now"] = strftime_now
        self.template = environment.from_string((directory / "chat_template.jinja").read_text())
        tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())
        self.template_tokens = {name: tokenizer_config[name] for name in ("bos_token", "eos_token", "pad_token", "unk_token") if isinstance(tokenizer_config.get(name), str)}
        self.template_defaults = dict(spec.get("template_defaults", {}))
        self.template_kwargs = set(spec.get("template_kwargs", []))
        self.reasoning_effort = dict(spec.get("reasoning_effort", {}))
        self.reasoning = spec["reasoning"]
        self.tool_calls = spec["tool_calls"]
        self.stop_token_ids = [self.token_id(text) for text in spec["stop_tokens"]]

    def token_id(self, text):
        token = self.tokenizer.token_to_id(text)
        if token is None:
            raise SystemExit(f"chat_frontend: {self.id} declares {text!r} but its tokenizer has no such token")
        return token

    def render(self, messages, tools, kwargs):
        context = dict(self.template_tokens)
        context.update(self.template_defaults)
        context.update(kwargs)
        try:
            return self.template.render(messages=messages, tools=tools or None, add_generation_prompt=True, **context)
        except jinja2.exceptions.TemplateError as error:
            raise RequestError(400, "invalid_messages", f"the chat template refused the request: {error}")


def normalize_arguments(arguments):
    if isinstance(arguments, dict):
        return arguments
    if arguments is None or arguments == "":
        return {}
    try:
        parsed = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        raise RequestError(400, "invalid_messages", "tool call arguments must be a JSON object")
    if not isinstance(parsed, dict):
        raise RequestError(400, "invalid_messages", "tool call arguments must be a JSON object")
    return parsed


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise RequestError(400, "invalid_messages", "messages must be a non-empty array")
    normalized = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise RequestError(400, "invalid_messages", "every message needs a string role")
        entry = dict(message)
        entry["role"] = ROLE_ALIASES.get(entry["role"], entry["role"])
        if entry.get("content") is None:
            entry["content"] = ""
        if entry.get("tool_calls"):
            calls = []
            for call in entry["tool_calls"]:
                call = dict(call)
                if isinstance(call.get("function"), dict):
                    function = dict(call["function"])
                    function["arguments"] = normalize_arguments(function.get("arguments"))
                    call["function"] = function
                calls.append(call)
            entry["tool_calls"] = calls
        normalized.append(entry)
    return normalized


def tool_parameter_types(tools):
    types = {}
    for tool in tools or []:
        function = tool.get("function", tool) if isinstance(tool, dict) else {}
        properties = (function.get("parameters") or {}).get("properties") or {}
        types[function.get("name")] = {key: (value or {}).get("type") for key, value in properties.items() if isinstance(value, dict)}
    return types


class OutputParser:
    def __init__(self, model, tools, reasoning_open, stops):
        self.reasoning_end = model.reasoning["end"]
        self.call = model.tool_calls
        self.types = tool_parameter_types(tools)
        self.stops = [stop for stop in stops if stop]
        self.state = "reasoning" if reasoning_open else "content"
        self.buffer = ""
        self.calls = 0
        self.stopped = False

    def parse_call(self, body):
        call = self.call
        head, _, rest = body.partition(call["key_start"])
        name = head.strip()
        arguments = {}
        rest = call["key_start"] + rest if rest else ""
        types = self.types.get(name, {})
        while rest.startswith(call["key_start"]):
            key, found, rest = rest[len(call["key_start"]):].partition(call["key_end"])
            if not found:
                return None
            rest = rest.lstrip()
            if not rest.startswith(call["value_start"]):
                return None
            value, found, rest = rest[len(call["value_start"]):].partition(call["value_end"])
            if not found:
                return None
            rest = rest.lstrip()
            key = key.strip()
            if types.get(key) == "string":
                arguments[key] = value
            else:
                try:
                    arguments[key] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[key] = value
        if not name or rest.strip():
            return None
        return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}

    def feed(self, text):
        self.buffer += text
        events = []
        while self.buffer and not self.stopped:
            if self.state == "reasoning":
                index = self.buffer.find(self.reasoning_end)
                if index < 0:
                    keep = longest_marker_prefix(self.buffer, [self.reasoning_end])
                    if len(self.buffer) > keep:
                        events.append(("reasoning", self.buffer[:len(self.buffer) - keep]))
                        self.buffer = self.buffer[len(self.buffer) - keep:]
                    break
                if index:
                    events.append(("reasoning", self.buffer[:index]))
                self.buffer = self.buffer[index + len(self.reasoning_end):]
                self.state = "content"
            elif self.state == "content":
                hits = [(self.buffer.find(marker), marker) for marker in [self.call["start"]] + self.stops]
                hits = [hit for hit in hits if hit[0] >= 0]
                if not hits:
                    keep = longest_marker_prefix(self.buffer, [self.call["start"]] + self.stops)
                    if len(self.buffer) > keep:
                        events.append(("content", self.buffer[:len(self.buffer) - keep]))
                        self.buffer = self.buffer[len(self.buffer) - keep:]
                    break
                index, marker = min(hits)
                if index:
                    events.append(("content", self.buffer[:index]))
                if marker != self.call["start"]:
                    self.buffer = ""
                    self.stopped = True
                    break
                self.buffer = self.buffer[index + len(marker):]
                self.state = "call"
            else:
                index = self.buffer.find(self.call["end"])
                if index < 0:
                    break
                body = self.buffer[:index]
                self.buffer = self.buffer[index + len(self.call["end"]):]
                self.state = "content"
                parsed = self.parse_call(body)
                if parsed is None:
                    events.append(("content", self.call["start"] + body + self.call["end"]))
                else:
                    events.append(("tool_call", parsed))
                    self.calls += 1
        return events

    def finish(self):
        events = []
        if self.buffer and not self.stopped:
            if self.state == "reasoning":
                events.append(("reasoning", self.buffer))
            elif self.state == "call":
                events.append(("content", self.call["start"] + self.buffer))
            else:
                events.append(("content", self.buffer))
        self.buffer = ""
        return events


class Frontend:
    def __init__(self, config, base):
        self.models = {}
        for spec in config["models"]:
            model = Model(spec, base)
            self.models[model.id] = model
        self.session = None

    async def start(self, app):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10))

    async def stop(self, app):
        await self.session.close()

    def model_for(self, body):
        name = body.get("model")
        if name in self.models:
            return self.models[name]
        if len(self.models) == 1 and name is None:
            return next(iter(self.models.values()))
        raise RequestError(404, "model_not_found", f"model {name!r} is not served here; served: {sorted(self.models)}")

    async def health(self, request):
        states = {}
        for model in self.models.values():
            try:
                async with self.session.get(model.engine + "/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    states[model.id] = {"status": response.status, "engine": await response.json(content_type=None)}
            except (aiohttp.ClientError, TimeoutError) as error:
                states[model.id] = {"status": 0, "error": str(error)}
        healthy = all(state["status"] == 200 for state in states.values())
        return web.json_response({"ok": healthy, "models": states}, status=200 if healthy else 503)

    async def liveliness(self, request):
        return web.json_response({"ok": True})

    async def list_models(self, request):
        created = int(time.time())
        return web.json_response({"object": "list", "data": [
            {"id": model.id, "object": "model", "created": created, "owned_by": "sparkpipe", "max_model_len": model.context_tokens}
            for model in self.models.values()]})

    def prepare(self, model, body):
        for field in UNSUPPORTED_FIELDS:
            if field in body and body[field] not in (None, False, 1):
                raise RequestError(400, "unsupported_parameter", f"{field} is not supported by this server")
        response_format = body.get("response_format")
        if response_format not in (None, {"type": "text"}):
            raise RequestError(400, "unsupported_parameter", "response_format other than text is not supported by this server")
        messages = normalize_messages(body.get("messages"))
        tools = body.get("tools") if body.get("tool_choice") != "none" else None
        kwargs = {}
        effort = body.get("reasoning_effort")
        if effort is not None:
            if effort not in model.reasoning_effort:
                raise RequestError(400, "unsupported_parameter", f"reasoning_effort must be one of {sorted(model.reasoning_effort)}")
            kwargs.update(model.reasoning_effort[effort])
        for key, value in (body.get("chat_template_kwargs") or {}).items():
            if key not in model.template_kwargs:
                raise RequestError(400, "unsupported_parameter", f"chat_template_kwargs.{key} is not accepted; accepted: {sorted(model.template_kwargs)}")
            kwargs[key] = value
        prompt = model.render(messages, tools, kwargs)
        prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False).ids
        if len(prompt_ids) >= model.context_tokens:
            raise RequestError(400, "context_length_exceeded", f"the prompt has {len(prompt_ids)} tokens and {model.id} serves {model.context_tokens}")
        limit = body.get("max_completion_tokens", body.get("max_tokens"))
        room = model.context_tokens - len(prompt_ids)
        max_tokens = room if limit is None else min(int(limit), room)
        if max_tokens <= 0:
            raise RequestError(400, "invalid_request", "max_tokens must be positive")
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        reasoning_open = prompt.rstrip().endswith(model.reasoning["start"])
        parser = OutputParser(model, tools, reasoning_open, stops)
        engine_body = {"prompt_token_ids": prompt_ids, "max_tokens": max_tokens, "stream": True, "stop_token_ids": model.stop_token_ids}
        return engine_body, parser, len(prompt_ids)

    async def engine_events(self, model, engine_body):
        try:
            async with self.session.post(model.engine + "/v1/completions", json=engine_body) as response:
                if response.status != 200:
                    text = await response.text()
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = error_body("engine_error", text.strip() or f"engine status {response.status}")
                    raise RequestError(response.status, payload.get("error", {}).get("code", "engine_error"), payload.get("error", {}).get("message", text))
                async for raw in response.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    yield json.loads(data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            raise RequestError(503, "engine_unavailable", f"the {model.id} engine at {model.engine} is unreachable: {error}")

    async def generate(self, model, engine_body, parser):
        decoder = DecodeStream(skip_special_tokens=False)
        finish = None
        usage = None
        async for chunk in self.engine_events(model, engine_body):
            if "error" in chunk:
                raise RequestError(502, "engine_error", json.dumps(chunk["error"]))
            text = []
            for token in chunk.get("tokens", []):
                if token in model.stop_token_ids:
                    continue
                piece = decoder.step(model.tokenizer, token)
                if piece:
                    text.append(piece)
            choices = chunk.get("choices") or [{}]
            finish = choices[0].get("finish_reason") or finish
            usage = chunk.get("usage") or usage
            for event in parser.feed("".join(text)):
                yield event
            if parser.stopped:
                finish = "stop"
                break
        for event in parser.finish():
            yield event
        yield ("finish", {"finish_reason": "tool_calls" if parser.calls else (finish or "stop"), "usage": usage})

    def usage_block(self, usage, prompt_tokens):
        usage = usage or {}
        completion = usage.get("completion_tokens", 0)
        prompt = usage.get("prompt_tokens", prompt_tokens)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion, "prompt_tokens_details": {"cached_tokens": cached}}

    def log(self, model, prompt_tokens, usage, started, first, finish):
        now = time.monotonic()
        usage = usage or {}
        completion = usage.get("completion_tokens", 0)
        decode_seconds = now - first if first else 0.0
        print(json.dumps({"event": "chat", "model": model.id, "prompt_tokens": prompt_tokens,
                          "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                          "completion_tokens": completion, "ttft_s": round(first - started, 3) if first else None,
                          "decode_tok_s": round((completion - 1) / decode_seconds, 2) if completion > 1 and decode_seconds > 0 else None,
                          "finish_reason": finish}), file=sys.stderr, flush=True)

    async def chat(self, request):
        started = time.monotonic()
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(error_body("invalid_json", "the request body is not JSON"), status=400)
        try:
            model = self.model_for(body)
            engine_body, parser, prompt_tokens = self.prepare(model, body)
            if body.get("stream"):
                return await self.chat_stream(request, model, body, engine_body, parser, prompt_tokens, started)
            content, reasoning, calls = [], [], []
            result = {}
            first = None
            async for kind, value in self.generate(model, engine_body, parser):
                first = first or time.monotonic()
                if kind == "content":
                    content.append(value)
                elif kind == "reasoning":
                    reasoning.append(value)
                elif kind == "tool_call":
                    calls.append(value)
                else:
                    result = value
        except RequestError as error:
            return error_response(error)
        message = {"role": "assistant", "content": "".join(content) or None}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if calls:
            message["tool_calls"] = calls
        self.log(model, prompt_tokens, result.get("usage"), started, first, result.get("finish_reason"))
        return web.json_response({"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                                  "model": model.id, "choices": [{"index": 0, "message": message, "finish_reason": result.get("finish_reason")}],
                                  "usage": self.usage_block(result.get("usage"), prompt_tokens)})

    async def chat_stream(self, request, model, body, engine_body, parser, prompt_tokens, started):
        identifier = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        response = None
        first = None
        calls = 0
        result = {}

        def chunk(delta, finish_reason=None, usage=None):
            payload = {"id": identifier, "object": "chat.completion.chunk", "created": created, "model": model.id,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            if usage is not None:
                payload["usage"] = usage
            return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()

        try:
            async for kind, value in self.generate(model, engine_body, parser):
                if response is None:
                    response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
                    await response.prepare(request)
                    await response.write(chunk({"role": "assistant", "content": ""}))
                    first = time.monotonic()
                if kind == "content" and value:
                    await response.write(chunk({"content": value}))
                elif kind == "reasoning" and value:
                    await response.write(chunk({"reasoning_content": value}))
                elif kind == "tool_call":
                    await response.write(chunk({"tool_calls": [dict(value, index=calls)]}))
                    calls += 1
                elif kind == "finish":
                    result = value
        except RequestError as error:
            if response is None:
                return error_response(error)
            await response.write(("data: " + json.dumps(error_body(error.code, error.message)) + "\n\n").encode())
            await response.write(b"data: [DONE]\n\n")
            return response
        except (ConnectionResetError, aiohttp.ClientConnectionResetError):
            self.log(model, prompt_tokens, result.get("usage"), started, first, "client_gone")
            return response
        usage = self.usage_block(result.get("usage"), prompt_tokens)
        await response.write(chunk({}, result.get("finish_reason"), usage))
        if include_usage:
            payload = {"id": identifier, "object": "chat.completion.chunk", "created": created, "model": model.id, "choices": [], "usage": usage}
            await response.write(("data: " + json.dumps(payload) + "\n\n").encode())
        await response.write(b"data: [DONE]\n\n")
        self.log(model, prompt_tokens, result.get("usage"), started, first, result.get("finish_reason"))
        return response


def main():
    parser = argparse.ArgumentParser(description="OpenAI chat front end for SparkPipe engines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    config_path = Path(arguments.config).resolve()
    frontend = Frontend(json.loads(config_path.read_text()), config_path.parent)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(frontend.start)
    app.on_cleanup.append(frontend.stop)
    app.router.add_get("/health", frontend.health)
    app.router.add_get("/health/liveliness", frontend.liveliness)
    app.router.add_get("/v1/models", frontend.list_models)
    app.router.add_post("/v1/chat/completions", frontend.chat)
    web.run_app(app, host=arguments.host, port=arguments.port, access_log=None)


if __name__ == "__main__":
    main()

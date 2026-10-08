import argparse
import asyncio
import codecs
import importlib.util
import json
import re
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
UNSUPPORTED_FIELDS = ("n", "logit_bias", "best_of", "echo", "suffix")
NEUTRAL_FIELDS = {"frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1, "min_p": 0}
MAX_TOP_LOGPROBS = 20
MAX_TOP_K = 1048576


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


ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')


def tag_attributes(text):
    return {key: value.replace("&quot;", '"').replace("&amp;", "&") for key, value in ATTRIBUTE.findall(text)}


class TokenizersVocabulary:
    def __init__(self, directory, spec):
        self.tokenizer = Tokenizer.from_file(str(directory / spec.get("file", "tokenizer.json")))

    def encode(self, text, special):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decoder(self):
        stream = DecodeStream(skip_special_tokens=False)
        return lambda token: stream.step(self.tokenizer, token)

    def token_id(self, text):
        return self.tokenizer.token_to_id(text)

    def piece(self, token):
        return self.tokenizer.decode([token], skip_special_tokens=False)


class TiktokenVocabulary:
    def __init__(self, directory, spec):
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        ranks = load_tiktoken_bpe(str(directory / spec["file"]))
        config = json.loads((directory / spec["special_tokens_from"]).read_text())
        specials = {entry["content"]: int(identifier) for identifier, entry in config.get("added_tokens_decoder", {}).items()}
        taken = set(specials.values())
        for identifier in range(len(ranks), len(ranks) + int(spec.get("reserved_special_tokens", 0))):
            if identifier not in taken:
                specials[f"<|reserved_token_{identifier}|>"] = identifier
        self.specials = specials
        self.encoding = tiktoken.Encoding(name=spec.get("name", directory.name), pat_str=spec["pattern"], mergeable_ranks=ranks, special_tokens=specials)

    def encode(self, text, special):
        return self.encoding.encode(text, allowed_special="all") if special else self.encoding.encode_ordinary(text)

    def decoder(self):
        utf8 = codecs.getincrementaldecoder("utf-8")("replace")
        return lambda token: utf8.decode(self.encoding.decode_single_token_bytes(token))

    def token_id(self, text):
        if text in self.specials:
            return self.specials[text]
        ids = self.encoding.encode_ordinary(text)
        return ids[0] if len(ids) == 1 else None

    def piece(self, token):
        return self.encoding.decode_single_token_bytes(token).decode("utf-8", "replace")


VOCABULARIES = {"tokenizers": TokenizersVocabulary, "tiktoken": TiktokenVocabulary}


class TemplateRenderer:
    def __init__(self, directory, spec, vocabulary):
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        environment.filters["tojson"] = tojson
        environment.globals["raise_exception"] = raise_exception
        environment.globals["strftime_now"] = strftime_now
        self.template = environment.from_string((directory / spec.get("file", "chat_template.jinja")).read_text())
        tokenizer_config = json.loads((directory / spec.get("tokenizer_config", "tokenizer_config.json")).read_text())
        self.template_tokens = {name: tokenizer_config[name] for name in ("bos_token", "eos_token", "pad_token", "unk_token") if isinstance(tokenizer_config.get(name), str)}
        self.vocabulary = vocabulary

    def render(self, messages, tools, kwargs):
        context = dict(self.template_tokens)
        context.update(kwargs)
        try:
            text = self.template.render(messages=messages, tools=tools or None, add_generation_prompt=True, **context)
        except jinja2.exceptions.TemplateError as error:
            raise RequestError(400, "invalid_messages", f"the chat template refused the request: {error}")
        return text, self.vocabulary.encode(text, True)


class SegmentRenderer:
    def __init__(self, directory, spec, vocabulary):
        path = directory / spec["module"]
        name = f"chat_renderer_{uuid.uuid4().hex}"
        loader = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(loader)
        sys.modules[name] = module
        loader.loader.exec_module(module)
        self.function = getattr(module, spec["function"])
        self.vocabulary = vocabulary

    def render(self, messages, tools, kwargs):
        try:
            segments = self.function(messages, tools or None, add_generation_prompt=True, **kwargs)
        except (ValueError, KeyError, TypeError, AssertionError) as error:
            raise RequestError(400, "invalid_messages", f"the chat renderer refused the request: {error}")
        ids = []
        for segment in segments:
            ids.extend(self.vocabulary.encode(segment.text, segment.allow_special))
        return "".join(segment.text for segment in segments), ids


RENDERERS = {"template": TemplateRenderer, "segments": SegmentRenderer}


class WarmGate:
    POLL_SECONDS = 1.0

    def __init__(self, model_id, engine, spec):
        if not isinstance(spec, dict) or set(spec) != {"depth", "wait_seconds", "activation_url"}:
            raise SystemExit(f"chat_frontend: {model_id} needs warm_queue with exactly depth, wait_seconds and activation_url (null when an operator starts the engine)")
        self.model_id = model_id
        self.engine = engine
        self.depth = int(spec["depth"])
        self.wait_seconds = float(spec["wait_seconds"])
        self.activation_url = spec["activation_url"]
        if self.depth <= 0 or self.wait_seconds <= 0 or (self.activation_url is not None and not isinstance(self.activation_url, str)):
            raise SystemExit(f"chat_frontend: {model_id} warm_queue depth and wait_seconds must be positive and activation_url a URL or null")
        self.event = asyncio.Event()
        self.poller = None
        self.waiting = 0
        self.activations = 0
        self.released = 0
        self.overflowed = 0
        self.expired = 0

    def state(self):
        return {"waiting": self.waiting, "depth": self.depth, "wait_seconds": self.wait_seconds, "activations": self.activations,
                "released": self.released, "overflowed": self.overflowed, "expired": self.expired}

    async def poll(self, session):
        if self.activation_url is not None:
            self.activations += 1
            try:
                async with session.post(self.activation_url, json={"model": self.model_id}, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    await response.read()
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                print(json.dumps({"event": "warm_activation_failed", "model": self.model_id, "error": str(error)}), file=sys.stderr, flush=True)
        while True:
            try:
                async with session.get(self.engine + "/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        self.event.set()
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(self.POLL_SECONDS)

    async def wait_ready(self, session, timeout):
        if timeout <= 0:
            self.expired += 1
            raise RequestError(503, "warm_queue_expired", f"the {self.model_id} engine did not become ready within {self.wait_seconds:g} s")
        if self.waiting >= self.depth:
            self.overflowed += 1
            raise RequestError(503, "warm_queue_full", f"the {self.model_id} engine is not ready and {self.waiting} requests already wait for it")
        if self.poller is None or self.poller.done():
            self.event.clear()
            self.poller = asyncio.ensure_future(self.poll(session))
        self.waiting += 1
        try:
            await asyncio.wait_for(asyncio.shield(self.event.wait()), timeout)
        except asyncio.TimeoutError:
            self.expired += 1
            raise RequestError(503, "warm_queue_expired", f"the {self.model_id} engine did not become ready within {self.wait_seconds:g} s")
        finally:
            self.waiting -= 1
        self.released += 1


class Model:
    def __init__(self, spec, base):
        directory = (base / spec["model_dir"]).resolve()
        self.id = spec["id"]
        self.context_tokens = int(spec["context_tokens"])
        self.engine = spec["engine"].rstrip("/")
        vocabulary = dict(spec.get("tokenizer", {"kind": "tokenizers"}))
        renderer = dict(spec.get("renderer", {"kind": "template"}))
        if vocabulary.get("kind") not in VOCABULARIES or renderer.get("kind") not in RENDERERS:
            raise SystemExit(f"chat_frontend: {self.id} tokenizer kind must be one of {sorted(VOCABULARIES)} and renderer kind one of {sorted(RENDERERS)}")
        self.vocabulary = VOCABULARIES[vocabulary["kind"]](directory, vocabulary)
        self.renderer = RENDERERS[renderer["kind"]](directory, renderer, self.vocabulary)
        self.template_defaults = dict(spec.get("template_defaults", {}))
        self.template_kwargs = set(spec.get("template_kwargs", []))
        self.reasoning_effort = dict(spec.get("reasoning_effort", {}))
        self.reasoning = spec["reasoning"]
        self.tool_calls = dict(spec["tool_calls"])
        self.tool_calls.setdefault("format", "key_value")
        if self.tool_calls["format"] not in ("key_value", "tagged"):
            raise SystemExit(f"chat_frontend: {self.id} tool_calls format must be key_value or tagged")
        self.content_markers = list(spec.get("content_markers", []))
        self.end_markers = list(spec.get("end_markers", []))
        self.stop_token_ids = [self.token_id(text) for text in spec["stop_tokens"]]
        self.gate = WarmGate(self.id, self.engine, spec.get("warm_queue"))

    def logprob(self, pair):
        text = self.vocabulary.piece(pair[0])
        return {"token": text, "logprob": -9999.0 if pair[1] is None else pair[1], "bytes": list(text.encode("utf-8"))}

    def token_id(self, text):
        token = self.vocabulary.token_id(text)
        if token is None:
            raise SystemExit(f"chat_frontend: {self.id} declares {text!r} but its tokenizer has no such token")
        return token

    def render(self, messages, tools, kwargs):
        context = dict(self.template_defaults)
        context.update(kwargs)
        return self.renderer.render(messages, tools, context)


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


def number(body, name, low, high, integer=False):
    value = body.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (integer and not isinstance(value, int)) or not low <= value <= high:
        kind = "an integer" if integer else "a number"
        raise RequestError(400, "invalid_request", f"{name} must be {kind} between {low} and {high}")
    return value


def sampling_fields(body):
    fields = {}
    temperature = number(body, "temperature", 0, 2)
    if temperature is not None and temperature != 0 and temperature < 0.0001:
        raise RequestError(400, "invalid_request", "temperature must be 0 or between 0.0001 and 2")
    top_p = number(body, "top_p", 0, 1)
    if top_p is not None and top_p <= 0:
        raise RequestError(400, "invalid_request", "top_p must be above 0 and at most 1")
    top_k = number(body, "top_k", -1, MAX_TOP_K, integer=True)
    seed = number(body, "seed", 0, 2 ** 64 - 1, integer=True)
    if temperature:
        fields["temperature"] = temperature
        if top_p is not None:
            fields["top_p"] = top_p
        if top_k is not None and top_k > 0:
            fields["top_k"] = top_k
        if seed is not None:
            fields["seed"] = seed
    logprobs = body.get("logprobs")
    if logprobs not in (None, True, False):
        raise RequestError(400, "invalid_request", "logprobs must be a boolean")
    top_logprobs = number(body, "top_logprobs", 0, MAX_TOP_LOGPROBS, integer=True)
    if top_logprobs is not None and not logprobs:
        raise RequestError(400, "invalid_request", "top_logprobs needs logprobs true")
    if logprobs:
        fields["logprobs"] = top_logprobs or 0
    return fields


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
        self.stops = [stop for stop in stops if stop] + model.end_markers
        self.end_markers = model.end_markers
        self.drops = model.content_markers
        self.ended = False
        self.state = "reasoning" if reasoning_open else "content"
        self.buffer = ""
        self.calls = 0
        self.stopped = False

    def parse_call(self, body):
        if self.call["format"] == "tagged":
            return self.parse_tagged_call(body)
        return self.parse_key_value_call(body)

    def typed(self, name, key, kind, value):
        if kind == "string" or (kind is None and self.types.get(name, {}).get(key) == "string"):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def parse_tagged_call(self, body):
        call = self.call
        head, found, rest = body.partition(call["attribute_end"])
        if not found:
            return None
        name = tag_attributes(head).get("tool")
        arguments = {}
        rest = rest.strip()
        while rest:
            if rest.startswith(call["object_start"]):
                _, found, rest = rest[len(call["object_start"]):].partition(call["attribute_end"])
                value, closed, rest = rest.partition(call["object_end"])
                if not found or not closed:
                    return None
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    return None
                if not isinstance(parsed, dict):
                    return None
                arguments.update(parsed)
            elif rest.startswith(call["argument_start"]):
                head, found, rest = rest[len(call["argument_start"]):].partition(call["attribute_end"])
                value, closed, rest = rest.partition(call["argument_end"])
                attributes = tag_attributes(head)
                if not found or not closed or "key" not in attributes:
                    return None
                arguments[attributes["key"]] = self.typed(name, attributes["key"], attributes.get("type"), value)
            else:
                return None
            rest = rest.strip()
        if not name:
            return None
        return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}

    def parse_key_value_call(self, body):
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
                markers = [self.call["start"]] + self.stops + self.drops
                hits = [(self.buffer.find(marker), marker) for marker in markers]
                hits = [hit for hit in hits if hit[0] >= 0]
                if not hits:
                    keep = longest_marker_prefix(self.buffer, markers)
                    if len(self.buffer) > keep:
                        events.append(("content", self.buffer[:len(self.buffer) - keep]))
                        self.buffer = self.buffer[len(self.buffer) - keep:]
                    break
                index, marker = min(hits)
                if index:
                    events.append(("content", self.buffer[:index]))
                if marker in self.drops and marker != self.call["start"]:
                    self.buffer = self.buffer[index + len(marker):]
                    continue
                if marker != self.call["start"]:
                    self.buffer = ""
                    self.stopped = True
                    self.ended = marker in self.end_markers
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
        self.control = config.get("control")
        if self.control is not None and set(self.control) != {"status_url", "activate_url"}:
            raise SystemExit("chat_frontend: control needs exactly status_url and activate_url")
        self.site = (base / config["site_dir"]).resolve() if config.get("site_dir") else None
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
            states[model.id]["warm_queue"] = model.gate.state()
        healthy = all(state["status"] == 200 for state in states.values())
        return web.json_response({"ok": healthy, "models": states}, status=200 if healthy else 503)

    async def liveliness(self, request):
        return web.json_response({"ok": True})

    async def control_status(self, request):
        if self.control is None:
            return web.json_response(error_body("not_found", "this front end has no model control"), status=404)
        try:
            async with self.session.get(self.control["status_url"], timeout=aiohttp.ClientTimeout(total=5)) as response:
                return web.json_response(await response.json(content_type=None), status=response.status)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as error:
            return web.json_response(error_body("control_unavailable", str(error)), status=503)

    async def control_activate(self, request):
        if self.control is None:
            return web.json_response(error_body("not_found", "this front end has no model control"), status=404)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(error_body("invalid_json", "the request body is not JSON"), status=400)
        model = body.get("model") if isinstance(body, dict) else None
        if model not in self.models:
            return web.json_response(error_body("model_not_found", f"model {model!r} is not served here; served: {sorted(self.models)}"), status=404)
        try:
            async with self.session.post(self.control["activate_url"], json={"model": model}, timeout=aiohttp.ClientTimeout(total=10)) as response:
                return web.json_response(await response.json(content_type=None), status=response.status)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as error:
            return web.json_response(error_body("control_unavailable", str(error)), status=503)

    async def site_root(self, request):
        raise web.HTTPFound("/site/playground.html")

    async def list_models(self, request):
        created = int(time.time())
        return web.json_response({"object": "list", "data": [
            {"id": model.id, "object": "model", "created": created, "owned_by": "sparkpipe", "max_model_len": model.context_tokens}
            for model in self.models.values()]})

    def prepare(self, model, body):
        for field in UNSUPPORTED_FIELDS:
            if field in body and body[field] not in (None, False, 1):
                raise RequestError(400, "unsupported_parameter", f"{field} is not supported by this server")
        for field, neutral in NEUTRAL_FIELDS.items():
            if body.get(field) not in (None, neutral):
                raise RequestError(400, "unsupported_parameter", f"{field} is not supported by this server; send {neutral} or leave it out")
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
        prompt, prompt_ids = model.render(messages, tools, kwargs)
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
        engine_body.update(sampling_fields(body))
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
                    if response.status == 503 and payload.get("status") == "starting":
                        raise RequestError(503, "engine_starting", f"the {model.id} engine is starting: {text}")
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

    async def engine_stream(self, model, engine_body):
        deadline = None
        while True:
            started = False
            try:
                async for chunk in self.engine_events(model, engine_body):
                    started = True
                    yield chunk
                return
            except RequestError as error:
                if started or error.code not in ("engine_starting", "engine_unavailable"):
                    raise
            now = asyncio.get_running_loop().time()
            deadline = deadline if deadline is not None else now + model.gate.wait_seconds
            await model.gate.wait_ready(self.session, deadline - now)

    def logprob_entries(self, model, chunk, expected):
        tokens = chunk.get("tokens", [])
        rows = chunk.get("token_logprobs")
        if rows is None or len(rows) != len(tokens):
            raise RequestError(502, "engine_error", "the engine returned tokens without their logprobs")
        entries = []
        for token, row in zip(tokens, rows):
            if len(row) != expected or row[0][0] != token:
                raise RequestError(502, "engine_error", "the engine returned logprobs that do not match its tokens")
            if token in model.stop_token_ids:
                continue
            entry = model.logprob(row[0])
            entry["top_logprobs"] = [model.logprob(pair) for pair in row[1:]]
            entries.append(entry)
        return entries

    async def generate(self, model, engine_body, parser):
        step = model.vocabulary.decoder()
        finish = None
        usage = None
        logprobs = engine_body.get("logprobs")
        async for chunk in self.engine_stream(model, engine_body):
            if "error" in chunk:
                raise RequestError(502, "engine_error", json.dumps(chunk["error"]))
            if logprobs is not None:
                entries = self.logprob_entries(model, chunk, logprobs + 1)
                if entries:
                    yield ("logprobs", entries)
            text = []
            for token in chunk.get("tokens", []):
                if token in model.stop_token_ids:
                    continue
                piece = step(token)
                if piece:
                    text.append(piece)
            choices = chunk.get("choices") or [{}]
            finish = choices[0].get("finish_reason") or finish
            usage = chunk.get("usage") or usage
            for event in parser.feed("".join(text)):
                yield event
            if parser.stopped and not parser.ended:
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
            content, reasoning, calls, logprobs = [], [], [], []
            result = {}
            first = None
            async for kind, value in self.generate(model, engine_body, parser):
                first = first or time.monotonic()
                if kind == "logprobs":
                    logprobs.extend(value)
                elif kind == "content":
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
        choice = {"index": 0, "message": message, "logprobs": {"content": logprobs} if "logprobs" in engine_body else None, "finish_reason": result.get("finish_reason")}
        return web.json_response({"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                                  "model": model.id, "choices": [choice], "usage": self.usage_block(result.get("usage"), prompt_tokens)})

    async def chat_stream(self, request, model, body, engine_body, parser, prompt_tokens, started):
        identifier = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        response = None
        first = None
        calls = 0
        result = {}

        def chunk(delta, finish_reason=None, usage=None, logprobs=None):
            choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            if logprobs is not None:
                choice["logprobs"] = {"content": logprobs}
            payload = {"id": identifier, "object": "chat.completion.chunk", "created": created, "model": model.id, "choices": [choice]}
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
                if kind == "logprobs":
                    await response.write(chunk({}, logprobs=value))
                elif kind == "content" and value:
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
    app.router.add_get("/v1/sparkpipe/status", frontend.control_status)
    app.router.add_post("/v1/sparkpipe/activate", frontend.control_activate)
    if frontend.site is not None:
        app.router.add_get("/", frontend.site_root)
        app.router.add_static("/site/", frontend.site)
    web.run_app(app, host=arguments.host, port=arguments.port, access_log=None)


if __name__ == "__main__":
    main()

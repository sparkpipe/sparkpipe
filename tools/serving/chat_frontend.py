import argparse
import asyncio
import codecs
import importlib.util
import ipaddress
import json
import os
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
CHANNEL_KEYS = ("message_start", "header_end", "message_end", "recipient_prefix", "reasoning_recipient", "content_recipient")
UNSUPPORTED_FIELDS = ("n", "logit_bias", "best_of", "echo", "suffix")
NEUTRAL_FIELDS = {"frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1, "min_p": 0}
MAX_TOP_LOGPROBS = 20
MAX_TOP_K = 1048576


def read_token(path):
    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        raise SystemExit(f"chat_frontend: {path} has mode {mode:o}; the token file must be readable by its owner only")
    token = Path(path).read_text().strip()
    if len(token) < 32:
        raise SystemExit(f"chat_frontend: the token in {path} must be at least 32 characters")
    return token


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
CALL_NAME = re.compile(r"\w+")
BARE_KEY = re.compile(r'[^\s{}\[\],:"]+')
SCALAR = json.JSONDecoder()


def tag_attributes(text):
    return {key: value.replace("&quot;", '"').replace("&amp;", "&") for key, value in ATTRIBUTE.findall(text)}


def skip_space(text, position):
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def delimited_string(text, position, delimiter):
    end = text.find(delimiter, position + len(delimiter))
    if end < 0:
        raise ValueError("unterminated string")
    return text[position + len(delimiter):end], end + len(delimiter)


def delimited_key(text, position, delimiter):
    position = skip_space(text, position)
    if text.startswith(delimiter, position):
        return delimited_string(text, position, delimiter)
    if text.startswith('"', position):
        key, end = SCALAR.raw_decode(text, position)
        return key, end
    match = BARE_KEY.match(text, position)
    if match is None:
        raise ValueError("missing key")
    return match.group(0), match.end()


def delimited_value(text, position, delimiter):
    position = skip_space(text, position)
    if text.startswith(delimiter, position):
        return delimited_string(text, position, delimiter)
    if text.startswith("{", position) or text.startswith("[", position):
        mapping = text[position] == "{"
        close = "}" if mapping else "]"
        result = {} if mapping else []
        position = skip_space(text, position + 1)
        if text.startswith(close, position):
            return result, position + 1
        while True:
            if mapping:
                key, position = delimited_key(text, position, delimiter)
                position = skip_space(text, position)
                if not text.startswith(":", position):
                    raise ValueError("missing colon")
                result[key], position = delimited_value(text, position + 1, delimiter)
            else:
                value, position = delimited_value(text, position, delimiter)
                result.append(value)
            position = skip_space(text, position)
            if text.startswith(",", position):
                position += 1
            elif text.startswith(close, position):
                return result, position + 1
            else:
                raise ValueError("missing separator")
    return SCALAR.raw_decode(text, position)


class TokenizersVocabulary:
    def __init__(self, directory, spec):
        self.adopt(Tokenizer.from_file(str(directory / spec.get("file", "tokenizer.json"))))

    @classmethod
    def from_tokenizer(cls, tokenizer):
        vocabulary = cls.__new__(cls)
        vocabulary.adopt(tokenizer)
        return vocabulary

    def adopt(self, tokenizer):
        self.tokenizer = tokenizer
        document = json.loads(tokenizer.to_str())
        document["added_tokens"] = []
        self.plain = Tokenizer.from_str(json.dumps(document))

    def encode(self, text, special):
        if special:
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        return self.plain.encode(text, add_special_tokens=False).ids

    def control_strings(self):
        return [token.content for token in self.tokenizer.get_added_tokens_decoder().values()]

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

    def control_strings(self):
        return list(self.specials)

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

    def render_text(self, messages, tools, context):
        try:
            return self.template.render(messages=messages, tools=tools or None, add_generation_prompt=True, **context)
        except jinja2.exceptions.TemplateError as error:
            raise RequestError(400, "invalid_messages", f"the chat template refused the request: {error}")
        except Exception as error:
            raise RequestError(400, "invalid_messages", f"the chat template could not render the request: {type(error).__name__}: {error}")

    def render(self, messages, tools, kwargs):
        context = dict(self.template_tokens)
        context.update(kwargs)
        text = self.render_text(messages, tools, context)
        specials = [token for token in self.vocabulary.control_strings() if token]
        carried = []

        def mark(value):
            if isinstance(value, str):
                if any(token in value for token in specials):
                    carried.append(value)
                    return f"\ue000{len(carried) - 1}\ue001"
                return value
            if isinstance(value, dict):
                if any(isinstance(key, str) and any(token in key for token in specials) for key in value):
                    raise RequestError(400, "invalid_messages", "a request key carries a control token string")
                return {key: mark(item) for key, item in value.items()}
            if isinstance(value, list):
                return [mark(item) for item in value]
            return value

        marked_messages, marked_tools = mark(messages), mark(tools)
        if not carried:
            return text, self.vocabulary.encode(text, True)
        pieces = re.split("\ue000([0-9]+)\ue001", self.render_text(marked_messages, marked_tools, context))
        ids, rebuilt = [], []
        for index, piece in enumerate(pieces):
            if index % 2 == 0:
                rebuilt.append(piece)
                ids.extend(self.vocabulary.encode(piece, True))
            else:
                rebuilt.append(carried[int(piece)])
                ids.extend(self.vocabulary.encode(carried[int(piece)], False))
        if "".join(rebuilt) != text:
            raise RequestError(400, "invalid_messages", "a message carries a control token string inside text the chat template rewrites")
        return text, ids


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
        except Exception as error:
            raise RequestError(400, "invalid_messages", f"the chat renderer refused the request: {type(error).__name__}: {error}")
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

    async def poll(self, session, headers):
        if self.activation_url is not None:
            self.activations += 1
            try:
                async with session.post(self.activation_url, json={"model": self.model_id}, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    await response.read()
                    if response.status >= 400:
                        print(json.dumps({"event": "warm_activation_refused", "model": self.model_id, "status": response.status}), file=sys.stderr, flush=True)
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

    async def wait_ready(self, session, timeout, headers):
        if timeout <= 0:
            self.expired += 1
            raise RequestError(503, "warm_queue_expired", f"the {self.model_id} engine did not become ready within {self.wait_seconds:g} s")
        if self.waiting >= self.depth:
            self.overflowed += 1
            raise RequestError(503, "warm_queue_full", f"the {self.model_id} engine is not ready and {self.waiting} requests already wait for it")
        if self.poller is None or self.poller.done():
            self.event.clear()
            self.poller = asyncio.ensure_future(self.poll(session, headers))
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
        self.max_output_tokens = int(spec["max_output_tokens"])
        self.sampling_defaults = dict(spec["sampling_defaults"])
        if set(self.sampling_defaults) - set(SAMPLING_DEFAULT_KEYS) or not str(spec.get("sampling_source", "")).strip():
            raise SystemExit(f"chat_frontend: {self.id} sampling_defaults takes only {', '.join(SAMPLING_DEFAULT_KEYS)} and needs a sampling_source")
        try:
            sampling_fields({}, self.sampling_defaults)
        except RequestError as error:
            raise SystemExit(f"chat_frontend: {self.id} sampling_defaults: {error.message}")
        if self.max_output_tokens <= 0:
            raise SystemExit(f"chat_frontend: {self.id} max_output_tokens must be positive")
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
        self.channels = spec.get("channels")
        if self.channels is not None and set(self.channels) != set(CHANNEL_KEYS):
            raise SystemExit(f"chat_frontend: {self.id} channels needs exactly {', '.join(CHANNEL_KEYS)}")
        if (self.channels is None) == ("reasoning" not in spec):
            raise SystemExit(f"chat_frontend: {self.id} needs either reasoning start and end markers or a channels protocol, not both")
        self.reasoning = spec.get("reasoning")
        self.tool_calls = dict(spec["tool_calls"])
        self.tool_calls.setdefault("format", "key_value")
        if self.tool_calls["format"] not in ("key_value", "tagged", "function_xml", "object"):
            raise SystemExit(f"chat_frontend: {self.id} tool_calls format must be key_value, tagged, function_xml or object")
        if self.tool_calls["format"] == "object" and not all(isinstance(self.tool_calls.get(key), str) and self.tool_calls[key] for key in ("name_prefix", "string_delimiter")):
            raise SystemExit(f"chat_frontend: {self.id} object tool calls need name_prefix and string_delimiter")
        self.content_markers = list(spec.get("content_markers", []))
        self.end_markers = list(spec.get("end_markers", []))
        self.stop_token_ids = [self.token_id(text) for text in spec["stop_tokens"]]
        self.gate = WarmGate(self.id, self.engine, spec.get("warm_queue"))
        self.memory = spec.get("chat_memory")
        if self.memory is not None and not isinstance(self.memory, dict):
            raise SystemExit(f"chat_frontend: {self.id} chat_memory must be an object")

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


SAMPLING_DEFAULT_KEYS = ("temperature", "top_p", "top_k")


def sampling_fields(body, defaults):
    fields = {}
    chosen = dict(defaults)
    chosen.update({key: body[key] for key in SAMPLING_DEFAULT_KEYS if body.get(key) is not None})
    temperature = number(chosen, "temperature", 0, 2)
    if temperature is not None and temperature != 0 and temperature < 0.0001:
        raise RequestError(400, "invalid_request", "temperature must be 0 or between 0.0001 and 2")
    top_p = number(chosen, "top_p", 0, 1)
    if top_p is not None and top_p <= 0:
        raise RequestError(400, "invalid_request", "top_p must be above 0 and at most 1")
    top_k = number(chosen, "top_k", -1, MAX_TOP_K, integer=True)
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


def call_entry(name, arguments):
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


class OutputParser:
    def __init__(self, model, tools, reasoning_open, stops):
        self.reasoning_start = model.reasoning["start"]
        self.reasoning_end = model.reasoning["end"]
        self.call = model.tool_calls
        self.types = tool_parameter_types(tools)
        self.stops = [stop for stop in stops if stop] + model.end_markers
        self.end_markers = model.end_markers
        self.drops = model.content_markers
        self.ended = False
        self.state = "reasoning" if reasoning_open else "content"
        self.fresh = not reasoning_open
        self.buffer = ""
        self.calls = 0
        self.stopped = False

    def parse_call(self, body):
        if self.call["format"] == "tagged":
            return self.parse_tagged_call(body)
        if self.call["format"] == "function_xml":
            return self.parse_function_xml_call(body)
        if self.call["format"] == "object":
            return self.parse_object_call(body)
        return self.parse_key_value_call(body)

    def parse_object_call(self, body):
        prefix = self.call["name_prefix"]
        name = CALL_NAME.match(body, len(prefix)) if body.startswith(prefix) else None
        if name is None or not body.startswith("{", name.end()):
            return None
        try:
            arguments, end = delimited_value(body, name.end(), self.call["string_delimiter"])
        except ValueError:
            return None
        if not isinstance(arguments, dict) or body[end:].strip():
            return None
        return call_entry(name.group(0), arguments)

    def argument_kind(self, attributes):
        flag = self.call.get("string_attribute")
        if flag is None:
            return attributes.get("type")
        if flag not in attributes:
            return None
        return "string" if attributes[flag] == "true" else "json"

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
        name = tag_attributes(head).get(call.get("name_attribute", "tool"))
        arguments = {}
        rest = rest.strip()
        while rest:
            if call.get("object_start") and rest.startswith(call["object_start"]):
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
                key = attributes.get(call.get("key_attribute", "key"))
                if not found or not closed or key is None:
                    return None
                arguments[key] = self.typed(name, key, self.argument_kind(attributes), value)
            else:
                return None
            rest = rest.strip()
        if not name:
            return None
        return call_entry(name, arguments)

    def parse_function_xml_call(self, body):
        call = self.call
        rest = body.strip()
        if not rest.startswith(call["function_start"]) or not rest.endswith(call["function_end"]):
            return None
        name, found, rest = rest[len(call["function_start"]):len(rest) - len(call["function_end"])].partition(call["attribute_end"])
        name = name.strip()
        if not found or not name:
            return None
        arguments = {}
        rest = rest.strip()
        while rest:
            if not rest.startswith(call["parameter_start"]):
                return None
            key, found, rest = rest[len(call["parameter_start"]):].partition(call["attribute_end"])
            value, closed, rest = rest.partition(call["parameter_end"])
            key = key.strip()
            if not found or not closed or not key:
                return None
            value = value[1:] if value.startswith("\n") else value
            value = value[:-1] if value.endswith("\n") else value
            arguments[key] = self.typed(name, key, None, value)
            rest = rest.strip()
        return call_entry(name, arguments)

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
        return call_entry(name, arguments)

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
                if self.fresh:
                    head = self.buffer.lstrip()
                    if not head or self.reasoning_start.startswith(head):
                        break
                    self.fresh = False
                    if head.startswith(self.reasoning_start):
                        self.buffer = head[len(self.reasoning_start):]
                        self.state = "reasoning"
                        continue
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


class ChannelParser(OutputParser):
    def __init__(self, model, tools, prompt, stops):
        self.channel = model.channels
        self.call = model.tool_calls
        self.types = tool_parameter_types(tools)
        self.stops = [stop for stop in stops if stop] + model.end_markers
        self.end_markers = model.end_markers
        self.drops = model.content_markers
        self.ended = False
        self.state = "header" if prompt.endswith(self.channel["message_start"]) else "between"
        self.target = "tool"
        self.buffer = ""
        self.calls = 0
        self.stopped = False

    def emit(self, events, text):
        if self.target == "reasoning":
            events.append(("reasoning", text))
        elif self.target == "content" or text.strip():
            events.append(("content", text))

    def recipient(self, header):
        prefix = self.channel["recipient_prefix"]
        if prefix not in header:
            return "content"
        name = header.split(prefix, 1)[1].split()
        name = name[0] if name else ""
        return "reasoning" if name == self.channel["reasoning_recipient"] else "content" if name == self.channel["content_recipient"] else "tool"

    def feed(self, text):
        self.buffer += text
        events = []
        start, end = self.channel["message_start"], self.channel["message_end"]
        while self.buffer and not self.stopped:
            if self.state == "header":
                index = self.buffer.find(self.channel["header_end"])
                if index < 0:
                    break
                self.target = self.recipient(self.buffer[:index])
                self.buffer = self.buffer[index + len(self.channel["header_end"]):]
                self.state = "body"
            elif self.state == "call":
                index = self.buffer.find(self.call["end"])
                if index < 0:
                    break
                body = self.buffer[:index]
                self.buffer = self.buffer[index + len(self.call["end"]):]
                self.state = "body"
                parsed = self.parse_call(body)
                if parsed is None:
                    events.append(("content", self.call["start"] + body + self.call["end"]))
                else:
                    events.append(("tool_call", parsed))
                    self.calls += 1
            else:
                markers = [start] if self.state == "between" else [end, start]
                if self.state == "body" and self.target != "reasoning":
                    markers += [self.call["start"]] + self.stops + self.drops
                hits = [hit for hit in ((self.buffer.find(marker), marker) for marker in markers) if hit[0] >= 0]
                if not hits:
                    keep = longest_marker_prefix(self.buffer, markers)
                    if len(self.buffer) > keep:
                        self.emit(events, self.buffer[:len(self.buffer) - keep])
                        self.buffer = self.buffer[len(self.buffer) - keep:]
                    break
                index, marker = min(hits)
                if index:
                    self.emit(events, self.buffer[:index])
                self.buffer = self.buffer[index + len(marker):]
                if marker == start:
                    self.state = "header"
                elif marker == end:
                    self.state, self.target = "between", "tool"
                elif marker == self.call["start"]:
                    self.state = "call"
                elif marker not in self.drops:
                    self.buffer = ""
                    self.stopped = True
                    self.ended = marker in self.end_markers
        return events

    def finish(self):
        events = []
        if self.buffer and not self.stopped:
            if self.state == "call":
                events.append(("content", self.call["start"] + self.buffer))
            elif self.state != "header":
                self.emit(events, self.buffer)
        self.buffer = ""
        return events


class Frontend:
    memory = None

    def __init__(self, config, base):
        self.models = {}
        for spec in config["models"]:
            model = Model(spec, base)
            self.models[model.id] = model
        self.control = config.get("control")
        if self.control is not None and set(self.control) not in ({"status_url"}, {"status_url", "force_url", "admin_url"}):
            raise SystemExit("chat_frontend: control needs status_url, and optionally both force_url and admin_url")
        controlled = self.control is not None or any(model.gate.activation_url is not None for model in self.models.values())
        if controlled != ("controller_token_file" in config):
            raise SystemExit("chat_frontend: controller_token_file is needed exactly when the config has control or an activation_url")
        self.controller_headers = {"Authorization": "Bearer " + read_token(base / config["controller_token_file"])} if controlled else {}
        self.site = (base / config["site_dir"]).resolve() if config.get("site_dir") else None
        self.session = None
        self.memory = None
        if config.get("chat_memory") is not None:
            loader = importlib.util.spec_from_file_location("sparkpipe_chat_memory", Path(__file__).with_name("chat_memory.py"))
            module = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(module)
            self.memory = module.Memory(config["chat_memory"], self)

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
            async with self.session.get(self.control["status_url"], headers=self.controller_headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                payload = await response.json(content_type=None)
                if isinstance(payload, dict):
                    payload["waiting"] = {model.id: model.gate.state()["waiting"] for model in self.models.values() if model.gate.state()["waiting"]}
                return web.json_response(payload, status=response.status)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as error:
            return web.json_response(error_body("control_unavailable", str(error)), status=503)

    async def control_switch_now(self, request):
        if self.control is None or "force_url" not in self.control:
            return web.json_response(error_body("not_found", "this front end has no Switch now"), status=404)
        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Bearer ") or not authorization[len("Bearer "):].strip():
            return web.json_response(error_body("unauthorized", "Switch now needs the admin's SparkPipe Chat login"), status=401)
        try:
            async with self.session.get(self.control["admin_url"], headers={"Authorization": authorization}, timeout=aiohttp.ClientTimeout(total=10)) as response:
                user = await response.json(content_type=None) if response.status == 200 else None
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as error:
            return web.json_response(error_body("control_unavailable", f"could not check the login: {error}"), status=503)
        if not isinstance(user, dict) or user.get("role") != "admin":
            return web.json_response(error_body("forbidden", "only the admin can use Switch now"), status=403)
        body = {}
        if request.can_read_body:
            try:
                body = await request.json()
            except json.JSONDecodeError:
                return web.json_response(error_body("invalid_json", "the request body is not JSON"), status=400)
        model = body.get("model") if isinstance(body, dict) else None
        if model is not None and model not in self.models:
            return web.json_response(error_body("model_not_found", f"model {model!r} is not served here; served: {sorted(self.models)}"), status=404)
        try:
            async with self.session.post(self.control["force_url"], json={"model": model} if model else {}, headers=self.controller_headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
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
        plan = self.plan(model, body, None)
        return plan["engine_body"], plan["parser"], plan["prompt_tokens"]

    def plan(self, model, body, headers):
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
        tools = body.get("tools")
        if tools is not None and (not isinstance(tools, list) or any(
                not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict) or
                not isinstance(tool["function"].get("name"), str) for tool in tools)):
            raise RequestError(400, "invalid_tools", "tools must be an array of function tools, each with a function object that has a string name")
        tool_choice = body.get("tool_choice")
        if tool_choice not in (None, "auto", "none"):
            raise RequestError(400, "unsupported_parameter", f"tool_choice {tool_choice!r} is not supported by this server; send auto or none")
        if tool_choice == "none":
            tools = None
        if body.get("stream") not in (None, True, False):
            raise RequestError(400, "invalid_request", "stream must be a boolean")
        kwargs = {}
        effort = body.get("reasoning_effort")
        if effort is not None:
            if not isinstance(effort, str) or effort not in model.reasoning_effort:
                raise RequestError(400, "unsupported_parameter", f"reasoning_effort must be one of {sorted(model.reasoning_effort)}")
            kwargs.update(model.reasoning_effort[effort])
        template_kwargs = body.get("chat_template_kwargs")
        if template_kwargs is not None and not isinstance(template_kwargs, dict):
            raise RequestError(400, "invalid_request", "chat_template_kwargs must be an object")
        for key, value in (template_kwargs or {}).items():
            if key not in model.template_kwargs:
                raise RequestError(400, "unsupported_parameter", f"chat_template_kwargs.{key} is not accepted; accepted: {sorted(model.template_kwargs)}")
            kwargs[key] = value
        session = None
        if self.memory is not None and headers is not None:
            session = self.memory.session(model, headers, messages, tools, kwargs)
            if session is not None:
                messages, tools = normalize_messages(session.messages), session.tools
        return self.finish_plan(model, body, messages, tools, kwargs, session)

    def finish_plan(self, model, body, messages, tools, kwargs, session):
        prompt, prompt_ids = model.render(messages, tools, kwargs)
        if len(prompt_ids) >= model.context_tokens:
            raise RequestError(400, "context_length_exceeded", f"the prompt has {len(prompt_ids)} tokens and {model.id} serves {model.context_tokens}")
        limit = number(body, "max_completion_tokens", 1, 2 ** 31, integer=True)
        if limit is None:
            limit = number(body, "max_tokens", 1, 2 ** 31, integer=True)
        if limit is not None and limit > model.max_output_tokens:
            raise RequestError(400, "invalid_request", f"max_tokens {limit} is above {model.id}'s output limit of {model.max_output_tokens} tokens")
        room = model.context_tokens - len(prompt_ids)
        max_tokens = min(model.max_output_tokens if limit is None else limit, room)
        if max_tokens <= 0:
            raise RequestError(400, "invalid_request", "max_tokens must be positive")
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        if not isinstance(stops, list) or not all(isinstance(stop, str) and stop for stop in stops):
            raise RequestError(400, "invalid_request", "stop must be a string or an array of non-empty strings")
        if model.channels is not None:
            parser = ChannelParser(model, tools, prompt, stops)
        else:
            parser = OutputParser(model, tools, prompt.rstrip().endswith(model.reasoning["start"].rstrip()), stops)
        engine_body = {"prompt_token_ids": prompt_ids, "max_tokens": max_tokens, "stream": True, "stop_token_ids": model.stop_token_ids}
        engine_body.update(sampling_fields(body, model.sampling_defaults))
        if self.memory is not None:
            engine_body["priority"] = 1
        return {"body": body, "engine_body": engine_body, "parser": parser, "prompt_tokens": len(prompt_ids), "messages": messages,
                "tools": tools, "kwargs": kwargs, "session": session}

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
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        raise RequestError(502, "engine_error", f"the {model.id} engine sent a stream event that is not JSON: {data[:120]}")
                    yield event
                raise RequestError(502, "engine_error", f"the {model.id} engine stream ended before [DONE]; the reply is incomplete")
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
            if self.memory is not None:
                self.memory.cancel_all()
            await model.gate.wait_ready(self.session, deadline - now, self.controller_headers)

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
        token_ids = []
        stop_string = False
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
                token_ids.append(token)
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
                stop_string = True
                break
        for event in parser.finish():
            yield event
        yield ("finish", {"finish_reason": "tool_calls" if parser.calls else (finish or "stop"), "usage": usage,
                          "engine_finish": "stop_string" if stop_string else (finish or "stop"), "token_ids": token_ids})

    async def respond(self, model, plan):
        session = plan["session"]
        if session is None:
            async for event in self.generate(model, plan["engine_body"], plan["parser"]):
                yield event
            return
        completion = 0
        while True:
            internal, visible, content, reasoning = [], [], [], []
            result = {}
            async for kind, value in self.generate(model, plan["engine_body"], plan["parser"]):
                if kind == "tool_call" and session.claims(value):
                    internal.append(value)
                    continue
                if kind == "finish":
                    result = value
                    continue
                if kind == "tool_call":
                    visible.append(value)
                elif kind == "content":
                    content.append(value)
                elif kind == "reasoning":
                    reasoning.append(value)
                yield kind, value
            usage = dict(result.get("usage") or {})
            completion += usage.get("completion_tokens", 0)
            if internal and not visible and session.can_continue():
                messages = session.continue_with("".join(content), "".join(reasoning), internal)
                plan = self.finish_plan(model, plan["body"], normalize_messages(messages), plan["tools"], plan["kwargs"], session)
                continue
            if usage:
                usage["completion_tokens"] = completion
            session.finished(plan["messages"], "".join(content), "".join(reasoning), visible, result.get("engine_finish"),
                             plan["engine_body"]["prompt_token_ids"], result.get("token_ids") or [])
            reason = "tool_calls" if visible else ("length" if result.get("engine_finish") == "length" else "stop")
            yield "finish", dict(result, usage=usage or result.get("usage"), finish_reason=reason)
            return

    async def memory_model_ready(self, model):
        if self.control is None:
            return True
        try:
            async with self.session.get(self.control["status_url"], headers=self.controller_headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                status = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
            return False
        return (isinstance(status, dict) and status.get("active") == model.id and status.get("state") == "idle"
                and not status.get("draining") and status.get("wanted") in (None, model.id))

    def memory_curator_ids(self, model, messages, tools, kwargs, suffix):
        prompt, ids = model.render(normalize_messages(messages), tools, kwargs)
        reasoning_open = model.reasoning is not None and prompt.rstrip().endswith(model.reasoning["start"].rstrip())
        if suffix and reasoning_open:
            ids = ids + model.vocabulary.encode(suffix, True)
            reasoning_open = False
        return ids, reasoning_open

    def memory_template_parts(self, model, messages, tools, kwargs, markers):
        context = dict(model.template_defaults)
        context.update(kwargs)
        messages = normalize_messages(messages)
        if isinstance(model.renderer, SegmentRenderer):
            pieces = [(segment.text, segment.allow_special) for segment in model.renderer.function(messages, tools or None, add_generation_prompt=True, **context)]
        else:
            pieces = [(model.renderer.render_text(messages, tools, dict(model.renderer.template_tokens, **context)), True)]
        parts, ids, texts, found = [], [], [], 0
        for text, special in pieces:
            while found < len(markers) and markers[found] in text:
                cut = text.index(markers[found])
                ids.extend(model.vocabulary.encode(text[:cut], special))
                texts.append(text[:cut])
                parts.append((ids, "".join(texts)))
                ids, texts = [], []
                text = text[cut + len(markers[found]):]
                found += 1
            ids.extend(model.vocabulary.encode(text, special))
            texts.append(text)
        parts.append((ids, "".join(texts)))
        return parts if found == len(markers) else None

    def memory_continuation_ids(self, model, messages, tools, kwargs, prompt_ids, generated_ids, instruction, suffix):
        reply, ask = "\ue000SPARKPIPE-REPLY\ue001", "\ue000SPARKPIPE-ASK\ue001"
        parts = self.memory_template_parts(model, list(messages) + [{"role": "assistant", "content": reply}, {"role": "user", "content": ask}], tools, kwargs, [reply, ask])
        if parts is None:
            return None
        generated = list(generated_ids)
        while generated and generated[-1] in model.stop_token_ids:
            generated.pop()
        between, after = parts[1][0], parts[2][0]
        overlap = next((size for size in range(min(len(between), len(generated)), 0, -1) if generated[-size:] == between[:size]), 0)
        reasoning_open = model.reasoning is not None and parts[2][1].rstrip().endswith(model.reasoning["start"].rstrip())
        ids = list(prompt_ids) + generated + between[overlap:] + model.vocabulary.encode(instruction, False) + after
        if suffix and reasoning_open:
            ids += model.vocabulary.encode(suffix, True)
            reasoning_open = False
        return ids, reasoning_open

    def memory_cut_ids(self, model, messages, tools, kwargs, sentinel):
        parts = self.memory_template_parts(model, messages, tools, kwargs, [sentinel])
        return parts[0][0] if parts else None

    async def memory_complete(self, model, ids, max_tokens, reasoning_open=False):
        body = {"prompt_token_ids": ids, "max_tokens": max_tokens, "stream": True, "stop_token_ids": model.stop_token_ids, "priority": 0, "temperature": 0.0}
        parser = OutputParser(model, None, reasoning_open, []) if model.channels is None else None
        step = model.vocabulary.decoder()
        content = []
        async for chunk in self.engine_events(model, body):
            text = "".join(piece for piece in (step(token) for token in chunk.get("tokens", []) if token not in model.stop_token_ids) if piece)
            if parser is None:
                content.append(text)
                continue
            content.extend(value for kind, value in parser.feed(text) if kind == "content")
        if parser is not None:
            content.extend(value for kind, value in parser.finish() if kind == "content")
        return "".join(content)

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
            plan = self.plan(model, body, request.headers)
        except RequestError as error:
            return error_response(error)
        if self.memory is not None:
            self.memory.enter(model)
        try:
            return await self.answer(request, model, body, plan, started)
        finally:
            if self.memory is not None:
                self.memory.leave(model)

    async def answer(self, request, model, body, plan, started):
        engine_body, prompt_tokens = plan["engine_body"], plan["prompt_tokens"]
        try:
            if body.get("stream"):
                return await self.chat_stream(request, model, body, plan, started)
            content, reasoning, calls, logprobs = [], [], [], []
            result = {}
            first = None
            async for kind, value in self.respond(model, plan):
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

    async def chat_stream(self, request, model, body, plan, started):
        prompt_tokens = plan["prompt_tokens"]
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
            async for kind, value in self.respond(model, plan):
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


def build(frontend):
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(frontend.start)
    app.on_cleanup.append(frontend.stop)
    app.router.add_get("/health", frontend.health)
    app.router.add_get("/health/liveliness", frontend.liveliness)
    app.router.add_get("/v1/models", frontend.list_models)
    app.router.add_post("/v1/chat/completions", frontend.chat)
    app.router.add_get("/v1/sparkpipe/status", frontend.control_status)
    app.router.add_post("/v1/sparkpipe/switch-now", frontend.control_switch_now)
    if frontend.site is not None:
        app.router.add_get("/", frontend.site_root)
        app.router.add_static("/site/", frontend.site)
    return app


def main():
    parser = argparse.ArgumentParser(description="OpenAI chat front end for SparkPipe engines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    try:
        loopback = ipaddress.ip_address(arguments.host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise SystemExit(f"chat_frontend: --host {arguments.host} is not a loopback address; the chat layer checks no API key, so it serves this host only and LiteLLM is the door")
    config_path = Path(arguments.config).resolve()
    frontend = Frontend(json.loads(config_path.read_text()), config_path.parent)
    web.run_app(build(frontend), host=arguments.host, port=arguments.port, access_log=None)


if __name__ == "__main__":
    main()

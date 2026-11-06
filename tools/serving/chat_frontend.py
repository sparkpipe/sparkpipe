import argparse
import asyncio
import codecs
import copy
import functools
import hashlib
import importlib.util
import ipaddress
import json
import os
import re
import resource
import sys
import time
import uuid
from array import array
from collections import Counter, OrderedDict
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
TOOL_CALL_RETRIES = 2


def read_token(path):
    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        raise SystemExit(f"chat_frontend: {path} has mode {mode:o}; the token file must be readable by its owner only")
    token = Path(path).read_text().strip()
    if len(token) < 32:
        raise SystemExit(f"chat_frontend: the token in {path} must be at least 32 characters")
    return token


class RequestError(Exception):
    def __init__(self, status, code, message, retry_after=None, details=None, overflow=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after
        self.details = details
        self.overflow = overflow


def load_sibling(name):
    loader = importlib.util.spec_from_file_location("sparkpipe_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


chat_replay = load_sibling("chat_replay")
chat_shapes = load_sibling("chat_shapes")
working_state = load_sibling("working_state")
chat_shapes.RequestError = RequestError
chat_shapes.chat_replay = chat_replay
KEEPALIVE_SECONDS = 10.0
BUSY_WAIT_SECONDS = 60.0
BUSY_RETRY_AFTER_SECONDS = 5
RESIDENCY_CACHE_SECONDS = 1.0
CACHE_BLOCK_TOKENS = 64
INTERACTIVE_RESERVE = 2
FAIR_SHARE = 0.75
PRIORITY_HEADER = "X-SparkPipe-Priority"
FLEET_HEADER = "X-SparkPipe-Fleet"
ACCOUNT_HEADER = "X-SparkPipe-Account"
MEMBER_HEADER = "X-SparkPipe-Key"
ALLOW_SWAP_HEADER = "X-SparkPipe-Allow-Swap"
ALLOW_SWAP_VALUES = frozenset({"1", "true"})
HEADER_VALUE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
FLEET_VALUE = re.compile(r"[A-Za-z0-9._-]{1,32}")
REQUEST_CLASS_HEADER = "x-claude-code-request-class"
BACKGROUND_CLASSES = frozenset({"compaction", "auxiliary"})
SUBAGENT_HEADER = "x-openai-subagent"
BACKGROUND_SUBAGENTS = frozenset({"compact"})
AFFINITY_HEADERS = ("X-Claude-Code-Session-Id", "x-session-affinity", "X-Session-Id")
AGENT_HEADERS = ("x-claude-code-agent-id", "x-parent-session-id")
DEFAULT_ALIASES = {"claude-*": None, "gpt-*": None}
ALIAS_TABLE_SIZE = 64
ALIAS_PATTERN_LENGTH = 128
ALIAS_STARS = 4
PACK_SEPARATOR = "+"
REPLAY_SENTINEL = "\ue000SPARKPIPE-REPLAY-{}\ue001"
REPLAY_MARK = re.compile("\ue000SPARKPIPE-REPLAY-([0-9]+)\ue001")
GENERATION_PROBE = "\ue000SPARKPIPE-PROBE\ue001"
GENERATION_CACHE = 64
DIRECTIVE_CHECK_SECONDS = 1.0
EXPIRY_SWEEP_SECONDS = 60.0
OWN_SCOPE = "own"


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def clean_header(value, pattern=HEADER_VALUE):
    return value if isinstance(value, str) and pattern.fullmatch(value) is not None else None


def harness_background(headers):
    return ((headers.get(REQUEST_CLASS_HEADER) or "").strip().lower() in BACKGROUND_CLASSES
            or (headers.get(SUBAGENT_HEADER) or "").strip().lower() in BACKGROUND_SUBAGENTS)


def glob_match(pattern, name):
    pieces = pattern.split("*")
    if len(pieces) == 1:
        return pattern == name
    head, tail = pieces[0], pieces[-1]
    if len(name) < len(head) + len(tail) or not name.startswith(head) or not name.endswith(tail):
        return False
    at, end = len(head), len(name) - len(tail)
    for piece in pieces[1:-1]:
        found = name.find(piece, at, end)
        if found < 0:
            return False
        at = found + len(piece)
    return True


def template_spans(pieces):
    spans, offset = [], 0
    for text, special in pieces:
        spans.append((offset, offset + len(text), special))
        offset += len(text)
    return spans


def find_in(spans, text, needle, begin, limit):
    for start, end, special in spans:
        if start >= limit:
            break
        if special and end > begin:
            found = text.find(needle, max(start, begin), min(end, limit))
            if found >= 0:
                return found
    return -1


def rfind_in(spans, text, needle, floor, before):
    for start, end, special in reversed(spans):
        if special and start < before:
            found = text.rfind(needle, max(start, floor), min(end, before))
            if found >= 0:
                return found
    return -1


def encode_span(model, pieces, begin, end):
    ids, offset = [], 0
    for text, special in pieces:
        low, high = max(begin, offset), min(end, offset + len(text))
        if low < high:
            ids.extend(model.renderer.cache.encode(text[low - offset:high - offset], special, model.context_tokens))
        offset += len(text)
    return ids


def message_handles(message):
    return list(message.get(chat_shapes.HANDLES) or []) + [call["id"] for call in message.get("tool_calls") or [] if isinstance(call, dict) and isinstance(call.get("id"), str)]


def without_handles(messages):
    return [{key: value for key, value in message.items() if key != chat_shapes.HANDLES} for message in messages]


EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EFFORT_DESCRIPTIONS = {"none": "No reasoning", "minimal": "Minimal reasoning", "low": "Fast responses with lighter reasoning",
                       "medium": "Balanced speed and reasoning depth", "high": "Deeper reasoning for complex problems",
                       "xhigh": "Extra reasoning depth", "max": "The most reasoning the model does"}
CODEX_TRUNCATION = {"mode": "tokens", "limit": 10000}


def effort_order(efforts):
    return sorted(efforts, key=lambda effort: (EFFORT_ORDER.index(effort) if effort in EFFORT_ORDER else len(EFFORT_ORDER), effort))


def codex_entry(model, index, resident, instructions):
    efforts = effort_order(model.reasoning_effort)
    default = "medium" if "medium" in efforts else (efforts[len(efforts) // 2] if efforts else None)
    return {"slug": model.id, "display_name": model.id, "description": "resident on this fleet" if resident else "offered on this fleet; loads on request",
            "default_reasoning_level": default,
            "supported_reasoning_levels": [{"effort": effort, "description": EFFORT_DESCRIPTIONS.get(effort, effort)} for effort in efforts],
            "shell_type": "unified_exec", "visibility": "list", "supported_in_api": True, "priority": index, "support_verbosity": False,
            "truncation_policy": dict(CODEX_TRUNCATION), "experimental_supported_tools": [], "apply_patch_tool_type": "freeform",
            "context_window": model.context_tokens, "max_context_window": model.context_tokens,
            "input_modalities": ["text", "image"] if model.images else ["text"], "base_instructions": instructions}


def replayed_tokens(plan):
    return (plan.get("replay") or {}).get("replayed", 0)


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def strftime_now(format_string):
    return time.strftime(format_string)


def error_body(code, message):
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


def error_headers(error):
    retry_after = error.retry_after if error.retry_after is not None else (5 if error.status == 503 else BUSY_RETRY_AFTER_SECONDS if error.status == 429 else None)
    headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
    if error.status == 429:
        headers["x-should-retry"] = "true"
    return headers or None


def error_response(error):
    body = error_body(error.code, error.message)
    if error.status >= 500:
        body["error"]["type"] = "server_error"
    body["error"].update(error.details or {})
    return web.json_response(body, status=error.status, headers=error_headers(error))


def anthropic_error_response(error):
    return web.json_response(chat_shapes.anthropic_error_body(error), status=error.status, headers=error_headers(error))


def responses_error_response(error):
    return web.json_response(chat_shapes.openai_error_body(error), status=error.status, headers=error_headers(error))


class Waiter:
    def __init__(self, priority, scope, member, order, future):
        self.priority, self.scope, self.member, self.order, self.future = priority, scope, member, order, future


class SequenceGate:
    def __init__(self, sequences):
        self.sequences = sequences
        self.used = 0
        self.held = Counter()
        self.held_members = Counter()
        self.waiters = []
        self.arrivals = 0

    def limit(self, priority):
        return self.sequences if priority >= INTERACTIVE_PRIORITY else max(1, self.sequences - INTERACTIVE_RESERVE)

    def share(self):
        return max(1, int(self.sequences * FAIR_SHARE))

    def capped(self, waiter, pool):
        if self.held[waiter.scope] >= self.share() and any(other.scope != waiter.scope for other in pool):
            return True
        member = (waiter.scope, waiter.member)
        return (waiter.member is not None and self.held_members[member] >= self.share() and
                any(other.scope == waiter.scope and other.member != waiter.member for other in pool))

    def grant(self):
        while True:
            ready = [waiter for waiter in self.waiters if self.used < self.limit(waiter.priority)]
            if not ready:
                return
            pool = [waiter for waiter in ready if waiter.priority >= INTERACTIVE_PRIORITY] or ready
            fair = [waiter for waiter in pool if not self.capped(waiter, pool)] or pool
            chosen = min(fair, key=lambda waiter: (self.held[waiter.scope], self.held_members[(waiter.scope, waiter.member)], waiter.order))
            self.waiters.remove(chosen)
            self.take(chosen.scope, chosen.member)
            chosen.future.set_result(True)

    def take(self, scope, member):
        self.used += 1
        self.held[scope] += 1
        self.held_members[(scope, member)] += 1

    async def acquire(self, priority, timeout, scope=OWN_SCOPE, member=None):
        self.arrivals += 1
        waiter = Waiter(priority, scope, member, self.arrivals, asyncio.get_running_loop().create_future())
        self.waiters.append(waiter)
        self.grant()
        try:
            await asyncio.wait_for(asyncio.shield(waiter.future), timeout)
        except asyncio.TimeoutError:
            if waiter.future.done():
                return
            self.waiters.remove(waiter)
            self.grant()
            raise RequestError(429, "busy", f"all {self.sequences} sequences were in use for {timeout:g} s", retry_after=BUSY_RETRY_AFTER_SECONDS)
        except asyncio.CancelledError:
            if waiter.future.done():
                self.free(scope, member)
            else:
                self.waiters.remove(waiter)
                waiter.future.cancel()
                self.grant()
            raise

    def free(self, scope, member):
        self.used -= 1
        for counter, name in ((self.held, scope), (self.held_members, (scope, member))):
            counter[name] -= 1
            if counter[name] <= 0:
                del counter[name]
        self.grant()

    async def release(self, scope=OWN_SCOPE, member=None):
        self.free(scope, member)

    def state(self):
        return {"sequences": self.sequences, "in_use": self.used, "waiting_interactive": sum(waiter.priority >= INTERACTIVE_PRIORITY for waiter in self.waiters),
                "waiting": len(self.waiters), "scopes": len(self.held)}


async def heartbeat(events, interval):
    iterator = events.__aiter__()
    pending = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield ("keepalive", None)
                continue
            task, pending = pending, None
            try:
                item = task.result()
            except StopAsyncIteration:
                return
            yield item
    finally:
        if pending is not None:
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, StopAsyncIteration, Exception):
                pass
        await iterator.aclose()


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
        self.spacing = {token["content"]: token["id"] for token in document["added_tokens"]
                        if not token["special"] and token["content"] and not token["content"].strip() and
                        not (token.get("lstrip") or token.get("rstrip") or token.get("single_word"))}
        self.spacing_pattern = re.compile("(" + "|".join(re.escape(text) for text in sorted(self.spacing, key=len, reverse=True)) + ")") if self.spacing else None
        document["added_tokens"] = []
        self.plain = Tokenizer.from_str(json.dumps(document))

    def encode(self, text, special):
        if special:
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        if self.spacing_pattern is None:
            return self.plain.encode(text, add_special_tokens=False).ids
        ids = []
        for index, piece in enumerate(self.spacing_pattern.split(text)):
            if index % 2 == 1:
                ids.append(self.spacing[piece])
            elif piece:
                ids.extend(self.plain.encode(piece, add_special_tokens=False).ids)
        return ids

    def control_strings(self):
        return [token.content for token in self.tokenizer.get_added_tokens_decoder().values() if token.content not in self.spacing]

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


CARRIED = re.compile("\ue000([0-9]+)\ue001")
PIECE_CACHE_BYTES = 512 << 20
TEMPLATE_PIECE_CACHE_BYTES = 0
INTERACTIVE_PRIORITY = 2
BULK_PRIORITY = 1
BACKGROUND_PRIORITY = 0
BULK_SUFFIX = "-bulk"


def context_target(body):
    value = body.get("num_ctx")
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value.strip())
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if value is None or value in (0, -1):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RequestError(400, "invalid_request", "num_ctx must be a positive whole number: the context chat memory keeps the prompt near")
    return value


def request_priority(model, body):
    return BULK_PRIORITY if body.get("model") == model.id + BULK_SUFFIX else INTERACTIVE_PRIORITY


@functools.lru_cache(maxsize=None)
def literal_pattern(strings):
    trie = {}
    for string in strings:
        node = trie
        for char in string:
            node = node.setdefault(char, {})
        node[None] = None

    def emit(node):
        if None in node:
            return ""
        branches = [re.escape(char) + emit(child) for char, child in sorted(node.items())]
        return branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"

    return re.compile(emit(trie)) if trie else None


def control_pattern(vocabulary):
    return literal_pattern(tuple(sorted({token for token in vocabulary.control_strings() if token})))


class PieceCache:
    def __init__(self, vocabulary, budget):
        self.vocabulary = vocabulary
        self.budget = budget
        self.entries = OrderedDict()
        self.bytes = 0

    def encode(self, piece, special, room):
        key = (piece, special)
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
            return entry[0]
        ids = array("I", self.vocabulary.encode(piece, special))
        size = sys.getsizeof(piece) + ids.itemsize * len(ids) + 128
        if size <= self.budget and len(ids) <= room:
            self.entries[key] = (ids, size)
            self.bytes += size
            while self.bytes > self.budget:
                self.bytes -= self.entries.popitem(last=False)[1][1]
        return ids


def carry_controls(messages, tools, vocabulary):
    pattern = control_pattern(vocabulary)
    carried = []

    def holds_control(text):
        return pattern is not None and pattern.search(text) is not None

    def mark(value):
        if isinstance(value, str):
            if holds_control(value):
                carried.append(value)
                return f"\ue000{len(carried) - 1}\ue001"
            return value
        if isinstance(value, dict):
            if any(isinstance(key, str) and holds_control(key) for key in value):
                raise RequestError(400, "invalid_messages", "a request key carries a control token string")
            return {key: mark(item) for key, item in value.items()}
        if isinstance(value, list):
            return [mark(item) for item in value]
        return value

    return mark(messages), mark(tools), carried


def uncarry(pieces, carried, expected):
    result = []
    for text, special in pieces:
        for index, piece in enumerate(CARRIED.split(text)):
            if index % 2 == 0:
                if piece:
                    result.append((piece, special))
            elif int(piece) < len(carried):
                result.append((carried[int(piece)], False))
            else:
                raise RequestError(400, "invalid_messages", "a message carries a control token string inside text the chat template rewrites")
    if "".join(text for text, _ in result) != expected:
        raise RequestError(400, "invalid_messages", "a message carries a control token string inside text the chat template rewrites")
    return result


def encode_pieces(cache, text, pieces, room):
    ids = []
    for piece, special in pieces:
        ids.extend(cache.encode(piece, special, room - len(ids)))
    return text, ids


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
        self.cache = PieceCache(vocabulary, TEMPLATE_PIECE_CACHE_BYTES)

    def render_text(self, messages, tools, context, generation=True):
        try:
            return self.template.render(messages=messages, tools=tools or None, add_generation_prompt=generation, **context)
        except jinja2.exceptions.TemplateError as error:
            raise RequestError(400, "invalid_messages", f"the chat template refused the request: {error}")
        except Exception as error:
            raise RequestError(400, "invalid_messages", f"the chat template could not render the request: {type(error).__name__}: {error}")

    def pieces(self, messages, tools, kwargs):
        context = dict(self.template_tokens)
        context.update(kwargs)
        text = self.render_text(messages, tools, context)
        marked_messages, marked_tools, carried = carry_controls(messages, tools, self.vocabulary)
        if not carried:
            return text, [(text, True)]
        return text, uncarry([(self.render_text(marked_messages, marked_tools, context), True)], carried, text)

    def generation_text(self, messages, tools, kwargs):
        context = dict(self.template_tokens)
        context.update(kwargs)
        return self.render_text(messages, tools, context, False), self.render_text(messages, tools, context)

    def render(self, messages, tools, kwargs, room):
        return encode_pieces(self.cache, *self.pieces(messages, tools, kwargs), room)


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
        self.cache = PieceCache(vocabulary, PIECE_CACHE_BYTES)

    def segments(self, messages, tools, kwargs, generation=True):
        try:
            return self.function(messages, tools or None, add_generation_prompt=generation, **kwargs)
        except Exception as error:
            raise RequestError(400, "invalid_messages", f"the chat renderer refused the request: {type(error).__name__}: {error}")

    def pieces(self, messages, tools, kwargs):
        segments = self.segments(messages, tools, kwargs)
        text = "".join(segment.text for segment in segments)
        marked_messages, marked_tools, carried = carry_controls(messages, tools, self.vocabulary)
        client = {token for token in self.vocabulary.control_strings() if token and any(token in value for value in carried)}
        if not any(segment.allow_special and any(token in segment.text for token in client) for segment in segments):
            return text, [(segment.text, segment.allow_special) for segment in segments]
        marked = self.segments(marked_messages, marked_tools, kwargs)
        return text, uncarry([(segment.text, segment.allow_special) for segment in marked], carried, text)

    def generation_text(self, messages, tools, kwargs):
        return tuple("".join(segment.text for segment in self.segments(messages, tools, kwargs, generation)) for generation in (False, True))

    def render(self, messages, tools, kwargs, room):
        return encode_pieces(self.cache, *self.pieces(messages, tools, kwargs), room)


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
        if self.tool_calls["format"] not in ("key_value", "tagged", "function_xml", "object", "json"):
            raise SystemExit(f"chat_frontend: {self.id} tool_calls format must be key_value, tagged, function_xml, object or json")
        if self.tool_calls["format"] == "object" and not all(isinstance(self.tool_calls.get(key), str) and self.tool_calls[key] for key in ("name_prefix", "string_delimiter")):
            raise SystemExit(f"chat_frontend: {self.id} object tool calls need name_prefix and string_delimiter")
        self.content_markers = list(spec.get("content_markers", []))
        self.end_markers = list(spec.get("end_markers", []))
        self.stop_token_ids = [self.token_id(text) for text in spec["stop_tokens"]]
        self.stop_strings = list(spec["stop_tokens"])
        self.openings = OrderedDict()
        self.gate = WarmGate(self.id, self.engine, spec.get("warm_queue"))
        sequences = spec.get("sequences")
        if isinstance(sequences, bool) or not isinstance(sequences, int) or sequences < 1:
            raise SystemExit(f"chat_frontend: {self.id} needs sequences, the engine's sequences at once (deployment/lanes.json)")
        self.sequences = SequenceGate(sequences)
        self.images = spec.get("images", False) is True
        self.memory = spec.get("chat_memory", {})
        if not isinstance(self.memory, dict):
            raise SystemExit(f"chat_frontend: {self.id} chat_memory must be an object of budget overrides")

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
        return self.renderer.render(messages, tools, context, self.context_tokens - 1)

    def generation_prompt(self, tools, kwargs):
        key = hashlib.sha256(chat_replay.canonical([tools or [], kwargs]).encode()).digest()
        if key in self.openings:
            self.openings.move_to_end(key)
            return self.openings[key]
        context = dict(self.template_defaults)
        context.update(kwargs)
        try:
            bare, full = self.renderer.generation_text([{"role": "user", "content": GENERATION_PROBE}], tools, context)
            opening = full[len(bare):] if full.startswith(bare) and len(full) > len(bare) else None
        except RequestError:
            opening = None
        self.openings[key] = opening
        while len(self.openings) > GENERATION_CACHE:
            self.openings.popitem(last=False)
        return opening


SOURCE_TAG = re.compile(r'<source id="([0-9]+)"((?: [a-z][a-z-]*="[^"]*")*)>')
SOURCE_ATTRIBUTE = re.compile(r' ([a-z][a-z-]*)="([^"]*)"')
CITATION_WRAPPER_KEYS = {"head_sha256", "open", "close", "separator"}


def parse_citations(text, wrapper):
    opening = text.find(wrapper["open"])
    if opening < 0 or hashlib.sha256(text[:opening].encode()).hexdigest() != wrapper["head_sha256"]:
        return None
    close = wrapper["close"] + wrapper["separator"]
    position, sources = opening + len(wrapper["open"]), []
    while True:
        tag = SOURCE_TAG.match(text, position)
        if tag is None:
            return None
        cursor = tag.end()
        while True:
            end = text.find("</source>", cursor)
            if end < 0:
                return None
            after = end + len("</source>")
            if text.startswith("\n<source id=", after) or text.startswith(close, after):
                break
            cursor = after
        sources.append((tag.group(1), tag.group(2), text[tag.end():end]))
        if text.startswith(close, after):
            base = text[after + len(close):]
            break
        position = after + 1
    head = text[:opening]
    rebuilt = head + wrapper["open"] + "\n".join(f'<source id="{n}"{attributes}>{body}</source>' for n, attributes, body in sources) + close + base
    return (head, sources, base) if rebuilt == text else None


def call_strings(arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return set()
    found = set()
    stack = [arguments]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return found


def canonical_citations(messages, wrapper):
    users = [index for index, message in enumerate(messages) if message.get("role") == "user"][-2:]
    for index in reversed(users):
        content = messages[index].get("content")
        texts = [content] if isinstance(content, str) else [part.get("text", "") for part in content or [] if isinstance(part, dict) and part.get("type") == "text"]
        parsed = [parse_citations(text, wrapper) for text in texts]
        if not parsed or any(item is None for item in parsed):
            continue
        sources = parsed[0][1]
        files = [source for source in sources if source[2]]
        markers = {}
        for n, attributes, body in sources:
            if not body:
                values = dict(SOURCE_ATTRIBUTE.findall(attributes))
                for key in ("resource-id", "name"):
                    if values.get(key):
                        markers.setdefault(values[key], n)

        def canonical(item):
            head, _, base = item
            if not files:
                return base
            return head + wrapper["open"] + "\n".join(f'<source id="{n}"{attributes}>{body}</source>' for n, attributes, body in files) + wrapper["close"] + wrapper["separator"] + base

        result = list(messages)
        if isinstance(content, str):
            result[index] = dict(messages[index], content=canonical(parsed[0]))
        else:
            replaced = iter(canonical(item) for item in parsed)
            result[index] = dict(messages[index], content=[dict(part, text=next(replaced)) if isinstance(part, dict) and part.get("type") == "text" else part for part in content])
        urls, labelled, used = {}, 0, set()
        for message in result[index + 1:]:
            for call in message.get("tool_calls") or []:
                urls[call.get("id")] = call_strings((call.get("function") or {}).get("arguments"))
        for position in range(index + 1, len(result)):
            message = result[position]
            if message.get("role") != "tool" or not isinstance(message.get("content"), str):
                continue
            ids = sorted({n for url, n in markers.items() if url in urls.get(message.get("tool_call_id"), set())}, key=int)
            if ids:
                used.update(ids)
                label = ", ".join(f"[{n}]" for n in ids)
                note = f"[This result is source {label}; cite it inline as {label}.]\n"
                result[position] = dict(message, content=note + message["content"], citation_label=note)
                labelled += 1
        return result, {"citation_markers": len(set(markers.values())), "citation_labelled": labelled,
                        "citation_unmatched": len(set(markers.values()) - used), "citation_files": len(files)}
    return messages, {}


PUBLIC_PATHS = frozenset({"/health/liveliness", "/v1/models", "/v1/chat/completions", "/v1/sparkpipe/status", "/v1/messages", "/v1/messages/count_tokens",
                          "/v1/responses", "/v1/tokenize", "/v1/detokenize", "/v1/models/aliases", "/v1/codex/models", "/v1/cache", "/api/hello"})
OPEN_PATHS = frozenset({"/health/liveliness", "/api/hello"})
ANTHROPIC_PATHS = frozenset({"/v1/messages", "/v1/messages/count_tokens"})
PUBLIC_KEY_PREFIX = "sk-sp-"
PUBLIC_BODY_BYTES_PER_TOKEN = 16
PUBLIC_BODY_OVERHEAD_BYTES = 1 << 20
PUBLIC_BODY_MIN_BYTES = 32 << 20
PUBLIC_KEY_PREFIX_BLOCKS = 1 << 16
STATUS_VIEW_FIELDS = ("state", "phase", "swap_elapsed_s", "phase_elapsed_s", "force_requested")
DRAINING_VIEW_FIELDS = ("live_requests", "waited_s", "limit_s", "quiet_s", "quiet_needed_s")
HISTORY_VIEW_FIELDS = ("finished", "seconds", "forced", "ok")


class PublicKeys:
    def __init__(self, path):
        self.path = path
        self.stamp = None
        self.entries = {}

    def lookup(self, key):
        if not key.startswith(PUBLIC_KEY_PREFIX):
            return None
        try:
            stamp = self.path.stat().st_mtime_ns
        except OSError:
            return None
        if stamp != self.stamp:
            try:
                document = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                return None
            self.entries = {entry["sha256"]: entry for entry in document.get("keys", []) if isinstance(entry, dict) and isinstance(entry.get("sha256"), str)}
            self.stamp = stamp
        entry = self.entries.get(hashlib.sha256(key.encode()).hexdigest())
        if entry is None or entry.get("revoked") or (entry.get("expires") is not None and entry["expires"] <= time.time()):
            return None
        return entry


def short_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()[:12] if value else None


def request_trace(headers, messages):
    headers = headers or {}
    names, results, rounds = {}, [], 0
    if len(messages) > 1 and messages[-2].get("role") == "tool" and isinstance(messages[-1].get("content"), list) and messages[-1].get("role") == "user":
        messages = messages[:-1]
    for message in messages:
        for call in message.get("tool_calls") or []:
            names[call.get("id")] = (call.get("function") or {}).get("name")
    for message in reversed(messages):
        role = message.get("role")
        if role == "tool" and rounds == 0:
            content = message.get("content")
            results.append([message.get("name") or names.get(message.get("tool_call_id")) or "tool",
                            len(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))])
        elif role == "assistant" and message.get("tool_calls"):
            rounds += 1
        elif role == "user":
            break
    trace = {"chat": short_hash(headers.get("X-OpenWebUI-Chat-Id")), "user_message": short_hash(headers.get("X-OpenWebUI-User-Message-Id")),
             "kind": "round" if results else "answer", "round": rounds}
    if results:
        trace["tool_chars"] = list(reversed(results))
    return trace


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


def tool_required_parameters(tools):
    required = {}
    for tool in tools or []:
        function = tool.get("function", tool) if isinstance(tool, dict) else {}
        listed = (function.get("parameters") or {}).get("required") or []
        required[function.get("name")] = [key for key in listed if isinstance(key, str)]
    return required


def call_entry(name, arguments):
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


class OutputParser:
    def __init__(self, model, tools, reasoning_open, stops):
        self.reasoning_start = model.reasoning["start"]
        self.reasoning_end = model.reasoning["end"]
        self.call = model.tool_calls
        self.types = tool_parameter_types(tools)
        self.required = tool_required_parameters(tools)
        self.stops = [stop for stop in stops if stop] + model.end_markers
        self.end_markers = model.end_markers
        self.drops = model.content_markers
        self.ended = False
        self.state = "reasoning" if reasoning_open else "content"
        self.fresh = not reasoning_open
        self.buffer = ""
        self.calls = 0
        self.stopped = False
        self.stop_marker = None
        self.fed = 0
        self.call_offset = None
        self.resume = None

    def call_opened(self, offset, state):
        self.call_offset = offset
        resume = copy.copy(self)
        resume.buffer = ""
        resume.state = state
        resume.fed = self.call_offset
        resume.resume = None
        self.resume = resume

    def bad_call(self, kind, text, repair=None):
        return ("bad_call", {"kind": kind, "text": text, "offset": self.call_offset, "resume": self.resume, "repair": repair})

    def unclosed_call(self):
        body = self.buffer[:len(self.buffer) - longest_marker_prefix(self.buffer, [self.call["end"]])]
        parsed = self.parse_call(body) if body.strip() else None
        if parsed is not None:
            name = parsed["function"]["name"]
            arguments = json.loads(parsed["function"]["arguments"])
            if name not in self.required or not isinstance(arguments, dict) or any(key not in arguments for key in self.required[name]):
                parsed = None
        return self.bad_call("unclosed", self.call["start"] + self.buffer, parsed)

    def parse_call(self, body):
        if self.call["format"] == "tagged":
            return self.parse_tagged_call(body)
        if self.call["format"] == "function_xml":
            return self.parse_function_xml_call(body)
        if self.call["format"] == "object":
            return self.parse_object_call(body)
        if self.call["format"] == "json":
            return self.parse_json_call(body)
        return self.parse_key_value_call(body)

    def parse_json_call(self, body):
        try:
            parsed = json.loads(body.strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or set(parsed) - {"name", "arguments"} or not isinstance(parsed.get("name"), str) or not parsed["name"]:
            return None
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        return call_entry(parsed["name"], arguments)

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
        self.fed += len(text)
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
                    self.stop_marker = marker
                    break
                self.call_opened(self.fed - len(self.buffer) + index, "content")
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
                    events.append(self.bad_call("unparsed", self.call["start"] + body + self.call["end"]))
                    self.stopped = True
                    break
                events.append(("tool_call", parsed))
                self.calls += 1
        return events

    def finish(self):
        events = []
        if not self.stopped and (self.buffer or self.state == "call"):
            if self.state == "reasoning":
                events.append(("reasoning", self.buffer))
            elif self.state == "call":
                events.append(self.unclosed_call())
            else:
                events.append(("content", self.buffer))
        self.buffer = ""
        return events


class ChannelParser(OutputParser):
    def __init__(self, model, tools, prompt, stops):
        self.channel = model.channels
        self.call = model.tool_calls
        self.types = tool_parameter_types(tools)
        self.required = tool_required_parameters(tools)
        self.stops = [stop for stop in stops if stop] + model.end_markers
        self.end_markers = model.end_markers
        self.drops = model.content_markers
        self.ended = False
        self.state = "header" if prompt.endswith(self.channel["message_start"]) else "between"
        self.target = "tool"
        self.buffer = ""
        self.calls = 0
        self.stopped = False
        self.stop_marker = None
        self.fed = 0
        self.call_offset = None
        self.resume = None

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
        self.fed += len(text)
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
                    events.append(self.bad_call("unparsed", self.call["start"] + body + self.call["end"]))
                    self.stopped = True
                    break
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
                opening = self.fed - len(self.buffer) + index
                self.buffer = self.buffer[index + len(marker):]
                if marker == start:
                    self.state = "header"
                elif marker == end:
                    self.state, self.target = "between", "tool"
                elif marker == self.call["start"]:
                    self.call_opened(opening, "body")
                    self.state = "call"
                elif marker not in self.drops:
                    self.buffer = ""
                    self.stopped = True
                    self.ended = marker in self.end_markers
                    self.stop_marker = marker
        return events

    def finish(self):
        events = []
        if not self.stopped and (self.buffer or self.state == "call"):
            if self.state == "call":
                events.append(self.unclosed_call())
            elif self.state != "header":
                self.emit(events, self.buffer)
        self.buffer = ""
        return events


class PrefixLedger:
    def __init__(self, block_tokens=64, capacity_blocks=1 << 21):
        self.block_tokens = block_tokens
        self.capacity_blocks = capacity_blocks
        self.blocks = OrderedDict()

    def chain(self, token_ids):
        data = array("I", token_ids).tobytes()
        width = 4 * self.block_tokens
        digest = b""
        for start in range(0, len(data) - width + 1, width):
            digest = hashlib.blake2b(digest + data[start:start + width], digest_size=16).digest()
            yield digest

    def seen(self, token_ids):
        matched, longest = 0, 0
        for digest in self.chain(token_ids):
            length = self.blocks.get(digest)
            if length is None:
                break
            matched += self.block_tokens
            longest = length
        return matched, longest

    def record(self, token_ids):
        total = len(token_ids)
        for digest in self.chain(token_ids):
            self.blocks[digest] = max(self.blocks.get(digest, 0), total)
            self.blocks.move_to_end(digest)
        while len(self.blocks) > self.capacity_blocks:
            self.blocks.popitem(last=False)


class Frontend:
    memory = None
    commands = None
    citation_wrapper = None
    prefixes = None
    control = None
    replay = None
    residency_cache = None
    residency_task = None
    control_error = None
    directive = None
    expiry_task = None
    codex_instructions = None

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
        wrapper = (config.get("client") or {}).get("citation_wrapper")
        if wrapper is not None and (not isinstance(wrapper, dict) or set(wrapper) != CITATION_WRAPPER_KEYS or not all(isinstance(wrapper[key], str) and wrapper[key] for key in CITATION_WRAPPER_KEYS)):
            raise SystemExit(f"chat_frontend: client.citation_wrapper needs exactly the non-empty strings {sorted(CITATION_WRAPPER_KEYS)}")
        self.citation_wrapper = wrapper
        public = config.get("public")
        self.public_header, self.public_keys, self.public_models = None, None, frozenset()
        if public is not None:
            if not isinstance(public, dict) or set(public) - {"header", "keys_file", "models"} or not isinstance(public.get("header"), str) or not public["header"] or not isinstance(public.get("keys_file"), str):
                raise SystemExit("chat_frontend: public needs a header and a keys_file, and optionally the public models")
            models = public.get("models", sorted(self.models))
            if not isinstance(models, list) or not set(models) <= set(self.models):
                raise SystemExit(f"chat_frontend: public models must be served models; served: {sorted(self.models)}")
            self.public_header, self.public_keys, self.public_models = public["header"], PublicKeys(base / public["keys_file"]), frozenset(models)
        self.public_prefixes = {}
        replay = config.get("replay") or {}
        if not isinstance(replay, dict) or set(replay) - {"directory", "budget_bytes", "directive"}:
            raise SystemExit("chat_frontend: replay takes directory (the replay store's working-state directory), budget_bytes and directive "
                             "(the working-state directive the controller writes on this host)")
        self.replay = chat_replay.ReplayStore(base / replay["directory"] if replay.get("directory") else None,
                                              int(replay.get("budget_bytes", chat_replay.DEFAULT_BUDGET_BYTES)))
        self.directive = base / replay["directive"] if replay.get("directive") else None
        codex = config.get("codex") or {}
        if not isinstance(codex, dict) or set(codex) - {"instructions_file"}:
            raise SystemExit("chat_frontend: codex takes instructions_file, the base instructions the Codex catalogue gives every model")
        self.codex_instructions = (base / codex["instructions_file"]).read_text() if codex.get("instructions_file") else None
        self.directive_state = None
        self.residency_cache = None
        self.residency_task = None
        self.control_error = None
        self.session = None
        self.memory = None
        self.tool_call_faults = {}
        if config.get("chat_memory") is not None:
            loader = importlib.util.spec_from_file_location("sparkpipe_chat_memory", Path(__file__).with_name("chat_memory.py"))
            module = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(module)
            self.memory = module.Memory(config["chat_memory"], self)
            self.memory_switch = module.memory_switch
            loader = importlib.util.spec_from_file_location("sparkpipe_chat_commands", Path(__file__).with_name("chat_commands.py"))
            commands = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(commands)
            self.commands = commands.Commands(self.memory, module, self)
            self.command_strip = commands.strip

    async def start(self, app):
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=None, sock_connect=10))
        self.prune_task = None
        if self.memory is not None and self.memory.chat_list:
            self.prune_task = asyncio.create_task(self.memory.prune_loop())
        self.expire_working_state(time.time(), True)
        self.expiry_task = asyncio.create_task(self.expiry_loop())

    async def stop(self, app):
        for task in (self.prune_task, self.expiry_task):
            if task is not None:
                task.cancel()
        await self.session.close()

    def expire_working_state(self, now, sweep):
        wiped, held_until = working_state.read_directive(self.directive) if self.directive is not None else (0, 0)
        changed = (wiped, held_until) != self.directive_state
        self.directive_state = (wiped, held_until)
        if not (changed or sweep):
            return 0
        try:
            dropped = self.replay.expire(lambda used: working_state.expired(used, wiped, held_until, now))
        except chat_replay.STORE_ERRORS as error:
            print(json.dumps({"event": "replay_expiry_failed", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr, flush=True)
            return 0
        if dropped:
            print(json.dumps({"event": "replay_expired", "dropped": dropped, "wiped": wiped, "held_until": held_until}), file=sys.stderr, flush=True)
        return dropped

    async def expiry_loop(self):
        swept = time.monotonic()
        while True:
            await asyncio.sleep(DIRECTIVE_CHECK_SECONDS)
            sweep = time.monotonic() - swept >= EXPIRY_SWEEP_SECONDS
            if sweep:
                swept = time.monotonic()
            self.expire_working_state(time.time(), sweep)

    def model_for(self, body):
        name = body.get("model")
        if name in self.models:
            return self.models[name]
        if isinstance(name, str) and name.endswith(BULK_SUFFIX) and name[:-len(BULK_SUFFIX)] in self.models:
            return self.models[name[:-len(BULK_SUFFIX)]]
        if len(self.models) == 1 and name is None:
            return next(iter(self.models.values()))
        raise RequestError(404, "model_not_found", f"model {name!r} is not served here; served: {sorted(self.models)}")

    def scope_of(self, key, headers):
        if key is None:
            return OWN_SCOPE
        if key.get("gateway") is not True:
            return "key:" + str(key.get("id"))
        account = headers.get(ACCOUNT_HEADER)
        if account is None:
            raise RequestError(400, "account_required", f"a gateway key must name the market account in {ACCOUNT_HEADER}")
        if not working_state.valid_account(account) or account == working_state.OWN_ACCOUNT:
            raise RequestError(400, "invalid_account", f"{ACCOUNT_HEADER} must be a market account id: 1 to 64 letters, digits, - or _")
        return "account:" + account

    @staticmethod
    def member_of(key, headers):
        if key is None:
            return None
        if key.get("gateway") is True:
            return clean_header(headers.get(MEMBER_HEADER))
        return key.get("id")

    @staticmethod
    def may_swap(key, headers):
        if key is None:
            return True
        if key.get("gateway") is True:
            return (headers.get(ALLOW_SWAP_HEADER) or "").strip().lower() in ALLOW_SWAP_VALUES
        return key.get("allow_swap") is True

    def control_failed(self, error):
        text = f"{type(error).__name__}: {error}"
        if text != self.control_error:
            print(json.dumps({"event": "control_unavailable", "error": text, "fallback": self.residency_cache is not None}), file=sys.stderr, flush=True)
        self.control_error = text

    async def read_residency(self):
        try:
            async with self.session.get(self.control["status_url"], headers=self.controller_headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                status = await response.json(content_type=None)
                if response.status != 200 or not isinstance(status, dict):
                    raise ValueError(f"the controller answered {response.status} with {type(status).__name__}")
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, ValueError) as error:
            self.control_failed(error)
            if self.residency_cache is None:
                raise RequestError(503, "control_unavailable", "the fleet controller is not answering; try again shortly")
            self.residency_cache = (time.monotonic(), self.residency_cache[1])
            return self.residency_cache[1]
        if self.control_error is not None:
            print(json.dumps({"event": "control_restored", "after": self.control_error}), file=sys.stderr, flush=True)
            self.control_error = None
        resident = [entry["id"] for entry in status.get("models") or [] if isinstance(entry, dict) and entry.get("active") is True and entry.get("id") in self.models]
        if status.get("active") in resident:
            resident.remove(status["active"])
            resident.insert(0, status["active"])
        view = {"resident": resident, "state": status.get("state"), "target": status.get("target")}
        self.residency_cache = (time.monotonic(), view)
        return view

    async def residency(self):
        if self.control is None:
            return None
        if self.residency_cache is not None and time.monotonic() - self.residency_cache[0] < RESIDENCY_CACHE_SECONDS:
            return self.residency_cache[1]
        if self.residency_task is None or self.residency_task.done():
            self.residency_task = asyncio.ensure_future(self.read_residency())
        return await asyncio.shield(self.residency_task)

    async def resident_ids(self):
        view = await self.residency()
        return list(self.models) if view is None else view["resident"]

    def alias_table(self, scope):
        stored = self.replay.aliases(scope) if self.replay is not None else None
        return dict(DEFAULT_ALIASES) if stored is None else stored

    @staticmethod
    def alias_target(table, name):
        if name in table:
            return True, table[name]
        best = None
        for pattern, target in table.items():
            if "*" in pattern and glob_match(pattern, name):
                literal = len(pattern) - pattern.count("*")
                if best is None or literal > best[0]:
                    best = (literal, target)
        return (False, None) if best is None else (True, best[1])

    async def resolve_model(self, name, key, scope, aliased=False):
        allowed = self.public_models if key is not None else frozenset(self.models)
        if isinstance(name, str) and len(name) > ALIAS_PATTERN_LENGTH:
            raise RequestError(404, "model_not_found", f"a model name has at most {ALIAS_PATTERN_LENGTH} characters; served: {sorted(allowed)}")
        if isinstance(name, str) and PACK_SEPARATOR in name:
            base, _, pack = name.partition(PACK_SEPARATOR)
            await self.resolve_model(base, key, scope, aliased)
            raise RequestError(404, "pack_not_found", f"this fleet has no pack {pack!r}; build one with POST /v1/packs")
        try:
            model = self.model_for({"model": name})
        except RequestError:
            model = None
        if model is not None:
            if key is not None and model.id not in allowed:
                raise RequestError(404, "model_not_found", f"model {name!r} is not served here; served: {sorted(allowed)}")
            return model, name != model.id and isinstance(name, str) and name.endswith(BULK_SUFFIX)
        found, target = self.alias_target(self.alias_table(scope), name) if isinstance(name, str) and not aliased else (False, None)
        if not found:
            raise RequestError(404, "model_not_found", f"model {name!r} is not served here and no alias names it; served: {sorted(allowed)}")
        if target is None:
            resident = [model_id for model_id in await self.resident_ids() if model_id in allowed]
            if not resident:
                raise RequestError(409, "model_not_loaded", f"{name!r} names the first resident model, and no model is resident",
                                   details={"resident": []})
            target = resident[0]
        return await self.resolve_model(target, key, scope, True)

    async def admit_model(self, model, key, headers):
        if self.may_swap(key, headers):
            return
        view = await self.residency()
        if view is None or model.id in view["resident"] or view["target"] == model.id:
            return
        if not view["resident"] and view["state"] != "swapping":
            return
        resident = view["resident"] or [view["target"]]
        raise RequestError(409, "model_not_loaded", f"{model.id} is not resident here, and loading it would unload {', '.join(str(item) for item in resident)}; "
                           "ask for a resident model, or use a key with allow_swap", details={"resident": view["resident"]})

    def hinted_priority(self, bulk, headers, key):
        value = (headers.get(PRIORITY_HEADER) or "").strip().lower()
        if value not in ("", "interactive", "background"):
            raise RequestError(400, "invalid_request", f"{PRIORITY_HEADER} must be interactive or background")
        if value:
            return INTERACTIVE_PRIORITY if value == "interactive" else BULK_PRIORITY
        if harness_background(headers):
            return BULK_PRIORITY
        if key is not None and key.get("priority") == "background":
            return BULK_PRIORITY
        return BULK_PRIORITY if bulk else INTERACTIVE_PRIORITY

    @staticmethod
    def affinity(headers, fallback=None):
        session = next((clean_header(headers.get(name)) for name in AFFINITY_HEADERS if clean_header(headers.get(name))), None) or clean_header(fallback)
        agent = next((clean_header(headers.get(name)) for name in AGENT_HEADERS if clean_header(headers.get(name))), None)
        return session, agent

    def replay_render(self, model, messages, tools, kwargs, scope, member, contexts):
        found = []
        try:
            for index, message in enumerate(messages):
                if message.get("role") == "assistant":
                    tokens = self.replay.find(scope, member, model.id, message_handles(message), chat_replay.view_digest(model.id, message), contexts[index])
                    if tokens is not None:
                        found.append((index, tokens))
        except chat_replay.STORE_ERRORS as error:
            print(json.dumps({"event": "replay_store_failed", "action": "find", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr, flush=True)
            return None
        if not found:
            return None
        opening = model.generation_prompt(tools, kwargs)
        anchor = max((control for control in model.vocabulary.control_strings() if control and opening and opening.startswith(control)), key=len, default=None)
        if anchor is None:
            return None
        marked = without_handles(messages)
        for number, (index, _) in enumerate(found):
            marked[index] = {"role": "assistant", "content": REPLAY_SENTINEL.format(number)}
            if messages[index].get("tool_calls"):
                marked[index]["tool_calls"] = messages[index]["tool_calls"]
        context = dict(model.template_defaults)
        context.update(kwargs)
        try:
            text, pieces = model.renderer.pieces(marked, tools, context)
        except RequestError:
            return None
        spans = template_spans(pieces)
        marks = [(match.start(), match.end(), int(match.group(1))) for match in REPLAY_MARK.finditer(text)]
        if [number for _, _, number in marks] != list(range(len(found))):
            return None
        regions, floor = [], 0
        for number, (_, tokens) in enumerate(found):
            at, after, _ = marks[number]
            limit = marks[number + 1][0] if number + 1 < len(marks) else len(text)
            start = rfind_in(spans, text, anchor, floor, at)
            close = min((position for position in (find_in(spans, text, stop, after, limit) for stop in model.stop_strings) if position >= 0),
                        default=-1) if start >= 0 else -1
            header = text[start:at]
            if close < 0 or not (opening.startswith(header) or header.startswith(opening)):
                return None
            regions.append((start, close, tokens))
            floor = close
        stops = set(model.stop_token_ids)
        ids = encode_span(model, pieces, 0, regions[0][0])
        replayed = 0
        for number, (start, close, tokens) in enumerate(regions):
            ids.extend(model.vocabulary.encode(opening, True))
            ids.extend(tokens)
            following = encode_span(model, pieces, close, regions[number + 1][0] if number + 1 < len(regions) else len(text))
            head, rest = chat_replay.splice(ids, following, stops)
            replayed = len(head)
            ids = head + rest
        return {"text": text, "ids": ids, "replayed": replayed, "turns": len(regions)}

    def replay_record(self, model, plan, assistant, result, handles, turn=None):
        if plan.get("scope") is None or self.replay is None or result.get("engine_finish") == "stop_string" or not result.get("token_ids"):
            return None
        try:
            return self.replay.put(plan["scope"], plan.get("member"), model.id, chat_replay.view_digest(model.id, assistant), plan["replay_context"],
                                   list(plan.get("forced_ids", [])) + list(result["token_ids"]), handles, turn)
        except chat_replay.STORE_ERRORS as error:
            print(json.dumps({"event": "replay_store_failed", "request": plan.get("request_id"), "error": f"{type(error).__name__}: {error}"}),
                  file=sys.stderr, flush=True)
            return None

    def store_safely(self, plan, action, *arguments):
        try:
            action(*arguments)
        except chat_replay.STORE_ERRORS as error:
            print(json.dumps({"event": "replay_store_failed", "request": plan.get("request_id"), "error": f"{type(error).__name__}: {error}"}),
                  file=sys.stderr, flush=True)

    def request_record(self, model, plan, result, usage, started, headers):
        timing = plan["timing"]
        first, last, admitted = timing.get("first"), timing.get("last"), timing.get("admitted", started)
        completion = usage["completion"]
        prefill_ms = round((first - admitted) * 1000) if first else None
        if prefill_ms is not None and self.replay is not None:
            self.store_safely(plan, self.replay.observe_prefill, model.id, usage["prompt"], usage["prompt"] - usage["cached"], prefill_ms)
        try:
            rate = self.replay.prefill_rate(model.id, usage["prompt"]) if self.replay is not None else None
        except chat_replay.STORE_ERRORS:
            rate = None
        engine = result.get("usage") or {}
        drafted = engine.get("draft_tokens")
        session, agent = plan.get("affinity") or (None, None)
        return {"request": plan["request_id"], "fleet": clean_header(headers.get(FLEET_HEADER), FLEET_VALUE), "model": model.id, "pack": None,
                "prompt_tokens": usage["prompt"], "cached_tokens": usage["cached"], "prefill_tokens": usage["prompt"] - usage["cached"],
                "replayed_tokens": replayed_tokens(plan),
                "queue_ms": round((admitted - started) * 1000), "prefill_ms": prefill_ms, "ttft_ms": round((first - started) * 1000) if first else None,
                "output_tokens": completion, "decode_tps": round((completion - 1) / (last - first), 2) if first and last and last > first and completion > 1 else None,
                "draft_accept": round(engine.get("accepted_draft_tokens", 0) / drafted, 3) if isinstance(drafted, int) and drafted > 0 else None,
                "cold_prefill_ms_estimate": round(usage["cached"] * rate) if rate is not None else None,
                "priority": "interactive" if plan["engine_body"].get("priority", INTERACTIVE_PRIORITY) >= INTERACTIVE_PRIORITY else "background",
                "session": session, "agent": agent}

    def early_headers(self, model, plan, headers):
        shown = {"X-SparkPipe-Request": plan["request_id"], "X-SparkPipe-Model": model.id, "X-SparkPipe-Prompt-Tokens": str(plan["prompt_tokens"]),
                 "X-SparkPipe-Replayed-Tokens": str(replayed_tokens(plan)),
                 "X-SparkPipe-Queue-Ms": str(round((plan["timing"].get("admitted", plan["timing"]["arrived"]) - plan["timing"]["arrived"]) * 1000))}
        fleet = clean_header(headers.get(FLEET_HEADER), FLEET_VALUE)
        if fleet:
            shown["X-SparkPipe-Fleet"] = fleet
        return shown

    def normalized_usage(self, plan, result, reasoning_tokens=0):
        block = self.usage_block(result.get("usage"), plan["prompt_tokens"], plan)
        prompt, cached = block["prompt_tokens"], block["prompt_tokens_details"]["cached_tokens"]
        committed = min(max(0, prompt // CACHE_BLOCK_TOKENS * CACHE_BLOCK_TOKENS - cached), prompt - cached)
        return {"prompt": prompt, "cached": cached, "committed": committed, "completion": block["completion_tokens"], "reasoning": reasoning_tokens}

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
        return web.json_response({"ok": healthy, "models": states, "tool_call_faults": self.tool_call_faults}, status=200 if healthy else 503)

    async def liveliness(self, request):
        return web.json_response({"ok": True})

    async def login_role(self, authorization):
        async with self.session.get(self.control["admin_url"], headers={"Authorization": authorization}, timeout=aiohttp.ClientTimeout(total=10)) as response:
            user = await response.json(content_type=None) if response.status == 200 else None
        return user.get("role") if isinstance(user, dict) else None

    def status_view(self, payload):
        def shown(model_id):
            return model_id if model_id is None or self.public_header is None or model_id in self.public_models else None
        view = {key: payload.get(key) for key in STATUS_VIEW_FIELDS}
        for key in ("active", "target", "wanted"):
            view[key] = shown(payload.get(key))
        draining = payload.get("draining")
        view["draining"] = dict({key: draining.get(key) for key in DRAINING_VIEW_FIELDS}, model=shown(draining.get("model"))) if isinstance(draining, dict) else None
        view["models"] = [{"id": entry.get("id"), "active": entry.get("active")} for entry in payload.get("models") or [] if isinstance(entry, dict) and shown(entry.get("id")) is not None]
        view["history"] = [dict({key: entry.get(key) for key in HISTORY_VIEW_FIELDS}, **{"from": shown(entry.get("from")), "to": shown(entry.get("to"))})
                           for entry in payload.get("history") or [] if isinstance(entry, dict)]
        view["waiting"] = {model_id: count for model_id, count in (payload.get("waiting") or {}).items() if shown(model_id) is not None}
        return view

    async def control_status(self, request):
        if self.control is None:
            return web.json_response(error_body("not_found", "this front end has no model control"), status=404)
        public = request.get("public_key") is not None
        authorization = request.headers.get("Authorization", "")
        try:
            admin = not public and authorization.startswith("Bearer ") and await self.login_role(authorization) == "admin"
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
            admin = False
        try:
            async with self.session.get(self.control["status_url"], headers=self.controller_headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                payload = await response.json(content_type=None)
                if not isinstance(payload, dict):
                    raise json.JSONDecodeError("the controller status is not an object", "", 0)
                payload["waiting"] = {model.id: model.gate.state()["waiting"] for model in self.models.values() if model.gate.state()["waiting"]}
                return web.json_response(payload if admin else self.status_view(payload), status=response.status)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as error:
            return web.json_response(error_body("control_unavailable", str(error) if admin else "the fleet controller is unreachable"), status=503)

    async def control_switch_now(self, request):
        if self.control is None or "force_url" not in self.control:
            return web.json_response(error_body("not_found", "this front end has no Switch now"), status=404)
        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Bearer ") or not authorization[len("Bearer "):].strip():
            return web.json_response(error_body("unauthorized", "Switch now needs the admin's SparkPipe Chat login"), status=401)
        try:
            role = await self.login_role(authorization)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as error:
            return web.json_response(error_body("control_unavailable", f"could not check the login: {error}"), status=503)
        if role != "admin":
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
        raise web.HTTPFound("/site/fleet.html")

    async def shown_models(self, key):
        try:
            resident = await self.resident_ids()
        except RequestError:
            resident = []
        shown = sorted((model for model in self.models.values() if key is None or model.id in self.public_models),
                       key=lambda model: resident.index(model.id) if model.id in resident else len(resident))
        return shown, resident

    async def list_models(self, request):
        created = int(time.time())
        key = request.get("public_key") if request is not None else None
        try:
            scope = self.scope_of(key, request.headers) if key is not None else OWN_SCOPE
        except RequestError as error:
            return error_response(error)
        shown, resident = await self.shown_models(key)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created))
        data = [{"id": name, "object": "model", "type": "model", "display_name": name, "created": created, "created_at": stamp, "owned_by": "sparkpipe",
                 "max_model_len": model.context_tokens, "context_window": model.context_tokens, "max_output_tokens": model.max_output_tokens,
                 "sequences": model.sequences.sequences, "reasoning_efforts": effort_order(model.reasoning_effort), "resident": model.id in resident}
                for model in shown for name in (model.id, model.id + BULK_SUFFIX)]
        return web.json_response({"object": "list", "data": data, "has_more": False, "first_id": data[0]["id"] if data else None,
                                  "last_id": data[-1]["id"] if data else None, "aliases": self.alias_table(scope), "packs": [], "custom_models": []})

    async def codex_models(self, request):
        if self.codex_instructions is None:
            return error_response(RequestError(404, "not_found", "this chat layer has no Codex catalogue; set codex.instructions_file"))
        shown, resident = await self.shown_models(request.get("public_key"))
        return web.json_response({"models": [codex_entry(model, index, model.id in resident, self.codex_instructions) for index, model in enumerate(shown)]})

    def prefix_ledger(self, model):
        if self.prefixes is None:
            self.prefixes = {}
        return self.prefixes.setdefault(model.id, PrefixLedger())

    def public_body_limit(self):
        context = max((self.models[model_id].context_tokens for model_id in self.public_models if model_id in self.models), default=0)
        return max(PUBLIC_BODY_MIN_BYTES, PUBLIC_BODY_BYTES_PER_TOKEN * context + PUBLIC_BODY_OVERHEAD_BYTES)

    def public_prefix_ledger(self, key, model):
        return self.public_prefixes.setdefault((key["id"], model.id), PrefixLedger(capacity_blocks=PUBLIC_KEY_PREFIX_BLOCKS))

    def prepare(self, model, body):
        plan = self.plan(model, body, None)
        return plan["engine_body"], plan["parser"], plan["prompt_tokens"]

    def plan(self, model, body, headers, scope=None, member=None):
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
        trace = request_trace(headers, messages)
        if self.citation_wrapper is not None and headers is not None and headers.get("X-OpenWebUI-Chat-Id"):
            messages, citations = canonical_citations(messages, self.citation_wrapper)
            trace.update(citations)
        memory_off = False
        identity, settings = None, {}
        if self.memory is not None:
            messages, memory_off = self.memory_switch(messages)
            if headers is not None:
                identity = self.commands.identity(headers)
                command = self.commands.match(messages)
                if command is not None:
                    trace.update(command=command[0])
                    return {"command": command, "identity": identity, "messages": messages, "memory_off": memory_off, "trace": trace}
                messages = self.command_strip(messages)
                settings = self.commands.settings(identity)
            memory_off = memory_off or settings.get("memory") == "off"
            if memory_off:
                trace.update(memory_off=True)
                messages = self.commands.compacted(identity, messages)
        target = context_target(body) or settings.get("target")
        if self.memory is not None and headers is not None and not memory_off:
            session = self.memory.session(model, headers, messages, tools, kwargs, target)
            if session is not None:
                messages, tools = normalize_messages(session.messages), session.tools
                trace.update(memory_lined=session.layout["lined"], memory_open=len(session.layout["open"]), memory_target=session.target)
        contexts = chat_replay.context_digests(model.id, tools, kwargs, messages) if scope is not None and session is None else None
        replay = self.replay_render(model, messages, tools, kwargs, scope, member, contexts) if contexts is not None else None
        plan = self.finish_plan(model, body, messages, tools, kwargs, session, (replay["text"], replay["ids"]) if replay is not None else None)
        plan.update(replay=replay, scope=scope if contexts is not None else None, member=member, replay_context=contexts[-1] if contexts else None)
        engine = plan["engine_body"]
        seen, seen_of = self.prefix_ledger(model).seen(engine["prompt_token_ids"])
        trace.update(seen_tokens=seen, seen_of=seen_of)
        trace.update(temperature=engine.get("temperature"), top_p=engine.get("top_p"), top_k=engine.get("top_k"), reasoning_effort=body.get("reasoning_effort"),
                     tools=len(tools or []), stream=bool(body.get("stream")))
        plan["trace"] = trace
        return plan

    def finish_plan(self, model, body, messages, tools, kwargs, session, rendered=None):
        prompt, prompt_ids = model.render(messages, tools, kwargs) if rendered is None else rendered
        if len(prompt_ids) >= model.context_tokens:
            raise RequestError(400, chat_shapes.OVERFLOW_CODE, f"the prompt has {len(prompt_ids)} tokens and {model.id} serves {model.context_tokens}",
                               overflow=(len(prompt_ids), model.context_tokens - 1))
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
        engine_body["priority"] = request_priority(model, body)
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

    async def engine_stream(self, model, engine_body, public):
        deadline = None
        while True:
            started = False
            events = self.engine_events(model, engine_body)
            try:
                async for chunk in events:
                    started = True
                    yield chunk
                return
            except RequestError as error:
                if started or error.code not in ("engine_starting", "engine_unavailable"):
                    raise
            finally:
                await events.aclose()
            now = asyncio.get_running_loop().time()
            deadline = deadline if deadline is not None else now + model.gate.wait_seconds
            if self.memory is not None and not public:
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

    async def generate_attempt(self, model, engine_body, parser, outcome, public, timing=None):
        step = model.vocabulary.decoder()
        finish = None
        usage = None
        token_ids = []
        ends = []
        start = None
        written = 0
        stop_string = False
        logprobs = engine_body.get("logprobs")
        stream = self.engine_stream(model, engine_body, public)
        try:
            async for chunk in stream:
                if "error" in chunk:
                    raise RequestError(502, "engine_error", json.dumps(chunk["error"]))
                if timing is not None and chunk.get("tokens"):
                    timing.setdefault("first", time.monotonic())
                    timing["last"] = time.monotonic()
                if logprobs is not None:
                    entries = self.logprob_entries(model, chunk, logprobs + 1)
                    if entries:
                        yield ("logprobs", entries)
                text = []
                for token in chunk.get("tokens", []):
                    token_ids.append(token)
                    if token not in model.stop_token_ids:
                        piece = step(token)
                        if piece:
                            text.append(piece)
                            written += len(piece)
                    ends.append(written)
                choices = chunk.get("choices") or [{}]
                finish = choices[0].get("finish_reason") or finish
                usage = chunk.get("usage") or usage
                if start is None:
                    start = parser.fed
                for event in parser.feed("".join(text)):
                    if event[0] == "bad_call":
                        outcome["fault"] = event[1]
                        break
                    yield event
                if "fault" in outcome:
                    break
                if parser.stopped and not parser.ended:
                    finish = "stop"
                    stop_string = True
                    break
        finally:
            await stream.aclose()
        if "fault" not in outcome:
            for event in parser.finish():
                if event[0] == "bad_call" and event[1]["kind"] == "unclosed" and event[1]["repair"] is not None and finish == "stop":
                    self.tool_call_fault(model, event[1], outcome.get("attempt", 0), "repaired")
                    parser.calls += 1
                    yield ("tool_call", event[1]["repair"])
                    continue
                if event[0] == "bad_call":
                    outcome["fault"] = event[1]
                    break
                yield event
        outcome.update(token_ids=token_ids, finish=finish, usage=usage, stop_string=stop_string)
        fault = outcome.get("fault")
        if fault is not None:
            base = parser.fed if start is None else start
            kept = 0
            while kept < len(ends) and base + ends[kept] <= fault["offset"]:
                kept += 1
            fault["tokens"] = token_ids[:kept]
            fault["boundary"] = base + (ends[kept - 1] if kept else 0) == fault["offset"]

    def tool_call_fault(self, model, fault, attempt, action):
        self.tool_call_faults[action] = self.tool_call_faults.get(action, 0) + 1
        print(json.dumps({"event": "tool_call_fault", "model": model.id, "kind": fault["kind"], "attempt": attempt, "action": action,
                          "text": fault["text"][:400]}, ensure_ascii=False), file=sys.stderr, flush=True)

    async def generate(self, model, engine_body, parser, public, timing=None):
        prompt = list(engine_body["prompt_token_ids"])
        budget = engine_body.get("max_tokens")
        body, kept, attempt = engine_body, [], 0
        while True:
            outcome = {"attempt": attempt}
            async for event in self.generate_attempt(model, body, parser, outcome, public, timing):
                yield event
            fault = outcome.get("fault")
            if fault is None:
                break
            kept = kept + fault["tokens"]
            left = None if budget is None else budget - len(kept)
            if fault["kind"] == "unclosed" and (outcome["finish"] == "length" or left == 0):
                self.tool_call_fault(model, fault, attempt, "cut_by_length")
                outcome.update(finish="length", token_ids=[], stop_string=False)
                parser = fault["resume"]
                break
            if (attempt >= TOOL_CALL_RETRIES or not body.get("temperature") or body.get("logprobs") is not None or not fault["boundary"] or
                    (left is not None and left <= 0)):
                self.tool_call_fault(model, fault, attempt, "refused")
                raise RequestError(502, "tool_call_invalid", f"the {model.id} reply has a tool call that does not parse, after {attempt} "
                                   f"regeneration(s): {fault['text'][:200]}")
            self.tool_call_fault(model, fault, attempt, "regenerated")
            attempt += 1
            parser = fault["resume"]
            body = dict(engine_body, prompt_token_ids=prompt + kept)
            if left is not None:
                body["max_tokens"] = left
            if body.get("seed") is not None:
                body["seed"] = (body["seed"] + attempt) % (1 << 64)
        token_ids = kept + outcome["token_ids"]
        usage = dict(outcome["usage"] or {})
        if usage and kept:
            usage["prompt_tokens"] = len(prompt)
            usage["completion_tokens"] = len(token_ids)
            usage["total_tokens"] = len(prompt) + len(token_ids)
            details = dict(usage.get("prompt_tokens_details") or {})
            if "cached_tokens" in details:
                details["cached_tokens"] = min(details["cached_tokens"], len(prompt))
                usage["prompt_tokens_details"] = details
        self.prefix_ledger(model).record(prompt + token_ids)
        finish = outcome["finish"]
        yield ("finish", {"finish_reason": "tool_calls" if parser.calls else (finish or "stop"), "usage": usage or None,
                          "engine_finish": "stop_string" if outcome["stop_string"] else (finish or "stop"), "token_ids": token_ids,
                          "stop_marker": parser.stop_marker if outcome["stop_string"] else None})

    async def respond(self, model, plan):
        session = plan["session"]
        if session is None:
            async for event in self.generate(model, plan["engine_body"], plan["parser"], "public_key" in plan, plan.get("timing")):
                yield event
            return
        completion = 0
        while True:
            internal, visible, content, reasoning = [], [], [], []
            result = {}
            async for kind, value in self.generate(model, plan["engine_body"], plan["parser"], False, plan.get("timing")):
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
        _, pieces = model.renderer.pieces(normalize_messages(messages), tools, context)
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

    def memory_count(self, model, text):
        return len(model.vocabulary.encode(text, False))

    def memory_cut_ids(self, model, messages, tools, kwargs, sentinel):
        parts = self.memory_template_parts(model, messages, tools, kwargs, [sentinel])
        return parts[0][0] if parts else None

    async def memory_complete(self, model, ids, max_tokens, reasoning_open=False, kind="background", trace=None):
        body = {"prompt_token_ids": ids, "max_tokens": max_tokens, "stream": True, "stop_token_ids": model.stop_token_ids, "priority": BACKGROUND_PRIORITY, "temperature": 0.0}
        parser = OutputParser(model, None, reasoning_open, []) if model.channels is None else None
        step = model.vocabulary.decoder()
        content, usage, generated, outcome, tokens = [], {}, 0, "error", []
        started = time.monotonic()
        try:
            async for chunk in self.engine_events(model, body):
                usage = chunk.get("usage") or usage
                generated += len(chunk.get("tokens", []))
                tokens.extend(chunk.get("tokens", []))
                text = "".join(piece for piece in (step(token) for token in chunk.get("tokens", []) if token not in model.stop_token_ids) if piece)
                if parser is None:
                    content.append(text)
                    continue
                content.extend(value for kind_, value in parser.feed(text) if kind_ == "content")
            outcome = "done"
            self.prefix_ledger(model).record(list(ids) + tokens)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            print(json.dumps(dict(trace or {}, event="memory_job", model=model.id, kind=kind, outcome=outcome, prompt_tokens=len(ids),
                                  cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"), completion_tokens=generated,
                                  seconds=round(time.monotonic() - started, 3))), file=sys.stderr, flush=True)
        if parser is not None:
            content.extend(value for kind_, value in parser.finish() if kind_ == "content")
        return "".join(content)

    def usage_block(self, usage, prompt_tokens, plan):
        usage = usage or {}
        completion = usage.get("completion_tokens", 0)
        prompt = usage.get("prompt_tokens", prompt_tokens)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if "public_key" in plan:
            cached = min(cached, plan["public_cached"])
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion, "prompt_tokens_details": {"cached_tokens": cached}}

    def record_public_prefix(self, model, plan, result):
        if "public_key" in plan:
            self.public_prefix_ledger(plan["public_key"], model).record(list(plan["engine_body"]["prompt_token_ids"]) + list(result.get("token_ids") or []))

    def log(self, model, prompt_tokens, usage, started, first, finish, trace=None):
        now = time.monotonic()
        usage = usage or {}
        completion = usage.get("completion_tokens", 0)
        decode_seconds = now - first if first else 0.0
        print(json.dumps(dict({"event": "chat", "model": model.id, "prompt_tokens": prompt_tokens,
                               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                               "completion_tokens": completion, "ttft_s": round(first - started, 3) if first else None,
                               "decode_tok_s": round((completion - 1) / decode_seconds, 2) if completion > 1 and decode_seconds > 0 else None,
                               "finish_reason": finish}, **(trace or {}))), file=sys.stderr, flush=True)

    async def chat(self, request):
        started = time.monotonic()
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(error_body("invalid_json", "the request body is not JSON"), status=400)
        public = request.get("public_key")
        if not isinstance(body, dict):
            return web.json_response(error_body("invalid_json", "the request body must be a JSON object"), status=400)
        try:
            scope = self.scope_of(public, request.headers) if public is not None else None
            model, bulk = await self.resolve_model(body.get("model"), public, scope or OWN_SCOPE)
            await self.admit_model(model, public, request.headers)
            plan = self.plan(model, body, None if public is not None else request.headers, scope, self.member_of(public, request.headers))
            if "engine_body" in plan:
                plan["engine_body"]["priority"] = self.hinted_priority(bulk, request.headers, public)
        except RequestError as error:
            return error_response(error)
        self.stamp(plan, started, request.headers)
        plan["gate"] = (scope or OWN_SCOPE, self.member_of(public, request.headers))
        if public is not None:
            plan["trace"]["key"] = public["id"]
            plan["public_key"] = public
            plan["public_cached"] = max(self.public_prefix_ledger(public, model).seen(plan["engine_body"]["prompt_token_ids"])[0], replayed_tokens(plan))
            return await self.admitted(request, model, body, plan, started, self.answer, error_response)
        if "command" in plan:
            name, argument = plan["command"]
            try:
                text = await self.commands.run(model, plan["identity"], plan["messages"], name, argument, plan["memory_off"])
            except RequestError as error:
                return error_response(error)
            print(json.dumps(dict(plan["trace"], event="chat_command", model=model.id, command=name)), file=sys.stderr, flush=True)
            return await self.reply_text(request, model, body, text)
        if self.memory is not None:
            self.memory.enter(model)
        try:
            return await self.admitted(request, model, body, plan, started, self.answer, error_response)
        finally:
            if self.memory is not None:
                self.memory.leave(model)

    def stamp(self, plan, started, headers, affinity=None):
        plan["request_id"] = "r_" + uuid.uuid4().hex[:24]
        plan["timing"] = {"arrived": started}
        plan["affinity"] = self.affinity(headers, affinity)

    async def admitted(self, request, model, body, plan, started, answer, failed):
        scope, member = plan.get("gate", (OWN_SCOPE, None))
        try:
            await model.sequences.acquire(plan["engine_body"].get("priority", INTERACTIVE_PRIORITY), BUSY_WAIT_SECONDS if "public_key" in plan else None,
                                          scope, member)
        except RequestError as error:
            return failed(error)
        plan["timing"]["admitted"] = time.monotonic()
        try:
            return await answer(request, model, body, plan, started)
        finally:
            await model.sequences.release(scope, member)

    def settle(self, model, plan, result, assistant, started, headers, shape, reasoning_tokens=0):
        usage = self.normalized_usage(plan, result, reasoning_tokens)
        if assistant is not None:
            self.replay_record(model, plan, assistant, result, message_handles(assistant))
        record = self.request_record(model, plan, result, usage, started, headers)
        print(json.dumps(dict(record, event="request_record", shape=shape, key=(plan.get("public_key") or {}).get("id")), ensure_ascii=False),
              file=sys.stderr, flush=True)
        return usage, record

    async def read_body(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise RequestError(400, "invalid_request", "the request body is not JSON")
        if not isinstance(body, dict):
            raise RequestError(400, "invalid_request", "the request body must be a JSON object")
        return body

    def responses_input(self, translated, scope, member):
        previous = []
        if translated["previous_response_id"] is not None:
            try:
                previous = self.replay.response_messages(scope, member, translated["previous_response_id"])
            except chat_replay.STORE_ERRORS as error:
                print(json.dumps({"event": "replay_store_failed", "action": "previous_response", "error": f"{type(error).__name__}: {error}"}),
                      file=sys.stderr, flush=True)
                raise RequestError(503, "store_unavailable", "the stored responses could not be read; try again")
            if previous is None:
                raise RequestError(400, "previous_response_not_found", f"there is no stored response {translated['previous_response_id']!r} for this account")
        system = [{"role": "system", "content": translated["instructions"]}] if translated["instructions"] else []
        translated["stored_from"] = len(system) + len(previous)
        translated["messages"] = system + previous + translated["input_messages"]
        if len(translated["messages"]) == len(system):
            raise RequestError(400, "invalid_request", "input must hold at least one item")
        return translated

    async def translate(self, request, shape):
        key = request.get("public_key")
        scope = self.scope_of(key, request.headers)
        body = await self.read_body(request)
        if shape == "messages":
            translated = chat_shapes.messages_request(body)
        else:
            translated = self.responses_input(chat_shapes.responses_request(body), scope, self.member_of(key, request.headers))
        model, bulk = await self.resolve_model(translated["model"], key, scope)
        if translated["images"] and not model.images:
            raise RequestError(400, "image_not_supported", f"{model.id} does not take images")
        return body, translated, model, bulk, key, scope

    def harness_plan(self, model, translated, scope, headers, key, bulk):
        messages = normalize_messages(translated["messages"])
        tools = translated["tools"] or None
        choice = translated["tool_choice"]
        forced = False
        if choice == "none":
            tools = None
        elif isinstance(choice, dict):
            tools = [tool for tool in tools or [] if tool["function"]["name"] == choice["name"]]
            if not tools:
                raise RequestError(400, "invalid_request", f"tool_choice names {choice['name']!r}, which is not among the tools")
            forced = True
        elif choice == "required":
            if not tools:
                raise RequestError(400, "invalid_request", "a required tool choice needs tools")
            forced = True
        if forced and model.channels is not None:
            raise RequestError(400, "unsupported_parameter", f"{model.id} takes tool_choice auto or none until grammar-constrained sampling lands")
        kwargs = {}
        effort = chat_shapes.effort_for(model.reasoning_effort, translated["effort"], translated["thinking"], translated["budget"])
        if effort is not None:
            kwargs.update(model.reasoning_effort[effort])
        member = self.member_of(key, headers)
        contexts = chat_replay.context_digests(model.id, tools, kwargs, messages)
        replay = self.replay_render(model, messages, tools, kwargs, scope, member, contexts)
        messages = without_handles(messages)
        text, ids = (replay["text"], replay["ids"]) if replay is not None else model.render(messages, tools, kwargs)
        forced_ids = []
        if forced:
            reasoning = model.reasoning
            opening = (reasoning.get("close", reasoning["end"]) if text.rstrip().endswith(reasoning["start"].rstrip()) else "") + model.tool_calls["start"]
            forced_ids = model.vocabulary.encode(opening, True)
            text, ids = text + opening, list(ids) + forced_ids
        limit = translated["max_tokens"]
        body = dict(translated["sampling"], max_tokens=min(limit, model.max_output_tokens) if limit else None, stop=translated["stops"])
        plan = self.finish_plan(model, body, messages, tools, kwargs, None, (text, ids))
        if forced:
            plan["parser"].feed(model.tool_calls["start"])
        plan["engine_body"]["priority"] = self.hinted_priority(bulk, headers, key)
        plan.update(scope=scope, member=member, replay_context=contexts[-1], replay=replay, forced_ids=forced_ids, gate=(scope, member),
                    trace={"shape": "harness", "tools": len(tools or []), "replayed": replayed_tokens({"replay": replay})})
        if key is not None:
            plan["public_key"] = key
            plan["public_cached"] = max(self.public_prefix_ledger(key, model).seen(ids)[0], replayed_tokens(plan))
        return plan

    async def messages(self, request):
        return await self.harness(request, "messages", anthropic_error_response)

    async def responses(self, request):
        return await self.harness(request, "responses", responses_error_response)

    async def harness(self, request, shape, failed):
        started = time.monotonic()
        try:
            body, translated, model, bulk, key, scope = await self.translate(request, shape)
            await self.admit_model(model, key, request.headers)
            plan = self.harness_plan(model, translated, scope, request.headers, key, bulk)
        except RequestError as error:
            return failed(error)
        self.stamp(plan, started, request.headers, translated.get("affinity"))
        plan.update(shape=shape, translated=translated, request_body=body)
        return await self.admitted(request, model, body, plan, started, self.harness_answer, failed)

    def responses_echo(self, body, translated):
        echo = {key: body.get(key) for key in ("instructions", "previous_response_id", "temperature", "top_p", "max_output_tokens", "metadata", "user")}
        echo.update(tools=body.get("tools") or [], tool_choice=body.get("tool_choice", "auto"), parallel_tool_calls=body.get("parallel_tool_calls", True),
                    store=translated["store"], reasoning=body.get("reasoning"), text=body.get("text") or {"format": {"type": "text"}}, object="response")
        return echo

    async def harness_answer(self, request, model, body, plan, started):
        shape, translated = plan["shape"], plan["translated"]
        failed = anthropic_error_response if shape == "messages" else responses_error_response
        if shape == "messages":
            turn = chat_shapes.MessagesTurn(translated["model"], plan["prompt_tokens"])
        else:
            turn = chat_shapes.ResponsesTurn(translated["model"], translated, self.responses_echo(body, translated))
        headers = self.early_headers(model, plan, request.headers)
        stream = translated["stream"]
        response, result, reasoning, first = None, {}, [], None

        async def write(frames):
            for frame in frames:
                await response.write(frame)

        async def open_stream():
            nonlocal response
            response = web.StreamResponse(headers=dict(headers, **{"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}))
            await response.prepare(request)
            await write(turn.start())

        try:
            async for kind, value in heartbeat(self.respond(model, plan), KEEPALIVE_SECONDS):
                if kind == "keepalive":
                    if stream:
                        if response is None:
                            await open_stream()
                        await response.write(turn.keepalive())
                    continue
                if kind == "finish":
                    result = value
                    continue
                if kind == "logprobs":
                    continue
                first = first or time.monotonic()
                if kind == "reasoning":
                    reasoning.append(value)
                if stream and response is None:
                    await open_stream()
                frames = turn.feed(kind, value)
                if stream:
                    await write(frames)
        except RequestError as error:
            self.log(model, plan["prompt_tokens"], result.get("usage"), started, first, error.code, plan.get("trace"))
            if response is None:
                return failed(error)
            await write(turn.failure(error))
            return response
        except (ConnectionResetError, aiohttp.ClientConnectionResetError):
            self.log(model, plan["prompt_tokens"], result.get("usage"), started, first, "client_gone", plan.get("trace"))
            return response
        except asyncio.CancelledError:
            self.log(model, plan["prompt_tokens"], result.get("usage"), started, first, "client_gone", plan.get("trace"))
            raise
        reasoning_tokens = len(model.vocabulary.encode("".join(reasoning), False)) if reasoning else 0
        usage = self.normalized_usage(plan, result, reasoning_tokens)
        if shape == "messages":
            stop_reason, stop_sequence = chat_shapes.anthropic_stop(result, result.get("stop_marker"))
            close = lambda record: turn.finish(stop_reason, stop_sequence, chat_shapes.anthropic_usage(usage), record)
            whole = lambda record: turn.body(stop_reason, stop_sequence, chat_shapes.anthropic_usage(usage), record)
        else:
            incomplete = chat_shapes.responses_incomplete(result)
            close = lambda record: turn.finish(incomplete, chat_shapes.responses_usage(usage), record)
            whole = lambda record: turn.body(incomplete, chat_shapes.responses_usage(usage), record)
        record = self.request_record(model, plan, result, usage, started, request.headers)
        frames = close(record) if stream else None
        reply = None if stream else whole(record)
        assistant = turn.assistant()
        self.replay_record(model, plan, assistant, result, message_handles(assistant), turn.turn)
        if shape == "responses" and translated["store"]:
            own = translated["messages"][translated["stored_from"]:] + [assistant]
            self.store_safely(plan, self.replay.put_response, plan["scope"], plan.get("member"), turn.id, translated["previous_response_id"], model.id, own)
        print(json.dumps(dict(record, event="request_record", shape=shape, key=(plan.get("public_key") or {}).get("id")), ensure_ascii=False),
              file=sys.stderr, flush=True)
        self.log(model, plan["prompt_tokens"], result.get("usage"), started, first, result.get("finish_reason"), plan.get("trace"))
        self.record_public_prefix(model, plan, result)
        if not stream:
            return web.json_response(reply, headers=headers)
        if response is None:
            await open_stream()
        await write(frames)
        await response.write_eof()
        return response

    async def count_tokens(self, request):
        try:
            _, translated, model, bulk, key, scope = await self.translate(request, "messages")
            plan = self.harness_plan(model, translated, scope, request.headers, key, bulk)
        except RequestError as error:
            return anthropic_error_response(error)
        return web.json_response({"input_tokens": plan["prompt_tokens"]})

    async def tokenize(self, request):
        key = request.get("public_key")
        try:
            scope = self.scope_of(key, request.headers)
            body = await self.read_body(request)
            model, _ = await self.resolve_model(body.get("model"), key, scope)
            if isinstance(body.get("text"), str):
                ids = model.vocabulary.encode(body["text"], False)
            elif isinstance(body.get("messages"), list):
                messages = normalize_messages(body["messages"])
                tools = body.get("tools") or None
                replay = self.replay_render(model, messages, tools, {}, scope, self.member_of(key, request.headers),
                                            chat_replay.context_digests(model.id, tools, {}, messages))
                ids = replay["ids"] if replay is not None else model.render(without_handles(messages), tools, {})[1]
            else:
                raise RequestError(400, "invalid_request", "send text, or messages (and tools) to render with the chat template")
        except RequestError as error:
            return error_response(error)
        return web.json_response({"model": model.id, "tokens": list(ids), "count": len(ids)})

    async def detokenize(self, request):
        key = request.get("public_key")
        try:
            body = await self.read_body(request)
            model, _ = await self.resolve_model(body.get("model"), key, self.scope_of(key, request.headers))
            tokens = body.get("tokens")
            if not isinstance(tokens, list) or not all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in tokens):
                raise RequestError(400, "invalid_request", "tokens must be an array of token ids")
            step = model.vocabulary.decoder()
            try:
                text = "".join(piece for piece in (step(token) for token in tokens) if piece)
            except Exception as error:
                raise RequestError(400, "invalid_request", f"the tokenizer refused the tokens: {error}")
        except RequestError as error:
            return error_response(error)
        return web.json_response({"model": model.id, "text": text})

    async def hello(self, request):
        return web.Response(status=200)

    async def get_aliases(self, request):
        try:
            scope = self.scope_of(request.get("public_key"), request.headers)
            table = self.alias_table(scope)
            resident = await self.resident_ids()
        except RequestError as error:
            return error_response(error)
        return web.json_response({"aliases": table, "first_resident": resident[0] if resident else None})

    async def put_aliases(self, request):
        key = request.get("public_key")
        try:
            scope = self.scope_of(key, request.headers)
            table = await self.read_body(request)
            if len(table) > ALIAS_TABLE_SIZE:
                raise RequestError(400, "invalid_request", f"an alias table has at most {ALIAS_TABLE_SIZE} entries")
            for pattern, target in table.items():
                if not pattern or len(pattern) > ALIAS_PATTERN_LENGTH or pattern.count("*") > ALIAS_STARS:
                    raise RequestError(400, "invalid_request", f"an alias is 1 to {ALIAS_PATTERN_LENGTH} characters with at most {ALIAS_STARS} *")
                if target is not None and (not isinstance(target, str) or len(target) > ALIAS_PATTERN_LENGTH):
                    raise RequestError(400, "invalid_request", "aliases map a name or a pattern with * to a model name, or to null for the first resident model")
                if target is not None and target.partition(PACK_SEPARATOR)[0] not in (self.public_models if key is not None else self.models):
                    raise RequestError(400, "invalid_request", f"alias {pattern!r} names {target!r}, which is not a model served here")
            self.replay.set_aliases(scope, table)
        except RequestError as error:
            return error_response(error)
        except chat_replay.STORE_ERRORS as error:
            print(json.dumps({"event": "replay_store_failed", "action": "set_aliases", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr, flush=True)
            return error_response(RequestError(503, "store_unavailable", "the alias table could not be saved; try again"))
        return web.json_response({"aliases": table})

    async def delete_cache(self, request):
        key = request.get("public_key")
        try:
            scope = self.scope_of(key, request.headers)
        except RequestError as error:
            return error_response(error)
        if request.query.get("storage") not in (None, "false", "true"):
            return error_response(RequestError(400, "invalid_request", "storage is true or false"))
        deleted = self.replay.forget(scope)
        if key is not None:
            for name in [name for name in self.public_prefixes if name[0] == key["id"]]:
                del self.public_prefixes[name]
        print(json.dumps({"event": "working_state_deleted", "scope": scope, "replay": deleted}), file=sys.stderr, flush=True)
        return web.json_response({"deleted": ["replay_store", "aliases", "stored_responses"],
                                  "not_deleted": ["engine_cache"] + (["storage"] if request.query.get("storage") == "true" else []),
                                  "note": "the engine's cache is deleted at the next change of user or 24 h after the last hour held"})

    async def reply_text(self, request, model, body, text):
        identifier = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not body.get("stream"):
            return web.json_response({"id": identifier, "object": "chat.completion", "created": created, "model": model.id,
                                      "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "logprobs": None, "finish_reason": "stop"}], "usage": usage})
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await response.prepare(request)
        for delta, finish in (({"role": "assistant", "content": text}, None), ({}, "stop")):
            payload = {"id": identifier, "object": "chat.completion.chunk", "created": created, "model": model.id,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if finish:
                payload["usage"] = usage
            await response.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    async def answer(self, request, model, body, plan, started):
        engine_body, prompt_tokens = plan["engine_body"], plan["prompt_tokens"]
        if body.get("stream"):
            return await self.chat_stream(request, model, body, plan, started)
        content, reasoning, calls, logprobs = [], [], [], []
        result = {}
        first = None
        try:
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
        except asyncio.CancelledError:
            self.log(model, prompt_tokens, result.get("usage"), started, first, "client_gone", plan.get("trace"))
            raise
        message = {"role": "assistant", "content": "".join(content) or None}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if calls:
            message["tool_calls"] = calls
        self.log(model, prompt_tokens, result.get("usage"), started, first, result.get("finish_reason"), plan.get("trace"))
        self.record_public_prefix(model, plan, result)
        _, record = self.settle(model, plan, result, message, started, request.headers, "chat")
        choice = {"index": 0, "message": message, "logprobs": {"content": logprobs} if "logprobs" in engine_body else None, "finish_reason": result.get("finish_reason")}
        return web.json_response({"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                                  "model": model.id, "choices": [choice], "usage": self.usage_block(result.get("usage"), prompt_tokens, plan),
                                  "sparkpipe": record}, headers=self.early_headers(model, plan, request.headers))

    async def chat_stream(self, request, model, body, plan, started):
        prompt_tokens = plan["prompt_tokens"]
        identifier = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        response = None
        first = None
        calls = 0
        result = {}
        content, reasoning, visible = [], [], []

        def chunk(delta, finish_reason=None, usage=None, logprobs=None, record=None):
            choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            if logprobs is not None:
                choice["logprobs"] = {"content": logprobs}
            payload = {"id": identifier, "object": "chat.completion.chunk", "created": created, "model": model.id, "choices": [choice]}
            if usage is not None:
                payload["usage"] = usage
            if record is not None:
                payload["sparkpipe"] = record
            return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()

        try:
            async for kind, value in heartbeat(self.respond(model, plan), KEEPALIVE_SECONDS):
                if response is None:
                    response = web.StreamResponse(headers=dict(self.early_headers(model, plan, request.headers), **{"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}))
                    await response.prepare(request)
                    await response.write(chunk({"role": "assistant", "content": ""}))
                if kind == "keepalive":
                    await response.write(b": keep-alive\n\n")
                    continue
                first = first or time.monotonic()
                if kind == "content":
                    content.append(value)
                elif kind == "reasoning":
                    reasoning.append(value)
                elif kind == "tool_call":
                    visible.append(value)
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
            self.log(model, prompt_tokens, result.get("usage"), started, first, "client_gone", plan.get("trace"))
            return response
        except asyncio.CancelledError:
            self.log(model, prompt_tokens, result.get("usage"), started, first, "client_gone", plan.get("trace"))
            raise
        usage = self.usage_block(result.get("usage"), prompt_tokens, plan)
        message = {"role": "assistant", "content": "".join(content) or None}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if visible:
            message["tool_calls"] = visible
        _, record = self.settle(model, plan, result, message, started, request.headers, "chat")
        await response.write(chunk({}, result.get("finish_reason"), usage, record=record))
        if include_usage:
            payload = {"id": identifier, "object": "chat.completion.chunk", "created": created, "model": model.id, "choices": [], "usage": usage, "sparkpipe": record}
            await response.write(("data: " + json.dumps(payload) + "\n\n").encode())
        await response.write(b"data: [DONE]\n\n")
        self.log(model, prompt_tokens, result.get("usage"), started, first, result.get("finish_reason"), plan.get("trace"))
        self.record_public_prefix(model, plan, result)
        return response


SERVER_OPTIONS = {"handler_cancellation": True}


def raise_open_files():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = hard if hard != resource.RLIM_INFINITY else 1 << 20
    while target > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            break
        except (ValueError, OSError):
            target //= 2
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def presented_key(headers):
    authorization = headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        return authorization[len("Bearer "):].strip()
    return headers.get("X-Api-Key", "").strip() or None


def public_door(frontend):
    @web.middleware
    async def door(request, handler):
        if frontend.public_header is None or request.headers.get(frontend.public_header) is None:
            return await handler(request)
        refuse = anthropic_error_response if request.path in ANTHROPIC_PATHS else (responses_error_response if request.path == "/v1/responses" else error_response)
        if request.path not in PUBLIC_PATHS:
            return refuse(RequestError(404, "not_found", "this path is not served to the public API"))
        if request.path in OPEN_PATHS:
            return await handler(request)
        presented = presented_key(request.headers)
        entry = frontend.public_keys.lookup(presented) if presented else None
        if entry is None:
            return refuse(RequestError(401, "invalid_api_key", "send a valid SparkPipe API key as Authorization: Bearer sk-sp-... or X-Api-Key: sk-sp-..."))
        request["public_key"] = entry
        if request.method in ("POST", "PUT"):
            if request.content_length is None:
                return refuse(RequestError(411, "length_required", "send the request body with a Content-Length"))
            limit = frontend.public_body_limit()
            if request.content_length > limit:
                return refuse(RequestError(413, "request_too_large", f"the request body is {request.content_length} bytes; the public API takes at most {limit}"))
        return await handler(request)
    return door


def build(frontend):
    app = web.Application(client_max_size=256 * 1024 * 1024, middlewares=[public_door(frontend)])
    app.on_startup.append(frontend.start)
    app.on_cleanup.append(frontend.stop)
    app.router.add_get("/health", frontend.health)
    app.router.add_get("/health/liveliness", frontend.liveliness)
    app.router.add_get("/v1/models", frontend.list_models)
    app.router.add_post("/v1/chat/completions", frontend.chat)
    app.router.add_post("/v1/messages", frontend.messages)
    app.router.add_post("/v1/messages/count_tokens", frontend.count_tokens)
    app.router.add_post("/v1/responses", frontend.responses)
    app.router.add_post("/v1/tokenize", frontend.tokenize)
    app.router.add_post("/v1/detokenize", frontend.detokenize)
    app.router.add_get("/v1/models/aliases", frontend.get_aliases)
    app.router.add_put("/v1/models/aliases", frontend.put_aliases)
    app.router.add_get("/v1/codex/models", frontend.codex_models)
    app.router.add_delete("/v1/cache", frontend.delete_cache)
    app.router.add_get("/api/hello", frontend.hello)
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
        raise SystemExit(f"chat_frontend: --host {arguments.host} is not a loopback address; the chat layer checks API keys only on requests the public door marks, so it serves this host only")
    config_path = Path(arguments.config).resolve()
    frontend = Frontend(json.loads(config_path.read_text()), config_path.parent)
    print(f"chat_frontend: open files limit {raise_open_files()}", file=sys.stderr, flush=True)
    web.run_app(build(frontend), host=arguments.host, port=arguments.port, access_log=None, **SERVER_OPTIONS)


if __name__ == "__main__":
    main()

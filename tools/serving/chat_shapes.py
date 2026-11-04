import json
import time
import uuid

RequestError = None

ANTHROPIC_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error",
                   409: "invalid_request_error", 411: "invalid_request_error", 413: "request_too_large", 429: "rate_limit_error",
                   500: "api_error", 502: "api_error", 503: "overloaded_error", 504: "api_error", 529: "overloaded_error"}
CONTRACT_CODES = frozenset({"fleet_required", "fleet_not_held", "model_not_loaded", "image_not_supported", "context_too_long", "busy",
                            "invalid_request", "invalid_key", "upstream_error"})
CONTRACT_RENAMES = {"context_length_exceeded": "context_too_long", "invalid_api_key": "invalid_key", "rate_limited": "busy"}
EMPTY_SCHEMA = {"type": "object", "properties": {}}
CUSTOM_INPUT = "input"
BUDGET_EFFORTS = ((4096, "low"), (16384, "medium"))
EFFORT_FALLBACKS = {"max": ("high",), "xhigh": ("high",), "minimal": ("minimal", "low"), "none": ("minimal", "low")}
SIGNATURE_PREFIX = "sp1."


def handle(prefix):
    return prefix + uuid.uuid4().hex[:24]


def sse(event, data):
    return ("event: " + event + "\ndata: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode()


def invalid(message, code="invalid_request", status=400):
    return RequestError(status, code, message)


def contract_code(code):
    return CONTRACT_RENAMES.get(code, code)


def anthropic_error_body(error):
    code = contract_code(error.code)
    kind = code if code in CONTRACT_CODES else ANTHROPIC_TYPES.get(error.status, "api_error" if error.status >= 500 else "invalid_request_error")
    detail = {"type": kind, "message": error.message}
    detail.update(getattr(error, "details", None) or {})
    return {"type": "error", "error": detail}


def openai_error_body(error):
    detail = {"message": error.message, "type": "server_error" if error.status >= 500 else "invalid_request_error",
              "code": contract_code(error.code), "param": None}
    detail.update(getattr(error, "details", None) or {})
    return {"error": detail}


def effort_for(labels, effort, thinking=None, budget=None):
    if isinstance(effort, str):
        if effort in labels:
            return effort
        for fallback in EFFORT_FALLBACKS.get(effort, ()):
            if fallback in labels:
                return fallback
        return None
    if thinking == "disabled":
        return next((label for label in ("none", "minimal") if label in labels), None)
    if thinking == "enabled" and isinstance(budget, int) and not isinstance(budget, bool):
        label = next((name for limit, name in BUDGET_EFFORTS if budget <= limit), "high")
        return label if label in labels else None
    return None


def joined(parts):
    if all(part.get("type") == "text" for part in parts):
        return "\n\n".join(part["text"] for part in parts)
    return parts


def block_list(content, where):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if content is None:
        return []
    if not isinstance(content, list) or not all(isinstance(block, dict) for block in content):
        raise invalid(f"{where} must be a string or an array of content blocks")
    return content


def text_only(content, where):
    texts = []
    for block in block_list(content, where):
        if block.get("type") != "text":
            raise invalid(f"{where} takes text blocks only, not {block.get('type')!r}")
        texts.append(str(block.get("text") or ""))
    return "\n\n".join(texts)


def image_part(block, request):
    source = block.get("source") or {}
    request["images"] = True
    if source.get("type") == "base64":
        return {"type": "image_url", "image_url": {"url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"}}
    if source.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url", "")}}
    raise invalid("an image block needs a base64 or url source")


def tool_result_content(content, request):
    parts = []
    for block in block_list(content, "tool_result content"):
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": str(block.get("text") or "")})
        elif kind == "image":
            parts.append(image_part(block, request))
        elif kind == "tool_reference":
            raise invalid("tool_reference blocks (MCP tool search) are not supported; turn that beta off")
        else:
            raise invalid(f"tool_result content takes text and image blocks, not {kind!r}")
    return joined(parts) if parts else ""


def assistant_from_blocks(content):
    texts, reasoning, calls = [], [], []
    for block in block_list(content, "assistant content"):
        kind = block.get("type")
        if kind == "text":
            texts.append(str(block.get("text") or ""))
        elif kind == "thinking":
            reasoning.append(str(block.get("thinking") or ""))
        elif kind == "redacted_thinking":
            continue
        elif kind == "tool_use":
            arguments = block.get("input")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict) or not isinstance(block.get("name"), str) or not isinstance(block.get("id"), str):
                raise invalid("a tool_use block needs a string id, a string name and an object input")
            calls.append({"id": block["id"], "type": "function", "function": {"name": block["name"], "arguments": arguments}})
        else:
            raise invalid(f"assistant content takes text, thinking and tool_use blocks, not {kind!r}")
    message = {"role": "assistant", "content": "".join(texts)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if calls:
        message["tool_calls"] = calls
    return message


def user_from_blocks(content, request):
    out, parts = [], []

    def flush():
        if parts:
            out.append({"role": "user", "content": joined(list(parts))})
            parts.clear()

    blocks = block_list(content, "user content")
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": str(block.get("text") or "")})
        elif kind == "image":
            parts.append(image_part(block, request))
        elif kind == "document":
            source = block.get("source") or {}
            if source.get("type") != "text":
                raise invalid("a document block is accepted with a text source only")
            parts.append({"type": "text", "text": str(source.get("data") or "")})
        elif kind == "tool_result":
            flush()
            if not isinstance(block.get("tool_use_id"), str):
                raise invalid("a tool_result block needs a string tool_use_id")
            out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": tool_result_content(block.get("content"), request)})
        elif kind == "tool_reference":
            raise invalid("tool_reference blocks (MCP tool search) are not supported; turn that beta off")
        else:
            raise invalid(f"user content takes text, image, document and tool_result blocks, not {kind!r}")
    flush()
    if not blocks:
        out.append({"role": "user", "content": ""})
    return out


def anthropic_tools(tools):
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise invalid("tools must be an array")
    internal = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise invalid("every tool must be an object")
        if tool.get("defer_loading"):
            raise invalid("defer_loading (MCP tool search) is not supported; turn that beta off")
        if tool.get("type") not in (None, "custom"):
            continue
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise invalid("every tool needs a string name")
        schema = tool.get("input_schema")
        internal.append({"type": "function", "function": {"name": tool["name"], "description": str(tool.get("description") or ""),
                                                          "parameters": schema if isinstance(schema, dict) else dict(EMPTY_SCHEMA)}})
    return internal


def anthropic_tool_choice(choice):
    if choice is None:
        return "auto"
    if not isinstance(choice, dict) or choice.get("type") not in ("auto", "any", "tool", "none"):
        raise invalid("tool_choice type must be auto, any, tool or none")
    if choice["type"] == "tool":
        if not isinstance(choice.get("name"), str):
            raise invalid("tool_choice of type tool needs a name")
        return {"name": choice["name"]}
    return {"auto": "auto", "any": "required", "none": "none"}[choice["type"]]


def stop_list(value, name):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(stop, str) and stop for stop in value):
        raise invalid(f"{name} must be an array of non-empty strings")
    return value


def sampling(body):
    return {key: body[key] for key in ("temperature", "top_p", "top_k", "seed") if body.get(key) is not None}


def messages_request(body):
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise invalid("model must be a string")
    request = {"images": False}
    messages = []
    system = body.get("system")
    if system is not None:
        text = text_only(system, "system")
        if text:
            messages.append({"role": "system", "content": text})
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise invalid("messages must be a non-empty array")
    for message in body["messages"]:
        if not isinstance(message, dict):
            raise invalid("every message must be an object")
        role = message.get("role")
        if role == "system":
            messages.append({"role": "system", "content": text_only(message.get("content"), "a system message")})
        elif role == "assistant":
            messages.append(assistant_from_blocks(message.get("content")))
        elif role == "user":
            messages.extend(user_from_blocks(message.get("content"), request))
        else:
            raise invalid("message roles must be user, assistant or system")
    thinking = body.get("thinking")
    thinking = thinking if isinstance(thinking, dict) else {}
    output = body.get("output_config")
    output = output if isinstance(output, dict) else {}
    if body.get("max_tokens") is not None and (isinstance(body["max_tokens"], bool) or not isinstance(body["max_tokens"], int) or body["max_tokens"] < 1):
        raise invalid("max_tokens must be a positive integer")
    if body.get("stream") not in (None, True, False):
        raise invalid("stream must be a boolean")
    request.update(model=body["model"], messages=messages, tools=anthropic_tools(body.get("tools")), tool_choice=anthropic_tool_choice(body.get("tool_choice")),
                   effort=output.get("effort"), thinking=thinking.get("type"), budget=thinking.get("budget_tokens"),
                   max_tokens=body.get("max_tokens"), stops=stop_list(body.get("stop_sequences"), "stop_sequences"), sampling=sampling(body),
                   stream=bool(body.get("stream")), custom_tools=frozenset())
    return request


def responses_text(content, request, where):
    parts = []
    for part in block_list(content, where):
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": str(part.get("text") or "")})
        elif kind == "input_image":
            request["images"] = True
            url = part.get("image_url")
            if not isinstance(url, str):
                raise invalid("an input_image part needs an image_url")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif kind == "refusal":
            parts.append({"type": "text", "text": str(part.get("refusal") or "")})
        else:
            raise invalid(f"{where} takes input_text, output_text and input_image parts, not {kind!r}")
    return joined(parts) if parts else ""


def output_text(output, request):
    if isinstance(output, str):
        return output
    return responses_text(output, request, "function_call_output output")


def call_arguments(arguments):
    if isinstance(arguments, dict):
        return arguments
    if arguments in (None, ""):
        return {}
    try:
        parsed = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        raise invalid("function_call arguments must be a JSON object")
    if not isinstance(parsed, dict):
        raise invalid("function_call arguments must be a JSON object")
    return parsed


def reasoning_text(item):
    content = [part.get("text", "") for part in item.get("content") or [] if isinstance(part, dict) and part.get("type") == "reasoning_text"]
    if content:
        return "".join(content)
    return "\n\n".join(part.get("text", "") for part in item.get("summary") or [] if isinstance(part, dict))


class Turn:
    def __init__(self):
        self.texts, self.reasoning, self.calls = [], [], []

    def message(self):
        message = {"role": "assistant", "content": "".join(self.texts)}
        if self.reasoning:
            message["reasoning_content"] = "\n\n".join(self.reasoning)
        if self.calls:
            message["tool_calls"] = self.calls
        return message


def responses_messages(items, request):
    out, turn = [], None

    def flush():
        nonlocal turn
        if turn is not None:
            out.append(turn.message())
            turn = None

    for item in items:
        if not isinstance(item, dict):
            raise invalid("every input item must be an object")
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            role = item.get("role")
            if role == "assistant":
                turn = turn or Turn()
                turn.texts.append(responses_text(item.get("content"), request, "an assistant message"))
                continue
            if role not in ("user", "system", "developer"):
                raise invalid("message roles must be user, assistant, system or developer")
            flush()
            out.append({"role": role, "content": responses_text(item.get("content"), request, f"a {role} message")})
        elif kind == "reasoning":
            if turn is not None and (turn.texts or turn.calls):
                flush()
            turn = turn or Turn()
            text = reasoning_text(item)
            if text:
                turn.reasoning.append(text)
        elif kind in ("function_call", "custom_tool_call"):
            if not isinstance(item.get("call_id"), str) or not isinstance(item.get("name"), str):
                raise invalid(f"a {kind} item needs a string call_id and name")
            arguments = call_arguments(item.get("arguments")) if kind == "function_call" else {CUSTOM_INPUT: str(item.get("input") or "")}
            turn = turn or Turn()
            turn.calls.append({"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": arguments}})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            flush()
            if not isinstance(item.get("call_id"), str):
                raise invalid(f"a {kind} item needs a string call_id")
            out.append({"role": "tool", "tool_call_id": item["call_id"], "content": output_text(item.get("output"), request)})
        else:
            raise invalid(f"input items of type {kind!r} are not supported; send message, reasoning, function_call and function_call_output items")
    flush()
    return out


def responses_tools(tools):
    if tools is None:
        return None, frozenset()
    if not isinstance(tools, list):
        raise invalid("tools must be an array")
    internal, custom = [], set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise invalid("every tool must be an object")
        kind = tool.get("type")
        if kind not in ("function", "custom"):
            continue
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise invalid("every function tool needs a string name")
        description = str(tool.get("description") or "")
        if kind == "function":
            parameters = tool.get("parameters")
            parameters = parameters if isinstance(parameters, dict) else dict(EMPTY_SCHEMA)
        else:
            custom.add(tool["name"])
            grammar = tool.get("format") if isinstance(tool.get("format"), dict) else {}
            if isinstance(grammar.get("definition"), str):
                description += "\n\nThe input follows this " + str(grammar.get("syntax") or "grammar") + " grammar:\n" + grammar["definition"]
            parameters = {"type": "object", "properties": {CUSTOM_INPUT: {"type": "string"}}, "required": [CUSTOM_INPUT]}
        internal.append({"type": "function", "function": {"name": tool["name"], "description": description, "parameters": parameters}})
    return internal, frozenset(custom)


def responses_tool_choice(choice):
    if choice in (None, "auto"):
        return "auto"
    if choice in ("none", "required"):
        return choice
    if isinstance(choice, dict) and choice.get("type") in ("function", "custom") and isinstance(choice.get("name"), str):
        return {"name": choice["name"]}
    if isinstance(choice, dict) and choice.get("type") in ("auto", "none", "required"):
        return choice["type"]
    raise invalid("tool_choice must be auto, none, required or a function by name")


def responses_request(body):
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise invalid("model must be a string")
    request = {"images": False}
    source = body.get("input")
    if isinstance(source, str):
        items = [{"type": "message", "role": "user", "content": source}]
    elif isinstance(source, list):
        items = source
    elif source is None:
        items = []
    else:
        raise invalid("input must be a string or an array of items")
    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise invalid("instructions must be a string")
    text = body.get("text")
    text_format = (text or {}).get("format") if isinstance(text, dict) else None
    if isinstance(text_format, dict) and text_format.get("type") not in (None, "text"):
        raise invalid("text.format other than text is not supported yet", "unsupported_parameter")
    reasoning = body.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, dict) else {}
    limit = body.get("max_output_tokens")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise invalid("max_output_tokens must be a positive integer")
    previous = body.get("previous_response_id")
    if previous is not None and not isinstance(previous, str):
        raise invalid("previous_response_id must be a string")
    if body.get("stream") not in (None, True, False):
        raise invalid("stream must be a boolean")
    tools, custom = responses_tools(body.get("tools"))
    request.update(model=body["model"], input_messages=responses_messages(items, request), instructions=instructions or None,
                   tools=tools, custom_tools=custom, tool_choice=responses_tool_choice(body.get("tool_choice")),
                   effort=reasoning.get("effort"), thinking=None, budget=None, max_tokens=limit, stops=[], sampling=sampling(body),
                   stream=bool(body.get("stream")), previous_response_id=previous, store=body.get("store") is not False,
                   affinity=body.get("prompt_cache_key") if isinstance(body.get("prompt_cache_key"), str) else None)
    return request


class MessagesTurn:
    def __init__(self, model_name, prompt_tokens):
        self.id = handle("msg_")
        self.model_name = model_name
        self.prompt_tokens = prompt_tokens
        self.blocks = []
        self.current = None
        self.pending = ""

    def start(self):
        usage = {"input_tokens": self.prompt_tokens, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}
        return [sse("message_start", {"type": "message_start", "message": {"id": self.id, "type": "message", "role": "assistant", "model": self.model_name,
                                                                           "content": [], "stop_reason": None, "stop_sequence": None, "usage": usage}})]

    def keepalive(self):
        return sse("ping", {"type": "ping"})

    def open(self, block, frames):
        self.close(frames)
        self.blocks.append(block)
        self.current = block
        shown = dict(block)
        if shown["type"] == "thinking":
            shown.update(thinking="", signature="")
        elif shown["type"] == "text":
            shown["text"] = ""
        elif shown["type"] == "tool_use":
            shown["input"] = {}
        frames.append(sse("content_block_start", {"type": "content_block_start", "index": len(self.blocks) - 1, "content_block": shown}))

    def delta(self, delta, frames):
        frames.append(sse("content_block_delta", {"type": "content_block_delta", "index": len(self.blocks) - 1, "delta": delta}))

    def close(self, frames):
        block, self.current = self.current, None
        if block is None:
            return
        if block["type"] == "thinking":
            block["signature"] = handle(SIGNATURE_PREFIX)
            self.delta({"type": "signature_delta", "signature": block["signature"]}, frames)
        frames.append(sse("content_block_stop", {"type": "content_block_stop", "index": len(self.blocks) - 1}))

    def feed(self, kind, value):
        frames = []
        if kind == "reasoning" and value:
            if self.current is None or self.current["type"] != "thinking":
                self.open({"type": "thinking", "thinking": ""}, frames)
            self.current["thinking"] += value
            self.delta({"type": "thinking_delta", "thinking": value}, frames)
        elif kind == "content" and value:
            if self.current is not None and self.current["type"] == "text":
                self.current["text"] += value
                self.delta({"type": "text_delta", "text": value}, frames)
            else:
                self.pending += value
                if self.pending.strip():
                    text, self.pending = self.pending, ""
                    self.open({"type": "text", "text": text}, frames)
                    self.delta({"type": "text_delta", "text": text}, frames)
        elif kind == "tool_call":
            self.pending = ""
            function = value["function"]
            arguments = function["arguments"]
            self.open({"type": "tool_use", "id": handle("toolu_"), "name": function["name"], "input": json.loads(arguments)}, frames)
            self.delta({"type": "input_json_delta", "partial_json": arguments}, frames)
            self.close(frames)
        return frames

    def finish(self, stop_reason, stop_sequence, usage, telemetry):
        frames = []
        self.close(frames)
        frames.append(sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                                            "usage": usage, "sparkpipe": telemetry}))
        frames.append(sse("message_stop", {"type": "message_stop"}))
        return frames

    def failure(self, error):
        return [sse("error", anthropic_error_body(error))]

    def body(self, stop_reason, stop_sequence, usage, telemetry):
        self.close([])
        return {"id": self.id, "type": "message", "role": "assistant", "model": self.model_name, "content": self.blocks,
                "stop_reason": stop_reason, "stop_sequence": stop_sequence, "usage": usage, "sparkpipe": telemetry}

    def assistant(self):
        return assistant_from_blocks(self.blocks)


def anthropic_usage(usage):
    return {"input_tokens": usage["prompt"] - usage["cached"] - usage["committed"], "cache_creation_input_tokens": usage["committed"],
            "cache_read_input_tokens": usage["cached"], "output_tokens": usage["completion"]}


def anthropic_stop(result, stop_marker):
    if result.get("finish_reason") == "tool_calls":
        return "tool_use", None
    if result.get("engine_finish") == "length":
        return "max_tokens", None
    if result.get("engine_finish") == "stop_string":
        return "stop_sequence", stop_marker
    return "end_turn", None


class ResponsesTurn:
    def __init__(self, model_name, request, echo):
        self.id = handle("resp_")
        self.model_name = model_name
        self.created = int(time.time())
        self.custom = request["custom_tools"]
        self.echo = echo
        self.items = []
        self.current = None
        self.pending = ""
        self.sequence = 0

    def frame(self, kind, payload):
        payload = dict(payload, type=kind, sequence_number=self.sequence)
        self.sequence += 1
        return sse(kind, payload)

    def response(self, status, usage=None, telemetry=None, error=None, incomplete=None):
        body = {"id": self.id, "object": "response", "created_at": self.created, "status": status, "model": self.model_name,
                "output": [dict(item) for item in self.items if status != "in_progress" or item.get("status") == "completed"],
                "usage": usage, "error": error, "incomplete_details": incomplete}
        body.update(self.echo)
        if telemetry is not None:
            body["sparkpipe"] = telemetry
        return body

    def start(self):
        return [self.frame("response.created", {"response": self.response("in_progress")}),
                self.frame("response.in_progress", {"response": self.response("in_progress")})]

    def keepalive(self):
        return b": keep-alive\n\n"

    def open(self, item, frames):
        self.close(frames)
        self.items.append(item)
        self.current = item
        shown = dict(item)
        if item["type"] == "reasoning":
            shown["summary"] = []
            shown.pop("encrypted_content", None)
        elif item["type"] == "message":
            shown["content"] = []
        elif item["type"] == "function_call":
            shown["arguments"] = ""
        frames.append(self.frame("response.output_item.added", {"output_index": len(self.items) - 1, "item": shown}))

    def close(self, frames):
        item, self.current = self.current, None
        if item is None:
            return
        index = len(self.items) - 1
        if item["type"] == "reasoning":
            text = item["summary"][0]["text"]
            item["encrypted_content"] = handle(SIGNATURE_PREFIX)
            frames.append(self.frame("response.reasoning_summary_text.done", {"item_id": item["id"], "output_index": index, "summary_index": 0, "text": text}))
            frames.append(self.frame("response.reasoning_summary_part.done", {"item_id": item["id"], "output_index": index, "summary_index": 0,
                                                                               "part": {"type": "summary_text", "text": text}}))
        elif item["type"] == "message":
            part = item["content"][0]
            frames.append(self.frame("response.output_text.done", {"item_id": item["id"], "output_index": index, "content_index": 0, "text": part["text"]}))
            frames.append(self.frame("response.content_part.done", {"item_id": item["id"], "output_index": index, "content_index": 0, "part": part}))
        item["status"] = "completed"
        frames.append(self.frame("response.output_item.done", {"output_index": index, "item": item}))

    def feed(self, kind, value):
        frames = []
        if kind == "reasoning" and value:
            if self.current is None or self.current["type"] != "reasoning":
                self.open({"type": "reasoning", "id": handle("rs_"), "summary": [{"type": "summary_text", "text": ""}], "status": "in_progress"}, frames)
                frames.append(self.frame("response.reasoning_summary_part.added", {"item_id": self.current["id"], "output_index": len(self.items) - 1,
                                                                                    "summary_index": 0, "part": {"type": "summary_text", "text": ""}}))
            self.current["summary"][0]["text"] += value
            frames.append(self.frame("response.reasoning_summary_text.delta", {"item_id": self.current["id"], "output_index": len(self.items) - 1,
                                                                                "summary_index": 0, "delta": value}))
        elif kind == "content" and value:
            if self.current is None or self.current["type"] != "message":
                self.pending += value
                if not self.pending.strip():
                    return frames
                value, self.pending = self.pending, ""
                self.open({"type": "message", "id": handle("msg_"), "status": "in_progress", "role": "assistant",
                           "content": [{"type": "output_text", "text": "", "annotations": []}]}, frames)
                frames.append(self.frame("response.content_part.added", {"item_id": self.current["id"], "output_index": len(self.items) - 1,
                                                                          "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}))
            self.current["content"][0]["text"] += value
            frames.append(self.frame("response.output_text.delta", {"item_id": self.current["id"], "output_index": len(self.items) - 1,
                                                                     "content_index": 0, "delta": value}))
        elif kind == "tool_call":
            self.pending = ""
            function = value["function"]
            call_id = handle("call_")
            if function["name"] in self.custom:
                text = str(json.loads(function["arguments"]).get(CUSTOM_INPUT, ""))
                self.open({"type": "custom_tool_call", "id": handle("ctc_"), "call_id": call_id, "name": function["name"], "input": text,
                           "status": "in_progress"}, frames)
            else:
                self.open({"type": "function_call", "id": handle("fc_"), "call_id": call_id, "name": function["name"],
                           "arguments": function["arguments"], "status": "in_progress"}, frames)
                index = len(self.items) - 1
                frames.append(self.frame("response.function_call_arguments.delta", {"item_id": self.current["id"], "output_index": index,
                                                                                     "delta": function["arguments"]}))
                frames.append(self.frame("response.function_call_arguments.done", {"item_id": self.current["id"], "output_index": index,
                                                                                    "arguments": function["arguments"]}))
            self.close(frames)
        return frames

    def finish(self, incomplete, usage, telemetry):
        frames = []
        self.close(frames)
        status = "incomplete" if incomplete else "completed"
        frames.append(self.frame("response.completed", {"response": self.response(status, usage, telemetry, incomplete=incomplete)}))
        return frames

    def failure(self, error):
        body = openai_error_body(error)["error"]
        return [self.frame("error", {"code": body["code"], "message": body["message"], "param": None}),
                self.frame("response.failed", {"response": self.response("failed", error={"code": body["code"], "message": body["message"]})})]

    def body(self, incomplete, usage, telemetry):
        self.close([])
        return self.response("incomplete" if incomplete else "completed", usage, telemetry, incomplete=incomplete)

    def assistant(self):
        scratch = {"images": False}
        return responses_messages(self.items, scratch)[0] if self.items else {"role": "assistant", "content": ""}


def responses_usage(usage):
    return {"input_tokens": usage["prompt"], "input_tokens_details": {"cached_tokens": usage["cached"]}, "output_tokens": usage["completion"],
            "output_tokens_details": {"reasoning_tokens": usage["reasoning"]}, "total_tokens": usage["prompt"] + usage["completion"]}


def responses_incomplete(result):
    return {"reason": "max_output_tokens"} if result.get("engine_finish") == "length" and result.get("finish_reason") != "tool_calls" else None

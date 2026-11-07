import asyncio
import importlib.util
import json
import re
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
HEADER = "X-SparkPipe-Public"
SPECIAL = ["<|im_start|>", "<|im_end|>"]
MARKERS = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"]
MARKER_SPLIT = re.compile("(" + "|".join(re.escape(marker) for marker in SPECIAL + MARKERS) + ")")
TEMPLATE = (
    "{%- if tools %}<|im_start|>system\n# Tools\n{% for tool in tools %}{{ tool | tojson }}\n{% endfor %}<|im_end|>\n{% endif %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'assistant' %}<|im_start|>assistant\n{{ message.content }}"
    "{%- for call in message.tool_calls or [] %}<tool_call>{{ call.function.name }}"
    "{%- for key, value in call.function.arguments.items() %}<arg_key>{{ key }}</arg_key><arg_value>{{ value if value is string else value | tojson }}</arg_value>{% endfor %}"
    "</tool_call>{% endfor %}<|im_end|>\n"
    "{% elif message.role == 'tool' %}<|im_start|>tool\n{{ message.content }}<|im_end|>\n"
    "{% else %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{% endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n<think>{% endif %}"
)
CODEX_INSTRUCTIONS = "You are a coding agent in the Codex CLI."
CORPUS = ["hello world, the weather in Paris is sunny today", "read the file a.txt and list the directory", "call the tool with path and pattern",
          "user assistant system tool think reply answer question", "The answer is 42. Let me check the file first."] * 20


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chat_frontend = load("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
api_keys = load("api_keys", ROOT / "tools" / "serving" / "api_keys.py")


def model_directory(directory):
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.trainers import BpeTrainer
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(CORPUS, BpeTrainer(vocab_size=600, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tokenizer.add_tokens([AddedToken(marker, special=False, normalized=False) for marker in MARKERS])
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    (directory / "chat_template.jinja").write_text(TEMPLATE)
    return directory


def model_spec(model_id, engine, sequences=2):
    return {"id": model_id, "model_dir": "model", "engine": engine, "context_tokens": 65536, "max_output_tokens": 4096,
            "sampling_defaults": {}, "sampling_source": "test", "warm_queue": {"depth": 8, "wait_seconds": 5, "activation_url": None},
            "sequences": sequences, "stop_tokens": ["<|im_end|>"], "template_defaults": {}, "template_kwargs": [],
            "reasoning_effort": {"low": {"effort": "low"}, "medium": {"effort": "medium"}, "high": {"effort": "high"}},
            "reasoning": {"start": "<think>", "end": "</think>"},
            "tool_calls": {"start": "<tool_call>", "end": "</tool_call>", "key_start": "<arg_key>", "key_end": "</arg_key>",
                           "value_start": "<arg_value>", "value_end": "</arg_value>"}}


def common_prefix(left, right):
    size = 0
    for a, b in zip(left, right):
        if a != b:
            break
        size += 1
    return size


class Engine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.replies = []
        self.requests = []
        self.committed = []
        self.delay = 0.0
        self.hold = None
        self.early_usage = True
        self.cache_all = False
        self.usage = []

    def tokens(self, text):
        ids = []
        for index, piece in enumerate(MARKER_SPLIT.split(text)):
            if index % 2:
                ids.append(self.tokenizer.token_to_id(piece))
            else:
                for char in piece:
                    ids.extend(self.tokenizer.encode(char, add_special_tokens=False).ids)
        return ids

    def reply(self, text, finish="stop"):
        ids = self.tokens(text)
        if finish == "stop":
            ids.append(self.tokenizer.token_to_id("<|im_end|>"))
        self.replies.append((ids, finish))

    async def completions(self, request):
        body = await request.json()
        self.requests.append(body)
        prompt = body["prompt_token_ids"]
        cached = max((common_prefix(prompt, context) for context in self.committed), default=0)
        cached = len(prompt) if self.cache_all else min(cached, len(prompt) - 1)
        self.usage.append({"prompt_tokens": len(prompt), "cached_tokens": cached})
        ids, finish = self.replies.pop(0) if self.replies else (self.tokens("ok") + [self.tokenizer.token_to_id("<|im_end|>")], "stop")
        ids = ids[:body["max_tokens"]]
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if self.hold is not None:
            await self.hold.wait()
        if self.delay:
            await asyncio.sleep(self.delay)
        for index, token in enumerate(ids):
            last = index + 1 == len(ids)
            event = {"tokens": [token], "choices": [{"finish_reason": finish if last else None}]}
            if last or (index == 0 and self.early_usage):
                sent = len(ids) if last else 1
                event["usage"] = {"prompt_tokens": len(prompt), "completion_tokens": sent, "total_tokens": len(prompt) + sent,
                                  "prompt_tokens_details": {"cached_tokens": cached}}
            await response.write(("data: " + json.dumps(event) + "\n\n").encode())
        self.committed.append(list(prompt) + list(ids[:-1]))
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def health(self, request):
        return web.json_response({"status": "ok"})


class Controller:
    def __init__(self, active, models):
        self.status = {"active": active, "state": "idle", "target": None, "wanted": None,
                       "models": [{"id": model_id, "active": model_id == active} for model_id in models]}
        self.activations = []
        self.fail = False

    async def get_status(self, request):
        if self.fail:
            return web.Response(status=502, text="Cannot connect to host 127.0.0.1:8440 ssl:default [Connect call failed]")
        return web.json_response(self.status)


async def serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


class Harness:
    def __init__(self, tmp, models=("alpha", "beta"), active="alpha", sequences=2, replay_store=True, control=True, extra=None):
        self.tmp = tmp
        self.extra = dict(extra or {})
        self.model_ids = models
        self.active = active
        self.sequences = sequences
        self.replay_store = replay_store
        self.control = control

    async def __aenter__(self):
        from tokenizers import Tokenizer
        directory = model_directory(self.tmp / "model")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.engine = Engine(self.tokenizer)
        engine_app = web.Application()
        engine_app.router.add_post("/v1/completions", self.engine.completions)
        engine_app.router.add_get("/health", self.engine.health)
        self.engine_runner, engine_base = await serve(engine_app)
        self.controller = Controller(self.active, self.model_ids)
        control_app = web.Application()
        control_app.router.add_get("/status", self.controller.get_status)
        self.control_runner, control_base = await serve(control_app)
        keys = self.tmp / "api_keys.json"
        if not keys.exists():
            self.key_id, self.key = api_keys.create(keys, 0, "harness")
            self.swap_id, self.swap_key = api_keys.create(keys, 0, "swapper", allow_swap=True)
            self.gateway_id, self.gateway_key = api_keys.create(keys, 0, "gateway", gateway=True)
            (self.tmp / "keys.json").write_text(json.dumps({"key": self.key, "swap_key": self.swap_key, "key_id": self.key_id, "gateway_key": self.gateway_key}))
        else:
            saved = json.loads((self.tmp / "keys.json").read_text())
            self.key, self.swap_key, self.key_id, self.gateway_key = saved["key"], saved["swap_key"], saved["key_id"], saved["gateway_key"]
        config = {"models": [model_spec(model_id, engine_base, self.sequences) for model_id in self.model_ids],
                  "public": {"header": HEADER, "keys_file": "api_keys.json", "models": list(self.model_ids)}}
        if self.control:
            token = self.tmp / "controller-token"
            token.write_text("t" * 40)
            token.chmod(0o600)
            config.update(controller_token_file="controller-token", control={"status_url": control_base + "/status"})
        if self.replay_store:
            config["replay"] = {"directory": "state/replay", "budget_bytes": 1 << 26, "directive": "working-state/directive"}
        (self.tmp / "codex_instructions.md").write_text(CODEX_INSTRUCTIONS)
        config["codex"] = {"instructions_file": "codex_instructions.md"}
        config.update(self.extra)
        self.frontend = chat_frontend.Frontend(config, self.tmp)
        self.front_runner, self.base = await serve(chat_frontend.build(self.frontend))
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc):
        await self.session.close()
        await self.front_runner.cleanup()
        await self.engine_runner.cleanup()
        await self.control_runner.cleanup()
        self.frontend.replay.close()

    def headers(self, swap=False, account=None, **extra):
        if account is not None:
            headers = {HEADER: "1", "Authorization": "Bearer " + self.gateway_key, "X-SparkPipe-Account": account, "X-SparkPipe-Fleet": "F1",
                       "X-SparkPipe-Allow-Swap": "1" if swap else "0"}
        else:
            headers = {HEADER: "1", "Authorization": "Bearer " + (self.swap_key if swap else self.key)}
        headers.update(extra)
        return headers

    def directive(self, wiped, held_until):
        path = self.tmp / "working-state" / "directive"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"sparkpipe-working-state 2\nwiped {int(wiped)}\nheld_until {int(held_until)}\nstorage_wiped {int(wiped)}\n")

    async def post(self, path, body, **headers):
        async with self.session.post(self.base + path, json=body, headers=self.headers(**headers)) as response:
            text = await response.text()
            return response.status, dict(response.headers), text

    def socket(self, path="/v1/responses", **headers):
        return self.session.ws_connect(self.base + path, headers=self.headers(**headers), max_msg_size=0)

    def prompt_text(self, index=-1):
        return self.tokenizer.decode(self.engine.requests[index]["prompt_token_ids"], skip_special_tokens=False)


def sse_events(text):
    events = []
    for frame in text.split("\n\n"):
        lines = [line for line in frame.split("\n") if line]
        if not lines:
            continue
        if all(line.startswith(":") for line in lines):
            events.append(("comment", None))
            continue
        name = next((line[len("event: "):] for line in lines if line.startswith("event: ")), None)
        data = next((line[len("data: "):] for line in lines if line.startswith("data: ")), None)
        events.append((name, json.loads(data) if data and data != "[DONE]" else data))
    return events


class Checks:
    def __init__(self, name):
        self.name = name
        self.failures = []

    def __call__(self, condition, message):
        if not condition:
            self.failures.append(message)

    def finish(self, summary):
        for failure in self.failures:
            print(f"FAIL {self.name}: {failure}")
        if self.failures:
            raise SystemExit(1)
        print(f"PASS {self.name}: {summary}")

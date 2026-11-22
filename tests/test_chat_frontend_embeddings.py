#!/usr/bin/env python3
"""The chat layer serves POST /v1/embeddings from an engine's embeddings call and offers the same to the memory program as the host method embed: a text is encoded with the tokens its tokenizer puts around a raw text, and token lists go to the engine as they are; a request larger than the model's batch goes to the engine in batches and the vectors come back in input order; float and base64 answers hold the engine's float32 values; dimensions reaches the engine and is bounded by the model's width; an unknown model, a missing or empty input, an input as long as the context and a malformed engine answer are refused with their own codes, and a starting engine is a 503; the public call runs at interactive priority and the memory host's at background priority; the host method answers base64 float32 vectors with the model, its width and the token count, a chat layer without an embedding model does not offer embed in its hello and answers it 404 unsupported."""
import asyncio
import base64
import importlib.util
import socket
from array import array
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
mc = chat_frontend.memory_connector
TOKEN = "k" * 48
WIDTH = 8
END = 31


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def fake_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    tokenizer = Tokenizer(WordLevel({"t%d" % index: index for index in range(32)}, unk_token="t0"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="$A t31", special_tokens=[("t31", END)])
    return tokenizer


def vector_of(ids, width):
    return array("f", [float(sum(ids) % 97 + 1), float(len(ids)), float(ids[-1])] + [0.5] * (width - 3)).tobytes()


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.mode = "ok"

    async def start(self):
        app = web.Application()
        app.router.add_post("/v1/embeddings", self.embeddings)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.port = free_port()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()
        return self

    async def embeddings(self, request):
        body = await request.json()
        self.calls.append(body)
        if self.mode == "starting":
            return web.json_response({"status": "starting", "phase": "connect"}, status=503)
        if self.mode == "malformed":
            return web.json_response({"object": "list", "data": [{"index": 0, "embedding": "AAAA"}], "usage": {"prompt_tokens": 1}})
        width = body.get("dimensions", WIDTH)
        data = [{"object": "embedding", "index": index, "embedding": base64.b64encode(vector_of(ids, width)).decode()} for index, ids in enumerate(body["input"])]
        return web.json_response({"object": "list", "data": list(reversed(data)), "model": "engine",
                                  "usage": {"prompt_tokens": sum(len(ids) for ids in body["input"]), "total_tokens": 0}})


def embedding_model(engine):
    model = chat_frontend.EmbeddingModel.__new__(chat_frontend.EmbeddingModel)
    model.id, model.engine, model.context_tokens, model.dimensions, model.batch_inputs = "embed-test", f"http://127.0.0.1:{engine.port}", 12, WIDTH, 3
    model.tokenizer = fake_tokenizer()
    return model


def frontend_with(models):
    frontend = chat_frontend.Frontend.__new__(chat_frontend.Frontend)
    frontend.models = {}
    frontend.embedding_models = {model.id: model for model in models}
    frontend.public_header = None
    return frontend


class Program:
    def __init__(self):
        self.hellos = []

    async def start(self):
        app = web.Application()
        app.router.add_post("/v1/hello", self.hello)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.port = free_port()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()
        return self

    async def hello(self, request):
        self.hellos.append(await request.json())
        return web.json_response({"protocol": mc.PROTOCOL, "name": "fake-memory", "version": "1", "methods": []})


async def host_call(session, connector, method, body, token=TOKEN):
    async with session.post("http://%s:%d/v1/%s" % (*connector.listen, method), json=body, headers={"Authorization": "Bearer " + token}) as response:
        return response.status, await response.json()


async def run():
    failures = []

    def check(condition, text):
        if not condition:
            failures.append(text)

    engine = await FakeEngine().start()
    model = embedding_model(engine)
    frontend = frontend_with([model])
    frontend.session = aiohttp.ClientSession()
    app = web.Application()
    app.router.add_post("/v1/embeddings", frontend.embeddings)
    runner = web.AppRunner(app)
    await runner.setup()
    port = free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    client = aiohttp.ClientSession()

    async def call(body):
        async with client.post(f"http://127.0.0.1:{port}/v1/embeddings", json=body) as response:
            return response.status, await response.json()

    status, answer = await call({"model": "embed-test", "input": "t1 t2 t3"})
    check(status == 200 and engine.calls[-1]["input"] == [[1, 2, 3, END]], f"a text is encoded with its tokenizer's end token: {engine.calls[-1:]}")
    check(engine.calls[-1].get("priority") == chat_frontend.INTERACTIVE_PRIORITY and engine.calls[-1]["encoding_format"] == "base64",
          "the public call asks the engine at interactive priority for base64")
    check(answer["object"] == "list" and answer["model"] == "embed-test" and answer["usage"] == {"prompt_tokens": 4, "total_tokens": 4}, f"usage counts the tokens: {answer}")
    check(answer["data"][0]["embedding"] == array("f", vector_of([1, 2, 3, END], WIDTH)).tolist(), "float answers hold the engine's float32 values")

    engine.calls.clear()
    inputs = ["t1", [4, 5], "t2 t2", [6], "t3 t4 t5", [7, 8, 9], "t9"]
    expected = [[1, END], [4, 5], [2, 2, END], [6], [3, 4, 5, END], [7, 8, 9], [9, END]]
    status, answer = await call({"input": inputs, "encoding_format": "base64"})
    check(status == 200 and [body["input"] for body in engine.calls] == [expected[0:3], expected[3:6], expected[6:7]], f"seven inputs go to the engine in batches of three: {engine.calls}")
    check([item["index"] for item in answer["data"]] == list(range(7)) and
          [base64.b64decode(item["embedding"]) for item in answer["data"]] == [vector_of(ids, WIDTH) for ids in expected], "vectors come back in input order")
    check(answer["usage"]["prompt_tokens"] == sum(len(ids) for ids in expected), "usage sums the batches")

    status, answer = await call({"input": [[1, 2, 3]], "dimensions": 4})
    check(status == 200 and engine.calls[-1]["dimensions"] == 4 and len(answer["data"][0]["embedding"]) == 4, "dimensions reaches the engine")
    status, answer = await call({"input": [1, 2, 3]})
    check(status == 200 and engine.calls[-1]["input"] == [[1, 2, 3]] and len(answer["data"]) == 1, "a flat list of token ids is one input")
    for body, code in (({"input": "t1", "dimensions": WIDTH + 1}, "invalid_request"), ({"input": "t1", "dimensions": 0}, "invalid_request"),
                       ({"input": "t1", "encoding_format": "hex"}, "invalid_request"), ({}, "invalid_request"), ({"input": []}, "invalid_request"),
                       ({"input": [""]}, "invalid_request"), ({"input": [[]]}, "invalid_request"), ({"input": [[1, -2]]}, "invalid_request"),
                       ({"input": [{"text": "t1"}]}, "invalid_request")):
        status, answer = await call(body)
        check(status == 400 and answer["error"]["code"] == code, f"{body} is a 400 {code}: {status} {answer}")
    status, answer = await call({"input": [list(range(12))]})
    check(status == 400 and answer["error"]["code"] == "context_length_exceeded", f"an input as long as the context is refused: {status} {answer}")
    status, answer = await call({"input": [list(range(11))]})
    check(status == 200, "an input one token under the context is embedded")
    status, answer = await call({"model": "other", "input": "t1"})
    check(status == 404 and answer["error"]["code"] == "model_not_found", f"an unknown embedding model is a 404: {status} {answer}")
    engine.mode = "starting"
    status, answer = await call({"input": "t1"})
    check(status == 503 and answer["error"]["code"] == "engine_starting", f"a starting engine is a 503: {status} {answer}")
    engine.mode = "malformed"
    status, answer = await call({"input": "t1"})
    check(status == 502 and answer["error"]["code"] == "engine_error", f"a malformed engine answer is a 502: {status} {answer}")
    engine.mode = "ok"

    program = await Program().start()
    listen = free_port()
    token_file = Path(ROOT / "build" / "test_chat_frontend_embeddings.token")
    token_file.parent.mkdir(exist_ok=True)
    token_file.write_text(TOKEN)
    token_file.chmod(0o600)
    settings = {"url": f"http://127.0.0.1:{program.port}", "host_listen": f"127.0.0.1:{listen}", "token_file": str(token_file)}
    connector = mc.Connector(settings, ROOT, chat_frontend.MemoryHost(frontend), lambda path: Path(path).read_text().strip())
    await connector.start()
    check("embed" in program.hellos[-1]["host_methods"], f"a chat layer with an embedding model offers embed: {program.hellos[-1]}")
    engine.calls.clear()
    status, answer = await host_call(client, connector, "embed", {"texts": ["t1 t2", "t3"]})
    check(status == 200 and answer["model"] == "embed-test" and answer["dimensions"] == WIDTH and answer["tokens"] == 5, f"embed answers the model, width and tokens: {status} {answer}")
    check([base64.b64decode(vector) for vector in answer.get("vectors", [])] == [vector_of([1, 2, END], WIDTH), vector_of([3, END], WIDTH)], "embed answers base64 float32 vectors in text order")
    check(engine.calls[-1].get("priority") == chat_frontend.BACKGROUND_PRIORITY, "the memory host embeds at background priority")
    status, answer = await host_call(client, connector, "embed", {"texts": ["t1"], "model": "embed-test", "dimensions": 4})
    check(status == 200 and answer["dimensions"] == 4 and len(base64.b64decode(answer["vectors"][0])) == 16, "embed takes a named model and dimensions")
    status, answer = await host_call(client, connector, "embed", {"texts": []})
    check(status == 400 and answer["error"]["code"] == "invalid_request", f"embed without texts is a 400: {status} {answer}")
    status, answer = await host_call(client, connector, "embed", {"texts": ["t1"], "model": "other"})
    check(status == 404 and answer["error"]["code"] == "model_not_found", f"embed with an unknown model is a 404: {status} {answer}")
    status, answer = await host_call(client, connector, "embed", {"texts": ["t1"]}, token="x" * 48)
    check(status == 401, "embed needs the memory token")
    await connector.stop()

    bare = frontend_with([])
    bare.session = frontend.session
    listen = free_port()
    connector = mc.Connector(dict(settings, host_listen=f"127.0.0.1:{listen}"), ROOT, chat_frontend.MemoryHost(bare), lambda path: Path(path).read_text().strip())
    await connector.start()
    check("embed" not in program.hellos[-1]["host_methods"] and "count" in program.hellos[-1]["host_methods"], f"a chat layer without an embedding model does not offer embed: {program.hellos[-1]}")
    status, answer = await host_call(client, connector, "embed", {"texts": ["t1"]})
    check(status == 404 and answer["error"]["code"] == "unsupported", f"embed is 404 unsupported there: {status} {answer}")
    await connector.stop()
    token_file.unlink()

    await client.close()
    await frontend.session.close()
    await runner.cleanup()
    await program.runner.cleanup()
    await engine.runner.cleanup()
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS chat frontend embeddings: texts and token lists, batches in order, float and base64, dimensions, refusals, priorities, and the memory host's embed")


if __name__ == "__main__":
    asyncio.run(run())

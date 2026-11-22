import asyncio
import base64
import importlib.util
import json
import socket
import sys
from pathlib import Path

sys.path.append(sys.argv[4])
import numpy as np
import aiohttp
from aiohttp import web

repo, service, model_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SPEC = importlib.util.spec_from_file_location("chat_frontend", repo / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
mc = chat_frontend.memory_connector
TOKEN = "k" * 48


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


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
        return web.json_response({"protocol": mc.PROTOCOL, "name": "live-check", "version": "1", "methods": []})


def vectors(items):
    return np.stack([np.frombuffer(base64.b64decode(item), dtype="<f4") for item in items])


async def run():
    data = json.loads((service / "check" / "corpus.json").read_text())
    direct = np.load(service / "check" / "ours.npy")
    model = chat_frontend.EmbeddingModel({"id": "qwen3-embedding-8b", "engine": "http://127.0.0.1:8491", "context_tokens": 2048, "dimensions": 4096,
                                          "model_dir": str(model_dir), "batch_inputs": 16}, Path("/"))
    frontend = chat_frontend.Frontend.__new__(chat_frontend.Frontend)
    frontend.models = {}
    frontend.embedding_models = {model.id: model}
    frontend.public_header = None
    frontend.session = aiohttp.ClientSession()
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_post("/v1/embeddings", frontend.embeddings)
    runner = web.AppRunner(app)
    await runner.setup()
    port = free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    client = aiohttp.ClientSession()
    texts = data["texts"]
    async with client.post(f"http://127.0.0.1:{port}/v1/embeddings", json={"model": "qwen3-embedding-8b", "input": texts, "encoding_format": "base64"}) as response:
        status, answer = response.status, await response.json()
    public = vectors([item["embedding"] for item in sorted(answer["data"], key=lambda item: item["index"])]) if status == 200 else None
    program = await Program().start()
    token_file = service / "check" / "memory.token"
    token_file.write_text(TOKEN)
    token_file.chmod(0o600)
    settings = {"url": f"http://127.0.0.1:{program.port}", "host_listen": f"127.0.0.1:{free_port()}", "token_file": str(token_file)}
    connector = mc.Connector(settings, Path("/"), chat_frontend.MemoryHost(frontend), lambda path: Path(path).read_text().strip())
    await connector.start()
    async with client.post("http://%s:%d/v1/embed" % connector.listen, json={"texts": texts[:64]}, headers={"Authorization": "Bearer " + TOKEN}) as response:
        host_status, host_answer = response.status, await response.json()
    hosted = vectors(host_answer["vectors"]) if host_status == 200 else None
    async with client.post("http://%s:%d/v1/embed" % connector.listen, json={"texts": texts[:4], "dimensions": 1024}, headers={"Authorization": "Bearer " + TOKEN}) as response:
        cut_status, cut_answer = response.status, await response.json()
    cut = vectors(cut_answer["vectors"]) if cut_status == 200 else None
    async with client.post("http://%s:%d/v1/embed" % connector.listen, json={"texts": ["word " * 3000]}, headers={"Authorization": "Bearer " + TOKEN}) as response:
        long_status, long_answer = response.status, await response.json()
    await connector.stop()
    token_file.unlink()
    expected_cut = direct[:4, :1024] / np.linalg.norm(direct[:4, :1024].astype(np.float64), axis=1, keepdims=True)
    print(json.dumps({
        "step": "chat_layer", "public_status": status, "public_texts": None if public is None else len(public),
        "public_rows_bit_equal_to_engine": None if public is None else int(np.all(public == direct, axis=1).sum()),
        "public_usage": answer.get("usage") if status == 200 else answer, "hello_host_methods": program.hellos[-1]["host_methods"],
        "embed_status": host_status, "embed_model": host_answer.get("model"), "embed_dimensions": host_answer.get("dimensions"), "embed_tokens": host_answer.get("tokens"),
        "embed_rows_bit_equal_to_engine": None if hosted is None else int(np.all(hosted == direct[:64], axis=1).sum()), "embed_rows": None if hosted is None else len(hosted),
        "embed_1024_status": cut_status, "embed_1024_shape": None if cut is None else list(cut.shape),
        "embed_1024_norms": None if cut is None else [float(np.linalg.norm(row)) for row in cut],
        "embed_1024_max_abs_from_cut_of_full": None if cut is None else float(np.abs(cut - expected_cut).max()),
        "too_long_status": long_status, "too_long_error": long_answer.get("error")}))
    await client.close()
    await frontend.session.close()
    await runner.cleanup()
    await program.runner.cleanup()


asyncio.run(run())

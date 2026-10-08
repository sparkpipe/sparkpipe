#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys
import time

import aiohttp
from aiohttp import web


class SwapError(Exception):
    pass


def substitute(template, values):
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def require(spec, keys, where):
    missing = [key for key in keys if key not in spec]
    if missing:
        raise SystemExit(f"model_swap: {where} needs {', '.join(missing)}")


class Swapper:
    def __init__(self, config):
        require(config, ("nodes", "ssh", "drop_kv", "memory_probe", "memory_margin_bytes", "models"), "the config")
        self.nodes = list(config["nodes"])
        self.ssh = list(config["ssh"])
        self.drop_kv = config["drop_kv"]
        self.memory_probe = config["memory_probe"]
        self.memory_margin = int(config["memory_margin_bytes"])
        self.ready_seconds = float(config.get("ready_seconds", 300))
        self.api_seconds = float(config.get("api_seconds", 120))
        self.drain_seconds = float(config.get("drain_seconds", 30))
        self.node_seconds = float(config.get("node_seconds", 240))
        self.models = {}
        for spec in config["models"]:
            require(spec, ("id", "kv_label", "node_memory_bytes", "node_reclaim", "node_start", "node_ready", "node_stop", "api_start", "api_stop", "health"), f"model {spec.get('id')!r}")
            self.models[spec["id"]] = spec
        self.active = None
        self.wanted = None
        self.state = "idle"
        self.phase = None
        self.target = None
        self.swap_started = None
        self.phase_started = None
        self.error = None
        self.history = []
        self.memory = None
        self.task = None
        self.session = None

    def status(self):
        now = time.time()
        return {"active": self.active, "state": self.state, "target": self.target, "wanted": self.wanted, "phase": self.phase,
                "swap_elapsed_s": round(now - self.swap_started, 1) if self.state == "swapping" and self.swap_started else None,
                "phase_elapsed_s": round(now - self.phase_started, 1) if self.state == "swapping" and self.phase_started else None,
                "error": self.error, "memory": self.memory, "models": [{"id": model_id, "active": model_id == self.active} for model_id in self.models],
                "history": self.history[-10:]}

    def enter(self, phase):
        self.phase = phase
        self.phase_started = time.time()
        print(json.dumps({"event": "swap_phase", "target": self.target, "phase": phase, "elapsed_s": round(self.phase_started - self.swap_started, 1)}), file=sys.stderr, flush=True)

    async def run(self, argv, timeout):
        process = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.DEVNULL,
                                                       stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return 124, f"timed out after {timeout:g} s"
        return process.returncode, output.decode(errors="replace").strip()

    async def local(self, command, timeout):
        return await self.run(["bash", "-c", command], timeout)

    async def fan(self, template, timeout, values):
        jobs = []
        for rank, host in enumerate(self.nodes):
            command = substitute(template, dict(values, rank=rank, host=host))
            jobs.append(self.run(self.ssh + [host, command], timeout))
        return await asyncio.gather(*jobs)

    def check(self, results, what):
        failed = [(self.nodes[rank], code, text[-300:]) for rank, (code, text) in enumerate(results) if code != 0]
        if failed:
            host, code, text = failed[0]
            raise SwapError(f"{what} failed on {len(failed)} node(s); first {host} exit {code}: {text}")

    async def available(self):
        results = await self.fan(self.memory_probe, 30, {})
        self.check(results, "memory probe")
        values = []
        for rank, (code, text) in enumerate(results):
            try:
                values.append(int(text.split()[-1]))
            except (IndexError, ValueError):
                raise SwapError(f"memory probe on {self.nodes[rank]} printed {text[-80:]!r}, not a byte count")
        return values

    async def admit(self, model):
        before = await self.available()
        need = int(model["node_memory_bytes"]) + self.memory_margin
        if any(value < need for value in before):
            for other in self.models.values():
                if other["id"] != model["id"]:
                    self.check(await self.fan(other["node_reclaim"], 120, {}), f"{other['id']} cold arena release")
            before = await self.available()
        short = [f"{self.nodes[rank]} {value / 2**30:.1f}" for rank, value in enumerate(before) if value < need]
        if short:
            raise SwapError(f"{model['id']} needs {need / 2**30:.1f} GiB available per node "
                            f"({int(model['node_memory_bytes']) / 2**30:.1f} GiB projected + {self.memory_margin / 2**30:.1f} GiB margin); "
                            f"short on {len(short)} node(s), GiB available: {', '.join(short)}")
        return before

    def measure(self, model, before, after):
        used = [b - a for b, a in zip(before, after)]
        self.memory = {"model": model["id"], "projected_gib": round(int(model["node_memory_bytes"]) / 2**30, 1),
                       "used_gib_max": round(max(used) / 2**30, 1), "used_gib_min": round(min(used) / 2**30, 1),
                       "available_gib_min": round(min(after) / 2**30, 1),
                       "tightest": self.nodes[min(range(len(after)), key=lambda rank: after[rank])]}
        print(json.dumps(dict(self.memory, event="swap_memory")), file=sys.stderr, flush=True)

    async def health(self, model):
        try:
            async with self.session.get(model["health"], timeout=aiohttp.ClientTimeout(total=5)) as response:
                payload = await response.json(content_type=None)
                return response.status, payload
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
            return 0, {}

    async def ready(self, model):
        status, payload = await self.health(model)
        return status == 200 and payload.get("connected_ranks") == len(self.nodes)

    async def drain(self, model):
        deadline = time.time() + self.drain_seconds
        while time.time() < deadline:
            status, payload = await self.health(model)
            if status == 0 or not payload.get("live_requests"):
                return
            await asyncio.sleep(1)
        print(json.dumps({"event": "swap_drain_expired", "model": model["id"], "drain_seconds": self.drain_seconds}), file=sys.stderr, flush=True)

    async def stop(self, model):
        await self.drain(model)
        code, text = await self.local(model["api_stop"], 60)
        if code != 0:
            raise SwapError(f"{model['id']} api stop exit {code}: {text[-300:]}")
        self.check(await self.fan(model["node_stop"], self.node_seconds, {}), f"{model['id']} engine stop")
        self.check(await self.fan(self.drop_kv, 60, {"label": model["kv_label"]}), f"{model['id']} KV pool release")

    async def start(self, model):
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.enter("starting engines")
        self.check(await self.fan(model["node_start"], 120, {"run_id": run_id}), f"{model['id']} engine start")
        self.enter("waiting for ranks")
        deadline = time.time() + self.ready_seconds
        while True:
            results = await self.fan(model["node_ready"], 30, {"run_id": run_id})
            states = [text.split()[-1] if code == 0 and text else "UNKNOWN" for code, text in results]
            dead = [self.nodes[rank] for rank, state in enumerate(states) if state == "DEAD"]
            if dead:
                raise SwapError(f"{model['id']} engines died on {', '.join(dead)}")
            if all(state == "READY" for state in states):
                break
            if time.time() > deadline:
                waiting = [self.nodes[rank] for rank, state in enumerate(states) if state != "READY"]
                raise SwapError(f"{model['id']} ranks not ready after {self.ready_seconds:g} s: {', '.join(waiting)}")
            await asyncio.sleep(2)
        if model.get("node_warm"):
            self.enter("warming weights")
            self.check(await self.fan(model["node_warm"], self.node_seconds, {"run_id": run_id}), f"{model['id']} weight warm")
        self.enter("connecting api")
        code, text = await self.local(model["api_start"], 60)
        if code != 0:
            raise SwapError(f"{model['id']} api start exit {code}: {text[-300:]}")
        deadline = time.time() + self.api_seconds
        while not await self.ready(model):
            if time.time() > deadline:
                raise SwapError(f"{model['id']} api did not connect all {len(self.nodes)} ranks within {self.api_seconds:g} s")
            await asyncio.sleep(1)

    async def swap(self, target):
        model = self.models[target]
        self.state = "swapping"
        self.target = target
        self.error = None
        self.swap_started = time.time()
        previous = self.active
        measured = None
        try:
            if await self.ready(model):
                self.active = target
            else:
                self.active = None
                for other in self.models.values():
                    if other["id"] != target:
                        self.enter(f"stopping {other['id']}")
                        await self.stop(other)
                self.enter(f"clearing {target}")
                code, text = await self.local(model["api_stop"], 60)
                if code != 0:
                    raise SwapError(f"{target} api stop exit {code}: {text[-300:]}")
                self.check(await self.fan(model["node_stop"], self.node_seconds, {}), f"{target} engine stop")
                self.enter("checking memory")
                before = await self.admit(model)
                await self.start(model)
                self.measure(model, before, await self.available())
                measured = self.memory
                self.active = target
            self.state = "idle"
        except SwapError as error:
            self.state = "failed"
            self.error = str(error)
        elapsed = round(time.time() - self.swap_started, 1)
        record = {"from": previous, "to": target, "ok": self.state == "idle", "seconds": elapsed, "error": self.error, "memory": measured,
                  "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.history.append(record)
        print(json.dumps(dict(record, event="swap_done")), file=sys.stderr, flush=True)
        self.phase = None
        self.target = None

    async def drive(self):
        while self.wanted is not None and self.wanted != self.active:
            target = self.wanted
            await self.swap(target)
            if self.state == "failed" and self.wanted == target:
                self.wanted = None
                break
        if self.wanted == self.active:
            self.wanted = None

    async def activate(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "body must be JSON with a model"}, status=400)
        model_id = body.get("model") if isinstance(body, dict) else None
        if model_id not in self.models:
            return web.json_response({"error": f"model {model_id!r} is not configured; configured: {sorted(self.models)}"}, status=404)
        if model_id == self.active and self.state != "swapping":
            if await self.ready(self.models[model_id]):
                return web.json_response(self.status(), status=200)
            self.active = None
        self.wanted = model_id
        if self.task is None or self.task.done():
            self.task = asyncio.ensure_future(self.drive())
        return web.json_response(self.status(), status=202)

    async def get_status(self, request):
        return web.json_response(self.status())

    async def startup(self, app):
        self.session = aiohttp.ClientSession()
        for model in self.models.values():
            if await self.ready(model):
                self.active = model["id"]
                break
        print(json.dumps({"event": "model_swap_ready", "active": self.active, "models": sorted(self.models)}), file=sys.stderr, flush=True)

    async def cleanup(self, app):
        await self.session.close()


def main():
    parser = argparse.ArgumentParser(description="Swap the fleet between configured models on demand.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8440)
    arguments = parser.parse_args()
    with open(arguments.config) as handle:
        swapper = Swapper(json.load(handle))
    app = web.Application()
    app.on_startup.append(swapper.startup)
    app.on_cleanup.append(swapper.cleanup)
    app.router.add_get("/status", swapper.get_status)
    app.router.add_post("/activate", swapper.activate)
    web.run_app(app, host=arguments.host, port=arguments.port, print=None)


if __name__ == "__main__":
    main()

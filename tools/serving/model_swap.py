#!/usr/bin/env python3
import argparse
import asyncio
import hmac
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web


class SwapError(Exception):
    pass


def substitute(template, values):
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def read_token(path):
    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        raise SystemExit(f"model_swap: {path} has mode {mode:o}; the token file must be readable by its owner only")
    token = Path(path).read_text().strip()
    if len(token) < 32:
        raise SystemExit(f"model_swap: the token in {path} must be at least 32 characters")
    return token


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
        self.fill_seconds = config.get("fill_seconds", 3600)
        self.drain_seconds = float(config.get("drain_seconds", 30))
        self.drain_quiet_seconds = float(config.get("drain_quiet_seconds", 0))
        self.node_seconds = float(config.get("node_seconds", 240))
        self.release_seconds = float(config.get("release_seconds", 60))
        self.hub_seconds = float(config.get("hub_seconds", 120))
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
        self.force_requested = False
        self.forced = False
        self.draining = None

    def status(self):
        now = time.time()
        return {"active": self.active, "state": self.state, "target": self.target, "wanted": self.wanted, "phase": self.phase,
                "swap_elapsed_s": round(now - self.swap_started, 1) if self.state == "swapping" and self.swap_started else None,
                "phase_elapsed_s": round(now - self.phase_started, 1) if self.state == "swapping" and self.phase_started else None,
                "error": self.error, "memory": self.memory, "models": [{"id": model_id, "active": model_id == self.active} for model_id in self.models],
                "draining": self.draining, "force_requested": self.force_requested, "history": self.history[-10:]}

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
        if any(value < need for value in before):
            self.check(await self.fan(model["node_reclaim"], 120, {}), f"{model['id']} own cold arena release")
            before = await self.available()
        deadline = time.time() + self.release_seconds
        while any(value < need for value in before) and time.time() < deadline:
            await asyncio.sleep(2)
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
        started = time.time()
        deadline = started + self.drain_seconds
        quiet_since = None
        try:
            while time.time() < deadline:
                if self.force_requested:
                    self.forced = True
                    print(json.dumps({"event": "swap_forced", "model": model["id"], "waited_s": round(time.time() - started, 1)}), file=sys.stderr, flush=True)
                    return
                status, payload = await self.health(model)
                if status == 0:
                    return
                live = payload.get("live_requests") or 0
                if not live and (payload.get("idle_ms") or 0) >= self.drain_quiet_seconds * 1000:
                    return
                if live:
                    quiet_since = None
                else:
                    quiet_since = quiet_since or time.time()
                    if time.time() - quiet_since >= self.drain_quiet_seconds:
                        if time.time() - started >= 1:
                            print(json.dumps({"event": "swap_drained", "model": model["id"], "waited_s": round(time.time() - started, 1)}), file=sys.stderr, flush=True)
                        return
                self.draining = {"model": model["id"], "live_requests": live, "waited_s": round(time.time() - started, 1),
                                 "quiet_s": round(time.time() - quiet_since, 1) if quiet_since else 0.0,
                                 "quiet_needed_s": self.drain_quiet_seconds, "limit_s": self.drain_seconds}
                await asyncio.sleep(1)
            print(json.dumps({"event": "swap_drain_expired", "model": model["id"], "drain_seconds": self.drain_seconds}), file=sys.stderr, flush=True)
        finally:
            self.draining = None

    async def stop(self, model, drain=True):
        if drain:
            await self.drain(model)
        code, text = await self.local(model["api_stop"], 60)
        if code != 0:
            raise SwapError(f"{model['id']} api stop exit {code}: {text[-300:]}")
        self.check(await self.fan(model["node_stop"], self.node_seconds, {}), f"{model['id']} engine stop")
        if model.get("hub_stop"):
            code, text = await self.local(model["hub_stop"], self.hub_seconds)
            if code != 0:
                raise SwapError(f"{model['id']} hub service stop exit {code}: {text[-300:]}")

    async def stop_all(self, models):
        for model in models:
            await self.drain(model)
        for model, (code, text) in zip(models, await asyncio.gather(*(self.local(model["api_stop"], 60) for model in models))):
            if code != 0:
                raise SwapError(f"{model['id']} api stop exit {code}: {text[-300:]}")
        joined = " && ".join(f"( {model['node_stop']} )" for model in models)
        self.check(await self.fan(joined, self.node_seconds, {}), f"{len(models)} models' engine stop")
        stopping = [model for model in models if model.get("hub_stop")]
        for model, (code, text) in zip(stopping, await asyncio.gather(*(self.local(model["hub_stop"], self.hub_seconds) for model in stopping))):
            if code != 0:
                raise SwapError(f"{model['id']} hub service stop exit {code}: {text[-300:]}")

    async def release(self, labels):
        joined = " && ".join(f"( {substitute(self.drop_kv, {'label': label})} )" for label in labels)
        self.check(await self.fan(joined, 60, {}), f"KV pool release of {', '.join(labels)}")

    async def start(self, model):
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        hub = asyncio.create_task(self.local(model["hub_start"], self.hub_seconds)) if model.get("hub_start") else None
        warm = None
        try:
            self.enter("starting engines")
            self.check(await self.fan(model["node_start"], 120, {"run_id": run_id}), f"{model['id']} engine start")
            if model.get("warm_with_engines"):
                warm = asyncio.create_task(self.fan(model["node_warm"], self.node_seconds, {"run_id": run_id}))
            await self.wait_for_ranks(model, run_id)
            if warm:
                self.enter("warming weights")
                self.check(await warm, f"{model['id']} weight warm")
            elif model.get("node_warm"):
                self.enter("warming weights")
                self.check(await self.fan(model["node_warm"], self.node_seconds, {"run_id": run_id}), f"{model['id']} weight warm")
            if hub:
                self.enter("starting hub services")
                code, text = await hub
                if code != 0:
                    raise SwapError(f"{model['id']} hub service start exit {code}: {text[-300:]}")
        except BaseException:
            for task in (hub, warm):
                if task:
                    task.cancel()
            raise
        self.enter("connecting api")
        code, text = await self.local(model["api_start"], 60)
        if code != 0:
            raise SwapError(f"{model['id']} api start exit {code}: {text[-300:]}")
        deadline = time.time() + self.api_seconds
        while not await self.ready(model):
            if time.time() > deadline:
                raise SwapError(f"{model['id']} api did not connect all {len(self.nodes)} ranks within {self.api_seconds:g} s")
            await asyncio.sleep(1)

    async def wait_for_ranks(self, model, run_id):
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
            await asyncio.sleep(1)

    async def stream_fill(self, model, tokens):
        engine = model["health"].rsplit("/", 1)[0] + "/v1/completions"
        body = {"prompt_token_ids": tokens, "max_tokens": 1, "temperature": 0}
        async with self.session.post(engine, json=body, timeout=aiohttp.ClientTimeout(total=self.fill_seconds)) as response:
            text = await response.text()
            if response.status != 200:
                raise SwapError(f"{model['id']} streamed fill answered {response.status}: {text[-300:]}")

    async def run_fill(self, model, tokens):
        fill_lane = {key: value for key, value in dict(model, **model["fill"]).items() if key not in ("hub_start", "hub_stop", "node_warm", "warm_with_engines")}
        began = time.time()
        marks = {}
        try:
            self.active = None
            self.enter("stopping engines")
            await self.stop(model)
            self.check(await self.fan(model["node_reclaim"], 120, {}), f"{model['id']} cold arena release")
            await self.start(fill_lane)
            self.enter("streaming the prompt")
            marks["start_s"] = round(time.time() - began, 1)
            await self.stream_fill(fill_lane, tokens)
            marks["fill_s"] = round(time.time() - began - marks["start_s"], 1)
            mark = time.time()
            self.enter("stopping the fill lane")
            await self.stop(fill_lane, drain=False)
            self.check(await self.fan(model["node_reclaim"], 120, {}), f"{model['id']} cold arena release")
            await self.start(model)
            marks["reload_s"] = round(time.time() - mark, 1)
            self.active = model["id"]
            self.state = "idle"
        except SwapError as error:
            self.state = "failed"
            self.error = str(error)
        except Exception as error:
            self.state = "failed"
            self.error = f"unexpected {type(error).__name__} during {self.phase}: {error}"
        record = dict(marks, model=model["id"], tokens=len(tokens), ok=self.state == "idle", error=self.error, seconds=round(time.time() - began, 1))
        print(json.dumps(dict(record, event="fill_done")), file=sys.stderr, flush=True)
        self.phase = None
        self.target = None
        return record

    async def fill(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "body must be JSON"}, status=400)
        model = self.models.get(body.get("model")) if isinstance(body, dict) else None
        tokens = body.get("prompt_token_ids") if isinstance(body, dict) else None
        if model is None or "fill" not in model or not isinstance(tokens, list) or not tokens:
            return web.json_response({"error": "fill takes a model with a fill lane and prompt_token_ids"}, status=400)
        if self.state == "swapping" or self.active != model["id"]:
            return web.json_response(dict(self.status(), error=f"{model['id']} must be the idle active model"), status=409)
        stride = model["fill"]["checkpoint_tokens"]
        keep = (len(tokens) - 2) // stride * stride
        if keep <= 0:
            return web.json_response({"error": f"a prompt under {stride + 2} tokens has no checkpoint to fill to"}, status=400)
        self.state = "swapping"
        self.target = model["id"]
        self.error = None
        self.swap_started = time.time()
        self.task = asyncio.ensure_future(self.run_fill(model, tokens[:keep]))
        record = await self.task
        return web.json_response(record, status=200 if record["ok"] else 502)

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
                others = [other for other in self.models.values() if other["id"] != target]
                self.enter("stopping engines")
                await self.stop_all(others)
                self.enter(f"clearing {target}")
                await self.stop(model)
                self.enter("releasing KV pools")
                await self.release(list(dict.fromkeys(other["kv_label"] for other in others)))
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
        except Exception as error:
            self.state = "failed"
            self.error = f"unexpected {type(error).__name__} during {self.phase}: {error}"
            print(json.dumps({"event": "swap_exception", "target": target, "phase": self.phase, "error": self.error}), file=sys.stderr, flush=True)
        elapsed = round(time.time() - self.swap_started, 1)
        record = {"from": previous, "to": target, "ok": self.state == "idle", "seconds": elapsed, "error": self.error, "memory": measured,
                  "forced": self.forced, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.force_requested = False
        self.forced = False
        self.history.append(record)
        print(json.dumps(dict(record, event="swap_done")), file=sys.stderr, flush=True)
        self.phase = None
        self.target = None

    async def clear(self):
        self.state = "swapping"
        self.target = None
        self.error = None
        self.swap_started = time.time()
        previous = self.active
        try:
            self.active = None
            self.enter("stopping engines")
            await self.stop_all(list(self.models.values()))
            self.enter("releasing KV pools")
            await self.release(list(dict.fromkeys(model["kv_label"] for model in self.models.values())))
            for model in self.models.values():
                self.enter(f"reclaiming {model['id']} cold arenas")
                self.check(await self.fan(model["node_reclaim"], 120, {}), f"{model['id']} cold arena release")
            self.state = "idle"
        except SwapError as error:
            self.state = "failed"
            self.error = str(error)
        except Exception as error:
            self.state = "failed"
            self.error = f"unexpected {type(error).__name__} during {self.phase}: {error}"
            print(json.dumps({"event": "deactivate_exception", "phase": self.phase, "error": self.error}), file=sys.stderr, flush=True)
        record = {"from": previous, "to": None, "ok": self.state == "idle", "seconds": round(time.time() - self.swap_started, 1), "error": self.error,
                  "memory": None, "forced": self.forced, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.force_requested = False
        self.forced = False
        self.history.append(record)
        print(json.dumps(dict(record, event="deactivate_done")), file=sys.stderr, flush=True)
        self.phase = None
        if self.wanted is not None:
            await self.drive()

    async def deactivate(self, request):
        if self.state == "swapping":
            return web.json_response(dict(self.status(), error="a switch is in progress; deactivate after it ends"), status=409)
        self.wanted = None
        self.state = "swapping"
        self.task = asyncio.ensure_future(self.clear())
        return web.json_response(self.status(), status=202)

    async def drive(self):
        while self.wanted is not None and self.wanted != self.active:
            target = self.wanted
            await self.swap(target)
            if self.state == "failed" and self.wanted == target:
                self.wanted = None
                break
        if self.wanted == self.active:
            self.wanted = None
        self.force_requested = False

    async def force(self, request):
        body = {}
        if request.can_read_body:
            try:
                body = await request.json()
            except json.JSONDecodeError:
                return web.json_response({"error": "body must be JSON"}, status=400)
        model_id = body.get("model") if isinstance(body, dict) else None
        if model_id is not None:
            if model_id not in self.models:
                return web.json_response({"error": f"model {model_id!r} is not configured; configured: {sorted(self.models)}"}, status=404)
            if model_id == self.active and self.state != "swapping":
                return web.json_response(self.status(), status=200)
            self.wanted = model_id
        elif self.state != "swapping" and self.wanted in (None, self.active):
            return web.json_response({"error": "no switch is waiting"}, status=409)
        self.force_requested = True
        print(json.dumps({"event": "swap_force_requested", "model": self.wanted, "draining": self.draining}), file=sys.stderr, flush=True)
        if self.task is None or self.task.done():
            self.task = asyncio.ensure_future(self.drive())
        return web.json_response(self.status(), status=202)

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


def build(swapper, token):
    expected = f"Bearer {token}".encode()

    @web.middleware
    async def authorize(request, handler):
        if not hmac.compare_digest(request.headers.get("Authorization", "").encode(), expected):
            return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)

    app = web.Application(middlewares=[authorize], client_max_size=64 * 1024 * 1024)
    app.on_startup.append(swapper.startup)
    app.on_cleanup.append(swapper.cleanup)
    app.router.add_get("/status", swapper.get_status)
    app.router.add_post("/activate", swapper.activate)
    app.router.add_post("/force", swapper.force)
    app.router.add_post("/deactivate", swapper.deactivate)
    app.router.add_post("/fill", swapper.fill)
    return app


def main():
    parser = argparse.ArgumentParser(description="Swap the fleet between configured models on demand.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8440)
    arguments = parser.parse_args()
    token = read_token(arguments.token_file)
    with open(arguments.config) as handle:
        swapper = Swapper(json.load(handle))
    web.run_app(build(swapper, token), host=arguments.host, port=arguments.port, print=None)


if __name__ == "__main__":
    main()

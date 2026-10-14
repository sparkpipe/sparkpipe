import argparse
import asyncio
import hmac
import json
import re
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web

NODE_STATUS = (
    "echo uptime_s=$(cut -d. -f1 /proc/uptime); "
    "echo load=$(cut -d' ' -f1-3 /proc/loadavg); "
    "echo mem_available_gib=$(awk '/MemAvailable/ {printf \"%.1f\", $2 / 1048576}' /proc/meminfo); "
    "echo root_disk_used=$(df -P / | awk 'NR == 2 {print $5}'); "
    "nvidia-smi --query-gpu=clocks.gr,utilization.gpu,temperature.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null | sed 's/^/gpu=/'; "
    "ps -eo pcpu=,comm= --sort=-pcpu | head -5 | sed 's/^ */top=/'"
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|MASTER_KEY)[A-Z0-9_]*)\s*([=:])\s*\S+"),
)
NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
CHAT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
ROOT = re.compile(r"(\$HOME/[^\s;&|]+/root)(?:/|\s|$)")


def redact(text):
    for pattern in SECRET_PATTERNS:
        if pattern.groups == 2:
            text = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", text)
        else:
            text = pattern.sub("[redacted]", text)
    return text


def clip(text, limit):
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters cut ...]\n{text[-half:]}"


class Gateway:
    def __init__(self, config):
        for key in ("token_file", "audit_log", "hub", "nodes", "ssh", "status_url", "logs", "logs_dir", "swap_config", "refuse", "refuse_on_nodes"):
            if key not in config:
                raise SystemExit(f"fleet_gateway: the config needs {key}")
        self.token = Path(config["token_file"]).read_text().strip()
        if len(self.token) < 32:
            raise SystemExit("fleet_gateway: the token must be at least 32 characters")
        self.audit = Path(config["audit_log"])
        self.hub = config["hub"]
        self.nodes = list(config["nodes"])
        self.hosts = [self.hub] + self.nodes
        self.ssh = list(config["ssh"])
        self.status_url = config["status_url"]
        self.logs = dict(config["logs"])
        self.logs_dir = Path(config["logs_dir"])
        self.refuse = [(re.compile(rule["pattern"]), rule["reason"]) for rule in config["refuse"]]
        self.refuse_on_nodes = [(re.compile(rule["pattern"]), rule["reason"]) for rule in config["refuse_on_nodes"]]
        self.max_output = int(config.get("max_output_chars", 20000))
        self.default_timeout = float(config.get("default_timeout_seconds", 60))
        self.max_timeout = float(config.get("max_timeout_seconds", 600))
        self.max_lines = int(config.get("max_log_lines", 500))
        self.agent_user = config.get("agent_user")
        self.agent_key = config.get("agent_key")
        self.agent_hub_prefix = list(config.get("agent_hub_prefix") or [])
        self.grant_seconds = float(config.get("grant_hours", 8)) * 3600
        self.grants = {}
        self.engine_logs = {}
        for model in json.loads(Path(config["swap_config"]).read_text())["models"]:
            match = ROOT.search(model["node_start"])
            if match:
                self.engine_logs[model["id"]] = match.group(1) + "/residentd.log"
        self.session = None

    async def start(self, app):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))

    async def stop(self, app):
        await self.session.close()

    def record(self, entry):
        self.audit.parent.mkdir(parents=True, exist_ok=True)
        with self.audit.open("a") as handle:
            handle.write(json.dumps(dict(entry, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")

    @web.middleware
    async def authorize(self, request, handler):
        if not hmac.compare_digest(request.headers.get("Authorization", "").encode(), f"Bearer {self.token}".encode()):
            return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)

    def agent_ready(self):
        return bool(self.agent_user and self.agent_key and self.agent_hub_prefix and Path(self.agent_key).is_file())

    def grant_for(self, chat_id):
        grant = self.grants.get(chat_id)
        if grant and grant["expires"] > time.time():
            return grant
        self.grants.pop(chat_id, None)
        return None

    async def execute(self, host, command, timeout, as_agent=False):
        if as_agent and host == self.hub:
            argv, cwd = self.agent_hub_prefix + ["bash", "-lc", command], "/"
        elif as_agent:
            argv, cwd = self.ssh + ["-i", self.agent_key, "-o", "IdentitiesOnly=yes", "-l", self.agent_user, host, command], None
        else:
            argv, cwd = (["bash", "-lc", command] if host == self.hub else self.ssh + [host, command]), None
        started = time.time()
        process = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.STDOUT, start_new_session=True, cwd=cwd)
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout)
            code = process.returncode
        except asyncio.TimeoutError:
            process.kill()
            output, _ = await process.communicate()
            code = None
        text = clip(redact(output.decode("utf-8", "replace")), self.max_output)
        return code, text, round(time.time() - started, 1)

    async def status(self, request):
        try:
            async with self.session.get(self.status_url) as response:
                swap = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as error:
            return web.json_response({"error": f"swap controller unavailable: {error}"}, status=503)
        keep = ("active", "state", "target", "wanted", "phase", "swap_elapsed_s", "draining", "error", "memory")
        summary = {key: swap.get(key) for key in keep}
        summary["recent_switches"] = (swap.get("history") or [])[-5:]
        return web.json_response(summary)

    async def node_status(self, request):
        results = await asyncio.gather(*(self.execute(host, NODE_STATUS, 20) for host in self.hosts))
        return web.json_response({host: {"exit": code, "status": text} for host, (code, text, _) in zip(self.hosts, results)})

    async def read_log(self, request):
        name = request.query.get("name", "")
        node = request.query.get("node", "")
        try:
            lines = max(1, min(int(request.query.get("lines", "100")), self.max_lines))
        except ValueError:
            return web.json_response({"error": "lines must be a number"}, status=400)
        if name == "engine":
            if node not in self.nodes:
                return web.json_response({"error": f"engine logs need node, one of {', '.join(self.nodes)}"}, status=400)
            try:
                async with self.session.get(self.status_url) as response:
                    active = (await response.json(content_type=None)).get("active")
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as error:
                return web.json_response({"error": f"swap controller unavailable: {error}"}, status=503)
            path = self.engine_logs.get(active)
            if path is None:
                return web.json_response({"error": f"no engine log known for the loaded model {active!r}"}, status=404)
            code, text, _ = await self.execute(node, f"tail -n {lines} {path}", 20)
            return web.json_response({"name": f"engine log of {active} on {node}", "exit": code, "text": text})
        if name in self.logs:
            path = Path(self.logs[name])
        elif name.startswith("api:") and NAME.match(name[4:]):
            path = self.logs_dir / f"{name[4:]}-kv-api.log"
        else:
            known = sorted(self.logs) + ["api:<name>", "engine"]
            return web.json_response({"error": f"unknown log {name!r}; known: {', '.join(known)}"}, status=404)
        if not path.is_file():
            return web.json_response({"error": f"{name} has no log file"}, status=404)
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 4 * 1024 * 1024))
            tail = handle.read().decode("utf-8", "replace").splitlines()[-lines:]
        return web.json_response({"name": name, "text": clip(redact("\n".join(tail)), self.max_output)})

    async def grant(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "the body must be JSON"}, status=400)
        chat_id = body.get("chat_id")
        approved_by = body.get("approved_by")
        if not isinstance(chat_id, str) or not CHAT_ID.match(chat_id):
            return web.json_response({"error": "chat_id must name the chat"}, status=400)
        if not isinstance(approved_by, str) or not approved_by:
            return web.json_response({"error": "approved_by must name the admin who granted access"}, status=400)
        if not self.agent_ready():
            return web.json_response({"error": f"the restricted user {self.agent_user or '(none)'} is not set up on the fleet yet; run deployment/hub/fleet_agent_setup.sh"}, status=503)
        expires = time.time() + self.grant_seconds
        self.grants[chat_id] = {"expires": expires, "approved_by": approved_by}
        self.record({"event": "grant", "chat_id": chat_id, "approved_by": approved_by, "user": self.agent_user, "hours": self.grant_seconds / 3600})
        return web.json_response({"chat_id": chat_id, "user": self.agent_user,
                                  "expires_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires))})

    async def grant_status(self, request):
        chat_id = request.query.get("chat_id", "")
        grant = self.grant_for(chat_id) if CHAT_ID.match(chat_id) else None
        return web.json_response({"chat_id": chat_id, "granted": grant is not None, "user": self.agent_user, "ready": self.agent_ready(),
                                  "expires_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(grant["expires"])) if grant else None})

    async def revoke(self, request):
        chat_id = request.query.get("chat_id", "")
        removed = self.grants.pop(chat_id, None) is not None
        if removed:
            self.record({"event": "revoke", "chat_id": chat_id})
        return web.json_response({"chat_id": chat_id, "revoked": removed})

    async def run(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "the body must be JSON"}, status=400)
        host = body.get("host")
        command = body.get("command")
        mode = body.get("as", "spec")
        chat_id = body.get("chat_id")
        if host not in self.hosts:
            return web.json_response({"error": f"host must be one of {', '.join(self.hosts)}"}, status=400)
        if not isinstance(command, str) or not command.strip() or len(command) > 4000:
            return web.json_response({"error": "command must be a non-empty string of at most 4000 characters"}, status=400)
        if mode == "agent":
            grant = self.grant_for(chat_id) if isinstance(chat_id, str) else None
            if grant is None:
                return web.json_response({"error": "this chat has no fleet grant, or it expired"}, status=403)
            if not self.agent_ready():
                return web.json_response({"error": f"the restricted user {self.agent_user or '(none)'} is not set up on the fleet yet"}, status=503)
            approved_by = grant["approved_by"]
        elif mode == "spec":
            approved_by = body.get("approved_by")
            if not isinstance(approved_by, str) or not approved_by:
                return web.json_response({"error": "approved_by must name the admin who approved the command"}, status=400)
        else:
            return web.json_response({"error": "as must be agent or spec"}, status=400)
        try:
            timeout = min(float(body.get("timeout") or self.default_timeout), self.max_timeout)
        except (TypeError, ValueError):
            return web.json_response({"error": "timeout must be a number of seconds"}, status=400)
        rules = self.refuse + (self.refuse_on_nodes if host != self.hub else [])
        user = self.agent_user if mode == "agent" else "spec"
        for pattern, reason in rules:
            if pattern.search(command):
                self.record({"event": "refused", "host": host, "user": user, "chat_id": chat_id, "command": command, "approved_by": approved_by, "reason": reason})
                return web.json_response({"error": f"refused: {reason}"}, status=403)
        self.record({"event": "run", "host": host, "user": user, "chat_id": chat_id, "command": command, "approved_by": approved_by, "timeout": timeout})
        code, text, seconds = await self.execute(host, command, timeout, as_agent=mode == "agent")
        self.record({"event": "done", "host": host, "user": user, "exit": code, "seconds": seconds, "output_chars": len(text)})
        return web.json_response({"host": host, "user": user, "exit": code, "timed_out": code is None, "seconds": seconds, "output": text})


def build(config):
    gateway = Gateway(config)
    app = web.Application(middlewares=[gateway.authorize])
    app.on_startup.append(gateway.start)
    app.on_cleanup.append(gateway.stop)
    app.router.add_get("/v1/status", gateway.status)
    app.router.add_get("/v1/nodes", gateway.node_status)
    app.router.add_get("/v1/logs", gateway.read_log)
    app.router.add_post("/v1/run", gateway.run)
    app.router.add_post("/v1/grant", gateway.grant)
    app.router.add_get("/v1/grant", gateway.grant_status)
    app.router.add_delete("/v1/grant", gateway.revoke)
    return app


def main():
    parser = argparse.ArgumentParser(description="Admin fleet access for SparkPipe Chat: status, logs and approved commands.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    app = build(json.loads(Path(arguments.config).read_text()))
    print(json.dumps({"event": "fleet_gateway_ready", "host": arguments.host, "port": arguments.port}), file=sys.stderr, flush=True)
    web.run_app(app, host=arguments.host, port=arguments.port, print=None, access_log=None)


if __name__ == "__main__":
    main()

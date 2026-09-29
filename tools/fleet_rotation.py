#!/usr/bin/env python3
import argparse
import concurrent.futures
import datetime
import fcntl
import http.server
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time

HOUR = 3600
BARE_RECLAIM = re.compile(r"--reclaim(?![-\w])")
PLACEHOLDER = re.compile(r"@[a-z_]+@")
ENGINE_KEYS = ("start", "stop", "up", "ready")
API_KEYS = ("start", "stop", "up", "health")


class Failure(Exception):
    pass


class ConfigError(Exception):
    pass


class Preempted(Exception):
    pass


def iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text):
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()


def fill(template, values):
    out = template
    for key, value in values.items():
        out = out.replace("@" + key + "@", str(value))
    left = PLACEHOLDER.findall(out)
    if left:
        raise ConfigError(f"unresolved placeholder {left[0]} in: {template[:120]}")
    return out


def walk_strings(node, path=""):
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from walk_strings(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk_strings(value, f"{path}[{index}]")


def require(condition, message):
    if not condition:
        raise ConfigError(message)


def validate_model(cfg, name, model):
    require(isinstance(model.get("title"), str), f"model {name}: title missing")
    if not model.get("runnable", False):
        require(isinstance(model.get("not_runnable"), str) and model["not_runnable"], f"model {name}: a model that is not runnable states why in not_runnable")
        return
    nodes = model.get("nodes")
    require(isinstance(nodes, list) and nodes, f"model {name}: nodes missing")
    require(all(h in cfg["fleet"] for h in nodes), f"model {name}: a node is not in fleet")
    require(isinstance(model.get("mem_gib"), (int, float)) and model["mem_gib"] > 0, f"model {name}: mem_gib missing")
    engine = model.get("engine", {})
    for key in ENGINE_KEYS:
        require(isinstance(engine.get(key), str) and engine[key], f"model {name}: engine.{key} missing")
    for key in ("ready_timeout_s", "stop_timeout_s", "grace_s"):
        require(isinstance(engine.get(key), (int, float)), f"model {name}: engine.{key} missing")
    require(isinstance(model.get("packs"), list) and model["packs"], f"model {name}: packs (reclaim-pack sha files) missing")
    api = model.get("api")
    if api is not None:
        for key in API_KEYS:
            require(isinstance(api.get(key), str) and api[key], f"model {name}: api.{key} missing")
        require(isinstance(api.get("port"), int), f"model {name}: api.port missing")
        require(isinstance(api.get("timeout_s"), (int, float)), f"model {name}: api.timeout_s missing")
        smoke = model.get("smoke")
        require(isinstance(smoke, dict) and smoke.get("path") and isinstance(smoke.get("body"), dict) and smoke.get("expect"), f"model {name}: smoke path/body/expect missing")
        require(isinstance(smoke.get("timeout_s"), (int, float)), f"model {name}: smoke.timeout_s missing")


def validate(cfg):
    for key in ("hub", "state_dir", "fleet", "floor_gib", "cycle", "slots", "fallback", "companions", "models", "commands", "ssh", "ssh_timeout_s", "poll_s", "rollback_models", "schedule_port", "lock"):
        require(key in cfg, f"config: {key} missing")
    models = cfg["models"]
    for name, model in models.items():
        validate_model(cfg, name, model)
    for key in ("mem_probe", "reclaim", "smoke"):
        require(isinstance(cfg["commands"].get(key), str), f"config: commands.{key} missing")
    require("--reclaim-pack" in cfg["commands"]["reclaim"] and "@packs@" in cfg["commands"]["reclaim"], "commands.reclaim must be a --reclaim-pack command over @packs@")
    for path, text in walk_strings(cfg):
        require(not BARE_RECLAIM.search(text), f"config{path}: node-global --reclaim is forbidden; use --reclaim-pack")
    slots = cfg["slots"]
    require(cfg["cycle"].get("slots") and all(s in slots for s in cfg["cycle"]["slots"]), "cycle.slots names an unknown slot")
    require(isinstance(cfg["cycle"].get("anchor_hour"), int), "cycle.anchor_hour missing")
    for name, slot in slots.items():
        require(slot.get("base") in models, f"slot {name}: base model unknown")
        fallback_slot = slot.get("fallback_slot")
        require(fallback_slot is None or fallback_slot in slots, f"slot {name}: fallback_slot unknown")
    fb = cfg["fallback"]
    require(fb in models and models[fb].get("runnable"), "fallback model must exist and be runnable")
    require(set(models[fb]["nodes"]) == set(cfg["fleet"]), "fallback model must span the whole fleet")
    require(models[fb].get("api") is not None, "fallback model needs an api")
    for name in cfg["companions"] + cfg["rollback_models"] + [cfg.get("default_companion")] * bool(cfg.get("default_companion")):
        require(name in models, f"companion/rollback model {name} unknown")
    require(all(models[m].get("runnable") for m in cfg["rollback_models"]), "rollback models must be runnable")
    for name in slots:
        seen = set()
        cur = name
        while cur is not None:
            require(cur not in seen, f"slot {name}: fallback_slot loop")
            seen.add(cur)
            cur = slots[cur].get("fallback_slot")
    return cfg


def load_config(path):
    with open(path) as f:
        return validate(json.load(f))


class Runner:
    def __init__(self, cfg, dry, out):
        self.cfg = cfg
        self.dry = dry
        self.out = out
        self.hub_local = socket.gethostname().split(".")[0] == cfg["hub"]

    def argv(self, host, command):
        if host == "hub" and self.hub_local:
            return ["bash", "-c", command]
        target = self.cfg["hub"] if host == "hub" else host
        return list(self.cfg["ssh"]) + [target, command]

    def run(self, host, command, timeout=None, stdin=None):
        if self.dry:
            self.out(f"DRY {host}: {command}")
            return 0, ""
        try:
            p = subprocess.run(self.argv(host, command), input=stdin, capture_output=True, text=True, timeout=timeout or self.cfg["ssh_timeout_s"])
        except subprocess.TimeoutExpired:
            return 124, f"timeout after {timeout or self.cfg['ssh_timeout_s']} s"
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    def each(self, commands, timeout=None):
        if not commands:
            return {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(commands)) as pool:
            futures = {host: pool.submit(self.run, host, command, timeout) for host, command in commands.items()}
            return {host: f.result() for host, f in futures.items()}


class Rotation:
    def __init__(self, cfg, runner, now=None, out=print):
        self.cfg = cfg
        self.run = runner
        self.dry = runner.dry
        self.out = out
        self.clock = now
        self.dir = os.path.expanduser(cfg["state_dir"])
        self.models = cfg["models"]
        self.fb = cfg["fallback"]
        self.honor_locks = True
        self.instance = self.slot_at(self.now())[0]

    def now(self):
        return self.clock if self.clock is not None else time.time()

    def path(self, name):
        return os.path.join(self.dir, name)

    def load_state(self):
        try:
            with open(self.path("state.json")) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_state(self, st):
        if self.dry:
            return
        os.makedirs(self.dir, exist_ok=True)
        tmp = self.path("state.json.tmp")
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path("state.json"))

    def append(self, name, line):
        if self.dry:
            return
        os.makedirs(self.dir, exist_ok=True)
        with open(self.path(name), "a") as f:
            f.write(line + "\n")

    def log(self, event, **fields):
        line = f"{iso(time.time())} {event}" + "".join(f" {k}={v}" for k, v in fields.items())
        self.out(line)
        self.append("rotation.log", line)

    def alert(self, st, level, message):
        line = f"{iso(time.time())} {level} {message}"
        self.log("ALERT", level=level, message=json.dumps(message))
        if st.get("alert_message") != message:
            self.append("ALERT", line)
        st["alert"] = line
        st["alert_message"] = message

    def pause_reason(self):
        for name, label in ((self.cfg["lock"]["pause_file"], "paused"), (self.cfg["lock"]["mirror_pause_file"], "paused-coord")):
            p = self.path(name)
            if os.path.exists(p):
                with open(p) as f:
                    return label, f"{label}: {f.read().strip()[:160]}"
        holder = self.path(self.cfg["lock"]["holder_file"])
        if os.path.exists(holder):
            with open(holder) as f:
                text = f.read().strip()
            if any(text.startswith(prefix) for prefix in self.cfg["lock"]["holder_prefixes"]):
                return "preempted", f"lead window: {text[:160]}"
        return None, None

    def checkpoint(self):
        if not self.honor_locks:
            return
        kind, reason = self.pause_reason()
        if kind:
            raise Preempted(kind, reason)

    def hold_off(self, st, kind, reason):
        if st.get("phase") != kind or st.get("lock") != reason:
            self.log("HOLD-OFF", kind=kind, reason=json.dumps(reason))
        st["phase"] = kind
        st["lock"] = reason
        st["resync"] = True

    def auto_pause(self, st, why):
        if not self.dry:
            os.makedirs(self.dir, exist_ok=True)
            with open(self.path(self.cfg["lock"]["pause_file"]), "w") as f:
                f.write(f"auto {iso(time.time())}: {why}\n")
        st["resync"] = True
        self.log("AUTO-PAUSE", why=json.dumps(why))

    def slot_at(self, t):
        hour = int(t // HOUR)
        names = self.cfg["cycle"]["slots"]
        return hour, names[(hour - self.cfg["cycle"]["anchor_hour"]) % len(names)]

    def effective_slot(self, st, instance, name):
        demoted = st.get("demoted", {}).get(str(instance), [])
        while True:
            slot = self.cfg["slots"][name]
            if self.models[slot["base"]].get("runnable") and name not in demoted:
                return name
            if slot.get("fallback_slot") is None:
                return None
            name = slot["fallback_slot"]

    def next_companion(self, after, skip=()):
        order = self.cfg["companions"]
        start = order.index(after) + 1 if after in order else 0
        for k in range(len(order)):
            name = order[(start + k) % len(order)]
            if self.models[name].get("runnable") and name not in skip:
                return name
        return self.cfg.get("default_companion")

    def companion_for(self, st, instance):
        chosen = st.setdefault("companions", {})
        key = str(instance)
        if key not in chosen:
            chosen[key] = self.next_companion(st.get("last_companion"), st.get("dropped", {}).get(key, []))
            if chosen[key] in self.cfg["companions"]:
                st["last_companion"] = chosen[key]
            for old in sorted(chosen, key=int)[:-24]:
                del chosen[old]
        return chosen[key]

    def target(self, st, t):
        instance, name = self.slot_at(t)
        override = st.get("override")
        companion = None
        if override and override.get("instance") == instance:
            name = override["slot"]
            companion = override.get("companion")
        if st.get("skip") == instance:
            return instance, f"{name}:skipped", [self.fb]
        if st.get("failed_instance") == instance:
            return instance, f"{name}:failed", [self.fb]
        eff = self.effective_slot(st, instance, name)
        if eff is None:
            return instance, f"{name}:unavailable", [self.fb]
        slot = self.cfg["slots"][eff]
        models = [slot["base"]]
        if slot.get("companion"):
            if companion is None:
                companion = self.companion_for(st, instance)
            if companion and companion not in st.get("dropped", {}).get(str(instance), []):
                models.append(companion)
        return instance, eff, models

    def predict(self, st, t, hours):
        shadow = json.loads(json.dumps(st))
        rows = []
        for k in range(hours):
            at = (int(t // HOUR) + k) * HOUR
            instance, slot, models = self.target(shadow, at)
            rows.append({"start": iso(at), "end": iso(at + HOUR), "slot": slot, "models": models})
        return rows

    def node_cmds(self, model, key, extra=None, section="engine"):
        m = self.models[model]
        out = {}
        for rank, host in enumerate(m["nodes"]):
            values = {"host": host, "rank": rank, "model": model}
            values.update(extra or {})
            out[host] = fill(m[section][key], values)
        return out

    def hub_cmd(self, model, key, extra=None):
        values = {"model": model}
        values.update(extra or {})
        return fill(self.models[model]["api"][key], values)

    def observe(self):
        runnable = [n for n, m in self.models.items() if m.get("runnable")]
        per_host = {}
        for name in runnable:
            for host, cmd in self.node_cmds(name, "up").items():
                per_host.setdefault(host, []).append(f"if ( {cmd} ) >/dev/null 2>&1; then echo '@@ {name} 1'; else echo '@@ {name} 0'; fi")
        hub_parts = [f"if ( {self.hub_cmd(n, 'up')} ) >/dev/null 2>&1; then echo '@@ {n} 1'; else echo '@@ {n} 0'; fi" for n in runnable if self.models[n].get("api")]
        commands = {h: "; ".join(parts) for h, parts in per_host.items()}
        if hub_parts:
            commands["hub"] = "; ".join(hub_parts)
        results = self.run.each(commands)
        seen = {}
        for host, (rc, text) in results.items():
            if self.dry:
                continue
            if rc != 0 and "@@" not in text:
                raise Failure(f"observe: {host} unreachable (rc {rc}: {text.strip()[:120]})")
            for line in text.splitlines():
                parts = line.split()
                if len(parts) == 3 and parts[0] == "@@":
                    seen.setdefault(parts[1], []).append(parts[2] == "1")
        state = {}
        for name in runnable:
            flags = seen.get(name, [])
            if self.dry:
                state[name] = "unknown"
            elif flags and all(flags):
                state[name] = "up"
            elif not any(flags):
                state[name] = "down"
            else:
                state[name] = "mixed"
        return state

    def mem(self, hosts):
        results = self.run.each({h: self.cfg["commands"]["mem_probe"] for h in hosts})
        out = {}
        for host, (rc, text) in results.items():
            if self.dry:
                out[host] = None
                continue
            match = re.search(r"-?\d+", text)
            if rc != 0 or not match:
                raise Failure(f"memory probe failed on {host} (rc {rc}: {text.strip()[:80]})")
            out[host] = int(match.group(0))
        return out

    def floor_check(self, when):
        avail = self.mem(self.cfg["fleet"])
        low = {h: v for h, v in avail.items() if v is not None and v < self.cfg["floor_gib"]}
        known = [v for v in avail.values() if v is not None]
        self.log("FLOOR", when=when, min_gib=min(known) if known else "dry", floor=self.cfg["floor_gib"])
        if low:
            raise Failure(f"MemAvailable below {self.cfg['floor_gib']} GiB {when}: " + " ".join(f"{h}={v}" for h, v in sorted(low.items())))
        return avail

    def fits(self, avail, outgoing, incoming):
        short = []
        for host, value in avail.items():
            if value is None:
                continue
            gain = sum(self.models[m]["mem_gib"] for m in outgoing if host in self.models[m]["nodes"])
            need = sum(self.models[m]["mem_gib"] for m in incoming if host in self.models[m]["nodes"])
            if value + gain - need < self.cfg["floor_gib"]:
                short.append(f"{host}={value}+{gain}-{need}")
        return short

    def precheck(self, model):
        m = self.models[model]
        if m["engine"].get("precheck"):
            results = self.run.each(self.node_cmds(model, "precheck"))
            bad = sorted(h for h, (rc, _) in results.items() if rc != 0)
            if bad:
                return f"engine precheck failed on {' '.join(bad)}"
        if m.get("api") and m["api"].get("precheck"):
            rc, text = self.run.run("hub", self.hub_cmd(model, "precheck"))
            if rc != 0:
                return f"api precheck failed (rc {rc}: {text.strip()[:120]})"
        return None

    def wait_nodes(self, model, key, timeout, grace=None):
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        pending = self.node_cmds(model, key, {"stamp": self.stamp})
        while True:
            self.checkpoint()
            results = self.run.each(pending)
            pending = {h: pending[h] for h, (rc, _) in results.items() if rc != 0}
            if not pending:
                return
            if grace is not None and time.monotonic() - started >= grace:
                alive = self.run.each({h: self.node_cmds(model, "up")[h] for h in pending})
                dead = sorted(h for h, (rc, _) in alive.items() if rc != 0)
                if dead:
                    raise Failure(f"{model}: engine exited on {' '.join(dead)} before {key}")
            if time.monotonic() >= deadline:
                raise Failure(f"{model}: {key} not reached within {timeout} s on {' '.join(sorted(pending))}")
            time.sleep(self.cfg["poll_s"])

    def wait_hub(self, model, key, timeout):
        deadline = time.monotonic() + timeout
        while True:
            self.checkpoint()
            rc, _ = self.run.run("hub", self.hub_cmd(model, key))
            if rc == 0:
                return
            if time.monotonic() >= deadline:
                raise Failure(f"{model}: api {key} not reached within {timeout} s")
            time.sleep(self.cfg["poll_s"])

    def reclaim(self, model):
        m = self.models[model]
        still = self.run.each(self.node_cmds(model, "up"))
        busy = sorted(h for h, (rc, _) in still.items() if rc == 0 and not self.dry)
        if busy:
            raise Failure(f"{model}: refusing reclaim-pack while the engine is up on {' '.join(busy)}")
        commands = {}
        for rank, host in enumerate(m["nodes"]):
            packs = " ".join(fill(p, {"host": host, "rank": rank, "model": model}) for p in m["packs"])
            commands[host] = fill(self.cfg["commands"]["reclaim"], {"packs": packs, "model": model, "warm": self.cfg["weightd_warm"], "socket": self.cfg["weightd_socket"]})
        results = self.run.each(commands)
        bad = sorted(f"{h}:rc{rc}" for h, (rc, _) in results.items() if rc != 0)
        if bad:
            raise Failure(f"{model}: reclaim-pack failed on {' '.join(bad)}")
        self.log("RECLAIM-PACK", model=model, nodes=len(results))

    def stop_model(self, model):
        m = self.models[model]
        self.checkpoint()
        self.log("STOP", model=model)
        if m.get("api"):
            self.run.run("hub", self.hub_cmd(model, "stop"))
            rc, _ = self.run.run("hub", self.hub_cmd(model, "up"))
            if rc == 0 and not self.dry:
                raise Failure(f"{model}: api still active after stop")
        results = self.run.each(self.node_cmds(model, "stop"))
        bad = sorted(h for h, (rc, _) in results.items() if rc != 0)
        if bad:
            raise Failure(f"{model}: stop failed on {' '.join(bad)}")
        deadline = time.monotonic() + m["engine"]["stop_timeout_s"]
        pending = self.node_cmds(model, "up")
        while pending:
            self.checkpoint()
            results = self.run.each(pending)
            pending = {h: pending[h] for h, (rc, _) in results.items() if rc == 0 and not self.dry}
            if pending and time.monotonic() >= deadline:
                raise Failure(f"{model}: engine still up after {m['engine']['stop_timeout_s']} s on {' '.join(sorted(pending))}")
            if pending:
                time.sleep(self.cfg["poll_s"])
        self.reclaim(model)
        self.log("STOPPED", model=model)

    def start_model(self, model):
        m = self.models[model]
        self.checkpoint()
        self.stamp = f"{int(time.time())}-{model}"
        self.log("START", model=model, stamp=self.stamp)
        up = self.run.each(self.node_cmds(model, "up"))
        cold = [h for h, (rc, _) in up.items() if rc != 0 or self.dry]
        avail = self.mem(cold)
        short = {h: v for h, v in avail.items() if v is not None and v < m["mem_gib"] + self.cfg["floor_gib"]}
        if short:
            raise Failure(f"{model}: needs {m['mem_gib']}+{self.cfg['floor_gib']} GiB, short on " + " ".join(f"{h}={v}" for h, v in sorted(short.items())))
        results = self.run.each({h: c for h, c in self.node_cmds(model, "start", {"stamp": self.stamp}).items()}, timeout=self.cfg["ssh_timeout_s"])
        bad = sorted(f"{h}:rc{rc}" for h, (rc, _) in results.items() if rc != 0)
        if bad:
            raise Failure(f"{model}: start failed on {' '.join(bad)}")
        self.wait_nodes(model, "ready", m["engine"]["ready_timeout_s"], m["engine"]["grace_s"])
        self.log("READY", model=model, nodes=len(m["nodes"]))
        if m.get("api"):
            rc, text = self.run.run("hub", self.hub_cmd(model, "start"))
            if rc != 0:
                raise Failure(f"{model}: api start rc {rc}: {text.strip()[:160]}")
            self.wait_hub(model, "health", m["api"]["timeout_s"])
            self.smoke(model)
        self.log("SERVING", model=model)

    def smoke(self, model):
        m = self.models[model]
        s = m["smoke"]
        command = fill(self.cfg["commands"]["smoke"], {"model": model, "port": m["api"]["port"], "path": s["path"], "body": shlex.quote(json.dumps(s["body"])), "timeout": s["timeout_s"]})
        rc, text = self.run.run("hub", command, timeout=s["timeout_s"] + 30)
        if self.dry:
            return
        if rc != 0 or s["expect"] not in text:
            raise Failure(f"{model}: smoke failed (rc {rc}, want '{s['expect']}'): {text.strip()[:200]}")
        self.log("SMOKE-PASS", model=model, expect=s["expect"])

    def healthy(self, model):
        m = self.models[model]
        if not m.get("api"):
            return True
        rc, _ = self.run.run("hub", self.hub_cmd(model, "health"))
        return rc == 0 or self.dry

    def converge(self, st, current, target, dirty, reason):
        outgoing = [m for m in current + dirty if m not in target or m in dirty]
        outgoing = list(dict.fromkeys(outgoing))
        incoming = [m for m in target if m not in current or m in dirty]
        incoming.sort(key=lambda m: m != self.fb)
        outgoing.sort(key=lambda m: m == self.fb)
        st["phase"] = "transition"
        st["transition"] = {"from": current, "to": target, "reason": reason, "started": iso(time.time())}
        self.save_state(st)
        self.log("TRANSITION-BEGIN", frm=",".join(current) or "-", to=",".join(target), reason=json.dumps(reason))
        try:
            self.floor_check("before")
            for model in outgoing:
                self.stop_model(model)
            for model in incoming:
                self.start_model(model)
            self.floor_check("after")
        except Preempted:
            raise
        except Exception as e:
            why = str(e) if isinstance(e, Failure) else f"{type(e).__name__}: {e}"
            self.alert(st, "ERROR", f"transition {','.join(current) or '-'} -> {','.join(target)} failed: {why}")
            return self.fallback(st, why)
        st["active"] = list(target)
        st["phase"] = "steady"
        st["transition"] = None
        st["since"] = iso(time.time())
        self.log("TRANSITION-OK", active=",".join(target))
        self.save_state(st)
        return True

    def fallback(self, st, why):
        st["failed_instance"] = self.instance
        st["phase"] = "recovering"
        st["transition"] = {"from": st.get("active") or [], "to": [self.fb], "reason": why, "started": iso(time.time())}
        self.save_state(st)
        self.log("FALLBACK-BEGIN", why=json.dumps(why))
        try:
            state = self.observe()
            for model in [m for m, s in state.items() if s in ("up", "mixed") and m != self.fb]:
                self.stop_model(model)
            self.start_model(self.fb)
            self.floor_check("after-fallback")
        except Preempted:
            raise
        except Exception as e:
            if not isinstance(e, Failure):
                e = f"{type(e).__name__}: {e}"
            try:
                st["active"] = [m for m, s in self.observe().items() if s == "up"]
            except Exception:
                st["active"] = []
            st["phase"] = "degraded"
            self.alert(st, "CRITICAL", f"fallback to {self.fb} failed: {e}; rotation auto-paused, fleet needs the lead")
            self.auto_pause(st, f"fallback failed: {e}")
            self.save_state(st)
            return False
        st["active"] = [self.fb]
        st["phase"] = "fallback"
        st["transition"] = None
        st["since"] = iso(time.time())
        self.log("FALLBACK-OK", active=self.fb)
        self.save_state(st)
        return False

    def plan(self, st, instance, current, dirty):
        avail = self.mem(self.cfg["fleet"])
        for _ in range(len(self.cfg["slots"]) + len(self.cfg["companions"]) + 2):
            _, slot, target = self.target(st, instance * HOUR)
            incoming = [m for m in target if m not in current or m in dirty]
            outgoing = [m for m in current + dirty if m not in target]
            problem = None
            culprit = None
            for model in incoming:
                problem = self.precheck(model)
                if problem:
                    culprit = model
                    break
            if not problem:
                short = self.fits(avail, outgoing, incoming)
                if short:
                    culprit = incoming[-1] if incoming else None
                    problem = "predicted MemAvailable below floor: " + " ".join(short)
            if not problem:
                return slot, target
            if culprit is None or culprit == self.fb:
                raise Failure(f"cannot plan slot {slot}: {problem}")
            base = self.cfg["slots"].get(slot, {}).get("base")
            key = str(instance)
            if culprit == base:
                st.setdefault("demoted", {}).setdefault(key, []).append(slot)
                self.alert(st, "WARN", f"slot {slot} demoted for this hour: {culprit}: {problem}")
            else:
                st.setdefault("dropped", {}).setdefault(key, []).append(culprit)
                st.setdefault("companions", {}).pop(key, None)
                self.alert(st, "WARN", f"companion {culprit} skipped this hour: {problem}")
        raise Failure("planning did not settle")

    def tick(self):
        st = self.load_state()
        kind, reason = self.pause_reason()
        if kind and self.dry:
            self.log("HOLD-OFF", kind=kind, reason=json.dumps(reason), dry_run=json.dumps("the real tick holds off; planning as if resumed"))
            st["resync"] = True
            self.honor_locks = False
        elif kind:
            self.hold_off(st, kind, reason)
            self.finish(st)
            return 0
        st["lock"] = None
        try:
            return self.step(st)
        except Preempted as e:
            kind, reason = e.args
            self.alert(st, "WARN", f"{st.get('phase')} {json.dumps(st.get('transition'))} stopped at a lock ({reason}); the fleet is left as it is and adopted when the lock clears")
            self.hold_off(st, kind, reason)
            self.finish(st)
            return 1

    def step(self, st):
        if st.get("phase") in ("transition", "recovering"):
            self.alert(st, "ERROR", f"previous {st['phase']} {json.dumps(st.get('transition'))} did not finish; falling back to {self.fb}")
            self.fallback(st, f"interrupted {st['phase']}")
            self.finish(st)
            return 1
        instance, slot, target = self.target(st, self.instance * HOUR)
        try:
            state = self.observe()
        except Failure as e:
            self.alert(st, "ERROR", f"{e}; no transition this tick")
            self.finish(st)
            return 1
        up = [m for m, s in state.items() if s == "up"]
        dirty = [m for m, s in state.items() if s == "mixed"]
        if self.dry:
            up = list(st.get("active") or [self.fb])
        if "active" not in st or st.get("resync"):
            self.log("ADOPT", up=",".join(up) or "-", mixed=",".join(dirty) or "-")
            st["active"] = up
            st["resync"] = False
            current = up
        else:
            current = list(st["active"])
            foreign = [m for m in up + dirty if m not in current]
            if foreign:
                self.alert(st, "ERROR", f"fleet changed outside the rotation (unexpected {','.join(foreign)}); pausing; resume after checking")
                self.auto_pause(st, "foreign change: " + ",".join(foreign))
                self.finish(st)
                return 1
            lost = [m for m in current if state.get(m) != "up"]
            if self.fb in lost:
                self.alert(st, "ERROR", f"{self.fb} is {state.get(self.fb)} while the rotation expects it serving; no transition until it is back or the lead pauses")
                self.finish(st)
                return 1
            if lost:
                self.alert(st, "ERROR", f"{','.join(lost)} stopped serving; falling back to {self.fb}")
                self.fallback(st, "lost " + ",".join(lost))
                self.finish(st)
                return 1
        if sorted(current) == sorted(target) and not dirty:
            sick = [m for m in current if not self.healthy(m)]
            if sick and sick != [self.fb]:
                self.alert(st, "ERROR", f"api health failed for {','.join(sick)}; falling back to {self.fb}")
                self.fallback(st, "unhealthy " + ",".join(sick))
                self.finish(st)
                return 1
            if sick:
                self.alert(st, "WARN", f"{self.fb} api health failed; leaving production to its owners")
            st["phase"] = st.get("phase") if st.get("phase") in ("steady", "fallback") else "steady"
            self.finish(st)
            return 0
        try:
            if current:
                slot, target = self.plan(st, instance, current, dirty)
            else:
                slot, target = "restore", [self.fb]
                self.log("RESTORE", message=json.dumps(f"nothing serving; restoring {self.fb} before any slot model"))
        except Failure as e:
            self.alert(st, "ERROR", str(e))
            if self.fb not in current:
                self.fallback(st, str(e))
            self.finish(st)
            return 1
        st["slot"] = slot
        if sorted(current) == sorted(target) and not dirty:
            self.finish(st)
            return 0
        ok = self.converge(st, current, target, dirty, f"slot {slot} {iso(instance * HOUR)}")
        self.finish(st)
        return 0 if ok else 1

    def status_line(self, st):
        instance, slot, target = self.target(json.loads(json.dumps(st)), self.now())
        nxt = self.predict(st, self.now() + HOUR, 1)[0]
        return (f"ROTATION {iso(self.now())} phase={st.get('phase', 'new')} slot={slot} active={','.join(st.get('active') or []) or '-'} "
                f"target={','.join(target)} next={nxt['slot']}@{nxt['start']}({','.join(nxt['models'])}) "
                f"lock={json.dumps(st.get('lock') or '-')} alert={json.dumps(st.get('alert') or '-')}")

    def schedule_document(self, st):
        rows = self.predict(st, self.now(), 12)
        active = st.get("active") or []
        models = []
        for name, m in self.models.items():
            nxt = next((r["start"] for r in rows if name in r["models"]), None)
            models.append({"id": name, "title": m["title"], "runnable": bool(m.get("runnable")), "validated": bool(m.get("validated")),
                           "active": name in active, "port": (m.get("api") or {}).get("port"), "next_slot_start": nxt,
                           "not_runnable": m.get("not_runnable")})
        return {"generated": iso(time.time()), "phase": st.get("phase"), "active": active, "slots": rows, "models": models}

    def finish(self, st):
        st["updated"] = iso(time.time())
        line = self.status_line(st)
        st["status"] = line
        self.save_state(st)
        if not self.dry:
            os.makedirs(self.dir, exist_ok=True)
            with open(self.path("STATUS.tmp"), "w") as f:
                f.write(line + "\n")
            os.replace(self.path("STATUS.tmp"), self.path("STATUS"))
            with open(self.path("schedule.json.tmp"), "w") as f:
                json.dump(self.schedule_document(st), f, indent=1)
            os.replace(self.path("schedule.json.tmp"), self.path("schedule.json"))
        self.out(line)


class Locked:
    def __init__(self, rotation):
        self.rotation = rotation
        self.fd = None

    def __enter__(self):
        if self.rotation.dry:
            return self
        os.makedirs(self.rotation.dir, exist_ok=True)
        self.fd = os.open(self.rotation.path("rotation.lock"), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            raise Failure("another fleet_rotation command holds the state lock")
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
        return False


def cmd_now(rot, st):
    print(rot.status_line(st))
    kind, reason = rot.pause_reason()
    print(f"lock: {reason or 'none'}")
    for row in rot.predict(st, rot.now(), 7):
        print(f"  {row['start']}  {row['slot']:<22} {', '.join(row['models'])}")
    try:
        with open(rot.path("rotation.log")) as f:
            tail = f.read().splitlines()[-8:]
    except FileNotFoundError:
        tail = []
    for line in tail:
        print(f"  | {line}")


def cmd_sync(cfg, runner, lanes):
    remote = cfg["state_dir"].replace("~", "$HOME", 1)
    lock = cfg["lock"]
    holder_path = os.path.join(lanes, "PERF_HOLDER")
    pause_path = os.path.join(lanes, "ROTATION_PAUSE")
    holder = open(holder_path).read() if os.path.exists(holder_path) else ""
    pause = open(pause_path).read() if os.path.exists(pause_path) else None
    parts = [f"mkdir -p {remote}/lock", f"printf %s {shlex.quote(holder)} > {remote}/{lock['holder_file']}.tmp && mv -f {remote}/{lock['holder_file']}.tmp {remote}/{lock['holder_file']}"]
    parts.append(f"printf %s {shlex.quote(pause or 'coord pause')} > {remote}/{lock['mirror_pause_file']}" if pause is not None else f"rm -f {remote}/{lock['mirror_pause_file']}")
    parts.append(f"cat {remote}/STATUS 2>/dev/null || echo 'ROTATION not installed'; tail -n 3 {remote}/ALERT 2>/dev/null | sed 's/^/ALERT-LOG /'")
    rc, text = runner.run("hub", " && ".join(parts[:3]) + "; " + parts[3])
    if rc != 0:
        text = f"ROTATION sync failed rc={rc}: {text.strip()[:200]}\n"
    if not runner.dry:
        with open(os.path.join(lanes, "ROTATION_STATUS.tmp"), "w") as f:
            f.write(text)
        os.replace(os.path.join(lanes, "ROTATION_STATUS.tmp"), os.path.join(lanes, "ROTATION_STATUS"))
    print(text, end="")
    return 0 if rc == 0 else 1


def serve_schedule(rot):
    class Handler(http.server.BaseHTTPRequestHandler):
        def send(self, code, doc):
            body = json.dumps(doc, indent=1).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                with open(rot.path("schedule.json")) as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                return self.send(503, {"error": {"code": "schedule_unavailable", "message": "the rotation has not written a schedule yet"}})
            if self.path in ("/schedule.json", "/"):
                return self.send(200, doc)
            if self.path == "/v1/models":
                return self.send(200, {"object": "list", "data": [dict(m, object="model", owned_by="sparkpipe") for m in doc["models"]]})
            return self.send(404, {"error": {"code": "not_found", "message": "GET /schedule.json or /v1/models"}})

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", rot.cfg["schedule_port"]), Handler)
    server.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Config-driven fleet rotation (docs/FLEET_ROTATION.md)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print every command, execute none, write no state")
    ap.add_argument("--at", help="evaluate the schedule at this UTC time (YYYY-MM-DDTHH:MM:SSZ)")
    ap.add_argument("--state-dir", help="override state_dir")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("tick")
    sub.add_parser("status")
    sub.add_parser("now")
    p = sub.add_parser("pause")
    p.add_argument("reason", nargs="*")
    sub.add_parser("resume")
    p = sub.add_parser("force")
    p.add_argument("slot")
    p.add_argument("companion", nargs="?")
    sub.add_parser("skip")
    p = sub.add_parser("schedule")
    p.add_argument("--hours", type=int, default=12)
    p = sub.add_parser("converge")
    p.add_argument("models", nargs="*")
    p.add_argument("--rollback", action="store_true")
    sub.add_parser("check-config")
    p = sub.add_parser("sync")
    p.add_argument("--lanes", required=True)
    sub.add_parser("serve-schedule")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError, ConfigError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if args.state_dir:
        cfg["state_dir"] = args.state_dir
    runner = Runner(cfg, args.dry_run, print)
    rot = Rotation(cfg, runner, parse_time(args.at) if args.at else None)
    if args.command == "check-config":
        print(f"config OK: {len(cfg['models'])} models, runnable {', '.join(n for n, m in cfg['models'].items() if m.get('runnable'))}")
        return 0
    if args.command == "sync":
        return cmd_sync(cfg, runner, args.lanes)
    if args.command == "serve-schedule":
        serve_schedule(rot)
        return 0
    if args.command == "pause":
        if not rot.dry:
            os.makedirs(rot.dir, exist_ok=True)
            with open(rot.path(cfg["lock"]["pause_file"] + ".tmp"), "w") as f:
                f.write(f"manual {iso(time.time())}: {' '.join(args.reason) or 'no reason given'}\n")
            os.replace(rot.path(cfg["lock"]["pause_file"] + ".tmp"), rot.path(cfg["lock"]["pause_file"]))
        rot.log("PAUSE", reason=json.dumps(" ".join(args.reason)))
        return 0
    if args.command in ("status", "now", "schedule"):
        st = rot.load_state()
        if args.command == "status":
            print(rot.status_line(st))
        elif args.command == "now":
            cmd_now(rot, st)
        else:
            print(json.dumps(rot.predict(st, rot.now(), args.hours), indent=1))
        return 0
    try:
        with Locked(rot):
            st = rot.load_state()
            if args.command == "tick":
                return rot.tick()
            instance = rot.instance
            if args.command == "resume":
                if not rot.dry and os.path.exists(rot.path(cfg["lock"]["pause_file"])):
                    os.remove(rot.path(cfg["lock"]["pause_file"]))
                st["resync"] = True
                rot.log("RESUME")
            elif args.command == "force":
                if args.slot not in cfg["slots"]:
                    print(f"unknown slot {args.slot}; slots: {', '.join(cfg['slots'])}", file=sys.stderr)
                    return 2
                if args.companion and not cfg["models"].get(args.companion, {}).get("runnable"):
                    print(f"companion {args.companion} is not a runnable model", file=sys.stderr)
                    return 2
                st["override"] = {"instance": instance, "slot": args.slot, "companion": args.companion}
                rot.log("FORCE", slot=args.slot, companion=args.companion or "-", until=iso((instance + 1) * HOUR))
            elif args.command == "skip":
                st["skip"] = instance
                rot.log("SKIP", instance=iso(instance * HOUR))
            elif args.command == "converge":
                target = cfg["rollback_models"] if args.rollback else args.models
                unknown = [m for m in target if not cfg["models"].get(m, {}).get("runnable")]
                if not target or unknown:
                    print(f"converge needs runnable models; not runnable: {' '.join(unknown) or '(none given)'}", file=sys.stderr)
                    return 2
                rot.honor_locks = False
                state = rot.observe()
                up = [m for m, s in state.items() if s == "up"]
                dirty = [m for m, s in state.items() if s == "mixed"]
                ok = rot.converge(st, up, list(target), dirty, "manual converge")
                rot.finish(st)
                return 0 if ok else 1
            rot.finish(st)
            return 0
    except Failure as e:
        print(f"refused: {e}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import collections
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys

KINDS = ("count", "last", "report")
SSH_TIMEOUT_S = 120


class ConfigError(Exception):
    pass


def load_spec(path, section):
    with open(path) as f:
        doc = json.load(f)
    if section not in doc:
        raise ConfigError(f"{path} has no section {section!r}")
    spec = doc[section]
    for key in ("log", "gates"):
        if key not in spec:
            raise ConfigError(f"{path} {section}: missing {key!r}")
    for i, g in enumerate(spec["gates"]):
        kind = g.get("kind")
        if kind not in KINDS:
            raise ConfigError(f"{section} gate {i}: kind must be one of {', '.join(KINDS)}")
        if not g.get("name") or not g.get("pattern"):
            raise ConfigError(f"{section} gate {i}: name and pattern are required")
        re.compile(g["pattern"])
        if kind == "count" and "min" not in g and "max" not in g:
            raise ConfigError(f"{section} gate {g['name']}: a count gate needs min or max")
        if kind == "last" and "equals" not in g:
            raise ConfigError(f"{section} gate {g['name']}: a last gate needs equals")
    errsite = spec.get("errsite")
    if errsite is not None:
        if not errsite.get("pattern") or not isinstance(errsite.get("allow"), list):
            raise ConfigError(f"{section} errsite: pattern and an allow list are required")
        for a in errsite["allow"]:
            re.compile(a)
    return spec


def remote_script(root, spec):
    log = shlex.quote(f"sparkdata/{root}/{spec['log']}")
    lines = [f"L=$HOME/{log}", "test -f \"$L\" || { echo NOLOG; exit 0; }"]
    for i, g in enumerate(spec["gates"]):
        pat = shlex.quote(g["pattern"])
        if g["kind"] == "count":
            lines.append(f"echo \"G{i} $(grep -c -E -- {pat} \"$L\")\"")
        else:
            lines.append(f"echo \"G{i} $(grep -o -E -- {pat} \"$L\" | tail -n 1)\"")
    if spec.get("errsite"):
        lines.append(f"grep -o -E -- {shlex.quote(spec['errsite']['pattern'])} \"$L\" | sort | uniq -c | sed 's/^ */E /'")
    lines.append("echo END")
    return "\n".join(lines) + "\n"


def parse_output(out):
    gates, errsites, complete, nolog = {}, collections.Counter(), False, False
    for line in out.splitlines():
        if line == "NOLOG":
            nolog = True
        elif line == "END":
            complete = True
        elif line.startswith("G"):
            head, _, value = line.partition(" ")
            if head[1:].isdigit():
                gates[int(head[1:])] = value
        elif line.startswith("E "):
            parts = line.split(" ", 2)
            if len(parts) == 3 and parts[1].isdigit():
                errsites[parts[2]] += int(parts[1])
    return {"gates": gates, "errsites": errsites, "complete": complete, "nolog": nolog}


def evaluate(spec, per_node):
    failures = []
    report = []
    for i, g in enumerate(spec["gates"]):
        values = {}
        for host, res in per_node.items():
            if res.get("error") or res["nolog"] or not res["complete"]:
                continue
            values[host] = res["gates"].get(i, "")
        if g["kind"] == "count":
            for host, v in values.items():
                try:
                    n = int(v)
                except ValueError:
                    failures.append(f"{host} {g['name']}: count {v!r} unreadable")
                    continue
                if "min" in g and n < g["min"]:
                    failures.append(f"{host} {g['name']}: {n} < min {g['min']}")
                if "max" in g and n > g["max"]:
                    failures.append(f"{host} {g['name']}: {n} > max {g['max']}")
            dist = collections.Counter(values.values())
            report.append(f"{g['name']}: " + ", ".join(f"{v} x{n}" for v, n in sorted(dist.items())))
        elif g["kind"] == "last":
            for host, v in values.items():
                if v != g["equals"]:
                    failures.append(f"{host} {g['name']}: last {v!r} want {g['equals']!r}")
            dist = collections.Counter(values.values())
            report.append(f"{g['name']}: " + ", ".join(f"{v or '(none)'} x{n}" for v, n in sorted(dist.items())))
        else:
            dist = collections.Counter(values.values())
            report.extend(f"{g['name']}: {n:3d} {v or '(none)'}" for v, n in sorted(dist.items()))
    for host, res in per_node.items():
        if res.get("error"):
            failures.append(f"{host}: {res['error']}")
        elif res["nolog"]:
            failures.append(f"{host}: no {spec['log']}")
        elif not res["complete"]:
            failures.append(f"{host}: incomplete output")
    errsite = spec.get("errsite")
    total = collections.Counter()
    for res in per_node.values():
        total.update(res.get("errsites", {}))
    unknown = []
    if errsite:
        allow = [re.compile(a) for a in errsite["allow"]]
        for site, n in sorted(total.items(), key=lambda kv: -kv[1]):
            known = any(a.search(site) for a in allow)
            report.append(f"errsite {n:6d} {site}{'' if known else '  UNKNOWN'}")
            if not known:
                unknown.append(site)
        for site in unknown:
            hosts = sorted(h for h, r in per_node.items() if site in r.get("errsites", {}))
            failures.append(f"ERRSITE-UNKNOWN {site} on {' '.join(hosts)}")
    return failures, report


def collect(nodes, root, spec, ssh):
    script = remote_script(root, spec)

    def one(host):
        try:
            r = subprocess.run(ssh + [host, "bash -s"], input=script, capture_output=True, text=True, timeout=SSH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return host, {"error": "ssh-timeout"}
        if r.returncode != 0:
            return host, {"error": f"ssh rc={r.returncode} {r.stderr.strip()[-120:]}"}
        return host, parse_output(r.stdout)

    with concurrent.futures.ThreadPoolExecutor(len(nodes)) as pool:
        return dict(pool.map(one, nodes))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check each node's engine log of the release root against a gate section of the checks file.")
    ap.add_argument("--checks", required=True)
    ap.add_argument("--section", required=True)
    args = ap.parse_args(argv)
    env = os.environ
    missing = [k for k in ("RELEASE_NODES", "RELEASE_ROOT_NAME") if not env.get(k)]
    if missing:
        print(f"engine_logs.py: missing {' '.join(missing)}", file=sys.stderr)
        return 2
    try:
        spec = load_spec(args.checks, args.section)
    except (OSError, ValueError, ConfigError, re.error) as e:
        print(f"engine_logs.py: {e}", file=sys.stderr)
        return 2
    nodes = env["RELEASE_NODES"].split()
    ssh = shlex.split(env.get("RELEASE_SSH") or "ssh -o BatchMode=yes -o ConnectTimeout=10")
    per_node = collect(nodes, env["RELEASE_ROOT_NAME"], spec, ssh)
    failures, report = evaluate(spec, per_node)
    for line in report:
        print(line)
    for line in failures:
        print(f"FAIL {line}")
    print(f"ENGINE-LOGS {args.section} " + ("PASS" if not failures else f"FAIL {len(failures)}") + f" on {len(nodes)} nodes")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

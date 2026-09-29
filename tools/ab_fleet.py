#!/usr/bin/env python3
"""Fleet placement and arm-root guards for accuracy A/B campaigns (docs/ACCURACY_AB.md)."""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "tools" / "fleet_node_agent.sh"
FORMAT = "sparkpipe-ab-fleet-v1"
MESH_LANES = 16
GIB = 1 << 30
ROOT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ENV_KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")
MEMORY_MAX = re.compile(r"^[1-9][0-9]*[KMGT]?$")


class FleetError(Exception):
    pass


def agent_headroom_gib(agent_path=AGENT):
    text = Path(agent_path).read_text()
    match = re.search(r'HEADROOM_GIB="\$\{FLEET_AGENT_HEADROOM_GIB:-([0-9]+)\}"', text)
    if match is None:
        raise FleetError(f"{agent_path}: no FLEET_AGENT_HEADROOM_GIB default found")
    return int(match.group(1))


def agent_fleet_hosts(agent_path=AGENT):
    text = Path(agent_path).read_text()
    match = re.search(r'^FLEET_HOSTS="([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise FleetError(f"{agent_path}: no FLEET_HOSTS found")
    return match.group(1).split()


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    check_spec(spec)
    return spec


def arm_arena(spec, arm):
    return spec["arms"][arm]["arena"]


def baseline_arenas(spec):
    return {name for name, arena in spec["arenas"].items() if arena.get("resident_in_baseline")}


def slot_order(slot):
    return list(slot.get("keep", [])) + list(slot.get("start", []))


def slot_arms(slot, optional=True):
    arms = slot_order(slot)
    if optional:
        arms += list(slot.get("optional", []))
    return arms


def arm_needs(spec, arms, evicted=()):
    resident = baseline_arenas(spec) - set(evicted)
    needs = []
    for arm in arms:
        a = spec["arms"][arm]
        need = a["residentd_gib"]
        if a["arena"] not in resident:
            need += spec["arenas"][a["arena"]]["gib"]
            resident.add(a["arena"])
        needs.append((arm, int(math.ceil(need))))
    return needs


def agent_need_gib(spec, arm):
    worst = 0
    for slot in spec["slots"]:
        for arms in (slot_arms(slot, False), slot_arms(slot, True)):
            if arm in arms:
                worst = max(worst, dict(arm_needs(spec, arms))[arm])
    if worst == 0:
        raise FleetError(f"arm {arm} is in no slot")
    return worst


def check_spec(spec):
    errors = []
    if spec.get("format") != FORMAT:
        errors.append(f"format must be {FORMAT}")
    for key in ("campaign", "nodes", "production", "free_lanes", "arenas", "arms", "slots", "protected_paths", "private_directory_members", "common_env"):
        if key not in spec:
            errors.append(f"missing {key}")
    if errors:
        raise FleetError("; ".join(errors))
    hosts = agent_fleet_hosts()
    if spec["nodes"] != hosts:
        errors.append(f"nodes {spec['nodes']} differ from the fleet agent's FLEET_HOSTS {hosts}")
    production = spec["production"]
    free = set(spec["free_lanes"])
    if production["lane"] in free:
        errors.append(f"production lane {production['lane']} is listed as free")
    for lane in free:
        if not 0 <= lane < MESH_LANES:
            errors.append(f"free lane {lane} is not a weightd mesh lane 0..{MESH_LANES - 1}")
    if production["arena"] not in spec["arenas"]:
        errors.append(f"production arena {production['arena']} is not declared")
    lanes = {}
    roots = {}
    for arm, a in spec["arms"].items():
        for key in ("root", "lane", "pool_bytes", "arena", "residentd_gib", "memory_max", "pack_root", "port_group"):
            if key not in a:
                errors.append(f"arm {arm}: missing {key}")
        if any(key not in a for key in ("root", "lane", "pool_bytes", "arena", "memory_max")):
            continue
        if not ROOT_NAME.match(a["root"]):
            errors.append(f"arm {arm}: root name {a['root']!r} is not a fleet root name")
        if a["root"] == production["root"]:
            errors.append(f"arm {arm}: root is production's root {production['root']}")
        if a["lane"] not in free:
            errors.append(f"arm {arm}: lane {a['lane']} is not in the free lanes {sorted(free)}")
        if a["lane"] in lanes:
            errors.append(f"arm {arm}: lane {a['lane']} is also arm {lanes[a['lane']]}'s lane")
        lanes[a["lane"]] = arm
        if a["root"] in roots:
            errors.append(f"arm {arm}: root {a['root']} is also arm {roots[a['root']]}'s root")
        roots[a["root"]] = arm
        if a["arena"] not in spec["arenas"]:
            errors.append(f"arm {arm}: arena {a['arena']} is not declared")
        if not MEMORY_MAX.match(str(a["memory_max"])):
            errors.append(f"arm {arm}: memory_max {a['memory_max']!r} is not a systemd size")
        if a["arena"] == production["arena"] and a["pool_bytes"] != production["pool_bytes"]:
            errors.append(f"arm {arm}: shares production's arena but its pool {a['pool_bytes']} differs from production's {production['pool_bytes']}; weightd refuses the attach")
    pools = {}
    for arm, a in spec["arms"].items():
        if "arena" not in a or "pool_bytes" not in a:
            continue
        other = pools.setdefault(a["arena"], (arm, a["pool_bytes"]))
        if other[1] != a["pool_bytes"]:
            errors.append(f"arms {other[0]} and {arm} share arena {a['arena']} with different pools {other[1]} and {a['pool_bytes']}; weightd refuses the second attach")
    for key in spec["common_env"]:
        if not ENV_KEY.match(key) or key.startswith("AGENT_"):
            errors.append(f"common_env key {key!r} is not an exported env key")
        if key in ("SPARK_WEIGHTD_LANE", "SPARK_WEIGHTD_EXPERT_POOL_BYTES"):
            errors.append(f"common_env must not set {key}; it is per arm")
    for member in spec["private_directory_members"]:
        if not isinstance(member, str) or not member:
            errors.append("private_directory_members entries must be member names")
    names = set()
    for slot in spec["slots"]:
        if slot.get("name") in names:
            errors.append(f"slot {slot.get('name')} appears twice")
        names.add(slot.get("name"))
        seen_arenas = baseline_arenas(spec)
        for arm in slot_arms(slot):
            if arm not in spec["arms"]:
                errors.append(f"slot {slot['name']}: unknown arm {arm}")
                continue
            a = spec["arms"][arm]
            if a.get("arena_owner") is False and a["arena"] not in seen_arenas:
                errors.append(f"slot {slot['name']}: arm {arm} attaches arena {a['arena']} before its owner is up")
            seen_arenas.add(a["arena"])
        arms = slot_arms(slot)
        slot_lanes = [spec["arms"][arm]["lane"] for arm in arms if arm in spec["arms"]]
        if len(set(slot_lanes)) != len(slot_lanes):
            errors.append(f"slot {slot['name']}: two arms share a weightd lane")
        groups = [spec["arms"][arm]["port_group"] for arm in arms if arm in spec["arms"]]
        if len(set(groups)) != len(groups):
            errors.append(f"slot {slot['name']}: two concurrent arms share a port group")
    if errors:
        raise FleetError("; ".join(errors))


def render_env(spec, arm):
    a = spec["arms"][arm]
    lines = [
        "AGENT_ROLE=dev",
        "AGENT_SYNC=local",
        f"AGENT_MEMORY_MAX={a['memory_max']}",
        f"AGENT_MEMORY_NEED_GIB={agent_need_gib(spec, arm)}",
        f"SPARK_WEIGHTD_LANE={a['lane']}",
        f"SPARK_WEIGHTD_EXPERT_POOL_BYTES={a['pool_bytes']}",
    ]
    for key, value in spec["common_env"].items():
        lines.append(f"{key}={value}")
    for key, value in a.get("env", {}).items():
        if not ENV_KEY.match(key) or key.startswith("AGENT_"):
            raise FleetError(f"arm {arm}: env key {key!r} is not an exported env key")
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def node_available(node, basis):
    key = "mem_available_gib" if basis == "live" else "window_available_gib"
    if key not in node:
        raise FleetError(f"memory snapshot has no {key} for a node")
    return int(math.floor(float(node[key])))


def place(spec, memory, basis="live", headroom=None, evicted=()):
    if headroom is None:
        headroom = agent_headroom_gib()
    lanes_live = {}
    for host in spec["nodes"]:
        node = memory["nodes"].get(host)
        if node is None or "error" in node:
            raise FleetError(f"memory snapshot has no reading for {host}")
        lanes_live[host] = set(node.get("lanes_in_use", []))
    report = {"basis": basis, "headroom_gib": headroom, "slots": []}
    for slot in spec["slots"]:
        entry = {"name": slot["name"], "fits": True, "optional_fits": True, "nodes": {}, "lane_conflicts": []}
        for with_optional in (False, True):
            arms = slot_arms(slot, with_optional)
            needs = arm_needs(spec, arms, evicted)
            total = sum(need for _, need in needs)
            fits_all = True
            for host in spec["nodes"]:
                avail = node_available(memory["nodes"][host], basis)
                used = 0.0
                steps = []
                for arm, need in needs:
                    ok = avail - used - need >= headroom
                    steps.append({"arm": arm, "need_gib": round(need, 2), "available_before_gib": round(avail - used, 2), "gate": ok})
                    used += need
                    fits_all = fits_all and ok
                margin = avail - total - headroom
                if not with_optional:
                    entry["nodes"][host] = {"available_gib": round(avail, 2), "sum_gib": round(total, 2), "margin_gib": round(margin, 2), "steps": steps}
                else:
                    entry["nodes"][host]["with_optional_margin_gib"] = round(margin, 2)
            if with_optional:
                entry["optional_fits"] = fits_all if slot.get("optional") else entry["fits"]
            else:
                entry["fits"] = fits_all
                entry["sum_gib"] = round(total, 2)
            if with_optional:
                entry["sum_with_optional_gib"] = round(total, 2)
        required = set(slot_arms(slot, False))
        for host in spec["nodes"]:
            for arm in slot_arms(slot):
                lane = spec["arms"][arm]["lane"]
                if lane in lanes_live[host]:
                    entry["lane_conflicts"].append(f"{host}: lane {lane} ({arm}{'' if arm in required else ', optional'}) is in use")
                    if arm in required:
                        entry["fits"] = False
                    entry["optional_fits"] = False
        entry["worst_node"] = min(entry["nodes"], key=lambda h: entry["nodes"][h]["margin_gib"])
        entry["worst_margin_gib"] = entry["nodes"][entry["worst_node"]]["margin_gib"]
        report["slots"].append(entry)
    return report


def format_place(report):
    lines = [f"placement basis={report['basis']} headroom={report['headroom_gib']} GiB (fleet agent rule, whole GiB: floor(MemAvailable) - sum(AGENT_MEMORY_NEED_GIB of concurrent arms) >= headroom on every node)"]
    for slot in report["slots"]:
        verdict = "FITS" if slot["fits"] else "DOES NOT FIT"
        line = f"slot {slot['name']}: {verdict} sum={slot['sum_gib']} GiB worst={slot['worst_node']} margin={slot['worst_margin_gib']} GiB"
        if slot["sum_with_optional_gib"] != slot["sum_gib"]:
            line += f"; with optional arms sum={slot['sum_with_optional_gib']} GiB {'FITS' if slot['optional_fits'] else 'DOES NOT FIT'}"
        lines.append(line)
        for conflict in slot["lane_conflicts"]:
            lines.append(f"  lane conflict: {conflict}")
        short = [h for h, n in slot["nodes"].items() if n["margin_gib"] < 0]
        if short:
            lines.append("  short: " + ", ".join(f"{h} ({slot['nodes'][h]['margin_gib']} GiB)" for h in short))
    return "\n".join(lines)


def home_path(home, relative):
    return os.path.realpath(os.path.join(home, relative))


def under(path, parent):
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def parse_env_file(path):
    values = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line:
            continue
        key, sep, value = line.partition("=")
        if not sep or not ENV_KEY.match(key):
            raise FleetError(f"{path}:{number}: not KEY=VALUE")
        values[key] = value
    return values


def check_private_directory(root_real, protected, member, value, where, fresh, errors, allow_absolute=False):
    if not isinstance(value, str) or not value:
        errors.append(f"{where}: {member} must be a non-empty path")
        return None
    if value.startswith("/"):
        if not allow_absolute:
            errors.append(f"{where}: {member}={value!r} must be relative to the runtime root")
            return None
        target = os.path.realpath(value)
    else:
        parts = value.split("/")
        if any(part in ("", ".", "..") for part in parts):
            errors.append(f"{where}: {member}={value!r} is not a normalized relative path")
            return None
        target = os.path.realpath(os.path.join(root_real, value))
    if not under(target, root_real) or target == root_real:
        errors.append(f"{where}: {member}={value!r} resolves to {target}, outside the arm root {root_real}")
    for path in protected:
        if under(target, path) or under(path, target):
            errors.append(f"{where}: {member}={value!r} resolves to {target}, inside protected {path}")
    if fresh and os.path.isdir(target) and os.listdir(target):
        errors.append(f"{where}: {member} {target} is not empty; every run needs an empty directory")
    return target


def root_check(spec, arm, root, home=None, fresh=False):
    home = home or os.path.expanduser("~")
    errors = []
    a = spec["arms"][arm]
    root_real = os.path.realpath(root)
    protected = [home_path(home, p) for p in spec["protected_paths"]]
    if os.path.realpath(os.path.join(home, "sparkdata", a["root"])) != root_real:
        errors.append(f"{root} is not ~/sparkdata/{a['root']}")
    for path in protected:
        if under(root_real, path) or under(path, root_real):
            errors.append(f"arm root {root_real} overlaps protected {path}")
    env_path = Path(root) / "agent.env"
    if not env_path.is_file():
        errors.append("agent.env missing")
    else:
        env = parse_env_file(env_path)
        want = parse_env_file_text(render_env(spec, arm))
        for key, value in want.items():
            if env.get(key) != value:
                errors.append(f"agent.env {key}={env.get(key)!r}, spec wants {value!r}")
    private = {}
    deployment = Path(root) / "model_resident.json"
    if not deployment.is_file():
        errors.append("model_resident.json missing")
    else:
        doc = json.loads(deployment.read_text())
        for node in doc.get("nodes", []):
            runtime_root = node.get("runtime_root", "")
            if not runtime_root.endswith("/sparkdata/" + a["root"]):
                errors.append(f"model_resident.json rank {node.get('rank_index')}: runtime_root {runtime_root!r} is not the arm root")
            backing = node.get("kv_backing_directory")
            if backing is not None and not under(os.path.normpath(backing), os.path.normpath(runtime_root)):
                errors.append(f"model_resident.json rank {node.get('rank_index')}: kv_backing_directory {backing!r} is outside the arm root")
            if backing is not None and os.path.realpath(runtime_root) == root_real:
                private["kv_backing_directory"] = check_private_directory(root_real, protected, "kv_backing_directory", backing, "model_resident.json", fresh, errors, True)
    stages = sorted(Path(root).glob("config*/stage_*.json"))
    if not any(stage.parent.name == "config" for stage in stages):
        errors.append("config/stage_*.json missing")
    for stage in stages:
        doc = json.loads(stage.read_text())
        for member in spec["private_directory_members"]:
            if member in doc:
                target = check_private_directory(root_real, protected, member, doc[member], f"{stage.parent.name}/{stage.name}", fresh, errors)
                if target is not None:
                    private.setdefault(member, target)
    return errors, private


def parse_env_file_text(text):
    values = {}
    for line in text.splitlines():
        if line:
            key, _, value = line.partition("=")
            values[key] = value
    return values


def roots_disjoint(private_by_arm):
    errors = []
    seen = []
    for arm, dirs in private_by_arm.items():
        for member, target in dirs.items():
            if target is None:
                continue
            for other_arm, other_member, other in seen:
                if under(target, other) or under(other, target):
                    errors.append(f"arm {arm} {member} {target} overlaps arm {other_arm} {other_member} {other}")
            seen.append((arm, member, target))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("spec-check")
    p.add_argument("spec")
    p = sub.add_parser("env")
    p.add_argument("spec")
    p.add_argument("arm")
    p = sub.add_parser("place")
    p.add_argument("spec")
    p.add_argument("memory")
    p.add_argument("--basis", choices=("live", "window"), default="live")
    p.add_argument("--headroom", type=int)
    p.add_argument("--evicted", action="append", default=[])
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("root-check")
    p.add_argument("spec")
    p.add_argument("arms", nargs="+", help="ARM=ROOT pairs, or one ARM followed by ROOT")
    p.add_argument("--home")
    p.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        if args.command == "spec-check":
            print(f"spec {spec['campaign']}: OK ({len(spec['arms'])} arms, {len(spec['slots'])} slots, lanes {sorted(a['lane'] for a in spec['arms'].values())})")
            for arm in spec["arms"]:
                print(f"  {arm}: root {spec['arms'][arm]['root']} lane {spec['arms'][arm]['lane']} need {agent_need_gib(spec, arm)} GiB")
            return 0
        if args.command == "env":
            if args.arm not in spec["arms"]:
                raise FleetError(f"unknown arm {args.arm}")
            sys.stdout.write(render_env(spec, args.arm))
            return 0
        if args.command == "place":
            if args.headroom is not None:
                print(f"HEADROOM OVERRIDE {args.headroom} GiB (agent default {agent_headroom_gib()} GiB): a lead decision", file=sys.stderr)
            for arena in args.evicted:
                if arena not in spec["arenas"]:
                    raise FleetError(f"unknown arena {arena}")
            memory = json.loads(Path(args.memory).read_text())
            report = place(spec, memory, args.basis, args.headroom, args.evicted)
            print(json.dumps(report, indent=1) if args.json else format_place(report))
            return 0 if all(slot["fits"] for slot in report["slots"]) else 3
        pairs = []
        if len(args.arms) == 2 and "=" not in args.arms[0]:
            pairs = [tuple(args.arms)]
        else:
            for item in args.arms:
                arm, sep, root = item.partition("=")
                if not sep:
                    raise FleetError(f"{item}: expected ARM=ROOT")
                pairs.append((arm, root))
        all_errors = []
        private = {}
        for arm, root in pairs:
            if arm not in spec["arms"]:
                raise FleetError(f"unknown arm {arm}")
            errors, dirs = root_check(spec, arm, root, args.home, args.fresh)
            all_errors += [f"{arm}: {e}" for e in errors]
            private[arm] = dirs
        all_errors += roots_disjoint(private)
        for error in all_errors:
            print("REFUSED " + error)
        if all_errors:
            return 1
        print("root-check OK: " + ", ".join(f"{arm}={root}" for arm, root in pairs))
        return 0
    except FleetError as error:
        print(f"ab_fleet: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

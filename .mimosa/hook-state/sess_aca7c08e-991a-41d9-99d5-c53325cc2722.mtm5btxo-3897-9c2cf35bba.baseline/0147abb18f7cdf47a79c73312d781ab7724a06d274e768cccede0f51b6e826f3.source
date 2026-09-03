#!/usr/bin/env python3
"""The hardware topology is one fact, three projections - prove all three agree.

examples/topologies/*.json is the single hardware description: node types,
compute nodes, fabrics, per-mode port assignments. tools/generate_topology.py
validates it and derives the C tables (deployment/) and the phase-6 runtime
configs (examples/runtime_configs/). What this gate pins down:

  1. every shipped topology validates - references exist, ranks are
     contiguous, per-mode port counts hold, every endpoint has an address
  2. the validation actually bites - mutated topologies are rejected
  3. the generated files are fresh - a hand edit of a runtime config or a
     stale table fails --check
  4. the emitted runtime projections carry the fields fabric_topology.c
     validates against (rail/switch/port counts per mode)
  5. the generated C compiles under -Wall -Wextra -Werror

No numpy, no third-party packages: same host constraint as the generator.
"""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import generate_topology  # noqa: E402

SCHEMA = json.loads((ROOT / "schema" / "hardware_topology.schema.json").read_text(encoding="utf-8"))
FAILURES = 0


def check(condition: bool, label: str) -> None:
    global FAILURES
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILURES += 1


def load_example(name: str) -> dict:
    return json.loads((ROOT / "examples" / "topologies" / name).read_text(encoding="utf-8"))


def rejects(document: dict, label: str) -> None:
    try:
        generate_topology.validate_topology(document, SCHEMA, "mutation")
    except generate_topology.TopologyError as error:
        print(f"  ok   {label} ({error})")
        return
    check(False, label)


def main() -> int:
    examples = sorted((ROOT / "examples" / "topologies").glob("*.json"))
    check(len(examples) == 3, f"three shipped topologies, got {len(examples)}")

    # 1. the shipped topologies validate and cover the three modes
    modes = set()
    for path in examples:
        document = json.loads(path.read_text(encoding="utf-8"))
        try:
            generate_topology.validate_topology(document, SCHEMA, path.name)
            print(f"  ok   {path.name} validates")
        except generate_topology.TopologyError as error:
            check(False, f"{path.name} validates ({error})")
        modes.add(document["topology"]["mode"])
    check(modes == {"ring", "single_switch", "dual_switch"}, f"all three modes covered: {sorted(modes)}")

    nodes_by_example = {
        path.name: len(json.loads(path.read_text(encoding="utf-8"))["compute_nodes"])
        for path in examples
    }
    check(nodes_by_example.get("ring_13node_bringup.json") == 13,
          "ring bring-up has the 13 July nodes")
    check(nodes_by_example.get("dual_switch_16node_production.json") == 16,
          "dual-switch production has 16 nodes")

    # 2. the validation bites: each mutation breaks one stated rule
    base = load_example("dual_switch_16node_production.json")

    mutated = copy.deepcopy(base)
    mutated["compute_nodes"][3]["node_type"] = "nonexistent"
    rejects(mutated, "unknown node type rejected")

    mutated = copy.deepcopy(base)
    mutated["compute_nodes"][0]["ports"][0]["fabric"] = "switch_c"
    rejects(mutated, "unknown fabric rejected")

    mutated = copy.deepcopy(base)
    mutated["compute_nodes"][5]["rank"] = 16
    rejects(mutated, "rank gap rejected")

    mutated = copy.deepcopy(base)
    mutated["compute_nodes"][5]["rank"] = 4
    rejects(mutated, "duplicate rank rejected")

    mutated = copy.deepcopy(base)
    mutated["compute_nodes"][2]["ports"] = mutated["compute_nodes"][2]["ports"][:1]
    rejects(mutated, "dual-switch node with one port rejected")

    mutated = copy.deepcopy(base)
    mutated["compute_nodes"][2]["ports"][1]["fabric"] = "switch_a"
    rejects(mutated, "dual-switch node with both ports on one switch rejected")

    mutated = copy.deepcopy(base)
    mutated["compute_nodes"][1]["ports"][0]["ipv4"] = base["compute_nodes"][0]["ports"][0]["ipv4"]
    rejects(mutated, "reused endpoint address rejected")

    mutated = copy.deepcopy(base)
    mutated["compute_nodes"][1]["ports"][0]["ipv4"] = "10.16.1.300"
    rejects(mutated, "invalid ipv4 rejected")

    ring = load_example("ring_13node_bringup.json")
    mutated = copy.deepcopy(ring)
    third_port = dict(mutated["compute_nodes"][0]["ports"][0])
    third_port["ipv4"] = "10.13.99.1"
    mutated["compute_nodes"][0]["ports"].append(third_port)
    rejects(mutated, "ring node with three ports rejected")

    single = load_example("single_switch_16node.json")
    mutated = copy.deepcopy(single)
    mutated["compute_nodes"][4]["ports"] = []
    rejects(mutated, "single-switch node with no port rejected")

    mutated = copy.deepcopy(base)
    del mutated["node_types"][0]["fp8_tflops"]
    rejects(mutated, "node type without compute characteristics rejected")

    # 3. generated files are fresh (byte-exact against the tree)
    fresh = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "generate_topology.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True)
    check(fresh.returncode == 0, f"generator --check: {fresh.stdout.strip()} {fresh.stderr.strip()}")

    # 4. the projections carry what fabric_topology.c validates
    expectations = {
        "ring_debug_single_rail.json": ("ring_debug_single_rail", 1, 0, 2, False),
        "single_switch_100g.json": ("single_switch_single_rail", 1, 1, 1, False),
        "dual_switch_dual_rail_future.json": ("dual_switch_dual_rail_future", 2, 2, 2, True),
    }
    for filename, (mode, rails, switches, ports, future) in expectations.items():
        config = json.loads((ROOT / "examples" / "runtime_configs" / filename).read_text(encoding="utf-8"))
        check(config["mode"] == mode
              and config["rail_count"] == rails
              and config["switch_count"] == switches
              and config["active_ports_per_node"] == ports,
              f"{filename} projects {mode} rails={rails} switches={switches} ports={ports}")
        if future:
            check(config.get("enabled") is False, f"{filename} stays fail-closed")
    ring_config = json.loads(
        (ROOT / "examples" / "runtime_configs" / "ring_debug_single_rail.json").read_text(encoding="utf-8"))
    check(ring_config.get("debug_only") is True, "ring projection stays debug-only")

    # 5. the generated C compiles clean, and its registry covers the examples
    with tempfile.TemporaryDirectory() as scratch:
        object_path = Path(scratch) / "tables.o"
        compile_run = subprocess.run(
            ["gcc", "-O2", "-Wall", "-Wextra", "-Werror",
             "-Iinclude", "-Ideployment/include", "-c",
             "deployment/src/spark_hardware_topology_tables.c", "-o", str(object_path)],
            cwd=ROOT, capture_output=True, text=True)
        check(compile_run.returncode == 0,
              f"generated tables compile: {compile_run.stderr.strip()}")
    tables = (ROOT / "deployment" / "src" / "spark_hardware_topology_tables.c").read_text(encoding="utf-8")
    for path in examples:
        name = json.loads(path.read_text(encoding="utf-8"))["topology"]["name"]
        check(f"spark_hardware_topology_{name}," in tables, f"registry carries {name}")

    print(f"\n{'FAIL' if FAILURES else 'PASS'} ({FAILURES} failures)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())

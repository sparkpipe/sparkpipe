#!/usr/bin/env python3
"""Lane-2 qwen38_max shared-socket deployment generator (TP16, multidev).

Builds the PRIVATE per-attempt model_resident deployment and the per-rank
stage configuration for developer lane 2 under the shared-socket protocol
of docs/MULTIDEV_QUICKSTART.md. The wrapper that drives this file is
tools/qwen38max_multidev_run_family.sh; it stages the output under
$SPARK_QUEUE_RUNTIME_ROOT so every runtime root, KV directory and listener
of this lane stays private to one queue attempt.

Topology: TP16 single stage, 16 world ranks over spark0..sparkf. Logical
rank = index in --nodes; the mesh map is the identity permutation
(SPARK_TP_MESH_RANKS=0,...,15), so rank i lives on spark{hex(i)} and its
pack is the OPERATOR-PLACED set (fleet inventory, Sep 12-15):
/home/{host}/sparkdata/qwenmax.nvfp4.tp16/packs/qwenmax.nvfp4.tp16.rank{i}.sp
with .experts + .receipt.json sidecars - grep the fleet pack inventory
before any warm read (the operator's expectation IS the map).

The stage configuration member set mirrors the qwen38_max serving
adapter's EXACT contract (tools/qwen38max_gen_deployment.py precedent:
schema_version, model_revision, stage_pack_path, max_sequence_positions,
tp_degree - a missing OR EXTRA member is a load error). The module takes
its remaining TP wiring from the deployment transport block plus the
residentd environment at admission; do not add members here.

Port ledger (lane 2 owns control 23032-23047, collective 53032-53047
(u16-valid, #1094), transport 64032-64047; every number below stays
inside those blocks):

  control_endpoint      23032 + rank   BOUND (residentd client listener;
                                      one per host, so the block is full).
  transport control     64032          TOPOLOGY ONLY under the shared
                        (base)         socket: the host-rdma backend and
                                      the module's embedded tp collective
                                      connect through the weightd mesh
                                      (SPARK_WEIGHTD_SOCKET + lane), so
                                      this binds nothing today; it stays
                                      inside lane 2's block so no other
                                      lane can collide.
  collective            53032..53047   RESERVED for this family's
                                      admission-time TCP collective shape
                                      if one ever binds (the module's
                                      embedded transport currently rides
                                      the weightd mesh); recorded here so
                                      the wrapper reserves the block with
                                      --ports and any future binding path
                                      is already lane-scoped.

Budgets: pack bytes (experts + full-resolution spine) ride the operator's
shared weightd arena - the lane-job device budget covers the resident side
only (KV, workspace, activations). Sizing record:
model-families/qwen38_max/smoke_experts.json +
tools/devcycle/lane_budget_calc.py (PR #1085 both-view numbers).

Fail-closed: host names validated, rank bounds checked, KV backing must be
finite, and --check mode must reproduce committed output byte for byte.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

LANE = 2
WORLD = 16
TP = 16
HEX = "0123456789abcdef"
HOSTS = [f"spark{HEX[i]}" for i in range(WORLD)]

CONTROL_BASE = 23032                            # 23032 .. 23047
COLLECTIVE_BASE = 53032                         # 53032 .. 53047 (u16-valid, #1094)
TRANSPORT_BASE = 64032                          # 64032 .. 64047

MESH_RANKS = ",".join(str(i) for i in range(WORLD))

# The compiled module's serving pin (modules/qwen38_max_resident_decode_stage/
# Makefile QWEN38_MODEL_REVISION). The serving adapter matches this EXACTLY
# against the module build; a drifted revision refuses to load.
MODEL_REVISION = "d2dc35658bcf77e66643428cb52e774cc3b5bd29"
NODE_TARGET = "cuda.sm121.qwen38.resident_decode_stage.fp8"

DEPLOYED_PACK_TEMPLATE = "/home/{host}/sparkdata/qwenmax.nvfp4.tp16/packs/qwenmax.nvfp4.tp16.rank{rank}.sp"
DEFAULT_KV_BACKING_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_KV_PAGE_CAPACITY = 16 * ((32768 + 63) // 64)


def host_of(rank: int) -> str:
    return HOSTS[rank]


def deployed_pack(rank: int) -> str:
    return DEPLOYED_PACK_TEMPLATE.format(host=host_of(rank), rank=rank)


def stage_config(rank: int) -> dict:
    """The adapter's EXACT member set - no additions (load error)."""
    return {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "stage_pack_path": "packs/qwenmax.nvfp4.tp16.rank%d.sp" % rank,
        "max_sequence_positions": 4096,
        "tp_degree": TP,
    }


def resident_deployment(runtime_root: str, weightd_socket: str,
                        kv_backing_bytes: int = DEFAULT_KV_BACKING_BYTES,
                        kv_page_capacity: int = DEFAULT_KV_PAGE_CAPACITY) -> dict:
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": runtime_root,
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/adapter.json",
            "kv_backing_directory": os.path.join(runtime_root, "kvcache"),
            "kv_backing_maximum_bytes": kv_backing_bytes,
            "control_endpoint": {
                "kind": "tcp",
                "host": host,
                "port": CONTROL_BASE + rank,
            },
        })
    return {
        "schema_version": 2,
        "coordinator_rank_index": 0,
        "adapter": {"shared_object_path": "lib/model_serving_adapter.so"},
        "driver": {
            "shared_object_path": "lib/model_driver.so",
            "program_name": "resident_decode",
        },
        "transport": {
            "shared_object_path": "lib/hidden_transport.so",
            "mode": "host-rdma",
            "control_port_base": TRANSPORT_BASE,
        },
        # Shared-socket lazy attach (multidev): the operator-tracked fleet
        # daemon owns byte residency; the module region loader serves pack
        # regions as zero-copy consumer-map slices through this socket.
        "weightd": {
            "socket_path": weightd_socket,
        },
        "runtime_limits": {
            "max_inflight_submissions": 4,
            "max_active_sequences": 16,
            "max_input_rows": 16,
            "resident_sequence_capacity": 16,
            # One logical page per 64-token block per resident sequence,
            # all physically resident (the family-wide pool law).
            "kv_logical_page_capacity": kv_page_capacity,
            "kv_physical_page_capacity": kv_page_capacity,
        },
        "nodes": nodes,
    }


def render(rank: int, runtime_root: str, weightd_socket: str,
           kv_backing_bytes: int) -> dict[str, str]:
    return {
        "deployment.json": json.dumps(resident_deployment(
            runtime_root, weightd_socket, kv_backing_bytes), indent=2) + "\n",
        "adapter.json": json.dumps(stage_config(rank), indent=2) + "\n",
    }


def budgets(source: str) -> int:
    """Emit 'expert_pool_bytes spine_bytes' (per-node, tp-sharded).

    The arena-side sizing record from the census manifest: the wrapper
    consumes this to default QMAX_EXPERT_POOL_BYTES/QMAX_SPINE_BUDGET_BYTES
    without any queue-cmd env syntax.
    """
    document = json.load(open(source, encoding="utf-8"))
    nodes = int(document["nodes"])
    if document.get("expert_shard") != "tp" or nodes < 1:
        raise SystemExit("budgets need a tp-sharded manifest with nodes >= 1")
    pool = -(-sum(int(e["bytes"]) for e in document["experts"]) // nodes)
    spine = -(-int(document["spine_bytes"]) // nodes)
    print(f"{pool} {spine}")
    return 0


def emit_wset(source: str, output: str) -> int:
    """Materialize the smoke-expert working set as a .wset binary.

    Raw little-endian (layer u32, expert u32) pairs, deduplicated and
    sorted - the format tools/weightd_warm.c --wset validates against the
    pack manifest. Source: model-families/qwen38_max/smoke_experts.json
    (machine-generated; PR #1085 census receipt).
    """
    document = json.load(open(source, encoding="utf-8"))
    if document.get("family") != "qwen38_max":
        raise SystemExit("wset source is not the qwen38_max manifest")
    pairs = sorted({(int(e["layer"]), int(e["expert"]))
                    for e in document["experts"]})
    with open(output, "wb") as handle:
        for layer, expert in pairs:
            handle.write(layer.to_bytes(4, "little"))
            handle.write(expert.to_bytes(4, "little"))
    print(json.dumps({"wset": output, "keys": len(pairs)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime-root",
                        help="private per-attempt runtime root")
    parser.add_argument("--weightd-socket")
    parser.add_argument("--output-dir")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--kv-backing-bytes", type=int,
                        default=DEFAULT_KV_BACKING_BYTES)
    parser.add_argument("--emit-wset", metavar="OUTPUT",
                        help="write the smoke-expert .wset and exit")
    parser.add_argument("--budgets", metavar="MANIFEST",
                        help="print 'expert_pool_bytes spine_bytes' "
                             "(per-node, tp-sharded) and exit")
    parser.add_argument("--wset-source", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model-families", "qwen38_max", "smoke_experts.json"))
    parser.add_argument("--check", action="store_true",
                        help="regenerate and compare against output-dir "
                             "instead of writing")
    arguments = parser.parse_args()

    if arguments.budgets:
        return budgets(arguments.budgets)
    if arguments.emit_wset:
        return emit_wset(arguments.wset_source, arguments.emit_wset)

    missing = [name for name, value in (
        ("--runtime-root", arguments.runtime_root),
        ("--weightd-socket", arguments.weightd_socket),
        ("--output-dir", arguments.output_dir),
        ("--rank", arguments.rank)) if value is None]
    if missing:
        raise SystemExit("the following arguments are required: "
                         + ", ".join(missing))
    if not 0 <= arguments.rank < WORLD:
        raise SystemExit(f"rank must be 0..{WORLD - 1}")
    if not re.fullmatch(r"/tmp/sparkqueue-[0-9a-f]{32}", arguments.runtime_root):
        raise SystemExit("runtime root must be the private queue namespace")
    if arguments.kv_backing_bytes <= 0:
        raise SystemExit("kv backing must be a finite positive cap")
    if not arguments.weightd_socket.startswith("/run/sparkpipe-weightd-shared/"):
        raise SystemExit("weightd socket must be the operator's shared unit")

    files = render(arguments.rank, arguments.runtime_root,
                   arguments.weightd_socket, arguments.kv_backing_bytes)
    output_dir = os.path.abspath(arguments.output_dir)
    if arguments.check:
        for name, text in files.items():
            current = open(os.path.join(output_dir, name), encoding="utf-8").read()
            if current != text:
                print(f"drift in {name}: regenerate with "
                      "tools/qwen38max_multidev_lane.py", file=sys.stderr)
                return 1
        print("multidev lane output matches")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    print(json.dumps({"rank": arguments.rank, "output": output_dir,
                      "control_port": CONTROL_BASE + arguments.rank,
                      "transport_base": TRANSPORT_BASE,
                      "pack": deployed_pack(arguments.rank)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

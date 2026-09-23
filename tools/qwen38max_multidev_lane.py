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
/home/{host}/sparkdata/qwenmax.nvfp4.tp16/packs/qwenmax.nvfp4.tp16.rank{i:x}.sp
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

# Placed-set pack names carry the HEX rank suffix (ranka..rankf for ranks
# 10-15, matching host names spark0..sparkf — the operator placement
# convention; decimal rank10..15 would miss on 6/16 nodes).
DEPLOYED_PACK_TEMPLATE = "/home/{host}/sparkdata/qwenmax.nvfp4.tp16/packs/qwenmax.nvfp4.tp16.rank{rank:x}.sp"
DEFAULT_KV_BACKING_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_KV_PAGE_CAPACITY = 16 * ((32768 + 63) // 64)


def host_of(rank: int) -> str:
    return HOSTS[rank]


def deployed_pack(rank: int) -> str:
    return DEPLOYED_PACK_TEMPLATE.format(host=host_of(rank), rank=rank)


def stage_config(rank: int) -> dict:
    """The adapter's EXACT member set - no additions (load error)."""
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "stage_pack_path": "packs/qwenmax.nvfp4.tp16.rank%x.sp" % rank,
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
        # Caps from the family adapter descriptor: max_inflight is 1
        # (SparkQwen38MaxServingDescriptor) - the loader rejects anything
        # above at deployment_validation (model_serving_adapter.c:232, the
        # attach-r15j / gemma4 launch-6 lesson; 4 was inherited from the
        # GLM TP16 template). active/rows/resident 16 sit inside the
        # descriptor's 512 caps.
        "runtime_limits": {
            "max_inflight_submissions": 1,
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


POOLS_MANIFEST_NAME = "smoke_experts_pools.json"


def pack_spine_allocation(pack: str) -> int:
    """Measure one rank pack's spine allocation exactly as the lazy tier
    sizes it (runtime/spark_weightd_manifest.c build_spine): the spans
    BETWEEN expert ranges, each padded to a cumulative 256-byte
    alignment. The naive spine/nodes division under-declares every rank
    because the spine holds REPLICATED tensors (embeddings, norms, head)
    that do not shard across TP ranks (rank 0 measures 14,761,125,376 B
    against a 9,420,097,264 B /16 budget - the r12p publish failure).
    """
    import struct
    sidecar_path = pack + ".experts"
    sidecar = open(sidecar_path, "rb").read()
    if sidecar[:4] != b"WEPX":
        raise SystemExit("budgets: bad sidecar magic in %s" % sidecar_path)
    count = struct.unpack_from("<I", sidecar, 8)[0]
    if len(sidecar) < 16 + count * 48:
        raise SystemExit("budgets: truncated sidecar %s" % sidecar_path)
    ranges = []
    for i in range(count):
        off, size = struct.unpack_from("<QQ", sidecar, 16 + i * 48 + 16)
        ranges.append((off, size))
    ranges.sort()
    pack_bytes = os.path.getsize(pack)
    cursor = 0
    allocation = 0
    spans = 0
    for offset, size in ranges:
        if offset > cursor:
            padding = (cursor - allocation) & 255
            allocation += padding + (offset - cursor)
            spans += 1
        cursor = offset + size
    if pack_bytes > cursor:
        padding = (cursor - allocation) & 255
        allocation += padding + (pack_bytes - cursor)
        spans += 1
    if spans == 0 or allocation == 0:
        raise SystemExit("budgets: %s has no spine spans" % sidecar_path)
    # +255: the lazy tier's budget check is (allocation + 255) > budget
    # (lazy_spine_load reserves alignment headroom on top of the aligned
    # allocation - r13p4 failed every node by exactly 255 bytes).
    return allocation + 255


def budgets(source: str, rank: int, pack: str = None) -> int:
    """Emit 'expert_pool_bytes raw_bytes spine_bytes' for ONE rank.

    The pool default is the CHUNK-BASIS working-set size for this exact
    rank from the pools manifest (model-families/qwen38_max/
    smoke_experts_pools.json): the 2 MiB chunk-union over every sidecar
    span of every smoke expert placed on the rank's pack - the same
    arithmetic the weightd lazy tier materializes at (VmmReserve:
    GRANULARITY_MINIMUM with a 2 MiB floor, runtime/spark_weightd.c).
    Raw division of the census bytes (sum/16) under-declares every rank
    (this family measures 1.40x-1.44x; lane 5's 2.15x and lane 0's 5.17x
    are their families' factors, not ours). The raw per-rank number is
    emitted second so receipts can carry BOTH bases. The spine default is
    MEASURED from this rank's placed pack (--pack; the exact lazy-tier
    arithmetic including replicated, non-sharded spine tensors); without
    a pack it falls back to the raw spine/nodes division, which
    under-declares replicated spines and must only size pin-all tiers.
    """
    document = json.load(open(source, encoding="utf-8"))
    nodes = int(document["nodes"])
    if document.get("expert_shard") != "tp" or nodes < 1:
        raise SystemExit("budgets need a tp-sharded manifest with nodes >= 1")
    if rank is None or not 0 <= rank < nodes:
        raise SystemExit("budgets need --rank in 0..%d" % (nodes - 1,))
    pools_path = os.path.join(os.path.dirname(os.path.abspath(source)),
                              POOLS_MANIFEST_NAME)
    try:
        pools = json.load(open(pools_path, encoding="utf-8"))
        chunked = int(pools["per_rank"][str(rank)])
        raw = int(pools["per_rank_raw"][str(rank)])
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(
            "budgets need %s with per_rank/per_rank_raw for rank %d "
            "(chunk-basis sizing record; missing key: %s)"
            % (pools_path, rank, error))
    if pack is not None:
        spine = pack_spine_allocation(pack)
    else:
        spine = -(-int(document["spine_bytes"]) // nodes)
    print(f"{chunked} {raw} {spine}")
    return 0


def emit_wset(source: str, output: str, rank: int = None) -> int:
    """Materialize the smoke-expert working set as a .wset binary.

    Raw little-endian (layer u32, expert u32) pairs, deduplicated and
    sorted - the format tools/weightd_warm.c --wset validates against the
    pack manifest. Source: model-families/qwen38_max/smoke_experts.json
    (machine-generated; PR #1085 census receipt).

    --rank N keeps only THIS rank's shard of the census: the packs are
    TP16 rank shards (global expert windows of 512/16), and weightd_warm
    rejects keys absent from the node's own pack manifest - the whole-
    census wset spans all 16 packs and is valid on none. Rank 0 filters
    to 104 keys (the pools manifest's per_rank_experts count for rank 0).
    """
    document = json.load(open(source, encoding="utf-8"))
    if document.get("family") != "qwen38_max":
        raise SystemExit("wset source is not the qwen38_max manifest")
    nodes = int(document.get("nodes") or 0)
    pairs = sorted({(int(e["layer"]), int(e["expert"]))
                    for e in document["experts"]})
    if rank is not None:
        if nodes < 1:
            raise SystemExit("wset rank filter needs a sharded manifest")
        routed = int(document.get("routed_expert_count") or 512)
        window = max(1, routed // nodes)
        pairs = [(layer, expert) for layer, expert in pairs
                 if expert // window == rank]
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
    parser.add_argument("--pack", metavar="PACK",
                        help="with --budgets: measure the spine default from "
                             "this rank pack's .experts sidecar (the exact "
                             "lazy-tier allocation; requires the placed pack)")
    parser.add_argument("--kv-backing-bytes", type=int,
                        default=DEFAULT_KV_BACKING_BYTES)
    parser.add_argument("--emit-wset", metavar="OUTPUT",
                        help="write the smoke-expert .wset and exit")
    parser.add_argument("--budgets", metavar="MANIFEST",
                        help="print 'expert_pool_bytes raw_bytes spine_bytes' "
                             "for --rank (chunk-basis pool from the sibling "
                             "pools manifest) and exit")
    parser.add_argument("--wset-source", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model-families", "qwen38_max", "smoke_experts.json"))
    parser.add_argument("--check", action="store_true",
                        help="regenerate and compare against output-dir "
                             "instead of writing")
    arguments = parser.parse_args()

    if arguments.budgets:
        return budgets(arguments.budgets, arguments.rank, arguments.pack)
    if arguments.emit_wset:
        return emit_wset(arguments.wset_source, arguments.emit_wset, arguments.rank)

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

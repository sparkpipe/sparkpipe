#!/usr/bin/env python3
"""Lane-9 ling shared-socket deployment generator (TP16, multidev).

Builds the PRIVATE per-attempt model_resident deployment and the per-rank
serving adapter configuration for developer lane 9 under the shared-socket
protocol of docs/MULTIDEV_QUICKSTART.md. The wrapper that drives this file
is tools/ling_multidev_run_family.sh; it stages the output under
$SPARK_QUEUE_RUNTIME_ROOT so every runtime root, KV directory and listener
of this lane stays private to one queue attempt.

Topology: TP16 single stage, 16 world ranks over spark0..sparkf. Logical
rank = index in --nodes; the mesh map is the identity permutation
(SPARK_TP_MESH_RANKS=0,...,15), so rank i lives on spark{hex(i)} and its
pack is the OPERATOR-PLACED set (fleet inventory, verified 2026-09-22 on
all 16 nodes with v2 .experts + .sha256 sidecars):
  bf16 arm  /home/{host}/sparkdata/ling.bf16.tp16/packs/ling.bf16.tp16.rank{i}.sp
  fp8 arm   /home/{host}/sparkdata/ling.fp8.tp16/packs/ling.fp8.tp16.rank{i}.sp
Zero ceph: the arms are placed; the wrapper only ever symlinks.
(lingfin.bf16.tp16 is placed on 15/16 - spark7 lacks it; tracked as the
lane-9 restore item, not consumed by this generator.)

The stage configuration member set mirrors the ling serving adapter's
EXACT contract (modules/ling_resident_decode_stage/source/
spark_ling_serving_adapter.c SparkLingServingConfigurationMembers; the
PR #1021 LING-T1 bring-up config is the semantic precedent): schema
version 3 with the full adaptive tp_collective (runtime/serving_adapter_template.c
adaptive member set - a missing OR EXTRA member is a load error). The
hidden-transport backend rides the operator's weightd mesh under the
shared socket (k3/qwen38max M1 finding), so only the tp_collective
listener actually binds; every other number below stays inside lane 9's
blocks so no future binding path can collide with another lane.

Port ledger (lane 9 owns control 23144-23159, collective 53144-53159
(u16-valid, #1094), transport 64144-64159, session 23744-23807; every
number below stays inside those blocks):

  control_endpoint      23144 + rank   BOUND (residentd client listener;
                                      one per host, so the block is full).
  tp_collective listen  53144 + rank   BOUND (host TCP collective; the
                                      ling adapter requires it at
                                      tp_degree 16, LING-T1 precedent).
  tp peer_ports         53144..53159   CONTIGUOUS (the adapter parses
                                      with require_contiguous_peer_ports).
  session_ports         23744 + col    TOPOLOGY ONLY under the shared
  session_ports_hc      23760 + col    socket: ApplyTopology records the
                                      matrices for the mesh path and they
                                      bind nothing today (k3 lane-3 M1
                                      finding, same runtime). Row r is
                                      host r's local table, so the shared
                                      row values cannot collide across
                                      hosts; they stay inside lane 9's
                                      reserved session block so a future
                                      verbs-qualified binding path is
                                      already lane-scoped.
  transport control     64144          TOPOLOGY ONLY (deployment schema
                        (base)         requires it; the host-rdma backend
                                      connects through the weightd socket
                                      and opens no TCP listener).

Budgets: pack bytes (experts + full-resolution spine) ride the operator's
shared weightd arena - the lane-job device budget covers the resident side
only (KV, workspace, activations). Sizing record lands with M2 as
model-families/ling/smoke_experts.json +
tools/devcycle/lane_budget_calc.py (both-view numbers).

Fail-closed: host names validated, rank bounds checked, codec in the
placed-arm set, KV backing must be finite, and --check mode must reproduce
committed output byte for byte.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

LANE = 9
WORLD = 16
TP = 16
HEX = "0123456789abcdef"
HOSTS = [f"spark{HEX[i]}" for i in range(WORLD)]

CONTROL_BASE = 23144                            # 23144 .. 23159
COLLECTIVE_BASE = 53144                         # 53144 .. 53159 (u16-valid, #1094)
TRANSPORT_BASE = 64144                          # 64144 .. 64159
SESSION_BASE = 23168 + 64 * LANE                # 23744 .. 23807 (64-number block)
SESSION_PORT_BASE = SESSION_BASE                # session_ports rows:  +0..15
SESSION_HC_BASE = SESSION_BASE + 16             # session_ports_hc rows: +16..31

MESH_RANKS = ",".join(str(i) for i in range(WORLD))

# The compiled module's serving pin (Makefile LING_MODEL_REVISION; the
# adapter refuses a drifted build). Shared by both placed arms - the fp8
# pack is the quantized emission of the same source snapshot.
MODEL_REVISION = "e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3"

CODECS = ("bf16", "fp8")                        # placed fleet arms
CODEC_NODE_TARGET = {
    codec: f"cuda.sm121.ling.resident_decode_stage.bf16.expert_{codec}"
    for codec in CODECS
}
# ling rank packs are HEX-named (rankb, not rank11): the stagepack
# canonical rule names the file after the spark node letter.
PACK_TEMPLATE = ("/home/{host}/sparkdata/ling.{codec}.tp16/packs/"
                 "ling.{codec}.tp16.rank{rank_hex}.sp")

DEFAULT_KV_BACKING_BYTES = 8 * 1024 * 1024 * 1024

# Nonzero by law: the stage module treats collective_identifier == 0 as
# "tp collective disabled" and refuses the multi-rank shape. Lane-tagged
# so two lanes can never join the same collective group.
COLLECTIVE_IDENTIFIER = 0x0009_0000_0000_0000


def host_of(rank: int) -> str:
    return HOSTS[rank]


def deployed_pack(rank: int, codec: str) -> str:
    return PACK_TEMPLATE.format(host=host_of(rank), codec=codec,
                                rank_hex=f"{rank:x}")


def session_row(base: int) -> list[list[int]]:
    """One 16x16 session matrix row set shared by every host.

    Row r belongs to host r only (TP16 identity: one rank per host), so
    identical row values bind at most once per host; diagonal is 0 and
    every off-diagonal value is nonzero inside lane 9's session block,
    exactly what SparkTpCollectiveLoadSessionPorts enforces.
    """
    return [[0 if column == row else base + column for column in range(TP)]
            for row in range(TP)]


def stage_config(rank: int, codec: str) -> dict:
    """The adapter's EXACT member set - no additions (load error)."""
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": codec,
        "stage_pack_path": f"packs/ling.{codec}.tp16.rank{rank:x}.sp",
        "max_sequence_positions": 32768,
        "execution_row_capacity": 128,
        "decode_split_context_threshold": 2048,
        "tp_degree": TP,
        "tp_rank": rank,
        "tp_collective": {
            "backend": "hidden_transport",
            "backend_module_path": "lib/hidden_transport.so",
            "algorithms": ["tree"],
            "collective_identifier": COLLECTIVE_IDENTIFIER,
            "listen_port": COLLECTIVE_BASE + rank,
            "connect_timeout_milli": 30000,
            "operation_timeout_milli": 30000,
            "peer_hosts": list(HOSTS),
            "peer_ports": [COLLECTIVE_BASE + peer for peer in range(TP)],
            "split_ring_min_payload_bytes": 0,
            "direct_all_to_all_max_payload_bytes": 0,
            "rail_peer_hosts": [list(HOSTS), list(HOSTS)],
            "step_rail_indices": [0] + [1] * (TP - 1),
            "session_ports": session_row(SESSION_PORT_BASE),
            "session_ports_hc": session_row(SESSION_HC_BASE),
        },
    }


def resident_deployment(runtime_root: str, weightd_socket: str, codec: str,
                        kv_backing_bytes: int = DEFAULT_KV_BACKING_BYTES) -> dict:
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": runtime_root,
            "node_target": CODEC_NODE_TARGET[codec],
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
        "eos_token_ids": [156895],
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
        "weightd": {"socket_path": weightd_socket},
        "runtime_limits": {
            "max_inflight_submissions": 4,
            "max_active_sequences": 16,
            "max_input_rows": 1024,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": 8192,
            "kv_physical_page_capacity": 8192,
        },
        "nodes": nodes,
    }


def render(rank: int, runtime_root: str, weightd_socket: str, codec: str,
           kv_backing_bytes: int) -> dict[str, str]:
    return {
        "deployment.json": json.dumps(resident_deployment(
            runtime_root, weightd_socket, codec, kv_backing_bytes), indent=1) + "\n",
        "adapter.json": json.dumps(stage_config(rank, codec), indent=1) + "\n",
    }


def emit_wset(source: str, output: str) -> int:
    """Materialize the smoke-expert working set as a .wset binary.

    Raw little-endian (layer u32, expert u32) pairs, deduplicated and
    sorted - the format tools/weightd_warm.c --wset validates against the
    pack manifest. Source: model-families/ling/smoke_experts.json
    (machine-generated; the M2 census receipt).
    """
    document = json.load(open(source, encoding="utf-8"))
    if document.get("family") != "ling":
        raise SystemExit("wset source is not the ling manifest")
    pairs = sorted({(int(e["layer"]), int(e["expert"]))
                    for e in document["experts"]})
    with open(output, "wb") as handle:
        for layer, expert in pairs:
            handle.write(layer.to_bytes(4, "little"))
            handle.write(expert.to_bytes(4, "little"))
    print(json.dumps({"wset": output, "keys": len(pairs)}))
    return 0


def budgets(source: str) -> int:
    """Emit 'expert_pool_bytes spine_bytes' (per-node, tp-sharded)."""
    document = json.load(open(source, encoding="utf-8"))
    nodes = int(document["nodes"])
    if document.get("expert_shard") != "tp" or nodes < 1:
        raise SystemExit("budgets need a tp-sharded manifest with nodes >= 1")
    pool = -(-sum(int(e["bytes"]) for e in document["experts"]) // nodes)
    spine = -(-int(document["spine_bytes"]) // nodes)
    print(f"{pool} {spine}")
    return 0


MANIFEST_MAGIC = 0x58504557
MANIFEST_VERSION = 2
MANIFEST_RECORD_BYTES = 48
SPINE_BUDGET_MARGIN_BYTES = 4 * 1024 * 1024


def manifest_spine_allocation(pack_base: str) -> int:
    """Replicate the weightd manifest builder's spine allocation exactly.

    runtime/spark_weightd_manifest.c build_spine: the spine spans are the
    byte gaps between the routed-expert ranges of <pack_base>.experts
    (header magic/version/count, then 48-byte offset-sorted records),
    compacted with 256-byte alignment derived from the file offsets. The
    lazy attach checks (allocation + 255) against the spine budget, so
    the budget must come from THIS value, never from a MiB-rounded
    estimate.
    """
    experts = pack_base + ".experts"
    with open(experts, "rb") as handle:
        head = handle.read(16)
        if len(head) != 16:
            raise SystemExit(f"manifest header short: {experts}")
        magic, version, count, _reserved = struct.unpack("<IIII", head)
        if magic != MANIFEST_MAGIC or version != MANIFEST_VERSION or count == 0:
            raise SystemExit(f"manifest is not v2-with-records: {experts}")
        ranges = []
        for _ in range(count):
            record = handle.read(MANIFEST_RECORD_BYTES)
            if len(record) != MANIFEST_RECORD_BYTES:
                raise SystemExit(f"manifest record short: {experts}")
            offset, size = struct.unpack("<QQ", record[16:32])
            if size == 0:
                raise SystemExit(f"manifest zero-byte range: {experts}")
            ranges.append((offset, offset + size))
    pack_bytes = os.path.getsize(pack_base)
    for start, end in ranges:
        if start >= pack_bytes or end > pack_bytes:
            raise SystemExit(f"manifest range outside the pack: {experts}")
    ranges.sort()
    cursor = 0
    allocation = 0
    for index in range(count + 1):
        end = ranges[index][0] if index < count else pack_bytes
        if end > cursor:
            padding = (cursor - allocation) & 255
            allocation += padding + (end - cursor)
        if index < count:
            cursor = ranges[index][1]
    return allocation


def spine_budget(pack_base: str) -> int:
    return manifest_spine_allocation(pack_base) + SPINE_BUDGET_MARGIN_BYTES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime-root",
                        help="private per-attempt runtime root")
    parser.add_argument("--weightd-socket")
    parser.add_argument("--output-dir")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--codec", choices=CODECS, default="bf16",
                        help="placed fleet arm (default bf16)")
    parser.add_argument("--kv-backing-bytes", type=int,
                        default=DEFAULT_KV_BACKING_BYTES)
    parser.add_argument("--emit-wset", metavar="OUTPUT",
                        help="write the smoke-expert .wset and exit")
    parser.add_argument("--budgets", metavar="MANIFEST",
                        help="print 'expert_pool_bytes spine_bytes' "
                             "estimated from the smoke manifest (advisory; "
                             "the spine budget must come from "
                             "--spine-budget)")
    parser.add_argument("--spine-budget", metavar="PACK_BASE",
                        help="print the spine budget derived from the "
                             "placed pack's .experts manifest (the exact "
                             "allocation the lazy attach checks, plus a "
                             "4 MiB margin) and exit")
    parser.add_argument("--wset-source", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model-families", "ling", "smoke_experts.json"))
    parser.add_argument("--check", action="store_true",
                        help="regenerate and compare against output-dir "
                             "instead of writing")
    arguments = parser.parse_args()

    if arguments.budgets:
        return budgets(arguments.budgets)
    if arguments.spine_budget:
        print(spine_budget(arguments.spine_budget))
        return 0
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
                   arguments.weightd_socket, arguments.codec,
                   arguments.kv_backing_bytes)
    output_dir = os.path.abspath(arguments.output_dir)
    if arguments.check:
        for name, text in files.items():
            current = open(os.path.join(output_dir, name), encoding="utf-8").read()
            if current != text:
                print(f"drift in {name}: regenerate with "
                      "tools/ling_multidev_lane.py", file=sys.stderr)
                return 1
        print("multidev lane output matches")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    print(json.dumps({"rank": arguments.rank, "codec": arguments.codec,
                      "output": output_dir,
                      "control_port": CONTROL_BASE + arguments.rank,
                      "collective_port": COLLECTIVE_BASE + arguments.rank,
                      "transport_base": TRANSPORT_BASE,
                      "pack": deployed_pack(arguments.rank, arguments.codec)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Lane-8 laguna shared-socket deployment generator (TP8xPP2, multidev).

Builds the PRIVATE per-attempt model_resident deployment and the per-rank
serving-adapter configuration for developer lane 8 under the shared-socket
protocol of docs/MULTIDEV_QUICKSTART.md. The wrapper that drives this file
is tools/laguna_multidev_run_family.sh; it stages the output under
$SPARK_QUEUE_RUNTIME_ROOT so every runtime root, KV directory and listener
of this lane stays private to one queue attempt.

Topology: 16 world ranks over spark0..sparkf, two pipeline stages of eight
TP ranks. Logical rank = index in --nodes; the mesh map is the identity
permutation (SPARK_TP_MESH_RANKS=0,...,15), so rank r lives on
spark{hex(r)}, PP stage r//8, TP rank r%8. The packs are the
OPERATOR-PLACED bf16 set (fleet inventory, Sep 17 - the prep2 placement
with ACC-2 batch receipts):

  /home/{host}/sparkdata/laguna-s-2.1.bf16.tp8pp2/packs/
      laguna_stage.tp8.pp2.stage{r//8}.rank{r}.lgsp

The filename rank is the GLOBAL rank (spark8 holds stage1.rank8, NOT
stage1.rank0 - verified on every node during the lane-8 M1 inventory;
the pack header's stage_index must equal rank//8 and the module checks
header vs deployment stage_index at load).

The configuration member set mirrors the laguna serving adapter's EXACT
contract (modules/laguna_resident_decode_stage/source/
spark_laguna_serving_adapter.c SparkLagunaServingConfigurationMembers +
SparkTpCollectiveValidateMembers adaptive_members - a missing OR EXTRA
member is a load error; the schema-3 deployment tree under
deployment/laguna-s-2.1.bf16.tp8pp2/ predates that contract and is NOT a
template for the private config). The deployment node stage_index carries
the GLOBAL rank: the adapter derives tp_rank = stage_index % 8 and the
module's pipeline stage = stage_index // 8 from it.

Port ledger (lane 8 owns control 23128-23143, collective 53128-53143
(u16-valid, #1094), transport 64128-64143, session block 23680-23743
(23168+64L per the template porting checklist); every number below stays
inside those blocks):

  control_endpoint      23128 + rank   BOUND (residentd client listener;
                                      one per host, so the block is full).
  tp_collective listen  53128 + tp     BOUND host TCP collective, GROUP-
                                      LOCAL: the eight ranks of a PP stage
                                      listen 53128..53135 and BOTH stages
                                      reuse those numbers because their
                                      ranks sit on disjoint hosts (the k3
                                      lane-3 group-local precedent).
  session matrix        23680..23735   TOPOLOGY ONLY under the shared
  (+ hc mirror)                        socket: SparkTpDeviceCollective
                                      transports through the weightd mesh
                                      (SPARK_WEIGHTD_SOCKET + lane), so
                                      these bind nothing today. The 8x8
                                      table packs 56 numbers into the
                                      lane-8 session block; the required
                                      session_ports_hc mirror reuses the
                                      same 56 numbers (both matrices are
                                      topology-only and the loader pins
                                      only nonzero/u16/diagonal-zero). If
                                      a verbs-qualified binding path ever
                                      lands, the per-host table must be
                                      re-planned and the session
                                      reservation widened (the k3 M1 port
                                      finding, verbatim).
  transport control     64128          TOPOLOGY ONLY (deployment schema
                        (base)         requires it; the host-rdma backend
                                      connects through the weightd socket
                                      and opens no TCP listener).

Budgets: pack bytes (experts + full-resolution spine) ride the operator's
shared weightd arena - the lane-job device budget covers the resident side
only (KV, workspace, activations). --budgets derives the arena-side
expert-pool / spine numbers from the pack's own .experts sidecar (expert
span sum and the complement), so the wrapper never hardcodes a byte count;
the M2 census manifest (model-families/laguna/smoke_experts.json) becomes
the sizing record on top of the same bases.

Fail-closed: host names validated, rank bounds checked, KV backing must be
finite, the collective identifier must be positive, and --check mode must
reproduce committed output byte for byte.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

LANE = 8
WORLD = 16
TP = 8
PP = 2
# Family geometry (model-families/laguna/include/sparkpipe/llm_defines.h):
# 48 layers over 2 pipeline stages. A rank's stage owns the contiguous
# 24-layer span; layer->stage is layer // 24 (NOT the rank->stage rank//8
# arithmetic - the two only coincide for ranks).
LAYER_COUNT = 48
LAYERS_PER_STAGE = LAYER_COUNT // PP
HEX = "0123456789abcdef"
HOSTS = [f"spark{HEX[i]}" for i in range(WORLD)]

CONTROL_BASE = 23128                            # 23128 .. 23143
COLLECTIVE_BASE = 53128                         # 53128 .. 53143 (u16-valid, #1094)
TRANSPORT_BASE = 64128                          # 64128 .. 64143
SESSION_BASE = 23168 + 64 * LANE                # 23680 .. 23743

MESH_RANKS = ",".join(str(i) for i in range(WORLD))

# The placed packs' identity (ACC-2 batch receipts beside every pack) and
# the module's serving pin; the adapter refuses a drifted build and the
# module refuses a pack whose header revision disagrees.
MODEL_REVISION = "0f573140834b11cfac0c2af97a101a7a69a13e22"
NODE_TARGET = "cuda.sm121.laguna.resident_decode_stage.bf16.expert_bf16"
EXPERT_CODEC = "bf16"

DEPLOYED_PACK_TEMPLATE = (
    "/home/{host}/sparkdata/laguna-s-2.1.bf16.tp8pp2/packs/"
    "laguna_stage.tp8.pp2.stage{stage}.rank{rank}.lgsp")
DEFAULT_KV_BACKING_BYTES = 8 * 1024 * 1024 * 1024

# The .experts sidecar contract shared with the lazy attach
# (runtime/spark_weightd_manifest.c): WEXP v2 header + 48-byte records.
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2


def host_of(rank: int) -> str:
    return HOSTS[rank]


def stage_of(rank: int) -> int:
    return rank // TP


def tp_rank_of(rank: int) -> int:
    return rank % TP


def group_hosts(rank: int) -> list[str]:
    first = stage_of(rank) * TP
    return HOSTS[first:first + TP]


def deployed_pack(rank: int) -> str:
    return DEPLOYED_PACK_TEMPLATE.format(
        host=host_of(rank), stage=stage_of(rank), rank=rank)


def session_table() -> list[list[int]]:
    """56 numbers packed into 23680..23735 with no per-host overlap.

    Row a is listened by the host whose TP rank is a (both PP stages reuse
    the table because they own disjoint hosts). The packing
    SESSION_BASE + 7a + (b if b < a else b - 1) keeps all 56 values inside
    the 64-number lane-8 session block with 8 numbers spare.
    """
    table = []
    for a in range(TP):
        row = []
        for b in range(TP):
            if a == b:
                row.append(0)
            else:
                row.append(SESSION_BASE + 7 * a + (b if b < a else b - 1))
        table.append(row)
    return table


def tp_collective(rank: int, collective_identifier: int) -> dict:
    """The hidden_transport tp_collective - the loader's EXACT member set."""
    peers = group_hosts(rank)
    return {
        "backend": "hidden_transport",
        "backend_module_path": "lib/hidden_transport.so",
        "algorithms": ["tree"],
        "collective_identifier": collective_identifier,
        "listen_port": COLLECTIVE_BASE + tp_rank_of(rank),
        "connect_timeout_milli": 300000,
        "operation_timeout_milli": 30000,
        "peer_hosts": peers,
        "peer_ports": [COLLECTIVE_BASE + peer for peer in range(TP)],
        "split_ring_min_payload_bytes": 0,
        "direct_all_to_all_max_payload_bytes": 0,
        "rail_peer_hosts": [peers, peers],
        "step_rail_indices": [0] + [1] * (TP - 1),
        "session_ports": session_table(),
        "session_ports_hc": session_table(),
    }


def adapter_config(rank: int, collective_identifier: int) -> dict:
    """The adapter's EXACT member set - no additions (load error)."""
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": EXPERT_CODEC,
        "stage_pack_path": "packs/" + os.path.basename(deployed_pack(rank)),
        "max_sequence_positions": 32768,
        "execution_row_capacity": 128,
        "decode_split_context_threshold": 2048,
        "tp_degree": TP,
        "tp_rank": tp_rank_of(rank),
        "tp_collective": tp_collective(rank, collective_identifier),
    }


def resident_deployment(runtime_root: str, weightd_socket: str,
                        collective_identifier: int,
                        kv_backing_bytes: int = DEFAULT_KV_BACKING_BYTES) -> dict:
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
        "eos_token_ids": [2, 24],
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
            "max_input_rows": 128,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": 8192,
            "kv_physical_page_capacity": 8192,
        },
        "nodes": nodes,
    }


def budgets(pack_path: str) -> int:
    """Emit 'expert_pool_bytes spine_bytes' for one rank pack.

    The pool base is the daemon's own acquire accounting: the WHOLE pack
    file charged in 2 MiB chunks (ceil(pack_bytes / 2 MiB) x 2 MiB) -
    the shared weightd refuses an acquire whose declared pool is under
    the pack's full chunk footprint (ACQUIRE-LOAD-STAGE stage=budget,
    chunk_count x 2 MiB = retained; measured on lane 8, the lane-4
    rank-3 cell found the same law). The spine base is the complement of
    the expert spans inside the pack (the build_spine arithmetic),
    derived from the .experts sidecar. Zero assumed numbers - a missing
    sidecar fails closed (generate it first with
    tools/laguna_multidev_experts_manifest.sh).
    """
    pack = os.path.abspath(pack_path)
    sidecar = pack + ".experts"
    if not os.path.isfile(pack):
        raise SystemExit(f"budgets: pack not found: {pack}")
    if not os.path.isfile(sidecar):
        raise SystemExit(f"budgets: .experts sidecar missing: {sidecar} "
                         "(generate it with "
                         "tools/laguna_multidev_experts_manifest.sh)")
    with open(sidecar, "rb") as handle:
        head = handle.read(16)
        if len(head) != 16:
            raise SystemExit("budgets: short sidecar header")
        magic, version, count, reserved = struct.unpack("<IIII", head)
        if magic != EXPERTS_MAGIC or version != EXPERTS_VERSION \
                or reserved != 0 or count == 0:
            raise SystemExit("budgets: not a v2 routed-expert sidecar")
        spans = []
        for _ in range(count):
            record = handle.read(48)
            if len(record) != 48:
                raise SystemExit("budgets: short sidecar record")
            _, _, _, _, offset, span_bytes = struct.unpack_from("<4I2Q", record)
            spans.append((offset, span_bytes))
        trailing = handle.read(1)
        if trailing:
            raise SystemExit("budgets: trailing bytes after last record")
    spans.sort()
    spine_bytes = 0
    cursor = 0
    pack_bytes = os.path.getsize(pack)
    for offset, span_bytes in spans:
        if offset < cursor or offset + span_bytes > pack_bytes:
            raise SystemExit("budgets: sidecar spans disagree with the pack")
        spine_bytes += offset - cursor
        cursor = offset + span_bytes
    spine_bytes += pack_bytes - cursor
    chunk = 2 * 1024 * 1024
    pool_bytes = -(-pack_bytes // chunk) * chunk
    # The spine budget must cover the loader's ALIGNED allocation:
    # lazy_spine_load cudaMallocs bytes + 255 and refuses a budget under
    # it (attach-007: spine 546,565,120 + 255 > budget 546,565,120 ->
    # CAPACITY_EXCEEDED at spark_weightd_lazy_pack.c:62).
    print(f"{pool_bytes} {spine_bytes + 256}")
    return 0


def smoke_budgets(source: str, rank: int) -> int:
    """Emit 'raw_bytes chunked_bytes' for one rank's smoke-expert set.

    The M3 warm-receipt sizing on both bases (the 2026-09-22 chunk-basis
    correction): raw = the exact per-rank span sum of this rank's STAGE's
    head pairs (a node holds every expert of its own stage's layers at
    the per-rank span); chunked = each span rounded up to the 2 MiB lazy
    pool chunk (runtime/spark_weightd.c: pool chunk = max(gpu allocation
    granularity, 2 MiB); measured 2 MiB on sm_121a). Laguna's small
    per-expert spans (w1 1.5 MiB, w2 0.75 MiB) make the chunk factor
    material - receipts carry both bases, never a factor.
    """
    document = json.load(open(source, encoding="utf-8"))
    if document.get("family") != "laguna":
        raise SystemExit("budgets source is not the laguna census manifest")
    bases = document["provenance"]["byte_bases"]
    spans = bases["per_rank_expert_span_bytes"]
    w1, w2 = int(spans["w1"]), int(spans["w2"])
    chunk = 2 * 1024 * 1024
    stage = stage_of(rank)
    raw = 0
    chunked = 0
    pairs = 0
    for entry in document["experts"]:
        if int(entry["layer"]) // LAYERS_PER_STAGE != stage:
            continue
        pairs += 1
        raw += w1 + w2
        chunked += -(-w1 // chunk) * chunk + -(-w2 // chunk) * chunk
    if pairs == 0:
        raise SystemExit(f"rank {rank}: the census head has no pairs for "
                         f"stage {stage}")
    print(f"{raw} {chunked}")
    return 0


def emit_wset(source: str, output: str, rank: int | None = None) -> int:
    """Materialize the smoke-expert working set as a .wset binary.

    Raw little-endian (layer u32, expert u32) pairs, deduplicated and
    sorted - the format tools/weightd_warm.c --wset validates against the
    pack manifest. Source: model-families/laguna/smoke_experts.json
    (machine-generated; the M2 census receipt).

    With --rank: filtered to that rank's PP-stage layers. weightd_warm
    (a) rejects any pair absent from the warmed pack's own manifest and
    (b) caps one --wset file at SPARK_WEIGHTD_LEASE_GROUPS_MAX (512)
    pairs - the warm side splits the filtered file into <=4096-byte
    (512-pair) chunks and warms them sequentially (the GLM pin-experts
    chunked-lease precedent).
    """
    document = json.load(open(source, encoding="utf-8"))
    if document.get("family") != "laguna":
        raise SystemExit("wset source is not the laguna census manifest")
    pairs = sorted({(int(e["layer"]), int(e["expert"]))
                    for e in document["experts"]})
    filtered = pairs
    if rank is not None:
        stage = stage_of(rank)
        filtered = [pair for pair in pairs
                    if pair[0] // LAYERS_PER_STAGE == stage]
        if not filtered:
            raise SystemExit(f"rank {rank}: the census head has no pairs "
                             f"for stage {stage}")
    with open(output, "wb") as handle:
        for layer, expert in filtered:
            handle.write(layer.to_bytes(4, "little"))
            handle.write(expert.to_bytes(4, "little"))
    print(json.dumps({"wset": output, "keys": len(filtered),
                      "rank_filter": rank}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime-root",
                        help="private per-attempt runtime root")
    parser.add_argument("--weightd-socket")
    parser.add_argument("--output-dir")
    parser.add_argument("--rank", type=int,
                        help="emit only this rank's adapter.json "
                             "(default: all sixteen)")
    parser.add_argument("--collective-identifier", type=int,
                        help="positive wire identifier for the tp "
                             "collective (the wrapper derives it from the "
                             "queue attempt id)")
    parser.add_argument("--kv-backing-bytes", type=int,
                        default=DEFAULT_KV_BACKING_BYTES)
    parser.add_argument("--budgets", metavar="PACK",
                        help="print 'expert_pool_bytes spine_bytes' "
                             "derived from the pack's .experts sidecar "
                             "and exit")
    parser.add_argument("--smoke-budgets", nargs=2, metavar=("MANIFEST", "RANK"),
                        help="print 'raw_bytes chunked_bytes' for this "
                             "rank's smoke-expert head pairs and exit")
    parser.add_argument("--emit-wset", metavar="OUTPUT",
                        help="write the smoke-expert .wset and exit")
    parser.add_argument("--wset-source", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model-families", "laguna", "smoke_experts.json"))
    parser.add_argument("--check", action="store_true",
                        help="regenerate and compare against output-dir "
                             "instead of writing")
    arguments = parser.parse_args()

    if arguments.budgets:
        return budgets(arguments.budgets)
    if arguments.smoke_budgets:
        source, rank_text = arguments.smoke_budgets
        if not rank_text.isdigit() or not 0 <= int(rank_text) < WORLD:
            raise SystemExit(f"rank must be 0..{WORLD - 1}")
        return smoke_budgets(source, int(rank_text))
    if arguments.emit_wset:
        rank = arguments.rank if arguments.rank is not None else None
        return emit_wset(arguments.wset_source, arguments.emit_wset, rank)

    missing = [name for name, value in (
        ("--runtime-root", arguments.runtime_root),
        ("--weightd-socket", arguments.weightd_socket),
        ("--output-dir", arguments.output_dir),
        ("--collective-identifier", arguments.collective_identifier)) if value is None]
    if missing:
        raise SystemExit("the following arguments are required: "
                         + ", ".join(missing))
    if arguments.rank is not None and not 0 <= arguments.rank < WORLD:
        raise SystemExit(f"rank must be 0..{WORLD - 1}")
    if not re.fullmatch(r"/tmp/sparkqueue-[0-9a-f]{32}", arguments.runtime_root):
        raise SystemExit("runtime root must be the private queue namespace")
    if arguments.kv_backing_bytes <= 0:
        raise SystemExit("kv backing must be a finite positive cap")
    if not arguments.weightd_socket.startswith("/run/sparkpipe-weightd-shared/"):
        raise SystemExit("weightd socket must be the operator's shared unit")
    if arguments.collective_identifier <= 0:
        raise SystemExit("collective identifier must be positive")

    ranks = range(WORLD) if arguments.rank is None else [arguments.rank]
    files = {
        "deployment.json": json.dumps(resident_deployment(
            arguments.runtime_root, arguments.weightd_socket,
            arguments.collective_identifier,
            arguments.kv_backing_bytes), indent=2) + "\n"}
    for rank in ranks:
        name = f"adapter.spark{HEX[rank]}.json" if arguments.rank is None \
            else "adapter.json"
        files[name] = json.dumps(adapter_config(
            rank, arguments.collective_identifier), indent=2) + "\n"
    output_dir = os.path.abspath(arguments.output_dir)
    if arguments.check:
        for name, text in files.items():
            current = open(os.path.join(output_dir, name), encoding="utf-8").read()
            if current != text:
                print(f"drift in {name}: regenerate with "
                      "tools/laguna_multidev_lane.py", file=sys.stderr)
                return 1
        print("multidev lane output matches")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    print(json.dumps({"rank": arguments.rank, "output": output_dir,
                      "control_port": None if arguments.rank is None
                      else CONTROL_BASE + arguments.rank,
                      "collective_id": arguments.collective_identifier,
                      "transport_base": TRANSPORT_BASE,
                      "session_base": SESSION_BASE,
                      "pack": None if arguments.rank is None
                      else deployed_pack(arguments.rank)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

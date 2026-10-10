#!/usr/bin/env python3
"""Lane-3 k3 shared-socket deployment generator (TP4xPP4, multidev).

Builds the PRIVATE per-attempt model_resident deployment and the per-rank
serving adapter configuration for developer lane 3 under the shared-socket
protocol of docs/MULTIDEV_QUICKSTART.md. The wrapper that drives this file
is tools/k3_multidev_run_family.sh; it stages the output under
$SPARK_QUEUE_RUNTIME_ROOT so every listener, runtime root, KV directory and
log of this lane stays private to one queue attempt.

Topology: 16 ranks over spark0..sparkf, four pipeline stages of four TP
ranks. Logical rank = index in --nodes; the mesh map is the identity
permutation 0,1,...,15 (SPARK_TP_MESH_RANKS), so rank i lives on
spark{hex(i)}, PP stage i//4, TP rank i%4, and stage s rank t pack
k3.stage{s}.rank0{t}.pack sits in that host's deployed pack directory.

Port ledger (lane 3 owns control 23048-23063, collective 53048-53063,
transport 64048-64063; every number below stays inside those blocks):

  control_endpoint      23048 + rank   BOUND (residentd client listener);
                                      one per host, so the block is full.
  53048..53051          UNUSED (the host TCP collective is gone).
  device session table  packed into    TOPOLOGY ONLY under the shared
                        53052..53063   socket: SparkTpDeviceCollectiveCreate
                                      transports through the weightd mesh
                                      (SPARK_WEIGHTD_SOCKET + lane), and
                                      libhidden_transport.so likewise
                                      broadcasts through the mesh, so
                                      these numbers bind nothing today.
                                      They stay inside lane 3's blocks so
                                      a future verbs-qualified path can
                                      not collide with another lane; if
                                      such a path ever binds them the
                                      per-host table must be re-planned.
  transport control     64062          TOPOLOGY ONLY (deployment schema
                        (wide base)    requires it; the band-1 "wide" device
                        ..64063 (base) collective for the fused gate_up
                                      reduce derives 64062, the hidden
                                      collective carries 64063; the
                                      host-rdma backend connects through the
                                      weightd socket and opens no TCP
                                      listener on either).

The k3 stage runner is fail-closed lazy: it requires the weightd socket,
SPARK_WEIGHTD_PACK_SHA256 (set by model_residentd from the single
packs/*.sha256 sidecar), SPARK_WEIGHTD_EXPERT_POOL_BYTES and
SPARK_WEIGHTD_SPINE_BUDGET_BYTES. The wrapper refuses to run without the
two budget envs; tools/devcycle/lane_budget_calc.py is the sizing record.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

LANE = 3
WORLD = 16
TP = 4
PP = 4
HEX = "0123456789abcdef"
HOSTS = [f"spark{HEX[i]}" for i in range(WORLD)]


CONTROL_BASE = 23048                            # 23048 .. 23063
COLLECTIVE_BASE = 53048                         # 53048 .. 53063 (u16-valid, #1094)
TRANSPORT_BASE = 64048                          # 64048 .. 64063

K3_SESSION_BLOCK_BASE = 18432             # PORT_LEDGER kimi k3 block, TP16 cells
DEVICE_SESSION_BASE = COLLECTIVE_BASE + 4 # packed 12-number table below


def select_topology(topology: str) -> None:
    global TOPOLOGY, TP, PP, DEPLOYED_PACK_TEMPLATE
    TOPOLOGY = topology
    if topology == "tp16":
        TP, PP = 16, 1
        DEPLOYED_PACK_TEMPLATE = (
            "/home/{host}/sparkdata/k3.mxfp4.tp16/packs/k3.stage0.rank{rank:02d}.pack")
    elif topology == "tp4pp4":
        TP, PP = 4, 4
        DEPLOYED_PACK_TEMPLATE = (
            "/home/{host}/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage{stage}.rank0{tp}.pack")
    else:
        raise SystemExit(f"topology {topology} is not tp4pp4 or tp16")


def select_lane(lane: int) -> None:
    global LANE, CONTROL_BASE, COLLECTIVE_BASE, TRANSPORT_BASE
    global DEVICE_SESSION_BASE
    if lane < 1 or lane > 15:
        raise SystemExit(f"lane {lane} outside 1..15 (lane 0 is production)")
    LANE = lane
    CONTROL_BASE = 23000 + 16 * lane
    COLLECTIVE_BASE = 53000 + 16 * lane
    TRANSPORT_BASE = 64000 + 16 * lane
    DEVICE_SESSION_BASE = COLLECTIVE_BASE + 4
MESH_RANKS = ",".join(str(i) for i in range(WORLD))

TOPOLOGY = "tp4pp4"
DEPLOYED_PACK_TEMPLATE = (
    "/home/{host}/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage{stage}.rank0{tp}.pack")

NODE_TARGET = "cuda.sm121.k3.resident_decode_stage.linear_bf16.expert_mxfp4.kv_bf16"
DEFAULT_KV_BACKING_BYTES = 8 * 1024 * 1024 * 1024
KV_IN_FLIGHT_MARGIN_BYTES = 2 * 2 * 1024 * 1024
K3_DEFINES = (Path(__file__).resolve().parents[1] / "model-families/k3/include/sparkpipe/spark_k3_llm_defines.h").read_text()


def k3_constant(name: str) -> int:
    return int(re.search(r"#define SPARK_K3_" + name + r" (\d+)u", K3_DEFINES).group(1))


def recurrent_page_bytes(topology: str) -> int:
    layers = k3_constant("MODEL_LAYER_COUNT")
    period = k3_constant("MODEL_ATTENTION_PERIOD")
    phase = k3_constant("MODEL_GLOBAL_ATTENTION_PHASE")
    kernel = k3_constant("MODEL_KDA_CONV_KERNEL")
    scalar = k3_constant("KV_BITS") // 8
    if topology == "tp16":
        tp, stages = 16, [(0, layers)]
    else:
        count = k3_constant("PP_STAGE_COUNT")
        base, remainder = divmod(layers, count)
        tp = 16 // count
        stages = [(stage * base + min(stage, remainder), base + (1 if stage < remainder else 0)) for stage in range(count)]
    heads = k3_constant("MODEL_KDA_HEAD_COUNT") // tp
    key = k3_constant("MODEL_KDA_HEAD_KEY_DIMENSION")
    value = k3_constant("MODEL_KDA_HEAD_VALUE_DIMENSION")
    per_layer = heads * key * value * k3_constant("MODEL_KDA_STATE_ELEMENT_BYTES") + heads * (2 * key + value) * kernel * scalar
    kda = [sum(1 for layer in range(first, first + count) if not (layer % period == phase or layer == layers - 1)) for first, count in stages]
    return max(kda) * per_layer


def kv_page_bytes(topology: str) -> int:
    layers = k3_constant("MODEL_LAYER_COUNT")
    period = k3_constant("MODEL_ATTENTION_PERIOD")
    phase = k3_constant("MODEL_GLOBAL_ATTENTION_PHASE")
    slot = (k3_constant("MODEL_MLA_LATENT_DIMENSION") + k3_constant("MODEL_MLA_UNROTATED_DIMENSION")) * (k3_constant("KV_BITS") // 8)
    if topology == "tp16":
        tp, stages = 16, [(0, layers)]
    else:
        count = k3_constant("PP_STAGE_COUNT")
        base, remainder = divmod(layers, count)
        tp = 16 // count
        stages = [(stage * base + min(stage, remainder), base + (1 if stage < remainder else 0)) for stage in range(count)]
    mla = [sum(1 for layer in range(first, first + count) if layer % period == phase or layer == layers - 1) for first, count in stages]
    return max(mla) * k3_constant("KV_PAGE_SLOTS") * slot // tp


def kv_backing_minimum(topology: str, sequences: int, physical_pages: int, logical_pages: int) -> int:
    spill = logical_pages - physical_pages + 2
    checkpoints = 2 * sequences + 2
    return spill * kv_page_bytes(topology) + checkpoints * recurrent_page_bytes(topology) + KV_IN_FLIGHT_MARGIN_BYTES
DEFAULT_KV_SNAPSHOT_BYTES = 8 * 1024 * 1024 * 1024
KV_PAGES_PER_SEQUENCE = 64   # adapter_config default; x SPARK_K3_KV_PAGE_SLOTS (64) tokens
MAX_ROWS = 2048

# The batch engine refuses a deployment with no EOS tokens (SCHEMA_ERROR at
# SparkModelBatchValidateConfiguration — cold14: status=6, tokens=0, the
# request never reached the tensors; adapter_ops stayed 0). Single source of
# truth is the authoritative contract, the same record the compiled driver's
# K3_EOS_TOKEN is generated from (tools/generate_k3_contract.py). Never
# hardcode the id here; fail closed if the contract cannot provide it.
CONTRACT = json.loads((Path(__file__).resolve().parents[1] /
                       "model_contracts/k3_authoritative.json").read_text())
try:
    K3_EOS_TOKEN_IDS = sorted(int(value) for value in CONTRACT["eos_token_ids"].values())
except (KeyError, TypeError, ValueError, AttributeError):
    raise SystemExit("k3 contract missing eos_token_ids; refusing to emit "
                     "a deployment the batch engine would reject")
if not K3_EOS_TOKEN_IDS or K3_EOS_TOKEN_IDS[0] <= 0:
    raise SystemExit("k3 contract eos_token_ids must hold positive ids")


def host_of(rank: int) -> str:
    return HOSTS[rank]


def stage_of(rank: int) -> int:
    return rank // TP


def tp_rank_of(rank: int) -> int:
    return rank % TP


def deployed_pack(rank: int) -> str:
    return DEPLOYED_PACK_TEMPLATE.format(
        host=host_of(rank), stage=stage_of(rank), tp=tp_rank_of(rank),
        rank=rank)


def session_table() -> list[list[int]]:
    """12 numbers packed into 53052..53063 with no per-host overlap.

    Row a is listened by the host whose TP rank is a (every PP stage
    reuses the table because stages own disjoint hosts). The packing
    53052 + 3a + (b if b < a else b - 1) keeps all twelve values inside
    the twelve numbers above 53048..53051.
    """
    if TOPOLOGY == "tp16":
        return [[0 if a == b else K3_SESSION_BLOCK_BASE + a * TP + b
                 for b in range(TP)] for a in range(TP)]
    table = []
    for a in range(TP):
        row = []
        for b in range(TP):
            if a == b:
                row.append(0)
            else:
                row.append(DEVICE_SESSION_BASE + 3 * a + (b if b < a else b - 1))
        table.append(row)
    return table


def group_hosts(rank: int) -> list[str]:
    first = stage_of(rank) * TP
    return HOSTS[first:first + TP]


DEFAULT_STATE_BUDGET_BYTES = 5 << 30


def adapter_config(rank: int, kv_pages: int = KV_PAGES_PER_SEQUENCE,
                   sequences: int = 16, rows: int | None = None,
                   state_budget_bytes: int = DEFAULT_STATE_BUDGET_BYTES) -> dict:
    tp = tp_rank_of(rank)
    config = {
        "stage_pack_path": deployed_pack(rank),
        "tp_degree": TP,
        "tp_rank": tp,
        "world_size": WORLD,
        "max_sequences": sequences,
        "max_rows": rows if rows is not None else sequences,
        "resident_capacity": sequences,
        "kv_pages": kv_pages,
        "state_budget_bytes": state_budget_bytes,
        "hidden": 7168,
        "device_collective": {
            "backend": "hidden_transport",
            "backend_module_path": "lib/hidden_transport.so",
            "local_host": host_of(rank),
            "collective_identifier": (LANE << 48) | stage_of(rank),
            "listen_port": TRANSPORT_BASE + 15,
            "connect_timeout_milli": 300000,
            "operation_timeout_milli": 30000,
            "peer_hosts": group_hosts(rank),
            "session_ports": session_table(),
            "wait_mode": "hardware",
        },
    }
    return config


CHAT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "model-families/k3/chat_template.json"
TOKENIZER_ASSET = "tokenizer/tokenizer.compiled"
TOKENIZER_VOCABULARY = 163840


def chat_template() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_model_resident_deployment import chat_template_value
    return chat_template_value(json.loads(CHAT_TEMPLATE_PATH.read_text()))


def resident_deployment(runtime_root: str, weightd_socket: str,
                        kv_backing_bytes: int = DEFAULT_KV_BACKING_BYTES,
                        kv_snapshot_bytes: int = DEFAULT_KV_SNAPSHOT_BYTES,
                        sequences: int = 16,
                        kv_pages: int = KV_PAGES_PER_SEQUENCE,
                        pipeline_transport: str = "host-rdma",
                        tokenizer_sha256: str | None = None,
                        rows: int | None = None,
                        physical_pages: int | None = None,
                        logical_pages: int | None = None) -> dict:
    nodes = []
    for rank, host in enumerate(HOSTS):
        root = runtime_root.format(host=host)
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": root,
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/adapter.json",
            "kv_backing_directory": os.path.join(root, "kvcache"),
            "kv_partition": "/",
            "kv_backing_maximum_bytes": kv_backing_bytes,
            "kv_snapshot_directory": os.path.join(root, "kvsnapshot"),
            "kv_snapshot_maximum_bytes": kv_snapshot_bytes,
            "control_endpoint": {
                "kind": "tcp",
                "host": host,
                "port": CONTROL_BASE + rank,
            },
        })
    return {
        "schema_version": 2,
        # EOS stop tokens are mandatory for the batch engine (count >= 1);
        # sourced from the authoritative k3 contract above.
        "eos_token_ids": K3_EOS_TOKEN_IDS,
        "coordinator_rank_index": 0,
        "adapter": {
            "shared_object_path": "lib/libk3_serving_adapter.so",
        },
        "driver": {
            "shared_object_path": "lib/libk3_serving_adapter.so",
            "program_name": "k3",
        },
        "transport": {
            "shared_object_path": "lib/hidden_pipeline.so"
            if pipeline_transport == "host-staged" else "lib/hidden_transport.so",
            "mode": pipeline_transport,
            "control_port_base": TRANSPORT_BASE,
        },
        "weightd": {
            "socket_path": weightd_socket,
        },
        # KV page capacities: residentd fails closed (INVALID_ARGUMENT)
        # when kv_physical < max_active_sequences or kv_logical <
        # resident_sequence_capacity (model_serving_adapter.c runtime-
        # limits check) — zeros do NOT mean "bind nothing" here. The honest
        # bound is the resident capacity times the adapter's
        # kv_pages_per_sequence (64 pages x 64 tokens = the 4096-token
        # per-sequence ceiling the seam already commits to).
        "runtime_limits": {
            "max_inflight_submissions": sequences,
            "max_active_sequences": sequences,
            "max_input_rows": rows if rows is not None else sequences,
            "resident_sequence_capacity": sequences,
            "kv_logical_page_capacity": logical_pages if logical_pages is not None else sequences * kv_pages,
            "kv_physical_page_capacity": physical_pages if physical_pages is not None else sequences * kv_pages,
        },
        "chat_template": chat_template(),
        **({"tokenizer": {"path": TOKENIZER_ASSET, "sha256": tokenizer_sha256,
                          "vocabulary_size": TOKENIZER_VOCABULARY}}
           if tokenizer_sha256 else {}),
        "nodes": nodes,
    }


def render(value: dict) -> str:
    return json.dumps(value, indent=2) + "\n"


def write_or_check(path: Path, text: str, check: bool) -> bool:
    if check:
        if not path.is_file():
            raise SystemExit(f"missing generated file: {path}")
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"generated file is stale: {path}")
        return False
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the lane-3 k3 TP4xPP4 shared-socket deployment")
    parser.add_argument("--runtime-root", required=True,
                        help="private per-attempt runtime root "
                             "($SPARK_QUEUE_RUNTIME_ROOT)")
    parser.add_argument("--weightd-socket", required=True,
                        help="shared weightd socket path")
    parser.add_argument("--output-dir", required=True,
                        help="directory receiving deployment.json and "
                             "adapter.json (created when missing)")
    parser.add_argument("--kv-backing-bytes", type=int,
                        default=None,
                        help="finite KV backing cap for the private root; it "
                             "must hold the spilled KV pages and two KDA "
                             "checkpoints per sequence plus two "
                             "(default: exactly that plus the in-flight margin)")
    parser.add_argument("--kv-snapshot-bytes", type=int,
                        default=DEFAULT_KV_SNAPSHOT_BYTES,
                        help="finite KV snapshot store cap under the root; "
                             "the KV binding refuses a deployment without "
                             "one (default %(default)d)")
    parser.add_argument("--rank", type=int, choices=range(WORLD), default=None,
                        help="emit only this rank's adapter.json "
                             "(default: all sixteen)")
    parser.add_argument("--lane", type=int, default=LANE,
                        help="weightd mesh lane and port block "
                             "(default %(default)d)")
    parser.add_argument("--sequences", type=int, default=16,
                        help="concurrent sequences per rank; KDA state and "
                             "KV scale with it (default %(default)d)")
    parser.add_argument("--kv-pages", type=int, default=KV_PAGES_PER_SEQUENCE,
                        help="64-token KV pages per sequence "
                             "(default %(default)d)")
    parser.add_argument("--rows", type=int, default=None,
                        help="input rows per submission, the prefill wave "
                             "width (default: --sequences)")
    parser.add_argument("--kv-physical-pages", type=int, default=None,
                        help="resident KV pages per rank (default: "
                             "sequences x kv-pages)")
    parser.add_argument("--kv-logical-pages", type=int, default=None,
                        help="KV pages per rank including the NVMe tier "
                             "(default: the physical pages)")
    parser.add_argument("--pipeline-transport",
                        choices=("host-rdma", "host-staged"),
                        default="host-rdma",
                        help="stage-to-stage hidden hand-off: the weightd "
                             "host-rdma module or the host-staged TCP module "
                             "(lib/hidden_pipeline.so) (default %(default)s)")
    parser.add_argument("--topology", choices=("tp4pp4", "tp16"),
                        default="tp4pp4",
                        help="rank layout: 4 PP stages of TP4, or one TP16 "
                             "group over the TP16 rank packs "
                             "(default %(default)s)")
    parser.add_argument("--tokenizer-sha256", default=None,
                        help="sha256 of the compiled publisher tokenizer the "
                             "API channel serves (runtime/" + TOKENIZER_ASSET +
                             "); omitted for residentd-only roots")
    parser.add_argument("--state-budget-bytes", type=int,
                        default=DEFAULT_STATE_BUDGET_BYTES,
                        help="per-rank budget for KDA state, windows, MLA KV "
                             "and scratch; the stage runner refuses a plan "
                             "over it (default %(default)d)")
    parser.add_argument("--check", action="store_true",
                        help="verify the outputs are current instead of "
                             "writing them")
    arguments = parser.parse_args()

    select_topology(arguments.topology)
    select_lane(arguments.lane)
    if arguments.kv_physical_pages is None:
        arguments.kv_physical_pages = arguments.sequences * arguments.kv_pages
    if arguments.kv_logical_pages is None:
        arguments.kv_logical_pages = max(arguments.kv_physical_pages, arguments.sequences * arguments.kv_pages)
    if arguments.kv_physical_pages < arguments.kv_pages:
        raise SystemExit(f"kv-physical-pages {arguments.kv_physical_pages} must hold one sequence's "
                         f"{arguments.kv_pages} kv-pages")
    if arguments.kv_logical_pages < max(arguments.kv_physical_pages, arguments.sequences * arguments.kv_pages):
        raise SystemExit(f"kv-logical-pages {arguments.kv_logical_pages} must cover the physical pages and "
                         f"{arguments.sequences} sequences x {arguments.kv_pages} kv-pages")
    if arguments.rows is not None and not arguments.sequences <= arguments.rows <= MAX_ROWS:
        raise SystemExit(f"rows must be within sequences..{MAX_ROWS}: a decode wave carries one row per sequence "
                         f"and the K3 adapter takes at most {MAX_ROWS} rows")
    minimum = kv_backing_minimum(arguments.topology, arguments.sequences, arguments.kv_physical_pages, arguments.kv_logical_pages)
    if arguments.kv_backing_bytes is None:
        arguments.kv_backing_bytes = minimum
    if arguments.kv_backing_bytes < minimum:
        raise SystemExit(f"kv-backing-bytes {arguments.kv_backing_bytes} cannot hold "
                         f"{arguments.kv_logical_pages - arguments.kv_physical_pages + 2} spilled "
                         f"{kv_page_bytes(arguments.topology)}-byte KV pages and "
                         f"{2 * arguments.sequences + 2} {recurrent_page_bytes(arguments.topology)}-byte "
                         f"KDA checkpoints; it needs {minimum}")
    if arguments.kv_snapshot_bytes <= 0:
        raise SystemExit("kv-snapshot-bytes must be positive and finite")
    if not 1 <= arguments.sequences <= 16:
        raise SystemExit("sequences must be within 1..16")
    if arguments.state_budget_bytes <= 0:
        raise SystemExit("state-budget-bytes must be positive")
    if not 1 <= arguments.kv_pages <= 16384:
        raise SystemExit("kv-pages must be within 1..16384")

    output = Path(arguments.output_dir)
    if not arguments.check:
        output.mkdir(parents=True, exist_ok=True)

    deployment = render(resident_deployment(
        arguments.runtime_root, arguments.weightd_socket,
        arguments.kv_backing_bytes, arguments.kv_snapshot_bytes, arguments.sequences, arguments.kv_pages,
        arguments.pipeline_transport, arguments.tokenizer_sha256,
        arguments.rows, arguments.kv_physical_pages, arguments.kv_logical_pages))
    wrote = write_or_check(output / "deployment.json", deployment,
                           arguments.check)

    ranks = range(WORLD) if arguments.rank is None else [arguments.rank]
    for rank in ranks:
        text = render(adapter_config(rank, arguments.kv_pages,
                                     arguments.sequences, arguments.rows,
                                     arguments.state_budget_bytes))
        name = f"adapter.{host_of(rank)}.json" if arguments.rank is None \
            else "adapter.json"
        wrote = write_or_check(output / name, text, arguments.check) or wrote

    action = "checked" if arguments.check else ("wrote" if wrote else "kept")
    print(f"k3 lane {LANE} deployment {action} under {output} "
          f"(mesh ranks {MESH_RANKS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
  tp_collective listen  53048 + tp     BOUND (host TCP collective; required
                                      by the k3 adapter at tp_degree 4).
                                      Group-local: the four ranks of a PP
                                      stage listen 53048..53051 and every
                                      stage reuses those numbers because
                                      its ranks are on disjoint hosts.
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
import sys
from pathlib import Path

LANE = 3
WORLD = 16
TP = 4
PP = 4
HEX = "0123456789abcdef"
HOSTS = [f"spark{HEX[i]}" for i in range(WORLD)]

# Numeric peer addresses for the host TP collective: SparkTpCollectiveCreate
# validates peers with inet_pton (IPv4 literals only — hostnames are
# INVALID_ARGUMENT at create). The fleet's sparkN names are static DNS
# (verified 2026-09-23: spark0=10.10.100.10 .. sparkf=10.10.100.25).
HOST_ADDRESSES = {f"spark{HEX[i]}": f"10.10.100.{10 + i}" for i in range(WORLD)}

CONTROL_BASE = 23048                            # 23048 .. 23063
COLLECTIVE_BASE = 53048                         # 53048 .. 53063 (u16-valid, #1094)
TRANSPORT_BASE = 64048                          # 64048 .. 64063

TP_COLLECTIVE_PORT = COLLECTIVE_BASE      # + tp rank (group-local, bound)
DEVICE_SESSION_BASE = COLLECTIVE_BASE + 4 # packed 12-number table below
MESH_RANKS = ",".join(str(i) for i in range(WORLD))

DEPLOYED_PACK_TEMPLATE = (
    "/home/{host}/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage{stage}.rank0{tp}.pack")

NODE_TARGET = "cuda.sm121.k3.resident_decode_stage.linear_bf16.expert_mxfp4.kv_bf16"
DEFAULT_KV_BACKING_BYTES = 8 * 1024 * 1024 * 1024
KV_PAGES_PER_SEQUENCE = 64   # adapter_config default; x SPARK_K3_KV_PAGE_SLOTS (64) tokens

# The batch engine refuses a deployment with no EOS tokens (SCHEMA_ERROR at
# SparkModelBatchValidateConfiguration — cold14: status=6, tokens=0, the
# request never reached the tensors; adapter_ops stayed 0). Single source of
# truth is the authoritative contract, the same record the compiled driver's
# K3_EOS_TOKEN is generated from (tools/generate_k3_contract.py). Never
# hardcode the id here; fail closed if the contract cannot provide it.
CONTRACT = json.loads((Path(__file__).resolve().parents[1] /
                       "model_contracts/k3_authoritative.json").read_text())
try:
    K3_EOS_TOKEN_IDS = [int(CONTRACT["tokens"]["end_of_text"])]
except (KeyError, TypeError, ValueError):
    raise SystemExit("k3 contract missing tokens.end_of_text; refusing to emit "
                     "a deployment the batch engine would reject")
if K3_EOS_TOKEN_IDS[0] <= 0:
    raise SystemExit("k3 contract tokens.end_of_text must be a positive id")


def host_of(rank: int) -> str:
    return HOSTS[rank]


def stage_of(rank: int) -> int:
    return rank // TP


def tp_rank_of(rank: int) -> int:
    return rank % TP


def deployed_pack(rank: int) -> str:
    return DEPLOYED_PACK_TEMPLATE.format(
        host=host_of(rank), stage=stage_of(rank), tp=tp_rank_of(rank))


def session_table() -> list[list[int]]:
    """12 numbers packed into 53052..53063 with no per-host overlap.

    Row a is listened by the host whose TP rank is a (every PP stage
    reuses the table because stages own disjoint hosts). The packing
    53052 + 3a + (b if b < a else b - 1) keeps all twelve values inside
    the twelve numbers left in the collective block after the bound
    tp_collective listeners take 53048..53051.
    """
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


def adapter_config(rank: int, kv_pages: int = 64) -> dict:
    tp = tp_rank_of(rank)
    return {
        "stage_pack_path": deployed_pack(rank),
        "tp_degree": TP,
        "tp_rank": tp,
        "world_size": WORLD,
        "max_sequences": 16,
        "max_rows": 16,
        "resident_capacity": 16,
        "kv_pages": kv_pages,
        "capture_graphs": 1,
        "hidden": 7168,
        "device_collective": {
            "backend": "hidden_transport",
            "backend_module_path": "lib/hidden_transport.so",
            "local_host": host_of(rank),
            "collective_identifier": 0x0003_0000_0000_0000 | stage_of(rank),
            "listen_port": TRANSPORT_BASE + 15,
            "connect_timeout_milli": 300000,
            "operation_timeout_milli": 30000,
            "peer_hosts": group_hosts(rank),
            "session_ports": session_table(),
        },
        "tp_collective": {
            "listen_port": TP_COLLECTIVE_PORT + tp,
            "connect_timeout_milli": 300000,
            "operation_timeout_milli": 30000,
            "collective_identifier": 1,
            "peers": [
                "{host}:{port}".format(
                    host=HOST_ADDRESSES[group_hosts(rank)[partner]],
                    port=TP_COLLECTIVE_PORT + partner)
                for partner in (tp ^ 1, tp ^ 2)
            ],
        },
    }


def resident_deployment(runtime_root: str, weightd_socket: str,
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
            "shared_object_path": "lib/hidden_transport.so",
            "mode": "host-rdma",
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
            "max_inflight_submissions": 16,
            "max_active_sequences": 16,
            "max_input_rows": 16,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": 16 * KV_PAGES_PER_SEQUENCE,
            "kv_physical_page_capacity": 16 * KV_PAGES_PER_SEQUENCE,
        },
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
                        default=DEFAULT_KV_BACKING_BYTES,
                        help="finite KV backing cap for the private root "
                             "(default %(default)d)")
    parser.add_argument("--rank", type=int, choices=range(WORLD), default=None,
                        help="emit only this rank's adapter.json "
                             "(default: all sixteen)")
    parser.add_argument("--check", action="store_true",
                        help="verify the outputs are current instead of "
                             "writing them")
    arguments = parser.parse_args()

    if arguments.kv_backing_bytes <= 0:
        raise SystemExit("kv-backing-bytes must be positive and finite")

    output = Path(arguments.output_dir)
    if not arguments.check:
        output.mkdir(parents=True, exist_ok=True)

    deployment = render(resident_deployment(
        arguments.runtime_root, arguments.weightd_socket,
        arguments.kv_backing_bytes))
    wrote = write_or_check(output / "deployment.json", deployment,
                           arguments.check)

    ranks = range(WORLD) if arguments.rank is None else [arguments.rank]
    for rank in ranks:
        text = render(adapter_config(rank))
        name = f"adapter.{host_of(rank)}.json" if arguments.rank is None \
            else "adapter.json"
        wrote = write_or_check(output / name, text, arguments.check) or wrote

    action = "checked" if arguments.check else ("wrote" if wrote else "kept")
    print(f"k3 lane {LANE} deployment {action} under {output} "
          f"(mesh ranks {MESH_RANKS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

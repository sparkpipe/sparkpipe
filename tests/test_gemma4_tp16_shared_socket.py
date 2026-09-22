#!/usr/bin/env python3
"""Lane-6 gemma4-31b TP16 shared-socket deployment contracts.

Validates tools/gemma4_tp16_gen_deployment.py output against the lane 6
port blocks (control 23096-23111, collective 53200-53215, transport
64096-64111; tools/devcycle/lane_assignments.json) and the adapter's exact
configuration members, then exercises tools/gemma4_tp16_shared_socket.sh
end to end in --dry-run against a synthetic checkout: layout, pack sidecar
fail-closed behavior, ${SPARK_QUEUE_RUNTIME_ROOT} substitution, and the
exactly-one-digest rule for shared weightd attachment.

Run: python3 tests/test_gemma4_tp16_shared_socket.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
GENERATOR = REPOSITORY / "tools/gemma4_tp16_gen_deployment.py"
WRAPPER = REPOSITORY / "tools/gemma4_tp16_shared_socket.sh"

RANKS = 16
LANE_CONTROL_BASE, LANE_CONTROL_END = 23096, 23111
LANE_COLLECTIVE_BASE, LANE_COLLECTIVE_END = 53200, 53215
LANE_TRANSPORT_BASE, LANE_TRANSPORT_END = 64096, 64111
MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
ADAPTER_MEMBERS = {"schema_version", "model_revision", "stage_pack_path",
                   "max_sequence_positions", "tp_degree"}
PACK_SIDECAR = re.compile(r"^[0-9a-f]{64}  gemma4_31b_tp16_rank[0-9a-f]_stage0.gemma4sp$")


def fail(message: str) -> None:
    print(f"test_gemma4_tp16_shared_socket: FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def generate(temporary: Path) -> Path:
    output = temporary / "deployment"
    subprocess.run([sys.executable, str(GENERATOR), "--output", str(output)],
                   check=True, capture_output=True)
    return output


def test_generator(output: Path) -> dict:
    deployment = json.loads((output / "model_resident.json").read_text())
    check(deployment["schema_version"] == 2, "model_resident schema_version")
    check(deployment["coordinator_rank_index"] == 0, "coordinator rank")
    check(deployment["eos_token_ids"] == [1, 106, 50], "eos token ids")
    check(deployment["adapter"]["shared_object_path"] == "lib/model_serving_adapter.so",
          "adapter path")
    check(deployment["driver"]["shared_object_path"] == "stages/stage_000/model_driver.so",
          "driver path")
    check(deployment["driver"]["program_name"] == "resident_decode", "program name")
    transport = deployment["transport"]
    check(transport["mode"] == "host-rdma", "transport mode")
    check(transport["control_port_base"] == LANE_TRANSPORT_BASE, "transport port base")
    check(LANE_TRANSPORT_BASE <= transport["control_port_base"] <= LANE_TRANSPORT_END,
          "transport base inside lane block")
    nodes = deployment["nodes"]
    check(len(nodes) == RANKS, "node count")
    for node in nodes:
        rank = node["rank_index"]
        check(node["stage_index"] == rank, f"rank {rank} stage index (unique rank/stage pairs)")
        check(node["node_target"] == "cuda.sm121.gemma4.31b.resident_decode_stage.bf16",
              f"rank {rank} node target")
        check(node["runtime_root"] == "${SPARK_QUEUE_RUNTIME_ROOT}",
              f"rank {rank} runtime root template")
        endpoint = node["control_endpoint"]
        check(endpoint["kind"] == "tcp", f"rank {rank} control kind")
        check(endpoint["host"] == f"spark{hex(rank)[2:]}", f"rank {rank} control host")
        expected_port = LANE_CONTROL_BASE + rank
        check(endpoint["port"] == expected_port, f"rank {rank} control port")
        check(LANE_CONTROL_BASE <= endpoint["port"] <= LANE_CONTROL_END,
              f"rank {rank} control port inside lane block")
        check(node["kv_backing_directory"].startswith("${SPARK_QUEUE_RUNTIME_ROOT}"),
              f"rank {rank} kv backing under runtime root")
    for rank in range(RANKS):
        stage = json.loads((output / "config" / f"stage_{rank:02d}.json").read_text())
        check(set(stage.keys()) == ADAPTER_MEMBERS,
              f"rank {rank} adapter members exact: {sorted(stage.keys())}")
        check(stage["schema_version"] == 3, f"rank {rank} schema version")
        check(stage["model_revision"] == MODEL_REVISION, f"rank {rank} revision")
        check(stage["tp_degree"] == RANKS, f"rank {rank} tp degree")
        check(stage["stage_pack_path"] ==
              f"packs/gemma4_31b_tp16_rank{hex(rank)[2:]}_stage0.gemma4sp",
              f"rank {rank} pack path (hex rank)")
        env = json.loads((output / "config" / f"env_{rank:02d}.json").read_text())
        check(env["SPARK_GEMMA4_TP_DEGREE"] == "16", f"rank {rank} env tp degree")
        check(env["SPARK_GEMMA4_TP_RANK"] == str(rank), f"rank {rank} env tp rank")
        check(env["SPARK_GEMMA4_TP_STANDALONE"] == "0", f"rank {rank} standalone")
        check(env["SPARK_GEMMA4_STAGE_TP_PORT_BASE"] == str(LANE_COLLECTIVE_BASE),
              f"rank {rank} collective base")
        check(env["SPARK_GEMMA4_STAGE_TP_HOSTS"].split(",") ==
              [f"spark{hex(r)[2:]}" for r in range(RANKS)], f"rank {rank} hosts")
        check(env["SPARK_GEMMA4_STAGE_TP_LOCAL_HOST"] == f"spark{hex(rank)[2:]}",
              f"rank {rank} local host")
        check(int(env["SPARK_GEMMA4_STAGE_TP_IDENTIFIER"]) > 0, f"rank {rank} identifier")
        cells = [int(cell) for cell in env["SPARK_GEMMA4_STAGE_TP_SESSION_PORTS"].split(",")]
        check(len(cells) == RANKS * RANKS, f"rank {rank} session matrix size")
        for row in range(RANKS):
            for column in range(RANKS):
                cell = cells[row * RANKS + column]
                if row == column:
                    check(cell == 0, f"rank {rank} session diag ({row},{column})")
                else:
                    check(LANE_COLLECTIVE_BASE <= cell <= LANE_COLLECTIVE_END,
                          f"rank {rank} session cell ({row},{column})={cell} inside lane block")
    return deployment


def synthetic_checkout(temporary: Path, deployment_tree: Path) -> Path:
    checkout = temporary / "checkout"
    release = checkout / "build/gemma4_31b_tp16"
    for artifact in ["bin/sparkpipe_model_residentd", "lib/model_serving_adapter.so",
                     "lib/hidden_transport.so", "stages/stage_000/model_driver.so"]:
        path = release / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic")
    (release / "SOURCE_COMMIT").write_text("0" * 40)
    (release / "SHA256SUMS").write_text("")
    tree = checkout / "deployment/gemma4_31b_tp16_lane6"
    shutil.copytree(deployment_tree, tree)
    packs = checkout / "packs"
    packs.mkdir()
    for rank in range(RANKS):
        name = f"gemma4_31b_tp16_rank{hex(rank)[2:]}_stage0.gemma4sp"
        (packs / name).write_bytes(b"0" * 1024)
        (packs / f"{name}.sha256").write_text(f"{'a' * 64}  {name}\n")
    return checkout


def run_wrapper(checkout: Path, temporary: Path, rank: int) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment.update({
        "SPARK_QUEUE_RANK": str(rank),
        "SPARK_QUEUE_SIZE": str(RANKS),
        "SPARK_QUEUE_MEMORY_MIB": "9792",
        "SPARK_QUEUE_RUNTIME_ROOT": str(temporary / f"runtime-{rank}"),
        "GEMMA4_RELEASE_DIR": "build/gemma4_31b_tp16",
        "GEMMA4_DEPLOYMENT_TREE": "deployment/gemma4_31b_tp16_lane6",
        "GEMMA4_PACK_DIR": str(checkout / "packs"),
        "GEMMA4_SHARED_WEIGHTD_SOCKET": str(temporary / "absent.sock"),
    })
    return subprocess.run(["bash", str(WRAPPER), "--dry-run"], cwd=checkout, env=environment,
                          capture_output=True, text=True)


def test_wrapper(deployment_tree: Path, temporary: Path) -> None:
    checkout = synthetic_checkout(temporary, deployment_tree)
    result = run_wrapper(checkout, temporary, 11)
    if result.returncode != 0:
        fail(f"wrapper dry-run rank 11 failed: {result.stderr}")
    root = temporary / "runtime-11"
    deployment = json.loads((root / "deployment.json").read_text())
    for node in deployment["nodes"]:
        check(node["runtime_root"] == str(root),
              "runtime root substituted in materialized deployment")
    check((root / "packs/gemma4_31b_tp16_rankb_stage0.gemma4sp").exists(),
          "rank 11 pack present (hex rank b)")
    sidecars = list((root / "packs").glob("*.sha256"))
    check(len(sidecars) == 1, f"exactly one sha256 sidecar, found {len(sidecars)}")
    check(PACK_SIDECAR.match(sidecars[0].read_text()) is not None,
          "sidecar names exactly its pack")
    check((root / "config/stage.json").exists(), "stage config materialized")
    check((root / "kv").is_dir(), "kv backing directory materialized (residentd requires a real dir)")
    wrapper_source = WRAPPER.read_text()
    check("SPARK_WEIGHTD_PACK_SHA256" in wrapper_source and
          "SPARK_WEIGHTD_ATTACH=1" in wrapper_source and
          "SPARK_WEIGHTD_IDENTITY_MODEL" in wrapper_source,
          "wrapper exports the pack-identity attach envs")
    check((root / "config/env.json").exists(), "module env materialized")

    # fail-closed: malformed sidecar
    (checkout / "packs/gemma4_31b_tp16_rankb_stage0.gemma4sp.sha256").write_text("nothex  x\n")
    result = run_wrapper(checkout, temporary, 11)
    check(result.returncode != 0, "malformed sidecar must fail closed")
    # fail-closed: missing sidecar entirely (the nvfp4 arm today)
    (checkout / "packs/gemma4_31b_tp16_rankb_stage0.gemma4sp.sha256").unlink()
    result = run_wrapper(checkout, temporary, 11)
    check(result.returncode != 0, "missing sidecar must fail closed")
    # fail-closed: rank out of range
    result = run_wrapper(checkout, temporary, 16)
    check(result.returncode != 0, "rank 16 must fail closed")
    # fail-closed: unbounded memory (the hard rule - refuse queue-less runs)
    environment = dict(os.environ)
    environment.update({
        "SPARK_QUEUE_RANK": "0", "SPARK_QUEUE_RUNTIME_ROOT": str(temporary / "runtime-unbounded"),
        "GEMMA4_RELEASE_DIR": "build/gemma4_31b_tp16", "GEMMA4_PACK_DIR": str(checkout / "packs"),
    })
    environment.pop("SPARK_QUEUE_MEMORY_MIB", None)
    result = subprocess.run(["bash", str(WRAPPER), "--dry-run"], cwd=checkout, env=environment,
                            capture_output=True, text=True)
    check(result.returncode != 0, "missing SPARK_QUEUE_MEMORY_MIB must fail closed (hard rule)")
    # fail-closed: missing release artifacts
    environment = dict(os.environ)
    environment.update({
        "SPARK_QUEUE_RANK": "0", "SPARK_QUEUE_RUNTIME_ROOT": str(temporary / "runtime-x"),
        "GEMMA4_RELEASE_DIR": "build/absent", "GEMMA4_PACK_DIR": str(checkout / "packs"),
    })
    result = subprocess.run(["bash", str(WRAPPER), "--dry-run"], cwd=checkout, env=environment,
                            capture_output=True, text=True)
    check(result.returncode != 0, "missing release must fail closed")
    # dry-run never requires the shared socket (runtime validation is gated
    # on the operator-established weightd; the non-dry path fails closed)
    check("shared weightd socket" not in result.stdout, "no socket requirement in dry-run")


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deployment_tree = generate(temporary)
        test_generator(deployment_tree)
        test_wrapper(deployment_tree, temporary)
    print("gemma4 tp16 shared-socket contracts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

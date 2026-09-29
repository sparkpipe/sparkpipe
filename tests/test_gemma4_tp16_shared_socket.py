#!/usr/bin/env python3
"""Lane-6 gemma4-31b TP16 shared-socket deployment contracts.

Validates tools/gemma4_tp16_gen_deployment.py output against the lane 6
port blocks (control 23096-23111, collective 53200-53215, transport
64096-64111; tools/devcycle/lane_assignments.json), the fleet weightd
socket and the adapter's exact configuration members, then exercises
tools/gemma4_tp16_shared_socket.sh against a synthetic checkout: the
--dry-run layout, ${SPARK_QUEUE_RUNTIME_ROOT} substitution and the
exactly-one-digest rule, each fail-closed case with its stated reason, and
a full launch through a stand-in residentd that records the environment
and arguments the wrapper hands it.

Run: python3 tests/test_gemma4_tp16_shared_socket.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
GENERATOR = REPOSITORY / "tools/gemma4_tp16_gen_deployment.py"
WRAPPER = REPOSITORY / "tools/gemma4_tp16_shared_socket.sh"

RANKS = 16
FLEET_WEIGHTD_SOCKET = "/tmp/spark_weightd.sock"
STALE_SHARED_UNIT_SOCKET = "/run/sparkpipe-weightd-shared/weightd.sock"
IDENTITY_MODEL = "cuda.sm121.gemma4.31b.resident_decode_stage.bf16"
MESH_RANKS = ",".join(str(rank) for rank in range(RANKS))
RESIDENTD_STAND_IN = """#!/usr/bin/env bash
printf 'ARG %s\\n' "$@"
env | sed 's/^/ENV /'
"""
LANE_CONTROL_BASE, LANE_CONTROL_END = 23096, 23111
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


def test_generator_refuses_foreign_socket(temporary: Path) -> None:
    for refused in (STALE_SHARED_UNIT_SOCKET, str(temporary / "private.sock")):
        output = temporary / "refused"
        result = subprocess.run([sys.executable, str(GENERATOR), "--output", str(output),
                                 "--weightd-socket", refused],
                                capture_output=True, text=True)
        check(result.returncode != 0 and
              f"weightd socket {refused} is not the fleet weightd" in result.stderr,
              f"generator must refuse {refused}: rc={result.returncode} {result.stderr}")
        check(not output.exists(), f"refused socket {refused} still wrote a tree")


def test_generator(output: Path) -> dict:
    deployment = json.loads((output / "model_resident.json").read_text())
    check(deployment["schema_version"] == 2, "model_resident schema_version")
    check(deployment["weightd"]["socket_path"] == FLEET_WEIGHTD_SOCKET,
          f"weightd socket {deployment['weightd']['socket_path']} must be the fleet weightd")
    check(deployment["coordinator_rank_index"] == 0, "coordinator rank")
    check(deployment["eos_token_ids"] == [1, 106, 50], "eos token ids")
    limits = deployment["runtime_limits"]
    check(limits["max_inflight_submissions"] == 1, "inflight within adapter cap")
    check(limits["max_input_rows"] <= 512, "input rows within adapter cap")
    env0 = json.loads((output / "config" / "env_00.json").read_text())
    check(limits["max_input_rows"] * 5376 * 2 + 16 <= 8 * 32768 + 64 or
          env0["SPARK_TP_WAIT_MODE"] == "hardware",
          "prefill frames wider than one mesh slot need the device-round (hardware) collective")
    check(limits["max_input_rows"] >= limits["max_active_sequences"],
          "input rows cover the active sequences")
    stage = json.loads((output / "config" / "stage_00.json").read_text())
    check(limits["max_sequence_positions"] == stage["max_sequence_positions"],
          "api context limit equals the engine's sequence positions")
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
        check(env["SPARK_TP_WAIT_MODE"] == "hardware",
              f"rank {rank} collective waits on the device like the fleet stations")
        check(set(env) == {"SPARK_GEMMA4_TP_DEGREE", "SPARK_GEMMA4_TP_RANK",
                           "SPARK_GEMMA4_TP_STANDALONE", "SPARK_GEMMA4_STAGE_TP_TIMEOUT_MS",
                           "SPARK_TP_WAIT_MODE"},
              f"rank {rank} env names only what the module reads: {sorted(env)}")
    return deployment


def synthetic_checkout(temporary: Path, deployment_tree: Path) -> Path:
    checkout = temporary / "checkout"
    release = checkout / "build/gemma4_31b_tp16"
    for artifact in ["bin/sparkpipe_model_residentd", "lib/model_serving_adapter.so",
                     "lib/hidden_transport.so", "stages/stage_000/model_driver.so"]:
        path = release / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic")
    residentd = release / "bin/sparkpipe_model_residentd"
    residentd.write_text(RESIDENTD_STAND_IN)
    residentd.chmod(residentd.stat().st_mode | stat.S_IXUSR)
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


def wrapper_environment(checkout: Path, temporary: Path, rank: int, **overrides) -> dict:
    environment = {key: value for key, value in os.environ.items()
                   if not (key.startswith("SPARK_QUEUE_") or key.startswith("GEMMA4_")
                           or key.startswith("SPARK_WEIGHTD_"))}
    environment.update({
        "SPARK_QUEUE_RANK": str(rank),
        "SPARK_QUEUE_SIZE": str(RANKS),
        "SPARK_QUEUE_MEMORY_MIB": "9792",
        "SPARK_QUEUE_RUNTIME_ROOT": str(temporary / f"runtime-{rank}"),
        "GEMMA4_RELEASE_DIR": "build/gemma4_31b_tp16",
        "GEMMA4_DEPLOYMENT_TREE": "deployment/gemma4_31b_tp16_lane6",
        "GEMMA4_PACK_DIR": str(checkout / "packs"),
    })
    for key, value in overrides.items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    return environment


def run_wrapper(checkout: Path, environment: dict, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(WRAPPER), *arguments], cwd=checkout, env=environment,
                          capture_output=True, text=True)


def expect_refusal(checkout: Path, environment: dict, reason: str, name: str) -> None:
    result = run_wrapper(checkout, environment, "--dry-run")
    check(result.returncode != 0 and reason in result.stderr,
          f"{name} must fail closed with '{reason}': rc={result.returncode} {result.stderr}")
    check(not Path(environment["SPARK_QUEUE_RUNTIME_ROOT"]).exists(),
          f"{name} created the runtime root before failing")


def set_tree_socket(checkout: Path, socket_path: str) -> None:
    tree = checkout / "deployment/gemma4_31b_tp16_lane6/model_resident.json"
    deployment = json.loads(tree.read_text())
    deployment["weightd"]["socket_path"] = socket_path
    tree.write_text(json.dumps(deployment, indent=1) + "\n")


def test_wrapper_launch(checkout: Path, temporary: Path) -> None:
    absent = str(temporary / "absent.sock")
    set_tree_socket(checkout, absent)
    environment = wrapper_environment(checkout, temporary, 12,
                                      GEMMA4_SHARED_WEIGHTD_SOCKET=absent)
    result = run_wrapper(checkout, environment)
    check(result.returncode != 0 and
          f"shared weightd socket is not present: {absent}" in result.stderr,
          f"launch without a live weightd socket must fail closed: {result.stderr}")
    live_path = temporary / "weightd.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(live_path))
        listener.listen(1)
        set_tree_socket(checkout, str(live_path))
        environment = wrapper_environment(checkout, temporary, 11,
                                          GEMMA4_SHARED_WEIGHTD_SOCKET=str(live_path))
        result = run_wrapper(checkout, environment)
    finally:
        listener.close()
        set_tree_socket(checkout, FLEET_WEIGHTD_SOCKET)
    check(result.returncode == 0, f"launch through the stand-in residentd failed: {result.stderr}")
    root = temporary / "runtime-11"
    arguments = [line[4:] for line in result.stdout.splitlines() if line.startswith("ARG ")]
    check(arguments == ["--deployment", str(root / "deployment.json"), "--rank-index", "11"],
          f"residentd arguments: {arguments}")
    exported = dict(line[4:].split("=", 1) for line in result.stdout.splitlines()
                    if line.startswith("ENV ") and "=" in line)
    expected = {
        "SPARK_WEIGHTD_ATTACH": "1",
        "SPARK_WEIGHTD_PACK_SHA256": "a" * 64,
        "SPARK_WEIGHTD_IDENTITY_MODEL": IDENTITY_MODEL,
        "SPARK_WEIGHTD_SOCKET": str(live_path),
        "SPARK_WEIGHTD_LANE": "6",
        "SPARK_TP_MESH_RANKS": MESH_RANKS,
        "SPARK_GEMMA4_TP_DEGREE": "16",
        "SPARK_GEMMA4_TP_RANK": "11",
        "SPARK_GEMMA4_TP_STANDALONE": "0",
    }
    for key, value in expected.items():
        check(exported.get(key) == value,
              f"residentd environment {key}={exported.get(key)!r}, expected {value!r}")
    deployment = json.loads((root / "deployment.json").read_text())
    check(deployment["weightd"]["socket_path"] == str(live_path),
          "the launched deployment names the attach socket")


def test_wrapper(deployment_tree: Path, temporary: Path) -> None:
    checkout = synthetic_checkout(temporary, deployment_tree)
    environment = wrapper_environment(checkout, temporary, 11)
    result = run_wrapper(checkout, environment, "--dry-run")
    if result.returncode != 0:
        fail(f"wrapper dry-run rank 11 failed: {result.stderr}")
    check(f"socket={FLEET_WEIGHTD_SOCKET} lane=6" in result.stdout,
          f"dry-run attaches through the fleet weightd: {result.stdout}")
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
    check(json.loads((root / "config/env.json").read_text()) ==
          json.loads((deployment_tree / "config/env_11.json").read_text()),
          "module env materialized from the rank's env json")
    shutil.rmtree(root)

    expect_refusal(checkout, wrapper_environment(checkout, temporary, 11,
                                                 SPARK_QUEUE_MEMORY_MIB=None),
                   "SPARK_QUEUE_MEMORY_MIB is unset/zero", "missing memory bound")
    expect_refusal(checkout, wrapper_environment(checkout, temporary, 11,
                                                 SPARK_QUEUE_MEMORY_MIB="0"),
                   "SPARK_QUEUE_MEMORY_MIB is unset/zero", "zero memory bound")
    expect_refusal(checkout, wrapper_environment(checkout, temporary, 16),
                   "SPARK_QUEUE_RANK 16 outside 0..15", "rank 16")
    expect_refusal(checkout, wrapper_environment(checkout, temporary, 11,
                                                 GEMMA4_RELEASE_DIR="build/absent"),
                   "release missing:", "missing release")
    expect_refusal(checkout, wrapper_environment(checkout, temporary, 11,
                                                 GEMMA4_DEPLOYMENT_TREE="deployment/absent"),
                   "deployment tree missing:", "missing deployment tree")
    expect_refusal(checkout, wrapper_environment(checkout, temporary, 11,
                                                 GEMMA4_SHARED_WEIGHTD_SOCKET=str(temporary / "other.sock")),
                   f"deployment tree weightd socket {FLEET_WEIGHTD_SOCKET} differs from the attach "
                   f"socket {temporary / 'other.sock'}", "attach socket differing from the tree")
    sidecar = checkout / "packs/gemma4_31b_tp16_rankb_stage0.gemma4sp.sha256"
    sidecar.write_text("nothex  x\n")
    expect_refusal(checkout, wrapper_environment(checkout, temporary, 11),
                   "malformed pack digest sidecar", "malformed sidecar")
    sidecar.write_text(f"{'a' * 64}  gemma4_31b_tp16_rank3_stage0.gemma4sp\n")
    expect_refusal(checkout, wrapper_environment(checkout, temporary, 11),
                   "sidecar names 'gemma4_31b_tp16_rank3_stage0.gemma4sp'", "sidecar naming another pack")
    sidecar.unlink()
    expect_refusal(checkout, wrapper_environment(checkout, temporary, 11),
                   "pack digest sidecar missing", "missing sidecar")
    sidecar.write_text(f"{'a' * 64}  gemma4_31b_tp16_rankb_stage0.gemma4sp\n")
    test_wrapper_launch(checkout, temporary)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deployment_tree = generate(temporary)
        test_generator(deployment_tree)
        test_generator_refuses_foreign_socket(temporary)
        test_wrapper(deployment_tree, temporary)
    print("gemma4 tp16 shared-socket contracts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

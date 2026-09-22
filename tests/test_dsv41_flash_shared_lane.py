#!/usr/bin/env python3
"""The dsv41_flash shared-lane deployment generator and family wrapper,
verified with the stdlib alone.

Lane 4 locks (PR #1083 lane_assignments.json / fleet registry band):
  1. the generated tree carries TP4 stage configs whose tp_rank/tp_degree
     and collective listener ports match the lane's node order and the
     53064-53079 collective block; control endpoints sit in 23064-23079;
     the transport base is 64064; every generated listener is inside a
     lane-4 reserved block.
  2. the stage config member set is exactly the dsv41_flash module node
     context (the family adapter must map these members and no others).
  3. the wrapper (prepare mode) builds the private runtime root with
     exactly one *.sha256 digest beside the rank pack (the shared
     weightd attach law), rewrites this rank's node onto the private
     root and the shared socket, and fail-closes on a missing sidecar,
     a missing environment variable or a duplicated digest.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "tools/dsv41_flash_gen_deployment.py"
WRAPPER = ROOT / "tools/dsv41_flash_shared_lane.sh"
LANE_HOSTS = ["spark4", "spark5", "spark6", "spark7"]
LANE_CONTROL = (23064, 23079)
LANE_COLLECTIVE = (53064, 53079)   # PR #1094 renumber (67064+ is not bindable)
LANE_TRANSPORT = (64064, 64079)

STAGE_MEMBERS = {
    "schema_version", "model_revision", "expert_weight_codec", "stage_count",
    "stage_index", "first_layer_index", "layer_count", "stage_pack_path",
    "max_sequence_positions", "execution_row_capacity",
    "resident_sequence_capacity", "pipeline_slot_count", "tp_degree",
    "tp_rank", "tp_collective",
}


def in_block(port, block):
    return block[0] <= port <= block[1]


def run(argv, env=None):
    return subprocess.run(argv, capture_output=True, text=True, env=env)


def main() -> int:
    failures = []
    tmp = tempfile.mkdtemp(prefix="dsv41-lane4-")
    tree = os.path.join(tmp, "deployment")

    result = run([sys.executable, str(GENERATOR), "--output", tree])
    if result.returncode != 0:
        print(result.stdout, result.stderr)
        return 1

    # -- the stage configs ------------------------------------------------
    for rank in range(4):
        stage = json.loads(
            Path(tree, "config", f"stage_{rank:02d}.json").read_text())
        if set(stage) != STAGE_MEMBERS:
            failures.append(
                f"stage {rank}: member set {sorted(set(stage) ^ STAGE_MEMBERS)} "
                "differs from the module node context")
        if stage["tp_degree"] != 4 or stage["tp_rank"] != rank:
            failures.append(f"stage {rank}: tp {stage['tp_degree']}/{stage['tp_rank']}")
        if stage["layer_count"] != 40 or stage["stage_count"] != 1:
            failures.append(f"stage {rank}: not the single 40-layer stage")
        if stage["expert_weight_codec"] != "mxfp4":
            failures.append(f"stage {rank}: codec {stage['expert_weight_codec']}")
        collective = stage["tp_collective"]
        if not in_block(collective["listen_port"], LANE_COLLECTIVE):
            failures.append(f"stage {rank}: collective listen {collective['listen_port']}")
        if collective["listen_port"] != 53064 + rank:
            failures.append(f"stage {rank}: listener not rank-offset")
        if collective["peer_hosts"] != LANE_HOSTS:
            failures.append(f"stage {rank}: peers {collective['peer_hosts']}")
        if collective["peer_ports"] != [53064 + r for r in range(4)]:
            failures.append(f"stage {rank}: peer ports {collective['peer_ports']}")
        for rail in collective["rail_peer_hosts"]:
            if len(rail) != 4:
                failures.append(f"stage {rank}: rail width {len(rail)}")
        if len(collective["step_rail_indices"]) != 4:
            failures.append(f"stage {rank}: step_rail_indices width")

    # -- the deployment ---------------------------------------------------
    deployment = json.loads(Path(tree, "model_resident.json").read_text())
    nodes = deployment["nodes"]
    if len(nodes) != 4:
        failures.append(f"deployment: {len(nodes)} nodes, want 4")
    for rank, node in enumerate(nodes):
        if node["rank_index"] != rank:
            failures.append(f"node {rank}: rank_index {node['rank_index']}")
        host = LANE_HOSTS[rank]
        if node["transport_host"] != host:
            failures.append(f"node {rank}: host {node['transport_host']}")
        if node["runtime_root"] != f"/home/{host}/sparkdata/dsv41flash.mxfp4.tp4":
            failures.append(f"node {rank}: runtime_root {node['runtime_root']}")
        if not node["kv_backing_directory"].startswith(node["runtime_root"]):
            failures.append(f"node {rank}: kv dir outside the runtime root")
        port = node["control_endpoint"]["port"]
        if not in_block(port, LANE_CONTROL) or port != 23064 + rank:
            failures.append(f"node {rank}: control port {port}")
    if not in_block(deployment["transport"]["control_port_base"], LANE_TRANSPORT):
        failures.append("transport base outside the lane block")
    if deployment["eos_token_ids"] != [1]:
        failures.append(f"eos {deployment['eos_token_ids']} != [1] (family header)")
    if "socket_path" not in deployment.get("weightd", {}):
        failures.append("deployment missing weightd.socket_path")
    if deployment["weightd"]["socket_path"] != "/run/sparkpipe-weightd-shared/weightd.sock":
        failures.append(
            f"weightd socket {deployment['weightd']['socket_path']} is not "
            "the fleet-wide shared daemon path")

    # -- the wrapper, prepare mode ----------------------------------------
    root = os.path.join(tmp, "runtime")
    packs = os.path.join(tmp, "packs")
    os.makedirs(root)
    os.makedirs(packs)
    Path(packs, "rank0.spstage").write_bytes(b"pack-bytes-0")
    Path(packs, "rank1.spstage").write_bytes(b"pack-bytes-1")
    Path(packs, "rank1.spstage.sha256").write_text("0" * 64 + "  rank1.spstage\n")
    Path(packs, "rank0.spstage.sha256").write_text("a" * 64 + "  rank0.spstage\n")
    # rank 2 has a pack but no digest sidecar: the attach law fail-close
    Path(packs, "rank2.spstage").write_bytes(b"pack-bytes-2")

    base_env = dict(os.environ,
                    SPARK_QUEUE_RUNTIME_ROOT=os.path.join(root, "r1"),
                    SPARK_QUEUE_RANK="1",
                    SPARK_WEIGHTD_SOCKET="/tmp/shared-weightd.sock")
    result = run(["sh", str(WRAPPER), "prepare", tree, packs], env=base_env)
    if result.returncode != 0:
        failures.append(f"wrapper prepare failed: {result.stderr}")
    else:
        prepared = Path(root, "r1")
        digests = list((prepared / "packs").glob("*.sha256"))
        if len(digests) != 1:
            failures.append(f"wrapper left {len(digests)} digests, want exactly 1")
        if not (prepared / "config/stage.json").exists():
            failures.append("wrapper did not install config/stage.json")
        if not (prepared / "kvcache").is_dir():
            failures.append("wrapper did not create the kv backing directory")
        private = json.loads((prepared / "deployment.json").read_text())
        node1 = [n for n in private["nodes"] if n["rank_index"] == 1][0]
        if node1["runtime_root"] != str(prepared):
            failures.append("wrapper did not rewrite rank 1 runtime_root")
        if private["weightd"]["socket_path"] != "/tmp/shared-weightd.sock":
            failures.append("wrapper did not bind the shared socket")
        node0 = [n for n in private["nodes"] if n["rank_index"] == 0][0]
        if node0["runtime_root"] == str(prepared):
            failures.append("wrapper rewrote another rank's node entry")
        if json.loads((prepared / "config/stage.json").read_text())["tp_rank"] != 1:
            failures.append("wrapper installed the wrong rank's stage config")

    # fail-closed: missing sidecar for rank 2
    bad_env = dict(base_env, SPARK_QUEUE_RANK="2",
                   SPARK_QUEUE_RUNTIME_ROOT=os.path.join(root, "r2"))
    result = run(["sh", str(WRAPPER), "prepare", tree, packs], env=bad_env)
    if result.returncode != 4:
        failures.append(f"missing sidecar exit {result.returncode}, want 4")

    # fail-closed: missing environment
    result = run(["sh", str(WRAPPER), "prepare", tree, packs])
    if result.returncode != 3:
        failures.append(f"missing env exit {result.returncode}, want 3")

    # unrelated sidecars in the source pack dir do not leak into the tree
    Path(packs, "extra.spstage").write_bytes(b"x")
    Path(packs, "extra.spstage.sha256").write_text("b" * 64 + "  extra.spstage\n")
    result = run(["sh", str(WRAPPER), "prepare", tree, packs], env=base_env)
    if result.returncode != 0:
        failures.append(f"unrelated source sidecar broke prepare: {result.stderr}")

    # fail-closed: a stray second digest inside the prepared packs/ dir
    # (stale runtime-root reuse) violates the one-digest law
    stray = Path(base_env["SPARK_QUEUE_RUNTIME_ROOT"], "packs", "stale.sha256")
    stray.write_text("c" * 64 + "  stale.spstage\n")
    result = run(["sh", str(WRAPPER), "prepare", tree, packs], env=base_env)
    if result.returncode != 4:
        failures.append(f"stray digest exit {result.returncode}, want 4")
    stray.unlink()

    for failure in failures:
        print(f"FAIL: {failure}")
    print(f"{'PASS' if not failures else 'FAIL'}: dsv41_flash shared lane "
          f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

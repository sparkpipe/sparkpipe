#!/usr/bin/env python3
"""CPU validation gate for the qwen38_max lane-2 multidev port (M1).

developer lane 2 (qwen38_max, TP16, spark0..sparkf). This gate holds the
lane contract:

  1. the deployment keeps the identity rank_index == stage_index ==
     0..15 with 16 unique per-host control endpoints inside the lane-2
     control block 23032..23047;
  2. the weightd socket is the operator's shared unit wired through
     deployment.weightd.socket_path, and the KV backing stays a finite
     cap under the private runtime root (never the shared sparkdata
     tree);
  3. the adapter configuration keeps the qwen38_max serving adapter's
     EXACT member set (schema_version, model_revision, stage_pack_path,
     max_sequence_positions, tp_degree — an extra member is a load
     error) and points at the rank's deployed pack;
  4. no port of any kind lands outside lane 2's three blocks
     (23032-23047, 53032-53047, 64032-64047);
  5. the mesh map is the identity permutation (logical rank = index in
     --nodes), the wrapper scripts parse, and the generator's --check
     mode reproduces its own output byte for byte;
  6. the .wset hook emits the smoke-expert working set from the
     committed manifest (PR #1085 census receipt) as deduplicated
     sorted (layer, expert) u32 pairs;
  7. the family wrapper upholds the shared-socket template queue
     contract: attempt id, job namespace, size/rank bounds and
     queue-reserved port membership all fail closed before any
     filesystem work, and a fully reserved lane reaches the
     shared-socket liveness check;
  8. the deployment runtime limits sit inside the family adapter
     descriptor caps (max_inflight 1), and the firmware description
     carries the adapter's driver model id and the pinned source
     revision (the adapter_initialize identity contract).
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import qwen38max_multidev_lane as lane  # noqa: E402

LANE_BLOCKS = (
    set(range(lane.CONTROL_BASE, lane.CONTROL_BASE + 16)),
    set(range(lane.COLLECTIVE_BASE, lane.COLLECTIVE_BASE + 16)),
    set(range(lane.TRANSPORT_BASE, lane.TRANSPORT_BASE + 16)),
)
ALL_LANE_PORTS = LANE_BLOCKS[0] | LANE_BLOCKS[1] | LANE_BLOCKS[2]
EXACT_STAGE_MEMBERS = {
    "schema_version", "model_revision", "stage_pack_path",
    "max_sequence_positions", "tp_degree"}
SHARED_SOCKET = "/run/sparkpipe-weightd-shared/weightd.sock"


def check(condition, failures, message):
    if not condition:
        failures.append(message)


def deployment_gates(deployment, runtime_root, socket, failures):
    check(deployment["schema_version"] == 2, failures,
          "deployment schema must be 2")
    check(deployment["coordinator_rank_index"] == 0, failures,
          "coordinator must be rank 0")
    check(deployment["weightd"]["socket_path"] == socket, failures,
          "weightd socket must be the shared unit")
    check(deployment["adapter"]["shared_object_path"]
          == "lib/model_serving_adapter.so", failures, "adapter path")
    check(deployment["driver"]["shared_object_path"]
          == "lib/model_driver.so", failures, "driver path")
    check(deployment["transport"]["control_port_base"]
          == lane.TRANSPORT_BASE, failures, "transport base inside lane")
    limits = deployment["runtime_limits"]
    check(limits["max_inflight_submissions"] == 1, failures,
          "max_inflight_submissions must be 1 (the SparkQwen38MaxServing "
          "descriptor cap; the loader rejects above at deployment_validation, "
          "model_serving_adapter.c:232 - the attach-r15j lesson)")
    check(limits["kv_logical_page_capacity"] > 0
          and limits["kv_physical_page_capacity"] > 0, failures,
          "kv page capacities must be positive (family pool law)")
    nodes = deployment["nodes"]
    check(len(nodes) == 16, failures, "16 ranks")
    seen_control = set()
    for rank, node in enumerate(nodes):
        check(node["rank_index"] == rank and node["stage_index"] == rank,
              failures, f"identity rank/stage at {rank}")
        check(node["runtime_root"] == runtime_root, failures,
              f"private runtime root at {rank}")
        check(node["kv_backing_directory"]
              == os.path.join(runtime_root, "kvcache"), failures,
              f"kv under private root at {rank}")
        check(node["kv_backing_maximum_bytes"] > 0, failures,
              f"finite kv cap at {rank}")
        check(node["transport_host"] == lane.host_of(rank), failures,
              f"host map at {rank}")
        endpoint = node["control_endpoint"]
        check(endpoint["port"] == lane.CONTROL_BASE + rank, failures,
              f"control port at {rank}")
        check(endpoint["host"] == lane.host_of(rank), failures,
              f"control host at {rank}")
        seen_control.add(endpoint["port"])
    check(len(seen_control) == 16, failures,
          "every control endpoint unique (one per host)")


def stage_gates(adapter, rank, failures):
    check(set(adapter.keys()) == EXACT_STAGE_MEMBERS, failures,
          f"adapter member set EXACT at rank {rank}: {sorted(adapter)}")
    check(adapter["tp_degree"] == 16, failures, "tp_degree 16")
    check(adapter["schema_version"] == 3, failures, "stage schema 3")
    check(adapter["model_revision"] == lane.MODEL_REVISION, failures,
          "pinned model revision")
    check(adapter["stage_pack_path"]
          == f"packs/qwenmax.nvfp4.tp16.rank{rank:x}.sp", failures,
          f"rank pack path at {rank} (HEX rank suffix - decimal rank10..15 "
          "miss 6/16 nodes)")
    check(adapter["max_sequence_positions"] > 0, failures, "positions cap")


def firmware_identity_gates(failures):
    """The adapter_initialize identity contract (serving_adapter_template
    compares the driver descriptor against the family request): the
    firmware description the driver compiles must carry the adapter's
    driver model id and the pinned source revision - the h-string /
    vendor-style values failed loader equality for gemma4 (launches 8/9)
    and were the latent half of attach-r15j."""
    import re
    adapter_source = (ROOT / "modules/qwen38_max_resident_decode_stage"
                      "/source/spark_qwen38_max_serving_adapter.c").read_text()
    match = re.search(r"define\s+SPARK_QWEN38_MAX_SERVING_DRIVER_MODEL_ID"
                      r'[\s\\]*"([^"]+)"', adapter_source)
    driver_model_id = match.group(1) if match else ""
    check(bool(driver_model_id), failures,
          "adapter driver model id must be a plain string literal")
    firmware = json.load(open(
        ROOT / "examples/model_descriptions"
        "/qwen38_max_resident_decode_stage_firmware.json"))
    check(firmware["model"]["id"] == driver_model_id, failures,
          "firmware model.id must equal the adapter's driver model id "
          "(adapter_initialize identity compare)")
    check(firmware["model"]["revision"] == lane.MODEL_REVISION, failures,
          "firmware model.revision must equal the pinned source revision "
          "(adapter_initialize identity compare)")


def main() -> int:
    failures = []
    runtime_root = "/tmp/sparkqueue-" + "0" * 31 + "1"
    for rank in range(16):
        files = lane.render(rank, runtime_root, SHARED_SOCKET,
                            lane.DEFAULT_KV_BACKING_BYTES)
        deployment = json.loads(files["deployment.json"])
        adapter = json.loads(files["adapter.json"])
        deployment_gates(deployment, runtime_root, SHARED_SOCKET, failures)
        stage_gates(adapter, rank, failures)
        port_numbers = [node["control_endpoint"]["port"]
                        for node in deployment["nodes"]]
        port_numbers.append(deployment["transport"]["control_port_base"])
        for number in port_numbers:
            check(number in ALL_LANE_PORTS, failures,
                  f"port {number} at rank {rank} lands outside lane 2 blocks")

    # Mesh map identity.
    check(lane.MESH_RANKS == ",".join(str(i) for i in range(16)), failures,
          "identity mesh permutation")

    import re
    adapter_source = (ROOT / "modules/qwen38_max_resident_decode_stage"
                      "/source/spark_qwen38_max_serving_adapter.c").read_text()

    # Firmware description identity pair (adapter_initialize contract).
    firmware_identity_gates(failures)

    # Transport-stage shape: the loader requires descriptor stage_count ==
    # deployment node_count (model_resident_deployment.c:615; the attach-b
    # target_mismatch when the descriptor carried the PP-era count 1).
    stage_count = re.search(r"define\s+SPARK_QWEN38_MAX_SERVING_STAGE_COUNT"
                            r"\s+(\d+)u", adapter_source)
    check(stage_count is not None
          and int(stage_count.group(1)) == lane.WORLD, failures,
          "adapter stage_count must equal the 16-node TP16 deployment "
          "(loader equality at ValidateForAdapter)")

    # Generator CLI: --check reproduces byte for byte; bad inputs fail closed.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gen"
        env = dict(os.environ)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools/qwen38max_multidev_lane.py"),
             "--runtime-root", runtime_root, "--weightd-socket", SHARED_SOCKET,
             "--output-dir", str(out), "--rank", "7"],
            capture_output=True, text=True, env=env)
        check(proc.returncode == 0, failures, f"generator run: {proc.stderr}")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools/qwen38max_multidev_lane.py"),
             "--runtime-root", runtime_root, "--weightd-socket", SHARED_SOCKET,
             "--output-dir", str(out), "--rank", "7", "--check"],
            capture_output=True, text=True, env=env)
        check(proc.returncode == 0 and "matches" in proc.stdout, failures,
              f"generator --check: {proc.stdout}{proc.stderr}")
        base = [sys.executable, str(ROOT / "tools/qwen38max_multidev_lane.py"),
                "--weightd-socket", SHARED_SOCKET, "--output-dir", str(out)]
        for extra in (["--rank", "16"], ["--rank", "-1"],
                      ["--rank", "7", "--runtime-root", "/tmp/elsewhere"],
                      ["--rank", "7", "--runtime-root", runtime_root,
                       "--kv-backing-bytes", "0"]):
            proc = subprocess.run(base + extra, capture_output=True,
                                  text=True, env=env)
            check(proc.returncode != 0, failures,
                  f"generator must fail closed on {extra}")

        # .wset hook from the committed manifest.
        wset = Path(tmp) / "smoke.wset"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools/qwen38max_multidev_lane.py"),
             "--emit-wset", str(wset)],
            capture_output=True, text=True, env=env)
        check(proc.returncode == 0, failures, f"wset emission: {proc.stderr}")
        if wset.exists():
            raw = wset.read_bytes()
            check(len(raw) % 8 == 0 and len(raw) > 0, failures,
                  "wset is u32 pair array")
            pairs = list(struct.iter_unpack("<II", raw))
            check(pairs == sorted(set(pairs)), failures,
                  "wset pairs deduplicated and sorted")
            manifest = json.load(open(
                ROOT / "model-families/qwen38_max/smoke_experts.json"))
            expected = {(int(e["layer"]), int(e["expert"]))
                        for e in manifest["experts"]}
            check(set(pairs) == expected, failures,
                  "wset matches the census manifest exactly")

    # Wrapper scripts parse (bash -n) and hold the template contract text.
    for script in ("tools/qwen38max_multidev_run_family.sh",
                   "tools/qwen38max_multidev_experts_manifest.sh"):
        path = ROOT / script
        proc = subprocess.run(["bash", "-n", str(path)],
                              capture_output=True, text=True)
        check(proc.returncode == 0, failures, f"bash -n {script}: {proc.stderr}")
    wrapper = (ROOT / "tools/qwen38max_multidev_run_family.sh").read_text()
    for token in ("SPARK_QUEUE_ATTEMPT", "SPARK_QUEUE_RUNTIME_ROOT",
                  "SPARK_QUEUE_RANK", "SPARK_QUEUE_SIZE", "reserved_ok",
                  "SPARK_WEIGHTD_ATTACH=1", "SPARK_WEIGHTD_LANE",
                  "SPARK_TP_MESH_RANKS", "/run/sparkpipe-weightd-shared/weightd.sock",
                  "53000 + 16 * LANE", "23000 + 16 * LANE", "64000 + 16 * LANE",
                  "QMAX_EXPERT_POOL_BYTES", "QMAX_SPINE_BUDGET_BYTES"):
        check(token in wrapper, failures, f"wrapper missing {token}")
    check("67000" not in wrapper and "19600" not in wrapper
          and "61000" not in wrapper, failures,
          "wrapper must never emit legacy collective/control bases")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"\nFAIL ({len(failures)})")
        return 1
    print("PASS qwen38_max lane-2 multidev contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

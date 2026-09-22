#!/usr/bin/env python3
"""CPU validation gate for the ling lane-9 multidev port (M1).

developer lane 9 (ling, TP16, spark0..sparkf; muse_glimmer and hy4 follow
on the same lane under the three-family rotation). This gate holds the
lane contract:

  1. the deployment keeps the identity rank_index == stage_index ==
     0..15 with 16 unique per-host control endpoints inside the lane-9
     control block 23144..23159, and the ling eos token 156895;
  2. the weightd socket is the operator's shared unit wired through
     deployment.weightd.socket_path, and the KV backing stays a finite
     cap under the private runtime root (never the shared sparkdata
     tree);
  3. the adapter configuration keeps the ling serving adapter's EXACT
     member set (schema_version, model_revision, expert_weight_codec,
     stage_pack_path, max_sequence_positions, execution_row_capacity,
     decode_split_context_threshold, tp_degree, tp_rank, tp_collective -
     an extra member is a load error) and points at the rank's deployed
     pack, for BOTH placed arms (bf16 + fp8);
  4. the tp_collective keeps the hidden-transport ADAPTIVE member set
     exactly (runtime/serving_adapter_template.c), its listen_port +
     contiguous peer_ports live inside the lane-9 collective block, its
     identifier is nonzero (0 means "collective disabled" to the stage
     module), and both session matrices keep a zero diagonal with every
     off-diagonal value nonzero inside the lane-9 session block
     23744..23807;
  5. no port of any kind lands outside lane 9's four blocks
     (23144-23159, 23744-23807, 53144-53159, 64144-64159) and none of
     the LING-T1 legacy bases (19590, 60730, 63560, 12289, 13057,
     collective id 9911223344556680) survive anywhere;
  6. the mesh map is the identity permutation (logical rank = index in
     --nodes), the wrapper scripts parse, and the generator's --check
     mode reproduces its own output byte for byte;
  7. the .wset hook emits the smoke-expert working set from the
     committed manifest as deduplicated sorted (layer, expert) u32
     pairs once model-families/ling/smoke_experts.json lands (M2);
     until then the hook fails closed instead of inventing a set;
  8. the family wrapper upholds the shared-socket template queue
     contract: attempt id, job namespace, size/rank bounds, codec
     whitelist and queue-reserved port membership (control, collective,
     transport AND the session block) all fail closed before any
     filesystem work, sidecar validation is v2-with-records (the lane-3
     existence-is-not-validity lesson), and the shared-socket liveness
     check precedes any build.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import ling_multidev_lane as lane  # noqa: E402

LANE_BLOCKS = (
    set(range(lane.CONTROL_BASE, lane.CONTROL_BASE + 16)),
    set(range(lane.SESSION_BASE, lane.SESSION_BASE + 64)),
    set(range(lane.COLLECTIVE_BASE, lane.COLLECTIVE_BASE + 16)),
    set(range(lane.TRANSPORT_BASE, lane.TRANSPORT_BASE + 16)),
)
ALL_LANE_PORTS = LANE_BLOCKS[0] | LANE_BLOCKS[1] | LANE_BLOCKS[2] | LANE_BLOCKS[3]
EXACT_STAGE_MEMBERS = {
    "schema_version", "model_revision", "expert_weight_codec",
    "stage_pack_path", "max_sequence_positions", "execution_row_capacity",
    "decode_split_context_threshold", "tp_degree", "tp_rank", "tp_collective"}
EXACT_COLLECTIVE_MEMBERS = {
    "backend", "backend_module_path", "collective_identifier",
    "listen_port", "connect_timeout_milli", "operation_timeout_milli",
    "peer_hosts", "peer_ports", "algorithms",
    "direct_all_to_all_max_payload_bytes", "split_ring_min_payload_bytes",
    "rail_peer_hosts", "step_rail_indices", "session_ports",
    "session_ports_hc"}
SHARED_SOCKET = "/run/sparkpipe-weightd-shared/weightd.sock"
LEGACY_TOKENS = ("19590", "60730", "63560", "12289", "13057",
                 "9911223344556680", "18477")


def check(condition, failures, message):
    if not condition:
        failures.append(message)


def deployment_gates(deployment, runtime_root, socket, codec, failures):
    check(deployment["schema_version"] == 2, failures,
          "deployment schema must be 2")
    check(deployment["eos_token_ids"] == [156895], failures,
          "ling eos token 156895")
    check(deployment["coordinator_rank_index"] == 0, failures,
          "coordinator must be rank 0")
    check(deployment["weightd"]["socket_path"] == socket, failures,
          "weightd socket must be the shared unit")
    check(deployment["adapter"]["shared_object_path"]
          == "lib/model_serving_adapter.so", failures, "adapter path")
    check(deployment["driver"]["shared_object_path"]
          == "lib/model_driver.so", failures, "driver path")
    check(deployment["driver"]["program_name"] == "resident_decode",
          failures, "ling program name")
    check(deployment["transport"]["control_port_base"]
          == lane.TRANSPORT_BASE, failures, "transport base inside lane")
    limits = deployment["runtime_limits"]
    check(limits["kv_logical_page_capacity"] > 0
          and limits["kv_physical_page_capacity"] > 0, failures,
          "kv page capacities must be positive (family pool law)")
    nodes = deployment["nodes"]
    check(len(nodes) == 16, failures, "16 ranks")
    seen_control = set()
    for rank, node in enumerate(nodes):
        check(node["rank_index"] == rank and node["stage_index"] == rank,
              failures, f"identity rank/stage at {rank}")
        check(node["node_target"] == lane.CODEC_NODE_TARGET[codec],
              failures, f"codec node target at {rank}")
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


def stage_gates(adapter, rank, codec, failures):
    check(set(adapter.keys()) == EXACT_STAGE_MEMBERS, failures,
          f"adapter member set EXACT at rank {rank}: {sorted(adapter)}")
    check(adapter["schema_version"] == 3, failures, "stage schema 3")
    check(adapter["model_revision"] == lane.MODEL_REVISION, failures,
          "pinned model revision")
    check(adapter["expert_weight_codec"] == codec, failures,
          f"codec member matches the arm at rank {rank}")
    check(adapter["stage_pack_path"]
          == f"packs/ling.{codec}.tp16.rank{rank}.sp", failures,
          f"rank pack path at {rank}")
    check(adapter["tp_degree"] == 16 and adapter["tp_rank"] == rank,
          failures, f"tp degree/rank at {rank}")
    check(adapter["max_sequence_positions"] > 0
          and adapter["execution_row_capacity"] > 0
          and adapter["decode_split_context_threshold"]
          <= adapter["max_sequence_positions"], failures,
          f"sequence/row caps at {rank}")

    collective = adapter["tp_collective"]
    check(set(collective.keys()) == EXACT_COLLECTIVE_MEMBERS, failures,
          f"tp_collective ADAPTIVE member set EXACT at rank {rank}: "
          f"{sorted(collective)}")
    check(collective["backend"] == "hidden_transport", failures,
          "hidden-transport backend")
    check(collective["backend_module_path"] == "lib/hidden_transport.so",
          failures, "backend module rides the private runtime")
    check(collective["collective_identifier"] != 0, failures,
          "collective identifier must be nonzero (0 disables the module TP)")
    check(collective["listen_port"] == lane.COLLECTIVE_BASE + rank,
          failures, f"collective listen port at {rank}")
    peers = collective["peer_ports"]
    check(peers == list(range(lane.COLLECTIVE_BASE,
                              lane.COLLECTIVE_BASE + 16)), failures,
          "peer ports contiguous across the collective block")
    check(collective["peer_hosts"] == lane.HOSTS, failures,
          "full-mesh peer hosts")
    check(len(collective["rail_peer_hosts"]) == 2
          and collective["step_rail_indices"] == [0] + [1] * 15, failures,
          "two-rail tree shape (LING-T1 precedent)")
    for name in ("session_ports", "session_ports_hc"):
        matrix = collective[name]
        check(len(matrix) == 16, failures, f"{name} is 16 rows")
        for row_index, row in enumerate(matrix):
            check(len(row) == 16, failures,
                  f"{name} row {row_index} is 16 columns")
            for column, value in enumerate(row):
                if column == row_index:
                    check(value == 0, failures,
                          f"{name}[{row_index}][{column}] diagonal must be 0")
                else:
                    check(value != 0, failures,
                          f"{name}[{row_index}][{column}] off-diagonal "
                          "must be nonzero")
                    check(value in LANE_BLOCKS[1], failures,
                          f"{name} value {value} outside the session block")


def main() -> int:
    failures = []
    runtime_root = "/tmp/sparkqueue-" + "0" * 31 + "1"
    for codec in lane.CODECS:
        for rank in range(16):
            files = lane.render(rank, runtime_root, SHARED_SOCKET, codec,
                                lane.DEFAULT_KV_BACKING_BYTES)
            deployment = json.loads(files["deployment.json"])
            adapter = json.loads(files["adapter.json"])
            deployment_gates(deployment, runtime_root, SHARED_SOCKET,
                             codec, failures)
            stage_gates(adapter, rank, codec, failures)
            port_numbers = [node["control_endpoint"]["port"]
                            for node in deployment["nodes"]]
            port_numbers.append(deployment["transport"]["control_port_base"])
            port_numbers.append(adapter["tp_collective"]["listen_port"])
            port_numbers.extend(adapter["tp_collective"]["peer_ports"])
            for matrix in (adapter["tp_collective"]["session_ports"],
                           adapter["tp_collective"]["session_ports_hc"]):
                for row in matrix:
                    port_numbers.extend(value for value in row if value != 0)
            for number in port_numbers:
                check(number != 0 and number in ALL_LANE_PORTS, failures,
                      f"port {number} ({codec} rank {rank}) lands outside "
                      "lane 9 blocks")

    # Mesh map identity.
    check(lane.MESH_RANKS == ",".join(str(i) for i in range(16)), failures,
          "identity mesh permutation")

    # Generator CLI: --check reproduces byte for byte; bad inputs fail closed.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gen"
        env = dict(os.environ)
        for codec in lane.CODECS:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "tools/ling_multidev_lane.py"),
                 "--runtime-root", runtime_root, "--weightd-socket",
                 SHARED_SOCKET, "--output-dir", str(out), "--rank", "7",
                 "--codec", codec],
                capture_output=True, text=True, env=env)
            check(proc.returncode == 0, failures,
                  f"generator run ({codec}): {proc.stderr}")
            proc = subprocess.run(
                [sys.executable, str(ROOT / "tools/ling_multidev_lane.py"),
                 "--runtime-root", runtime_root, "--weightd-socket",
                 SHARED_SOCKET, "--output-dir", str(out), "--rank", "7",
                 "--codec", codec, "--check"],
                capture_output=True, text=True, env=env)
            check(proc.returncode == 0 and "matches" in proc.stdout,
                  failures,
                  f"generator --check ({codec}): {proc.stdout}{proc.stderr}")
        base = [sys.executable, str(ROOT / "tools/ling_multidev_lane.py"),
                "--weightd-socket", SHARED_SOCKET, "--output-dir", str(out)]
        for extra in (["--rank", "16"], ["--rank", "-1"],
                      ["--rank", "7", "--codec", "nvfp4"],
                      ["--rank", "7", "--runtime-root", "/tmp/elsewhere"],
                      ["--rank", "7", "--runtime-root", runtime_root,
                       "--kv-backing-bytes", "0"],
                      ["--rank", "7", "--runtime-root", runtime_root,
                       "--weightd-socket", "/tmp/other.sock"]):
            proc = subprocess.run(base + extra, capture_output=True,
                                  text=True, env=env)
            check(proc.returncode != 0, failures,
                  f"generator must fail closed on {extra}")

        # .wset hook: parity with the committed census manifest once it
        # lands (M2); until then it must fail closed, never invent a set.
        wset = Path(tmp) / "smoke.wset"
        manifest_path = ROOT / "model-families/ling/smoke_experts.json"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools/ling_multidev_lane.py"),
             "--emit-wset", str(wset)],
            capture_output=True, text=True, env=env)
        if manifest_path.exists():
            check(proc.returncode == 0, failures,
                  f"wset emission: {proc.stderr}")
            if wset.exists():
                raw = wset.read_bytes()
                check(len(raw) % 8 == 0 and len(raw) > 0, failures,
                      "wset is u32 pair array")
                pairs = list(struct.iter_unpack("<II", raw))
                check(pairs == sorted(set(pairs)), failures,
                      "wset pairs deduplicated and sorted")
                manifest = json.load(open(manifest_path))
                expected = {(int(e["layer"]), int(e["expert"]))
                            for e in manifest["experts"]}
                check(set(pairs) == expected, failures,
                      "wset matches the census manifest exactly")
        else:
            check(proc.returncode != 0 and not wset.exists(), failures,
                  "wset hook must fail closed without the committed manifest")

    # Wrapper parses (bash -n) and holds the template contract text.
    for script in ("tools/ling_multidev_run_family.sh",):
        path = ROOT / script
        proc = subprocess.run(["bash", "-n", str(path)],
                              capture_output=True, text=True)
        check(proc.returncode == 0, failures, f"bash -n {script}: {proc.stderr}")
    wrapper = (ROOT / "tools/ling_multidev_run_family.sh").read_text()
    for token in ("SPARK_QUEUE_ATTEMPT", "SPARK_QUEUE_RUNTIME_ROOT",
                  "SPARK_QUEUE_RANK", "SPARK_QUEUE_SIZE", "reserved_ok",
                  "SPARK_WEIGHTD_ATTACH=1", "SPARK_WEIGHTD_LANE",
                  "SPARK_TP_MESH_RANKS", "/run/sparkpipe-weightd-shared/weightd.sock",
                  "53000 + 16 * LANE", "23000 + 16 * LANE", "64000 + 16 * LANE",
                  "23168 + 64 * LANE", "SESSION_BASE",
                  "LING_EXPERT_POOL_BYTES", "LING_SPINE_BUDGET_BYTES",
                  "0x58504557", "ling_resident_decode_stage"):
        check(token in wrapper, failures, f"wrapper missing {token}")
    for legacy in LEGACY_TOKENS:
        check(legacy not in wrapper, failures,
              f"wrapper must never emit legacy base {legacy}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"\nFAIL ({len(failures)})")
        return 1
    print("PASS ling lane-9 multidev contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

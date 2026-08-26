#!/usr/bin/env python3
"""Regression: DSV4 Pro TP4xPP4 stage must fit exact-32K prompt + 256 output."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "examples" / "deployments" / "dsv4_pro_tp4_pp4_stage.json"
SPEC = ROOT / "examples" / "deployments" / "dsv4_pro_tp4_pp4_host_rdma.spec.json"


def main() -> int:
    stage = json.loads(STAGE.read_text(encoding="utf-8"))
    spec = json.loads(SPEC.read_text(encoding="utf-8"))

    ceiling = stage["max_sequence_positions"]
    assert isinstance(ceiling, int) and not isinstance(ceiling, bool)
    assert ceiling == 33024, ceiling

    # Capacity equation: prompt + output = 32768 + 256 = 33024.
    assert 32768 + 256 <= ceiling
    assert 33025 > ceiling, "33025 positions must exceed the declared ceiling"

    # Retained topology and transport fields unchanged.
    tp = stage["tp_collective"]
    assert stage["model_revision"] == "official Hugging Face config observed 2026-07-31"
    assert stage["stage_pack_path"] == "packs/dsv4_pro_tp4_pp4_stage.spstage"
    assert stage["cuda_graph_count_by_pp_stage"] == [49, 46, 46, 46]
    assert tp["backend"] == "hidden_transport"
    assert tp["backend_module_path"] == "lib/hidden_transport.so"
    assert tp["collective_identifier"] == 134217730
    assert tp["listen_port"] == 64620
    assert len(tp["peer_hosts"]) == 16 and len(tp["peer_ports"]) == 16
    assert tp["peer_ports"][0] == 64620 and tp["peer_ports"][-1] == 64635
    assert tp["algorithms"] == [
        "recursive_doubling", "direct_all_to_all", "counter_rotating_split_ring",
    ]
    assert len(tp["rail_peer_hosts"]) == 2
    assert all(len(rail) == 16 for rail in tp["rail_peer_hosts"])
    assert tp["step_rail_indices"] == [0, 1, 1]
    assert spec["topology"]["runtime_dataset"] == "dsv4_pro.tp4pp4"
    assert spec["transport"]["shared_object_path"] == (
        "lib/libhidden_transport_spark_host_rdma_verbs.so"
    )
    assert len(spec["topology"]["rank_hosts"]) == 16
    assert spec["topology"]["stage_indices"] == list(range(16))

    print("PASS dsv4_pro exact-32K stage capacity (33024 >= 32768 + 256)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

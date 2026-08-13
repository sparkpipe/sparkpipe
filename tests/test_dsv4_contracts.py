#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run(
        ["python3", str(ROOT / "tools" / "generate_dsv4_contracts.py"), "--check"],
        check=True,
        cwd=ROOT,
    )
    flash = json.loads((ROOT / "model_contracts" / "dsv4_flash.json").read_text(encoding="utf-8"))
    pro = json.loads((ROOT / "model_contracts" / "dsv4_pro.json").read_text(encoding="utf-8"))
    assert flash["model"]["layer_count"] == 43
    assert pro["model"]["layer_count"] == 61
    assert len(flash["attention"]["compression_ratios"]) == 46
    assert len(pro["attention"]["compression_ratios"]) == 62
    assert flash["attention"]["compression_ratios"][:2] == [0, 0]
    assert pro["attention"]["compression_ratios"][:2] == [128, 128]
    assert flash["attention"]["compression_ratios"][-3:] == [0, 0, 0]
    assert flash["model"]["mtp_layer_count"] == 0
    assert flash["dspark"]["layer_count"] == 3
    assert flash["runtime"]["packed_mtp_layer_count"] == 0
    assert flash["source_index_sha256"] == (
        "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b")
    source_files = flash["source_files"]
    shard_names = [
        f"model-{index:05d}-of-00048.safetensors" for index in range(1, 49)
    ]
    assert sorted(source_files) == sorted(
        ["model.safetensors.index.json"] + shard_names)
    assert len({source_files[name]["sha256"] for name in shard_names}) == 48
    assert all(len(source_files[name]["sha256"]) == 64 for name in source_files)
    assert sum(source_files[name]["bytes"] for name in shard_names) == 166886535336
    assert flash["source_indexed_payload_bytes"] == 166878536440
    assert flash["source_shard_count"] == 48
    header = (ROOT / "model-families" / "dsv4" / "include" / "sparkpipe" /
              "spark_dsv4_model.h").read_text(encoding="utf-8")
    assert "SparkDsv4ModelCompressionRatios[43u]" in header
    description = json.loads(
        (ROOT / "examples" / "model_descriptions" /
         "dsv4_resident_decode_stage_firmware.json").read_text(encoding="utf-8")
    )
    scheduling = description["stages"][0]["programs"][0]["scheduling"]
    assert scheduling["max_active_slots"] == 1024
    assert scheduling["max_resident_sequences"] == 16384
    b1_description = json.loads(
        (ROOT / "examples" / "model_descriptions" /
         "dsv4_resident_decode_stage_firmware_b1.json").read_text(
             encoding="utf-8")
    )
    b1_program = b1_description["stages"][0]["programs"][0]
    assert b1_program["max_inflight"] == 1
    assert b1_program["operations"][0]["module"].endswith(".b1.v3")
    assert b1_program["scheduling"]["max_active_slots"] == 1
    assert b1_program["scheduling"]["max_resident_sequences"] == 1
    b1_description_path = (
        ROOT / "examples" / "model_descriptions" /
        "dsv4_resident_decode_stage_firmware_b1.json")
    b1_digest = hashlib.sha256(b1_description_path.read_bytes()).hexdigest()
    assert f'#define SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B1 "{b1_digest}"' in header
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "DSV4_TP4_PP4_B1_SERVING_ADAPTER" in makefile
    assert "-USPARK_BATCH_BUCKET -DSPARK_BATCH_BUCKET=1u" in makefile
    tp4_pp4_spec = json.loads(
        (ROOT / "examples" / "deployments" /
         "dsv4_flash_tp4_pp4_b1_host_rdma.spec.json").read_text(
             encoding="utf-8")
    )
    assert tp4_pp4_spec["adapter"]["shared_object_path"] == (
        "lib/model_serving_adapter.so")
    assert tp4_pp4_spec["transport"]["shared_object_path"] == (
        "lib/hidden_transport.so")
    assert tp4_pp4_spec["runtime_limits"]["kv_logical_page_capacity"] == 32
    assert tp4_pp4_spec["runtime_limits"]["kv_physical_page_capacity"] == 32
    tp4_pp4_gpudirect_spec = json.loads(
        (ROOT / "examples" / "deployments" /
         "dsv4_flash_tp4_pp4_b1_gpudirect_rdma.spec.json").read_text(
             encoding="utf-8")
    )
    assert tp4_pp4_gpudirect_spec["transport"]["mode"] == "gpudirect-rdma"
    assert tp4_pp4_gpudirect_spec["topology"] == tp4_pp4_spec["topology"]
    assert tp4_pp4_gpudirect_spec["runtime_limits"] == (
        tp4_pp4_spec["runtime_limits"])
    tp4_pp4_stage = json.loads(
        (ROOT / "examples" / "deployments" /
         "dsv4_flash_tp4_pp4_stage.json").read_text(encoding="utf-8")
    )
    assert tp4_pp4_stage["tp_collective"]["connect_timeout_milli"] >= 120000
    assert tp4_pp4_stage["tp_collective"]["backend"] == "hidden_transport"
    assert tp4_pp4_stage["tp_collective"]["backend_module_path"] == (
        "lib/hidden_transport.so")
    assert tp4_pp4_stage["tp_collective"]["peer_hosts"] == [
        f"spark{index:x}-mgmt" for index in range(16)]
    assert tp4_pp4_stage["tp_collective"]["algorithms"] == [
        "recursive_doubling", "counter_rotating_split_ring"]
    assert tp4_pp4_stage["tp_collective"][
        "split_ring_min_payload_bytes"] == 640 * 1024
    assert tp4_pp4_stage["tp_collective"]["rail_peer_hosts"] == [
        [f"10.10.200.{index}" for index in range(16)],
        [f"10.10.100.{index + 10}" for index in range(16)],
    ]
    assert tp4_pp4_stage["tp_collective"]["step_rail_indices"] == [0, 1, 1]
    tp4_stage = json.loads(
        (ROOT / "examples" / "deployments" /
         "dsv4_flash_tp4_stage.json").read_text(encoding="utf-8")
    )
    assert tp4_stage["cuda_graph_count_by_pp_stage"] == [87]
    assert tp4_stage["tp_collective"]["backend"] == "hidden_transport"
    assert tp4_stage["tp_collective"]["peer_hosts"] == [
        f"spark{index:x}-mgmt" for index in range(4)]
    tp4_spec = json.loads(
        (ROOT / "examples" / "deployments" /
         "dsv4_flash_tp4_b1_host_rdma.spec.json").read_text(
             encoding="utf-8")
    )
    assert tp4_spec["topology"]["rank_hosts"] == [
        f"spark{index:x}" for index in range(4)]
    assert tp4_spec["topology"]["stage_indices"] == list(range(4))
    for release_template in ("dsv4_tp4_b1_template",
                             "dsv4_tp4_pp4_b1_template"):
        release = json.loads(
            (ROOT / "examples" / "release" / release_template /
             "sparkpipe.json").read_text(encoding="utf-8"))
        assert not any(entry["path"].endswith("libnccl.so.2")
                       for entry in release["files"])
        assert not any(value.startswith("NCCL_")
                       for value in release["roles"][0]["env"])
    tp16_release = json.loads(
        (ROOT / "examples" / "release" / "dsv4_tp16_b1_template" /
         "sparkpipe.json").read_text(encoding="utf-8"))
    assert "NCCL_IB_GID_INDEX=0" in tp16_release["roles"][0]["env"]
    tp4_release = json.loads(
        (ROOT / "examples" / "release" / "dsv4_tp4_b1_template" /
         "sparkpipe.json").read_text(encoding="utf-8"))
    assert tp4_release["rank_count"] == 4
    assert tp4_release["max_active_sequence_count"] == 1
    assert not any(value.startswith("NCCL_IB_HCA=") or
                   value.startswith("NCCL_IB_GID_INDEX=") or
                   value.startswith("NCCL_SOCKET_IFNAME=")
                   for value in tp4_release["roles"][0]["env"])
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "DSV4_TP4_B1_SERVING_ADAPTER" in makefile
    stage_source = (
        ROOT / "modules" / "dsv4_resident_decode_stage" / "source" /
        "spark_dsv4_resident_decode_stage_module.c").read_text(
            encoding="utf-8")
    assert re.search(
        r"if \( state->tp_credit_binding_count != 0u \)\s*\{\s*"
        r"configuration\.credit_bindings = state->tp_credit_bindings;\s*"
        r"configuration\.credit_binding_count = "
        r"state->tp_credit_binding_count;\s*\}",
        stage_source,
    )
    assert "SparkTpDeviceCollectiveApplyTopology(" in stage_source
    adapter_source = (
        ROOT / "modules" / "dsv4_resident_decode_stage" / "source" /
        "spark_dsv4_serving_adapter.c").read_text(encoding="utf-8")
    assert "SparkTpDeviceCollectiveSliceTopology(" in adapter_source
    deployment = json.loads(
        (ROOT / "examples" / "deployments" /
         "dsv4_flash_pp13_host_rdma.spec.json").read_text(encoding="utf-8")
    )
    limits = deployment["runtime_limits"]
    assert scheduling["max_active_slots"] >= limits["max_active_sequences"]
    assert scheduling["max_resident_sequences"] >= limits["resident_sequence_capacity"]
    assert pro["attention"]["compression_ratios"][-1] == 0
    assert flash["qualification"]["cuda_target"] == "sm_121a"
    assert pro["qualification"]["cuda_target"] == "sm_121a"
    print("PASS DSV4 Flash and Pro generated contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Gate the single-source serving-profile derivation.

Red cases reproduce the #1210 drift incident (each hand-pinned value that
booted wrong or served wrong); green cases pin the B1 and B8 derivations
and the verify() drift detector.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import spark_serving_profile as sp  # noqa: E402


def main() -> int:
    checks = 0

    def check(name, condition):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(f"FAILED: {name}")
        print(f"ok   {name}")

    b1 = sp.profile("B1")
    b8 = sp.profile("B8")

    # B1 derivation matches the qualified station profile (the working set).
    check("B1 sequences", b1["max_active_sequences"] == 1)
    check("B1 kv pages 128", b1["kv_logical_page_capacity"] == 128)
    check("B1 backing 2GiB", b1["kv_backing_maximum_bytes"] == 2147483648)
    check("B1 input rows are the wave width, not the sequence count", b1["max_input_rows"] == sp.PROFILE_ROW_CAPACITY)
    check("B1 row capacity = profile row capacity under the mesh cap", b1["execution_row_capacity"] == sp.PROFILE_ROW_CAPACITY == 128 and sp.PROFILE_ROW_CAPACITY <= sp.MESH_MAX_BATCH_ROWS)

    # B8 derivation equals the map recovered in #1210.
    check("B8 sequences", b8["max_active_sequences"] == 8)
    check("B8 resident capacity", b8["resident_sequence_capacity"] == 8)
    check("B8 kv pages 1024", b8["kv_logical_page_capacity"] == 1024)
    check("B8 backing 16GiB", b8["kv_backing_maximum_bytes"] == 17179869184)
    check("B8 input rows >= sequences", b8["max_input_rows"] >= 8)
    check("B8 row capacity = profile row capacity", b8["execution_row_capacity"] == 128)

    check("input rows >= sequences always",
          all(sp.derive(n)["max_input_rows"] >= n for n in range(1, 17)))
    check("a wide prefill wave is kept for any sequence count",
          all(sp.derive(n, 32768, rows=1024)["max_input_rows"] == 1024 == sp.derive(n, 32768, rows=1024)["execution_row_capacity"]
              for n in (1, 2, 8, 16)))
    for rows, sequences in ((8, 16), (sp.MESH_MAX_BATCH_ROWS + 1, 1), (0, 1)):
        try:
            sp.derive(sequences, 512, rows=rows)
            check(f"a {rows}-row wave for {sequences} sequences is refused", False)
        except ValueError:
            check(f"a {rows}-row wave for {sequences} sequences is refused", True)
    check("physical <= logical pages always",
          all(sp.derive(n)["kv_physical_page_capacity"] <= sp.derive(n)["kv_logical_page_capacity"]
              for n in range(1, 17)))

    # A recurrent model keeps one state record per logical page in its KV
    # backing; the binding refuses a deployment whose backing cannot hold
    # them, so the derivation caps the pages by the backing budget.
    budget = 137438953472
    record = 9330688
    capped = sp.derive(16, 32768, record, budget)
    check("recurrent pages capped by the backing budget",
          capped["kv_logical_page_capacity"] == capped["kv_physical_page_capacity"] == (budget - 2 * sp.KV_PAGE_BYTES) // record)
    check("recurrent pages still cover every resident sequence",
          capped["kv_logical_page_capacity"] >= 16 * 32768 // sp.KV_PAGE_TOKENS)
    check("recurrent budget fits the binding rule",
          capped["kv_logical_page_capacity"] * record + 2 * sp.KV_PAGE_BYTES <= budget)
    try:
        sp.derive(16, 32768, record, 16 * 32768 // sp.KV_PAGE_TOKENS * record - 1)
        check("a budget below the resident sequences is refused", False)
    except ValueError:
        check("a budget below the resident sequences is refused", True)
    check("a model without recurrent state keeps the pool multiple",
          sp.derive(16, 32768)["kv_logical_page_capacity"] == sp.KV_POOL_PAGES_PER_RESIDENT_PAGE * 16 * 32768 // sp.KV_PAGE_TOKENS)

    page = 67672
    stride = 4096
    strided = sp.derive(16, 262144, record, budget, rows=1024, checkpoint_tokens=stride, page_bytes=page)
    resident = 16 * 262144 // sp.KV_PAGE_TOKENS
    spill = strided["kv_logical_page_capacity"] - strided["kv_physical_page_capacity"]
    check("a strided recurrent model keeps every resident page on the device",
          strided["kv_physical_page_capacity"] == resident)
    check("a strided recurrent model is not capped at one state record per logical page",
          strided["kv_logical_page_capacity"] > (budget - 2 * sp.KV_PAGE_BYTES) // record)
    check("the logical pool stays within the pool multiple",
          strided["kv_logical_page_capacity"] <= sp.KV_POOL_PAGES_PER_RESIDENT_PAGE * resident)
    check("spilled pages plus every checkpoint of every full-length sequence fit the backing",
          (spill + sp.KV_BACKING_IN_FLIGHT_PAGES) * page + sp.checkpoint_slots(16, 262144, stride) * record <= strided["kv_backing_maximum_bytes"] == budget)
    small = sp.checkpoint_slots(16, 262144, stride) * record + 10 * page
    tight = sp.derive(16, 262144, record, small, rows=1024, checkpoint_tokens=stride, page_bytes=page)
    check("a tight backing spills only what it can hold",
          tight["kv_logical_page_capacity"] - tight["kv_physical_page_capacity"] == 10 - sp.KV_BACKING_IN_FLIGHT_PAGES)
    try:
        sp.derive(16, 262144, record, sp.checkpoint_slots(16, 262144, stride) * record - 1, rows=1024, checkpoint_tokens=stride, page_bytes=page)
        check("a backing below the checkpoints of the resident sequences is refused", False)
    except ValueError:
        check("a backing below the checkpoints of the resident sequences is refused", True)
    for name, recurrent, page_size, checkpoint in (("without a record size", 0, page, stride),
                                                  ("without a page size", record, 0, stride),
                                                  ("off the page grain", record, page, stride + 1)):
        try:
            sp.derive(16, 262144, recurrent, budget, rows=1024, checkpoint_tokens=checkpoint, page_bytes=page_size)
            check(f"a checkpoint stride {name} is refused", False)
        except ValueError:
            check(f"a checkpoint stride {name} is refused", True)

    sys.path.insert(0, str(ROOT / "tools"))
    import glm5_next_gen_deployment as flash
    committed = json.loads((ROOT / "deployment/glm5_next_tp16/model_resident.json").read_text())
    limits = committed["runtime_limits"]
    flash_spill = limits["kv_logical_page_capacity"] - limits["kv_physical_page_capacity"]
    flash_slots = sp.checkpoint_slots(limits["max_active_sequences"], limits["max_sequence_positions"], flash.kv_geometry_constant("CHECKPOINT_TOKENS"))
    check("the GLM Flash page bytes match the binding's measured 67672-byte TP16 page", flash.kv_page_bytes() == page)
    check("the committed GLM Flash deployment fits its spilled pages and checkpoints in every node's backing",
          flash.recurrent_page_bytes() == record and
          all((flash_spill + sp.KV_BACKING_IN_FLIGHT_PAGES) * flash.kv_page_bytes() + flash_slots * flash.recurrent_page_bytes()
              <= node["kv_backing_maximum_bytes"] for node in committed["nodes"]))
    check("the committed GLM Flash deployment prefills in mesh-wide waves",
          limits["max_input_rows"] == sp.MESH_MAX_BATCH_ROWS)

    # Drift detector: the incident's actual drifted configs must be caught.
    drifted_limits = dict(b8)
    drifted_limits["kv_logical_page_capacity"] = 128  # hand-pinned leftover
    drifted_stage = {"max_sequence_positions": 512, "execution_row_capacity": 1}
    findings = sp.verify(drifted_limits, drifted_stage)
    check("drift detected (kv pages)", any("kv_logical" in f for f in findings))
    check("drift detected (row capacity)", any("execution_row_capacity" in f for f in findings))
    check("clean config passes", sp.verify(dict(b1), {
        "max_sequence_positions": 512, "execution_row_capacity": b1["execution_row_capacity"]}) == [])

    # CLI smoke: verify mode exits 1 on drift.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        dep = pathlib.Path(td) / "deployment.json"
        dep.write_text(json.dumps({"runtime_limits": drifted_limits}))
        result = subprocess.run([sys.executable, str(ROOT / "tools/spark_serving_profile.py"),
                                 "--verify-deployment", str(dep)], capture_output=True, text=True)
        check("CLI flags drift", result.returncode == 1 and "DRIFT" in result.stdout)

    print(f"PASS {checks} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

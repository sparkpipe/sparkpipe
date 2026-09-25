#!/usr/bin/env python3
"""Gate the single-source serving-profile derivation.

Red cases reproduce the #1210 drift incident (each hand-pinned value that
booted wrong or served wrong); green cases pin the B1 and B8 derivations
and the verify() drift detector.
"""
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
    check("B1 input rows 1", b1["max_input_rows"] == 1)
    check("B1 row capacity = mesh cap", b1["execution_row_capacity"] == 128)

    # B8 derivation equals the map recovered in #1210.
    check("B8 sequences", b8["max_active_sequences"] == 8)
    check("B8 resident capacity", b8["resident_sequence_capacity"] == 8)
    check("B8 kv pages 1024", b8["kv_logical_page_capacity"] == 1024)
    check("B8 backing 16GiB", b8["kv_backing_maximum_bytes"] == 17179869184)
    check("B8 input rows >= sequences", b8["max_input_rows"] >= 8)
    check("B8 row capacity mesh-capped", b8["execution_row_capacity"] == 128)

    # Invariants the validators enforce must hold by construction.
    check("input rows >= sequences always",
          all(sp.derive(n)["max_input_rows"] >= n for n in range(1, 17)))
    check("physical <= logical pages always",
          all(sp.derive(n)["kv_physical_page_capacity"] <= sp.derive(n)["kv_logical_page_capacity"]
              for n in range(1, 17)))

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
    import json, tempfile
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

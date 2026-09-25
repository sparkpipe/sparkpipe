#!/usr/bin/env python3
"""Single-source derivation of every serving-profile constant.

One seed — MAX_ACTIVE_SEQUENCES (with MAX_SEQUENCE_POSITIONS and the
compile-time ceilings) — derives every dependent constant. Hand-pinning
any dependent value in a deployment/stage config is drift; incident
#1210 found six such constants drifted across three files. All profile
emitters must go through derive(); verify() rejects drifted configs.

Compile-time ceilings (keep in sync with the C headers; verify() checks
the derived values against them):
  SPARK_WEIGHTD_MESH_MAX_BATCH_ROWS = 128   (include/sparkpipe/spark_weightd.h)
  SPARK_GLM5_NEXT_..._MAX_INPUT_ROW_COUNT = 65536 (module firmware)
  KV page granularity = 64 positions/page
"""
from __future__ import annotations

MESH_MAX_BATCH_ROWS = 128
FIRMWARE_MAX_INPUT_ROWS = 65536
KV_PAGE_TOKENS = 64
# 16 rank-pages per sequence-page (the multi-rank KV layout; module contract).
KV_RANK_PAGES_PER_PAGE = 16
# Bytes per KV page (payload + recurrent state; module formula — the
# module recomputes this and rejects a smaller backing budget).
KV_PAGE_BYTES = 2 * 1024 * 1024
# Backing headroom over the strict pages*page-bytes minimum: the qualified
# B1 profile shipped 2 GiB for a 128-page (256 MiB strict) budget. The
# module only enforces the minimum; the factor preserves the working
# profile's headroom so regenerated configs match byte-for-byte.
KV_BACKING_HEADROOM = 8

PROFILES = {
    # name: (max_active_sequences, max_sequence_positions)
    "B1": (1, 512),
    "B8": (8, 512),
}


def derive(sequences: int, positions: int = 512) -> dict:
    if sequences < 1 or positions < 1:
        raise ValueError("sequences and positions must be positive")
    pages_per_sequence = (positions + KV_PAGE_TOKENS - 1) // KV_PAGE_TOKENS
    kv_pages = KV_RANK_PAGES_PER_PAGE * sequences * pages_per_sequence
    row_capacity = min(MESH_MAX_BATCH_ROWS, FIRMWARE_MAX_INPUT_ROWS)
    return {
        "max_active_sequences": sequences,
        "resident_sequence_capacity": sequences,
        "max_inflight_submissions": min(4, sequences),
        # Validator invariant: max_input_rows >= sequences; prefill waves
        # ride the mesh, so the mesh batch cap is the ceiling. The
        # multi-chunk prefill continuation bug (#1210) forces 1 until the
        # engine fix lands; derived, not hand-pinned.
        "max_input_rows": 1 if sequences == 1 else min(sequences, row_capacity),
        "kv_logical_page_capacity": kv_pages,
        "kv_physical_page_capacity": kv_pages,
        "kv_backing_maximum_bytes": kv_pages * KV_PAGE_BYTES * KV_BACKING_HEADROOM,
        "execution_row_capacity": row_capacity,
        "max_sequence_positions": positions,
    }


def profile(name: str) -> dict:
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; known: {sorted(PROFILES)}")
    sequences, positions = PROFILES[name]
    return derive(sequences, positions)


def verify(runtime_limits: dict, stage_config: dict, nodes_backing=None) -> list:
    """Return a list of drift findings ([] = consistent)."""
    positions = stage_config.get("max_sequence_positions") or 512
    try:
        want = derive(runtime_limits.get("max_active_sequences", 0), positions)
    except ValueError as error:
        return [f"cannot derive a consistent profile: {error}"]
    got = dict(runtime_limits)
    got["execution_row_capacity"] = stage_config.get("execution_row_capacity", 0)
    got["max_sequence_positions"] = stage_config.get("max_sequence_positions", 0)
    # kv_backing_maximum_bytes lives per-node in the deployment, not in
    # runtime_limits; accept it from either position.
    if got.get("kv_backing_maximum_bytes") is None and nodes_backing is not None:
        got["kv_backing_maximum_bytes"] = nodes_backing
    findings = []
    for key, expected in want.items():
        actual = got.get(key)
        if actual != expected:
            findings.append(f"{key}: config has {actual}, derivation requires {expected}")
    return findings


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument("--verify-deployment", type=str,
                        help="path to deployment.json; also needs --verify-stage")
    parser.add_argument("--verify-stage", type=str)
    args = parser.parse_args()
    if args.verify_deployment:
        deployment = json.load(open(args.verify_deployment))
        stage = json.load(open(args.verify_stage)) if args.verify_stage else {}
        nodes = deployment.get("nodes") or []
        backing = nodes[0].get("kv_backing_maximum_bytes") if nodes else None
        findings = verify(deployment.get("runtime_limits", {}), stage, backing)
        if findings:
            for f in findings:
                print(f"DRIFT: {f}")
            raise SystemExit(1)
        print("profile consistent")
    elif args.profile:
        print(json.dumps(profile(args.profile), indent=2))
    else:
        parser.error("--profile or --verify-deployment required")

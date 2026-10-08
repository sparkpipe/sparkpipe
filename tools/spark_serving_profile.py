#!/usr/bin/env python3
"""Single-source derivation of every serving-profile constant.

One seed — MAX_ACTIVE_SEQUENCES (with MAX_SEQUENCE_POSITIONS and the
compile-time ceilings) — derives every dependent constant. Hand-pinning
any dependent value in a deployment/stage config is drift; incident
#1210 found six such constants drifted across three files. All profile
emitters must go through derive(); verify() rejects drifted configs.

Compile-time ceilings (keep in sync with the C headers; verify() checks
the derived values against them):
  SPARK_WEIGHTD_MESH_MAX_BATCH_ROWS = 1024  (include/sparkpipe/spark_weightd.h)
  SPARK_GLM5_NEXT_..._MAX_INPUT_ROW_COUNT = 65536 (module firmware)
  KV page granularity = 64 positions/page
"""
from __future__ import annotations

MESH_MAX_BATCH_ROWS = 1024
PROFILE_ROW_CAPACITY = 128
FIRMWARE_MAX_INPUT_ROWS = 65536
KV_PAGE_TOKENS = 64
KV_POOL_PAGES_PER_RESIDENT_PAGE = 16
# Bytes per KV page (payload + recurrent state; module formula — the
# module recomputes this and rejects a smaller backing budget).
KV_PAGE_BYTES = 2 * 1024 * 1024
# Backing headroom over the strict pages*page-bytes minimum: the qualified
# B1 profile shipped 2 GiB for a 128-page (256 MiB strict) budget. The
# module only enforces the minimum; the factor preserves the working
# profile's headroom so regenerated configs match byte-for-byte.
KV_BACKING_HEADROOM = 8

# The runtime_limits members runtime/model_resident_deployment.c accepts;
# the other derived values live in the stage config
# (execution_row_capacity) and on each node (kv_backing_maximum_bytes).
DEPLOYMENT_RUNTIME_MEMBERS = (
    "max_active_sequences",
    "resident_sequence_capacity",
    "max_inflight_submissions",
    "max_input_rows",
    "kv_logical_page_capacity",
    "kv_physical_page_capacity",
    "max_sequence_positions",
)

PROFILES = {
    # name: (max_active_sequences, max_sequence_positions)
    "B1": (1, 512),
    "B8": (8, 512),
}


KV_BACKING_IN_FLIGHT_PAGES = 2


def checkpoint_slots(sequences: int, positions: int, checkpoint_tokens: int) -> int:
    return sequences * ((positions + checkpoint_tokens - 1) // checkpoint_tokens + 1)


def derive(sequences: int, positions: int = 512, recurrent_page_bytes: int = 0, backing_bytes: int = 0,
           rows: int = PROFILE_ROW_CAPACITY, checkpoint_tokens: int = 0, page_bytes: int = 0) -> dict:
    if sequences < 1 or positions < 1:
        raise ValueError("sequences and positions must be positive")
    if rows < sequences or rows > min(MESH_MAX_BATCH_ROWS, FIRMWARE_MAX_INPUT_ROWS):
        raise ValueError(f"a {rows}-row wave must hold every one of {sequences} sequences and fit the "
                         f"{min(MESH_MAX_BATCH_ROWS, FIRMWARE_MAX_INPUT_ROWS)}-row mesh batch")
    pages_per_sequence = (positions + KV_PAGE_TOKENS - 1) // KV_PAGE_TOKENS
    resident_pages = sequences * pages_per_sequence
    kv_pages = KV_POOL_PAGES_PER_RESIDENT_PAGE * resident_pages
    physical_pages = kv_pages
    backing = kv_pages * KV_PAGE_BYTES * KV_BACKING_HEADROOM
    if checkpoint_tokens:
        if not recurrent_page_bytes or not page_bytes or checkpoint_tokens % KV_PAGE_TOKENS:
            raise ValueError("a checkpoint stride needs the recurrent record bytes, the KV page bytes and a whole number of pages")
        state_bytes = checkpoint_slots(sequences, positions, checkpoint_tokens) * recurrent_page_bytes
        spill_pages = (backing_bytes - state_bytes) // page_bytes - KV_BACKING_IN_FLIGHT_PAGES if backing_bytes > state_bytes else -1
        if spill_pages < 0:
            raise ValueError(f"a {backing_bytes}-byte KV backing budget cannot hold the {state_bytes} bytes of recurrent "
                             f"checkpoints that {sequences} sequences of {positions} positions keep")
        physical_pages = resident_pages
        kv_pages = resident_pages + min(spill_pages, kv_pages - resident_pages)
        backing = backing_bytes
    elif recurrent_page_bytes:
        state_pages = (backing_bytes - 2 * KV_PAGE_BYTES) // recurrent_page_bytes if backing_bytes > 2 * KV_PAGE_BYTES else 0
        if state_pages < resident_pages:
            raise ValueError(f"a {backing_bytes}-byte KV backing budget holds the {recurrent_page_bytes}-byte recurrent "
                             f"record of {state_pages} pages; the resident sequences need {resident_pages}")
        kv_pages = min(kv_pages, state_pages)
        physical_pages = kv_pages
        backing = kv_pages * KV_PAGE_BYTES * KV_BACKING_HEADROOM
    return {
        "max_active_sequences": sequences,
        "resident_sequence_capacity": sequences,
        "max_inflight_submissions": min(4, sequences),
        "max_input_rows": rows,
        "kv_logical_page_capacity": kv_pages,
        "kv_physical_page_capacity": physical_pages,
        "kv_partition": "/",
        "kv_backing_maximum_bytes": backing,
        "execution_row_capacity": rows,
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
        want = derive(runtime_limits.get("max_active_sequences", 0), positions,
                      rows=runtime_limits.get("max_input_rows", PROFILE_ROW_CAPACITY))
    except ValueError as error:
        return [f"cannot derive a consistent profile: {error}"]
    got = dict(runtime_limits)
    got["execution_row_capacity"] = stage_config.get("execution_row_capacity", 0)
    got["max_sequence_positions"] = stage_config.get("max_sequence_positions", 0)
    # kv_backing_maximum_bytes lives per-node in the deployment, not in
    # runtime_limits; accept it from either position.
    if got.get("kv_backing_maximum_bytes") is None and nodes_backing is not None:
        got["kv_backing_maximum_bytes"] = nodes_backing
    if got.get("kv_backing_maximum_bytes") == 0:
        want.pop("kv_backing_maximum_bytes")
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

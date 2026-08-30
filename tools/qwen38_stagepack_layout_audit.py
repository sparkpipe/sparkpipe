#!/usr/bin/env python3
"""CPU audit: is the qwen38_max stagepack wire layout ACCURATE?

The pack format exists in three places that must agree byte-for-byte:
the C family header (loaded by the module), the Python packer's
struct layout (writes packs), and the Python verifier's layout (reads
them). This tool compiles tools/qwen38_stagepack_layout_probe.c against
a chosen C header and compares the probe's offsetof table against a
chosen Python header layout, field by field. Exit 0 = accurate (the two
sides agree on every field offset and size); exit 1 = mismatch, with the
divergent fields named.

No GPU is involved: the stagepack contract is checkable entirely on the
build host, which is the point -- a layout drift must be a red stop
BEFORE any multi-hundred-GiB pack is built from a checkpoint.

Usage:
  qwen38_stagepack_layout_audit.py --probe-bin PROBE \\
      --python-layout {v1,v2-as-built,v2-tail} [--label LABEL]

The probe binary is built once (see the report or the Makefile snippet
in docs/AGENT_LANE_BRIEFS/reports/qwen38max-v2-cpu-audit-2026-08-30.md):
  cc -std=c11 -o /tmp/probe tools/qwen38_stagepack_layout_probe.c \\
     -Imodules/qwen38_max_resident_decode_stage/source \\
     -Imodules/qwen38_max_resident_decode_stage/include \\
     -Imodel-families/qwen38_max/include -Iinclude
"""
import argparse
import subprocess
import sys

# The three layouts that have existed on the branch graph, expressed as
# (field, struct-format) lists exactly as the Python packers define them.
PY_V1 = [  # main's family packer today: 120-byte v1 header
    ("magic", "I"), ("format_version", "I"), ("header_bytes", "I"),
    ("directory_entry_bytes", "I"), ("tensor_count", "I"),
    ("hidden_dimension", "I"), ("layer_count", "I"),
    ("first_layer_index", "I"), ("total_layer_count", "I"),
    ("attention_period", "I"), ("full_attention_phase", "I"),
    ("gdn_key_head_count", "I"), ("gdn_value_head_count", "I"),
    ("gdn_head_key_dimension", "I"), ("gdn_head_value_dimension", "I"),
    ("gdn_conv_kernel", "I"), ("attn_query_head_count", "I"),
    ("attn_kv_head_count", "I"), ("attn_head_dimension", "I"),
    ("attn_rope_dimension", "I"), ("routed_expert_count", "I"),
    ("experts_per_token", "I"), ("expert_intermediate_dimension", "I"),
    ("output_vocab_count", "I"), ("mxfp4_group_size", "I"),
    ("mtp_layer_count", "I"), ("directory_offset", "Q"),
    ("file_bytes", "Q"),
]
PY_V2_AS_BUILT = [  # the lane's v2 python packer: tp fields before the u64s
    *PY_V1[:-2], ("tp_degree", "I"), ("tp_rank", "I"),
    ("directory_offset", "Q"), ("file_bytes", "Q"),
]
PY_V2_TAIL = [  # the rebased C header: 120-byte common prefix + 8-byte TP tail
    *PY_V1, ("tp_degree", "I"), ("tp_rank", "I"),
]
PY_LAYOUTS = {
    "v1": PY_V1,
    "v2-as-built": PY_V2_AS_BUILT,
    "v2-tail": PY_V2_TAIL,
}


def python_offsets(layout):
    """Field -> (offset, size) with C natural-alignment rules (LP64)."""
    offsets = {}
    cursor = 0
    for name, code in layout:
        size = 8 if code == "Q" else 4
        align = size
        cursor = (cursor + align - 1) & ~(align - 1)
        offsets[name] = (cursor, size)
        cursor += size
    return offsets, cursor


def probe_offsets(probe_bin):
    """Field -> (offset, size) from the compiled C probe's output."""
    out = subprocess.run([probe_bin], capture_output=True, text=True, check=True)
    meta = {}
    offsets = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts[0] == "probe" and parts[1] in ("format_version",):
            meta["format_version"] = int(parts[2])
        elif parts[0] == "header":
            offsets[parts[1]] = (int(parts[2]), int(parts[3]))
    return meta, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-bin", required=True)
    ap.add_argument("--python-layout", choices=sorted(PY_LAYOUTS), required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    meta, c_offsets = probe_offsets(args.probe_bin)
    p_offsets, py_total = python_offsets(PY_LAYOUTS[args.python_layout])

    c_version = meta.get("format_version")
    label = args.label or f"C header vs python {args.python_layout}"
    print(f"== {label}")
    print(f"   C format_version={c_version} fields={len(c_offsets)}; "
          f"python layout={args.python_layout} fields={len(p_offsets)}")

    bad = []
    for name, (off, size) in sorted(p_offsets.items(), key=lambda kv: kv[1]):
        c = c_offsets.get(name)
        if c is None:
            bad.append((name, "absent in C header", (off, size)))
        elif c != (off, size):
            bad.append((name, f"C@{c[0]} size {c[1]}", f"py@{off} size {size}"))
    for name in sorted(set(c_offsets) - set(p_offsets)):
        bad.append((name, f"C@{c_offsets[name][0]}", "absent in python layout"))

    if bad:
        print(f"   VERDICT: INACCURATE -- {len(bad)} divergent field(s):")
        for name, c_side, p_side in bad:
            print(f"     {name}: C {c_side} vs python {p_side}")
        return 1
    print("   VERDICT: accurate -- every field offset and size agrees.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

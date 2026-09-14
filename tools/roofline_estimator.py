#!/usr/bin/env python3
"""SparkPipe decode roofline estimator (canonical).

Parses stagepack wire layouts (magics, field order, tensor-kind numbering)
straight from the repo's format headers, reads per-arm pack manifests +
receipts read-only from the nodes, parses /Users/mac/sparkpipe-coord/
MEMORY_MODEL.md rows, and emits the per-lane decode roofline:

    bytes/token/rank = embedding row + spine stream/B + lm head/B
                     + touched experts + KV read/append + GDN state R/W
                     + activation round-trips

    % of roofline = receipted tok/s * analytic bytes/token / MEASURED BW

MEASURED BW is the sparkcap stream probe (tools/sparkcap_stream_probe.c,
userspace CPU-side lower bound); SPEC BW is the platform spec number.
Touched-expert law: N_ranks_moe_bytes * (1-(1-k/N)^B); at B=1 this equals
moe_bytes_on_rank * k / N for replicated and expert-sharded placements
alike.  HOT prices expert bytes at memory BW; COLD (lease-streamed) at
NVMe BW.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import struct
import subprocess
import sys
from typing import Any

SPEC_BW_GB_PER_S_DEFAULT = 273.0
FORMAT_HEADER_GLOBS = [
    "include/sparkpipe/spark_stagepack_format.h",
    "modules/*/source/spark_*_stagepack_format.h",
]
WEIGHT_FORMAT_ELEMENT_BYTES = {
    "SPARK_STAGEPACK_FORMAT_WEIGHT_BF16": 2.0,
    "SPARK_STAGEPACK_FORMAT_WEIGHT_F32": 4.0,
    "SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128": 1.0,
    "SPARK_STAGEPACK_FORMAT_WEIGHT_I64": 8.0,
    "SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED": 0.5,
}
ACTIVATION_TOUCH_NAMES = [
    "input-norm", "qkv/gdn-in write+read", "conv-state read+write",
    "attn/gdn-out write", "o_proj write+read", "residual+post-norm",
    "router", "gather write+read", "expert-out write+read",
    "shared-expert write+read", "scatter-add", "next-block norm",
]
LANE_PROBES = {
    "qwen38max": ["qwen38max", "qwenmax"],
    "qwen3flash": ["qwen38flash", "qwen3flash", "qwen4flash", "qwenflash"],
    "qwen3827b": ["qwen3827b", "qwen27b"],
}


def repo_header_paths(repo_root: str) -> list[str]:
    paths: list[str] = []
    for pattern in FORMAT_HEADER_GLOBS:
        paths.extend(sorted(glob.glob(os.path.join(repo_root, pattern))))
    return paths


def parse_c_number(text: str) -> int | None:
    text = text.strip().rstrip("uU")
    try:
        return int(text, 0)
    except ValueError:
        return None


def parse_defines(text: str) -> dict[str, int]:
    defines: dict[str, int] = {}
    for match in re.finditer(r"^#define\s+(\w+)\s+(0x[0-9a-fA-F]+[uU]?|\d+[uU]?)\s*$", text, re.M):
        value = parse_c_number(match.group(2))
        if value is not None:
            defines[match.group(1)] = value
    return defines


def parse_struct_fields(text: str, struct_name: str) -> list[tuple[str, int]]:
    match = re.search(
        r"typedef struct\s+" + struct_name + r"\s*\{(.*?)\}\s*" + struct_name + r"\s*;",
        text,
        re.S,
    )
    if match is None:
        return []
    fields: list[tuple[str, int]] = []
    for line in match.group(1).splitlines():
        field = re.match(r"\s*uint(32|64)_t\s+(\w+)\s*;", line)
        if field:
            fields.append((field.group(2), int(field.group(1)) // 8))
    return fields


def parse_tensor_kind_names(text: str, name_value: dict[str, int]) -> dict[str, int]:
    local: dict[str, int] = {}
    for match in re.finditer(r"typedef enum\s+\w*\s*\{(.*?)\}\s*\w*\s*;", text, re.S):
        next_value = 0
        for entry in match.group(1).split(","):
            piece = re.match(r"\s*(\w+)\s*(?:=\s*([^,]+))?", entry)
            if not piece:
                continue
            raw = piece.group(2)
            value = parse_c_number(raw) if raw else next_value
            if value is None and raw:
                value = name_value.get(raw.strip())
            if value is None:
                continue
            next_value = value + 1
            if "_TENSOR_" in piece.group(1):
                local[piece.group(1)] = value
                name_value[piece.group(1)] = value
    for match in re.finditer(r"^#define\s+(\w+)\s+(\w+)\s*$", text, re.M):
        source, target = match.group(1), match.group(2)
        if "_TENSOR_" in source and target in name_value and source not in local:
            local[source] = name_value[target]
            name_value[source] = name_value[target]
    return local


def parse_formats(repo_root: str) -> dict[str, Any]:
    defines: dict[str, int] = {}
    families: dict[str, dict[str, Any]] = {}
    name_value: dict[str, int] = {}
    texts: list[tuple[str, str]] = []
    for path in repo_header_paths(repo_root):
        with open(path, "r", encoding="utf-8") as handle:
            texts.append((path, handle.read()))
    for _, text in texts:
        parse_tensor_kind_names(text, name_value)
    for _, text in texts:
        file_defines = parse_defines(text)
        defines.update(file_defines)
        file_kinds = {value: name for name, value in parse_tensor_kind_names(text, name_value).items()}
        for magic_name, magic in file_defines.items():
            if not magic_name.endswith("_STAGEPACK_MAGIC"):
                continue
            candidates = re.findall(r"typedef struct\s+(Spark\w*StagePackHeader\w*)\s*\{", text)
            candidates.append("SparkStagePackHeaderCommon")
            for struct_name in candidates:
                fields = parse_struct_fields(text, struct_name)
                if fields:
                    families[str(magic)] = {"fields": fields, "kinds": file_kinds}
                    break
    element_bytes = {
        value: WEIGHT_FORMAT_ELEMENT_BYTES[name]
        for name, value in defines.items()
        if name in WEIGHT_FORMAT_ELEMENT_BYTES
    }
    return {"defines": defines, "families": families, "element_bytes": element_bytes}


def special_kind_ids(kinds: dict[int, str]) -> dict[str, set[int]]:
    wanted = {"embedding": "EMBEDDING", "lm_head": "LM_HEAD", "moe": ("MOE_W1", "MOE_W3", "MOE_DOWN")}
    ids: dict[str, set[int]] = {"embedding": set(), "lm_head": set(), "moe": set(), "lookup": set()}
    for value, name in kinds.items():
        if "_PLE_" in name or "_NGRAM" in name:
            ids["lookup"].add(value)
            continue
        for key, token in wanted.items():
            tokens = token if isinstance(token, tuple) else (token,)
            if any(name.endswith("_" + item) for item in tokens):
                ids[key].add(value)
    return ids


def ssh_run(node: str, command: str) -> str:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, command],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def ssh_read_bytes(node: str, path: str, offset: int, count: int) -> bytes:
    command = "tail -c +%d '%s' 2>/dev/null | head -c %d | base64" % (offset + 1, path, count)
    return base64.b64decode(ssh_run(node, command).strip())


def discover_packs(node: str, arm: str) -> list[str]:
    try:
        output = ssh_run(node, "ls -1 ~/sparkdata/%s/packs/* 2>/dev/null" % arm)
    except subprocess.CalledProcessError:
        return []
    return [
        line.strip()
        for line in output.splitlines()
        if line.strip() and not line.strip().endswith((".sha256", ".json", ".experts", ".backup"))
    ]


def family_for_header(magic: int, header_bytes: int, formats: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, int]]]:
    family = formats["families"].get(str(magic))
    if family is None:
        raise ValueError("unknown stagepack magic %d (%#x)" % (magic, magic))
    fields = family["fields"]
    if header_bytes == sum(width for _, width in fields):
        return family, fields
    donor = None
    for other in formats["families"].values():
        if sum(width for _, width in other["fields"]) == header_bytes:
            donor = other["fields"]
            break
    base_u32 = [entry for entry in fields if entry[1] == 4]
    wire_u32 = (header_bytes - 16) // 4
    extra = wire_u32 - len(base_u32)
    if extra < 0 or (donor is None and extra == 0):
        raise ValueError("wire header %d bytes has no layout for magic %#x" % (header_bytes, magic))
    if donor is not None:
        tail_u32 = [entry for entry in donor if entry[1] == 4][len(base_u32):]
    else:
        tail_u32 = [("tail_u32_%d" % index, 4) for index in range(extra)]
    u64s = [entry for entry in fields if entry[1] == 8]
    return family, base_u32 + tail_u32 + u64s


def fetch_pack(node: str, path: str, formats: dict[str, Any]) -> dict[str, Any]:
    size = int(ssh_run(node, "stat -c %%s '%s'" % path).strip())
    head = ssh_read_bytes(node, path, 0, 128)
    magic, format_version, header_bytes, entry_bytes, tensor_count = struct.unpack_from("<5I", head, 0)
    if header_bytes < 120 or header_bytes > 128:
        raise ValueError("unsupported wire header_bytes %d" % header_bytes)
    family_for_header(magic, header_bytes, formats)
    head = head[:header_bytes]
    directory_offset = struct.unpack_from("<Q", head, header_bytes - 16)[0]
    directory = ssh_read_bytes(node, path, directory_offset, tensor_count * entry_bytes)
    if len(directory) != tensor_count * entry_bytes:
        raise ValueError("short directory read: %d of %d bytes" % (len(directory), tensor_count * entry_bytes))
    receipt: dict[str, Any] = {}
    for suffix in (".receipt.json", ".packer-receipt.json"):
        try:
            receipt = json.loads(ssh_read_bytes(node, path + suffix, 0, 1 << 16).decode("utf-8"))
            break
        except Exception:
            continue
    experts_version = None
    experts_records = None
    try:
        experts_header = ssh_read_bytes(node, path + ".experts", 0, 16)
        experts_magic, experts_version, _, experts_records = struct.unpack_from("<4sHHI", experts_header, 0)
        assert experts_magic == b"WEPX"
    except Exception:
        pass
    return {
        "path": path,
        "node": node,
        "file_bytes": size,
        "header_bytes": header_bytes,
        "entry_bytes": entry_bytes,
        "tensor_count": tensor_count,
        "magic": magic,
        "format_version": format_version,
        "head_b64": base64.b64encode(head).decode("ascii"),
        "directory_b64": base64.b64encode(directory).decode("ascii"),
        "directory_offset": directory_offset,
        "receipt": receipt,
        "experts_version": experts_version,
        "experts_records": experts_records,
    }


def decode_pack(pack: dict[str, Any], formats: dict[str, Any]) -> dict[str, Any]:
    family, fields = family_for_header(pack["magic"], pack["header_bytes"], formats)
    head = base64.b64decode(pack["head_b64"])
    header: dict[str, int] = {}
    offset = 0
    for name, width in fields:
        header[name] = struct.unpack_from("<Q" if width == 8 else "<I", head, offset)[0]
        offset += width
    directory = base64.b64decode(pack["directory_b64"])
    ids = special_kind_ids(family["kinds"])
    totals = {"embedding": 0, "lm_head": 0, "moe": 0, "spine": 0, "lookup": 0}
    layers: set[int] = set()
    moe_layers: set[int] = set()
    lookup_rows = 0
    lookup_row_bytes_sum = 0.0
    accounted = pack["header_bytes"] + pack["tensor_count"] * pack["entry_bytes"]
    for index in range(pack["tensor_count"]):
        base = index * pack["entry_bytes"]
        tensor_kind, layer_index = struct.unpack_from("<II", directory, base)
        rows = struct.unpack_from("<I", directory, base + 12)[0]
        payload_bytes = struct.unpack_from("<Q", directory, base + 32)[0]
        scale_bytes = struct.unpack_from("<Q", directory, base + 48)[0]
        total = payload_bytes + scale_bytes
        accounted += total
        if tensor_kind in ids["embedding"]:
            totals["embedding"] += total
        elif tensor_kind in ids["lm_head"]:
            totals["lm_head"] += total
        elif tensor_kind in ids["lookup"]:
            totals["lookup"] += total
            if rows:
                lookup_rows += rows
                lookup_row_bytes_sum += payload_bytes / rows
        elif tensor_kind in ids["moe"]:
            totals["moe"] += total
            moe_layers.add(layer_index)
        else:
            totals["spine"] += total
            layers.add(layer_index)
    padding_slack = pack["tensor_count"] * 256 + 4096
    gap = pack["file_bytes"] - accounted
    return {
        "header": header,
        "totals": totals,
        "tensor_layers": len(layers),
        "moe_layer_count": len(moe_layers),
        "lookup_rows": lookup_rows,
        "lookup_row_bytes": lookup_row_bytes_sum / lookup_rows if lookup_rows else 0.0,
        "accounted_bytes": accounted,
        "gap_bytes": gap,
        "gap_fatal": gap > padding_slack,
    }


def parse_memory_model(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 4 or set(cells[0]) <= {"-", " ", ":"}:
                continue
            rows.append({"driver": cells[0], "cells": cells})
    return rows


def lane_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def memory_model_row(rows: list[dict[str, Any]], lane: str) -> dict[str, Any] | None:
    for row in rows:
        key = lane_key(row["driver"])
        if any(probe in key for probe in LANE_PROBES.get(lane, [lane])):
            return row
    return None


def arm_lane(arm: str) -> str:
    key = lane_key(arm)
    for lane, probes in LANE_PROBES.items():
        if any(probe in key for probe in probes):
            return lane
    return key


def roofline(args: argparse.Namespace, pack: dict[str, Any], decoded: dict[str, Any]) -> dict[str, Any]:
    header = decoded["header"]
    totals = decoded["totals"]
    tp_degree = int(pack["receipt"].get("tp_degree") or header.get("tp_degree") or args.tp_degree or 1)
    layer_count = int(header.get("layer_count") or 1)
    period = max(1, int(header.get("attention_period") or 4))
    attn_layers = max(1, layer_count // period)
    gdn_layers = max(0, layer_count - attn_layers)
    hidden = int(header.get("hidden_dimension") or 0)
    kv_heads = int(header.get("attn_kv_head_count") or 1)
    head_dim = int(header.get("attn_head_dimension") or 0)
    gdn_value_heads = int(header.get("gdn_value_head_count") or 0)
    gdn_key_dim = int(header.get("gdn_head_key_dimension") or 0)
    gdn_value_dim = int(header.get("gdn_head_value_dimension") or 0)
    routed_experts = int(header.get("routed_expert_count") or 0)
    experts_per_token = int(header.get("experts_per_token") or 0)
    local_kv_heads = max(1, -(-kv_heads // tp_degree))
    local_gdn_heads = max(1, -(-gdn_value_heads // tp_degree))
    batch = args.batch
    context = args.context
    amortize = 1.0 / batch
    hidden_row_bytes = hidden * args.kv_element_bytes

    touched_fraction = 1.0 - (1.0 - experts_per_token / routed_experts) ** batch if routed_experts else 0.0
    expert_bytes = totals["moe"] * touched_fraction
    lookup_bytes = decoded["lookup_rows"] and args.lookup_rows_per_token * decoded["lookup_row_bytes"] or 0.0
    kv_read = context * local_kv_heads * head_dim * args.kv_element_bytes * attn_layers
    kv_append = local_kv_heads * head_dim * args.kv_element_bytes * attn_layers
    gdn_state = gdn_layers * local_gdn_heads * gdn_key_dim * gdn_value_dim * args.gdn_state_bytes * 2
    activations = args.activation_roundtrips * hidden_row_bytes * layer_count
    embedding_row = hidden_row_bytes

    per_token_hot = (
        embedding_row
        + (totals["spine"] + totals["lm_head"]) * amortize
        + expert_bytes
        + lookup_bytes
        + kv_read
        + kv_append
        + gdn_state
        + activations
    )
    measured_bw = args.measured_bw * 1e9
    spec_bw = args.spec_bw * 1e9
    nvme_bw = args.nvme_bw * 1e9
    non_expert_bytes = per_token_hot - expert_bytes
    ceiling_hot = measured_bw / per_token_hot if per_token_hot else 0.0
    ceiling_spec = spec_bw / per_token_hot if per_token_hot else 0.0
    cold_seconds = non_expert_bytes / measured_bw + expert_bytes / nvme_bw
    ceiling_cold = 1.0 / cold_seconds if cold_seconds else 0.0
    percent_hot = args.tokps * per_token_hot / measured_bw * 100.0 if args.tokps else None
    components = {
        "embedding_row": embedding_row,
        "spine_stream": totals["spine"] * amortize,
        "lm_head": totals["lm_head"] * amortize,
        "experts_touched_hot": expert_bytes,
        "lookup_touched": lookup_bytes,
        "kv_read": kv_read,
        "kv_append": kv_append,
        "gdn_state_rw": gdn_state,
        "activation_round_trips": activations,
    }
    dominant = max(components, key=components.get) if per_token_hot else "n/a"
    lane = arm_lane(os.path.basename(pack["path"]))
    model_row = memory_model_row(parse_memory_model(args.memory_model), lane)
    return {
        "lane": lane,
        "arm": os.path.basename(pack["path"]),
        "node": pack["node"],
        "tp_degree": tp_degree,
        "layers_on_rank": layer_count,
        "hidden": hidden,
        "routed_experts": routed_experts,
        "experts_per_token": experts_per_token,
        "pack_bytes": pack["file_bytes"],
        "totals_on_rank": totals,
        "components_bytes_per_token": components,
        "bytes_per_token_hot": per_token_hot,
        "measured_bw_gb_per_s": args.measured_bw,
        "spec_bw_gb_per_s": args.spec_bw,
        "nvme_bw_gb_per_s": args.nvme_bw,
        "ceiling_tok_s_hot_measured": ceiling_hot,
        "ceiling_tok_s_hot_spec": ceiling_spec,
        "ceiling_tok_s_cold_nvme": ceiling_cold,
        "receipted_tok_s": args.tokps,
        "receipted_source": args.tokps_source,
        "percent_of_measured_roofline_hot": percent_hot,
        "dominant_term": dominant,
        "memory_model_row": " | ".join(model_row["cells"][:5]) if model_row else None,
    }


def print_table(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        print("== [%s] %s (%s, TP%d, %d layers on rank, pack %.1f GB)" % (
            entry["lane"], entry["arm"], entry["node"], entry["tp_degree"],
            entry["layers_on_rank"], entry["pack_bytes"] / 1e9,
        ))
        print("   bytes/token/rank (hot) = %s" % " + ".join(
            "%s %.3f MB" % (name, value / 1e6)
            for name, value in entry["components_bytes_per_token"].items()
            if value
        ))
        print("   TOTAL %.3f GB/token/rank" % (entry["bytes_per_token_hot"] / 1e9))
        print("   SPEC %.0f GB/s -> %.2f tok/s | MEASURED %.1f GB/s -> %.2f tok/s | COLD NVMe %.1f GB/s -> %.2f tok/s" % (
            entry["spec_bw_gb_per_s"], entry["ceiling_tok_s_hot_spec"],
            entry["measured_bw_gb_per_s"], entry["ceiling_tok_s_hot_measured"],
            entry["nvme_bw_gb_per_s"], entry["ceiling_tok_s_cold_nvme"],
        ))
        if entry["receipted_tok_s"]:
            print("   RECEIPTED %.2f tok/s (%s) -> %.1f%% of MEASURED roofline (hot)" % (
                entry["receipted_tok_s"], entry["receipted_source"],
                entry["percent_of_measured_roofline_hot"],
            ))
        else:
            print("   RECEIPTED N/A (no tok/s receipt for this arm; never fabricated)")
        print("   dominant term: %s" % entry["dominant_term"])
        if entry["memory_model_row"]:
            print("   MEMORY_MODEL: %s" % entry["memory_model_row"])
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--memory-model", default="/Users/mac/sparkpipe-coord/MEMORY_MODEL.md")
    parser.add_argument("--node", default="spark0")
    parser.add_argument("--arm", action="append", default=[], help="arm dir under ~/sparkdata/<arm>/packs (repeatable)")
    parser.add_argument("--pack", action="append", default=[], help="explicit pack path (repeatable)")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--from-cache", default=None)
    parser.add_argument("--spec-bw", type=float, default=SPEC_BW_GB_PER_S_DEFAULT)
    parser.add_argument("--measured-bw", type=float, required=True)
    parser.add_argument("--nvme-bw", type=float, default=5.1)
    parser.add_argument("--tokps", type=float, default=None)
    parser.add_argument("--tokps-source", default="")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=None)
    parser.add_argument("--kv-element-bytes", type=float, default=2.0)
    parser.add_argument("--gdn-state-bytes", type=float, default=4.0)
    parser.add_argument("--lookup-rows-per-token", type=float, default=2048.0,
                        help="rows touched per token in lookup tables (PLE/ngram); default = the qwen4-flash indexer budget")
    parser.add_argument("--activation-roundtrips", type=int, default=len(ACTIVATION_TOUCH_NAMES),
                        help="hidden-row R+W pairs per layer: " + ", ".join(ACTIVATION_TOUCH_NAMES))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    formats = parse_formats(args.repo)
    if not formats["families"]:
        raise SystemExit("no stagepack layouts parsed from %s" % args.repo)
    if args.from_cache:
        with open(args.from_cache, "r", encoding="utf-8") as handle:
            packs = json.load(handle)
    else:
        packs = []
        for pack_path in args.pack:
            packs.append(fetch_pack(args.node, pack_path, formats))
        for arm in args.arm:
            for pack_path in discover_packs(args.node, arm):
                try:
                    packs.append(fetch_pack(args.node, pack_path, formats))
                except Exception as error:
                    print("skip %s: %s" % (pack_path, error), file=sys.stderr)
        if args.cache:
            with open(args.cache, "w", encoding="utf-8") as handle:
                json.dump(packs, handle, indent=1)
    entries = []
    for pack in packs:
        decoded = decode_pack(pack, formats)
        if decoded["gap_fatal"]:
            raise SystemExit(
                "byte accounting gap in %s: parsed %d of %d bytes"
                % (pack["path"], decoded["accounted_bytes"], pack["file_bytes"])
            )
        entries.append(roofline(args, pack, decoded))
    if args.json:
        print(json.dumps({"format_magics": sorted(formats["defines"].items()), "lanes": entries}, indent=1))
    else:
        print_table(entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())

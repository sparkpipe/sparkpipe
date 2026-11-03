#!/usr/bin/env python3
import argparse
import json
import struct
import sys

from pack_restamp import restamp, sidecar_digest

MAGIC = 0x33584C47
FORMAT_VERSION = 1
HEADER_BYTES = 264
ENTRY_BYTES = 64
EXPERT_CODEC_OFFSET = 16 * 4
TENSOR_COUNT_INDEX = 6
EXPERT_UP_GATE = 22
EXPERT_DOWN = 23
PAYLOAD_PACKED = 4
CODECS = {"bf16": 1, "fp8": 5}


def read_pack(path):
    with open(path, "rb") as handle:
        header = handle.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES:
            raise SystemExit(f"{path}: shorter than a glm5_next stage pack header")
        fields = struct.unpack_from("<20I", header, 0)
        directory_offset, file_bytes = struct.unpack_from("<QQ", header, 80)
        if fields[0] != MAGIC or fields[1] != FORMAT_VERSION or fields[2] != HEADER_BYTES or fields[3] != ENTRY_BYTES:
            raise SystemExit(f"{path}: not a glm5_next v{FORMAT_VERSION} stage pack")
        handle.seek(directory_offset)
        directory = handle.read(fields[TENSOR_COUNT_INDEX] * ENTRY_BYTES)
    if len(directory) != fields[TENSOR_COUNT_INDEX] * ENTRY_BYTES:
        raise SystemExit(f"{path}: the directory is truncated")
    experts = []
    for index in range(fields[TENSOR_COUNT_INDEX]):
        kind, layer, payload_type, codec, encoding, _, _, _, _, _, _, scale_bytes = struct.unpack_from("<8I4Q", directory, index * ENTRY_BYTES)
        if kind in (EXPERT_UP_GATE, EXPERT_DOWN):
            experts.append({"kind": kind, "layer": layer, "payload_type": payload_type, "codec": codec, "encoding": encoding, "scale_bytes": scale_bytes})
    return header, fields[16], experts, file_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack")
    parser.add_argument("--expert-codec", required=True, choices=sorted(CODECS))
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    target = CODECS[arguments.expert_codec]
    header, stamped, experts, file_bytes = read_pack(arguments.pack)
    report = {"pack": arguments.pack, "header_expert_codec": stamped, "routed_expert_entries": len(experts),
              "file_bytes": file_bytes, "sidecar": sidecar_digest(arguments.pack)}
    if not experts:
        raise SystemExit(f"{arguments.pack}: no routed-expert entries carry a codec to stamp")
    odd = [entry for entry in experts if entry["payload_type"] != PAYLOAD_PACKED or entry["codec"] != target or
           (target == CODECS["bf16"] and (entry["encoding"] != 0 or entry["scale_bytes"] != 0))]
    if odd:
        raise SystemExit(f"{arguments.pack}: {len(odd)} routed-expert entries are not {arguments.expert_codec}, first {odd[0]}; the payload must be repacked")
    if stamped == target:
        report["action"] = "already stamped"
        print(json.dumps(report))
        return 0
    if not arguments.apply:
        report["action"] = "would restamp"
        print(json.dumps(report))
        return 0
    receipt = restamp(arguments.pack, header, EXPERT_CODEC_OFFSET, struct.pack("<I", stamped), struct.pack("<I", target),
                      f"expert-codec-{stamped}", dict(report, to_expert_codec=target))
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import json
import struct
import sys

from pack_restamp import restamp, sidecar_digest

MAGIC = 0x32534C47
FORMAT_VERSION = 3
HEADER_BYTES = 264
REVISION_OFFSET = 96
REVISION_BYTES = 65
CONTRACT_OFFSET = REVISION_OFFSET + REVISION_BYTES
DIGEST_BYTES = 32


def read_header(path):
    with open(path, "rb") as handle:
        header = handle.read(HEADER_BYTES)
    magic, version, header_bytes = struct.unpack_from("<3I", header, 0)
    if magic != MAGIC or version != FORMAT_VERSION or header_bytes != HEADER_BYTES:
        raise SystemExit(f"{path}: not a glm52 v{FORMAT_VERSION} stage pack")
    revision = header[REVISION_OFFSET:REVISION_OFFSET + REVISION_BYTES].split(b"\0")[0].decode()
    contract = header[CONTRACT_OFFSET:CONTRACT_OFFSET + DIGEST_BYTES].hex()
    return header, revision, contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--from-contract", required=True)
    parser.add_argument("--to-contract", required=True)
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    for value in (arguments.from_contract, arguments.to_contract):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise SystemExit("contracts are 64 lowercase hex digits")
    header, revision, contract = read_header(arguments.pack)
    report = {"pack": arguments.pack, "revision": revision, "contract": contract, "sidecar": sidecar_digest(arguments.pack)}
    if revision != arguments.revision:
        raise SystemExit(f"{arguments.pack}: header revision {revision} != {arguments.revision}")
    if contract == arguments.to_contract:
        report["action"] = "already stamped"
        print(json.dumps(report))
        return 0
    if contract != arguments.from_contract:
        raise SystemExit(f"{arguments.pack}: header contract {contract} is neither the source nor the target identity")
    if not arguments.apply:
        report["action"] = "would restamp"
        print(json.dumps(report))
        return 0
    receipt = restamp(arguments.pack, header, CONTRACT_OFFSET, bytes.fromhex(arguments.from_contract), bytes.fromhex(arguments.to_contract),
                      arguments.from_contract[:8], dict(report, to_contract=arguments.to_contract))
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())

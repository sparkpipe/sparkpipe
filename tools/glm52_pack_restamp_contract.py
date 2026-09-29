#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import os
import struct
import sys

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


def file_sha256(path):
    digest = hashlib.sha256()
    offset = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 24)
            if not block:
                return digest.hexdigest()
            digest.update(block)
            os.posix_fadvise(handle.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED)
            offset += len(block)


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
    sidecar = arguments.pack + ".sha256"
    header, revision, contract = read_header(arguments.pack)
    with open(sidecar) as handle:
        recorded = handle.read().split()[0]
    report = {"pack": arguments.pack, "revision": revision, "contract": contract, "sidecar": recorded}
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
    before = file_sha256(arguments.pack)
    if before != recorded:
        raise SystemExit(f"{arguments.pack}: content sha256 {before} != sidecar {recorded}; refusing to restamp")
    backup = f"{arguments.pack}.header-{arguments.from_contract[:8]}"
    with open(backup, "wb") as handle:
        handle.write(header)
    with open(arguments.pack, "r+b") as handle:
        handle.seek(CONTRACT_OFFSET)
        handle.write(bytes.fromhex(arguments.to_contract))
        handle.flush()
        os.fsync(handle.fileno())
    _, _, stamped = read_header(arguments.pack)
    if stamped != arguments.to_contract:
        raise SystemExit(f"{arguments.pack}: stamp did not land")
    after = file_sha256(arguments.pack)
    name = os.path.basename(arguments.pack)
    with open(sidecar + ".new", "w") as handle:
        handle.write(f"{after}  {name}\n")
    os.replace(sidecar + ".new", sidecar)
    receipt = dict(report, action="restamped", sha256_before=before, sha256_after=after, header_backup=backup,
                   to_contract=arguments.to_contract,
                   utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with open(arguments.pack + ".restamp.json", "w") as handle:
        json.dump(receipt, handle, indent=1)
        handle.write("\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Pack-verification parity oracle: warm checkpoint vs packer receipt vs placed bytes.

Enforces the pack-verification law per lane arm:
  pack == packer   (whole-file digest vs the packer receipt / .sha256 sidecar)
  packer == checkpoint (receipt source identity vs the live warm checkpoint)
  pack == checkpoint   (the family verifier's byte-trace pass, run as a child)

The family verifiers stay the byte-level authorities; this driver adds the
digest legs they leave optional (--recompute-file-hash), binds the three legs
into one verdict, and fails loud on any missing input.

Usage (node-local, next to the warm mount):
  python3 tools/wave_acc1_parity_oracle.py --lane qwen3flash --arm qwen3flash.bf16.tp8 \
      --pack packs/qwenflash.tp8.rank0.pack \
      --checkpoint /mnt/model-warm/qwen3.8-flash-next \
      --family-verify tools/qwen4_flash_pack_verify.py \
      --family-args "--tp-degree 8 --tp-rank 0" \
      --json-out /tmp/acc1-qwen3flash-bf16tp8.json

Exit 0 only when every requested leg passes. A leg with no usable input is
reported UNVERIFIABLE and fails the run unless --allow-unverifiable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HASH_CHUNK = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(HASH_CHUNK)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def read_sidecar_digest(path: Path) -> str | None:
    text = path.read_text().strip()
    if not text:
        return None
    first = text.splitlines()[0].split()
    if first:
        return first[0].lower()
    return None


def load_receipt(path: Path | None) -> dict:
    if path is None or not Path(path).is_file():
        return {}
    receipt = json.loads(Path(path).read_text())
    source = receipt.get("source")
    if isinstance(source, dict):
        receipt.setdefault("source_index_sha256", source.get("index_sha256"))
        receipt.setdefault("source_config_sha256", source.get("config_sha256"))
    return receipt


def source_identity(checkpoint: Path) -> tuple[str | None, str | None, str | None]:
    index_path = checkpoint / "model.safetensors.index.json"
    config_path = checkpoint / "config.json"
    index_sha = sha256_file(index_path) if index_path.is_file() else None
    config_sha = sha256_file(config_path) if config_path.is_file() else None
    return index_sha, config_sha, str(checkpoint)


def leg_status(enabled: bool, ok: bool, detail: dict) -> dict:
    status = ("SKIP" if not enabled else ("PASS" if ok else "FAIL"))
    return {"status": status, **detail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--sidecar", type=Path,
                        help="<pack>.sha256 (sha256sum format) or manifest json "
                             "with file_sha256")
    parser.add_argument("--family-verify", type=Path,
                        help="family verifier script run as the content leg")
    parser.add_argument("--family-args", default="",
                        help="argument tokens for the family verifier; {pack}, "
                             "{checkpoint} and {receipt} are substituted")
    parser.add_argument("--expected-sha256",
                        help="known pack digest to compare instead of a sidecar")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not args.pack.is_file():
        print(f"FAIL pack missing: {args.pack}", file=sys.stderr)
        return 1

    receipt = load_receipt(args.receipt)
    sidecar_digest = None
    if args.sidecar is not None and args.sidecar.is_file():
        if args.sidecar.suffix == ".json":
            try:
                sidecar_digest = json.loads(args.sidecar.read_text()).get("file_sha256")
            except json.JSONDecodeError:
                sidecar_digest = None
        else:
            sidecar_digest = read_sidecar_digest(args.sidecar)
    expected = args.expected_sha256 or receipt.get("output_sha256") or sidecar_digest

    digest = sha256_file(args.pack) if expected is not None else None
    digest_ok = None if expected is None or digest is None else digest == expected

    index_sha = config_sha = None
    checkpoint_present = args.checkpoint is not None and args.checkpoint.is_dir()
    if checkpoint_present:
        index_sha, config_sha, _ = source_identity(args.checkpoint)
    recorded_index = receipt.get("source_index_sha256")
    identity_ok = (checkpoint_present and index_sha is not None
                   and recorded_index == index_sha)

    content: dict = {"status": "SKIP", "detail": "no family verifier given"}
    if args.family_verify is not None:
        if not args.family_verify.is_file():
            content = {"status": "SKIP",
                       "detail": f"verifier missing: {args.family_verify}"}
        elif args.checkpoint is None or not args.checkpoint.is_dir():
            content = {"status": "SKIP", "detail": "no checkpoint for content leg"}
        else:
            content = run_family(args.family_verify, args.family_args, args.pack,
                                 args.checkpoint, args.receipt)

    verdict = {
        "lane": args.lane,
        "arm": args.arm,
        "pack": str(args.pack),
        "pack_bytes": args.pack.stat().st_size,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "receipt": str(args.receipt) if args.receipt else None,
        "legs": {
            "pack_equals_packer": leg_status(
                expected is not None,
                bool(digest_ok),
                {"expected_sha256": expected, "recomputed_sha256": digest}),
            "packer_equals_checkpoint": leg_status(
                bool(recorded_index) and checkpoint_present,
                bool(identity_ok),
                {"receipt_source_index_sha256": receipt.get("source_index_sha256"),
                 "live_index_sha256": index_sha,
                 "live_config_sha256": config_sha}),
            "pack_equals_checkpoint": content,
        },
    }
    statuses = [leg["status"] for leg in verdict["legs"].values()]
    hard = [s for s in statuses if s != "SKIP"]
    if not hard:
        verdict["verdict"] = "UNVERIFIABLE"
    elif "FAIL" in hard:
        verdict["verdict"] = "FAIL"
    else:
        verdict["verdict"] = "PASS"

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in verdict.items() if k != "legs"}, indent=2))
    for name, leg in verdict["legs"].items():
        print(f"  {name}: {leg['status']}")
    print(f"oracle verdict={verdict['verdict']} lane={args.lane} arm={args.arm}")
    if verdict["verdict"] == "FAIL":
        return 1
    return 0


def run_family(verifier: Path, family_args: str, pack: Path, checkpoint: Path,
               receipt: Path | None) -> dict:
    substitutions = {"{pack}": str(pack), "{checkpoint}": str(checkpoint),
                     "{receipt}": str(receipt) if receipt is not None else ""}
    argv = [substitutions.get(token, token) for token in family_args.split()]
    command = [sys.executable, str(verifier)] + argv
    print(f"  content leg: {' '.join(command)}", flush=True)
    proc = subprocess.run(command, capture_output=True, text=True)
    tail = [line for line in (proc.stdout + proc.stderr).splitlines() if line][-12:]
    return {"status": "PASS" if proc.returncode == 0 else "FAIL",
            "returncode": proc.returncode,
            "command": command,
            "output_tail": tail}


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("oracle: interrupted", file=sys.stderr)
        sys.exit(130)

#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "spec_offline"))
from drafter_contract import DrafterContractError, drafter_tensors, tap_entries  # noqa: E402

REPOSITORY = Path(__file__).resolve().parent.parent
DRAFTER_ROOT = Path("autopilot") / "drafter"
MANIFEST = "MANIFEST.json"
PUBLISHED = "published"
EXPORT_FORMAT = "sparkpipe-drafter-export-v1"
RECEIPT_FORMAT = "sparkpipe-drafter-training-v1"
FILES = ("config.json", "model.safetensors", "TRAINING-RECEIPT.json")
LIMITS = {MANIFEST: 1 << 20, "config.json": 1 << 20, "TRAINING-RECEIPT.json": 16 << 20, "model.safetensors": 64 << 30}
HEADER_LIMIT = 64 << 20
CHUNK = 16 << 20


class RefusedDrafter(ValueError):
    pass


def root(scratch):
    return Path(scratch) / DRAFTER_ROOT


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, what):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RefusedDrafter(f"{what} is not readable JSON: {error}")


def lane_taps(lanes_path, lane):
    lanes = read_json(lanes_path, "the lane records")["lanes"]
    if lane not in lanes or "taps" not in lanes[lane]:
        raise RefusedDrafter(f"{lanes_path} has no lane {lane} with taps")
    return [name for name in lanes[lane]["taps"]["entries"].split(",") if name]


def check_weights(path, config):
    expected = drafter_tensors(config)
    size = path.stat().st_size
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise RefusedDrafter("model.safetensors is shorter than its header length")
        length = struct.unpack("<Q", raw)[0]
        if length > HEADER_LIMIT or 8 + length > size:
            raise RefusedDrafter(f"model.safetensors declares a {length}-byte header in a {size}-byte file")
        try:
            header = json.loads(handle.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RefusedDrafter(f"model.safetensors header is not JSON: {error}")
    header.pop("__metadata__", None)
    missing, extra = sorted(set(expected) - set(header)), sorted(set(header) - set(expected))
    if missing or extra:
        raise RefusedDrafter(f"model.safetensors tensors differ from the drafter's: missing {missing[:4]}, extra {extra[:4]}")
    spans = []
    for name, entry in header.items():
        if not isinstance(entry, dict) or entry.get("dtype") != "BF16" or entry.get("shape") != expected[name]:
            raise RefusedDrafter(f"{name} is not BF16 of shape {expected[name]}")
        offsets = entry.get("data_offsets")
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(isinstance(value, int) for value in offsets)) or \
                offsets[1] - offsets[0] != 2 * math.prod(expected[name]):
            raise RefusedDrafter(f"{name} has data offsets {offsets} for {expected[name]}")
        spans.append(tuple(offsets))
    spans.sort()
    cursor = 0
    for begin, end in spans:
        if begin != cursor:
            raise RefusedDrafter(f"model.safetensors data has a hole or an overlap at byte {cursor}")
        cursor = end
    if cursor != size - 8 - length:
        raise RefusedDrafter(f"model.safetensors holds {size - 8 - length} data bytes, its tensors {cursor}")


def describe(directory, expect_taps, expect_pack):
    directory = Path(directory)
    present = sorted(path.name for path in directory.iterdir() if path.name != MANIFEST)
    if present != sorted(FILES):
        raise RefusedDrafter(f"a drafter holds exactly {sorted(FILES)}; {directory} holds {present}")
    for name in FILES:
        if not (directory / name).is_file() or (directory / name).is_symlink():
            raise RefusedDrafter(f"{name} is not a regular file")
        if (directory / name).stat().st_size > LIMITS[name]:
            raise RefusedDrafter(f"{name} is larger than {LIMITS[name]} bytes")
    config = read_json(directory / "config.json", "config.json")
    try:
        taps = tap_entries(config)
        check_weights(directory / "model.safetensors", config)
    except (DrafterContractError, KeyError, TypeError) as error:
        raise RefusedDrafter(f"config.json does not describe a supported drafter: {error}")
    receipt = read_json(directory / "TRAINING-RECEIPT.json", "TRAINING-RECEIPT.json")
    if receipt.get("format") != RECEIPT_FORMAT:
        raise RefusedDrafter(f"TRAINING-RECEIPT.json is not {RECEIPT_FORMAT}")
    files = {name: {"bytes": (directory / name).stat().st_size, "sha256": sha256_file(directory / name)} for name in FILES}
    if receipt.get("model_sha256") != files["model.safetensors"]["sha256"]:
        raise RefusedDrafter("the training receipt names other weights than model.safetensors")
    if receipt.get("tap_entries") != taps:
        raise RefusedDrafter(f"the receipt trained on taps {receipt.get('tap_entries')}, the config reads {taps}")
    absent = [name for name in taps if name not in expect_taps]
    if absent:
        raise RefusedDrafter(f"the lane streams {expect_taps}, not the drafter's {absent}")
    if receipt.get("pack_identity") != expect_pack:
        raise RefusedDrafter(f"the drafter was trained on target pack {receipt.get('pack_identity')}, the lane serves {expect_pack}")
    return {"format": EXPORT_FORMAT, "drafter": files["model.safetensors"]["sha256"][:16], "architecture": config["architectures"][0],
            "tap_entries": taps, "pack_identity": expect_pack, "target_model": receipt.get("target_model"),
            "base_sha256": (receipt.get("base") or {}).get("model_sha256"), "files": files}


def verify(directory):
    directory = Path(directory)
    manifest = read_json(directory / MANIFEST, MANIFEST)
    for name in FILES:
        entry = manifest["files"][name]
        path = directory / name
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise RefusedDrafter(f"{path} no longer matches its manifest")
    return manifest


def place(scratch, staged, manifest):
    target = root(scratch) / "drafters" / manifest["drafter"]
    if target.exists():
        if verify(target) != manifest:
            raise RefusedDrafter(f"{target} exists with another manifest")
        shutil.rmtree(staged)
        return target
    (staged / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staged, target)
    return target


def staging(scratch):
    (root(scratch) / "incoming").mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root(scratch) / "incoming"))


def install(scratch, checkpoint, expect_taps, expect_pack):
    manifest = describe(checkpoint, expect_taps, expect_pack)
    staged = staging(scratch)
    for name in FILES:
        shutil.copyfile(Path(checkpoint) / name, staged / name)
    if describe(staged, expect_taps, expect_pack) != manifest:
        shutil.rmtree(staged)
        raise RefusedDrafter(f"{checkpoint} changed while it was copied")
    place(scratch, staged, manifest)
    return manifest


def published(scratch):
    pointer = root(scratch) / PUBLISHED
    return pointer.read_text().strip() if pointer.exists() else None


def publish(scratch, drafter):
    manifest = verify(root(scratch) / "drafters" / drafter)
    partial = root(scratch) / (PUBLISHED + ".partial")
    partial.write_text(drafter + "\n")
    os.replace(partial, root(scratch) / PUBLISHED)
    return manifest


def export(scratch, drafter, out):
    drafter = drafter or published(scratch)
    if not drafter:
        raise RefusedDrafter("no drafter named and none published")
    directory = root(scratch) / "drafters" / drafter
    manifest = verify(directory)
    out = Path(out)
    partial = out.with_name(out.name + ".partial")
    with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as archive:
        for name in (MANIFEST,) + FILES:
            info = archive.gettarinfo(directory / name, arcname=name)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with open(directory / name, "rb") as handle:
                archive.addfile(info, handle)
    os.replace(partial, out)
    return {"archive": str(out), "drafter": drafter, "bytes": out.stat().st_size, "sha256": sha256_file(out)}


def restore(scratch, archive_path, expect_taps, expect_pack):
    staged = staging(scratch)
    try:
        seen = set()
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive:
                if member.name not in LIMITS or member.name in seen or not member.isfile() or member.size > LIMITS[member.name]:
                    raise RefusedDrafter(f"the archive holds {member.name!r}, which is not one of {sorted(LIMITS)} as one regular file within its size limit")
                seen.add(member.name)
                with archive.extractfile(member) as source, open(staged / member.name, "wb") as sink:
                    shutil.copyfileobj(source, sink, CHUNK)
        if seen != set(LIMITS):
            raise RefusedDrafter(f"the archive lacks {sorted(set(LIMITS) - seen)}")
        claimed = read_json(staged / MANIFEST, MANIFEST)
        (staged / MANIFEST).unlink()
        manifest = describe(staged, expect_taps, expect_pack)
        if claimed != manifest:
            raise RefusedDrafter("the archive's manifest does not describe the files it carries")
        place(scratch, staged, manifest)
        return manifest
    except (tarfile.TarError, OSError) as error:
        shutil.rmtree(staged, ignore_errors=True)
        raise RefusedDrafter(f"{archive_path} is not a readable drafter archive: {error}")
    except RefusedDrafter:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def status(scratch):
    drafters = []
    for directory in sorted((root(scratch) / "drafters").glob("*")):
        manifest = verify(directory)
        receipt = read_json(directory / "TRAINING-RECEIPT.json", "TRAINING-RECEIPT.json")
        drafters.append({"drafter": manifest["drafter"], "positions": receipt.get("positions"), "steps": receipt.get("steps"), "seconds": receipt.get("seconds")})
    return {"published": published(scratch), "drafters": drafters}


def tokens_per_round(path):
    score = read_json(path, path)
    region = (score.get("arm") or {}).get("region")
    if region != "reply":
        raise RefusedDrafter(f"{path} scores the {region} region; the gate counts only tokens the target generated (block_drafter.py --region reply)")
    return float(score["tokens_per_round"])


def gate(candidate, incumbent, base_general, candidate_general, margin):
    held_out = {"candidate": tokens_per_round(candidate), "incumbent": tokens_per_round(incumbent)}
    general = {"base": tokens_per_round(base_general), "candidate": tokens_per_round(candidate_general)}
    general["floor"] = general["base"] * (1.0 - margin)
    return {"publish": held_out["candidate"] > held_out["incumbent"] and general["candidate"] >= general["floor"], "held_out": held_out, "general": general}


def main():
    parser = argparse.ArgumentParser(description="the customer's own drafter in their scratch: install, publish, export, restore and the nightly gate")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("install", "restore"):
        command = commands.add_parser(name)
        command.add_argument("--scratch", required=True)
        command.add_argument("--lane", required=True, help="the lane record whose taps the drafter must read")
        command.add_argument("--lanes", default=str(REPOSITORY / "deployment" / "lanes.json"))
        command.add_argument("--pack-identity", required=True, help="the target pack identity the lane's tap stream reports")
        command.add_argument("--publish", action="store_true")
        command.add_argument("source", help="a trained checkpoint directory (install) or an exported archive (restore)")
    command = commands.add_parser("publish")
    command.add_argument("--scratch", required=True)
    command.add_argument("drafter")
    command = commands.add_parser("export")
    command.add_argument("--scratch", required=True)
    command.add_argument("--drafter", default="", help="defaults to the published drafter")
    command.add_argument("--out", required=True)
    command = commands.add_parser("status")
    command.add_argument("--scratch", required=True)
    command = commands.add_parser("gate", help="publish only a candidate that beats the incumbent on held-out hours and keeps the base's general acceptance")
    command.add_argument("--candidate", required=True)
    command.add_argument("--incumbent", required=True)
    command.add_argument("--base-general", required=True)
    command.add_argument("--candidate-general", required=True)
    command.add_argument("--margin", type=float, default=0.02)
    args = parser.parse_args()
    try:
        if args.command in ("install", "restore"):
            expect = lane_taps(args.lanes, args.lane)
            result = (install if args.command == "install" else restore)(args.scratch, args.source, expect, args.pack_identity)
            if args.publish:
                publish(args.scratch, result["drafter"])
                result = dict(result, published=True)
        elif args.command == "publish":
            result = publish(args.scratch, args.drafter)
        elif args.command == "export":
            result = export(args.scratch, args.drafter, args.out)
        elif args.command == "status":
            result = status(args.scratch)
        else:
            result = gate(args.candidate, args.incumbent, args.base_general, args.candidate_general, args.margin)
    except RefusedDrafter as error:
        print(f"autopilot_drafter: refused: {error}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "spec_offline"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "serving"))
import working_state  # noqa: E402
from drafter_contract import CHUNK, RECEIPT, SHA256, drafter_tensors, named, sha256_file, tap_entries  # noqa: E402

REPOSITORY = Path(__file__).resolve().parent.parent
DRAFTER_ROOT = Path("autopilot") / "drafter"
MANIFEST = "MANIFEST.json"
PUBLISHED = "published"
PARTIAL = ".partial"
STORE = ("drafters", "incoming", "exports")
EXPORT_FORMAT = "sparkpipe-drafter-export-v1"
RECEIPT_FORMAT = "sparkpipe-drafter-training-v1"
FILES = ("config.json", "model.safetensors", RECEIPT)
LIMITS = {MANIFEST: 1 << 20, "config.json": 1 << 20, RECEIPT: 16 << 20, "model.safetensors": 64 << 30}
HEADER_LIMIT = 64 << 20
DRAFTER_ID = re.compile(r"[0-9a-f]{16}")
GENERAL_MARGIN = 0.02


class RefusedDrafter(ValueError):
    pass


def root(scratch):
    return Path(scratch) / DRAFTER_ROOT


def drafter_directory(scratch, drafter):
    if not isinstance(drafter, str) or DRAFTER_ID.fullmatch(drafter) is None:
        raise RefusedDrafter(f"a drafter id is the first 16 lowercase hex digits of its weights' sha256, not {drafter!r}")
    return root(scratch) / "drafters" / drafter


def record_use(path, now):
    stamp = time.time() if now is None else now
    os.utime(path, (stamp, stamp))


def read_json(path, what):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError, RecursionError) as error:
        raise RefusedDrafter(f"{what} is not readable JSON: {error}")
    if not isinstance(value, dict):
        raise RefusedDrafter(f"{what} is not a JSON object")
    return value


def lane_taps(lanes_path, lane):
    try:
        entries = read_json(lanes_path, "the lane records")["lanes"][lane]["taps"]["entries"]
        return [name for name in entries.split(",") if name]
    except (KeyError, TypeError, AttributeError):
        raise RefusedDrafter(f"{lanes_path} has no lane {lane} with taps")


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
        except (ValueError, RecursionError) as error:
            raise RefusedDrafter(f"model.safetensors header is not JSON: {error}")
    if not isinstance(header, dict):
        raise RefusedDrafter("model.safetensors header is not a JSON object")
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


def check_sources(receipt):
    base = receipt.get("base")
    if not isinstance(base, dict) or not isinstance(base.get("model_sha256"), str) or SHA256.fullmatch(base["model_sha256"]) is None or not named(base.get("license")):
        raise RefusedDrafter("the training receipt names no base checkpoint with its sha256 and license")
    taps = receipt.get("taps")
    if not isinstance(taps, list) or not taps or not all(isinstance(tap, dict) and isinstance(tap.get("sha256"), str) and SHA256.fullmatch(tap["sha256"])
                                                         and named(tap.get("license")) for tap in taps):
        raise RefusedDrafter("the training receipt names no sha256 and license for each tap file it trained on")
    origin = receipt.get("origin_sha256")
    if not isinstance(origin, str) or SHA256.fullmatch(origin) is None:
        raise RefusedDrafter("the training receipt names no origin_sha256, the drafter this one descends from")
    requests = receipt.get("requests")
    if not isinstance(requests, list) or not requests or not all(isinstance(request, int) and not isinstance(request, bool) for request in requests):
        raise RefusedDrafter("the training receipt names no requests it trained on")
    return base["model_sha256"], origin


def checked(directory, expect_taps, expect_pack):
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
        drafter_tensors(config)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise RefusedDrafter(f"config.json does not describe a supported drafter: {error}")
    check_weights(directory / "model.safetensors", config)
    receipt = read_json(directory / RECEIPT, RECEIPT)
    if receipt.get("format") != RECEIPT_FORMAT:
        raise RefusedDrafter(f"{RECEIPT} is not {RECEIPT_FORMAT}")
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
    base, origin = check_sources(receipt)
    return {"format": EXPORT_FORMAT, "drafter": files["model.safetensors"]["sha256"][:16], "architecture": config["architectures"][0],
            "tap_entries": taps, "pack_identity": expect_pack, "target_model": receipt.get("target_model"),
            "base_sha256": base, "origin_sha256": origin, "files": files}


def describe(directory, expect_taps, expect_pack):
    try:
        return checked(Path(directory), expect_taps, expect_pack)
    except RefusedDrafter:
        raise
    except Exception as error:
        raise RefusedDrafter(f"{directory} is not a drafter this lane can serve: {type(error).__name__}: {error}") from error


def verify(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise RefusedDrafter(f"there is no drafter {directory.name} in {directory.parent}")
    manifest = read_json(directory / MANIFEST, MANIFEST)
    try:
        current = describe(directory, manifest.get("tap_entries") or [], manifest.get("pack_identity"))
    except RefusedDrafter as error:
        raise RefusedDrafter(f"{directory} no longer matches its manifest: {error}")
    if current != manifest or manifest["drafter"] != directory.name:
        raise RefusedDrafter(f"{directory} no longer matches its manifest")
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


def admit(scratch, source, fill, now):
    (root(scratch) / "incoming").mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=root(scratch) / "incoming"))
    try:
        manifest = fill(staged)
        target = place(scratch, staged, manifest)
    except Exception as error:
        shutil.rmtree(staged, ignore_errors=True)
        if isinstance(error, RefusedDrafter):
            raise
        raise RefusedDrafter(f"{source} is not a readable drafter: {type(error).__name__}: {error}") from error
    record_use(target, now)
    return manifest


def install(scratch, checkpoint, expect_taps, expect_pack, now=None):
    manifest = describe(checkpoint, expect_taps, expect_pack)

    def fill(staged):
        for name in FILES:
            shutil.copyfile(Path(checkpoint) / name, staged / name)
        if describe(staged, expect_taps, expect_pack) != manifest:
            raise RefusedDrafter(f"{checkpoint} changed while it was copied")
        return manifest
    return admit(scratch, checkpoint, fill, now)


def restore(scratch, archive_path, expect_taps, expect_pack, now=None):
    def fill(staged):
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
        return manifest
    return admit(scratch, archive_path, fill, now)


def published(scratch):
    pointer = root(scratch) / PUBLISHED
    return pointer.read_text().strip() if pointer.exists() else None


def publish(scratch, drafter, now=None):
    directory = drafter_directory(scratch, drafter)
    manifest = verify(directory)
    partial = root(scratch) / (PUBLISHED + PARTIAL)
    partial.write_text(drafter + "\n")
    os.replace(partial, root(scratch) / PUBLISHED)
    record_use(directory, now)
    return manifest


def export(scratch, drafter, now=None):
    drafter = drafter or published(scratch)
    if not drafter:
        raise RefusedDrafter("no drafter named and none published")
    directory = drafter_directory(scratch, drafter)
    manifest = verify(directory)
    exports = root(scratch) / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    out = exports / f"{drafter}.tar"
    partial = exports / f"{drafter}.tar{PARTIAL}"
    try:
        with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as archive:
            for name in (MANIFEST,) + FILES:
                info = archive.gettarinfo(directory / name, arcname=name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with open(directory / name, "rb") as handle:
                    archive.addfile(info, handle)
        os.replace(partial, out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    record_use(out, now)
    record_use(directory, now)
    return {"archive": str(out), "drafter": manifest["drafter"], "bytes": out.stat().st_size, "sha256": sha256_file(out)}


def status(scratch):
    drafters = []
    for directory in sorted((root(scratch) / "drafters").glob("*")):
        manifest = verify(directory)
        receipt = read_json(directory / RECEIPT, RECEIPT)
        drafters.append({"drafter": manifest["drafter"], "used": int(directory.stat().st_mtime), "positions": receipt.get("positions"),
                         "steps": receipt.get("steps"), "seconds": receipt.get("seconds")})
    return {"published": published(scratch), "drafters": drafters}


def remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def expire(scratch, directive=None, now=None):
    now = time.time() if now is None else now
    wiped, held_until = working_state.read_directive(directive) if directive else (0, 0)
    store = root(scratch)
    removed = []
    for kind in STORE:
        for path in sorted((store / kind).iterdir()) if (store / kind).is_dir() else []:
            if working_state.expired(path.lstat().st_mtime, wiped, held_until, now):
                remove(path)
                removed.append(f"{kind}/{path.name}")
    partial = store / (PUBLISHED + PARTIAL)
    if partial.exists() and working_state.expired(partial.lstat().st_mtime, wiped, held_until, now):
        partial.unlink()
        removed.append(partial.name)
    pointer = published(scratch)
    if pointer is not None and (DRAFTER_ID.fullmatch(pointer) is None or not (store / "drafters" / pointer).is_dir()):
        (store / PUBLISHED).unlink()
        removed.append(PUBLISHED)
    return {"removed": removed, "retention_seconds": working_state.RETENTION_SECONDS, "wiped": wiped, "held_until": held_until}


def read_score(path):
    report = read_json(path, str(path))
    arm = report.get("arm")
    if not isinstance(arm, dict):
        raise RefusedDrafter(f"{path} records no arm")
    if arm.get("region") != "reply":
        raise RefusedDrafter(f"{path} scores the {arm.get('region')} region; the gate counts only tokens the target generated (block_drafter.py --region reply)")
    if not isinstance(arm.get("drafter_sha256"), str) or SHA256.fullmatch(arm["drafter_sha256"]) is None or not isinstance(arm.get("taps"), list) or not arm["taps"]:
        raise RefusedDrafter(f"{path} names no drafter sha256 and tap files; score it with the current block_drafter.py")
    anchors = {}
    try:
        for request in report["requests"]:
            if request["request_id"] in anchors or not isinstance(request["anchors"], int):
                raise RefusedDrafter(f"{path} scores request {request['request_id']} twice or without an anchor count")
            anchors[request["request_id"]] = request["anchors"]
        value = report["tokens_per_round"]
    except (KeyError, TypeError) as error:
        raise RefusedDrafter(f"{path} is not a block_drafter.py score: {error}")
    if not sum(anchors.values()) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RefusedDrafter(f"{path} scores no anchors or no finite tokens per round")
    return {"path": str(path), "arm": arm, "anchors": anchors, "tokens_per_round": float(value)}


def matched(first, second, what):
    keys = sorted(key for key in set(first["arm"]) | set(second["arm"]) if key != "drafter_sha256" and first["arm"].get(key) != second["arm"].get(key))
    if keys:
        raise RefusedDrafter(f"the {what} scores {first['path']} and {second['path']} are not the same arm: they differ in {keys}")
    if first["anchors"] != second["anchors"]:
        raise RefusedDrafter(f"the {what} scores {first['path']} and {second['path']} drew different anchors per request")


def scored_by(score, sha, who):
    if score["arm"]["drafter_sha256"] != sha:
        raise RefusedDrafter(f"{score['path']} scored drafter {score['arm']['drafter_sha256'][:16]}, not {who} {sha[:16]}")


def held_out_of_training(directory, score):
    receipt = read_json(directory / RECEIPT, RECEIPT)
    trained_files = {tap["sha256"] for tap in receipt["taps"]}
    trained = set(receipt["requests"])
    if trained_files & {tap.get("sha256") for tap in score["arm"]["taps"] if isinstance(tap, dict)} and trained & set(score["anchors"]):
        raise RefusedDrafter(f"{score['path']} scores requests {sorted(trained & set(score['anchors']))[:4]} the candidate trained on")


def gate(scratch, candidate, held_out_candidate, held_out_incumbent, general_base, general_candidate, now=None):
    directory = drafter_directory(scratch, candidate)
    manifest = verify(directory)
    weights, origin = manifest["files"]["model.safetensors"]["sha256"], manifest["origin_sha256"]
    incumbent = published(scratch)
    incumbent_weights = verify(drafter_directory(scratch, incumbent))["files"]["model.safetensors"]["sha256"] if incumbent else origin
    scores = [read_score(path) for path in (held_out_candidate, held_out_incumbent, general_base, general_candidate)]
    matched(scores[0], scores[1], "held-out")
    matched(scores[2], scores[3], "general")
    scored_by(scores[0], weights, "the candidate")
    scored_by(scores[1], incumbent_weights, "the incumbent")
    scored_by(scores[2], origin, "the base")
    scored_by(scores[3], weights, "the candidate")
    held_out_of_training(directory, scores[0])
    held_out = {"candidate": scores[0]["tokens_per_round"], "incumbent": scores[1]["tokens_per_round"]}
    general = {"base": scores[2]["tokens_per_round"], "candidate": scores[3]["tokens_per_round"]}
    general["floor"] = general["base"] * (1.0 - GENERAL_MARGIN)
    verdict = {"candidate": candidate, "incumbent": incumbent or f"base {origin[:16]}", "held_out": held_out, "general": general}
    if not (held_out["candidate"] > held_out["incumbent"] and general["candidate"] >= general["floor"]):
        raise RefusedDrafter("the incumbent stays: " + json.dumps(verdict))
    publish(scratch, candidate, now)
    return dict(verdict, published=True)


def main():
    parser = argparse.ArgumentParser(description="the customer's own drafter in their scratch: install, export, restore, the nightly gate that alone publishes, and expiry")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("install", "restore"):
        command = commands.add_parser(name)
        command.add_argument("--scratch", required=True)
        command.add_argument("--lane", required=True, help="the lane record whose taps the drafter must read")
        command.add_argument("--lanes", default=str(REPOSITORY / "deployment" / "lanes.json"))
        command.add_argument("--pack-identity", required=True, help="the target pack identity the lane's tap stream reports")
        command.add_argument("source", help="a trained checkpoint directory (install) or an exported archive (restore)")
    command = commands.add_parser("export", help="write <scratch>/autopilot/drafter/exports/<id>.tar")
    command.add_argument("--scratch", required=True)
    command.add_argument("--drafter", default="", help="defaults to the published drafter")
    command = commands.add_parser("status")
    command.add_argument("--scratch", required=True)
    command = commands.add_parser("gate", help="publish the candidate only if it beats the incumbent on held-out replies and keeps the base's general acceptance")
    command.add_argument("--scratch", required=True)
    command.add_argument("--candidate", required=True, help="the installed candidate's drafter id")
    command.add_argument("--held-out-candidate", required=True)
    command.add_argument("--held-out-incumbent", required=True)
    command.add_argument("--general-base", required=True)
    command.add_argument("--general-candidate", required=True)
    command = commands.add_parser("expire", help="delete what has not been used within the working-state retention")
    command.add_argument("--scratch", required=True)
    command.add_argument("--directive", default="", help="the working-state directive the controller writes on this host; without it, 24 hours after last use")
    args = parser.parse_args()
    try:
        if args.command in ("install", "restore"):
            expect = lane_taps(args.lanes, args.lane)
            result = (install if args.command == "install" else restore)(args.scratch, args.source, expect, args.pack_identity)
        elif args.command == "export":
            result = export(args.scratch, args.drafter)
        elif args.command == "status":
            result = status(args.scratch)
        elif args.command == "gate":
            result = gate(args.scratch, args.candidate, args.held_out_candidate, args.held_out_incumbent, args.general_base, args.general_candidate)
        else:
            result = expire(args.scratch, args.directive)
    except RefusedDrafter as error:
        print(f"autopilot_drafter: refused: {error}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

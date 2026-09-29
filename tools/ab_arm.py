#!/usr/bin/env python3
"""Quantization A/B arm descriptor (sparkpipe-quant-arm-v1).

Python mirror of include/sparkpipe/spark_quant_arm.h: the same members, the
same refusals, the same canonical JSON and therefore the same arm_digest.
tests/test_ab_arm.py cross-checks both implementations on shared fixtures.

usage:
  ab_arm.py digest ARM_JSON
  ab_arm.py canonical ARM_JSON
  ab_arm.py axes ARM_A ARM_B        which axes (spine, E, K, D) differ
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

FORMAT = "sparkpipe-quant-arm-v1"
MAX_RANKS = 64
TEXT_BYTES = 256
TOKEN_BYTES = 32
ID_BYTES = 256

ROOT = ("format", "arm_id", "model", "revision", "topology", "spine", "expert", "kv", "drafter",
        "pack_sha256", "artifacts")
TOPOLOGY = ("tp", "pp", "kv_shard")
SPINE = ("frame", "source", "spine_digest")
EXPERT = ("codec", "label", "producer", "source", "recipe_sha256")
KV = ("latent", "index", "state", "group", "mode")
DRAFTER = ("kind", "label", "codec", "head", "sidecar_sha256")
ARTIFACTS = ("module_archive_sha256", "driver_sha256", "adapter_sha256")
EXPERT_CODECS = ("bf16", "fp8", "nvfp4", "mxfp4", "int8", "int7", "int6")
PRODUCERS = ("publisher", "community", "experiment")
KV_LATENT = ("bf16", "fp8", "mxfp4")
KV_INDEX = ("bf16", "fp8")
KV_STATE = ("fp32", "bf16")
KV_MODES = ("sim", "store")
DRAFTER_KINDS = ("none", "lookup", "mtp", "dflash", "oracle", "adversary", "random")
DRAFTER_HEADS = ("bf16", "fp8")
WEIGHTED_DRAFTERS = ("mtp", "dflash")
HEX = re.compile(r"[0-9a-f]{64}\Z")
LABEL = re.compile(r"[a-z0-9]+\Z")
MODEL = re.compile(r"[a-z0-9_-]+\Z")
FRAME = re.compile(r"S[0-9]+\Z")
PLAIN = re.compile(r"[\x20-\x7e]+\Z")
AXES = ("spine", "E", "K", "D")


class ArmError(ValueError):
    pass


def _object(value, name, members):
    if not isinstance(value, dict):
        raise ArmError(f"arm: {name} must be an object")
    if set(value) != set(members) or len(value) != len(members):
        raise ArmError(f"arm: {name} members are not exactly the {FORMAT} set")
    return value


def _text(value, name, limit=TEXT_BYTES):
    if not isinstance(value, str):
        raise ArmError(f"arm: {name} must be a string")
    if not value or len(value) > limit or not PLAIN.match(value) or '"' in value or "\\" in value:
        raise ArmError(f"arm: {name} must be 1-{limit} printable ASCII characters without quote or backslash")
    return value


def _enum(value, name, allowed):
    _text(value, name, TOKEN_BYTES)
    if value not in allowed:
        raise ArmError(f"arm: {name} value {value} is not a known token")
    return value


def _hex(value, name):
    _text(value, name, 64)
    if not HEX.match(value):
        raise ArmError(f"arm: {name} must be 64 lowercase hex digits")
    return value


def _unsigned(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 999999999:
        raise ArmError(f"arm: {name} must be a canonical unsigned integer")
    return value


def _hex_array(value, name, count):
    if not isinstance(value, list):
        raise ArmError(f"arm: {name} must be an array")
    if len(value) != count:
        raise ArmError(f"arm: {name} has {len(value)} entries, expected {count}")
    return [_hex(item, name) for item in value]


def _source(value):
    at = value.find("@")
    return at > 0 and at + 1 < len(value) and "@" not in value[at + 1:]


def kv_token(codec: str, group: int) -> str:
    return codec if codec == "bf16" else f"{codec}g{group}"


def expected_id(arm: dict) -> str:
    kv = arm["kv"]
    return (f"{arm['model']}.{arm['spine']['frame']}.e-{arm['expert']['label']}"
            f".k-{kv_token(kv['latent'], kv['group'])}/{kv_token(kv['index'], kv['group'])}/{kv['state']}"
            f".d-{arm['drafter']['label']}")


def _reject_duplicates(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ArmError("arm: root members are not exactly the sparkpipe-quant-arm-v1 set (duplicate member)")
        seen[key] = value
    return seen


def _reject_constant(token):
    raise ArmError(f"arm: not valid JSON ({token})")


def validate(arm) -> dict:
    _object(arm, "root", ROOT)
    fmt = _text(arm["format"], "format", TOKEN_BYTES)
    if fmt != FORMAT:
        raise ArmError(f"arm: format {fmt} is not {FORMAT}")
    _text(arm["arm_id"], "arm_id", ID_BYTES)
    _text(arm["model"], "model", TOKEN_BYTES)
    _text(arm["revision"], "revision")
    topology = _object(arm["topology"], "topology", TOPOLOGY)
    tp = _unsigned(topology["tp"], "tp")
    pp = _unsigned(topology["pp"], "pp")
    kv_shard = _unsigned(topology["kv_shard"], "kv_shard")
    if tp == 0 or pp == 0 or tp > MAX_RANKS or tp * pp > MAX_RANKS:
        raise ArmError(f"arm: topology tp={tp} pp={pp} is outside 1..{MAX_RANKS} ranks")
    ranks = tp * pp
    spine = _object(arm["spine"], "spine", SPINE)
    _text(spine["frame"], "frame", TOKEN_BYTES)
    _text(spine["source"], "source")
    _hex_array(spine["spine_digest"], "spine_digest", ranks)
    expert = _object(arm["expert"], "expert", EXPERT)
    _enum(expert["codec"], "codec", EXPERT_CODECS)
    _text(expert["label"], "label", TOKEN_BYTES)
    _enum(expert["producer"], "producer", PRODUCERS)
    _text(expert["source"], "source")
    if expert["recipe_sha256"] is not None:
        _hex(expert["recipe_sha256"], "recipe_sha256")
    kv = _object(arm["kv"], "kv", KV)
    _enum(kv["latent"], "latent", KV_LATENT)
    _enum(kv["index"], "index", KV_INDEX)
    _enum(kv["state"], "state", KV_STATE)
    _unsigned(kv["group"], "group")
    _enum(kv["mode"], "mode", KV_MODES)
    drafter = _object(arm["drafter"], "drafter", DRAFTER)
    _enum(drafter["kind"], "kind", DRAFTER_KINDS)
    _text(drafter["label"], "label", TOKEN_BYTES)
    _text(drafter["codec"], "codec", TOKEN_BYTES)
    _text(drafter["head"], "head", TOKEN_BYTES)
    if not isinstance(drafter["sidecar_sha256"], list):
        raise ArmError("arm: sidecar_sha256 must be an array")
    if len(drafter["sidecar_sha256"]) > MAX_RANKS:
        raise ArmError(f"arm: sidecar_sha256 has more than {MAX_RANKS} entries")
    for item in drafter["sidecar_sha256"]:
        _hex(item, "sidecar_sha256")
    _hex_array(arm["pack_sha256"], "pack_sha256", ranks)
    artifacts = _object(arm["artifacts"], "artifacts", ARTIFACTS)
    for name in ("module_archive_sha256", "driver_sha256", "adapter_sha256"):
        _hex(artifacts[name], name)
    if kv_shard > 1:
        raise ArmError("arm: topology kv_shard must be 0 or 1")
    if not MODEL.match(arm["model"]):
        raise ArmError("arm: model must be lowercase letters, digits, dash or underscore")
    if not FRAME.match(spine["frame"]):
        raise ArmError("arm: spine frame must be S followed by digits")
    if not _source(spine["source"]) or not _source(expert["source"]):
        raise ArmError("arm: spine and expert sources must be repo@revision")
    if not LABEL.match(expert["label"]) or not expert["label"].startswith(expert["codec"]):
        raise ArmError(f"arm: expert label {expert['label']} must be lowercase alphanumeric and start with the codec {expert['codec']}")
    if expert["producer"] == "experiment" and expert["recipe_sha256"] is None:
        raise ArmError("arm: an experiment expert codec needs its recipe_sha256")
    bf16_kv = kv["latent"] == "bf16" and kv["index"] == "bf16"
    if bf16_kv and kv["group"] != 0:
        raise ArmError("arm: kv group must be 0 when latent and index are bf16")
    if not bf16_kv and kv["group"] not in (32, 64, 128):
        raise ArmError(f"arm: kv group {kv['group']} must be 32, 64 or 128 for a quantized latent or index")
    if kv["latent"] == "mxfp4" and kv["group"] != 32:
        raise ArmError("arm: an mxfp4 latent needs group 32")
    if bf16_kv and kv["state"] == "fp32" and kv["mode"] != "store":
        raise ArmError("arm: the unquantized kv reference must use mode store")
    weighted = drafter["kind"] in WEIGHTED_DRAFTERS
    if not weighted and (drafter["codec"] != "none" or drafter["head"] != "none" or drafter["sidecar_sha256"]):
        raise ArmError(f"arm: drafter {drafter['kind']} carries no weights, so codec and head must be none and sidecar_sha256 empty")
    if weighted and (drafter["codec"] not in EXPERT_CODECS or drafter["head"] not in DRAFTER_HEADS
                     or len(drafter["sidecar_sha256"]) != ranks):
        raise ArmError(f"arm: drafter {drafter['kind']} needs a weight codec, a bf16 or fp8 head and one sidecar sha256 per rank")
    if not LABEL.match(drafter["label"]) or not drafter["label"].startswith(drafter["kind"]):
        raise ArmError(f"arm: drafter label {drafter['label']} must be lowercase alphanumeric and start with the kind {drafter['kind']}")
    expected = expected_id(arm)
    if expected != arm["arm_id"]:
        raise ArmError(f"arm: arm_id {arm['arm_id']} does not match its fields (expected {expected})")
    return arm


def parse_text(text: str) -> dict:
    try:
        arm = json.loads(text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant,
                         parse_float=lambda token: _reject_constant(token))
    except json.JSONDecodeError as error:
        raise ArmError(f"arm: not valid JSON ({error})") from error
    return validate(arm)


def load(path) -> dict:
    return parse_text(Path(path).read_text(encoding="ascii"))


def canonical(arm: dict) -> bytes:
    return json.dumps(validate(arm), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def arm_digest(arm: dict) -> str:
    return hashlib.sha256(canonical(arm)).hexdigest()


def pack_set_sha256(arm: dict) -> str:
    return hashlib.sha256(json.dumps(arm["pack_sha256"], separators=(",", ":")).encode("ascii")).hexdigest()


def kv_text(arm: dict) -> str:
    kv = arm["kv"]
    return f"{kv['latent']}/{kv['index']}/{kv['state']}/{kv['group']}/{kv['mode']}"


def axis_members(arm: dict) -> dict:
    return {
        "spine": (arm["spine"]["frame"], arm["spine"]["source"], tuple(arm["spine"]["spine_digest"]), arm["revision"]),
        "E": tuple(arm["expert"][key] for key in EXPERT),
        "K": tuple(arm["kv"][key] for key in KV),
        "D": (tuple(arm["drafter"][key] for key in ("kind", "label", "codec", "head")),
              tuple(arm["drafter"]["sidecar_sha256"])),
    }


def differing_axes(left: dict, right: dict) -> list:
    a, b = axis_members(left), axis_members(right)
    return [axis for axis in AXES if a[axis] != b[axis]]


def main(argv) -> int:
    if len(argv) == 3 and argv[1] in ("digest", "canonical"):
        arm = load(argv[2])
        if argv[1] == "digest":
            print(arm_digest(arm))
        else:
            sys.stdout.write(canonical(arm).decode("ascii"))
        return 0
    if len(argv) == 4 and argv[1] == "axes":
        print(" ".join(differing_axes(load(argv[2]), load(argv[3]))) or "none")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except ArmError as error:
        print(f"ab_arm: REFUSED: {error}", file=sys.stderr)
        sys.exit(1)

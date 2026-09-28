#!/usr/bin/env python3
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACTS = {
    "ling": REPOSITORY / "model_contracts/ling_authoritative.json",
    "lingfin": REPOSITORY / "model_contracts/lingfin_authoritative.json",
}
MODELING = REPOSITORY / "model_contracts/references/modeling_ling_bailing_moe_v3.py"
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
SHARD = re.compile(r"model(-mtp)?-(\d{5})-of-(\d{5})\.safetensors")


def release_problems(label, release, modeling_sha256):
    problems = []
    if not HEX40.fullmatch(release.get("revision", "")):
        problems.append(f"{label}: revision is not a 40-hex commit")
    shards = release.get("shards", {})
    if release.get("shard_count") != len(shards):
        problems.append(f"{label}: shard_count {release.get('shard_count')} != {len(shards)} pinned shards")
    if release.get("shard_total_bytes") != sum(entry["bytes"] for entry in shards.values()):
        problems.append(f"{label}: shard_total_bytes differs from the pinned shard sizes")
    stacks = {}
    for name, entry in shards.items():
        match = SHARD.fullmatch(name)
        if match is None:
            problems.append(f"{label}: {name} is not a safetensors shard name")
            continue
        stacks.setdefault(match.group(1), []).append((int(match.group(2)), int(match.group(3))))
        if not HEX64.fullmatch(entry.get("sha256", "")) or entry.get("bytes", 0) <= 0:
            problems.append(f"{label}: {name} lacks a sha256 or a positive size")
    for prefix, numbers in stacks.items():
        total = numbers[0][1]
        indices = sorted(index for index, _ in numbers)
        if indices not in (list(range(1, total + 1)), list(range(total))) or any(count != total for _, count in numbers):
            problems.append(f"{label}: shard set{prefix or ''} is not a contiguous {total}-of-{total} set")
    files = release.get("file_sha256", {})
    for name, digest in files.items():
        if not HEX64.fullmatch(digest):
            problems.append(f"{label}: {name} sha256 is not 64 hex digits")
    if files.get("modeling_bailing_moe_v3.py") != modeling_sha256:
        problems.append(f"{label}: modeling sha256 differs from the committed reference modeling")
    return problems


def contract_problems(name, contract, modeling_sha256):
    problems = []
    if not contract.get("freeze_status", "").startswith("FROZEN"):
        problems.append(f"{name}: freeze_status is not FROZEN")
    source = contract.get("source", {})
    if source.get("revision") != contract.get("source_revision"):
        problems.append(f"{name}: source.revision != source_revision")
    if source.get("repository") != contract.get("model_id"):
        problems.append(f"{name}: source.repository != model_id")
    problems += release_problems(f"{name}.source", source, modeling_sha256)
    for variant, release in contract.get("variants", {}).items():
        problems += release_problems(f"{name}.variants.{variant}", release, modeling_sha256)
    tokenizer = contract.get("tokenizer", {})
    for key, file_name in (("tokenizer_json_sha256", "tokenizer.json"), ("tokenizer_config_sha256", "tokenizer_config.json"), ("chat_template_sha256", "chat_template.jinja")):
        if tokenizer.get(key) != source.get("file_sha256", {}).get(file_name):
            problems.append(f"{name}: tokenizer.{key} differs from source.file_sha256[{file_name}]")
    if tokenizer.get("end_of_text_token_id") != contract["model"]["end_of_text_token_id"]:
        problems.append(f"{name}: tokenizer eos differs from model eos")
    return problems


def mutated(contract, edit):
    copy_of = copy.deepcopy(contract)
    edit(copy_of)
    return copy_of


def main():
    modeling_sha256 = hashlib.sha256(MODELING.read_bytes()).hexdigest()
    contracts = {name: json.loads(path.read_text()) for name, path in CONTRACTS.items()}
    failures = []
    for name, contract in contracts.items():
        failures += contract_problems(name, contract, modeling_sha256)
    if contracts["ling"]["source"]["shard_count"] != 24 or contracts["ling"]["variants"]["fp8"]["shard_count"] != 24 or contracts["lingfin"]["source"]["shard_count"] != 65:
        failures.append("shard counts differ from the published releases (24 bf16, 24 fp8, 65 Fin)")
    ling = contracts["ling"]
    first = sorted(ling["source"]["shards"])[0]
    last = sorted(ling["source"]["shards"])[-1]
    controls = {
        "dropped shard": lambda c: c["source"]["shards"].pop(last),
        "bad digest": lambda c: c["source"]["shards"][first].update(sha256="0" * 63),
        "size drift": lambda c: c["source"]["shards"][first].update(bytes=c["source"]["shards"][first]["bytes"] + 1),
        "revision drift": lambda c: c.update(source_revision="e" * 40),
        "modeling drift": lambda c: c["source"]["file_sha256"].update({"modeling_bailing_moe_v3.py": "1" * 64}),
        "tokenizer drift": lambda c: c["tokenizer"].update(tokenizer_json_sha256="2" * 64),
        "fp8 count drift": lambda c: c["variants"]["fp8"].update(shard_count=23),
        "unfrozen": lambda c: c.update(freeze_status="pre-freeze"),
    }
    for label, edit in controls.items():
        if not contract_problems("ling", mutated(ling, edit), modeling_sha256):
            failures.append(f"negative control not convicted: {label}")
    if failures:
        for failure in failures:
            print("FAIL", failure)
        return 1
    print(f"PASS ling contracts frozen ({sum(len(c['source']['shards']) for c in contracts.values()) + len(ling['variants']['fp8']['shards'])} shard pins, {len(controls)} negative controls convicted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

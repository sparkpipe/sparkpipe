#!/usr/bin/env python3
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEMPLATE_SYMBOLS = [
    "SparkServingAdapterTemplateLoadTpCollective",
    "SparkServingAdapterTemplateReservePending",
    "SparkServingAdapterTemplateLoadDriver",
]

FAMILIES = {
    "dsv4_resident_decode_stage": None,
    "glm52_resident_decode_stage": None,
    "qwen38_27b_resident_decode_stage": None,
    "k3_resident_decode_stage": "exempt:pack-API-module (verified not the pasted shape; dry-final report)",
    "glm5_next_resident_decode_stage": "wave-2 (post-closeout)",
    "qwen38_max_resident_decode_stage": "wave-2",
    "qwen4_flash_resident_decode_stage": "wave-2",
    "ling_resident_decode_stage": None,
    "laguna_resident_decode_stage": "wave-2 (mod-infra-mid migration notes: TP-config loader + MASK_CONDITIONAL/ADAPTIVE_COMBOS policies)",
}

FORK_SIGNATURES = [
    re.compile(r'json_member[^;]*"tp_collective"[^;]*\)', re.I),
    re.compile(r'find_object[^;]*"tp_collective"', re.I),
    re.compile(r'listen_port[^;]{0,80}connect_timeout_milli[^;]{0,40}==', re.I),
]

PACK_SYNTHESIZE_CORE = "sparkpipe/spark_pack_synthesize_common.h"
PACK_SYNTHESIZE_OFFENDERS = {
    "dsv41_flash_pack_synthesize.c": "own generator",
    "gemma4_pack_synthesize.c": "own generator",
    "muse_glimmer_pack_synthesize.c": "own generator",
}

SHARED_HEADERS = [
    ("model-families/*/include/sparkpipe/spark_*_work_control.h", ("sparkpipe/spark_work_control_api.h",), {}),
    ("modules/*/include/sparkpipe/spark_*_batch_tuning.h", ("sparkpipe/spark_batch_variant_tuning_common.h", "spark_glm_batch_tuning.h"), {
        "spark_dsv4_batch_tuning.h": "own bucket ladder",
        "spark_k3_batch_tuning.h": "own bucket ladder",
    }),
]


def ratchet(label, name, adopted, offenders):
    if adopted and name in offenders:
        return [f"FAIL {name}: now {label}; remove it from the offender list"]
    if not adopted and name not in offenders:
        return [f"FAIL {name}: not {label}"]
    if not adopted:
        print(f"  known-offender {name}: {offenders[name]}")
    return []


def check_adapters():
    failures = []
    for family, wave in sorted(FAMILIES.items()):
        adapter = ROOT / "modules" / family / "source"
        sources = list(adapter.glob("*serving_adapter.c")) if adapter.exists() else []
        if not sources:
            continue
        text = "\n".join(p.read_text(errors="replace") for p in sources)
        consumed = any(sym in text for sym in TEMPLATE_SYMBOLS)
        forked = any(rx.search(text) for rx in FORK_SIGNATURES)
        if consumed and not forked:
            continue
        if wave is None:
            failures.append(
                f"FAIL {family}: adapter does not consume the template"
                f" (symbols {'present' if consumed else 'absent'},"
                f" local-parser signature {'present' if forked else 'absent'})"
                " — adopted families must not regress to private copies"
            )
        elif wave.startswith("exempt"):
            print(f"  exempt {family}: {wave}")
        else:
            print(f"  known-offender {family}: adoption scheduled in {wave}")
    return failures


def check_pack_synthesize():
    failures = []
    paths = sorted(ROOT.glob("modules/*/tools/*pack_synthesize.c"))
    if not paths:
        return ["FAIL no pack synthesizers found under modules/*/tools"]
    for path in paths:
        failures += ratchet(f"including {PACK_SYNTHESIZE_CORE}", path.name, PACK_SYNTHESIZE_CORE in path.read_text(errors="replace"), PACK_SYNTHESIZE_OFFENDERS)
    return failures


def check_shared_headers():
    failures = []
    for pattern, shared, offenders in SHARED_HEADERS:
        paths = sorted(ROOT.glob(pattern))
        if not paths:
            failures.append(f"FAIL no headers match {pattern}")
        for path in paths:
            text = path.read_text(errors="replace")
            failures += ratchet(f"including {' or '.join(shared)}", path.name, any(name in text for name in shared), offenders)
    return failures


def check_firmware_structs():
    bodies = collections.defaultdict(list)
    for path in sorted(ROOT.glob("modules/*/include/sparkpipe/spark_*_resident_decode_stage_firmware.h")):
        family = path.name[len("spark_"):-len("_resident_decode_stage_firmware.h")]
        text = path.read_text(errors="replace")
        camel = collections.Counter(re.findall(r"Spark([A-Z][A-Za-z0-9_]*?)ResidentDecodeStage", text)).most_common(1)
        if not camel:
            continue
        for match in re.finditer(r"typedef struct \w+\s*\{.*?\}\s*(\w+)\s*;", text, re.S):
            body = match.group(0).replace("Spark" + camel[0][0], "SparkFAMILY").replace("SPARK_" + family.upper() + "_", "SPARK_FAMILY_")
            bodies[re.sub(r"\s+", " ", body)].append(f"{path.name}:{match.group(1)}")
    for path in sorted(ROOT.glob("include/sparkpipe/family/abi/*.h")):
        text = re.sub(r"SPARK_ABI_TYPE\((\w+)\)", r"SparkFAMILY\1", path.read_text(errors="replace"))
        for match in re.finditer(r"typedef struct \w+\s*\{.*?\}\s*(\w+)\s*;", text, re.S):
            bodies[re.sub(r"\s+", " ", match.group(0))].append(f"{path.name}:{match.group(1)}")
    return [f"FAIL identical firmware ABI structs {', '.join(names)}: include the family/abi template instead of a copy" for names in bodies.values() if len(names) > 1]


def main():
    print("template-adoption gate:")
    failures = check_adapters() + check_pack_synthesize() + check_shared_headers() + check_firmware_structs()
    for f in failures:
        print(f"  {f}")
    if failures:
        print(f"FAIL ({len(failures)}) pattern-exists-but-not-adopted violations")
        return 1
    print("PASS shared patterns consumed (or scheduled via the ratcheting offender set)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

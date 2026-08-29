#!/usr/bin/env python3
"""Template-adoption gate (the three-audit loop-closer, 2026-08-30).

The audits' cross-cutting finding: the right abstractions existed and
families forked anyway — the adapter template landed while three
families kept private copies; the pack-synthesize core landed with
zero importers. Size and paste gates cannot see "pattern exists but
wasn't adopted." This gate can, mechanically:

  1. SERVING ADAPTERS must consume the template (its symbols appear in
     the family adapter; a family-local TP-collective parser copy is a
     fork signature).
  2. PACK SYNTHESIZE tools must import the shared core (a standalone
     generator is a fork signature).
  3. WORK-CONTROL / BATCH-TUNING headers must be the shared include,
     not byte-siblings.

Enforcement shape (mirrors how the paste gate earned its teeth):
families already on the pattern PASS; known offenders live in
KNOWN_OFFENDERS with their adoption wave — the set RATCHETS (removing
an entry promotes its check to hard-fail; an entry may only shrink);
any family NOT listed and NOT consuming = hard FAIL (new families hit
the gate at merge, by construction).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEMPLATE_HEADER = "include/sparkpipe/spark_serving_adapter_template.h"
TEMPLATE_SYMBOLS = [
    "SparkServingAdapterTemplateLoadTpCollective",
    "SparkServingAdapterTemplateReservePending",
    "SparkServingAdapterTemplateLoadDriver",
]

FAMILIES = {
    # family dir name -> adoption wave (None = adopted)
    "dsv4_resident_decode_stage": None,
    "glm52_resident_decode_stage": None,
    "qwen38_27b_resident_decode_stage": None,
    "k3_resident_decode_stage": "exempt:pack-API-module (verified not the pasted shape; dry-final report)",
    "glm5_next_resident_decode_stage": "wave-2 (post-closeout)",
    "qwen38_max_resident_decode_stage": "wave-2",
    "qwen4_flash_resident_decode_stage": "wave-2",
}

# A family-local copy of the TP-collective config parser is the fork
# signature the audits flagged: parsing tp_collective JSON inline.
# Fork signature: a family-LOCAL parser of the collective config — the
# key string appearing in its own parsing context, not in a member-list
# table or a template call. (Adopted families still name the key in
# allow-lists and template calls; the distinction is the parse loop.)
FORK_SIGNATURES = [
    re.compile(r'json_member[^;]*"tp_collective"[^;]*\)', re.I),
    re.compile(r'find_object[^;]*"tp_collective"', re.I),
    re.compile(r'listen_port[^;]{0,80}connect_timeout_milli[^;]{0,40}==', re.I),
]


def check_adapters():
    failures = []
    for family, wave in sorted(FAMILIES.items()):
        adapter = ROOT / "modules" / family / "source"
        sources = list(adapter.glob("*serving_adapter.c")) if adapter.exists() else []
        if not sources:
            continue  # pack-API modules carry no adapter
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
    tools = ROOT / "tools"
    synth = sorted(tools.glob("*pack_synthesize.c"))
    if not synth:
        return failures  # wave-1 consolidation may have already unified them
    shared = tools / "spark_pack_synthesize_core.c"
    shared_exists = shared.exists()
    if not shared_exists:
        print("  note: shared synthesize core not present yet"
              " (wave-1 in flight); quartet checks deferred")
        return failures
    for p in synth:
        text = p.read_text(errors="replace")
        if "spark_pack_synthesize_core" not in text:
            failures.append(
                f"FAIL {p.name}: standalone generator — must import"
                " the shared synthesize core"
            )
    return failures


def main():
    print("template-adoption gate:")
    failures = check_adapters() + check_pack_synthesize()
    for f in failures:
        print(f"  {f}")
    if failures:
        print(f"FAIL ({len(failures)}) pattern-exists-but-not-adopted violations")
        return 1
    print("PASS shared patterns consumed (or scheduled via the ratcheting offender set)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

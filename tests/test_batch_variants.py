#!/usr/bin/env python3
"""The batch-variant contract, enforced at the source level.

One source tree, N compiled modules: each resident-stage module Makefile
opts in with MODULE_BATCH_VARIANT_BUCKETS and modules/
resident_decode_stage_rules.mk emits lib<family>_resident_decode_stage[_<codec>]_b1/
_b2/.../_b1024.a - one archive per power of two - from the same translation
units with -DSPARK_BATCH_BUCKET=<n> the only difference, and every
per-bucket constant lives in the family's spark_<model>_batch_tuning.h. A
bucket is a capacity ceiling - b8 serves 1-8 rows - and runtime selection
takes the tightest ceiling at or above the microbatch, so a live batch pads
to at most twice itself; B1024 is the unflagged build and keeps the family's
published unbucketed module identifier.

What fails here:
  * a per-bucket fork: a second bucket-rule template, a hand-written bucket
    rule, or a literal bucket flag anywhere in the shared rules file
  * a second spelling of any name: bucket list, module-ID prefix/suffix, or
    archive name written twice
  * a make target declared .PHONY but never defined (the variant targets
    were born exactly that way)
  * a firmware capacity ceiling that no longer tracks the compiled bucket
  * a selection function that disagrees with the ceiling contract - checked
    by compiling and running a host probe per bucket, no CUDA involved
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_MAKEFILE = os.path.join(
    ROOT, "modules/resident_decode_stage_rules.mk")
GLM52_MODULE_MAKEFILE = os.path.join(
    ROOT, "modules/glm52_resident_decode_stage/Makefile")
DSV4_MODULE_MAKEFILE = os.path.join(
    ROOT, "modules/dsv4_resident_decode_stage/Makefile")
GLM52_TUNING_HEADER = os.path.join(
    ROOT, "modules/glm52_resident_decode_stage/include/sparkpipe",
    "spark_glm52_batch_tuning.h")
K3_TUNING_HEADER = os.path.join(
    ROOT, "modules/k3_resident_decode_stage/include/sparkpipe",
    "spark_k3_batch_tuning.h")
DSV4_TUNING_HEADER = os.path.join(
    ROOT, "modules/dsv4_resident_decode_stage/include/sparkpipe",
    "spark_dsv4_batch_tuning.h")
GLM52_FIRMWARE_HEADER = os.path.join(
    ROOT, "modules/glm52_resident_decode_stage/include/sparkpipe",
    "spark_glm52_resident_decode_stage_firmware.h")
DSV4_FIRMWARE_HEADER = os.path.join(
    ROOT, "modules/dsv4_resident_decode_stage/include/sparkpipe",
    "spark_dsv4_resident_decode_stage_firmware.h")
GLM52_FIRMWARE_JSON = os.path.join(
    ROOT, "examples/model_descriptions",
    "glm52_resident_decode_stage_mxfp4_firmware.json")
DSV4_FIRMWARE_JSON = os.path.join(
    ROOT, "examples/model_descriptions",
    "dsv4_resident_decode_stage_firmware.json")
TOP_MAKEFILE = os.path.join(ROOT, "Makefile")

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
# The grouped tile height at each bucket ceiling, the busiest group priced at
# twice the mean: all three families land at 16 from b1 through b256, 32 at
# b512, 64 at b1024 - the peak-row count doubles with the bucket, so the tile
# height climbs the ladder monotonically and never steps down.
EXPECTED_TILE_M = {bucket: 16 for bucket in BUCKETS}
EXPECTED_TILE_M[512] = 32
EXPECTED_TILE_M[1024] = 64
GLM52_ID_PREFIX = ("spark.glm52.resident_decode_stage.bf16.expert_mxfp4."
                   "h6144.l78.e256.k8")
GLM52_ID_SUFFIX = "v2"
K3_ID_PREFIX = ("spark.k3.resident_decode_stage.mxfp4_routed_bf16_rest."
                "h7168.l93.kda69.mla24")
K3_ID_SUFFIX = "v2"
DSV4_ID_PREFIX = ("spark.dsv4.flash.resident_decode_stage.linear_fp8."
                  "expert_mxfp4.kv_bf16.h4096.l43.e256.k6.ga0731")
DSV4_ID_SUFFIX = "v3"

FAILURES = []


def report(kind, path, detail=""):
    FAILURES.append(f"  {kind:22s} {path} {detail}".rstrip())


def module_id(prefix, bucket, suffix):
    return f"{prefix}.b{bucket}.{suffix}"


def check_rules_makefile():
    """One template, one instantiation, no literal buckets, targets defined."""
    text = open(RULES_MAKEFILE).read()
    rel = os.path.relpath(RULES_MAKEFILE, ROOT)

    if text.count("define SPARK_MODULE_BATCH_VARIANT_RULES") != 1:
        report("recipe template", rel, "template must be defined exactly once")
    foreach = ("$(foreach bucket,$(MODULE_BATCH_VARIANT_BUCKETS),"
               "$(eval $(call SPARK_MODULE_BATCH_VARIANT_RULES,$(bucket))))")
    if text.count(foreach) != 1:
        report("recipe template", rel,
               "template must be instantiated exactly once, by the bucket list")

    # A literal bucket anywhere - a flag, an object name, an archive name -
    # is a hand-written variant rule, i.e. the fork this system exists to
    # prevent. The template may name buckets only through $(1).
    for bucket in BUCKETS:
        for pattern in (f"SPARK_BATCH_BUCKET={bucket}",
                        f"_b{bucket}.o", f"_b{bucket}.a"):
            if pattern in text:
                report("literal bucket", rel,
                       f"'{pattern}' appears; buckets flow through $(1) only")

    # Declared-but-undefined targets: .PHONY is a promise, not a target.
    for target in ("variants", "cold_variants", "publish_variants"):
        if not re.search(rf"^{target}:", text, re.M):
            report("missing target", rel, f"'{target}' is declared, not defined")
    if "-include" in text and "MODULE_BATCH_VARIANT_DEPENDENCIES" not in \
            text.split("-include")[-1]:
        report("dependencies", rel,
               "variant .d files are collected but never -included")
    # b1024 must not publish under a second identity: the unflagged archive
    # IS the b1024 build and already publishes under the unbucketed ID.
    if "$(filter-out 1024,$(MODULE_BATCH_VARIANT_BUCKETS))" not in text:
        report("publish set", rel,
               "publish_variants must skip the b1024 re-publish")
    return 1


def check_family_makefile(path, family):
    """The family's opt-in: one bucket list, identifier prefix/suffix once."""
    text = open(path).read()
    rel = os.path.relpath(path, ROOT)
    upper = family.upper()

    expected_list = " ".join(str(bucket) for bucket in BUCKETS)
    bucket_lists = re.findall(
        r"^MODULE_BATCH_VARIANT_BUCKETS \?= ([0-9 ]+)$", text, re.M)
    if bucket_lists != [expected_list]:
        report("bucket list", rel,
               f"expected exactly one '{expected_list}', got {bucket_lists}")
    if text.count("MODULE_IDENTIFIER_PREFIX :=") != 1 or \
            text.count("MODULE_IDENTIFIER_SUFFIX :=") != 1:
        report("module id", rel, "prefix and suffix must each be written once")
    if "MODULE_IDENTIFIER := $(MODULE_IDENTIFIER_PREFIX)." \
            "$(MODULE_IDENTIFIER_SUFFIX)" not in text:
        report("module id", rel,
               "the unbucketed ID must compose the prefix and suffix")
    # The full unbucketed ID respelled literally would be a second source of
    # the same name.
    globals_ = globals()
    id_prefix = globals_[f"{upper}_ID_PREFIX"]
    id_suffix = globals_[f"{upper}_ID_SUFFIX"]
    if f"{id_prefix}.{id_suffix}" in text:
        report("module id", rel,
               "the composed unbucketed ID must not be respelled")
    return 1


def check_tuning_header(path, family, id_prefix, id_suffix):
    """The family's per-bucket truth, spelled once."""
    rel = os.path.relpath(path, ROOT)
    if not os.path.isfile(path):
        report("missing header", rel)
        return
    text = open(path).read()
    upper = family.upper()

    if family == "dsv4":
        if "#define SPARK_BATCH_BUCKET" in text:
            report("explicit bucket", rel,
                   "DSV4 must fail when the build omits its bucket")
        if not text.startswith("#pragma once\n") or "#ifndef" in text:
            report("header guard", rel,
                   "DSV4 uses pragma once and no ifndef fallback")
    elif "#define SPARK_BATCH_BUCKET 1024u" not in text:
        report("default bucket", rel, "the unflagged build must be b1024")
    # The guard closes the set: every bucket named exactly as the flag spells
    # it, so a typo'd -DSPARK_BATCH_BUCKET is a build error, not a silent tune.
    guard = re.search(r"#if SPARK_BATCH_BUCKET != 1u.*?#error", text, re.S)
    if guard is None or any(
            f"SPARK_BATCH_BUCKET != {bucket}u" not in guard.group(0)
            for bucket in BUCKETS):
        report("bucket guard", rel,
               "the #error guard must name all eleven buckets")
    prefix_defs = re.findall(
        rf"^#define SPARK_{upper}_BATCH_VARIANT_MODULE_ID_PREFIX \\$",
        text, re.M)
    suffix_defs = re.findall(
        rf"^#define SPARK_{upper}_BATCH_VARIANT_MODULE_ID_SUFFIX \\$",
        text, re.M)
    if len(prefix_defs) != 1 or len(suffix_defs) != 1:
        report("module id", rel, "prefix and suffix must each be defined once")
    for bucket in BUCKETS:
        macro = f"SPARK_{upper}_BATCH_VARIANT_MODULE_ID_B{bucket}"
        if macro not in text:
            report("module id", rel, f"{macro} is missing")
        # The bucket ID composes prefix and suffix; a full literal spelling
        # of any bucket ID would be a second source of the same name.
        if f'"{id_prefix}.b{bucket}' in text:
            report("module id", rel,
                   f"b{bucket} ID respelled literally; compose it")
    if f"SPARK_{upper}_BATCH_TUNING_GROUPED_TILE_M" not in text:
        report("tile height", rel, "the grouped tile-M macro is missing")
    return 1


def check_firmware_identity():
    """Capacity tracks the bucket; the JSONs name the unflagged b1024 build."""
    text = open(GLM52_FIRMWARE_HEADER).read()
    rel = os.path.relpath(GLM52_FIRMWARE_HEADER, ROOT)
    if '#include "sparkpipe/spark_glm52_batch_tuning.h"' not in text:
        report("firmware wiring", rel, "the firmware header must include the "
               "batch-tuning header")
    if not re.search(
            r"#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT \\\n"
            r"\s+SPARK_BATCH_BUCKET", text):
        report("firmware capacity", rel,
               "the active-sequence ceiling must be the compiled bucket")

    text = open(DSV4_FIRMWARE_HEADER).read()
    rel = os.path.relpath(DSV4_FIRMWARE_HEADER, ROOT)
    if '#include "sparkpipe/spark_dsv4_batch_tuning.h"' not in text:
        report("firmware wiring", rel, "the firmware header must include the "
               "batch-tuning header")
    if not re.search(
            r"#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT \\\n"
            r"\s+SPARK_DSV4_BATCH_TUNING_SEQUENCE_CEILING", text):
        report("firmware capacity", rel,
               "the active-sequence ceiling must come from the tuning header")

    # The unflagged build IS the b1024 module and keeps the unbucketed ID the
    # model descriptions publish; a bucketed ID appears only when a variant
    # archive publishes. Drift here means the compiled module and the model
    # description name different artifacts.
    for json_path, id_prefix, id_suffix in (
            (GLM52_FIRMWARE_JSON, GLM52_ID_PREFIX, GLM52_ID_SUFFIX),
            (DSV4_FIRMWARE_JSON, DSV4_ID_PREFIX, DSV4_ID_SUFFIX)):
        firmware_json = open(json_path).read()
        rel = os.path.relpath(json_path, ROOT)
        if f'"module": "{id_prefix}.{id_suffix}"' not in firmware_json:
            report("firmware id", rel,
                   f"expected module '{id_prefix}.{id_suffix}'")


def check_top_level_makefile():
    text = open(TOP_MAKEFILE).read()
    rel = "Makefile"
    for target, delegation in (
            ("cuda_glm52_resident_decode_stage_variants",
             "modules/glm52_resident_decode_stage variants"),
            ("cuda_dsv4_resident_decode_stage_variants",
             "modules/dsv4_resident_decode_stage variants")):
        if not re.search(rf"^{target}:", text, re.M):
            report("variant set", rel, f"the {target} target is missing")
        if delegation not in text:
            report("variant set", rel,
                   f"{target} must delegate to {delegation}")
    if "tests/test_batch_variants.py" not in text:
        report("test wiring", rel, "this test must be in PYTHON_TESTS")
    return 1


PROBE_TEMPLATE = r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// The dsv4 model header first: the firmware/tuning headers pick up its
// geometry by the module's model-header-first pattern.
#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_glm52_batch_tuning.h"
#include "sparkpipe/spark_k3_batch_tuning.h"
#include "sparkpipe/spark_dsv4_batch_tuning.h"

// The ceiling contract, executed: smallest built bucket >= the request, 0
// above b1024 (no silent oversubscription of the pools the ceiling sizes).
static uint32_t expected_ceiling(uint32_t request)
{
    uint32_t bucket = 1u;
    if (request == 0u || request > 1024u)
        return(0u);
    while (bucket < request)
        bucket *= 2u;
    return(bucket);
}

static void check_ceiling(uint32_t (*ceiling)(uint32_t),
                          const char *(*id_of)(uint32_t),
                          const char *prefix, const char *suffix)
{
    char expected[192];
    uint32_t request;
    uint32_t index;
    // Every legal request, not just the rungs and their midpoints: a missed
    // rung in the ladder silently serves a microbatch under a ceiling twice
    // its size - the pool waste the eleven-bucket set exists to remove.
    for (request = 0u; request <= 1025u; ++request)
        assert(ceiling(request) == expected_ceiling(request));
    assert(id_of(0u) == 0);
    assert(id_of(3u) == 0);
    assert(id_of(100u) == 0);
    {
        const uint32_t buckets[11] = {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u,
                                      256u, 512u, 1024u};
        for (index = 0u; index < 11u; ++index)
        {
            snprintf(expected, sizeof(expected), "%s.b%u.%s",
                     prefix, buckets[index], suffix);
            assert(strcmp(id_of(buckets[index]), expected) == 0);
        }
    }
}

int main(void)
{
    assert(SPARK_BATCH_BUCKET == EXPECTED_COMPILED_BUCKET);
    // The firmware capacity ceilings ARE the compiled bucket: this is what
    // makes a variant archive more than a rename.
    assert(SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT ==
           EXPECTED_COMPILED_BUCKET);
    assert(SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT ==
           EXPECTED_SEQUENCE_CEILING);
    check_ceiling(SparkGlm52BatchVariantBucketCeiling,
                  SparkGlm52BatchVariantModuleId,
                  "GLM52_PREFIX", "GLM52_SUFFIX");
    check_ceiling(SparkK3BatchVariantBucketCeiling,
                  SparkK3BatchVariantModuleId,
                  "K3_PREFIX", "K3_SUFFIX");
    check_ceiling(SparkDsv4BatchVariantBucketCeiling,
                  SparkDsv4BatchVariantModuleId,
                  "DSV4_PREFIX", "DSV4_SUFFIX");
    assert(strcmp(SPARK_GLM52_BATCH_TUNING_MODULE_ID,
                  "GLM52_PREFIX.bEXPECTED_BUCKET_ID.GLM52_SUFFIX") == 0);
    assert(strcmp(SPARK_K3_BATCH_TUNING_MODULE_ID,
                  "K3_PREFIX.bEXPECTED_BUCKET_ID.K3_SUFFIX") == 0);
    assert(strcmp(SPARK_DSV4_BATCH_TUNING_MODULE_ID,
                  "DSV4_PREFIX.bEXPECTED_BUCKET_ID.DSV4_SUFFIX") == 0);
    assert(SPARK_GLM52_BATCH_TUNING_GROUPED_TILE_M == EXPECTED_TILE_M);
    assert(SPARK_K3_BATCH_TUNING_GROUPED_TILE_M == EXPECTED_TILE_M);
    assert(SPARK_DSV4_BATCH_TUNING_GROUPED_TILE_M == EXPECTED_TILE_M);
    return(0);
}
"""


def probe_source(bucket):
    source = PROBE_TEMPLATE
    source = source.replace("EXPECTED_COMPILED_BUCKET", f"{bucket}u")
    source = source.replace("EXPECTED_SEQUENCE_CEILING",
                            f"{bucket}u")
    source = source.replace("EXPECTED_BUCKET_ID", str(bucket))
    source = source.replace("EXPECTED_TILE_M", f"{EXPECTED_TILE_M[bucket]}u")
    source = source.replace('"GLM52_PREFIX', f'"{GLM52_ID_PREFIX}')
    source = source.replace('GLM52_SUFFIX"', f'{GLM52_ID_SUFFIX}"')
    source = source.replace('"K3_PREFIX', f'"{K3_ID_PREFIX}')
    source = source.replace('K3_SUFFIX"', f'{K3_ID_SUFFIX}"')
    source = source.replace('"DSV4_PREFIX', f'"{DSV4_ID_PREFIX}')
    source = source.replace('DSV4_SUFFIX"', f'{DSV4_ID_SUFFIX}"')
    return source


def check_selection_contract():
    """Compile and run the ceiling/ID/tile contract per bucket, host-only."""
    compiler = shutil.which("cc") or shutil.which("clang") or \
        shutil.which("gcc")
    if compiler is None:
        report("probe skipped", "tests/test_batch_variants.py",
               "no C compiler; source checks above still stand")
        return
    include_flags = [
        "-I.", "-Iinclude", "-Imodel-families/glm52/include",
        "-Imodel-families/dsv4/include",
        "-Imodules/glm52_resident_decode_stage/include",
        "-Imodules/k3_resident_decode_stage/include",
        "-Imodules/dsv4_resident_decode_stage/include",
    ]
    # The glm52 variant module ID names the expert codec; the probe compiles
    # the mxfp4 spelling, the one the mxfp4 model description publishes.
    codec_flag = '-DGLM52_EXPERT_CODEC_NAME="mxfp4"'
    with tempfile.TemporaryDirectory() as scratch:
        for bucket in BUCKETS:
            source_path = os.path.join(scratch, f"probe_b{bucket}.c")
            binary_path = os.path.join(scratch, f"probe_b{bucket}")
            with open(source_path, "w") as handle:
                handle.write(probe_source(bucket))
            compile_command = [
                compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                f"-DSPARK_BATCH_BUCKET={bucket}u", codec_flag,
                *include_flags, source_path, "-o", binary_path,
            ]
            built = subprocess.run(
                compile_command, cwd=ROOT, capture_output=True, text=True)
            if built.returncode != 0:
                report("probe compile", f"bucket b{bucket}",
                       built.stderr.strip().splitlines()[-1]
                       if built.stderr else "")
                continue
            run = subprocess.run([binary_path], capture_output=True)
            if run.returncode != 0:
                report("probe run", f"bucket b{bucket}",
                       "the compiled selection contract asserted")

        # DSV4 has no implicit bucket.  Its root and module Makefiles name
        # B1024 explicitly, while every generated variant supplies its rung.
        default_source = os.path.join(scratch, "probe_default.c")
        with open(default_source, "w") as handle:
            handle.write(
                '#include "sparkpipe/spark_dsv4_model.h"\n'
                '#include "sparkpipe/spark_dsv4_batch_tuning.h"\n'
                'int main(void) { return(0); }\n'
            )
        built = subprocess.run(
            [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", codec_flag,
             *include_flags, default_source, "-o", os.devnull],
            cwd=ROOT, capture_output=True, text=True)
        if built.returncode == 0:
            report("probe default", "unflagged DSV4 build",
                   "a missing bucket must fail loudly")

        # A bucket outside the set must not compile at all: the variant list
        # is closed, so a typo'd bucket is a build error, not a silent tune.
        rogue_source = os.path.join(scratch, "probe_rogue.c")
        with open(rogue_source, "w") as handle:
            handle.write(probe_source(8))
        built = subprocess.run(
            [compiler, "-std=c11", f"-DSPARK_BATCH_BUCKET=3u", codec_flag,
             *include_flags, rogue_source, "-o", os.devnull],
            cwd=ROOT, capture_output=True, text=True)
        if built.returncode == 0:
            report("probe guard", "-DSPARK_BATCH_BUCKET=3",
                   "an unbuilt bucket compiled; the #error guard is broken")
    return 1


def main():
    check_rules_makefile()
    check_family_makefile(GLM52_MODULE_MAKEFILE, "glm52")
    check_family_makefile(DSV4_MODULE_MAKEFILE, "dsv4")
    check_tuning_header(GLM52_TUNING_HEADER, "glm52",
                        GLM52_ID_PREFIX, GLM52_ID_SUFFIX)
    check_tuning_header(K3_TUNING_HEADER, "k3", K3_ID_PREFIX, K3_ID_SUFFIX)
    check_tuning_header(DSV4_TUNING_HEADER, "dsv4",
                        DSV4_ID_PREFIX, DSV4_ID_SUFFIX)
    check_firmware_identity()
    check_top_level_makefile()
    check_selection_contract()
    print("glm52 + dsv4 + k3 batch variants: one source, eleven buckets B1..B1024")
    if FAILURES:
        print(f"\n{len(FAILURES)} batch-variant contract failure(s):")
        print("\n".join(FAILURES))
        return 1
    print("\nthe batch-variant contract holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())

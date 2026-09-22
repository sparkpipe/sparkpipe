#!/usr/bin/env python3
"""Source contracts for the weightd_warm dsv41_flash family hook (lane 4 M3).

The dsv41_flash module pins only the identity model tag
(SPARK_DSV41_FLASH_MODULE_TAG) and sends revision/topology from its node
context with geometry left at 0. The warmer's --family dsv41_flash hook
must mirror exactly that slice or the daemon keys the arena differently
and the resident attach fails. These tests pin the drift surface:

1. The tag pinned by the warmer equals the module's tag define.
2. The hook must NOT require --world-rank (that requirement is
   dsv4_pro-specific) and must not touch revision/topology/geometry.
3. Functional: --identity-print shows the pinned tag with the argument
   revision/topology and geometry 0; the default path stays
   glm5_next_stage byte-identical (built offline with link stubs when a
   C compiler is available; skipped otherwise).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WARM = ROOT / "tools" / "weightd_warm.c"
MODULE = (ROOT / "modules" / "dsv41_flash_resident_decode_stage" / "source" /
          "spark_dsv41_flash_resident_decode_stage_module.c")


def module_tag() -> str:
    text = MODULE.read_text()
    match = re.search(r'#define\s+SPARK_DSV41_FLASH_MODULE_TAG\s+"([^"]+)"', text)
    if match is None:
        raise AssertionError("module tag define not found")
    return match.group(1)


class WarmDsv41FlashSourceContract(unittest.TestCase):
    def test_warmer_pins_the_module_tag(self):
        warm = WARM.read_text()
        self.assertIn('strcmp(family,"dsv41_flash") == 0', warm)
        self.assertIn('strcpy(request.identity.model,"%s")' % module_tag(), warm)

    def test_world_rank_requirement_stays_dsv4_pro_specific(self):
        warm = WARM.read_text()
        self.assertIn(
            'strcmp(family,"dsv4_pro") == 0 && !world_rank_given', warm)

    def test_unknown_family_lists_all(self):
        # k3 (#1131) and ling (lane 9) appended to the whitelist; the
        # pin follows the current full list so a silent removal of any
        # family fails here.
        warm = WARM.read_text()
        self.assertIn(
            'unknown family %s (dsv4_pro, dsv41_flash, k3, ling)', warm)


class Dsv41FlashIdentityPrintFunctional(unittest.TestCase):
    def test_identity_print(self):
        if shutil.which("cc") is None:
            self.skipTest("no C compiler available")

        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "mini.spstage"
            pack.write_bytes(bytes(64))          # header never parsed on this path
            stubs = Path(tmp) / "stubs.c"
            stubs.write_text("""
#include "sparkpipe/spark_weightd.h"
SparkStatus SparkWeightdClientAcquire(SparkWeightdClient *c,uint64_t g,
    const SparkWeightdExpertKey *k,uint32_t n,
    SparkWeightdWorkingSetResult *r,uint64_t t)
    { (void)c;(void)g;(void)k;(void)n;(void)r;(void)t;
      return SPARK_STATUS_UNSUPPORTED; }
SparkStatus SparkWeightdClientAttachLazy(SparkWeightdClient *c,
    const SparkWeightdLazyAttachRequest *q,
    SparkWeightdLazyAttachResult *r,uint64_t t)
    { (void)c;(void)q;(void)r;(void)t; return SPARK_STATUS_UNSUPPORTED; }
void SparkWeightdClientClose(SparkWeightdClient *c) { (void)c; }
SparkStatus SparkWeightdClientConnect(const char *s,
    SparkWeightdClient **c,SparkWeightdHelloResult *h)
    { (void)s;(void)c;(void)h; return SPARK_STATUS_UNSUPPORTED; }
SparkStatus SparkWeightdClientRelease(SparkWeightdClient *c,uint64_t g,
    uint64_t l,SparkWeightdWorkingSetResult *r,uint64_t t)
    { (void)c;(void)g;(void)l;(void)r;(void)t;
      return SPARK_STATUS_UNSUPPORTED; }
SparkStatus SparkWeightdClientReclaim(SparkWeightdClient *c,
    SparkWeightdReclaimResult *r,uint64_t t)
    { (void)c;(void)r;(void)t; return SPARK_STATUS_UNSUPPORTED; }
""")
            binary = Path(tmp) / "warm"
            build = subprocess.run(
                ["cc", "-O2", f"-I{ROOT / 'include'}",
                 f"-I{ROOT / 'model-families' / 'dsv4' / 'include'}",
                 "-o", str(binary), str(WARM),
                 str(ROOT / "model-families" / "dsv4" / "src" /
                     "spark_dsv4_parallel_shape.c"),
                 str(ROOT / "src" / "spark_ck128.c"),
                 str(ROOT / "runtime" / "spark_weightd_manifest.c"),
                 str(stubs)],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)

            revision = "dba1be0a40aa45a94ad051997016db3960a90277"
            run = subprocess.run(
                [str(binary), "/x", str(pack), "0" * 64, revision, "4",
                 "--family", "dsv41_flash", "--identity-print"],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            fields = dict(pair.split("=", 1)
                          for pair in run.stdout.split()[1:])
            self.assertEqual(fields["model"], module_tag())
            self.assertEqual(fields["revision"], revision)
            self.assertEqual(fields["topology"], "4")
            self.assertEqual(fields["geometry"], "0")   # module never sets it

            default = subprocess.run(
                [str(binary), "/x", str(pack), "0" * 64, revision, "4",
                 "--identity-print"],
                capture_output=True, text=True)
            self.assertEqual(default.returncode, 0, default.stderr)
            fields = dict(pair.split("=", 1)
                          for pair in default.stdout.split()[1:])
            self.assertEqual(fields["model"], "glm5_next_stage")

            for extra_arguments, expect in (
                    (["--family", "dsv41_unknown"], 2),
                    (["--family", "dsv4_pro"], 2)):     # still needs --world-rank
                rejected = subprocess.run(
                    [str(binary), "/x", str(pack), "0" * 64, revision, "4",
                     "--identity-print", *extra_arguments],
                    capture_output=True, text=True)
                self.assertEqual(rejected.returncode, expect, rejected.stderr)


if __name__ == "__main__":
    unittest.main()

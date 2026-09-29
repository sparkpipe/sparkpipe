#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WARM = ROOT / "tools" / "weightd_warm.c"
MODULE_TAGS = {
    None: (ROOT / "modules" / "glm5_next_resident_decode_stage" / "source" /
           "spark_glm5_next_resident_decode_stage_module.c",
           "SPARK_GLM5_NEXT_MODULE_TAG"),
    "dsv41_flash": (ROOT / "modules" / "dsv41_flash_resident_decode_stage" /
                    "source" /
                    "spark_dsv41_flash_resident_decode_stage_module.c",
                    "SPARK_DSV41_FLASH_MODULE_TAG"),
}
STUBS = """
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_receipt.h"
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
SparkStatus SparkWeightdClientReclaimPack(SparkWeightdClient *c,
    const char *p,SparkWeightdReclaimResult *r,uint64_t t)
    { (void)c;(void)p;(void)r;(void)t; return SPARK_STATUS_UNSUPPORTED; }
SparkStatus SparkWeightdPackDigestRead(const char *p,char h[65])
    { (void)p;(void)h; return SPARK_STATUS_UNSUPPORTED; }
"""
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
DIGEST = "0123456789abcdef" * 4
PACK_BYTES = 4096


def module_tag(family):
    source, define = MODULE_TAGS[family]
    match = re.search(r'#define\s+%s\s+"([^"]+)"' % define,
                      source.read_text())
    if match is None:
        raise AssertionError("%s not defined in %s" % (define, source))
    return match.group(1)


class WeightdWarmFamilyIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        work = Path(cls.directory.name)
        stubs = work / "stubs.c"
        stubs.write_text(STUBS)
        cls.binary = work / "weightd_warm"
        build = subprocess.run(
            ["cc", "-std=c11", "-D_GNU_SOURCE", "-Wall", "-Wextra",
             "-Werror", "-O2", f"-I{ROOT / 'include'}",
             f"-I{ROOT / 'model-families' / 'dsv4' / 'include'}",
             "-o", str(cls.binary), str(WARM),
             str(ROOT / "model-families" / "dsv4" / "src" /
                 "spark_dsv4_parallel_shape.c"),
             str(ROOT / "runtime" / "spark_weightd_manifest.c"),
             str(stubs)],
            capture_output=True, text=True)
        if build.returncode != 0:
            cls.directory.cleanup()
            raise AssertionError("weightd_warm build failed:\n" +
                                 build.stderr)
        cls.pack = work / "stage.pack"
        cls.pack.write_bytes(bytes(PACK_BYTES))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_warm(self, *extra):
        environment = {key: value for key, value in os.environ.items()
                       if key != "SPARK_WEIGHTD_EXPERT_POOL_BYTES"}
        return subprocess.run(
            [str(self.binary), "/nonexistent/weightd.sock", str(self.pack),
             DIGEST, REVISION, "16", "--identity-print", *extra],
            env=environment, capture_output=True, text=True)

    def identity(self, *extra):
        run = self.run_warm(*extra)
        self.assertEqual(run.returncode, 0, run.stderr)
        words = run.stdout.split()
        self.assertEqual(words[0], "identity", run.stdout)
        return dict(word.split("=", 1) for word in words[1:])

    def test_module_tag_families_keep_argument_revision_and_topology(self):
        for family in MODULE_TAGS:
            with self.subTest(family=family):
                extra = () if family is None else ("--family", family)
                fields = self.identity(*extra)
                self.assertEqual(fields, {
                    "model": module_tag(family),
                    "revision": REVISION,
                    "topology": "16",
                    "geometry": "0",
                    "arena_bytes": str(PACK_BYTES),
                    "pack_sha256": DIGEST,
                })

    def test_family_argument_position_does_not_change_identity(self):
        before = subprocess.run(
            [str(self.binary), "--family", "dsv41_flash", "--identity-print",
             "/nonexistent/weightd.sock", str(self.pack), DIGEST, REVISION,
             "16"], capture_output=True, text=True)
        self.assertEqual(before.returncode, 0, before.stderr)
        self.assertEqual(before.stdout,
                         self.run_warm("--family", "dsv41_flash").stdout)

    def test_unknown_family_is_refused(self):
        for family in ("dsv41_unknown", "glm5_next", ""):
            with self.subTest(family=family):
                run = self.run_warm("--family", family)
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertEqual(run.stdout, "")

    def test_dsv4_pro_requires_world_rank(self):
        run = self.run_warm("--family", "dsv4_pro")
        self.assertEqual(run.returncode, 2, run.stdout)
        self.assertEqual(run.stdout, "")

    def test_invalid_identity_arguments_are_refused(self):
        for digest in ("0" * 63, "g" * 64):
            with self.subTest(digest=digest):
                run = subprocess.run(
                    [str(self.binary), "/nonexistent/weightd.sock",
                     str(self.pack), digest, REVISION, "16",
                     "--family", "dsv41_flash", "--identity-print"],
                    capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertEqual(run.stdout, "")


if __name__ == "__main__":
    unittest.main()

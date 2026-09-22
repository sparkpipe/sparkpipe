#!/usr/bin/env python3
"""Source contracts for the weightd_warm family identity (lane 5 M3).

The dsv4_pro warm identity must equal the DSV4 Pro module's attach slice
byte-for-byte or the daemon keys the arena differently and the resident
attach fails. The derivation shares the family shape source, so the values
match by construction; these tests pin the two things that could still
drift silently:

1. The geometry hash FIELD ORDER in tools/weightd_warm.c must mirror
   SparkDsv4ModuleWeightdAttach in the module source.
2. The rank->shape mapping (tp_rank = R % 4, pp_stage = R / 4) must mirror
   SparkDsv4ServingTpRank/PpStageIndex in the serving adapter.
3. Functional: --identity-print derives distinct topologies per rank and a
   stable geometry per pack (built offline with link stubs when a C
   compiler is available; skipped otherwise).
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WARM = ROOT / "tools" / "weightd_warm.c"
MODULE = (ROOT / "modules" / "dsv4_resident_decode_stage" / "source" /
          "spark_dsv4_resident_decode_stage_module.c")
ADAPTER = (ROOT / "modules" / "dsv4_resident_decode_stage" / "source" /
           "spark_dsv4_serving_adapter.c")

# The module's hash order: field names in SparkDsv4ModuleWeightdAttach.
MODULE_ORDER = [
    "format_version", "codec_abi_version", "linear_weight_codec",
    "expert_weight_codec", "kv_cache_codec", "tensor_count",
    "first_layer_index", "layer_count", "total_layer_count",
    "hidden_dimension", "vocab_count", "routed_expert_count",
    "mtp_layer_count", "file_bytes",
]


class WarmModuleSourceContract(unittest.TestCase):
    def test_geometry_hash_field_order_matches_module(self):
        module_text = MODULE.read_text()
        attach = module_text.split("SparkDsv4ModuleWeightdAttach", 1)[1]
        module_fields = re.findall(
            r"SparkHashBytes\(geometry,&header->([a-z_]+)", attach)
        self.assertEqual(module_fields, MODULE_ORDER)

        warm_text = WARM.read_text()
        warm_fields = re.findall(
            r"SparkHashBytes\(geometry,&(?:header|wide)\[(\d+|1)\]", warm_text)
        # warm indexes the raw little-endian header: [1] format_version,
        # [4] codec_abi, [5] linear, [6] expert, [7] kv, [8] tensor_count,
        # [9] first_layer, [10] layer_count, [11] total_layers, [12] hidden,
        # [13] vocab, [14] experts, [15] mtp, wide[1] file_bytes.
        expected_indices = ["1", "4", "5", "6", "7", "8", "9", "10", "11",
                            "12", "13", "14", "15", "1"]
        self.assertEqual(warm_fields, expected_indices)

    def test_rank_shape_mapping_matches_adapter(self):
        adapter = ADAPTER.read_text()
        self.assertIn("world_rank / SPARK_DSV4_SERVING_TP_DEGREE", adapter)
        self.assertIn("world_rank % SPARK_DSV4_SERVING_TP_DEGREE", adapter)
        warm = WARM.read_text()
        self.assertIn("shape.tp_rank = world_rank % 4u", warm)
        self.assertIn("shape.pp_stage_index = world_rank / 4u", warm)
        self.assertIn("shape.tp_degree = 4u", warm)
        self.assertIn("shape.pp_stage_count = 4u", warm)

    def test_family_identity_values(self):
        warm = WARM.read_text()
        self.assertIn('strcpy(identity->model,"dsv4")', warm)
        self.assertIn("identity->revision[0] = '\\0'", warm)
        self.assertIn("(uint32_t)config.configuration_hash", warm)


class IdentityPrintFunctional(unittest.TestCase):
    def test_identity_print(self):
        if shutil.which("cc") is None:
            self.skipTest("no C compiler available")

        header = [0x34565344, 1, 80, 56, 1, 5, 7, 9, 10, 0, 1, 61, 7168,
                  129280, 384, 3, 80, 1 << 20]
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "mini.spstage"
            with pack.open("wb") as handle:
                handle.write(struct.pack("<16I2Q", *header))
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

            import os
            env = dict(os.environ,
                       SPARK_WEIGHTD_EXPERT_POOL_BYTES="1073741824")
            geometries = set()
            topologies = set()
            for rank in (0, 1, 6, 15):
                run = subprocess.run(
                    [str(binary), "/x", str(pack), "0" * 64, "", "16",
                     "--family", "dsv4_pro", "--world-rank", str(rank),
                     "--identity-print"],
                    env=env, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                fields = dict(pair.split("=", 1)
                              for pair in run.stdout.split()[1:])
                self.assertEqual(fields["model"], "dsv4")
                self.assertEqual(fields["revision"], "")
                geometries.add(fields["geometry"])
                topologies.add(fields["topology"])
            self.assertEqual(len(geometries), 1)      # same pack header
            self.assertEqual(len(topologies), 4)      # rank-dependent


if __name__ == "__main__":
    unittest.main()

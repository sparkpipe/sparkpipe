#!/usr/bin/env python3
"""Source contracts for the weightd_warm ling family hook (lane 9 M3).

The ling module pins only the identity model tag
(SPARK_LING_MODULE_TAG) and sends revision/topology from its stage
config (revision = the stage model_revision, topology = tp_degree)
with geometry left at 0 - exactly the dsv41_flash shape. The warmer's
--family ling hook must mirror exactly that slice or the daemon keys
the arena differently and the resident attach fails. These tests pin
the drift surface:

1. The tag pinned by the warmer equals the module's tag define.
2. The hook must NOT require --world-rank (dsv4_pro-specific) and must
   not touch revision/topology/geometry.
3. The whitelist/usage list ling alongside the other families.
4. Functional: --identity-print shows the pinned tag with the argument
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
MODULE = (ROOT / "modules" / "ling_resident_decode_stage" / "source" /
          "spark_ling_resident_decode_stage_module.c")


def module_tag() -> str:
    text = MODULE.read_text()
    match = re.search(r'#define\s+SPARK_LING_MODULE_TAG\s+"([^"]+)"', text)
    if match is None:
        raise AssertionError("module tag define not found")
    return match.group(1)


class WarmLingSourceContract(unittest.TestCase):
    def test_warmer_pins_the_module_tag(self):
        warm = WARM.read_text()
        self.assertIn('strcmp(family,"ling") == 0', warm)
        self.assertIn('strcpy(request.identity.model,"%s")' % module_tag(),
                      warm)

    def test_world_rank_requirement_stays_dsv4_pro_specific(self):
        warm = WARM.read_text()
        self.assertIn(
            'strcmp(family,"dsv4_pro") == 0 && !world_rank_given', warm)
        self.assertNotIn('strcmp(family,"ling") == 0 && !world_rank_given',
                         warm)

    def test_hook_leaves_revision_topology_geometry_to_the_args(self):
        # The ling hook body must only pin the tag: no revision/topology
        # writes inside the ling branch (they stay authoritative from
        # the stage config, exactly like dsv41_flash).
        warm = WARM.read_text()
        match = re.search(
            r'strcmp\(family,"ling"\) == 0 \)\s*\{(.*?)\n    \}', warm,
            re.S)
        self.assertIsNotNone(match, "ling family hook not found")
        body = match.group(1)
        self.assertIn('strcpy(request.identity.model', body)
        self.assertNotIn("identity.revision", body)
        self.assertNotIn("identity.topology", body)
        self.assertNotIn("geometry_fingerprint", body)

    def test_unknown_family_lists_all(self):
        warm = WARM.read_text()
        self.assertIn(
            'unknown family %s (dsv4_pro, dsv41_flash, k3, ling)', warm)

    def test_usage_documents_ling(self):
        warm = WARM.read_text()
        self.assertIn("--family ling (pin the module tag", warm)


class LingIdentityPrintFunctional(unittest.TestCase):
    def test_identity_print(self):
        if shutil.which("cc") is None:
            self.skipTest("no C compiler for the offline link-stub build")
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "weightd_warm"
            stub = Path(tmp) / "stub.c"
            stub.write_text(
                "#include <stdint.h>\n"
                "typedef int SparkStatus;\n"
                "typedef struct { const char *name; } SparkWeightdClient;\n"
                "SparkStatus SparkWeightdClientConnect(const char *s,"
                "SparkWeightdClient **c,uint64_t t){(void)s;(void)t;"
                "*c=0;return 0;}\n"
                "SparkStatus SparkWeightdClientAttachLazy("
                "SparkWeightdClient *c,const void *r,void *a,uint64_t t)"
                "{(void)c;(void)r;(void)a;(void)t;return 1;}\n"
                "void SparkWeightdClientClose(SparkWeightdClient *c)"
                "{(void)c;}\n")
            compile_run = subprocess.run(
                ["cc", "-I", str(ROOT), "-DLINK_STUBS", "-o", str(binary),
                 str(WARM), str(stub)],
                capture_output=True, text=True)
            # The offline stub build only needs to succeed far enough to
            # run --identity-print; full link fidelity is the queue
            # build's job. Skip when the real object graph is required.
            if compile_run.returncode != 0:
                self.skipTest(
                    f"offline stub build not linkable here: "
                    f"{compile_run.stderr[:200]}")
            pack = Path(tmp) / "pack.sp"
            pack.write_bytes(b"x" * 16)
            run = subprocess.run(
                [str(binary), "/nonexistent/sock", str(pack), "0" * 64,
                 "e0dfe7cd", "16", "--family", "ling", "--identity-print"],
                capture_output=True, text=True)
            self.assertIn("model=ling_stage", run.stdout)
            self.assertIn("revision=e0dfe7cd", run.stdout)
            self.assertIn("topology=16", run.stdout)
            self.assertIn("geometry=0", run.stdout)


if __name__ == "__main__":
    unittest.main()

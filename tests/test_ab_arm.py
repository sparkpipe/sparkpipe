#!/usr/bin/env python3
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_arm
import ab_dry_run

CLI = ROOT / "build" / "sparkpipe_quant_arm"


def base_arm():
    return ab_dry_run.arm_descriptor("flash", "S1", "nvfp4", "nvfp4nv", "community")


def variants():
    arms = {"base": base_arm()}
    arm = base_arm()
    arm["kv"].update(latent="fp8", group=128, mode="sim")
    arm["arm_id"] = ab_arm.expected_id(arm)
    arms["kv"] = arm
    arm = base_arm()
    arm["drafter"] = {"kind": "mtp", "label": "mtp8", "codec": "fp8", "head": "bf16", "sidecar_sha256": [ab_dry_run.hexid("side", r) for r in range(4)]}
    arm["arm_id"] = ab_arm.expected_id(arm)
    arms["drafter"] = arm
    arm = base_arm()
    arm["expert"].update(codec="int8", label="int8rtn", producer="experiment", recipe_sha256=ab_dry_run.hexid("recipe"))
    arm["arm_id"] = ab_arm.expected_id(arm)
    arms["experiment"] = arm
    return arms


def invalid():
    out = {}
    arm = base_arm()
    arm["note"] = "x"
    out["extra member"] = arm
    arm = base_arm()
    arm["arm_id"] = "flash.S1.e-fp8.k-bf16/bf16/fp32.d-none"
    out["arm id mismatch"] = arm
    arm = base_arm()
    arm["spine"]["spine_digest"] = arm["spine"]["spine_digest"][:3]
    out["short spine"] = arm
    arm = base_arm()
    arm["topology"]["tp"] = True
    out["bool integer"] = arm
    arm = base_arm()
    arm["kv"]["group"] = 64
    out["bf16 group"] = arm
    arm = base_arm()
    arm["expert"]["producer"] = "experiment"
    out["experiment without recipe"] = arm
    arm = base_arm()
    arm["drafter"]["codec"] = "fp8"
    out["weightless drafter codec"] = arm
    arm = base_arm()
    arm["spine"]["spine_digest"][0] = arm["spine"]["spine_digest"][0].upper()
    out["upper hex"] = arm
    arm = base_arm()
    arm["revision"] = "bad\\revision"
    out["backslash"] = arm
    return out


class ArmTest(unittest.TestCase):
    def test_digest_is_stable_under_formatting(self):
        arm = base_arm()
        compact = ab_arm.parse_text(json.dumps(arm))
        pretty = ab_arm.parse_text(json.dumps(dict(reversed(list(arm.items()))), indent=3))
        self.assertEqual(ab_arm.arm_digest(compact), ab_arm.arm_digest(pretty))
        self.assertEqual(len(ab_arm.arm_digest(arm)), 64)

    def test_refusals(self):
        for name, arm in invalid().items():
            with self.assertRaises(ab_arm.ArmError, msg=name):
                ab_arm.parse_text(json.dumps(arm))
        with self.assertRaises(ab_arm.ArmError):
            ab_arm.parse_text('{"format": "sparkpipe-quant-arm-v1", "format": "x"}')
        with self.assertRaises(ab_arm.ArmError):
            ab_arm.parse_text(json.dumps(base_arm()).replace('"tp": 4', '"tp": 4.0'))

    def test_axes(self):
        arms = variants()
        self.assertEqual(ab_arm.differing_axes(arms["base"], arms["kv"]), ["K"])
        self.assertEqual(ab_arm.differing_axes(arms["base"], arms["drafter"]), ["D"])
        self.assertEqual(ab_arm.differing_axes(arms["base"], arms["experiment"]), ["E"])
        both = copy.deepcopy(arms["kv"])
        both["expert"] = copy.deepcopy(arms["experiment"]["expert"])
        both["arm_id"] = ab_arm.expected_id(both)
        self.assertEqual(ab_arm.differing_axes(arms["base"], both), ["E", "K"])
        spine = copy.deepcopy(arms["base"])
        spine["spine"]["spine_digest"][5 % 4] = "0" * 64
        self.assertEqual(ab_arm.differing_axes(arms["base"], spine), ["spine"])

    @unittest.skipUnless(CLI.exists(), "build/sparkpipe_quant_arm not built on this host")
    def test_c_and_python_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, arm in variants().items():
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps(arm, indent=2))
                digest = subprocess.run([str(CLI), "--digest", str(path)], capture_output=True, text=True, check=True).stdout.strip()
                canonical = subprocess.run([str(CLI), "--canonical", str(path)], capture_output=True, text=True, check=True).stdout
                self.assertEqual(canonical.encode(), ab_arm.canonical(arm), name)
                self.assertEqual(digest, ab_arm.arm_digest(arm), name)
                summary = json.loads(subprocess.run([str(CLI), "--summary", str(path)], capture_output=True, text=True, check=True).stdout)
                self.assertEqual(summary["pack_set_sha256"], ab_arm.pack_set_sha256(arm))
                self.assertEqual(summary["arm_kv"], ab_arm.kv_text(arm))
            for name, arm in invalid().items():
                path = Path(directory) / "invalid.json"
                path.write_text(json.dumps(arm))
                result = subprocess.run([str(CLI), "--digest", str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, name)
                self.assertIn("REFUSED", result.stderr, name)


if __name__ == "__main__":
    unittest.main(verbosity=1)

#!/usr/bin/env python3
import copy
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools/gen_geometry_header.py"


def generator():
    spec = importlib.util.spec_from_file_location("gen_geometry_header", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeometryHeaderGenerator(unittest.TestCase):
    def check(self, *arguments):
        return subprocess.run([sys.executable, str(GENERATOR), *arguments, "--check"], cwd=ROOT, capture_output=True, text=True)

    def test_every_output_is_byte_identical(self):
        module = generator()
        self.assertTrue(module.FAMILIES)
        for family in sorted(module.FAMILIES):
            with self.subTest(family=family):
                run = self.check("--family", family)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_contract_changes_reach_the_output(self):
        module = generator()
        for family in sorted(module.FAMILIES):
            with self.subTest(family=family):
                contract = module.load_contract(module.FAMILIES[family]["contract"])
                tracked = (ROOT / module.FAMILIES[family]["header"]).read_text(encoding="utf-8")
                self.assertEqual(module.render_header(family, contract), tracked)
                changed = copy.deepcopy(contract)
                changed["model"]["hidden_dimension"] += 128
                self.assertNotEqual(module.render_header(family, changed), tracked)

    def test_drift_fails_the_check(self):
        module = generator()
        family = sorted(module.FAMILIES)[0]
        relative = module.FAMILIES[family]["header"]
        rendered = module.render_header(family, module.load_contract(module.FAMILIES[family]["contract"]))
        with tempfile.TemporaryDirectory() as directory:
            module.ROOT = Path(directory)
            target = module.ROOT / relative
            target.parent.mkdir(parents=True)
            target.write_text(rendered, encoding="utf-8")
            self.assertTrue(module.write_or_check(relative, rendered, True))
            target.write_text(rendered + "\n", encoding="utf-8")
            self.assertFalse(module.write_or_check(relative, rendered, True))
            target.unlink()
            self.assertFalse(module.write_or_check(relative, rendered, True))

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ValidatorDigest(unittest.TestCase):
    def tree(self, directory):
        root = Path(directory)
        (root / "tools").mkdir()
        shutil.copy(ROOT / "tools/validator_digest.py", root / "tools/validator_digest.py")
        templates = root / "include/sparkpipe/family/validation"
        templates.mkdir(parents=True)
        (templates / "outer.h").write_text('#include "sparkpipe/family/validation/inner.h"\n#include <stdio.h>\n')
        (templates / "inner.h").write_text("static int inner = 1;\n")
        (templates / "unused.h").write_text("static int unused = 1;\n")
        validator = root / "modules/m/validation/spark_m_cuda_validation.cu"
        validator.parent.mkdir(parents=True)
        validator.write_text('#include "sparkpipe/family/validation/outer.h"\nint main(void) { return 0; }\n')
        return root, templates, validator

    def digest(self, root, validator):
        return subprocess.run([sys.executable, str(root / "tools/validator_digest.py"), str(validator)], capture_output=True, text=True)

    def test_template_edits_change_the_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root, templates, validator = self.tree(directory)
            first = self.digest(root, validator)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertRegex(first.stdout.strip(), r"^[0-9a-f]{64}$")
            (templates / "unused.h").write_text("static int unused = 2;\n")
            self.assertEqual(self.digest(root, validator).stdout, first.stdout)
            (templates / "inner.h").write_text("static int inner = 2;\n")
            second = self.digest(root, validator)
            self.assertNotEqual(second.stdout, first.stdout)
            validator.write_text(validator.read_text() + "\n")
            self.assertNotEqual(self.digest(root, validator).stdout, second.stdout)

    def test_a_missing_template_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root, templates, validator = self.tree(directory)
            (templates / "inner.h").unlink()
            result = self.digest(root, validator)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("inner.h", result.stderr)

    def test_every_module_validator_digests(self):
        validators = sorted(ROOT.glob("modules/*/validation/spark_*_cuda_validation.cu"))
        self.assertTrue(validators)
        for validator in validators:
            result = self.digest(ROOT, validator)
            self.assertEqual(result.returncode, 0, f"{validator}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()

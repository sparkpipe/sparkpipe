import os
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (".h", ".c", ".cu", ".cuh", ".cpp", ".hpp")
SKIPPED = ("build", ".git", "tests")
PLATFORM_SHIMS = {"_POSIX_C_SOURCE", "_GNU_SOURCE", "MSG_NOSIGNAL", "POLLRDHUP", "F_GETPATH", "HWCAP_SHA2", "O_DIRECT", "MFD_CLOEXEC"}
PENDING = {}
GUARD = re.compile(r"^\s*#\s*(?:ifndef\s+(\w+)|if\s+!\s*defined\s*\(\s*(\w+)\s*\))\s*$")
DEFINE = re.compile(r"^\s*#\s*define\s+(\w+)\s+\S")


def build_defaults():
    found = set()
    for directory, subdirectories, files in os.walk(ROOT):
        relative = Path(directory).relative_to(ROOT)
        if relative.parts[:1] and relative.parts[0] in SKIPPED:
            subdirectories[:] = []
            continue
        for name in files:
            if not name.endswith(SOURCES):
                continue
            lines = (Path(directory) / name).read_text(errors="replace").splitlines()
            for line, following in zip(lines, lines[1:]):
                guard, define = GUARD.match(line), DEFINE.match(following)
                if guard and define and define.group(1) == (guard.group(1) or guard.group(2)):
                    found.add((str(relative / name), define.group(1)))
    return found


class NoBuildDefaults(unittest.TestCase):
    def test_every_build_setting_is_named_by_its_build(self):
        found = build_defaults()
        self.assertEqual(sorted((path, macro) for path, macro in found if macro not in PLATFORM_SHIMS and (path, macro) not in PENDING), [])

    def test_pending_list_only_shrinks(self):
        self.assertEqual(sorted(set(PENDING) - build_defaults()), [])


if __name__ == "__main__":
    unittest.main()

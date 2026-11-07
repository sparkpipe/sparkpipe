#!/usr/bin/env python3
"""No tool creates a symlink to a weight file or a configuration.

A scan of every tracked tool, script and deployment file finds no code that
creates a symlink, apart from the three operating-system and archived-replay
sites listed with their reasons; the list may only shrink. No tool names the
shared config/stage.json that per-node links used to resolve.
"""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK_SITE = re.compile(r"(^|[^A-Za-z_.])ln\s+(-[A-Za-z]*s[A-Za-z]*\b|--symbolic)|os\.symlink\(|\.symlink_to\(|(^|[^A-Za-z_.])symlink(at)?\(")
UNSCANNED = ("tests/", "qualification/", "docs/", "references/", "experiments/")
ALLOWED = {
    "tools/install_cuda13_compile_only_debian13.sh": "points the system /usr/local/cuda at the installed CUDA 13.3 toolkit",
    "tools/devcycle/ds4_spark_brickproof.py": "masks a systemd unit with the standard /dev/null link",
    "tools/replay_shared_serving_release.py": "replays the archived 2026-09-24 release b690c3a57 byte for byte; its archived stage-assets.sh reads the source checkout through this link",
}


def tracked():
    names = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                           capture_output=True, check=True).stdout.decode().split("\0")
    return sorted(name for name in names if name and not name.startswith(UNSCANNED) and not name.endswith(".md")
                  and (ROOT / name).is_file() and not (ROOT / name).is_symlink())


class NoWeightLinks(unittest.TestCase):
    def test_no_tool_creates_a_link_outside_the_reasoned_list(self):
        sites = {}
        for name in tracked():
            text = (ROOT / name).read_bytes().decode("utf-8", "ignore")
            hits = [number for number, line in enumerate(text.splitlines(), 1) if LINK_SITE.search(line) and "is_symlink" not in line]
            if hits:
                sites[name] = hits
        self.assertEqual({name: lines for name, lines in sites.items() if name not in ALLOWED}, {})
        self.assertEqual(sorted(set(ALLOWED) - set(sites)), [], "an allowed link site is gone; drop it from ALLOWED")
        named = [name for name in tracked() if name.startswith("tools/") and '"config/stage.json"' in (ROOT / name).read_text(errors="ignore")]
        self.assertEqual(named, [])


if __name__ == "__main__":
    unittest.main()

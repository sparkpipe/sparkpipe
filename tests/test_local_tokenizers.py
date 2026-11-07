#!/usr/bin/env python3
"""Every lane reads its tokenizer from the hub's local NVMe.

Cold and warm storage can leave at any moment, so no lane tool or hub
deployment file names them, a model record that names one stops the tool at
import, and the serving configuration carries no cold path, even as a
citation.
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane

SCANNED = ("tools/stage_lane.py", "tools/stage_lane.sh", "tools/lane_config.py", "tools/k3_lane.sh", "deployment/lanes.json",
           "deployment/hub/model_swap.json", "deployment/hub/chat_frontend.json")


class LocalTokenizers(unittest.TestCase):
    def test_every_model_names_a_local_tokenizer(self):
        named = {name: model["tokenizer_json"] for name, model in stage_lane.MODELS.items() if model.get("tokenizer_json")}
        self.assertGreaterEqual(len(named), 19)
        for name, path in named.items():
            with self.subTest(model=name):
                self.assertTrue(path.startswith("/srv/workspace/sparkpipe/tokenizers/"), path)

    def test_lane_tools_name_no_cold_or_warm_path(self):
        for name in SCANNED:
            with self.subTest(file=name):
                text = (ROOT / name).read_text()
                self.assertNotIn("/mnt/cold-raid", text)
                self.assertNotIn("/mnt/model-warm", text)

    def test_a_cold_tokenizer_stops_the_tool(self):
        probe = ("import re, sys; sys.path.insert(0, 'tools'); source = open('tools/stage_lane.py').read();"
                 "source = source.replace('/srv/workspace/sparkpipe/tokenizers/', '/mnt/cold-raid6/models/', 1);"
                 "exec(compile(source, 'stage_lane', 'exec'), {'__name__': 'stage_lane', '__file__': 'tools/stage_lane.py'})")
        run = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("is not on the hub's local NVMe", run.stderr + run.stdout)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("block_draftd", ROOT / "tools" / "block_draftd.py")
block_draftd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(block_draftd)


def main():
    failures = []
    latent_module = importlib.import_module("latent_block_drafter")
    built = []

    def stand_in(self, directory, shard, names, device):
        built.append(directory)

    latent_module.LatentBlockDrafter.__init__ = stand_in
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text(json.dumps({"dflash_config": {"attention_mode": "mla"}}))
        drafter = block_draftd.open_drafter(directory, None, {}, "cpu")
    if built != [directory]:
        failures.append("a latent checkpoint should open through the latent drafter")
    if not isinstance(drafter, block_draftd.LatentBlockDrafter):
        failures.append(f"the drafter open_drafter builds ({type(drafter).__module__}.{type(drafter).__name__}) is not the class the draft server checks for latent drafting ({block_draftd.LatentBlockDrafter.__module__}), so the server would build the block graph for a latent drafter")
    copies = [name for name in sys.modules if name.endswith("block_drafter") or name.endswith("tapstream")]
    if len(copies) != len(set(name.rsplit(".", 1)[-1] for name in copies)):
        failures.append(f"the draft server loads a drafter module twice under different names: {sorted(copies)}")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS the draft server and open_drafter share one latent drafter class, so a DFlash2 checkpoint gets the latent draft graph")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

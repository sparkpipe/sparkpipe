#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def digest_file(pack: str, root: str) -> str:
    packs = os.path.join(root, "packs")
    return os.path.join(packs, "pack.sha256") if os.path.dirname(pack) == packs else pack + ".sha256"


def main() -> int:
    lanes = json.loads((ROOT / "deployment" / "lanes.json").read_text())["lanes"]
    records = sys.argv[1:] or sorted(lanes)
    with tempfile.TemporaryDirectory() as scratch:
        for record in records:
            if record not in lanes:
                print(f"release_packs: {record} is not a lane record", file=sys.stderr)
                return 2
            out = os.path.join(scratch, record)
            subprocess.run([sys.executable, str(ROOT / "tools" / "lane_config.py"), "render", record, out], check=True, stdout=subprocess.DEVNULL)
            for name in sorted(os.listdir(out)):
                if not (name.startswith("adapter.") and name.endswith(".json")):
                    continue
                host = name[len("adapter."):-len(".json")]
                pack = json.loads(pathlib.Path(out, name).read_text()).get("stage_pack_path")
                if pack:
                    print(host, pack, digest_file(pack, lanes[record]["runtime_root"].format(host=host)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

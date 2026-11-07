#!/usr/bin/env python3
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def digest_file(pack: str, root: str) -> str:
    packs = os.path.join(root, "packs")
    return os.path.join(packs, "pack.sha256") if os.path.dirname(pack) == packs else pack + ".sha256"


def check_script(host: str, listing: str) -> str:
    lines = []
    for row in pathlib.Path(listing).read_text().splitlines():
        name, pack, digest = row.split(" ")
        if name == host:
            lines.append(f"test -s {shlex.quote(pack)} -a -s {shlex.quote(digest)} -a -s {shlex.quote(pack + '.experts')} || echo MISSING {shlex.quote(pack)}")
    return "; ".join(lines) if lines else "echo MISSING no-pack-for-" + shlex.quote(host)


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        print(check_script(sys.argv[2], sys.argv[3]))
        return 0
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
                if not (name.startswith("adapter.") and name.endswith(".json")) or name.startswith("adapter.fill."):
                    continue
                host = name[len("adapter."):-len(".json")]
                adapter = json.loads(pathlib.Path(out, name).read_text())
                pack = adapter.get("stage_pack_path")
                fill = pathlib.Path(out, f"adapter.fill.{host}.json")
                partner = (json.loads(fill.read_text()) if fill.exists() else adapter).get("streamed_prefill", {}).get("partner_pack")
                if partner:
                    print(host, partner, partner + ".sha256")
                if pack:
                    digest = digest_file(pack, lanes[record]["runtime_root"].format(host=host))
                    if any(character.isspace() for character in pack + digest):
                        print(f"release_packs: {record} on {host} names {pack!r}; the listing is space-separated", file=sys.stderr)
                        return 2
                    print(host, pack, digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

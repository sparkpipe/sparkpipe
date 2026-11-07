#!/usr/bin/env python3
import argparse
import json
import pathlib
import shlex
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import lane_config
import stage_lane

SSH = ("ssh", "-o", "BatchMode=yes")


def pairs(record):
    with tempfile.TemporaryDirectory() as scratch:
        directory = pathlib.Path(scratch) / "lane"
        lane_config.render(record, directory)
        name = "adapter.fill" if (directory / "deployment.fill.json").exists() else "adapter"
        adapters = {host: json.loads((directory / f"{name}.{host}.json").read_text()) for host in stage_lane.HOSTS}
    if "streamed_prefill" not in adapters[stage_lane.HOSTS[0]]:
        raise SystemExit(f"{record}: this lane record has no streamed_prefill")
    for rank, host in enumerate(stage_lane.HOSTS):
        partner = stage_lane.HOSTS[rank ^ 1]
        yield host, partner, adapters[partner]["stage_pack_path"], adapters[host]["streamed_prefill"]["partner_pack"]


def fetch(host, partner, source, target):
    part = target + ".part"
    quoted = {name: shlex.quote(value) for name, value in (("source", source), ("target", target), ("part", part))}
    script = (f"set -e; test -f {quoted['target']}.verified-copy && test -f {quoted['target']} && {{ echo present; exit 0; }}; "
              f"mkdir -p $(dirname {quoted['target']}); "
              f"ssh -o BatchMode=yes -c aes128-gcm@openssh.com {partner} dd if={quoted['source']} iflag=direct bs=16M status=none | "
              f"dd of={quoted['part']} oflag=nocache conv=fdatasync bs=16M status=none; "
              f"scp -q -o BatchMode=yes {partner}:{quoted['source']}.experts {quoted['target']}.experts; "
              f"scp -q -o BatchMode=yes {partner}:{quoted['source']}.sha256 {quoted['target']}.sha256; "
              f"want=$(cut -c1-64 {quoted['target']}.sha256); got=$(sha256sum {quoted['part']} | cut -c1-64); "
              f"[ \"$want\" = \"$got\" ] || {{ echo \"digest mismatch want $want got $got\"; exit 3; }}; "
              f"mv {quoted['part']} {quoted['target']}; echo $got > {quoted['target']}.verified-copy; echo copied")
    return subprocess.Popen([*SSH, host, script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main():
    parser = argparse.ArgumentParser(description="Copy each rank's rank ^ 1 pack to where the lane record's streamed prefill reads it")
    parser.add_argument("record")
    args = parser.parse_args()
    failed = 0
    running = [(host, fetch(host, partner, source, target)) for host, partner, source, target in pairs(args.record)]
    for host, process in running:
        output = process.communicate()[0].strip()
        print(f"{host}: {output}")
        failed += process.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

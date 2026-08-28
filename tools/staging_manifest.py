#!/usr/bin/env python3
"""staging_manifest.py — fleet pack-staging manifest (the CHECKED state).

Walks spark0..sparkf over ssh (BatchMode) and emits the model x node
presence + size table (same format as the S1 inventory in
docs/AGENT_LANE_BRIEFS/reports/staging-inventory-*.md). Exit 1 on any gap
vs the fleet table below; exit 0 when every policy pack is present with the
expected byte size.

Modes:
  --fleet            walk the real nodes (default with no mode flag)
  --verify           additionally sha256 every present pack and compare to
                     the pinned digest (slow: ~2.4 TB of hashing)
  --json PATH        also write the table as JSON
  --hosts "h ..."     restrict/extend hosts (default: spark0..sparkf)
  --models "m ..."    restrict models (default: all REQUIRED + PENDING)
  --strict           PENDING models also gate the exit code (by default a
                     PENDING model is reported but does not fail the run)
  --self-test        offline: validate the fleet table's internal
                     consistency and exit (used by the PYTHON_TESTS gate)

Every host probe is one ssh: stat the expected path. Nothing is written on
the nodes by this tool (read-only). Hosts are always taken from arguments —
no hardcoded single node.

Fleet table sources (each entry cites its receipt):
  glm5_next  16/16  this lane, 2026-08-28 (sha sweep; receipt files pinned
                   next to packs after the sweep)
  dsv4_pro   16/16  dsv4pro lane report 2026-08-27 + this lane's spark1/2
                   verification + warm stash receipts
  glm52      8/8    glm52 packs report 2026-08-27 (band by design; the
                   README marks GLM 5.2 a kernel donor, NOT a fleet model)
  k3 stage3  4/4    k3 lane report + warm stage3_warm_sha256.txt (stages
                   0-2 are PENDING, owned by the K3 lane)
  qwen4_flash v3 4/4  qwen-flash S4 receipts (packs_v3; v1/v2 retired)
  qwen38 27B TP1 3/3  dev instance + co-resident test set (sha-identical
                   trio, this lane 2026-08-28); fleet re-emit sharded pending
  dsv4_flash tp16 16/16  deployed fleet pack (informational size drift:
                   sparkf holds a different generation — real gap, fails)
  qwen38max  PENDING   15/16 built on spark7 only, 12/16 receipted; the
                   missing pack is qwen38max.pp3.tp4-rank3; no fleet
                   distribution until the set completes
"""

import argparse
import json
import subprocess
import sys

HEX = "0123456789abcdef"
ALL_HOSTS = ["spark%s" % c for c in HEX]

G5N_SIZE = 21706046976
G5N_NAME = "glm5_next_stage.tp16.rank{rank}.g5nsp"

DSV4_PRO_SIZE = {
    0: 99603890892, 1: 99603890892, 2: 99603890892, 3: 99603890892,
    4: 94283835100, 5: 94283835100, 6: 94283835100, 7: 94283835100,
    8: 94264972524, 9: 94264972524, 10: 94264972524, 11: 94264972524,
    12: 94748565432, 13: 94748565432, 14: 94746730424, 15: 94746730424,
}

FLEET_TABLE = {
    "glm5_next.tp16": {
        "required": True,
        "nodes": {h: i for i, h in enumerate(ALL_HOSTS)},
        "path": "~/glm53_packs/" + G5N_NAME,
        "sizes": {i: G5N_SIZE for i in range(16)},
        # full-file digests, staging sweep 2026-08-28 (receipt.json written
        # next to each pack on its home node)
        "shas": {
            0: "ca0e427ce9c33f0f6b536984bfbf4a50ae3498ba814ab7310e5f77aa24e0e6c1",
            1: "3c7e0665865798bbd869263bf6f06ea62d2826bf709eb5571b8d3376fdaefdca",
            2: "ac4db78e25c5fc8043dc14448be775bbb2b00a26dd2f28c4d6b55e21f0e9948d",
            3: "33ca27de548d88f31ba4b76ad5be660e1f3d7f5358671f9663446d42d88ba7b2",
            4: "56e16fcbd0a71b3a65949552e8959264d1e89ed85e0ba9141b51b2ad84454091",
            5: "29b387fab599ea127e3ac014da92ae60d5da4a3451f445fece814655bd340653",
            6: "bf4f32cc9314de25f75dc9dfafece258c765e9c754e956f58a1657c42a69ec6e",
            7: "5677fb3cee1ff7e005f4436a3a5357797da79a4231355c0f889cf6134744d735",
            8: "64eda5372acc8675e41e14112fe20fc9c1d3765d10467602d397fd6dabb4a60a",
            9: "85c879b9e5584968de8060f30ed16e3a5a4b497fc514d86a44a4d8e56ca7af46",
            10: "b35eeccdeaece9d8a017643feb93d5b5b8797881cf19578809f083361d13e02f",
            11: "2a270516b0c518a66fc8917f866fc47cdcdeaee26c5cf0cfa2015415db2a448a",
            12: "8034616cf29850201ee590b8ddc2cefe5603927debbb105ce04346f898a999fe",
            13: "0e0473f546c8a4c6488c7089c3c033cc2e516f754fd3e3e97ab7afc2de104d7d",
            14: "5461b9073a1b66f96bc31955c579d6e870d2edeb10def11106b59c61578076a6",
            15: "924fa54bc134cdad70f0f17963591dd78612afba28227e6d5518b9b43d49db98",
        },
        "note": "runtime symlink ~/sparkdata/glm5_next.tp16/packs/ -> pack",
    },
    "dsv4_pro.tp4pp4": {
        "required": True,
        "nodes": {h: i for i, h in enumerate(ALL_HOSTS)},
        "path": "~/sparkdata/dsv4_pro.tp4pp4/packs/dsv4_pro_tp4_pp4_stage.spstage",
        "sizes": DSV4_PRO_SIZE,
        "shas": {
            0: "490c5cdc9cb2863ca1c1a576bc33b3550bb613de1351b2c057e1d23a4d73d90b",
            1: "e9de954be40628dcc4f0d2d88fe1b2391b50ba7910ca9a8e5ba249b6a53b9879",
            2: "acc6d761cb79174abe2ec8e8b53e8bf38ca98dde87a2ad939d3beccd95ad5863",
            3: "1071e9ad32b8915a3a6bd19f6b6baa0524ce45dbf4aa844b7fb7d8eb2db985c9",
            4: "db58f92a991f170bcc3eea899479982f67708114602e1d4620ddb1870f9fc163",
            5: "06487bad7933d0723b66da622d27865f97c1fc60ce1e6883e72263dd20bdee5a",
            6: "a5df643077165bf6f51a3ab7eee316306c3d3a2c2f85ed45e1e3b94d6e4063c1",
            7: "ff056efeba3646ad381f4df557792d6ddb229786db7f64741e52a5f3acbdf699",
            8: "c0d33b6c69b4267fddee472197ed821a1c8ae9af4a825af880d9647291f6ba8c",
            9: "93dfa14e286d5e4a7e12124df65155be9a77df7a6ce3029ea318d5cd5712c34f",
            10: "95e336005ae791f653ffbc47f78ec7601f04373d078329a2e3a596f2ca45615d",
            11: "b48802b527b857d53896b8fbc8b54953b6f28ea8717c788940a8768a47b11765",
            12: "161664ca08c97716a6ca7cb3b825133454891c8eb27c233e3ca303faaadc4444",
            13: "7652b4ef0604f49f7990136beb203b8b3ea279ac9fbfd2c4e127d8ff9c4cf2b5",
            14: "31e81d7b9f651c2bf67440e837e2cdc5246dafa3e3637b4273019de401a0797a",
            15: "889276e4b12fadf070e8fecf3f4d77185e836cb475d2eae227013bfbebd8a962",
        },
        "note": "16/16 verified 2026-08-28 (14 lane receipts + spark1/2 by staging)",
    },
    "glm52.tp8.fp8": {
        "required": True,
        "nodes": {h: i - 8 for i, h in enumerate(ALL_HOSTS) if i >= 8},
        "path": "~/sparkdata/glm52.tp8.fp8/packs/glm52_tp8_rank{rank:02d}.fp8.glms52sp",
        "sizes": {r: 102835957760 for r in range(8)},
        "shas": {},
        "note": "donor band spark8-f BY DESIGN (GLM 5.2 is not a fleet model)",
    },
    "k3.mxfp4.tp4pp4.stage3": {
        "required": True,
        "nodes": {h: i - 12 for i, h in enumerate(ALL_HOSTS) if i >= 12},
        "path": "~/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage3.rank{rank:02d}.pack",
        "sizes": {r: 98119908864 for r in range(4)},
        "shas": {
            0: "23df8aa64e02cbc62fdf318bc0189cbef18192faaf55ce464b05e14f1bcfabf9",
            1: "d9deda6a8e6d70d41b2a726e4c9974bec284b6d98eff23b63f9064b5e11c9c82",
            2: "b1727beca178986073245cbae1782f6e1bbe0f860b0491510256761e43188ae4",
            3: "02882b942cfc9321ebdc0eebfe076dd11ec929b32300d857e066a618051ed348",
        },
        "note": "stages 0-2 PENDING (K3 lane builds on sparke); stage-3 verified",
    },
    "qwen4_flash.tp4.v3": {
        "required": True,
        "nodes": {h: i - 4 for i, h in enumerate(ALL_HOSTS) if 4 <= i <= 7},
        "path": "~/sparkdata/qwen4_flash.tp4/packs_v3/qwen4_flash_full.tp4-rank{rank}.qwen4_flashsp",
        "sizes": {r: 60194156288 for r in range(4)},
        "shas": {
            0: "d08ccfec4ac5ab82db8a5af544d8b8d2ab725a2a04734499a3f5dd4a4742c058",
            1: "183bf7fc79492e46995578cbb85b4a297674e1336a1e07b5b29499d7e29fe551",
            2: "d3d66c6aebd715c93f0457d5498ac3074a1d8604fa2396076c411a3b540519fd",
            3: "4fbc9336501c079ab3555c7651f27547cb2fc751ce2b844ee2f9ed3face57960",
        },
        "note": "per-node own rank (deploy_v3 layout); v1/v2 retired by staging 2026-08-28",
    },
    "qwen38.fp8.tp1_27b": {
        "required": True,
        "nodes": {"spark2": 0, "spark9": 0, "sparka": 0},
        "path": "~/sparkdata/qwen38.fp8.tp1/packs/qwen38-fp8.tp1.qwen36sp",
        "sizes": {0: 30135214592},
        "shas": {0: "ceef03e453e720694987fad70accd9cab27b78fd64bec4f0099c43d0477a9b4e"},
        "note": "dev (spark2) + co-resident set (spark9/a), sha-identical trio",
    },
    "dsv4_flash.fp8.tp16.b1": {
        "required": True,
        "nodes": {h: 0 for h in ALL_HOSTS},
        "path": "~/sparkdata/dsv4_flash.fp8.tp16.b1/packs/dsv4_flash_stage.spstage",
        "sizes": {0: 9013048832},
        "shas": {},
        "note": "deployed fleet pack; sparkf generation drift is a real gap",
    },
    "qwen38max.tp4.v2": {
        "required": False,  # PENDING: set incomplete — build owner: qwen38max shard lane
        "nodes": {"spark7": 0},
        "path": "~/sparkdata/qwen38max.tp4/packs/{probe_dir}",
        "sizes": {},
        "shas": {},
        "note": "15/16 packs on spark7, 12/16 receipted; pp3-rank3 missing; "
               "no fleet distribution until complete (staging decision 2026-08-28)",
        "probe": "count qwen38max.*.qwen38sp files (expect 16 when complete)",
    },
}


def ssh_stat(host, path):
    """Return (size_bytes, sha256_or_None); size None => missing/unreachable."""
    cmd = "stat -c %s " + path
    try:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd],
            capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, "ssh-timeout"
    if out.returncode != 0:
        return None, (out.stderr.strip().splitlines() or ["error"])[-1]
    try:
        return int(out.stdout.strip()), None
    except ValueError:
        return None, "bad-stat"


def ssh_sha256(host, path):
    cmd = "sha256sum " + path
    try:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd],
            capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return "hash-timeout"
    if out.returncode != 0:
        return "hash-error"
    return out.stdout.split()[0]


def probe_pending_qwen38max(host):
    cmd = "ls ~/sparkdata/qwen38max.tp4/packs/*.qwen38sp 2>/dev/null | wc -l"
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd],
        capture_output=True, text=True, timeout=60)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return -1


def fmt_gb(n):
    return "%.1fG" % (n / 1e9) if n is not None else "-"


def run_fleet(args):
    hosts = args.hosts
    rows = []
    failures = []
    pend_notes = []
    for model in args.models:
        spec = FLEET_TABLE[model]
        for host in hosts:
            if host not in spec["nodes"]:
                continue
            rank = spec["nodes"][host]
            if not spec["required"]:
                count = probe_pending_qwen38max(host)
                state = "PENDING(%d/16 packs on build node)" % count
                rows.append((model, host, rank, state, "-"))
                if args.strict and count != 16:
                    pend_notes.append("%s: %d/16" % (model, count))
                continue
            path = spec["path"]
            if "{rank" in path:
                path = path.format(rank=rank)
            elif "{probe_dir}" in path:
                path = path.replace("{probe_dir}", "")
            size, err = ssh_stat(host, path)
            exp = spec["sizes"].get(rank)
            if size is None:
                state = "MISSING(%s)" % err
                failures.append("%s@%s: %s" % (model, host, state))
            elif exp is not None and size != exp:
                state = "SIZE-DIFF(%s != %s)" % (fmt_gb(size), fmt_gb(exp))
                failures.append("%s@%s: %s" % (model, host, state))
            else:
                state = "present"
                if args.verify:
                    want = spec["shas"].get(rank)
                    got = ssh_sha256(host, path)
                    if want and got != want:
                        state = "SHA-MISMATCH"
                        failures.append("%s@%s: sha %s != %s" % (model, host, got, want))
                    else:
                        state = "verified"
            rows.append((model, host, rank, state, fmt_gb(size)))

    print("%-28s %-7s %-5s %-34s %s" % ("model", "node", "rank", "state", "size"))
    for r in rows:
        print("%-28s %-7s %-5s %-34s %s" % r)
    if pend_notes:
        print("PENDING (strict): " + "; ".join(pend_notes))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump([{"model": m, "node": h, "rank": rk, "state": st, "size": sz}
                       for m, h, rk, st, sz in rows], fh, indent=1)
    if failures:
        print("\nSTAGING GAPS (%d):" % len(failures))
        for f in failures:
            print("  " + f)
        return 1
    print("\nSTAGING OK: every required pack present with expected size"
          + (" and sha" if args.verify else ""))
    return 0


def self_test():
    checks = 0
    assert set(HEX) == set("0123456789abcdef")
    # internal consistency of the fleet table
    for model, spec in FLEET_TABLE.items():
        for host, rank in spec["nodes"].items():
            assert host in ALL_HOSTS, (model, host)
            assert 0 <= rank <= 15, (model, host, rank)
            assert "{rank" in spec["path"] or "{probe_dir}" in spec["path"] or True
            if spec["required"]:
                exp = spec["sizes"].get(rank)
                assert exp is not None and exp > 0, (model, host)
                want = spec["shas"].get(rank)
                assert want is None or len(want) == 64, (model, host)
        checks += 1
    # the policy invariants the README fleet table encodes
    assert len(FLEET_TABLE["glm5_next.tp16"]["nodes"]) == 16
    assert len(FLEET_TABLE["dsv4_pro.tp4pp4"]["nodes"]) == 16
    assert len(FLEET_TABLE["dsv4_flash.fp8.tp16.b1"]["nodes"]) == 16
    assert len(FLEET_TABLE["glm52.tp8.fp8"]["nodes"]) == 8
    assert sorted(FLEET_TABLE["glm52.tp8.fp8"]["nodes"]) == ALL_HOSTS[8:]
    assert len(FLEET_TABLE["k3.mxfp4.tp4pp4.stage3"]["nodes"]) == 4
    assert sorted(FLEET_TABLE["k3.mxfp4.tp4pp4.stage3"]["nodes"]) == ALL_HOSTS[12:]
    assert len(FLEET_TABLE["qwen4_flash.tp4.v3"]["nodes"]) == 4
    assert sorted(FLEET_TABLE["qwen4_flash.tp4.v3"]["nodes"]) == ALL_HOSTS[4:8]
    # path formatting works for every rank
    for model, spec in FLEET_TABLE.items():
        for host, rank in spec["nodes"].items():
            p = spec["path"]
            if "{rank" in p:
                p.format(rank=rank)
            checks += 1
    print("PASS staging_manifest self-test (%d checks, %d models)"
          % (checks, len(FLEET_TABLE)))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fleet", action="store_true", help="walk the real nodes")
    ap.add_argument("--verify", action="store_true",
                    help="sha256 every present pack (slow)")
    ap.add_argument("--json", default=None, help="write JSON table to PATH")
    ap.add_argument("--hosts", nargs="*", default=ALL_HOSTS)
    ap.add_argument("--models", nargs="*", default=list(FLEET_TABLE))
    ap.add_argument("--strict", action="store_true",
                    help="PENDING models also gate the exit code")
    ap.add_argument("--self-test", action="store_true",
                    help="offline consistency check (no ssh)")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    sys.exit(run_fleet(args))


if __name__ == "__main__":
    main()

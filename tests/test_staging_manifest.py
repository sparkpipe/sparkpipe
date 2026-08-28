#!/usr/bin/env python3
"""Offline gate for tools/staging_manifest.py (PYTHON_TESTS registration).

The fleet walk itself needs ssh to the sparks and runs from the controller
(`python3 tools/staging_manifest.py --fleet [--verify]`); this test keeps
the CHECKED STATE machinery healthy without nodes: it validates the fleet
table's invariants, exercises the table/format logic end to end against
simulated node catalogs (including the negative cases that must exit 1),
and proves the CLI wiring.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "staging_manifest.py"

spec = importlib.util.spec_from_file_location("staging_manifest", TOOL)
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)

checks = 0


def ok(label, cond):
    global checks
    assert cond, label
    checks += 1


# 1. Self-test subcommand exits 0.
r = subprocess.run([sys.executable, str(TOOL), "--self-test"],
                   capture_output=True, text=True)
ok("self-test exit 0", r.returncode == 0)
ok("self-test reports PASS", "PASS staging_manifest self-test" in r.stdout)

# 2. Fleet table policy invariants (the all-16 policy shape).
t = sm.FLEET_TABLE
ok("glm5_next all 16", len(t["glm5_next.tp16"]["nodes"]) == 16)
ok("dsv4_pro all 16", len(t["dsv4_pro.tp4pp4"]["nodes"]) == 16)
ok("dsv4_flash all 16", len(t["dsv4_flash.fp8.tp16.b1"]["nodes"]) == 16)
ok("glm52 donor band = spark8-f", sorted(t["glm52.tp8.fp8"]["nodes"]) == sm.ALL_HOSTS[8:])
ok("k3 stage3 = sparkc-f", sorted(t["k3.mxfp4.tp4pp4.stage3"]["nodes"]) == sm.ALL_HOSTS[12:])
ok("qwen4_flash v3 = spark4-7", sorted(t["qwen4_flash.tp4.v3"]["nodes"]) == sm.ALL_HOSTS[4:8])
ok("qwen38max is PENDING (not required)", t["qwen38max.tp4.v2"]["required"] is False)

# 3. Every required cell pins a size; every pinned sha is a sha256.
for model, spec_ in t.items():
    for host, rank in spec_["nodes"].items():
        if spec_["required"]:
            ok("%s/%s size pinned" % (model, host),
               spec_["sizes"].get(rank, 0) > 0)
            sha = spec_["shas"].get(rank)
            ok("%s/%s sha wellformed" % (model, host),
               sha is None or (len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)))

# 4. Simulated walk: a complete catalog exits 0, each defect class exits 1.
def fake_stat_catalog(catalog):
    def ssh_stat(host, path):
        size = catalog.get((host, path))
        return (size, None) if size is not None else (None, "No such file")
    return ssh_stat


good = {}
for model, spec_ in t.items():
    if not spec_["required"]:
        continue
    for host, rank in spec_["nodes"].items():
        p = spec_["path"]
        if "{rank" in p:
            p = p.format(rank=rank)
        good[(host, p)] = spec_["sizes"][rank]

import argparse

def run_walk(catalog, models=None):
    sm.ssh_stat = fake_stat_catalog(catalog)
    sm.probe_pending_qwen38max = lambda host: 16
    args = argparse.Namespace(
        hosts=sm.ALL_HOSTS, models=models or list(sm.FLEET_TABLE),
        verify=False, json=None, strict=False)
    return sm.run_fleet(args)


class GapCatcher(Exception):
    """run_fleet calls sys.exit; catch it to read the code."""


import contextlib
import io


def quiet_walk(catalog, models=None):
    """Run the walk with the table silenced; return its exit code."""
    with contextlib.redirect_stdout(io.StringIO()):
        return run_walk(catalog, models=models)


code = quiet_walk(good)
ok("complete catalog exits 0", code == 0)

missing = dict(good)
missing.pop(("sparkf", t["glm5_next.tp16"]["path"].format(rank=15)))
code = quiet_walk(missing)
ok("missing pack exits 1", code == 1)

wrong = dict(good)
wrong[("sparkc", "~/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage3.rank00.pack")] = 1
code = quiet_walk(wrong)
ok("size drift exits 1", code == 1)

sub = {k: v for k, v in good.items()
       if k[0] in ("spark4", "spark5", "spark6", "spark7")}
sub[("spark4", t["qwen4_flash.tp4.v3"]["path"].format(rank=0))] = 60194156288
code = quiet_walk(sub, models=["qwen4_flash.tp4.v3"])
ok("single-model subset walk exits 0", code == 0)

# 5. --hosts restriction works (no spark8+ hosts probed for glm52 subset).
sm.ssh_stat = fake_stat_catalog(good)
seen = []


def spy_stat(host, path):
    seen.append(host)
    return fake_stat_catalog(good)(host, path)


sm.ssh_stat = spy_stat
args = argparse.Namespace(hosts=["spark8"], models=["glm52.tp8.fp8"],
                          verify=False, json=None, strict=False)
with contextlib.redirect_stdout(io.StringIO()):
    sm.run_fleet(args)
ok("--hosts restricts probes", seen == ["spark8"])

print("PASS staging_manifest offline gate (%d checks)" % checks)

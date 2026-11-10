#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane

MODEL = "dsv41_flash"
LANE = 10
HEX = "0123456789abcdef"
FAKE_SSH = """#!/usr/bin/env bash
command="${@: -1}"
case "$command" in
  "systemctl --user is-active "*) echo inactive; exit 3 ;;
esac
exec bash -c "${command//\\/home\\//$FAKE_ROOT/home/}"
"""
FAKE_SCP = """#!/usr/bin/env bash
destination="${@: -1}"
target="$FAKE_ROOT/home/${destination#*:/home/}"
sources=()
for argument in "$@"; do case "$argument" in -*) ;; *) sources+=("$argument") ;; esac; done
unset 'sources[${#sources[@]}-1]'
for source in "${sources[@]}"; do
  if [ "$(basename "$source")" = weightd_spine_budget.py ]; then
    if [ -d "$target" ]; then out="$target/weightd_spine_budget.py"; else out="$target"; fi
    printf 'import os, sys\\nprint(os.path.getsize(os.path.realpath(sys.argv[1])) + 7)\\n' > "$out"
  else
    cp "$source" "$target"
  fi
done
"""
SYSTEMD_RUN = """#!/usr/bin/env bash
printf '%s\\n' "$*" > "$FAKE_ROOT/systemd-run.args"
"""


def write(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def main():
    model = stage_lane.MODELS[MODEL]
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        fake = pathlib.Path(scratch)
        tools = fake / "bin"
        write(tools / "ssh", FAKE_SSH, 0o755)
        write(tools / "scp", FAKE_SCP, 0o755)
        write(tools / "systemd-run", SYSTEMD_RUN, 0o755)
        write(tools / "systemctl", "#!/usr/bin/env bash\nexit 0\n", 0o755)
        firmware = fake / "firmware"
        for name in ("sparkpipe_model_residentd", "weightd_warm", "libhidden_transport_spark_host_rdma_verbs.so", model["adapter"]):
            write(firmware / name, "binary\n")
        write(firmware / "runtime_libs" / "libsparkpipe_device_cuda.so", "binary\n")
        digests, sizes = {}, {}
        for rank in range(16):
            host = f"spark{HEX[rank]}"
            pack = fake / "home" / host / model["pack"].format(rank=rank)
            pack.parent.mkdir(parents=True, exist_ok=True)
            sizes[rank] = 4096 + rank
            pack.write_bytes(bytes([rank]) * sizes[rank])
            digests[rank] = f"{rank:02x}" * 32
            (pack.parent / f"{pack.name}.sha256").write_text(f"{digests[rank]}  {pack.name}\n")
            (pack.parent / f"{pack.name}.experts").write_bytes(b"\0" * 64)
            for name in model.get("pack_siblings", ()):
                (pack.parent / name).write_bytes(b"x")
        environment = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", FAKE_ROOT=str(fake), STAGE_MODEL=MODEL, STAGE_LANE=str(LANE),
                           STAGE_FIRMWARE=str(firmware), STAGE_WEIGHTD_SOCKET="/tmp/spark_weightd.sock", STAGE_MEMORY_MAX="1G")
        result = subprocess.run(["bash", str(ROOT / "tools" / "stage_lane.sh"), "setup"], env=environment, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            failures.append(f"setup exited {result.returncode}: {result.stderr.strip()[-400:]}")
        for rank in range(16):
            host = f"spark{HEX[rank]}"
            root = fake / "home" / host / "stage-lanes" / MODEL / f"lane{LANE}" / "root"
            adapter = root / "config" / "adapter.json"
            if not adapter.exists():
                failures.append(f"{host}: setup wrote no config/adapter.json")
                continue
            configured = pathlib.Path(json.loads(adapter.read_text())["stage_pack_path"].replace("/home/", f"{fake}/home/", 1))
            digests_in_packs = sorted(path.name for path in (root / "packs").iterdir() if path.name.endswith(".sha256"))
            if digests_in_packs != ["pack.sha256"]:
                failures.append(f"{host}: packs/ holds {digests_in_packs}; residentd reads exactly one packs/*.sha256")
            if not configured.exists() or not pathlib.Path(f"{configured}.experts").exists():
                failures.append(f"{host}: the configured pack {configured.name} or its .experts is missing")
                continue
            if (root / "packs" / "pack.sha256").read_text() != digests[rank]:
                failures.append(f"{host}: packs/pack.sha256 is not the pack's digest")
            for local in (root / "lane.sh", adapter):
                local.write_text(local.read_text().replace("/home/", f"{fake}/home/"))
            script = root / "lane.sh"
            (fake / "systemd-run.args").unlink(missing_ok=True)
            started = subprocess.run(["bash", str(script), "start", f"run{rank}"], env=environment, capture_output=True, text=True, timeout=60)
            arguments = (fake / "systemd-run.args").read_text() if (fake / "systemd-run.args").exists() else ""
            if started.returncode != 0 or f"SPARK_WEIGHTD_SPINE_BUDGET_BYTES={sizes[rank] + 7} " not in arguments:
                failures.append(f"{host}: lane.sh start exited {started.returncode} ({started.stderr.strip()[-200:]}) or launched without the configured pack's spine budget: {arguments.strip()[:160]}")
            after = sorted(path.name for path in (root / "packs").iterdir() if path.name.endswith(".sha256"))
            if after != ["pack.sha256"] or (root / "packs" / "pack.sha256").read_text() != digests[rank]:
                failures.append(f"{host}: lane.sh start left packs/ with {after} or changed the digest")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS stage_lane dense setup: a dense lane's packs/ holds exactly one digest, packs/pack.sha256, "
          "which is the pack's own; lane.sh start keeps that digest and launches with the configured pack's spine budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())

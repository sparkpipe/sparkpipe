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
[ "${destination#*:}" = "$destination" ] && exit 1
case "$destination" in *:/home/*) ;; *) target="$FAKE_ROOT/home/${destination%%:*}/${destination#*:}" ;; esac
sources=()
for argument in "$@"; do case "$argument" in -*) ;; *) sources+=("$argument") ;; esac; done
unset 'sources[${#sources[@]}-1]'
for source in "${sources[@]}"; do
  if [ "$(basename "$source")" = weightd_spine_budget.py ]; then
    if [ -d "$target" ]; then out="$target/weightd_spine_budget.py"; else out="$target"; fi
    printf 'import os, sys\\nprint(os.path.getsize(sys.argv[1]))\\n' > "$out"
  else
    cp "$source" "$target"
  fi
done
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
        firmware = fake / "firmware"
        for name in ("sparkpipe_model_residentd", "weightd_warm", "libhidden_transport_spark_host_rdma_verbs.so", model["adapter"]):
            write(firmware / name, "binary\n")
        digests = {}
        for rank in range(16):
            host = f"spark{HEX[rank]}"
            pack = fake / "home" / host / model["pack"].format(rank=rank)
            pack.parent.mkdir(parents=True, exist_ok=True)
            pack.write_bytes(bytes([rank]) * (4096 + rank))
            digests[rank] = f"{rank:02x}" * 32
            (pack.parent / f"{pack.name}.sha256").write_text(f"{digests[rank]}  {pack.name}\n")
            (pack.parent / f"{pack.name}.experts").write_bytes(b"\0" * 64)
            for target in model.get("extra_packs", {}).values():
                extra = fake / "home" / host / target.format(rank=rank)
                extra.parent.mkdir(parents=True, exist_ok=True)
                extra.write_bytes(b"x")
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
            missing = [str(path) for path in (configured, pathlib.Path(f"{configured}.sha256"), pathlib.Path(f"{configured}.experts")) if not path.exists() or path.stat().st_size == 0]
            if missing:
                failures.append(f"{host}: the configured pack lacks what lane.sh start and the release precheck read: {missing}")
                continue
            if pathlib.Path(f"{configured}.sha256").read_text() != digests[rank]:
                failures.append(f"{host}: {configured.name}.sha256 is not the pack's digest")
            if (root / "packs" / "pack.sha256").read_text() != digests[rank]:
                failures.append(f"{host}: packs/pack.sha256 is not the pack's digest")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS stage_lane dense setup: every rank's configured pack has the digest and the manifest beside it that lane.sh start and the release precheck read, "
          "and the digest is the pack's own")
    return 0


if __name__ == "__main__":
    sys.exit(main())

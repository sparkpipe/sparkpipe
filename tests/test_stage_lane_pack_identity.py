#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
RECORD = "qwen3.8-27b"
SYSTEMD_RUN = """#!/usr/bin/env bash
printf '%s\\n' "$*" > "$FAKE_HOME/systemd-run.args"
"""
SYSTEMCTL = "#!/usr/bin/env bash\nexit 0\n"


def write(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def pack(directory, name, size, digest):
    path = directory / name
    path.write_bytes(b"\0" * size)
    (directory / f"{name}.sha256").write_text(f"{digest}  {name}\n")
    return path


def start(script, root, home, adapter_pack):
    (root / "config" / "adapter.json").write_text(json.dumps({"stage_pack_path": str(adapter_pack)}))
    environment = dict(os.environ, PATH=f"{home / 'bin'}:{os.environ['PATH']}", FAKE_HOME=str(home))
    result = subprocess.run(["bash", str(script), "start", "run1" if not (root / "logs").exists() else f"run{len(list((root / 'logs').iterdir())) + 1}"],
                            env=environment, capture_output=True, text=True)
    (root / "logs" / "placeholder").mkdir(parents=True, exist_ok=True)
    return result, (home / "systemd-run.args").read_text() if (home / "systemd-run.args").exists() else ""


def main():
    failures = []
    with tempfile.TemporaryDirectory() as temporary:
        base = pathlib.Path(temporary)
        rendered, root, home, packs = base / "rendered", base / "root", base / "home", base / "sparkdata"
        rendered.mkdir()
        packs.mkdir()
        subprocess.run([sys.executable, str(ROOT / "tools" / "lane_config.py"), "render", RECORD, str(rendered)], check=True, capture_output=True)
        lane_root = json.loads((ROOT / "deployment" / "lanes.json").read_text())["lanes"][RECORD]["runtime_root"].format(host="spark0")
        script = rendered / "lane.spark0.sh"
        text = script.read_text()
        if f"cd {lane_root} " not in text:
            failures.append(f"the rendered lane.sh does not enter {lane_root}")
        script.write_text(text.replace(f"cd {lane_root} ", f"cd {root} "))
        for name in ("bin", "packs", "config"):
            (root / name).mkdir(parents=True)
        write(home / "bin" / "systemd-run", SYSTEMD_RUN, 0o755)
        write(home / "bin" / "systemctl", SYSTEMCTL, 0o755)
        (root / "packs" / "pack.sha256").write_text("f" * 64)
        first = pack(packs, "fp8.rank00.pack", 1000, "a" * 64)
        second = pack(packs, "bf16.rank00.pack", 3000, "b" * 64)
        for adapter_pack, digest in ((first, "a" * 64), (second, "b" * 64)):
            result, arguments = start(script, root, home, adapter_pack)
            if result.returncode != 0:
                failures.append(f"start with {adapter_pack.name} exited {result.returncode}: {result.stderr.strip()[-300:]}")
                continue
            if (root / "packs" / "pack.sha256").read_text() != digest:
                failures.append(f"start with {adapter_pack.name} left packs/pack.sha256 {(root / 'packs' / 'pack.sha256').read_text()!r}, not the configured pack's digest")
            if "SPARK_" in arguments or "--deployment deployment.json" not in arguments:
                failures.append(f"start with {adapter_pack.name} launched residentd with SPARK_ inputs or without its deployment ({arguments.strip()[:120]})")
        printed = subprocess.run(["bash", str(script), "pack"], capture_output=True, text=True).stdout.strip()
        if printed != str(second):
            failures.append(f"lane.sh pack printed {printed!r}, not the configured {second}")
        (packs / "bf16.rank00.pack.sha256").unlink()
        result, _ = start(script, root, home, second)
        if result.returncode == 0:
            failures.append("start with a configured pack that has no .sha256 sidecar succeeded")
        if (root / "packs" / "pack.sha256").read_text() != "b" * 64:
            failures.append("a refused start changed packs/pack.sha256")
    swap = json.loads((ROOT / "deployment" / "hub" / "model_swap.json").read_text())
    lanes = json.loads((ROOT / "deployment" / "lanes.json").read_text())["lanes"]
    for entry in swap["models"]:
        warm = entry.get("node_warm", "")
        if lanes.get(entry["id"], {}).get("launcher") == "stage_lane" and warm and ("readlink" in warm or "$(./lane.sh pack)" not in warm):
            failures.append(f"{entry['id']}: the swap entry warms {warm.split('weightd_warm', 1)[-1][:80]!r}, not the pack its config names ($(./lane.sh pack))")
    for failure in failures:
        print("FAIL " + failure)
    if failures:
        return 1
    print("PASS stage lane pack identity: lane.sh start takes the expected pack digest from the pack config/adapter.json names and hands residentd no SPARK_ input, "
          "so a pushed pack switch launches against the new pack, and a configured pack without a sidecar refuses to start; lane.sh pack prints that pack, "
          "and every stage lane's swap-controller warm command warms it")
    return 0


if __name__ == "__main__":
    sys.exit(main())

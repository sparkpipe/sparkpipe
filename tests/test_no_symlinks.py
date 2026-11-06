import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXEMPT_PREFIXES = ("tests/",)
EXEMPT_SUFFIXES = (".md",)
ALLOWED = (
    "tools/install_cuda13_compile_only_debian13.sh",
)
PENDING = (
    "deployment/hub/model_swap.json",
    "tools/devcycle.sh",
    "tools/devcycle/bbench.sh",
    "tools/devcycle/ds4_spark_brickproof.py",
    "tools/devcycle/r3flash_exact_cell.sh",
    "tools/devcycle/templates/run-family-job.sh.template",
    "tools/fleet_node_agent.sh",
    "tools/fleet_serve.sh",
    "tools/glm5_next_coldlaunch_ab.sh",
    "tools/glm5_next_pack_swap.sh",
    "tools/inference_smoke.py",
    "tools/k3_lane.sh",
    "tools/k3_multidev_run_family.sh",
    "tools/lag_t1_launch.sh",
    "tools/lane_run_log.sh",
    "tools/metal_gates.sh",
    "tools/mimo26_model_inputs.py",
    "tools/multi_dev_orchestrate.py",
    "tools/replay_shared_serving_release.py",
    "tools/stage_lane.sh",
)
LINK_CALL = re.compile(
    r"os\.symlink\s*\("
    r"|\.symlink_to\s*\("
    r"|(?<![\w.])symlink(?:at)?\s*\("
    r"|(?<![\w./-])ln\s+-[A-Za-z]*s[A-Za-z]*(?![\w-])"
)
PLANTED = (
    "os.symlink(source, target)",
    "path.symlink_to(source)",
    "if ( symlink(a,b) != 0 )",
    "symlinkat(a, fd, b);",
    "ln -s a b",
    "cd x && ln -sfn $pack $root/packs/",
    '"ln -sf stage.json config/stage.json"',
)
CLEAN = (
    "path.is_symlink()",
    "os.readlink(path)",
    "S_ISLNK(st.st_mode)",
    "unlink(path)",
    "os.link(temporary, path)",
    "kiln -s",
    "--ln-s",
    "ln a b",
    "lnk -s a b",
)


def listed_files():
    run = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True)
    if run.returncode != 0:
        stderr = run.stderr.decode("utf-8", "replace").splitlines()
        return None, f"git ls-files exited {run.returncode}: {stderr[0] if stderr else ''}"
    entries = run.stdout.decode("utf-8", "surrogateescape").split("\0")
    return sorted({entry for entry in entries if entry}), None


def exempt(relative):
    return relative.startswith(EXEMPT_PREFIXES) or relative.endswith(EXEMPT_SUFFIXES)


def main():
    failures = 0
    for planted in PLANTED:
        if LINK_CALL.search(planted) is None:
            print(f"  FAIL the link pattern does not see {planted!r}")
            failures += 1
    for clean in CLEAN:
        if LINK_CALL.search(clean) is not None:
            print(f"  FAIL the link pattern sees a link in {clean!r}")
            failures += 1
    entries, error = listed_files()
    if error is not None:
        print(f"  FAIL file list: {error}")
        return 1
    creators = set()
    checked = 0
    for relative in entries:
        path = ROOT / relative
        if path.is_symlink():
            print(f"  FAIL {relative}: the repository holds a symbolic link")
            failures += 1
            continue
        if not path.is_file() or exempt(relative):
            continue
        checked += 1
        text = path.read_text(errors="surrogateescape")
        for number, line in enumerate(text.splitlines(), 1):
            if LINK_CALL.search(line):
                creators.add(relative)
                if relative not in PENDING and relative not in ALLOWED:
                    print(f"  FAIL {relative}:{number}: creates a link; reference the data by path in configuration, or copy a small file")
                    failures += 1
    for entry in PENDING + ALLOWED:
        if entry not in creators:
            print(f"  FAIL {entry} creates no link or is gone: remove it from {'PENDING' if entry in PENDING else 'ALLOWED'}")
            failures += 1
    for entry in PENDING:
        if entry in ALLOWED:
            print(f"  FAIL {entry} is both pending and allowed")
            failures += 1
    print(f"files checked {checked}, link creators pending conversion {len(PENDING)}, allowed {len(ALLOWED)}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nPASS no symbolic links to data")
    return 0


if __name__ == "__main__":
    sys.exit(main())

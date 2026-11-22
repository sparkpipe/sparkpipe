#!/usr/bin/env python3
"""Run scripts/release_build.sh against a fake build host and hub: the release directory holds every lane adapter with its kernels
directory beside it, SHA256SUMS and SHA16 cover each kernels file, and an adapter built without its kernels directory refuses the
release, naming it, before the release directory is written."""
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release_build.sh"
SHIMS = {
    "ssh": """#!/usr/bin/env bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host="$1"; shift
mkdir -p "$FAKE_FLEET/$host"
HOME="$FAKE_FLEET/$host" exec bash -c "$*"
""",
    "scp": """#!/usr/bin/env bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host="${2%%:*}"
target="${2#*:}"
case "$target" in /*) ;; *) target="$FAKE_FLEET/$host/$target" ;; esac
mkdir -p "$(dirname "$target")" && cp "$1" "$target"
""",
    "make": """#!/usr/bin/env bash
for target in "$@"; do
	case "$target" in
		build/*)
			mkdir -p "$(dirname "$target")" && printf 'built %s\\n' "$target" > "$target"
			case "$target" in
				*_serving_adapter.so)
					case " ${FAKE_NO_KERNELS:-} " in *" $(basename "$target") "*) ;; *)
						mkdir -p "$target.kernels"
						printf 'cubin of %s\\n' "$target" > "$target.kernels/stage_runner.cubin"
						printf 'manifest of %s\\n' "$target" > "$target.kernels/stage_runner.manifest" ;;
					esac ;;
			esac ;;
	esac
done
""",
    "shasum": """#!/usr/bin/env python3
import hashlib, sys
for name in [argument for argument in sys.argv[1:] if argument not in ("-a", "256")]:
    print(hashlib.sha256(open(name, "rb").read()).hexdigest() + "  " + name)
""",
    "sha256sum": """#!/usr/bin/env python3
import hashlib, sys
for name in sys.argv[1:]:
    print(hashlib.sha256(open(name, "rb").read()).hexdigest() + "  " + name)
""",
}


def check(condition, message, result=None):
    if not condition:
        detail = "" if result is None else f"\n--- stdout\n{result.stdout[-2000:]}\n--- stderr\n{result.stderr[-2000:]}"
        raise SystemExit(f"FAIL test_release_build: {message}{detail}")


def run(work, no_kernels=""):
    shims = work / "shims"
    shims.mkdir(exist_ok=True)
    for name, body in SHIMS.items():
        (shims / name).write_text(body)
        (shims / name).chmod(0o755)
    out = work / "release"
    env = dict(os.environ, PATH=f"{shims}:{os.environ['PATH']}", FAKE_FLEET=str(work / "fleet"), REPO=str(ROOT), RELEASE_OUT=str(out),
               RELEASE_WORK=str(work / "scratch"), HUB_WORK=str(work / "fleet" / "hub-work"), BUILD_HOST="builder", FAKE_NO_KERNELS=no_kernels)
    env.pop("HOSTED", None)
    result = subprocess.run(["bash", str(SCRIPT), "HEAD"], env=env, capture_output=True, text=True, timeout=600)
    return result, out


def adapters_of(result):
    for line in result.stdout.splitlines():
        if line.startswith("release_build: adapters from deployment/lanes.json at "):
            return sorted(name[len("build/"):] for name in line.split(": ", 2)[2].split())
    return []


def main():
    with tempfile.TemporaryDirectory(prefix="release-build-") as directory:
        work = pathlib.Path(directory)
        result, out = run(work)
        check(result.returncode == 0, "release_build failed on a build host that installs every kernels directory", result)
        adapters = adapters_of(result)
        check(len(adapters) > 1, f"release_build named {adapters} as the lane adapters", result)
        listed = {}
        for line in (out / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            listed[name] = digest
        short = {line[16:].strip() for line in (out / "SHA16").read_text().splitlines()}
        for adapter in adapters:
            check((out / adapter).is_file(), f"the release has no {adapter}", result)
            for name in ("stage_runner.cubin", "stage_runner.manifest"):
                relative = f"{adapter}.kernels/{name}"
                path = out / relative
                check(path.is_file(), f"the release has no {relative} beside {adapter}", result)
                check(listed.get(relative) == hashlib.sha256(path.read_bytes()).hexdigest(), f"SHA256SUMS does not cover {relative}", result)
                check(relative in short, f"SHA16 does not list {relative}", result)
        for name, digest in listed.items():
            check(hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, f"SHA256SUMS lists a wrong digest for {name}", result)

    with tempfile.TemporaryDirectory(prefix="release-build-") as directory:
        work = pathlib.Path(directory)
        first, _ = run(work)
        missing = adapters_of(first)[0]
        work = work / "again"
        work.mkdir()
        result, out = run(work, no_kernels=missing)
        check(result.returncode != 0, f"release_build exited 0 with {missing} built without its kernels directory", result)
        check(f"{missing}.kernels" in result.stderr, f"release_build did not name {missing}.kernels", result)
        check(not out.exists(), "release_build wrote the release directory before refusing", result)
    print("PASS test_release_build: the release directory holds every lane adapter with its kernels directory, SHA256SUMS and SHA16 cover "
          "each kernels file, and an adapter built without its kernels directory refuses the release before anything is written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

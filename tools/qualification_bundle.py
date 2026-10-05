#!/usr/bin/env python3
"""Assemble one immutable qualification bundle for a release.

The bundle holds the merged commit, the release generation, the package and
driver hashes from the release root's SHA256SUMS, every rank's identity, the
token stream, accuracy and performance receipts, route counters and the drained
queue state. Every part is required. Each rank must report the driver the
release ships, and the drained state must show no live requests and no latched
failure. The output directory must not exist; the bundle is written once with
its own SHA256SUMS.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

PARTS = ("token_stream", "accuracy", "performance", "route_counters")


class BundleError(Exception):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path, label):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise BundleError("%s %s is not readable JSON: %s" % (label, path, error))


def release_hashes(release_root):
    sums = Path(release_root) / "SHA256SUMS"
    if not sums.is_file():
        raise BundleError("release root %s has no SHA256SUMS" % release_root)
    hashes = {}
    for line in sums.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line.strip())
        if match is None:
            raise BundleError("malformed SHA256SUMS line: %r" % line)
        hashes[match.group(2).removeprefix("./")] = match.group(1)
    drivers = {path: digest for path, digest in hashes.items() if path.endswith("model_driver.so")}
    if not drivers:
        raise BundleError("SHA256SUMS lists no model_driver.so")
    return hashes, drivers


def check_ranks(ranks, drivers):
    if not isinstance(ranks, list) or not ranks:
        raise BundleError("rank identities must be a non-empty list")
    seen = set()
    for record in ranks:
        for key in ("rank_index", "host", "driver_sha256", "pack_sha256"):
            if key not in record:
                raise BundleError("rank identity %s lacks %s" % (record, key))
        if record["rank_index"] in seen:
            raise BundleError("rank %s appears twice" % record["rank_index"])
        seen.add(record["rank_index"])
        if record["driver_sha256"] not in drivers.values():
            raise BundleError("rank %s runs driver %s, which the release does not ship" % (record["rank_index"], record["driver_sha256"]))
    if seen != set(range(len(ranks))):
        raise BundleError("rank identities must cover ranks 0..%d exactly" % (len(ranks) - 1))


def check_drained(state):
    if state.get("live_requests") != 0:
        raise BundleError("the engine still holds %s live requests" % state.get("live_requests"))
    if state.get("status") != "ok" or state.get("engine_status") != "ok":
        raise BundleError("the drained state is not healthy: %s" % state)
    if state.get("connected_ranks") != state.get("ranks"):
        raise BundleError("the drained state reports %s of %s ranks connected" % (state.get("connected_ranks"), state.get("ranks")))


def build(arguments):
    if re.fullmatch(r"[0-9a-f]{40}", arguments.commit) is None:
        raise BundleError("--commit must be a full 40-hex merged commit")
    if not arguments.generation:
        raise BundleError("--generation is required")
    output = Path(arguments.output)
    if output.exists():
        raise BundleError("%s exists; a qualification bundle is written once" % output)
    hashes, drivers = release_hashes(arguments.release_root)
    ranks = load_json(arguments.rank_identities, "rank identities")
    check_ranks(ranks, drivers)
    drained = load_json(arguments.drained_state, "drained state")
    check_drained(drained)
    parts = {}
    for name in PARTS:
        path = getattr(arguments, name)
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise BundleError("%s receipt %s is missing or empty" % (name, path))
        parts[name] = path
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    copied = {}
    for name, path in list(parts.items()) + [("rank_identities", arguments.rank_identities), ("drained_state", arguments.drained_state)]:
        target = staging / (name + Path(path).suffix)
        shutil.copyfile(path, target)
        copied[name] = {"file": target.name, "sha256": sha256_file(target), "source": str(path)}
    manifest = {
        "kind": "sparkpipe.qualification-bundle.v1",
        "commit": arguments.commit,
        "generation": arguments.generation,
        "release_root": str(arguments.release_root),
        "package_sha256": hashes,
        "driver_sha256": drivers,
        "rank_count": len(ranks),
        "parts": copied,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    lines = ["%s  %s" % (sha256_file(path), path.name) for path in sorted(staging.iterdir())]
    (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    for path in staging.iterdir():
        path.chmod(0o444)
    staging.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--rank-identities", required=True)
    parser.add_argument("--drained-state", required=True)
    for name in PARTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True, dest=name)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    try:
        manifest = build(arguments)
    except BundleError as error:
        print("qualification bundle refused: %s" % error, file=sys.stderr)
        return 1
    print("qualification bundle %s commit=%s generation=%s ranks=%d" % (arguments.output, manifest["commit"], manifest["generation"], manifest["rank_count"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

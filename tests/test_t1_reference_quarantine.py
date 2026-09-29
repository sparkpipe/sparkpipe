import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from t1_reference_common import (read_fixture, sha256_file,  # noqa: E402
                                 write_fixture, write_manifest)

COMPARE = os.path.join(ROOT, "tools", "t1_reference_compare.py")
T1_ROOT = os.path.join(ROOT, "qualification", "t1_reference")
DECODER_FEED_FIX_UTC = "2026-09-19T11:22:17Z"
STALE_E2M1_DIRECTORIES = (
    os.path.join(ROOT, "qualification", "t1_reference", "qwen38_max"),
    os.path.join(ROOT, "qualification", "t1_reference", "qwen38_27b", "nvfp4a16"),
)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def verify(directory):
    return subprocess.run([sys.executable, COMPARE, "verify-manifest", "--fixture-dir", directory],
                          capture_output=True, text=True)


def compare(reference, candidate):
    return subprocess.run([sys.executable, COMPARE, "compare", "--reference", reference,
                           "--candidate", candidate], capture_output=True, text=True)


def refuses_read(path):
    try:
        read_fixture(path)
    except ValueError as error:
        return "quarantined" in str(error)
    return False


def committed_manifests():
    for directory, _, files in sorted(os.walk(T1_ROOT)):
        if "MANIFEST.json" in files:
            yield directory, json.load(open(os.path.join(directory, "MANIFEST.json")))


def check_decoder_feed(directory, manifest):
    generated = manifest.get("generated_utc", "")
    expect(generated, f"{directory}: the manifest records no generated_utc")
    if generated >= DECODER_FEED_FIX_UTC:
        return False
    quarantine = manifest.get("quarantine") or {}
    expect(sorted(quarantine.get("fixtures", [])) == sorted(manifest["fixtures"]),
           f"{directory}: generated {generated}, before the decoder double-feed fix; every fixture must be quarantined")
    expect("double-feed" in quarantine.get("reason", ""),
           f"{directory}: the quarantine must name the decoder double-feed")
    result = verify(directory)
    expect(result.returncode == 1 and "quarantined" in result.stdout,
           f"{directory}: verify-manifest must fail on the pre-fix fixtures: {result.stdout}")
    return True


def synthetic(workspace):
    directory = os.path.join(workspace, "family")
    os.makedirs(directory)
    path = os.path.join(directory, "synth.t1r")
    write_fixture(path, {"prompt_token_ids": np.arange(4, dtype=np.int32),
                         "generated_token_ids": np.arange(2, dtype=np.int32)})
    manifest = {"family": "synthetic",
                "fixtures": {"synth.t1r": {"bytes": os.path.getsize(path),
                                           "sha256": sha256_file(path)}}}
    write_manifest(os.path.join(directory, "MANIFEST.json"), manifest)
    return directory, path, manifest


def main():
    for directory in STALE_E2M1_DIRECTORIES:
        manifest = json.load(open(os.path.join(directory, "MANIFEST.json")))
        expect(sorted(manifest["quarantine"]["fixtures"]) == sorted(manifest["fixtures"]),
               f"{directory}: every stale e2m1 fixture must be quarantined")
        expect("e2m1" in manifest["quarantine"]["reason"],
               f"{directory}: the quarantine must name the e2m1 table")
        result = verify(directory)
        expect(result.returncode == 1 and "quarantined" in result.stdout,
               f"{directory}: verify-manifest must fail on quarantine: {result.stdout}")
        for name in manifest["fixtures"]:
            path = os.path.join(directory, name)
            expect(refuses_read(path), f"{path}: a quarantined fixture must not be read")
            refused = compare(path, path)
            expect(refused.returncode != 0 and "quarantined" in refused.stderr,
                   f"{path}: compare must refuse a quarantined reference")
    stale_feed = [os.path.relpath(directory, T1_ROOT) for directory, manifest in committed_manifests()
                  if check_decoder_feed(directory, manifest)]
    expect("glm5_next" in stale_feed, f"glm5_next predates the double-feed fix and must be quarantined: {stale_feed}")
    glm53flash = os.path.join(T1_ROOT, "glm53flash")
    current = verify(glm53flash)
    expect(current.returncode == 0, f"{glm53flash}: the GLM-5.3 Flash fixtures must verify: {current.stdout}")
    workspace = tempfile.mkdtemp(prefix="t1ref-quarantine-")
    try:
        directory, path, manifest = synthetic(workspace)
        clean = verify(directory)
        expect(clean.returncode == 0, f"clean manifest must verify: {clean.stdout}")
        expect(not refuses_read(path), "a clean fixture must read")
        manifest["quarantine"] = {"reason": "synthetic stale decode", "fixtures": ["synth.t1r"]}
        write_manifest(os.path.join(directory, "MANIFEST.json"), manifest)
        held = verify(directory)
        expect(held.returncode == 1 and "synthetic stale decode" in held.stdout,
               f"quarantine must fail verify-manifest and name the reason: {held.stdout}")
        expect(refuses_read(path), "quarantine must refuse read_fixture")
        manifest["quarantine"] = {"reason": "", "fixtures": ["synth.t1r"]}
        write_manifest(os.path.join(directory, "MANIFEST.json"), manifest)
        expect(verify(directory).returncode == 1, "a quarantine without a reason must fail verify")
        try:
            read_fixture(path)
            raise AssertionError("a quarantine without a reason must refuse the read")
        except ValueError as error:
            expect("reason" in str(error), f"malformed quarantine must be named: {error}")
        manifest["quarantine"] = {"reason": "synthetic", "fixtures": ["absent.t1r"]}
        write_manifest(os.path.join(directory, "MANIFEST.json"), manifest)
        stray = verify(directory)
        expect(stray.returncode == 1 and "not in the manifest" in stray.stdout,
               f"a quarantine naming an unknown fixture must fail: {stray.stdout}")
        expect(not refuses_read(path), "an unlisted fixture stays readable")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    print("PASS t1_reference quarantine: stale e2m1 fixtures refused by verify-manifest, "
          "read_fixture and compare; fixtures older than the decoder double-feed fix quarantined "
          f"({', '.join(stale_feed)}); glm53flash verifies; clean, quarantined and malformed synthetic manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

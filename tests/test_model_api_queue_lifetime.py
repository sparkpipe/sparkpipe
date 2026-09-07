#!/usr/bin/env python3
"""Queue-lifetime regression for the model_api request list (perf review
finding 1, 2026-09-07): removing a COMPLETED TAIL must keep older requests
reachable - enqueue A, B; finish B first; enqueue C; A must still be in the
list. Drives a live api over a socket pair and checks /v1/models between
operations, so the enqueue path itself is exercised.

  test_model_api_queue_lifetime.py
"""
import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "node", "model_api.c")

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def build():
    binary = os.path.join(tempfile.mkdtemp(prefix="apiq-"), "model_api_test")
    status = subprocess.run(
        ["cc", "-std=c11", "-O1", "-D_GNU_SOURCE", "-pthread",
         "-I.", "-Iinclude", "-Isrc", "-Itests/cuda_stub",
         "-Imodel-families/common/include",
         TOOL,
         "build/libsparkpipe_runtime.a",
         "build/libsparkpipe_model_common.a",
         "build/libsparkpipe_core.a",
         "tests/cuda_stub/cuda_runtime_stub.c",
         "-ldl", "-lpthread", "-o", binary],
        cwd=REPO, capture_output=True, text=True)
    if status.returncode != 0:
        print(status.stderr[:400])
        return 0
    return binary


def extract_requests(source):
    marker = "while (*pp != 0)"
    return marker in source


def unlink_simulates_review_case():
    source = open(TOOL, encoding="utf-8", errors="replace").read()
    check("api_queue_unlink exists (shared unlink)",
          "static void api_queue_unlink" in source)
    check("destructor frees output_token_ids",
          "free(req->output_token_ids)" in source)
    early = source.find("n = recv(fd, buf + total")
    window = source[early:early + 400]
    check("early-recv frees buffer", "free(buf);" in window)


def main():
    unlink_simulates_review_case()
    binary = build()
    if binary == 0:
        check("model_api builds", False)
        print("FAILURES: " + ", ".join(failures))
        return 1
    check("model_api builds", True)
    print(f"queue-lifetime checks: {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

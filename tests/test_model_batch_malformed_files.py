#!/usr/bin/env python3
"""Malformed batch files are rejected cleanly by sparkpipe_model_batch.

Every --batch input that fails validation must exit 1 with a single
sparkpipe_model_batch_status line and the documented SparkStatus - never a
signal. The over-capacity case is the load-bearing regression: ParseRequests
used to publish request_count before the capacity check could bail, so
FileDestroy walked a NULL array on teardown (SIGSEGV, found by feeding the
binary a two-request file against request_capacity=1).

Status codes pinned from include/sparkpipe/spark_status.h:
  1 INVALID_ARGUMENT   2 CAPACITY_EXCEEDED  4 NOT_FOUND
  5 PARSE_ERROR        6 SCHEMA_ERROR       16 DUPLICATE

The valid control asserts only "exit 1 with a status line": its engine-connect
failure depends on an empty runtime root, not on the loader.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(ROOT, "build/sparkpipe_model_batch")
DEPLOYMENT = os.path.join(
    ROOT, "examples/deployments/dsv4_flash_pp13_host_rdma.json")

INVALID_ARGUMENT = 1
CAPACITY_EXCEEDED = 2
NOT_FOUND = 4
PARSE_ERROR = 5
SCHEMA_ERROR = 6
DUPLICATE = 16


def base_document(**overrides):
    document = {
        "schema_version": 1,
        "connect_timeout_ms": 1000,
        "request_capacity": 4,
        "max_context_tokens": 4096,
        "max_prefill_rows_per_submission": 1,
        "maximum_messages_per_rank_per_progress": 8,
        "maximum_new_submissions_per_progress": 1,
        "stop_token_ids": [],
        "requests": [{
            "request_id": 760130,
            "sequence_id": 760130,
            "priority": 0,
            "output_token_budget": 8,
            "prompt_token_ids": [0, 3476, 477, 18068],
        }],
    }
    document.update(overrides)
    return document


def request(**overrides):
    entry = {
        "request_id": 760130,
        "sequence_id": 760130,
        "priority": 0,
        "output_token_budget": 8,
        "prompt_token_ids": [0, 3476, 477, 18068],
    }
    entry.update(overrides)
    return entry


def cases():
    # (name, document-or-None, raw-text-or-None, expected SparkStatus)
    return [
        ("not_json", None, "hello world this is not json\n", PARSE_ERROR),
        ("truncated", None, json.dumps(base_document())[:120], PARSE_ERROR),
        ("empty_file", None, "", PARSE_ERROR),
        ("root_array", None, "[1,2,3]", SCHEMA_ERROR),
        ("wrong_schema_version", base_document(schema_version=2), None,
         SCHEMA_ERROR),
        ("missing_member",
         {k: v for k, v in base_document().items()
          if k != "max_context_tokens"},
         None, SCHEMA_ERROR),
        ("unknown_member", dict(base_document(), extra_field=1), None,
         SCHEMA_ERROR),
        ("empty_requests", base_document(requests=[]), None,
         CAPACITY_EXCEEDED),
        ("over_capacity",
         base_document(request_capacity=1,
                       requests=[request(request_id=1, sequence_id=1),
                                 request(request_id=2, sequence_id=2)]),
         None, CAPACITY_EXCEEDED),
        ("empty_prompt_tokens",
         base_document(requests=[request(prompt_token_ids=[])]), None,
         CAPACITY_EXCEEDED),
        ("prompt_too_long",
         base_document(requests=[request(prompt_token_ids=[1] * 5000)]), None,
         CAPACITY_EXCEEDED),
        ("negative_token", None,
         json.dumps(base_document()).replace("18068", "-5"), SCHEMA_ERROR),
        ("float_token", None,
         json.dumps(base_document()).replace("18068", "1.5"), SCHEMA_ERROR),
        ("string_token", None,
         json.dumps(base_document()).replace("18068", '"abc"'),
         INVALID_ARGUMENT),
        ("bool_token", None,
         json.dumps(base_document()).replace("18068", "true"), SCHEMA_ERROR),
        ("null_token", None,
         json.dumps(base_document()).replace("18068", "null"), SCHEMA_ERROR),
        ("duplicate_request_id",
         base_document(requests=[request(request_id=9, sequence_id=1),
                                 request(request_id=9, sequence_id=2)]),
         None, DUPLICATE),
        ("duplicate_sequence_id",
         base_document(requests=[request(request_id=1, sequence_id=9),
                                 request(request_id=2, sequence_id=9)]),
         None, DUPLICATE),
        ("zero_budget",
         base_document(requests=[request(output_token_budget=0)]), None,
         CAPACITY_EXCEEDED),
        ("budget_overflow",
         base_document(requests=[request(output_token_budget=4095)]), None,
         CAPACITY_EXCEEDED),
        ("string_timeout", None,
         json.dumps(base_document()).replace(
             '"connect_timeout_ms": 1000', '"connect_timeout_ms": "1000"'),
         INVALID_ARGUMENT),
        ("negative_timeout", base_document(connect_timeout_ms=-5), None,
         SCHEMA_ERROR),
        ("u32_overflow", base_document(max_context_tokens=4294967296), None,
         CAPACITY_EXCEEDED),
        ("requests_not_array", base_document(requests={"a": 1}), None,
         SCHEMA_ERROR),
        ("request_not_object", base_document(requests=[42]), None,
         SCHEMA_ERROR),
        ("duplicate_stop_tokens", base_document(stop_token_ids=[7, 7]), None,
         DUPLICATE),
        ("too_many_stop_tokens",
         base_document(stop_token_ids=list(range(64))), None,
         CAPACITY_EXCEEDED),
        ("zero_request_id",
         base_document(requests=[request(request_id=0)]), None,
         CAPACITY_EXCEEDED),
        ("null_requests", None,
         json.dumps(base_document()).replace('[{', 'null, [{'), PARSE_ERROR),
        ("missing_file", None, "@@ABSENT@@", NOT_FOUND),
        ("directory_as_batch", None, "@@DIRECTORY@@", NOT_FOUND),
        ("valid_control", base_document(), None, None),
    ]


def main():
    if os.path.exists(BINARY) is False or             os.path.getmtime(node_source()) > os.path.getmtime(BINARY):
        subprocess.run(
            ["make", "-s", "build/sparkpipe_model_batch"],
            cwd=ROOT, check=True)
    failures = []
    with tempfile.TemporaryDirectory() as workdir:
        runtime_root = os.path.join(workdir, "empty_runtime")
        os.makedirs(runtime_root)
        for name, document, raw, expected in cases():
            path = materialize(workdir, name, document, raw)
            completed = subprocess.run(
                [BINARY, "--deployment", DEPLOYMENT,
                 "--runtime-root", runtime_root, "--batch", path],
                capture_output=True, text=True, timeout=30)
            match = re.search(
                r"sparkpipe_model_batch_status=(\d+)", completed.stderr)
            observed = int(match.group(1)) if match else None
            if completed.returncode < 0:
                failures.append(
                    f"{name}: killed by signal {-completed.returncode}")
            elif expected is None:
                if observed is None:
                    failures.append(f"{name}: no status line on stderr")
            elif completed.returncode != 1:
                failures.append(
                    f"{name}: exit {completed.returncode}, want 1")
            elif observed != expected:
                failures.append(
                    f"{name}: status {observed}, want {expected}")
    if failures:
        for failure in failures:
            print("FAIL", failure, file=sys.stderr)
        return 1
    print(f"ok: {len(cases())} malformed/edge batch files rejected cleanly")
    return 0


def node_source():
    return os.path.join(ROOT, "node/model_batch.c")


def materialize(workdir, name, document, raw):
    path = os.path.join(workdir, name + ".json")
    if raw == "@@ABSENT@@":
        return path
    if raw == "@@DIRECTORY@@":
        os.makedirs(path, exist_ok=True)
        return path
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(raw if raw is not None else json.dumps(document, indent=2))
    return path


if __name__ == "__main__":
    sys.exit(main())

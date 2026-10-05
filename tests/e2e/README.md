# End-to-end system tests

These tests drive the public HTTP API of a deployed model (and, for the
fleet tier, the rotation tooling) the way a user does. They exist because
unit tests prove components, not readiness: a green unit suite says
nothing about whether JIT KV, prefix sharing, continuous batching, model
toggle, KV park/restore or speculation actually hold on a live fleet.

Each test states the system behaviour it proves, and passes only on
evidence: response bodies AND the engine's `request_measurements` lines
(`node/model_api.c` writes them to the API log; `cached_prompt_tokens`,
`stale_prefix_recomputes`, `finish_reason`, `deadline_expired` and
friends are the receipts). A test that cannot obtain its evidence fails;
nothing is inferred from the absence of an error.

## Running

Against a deployed model (Tier A only needs the API to be up):

    python3 tests/e2e/e2e.py --config tests/e2e/e2e_config.json \
        --model glmfull --tier A

Tiers:

- **A — serving contract** (single model, no fleet churn): health and
  identity, completion contract, greedy determinism, seeded sampling,
  streaming order, logprobs, JIT KV prefix sharing, continuous batching
  with staggered arrivals and odd sizes, large prefill under decode,
  cancellation and slot reuse, error surfaces, long-context pass-key.
- **B — fleet lifecycle** (drives `tools/fleet_rotation.py`; pauses the
  rotation timer and restores it after): model toggle under the 60 s
  promotion gate, KV park and resume with restored-prefix evidence and
  greedy-identical continuation, co-residency independence.
- **C — speculation**: greedy equivalence with speculation on/off and
  the payoff receipt (tok/s both ways). Performance is recorded, not
  asserted; greedy equality is asserted.

`--tier A,B` selects tiers, `--only name` runs one test, `--list`
prints the registry. Tier B/C tests report `SKIPPED` with the missing
configuration key when the config has no `rotation`/`speculation`
block; a skip is printed and counted, never silently green.

Exit code is the number of failed tests (capped at 125). Receipts are
written per test as JSON under `--receipt-dir`
(default `qualification/e2e/<utc-timestamp>/`): every request body,
response, matching measurement line and the wall-clock timings needed to
re-derive the verdict. A red test names the offending evidence in its
output line.

## Self-test (hermetic, CI)

`python3 tests/e2e/e2e.py --selftest` runs every Tier A test against an
in-process fake API that implements the documented contract, including a
prefix-cache emulation that feeds real `request_measurements` lines.
`tests/test_e2e_harness.py` (registered in `make test`) asserts the
self-test passes. This proves the harness; it is explicitly not a claim
about any fleet.

## Configuration

See `e2e_config.example.json`. Per model: `host`, `port`, `chat`
(true → `/v1/chat/completions`), `api_log` (local path or, with
`ssh_host`, a remote path), `ranks` (expected TP size), `long_context`
(max positions; gates the pass-key depth), and optional
`logprobs_supported`. Fleet blocks: `rotation` (host + tool path +
model slot names) and `speculation` (spec-on/spec-off endpoints).

## Prompt fixtures

Tests address the engine with `prompt_token_ids` arrays (the API accepts
them directly), generated deterministically from a fixed LCG — no
checkpoint or tokenizer dependency, identical across runs and models.

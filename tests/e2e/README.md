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

## Known-failure-class coverage

Each Tier A test beyond the core contract guards a failure class that
has shipped in public inference stacks (and in this fleet's own ledger):

- `context_boundaries` — off-by-one at the position limit: exact-fit
  must serve, one-over must name `context_length_exceeded` (the
  crash-or-garbage-at-exactly-N class).
- `edge_shapes` — degenerate shapes: `max_tokens=1`, single-token
  prompt, `prompt`+`prompt_token_ids` together rejected, empty prompt
  rejected (hang-or-garbage class).
- `stop_token_edges` — stop token as the first generated token, mid-run
  stop (tokens before the stop must be an exact prefix of the unstopped
  run), never-hit stop (finish `length`); the missed-boundary-check and
  stop-changes-history classes.
- `duplicate_concurrent` — the same prompt submitted four times
  concurrently: identical greedy outputs and four distinct request ids
  (prefix-cache insert races corrupting concurrent duplicates).
- `batch_invariance` — one greedy probe solo, under 8-way concurrent
  load, and after the load: byte-identical (batched-numerics changing
  outputs by occupancy).
- `cancel_storm` — 8 of 16 streams aborted after their first token;
  survivors must equal their references and the engine must stay
  healthy (cancellation tearing shared state).
- `error_recovery` — oversized, invalid-temperature, malformed-body and
  1 ms-deadline requests each followed by a healthy completion (the
  one-bad-request-wedges-the-engine class; this fleet's completion-4xx
  latch was exactly this).
- `churn_stability` — 120 sequential requests over a rotating pool:
  correct lengths, `live_requests` returns to baseline, evidence
  request ids never repeat (slow slot/id leaks that benchmarks miss).
- `utf8_stream_integrity` — multibyte-heavy text: the concatenation of
  stream pieces must equal the non-stream text and be valid UTF-8
  (chunk-boundary codepoint splits/drops). Config-gated on
  `text_supported`.
- `model_isolation` — the same prompt to two configured endpoints
  simultaneously; evidence adapter identities must match their
  endpoints (cross-model response mixing under load). Config-gated on
  `isolation_peer`.
- `evidence_integrity` — every measurement line: unique increasing
  request ids, 64-hex prompt digests, cached <= prompt, known
  finish_reason values (silent drop / malformed evidence class).

## Prompt fixtures

Tests address the engine with `prompt_token_ids` arrays (the API accepts
them directly), generated deterministically from a fixed LCG — no
checkpoint or tokenizer dependency, identical across runs and models.

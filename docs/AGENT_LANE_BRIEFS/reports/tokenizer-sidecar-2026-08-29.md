# tokenizer-sidecar — Phase 4: text-in/text-out at the API edge

Lane: `lane/tokenizer-sidecar` (worktree /tmp/lane-toksidecar), one commit
`3bd257e` on origin/main `86431e3`. Host-only: no GPU, no reservations, no
fleet. Requested integration; main was never merged down.

## What landed

### The front door (node/model_api.c)

`POST /v1/completions` now accepts `{"prompt": "text"}` when the deployment
loaded a tokenizer sidecar; the response gains additive `"text"` beside the
unchanged `"tokens"`/`"status"`, so the event-stream shape is backward
compatible for token-id clients. The bounded edge is exactly text→ids before
the engine and ids→text after the events — nothing textual enters the engine
loop. Stop awareness: per-request `stop_token_ids` keep working unchanged
(the engine still sees ids), and the decoded text cuts at the first engine-EOS
or per-request stop id, so the stops survive the text round trip.

- No sidecar + text prompt → loud `400` with
  `"code":"tokenizer_unavailable"` naming the missing deployment config
  entry. Never a silent tokenization. `prompt_token_ids` remains accepted on
  every deployment; both prompt forms together → `400 ambiguous_prompt`.
- `/health` reports `"tokenizer":true|false`.
- A deployment that NAMES a tokenizer asset which cannot load refuses to
  start (`exit 1`, stderr names the asset): a promised text surface never
  silently degrades to token-id-only.

### The sidecar (text/tokenizer_sidecar.c + spark_tokenizer_sidecar.h)

Model-neutral shared infrastructure (dry-law clean): a deployment references
a tokenizer ASSET and the sidecar loads it, detects the format by content
(`{` → HuggingFace tokenizer.json, compiled-file magic → compiled, else
tiktoken ranks), and exposes `SparkTokenizerSidecarEncodeText` /
`SparkTokenizerSidecarDecodeText` with a per-token byte bound computed at
load for buffer sizing. Encode/decode take caller-owned workspaces, so
concurrent connection threads share the loaded tokenizer read-only.

### The deployment reference (additive, schema-compatible)

`model_resident.json` gains the one OPTIONAL root member
`"tokenizer": {"path": "tokenizer/tokenizer.json"}` — a tokenizer directory
shipped beside the pack, resolved against the serving host's runtime root
like every pack asset. Old 7-member deployment files parse byte-identically
(the root member check moved from exact-count to known-set+required-present;
duplicates and unknowns still rejected). `SPARK_MODEL_RESIDENT_DEPLOYMENT_SCHEMA_VERSION`
stays 2.

### Tokenizer engine upgrades (text/tokenizer.c) — what the real families needed

- The **digit-runs Split variant** (`\p{N}{1,3}`) of the GPT-4o-style
  pretokenization pattern now loads beside the single-digit variant; the
  scanner emits digit pieces of up to three. The GLM family asset carries the
  {1,3} form — without this variant the encode diverges from tiktoken.
- **`ignore_merges`** (HF model flag): a pretokenized piece found wholly in
  the vocabulary encodes as that single token (glyph-encoded lookup before
  the merge loop).
- **Tiktoken ranks loader** (`SparkTokenizerLoadTiktokenRanks`): the
  Kimi-K3 format (`base64(piece) rank` lines, 163,584 entries). That format
  ships no merges list — merge priority IS the vocabulary id of the
  concatenated piece, resolved by text lookup; the exact tiktoken rule,
  pinned by a fixture where a reachable-vocab piece must NOT be invented.

### Proofs (both registered in offline-gates)

- `tests/test_tokenizer_sidecar.c`: format detection, round trips (empty,
  whitespace, digits, contractions, CJK/emoji/combining marks), stop
  boundaries, compiled-format round trip, ranks-merge semantics — and THE
  GROUND TRUTH: the 92 ds4_eval cases joined case-id to case-id with their
  rendered prompts; `encode(text)==prompt_token_ids` and
  `decode(ids)==text` for **92/92** through the committed real GLM asset.
  Skips with a notice if an asset is absent.
- `tests/test_model_api_text.c`: the REAL `build/sparkpipe_model_api` over
  HTTP on the host resident stack (3× residentd, test adapter's deterministic
  4200+ ids): the 400 contract, token-id compatibility (historical shape
  byte-preserved), `text == decode(tokens)` identity, stops bounding the
  stream, `/health` both ways, and fatal startup on a missing asset.

### Ground-truth assets (qualification/ds4_eval/tokenizer/)

The two real tokenizer assets, committed beside the fixtures they validate
(excluded from package + ratchet): `glm-5.3-flash-tokenizer.json`
(zai-org/GLM-5.3-Flash @ `04c4e9e`, 20,217,442 bytes,
sha256 `19e7736...a82d`) and `kimi-k3-tiktoken.model` (moonshotai/Kimi-K3 @
`a590ce0`, 2,795,286 bytes, sha256 `b6c497...5103`). Provenance + hashes in
the README there.

### Three REAL concurrency defects fixed in model_api (pre-existing, first-ever gate coverage)

The e2e test is the first gate that has ever driven model_api; it found:

1. The disconnect-aware wait held `req->mutex` across the loop AND relocked
   it per iteration — every request slower than the 250ms poll froze its
   connection thread forever holding the mutex (sampled: three threads
   parked at the iteration lock). Fixed: no lock across the loop; the
   per-iteration lock covers only predicate+timedwait.
2. `api_event`'s per-request-stop branch called `SparkModelBatchEngineCancel`
   while holding `queue_mutex`; the engine fires completion events
   synchronously from Cancel and the callback re-acquires that mutex.
3. The disconnect branch called Cancel while holding `req->mutex`; same
   callback takes `r->mutex` under `queue_mutex` — the worker/connection
   ABBA (sample receipt: worker in api_event holding queue_mutex waiting
   r->mutex, connection holding r->mutex waiting queue_mutex).

Both Cancel sites now run lock-free under the existing `inflight` deferral
contract. Token-id behavior on the stop path is unchanged (the engine still
stops; the API still emits the token that matched).

## Gates, ratchet, artifacts

- `make offline-gates`: **PASS** fresh (`rm -rf build` first), one run, end
  to end — 178 test binaries, nvcc artifacts SKIP per contract. Log:
  /tmp/toksidecar_offline_gates.log. e2e additionally run 3× standalone:
  PASS 3/3.
- Dry-law: PASS (shared infra model-neutral).
- Ratchet: 227633 → **229008 EXACT** (+1375), in-commit justification in
  tests/test_code_size.py (sidecar, engine upgrades, deployment member, API
  edge incl. the concurrency repairs; tests and qualification assets excluded
  by construction).
- SHA256SUMS + PACKAGE_MANIFEST.json regenerated with the tools; verify:
  "package manifest, payload, and checksums match".

## Honesty notes

- The ground-truth pairing relies on the kimi-k3 archive's `rendered_prompt`
  being the exact text behind the glm5.3-flash fixture's ids. The join is
  92/92 on case id, the test re-asserts the id equality per case, and both
  directions of the round trip pass — but the ids' ultimate provenance is
  the fixture generator on spark1 (real tiktoken), not reproducible on this
  host; no tiktoken/tokens library exists offline here. The committed asset
  + fixture join is now itself the reproducible proof.
- `ignore_merges` did not change any of the 92 cases (plain rank merge loop
  already agreed with tiktoken on that vocab); it is implemented for HF
  semantic correctness and pinned by the synthetic fixture, not by the
  ground truth.
- The compiled tokenizer file format does not carry the new
  `ignore_merges`/`rank_ordered_merges` flags (v2 layout untouched; both
  default false on compiled load). Text assets (HF json / tiktoken ranks)
  are the sidecar formats of record; a compiled-asset deployment would lose
  those two behaviors. Follow-up if compiled assets ever need them.
- Kimi-K3's full pretokenization regex (Han runs, upper/lower case-split
  branches) is NOT implemented; ranks assets encode through the shared
  splitter variants. The committed K3 proof is round-trip identity +
  byte-id pins, not against tiktoken-generated ids (none exist in-repo for
  that family). GLM-family ground truth is the binding 92-case proof.
- The manifest regen captured the concurrent drift in
  docs/AGENT_LANE_BRIEFS/reports/coordinator-log.md and
  runs/reservations.json (live files that moved after main's last regen);
  the merge-window regen should be re-run at integration as usual.
- Push needed `http.postBuffer 524288000` (the 23MB of assets exceeded the
  default; first push died HTTP 400 mid-pack, retried once per rule with the
  buffer raised).

## Integration request

Please review and merge `lane/tokenizer-sidecar` (pushed to origin, one
commit `3bd257e` on `86431e3`). Never merged main here.

Deployment rollout note for the merge coordinator: turning text serving on
for a fleet deployment is adding
`"tokenizer": {"path": "tokenizer/tokenizer.json"}` to its
`model_resident.json` and dropping the asset under the runtime root; without
that member nothing changes for existing deployments. LiteLLM's native chat
path (the named Phase-4 dependency) can now be pointed at a text-serving
deployment.

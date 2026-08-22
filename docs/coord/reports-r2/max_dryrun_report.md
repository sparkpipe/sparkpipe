# Qwen 3.8 Max — bring-up runbook local dry-run (round 2)

Task: dry-run the round-1 bring-up runbook (`docs/coord/reports-r1/max_report.md` §2)
locally — pack synthesis command chain, deployment json, smoke-test steps. Host:
this macOS workstation (no CUDA, no nvcc, no GPU), so every step that needs the
CUDA-linked module was replaced by an independent static/binary validation and is
labeled accordingly. Everything labeled **measured** below actually ran here.

Dry-run workspace: `.agents/coord/max_dryrun/` (synthesize binary, independent
validator, stub checkpoint). Synthetic packs were deleted after validation
(226 GB scratch); sizes and results are recorded here.

---

## 1. Tier 1 — pack synthesis command chain

**Measured PASS (with two defects).** Built `tools/qwen38_pack_synthesize.c` with
plain `cc` (no repo headers outside the module needed) and ran the runbook's
synthesis chain with seed 7:

| slice | file | bytes | tensors | independent validation |
|---|---|---|---|---|
| 0+1 (GDN) | q38_synth_l0_1.qwen38sp | 30,830,447,104 | 20 | PASS |
| 1+2 (GDN+GDN) | q38_synth_l1_2.qwen38sp | 53,523,944,192 | 38 | PASS |
| 2+3 (GDN+ATTN) | q38_synth_l2_3.qwen38sp | 53,486,031,360 | 35 | PASS |
| 90+2 (GDN+ATTN tail) | q38_synth_l90_2.qwen38sp | 88,615,541,504 | 58 | PASS |

The validator (`.agents/coord/max_dryrun/validate_qwen38_pack.py`, constants
transcribed from `spark_qwen38_model.h` / `spark_qwen38_stagepack_format.h`,
not imported) checks: header field-by-field vs pinned geometry, tensor count vs
the format's expected-count rule, per-entry shape/format/payload/scale
arithmetic, 256 B alignment with no gaps/overlap/out-of-bounds regions, full
slice coverage including the MTP/embedding/head tail rules, and a byte-exact
spot check of one f32 tensor against an independent xorshift64* fill
reimplementation. All four packs pass everything.

### D1 (measured): default full-stack synthesis fails, exit 4

`qwen38_pack_synthesize --dry-run` with no slice args (the natural "synthesize
the whole stage" reading) dies with `directory build failed`, exit 4. Root
cause: the full 92-layer stack needs **1702** directory entries (92×10 + 69 GDN
layers ×9 + 23 attn layers ×6 + embedding + 22 head/MTP tail) against the tool's
hard-coded `SPARK_QWEN38_SYNTHESIZE_MAX_TENSORS 1024`
(`tools/qwen38_pack_synthesize.c:28`) — the append fails and surfaces as a
generic directory error. Not on the runbook's critical path (only 1–2-layer
slices fit a GB10 anyway, §R1), but the failure mode is confusing and the fix is
one constant (≥2048) or an explicit early refusal.

### D2 (measured): runbook's Tier-1 execute defaults claim is wrong

Runbook says "defaults already match this slice: … LAYER_COUNT=2" for the
2-layer synthetic smoke. Actually `tests/test_qwen38_execute.c:39` defaults
`TEST_QWEN38_LAYER_COUNT` to **"1"**. The l1_2 smoke must export
`TEST_QWEN38_LAYER_COUNT=2` explicitly or the module sees a slice mismatch.
One-line runbook correction.

Tool hygiene (re-confirmed from source): docstring says "Qwen 3.6 27B", default
`--layer-count` is 64 (not 92), and `--bf16`/`--dry-run` are undocumented.

Not runnable locally (needs nvcc/sm_121a + CUDA runtime): `make archive`
(target exists via `modules/resident_decode_stage_rules.mk`),
`test_qwen38_pack_load`, `test_qwen38_execute`. The validator above is the
declared stand-in; the real Tier-1/2 gates still need the target box.

## 2. Tier 2 — real-FP8 packer CLI

**Static PASS.** `tools/qwen38_stagepack.py --help` shows exactly the runbook's
flags (`--checkpoint --output --first-layer --layer-count --receipt`). Requires
the checkpoint + CUDA module at run time; not executable here. The stale-BF16
docstring finding from round 1 stands (usage text still says "BF16 safetensors"
while the code packs vendor FP8).

## 3. Tier 3 — TP env contract

Re-confirmed statically (round-1 session measured the greps): the module reads
exactly the runbook's 8 `SPARK_QWEN38_STAGE_TP_*` variables;
`spark_qwen38_serving_adapter.c` reads none of them (§1.2 item 3). The transport
backend path in the runbook,
`build/libhidden_transport_spark_host_rdma_verbs.so`, is a real root-Makefile
target (line 80). Ports 66620/66621 sit inside the registered qwen38max block
(`fleet_registry.json`: control 22480, collective 66620, transport-control
63700, status placeholder). The CAPACITY_EXCEEDED collective blocker cannot be
probed without a GPU; it stands as the Tier-3 gate.

## 4. Tier 4 — deployment json chain

### D3 (measured): the qwen38 TP4xPP4 spec does not generate

`python3 tools/generate_model_resident_deployment.py --specification
examples/deployments/qwen38_fp8_tp4_pp4_host_rdma.spec.json …` **fails**:
`topology members differ: missing=['transport_hosts']` (generator enforces the
full key set at `tools/generate_model_resident_deployment.py:36-38,239`). This
is repo-wide, not qwen38-specific: **10 of 13 committed specs fail** the same
check (only `dsv4_flash_pp13`, `dsv4_pro_tp4_pp4`, `qwen36_tp4` pass). The
generator also carries dead handling for an absent `transport_hosts`
(lines 257-259 default it to rank_hosts — unreachable after the strict check).
Fix options: add `"transport_hosts"` (= rank_hosts) to the spec(s), or exempt
that key from `exact_object`. Verified: a temp copy of the qwen38 spec with
`transport_hosts = rank_hosts` generates cleanly end-to-end.

### D4 (measured, new): no build seam produces any qwen38 serving adapter .so

Round 1 flagged a *name* mismatch; the dry-run shows it is worse: the root
Makefile defines serving-adapter shared-library targets for dsv4 (seven
variants), qwen36 and k3 — **zero qwen38 targets**. Nothing compiles
`modules/qwen38_resident_decode_stage/source/spark_qwen38_serving_adapter.c`.
Both the spec and the generated json reference
`lib/libqwen38_tp4_pp4_serving_adapter.so`; no `make` invocation can produce it.
A target mirroring the k3/qwen36 pattern (~6 lines) is missing.

### D5 (measured): release template and generated deployment disagree on the adapter file

The generated `model_resident.json` (adapter path
`lib/libqwen38_tp4_pp4_serving_adapter.so`) is inconsistent with the committed
release template `examples/release/qwen38_tp4_pp4_b1_template/sparkpipe.json`,
whose file list ships the qwen36-style `lib/model_serving_adapter.so`. A release
tree laid out from the template would not contain the file the deployment json
loads. Same family as round-1 correction item 4, now pinned to exact strings.

Generated-json sanity (from the patched-spec run): schema v2, 16 nodes,
control 22480, transport host-rdma, `runtime_limits` = max_inflight 4 /
max_active_sequences **1024** / resident_sequence_capacity **4096** /
kv_logical_page_capacity 1,048,576 — i.e. the round-1 R4 KV-sizing explosion and
the 409-row module ceiling (vs advertised batch) are concrete in the very
artifact Tier 4 would boot. b1 intent remains unexpressed.

Stage json (`qwen38_fp8_tp4_pp4_stage.json`): internally consistent —
`model_revision` matches the Makefile pin `d2dc3565…`, ctx 262144, tp_degree 4,
collectives 66620-66635, rails 10.10.200.x/10.10.100.x, cuda_graph_count
[1,1,1,1]. Round-1 correction item 3 stands unverified: `peer_hosts` are all
`sparkN-mgmt`, which collides with the HARDWARE_TOPOLOGY "inference never on
mgmt" rule and the host-rdma interface binding.

## 5. Scripts and registry

`qwen38_tp4_build.sh` / `qwen38_tp4_deploy.sh` drive **qwen36** artifacts end to
end (module dir, firmware description, `.qwen36sp` packs, `SPARK_QWEN36_*` env,
rank jsons) under qwen38 filenames — the wrong-artifact deploy hazard (round-1
R8) is confirmed by direct read; they are not usable as-is for this model.
Registry slot `qwen38max` matches the runbook's port block exactly.

---

## Bottom line

The runbook's Tier 1 synthesis chain is sound (all slices synthesize and pass
independent byte-level validation; the only defects are the 1024-entry cap and a
wrong defaults claim). Tier 2's CLI is correct as documented. Tier 4 **cannot
boot as committed**: the spec fails generation (D3, one-line fix, verified),
nothing builds the adapter library (D4), and the release template disagrees with
the generated deployment on the adapter filename (D5). Recommended order before
any hardware window: land D3 (+ the generator dead-code cleanup), add the
qwen38 adapter make target (D4), reconcile D5 naming, fix the D1/D2 nits, then
re-run this dry-run chain green end-to-end.

Artifacts kept: `.agents/coord/max_dryrun/{qwen38_pack_synthesize,
validate_qwen38_pack.py, stub_ckpt/}`. Synthetic packs deleted post-validation
(226 GB); regenerate with the commands in §1 (seed 7).

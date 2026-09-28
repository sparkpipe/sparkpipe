# Driver handoff, 2026-09-28

This handoff comes from the driver session that ran iterations i19-i33.
That session wrote it against main `338c202` and revised it at `b69eb69`.
Facts the lead dev verified on 2026-09-28 and the operator's answers from the
same day are folded in.

It is a dated snapshot. The current authorities are:

- [`../PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md) for measurements;
- [`../TECHDEBT.md`](../TECHDEBT.md) for open work;
- [`ROADMAP.md`](ROADMAP.md) for milestone order;
- [`GOALS.md`](GOALS.md) for operator directives.

Claims this handoff carries without re-checking are marked "(handoff)".

## Merge state

Every iteration is on `origin/main` (git, 2026-09-28):

| Iterations | PR | Merge commit |
| --- | --- | --- |
| i19 path-independent numerics | #1240 | `0c90e19` |
| i20 position-limit fix, reject diagnostics | #1243 | `56242e2` |
| i21 every decode module builds for sm_121a and links | #1244 | `2e12a86` |
| i22 glm5_next TP config through the shared loader | #1245 | `13683bb` |
| i23 58 more tests in CI; glm5_next DFlash2 tap retired | #1246 | `c06291f` |
| i24 ten drivers on the weightd mesh collective | #1247 | `1d62046` |
| i25 common FP32 combines for every mesh TP driver | #1248 | `36a7883` |
| i26 TP opens become two family templates | #1249 | `a790ce9` |
| i27 identical validator, serving and module code into templates | #1250 | `05a0b25` |
| i28 one JIT KV tier for three families; qwen4_flash prefills | #1251 | `53d33dd` |
| i29 KV frame combines, i30 defaults and adapters | #1254 | `ed03719` |
| i31 adapter rule, i32 build defaults and KV LRU, i33 GLM twin folds | #1257 | `aa818fd` |

The same day also merged the chunked single-sequence rounds (#1255), the
COMPSEC-17 harness fix (#1256), model-neutral mesh kernel names (#1259) and
host tests on glibc 2.43/CUDA hosts (#1258, `bbf5432`).

The fleet still runs engines built from `dd3526b`, the head of the #1243
branch. It holds i19 and i20 and none of i21-i33. None of i21-i33 has
hardware evidence yet.

## Milestone status

| Milestone | Status on 2026-09-28 |
| --- | --- |
| M1 GLM 5.3 Flash at 80% of roofline | See the notes after this table. |
| M2 batch scale | Not run. Decode graphs go to 64 rows (i7). The context-parallel DSA indexer is opt-in (i8). Every sequence reserves full-context KV. Batched serving is not batch-invariant. (handoff) |
| M3 serving completeness | Seeded temperature sampling exists for glm5_next only (i12b). API streaming, priorities and deadlines exist (i12), but the engine never receives the deadline. MTP drafts skip decode chains. There are no logprobs. (handoff) |
| M4 one collective platform | Ten drivers are on the weightd mesh (i24) with common FP32 combines (i25). Only k3 still builds the old hidden transport. Every hardware-wait round crosses the weightd CPU relay twice (TECHDEBT). One rail is wired. (handoff) |
| M5 placement | A TP4 x PP4 generator exists. The glm5_next adapter hard-codes TP16 (`spark_glm5_next_serving_adapter.c:37`). The PP route can wedge after a peer failure. (handoff) |
| M6 residency, multi-model | Two lanes are deployed. Eight lanes fail at a 4 GB `cudaHostRegister` inside the engine but not standalone. There is no catalog, promotion or pooled store. (handoff) GLM graphs need every expert pinned (`78c2c21`), and relocatable expert graphs are not built. |
| M7 models, code health | 13 modules build and link (i21), and twins are folded (i26-i33). About 17,400 of about 41,000 module source lines recur elsewhere. There are 33 packers. (handoff) |
| M8 beyond CUDA | The device layer is `SparkMemoryBuffer` only. The Mac Studios are on order; the handoff expected them in October 2026. |
| M9 provider network | Only the LiteLLM door and the static site and playground exist. It needs batch-invariant numerics, logprobs and promotion. (handoff) |

M1 notes:

- B1 measured 36 tok/s (128 tokens in 3.53 s). The step inside 8-step graph
  chains is 23-24.5 ms/token. The deployment was `dd3526b` with pinned
  experts and hardware wait.
- B8 across 8 streams measured 114.8-129.9 tok/s (2026-09-26/27).
- Eight concurrent COMPSEC prompts, thinking on, 128 tokens each, measured
  89.7 tok/s aggregate (lead dev). That is a different fixture from the B8
  sweep receipts and cannot be compared with them.
- Exits: B8 at most 26 ms per step, B1 at most 10 ms per token.

In the committed 2026-09-27 rank-0 window, one 10 s interval ran 171 waves
with `run_ms` 8835 and `peer_wait_ms` 2987. That is about 52 ms per B8 wave:
17.5 ms of peer wait, about 34 ms of the rest of the run, and about 7 ms
outside `run_ms` (arithmetic). Against the 21 ms floor, the B8 exit needs
overlap (two micro-batches, chains) as well as smaller costs.

The M1 levers, in order:

1. Collective waits: GPU-initiated RDMA, the pair link for the first
   reduction level, and one predeclared collective program per slot.
2. Kernels: the context-parallel DSA indexer on by default, then
   context-parallel latent attention.
3. Host time between waves: resident chains and device-fed tokens.
4. Prefill off the decode path.

## Drivers

| Driver | Last hardware evidence | Next step |
| --- | --- | --- |
| glm5_next (GLM 5.3 Flash) | TP16: B1 36 tok/s, B8 125-130 tok/s; COMPSEC-17 14/17 (2026-09-28) | Requalify on main with i19-i33: greedy tokens unchanged, a B8 sweep and a B1 run. Description hash (i37). Relocatable expert graphs (ROADMAP M6). |
| glm52 (GLM 5.3 Full) | none on GLM 5.3 | Move the generators and packer from GLM 5.2 revision `b4734de4` to GLM 5.3. Five firmware descriptions already name `935644c0`, and the mismatch risks `TARGET_MISMATCH`. Apply the committed prefix restore in its `ValidateSequenceContinuity`. |
| dsv4 (DSV4 Flash; DSV4 Pro 0813) | TP4 B1 about 40 tok/s at 128 output tokens (2026-08-28) | Both registry entries were deprecated and their packs deleted on 2026-09-23 (`83519ad`). The combines changed in i25, so token hash `211462f2…` may legitimately change; requalify against the reference with a precision receipt. Add `ADAPTER_SOURCE` (i36). DSV4 Pro 0813 is the last driver in the order. |
| dsv41_flash (DSV4.1 Flash) | none; no serving adapter | Add the adapter (i36). It outranks DSV4 Pro 0813. |
| qwen38_27b | TP1 B1 24.5 tok/s with DFlash2, 7.7-8.45 without (2026-08-28) | Add `ADAPTER_SOURCE` (i36), then requalify. Internal use only (license). |
| qwen38_max | TP4 x PP4 B1 1.29 tok/s (anchors, 2026-08) | Firmware JSON for release (i36). Internal use only. |
| qwen4_flash (Qwen 3.8 Flash) | none; packs held at the operator's pause since 2026-08-29 | Prefill was refused from 2026-09-19 until i28 fixed it. Firmware JSON (i36). Resuming is an open question. Internal use only. |
| k3 (Kimi K3) | single stage B1 18.0 tok/s | Move onto the weightd mesh; add a module Makefile (i36). K3 takes DSV4 Pro 0813's old slot in the order. |
| minimax | none | A first hardware run. |
| gemma4 | none | Its `Makefile.moe` has no adapter rule; the root Makefile builds the 26B adapter. Add `ADAPTER_SOURCE` (i36). |
| laguna | none; the layer-7 runner fails a bf16-experts static assert (handoff) | Fix the runner; firmware JSON and description hash (i36, i37). |
| ling (Ling 3.0, finance fine-tune) | none | Apply the committed prefix restore in its inline continuity loop (i38). Firmware JSON and description hash (i36, i37). Ling 2.x is dropped. |
| muse_glimmer | none | Its adapter comes from the root Makefile. Add `ADAPTER_SOURCE` (i36). |
| hy4 | none; no serving adapter | Add the adapter (i36). |
| mimo26 (MiMo 2.6) | none; the module is a 137-line stage-pack header | Build the driver against contracts `mimo26_flash` and `mimo26_pro`. |

The proposed requalification order was: glm5_next regression, dsv4 TP4 B1,
qwen38_27b, the first qwen4_flash wave, glm52 as GLM 5.3 Full, then the rest.
The operator's model direction in GOALS.md overrides it where the two
conflict.

## No-hardware iterations i34-i41

These are proposed and not yet started.

- **i34 twin and duplication ratchet.** Add `tests/test_no_new_twins.py`
  with a twin scanner and before/after object builds. `tools/host_codegen_diff.py`
  and `tools/dup_report.py` already exist.
- **i35 the last `#ifndef` default.** `DSPARK_SPEC_STEP` in the dsv4 builds.
- **i36 every driver releasable by `tools/module_build_release.sh`:**
  - firmware JSON for laguna, ling, qwen38_max and qwen4_flash;
  - `ADAPTER_SOURCE` for dsv4, muse_glimmer, qwen38_27b and the gemma4 26B;
  - adapters for hy4 and dsv41_flash;
  - a k3 module Makefile.
- **i37 description hashes** for glm5_next, laguna and ling through
  `serving_adapter_template.c`.
- **i38 first behaviour-drift folds.** The two `ModuleValidateFrameContext`
  definitions differ only in the prefill refusal, and the decode-only header
  also holds `ModuleConsumeHiddenInput`. The two `ValidateSequenceContinuity`
  versions differ in prefix restore:
  - the loader shared by glm5_next and laguna applies a committed restore and
    refuses a mismatched one;
  - the inline loop in glm52 and ling does neither, which is a correctness
    bug for those two drivers.
- **i39** the validator digest covers included templates.
- **i40** register the nine Python tests that sit outside `make test`: the
  stream synthetic packs in the suite, and the five ssh fleet tests as a
  separate target.
- **i41** the universal packer's first emitter.

## Operator questions

| Question | Answer |
| --- | --- |
| Merge state | Answered. i19-i33 are all on main (table above). |
| Fleet runs: a B8 sweep and B1 on main with i19-i33; the station wedge on #1229 | Open. The only 2026-09-28 fleet numbers are from `dd3526b` (pre-i21). |
| glm52 checkpoint revision | Answered. GLM 5.2 weights are deprecated and not considered, so glm52 moves to GLM 5.3. |
| Requalification and driver order | Partly answered. DSV4.1 Flash outranks DSV4 Pro 0813, and DSV4 Pro 0813 goes last. K3 takes DSV4 Pro 0813's old slot. DSV4.1 Pro is to be supported when released. |
| Resume Qwen 3.8 Flash | Open. Qwen models may serve internally but not on the external API. |
| Licensing | Answered. Revenue will not come near $20M, so only Qwen is an issue. Qwen is kept off the external API service; internal use is fine. |
| MiMo version | Answered. MiMo 2.6. |
| Ling version | Answered. Ling 3.0 and its finance fine-tune; drop Ling 2.x. |
| How fast can K3 get? | Open. SparkPipe measured 18.0 tok/s single stage. gb10-vllm announced 23.59 t/s without speculation on 16 Sparks. |
| A video generation model | Open. The operator asked why a video generation model would be wrong and whether the Sparks cannot run one. The notes do not say which model was set aside. |
| Topology | Answered. No fixed topology: hill-climb TP16, PP16 and TP4 x PP4 on tok/s and latency (GOALS.md). |
| Speculation direction | Answered as direction. A multi-drafter tree with conditional truncation and extension, aiming for TensorFold-class gains at B8 and B64 (GOALS.md). The truncation condition is open. |
| Mac Studio arrival | Open. They are on order. |
| Provider network: tailnet control server, payout rails, identity checks | Open. |

## Working rules

- Claude is lead dev. The operator merges PRs by hand, and PRs may stack.
  GitHub commands go through `tools/sparkpipe_github_pat.sh` (AGENTS.md).
- Code style:
  - no comments in code;
  - compact Allman, with no blank lines inside functions and locals at the
    top;
  - one-line calls and signatures, and functions of at most 50 lines;
  - no `malloc`, no N² loops, no `#ifndef` defaults, no silent fallbacks;
  - shared code has model-neutral names (`tests/test_dry_law.py`).
- A template move needs identical objects. Land the behaviour commit first,
  then the move, and check it with `tools/host_codegen_diff.py`.
- Every commit passes `tests/test_dry_law.py`, `tests/test_memory_contracts.py`,
  `tests/test_no_build_defaults.py` and `tools/source_package_gate.sh`.
  `PACKAGE_MANIFEST.json` and `SHA256SUMS` are generated and never committed
  (AGENTS.md).
- Run the host suite after deleting `build/test_*`, `build/test_modules` and
  `build/lib*_serving_adapter.so`. The sm_121a gate is
  `tools/cuda13_sm121a_compile_gate.sh`.
- Where to build:
  - The workstation's git 2.10 and make 3.81 cannot build the tree.
  - Host tests run on the rtx5090.
  - aarch64 sm_121a builds run on sparkf (`~/g5n-rd-build`).
  - Every release also needs an x86 build of the API for the hub, because the
    release root's aarch64 API cannot run there.

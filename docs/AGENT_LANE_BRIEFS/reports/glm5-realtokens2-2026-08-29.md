# glm5-realtokens2 — the HC-wide twin fix deployed and verified; the all-zero token bug is now localized DEEPER (layer-34 attention), one layer below the fixed reduce

Lane: glm5_next (attempt 2). Branch `lane/glm5-realtokens2` (worktree
/tmp/lane-g5rt2). Every claim below = command + raw output.

## How this serves the scoreboard

GLM 5.3 Flash is the only TP16 family whose first serving cell is
blocked on value-correctness (status-0 all-zero tokens). This session
verified the coordinator's HC-width root-cause fix end to end on the
live fleet — the cross-rank reduce is now proven correct with device
receipts — and drove the remaining zero-token defect from "somewhere in
the stack" to a single layer's attention phase. M5 (exact-32K B1) and
COMPSEC-17 stay blocked on real tokens; the fixtures are pre-tokenized
(main 93a3a0b) so the gate fires the moment tokens land.

## R1 — the committed fix (8043d83) could not build and would not have worked

Source on spark0 verified synced to main first (blob hashes of module.c,
serving_adapter.c, cuda.cu, Makefile all equal `git show main:<path> |
git hash-object --stdin`). Then the mandated host syntax gate:

```
$ cc -fsyntax-only [house flags, tests/cuda_stub] .../spark_glm5_next_resident_decode_stage_module.c
spark_glm5_next_resident_decode_stage_module.c:1549:41: error: ‘SparkTpDeviceCollectiveConfig’ has no member named ‘memory_mode’
 1549 |                         configuration_hc.memory_mode = SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE;
spark_glm5_next_resident_decode_stage_module.c:1562:16: error: implicit declaration of function ‘SparkGlm5NextModuleReduceHiddenWide’
spark_glm5_next_resident_decode_stage_module.c:2205:9: error: this ‘if’ clause does not guard... [-Werror=misleading-indentation]
```

Semantic holes beyond the compile errors (all fixed in 523bcaa):

1. `memory_mode` does not exist on `SparkTpDeviceCollectiveConfig`
   (include/sparkpipe/spark_tp_device_collective.h) — the mode is derived
   INSIDE Create from backend capabilities. This fleet's host-verbs
   transport probes to `MEMORY_MODE_MAPPED_HOST`; the committed twin's
   device-only bindings with `transport = device pointer` could never
   satisfy it.
2. `configuration_hc` was never populated (credit_count, width, ports,
   identifier all zero) → `hc_total == 0` → the twin silently never
   created → EVERY hidden reduce would return INTERNAL_ERROR.
3. `local_hidden_dimension` was never scaled by HC — even a created twin
   would re-price one hidden-width per row (payload = rows ×
   collective->local_hidden_dimension, tp_device_collective.c:379), i.e.
   the original bug, unchanged.
4. No `combine_bf16_function` on the twin; shared collective_identifier
   and control_port_base with the narrow instance (route names embed the
   identifier; each step claims PORT_STRIDE×degree ports — sharing
   collides at open).

Fix commit 523bcaa (lane): twin fully mirrors the narrow instance's
proven pattern (probed memory mode, device arena + pinned mapped-host
transport aliases + BINDING_KNOWN_FLAGS in mapped mode, combine fns,
MAX_BINDING_COUNT guard, state-tracked buffers freed on Destroy), with
`local_hidden_dimension = HIDDEN × HC_MULT`, `collective_identifier+1`,
`control_port_base+512` (SPARK_GLM5_NEXT_TP_COLLECTIVE_HC_PORT_STRIDE).
Also adds `tests/test_glm5_next_module_host_syntax.py` (the gate that
would have caught 8043d83; mirrors the dsv4 one). Ratchet 216634 →
216722 justified in-commit.

## Build receipts (spark0:~/g5rt2-src, git checkout from bundle of main b415bba3)

```
$ python3 tests/test_glm5_next_module_host_syntax.py
PASS GLM5_NEXT resident module host syntax

$ make publish NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a EXPERT_CODEC=fp8 \
    MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 \
    CONTRACT_SHA256=a40e9ec5fbfb0c1a180162c9d82915c887e8549fbd779c9f5dacb780a1498db4 \
    STAGE_PACK_PATH=/home/spark0/glm53_packs/glm5_next_stage.tp16.rank0.g5nsp
glm5_next validator: PASS (0 failures)
module=spark.glm5_next.resident_decode_stage.bf16.expert_fp8.h4096.l45.kda34.e288.k8.v2
artifact=226b079cdac8e873871d652b43aa48629d099a575fff4a9d583875ffe77a8d27
validation=executed
```

(a40e9ec5... = sha256(examples/model_descriptions/
glm5_next_resident_decode_stage_fp8_firmware.json), verified; the same
value the deployed binaries embed.) Every probe rebuild re-PASSed
(validator full pass at each of c7547db, 02cd427, 5ef4ffa, 5b227c0,
16cc7d9). Driver compiles:

```
$ build/sparkpipe_model_compile --model examples/model_descriptions/glm5_next_resident_decode_stage_fp8_firmware.json \
    --stage glm5_next_resident_decode_stage --library build/module_library --output ... \
    --cc /usr/bin/cc --include include --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
    --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl \
    --cc-arg -pthread --cc-arg /home/spark0/g5rt2-src/build/libsparkpipe_model_common.a
driver=.../model_driver.so model_sha256=a40e9ec5... driver_sha256=4f3b8f33e70cbc15ccd7b6d47b3f3870c60bec45ab0702129621b2be8dcbd169
```

NOTE: `-lmodel_common` alone fails (`/usr/bin/ld: cannot find
-lmodel_common`); passing the archive by absolute path links. api +
residentd binaries rebuilt from the same source (`make
build/sparkpipe_model_api build/sparkpipe_model_residentd`) — api
carries the /v1/models + error-shape build (coordinator addition):
api sha fa0891dc..., residentd b5762b47..., adapter 88b58b0f....

## R2 — deploy receipts

First fan-out hit ETXTBSY on the running daemons' executables; deploy is
write-temp + atomic `mv` (rename works on busy executables). Verify:

```
$ for h in spark0..sparkf; do ssh $h sha256sum .../lib/model_driver.so ...; done | sort | uniq -c
     16 4f3b8f33e70cbc15ccd7b6d47b3f3870c60bec45ab0702129621b2be8dcbd169   (driver)
     16 88b58b0fcfd99311e0c23bd97a8c54c2936f6ad559df30f340af00d257b49c52   (adapter)
     16 fa0891dc7c8601819141dc53f0a31dc1a6debab89f25f68c1914231d0f287663   (api)
     16 b5762b4701f52c81c7be8a068dd8034884b0ab4a053ea3f75f15c6dd52c68172   (residentd)
```

Pre-deploy binaries backed up as *.pre-rt2 in each runtime root.

## R3 — wave receipts

- Stop is cwd-scoped TERM (only pids whose /proc/<pid>/cwd ==
  /home/<host>/sparkdata/glm5_next.tp16) — spark5's dry-template2 lane
  processes (cwd /tmp/dry2-*, /home/spark5/sparkpipe) untouched.
- First wave 0/16: launching via `ssh host "cd ... && ... nohup ... &"`
  without holding the session killed the child BEFORE exec (sshd HUP) —
  residentd.log didn't even exist on the ranks. WORKING launch pattern:
  `ssh host "cd $rr && rm -f residentd.log && setsid nohup env
  LD_LIBRARY_PATH=$rr/lib ./bin/sparkpipe_model_residentd ... > residentd.log
  2>&1 < /dev/null & sleep 2"` — setsid + a 2s session hold.
- A double-launch left a duplicate rank-4 residentd that IGNORED TERM
  (SigCgt has SIGTERM; found sleeping in poll_schedule_timeout, 4
  threads) — the exact TERM-immune class the README predicts. Per the
  no-kill protocol I captured diagnostics and did NOT escalate; the
  process self-resolved (exited) after ~25 min and spark4 needed no
  reboot. The whole wave then timed out once waiting for it
  (`hidden_spark_rdma_open_timeout route=tp-device...2.0.4 role=sender
  host=spark4 port=63772 waited_ms=180000` → LoadDriver rc=15 →
  route_failed) and was re-run clean.
- FINAL: wave ready 16/16 (`grep -c "model_residentd ready"
  residentd.log` = 1 on every rank), api on spark0:8433 healthy:
  `{"status":"ok","served":0}` + `model_api ready port=8433`.

## R4 — THE CURL: status 0, tokens still all zero — root cause driven one layer deeper, with device receipts

```
$ curl -s -X POST http://localhost:8433/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt_token_ids":[154819,11,1875,525],"max_tokens":16}'
{"object":"text_completion","tokens":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"status":0}
```

Instrumented driver (G5N-PROBE, env SPARK_GLM5_NEXT_PROBE=1, diag-only
commits c7547db..16cc7d9) — evidence ladder, all from the live fleet:

1. Token flow is correct:
   `G5N-PROBE batch token_ids rows 4: 154819 11 1875 525`
2. THE WIDE TWIN WORKS — post-embed-reduce hidden is NON-ZERO and
   IDENTICAL on both audited ranks (this is the brief's root-cause fix,
   verified on device):
   ```
   spark0 (rank 0):  G5N-PROBE post-embed-reduce hidden row0 first16 bf16sum 470600 first8 14874 47853 48089 15144 47849 15375 13881 15386
   sparkf (rank 15): G5N-PROBE post-embed-reduce hidden row0 first16 bf16sum 470600 first8 14874 47853 48089 15144 47849 15375 13881 15386
   ```
3. The head receives EXACTLY 0.0 for every candidate on every rank:
   `G5N-PROBE head score 0.000000 token 0 maxloc 80000000ffffffff rank 0/16`
   (0x80000000 = ordered +0.0; low word = token 0). With all-equal
   scores the pack kernel's documented tiebreak ("the token is inverted
   so equal scores keep the LOWEST token id",
   SparkGlm5NextHeadMaxlocPackKernel) deterministically emits token 0.
4. Per-layer hidden checksums (row 0, elements [0,1024)): healthy and
   growing through layer 34's attention entry, EXACT ZERO from layer
   35's:
   ```
   layer 33 hidden bf16sum 49176039   layer 34 hidden bf16sum 49671706
   layer 35 hidden bf16sum 0          ... layers 35-44 all 0
   ```
5. Splitting layer 34: hidden at MLP entry is ALL bf16 0x8000 (= -0.0;
   sum of 1024×32768 = 33554432 exactly):
   ```
   G5N-PROBE layer 34 hidden row0 bf16sum 49671706          (attention entry)
   G5N-PROBE layer 34 attn_out bf16sum 0                    (after narrow attn reduce)
   G5N-PROBE layer 34 mlp-entry hidden bf16sum 33554432     = 1024 × 0x8000
   G5N-PROBE layer 34 pre-wide-submit hidden bf16sum 0      (MLP kernel wrote through)
   G5N-PROBE layer 34 post-mlp (post-reduce) bf16sum 0
   ```
   → the layer-34 ATTENTION produced a zero sublayer output and the HC
   post placement wrote sign-flipped zeros (0 × negative placement
   coefficients) over the residual streams; the MLP then normalized
   zeros to zeros. The wide reduce is innocent (pre-submit == 0).
6. Attention partials are ZERO at layers 33-35 but NON-ZERO at 0-4
   (`layer 0 attn_out bf16sum 16195095` ... `layer 4 attn_out bf16sum
   32744652`) — the attention-output death starts somewhere in layers
   5..32 (unlocated) and layer 34 (KDA ordinal 26) is where the residual
   itself finally dies. Layer 34's pack tensors are ALL healthy — every
   one of the 25 kinds at layer 34 checksummed nonzero vs layer 33
   (router 524288/524288, experts ~1048573/1048576, KDA 26-36 all
   matching), so this is runtime/kernel/state, not weights.
7. Pack exonerated wholesale: EMBEDDING rows fully populated (token 11 =
   4096/4096 nonzero, verified on rank 0), FINAL_NORM 4096/4096,
   LM_HEAD 32768/32768.

## Remaining blocker (crisp handoff)

Attention sublayer output is zero by mid-stack (between layers 5 and 32,
unlocated) and kills the residual outright at layer 34 (KDA ordinal 26)
via -0.0 placement. Next probes, in order:
1. attn_out checksum for EVERY layer 5..32 (same one-line probe
   extension; finds the exact layer where partials die).
2. At that layer: KDA vs DSA class, then its state/conv-window pool
   indexing (kda_ordinal_by_local_layer / index_ordinal_by_local_layer)
   and the HC post kernel's placement sign path.
3. The hidden GROWTH is suspiciously linear (+~500K bf16sum per layer);
   worth checking the residual placement scale once the attention death
   is fixed.

M5 + COMPSEC-17 remain blocked behind this; fixtures are ready
(qualification/ds4_eval/quality-fixtures-glm5.3-flash.json).

## Fleet state (session end)

Standing 16/16 on the CLEAN corrected-twin driver (4f3b8f33, no probe
stalls) + new api (fa0891dc) on spark0:8433, healthy:

```
$ curl -s http://localhost:8433/health
{"status":"ok","served":0}
$ curl -s -X POST http://localhost:8433/v1/completions -H "Content-Type: application/json" \
    -d '{"prompt_token_ids":[154819,11,1875,525],"max_tokens":8}'
{"object":"text_completion","tokens":[0,0,0,0,0,0,0,0],"status":0}
```

(the zeros are the known layer-34 defect above, not a serving fault.)
The probe runs' own residentd.log files were rotated by the final clean
wave; every probe line cited above is captured verbatim in this report.

## INTEGRATION REQUEST

None new. The five G5N-PROBE commits are diag-only (env-gated, print-only)
and stay on the lane branch until the zero-token bug closes; the family
host-syntax gate test is mergeable as-is.

## Process findings worth keeping

- The mandated pre-nvcc syntax gate caught a main-branch commit that
  could not compile AND would have silently degraded the fleet (twin
  never created → all reduces INTERNAL_ERROR). Keep gating every fix
  commit that touches host sources.
- `ssh host "... &"` without a session hold loses the child before exec;
  the committed tools/glm5_next_wave.sh has the same latent bug (its
  stop/start worked here only because... its start branch does NOT hold
  the session). Fixed pattern documented above; the wave script wants
  the setsid+hold treatment before its next user.

# HANDOFF: glm53flash transport rewrite — next session pickup

## Where the code is

Repo: ~/lane-glm53 (lane branch lane/glm53-p0, tip b4ec228)

### What was rewritten

| File | Before | After | What it does now |
|------|--------|-------|------------------|
| ring/transport/rdma.cu | 5,831 | 275 | Thin weightd IPC proxy DSO |
| ring/transport/tp_device_collective.c | 3,716 | 437 | Shared-memory dataflow collective |
| node/weightd_mesh.c | 0 (new) | 500 | Universal all-to-all mesh (30 QPs/node) |
| Module | 3,340 | 3,165 | Credit bindings stripped |

Total: 9,547 → 1,212 lines

### What works
- Universal mesh: 16/16 nodes wired, 30 QPs each, once per boot
- Fleet deploy cycle: READY 16/16 in 2 SECONDS
- Module boots, reports ready, API connects
- Build: ~2s transport-only, ~70s module change

### THE ONE BLOCKER

Engine's MESH_WRITE IPC never reaches weightd daemon.

Evidence:
- MESH-SUBMIT printed in residentd log (Submit IS called)
- MESH-WRITE-FAIL never printed (MESH_WRITE returned OK)
- WD-MESH-WRITE count=0 in weightd log (daemon never sees it)
- Result: spin-wait hangs → 5s timeout → model status 4

### Three debug approaches

1. Print inside SparkWeightdClientMeshWrite before/after Exchange (runtime/spark_weightd.c ~line 2723)
2. Print at top of weightd server request dispatch (search KIND_MESH_WRITE ~line 1877)
3. Share lazy_pack's weightd client instead of separate connection

### Most likely root causes

1. sizeof(SparkWeightdClient) mismatch — engine declares char client[4096]
2. SparkWeightdClientExchange doesn't handle kind=21
3. Server Step drops unknown kinds before handler processes them

### Build notes (CRITICAL)

Module library cache MUST be cleared on every engine change:
  rm -rf build/module_library modules/glm5_next_resident_decode_stage/build

Full build command:
  ssh sparkf 'cd ~/sparkpipe-build && export PATH="/usr/local/cuda/bin:$PATH" && git fetch -q origin lane/glm53-p0 && git reset -q --hard origin/lane/glm53-p0 && rm -rf build/module_library modules/glm5_next_resident_decode_stage/build && tools/module_build_release.sh glm5_next_resident_decode_stage fp8 glm53flash.fp8.tp16 84c6a6aa9497188e15a635ba793b0f95a79b1033 model_contracts/glm53_flash_authoritative.json'

Deploy is automatic (1s agent loop, kill -9, restart).
Fleet check: ssh sparkf 'grep -l ready ~/current/*.json | wc -l'

### Mesh buffer layout

1MB buffer in weightd per node:
[0..bytes): scratch (sender writes vector + seq at bytes..bytes+8)
[bytes+8..]: per-peer receive slots at (peer_index+1)*(bytes+8)

Sender: MESH_WRITE(peer_rank, 0, (my_index+1)*(bytes+8), bytes+8, timeout)
Receiver: spin on *(uint64_t*)(buffer + (peer+1)*(bytes+8) + bytes) >= ordinal+1

### After IPC fix

1. Serving E2E test
2. Decode-only benchmark
3. Inference hill climb (baseline 0.65 tok/s, target 40 tok/s)
4. Replace CPU sum with GPU kernel (first optimization)

### Weightd restart (mesh binary changes)

  for h in spark0..sparkf; do ssh $h 'kill -9 $(pgrep -x sparkpipe_weightd); rm -f /tmp/weightd-mesh/.shipped'; done
  # agents restart weightd (15s mesh wiring)
  # then restart residentds:
  for h in spark0..sparkf; do ssh $h 'kill -9 $(pgrep -f bin/sparkpipe_model_residentd)'; done
  # agents restart residentds (~2s)

### Cleanup after IPC works

- Remove debug prints (MESH-SUBMIT, MESH-WRITE-FAIL, WD-MESH-WRITE)
- Delete rdma_control.c from DSO build (dead code)
- Delete tp_device_collective_nccl.c (dead code)
- Replace CPU bf16 sum with GPU kernel


## Hill-climb iteration 2 (09-11 ~03:10, lane 7d54c8c)

- Fixed sparkf weightd segfault-loop: the server step's mesh poll raced the mesh
  thread's construction (TryWire on NULL QPs); poll is now gated on mesh_active.
  Loaded nodes (sparkf, the build host) hit the window; that was both fleet
  wedges tonight.
- Engine shapes measured for 32-token B1 requests: sync+CPU-reduce 20.4s (kept),
  async-worker+GPU-combine 24.0s, inline+GPU-combine 33.6s (combine kernels and
  staging queue behind the module's busy stream). Worker/staging machinery
  deleted; engine = inline round, 2 broadcasts, CPU bf16 sum.
- Phase data stands: engine round 0.5-1.3ms; ~6ms/round is module-side wave
  orchestration. NEXT: nsys one request on a node, kernel-gap map of the module
  wave path.


## Hill-climb iteration 3 (09-11 ~04:15, lane 851858a — analysis iteration)

- gdb stack sampling during a request: the busy thread sits in map_import_lease
  (weightd lease IPC) + CUDA; strace -c: 151,700 ioctl calls in 20s (98% of
  syscall time, 22us each), 546 recvmsg. Per token: ~12.6K launch ioctls,
  ~160/layer at B1. The module has NO CUDA-graph machinery (grep: zero).
- Root cause chain: each synchronous round drains the pipeline (D2H event sync
  waits for all prior module kernels), so every subsequent kernel launch waits
  on an idle GPU — launch ioctls become waiting round-trips. The old 71ms/token
  (14 tok/s) could not pay 12.6K x 22us; the bubbles are INHERENT to CPU-mediated
  reduces at B1's serial chain, not fixable by engine reshaping (bake-off in
  iteration 2 proved: 20.4 / 24.0 / 33.6s all within the same regime).
- DEFINITIVE LEVER: GPU-resident mesh. Stage A: weightd allocates the mesh
  region as GPU memory (links cudart already), ibv_reg_mr on it, exports via
  CUDA IPC handle on the lazy-attach reply (SCM_RIGHTS carries the IPC fd);
  engine cudaIpcOpenMemHandle -> mesh slots become GPU pointers; local vector
  copies device->mesh-slot on-stream (no event sync, no CPU visibility needed
  for the payload); CPU stamps seq + broadcasts from weightd's own mapping
  (GB10 coherent); spin + CPU sum read GPU memory over C2C (512KB/round,
  tens of us). D2H/H2D and drained-stream bubbles disappear.
  Stage B: GPU polling kernel for seqs + module combine kernel on mesh slots,
  then CUDA graph capture eligibility returns.
- Also measured: 546 recvmsg/request = lease IPC is NOT the bottleneck at B1.


## Hill-climb iteration 4 (09-11 ~05:20, lane bfcf16b)

- Idle-control strace: ZERO ioctls in 10s idle -> the 151.7K ioctls/request are
  all request-driven; launch-submission tax CONFIRMED (12.6K/token, ~160/layer).
- Merged broadcast IPC landed: MESH_BROADCAST wire carries seq_value +
  seq_remote_offset; the daemon posts the 8-byte seq RDMA from a dedicated
  registered staging word alongside the payload fanout. One unix round trip per
  round instead of two (second was 100-950us). Engine local seq stamp removed
  (value rides the IPC). Functionally verified (tokens returned, 16/16).
- Warm perf number BLOCKED by serving-reliability defects that now gate all
  measurement: (1) after any residentd restart the API's engine_connect retries
  ~22s x 5 (model_batch_engine.c:1090 status=4) and must be killed to recover;
  (2) resident-slot exhaustion still wedges after ~16 requests. Fix these two
  BEFORE the next perf iteration or every measurement costs 5+ minutes of
  unwedging.
- Next levers unchanged: Stage A/B GPU-resident mesh + graphs for the launch
  tax; these two serving fixes are now first in the queue.


## Hill-climb iteration 5 (09-11 ~06:20, lane 6855757) — dev-cycle reliability

- Slot exhaustion FIXED: idle retained slots now evict on claim (binding +
  continuation lease reset when a new sequence takes over an idle slot; BUSY
  only if actively owned). Stress: 25/25 sequential requests served, zero
  restarts, fleet 16/16 throughout.
- API self-heals after residentd restarts (1s retry cadence to a 120s deadline;
  each failed connect blocks ~22s against a booting residentd — acceptable, no
  kill needed; just wait after a wave).
- Warm baseline re-established with merged-IPC broadcast live: 20.3s/32 tok;
  broadcast phase now 130-145us (was 160-950us). One 34s outlier right after
  the 25-request stress (retained-KV state / background) — re-measure before
  trusting any single reading.
- Dev cycle now: module commit -> ~2min wave (weightd waves ~90s, module-only
  ~30s) -> API self-heals -> unlimited requests. Next: Stage A GPU-resident
  mesh, then graphs (launch tax is the proven 10x).


## OPERATOR DIRECTIVE + STAGE A/B BUILD ORDER (09-11 ~06:50): "as fast as before, in the new architecture"

Target: 14 tok/s plain B1 (71ms/token). Current 1.58 (634ms/token). The gap
decomposition (all measured): launch ioctls ~277ms/token (12.6K x 22us,
serialized because each blocking round drains the pipeline and the module
cannot submit ahead), module CPU + engine ~remainder. The OLD system hit
71ms because its STREAM-ORDERED async collective kept the module's 4 inflight
slots pipelining: wave N+1's kernels were submitted while wave N's reduce flew.
Restoring that under the weightd-mesh architecture = GPU-resident mesh + all
device-side round mechanics. Function-level build order:

1. weightd_mesh.c: mesh region becomes GPU memory — cudaMalloc(2GB) in Init
   (link -lcudart already present), ibv_reg_mr on the pointer (GPU-Direct,
   the proven NIC-DMA->GPU path), cudaIpcGetMemHandle -> store the handle
   bytes. Keep the host doorbell region: cudaHostAlloc(mapped, 4KB) per node
   + its own ibv MR; doorbell layout: one uint64 per (band, rank) = seq
   published; ALSO export doorbell addr+rkey via the mesh RECORD so peers'
   weightds can RDMA seq words into it (records gain doorbell_rkey/addr;
   TryWire stamps them into qp_info).
2. lazy attach: reply gains the CUDA IPC handle (rides the same SCM_RIGHTS
   tier, 64 bytes) + doorbell host address (engine-local alias not needed:
   the engine only writes its OWN doorbell via the H2H staged memcpy below).
3. Engine round becomes FIVE stream-ordered ops, ZERO IPCs, no CPU sync:
   a. cudaMemcpyAsync(gpu_slot <- local_device, bytes, D2D, stream)
   b. cudaMemcpyAsync(own_host_doorbell <- pinned_seq_staging, 8, H2H, stream)
      (pinned staging holds ordinal+1, CPU-written before enqueue)
   c. module WAIT kernel: spins until every peer slot-end GPU word >= ordinal+1
   d. cudaMemcpyAsync(full_device <- own gpu_slot, D2D) + 15x module combine
      kernels on peer gpu slots (config.combine_bf16_function — device ptrs)
   e. module DONE kernel: writes host-mapped done word for this ordinal
   Completion watcher thread (CPU, plain host reads) fires the module
   completion callback when done advances (CUDA calls allowed there).
4. weightd: per-round IPC deleted — a doorbell poller (mesh thread, already
   exists: fold into SparkWeightdMeshPoll) reads own host doorbells; on
   advance, posts payload RDMA (GPU->peers GPU, offset = band+rank*32MB) +
   seq RDMA (8B GPU slot-end AND 8B peer host doorbells at rank offset, from
   the seq staging word — dual sge/post). Doorbell read = plain load, no ioctl.
5. Module additions (nvcc side): SparkGlm5NextModuleMeshWait(ctx, slot_base,
   ordinal, degree, stream) and SparkGlm5NextModuleMeshDone(ctx, host_word,
   ordinal, stream) — expose via config fn pointers like combine.
6. Numerics: combine kernels in peer order (module reference path) replaces
   the CPU f32 sum — also the accuracy gate's correct reference.

Expected: launches amortize in a deep stream (~5us submissions), rounds ~
network+kernel latency (~200-500us), module slots pipeline again. Stage B+
graphs remain the further 2x if needed after this lands.

Measurement gates: after (1-4) expect <= 8s/32tok; after module kernels
<= 5s; then attack prefill (chunked) and re-run COMPSEC-17.
Danger notes: cudaMalloc in weightd needs the primary context (call
cudaSetDevice(0)+cudaFree(0) trick in main before the mesh thread); IPC
handle lifetime = weightd process (mesh re-wire keeps allocation, only QPs
re-transition); engine must cudaIpcCloseMemHandle at Destroy.


## Hill-climb iteration 6 (09-11 ~07:40, lane 8c57291) — third reliability defect

- Found: the api connects its engine ONLY at startup — a residentd restart
  mid-connection left every request failing status 4 forever (engine_completed=0,
  boot_pid pinned to the dead residentd). Fix: the agent's ensure_api kills any
  api that PREDATES the running residentd (start clock ticks from /proc stat
  field 22; /proc dir mtime proved unreliable) and the gate relaunches it.
- sparke REBOOTED (~262s uptime; the 15-min boot guard held autospawn by
  design). Fleet rides 15/16 until its window opens, then mesh re-ships records
  and everything converges. NEXT RUN: verify serving end-to-end post-reboot,
  measure warm 32-tok (expect ~20.3s baseline), then START THE STAGE A/B BUILD
  per the function-level order above (GPU-resident mesh + stream-ordered
  rounds) — that is the operator directive.


## Hill-climb iteration 7 (09-11 ~08:20, lane e1e4c52) — doorbell machinery landed

- Engines now publish (seq,bytes) to a doorbell page appended to the mesh
  region; weightd's mesh thread polls (20us) and posts payload+seq RDMA to all
  peers. The unix socket is OFF the round path; the client broadcast call and
  its IO-exit path are deleted from the engine. This is the Stage-B trigger
  machinery (GPU-move reuses it verbatim).
- Measurement: doorbell 30-31.6s vs TODAY'S pre-build baseline 30.9s — parity
  (the ~130us IPC saving is noise). FLEET DRIFTED +10s vs last night's 20.3s
  (d2h phase 304us today vs 120us then) — environmental, NOT the doorbell.
  NEXT RUN FIRST: diagnose the drift (per-phase compare, check sparke
  post-reboot state, GPU clocks/contention, concurrent actor builds) before
  trusting any perf deltas; then continue Stage A (GPU-resident mesh per the
  function-level order above — doorbell loop becomes the GPU-posting poller
  unchanged).

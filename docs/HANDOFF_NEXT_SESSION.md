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

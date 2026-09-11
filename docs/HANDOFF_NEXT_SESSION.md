# HANDOFF: glm53flash SOTA hill climb — session pickup (09-11 ~16:30)

Lane: lane/glm53-p0, tip b0403a7. Fleet 16/16, serving STABLE (20/20 stress,
zero segfaults). Timer: 15-minute cadence (automation-779eb59b), prompt carries
the full work order. Goal: 14+ tok/s plain B1 (no speculation), prefill
seconds-class, KV warm ~0.

## CURRENT MEASURED STATE

- B1 decode: 20.26s / 32 tok = 1.58 tok/s warm (small 6-tok prompt).
- COMPSEC-17 (372-tok question): cold prefill 209s (562ms/tok, sequential —
  no chunked batch prefill), warm re-run 29s residual, warm decode 1.81 tok/s.
- Engine round (phase-instrumented, pre-doorbell build): d2h ~120us,
  broadcast 130-145us (after merged IPC; was 160-950), spin ~60us,
  combine ~180us. Round total 0.5-1.3ms. Old-system reference: 447us allreduce.
- Launch tax (strace-proven, the dominant cost): 12.6K CUDA launch ioctls per
  token at ~22us, serialized because each synchronous round drains the
  pipeline (~160 launches/layer wait on an idle GPU). ~277ms/token.
- Numerics: tokens are all-zeros / repeated garbage — B1 accuracy gate OPEN.

## SHORT-TERM GOAL (operator-set): the seven remaining tasks

1. RE-LAND host-registered stream-ordered engine. Code preserved at git
   426e038 (engine round) — adapt to the CURRENT engine shape (which now has
   the completion drain thread + queue in tp_device_collective.c). Key swap:
   PrepareReceiveBf16 cudaHostRegister's the 2GB mesh mapping ONCE (probe
   verified 0.66s; kernels read host-registered memory zero-copy at
   0.49ms/32MB), round becomes pure stream-ordered enqueues: D2P copy (no
   sync), staged doorbell H2H copies, CPU spin on peer slot-end words,
   own-slot H2D, module combine kernels via the registered pointer.
   rc=712 that forced its revert = the completion-recursion corruption NOW
   FIXED (see ledger). Expect boot to work this time.
2. Stage B: module MeshWait kernel (GPU polls peer slot-ends on-stream) +
   combine kernels on the registered mesh; removes the last CPU spin.
3. CUDA graph capture of whole layers (legal once no CPU syncs inside) —
   amortizes the launch tax; this is the 10x.
4. u64-maxloc path in the stream-ordered engine (head-max op returns
   UNSUPPORTED in the 426e038 shape) — GATES task 1.
5. Chunked batch prefill via the module batch ladder (buckets to 1024).
6. KV warm residual 29s -> ~0 (64-token cliff suspect).
7. Numerics B1 gate (GPU combine is the reference path).
Order: 4 gates 1; 1 gates 2-3; 5-7 interleave on any window.
Overlap thesis (operator-endorsed): the allreduce is NOT the gap (300-500us,
old ring 447us); serialization multiplies it via the launch tax. Slot
pipelining (4 inflight slots, per-slot streams) + stream ordering hide it
behind other-wave compute; graphs amortize launches; steady state = compute
bound, allreduce exposure ~0.

## FIXED-BUG LEDGER (do not re-diagnose; recipes at bottom)

- Completion recursion segfault: inline completion re-entered the module chain
  (ReduceHiddenWide -> SubmitInternal -> completion -> TpChainAdvance ->
  ReduceHiddenWide...) -> stack exhaustion. FIX: completion drain thread +
  queue (161b5fb).
- QP flush-error permanence: ONE transient RDMA failure puts the RC QP in
  error forever; later rounds all time out (round 3+ stall, both directions;
  was ALSO the "+10s fleet drift"). FIX: doorbell loop rewires via TryWire +
  retries post once; CQ drain rewires on IBV_WC_WR_FLUSH_ERR (e8cc397).
- Slot exhaustion wedge: idle retained slots evict on claim (db025fa).
- API predates residentd: agent kills any API older than the running residentd
  (proc start ticks, /proc stat field 22).
- Mesh poll raced mesh-thread init (NULL QPs -> segfault at boot on loaded
  nodes): poll gated on mesh_active.
- sparkf self-ssh host key: agent keyscans when connect fails.
- Record shipping races: content-hash keyed (.shipped_sha), 10s peer record
  re-fetch, records deleted on weightd (re)start.
- peermem CANNOT load on kernel 6.17 (modprobe EINVAL; module 580.159.03
  present) -> cudaMalloc EFAULT + VMM-dmabuf EINVAL both dead. The
  host-registered design (task 1) needs none of it. VMM code at 4349579 if
  peermem ever lands.
- Spine re-certification: full sha ONCE per pack state, then ck128 vs receipt
  in /tmp/spark-weightd-spine/. weightd wave 6min -> ~90s.
- env.local DELETED (was silently defaulting MTP ON; pool/spine budgets now
  stated module defaults; MTP defaults OFF).
- Old mislabels corrected: "2GB register wedge" was the spine hash; "253k-pass
  runaway" was the ~kHz idle steploop; rc=712 was the recursion corruption.

## BUILD / DEPLOY / MEASURE (the fast cycle)

Build on sparkf (worktree ~/sparkpipe-build):
  ssh sparkf 'cd ~/sparkpipe-build && export PATH="/usr/local/cuda/bin:$PATH" && \
    git fetch -q origin lane/glm53-p0 && git reset -q --hard origin/lane/glm53-p0 && \
    rm -rf build/module_library modules/glm5_next_resident_decode_stage/build && \
    tools/module_build_release.sh glm5_next_resident_decode_stage fp8 glm53flash.fp8.tp16 \
    84c6a6aa9497188e15a635ba793b0f95a79b1033 model_contracts/glm53_flash_authoritative.json'
Edit in /Users/mac/lane-glm53 (verify HEAD/index first — concurrent actor!),
commit + push ONLY via /Users/mac/sparkpipe/tools/sparkpipe_github_pat.sh
(never print the PAT; identity must be sparkpipe).
Deploy automatic (1s agent loop). Poll: ssh sparkf 'grep -l ready ~/current/*.json |
wc -l' until 16 (module-only wave ~30s, weightd wave ~90s with receipts).
API self-heals after waves (predates guard; ~22s connect blocking vs booting
residentd is normal — WAIT, don't kill). Then wait for 'api ready' in
~/sparkdata/glm53flash.fp8.tp16/api.log.
Measure: 32-tok curl on spark0:8433 (prompt_token_ids; NO text prompts — no
tokenizer sidecar). COMPSEC split: /tmp/compsec_prefill_decode.py + fixtures
on spark0 (re-scp from a checkout if missing). NEVER benchmark mid-rollout.
perf is LOCKED (paranoid=4), no nsys on nodes.

## DEBUG RECIPES

- Segfault live-catch: nohup gdb -p <residentd-pid> -batch -ex "handle SIGSEGV
  stop nopass" -ex continue -ex "bt 8" > /tmp/segfault.bt 2>&1 & then one curl.
- Stack sampling: gdb -p PID -batch -ex "thread apply all bt 4" (repeat).
- Syscall census: strace -f -p PID -e trace=ioctl -c -w (idle control first!).
- Mesh memory dump: python3 seek on weightd's /proc/PID/fd/<memfd>
  (band*16*32MB + slot*32MB + 32MB-8 for slot-end seq words).
- QP health: grep WD-MESH-CQERR in ~/weightd.log (status 5 = flush = the
  repaired class; if it returns with stalls, the rewire path regressed).
- Mac->spark ssh is FLAKY under bursts: retry the command, don't diagnose ssh.

## ARCHITECTURE FACTS (the deployed engine today)

- Mesh: 2GB CPU memfd shared engine<->weightd (SCM_RIGHTS on lazy attach);
  4 bands x 16 slots x 32MB; band = collective_identifier & 3 (identifiers are
  64-bit uniquifiers — NEVER small ints); slot = rank; seq stamped at fixed
  slot-end-8. Doorbell page appended after the 2GB: engine publishes
  (seq,bytes) per (band,rank); weightd's mesh thread polls 20us and posts
  payload+seq RDMA to all peers. ZERO per-round IPC.
- Engine (tp_device_collective.c ~450 lines): inline round (D2H+sync,
  doorbell publish, spin, CPU bf16 sum, H2D) + completion drain thread.
  Crash contract: engine _exits on weightd IO errors.
- weightd: 30 QPs/node on rocep1s0f1 (switch port), TryWire re-wires on peer
  record change, records carry doorbell addr+rkey, spine receipts on disk.
- Deployment identifiers: env deleted; MTP off by default; module reads
  pool/spine budgets with stated defaults.

## GIT ARCHAEOLOGY

- 426e038: host-registered engine round (task 1 source) + diagnostic print.
- 4349579: full VMM/dmabuf attempt (dead without peermem).
- 19d24ed: last pre-VMM known-good engine (current == byte-identical + fixes).
- PR #913 merged lane->main (main carries the platform; #912 was the stale
  merge that started the alignment wave).

## Standing cleanup (any window): failed-request API silence; full Mimosa
rerun (hook flagged enobufs all session); rdma_control.c out of the DSO
build; sparke reboot cause; strip leftover MESH-PHASE prints if any remain.

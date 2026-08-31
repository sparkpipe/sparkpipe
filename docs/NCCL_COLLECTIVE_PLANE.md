# The NCCL collective plane — design (coordinator, 2026-08-31)

Measured facts this design stands on (16 ranks, verified, receipts in
NCCL_16WIDE_RECEIPTS.md + this doc's table):

  size    per-op    busbw/node   % of 400Gbps
  8KB     105.9us   1.2 Gbps     0.29%
  14KB    103.3us   2.1 Gbps     0.5%
  80KB    217.4us   5.6 Gbps     1.4%
  1MB     885.7us   17.6 Gbps    4.4%
  4MB     1807.2us  34.8 Gbps    8.7%

  The fabric itself proves ~90% (the flash dev's 2-node proof). The gap
  is algorithm + message size, not the wire.

## Diagnosis: three regimes

1. LATENCY regime (B1-B8, messages 8-80KB): the ring's 2(N-1) serialized
   hops dominate; 100us floor at 16 ranks regardless of size. Serving
   today sits here at ~0.06% fabric utilization - and that is FINE, B1
   is not fabric-bound, it is serialization-bound (the 95 host-tier
   round trips). NCCL's value here is stream-ordering (overlap), not
   bandwidth.
2. AMORTIZATION regime (B16-B128, 200KB-2MB): per-byte cost falls ~10x;
   aggregate tok/s scales with B as weight streaming amortizes.
3. BANDWIDTH regime (B256+, multi-MB): fabric-bound. TODAY'S PATTERN
   DIES HERE: 95 separate 14KBxB collectives at 8.7% busbw = ~180
   tok/s ceiling. The fix is traffic shape, not speed: fewer, bigger,
   overlapped ops (fused groups), and busbw tuning toward 150+Gbps.

## The design: five layers, one config flip

### L1 Comm lifecycle - in-process, zero choreography
backend:"nccl" in the stage config (member set already parses). At
adapter init: rank0 calls ncclGetUniqueId() IN THE RESIDENTD PROCESS and
broadcasts the 128-byte id over the EXISTING hidden-transport control
channel (peer_hosts/listen_port are already in every config and up
before collectives start - the channel that carries the host-tier
rendezvous today). Every rank ncclCommInitRank locally. NO files, NO
ssh, NO fanout - the entire 16-rank choreography that made the bench
"hacky" disappears into the serving stack. The env pins
(NCCL_SOCKET_IFNAME=enp1s0f1np1, NCCL_IB_HCA=rocep1s0f1,
NCCL_IB_GID_INDEX=3) become BACKEND DEFAULTS, probed at init (enumerate
ifaces; fail loud with the exact recipe string if absent). One comm per
deployment; teardown at shutdown.

### L2 The collective ABI - unchanged surface, nccl underneath
SparkTpDeviceCollective submit/completion stays the module-facing
interface (the ABI-13 surface). The nccl backend implements it:
ncclGroupStart/End fuses a step's submissions; ncclAllReduce/Reduce/
Broadcast issue ON THE MODULE'S EXECUTION STREAM; completion is
stream-ordered (cudaLaunchHostFunc -> the existing completion_function,
the K3RunnerTpCompletion pattern). The 9 host-tier stream-syncs in
glm5_next simply never execute on this backend. Per-op latency at
serving sizes ~103us AND stream-ordered = pipelined behind compute.

### L3 Size-adaptive algorithm - push the small-message floor
Below a measured crossover (~128KB, to be pinned by receipts):
the DIRECT path - one hop, 15-way parallel exchange + local reduce
(#760's d2a is the hidden-transport version; projected ~15-30us at 16
ranks vs the ring's 103us). Above: NCCL ring, tuned for the dual rails
(NCCL_NRINGS=2; evaluate the ext-net RoCE plugin for busbw past 35
Gbps). The crossover is a config constant backed by a receipt table,
never a guess. Both paths implement the same ABI; the backend picks
per submission by payload bytes.

### L4 Traffic reduction - the actual roofline unlock at large B
Fuse aggressively: the P2 "predeclared collective program" - group the
whole step's exchanges (ncclGroup semantics or the fused frames) so B
large means FEWER x BIGGER ops, riding the per-byte curve (14KB: 2.1
Gbps vs 4MB: 34.8 Gbps = 16x better per byte). Target shape at B1024:
~10-15 multi-MB groups instead of 95 medium ops.

### L5 Overlap - step time becomes max(), not sum
Stream-ordered NCCL + the async loop (p1d2, landed) + per-layer
pipelining: compute of layer k+1 overlaps the wire of layer k. The B1
projection depends on this: serialized-collectives B1 ~20 tok/s;
overlapped ~25+ tok/s. At large B overlap decides whether the fabric
or the compute is the visible wall.

## Exactness policy (decided before timing)
NCCL's ring reduction order differs from recursive doubling - outputs
will NOT be bit-identical to the host-tier reference. Policy: the
T-gate compares NCCL-backend output to the host-tier output within the
bf16-rounding tolerance band already used by the split-K cell
(max_rel <= 3e-2 with deterministic re-runs); the reference for FUTURE
gates becomes whichever backend is deployed. Mismatch beyond tolerance
= RED stop, per the lane rule.

## Rollout
1. glm5_next first (it has the async loop; the rc=1 module-create fix
   is its prerequisite - see ENGAGEMENT_HANDOFF.md).
2. qwen38_27b together with its async port: the port removes the
   host-tier syncs the nccl backend would otherwise keep hitting -
   one landing, both fixes.
3. The B-ladder cell (B1/8/16/64/256) measuring aggregate tok/s AND
   per-node fabric Gbps - the table that answers "how close to the
   roofline" at every batch size, before and after L3/L4 tuning.

## Projections (from receipts, honestly labeled)
B1 (32K ctx): 78.7ms step - 18.7ms (split-K) - 1.4ms (screened head)
= 58.6ms; collectives 19.5ms serialized -> 9.8ms NCCL (serialized) or
~0 (overlapped): 20-26 tok/s (vs 12.7 today).
B16: step ~50-90ms for 16 tokens (weights amortize, comm ~300-500us/op
serialized or hidden): ~250-350 tok/s aggregate.
B64: 400-700 tok/s aggregate (comm ~40-60% of step; overlap matters).
B256+: fabric regime - L4 fusion decides; with 95-op shape ~180 tok/s
ceiling, with 10-15 fused multi-MB groups ~500-1000+ tok/s aggregate.
Fabric utilization: today's serving 0.06%; NCCL bench 1.4% @80KB,
8.7% @4MB; fabric proves 90% - L3/L4 exist to close that gap.

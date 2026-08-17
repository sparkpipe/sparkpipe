# Qwen 3.8 Max TP16 status - expert-sharded MoE foundation

TP sharding progress: the module-side wiring is LANDED and compile-verified;
the two-process smoke reaches SparkTpDeviceCollectiveCreate and is blocked on
one backend contract detail.

## Landed (this round)

1. SparkTpDeviceCollective wired into the module (env-driven):
   SPARK_QWEN38_STAGE_TP_DEGREE / _TP_RANK / _TP_BACKEND_PATH /
   _TP_IDENTIFIER / _TP_PORT_BASE / _TP_HOSTS (comma list) /
   _TP_LOCAL_HOST / _TP_TIMEOUT_MS. Recursive-doubling mask; the
   module provides the combine kernel (SparkQwen38LaunchTpCombineAdd,
   dest += src bf16 pairs in f32), credit bindings (route x 2 credits,
   max-512-row sizing), memory-mode probe + mapped-host credit buffers,
   and the submission path with STREAM_ORDERED_COMPLETION + a host wait
   on the completion flag.
2. Expert-sharded MoE: the grouped tile and scalar launchers take
   (tp_degree, tp_rank) and shift the expert-major payload/scales/
   offsets/prefix pointers by rank x (512/tp) - all row/output indexing
   stays GLOBAL, so the peer ranks' rows remain consistent and the
   pair-reduce reads the full inverse map. The slot buffer is zeroed
   per frame so each rank's pair-reduce sums only its experts.
3. The residual all-reduce reduces the DELTA right after the pair
   reduce - BEFORE the replicated shared expert and the residual add -
   so the shared expert and the base hidden are never double-counted
   by the reduce (the first placement reduced the hidden and would
   have doubled both).
4. tp_degree>1 refusal removed; tp validates 1..512 (any divisor of
   the expert count). tp=1 remains byte-identical (smokes pass).

## Blocked: SparkTpDeviceCollectiveCreate returns CAPACITY_EXCEEDED

Status trace on spark4 (two-process, same-host, rail IP 10.10.200.4,
backend libhidden_transport_spark_host_rdma_verbs.so, ports 66620/66621):
ApplyTopology OK, CreditBindingRouteCount OK, ProbeMemoryMode OK (the .so
loads), Validate* all pass, Create -> CAPACITY_EXCEEDED. The remaining
suspects, in order for the next session:
1. The backend .so's initialize: its buffer/queue sizing against the
   config (max_active_sequence_count 512, credit_count 2, route_count 1)
   - read modules/../ring/transport/rdma.cu's create or the endpoint
   memory contract.
2. The collective's own credit-plane math with the mapped-host mode
   (the probe result on this host - check which mode it returned and
   whether the binding flags/buffers match).
3. Compare the qwen38 config field-by-field with dsv4's working
   SparkDsv4ModuleInitializeTpCollective (the pattern this port
   follows) - the adapter-side topology there carries rail hosts and
   thresholds that the .so may require even for recursive doubling.

## Next increments after the smoke passes

- tp=2/4 two- and four-process smokes (l1 GDN + l2_3 attention packs),
  then tp=16.
- Attention head-sliced projections + the strided o_proj (or the
  all-gather variant) + per-rank KV strides (the head-parallel kernels
  already take tp_degree/tp_rank; 4 KV heads need the 4x-replication
  extension for tp>4).
- GDN channel slicing (the fused qkv needs per-rank q/k/v contiguous
  views + a runtime conv channel count).
- Per-request latency measurements at tp=2/4/16 (target 10+ tok/s).

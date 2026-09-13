# SparkPipe Constant Audit — 2026-09-13

Full-codebase sweep for hardcoded constants that are (a) derived values
restated as literals, or (b) shared values stated independently in more
than one place. Law (operator, 2026-09-12): every derived constant
derives in code from one #define; cross-domain ties are _Static_asserts;
scripts parse the header. Each violation below is a live drift hazard —
two of them already cost debugging days (the 4 GB geometry set, the
miss-record packing).

## The minimal generative set (stated config; everything else derives)

Domain constants live in exactly one header each:

- `model-families/<model>/include/sparkpipe/spark_<model>_model.h` —
  model geometry (layers, hidden, experts, top-k, first routed layer,
  HC mult, KV dims, vocab, context).
- `include/sparkpipe/spark_weightd.h` — mesh geometry
  (MAX_LANES, MAX_BATCH_ROWS, ROW_BYTES_MAX, RANKS_PER_BAND,
  SLOTS_PER_RANK derive these), lease budget (LEASE_COUNT_MAX,
  LEASE_GROUPS_MAX), doorbell layout (entry stride, cell indices).
- `ring/transport/tp_device_collective.c` — transport addressing
  (CHAIN_ID_BITS, CHAIN_ROUND_BITS, WAVE_STRIDE_BITS, STAGING_SETS,
  control-cell numbering derived from the doorbell header).
- `include/sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h`
  — module capacities (rows, slots, MTP depth).

## Violations found (this sweep)

Ranked by drift risk; each entry lists every site so fixes can be
verified by grep-to-zero.

1. Miss-record packing stride `512` — restates nothing; it is a
   round-padded stand-in for MOE_EXPERT_COUNT (288). Sites:
   `modules/.../spark_glm5_next_resident_decode_stage_module.c`
   (decode `/512u`,`%512u`, x1 each), `modules/.../cuda/layer.cuh`
   (encode `*512u`, x1). Fix: derive the stride from
   SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT rounded to a power of two in
   ONE define; _Static_assert it exceeds the expert count.
2. Miss-ring capacity `64` and ring buffer `512u` bytes — kernel masks
   `& 63u` (layer.cuh), host compares `< 64u` and memsets/allocates
   `512u` bytes (module, x6 sites). The 512-byte buffer, 64-record cap,
   and 2-word header are three independent statements of one structure.
   Fix: one MISS_RING_CAPACITY define + derived byte size; share between
   host and kernel headers.
3. Union lease caps `2400` and `1888` — module (array size, three
   comparisons, one margin expression). These encode the lease-budget
   arithmetic (LEASE_GROUPS_MAX × LEASE_COUNT_MAX minus safety margin)
   but are stated as bare literals. Fix: derive from
   SPARK_WEIGHTD_LEASE_COUNT_MAX × SPARK_WEIGHTD_LEASE_GROUPS_MAX with
   a named margin define; _Static_assert ordering.
4. Doorbell control-cell indices `100`/`101` — transport
   (`(100u + 2u*band)*24u`, `(101u + 2u*band)*24u`). Magic base offsets
   in the doorbell page. Fix: named CELL_BASE/CELL_CANCEL defines in
   the weightd header next to the doorbell layout they belong to.
5. Peer count `15` (`SPARK_WEIGHTD_MESH_PEERS` in weightd_mesh.c) —
   restates RANKS_PER_BAND(16) − 1. Fix: derive from the mesh header.
6. Peer masks `0xffffu` (transport, x2) — restates 16 ranks. Fix:
   derive `((1u << RANKS_PER_BAND) - 1u)`.
7. Doorbell entry stride `24u` — the header macro derives it, but
   weightd_mesh.c scans entries as `index * 3u` u64 words and the
   transport computes `* 24u` directly for control cells. Fix: one
   DOORBELL_ENTRY_BYTES define; assert 3-word layout where scanned.
8. Chain-key round bits `16` appears as `65536ull`/`0xffffull` masks in
   the daemon's jump resync (`node/weightd_mesh.c`) — restates
   CHAIN_ROUND_BITS from the transport. Cross-domain tie: export the
   bits in the weightd header or _Static_assert equality.

## Clean (verified this sweep)

- Mesh slot geometry: SLOT_BYTES derives from batch law; BANDS from
  MAX_LANES; doorbell page size asserted. `tools/s` parses the header.
- ChainKey encoding in the transport: all from CHAIN_ID_BITS /
  CHAIN_ROUND_BITS / WAVE_STRIDE_BITS.
- Model geometry: single model header; the module validates config
  against it at boot and ties row width to the mesh law at compile
  time.
- CQ depth, staging sets, spin cap: named defines at point of use.

## Not literals but drift-adjacent (recorded, lower priority)

- Kernel `cover_stride` recomputed at each site from the expert count
  instead of reading one derived define (host computes, kernel
  recomputes — same formula, two statements).
- The stable-read retry count (64) in the daemon doorbell scan and the
  CQ drain batch (64) are independent 64s — same value by coincidence,
  not by derivation; fine while unrelated, but they must not grow a
  dependency.

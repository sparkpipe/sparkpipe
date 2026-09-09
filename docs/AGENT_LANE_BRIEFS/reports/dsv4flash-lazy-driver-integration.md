# DSV4 Flash lazy driver integration — design (pre-implementation)

Status: DESIGN for the driver-side lazy consumer, written before hot-path
surgery per the #827 review (plan-first for lazy). Daemon side is complete
and receipted (ATTACH_LAZY + leases + EXPORT_LEASE_BATCH on the placed
rank1 pack, dsv4flash-lazy-smoke4).

## Current eager path (what changes)

`SparkDsv4ModuleLoadEntry` binds every tensor to either
- `state->weightd_arena_base + offset` (weightd attached; whole 22G loaded), or
- a direct pack read (`SparkStageModuleLoadDeviceRegion`).

Expert kernel inputs are the bound `experts_w1/w2/w3` LinearViews. The
route is computed ON GPU per layer (`SparkDsv4LaunchGateRoute` →
`indices_u32`/`route_packed_row` device buffers); the host never sees it —
which is why a host-driven ACQUIRE cannot be inserted without a route
readback.

## Lazy design (SPARK_DSV4_EXPERT_LAZY=1 runtime env, default off)

1. **Attach**: `SparkWeightdAttachPackLazyEnv` (shared helper) replaces the
   eager attach when the gate is on. Consumer reserves an arena-sized VA
   (`cuMemAddressReserve`) once at load; positions stay canonical
   (chunk_index × chunk_bytes), so `experts_w*` binding arithmetic is
   unchanged — but expert ranges are UNCOMMITTED until acquired.
2. **Spine**: all non-expert tensors keep the direct pack read (they are
   the manifest's spine complement; small — attention/index/norm/dense).
   `SparkWeightdManifestSpineSlice` provides pack-offset → compact-offset
   translation if the daemon later serves spine bytes; until then the pack
   read needs no daemon involvement.
3. **Per-layer ACQUIRE**: after `LaunchGateRoute`, the module copies the
   routed expert ids D2H (topk u32s, 24 B — one small sync per layer),
   builds keys with `SparkWeightdRouteKeys`, `ClientAcquire` (daemon
   commits chunks + copies ranges H2D, synchronous), then launches the
   expert kernels against the now-committed arena VA.
4. **Release**: per the lease contract, after the layer's completion event
   — the module already owns a completion callback seam
   (`SparkDsv4ValidationCompletion`-shaped); release there.
5. **Fail-closed**: missing `.experts`, daemon unreachable, ACQUIRE error
   (capacity/NOT_FOUND/drift), or VA reserve failure → the layer returns
   the named status. No eager re-read path exists behind the gate.

## Costs and interactions (honest)

- **Route readback**: 43 syncs/token. Measured analog: the dspark tap
  (3 syncs) cost ~5 ms/token. 43 × ~120 µs ≈ 5 ms/token at B1 — the price
  of host-driven lazy at B1 on TOP of the ~24 ms eager step. Net win only
  vs a 22 G cold load; the point of lazy is coexistence + startup, not B1
  throughput.
- **CUDA graphs**: capture forbids the per-layer readback+ACQUIRE. Lazy
  decode therefore runs graphs-OFF — but the stage config schema asserts
  `cuda_graph_count_by_pp_stage = [130]` (3×43+1 islands, adapter-checked).
  REQUIRED companion change: the adapter must accept the lazy gate's
  graphs-0 configuration (a family adapter change, this lane) — else lazy
  and the config schema are mutually exclusive.
- **Pool sizing**: declared budget must cover the per-token working set
  (6 experts × 43 layers × (256 KiB payload + 16 KiB scale) ≈ 69 MiB) plus
  headroom; the 384 MiB pool receipted in dsv4flash-lazy-smoke4 is 5.5×.
- **B8+/prefill**: batch routes hit more experts/token; the working-set
  union grows toward the full arena — lazy is a B1/coexistence feature by
  construction (matches #816's B1/B8 split).

## Sequencing

1. This design reviewed (coordinator + operator).
2. Adapter: accept graphs-0 under the lazy env gate (family-local).
3. Module: the 5-point wiring above (family-local; shared client functions
   only — no new daemon protocol).
4. Receipts: spark2 single-node consumer gate (the scanner-blocked tool —
   or in-runner validation legs), then the two-consumer bar (second
   consumer = a second dsv4 identity or glm5_next's lazy consumer).

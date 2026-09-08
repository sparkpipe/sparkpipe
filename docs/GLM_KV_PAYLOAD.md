# GLM Flash cache payload correction

2026-09-08, based on main `55006656a730a8c75a78fbf01c4906ae170fe231`.

The native GLM KV and index allocations contain layer slabs: each slab holds
all physical pages for one layer. The common arena describes backing pages
whose payload contains one physical page from every local layer. Previously
the GLM copy hook transferred this payload as one contiguous device range.
That range instead contains neighboring pages in a layer. The page store also
omitted the index allocation entirely. Checking total allocation size cannot
detect either error.

`SparkKvPageStoreCopyLayered` now gathers/scatters the layer slices through the
existing hardware copy callback. It validates geometry and integer/address
bounds before issuing any copies and propagates copy errors. This common
algorithm has no CUDA dependency and supports padding between layer slabs.
The GLM hook supplies native layout and translates the arena's packed block
address into a physical page index. CUDA allocation and kernel addressing stay
layer-major. The existing key/value payload mechanism carries main KV and
index state respectively; staging and backing page sizes include both. A new
layout fingerprint prevents old backing payloads from being treated as this
format.

## Evidence

- `make build/test_kv_page_layout && build/test_kv_page_layout`: three layers,
  five pages, layer padding, round trips preserving every untouched byte,
  invalid/overflow geometry rejected before copying, copy failure propagation.
- `python3 tests/test_glm5_next_stage_context.py`: actual GLM copy hook packs
  and restores both KV and index regions across three layers and five pages.
  Replacing just the new hook with the old contiguous copy makes this test
  exit 2 at its payload check; the corrected hook passes. The harness uses a
  host memcpy implementation of the CUDA copy boundary.
- `build/test_kv_cache` and `build/test_kv_model_table`: common arena and backend
  configuration checks pass.

These are host correctness results. They do not prove CUDA transfer performance,
distributed numerical accuracy, or full prefix reuse. Full GLM integration
still needs dynamic logical/physical mappings and complete KDA, convolution and
continuity state restoration, followed by prefix-hit versus uninterrupted
execution checks. Do not enable a cache capability merely because page copies
are now correct.

## Lesson for later drivers

Trace allocation, kernel indexing, arena addressing and backing payload layout
together. Test multiple layers and pages with distinguishable contents before
optimizing transfers. Put the gather/scatter algorithm in common code and keep
native geometry and hardware copying behind explicit hooks. Once the serving
path is correct, measure per-layer transfer overhead before changing the native
layout or introducing a device packing operation.

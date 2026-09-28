# Relocatable CUDA graphs (B1: in-process relocation)

An instantiated CUDA graph can be moved to other buffers by patching its
pointer arguments in place, with no re-capture. Two uses:

- one graph can serve several execution slots;
- a graph keeps working after its buffers move (a weight-map base remap, a
  mesh window, a re-allocated slot).

The code is model-neutral: `include/sparkpipe/spark_graph_reloc.h`,
`runtime/spark_graph_reloc.c` (registry, image, validation, rebase) and
`runtime/spark_graph_reloc_cuda.c` (the CUDA node walker and exec patcher).

## Model

- **Region registry.** `Register(id, base, bytes, class, slot)`. `slot` is an
  execution slot, or `SPARK_GRAPH_RELOC_SLOT_SHARED` for memory that every slot
  uses (weights, the weight map, collective windows). `Seal` sorts the
  regions. It refuses overlapping regions and a duplicate `(id, slot)`.
  - The end of a region is inclusive, so one-past-end pointers classify.
  - A word equal to one region's end *and* the next region's base is
    **ambiguous**, and the walker refuses it. Callers must keep a guard gap
    between registered buffers. The GPU test allocates 512 extra bytes per
    buffer for this.
- **Walk.** `SparkGraphRelocCaptureCuda` walks the captured graph and reads
  every node:
  - For a kernel node, it takes the parameter layout from
    `cuFuncGetParamInfo`, then scans every 8-byte-aligned word of every
    parameter. That includes pointers inside structs passed by value.
  - For memcpy, memset and host nodes, it scans their pointer fields.
  - Event nodes are compared as constants.

  Each word is classified:
  - a constant;
  - a **site** `(region id, offset)`;
  - **unknown**, when the word falls inside the registry window or the CUDA
    pointer probe knows it. The capture is refused with `GRAPH-RELOC-UNKNOWN`.

  A site in another slot's region is refused with `GRAPH-RELOC-SLOT`. Memory
  allocation nodes, array copies and `extra` launches are refused.
- **Two-capture validation.** `ValidatePair` compares captures of the same
  shape made on two slots. The site lists must match, with the same region ids
  and offsets, and every non-site byte must be identical. A scalar that
  happens to equal a region address is caught here.
- **Rebase and apply.** `Rebase` writes each site as the target registry's
  `base + offset` for the target slot. `ApplyCuda` pushes the changed nodes
  with `cudaGraphExec{Kernel,Memcpy,Memset,Host}NodeSetParams`.
  - Every exec owns a **held** blob: the parameter bytes that exec holds right
    now. `HeldReset` seeds it from the capture image when the exec is
    instantiated. `ApplyCuda` compares the target blob with the held blob, not
    with the capture, and copies each node it patched into the held blob. A
    round trip (slot 0 → 1 → 0, or a base moved and moved back) therefore
    patches the nodes back. Diffing against the capture would report 0
    changes on the way back and leave the exec on the other slot's buffers.
  - `force=1` patches every node that has a site, whatever the held blob says.
  - The held blob must not alias the target blob or the capture image.
    `SparkGraphRelocWorkspace` carves one held blob per image
    (`held_blob[0]`, `held_blob[1]`).

## Tests

- `build/test_graph_reloc` (host, in `make test`, CUDA stub). It covers the
  registry, classification, the walker over every node kind, planted scalars,
  refusals and rebase targets.
- `make test-graph-reloc-gpu` (sparkf, sm_121a) runs real CUDA graphs: a
  memset, two kernels (one of them takes a struct holding a pointer by value)
  and a D2H copy.
  - The captures on slot 0 and slot 1 validate as a pair (9 sites).
  - The slot-0 exec, patched to slot 1, gives output bit-identical to a fresh
    slot-1 capture.
  - The shared weight base moves to a copy while the old base is poisoned with
    NaN. The patched exec matches a fresh capture on the new base bitwise.
  - An unregistered device pointer in a kernel argument is refused.
  - Round trip: the exec patched to slot 1 is patched back to slot 0 with
    `force=0`. All 4 nodes are re-patched, and the output is bit-identical to
    the original slot-0 run. Re-applying the same target patches 0 nodes.
  - Timing for this 4-node graph: capture+instantiate against rebase+patch.
    The patch loop alternates slot 0 and slot 1 with `force=0`, so every
    iteration really patches all 4 nodes.

## Integration prerequisites

- **Keep the template graph alive.** `cudaGraphExec*NodeSetParams` takes node
  handles from the source `cudaGraph_t`. The image stores those handles, so
  the template graph must outlive every patch of every exec made from it. A
  caller that destroys the graph right after instantiate must stop doing so
  before it uses `ApplyCuda`.
- **No adjacent regions.** A word equal to one region's inclusive end and the
  next region's base is refused as ambiguous. Sub-allocations of a ledger
  that sit back to back need a guard gap between them, or must be registered
  as one region, before a graph that uses them can be walked.

## Not yet

- Registering the GLM stage buffers: the ledger, KV/KDA, slot scratch, the
  weight map (`map->base`), collective windows, and the cover, ring and
  snapshot.
- Sharing one exec across slots in the module.
- B2: saved `.spg` artifacts keyed by build, pack, rank and shape.

Measure `capture_ms` in G5N-WAVE-TIMING first. B1 cuts engine-ready time, not
per-token time.

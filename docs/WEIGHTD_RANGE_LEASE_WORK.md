# Multi-range expert acquisition work in progress

The real GLM Flash TP16 pack has 12,096 logical experts and 24,192 weight ranges before scale ranges. The old 4,096-entry single-range ENSURE interface cannot represent this. Do not increase its cap or treat the first matching range as the whole expert.

The shared version-2 manifest parser now represents explicitly typed ranges grouped by `(layer, expert)`, with bounded startup allocations and binary-search lookup. It validates complete file framing, range bounds, non-overlap and unique range kinds. The format uses a 16-byte header and 48-byte range records, documented in `spark_weightd_manifest.h`. Version 1 is not guessed or converted. Range kinds are producer-defined stable identifiers; the producer and consumer must agree on them, including separate scale ranges.

This branch is incomplete and must remain a draft. The parser is built into the shared runtime but is not yet used by the daemon. Remaining integration:

- Generate complete weight and scale ranges from the actual codec layout, validate against the shared parser, and atomically publish `.experts` without destroying an existing artifact on failure.
- Replace single-range ENSURE with generation-scoped working-set acquisition. Return all ranges plus a lease; do not expose successful partial acquisition.
- Plan memory against the union of required chunks and protect every acquired range before evicting unleased ranges. Use bounded staging and preserve correctness on allocation, read, checksum and mapping failures.
- Export only ranges covered by the consumer's lease and map them into its own CUDA virtual address space. Spine residency is separate from routed expert working sets.
- Hold the lease through actual GPU completion, then unmap and release. Handle cancellation and disconnect without releasing backing storage still used by a GPU.
- Exercise two real driver consumers, eviction pressure, corruption, cancellation and restart. Retain numerical and residency receipts from clean merged-main deployment before claiming hardware qualification.

`tests/test_weightd_manifest.py` currently checks 12,096 logical groups with four ranges each (two weights and two scale fixtures), exact lookup and malformed files. These are host parser tests, not a claim of lazy GPU functionality or complete scale-layout validation.

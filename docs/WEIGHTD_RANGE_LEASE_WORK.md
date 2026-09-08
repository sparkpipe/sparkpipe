# Multi-range expert acquisition work in progress

The real GLM Flash TP16 pack has 12,096 logical experts and 24,192 weight ranges before scale ranges. The old 4,096-entry single-range ENSURE interface cannot represent this. Do not increase its cap or treat the first matching range as the whole expert.

The shared version-2 manifest parser now represents explicitly typed ranges grouped by `(layer, expert)`, with bounded startup allocations and binary-search lookup. It validates complete file framing, range bounds, non-overlap and unique range kinds. The format uses a 16-byte header and 48-byte range records, documented in `spark_weightd_manifest.h`. Version 1 is not guessed or converted. Range kinds are producer-defined stable identifiers; the producer and consumer must agree on them, including separate scale ranges.

This branch is incomplete and must remain a draft. The parser is built into the shared runtime but is not yet used by the daemon. Remaining integration:

The shared lease table now supplies serialized owner-scoped working-set pin
accounting. Acquisition deduplicates keys and commits all pins only after
validation; release checks owner and a monotonically increasing identifier.
An active lease prevents table destruction. Storage is allocated at table
creation, with no acquire/release allocations. Each lease admits up to 512
logical experts, with 64 concurrent leases. Larger prewarm sets require
multiple leases. The daemon must scope the table to an arena generation and
must not reuse connection owner identifiers.

These are accounting invariants only. The table does not establish GPU
completion, load weights, export memory, or make disconnect safe. Those
operations must be wired to it before any claim of protected GPU residency.

- Qualify the new FP8/BF16 generator against corrected packs from PR #843. It now emits payload and scale ranges, validates the complete manifest with the shared parser, uses a fixed 64 KiB read buffer, and publishes exclusively through a temporary file. Unsupported codecs are explicit errors. NVFP4 needs its separate global-scale and block-scale ranges implemented before use. Build with `make build/glm5_next_experts_manifest`; do not run against the known interleaved legacy packs.
- Replace single-range ENSURE with generation-scoped working-set acquisition. Return all ranges plus a lease; do not expose successful partial acquisition.
- Plan memory against the union of required chunks and protect every acquired range before evicting unleased ranges. Use bounded staging and preserve correctness on allocation, read, checksum and mapping failures.
- Export only ranges covered by the consumer's lease and map them into its own CUDA virtual address space. Spine residency is separate from routed expert working sets.
- Hold the lease through actual GPU completion, then unmap and release. Handle cancellation and disconnect without releasing backing storage still used by a GPU.
- Exercise two real driver consumers, eviction pressure, corruption, cancellation and restart. Retain numerical and residency receipts from clean merged-main deployment before claiming hardware qualification.

`tests/test_weightd_manifest.py` currently checks 12,096 logical groups with four ranges each (two weights and two scale fixtures), exact lookup and malformed files. These are host parser tests, not a claim of lazy GPU functionality or complete scale-layout validation.

`build/test_weightd_lease` checks shared pins across owners, duplicate keys,
failure without partial pins, stale/wrong-owner release, capacity bounds and
identifier exhaustion. It is a host accounting test, not a CUDA lifetime gate.

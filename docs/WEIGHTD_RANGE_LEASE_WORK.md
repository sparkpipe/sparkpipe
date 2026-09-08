# Multi-range expert acquisition work in progress

The real GLM Flash TP16 pack has 12,096 logical experts and 24,192 weight ranges before scale ranges. The old 4,096-entry single-range ENSURE interface cannot represent this. Do not increase its cap or treat the first matching range as the whole expert.

The shared version-2 manifest parser now represents explicitly typed ranges grouped by `(layer, expert)`, with bounded startup allocations and binary-search lookup. It validates complete file framing, range bounds, non-overlap and unique range kinds. The format uses a 16-byte header and 48-byte range records, documented in `spark_weightd_manifest.h`. Version 1 is not guessed or converted. Range kinds are producer-defined stable identifiers; the producer and consumer must agree on them, including separate scale ranges.

This branch is incomplete and must remain a draft. The daemon now uses the v2
parser for lazy attachment. IPC ABI 2 adds bounded whole-working-set ACQUIRE and
owner-scoped RELEASE. Legacy single-range ENSURE returns UNSUPPORTED; the old
whole-arena export rejects lazy arenas. Lease-scoped export returns the sorted
physical chunk union, explicit chunk indices, and up to 64 descriptors per
response. Consumer mapping/lifetime integration must land before this branch
is usable for lazy drivers.

The shared lease table now supplies serialized owner-scoped working-set pin
accounting. Acquisition deduplicates keys and commits all pins only after
validation; release checks owner and a monotonically increasing identifier.
An active lease prevents table destruction. Storage is allocated at table
creation, with no acquire/release allocations. Each lease admits up to 512
logical experts, with 64 concurrent leases. Larger prewarm sets require
multiple leases. The daemon must scope the table to an arena generation and
must not reuse connection owner identifiers.

The daemon now pins the complete key set before planning the union of required
physical chunks, includes other owners' pins in its capacity check, evicts only
unpinned groups, and loads all ranges through a 64 KiB buffer. Every range must
pass ck128 before the set is committed. Allocation/read/hash failures release
the new lease and undo newly allocated chunks. Each connection gets a
non-reused owner ID. Explicit detach is BUSY while that owner has active leases.

This still does not establish GPU completion or provide the production
consumer mapping/lifetime helper.
Disconnected owners retain their pins, including across connection-slot reuse;
they cannot be automatically reclaimed without a completion/death proof. This
is an explicit unfinished recovery requirement, not a completed debug system.

- Qualify the new FP8/BF16 generator against corrected packs from PR #843. It now emits payload and scale ranges, validates the complete manifest with the shared parser, uses a fixed 64 KiB read buffer, and publishes exclusively through a temporary file. Unsupported codecs are explicit errors. NVFP4 needs its separate global-scale and block-scale ranges implemented before use. Build with `make build/glm5_next_experts_manifest`; do not run against the known interleaved legacy packs.
- Wire lease-scoped exports into the shared consumer mapping/lifetime helper. Acquisition returns a lease only after every requested expert range loads; export checks the connection owner and returns the exact physical chunk union at allocation granularity.
- Extend fault injection to read, copy and map/unmap failures. First/second allocation failure, partial export failure, capacity rejection, all-range copying and corruption rollback have host-stub coverage; real CUDA qualification remains open.
- Export only ranges covered by the consumer's lease and map them into its own CUDA virtual address space. Spine residency is separate from routed expert working sets.
- Hold the lease through actual GPU completion, then unmap and release. Handle cancellation and disconnect without releasing backing storage still used by a GPU.
- Exercise two real driver consumers, eviction pressure, corruption, cancellation and restart. Retain numerical and residency receipts from clean merged-main deployment before claiming hardware qualification.

`tests/test_weightd_manifest.py` currently checks 12,096 logical groups with four ranges each (two weights and two scale fixtures), exact lookup and malformed files. These are host parser tests, not a claim of lazy GPU functionality or complete scale-layout validation.

`build/test_weightd_lease` checks shared pins across owners, duplicate keys,
failure without partial pins, stale/wrong-owner release, capacity bounds and
identifier exhaustion. It is a host accounting test, not a CUDA lifetime gate.

`build/test_weightd_working_set` runs the real daemon/client IPC with CUDA stubs:
four ranges per expert, shared chunks, two owners, duplicate keys, pool pressure,
wrong-owner release, detach refusal, first/second allocation rollback, corruption,
and disconnected-owner pin retention. It also imports descriptors into a
separate CUDA-stub virtual address, checks weights there, verifies wrong-owner
and stale-lease export rejection, and exercises 65 chunks across two responses.
These remain host-stub tests, not real consumer GPU qualification. The legacy
single-range `test_weightd_expert` fixtures and their callers still require ABI-2
migration before the complete suite can pass.

The existing eager `test_weightd_map` also fails its 2 MiB chunk-size assertion
on unchanged main dd9bfdb (eager arenas use 64 MiB chunks). This baseline fixture
drift was reproduced independently; do not claim the complete mapping suite is
green from the working-set test. It needs correction alongside export tests.

`build/test_weightd_fd_frames` checks 64-FD frames, oversized ancillary cleanup,
truncation rejection, CLOEXEC, and lease-response identity/index/count checks.
The receive buffer accommodates the kernel descriptor limit before applying
the protocol cap, so surplus descriptors can be explicitly closed. Server
exports account each descriptor as it is created, including failure midway
through a batch.

### Shared consumer mapping lifecycle (draft)

The shared `spark_weightd_map` helper reserves consumer-local VA and imports the exact leased chunk union read-only. Startup creates its metadata and completion events. BeginUse marks a lease in flight; Release refuses it until RecordCompletion has recorded an event and that event completes. Overlapping leases share local chunks, and the last local owner unmaps/releases before sending daemon RELEASE. Unused acquisitions can be cancelled. Failed import cleanup retains a nonzero identifier when further cleanup is required. Calls are serialized on the creating CUDA context; callers must join every using stream before recording completion.

The working-set host test exercises the helper through real client/server IPC with CUDA stubs: separate consumer VA, overlapping leases, destroy while busy, release before recording, record failure, pending completion, completed release, stale release, unused cancellation, and partial export rollback. This is not GPU qualification. Production driver/spine integration, real GPU event proof, disconnected-owner recovery, and an overall multi-batch acquisition deadline remain incomplete.

Additional host fault gates cover failure on the second CUDA import, unmap failure retaining the remote pin, and successful cleanup retry. Once event teardown begins, the map rejects new acquisitions even if destruction fails. CUDA-stub gates do not establish GPU correctness.

The shared attach gate now rejects a configured socket combined with ATTACH=0, malformed ATTACH values, and ATTACH=1 without a socket. The common stage loader propagates configuration errors before device allocation, closing the standalone-driver bypass of the supervised launchers configuration check. Unconfigured standalone direct loading remains explicit; this change does not yet convert the eager shared stage path to lazy spine/expert mapping. Host attach and stage-module tests pass, including zero allocation on conflicting configuration.

### Non-expert span derivation

The shared manifest parser now retains the sorted exact complement of routed expert ranges as spine spans, plus total spine bytes. It derives these at startup while expert ranges are sorted for overlap validation; no second manifest format or hot-path sort is introduced. Tests cover 48,384 ranges, complete expert coverage, adjacent ranges, leading/trailing gaps, and offsets above 1 TiB. These spans include headers and alignment padding and have no independent digest: the loader must validate pack identity before publication.

This is metadata groundwork, not a loaded spine. Do not map every complement span at its original pack VA: tiny padding gaps could pin nearly every expert physical chunk. The production spine must use separate compact storage or explicit non-expert tensor allocation, with source-offset lookup, while routed experts retain the sparse leased VA. Loading/export, lifetime integration, and real driver tests remain open.

Compact spine offsets now preserve source alignment modulo 256, with overflow-checked allocation sizing. SparkWeightdManifestSpineSlice translates wholly non-expert slices using binary search and rejects expert bytes, crossing slices, zero lengths, and overflow. Host tests exhaustively check all slices of a small mixed pack in addition to GLM-sized manifest coverage. The spine payload still needs loading and identity validation before any pointer publication; this metadata is not serving qualification.

### Compact spine payload loader

SparkWeightdSpineLoad now streams one open pack descriptor through a 64 KiB stack buffer, hashes the full pack with SHA256, and copies only spine spans into caller-owned compact GPU storage. It rejects insufficient capacity before copying, preserves file position via pread, checks regular-file size and nanosecond metadata stability, and returns failure on identity mismatch. The caller must quarantine partial output until success and own allocation/GPU lifetime; this primitive does not allocate or publish pointers.

The host working-set test with CUDA stubs verifies actual nonzero data placement, untouched alignment gaps, undersized capacity, unchanged fd position, and corruption of an expert byte failing full-pack SHA. The combined map/spine working-set test passes ASan/UBSan. No GPU qualification or driver integration is claimed. Full-pack disk reads for hashing remain a startup cost until authenticated independent spine checksums are available; routed expert bytes are not copied into spine GPU storage.

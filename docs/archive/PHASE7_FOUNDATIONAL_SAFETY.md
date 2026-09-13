# Phase 7: Foundational Storage, Memory, KV, and Package Safety

Phase 7 addresses the defects that could corrupt state or make a broken source package appear qualified before work continues on model execution and distributed commit.

## Runtime arena ownership

Arena allocation now returns a handle containing arena class, slot, usable bytes, and a 64-bit per-slot generation. Release validates the complete handle. A stale handle cannot release a newer allocation that reused the same address. Metadata layout is overflow checked and aligned, descriptor classes are compared after rounding, total allocation is checked against `SIZE_MAX`, and generation exhaustion fails closed.

## NVMe ownership and visibility

The NVMe tier now has an explicit write lifecycle:

```text
ReserveWrite -> device write -> CommitWrite
                         \-> AbortWrite
```

Reserved records are not readable or hash-indexed. Cancellation retains staging-buffer ownership until the device returns a terminal result. The pending-queue header has its own table region, table and staging capacities are explicit, the queue is a bounded `(deadline, FIFO-order)` min-heap, and bandwidth arithmetic saturates rather than wrapping. The public API states the single-owner-thread rule and alignment requirements.

## Required KV access

A required KV read or write now validates the complete view, sequence, logical page, mapping, and physical page. The first device failure publishes a structured error and traps the CUDA stream. Dense attention, selected-position attention, sparse scoring, sparse summary maintenance, and sparse refinement no longer silently turn a missing required page into a plausible output. Tail positions are skipped only after an explicit context-length comparison.

The canonical `LmKvViewInitialize` function requires pool size, page-table extent, sequence count, physical-page count, and an error record. Production model-package integration must use an equivalently complete view construction before GPU qualification.

## Sliding-window and GQA contracts

MiMo sliding attention now builds the exact contiguous position list on device. GQA rejects invalid head geometry instead of depending on integer truncation. This closes the immediate missing-producer and out-of-bounds contract, but the common GQA implementation still needs a split-key, KV-reuse schedule for performance.

## Numerical build policy

Global `--use_fast_math` was removed from all CUDA build surfaces. Approximate intrinsics may still be selected locally by a kernel, but that choice is now visible in source and can be included in package-specific qualification.

## Package identity

The source package is generated without Git, generated build products, Python caches, compiled objects, nested archives, or raw qualification logs. `PACKAGE_MANIFEST.json` identifies the exact source payload. `SHA256SUMS` covers that payload plus the manifest. The verifier requires exact set equality and rejects duplicate paths, unsafe paths, symlinks, compiled artifacts, qualification evidence, and nested archives in strict mode.

The deterministic archive builder emits one safe root with fixed ownership, modes, timestamps, ordering, and gzip metadata. Qualification logs and receipts are emitted outside the source archive and identify the archive by SHA-256.

## Scope boundary

This phase does not claim that K3, Qwen 3.6, DSV4 Flash, or DSV4 Pro have complete shipping executors. It does not claim that GLM 5.2 is final-link complete. It also does not close all-rank commit, final-event acknowledgement, asynchronous resident admission, CUDA 13 compilation, or Blackwell execution.

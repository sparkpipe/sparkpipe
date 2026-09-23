# Reproducing and recovering the shared-serving release

The September 22 result was 13.521 aggregate decode tokens/s across eight GLM
instances, about 1.69–1.74 tokens/s per instance. It was a 32-token smoke workload,
not an eight-family qualification or a long-running serving soak. The earlier
isolated API measurement was 12.81 tokens/s. Do not compare those denominators.

The release is `shared-serving-20260922`, source `b690c3a57005981fec1a2f611d95ec946aa3ddf7`
(tree identical to merge `54841ab38160e63dd9e15def5a8488229376f381`). Use the
published archive and verify its checksum; a directory name or `SOURCE_COMMIT`
file does not prove that its binaries have not been replaced.

## September 24 reproduction

All runs used the original release binaries, weights, 336-key input working set,
reference tokens, full pinned expert pool, hardware waits, CUDA graphs, eight
explicit lanes, B1/one input row, context 512, 128 KV pages and 2 GiB KV backing
per resident. Device reservation was 61,952 MiB per Spark. A successful run means
16 PASS receipts, 256 matching tokens, overlapping decode and 144 owned PIDs gone.

| Attempt | Host reservation | Change | Outcome |
| --- | --- | --- | --- |
| `293cd4a720bf41b5a635a95229f51326` | 16 GiB | Exact original profile | Spark4 resident 6 failed CUDA context initialization |
| `6d897973702e44f486423f1398595b31` | 32 GiB | Host budget only | Spark4 resident 1 failed the same CUDA call below the new memory limit |
| `a40c4f0eb26841539d85f34baef88f27` | 32 GiB | Explicit cache flush on Spark4 only | PASS, 13.096 aggregate tokens/s |
| `06c8acb613b4465e91f9866b5f377ebe` | 32 GiB | Ordinary restart, no cache flush | PASS, 13.331 aggregate tokens/s |

The first failing call was `cudaStreamCreateWithFlags` in resident startup, before
model execution. The kernel logged `NV_ERR_NO_MEMORY` from graphics-context
allocation. Raising the cgroup budget alone did not fix it: the second failure
peaked at 13.2 GiB under a 32 GiB limit. After an explicit cache flush, the same
binaries initialized eight residents and completed inference. This establishes
cache-state-sensitive CUDA startup, not an allreduce numerical failure or proof
that a larger host reservation is necessary.

NVIDIA documents an explicit debugging workaround for DGX Spark memory
reclamation: [known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
During an operator-controlled drained window, preserve the failed logs and run
`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` on the affected node before
retrying the same profile. This is not automatic in the replay tool or server.
A repeat failure must remain visible. CUDA startup now prints the failing stream,
rank, stage, numeric CUDA error and error text. Smoke shutdown preserves the
original failure and records cleanup failures separately.

## Replay from the one fleet controller

Use `mac@mac-studio` and the existing `~/.sparkpipe/queue` controller. This replay
starts private weightd processes and therefore requires an operator maintenance
window: drain model jobs, stop the tracked shared daemons, then untrack them.
The tool does not stop another developer's processes or create a second queue.
Restore and track the shared services when the window ends.

Download these two assets from the
[release](https://github.com/sparkpipe/sparkpipe/releases/tag/shared-serving-20260922)
into one directory using `tools/sparkpipe_github_pat.sh gh release download`:

- `sparkpipe-shared-serving-linux-arm64-cuda13.tar.gz`
- `shared-serving-qualification.tar.gz`

From the checkout containing this tool, on the controller:

```sh
python3 tools/replay_shared_serving_release.py \
  --assets /absolute/path/to/release-assets \
  --id release-replay-unique-id --replicas 8 --host-memory-mib 32768
python3 tools/spark_queue.py status --id release-replay-unique-id
```

The command verifies both archives, fetches and syncs exact release source,
stages immutable binaries and the original working-set input, reserves memory
and ports, and submits the existing `inference_smoke.py` worker. It uses the same
reference tokens; it does not regenerate expected outputs from the tested run.
Each invocation needs a new ID. It neither retries failures nor flushes caches.
For the returned attempt ID, retain `/tmp/sparkqueue-ATTEMPT/receipt.json` and
logs on all 16 hosts. Check PASS, `tokens=256`, and absence of every `owned_pids`
entry. Only the coordinator receipt has token timing. Aggregate throughput counts
tokens inside the common overlapping decode interval across the eight instances.

The original warm input has SHA256
`fbd8e8a4bff8ec99840b9bd1be5039d7ca73d305bb92ee0bafd5552bbbfcd06b`.
A full-pin run writes a larger working set into its private runtime afterward;
that output must not replace the fixed input for subsequent reproductions.

## Deployment differences found during recovery

The live GLM service used different resident and weightd binary hashes while its
shared-serving directory still declared the original source commit. Its graph
path was disabled and deployment declared 128 input rows, 16 active sequences,
and 32K context. Those settings were outside the released smoke profile.

Its first reduction submitted eight rows for one logical sequence. The wide HC
payload was `8 * 16384 * 2 = 262144` bytes, plus a 16-byte slot header, exceeding
the 256 KiB B1 mesh slot. That produces `CAPACITY_EXCEEDED` before combining.
Removing that capacity guard would corrupt the protocol. Use the explicitly
qualified profile to reproduce the release; larger B1 prefill requires a tested
chunked collective implementation, not a silently clamped configuration.

The shared daemon also had a 90 GiB device cap while the controller reserved
28.5 GiB for it. Keep the actual cap plus overhead, finite host bound and queue
reservation consistent. Verify `SHA256SUMS` before starting services and do not
copy new executables over a directory advertising an older release.

# GLM 5.2 Mooncake KV Backing Tier

This path replaces the direct-file KV swap implementation with a persistent,
stage-local Mooncake service. It is selected explicitly and never falls back to
the direct-file backend.

## Ownership

- One `mooncake_client` process owns memory and local NVMe on every Spark.
- One `mooncake_master` may run on spark0 for placement metadata. KV payloads
  remain on the Spark that owns the PP stage.
- The CUDA resident loads `libkv_mooncake.so` as a DummyClient adapter.
  The adapter connects only to the same-node real client.
- The real client survives SparkPipe resident/module releases. A SparkPipe
  restart therefore does not discard Mooncake state.

Keys include the model fingerprint, cache-layout fingerprint, rank, sequence,
and logical block. A model or layout change cannot accidentally consume old KV.
Each KV record also retains the existing rank/layer/layout header validation.

## JIT Pipeline

The CUDA resident snapshots the front of its work queue and announces three
packets by default. The builder resolves their nonresident blocks through the
authoritative WorkControl directory, de-duplicates them, and submits a batch
GET into provider-owned shared memory. The CUDA builder registers that exact
arena once at startup; DummyClient cannot register an arbitrary
`cudaHostAlloc` pointer.

Mooncake RPC and SSD work run only on adapter worker threads. `poll()` only
observes completion. The resident thread never performs a Mooncake call or a
filesystem read. When the packet reaches the front, its load callback either:

1. copies an already-prefetched record H2D and proceeds, or
2. returns `BUSY`, leaving the packet in the resident queue while I/O completes.

Stores copy D2H into separately owned pinned slots. WorkControl may evict the
GPU block, but the slots remain owned until Mooncake confirms the PUT. A later
GET cannot reuse those slots or pass the pending store.

`poll()` returns `BUSY` while the batch is in flight, `NOT_FOUND` only for a
batch id the provider does not know, and `OK` when a completion is delivered;
the batch's own outcome (including a GET miss as `NOT_FOUND`) travels in
`completion.status`, never in the poll return. `destroy()` joins workers and
can therefore block behind an in-flight PUT retry tail (bounded at twelve
seconds of `NO_AVAILABLE_HANDLE` backoff).

The lookahead is explicit and bounded to `1..8`; production defaults to `3`.
That value is a ceiling, not a fixed read-ahead depth. Packet zero always has
priority. Later packets are admitted in queue order only while their cumulative
de-duplicated nonresident blocks fit both the fixed staging batch and the
currently unallocated physical KV blocks. At zero physical headroom, no future
packet is prefetched.
The batch is bounded to 128 blocks and there are two provider batches, one GET
and one PUT. There is no per-token allocation and no per-token registration.

## Build

Build Mooncake first, then build the adapter. The target fails if Mooncake is
not provided.

```bash
make kv_mooncake \
  MOONCAKE_ROOT=/opt/mooncake \
  MOONCAKE_DEP_INCLUDE=/opt/mooncake/local/include
```

The KV provider ABI is version 2 and the CUDA builder ABI is version 17.
Rebuild and release the resident, provider, and builder together; mixed
generations fail the ABI gate.

## Local Services

Use loopback for the master and real-client RPC endpoints. The data-plane host
may use the wired Spark interface, but the real-client RPC port must not be
publicly exposed.

The recommended persistent SSD backend is `bucket_storage_backend`. It recovers
objects after real-client restart while bounding inode count by bucket count.
`offset_allocator_storage_backend` is faster but truncates its data file when
the real client restarts, so it is suitable only when cold-start cache loss is
acceptable.

Example service parameters:

```bash
MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/mnt/nvme/mooncake/glm52-rank2 \
MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend \
MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS=1 \
mooncake_client \
  --host=127.0.0.1 \
  --port=50052 \
  --master_server_address=127.0.0.1:50051 \
  --metadata_server=http://127.0.0.1:8080/metadata \
  --protocol=tcp \
  --global_segment_size='4 GB' \
  --local_buffer_size=0 \
  --threads=4 \
  --enable_offload=true
```

The resident must provide every Mooncake argument. The fingerprints are
release-generated nonzero 64-bit integers, expressed in decimal.

```text
--kv-store-module /opt/sparkpipe/lib/libkv_mooncake.so
--kv-store-service 127.0.0.1:50052
--kv-store-ipc-socket @mooncake_client_50052.sock
--kv-store-blocks 1048576
--kv-store-batch-blocks 128
--kv-store-workers 2
--kv-store-lookahead 3
--kv-store-model-fingerprint MODEL_U64
--kv-store-layout-fingerprint LAYOUT_U64
--kv-store-client-memory 4294967296
--kv-store-local-buffer 1073741824
```

Supplying both `--kv-nvme-path` and `--kv-store-module`, omitting a required
Mooncake field, or failing module/service health validation aborts startup.
The provider retries only Mooncake `NO_AVAILABLE_HANDLE` results, only for the
individual keys that failed, and only inside its worker threads. Every other
Mooncake error fails closed.

## Qualification

Before release, run these gates from a pulled PR commit:

1. host unit suite and Mooncake adapter build;
2. single-Spark PUT/GET with less than 20 GiB resident memory;
3. batch sizes 1, 32, and 128 with byte-for-byte record validation;
4. real-client restart recovery with the bucket backend;
5. forced-capacity eviction and reload;
6. PP13 correctness parity with lookahead 1 and 3;
7. queue-load test proving the resident event loop remains responsive.

The adapter, provider-owned arena, asynchronous two-block PUT/GET, and exact
128 KiB parity pass against Mooncake commit
`d65bbd9810bb8322cd37aa18e50141760e050949`. In the initial 16 MiB pressure
test, that Mooncake revision repeatedly deferred objects for disk offload but
did not create SSD files. Forced-capacity reload, restart recovery, and PP13
parity remain unqualified. Until those gates pass, this path is `NOT VALIDATED`
for production inference.

#pragma once
#include <stdint.h>
#include "sparkpipe/spark_weightd_manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

// Startup only. Manifest must be the validated immutable layout of this pack.
// Caller owns an aligned destination of capacity bytes, and must quarantine it
// until success. Failure may leave partial bytes; never publish them. Caller
// controls allocation budget and GPU-use lifetime. Reads preserve fd position.
// SHA256 validates the entire pack, so startup still reads expert bytes from
// disk, but only compact non-expert spans are copied to GPU memory.
SparkStatus SparkWeightdSpineLoad(int32_t fd,const SparkWeightdManifest *manifest,uint64_t pack_bytes,const char *sha256,void *destination,uint64_t capacity);

// Daemon side (weightd materialization only): after the daemon has hashed
// the whole pack image and matched SHA256 against `expected` with the file
// stat stable across the read, record the spine fast-path receipt with the
// DAEMON_SHA proof basis so the client's first spine load copies spans
// without re-hashing the pack (the daemon proof makes that hash strictly
// redundant; ck128-sidecar modes must NOT call this - there the client
// hash is the only SHA256 proof). Best-effort at the call site: a failure
// only costs the one-time client hash, never correctness.
SparkStatus SparkWeightdSpineReceiptRecordDaemon(int32_t fd,const char *expected,
    const uint8_t sha_digest[32]);

#ifdef __cplusplus
}
#endif

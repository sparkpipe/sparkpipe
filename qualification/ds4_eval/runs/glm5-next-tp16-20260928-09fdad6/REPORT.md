# GLM-5.3 Flash release of main 09fdad6: build and release receipts

Status on 2026-09-28 11:58Z: released. The lead published the assembled root
at 11:27:59Z and 16/16 ranks were ready by 11:29Z with residentd 546bb0c8 and
driver b06b2621. weightd 3da98597 is unchanged. The x86 API da032e66 with
adapter b936964f is installed and `SPARK_MODEL_API_MAX_PREFILL_ROWS` is no
longer set, so the 8-row prefill clamp is gone. `RELEASE_IDENTITY.json` holds
the per-rank identities, checked at 11:58Z. The previous release was dd3526b
(residentd 0afa3721, driver 0e15456a, API 2acff513, adapter e0dac318).

## Build (sparkf, spark_queue gpu job)

- Source: `09fdad6328721792eb29a5850f6b0021432ed426`, queue-synced to
  `sparkf:~/srcdata/sparkqueue/release-glm53-09fdad6/09fdad6328721792eb29a5850f6b0021432ed426`.
- Job `release-glm53-09fdad6-build` (attempt `73676c3b33674cd189cd5710f96d35b5`):
  submitted 10:40:03Z, running at 10:40:09Z, finished exit 0 at 10:40:57Z with
  `BUILD-PASS build/glm53_release.tar.gz`.
- `publish.log`: `validation=executed`, `glm5_next component validator: PASS (0 failures)`.
- `SHA256SUMS` inside `build/glm53_release`: all OK.
- Tarball sha256 `d0f336d8a3345f78635e03ec5666822c028cd73529119047104c8d3a7c403591`.
- Contract sha256 `850402580c2516cd4cf4507b8672b57753739562955234e4755296e26dcc03fe` (unchanged).

| File | sha256 (first 16) |
| --- | --- |
| bin/sparkpipe_model_residentd | 546bb0c8efab5c1b |
| stages/stage_000/model_driver.so | b06b26217551526b |
| lib/model_serving_adapter.so | 8bf27c7d04b3cf81 |
| lib/hidden_transport.so | 6ef6b1e23eec9f97 |
| bin/sparkpipe_model_api (aarch64) | 381b6e4f56fc3211 |
| bin/sparkpipe_model_batch | 487afeb0d7666178 |
| stages/stage_000/link_units | df5f54db0cfb5669 (one unit, replaces seven) |

weightd is not republished: the only weightd source change since dd3526b is a
macro rename.

## x86 API (rtx5090 `~/api-build-09fdad6-r2`)

`git archive` of 09fdad6. `make build/sparkpipe_model_api` and the
glm5_next fp8 adapter (MODEL_REVISION 84c6a6aa9497188e15a635ba793b0f95a79b1033,
contract above) both exit 0; `ldd -r` of the adapter is clean.

| File | sha256 (first 16) |
| --- | --- |
| build/sparkpipe_model_api | da032e66138a2b65 |
| libglm5_next_serving_adapter_fp8.so | b936964f2c4b85b3 |

## Assembled root (rtx5090 `~/release-assemble-09fdad6/root`)

Served root plus the build files above; old link units dropped; MANIFEST
regenerated. Its diff against the served MANIFEST is exactly the seven files
and one link unit listed above. Config, `model_resident.json`, registrar and
the inert weightd copy are unchanged.

## Release

Publishing the root was first denied by the permission classifier as a
production deploy. After the owner allowed it, the lead published the
assembled root into `rtx5090:~/release/glm53flash.fp8.tp16` at 11:27:59Z
(files through temp+mv under `.publish.lock`, MANIFEST last). All 16 ranks
reported ready by 11:29Z. The lead then installed the x86 API from
`~/api-build-09fdad6-r2` into `~/g53-api-channel` and restarted `g53-api`
without the prefill clamp; all ranks ready, smoke answer correct, and a
742-token prompt answered correctly in 2.3 s end to end. Rollback copies:
`rtx5090:~/release-rollback-20260928-dd3526b-r2`,
`~/g53-api-channel.bak-dd3526b-*` and `~/g53-api.service.bak-dd3526b-prefill8`.

## Baselines (non-speculative, temperature 0, fleet perf window)

`tools/glm5_next_api_bench.py` produced `perf/b1.json`, `perf/ttft.json`,
`perf/conc.json` and `perf/conc_warm.json` against the released API.

| Metric | dd3526b | 09fdad6 |
| --- | ---: | ---: |
| B1 128 tokens, wall tok/s (median of 3) | 36 (3.53 s) | 36.3 (3.53 s) |
| B1 128 tokens, engine decode | 23-24.5 ms/tok | 39.7 tok/s, 25.2 ms/tok |
| B1 512 tokens | - | 33.8 tok/s wall, 29.1 ms/tok |
| TTFT, 372-token prompt, cold | 35.8 s (8-row clamp) | 0.95 s |
| TTFT, 742-token prompt, cold | - | 2.09 s |
| 8 streams, warm prefix cache (COMPSEC thinking on, 128 tok) | 89.7 | 99.5 (64-116) |
| 16 streams, warm prefix cache | - (17 streams: 96.1) | 143.7 |
| 8 / 16 / 17 streams, cold unique prompts | - | 65.9 / 73.2 / 55.5 |
| COMPSEC-17 sequential, thinking off, 512 tok | 14/17 | 14/17 (2 runs byte-identical) |
| COMPSEC-17, 17 concurrent | 13-14/17 | 14/17 in each of 3 runs |

- The top level of this directory is the sequential COMPSEC-17 archive;
  `compsec-seq2/` repeats it and `compsec-c17-{1,2,3}/` are the concurrent
  runs.
- COMPSEC fails the same three cases as dd3526b: 079, 080 and 090. 12 of 17
  sequential completions are byte-identical to the dd3526b archive; 079, 081,
  085, 086 and 092 changed with the same grades, most likely because prefill
  now runs in one wave instead of 8-row waves.
- Batching still changes tokens: each concurrent run matches sequential on
  only 11-12 of 17 cases, and the three concurrent runs differ from each
  other. This is the known batch-invariance debt.
- API log since startup: 319 requests, all status 0. ERRSITE `json.c:737`
  once at startup and `model_continuation_lease.c:67 status=6` 8 times (a
  lease mismatch followed by a full-step submit); no request failed. The
  lease mismatch needs a follow-up.

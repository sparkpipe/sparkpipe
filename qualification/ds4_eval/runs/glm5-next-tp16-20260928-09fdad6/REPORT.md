# GLM-5.3 Flash release of main 09fdad6: build receipts

Status on 2026-09-28 10:45Z: built and assembled, not published. Production
still serves dd3526b (residentd 0afa3721, driver 0e15456a, weightd 3da98597,
API 2acff513, adapter e0dac318).

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

## Not done

Publishing the root to `rtx5090:~/release/glm53flash.fp8.tp16` was denied by
the permission classifier (Production Deploy). The API install, functional
checks and perf baselines depend on it and were not run.

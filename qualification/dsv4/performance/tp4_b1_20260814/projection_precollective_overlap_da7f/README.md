# DSV4 B1 projection-precollective overlap qualification

Final production decision: **rejected**. The isolated actual-shape probe was
byte-exact and repeatably positive, but the cached-prefill full-ring TP4 B1 O128
test was byte-exact and slower in all three alternating pairs. The production
implementation and its source contract were therefore removed; the serial
attention-prologue schedule remains authoritative.

This receipt tested one narrow scheduling change: while the TP projection
collective was in flight, run only work that is independent of its reduced
activation. The candidate moved CSA index-weight projection and BF16-to-F32
widening plus attention-index construction onto the existing generic CUDA
fork, then joined before any reduced-activation consumer.

No target, spine, activation, or KV precision changed. The serial path remains
unchanged. There is no compatibility fallback or feature gate.

The actual-shape probe is intentionally not an end-to-end token-rate claim. It
uses the production B1 kernel shapes for all 43 query consumers, 21 CSA index
query/weight consumers, and an 8 KiB host-signaled collective window. Across
31 alternating control/candidate pairs per process, the retained schedule was
byte-exact for every BF16, F32, and metadata output and gained:

| Run | 44 us collective | 60 us collective |
| --- | ---: | ---: |
| clean | +8.009041% | +8.080137% |
| replay | +8.263612% | +8.207596% |

That isolated result did not transfer to end-to-end decode. With one real
request, 128 cached prompt tokens, 128 generated tokens, no speculation, and
127 timed decode intervals on `spark4`-`spark7`, the serial control averaged
40.4163919596 tok/s and the overlap candidate averaged 40.1834799014 tok/s:

| Pair | Serial control | Overlap candidate | Delta |
| --- | ---: | ---: | ---: |
| 1 | 40.4345160979 | 40.1789259319 | -0.632108878% |
| 2 | 40.3724330500 | 39.9859230660 | -0.957361137% |
| 3 | 40.4422267310 | 40.3855907064 | -0.140041806% |
| **Mean** | **40.4163919596** | **40.1834799014** | **-0.576281174%** |

All O128 runs produced the exact canonical token vector with CSV-plus-newline
SHA-256 `211462f2525f73b76137ee1ce9bd4e015ad8a3fd825a7c45d38fff0488598083`.
The overlap candidate also passed the O24 gate with canonical hash
`18cd1dfc7c7fce04adf200916184cee952a185367412a1780c0d28486c42cbd8`.
See `full_ring_rejection_5e0a3b14/summary.json` and its seven raw receipts.

The standard B1 GA validator also passed against the validated PP13 stage pack
for source revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`. Its focused
precollective case compares the serial and generic-fork schedules byte for byte
using the production BF16 linear, widen, and attention-index kernels.

The tested archive was published into a fresh scratch module library with
validation executed, compiled into `model_driver.so` with explicit CUDA link
arguments, and inspected successfully. Nothing was deployed.

## Exact commands

The source snapshot was transferred from the worktree root:

```sh
ssh -o BatchMode=yes spark1 'mkdir -p /tmp/dsv4-projection-final2-20260814/source /tmp/dsv4-projection-final2-20260814/build'
rsync -a --exclude=.git --exclude=build ./ spark1:/tmp/dsv4-projection-final2-20260814/source/
```

The B1 archive was built with:

```sh
ssh -o BatchMode=yes spark1 "make -C /tmp/dsv4-projection-final2-20260814/source/modules/dsv4_resident_decode_stage archive MODULE_BATCH_VARIANT_BUCKETS= BUILD_DIRECTORY=/tmp/dsv4-projection-final2-20260814/build/objects MODULE_ARCHIVE=/tmp/dsv4-projection-final2-20260814/build/libdsv4_resident_decode_stage_b1.a MODULE_COMPILE_FLAGS='-DSPARK_DSV4_MODULE_BUILD=1 -DSPARK_BATCH_BUCKET=1u' CUDA_ARCH=sm_121a"
```

The stage pack was inspected with:

```sh
ssh -o BatchMode=yes spark1 'python3 /tmp/dsv4-projection-final2-20260814/source/tools/dsv4_stagepack.py --verify-pack /home/spark1/sparkdata/dsv4_flash.fp8.pp13.b16/packs/dsv4_flash_stage.spstage'
```

The configuration hash was
`056da5f4e95d94bcb879407fdd63db5f5e67a58443e9433c5d3c5d403bc77adc`.
The validator and publisher used this exact environment:

```text
SPARK_MODULE_BATCH_BUCKET=1
SPARK_DSV4_STAGE_PACK_PATH=/home/spark1/sparkdata/dsv4_flash.fp8.pp13.b16/packs/dsv4_flash_stage.spstage
SPARK_DSV4_STAGE_COUNT=13
SPARK_DSV4_STAGE_INDEX=1
SPARK_DSV4_STAGE_FIRST_LAYER=3
SPARK_DSV4_STAGE_LAYER_COUNT=3
SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES=1
SPARK_DSV4_STAGE_MAX_SEQ=4736
SPARK_DSV4_STAGE_PIPELINE_SLOTS=1
SPARK_DSV4_STAGE_PHYSICAL_PAGES=128
SPARK_DSV4_STAGE_LOGICAL_PAGES=128
SPARK_DSV4_STAGE_MTP=0
SPARK_DSV4_STAGE_GRAPHS=0
SPARK_DSV4_REFERENCE_MANIFEST_SHA256=9ef837975bc4ddbd3cf0de0ea19c59c2c4c8a3750a8b8f302a19df0e09f39fa3
SPARK_DSV4_CUDA_VALIDATOR_SHA256=bd1b0f2fad0c5342b4c2a3f6467582427151ea3958c5813286b40098e79b52a5
SPARK_DSV4_REFERENCE_VERIFIER_SHA256=f66953937bceecf5b48f7cacec617a0a84856d95ff56d9e9e49c994863e289ee
CUDA_ARCH=sm_121a
```

With that environment exported, the validator command was:

```sh
/tmp/dsv4-projection-final2-20260814/source/modules/dsv4_resident_decode_stage/validation/validate_dsv4_resident_decode_stage_cuda.sh 056da5f4e95d94bcb879407fdd63db5f5e67a58443e9433c5d3c5d403bc77adc /tmp/dsv4-projection-final2-20260814/build/libdsv4_resident_decode_stage_b1.a
```

The fresh scratch publish command was:

```sh
/tmp/dsv4-projection-final2-20260814/source/build/sparkpipe_module_publish --library /tmp/dsv4-projection-final2-20260814/library --module spark.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16.h4096.l43.e256.k6.ga0731.b1.v4 --target cuda.sm121.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16 --link-unit /tmp/dsv4-projection-final2-20260814/build/libdsv4_resident_decode_stage_b1.a --recipe dsv4.resident_decode_stage.sm_121a.gpu.config_056da5f4e95d94bcb879407fdd63db5f5e67a58443e9433c5d3c5d403bc77adc.v1 --initialize SparkDsv4ResidentDecodeStageInitialize --execute SparkDsv4ResidentDecodeStageExecute --admit SparkDsv4ResidentDecodeStageAdmit --snapshot SparkDsv4ResidentDecodeStageSnapshot --destroy SparkDsv4ResidentDecodeStageDestroy --validator /tmp/dsv4-projection-final2-20260814/source/modules/dsv4_resident_decode_stage/validation/validate_dsv4_resident_decode_stage_cuda.sh --validator-arg 056da5f4e95d94bcb879407fdd63db5f5e67a58443e9433c5d3c5d403bc77adc
```

The successful driver compile used explicit CUDA and C++ runtime link args:

```sh
/tmp/dsv4-projection-final2-20260814/source/build/sparkpipe_model_compile --model /tmp/dsv4-projection-final2-20260814/source/examples/model_descriptions/dsv4_resident_decode_stage_firmware_b1.json --stage dsv4_resident_decode_stage --library /tmp/dsv4-projection-final2-20260814/library --output /tmp/dsv4-projection-final2-20260814/driver2 --include /tmp/dsv4-projection-final2-20260814/source/include --cc cc --cc-arg -L/usr/local/cuda/lib64 --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -ldl --cc-arg -lm --cc-arg -pthread
```

The descriptor inspection command was:

```sh
/tmp/dsv4-projection-final2-20260814/source/build/sparkpipe_driver_inspect /tmp/dsv4-projection-final2-20260814/driver2/model_driver.so cuda.sm121.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16
```

See `qualification.json` for artifact hashes, `clean.log` and `replay.log`
for raw timings, `validator.log` for the GA pass, and the retained module
record and compiled manifest for package identity.

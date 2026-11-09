# #1901 on lenovo7: CUDA build and gates (2026-11-09)

The builds and gates ran on lenovo7, the build host (one GB10, sm_121a, nvcc from `/usr/local/cuda/bin`). They compare main's head `fded9e032` with this PR's head `ce30c377b`. main's head is in the PR's history, so the comparison covers exactly the PR's changes. The gates ran under a LENOVO HOLD from 21:14 to 21:52 KST. The same files stay on lenovo7 in `~/build-rp/`.

## Build

```sh
git archive fded9e032 | ssh lenovo7 'mkdir -p ~/build-rp/main-fded9e032-src && cd ~/build-rp/main-fded9e032-src && tar -xf -'
git archive ce30c377b | ssh lenovo7 'mkdir -p ~/build-rp/pr1901-ce30c377b-src && cd ~/build-rp/pr1901-ce30c377b-src && tar -xf -'
ssh lenovo7 '~/build-rp/build.sh main-fded9e032-src & ~/build-rp/build.sh pr1901-ce30c377b-src & wait'
```

`build.sh` runs the release build's own build-host command, `make -j8 $NODE $SO` (runbook section 4.1):
- `$NODE` is residentd, `weightd_warm`, the RDMA transport, `sparkpipe_weightd`, `weightd_receipt`, `weightdctl` and `sparkpipe_model_batch`;
- `$SO` is every lane adapter that `tools/lane_config.py release-row` names in `deployment/lanes.json`. `targets-main.txt` and `targets-pr1901.txt` list the 20 adapters.

Both builds exited 0: main in 121 s, this PR in 108 s.

## Gates

1. **SASS of every kernel in the 20 adapters:** `python3 sass_diff.py main-fded9e032-src pr1901-ce30c377b-src <adapters>`. The script hashes each kernel's full `cuobjdump -sass` text, instruction encodings included.

   `sass-summary.txt`: 4,564 kernels, 4,544 byte-identical. The 20 that differ are one kernel, `LmSampleRowsKernel<1024>`, once per adapter, because the sampler formulas moved into `inference/kernels/kernel_math.cuh`. Every head, attention, expert and KV kernel is identical.
2. **The changed paths, on both builds:** `changed-paths.tsv` has each test's last line, and whether its whole output is the same on both builds.

   | test | covers | result |
   |---|---|---|
   | `test_sample_threads_cuda` | the sampler kernel on the GPU | passes, byte-identical across the builds |
   | `test_sample_host` | the sampler's draws | passes, identical; run without `python3 -I`, since it imports `host_cuda_compiler` from the repository |
   | `test_kv_snapshot_cuda` | KV save and restore on the GPU | passes, 0 bitwise differences; only its timing field differs |
   | `test_kv_shard_host` | KV sharding | passes, identical |
   | `test_kv_failure_host` | KV failure handling | passes, identical |
   | `test_device_contract_cuda` | the CUDA backend; on this PR, memory and queue adoption | passes, identical |
   | `test_head_certified_cuda` | the output head | passes, identical |
3. **Every `tests/*_cuda.py`:** `cuda-sweep.sh` runs each test on both builds back to back, 90 s per run, starting no test after 22:08.

   `cuda-sweep.tsv`: 55 tests, 110 runs from 21:27:48 to 21:51:37, every exit status 0. 41 outputs are byte-identical across the builds. The other 14 differ only in timings: every result field (error bounds, peaks, bit-equality checks) is equal.

## Why no serving test ran on one GB10

`deployment/lanes.json` has 20 lanes, all on the 16-rank mesh, and neither main nor this PR has a CUDA serving adapter below TP16. So no model can be served on one GB10. A dev-lane trial on Fleet 1 is not allowed either:
- the runbook forbids a second weightd on a production node (section 3);
- attaching dev lanes to the production weightd is not documented;
- this PR moves weightd's sources and adopts the weightd pool.

The TP16 serving path is exercised by the release kit's qualify step after merge (runbook section 4): it activates the build, captures and compares against the old one, runs COMPSEC and the perf cases, and rolls back on failure.

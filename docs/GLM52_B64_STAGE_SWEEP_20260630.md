# GLM52 B64 Stage Sweep - 2026-06-30

This records the current B64 routed-stage measurements after rebuilding the B12x resident MoE packs with the corrected pack ABI.

The sweep used one warmup run and two measured runs per stage. The selected value is the best measured run, not the warmup. The cold warmup and slower measured attempts remain in the remote logs named below.

## Result

| stage | host | stage ms | per-layer ms | filled-pipeline tok/s | submissions |
| --- | --- | ---: | ---: | ---: | ---: |
| 11:8 | spark2 | 58.703583 | 7.337948 | 1090.223 | 8 |
| 19:8 | spark2 | 58.214112 | 7.276764 | 1099.390 | 8 |
| 27:8 | spark0 | 61.400895 | 7.675112 | 1042.330 | 8 |
| 35:8 | spark1 | 60.243680 | 7.530460 | 1062.352 | 8 |
| 43:8 | spark7 | 70.719456 | 8.839932 | 904.984 | 8 |
| 51:8 | spark2 | 60.860415 | 7.607552 | 1051.587 | 8 |
| 59:8 | spark2 | 59.550304 | 7.443788 | 1074.722 | 8 |
| 67:8 | spark2 | 59.514080 | 7.439260 | 1075.376 | 8 |
| 75:3 | spark2 | 21.913504 | 7.304501 | 2920.574 | 3 |

Slowest measured B64 stage:

```text
stage=43:8
stage_ms=70.719456
filled_pipeline_tok_s=904.984
```

For a filled B64 pipeline, this stage currently sets the aggregate routed-pipeline ceiling at about 905 tok/s before transport and scheduler overhead.

## Notes

- The B12x packs for 27:8, 35:8, and 43:8 were generated on spark0, spark1, and spark7 respectively, using local `.spb12x` output and control-plane-only SSH.
- The `.spb12x` bodies were not copied over SSH.
- The generated AOT metadata originally contained Spark2 absolute runtime paths. Worker validation needed `tvm_ffi_flags.mk` and `runtime_link_args.txt` relocated to the worker user path.
- CUDA graph replay was disabled for this sweep.
- The current execution still reports 8 submissions for each 8-layer stage; true stage-slice graph capture remains a separate optimization target.

## Evidence Paths

```text
spark2:/tmp/sparkpipe_glm52_live_main_20260630/build/glm52_stage_bucket_sweep_0011_0018/glm52_stage_bucket_sweep.tsv
spark2:/tmp/sparkpipe_glm52_live_main_20260630/build/glm52_stage_bucket_sweep_0019_0026/glm52_stage_bucket_sweep.tsv
spark0:/tmp/sparkpipe_glm52_live_main_20260630/build/glm52_stage_bucket_sweep/glm52_stage_bucket_sweep.tsv
spark1:/tmp/sparkpipe_glm52_live_main_20260630/build/glm52_stage_bucket_sweep/glm52_stage_bucket_sweep.tsv
spark7:/tmp/sparkpipe_glm52_live_main_20260630/build/glm52_stage_bucket_sweep/glm52_stage_bucket_sweep.tsv
spark2:/tmp/sparkpipe_glm52_live_main_20260630/build/glm52_stage_bucket_sweep_0059_0066/glm52_stage_bucket_sweep.tsv
spark2:/tmp/sparkpipe_glm52_live_main_20260630/build/glm52_stage_bucket_sweep_0067_0074/glm52_stage_bucket_sweep.tsv
spark2:/tmp/sparkpipe_glm52_live_main_20260630/build/glm52_stage_bucket_sweep_0075_0077_v3/glm52_stage_bucket_sweep.tsv
```

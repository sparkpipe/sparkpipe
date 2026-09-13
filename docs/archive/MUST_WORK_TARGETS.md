# Mandatory model targets

`model_contracts/must_work_targets.json` is the release scope. A model family is
not considered supported merely because its module compiles or its source tree
exists.

The mandatory targets are:

1. Kimi K3 with MXFP4 routed-expert weights, BF16 routed activations and BF16
   non-expert tensors.
2. GLM 5.2 with FP8 E4M3 routed-expert weights and BF16 activations/non-expert
   tensors.
3. Qwen 3.6 27B in BF16.
4. DeepSeek V4 Flash in its checkpoint-native FP4-expert / FP8-non-expert mixed
   precision.
5. DeepSeek V4 Pro in its checkpoint-native FP4-expert / FP8-non-expert mixed
   precision.

Every target requires exact CUDA 13 compilation for `sm_121a`, numerical
qualification on the target Spark, queue/reconnect fault tests, and retained
performance receipts before `production_ready` may become true.

The first deployment fabric is one rail. Debugging may use a direct-cable ring;
the first switched deployment uses one MikroTik 100 Gbit/s switch. A second
switch and dual-rail scheduling are a future topology and must not be selected
silently by current packages.

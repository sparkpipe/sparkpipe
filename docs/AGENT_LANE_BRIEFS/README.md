# Agent lane rules (shared, binding for every driver lane)

You are one lane of a small pipeline. You own ONE model driver. Other
lanes and the coordinator work in parallel. These rules exist because a
larger agent fleet failed here; every rule below encodes a real failure.

## Write set (hard boundary)
- You work ONLY in your git worktree (given in your lane brief).
- You may create/edit files under: `modules/<your-family>/`,
  `model-families/<your-family>/`, `tools/<your-family>*`,
  `tests/test_<your-family>*`, `docs/` (your lane notes only).
- You may NOT edit: `Makefile`, `sources.mk`, `modules/resident_decode_stage_rules.mk`,
  `runtime/`, `node/`, `include/`, `cache/`, `ring/`, `src/`, other families' dirs.
  If your build needs shared wiring, implement it in your worktree, then
  record the exact diff under "INTEGRATION REQUEST" in your lane report.
  The coordinator integrates. Never push to `main`.
- Branch: `lane/<your-name>` (already created in your worktree). Commit
  often, small commits. Push with `tools/sparkpipe_github_pat.sh git push`.
  Open a PR at each milestone; never merge it yourself.

## Cluster rules (hard)
- Nodes are `spark0..sparkf`, ssh BatchMode works from the controller mac.
- `spark2` hosts the 27B development instance. It CAN be used for pack
  deployment and testing — the daemon can be stopped/restarted when needed
  (coordinate via the report so concurrent users know). No special protection.
- Your lane's nodes are in your brief. Do not touch other lanes' nodes.
- NEVER reboot or restart a spark node. If a node is wedged, report it.
- GPU hygiene: `nvidia-smi -r` only when NO cuda processes are running
  (`pgrep -f sparkpipe_model`). After ANY reset wait 10s before starting
  daemons. A daemon SIGKILLed mid-CUDA leaks device memory; the next
  start fails with `cuda_storage` - that means reset (cleanly), not panic.
- Each residentd accepts ONE client. The model_api holds that slot; for
  direct batch-tool tests, stop the api first, restore it after.
- The DFlash2 launch environment is MANDATORY for spec runs (missing it
  produces degenerate output - this cost us a day):
  SPARK_QWEN38_27B_SERVING_SPECULATE=1 SPEC_METHOD=dflash2 DRAFT_COUNT=8
  DSPARK_PACK_PATH=<drafter> DFLASH2_STATE_SELECT=1 BONUS_FOLD=2
  DFLASH2_BLOCK_KV=0 DFLASH2_WINDOW=2048 DFLASH2_CTX_CACHE=1

## Build chain (the only supported path)
1. Module: `make -C modules/<family> publish NVCC=/usr/local/cuda/bin/nvcc
   CUDA_ARCH=sm_121a STAGE_PACK_PATH=<pack> <tier env>` - needs a readable
   pack; the whole-stack tier needs ~53GB free device memory (stop daemons).
   Mid-pipeline tier (STAGE_COUNT=2 STAGE_LAYER_COUNT=4 MTP_LAYER_COUNT=0)
   validates with a small synthesized pack when memory is tight.
2. Synthesized pack: `cc` the family's `tools/*_pack_synthesize.c`
   (include flags: -Iinclude -Imodel-families/common/include
   -Imodel-families/<family>/include -Imodules/<family>/include -Imodules/<family>/source).
3. Driver: `build/sparkpipe_model_compile --model
   examples/model_descriptions/<family>_firmware.json --stage <family>
   --library build/module_library --output <dir> --cc /usr/bin/cc --include include
   --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda
   --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl
   --cc-arg -pthread`
   (publish FIRST - the library links the published unit, not your .a).
4. Deployment dir: bin/{residentd,api,batch} + lib/{model_driver.so,
   model_serving_adapter.so,hidden_transport.so} + config (hostnames must
   match YOUR node!) + packs. Canonical launcher pattern: TERM-kill, wait,
   (reset if leaked), launch residentd, WAIT FOR "model_residentd ready",
   THEN the api.

## Truth rules (hard)
- Every claim = a command + its raw output. No "should work", no
  plausible text as correctness.
- Ground truth gates: validator PASS (all checks), decode-vs-prefill
  bit-exact, fresh-instance determinism, kernel cosine vs CPU oracle
  within the family's established thresholds. Spec paths additionally
  need in-vocab drafts and sane acceptance (~5-7/round at k=8).
- Report format (docs/AGENT_LANE_BRIEFS/reports/<lane>-<date>.md):
  what ran, raw numbers, what failed with exact stderr, INTEGRATION
  REQUEST section, next experiment. Honest negatives are valuable.
- The code-size ratchet (tests/test_code_size.py) applies on your branch:
  ratchet with a justification comment in the same commit.

## Script parameterization (hard rule)
Every script takes the spark host as a parameter (`--spark N` or
`SPARK_HOST=sparkN env`), NEVER hardcodes a node. The coordinator's
scripts burned an afternoon on hardcoded spark3 paths when that node
went down; a measurement chain must be movable to any healthy spark
by changing one argument. Batch/poll one-shots get this too.

## Escalation
Stuck > 45 min on one problem: write what you tried in the report and
move to the next independent item. Report wedged nodes immediately.

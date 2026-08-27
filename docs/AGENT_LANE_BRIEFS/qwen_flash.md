# Lane brief: Qwen 3.8 Flash (qwen4_exp) - PILOT LANE

Worktree: /tmp/lane-qwenflash
Branch: lane/qwen-flash (already created from main)
Your nodes: spark4, spark5, spark6, spark7 (TP4). spark3 is the
coordinator's bench - do not run daemons there. spark2 = prod, untouchable.

## Mission
Bring Qwen 3.8 Flash to its first validated, serving-ready build, reusing
the qwen38_max module family (same 3:1 linear:full layer pattern, same
512-expert/top-10 MoE shape - this is a sibling, not a new architecture).

## Source (verified complete, warm+cold dual-archived)
/mnt/model-warm/qwen3.8-flash-next (and cold /mnt/cold-raid6/models/qwen3.8-flash-next)
- arch: Qwen4ExpForConditionalGeneration (model_type qwen4_exp)
- text: hidden 2560, 48 layers (3 linear : 1 full), 24 heads / 2 KV,
  linear attn 16k/48v heads x128, MoE 512 experts top-10, moe_int 640,
  shared 640, MTP 1 layer, max_pos 262144, vocab 248320, bf16 unquantized,
  vision encoder 27L/1152 (out of scope for lane 1)
- 149 files, 131 shards, 336 GiB. There is a vision encoder - SKIP it
  (text stack only for this lane).

## Milestone ladder (each rung = verifiable exit, commit + report)
M1 Contract freeze: model_contracts/qwen4_flash_authoritative.json with
   pinned file sha256s (config.json + index + shard digests from the warm
   copy; hash a strided sample of shards fully and all small files, note
   the sampling). Test: `python3 tools/verify_source_archive.py` style
   check or a fresh small verifier in your write set.
M2 Geometry header: model-families/qwen4_flash/include/.../spark_qwen4_flash_model.h
   + runtime contract, generated from the config (mirror qwen38_max's
   header shape). Conformance test binds header to config.json.
M3 Module skeleton: modules/qwen4_flash_resident_decode_stage/ copied from
   qwen38_max's module, re-parameterized (expect: dims, layer counts, head
   counts, vocab, moe_int 640 vs 2048, 48 vs 92 layers). Compile clean
   on YOUR node (nvcc sm_121a, mid-pipeline tier, synthesized pack).
   Exit: validator PASS on the 4-layer slice (bit-exact decode-vs-prefill,
   determinism) with SYNTHETIC weights.
M4 Packer: tools/qwen4_flash_stagepack.py from the warm source -> MX FP8
   format-6 packs, TP4 rank split (mirror tools/qwen38_stagepack.py +
   spark_pack_common.py). Exit: pack verifier passes per rank; sizes
   sane (~42GB/rank class).
M5 Whole-stack validation on spark4 (TP_STANDALONE rank0, FP8 packs):
   validator PASS with real weights incl. MTP draft in-vocab.
M6 TP4 parity smoke on spark4-7: 4-rank launch, short B1 correctness run,
   record wall time + token stream hash.
M7 Serving: deployment on your nodes, no-spec first (record stream hash
   of the canonical 128-token real-text prompt - embed YOUR canonical
   prompt in the report), then MTP-1 spec.
Do NOT chase the perf targets yet - correctness and the data trail first.

## Known traps (paid for in blood)
- The adapter source defaults to TP4; TP1 builds need
  -DSPARK_QWEN38_27B_SERVING_TP_DEGREE=1u-style flags (see lane rules).
- Pack file names/extensions must match module expectations exactly.
- model_compile links the LIBRARY: publish before compile, every time.
- Node hostnames in config/ must match the node you run on.
- Kernels: qwen38_max's linear-attention + MoE kernels are your starting
  point - do not rewrite them; re-parameterize. If a kernel assumes
  dimensions that do not divide cleanly, STOP and report (INTEGRATION
  REQUEST) rather than forking the kernel.

## Report
docs/AGENT_LANE_BRIEFS/reports/qwen-flash-<date>.md after every milestone.

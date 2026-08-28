# Qwen Max sharded-pack sprint — lane report (2026-08-28)

Worktree /tmp/lane-qwen38max-shard, branch lane/qwen38max-shard.
Build/validate node spark7, pack build node spark0 (spark5's ceph client
is STILL wedged - 5 MB/s bulk vs 515/887 MB/s on spark0/7, evidence in
the qwen-flash lane report and re-measured this session). Scope updated
mid-sprint per coordinator: the AMD Quark MXFP4 checkpoint replaces our
own requantizer (S5) and the FP8-parity experiment (S6).

## Headline

The qwen38_max module was UNBUILDABLE at sprint start (the .cu failed
nvcc with 28 errors - an incomplete 27b rename - and its launcher symbols
never matched the module's declarations, so it could never link either).
It now builds clean on sm_121a, has the family's first validation
harness (full PASS on spark7, including the MXFP4 expert path and the
TP4 rank-local kernel geometry), accepts format-v2 sharded packs from
its own loader on REAL checkpoint data, and the packer/verifier pair
round-trips real ranks byte-exactly. The 16-rank build waits only on the
AMD checkpoint download (in progress, numbers below).

## Scope change: MXFP4 source = AMD Quark, not our requantizer (approved)

No official Qwen 4-bit exists (Qwen ships BF16 + FP8 only). Verified from
the HF API and the checkpoint's own config.json/safetensors headers:

  * amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 (5976 downloads, rev
    13dc9676): quantizes ONLY the routed experts; the exclude list keeps
    every GDN, attention, router, shared-expert, MTP and lm_head tensor
    BF16 - exactly our contract ladder. Payload U8 [rows, cols/2] +
    E8M0 scale U8 [rows, cols/32] group-32: BYTE-IDENTICAL to our kernel
    decode contract (SparkLmDecodeE2m1 x SparkLmDecodeE8m0). Consumed
    verbatim, no requantization, 1.248 TiB total.
  * NVFP4 was evaluated and DEFERRED (coordinator approved): I compiled
    the kind::mxf4nvf4.scale_vec::4X ... ue4m3 m16n8k64 atom for sm_121a
    on spark7 - ptxas ACCEPTS it (nvcc exit 0, runs) - but zero in-tree
    kernels decode ue4m3 group-16 scales and the compressed-tensors NVFP4
    recipes carry two-level scales; ~500-800 new lines vs MXFP4's zero.
    The v2 pack format is codec-agnostic (per-entry weight_format +
    scale_group_size), so NVFP4 lands later as a drop-in precision
    upgrade with zero wire change. My probe is the enabling evidence.
  * dsv4 precedent checked: both dsv4 contracts pin routed_expert_
    weight_codec "mxfp4_e2m1" (the nvfp4 codec id exists in the table,
    no contract uses it).

## Milestones

### S1 validation harness - DONE, full PASS on spark7

modules/qwen38_max_resident_decode_stage/validation/ (ported from the
27b template at Max geometry). All numbers from the PASS run
(smid-pipeline tier: STAGE_COUNT=2, stage 0, layers 4, 58.4 GiB
synthesized v2 pack with MXFP4 experts, MAX_ACTIVE_SEQUENCES=8):

```
check=decay_gate       elements=0 relative_l2=7.10e-08 cosine=1
check=write_gate       elements=0 relative_l2=3.97e-08 cosine=1
check=gdn_step_output  elements=0 relative_l2=1.64e-03 cosine=0.999998660
check=gdn_step_state   elements=0 relative_l2=6.38e-08 cosine=1        <- bit-carry-equal
check=gated_norm       elements=0 relative_l2=1.65e-03 cosine=0.999998637
check=attn_decode      elements=0 relative_l2=1.71e-03 cosine=0.999998540
check=gdn_chunk_output elements=0 relative_l2=1.66e-03 cosine=0.999998623
check=gdn_chunk_state  elements=0 relative_l2=6.73e-07 cosine=1
check=moe_mxfp4        elements=0 relative_l2=1.39e-03 cosine=0.999999034
check=gdn_step_tp4     elements=0 relative_l2=1.65e-03 cosine=0.999998644
check=module_admit_snapshot admit=fail_closed snapshot=fail_closed
check=module_determinism bit_exact=1
qwen38_max_validation PASS
```

The moe_mxfp4 check is the production codec's kernel gate: route build +
B1 W13/W2 + pair reduce over a tp4 rank-0 shard view (128 experts),
compared against a CPU oracle that replicates the B1 kernels' EXACT
semantics - FP8-E4M3/UE8M0 128-block activation QDQ (both stages),
bf16-RNE intermediate rounds, the 10.0 SwiGLU limit clamps, and fmaf
dots. Getting there found two real bugs: the 27b template's TRUNCATING
bf16 converter (outputs need round-to-nearest-even; it was a 2x-ulp
systematic bias -> moe cosine 0.99977 -> 0.999999 after the fix) and
oracles must consume bf16-rounded inputs (gdn_step_state 1.3e-3 ->
6e-8). Decode-vs-prefill at module tier is impossible by design: the
module fails prefill frames closed; the equivalent mathematical gate
(chunk kernels vs recurrence oracle, bit-carry-equal state) runs at the
kernel tier instead (gdn_chunk_* checks).

### S2 wire format v2 - DONE

128-byte header (tp_degree, tp_rank; FORMAT_VERSION 2), the shard-axis
table (expert rows / head rows / composed GDN q|k|v channels / input
columns / head columns / replicated), a feasibility gate (degrees 1, 2,
4), and rank-sharded resolution where tp_degree 1 reproduces v1 shapes
exactly. Files: spark_qwen38_max_stagepack_format.h, packer,
verifier, synthesizer.

### S3 sharded loader + MXFP4 acceptance - DONE, proven on real data

ValidateEntry resolves per-rank shapes; the expert natural-format
exception now admits MXFP4-E2M1 group-32 (the AMD packs) alongside the
vendor FP8 and synth BF16; non-expert tensors are BF16-natural, which
the AMD source satisfies verbatim (the coordinator's question: yes, the
existing BF16 non-expert path works unchanged). LoadPack pins the
header's tp fields against the configured rank and names them on
mismatch. Real-data proof (vendor FP8 source, tp4 rank1, 1-layer,
10.10 GiB, 20 entries): the module at SPARK_QWEN38_MAX_STAGE_TP=4/1
loads the pack completely - zero pack_geometry_mismatch /
pack_entry_invalid / coverage errors - and stops at the collective
memory-mode probe, the correct fail-closed without peers:

```
qwen38_stage tp_probe_memory_mode_failed status=13
qwen38_stage initialize_failed status=13
```

### S4 kernel TP indexing - DONE, validated

Attention prepare/decode were already TP-aware; the GDN kernels
(conv update, decay/beta, step, gated norm, all six chunk kernels) now
derive rank-local geometry from tp_degree (whole-head cuts, GVA 8:1 and
per-head dims invariant); pools, conv tails, KV-cache strides and every
head-shaped slot buffer are rank-local; the GDN/attention output
projections all-reduce their input-sharded partials (the audit's second
per-layer collective); the MoE route builds tile prefixes at the B1
tile widths for MXFP4 and the previously DEAD MXFP4 launchers
(SparkQwen38MaxLaunchFusedExpertW13Act/ExpertDown) are rank-aware
(full-width pointer shift vs sharded base 0) and dispatched from RunMoe;
the FP8 grouped launchers accept both pack generations. Kernel-tier
proof: gdn_step_tp4 (rank-0 local heads vs the oracle on the same slice)
passes at the tp1 check's thresholds, and moe_mxfp4 runs the whole
rank-0 shard path. Cross-rank numerics need 4 live ranks = the fleet
window. Found and fixed en route: ExpertDown's rows parameter must be
the SOURCE row count (an admitted B1 decode shape), not rows*topk.

### Packer + verifier - DONE, round-trip PASS on real data

tools/qwen38_stagepack.py: v2 header, --tp-degree/--tp-rank with the C
header's exact shard axes (including the composed GDN q|k|v channel cut
and F32 A_log/dt_bias column slices widened from BF16), and
--source-format quark-mxfp4 (E2M1+E8M0 bytes copied VERBATIM - the
checkpoint layout IS the kernel layout) alongside the vendor FP8 path.
tools/qwen38_pack_verify.py streams (rank packs are ~80 GiB): header
geometry + tp fields, directory shape/format/bounds against the packer's
own table, exact coverage, byte-exact sampled traces recomputed through
the packer's copy plans, and an MXFP4 dequant sanity probe. Real-data
round trip (FP8 source, tp4 rank1, 1 layer):

```
qwen38_stagepack slice=0+1 tp=4/1 tensors=20 file_gib=10.10
trace kind=0 layer=global slice_bytes=4068474880 receipt=verified   (embedding)
trace kind=6 layer=0 expert=128 bytes=16777216 receipt=verified     (W1)
... 14/14 traces byte-exact ...
PASS q38max_fp8_l0_tp4r1.qwen38sp: header geometry, 20 directory entries
     (tp 4/1), 14 byte-traced samples receipt=verified
```

Two packer bugs found by that round trip, both fixed: the tp>1 copy
path fell through to a row cut for REPLICATED tensors (embedding read
off the end of the file), and _full_shape widened replicated kinds.

### S5/S6 - DROPPED (coordinator-approved)

The AMD quantization is consumed verbatim (no requantizer), and there is
no FP8 twin of those weights to parity against; the validator's exact
dequant oracle (moe_mxfp4) is the stronger and sufficient gate.

### S7 16-rank build/deploy - IN PROGRESS (download-bound)

tools/qwen38max_build_ranks.sh (build+verify the 16 packs on the hub,
one PP stage x one TP rank each, then deploy one pack per node: sparkN
gets world rank N). The AMD checkpoint download to
/mnt/model-warm/packbuild/qwen38max/amd-mxfp4 is the long pole:
HF's CDN throttles long-lived connections (fresh stream 44 MB/s while
24-hour-old streams crawl at ~2 MB/s), so the downloader cycles
connections (--speed-limit 4M --speed-time 20, resume -C -), split
across spark7+spark0 by shard parity. Measured 88-89 MB/s aggregate at
24 workers; 1.248 TiB total. At last measurement: 182 GiB down, ETA
~4h. spark5 excluded (ceph wedge).

## Honest negatives

  * Module-tier decode-vs-prefill: impossible (module is decode-only by
    design); the chunk-vs-recurrence kernel check carries the
    mathematical claim instead. Prefill execution is serving-adapter
    work, out of this sprint's 7 files.
  * Module-tier validation at tp>1 needs the live 4-rank collective -
    single-node runs stop at the collective probe BY DESIGN (loader
    acceptance still proven, see S3). The fleet window owns the
    cross-rank numerics gate.
  * The whole-stack TP1 synth tier would need a 296 GiB pack (92 layers
    x full-width MXFP4 experts) - does not fit any node; the module tier
    validated at the mid-pipeline tier, and whole-stack waits for the
    real ~80 GiB rank packs.
  * The moe_mxfp4 thresholds (2e-2/0.999) are the family's bf16-output
    class bounds; measured 1.39e-3/0.999999.
  * spark5 ceph client still wedged (coordinator item, standing from the
    flash lane's report).

## INTEGRATION REQUEST

  1. (coordinator) spark5 ceph wedge - still 5 MB/s bulk reads; this
     lane measured it again this session. Packs build on spark0 instead.
  2. (coordinator) The AMD Quark checkpoint download (~1.25 TiB to
     /mnt/model-warm/packbuild/qwen38max/amd-mxfp4) is the sprint's long
     pole; if another egress path exists (mirror, direct trunk), it
     would compress the remaining hours.
  3. (driver follow-up) The qwen38_max serving stack (residentd/api
     deployment configs + driver compile for the 16-node TP4xPP4
     topology) is not in this lane's 7-file scope; the packs, loader and
     kernels are ready for it.
  4. (platform follow-up, approved) NVFP4 precision upgrade: probe
    result on file (ue4m3 mxf4nvf4 MMA assembles for sm_121a); needs
    ue4m3-group16 scale decode in the expert kernels, a pack codec id,
    and two-level scale handling.

## Reproduce

  * Validator (spark7): env as in modules/qwen38_max_resident_decode_
    stage/validation/validate_qwen38_max_resident_decode_stage_cuda.sh,
    pack from modules/qwen38_max_resident_decode_stage/tools/
    qwen38_max_pack_synthesize.c --first-layer 0 --layer-count 4
    (defaults to MXFP4 experts, v2 header).
  * Rank build+verify: tools/qwen38max_build_ranks.sh --spark spark0
    (16 packs, verifies each against the checkpoint).
  * Loader acceptance single-node: SPARK_QWEN38_MAX_STAGE_TP_DEGREE=4
    RANK=r with the rank pack; expect tp_probe_memory_mode_failed (no
    peers) AFTER a clean pack load.

## Commits (this lane session)

  12b9459 format v2 + sharded loader + TP kernels (S2-S4)
  66f3bed complete the 27b->max cuda rename (module was uncompilable)
  6b669eb validation harness (S1) - PASS on spark7
  eac55a4 packer v2 + quark-mxfp4 codec + verifier
  eef435a packer replicated-tensor fixes, verified on real data
  (this commit) lane report; build/deploy results appended below when
  the download lands.

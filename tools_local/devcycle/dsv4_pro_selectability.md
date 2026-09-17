# DSV4 Pro topology and expert-codec selection

DSV4 Pro ships as TP4xPP4 with MXFP4-E2M1 experts (the deployed default).
Both axes are selectable at build/deploy time; this doc records the knobs,
what is implemented, and what remains.

## Topology selection

The serving topology is a compile-time flag on the serving adapter
(-DSPARK_DSV4_SERVING_TOPOLOGY), with the adapter id and geometry deriving
from it in spark_dsv4_serving_adapter.c:

| Topology flag | Adapter id (pro) | Geometry | Status |
| --- | --- | --- | --- |
| 404 | spark.dsv4.pro.serving-adapter.tp4-pp4.v1 | TP4 x PP4, 16 ranks, 16/15/15/15 layers | DEPLOYED (current) |
| 404 (B1) | tp4-pp4 id | bucket-1 variant | builds exist |

Topologies 16 and 4 are flash-only; a pro build with either flag is a
compile-time #error in spark_dsv4_serving_adapter.c.

The root Makefile builds the pro adapter targets per topology
(DSV4_PRO_TP4_PP4_SERVING_TOPOLOGY_FLAGS etc.). The pack side is already
topology-parameterized: tools/dsv4_tp4_pp4_stagepacks.py --model pro drives the
sharder with any TP degree and PP stage count (TP4xPP4 defaults). A new
topology deployment needs: adapter variant build + rank packs from the full
pack + a deployment spec with the matching adapter id + stage JSON
(peer ports, rails, cuda_graph_count_by_pp_stage = 3L+1) + a fleet registry
entry.

## Expert weight codec selection

The expert codec is a compile-time constant of the module
(SPARK_DSV4_PRO_EXPERT_WEIGHT_CODEC) surfaced through Makefile.pro:

    make -C modules/dsv4_resident_decode_stage -f Makefile.pro         PRO_EXPERT_CODEC=mxfp4|fp8_e4m3 ...

- mxfp4 (default): module id ...linear_fp8.expert_mxfp4.kv_bf16... - the
  deployed artifact set, unchanged.
- fp8_e4m3: adds -DSPARK_DSV4_PRO_EXPERT_CODEC_FP8_E4M3=1 (the pro model
  header switches the codec) and the module id becomes
  ...expert_fp8.kv_bf16...

Pack side: tools/dsv4_pro_expert_requant.py converts any pack's routed
expert records from MXFP4-E2M1 to FP8-E4M3 (per-128 F32 scales, the same
convention as the FP8 linears; header expert codec id 7 -> 5). The
conversion is lossy by construction (the checkpoint was trained into the
MXFP4 grid), so FP8-expert packs are VARIANT artifacts, never the baseline.

## Status matrix

| Piece | mxfp4 (default) | fp8_e4m3 variant |
| --- | --- | --- |
| Model header codec define | done | done (build flag) |
| Module id / target derivation | done | done (Makefile.pro) |
| Packer/pack production | done (authoritative) | done (requant tool) |
| Pack header codec id | 7 | 5 |
| Module expert kernels | done (SM121 MXFP4 fused W13/W2) | MISSING - the SM121 expert kernels are MXFP4-hardwired (SparkLmSm121LoadMxf4B, E8M0 scales); an FP8-E4M3 fused expert kernel variant is the remaining kernel work. Until then FP8 packs cannot run. |
| Sharder (rank packs) | done (TP4/PP4) | MISSING - the sharder's FP4 column slicing assumes 2-elements-per-byte; an FP8-aware sharder variant is needed before rank-level FP8 packs. |
| GPU validation | done (val4 + valtail, real token) | not possible until the kernel variant exists |

## Verified conversion result (round 11)

tools/dsv4_pro_expert_requant.py converted the val4 slice pack on spark3:
/home/spark3/pro-repo/dsv4_pro.val4.fp8experts.spstage - 108,151,344,456
bytes (source 57,382,440,264), header codecs (fp8_e4m3, fp8_e4m3, bf16),
12 expert records (4 layers x W1/W2/W3) converted to FP8-E4M3 with per-128
F32 scales, and all 107 non-expert records byte-identical to the source
(verified by direct payload comparison). The same tool converts the full
pack or any rank slice; wall time ~6 min per expert record (~70 min for the
full 865 GB pack).

## Registry and naming

The fleet registry entry "dsv4-pro" describes the TP4xPP4/mxfp4 deployment.
A variant deployment adds a new entry (e.g. dsv4-pro-fp8) with its own
runtime_root/dataset name, pack set, and module target - the fleet_swap
mechanism treats it as a separate selectable model.

# Stagepack naming standard

This is the naming authority for pack directories and runtime roots (operator
directive 2026-09-04). `tools/stagepack_naming.json` is the machine-readable
source for the token lists and the legacy map; where this file and the JSON
differ, the JSON wins and this file is wrong. The deployed GLM 5.3 Flash root
follows the scheme: `glm53flash.fp8.tp16`
(`deployment/glm5_next_tp16/model_resident.json`).

## The scheme

    arm      = <model>.<quant>.<topo>
    model    = a token from "models" in the JSON
    quant    = a token from "quants": bf16 | fp8 | nvfp4 | nvfp4a16 | mxfp4 | iq1m
               bf16 means ALL-BF16 (native bf16 experts); a bf16-spine +
               fp8-expert package is named fp8. Never carry one quant's data
               under another quant's name (operator ruling 2026-09-05, 4969a89).
    topo     = a token from "topologies": tp4 | tp8 | tp16 | tp4pp4
    layout   = ~/sparkdata/<arm>/packs/
    pack     = <arm>.rank<h>.sp   (h one hex digit 0-f: rank h lives on spark<h>)
    sidecars = <pack>.sha256   (placement digest, the receipt chain)
               <pack>.ck128    (load digest, optional)
               <pack>.experts  (lazy manifest: layer/expert/offset/bytes/ck128)

Model tokens in the JSON on 2026-09-28: glm53flash (glm5_next, GLM 5.3 Flash),
glm53full, dsv4flash, dsv4pro, dsv41flash, qwen3flash (Qwen 3.8 Flash Next),
qwen27b, qwenmax, k3, hy4, ling, lingfin, mimo26pro, mimo26flash. Registered
arms: `mimo26pro.mxfp4.tp8`, `mimo26flash.mxfp4.tp4`.

weightd does not read the arm name. Its identity is the model and revision
strings (`SPARK_WEIGHTD_IDENTITY_MODEL` and `SPARK_WEIGHTD_IDENTITY_REVISION`,
or the driver's defaults), the pack SHA-256 (`SPARK_WEIGHTD_PACK_SHA256`), the
topology and the geometry fingerprint (`runtime/spark_weightd_attach.c`).

## Roots in use that do not conform (2026-09-28)

Each needs a token or topology decision in the JSON before a rename:

| Root | Deviation | Source |
| --- | --- | --- |
| `laguna-s-2.1.bf16.tp8pp2` | no laguna token; `tp8pp2` is not a listed topology; packs `laguna_stage.tp8.pp2.stage<S>.rank<R>.lgsp` | `tools/lag_t1_launch.sh`, `tools/devcycle/laguna_warm_receipt.sh` |
| `muse.tp16.bf16` | topology before quant; no muse token | `tools/muse_gen_deployment.py` |
| `gemma4_31b.bf16.tp16` | no gemma4 token; packs `gemma4_31b_tp16_rank<hex>_stage0.gemma4sp` | `tools/gemma4_tp16_gen_deployment.py`, `tools/gemma4_build_release.sh` |
| `qwen38max.tp16` | no quant; the registered token is `qwenmax` | `tools/qwen38max_multidev_run_family.sh` |
| `minimax.text.bf16.tp4` | no minimax token; packs `.rank<r>.mntx` | `tools/spark_station_profile.py` |

The JSON's `rename_executed` note lists further trees still under legacy names.

## Legacy map

`legacy_map` in the JSON maps each legacy directory to its canonical arm, and
`tools/rename_stagepacks.sh <node>` applies the same table on a node. The
fleet rename (07f0cc2) ran on all 16 nodes. On 2026-09-05 the bf16 ruling
(4969a89) removed the fp8-content arm that the rename had placed under
`glm53flash.bf16.tp16` and gave that name to the true BF16 build. The map
therefore sends `glm5_next.tp16` (FP8 content) to `glm53flash.fp8.tp16`, and
the script refuses to rename into a directory that already exists instead of
nesting one tree inside another.

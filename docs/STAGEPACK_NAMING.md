# Stagepack naming standard (operator directive 2026-09-04)

One scheme, no silly differences. Every generated stagepack tree conforms;
legacy names map through `tools/stagepack_naming.json` until the physical
rename lands (stagepack dev task: one mv script + receipt re-issue).

## The scheme

    model registry token (one per FAMILY, fixed set):
        glm53flash   (= glm5_next / glm-5.3-flash; NOT glm53full)
        glm53full
        dsv4flash    (NOT dsv4pro)
        dsv4pro
        qwen3flash   (qwen3.8-flash-next)
        qwen27b      (the 27B family)
        qwenmax
        k3
        hy4

    arm      = <model>.<quant>.<topo>
    quant    = bf16 | fp8 | nvfp4 | nvfp4a16 | mxfp4 | iq1m
    topo     = tp4 | tp8 | tp16 | tp4pp4
    layout   = ~/sparkdata/<arm>/packs/
    pack     = <arm>.rank<h>.sp      (h single hex digit 0-9a-f: rank h
                                      lives on spark<h> - the pack name
                                      states its own node)
    sidecars = <pack>.sha256  (placement digest - the receipt chain)
               <pack>.ck128    (load digest, optional)
               <pack>.experts  (lazy-manifest: layer/expert/offset/bytes/ck128)

    weightd identity: model = <arm>, revision = <checkpoint-tag>.

## Legacy -> canonical map

Entries marked QUANT? carry no quant token in the legacy name; stagepack
dev fills the true value at rename time (nothing is silently guessed).

    hy4 fp8 safetensors (model-fp8-tp16-rank-NN.safetensors) -> hy4.fp8.tp16
    qwen4_flash.tp4            -> qwen3flash.QUANT?.tp4
    qwenflash.tp8              -> qwen3flash.QUANT?.tp8
    qwenflash.tp8.fp8          -> qwen3flash.fp8.tp8
    qwenflash.tp8.nvfp4        -> qwen3flash.nvfp4.tp8
    qwenflash.tp4pp4           -> qwen3flash.QUANT?.tp4pp4
    qwenflash.tp4pp4.fp8       -> qwen3flash.fp8.tp4pp4
    dsv4flash.tp16             -> dsv4flash.QUANT?.tp16
    dsv4flash.tp4pp4           -> dsv4flash.QUANT?.tp4pp4
    dsv4_pro.tp16              -> dsv4pro.QUANT?.tp16
    dsv4_pro.tp4pp4            -> dsv4pro.QUANT?.tp4pp4
    glm5_next.tp16             -> glm53flash.QUANT?.tp16
    glm5_next.tp4pp4           -> glm53flash.QUANT?.tp4pp4
    glm5_next.tp8.fp8          -> glm53flash.fp8.tp8
    glm5_next.bf16.tp16        -> glm53flash.bf16.tp16
    glm53full.bf16.tp4pp4      -> glm53full.bf16.tp4pp4
    glm53full.fp8.tp4pp4       -> glm53full.fp8.tp4pp4
    glm53full.nvfp4.tp4pp4     -> glm53full.nvfp4.tp4pp4
    k3.mxfp4.tp4pp4            -> k3.mxfp4.tp4pp4
    qwen27b.tp4pp4             -> qwen27b.QUANT?.tp4pp4
    qwen38_27b.tp4pp4          -> qwen27b.QUANT?.tp4pp4   (or a distinct
                                 checkpoint - stagepack dev confirms)
    qwen38-27b.nvfp4a16.tp4    -> qwen27b.nvfp4a16.tp4

Also banned after the rename: `packs_v4/` (always `packs/`), mixed rank
formats (`rank-00`, `rank00`, `rank0`, `rank00`-style decimals -> single
hex digit `rank<h>`), mixed extensions (`.g5nsp`, `.pack`, `.safetensors`
-> `.sp`).
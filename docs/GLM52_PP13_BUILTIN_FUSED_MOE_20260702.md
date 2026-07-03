# GLM52 PP13 Built-in Fused MoE Router Path

This note records the PP13 fused MoE performance pass for the fixed six-layer production plan.

## Scope

The completed path is intentionally narrow and connected:

- exact PP13 six-layer stage plans can advertise a built-in fused MoE capability;
- production exact plans no longer need an external fused-MoE callback merely to be accepted;
- routed layers in those stages must have production-fast router plans and production expert plans;
- the built-in exact PP13 launcher marks routed layers as requiring the fast MoE router path;
- a fused-stage MoE plan without either an external implementation or the built-in backing capability is rejected.

## Kernel path

The production path keeps the router projection on the fast linear-plan backend instead of replacing it with a scalar per-expert dot-product kernel.

For NVFP4/B12x routed layers:

```text
router tensor-core projection -> router logits
router logits + score bias -> FlashInfer B12x fused top-k / expert / combine
```

For FP8 routed layers:

```text
router tensor-core projection -> router logits
router logits + score bias -> FP8 MoE plan fused top-k / expert / combine
```

The FP8 path no longer runs Sparkpipe's standalone top-k kernel before calling the FP8 MoE plan. The FP8 plan already requires the fused-top-k capability, so Sparkpipe now passes router logits to that production plan and lets the plan own top-k, dispatch, expert compute, and combine.

The standalone Sparkpipe top-k kernel remains only for router-only validation/debug layers, not for production routed FP8 MoE execution.

## Validation behavior

When the built-in fused MoE capability is requested, every routed layer in the slice must validate:

- production-fast router linear plan;
- B12x FlashInfer dispatch/expert plan for NVFP4 routed layers;
- FP8 expert tensor-core plan with fused top-k capability for FP8 routed layers.

Dense or attention-only layers are accepted because they do not execute routed MoE work.

## Expected impact

This pass removes the avoidable standalone top-k launch from FP8 routed MoE execution and makes the PP13 built-in fused MoE contract real for both 4-bit and 8-bit routed variants. The exact speedup must be measured on Spark2 because this container cannot compile or run the CUDA target.

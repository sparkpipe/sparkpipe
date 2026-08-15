# GLM 5.2 deployment slot + measurement rules

Fleet coexistence (per fleet architecture): ports and measurement
exclusivity are the constraints, not memory.

## Port block registry

| Model/slot | Hosts | Control | Collective | Transport ctrl |
|---|---|---|---|---|
| DSV4 Flash TP4 (always-on) | spark4-7 | 18480 | 62620-3 | 59700 |
| Qwen 27B PP16 (always-on) | spark0-3 | 17480 | 61620-3 | 58700 |
| **GLM 5.2 slot A** | spark8-f | **19480** | **63620-3** | **60700** |
| Big slot B (full-16) | all | 20480 | 64620-3 | 61700 |

GLM 5.2 is a "big model": TP4 work uses 4 hosts from the spark8-f slot A
block; TP16/PP16 topologies take the full fleet (slot B) in their slice.

## Measurement cleanliness

- Dev-active coexistence (residentd up, idle) is fine anytime.
- A measured B1 window is EXCLUSIVE on its hosts: stop sibling residentds
  there, measure, restore. Promotion is <60 s.
- GLM B1 receipts are only valid when spark8-f (or whichever slice is used)
  carries no other residentd traffic during the window.

## Current state (build phase)

- Build/data host: sparke (checkpoint, pack build, module compile) - no
  residentd, no measurement contention.
- Pack build: full 78-layer glm52sp, streaming, two-source
  (spine = zai-org/GLM-5.2 BF16 master b4734de4; experts =
  zai-org/GLM-5.2-FP8), output /home/sparke/srcdata/glm52_full.fp8.glms52sp.
- Next: model driver compile (fp8 firmware), residentd deploy with the
  slot-A port block, layer-0 gates, then the B1 baseline.

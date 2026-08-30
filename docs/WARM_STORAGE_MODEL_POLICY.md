# Warm-Storage Model Variant Policy (operator directive 2026-08-30)

THE RULE: warm storage holds EXACTLY the variants we support —
everything wanted is present, nothing unwanted remains. Each variant
in warm storage gets a stagepack + model descriptor.

## The audit (warm inventory vs support intent)

### KEEP — serving variants (stagepacks built/building)

| warm dir | size | serving role | provenance |
|---|---|---|---|
| glm-5.3-bf16 | 1.4T | glm5.3-full 16-bit arm (packs placed) | OFFICIAL zai-org |
| glm-5.3-fp8 | 704G | glm5.3-full FP8 serving arm (placed) | OFFICIAL zai-org |
| glm-5.3-nvfp4-radixark | 433G | glm5.3-full hybrid arm (placed) | vetted community (RadixArk) |
| glm-5.3-flash | 306G | glm5.3-flash FP8 serving set (placed) | OFFICIAL zai-org |
| glm-5.3-flash-bf16-official | 599G | flash BF16 reference | OFFICIAL zai-org |
| glm-5.3-flash-nvfp4-redhatai | 185G | flash hybrid (per policy table) | vetted community (RedHatAI) |
| qwen3.8-max-nvfp4-radixark-bf16-spine | 1.4T | qwen-max 4-bit (the ONLY fitting form; PP16 placed) | vetted community |
| qwen3.8-flash-next | 336G | qwen-flash BF16 serving (TP8 placed) | official lineage |
| qwen3.8-27b-fp8 | 29G | 27B serving (TP4 placed, verified) | OFFICIAL Qwen FP8 |
| deepseek-v4-flash-0731 | 156G | dsv4-flash (TP16 placed) | official preview |
| deepseek-v4-pro-0813-ga | 832G | dsv4-pro GA (TP16 pending ext) | OFFICIAL GA |
| kimi-k3 | 1.5T | k3 MXFP4 (warm-build in flight) | OFFICIAL moonshot (native MXFP4 QAT) |

### KEEP — drafter/speculator corpus (the tournament portfolio)

kimi-k3-dspark-redhatai/inferact/radixark, kimi-k3-dflash-*,
qwen3.8-*-dspark/dflash, glm-5.3-flash-dflash2, deepseek-v4-flash-
dflash-redhatai (all ≤9G each; the speculation portfolio's sources).

### REMOVE — superseded or unusable (candidates, operator-confirm)

| warm dir | size | why |
|---|---|---|
| glm-5.2-fp8 | 704G | glm5.2 deprecated (kernel donor only; donor role needs repo, not warm weights) |
| qwen3.8-2.4t-a95b-fp8 | 2.3T | qwen-max FP8 = 156G/rank — BREAKS the 110GiB law; superseded by the nvfp4 arm |
| kimi-k3-nvfp4-redhatai | 1.5T | duplicate of the official moonshot MXFP4 (NVIDIA's nvidia/Kimi-K3-NVFP4 is the vetted alt if wanted) |
| deepseek-v4-pro-0813-nvfp4-jarrelscy | 877G | community alt of the GA (keep ONE pro variant) |
| deepseek-v4-flash-official-fp8-mixed | 275G | the quality pick FAILED the packer contract (tensor inventory); revisit = contract work, not a kept source |
| deepseek-v4-flash-official-fp4-fp8 | 149G | superseded by 0731 |
| deepseek-v4-flash-0731-nvfp4-mjpansa | 164G | community alt; 0731 is the packer-native source |
| qwen3.8-27b-nvfp4a16-bf16-spine | 29G | needs module vertical; FP8 arm serves |
| qwen3.8-27b-nvfp4-* (inferact/radixark/bf16-lmhead/redhatai/plain) | ~120G | five community 27B variants — one spine MAY be kept for the future vertical; the rest superseded by OFFICIAL fp8 |
| qwen3.8-flash-next-fp8 / -nvfp4-radixark | 173G+126G | flash BF16 serves; the quant alts violate no-need rule until quality demands |

REMOVAL TOTAL (operator-confirm): ~6.7T of 19T (packbuild/ 3.4T is
ACTIVE state — the k3 base; cleaned in Phase B).

## Internet check — useful variants we may be MISSING

- **nvidia/Kimi-K3-NVFP4** (~1.4T): the vetted NVIDIA conversion of
  K3's experts — a HIGHER-accuracy 4-bit than MXFP4 (E4M3 block
  scales). Candidate to REPLACE kimi-k3 (1.5T) as the k3 4-bit arm.
- **nvidia/DeepSeek-V4-Pro-NVFP4 / -Flash-nvfp4-DSSpark**: official
  NVIDIA NVFP4 conversions — DSSpark variant aligns with our DSpark
  speculator work. Candidates if the GA pro path stalls.
- **Official GLM-5.3 NVFP4**: does NOT exist (search-confirmed; only
  community RadixArk/ressl/etc.) — our radixark pick stands as the
  best available.
- **nvidia/Qwen3.6-27B-NVFP4**: prior-gen; skip unless wanted.

## Stagepack + descriptor status per kept variant

glm5.3-full ×3 (placed) · glm5.3-flash ×3 (fp8 placed; bf16/hybrid
packs TODO) · qwen-max PP16 (placed) · qwen-flash TP8 (placed) ·
27B TP4 (placed+verified) · dsv4-flash TP16 (placed) · dsv4-pro
TP16 (Phase A unit 2) · k3 (Phase A unit 1, in flight).
Model descriptors: model_contracts/*_authoritative.json exist for
glm5.3 (full+flash), dsv4 (flash+pro), k3, qwen38 families; qwen-
flash/27B/max descriptors verified through their packers' contract
flags. GAP: none blocking; each kept variant maps to a contract.

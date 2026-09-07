# dsv4 triple-drafter analysis: why the farm port stopped, and the shape that works

Date: 2026-09-04. Trigger: "do all 3 combined" (MTP + DSPARK + DFlash on one target).

## The inventory answer

- glm5.3 flash: MTP (layer 45, byte-exact on hardware) + DFlash2 (farm). No DSPARK weights exist
  in the checkpoint — a triple needs a trained dspark head that does not exist.
- qwen38_27b: three methods (mtp/dspark/dflash2) but dspark and dflash2 are the same drafter
  weights — correlated, does not deliver source diversity.
- dsv4 flash: the only real triple — MTP (3 layers) + DSPARK (block 5, markov rank 256), both in
  the target pack and live, plus the independent redhatai DFlash v1 checkpoint, now ported to the
  farm (SPEC_DFLASH1 = bit 64, 2.63 ms/block, 3.4 GiB, gates green).

## What the dsv4 MTP/DSPARK actually is (module source + pack manifest)

The dsv4 draft is not a small drafter: it is the full dsv4 target architecture stacked 3 layers —
HC sinkhorn stream mixing, fp8-e4m3 block-scaled LoRA projections, attention with learned sink
over a 128-token ring, a full 256-expert FP4 MoE per draft layer, then the Markov chain head.
"MTP" and "DSPARK" are two extractions from ONE 3-layer checkpoint (block-row argmax vs
markov-chained), not two weight sets. A blind torch port of that stack (no oracle on the 5090,
the full lm_head is not even in the TP16 rank-0 pack — only an 8192-row shard) would be a wrong
drafter, so it was deliberately not attempted. Manifest tool + full tensor dump:
/home/spec/draft_service/dsv4_pack_manifest.{py,txt} on the 5090.

## The shape that works

MTP/DSPARK stay on-target (the dsv4 module already runs them — that IS the live dspark loop).
The farm supplies DFlash1 (and the tap-free sources) remotely. Composition v1 is per-cycle source
selection (chain verify unchanged); tree-merged verify is the later phase.

## The tap-spec conflict (protocol consequence)

dflash1 wants 5x16384 (hyper-connection residual, layers 3/13/23/32/42); dspark/MTP want 3x4096
(layers 40/41/42). One tap payload per request cannot serve both — irrelevant while MTP/DSPARK
stay on-target, but if a second tap-conditioned remote source ever lands, the protocol needs
per-source tap sections (a v2 change, not hacked in).

#!/usr/bin/env bash
# dsv4flash.tp16 release build (dsv4flash-lane lane): publish the module
# with GPU validation and assemble the coherent generation.
#
# Publish shape = the pp13 3-layer slice (13/1/3+3) against the regenerated
# validation slice — the configuration every prior dsv4 module publish used
# (family validators on main are TP1/B1-only by construction; rank packs
# cannot take the TP1 slice position checks). DSV4_VALIDATION_PACK is
# produced by the chained val-slice job (tools/dsv4flash_val_slice_run.sh).
set -euo pipefail
export DSV4_VALIDATION_PACK=/home/spark2/lane-dsv4flash-build/build/dsv4_val3/dsv4_flash_val3.spstage
export DSV4_PUBLISH_STAGE_COUNT=13
export DSV4_PUBLISH_STAGE_INDEX=1
export DSV4_PUBLISH_STAGE_FIRST_LAYER=3
export DSV4_PUBLISH_STAGE_LAYER_COUNT=3
export SPARK_DSV4_STAGE_TP_DEGREE=1
export SPARK_DSV4_STAGE_TP_RANK=0
bash tools/dsv4flash_build_release.sh 2>&1

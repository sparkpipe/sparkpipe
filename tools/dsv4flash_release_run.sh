#!/usr/bin/env bash
# dsv4flash.tp16 release build (dsv4flash-lane lane): publish the module
# with GPU validation (TP16 shape via the validator TP envs) and assemble
# the coherent generation. DSV4_VALIDATION_PACK = the placed rank1 pack on
# this node (sha-verified placement bytes). Run from the CHECKOUT ROOT
# (queue --cwd points there; this file lives at tools/ in the checkout).
set -euo pipefail
export DSV4_VALIDATION_PACK=/home/spark2/sparkdata/dsv4flash.tp16/packs/dsv4flash.tp16.rank1.spstage
export DSV4_PUBLISH_STAGE_COUNT=16
export DSV4_PUBLISH_STAGE_INDEX=1
export SPARK_DSV4_STAGE_TP_DEGREE=16
export SPARK_DSV4_STAGE_TP_RANK=1
bash tools/dsv4flash_build_release.sh 2>&1

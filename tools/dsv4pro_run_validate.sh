#!/bin/bash
cd /home/spark6/dsv4pro_checkout
export PATH=/usr/local/cuda/bin:$PATH
PACK=$HOME/sparkdata/dsv4_pro.tp4pp4/packs/dsv4_pro.tp4_pp4.rank00.spstage
make -C modules/dsv4_resident_decode_stage -f Makefile.pro validate \
  PRO_EXPERT_CODEC=mxfp4 PRO_KV_CODEC=bf16 \
  STAGE_PACK_PATH="$PACK" \
  STAGE_COUNT=4 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=16 \
  MAX_ACTIVE_SEQUENCES=1024 MAX_SEQUENCE_POSITIONS=33024 \
  PIPELINE_SLOT_COUNT=13 PHYSICAL_PAGE_CAPACITY=1024 \
  LOGICAL_PAGE_CAPACITY=16384 MTP_LAYER_COUNT=0 CUDA_GRAPH_COUNT=0
echo VALIDATE-RC=$?

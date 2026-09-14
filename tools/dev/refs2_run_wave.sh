#!/bin/bash
set -u
cd ~/refs2
run() {
  fam=$1; ckpt=$2; runid=$3
  env OMP_NUM_THREADS=20 python3 tools/t1_reference_decoder.py \
    --family "$fam" --checkpoint "$ckpt" \
    --header "model-families/$fam/include/sparkpipe/llm_defines.h" \
    --prompts "qualification/t1_reference/$fam/prompts.json" \
    --output "$HOME/refs2_out_$runid" > "$HOME/refs2_${fam}_$runid.log" 2>&1
  echo "$fam $runid exit=$?"
}
CK=/home/spark1/refs2_ckpt
for fam in ling gemma4 laguna; do
  case $fam in
    ling) ckpt=$CK/ling-3.0-flash;;
    gemma4) ckpt=$CK/gemma-4-31b-it;;
    laguna) ckpt=$CK/laguna-s-2.1;;
  esac
  run "$fam" "$ckpt" a
  run "$fam" "$ckpt" b
done
echo ALL_DONE

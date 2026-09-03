#!/bin/sh
cd ~/hy4-fp8-packs || exit 1
pgrep -f hy4_fp8_pack_verify >/dev/null && echo verify-running && exit 0
rm -f verify0.log verify7.log
python3 hy4_fp8_pack_verify.py --checkpoint /mnt/model-warm/hy4-preview-fp8-official --pack-dir . --rank 0 --samples 4 > verify0.log 2>&1
python3 hy4_fp8_pack_verify.py --checkpoint /mnt/model-warm/hy4-preview-fp8-official --pack-dir . --rank 7 --samples 4 > verify7.log 2>&1
echo VERIFY_DONE >> verify0.log

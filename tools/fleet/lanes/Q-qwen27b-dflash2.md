GOAL: promote the BEST qwen38-27B driver (origin/main has modules/qwen38_27b_resident_decode_stage + API-level improvements; keep that API work) and add the new SGLang DFlash2 improvement (BF16 lm_head claim, ~50 tok/s).
Verify precision parity with our mixed/FP8 pack before porting. Hosts spark2/spark3. dsh clone reproduced spark2 results on spark3 - mine ~/dsh.sparkpipe for its receipts first.

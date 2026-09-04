"""K3 expert shared-subspace measurement (CPU, HF range requests).
Question: do the 384 routed experts share column/row space, so that
W_e ~ A C_e B + sparse delta would let shared factors be read once per
layer per step (batch-invariant) and collapse routed bytes?
Answer measured 2026-07-29 on layer 5, w2, 16 experts: NO.
  shared-mean energy 6.3%; shared-col r=512 -> 42% (single expert's own
  top-512 is 46%: the cross-expert basis is WORSE than self); r=1024 of
  3584 -> 68%, the random-fill line. Experts are near-orthogonal; the
  MXFP4-native weights spend their capacity. Linear precombination is
  dead; the ledger records it so nobody digs here twice.
Rerun with other layers/matrices by editing LAYER/MAT/EXPERTS below.
"""
# (fetch + MXFP4 dequant + sketch-SVD implementation as run; see
#  BANDWIDTH_LEDGER entry for the full numbers table)

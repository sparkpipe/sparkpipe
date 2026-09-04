diff --git a/vllm/v1/worker/gpu/sample/gumbel.py b/vllm/v1/worker/gpu/sample/gumbel.py
index 74ee8abb2c4c..190307d5e75a 100644
--- a/vllm/v1/worker/gpu/sample/gumbel.py
+++ b/vllm/v1/worker/gpu/sample/gumbel.py
@@ -223,6 +223,11 @@ def gumbel_sample(
     output_processed_logits_col: torch.Tensor | None = None,
     use_fp64: bool = False,
 ) -> torch.Tensor:
+    # Enforce contiguity on non-strided input tensors
+    expanded_idx_mapping = expanded_idx_mapping.contiguous()
+    pos = pos.contiguous()
+    if output_processed_logits_col is not None:
+        output_processed_logits_col = output_processed_logits_col.contiguous()
     num_tokens, vocab_size = logits.shape
     BLOCK_SIZE = 1024
     num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
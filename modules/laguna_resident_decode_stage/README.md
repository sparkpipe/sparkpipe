# laguna_resident_decode_stage

CUDA resident decode-stage firmware and serving adapter for Laguna, with BF16
linear weights and a routed-expert codec chosen at build time with
`EXPERT_CODEC` (bf16, int6, int7, int8, fp8, nvfp4 or mxfp4).

## Implementation notes

### Lazy-attach manifest check (`SparkLagunaManifestCheck`)

- Every routed-expert entry in the stage pack directory
  (`SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_GATE_UP` and
  `SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_DOWN`) must have `weight_codec` equal
  to the codec this build was compiled for. That codec is
  `LAGUNA_EXPERT_WEIGHT_CODEC`, which the `Makefile` sets from
  `EXPERT_CODEC` to the matching `SparkWeightCodec` value: bf16 1, int6 2,
  int7 3, int8 4, fp8 5, nvfp4 6, mxfp4 7. A pack built for another codec
  fails with `SPARK_STATUS_UNSUPPORTED`.
- Plane 0 is the payload and plane 1 is the scale plane. Quantized packs
  carry a scale plane. BF16 expert packs written by `tools/laguna_stagepack.py`
  have `scale_bytes` 0, and the check skips plane 1 for them, because
  `SparkLagunaManifestPlane` rejects a plane with zero bytes.
- For each plane it walks, `SparkLagunaManifestPlane` requires, for every
  expert, a range of kind `tensor_kind * 2 + plane` in the manifest group of
  that (layer, expert), with offset `plane offset + expert * per-expert
  bytes` and length `plane bytes / group_count`.
- The total number of ranges walked must equal the manifest's
  `range_count`; otherwise the check returns `SPARK_STATUS_SCHEMA_ERROR`.

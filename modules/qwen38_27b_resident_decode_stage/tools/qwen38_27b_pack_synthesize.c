/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "spark_qwen38_27b_stagepack_format.h"

/*
 * Synthetic qwen38_27b stage pack writer, slice-aware.
 *
 * Emits every tensor one pipeline STAGE will demand, at the geometry the
 * module was compiled for, with reproducible pseudo-random contents: the
 * loader, the shape table, the slice arithmetic and the layer walk run end
 * to end against it. This file is the family configuration — stage-pack
 * types, the dense-FFN per-layer kind inventory, and the quantization
 * policy; the machinery is the shared generator core.
 *
 * Family facts (recorded, not re-derived): the shape resolver takes the
 * stage count (pinned 1u here, as pasted); quantization is
 * quantizable-gated (only shape.quantizable kinds go MXFP4, everything
 * else keeps its natural format); the MTP tail carries 4 globals.
 */

#define SPARK_SYNTH_QWEN_TEMPLATE 1
#define SPARK_SYNTH_TOOL_NAME "qwen38_27b_pack_synthesize"
#define SPARK_SYNTH_MAX_TENSORS 1024u
#define SPARK_SYNTH_CHUNK_BYTES (8u * 1024u * 1024u)
#define SPARK_SYNTH_ALIGN_UNIT SPARK_QWEN38_27B_STAGEPACK_PAYLOAD_ALIGNMENT
#define SPARK_SYNTH_ENTRY_T SparkQwen38_27bStagePackEntry
#define SPARK_SYNTH_HEADER_T SparkQwen38_27bStagePackHeader
#define SPARK_SYNTH_SHAPE_T SparkQwen38_27bStagePackTensorShape
#define SPARK_SYNTH_HEADER_BYTES SPARK_QWEN38_27B_STAGEPACK_HEADER_BYTES
#define SPARK_SYNTH_ENTRY_BYTES SPARK_QWEN38_27B_STAGEPACK_ENTRY_BYTES
#define SPARK_SYNTH_RESOLVE_SHAPE(kind, layer, is_global, shape) \
	(SparkQwen38_27bStagePackResolvedShape((kind),(layer),(is_global),1u,(shape)) < 0)
#define SPARK_SYNTH_SELECT_FORMAT(quantize, shape) \
	(((quantize) != 0u && (shape).quantizable != 0u) ? SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 : (shape).natural_format)
#define SPARK_SYNTH_SCALE_GROUP_SIZE(format) \
	((format) == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 ? 32u : 0u)
#define SPARK_SYNTH_PAYLOAD_KIND(entry) ((entry)->weight_format)
#define SPARK_SYNTH_PACKED_FORMAT SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1
#define SPARK_SYNTH_PACKED_NAN_MASK 0u
#define SPARK_SYNTH_F32_FORMAT SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32
#define SPARK_SYNTH_PAYLOAD_BYTES(format, rows, columns) \
	SparkQwen38_27bStagePackPayloadBytes((format),(rows),(columns))
#define SPARK_SYNTH_SCALE_BYTES(format, rows, columns) \
	SparkQwen38_27bStagePackScaleBytes((format),(rows),(columns))
#define SPARK_SYNTH_FILL_SCALE(entry, buffer, bytes, state) \
	SparkSynthFillScaleBytes((buffer),(bytes),(state))
#define SPARK_SYNTH_MTP_LAYER SPARK_QWEN38_27B_STAGEPACK_MTP_LAYER
#define SPARK_SYNTH_GLOBAL_LAYER SPARK_QWEN38_27B_STAGEPACK_GLOBAL_LAYER
#define SPARK_SYNTH_TENSOR_EMBEDDING SPARK_QWEN38_27B_STAGEPACK_TENSOR_EMBEDDING
#define SPARK_SYNTH_TENSOR_FINAL_NORM SPARK_QWEN38_27B_STAGEPACK_TENSOR_FINAL_NORM
#define SPARK_SYNTH_TENSOR_LM_HEAD SPARK_QWEN38_27B_STAGEPACK_TENSOR_LM_HEAD
#define SPARK_SYNTH_MODEL_LAYER_COUNT SPARK_QWEN38_27B_MODEL_LAYER_COUNT
#define SPARK_SYNTH_MODEL_IS_GDN(layer) (SPARK_QWEN38_27B_MODEL_LAYER_IS_GDN(layer) != 0u)
#define SPARK_SYNTH_EXPECTED_TENSOR_COUNT(first, count) \
	SparkQwen38_27bStagePackExpectedTensorCount((first),(count))
#define SPARK_SYNTH_EXPECTED_GEOMETRY(header, first, count) \
	SparkQwen38_27bStagePackExpectedGeometry((header),(first),(count))
#define SPARK_SYNTH_EVERY_LAYER_KINDS \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTENTION_NORM,SPARK_QWEN38_27B_STAGEPACK_TENSOR_MLP_NORM, \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE,SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_UP, \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_DOWN
#define SPARK_SYNTH_GDN_LAYER_KINDS \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_QKV,SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_GATE, \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_BETA,SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_DECAY, \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_OUTPUT,SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_CONV_WEIGHT, \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_A_LOG,SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_DT_BIAS, \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_NORM
#define SPARK_SYNTH_ATTN_LAYER_KINDS \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY,SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY, \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_VALUE,SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_OUTPUT, \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY_NORM,SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY_NORM
#define SPARK_SYNTH_MTP_GLOBAL_KINDS \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FC,SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_EMBED_NORM, \
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_HIDDEN_NORM,SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FINAL_NORM

#include "sparkpipe/spark_pack_synthesize_common.h"

#pragma once

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_weight_codec.h"

/* glm5_next stage pack (.g5nsp): the glm52sp v3 wire layout with the
 * glm5_next tensor-kind set - the hybrid KDA/DSA layer classes, the
 * hyper-connection weights, the indexer kpool compressor, and the MTP
 * head. Produced by tools/glm5_next_pack_synthesize.c (validation /
 * synthesized packs) and tools/glm5_next_resident_stagepack.py (real
 * checkpoint packs, M4). */
#define SPARK_GLM5_NEXT_STAGEPACK_MAGIC UINT32_C(0x33584C47)
#define SPARK_GLM5_NEXT_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES 256u
#define SPARK_GLM5_NEXT_STAGEPACK_MODEL_REVISION_BYTES 65u
#define SPARK_GLM5_NEXT_STAGEPACK_SHA256_BYTES 32u
#define SPARK_GLM5_NEXT_STAGEPACK_FLAG_MTP UINT32_C(0x00000001)
#define SPARK_GLM5_NEXT_STAGEPACK_KNOWN_FLAGS SPARK_GLM5_NEXT_STAGEPACK_FLAG_MTP

typedef enum SparkGlm5NextStagePackPayloadType
{
    SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16 = 1,
    SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32 = 2,
    SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_U32 = 3,
    SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_PACKED_WEIGHT = 4
} SparkGlm5NextStagePackPayloadType;

typedef enum SparkGlm5NextStagePackTensorKind
{
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EMBEDDING = 0,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_FINAL_NORM = 1,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD = 2,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM = 3,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A = 4,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A_NORM = 5,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_B = 6,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A = 7,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A_NORM = 8,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED = 9,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_VALUE = 10,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_OUTPUT = 11,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_POST_ATTN_NORM = 12,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_Q = 13,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_K = 14,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_HEAD = 15,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT = 16,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_BIAS = 17,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_GATE_UP = 18,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_DOWN = 19,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER = 20,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER_CORRECTION = 21,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE = 22,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN = 23,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_GATE_UP = 24,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_DOWN = 25,
    /* KDA linear-attention layers (34): the pack-V2 fusions. */
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_QKV_BETA = 26,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_GATE_DOWN = 27,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_UP = 28,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_GATE_UP = 29,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_Q_CONV = 30,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_K_CONV = 31,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_V_CONV = 32,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_BIAS = 33,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_HEAD_LOG_SCALE = 34,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT_NORM = 35,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT = 36,
    /* Hyper-connections on every weight layer (not the MTP layer). fn is
     * stored F32 (the checkpoint's BF16 rows upcast once at pack time). */
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_FN = 37,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_BASE = 38,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_SCALE = 39,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_FN = 40,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_BASE = 41,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_SCALE = 42,
    /* Indexer kpool compressor on every DSA layer. ape is F32 (the
     * checkpoint BF16 upcast at pack time; the pool kernel reads f32). */
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_APE = 43,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_GATE = 44,
    /* MTP head tensors (layer 45 only, spec path). */
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_EH_PROJ = 45,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_ENORM = 46,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_HNORM = 47,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_SHARED_NORM = 48,
    SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KIND_COUNT = 49
} SparkGlm5NextStagePackTensorKind;

typedef struct SparkGlm5NextStagePackHeader
{
    uint32_t magic;
    uint32_t format_version;
    uint32_t header_bytes;
    uint32_t directory_entry_bytes;
    uint32_t codec_abi_version;
    uint32_t flags;
    uint32_t tensor_count;
    uint32_t stage_count;
    uint32_t stage_index;
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t total_layer_count;
    uint32_t hidden_dimension;
    uint32_t vocab_count;
    uint32_t routed_expert_count;
    uint32_t linear_weight_codec;
    uint32_t expert_weight_codec;
    uint32_t kv_cache_codec;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t directory_offset;
    uint64_t file_bytes;
    char model_revision[SPARK_GLM5_NEXT_STAGEPACK_MODEL_REVISION_BYTES];
    uint8_t contract_sha256[SPARK_GLM5_NEXT_STAGEPACK_SHA256_BYTES];
    uint8_t source_config_sha256[SPARK_GLM5_NEXT_STAGEPACK_SHA256_BYTES];
    uint8_t pack_recipe_sha256[SPARK_GLM5_NEXT_STAGEPACK_SHA256_BYTES];
} SparkGlm5NextStagePackHeader;

typedef struct SparkGlm5NextStagePackEntry
{
    uint32_t tensor_kind;
    uint32_t layer_index;
    uint32_t payload_type;
    uint32_t weight_codec;
    uint32_t scale_encoding;
    uint32_t group_count;
    uint32_t rows;
    uint32_t columns;
    uint64_t payload_offset;
    uint64_t payload_bytes;
    uint64_t scale_offset;
    uint64_t scale_bytes;
} SparkGlm5NextStagePackEntry;

typedef struct SparkGlm5NextStagePackTensorShape
{
    uint32_t payload_type;
    uint32_t weight_codec;
    uint32_t scale_encoding;
    uint32_t group_count;
    uint32_t rows;
    uint32_t columns;
} SparkGlm5NextStagePackTensorShape;

#define SPARK_GLM5_NEXT_STAGEPACK_HEADER_BYTES ((uint32_t)sizeof(SparkGlm5NextStagePackHeader))
#define SPARK_GLM5_NEXT_STAGEPACK_ENTRY_BYTES ((uint32_t)sizeof(SparkGlm5NextStagePackEntry))

static inline uint32_t SparkGlm5NextStagePackLayerIsDense(uint32_t layer_index)
{
    return(layer_index < SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextStagePackLayerIsKda(uint32_t layer_index)
{
    return(layer_index < SPARK_GLM5_NEXT_MODEL_LAYER_COUNT &&
        SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer_index) ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextStagePackLayerIsDsa(uint32_t layer_index)
{
    return(layer_index < SPARK_GLM5_NEXT_MODEL_LAYER_COUNT &&
        !SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer_index) ? 1u : 0u);
}

/* The MTP layer index is a DSA-class layer carrying the MTP head tensors
 * and NO hyper-connections. */
static inline uint32_t SparkGlm5NextStagePackLayerIsMtp(uint32_t layer_index)
{
    return(layer_index == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextStagePackKindIsGlobal(uint32_t tensor_kind)
{
    return(tensor_kind <= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextStagePackKindIsIndexer(uint32_t tensor_kind)
{
    return((tensor_kind >= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_Q &&
            tensor_kind <= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_BIAS) ||
        tensor_kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_APE ||
        tensor_kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_GATE
            ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextStagePackKindIsMla(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A &&
           tensor_kind <= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_OUTPUT ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextStagePackKindIsKda(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_QKV_BETA &&
           tensor_kind <= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextStagePackKindIsHc(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_FN &&
           tensor_kind <= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_SCALE ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextStagePackKindIsMtp(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_EH_PROJ &&
           tensor_kind <= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_SHARED_NORM ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextStagePackKindIsDense(uint32_t tensor_kind)
{
    return(tensor_kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_GATE_UP ||
           tensor_kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_DOWN ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextStagePackKindIsRouted(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER &&
           tensor_kind <= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_DOWN ? 1u : 0u);
}

static inline void SparkGlm5NextStagePackShapeBf16(SparkGlm5NextStagePackTensorShape *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
    shape->payload_type = SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16;
    shape->weight_codec = SPARK_WEIGHT_CODEC_BF16;
    shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
    shape->group_count = groups;
    shape->rows = rows;
    shape->columns = columns;
}

static inline void SparkGlm5NextStagePackShapeF32(SparkGlm5NextStagePackTensorShape *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
    shape->payload_type = SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32;
    shape->weight_codec = SPARK_WEIGHT_CODEC_NONE;
    shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
    shape->group_count = groups;
    shape->rows = rows;
    shape->columns = columns;
}

/* TP sharding policy (must match the packer exactly):
 *   row-sharded - the output dimension splits by ranks (heads, experts,
 *     vocab rows, or the fused per-head sections)
 *   column-sharded - the INPUT dimension splits (down projections read a
 *     rank slice of the intermediate and all-reduce)
 *   replicated - every rank carries the whole tensor (norms, q_a, kv_a,
 *     kv_b, indexer, router, HC, compressor, decay/gate bottleneck).
 * The attention OUT projections (ATTN_OUTPUT, KDA_OUT) are checkpoint
 * [hidden, heads*dim] = OUTPUT-hidden x INPUT-width, the down-projection
 * shape family: the rank slice is of the INPUT columns (this rank's
 * heads) and the out-GEMM lands the full-width rank partial the chain
 * reduces. Row-sharding them silently transposed the block (pack
 * [hidden/tp, width] consumed as [hidden, width/tp]): bounded garbage in
 * every attention partial from layer 0 - the cold-first-request
 * degeneration; TP1-invariant, which is why the M3 gates passed. */
static inline uint32_t SparkGlm5NextStagePackTpShardsRows(uint32_t tensor_kind)
{
    switch ( tensor_kind )
    {
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EMBEDDING:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_B:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_GATE_UP:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_GATE_UP:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_QKV_BETA:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_UP:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_GATE_UP:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_Q_CONV:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_K_CONV:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_V_CONV:
        return(1u);
    default:
        return(0u);
    }
}

static inline uint32_t SparkGlm5NextStagePackTpShardsCols(uint32_t tensor_kind)
{
    switch ( tensor_kind )
    {
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_OUTPUT:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_DOWN:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_DOWN:
    /* the per-channel KDA vectors shard along their channel/head axis
     * (columns at rows=1); rows-sharding a 1-row tensor cannot divide. */
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_BIAS:
    case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_HEAD_LOG_SCALE:
        return(1u);
    default:
        return(0u);
    }
}

/* Pack header TP identity: reserved0 = tp_degree, reserved1 = tp_rank. */
static inline uint32_t SparkGlm5NextStagePackHeaderTpDegree(const SparkGlm5NextStagePackHeader *header)
{
    return(header != 0 ? header->reserved0 : 0u);
}

static inline uint32_t SparkGlm5NextStagePackHeaderTpRank(const SparkGlm5NextStagePackHeader *header)
{
    return(header != 0 ? header->reserved1 : 0u);
}

/* Expected-shape table (the complexity lane's conjunction-soup conversion:
 * the 30-case switch became data). One row per tensor kind; payload_type 0
 * marks an uncovered kind, which the function refuses with the old switch
 * default (-6) — a future enum member without a table row fails loudly
 * instead of silently shape-matching. */
typedef struct SparkGlm5NextStagePackShapeSpec
{
    uint32_t payload_type;
    uint32_t weight_codec;
    uint32_t scale_encoding;
    uint32_t group_count;
    uint32_t rows;
    uint32_t columns;
    uint32_t codec_from_arg; /* expert packed weights: codec rides the entry */
} SparkGlm5NextStagePackShapeSpec;

static const SparkGlm5NextStagePackShapeSpec SPARK_GLM5_NEXT_STAGEPACK_SHAPE_TABLE[SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KIND_COUNT] = {
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EMBEDDING] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_FINAL_NORM] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A_NORM] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_B] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_MLA_QUERY_B_DIMENSION, SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A_NORM] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT, SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION, SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_VALUE] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT, SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION, SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_OUTPUT] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, SPARK_GLM5_NEXT_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_POST_ATTN_NORM] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_Q] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION, SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_K] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_HEAD] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_BIAS] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_GATE_UP] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 2u * SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_DOWN] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER_CORRECTION] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_PACKED_WEIGHT, 0u, 0u, SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT, 2u * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 1u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_PACKED_WEIGHT, 0u, 0u, SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION, 1u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_GATE_UP] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 2u * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_DOWN] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_QKV_BETA] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 2u * SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION + SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION + SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_GATE_DOWN] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 2u * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_UP] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION, SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_GATE_UP] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION, SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_Q_CONV] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION, SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_K_CONV] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION, SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_V_CONV] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION, SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_BIAS] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_HEAD_LOG_SCALE] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT_NORM] =
        /* F32 in the pack (k3 donor convention): the gated norm instantiates
         * Weight=float, and a bf16 store made the kernel read past the
         * tensor - weights 64..127 came back as whatever followed, and the
         * upper half of every head's output silently zeroed. */
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_FN] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION, SPARK_GLM5_NEXT_MODEL_HC_FN_COLUMNS, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_BASE] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_SCALE] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_FN] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION, SPARK_GLM5_NEXT_MODEL_HC_FN_COLUMNS, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_BASE] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_SCALE] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_APE] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL, SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_GATE] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_EH_PROJ] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 2u * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_ENORM] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_HNORM] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_SHARED_NORM] =
        {SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, 0u},
};

/* Layer/kind compatibility rules (unchanged semantics: -3 for a kind the
 * layer class cannot carry, -4 for the dense/routed class mismatch). */
static inline int32_t SparkGlm5NextStagePackCheckLayerKind(uint32_t layer_index,uint32_t tensor_kind)
{
    uint32_t mtp_layer = SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX;
    uint32_t in_mtp = layer_index == mtp_layer;
    if ( SparkGlm5NextStagePackKindIsMla(tensor_kind) != 0u &&
         SparkGlm5NextStagePackLayerIsDsa(layer_index) == 0u && !in_mtp )
        return(-3);
    if ( SparkGlm5NextStagePackKindIsKda(tensor_kind) != 0u &&
         SparkGlm5NextStagePackLayerIsKda(layer_index) == 0u )
        return(-3);
    if ( SparkGlm5NextStagePackKindIsIndexer(tensor_kind) != 0u &&
         SparkGlm5NextStagePackLayerIsDsa(layer_index) == 0u && !in_mtp )
        return(-3);
    if ( SparkGlm5NextStagePackKindIsHc(tensor_kind) != 0u && in_mtp )
        return(-3);
    if ( SparkGlm5NextStagePackKindIsMtp(tensor_kind) != 0u && !in_mtp )
        return(-3);
    if ( SparkGlm5NextStagePackKindIsDense(tensor_kind) !=
             SparkGlm5NextStagePackLayerIsDense(layer_index) &&
         (SparkGlm5NextStagePackKindIsDense(tensor_kind) != 0u ||
          SparkGlm5NextStagePackKindIsRouted(tensor_kind) != 0u) )
        return(-4);
    return(0);
}

static inline int32_t SparkGlm5NextStagePackExpectedShape(uint32_t tensor_kind,uint32_t layer_index,uint32_t expert_codec,uint32_t tp_degree,SparkGlm5NextStagePackTensorShape *shape)
{
    const SparkGlm5NextStagePackShapeSpec *spec;
    uint32_t global;
    uint32_t mtp_layer = SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX;
    if ( shape == 0 || tensor_kind >= SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KIND_COUNT || tp_degree == 0u )
        return(-1);
    memset(shape,0,sizeof(*shape));
    global = SparkGlm5NextStagePackKindIsGlobal(tensor_kind);
    if ( (global != 0u && layer_index != SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER) ||
         (global == 0u && layer_index > mtp_layer) )
        return(-2);
    if ( layer_index <= mtp_layer && layer_index != SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER )
    {
        int32_t allowed = SparkGlm5NextStagePackCheckLayerKind(layer_index,tensor_kind);
        if ( allowed != 0 )
            return(allowed);
    }
    spec = &SPARK_GLM5_NEXT_STAGEPACK_SHAPE_TABLE[tensor_kind];
    if ( spec->payload_type == 0u )
        return(-6);
    if ( spec->codec_from_arg != 0u )
    {
        if ( expert_codec < SPARK_WEIGHT_CODEC_INT6 || expert_codec > SPARK_WEIGHT_CODEC_MXFP4_E2M1 )
            return(-5);
        shape->payload_type = spec->payload_type;
        shape->weight_codec = expert_codec;
        shape->scale_encoding = SparkWeightCodecScaleEncoding(expert_codec);
    }
    else
    {
        shape->payload_type = spec->payload_type;
        shape->weight_codec = spec->weight_codec;
        shape->scale_encoding = spec->scale_encoding;
    }
    shape->group_count = spec->group_count;
    shape->rows = spec->rows;
    shape->columns = spec->columns;
    if ( SparkGlm5NextStagePackTpShardsRows(tensor_kind) != 0u )
    {
        if ( shape->rows == 0u || shape->rows % tp_degree != 0u )
            return(-7);
        shape->rows /= tp_degree;
    }
    if ( SparkGlm5NextStagePackTpShardsCols(tensor_kind) != 0u )
    {
        if ( shape->columns == 0u || shape->columns % tp_degree != 0u )
            return(-7);
        shape->columns /= tp_degree;
    }
    return(0);
}

static inline uint64_t SparkGlm5NextStagePackExpectedPayloadBytes(const SparkGlm5NextStagePackTensorShape *shape)
{
    uint64_t elements;
    if ( shape == 0 || shape->group_count == 0u || shape->rows == 0u || shape->columns == 0u || shape->group_count > UINT64_MAX / shape->rows || (uint64_t)shape->group_count * shape->rows > UINT64_MAX / shape->columns )
        return(0u);
    elements = (uint64_t)shape->group_count * shape->rows * shape->columns;
    if ( shape->payload_type == SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16 )
        return(elements > UINT64_MAX / 2u ? 0u : elements * 2u);
    if ( shape->payload_type == SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32 || shape->payload_type == SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_U32 )
        return(elements > UINT64_MAX / 4u ? 0u : elements * 4u);
    return(shape->payload_type == SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_PACKED_WEIGHT ? SparkWeightCodecPayloadBytes(shape->weight_codec,(uint64_t)shape->group_count * shape->rows,shape->columns) : 0u);
}

static inline uint64_t SparkGlm5NextStagePackExpectedScaleBytes(const SparkGlm5NextStagePackTensorShape *shape)
{
    return(shape != 0 && shape->payload_type == SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_PACKED_WEIGHT ? SparkWeightCodecScaleBytes(shape->weight_codec,shape->group_count,shape->rows,shape->columns) : 0u);
}

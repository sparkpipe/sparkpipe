#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <pthread.h>

#include "runtime/launch.h"
#include "inference/runner/stage_model_cuda.cuh"
#include "sparkpipe/spark_dsv41_flash_family.h"
#include "sparkpipe/spark_dsv41_flash_stage_model.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_row_file.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/kernels/cell_mma.cuh"
#include "inference/kernels/cell_pack.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/hc_site.cuh"
#include "inference/kernels/index_score.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/mx_gemm.cuh"
#include "inference/kernels/cell_mx.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "inference/kernels/formats/mxfp8.cuh"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_dsv41_flash_stagepack.h"
#include "dsv41_flash_stage_kernels.cuh"
#include "dsv41_flash_attend.cuh"

#define D41_TP SPARK_DSV41_FLASH_STAGE_TP_DEGREE
#define D41_KV_TP SPARK_DSV41_FLASH_STAGE_KV_DEGREE
#define D41_ATP SPARK_DSV41_FLASH_STAGE_ATTENTION_DEGREE
#define D41_ROWS (D41_ATP == 1u && D41_TP > 1u)
#define D41_WHOLE (D41_KV_TP == 1u)
#define D41_KV_RANK(state) (D41_WHOLE ? 0u : (state)->tp_rank)
#define D41_ATTENTION_RANK(state) ((state)->tp_rank % D41_ATP)
#define D41_RECORDED_ROWS(state) (D41_WHOLE && (state)->max_rows > D41_WHOLE_RECORD_ROWS ? D41_WHOLE_RECORD_ROWS : (state)->max_rows)
#define D41_ENGRAM_DISK (D41_TP != 16u)
#define D41_HIDDEN SPARK_DSV41_FLASH_STAGE_HIDDEN
#define D41_LAYERS 40u
#define D41_VOCAB 129280u
#define D41_HEADS 64u
#define D41_HEAD_DIM 512u
#define D41_ROPE 64u
#define D41_QLORA 1280u
#define D41_OLORA 1024u
#define D41_OGROUPS 8u
#define D41_WINDOW 128u
#define D41_INDEX_HEADS 32u
#define D41_INDEX_DIM 128u
#define D41_INDEX_TOPK 512u
#define D41_EXPERTS 384u
#define D41_TOPK 6u
#define D41_MOE_INTER 2304u
#define D41_ROUTE_SCALE 1.5f
#define D41_SWIGLU_LIMIT 10.0f
#define D41_EPS 1e-20f
#define D41_HC_ITERATIONS 20u
#define D41_HC_EPS 1e-6f
#define D41_ROPE_THETA 10000.0
#define D41_COMPRESS_THETA 160000.0
#define D41_YARN_FACTOR 16.0
#define D41_YARN_ORIGINAL 65536.0
#define D41_BETA_FAST 32.0
#define D41_BETA_SLOW 1.0
#define D41_CANDIDATE_SOURCE 20u
#define D41_CANDIDATE_BLOCKS 2048u
#define D41_CANDIDATE_ROWS (D41_CANDIDATE_BLOCKS * D41_GRAIN)
#define D41_ENGRAM_HEAD_DIM 256u
#define D41_ENGRAM_COLUMNS 24u
#define D41_ENGRAM_MULTIPLIERS 4u
#define D41_ENGRAM_ORDERS 3u
#define D41_ENGRAM_HEADS 8u
#define D41_ENGRAM_KV_ROWS ((D41K_HC + 1u) * D41_HIDDEN)
#define D41_ENGRAM_EMBED (D41_ENGRAM_COLUMNS * D41_ENGRAM_HEAD_DIM)
#define D41_ENGRAM_KV_LOCAL (D41_ENGRAM_KV_ROWS / D41_TP)
#define D41_FP8_BLOCK 32u
#define D41_SHARED_GRANULE 128u
#define D41_THREADS 256u
#define D41_SLOT_BYTES SPARK_DSV41_FLASH_STAGE_SLOT_BYTES
#define D41_PAGE_TOKENS SPARK_DSV41_FLASH_STAGE_PAGE_TOKENS
#define D41_GRAIN SPARK_DSV41_FLASH_STAGE_SHARD_GRAIN
static_assert(D41_GRAIN == 8u, "a DSV4.1 candidate block is one context shard grain, so each block lives on one rank");
static_assert(D41_FP8_BLOCK == SPARK_DSV41_FLASH_STAGEPACK_FP8_BLOCK, "the stage reads fp8 scales in the pack's blocks");
static_assert(D41_HIDDEN == SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION && D41_LAYERS == SPARK_DSV41_FLASH_MODEL_LAYER_COUNT &&
	D41_VOCAB == SPARK_DSV41_FLASH_MODEL_OUTPUT_VOCAB_COUNT && D41_HEADS == SPARK_DSV41_FLASH_MODEL_ATTENTION_HEAD_COUNT &&
	D41_HEAD_DIM == SPARK_DSV41_FLASH_MODEL_HEAD_DIMENSION && D41_HEAD_DIM == SPARK_DSV41_FLASH_MODEL_KV_LATENT_DIMENSION &&
	D41_ROPE == SPARK_DSV41_FLASH_MODEL_QK_ROPE_HEAD_DIMENSION && D41_QLORA == SPARK_DSV41_FLASH_MODEL_QUERY_LORA_RANK &&
	D41_OLORA == SPARK_DSV41_FLASH_MODEL_OUTPUT_LORA_RANK && D41_OGROUPS == SPARK_DSV41_FLASH_MODEL_OUTPUT_GROUP_COUNT &&
	D41_INDEX_HEADS == SPARK_DSV41_FLASH_MODEL_INDEX_HEAD_COUNT && D41_INDEX_DIM == SPARK_DSV41_FLASH_MODEL_INDEX_HEAD_DIMENSION &&
	D41_EXPERTS == SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT && D41_TOPK == SPARK_DSV41_FLASH_MODEL_EXPERTS_PER_TOKEN &&
	D41_MOE_INTER == SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION,
	"the stage reads the tensors the pack contract shapes");
#define D41_A2A_HEADS SPARK_DSV41_FLASH_STAGE_A2A_HEADS
#define D41_LOCAL_HEADS (D41_HEADS / D41_ATP)
#define D41_LOCAL_INDEX_HEADS (D41_INDEX_HEADS / D41_ATP)
#define D41_LOCAL_EXPERTS (D41_EXPERTS / D41_TP)
#define D41_EXPERT_SLICE (D41_MOE_INTER / D41_TP)
#define D41_HIDDEN_SLICE (D41_HIDDEN / D41_TP)
#define D41_MOE_CODES (D41_TP != 16u && D41_EXPERT_SLICE % (2u * D41_FP8_BLOCK) == 0u ? 1u : 0u)
#define D41_MOE_ROUTE_ELEMENTS (D41_MOE_CODES != 0u ? D41_EXPERT_SLICE / 2u + D41_EXPERT_SLICE / (2u * D41_FP8_BLOCK) : D41_EXPERT_SLICE)
#define D41_SHARED_ROUTED D41_MOE_CODES
#define D41_SHARED_SLICE_MAX ((D41_MOE_INTER / D41_SHARED_GRANULE + D41_TP - 1u) / D41_TP * D41_SHARED_GRANULE)
#define D41_MOE_SHARED_ELEMENTS (D41_SHARED_ROUTED != 0u ? (D41_SHARED_SLICE_MAX / 2u + D41_SHARED_SLICE_MAX / (2u * D41_FP8_BLOCK) + 7u) / 8u * 8u : 0u)
#define D41_MOE_ROW_ELEMENTS (D41_SHARED_ROUTED != 0u ? (D41_MOE_SHARED_ELEMENTS + D41_TOPK * D41_MOE_ROUTE_ELEMENTS + 7u) / 8u * 8u : \
	(D41_TOPK * D41_MOE_ROUTE_ELEMENTS + D41_HIDDEN_SLICE + 7u) / 8u * 8u)
#define D41_WO_A_ROWS (D41_OGROUPS * D41_OLORA / D41_ATP)
#define D41_RING_ROWS (D41_WINDOW / D41_KV_TP)
#define D41_VERIFY_ROWS SPARK_DSV41_FLASH_STAGE_VERIFY_ROWS
#define D41_PAIR_CACHES 3u
#define D41_SELECTION_CHUNK 64u
#define D41_INDEX_PACK ((D41_LOCAL_INDEX_HEADS * (D41_INDEX_DIM + 1u) + 7u) / 8u * 8u)
#define D41_GROUP_WIDTH (D41_HEADS / D41_OGROUPS * D41_HEAD_DIM)
#define D41_LOCAL_GROUPS (D41_ATP > D41_OGROUPS ? 1u : D41_OGROUPS / D41_ATP)
#define D41_GROUP_ROWS (D41_WO_A_ROWS / D41_LOCAL_GROUPS)
#define D41_ENGRAM_KV_PARTS SPARK_DSV41_FLASH_STAGE_ENGRAM_KV_PARTS
#define D41_ENGRAM_CACHE_ROWS 1048576u
#define D41_ENGRAM_READERS 4u
#define D41_SELECT_BYTES (1ull << 30)
#define D41_ENGRAM_TABLES_MAGIC 0x45313444u
#define D41_GEMM_TILE_N 128u
#define D41_ACT_BLOCKS 4096u
#define D41_Q_CODE_ELEMENTS ((D41_QLORA + D41_QLORA / D41_FP8_BLOCK) / 2u)
#define D41_SKINNY_EXPERT_ROWS 16u
#define D41_ATTEND_TILES 2u
#define D41_OWN_ROWS_MIN (D41_WHOLE ? 2048u : UINT32_MAX)
#define D41_FFN_GATHER_ELEMENTS ((D41_HIDDEN + D41_HIDDEN / D41_FP8_BLOCK + 8u * D41_TOPK + 15u) / 16u * 8u)
#define D41_BACKUP_LAYERS (D41_ROWS ? D41_CANDIDATE_SOURCE : 0u)
#define D41_BACKUP_COPIES (D41_ROWS && D41_TP >= 4u ? 2u : 1u)
#define D41_HELD_EXPERTS ((1u + D41_BACKUP_COPIES) * D41_LOCAL_EXPERTS)
#define D41_BACKUP_PASS (D41_LOCAL_EXPERTS / 2u)
#define D41_BACKUP_PAIR 2u
typedef D41ExchangeCounts<D41_TP, D41_EXPERTS> D41Counts;
typedef D41ExchangeTotals<D41_TP, D41_EXPERTS> D41Totals;
static_assert(D41_HELD_EXPERTS < 255u && D41_BACKUP_LAYERS <= 64u && sizeof(D41Counts) % 16u == 0u &&
	offsetof(D41Counts, routes) % 16u == 0u, "held groups fit a byte, filled layers a 64-bit mask, and a rank's counts gather as whole 16-byte rows");
#define D41_OWNER_OUTPUT_ELEMENTS ((D41_WO_A_ROWS + D41_WO_A_ROWS / D41_FP8_BLOCK) / 2u)
#define D41_OWN_ROWS_STEP SPARK_DSV41_FLASH_STAGE_OWN_ROWS_STEP
#define D41_WINDOW_EDGE_ROWS 256u
#define D41_WHOLE_RECORD_ROWS (D41_LOCAL_HEADS > D41A_QUERIES ? 2u * D41_OWN_ROWS_STEP : SPARK_STAGE_RUNNER_GRAPH_ROWS)
#define D41_MX_WIDTH (D41_ENGRAM_EMBED > D41_HIDDEN ? D41_ENGRAM_EMBED : D41_HIDDEN)
#define D41_NORM_SPLIT_ROWS 16u
#define D41_TAIL_MIN_ROWS (LM_SKINNY_ROWS_WIDE > D41_SKINNY_EXPERT_ROWS ? LM_SKINNY_ROWS_WIDE : D41_SKINNY_EXPERT_ROWS)
#define D41_RECORD_ROW_ELEMENTS (D41_LOCAL_HEADS * D41K_RECORD * (uint32_t)(sizeof(float) / sizeof(uint16_t)))
#define D41_ENGRAM_SPLIT_ROWS SPARK_DSV41_FLASH_STAGE_ENGRAM_SPLIT_ROWS
#define D41_ENGRAM_HALF (D41_ENGRAM_EMBED / 2u)

#define D41_RING_ELEMENTS (D41_RING_ROWS * D41_HEAD_DIM)
#define D41_RING_LANDING_ROWS ((D41_LAYERS * D41_RING_ELEMENTS + D41_HIDDEN - 1u) / D41_HIDDEN)
#define D41_PAIR_RECORD_ROWS ((uint32_t)(SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / (2u * D41_RECORD_ROW_ELEMENTS * sizeof(uint16_t))))
static_assert(D41_WHOLE || D41_PAIR_RECORD_ROWS >= 1u, "a pair-duplicated record row fits a staging slot");
#define D41_PAIR_RECORD_CHOSEN_ROWS 1u
static_assert(D41_WHOLE || D41_PAIR_RECORD_CHOSEN_ROWS <= D41_PAIR_RECORD_ROWS, "pair-duplicated records are chosen only where they fit a staging slot");
static_assert(D41_WHOLE || (D41_RECORD_ROW_ELEMENTS <= D41_HIDDEN && D41_RECORD_ROW_ELEMENTS % 8u == 0u),
	"one row of attention records fits a collective row and keeps 16-byte copies");
static_assert(D41_ENGRAM_EMBED % 2u == 0u && D41_ENGRAM_HALF <= D41_HIDDEN && D41_ENGRAM_HALF % 8u == 0u,
	"an engram embedding row splits into two collective rows");
static_assert((D41_WHOLE ? D41_INDEX_PACK : D41_LOCAL_HEADS * D41_HEAD_DIM + D41_INDEX_PACK) <= D41_HIDDEN && D41_INDEX_PACK % 8u == 0u,
	"the index query pack rides in a collective row, with the queries where the context is cut across ranks, and keeps 16-byte copies");
static_assert(D41_TP == 16u || D41_TP == 8u || D41_TP == 4u, "the stage is built for sixteen, eight or four ranks");
static_assert(D41_ATP != 16u || (D41_INDEX_PACK == 264u && D41_LOCAL_GROUPS == 1u && D41_GROUP_ROWS == 512u && D41_ENGRAM_KV_PARTS == 1u),
	"sixteen ranks keep the exchange rows their packs and caches were made with");
static_assert(D41_WHOLE == (D41_ATP <= D41_OGROUPS) && D41_WINDOW % D41_KV_TP == 0u && D41_GROUP_ROWS % D41_FP8_BLOCK == 0u &&
	D41_LOCAL_HEADS * D41_HEAD_DIM == D41_GROUP_WIDTH * D41_LOCAL_GROUPS / (D41_ATP > D41_OGROUPS ? 2u : 1u),
	"a rank that keeps the whole context holds whole output groups, and a rank of sixteen holds half of one");
static_assert(D41_WINDOW == D41K_WINDOW_KEYS && (!D41_WHOLE || D41_RING_ROWS == D41_WINDOW), "a rank that keeps the whole window folds all of its keys in one pass");
static_assert(D41_ENGRAM_KV_LOCAL % D41_ENGRAM_KV_PARTS == 0u && D41_ENGRAM_KV_LOCAL / D41_ENGRAM_KV_PARTS <= D41_HIDDEN &&
	D41_ENGRAM_KV_LOCAL / D41_ENGRAM_KV_PARTS % 8u == 0u, "a rank's engram key/value slice gathers as whole collective rows");
static_assert(D41_HEAD_DIM % D41_THREADS == 0u && (D41_HEAD_DIM - D41_ROPE) % D41_FP8_BLOCK == 0u && D41_ROPE == 2u * D41K_ROPE_PAIRS,
	"the key/value norm covers whole rows per thread block, and its rotary pairs sit in whole fp8 groups of one warp");
static_assert(D41_MOE_INTER % D41_TP == 0u && D41_HIDDEN % D41_TP == 0u && D41_EXPERT_SLICE % 16u == 0u && D41_HIDDEN_SLICE % 32u == 0u,
	"every rank holds whole 16-neuron cells of each expert slice and whole 32-wide tensor-core tiles of the down slice");
static_assert(D41_MOE_ROW_ELEMENTS <= D41_HIDDEN && D41_MOE_ROW_ELEMENTS % 8u == 0u && D41_HIDDEN_SLICE % 8u == 0u,
	"the MoE exchange rows fit a collective row and keep 16-byte copies");
static_assert(D41_MOE_INTER % D41_THREADS == 0u, "the activation assembly covers whole fp8 groups with full warps");
static_assert(D41_MOE_CODES == 0u || (D41_EXPERT_SLICE % 16u == 0u && (D41_EXPERT_SLICE / D41_FP8_BLOCK) % 2u == 0u && D41_MOE_INTER % 128u == 0u),
	"the down projection's fp8 rows assemble from 16-byte code vectors and 2-byte scale pairs of one source each and split into 128-wide tiles");
static_assert(!D41_WHOLE || (D41_HIDDEN % D41_FP8_BLOCK == 0u && D41_FFN_GATHER_ELEMENTS <= D41_HIDDEN && 2u * D41_FFN_GATHER_ELEMENTS <= D41_TP * D41_MOE_ROW_ELEMENTS),
	"an owner's expert input rows cross as fp8 codes, scales and routes inside the MoE exchange buffers");
static_assert(!D41_WHOLE || (D41_WO_A_ROWS % LM_MX_TILE_K == 0u && D41_OWNER_OUTPUT_ELEMENTS % 8u == 0u && D41_OWNER_OUTPUT_ELEMENTS <= D41_HIDDEN &&
	D41_OWNER_OUTPUT_ELEMENTS <= D41_TP * D41_MOE_ROW_ELEMENTS && D41_WO_A_ROWS / 2u <= D41_HIDDEN),
	"an owner's attention output rows cross as fp8 codes with their scales inside the MoE exchange buffers, and every rank's o_b rows gather as collective rows");
static_assert(D41_SHARED_ROUTED == 0u || (D41_SHARED_GRANULE % (4u * D41_FP8_BLOCK) == 0u && D41_HIDDEN_SLICE % D41_FP8_BLOCK == 0u),
	"a routed shared slice is whole 16-byte rows of fp8 codes with 4-byte scale rows, and each rank's down columns are whole fp8 row blocks");
static_assert(D41_WINDOW_EDGE_ROWS >= D41_WINDOW + D41_OWN_ROWS_STEP - 1u && D41_HEAD_DIM == D41K_LATENT,
	"a rank's last rows cover the next rank's window and the step's last window past any padding rows");
static_assert(D41_ENGRAM_HEAD_DIM == D41K_ENGRAM_DIM && D41K_ENGRAM_RECORD_BYTES % 16u == 0u, "an engram record holds one fp8 row, its scales and padding to 16 bytes");
static_assert(D41_ENGRAM_KV_ROWS % (D41_TP * D41_FP8_BLOCK) == 0u, "the engram key/value rows split into whole fp8 blocks per rank");
static_assert(D41_ENGRAM_KV_ROWS == SPARK_DSV41_FLASH_ENGRAM_KV_ROWS && D41_ENGRAM_EMBED == SPARK_DSV41_FLASH_ENGRAM_EMBED_COLUMNS &&
	D41_ENGRAM_HEAD_DIM == SPARK_DSV41_FLASH_MODEL_ENGRAM_HEAD_DIMENSION && D41K_HC == SPARK_DSV41_FLASH_MODEL_HC_MULT, "the stage reads the engram shard the format check accepts");
static_assert(D41_MOE_INTER % D41_SHARED_GRANULE == 0u && D41_SHARED_GRANULE % D41_FP8_BLOCK == 0u,
	"the shared expert width splits into whole fp8 blocks and whole dense GEMM k tiles");
static_assert(D41K_HC == LM_HC_SITE_STREAMS, "the stream expansion writes every hyper-connection stream");

using D41Ratio2Kv = LmKvGeometry<D41_SLOT_BYTES, D41_PAGE_TOKENS / 2u, true>;
using D41Ratio1Kv = LmKvGeometry<D41_SLOT_BYTES, D41_PAGE_TOKENS, true>;

typedef SparkDsv41FlashStagePackHeader D41PackHeader;
typedef SparkDsv41FlashStagePackEntry D41PackEntry;

typedef struct D41Fp8Linear
{
	const uint8_t *weight, *scale;
	const uint16_t *wide;
} D41Fp8Linear;

typedef struct D41Mx
{
	uint8_t *codes, *scales;
} D41Mx;

typedef struct D41Layer
{
	const uint16_t *attn_norm, *ffn_norm, *q_norm, *kv_norm;
	D41Fp8Linear q_a, q_b, kv_a, o_a[D41_LOCAL_GROUPS], o_b, idx_q_b;
	const uint16_t *idx_wk, *idx_wp, *idx_kn, *c_wkv, *c_wgate, *c_norm;
	const float *sink, *group_sink, *hc_attn_fn, *hc_attn_base, *hc_attn_scale, *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
	const uint16_t *router;
	const float *router_bias;
	const uint8_t *e_w13, *e_w13_scale, *e_w2, *e_w2_scale;
	const uint8_t *e_w13_cells, *e_w2_cells, *e_w13_backup, *e_w2_backup;
	D41Fp8Linear sh_w1, sh_w2, sh_w3, sh_w2_part[D41_SHARED_ROUTED != 0u ? D41_TP : 1u];
} D41Layer;

struct D41ModelState;

typedef struct D41EngramCall
{
	struct D41ModelState *state;
	uint32_t which, rows;
} D41EngramCall;

typedef struct D41EngramAhead
{
	pthread_t thread;
	pthread_mutex_t lock;
	pthread_cond_t wake, done;
	uint32_t started, stop, busy, pending, finished, failed, rows;
} D41EngramAhead;

typedef struct D41Engram
{
	const uint8_t *payload, *scale;
	D41Fp8Linear wkv, wkv_whole;
	const uint16_t *q_weight, *k_weight;
	uint64_t first_row, local_rows, part, table_rows;
	int64_t *multipliers, *primes, *offsets;
} D41Engram;

struct D41ModelState : CudaStageState
{
	const SparkStageRunnerConfiguration *configuration;
	uint32_t context_slots, max_context;
	FILE *pack_file;
	D41PackHeader header;
	D41PackEntry *entries;
	SparkWeightdLazyPack *engram_pack;
	char engram_path[1024], engram_sha256[72];
	uint64_t engram_bytes;
	D41PackEntry engram_entries[8];
	int32_t engram_pad;
	int32_t *token_map;
	D41Layer layers[D41_LAYERS];
	D41Engram engram[2];
	float *full_frequency, *compress_frequency;
	uint8_t *fp8_side;
	uint64_t fp8_side_bytes;
	uint8_t *cells;
	uint64_t cells_bytes;
	uint64_t record_chunk_floats, a2a_units;
	uint32_t local_score_stride, global_score_stride, step_score_stride, step_ratio2_stride, step_blocks;
	uint32_t candidate_blocks, local_block_stride, global_block_stride, local_block_keep, global_block_keep, block_round_parts;
	uint16_t *streams, *x, *xq, *q_latent, *qrq, *q_local, *q_gathered, *kv, *window_kv;
	uint16_t *own_x, *own_out, *own_q, *window_edges, *streams_whole;
	float *own_c;
	uint32_t *own_selected, *exchange_slot, *exchange_slot_count, *exchange_counts_host, *route_output_row;
	D41Counts *exchange_count, *exchange_counts;
	D41Totals *exchange_totals, *exchange_totals_host;
	uint16_t *exchange_quota, *exchange_quota_host, *expert_at_host;
	uint8_t *expert_backup, *expert_group, *owner_host, *backup_host, *group_host, *backup_cells;
	uint64_t backup_group13, backup_group2, backup_filled;
	cudaEvent_t counts_ready;
	cudaStream_t shared_stream;
	cudaEvent_t shared_fork, shared_join;
	uint32_t exchange_rows, exchange_live, shuffle_rows, moe_rows;
	D41ReceivedRows<D41_TP> exchange_received, shuffle_received;
	uint8_t *expert_owner, *expert_slot;
	int64_t *engram_ids_next;
	uint32_t *engram_slot, *engram_counts, engram_capacity[2];
	uint8_t *engram_send, *engram_received;
	D41Mx mx;
	float *mixes, *pre_attn, *post_attn, *comb_attn, *post_ffn, *comb_ffn, *pre_head, *pre_whole;
	uint32_t own_first, own_rows, own_pad, o_b_ready, owner_reduced, x_complete, tail_owned;
	uint8_t *o_b_all, *o_b_all_scale;
	float *c_kv, *c_score;
	uint16_t *c_latent, *c_latent_store, *idx_key, *idx_pack_local, *idx_query, *idx_w_full;
	uint32_t *emit_row, *bound_ratio2, *bound_ratio1, *length_ratio2, *length_ratio1, *row_lane, *row_ordinal, *tail_row_begin;
	uint32_t tail_target, tail_first_position, compress_done, row_invariant;
	uint32_t local_attention, local_upper, landing_keys[2], landing_rows[2];
	int16_t tap_output[D41_LAYERS];
	uint16_t *tap_slab;
	uint64_t tap_stride;
	uint32_t *tap_first_row;
	uint32_t tap_reach;
	uint64_t landing_stride[2];
	uint8_t *landing;
	uint16_t *ring_send, *ring_landing;
	uint32_t *gather_plan, *split_selected, *split_offsets;
	float *local_scores, *topk_values;
	uint32_t *local_selected, *selected, *topk_positions;
	uint64_t topk_entries;
	uint2 *candidates, *candidates_gathered;
	float *block_scores;
	uint32_t *block_local_selected, *block_selected;
	uint2 *block_candidates, *block_gathered;
	uint8_t *block_mask;
	uint16_t *verify_window;
	float *verify_c_kv, *verify_c_score;
	float *records_send, *records_received;
	uint16_t *attn_merged, *attn_gathered, *group_input, *wo_a_out, *partial;
	float *pre_init;
	uint32_t max_positions;
	float *router_logits, *route_weight, *packed_weight;
	uint32_t *route_expert, *route_global, *group_row_offset, *route_packed_row, *route_source_token, *tile_prefix_up, *tile_prefix_down;
	uint16_t *expert_gate_up, *expert_act, *expert_slice, *moe_send, *moe_received, *moe_final, *moe_gathered;
	uint16_t *moe_out, *shared_gate, *shared_up, *shared_act, *shared_out;
	int64_t *engram_ids;
	uint16_t *engram_embed, *engram_kv, *engram_kv_local, *engram_kv_gathered;
	SparkRowFile *engram_file[2];
	int64_t *engram_ids_host;
	uint8_t *engram_rows, *engram_rows_host, *engram_resident;
	uint64_t engram_resident_bytes;
	D41EngramCall *engram_calls;
	D41EngramAhead engram_ahead;
	int64_t *engram_ahead_ids_host[2], *engram_ahead_ids[2];
	uint8_t *engram_ahead_rows_host[2], *engram_ahead_rows[2];
	uint32_t engram_ahead_step;
	volatile uint32_t engram_failed;
	uint32_t select_rows;
	uint16_t *wo_a_group;
	uint32_t shared_first, shared_width, moe_whole;
	uint8_t *recurrent;
	uint64_t ring_bytes, compressor_bytes, history_bytes, recurrent_bytes;
	LmKvAccessError *host_error;
	LmKvShardView latent_views[4], index_views[4];
	FILE *dump;
	uint16_t *dump_streams;
	uint32_t *dump_routes;
	float *dump_weights;
	SparkKvShard shard;
};

static uint32_t D41Blocks(uint64_t elements)
{
	const uint64_t blocks = (elements + D41_THREADS - 1u) / D41_THREADS;
	return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
}

static SparkStatus D41LaunchedAt(uint32_t line)
{
	const cudaError_t error = cudaPeekAtLastError();
	if ( error == cudaSuccess )
		return SPARK_STATUS_OK;
	fprintf(stderr, "sparkpipe_dsv41_flash: CUDA error seen at line %u: %s\n", line, cudaGetErrorString(error));
	return SPARK_STATUS_INTERNAL_ERROR;
}

#define D41Launched() D41LaunchedAt(__LINE__)

static int32_t D41Project(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32,
	uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, uint32_t row_invariant, cudaStream_t stream)
{
	if ( rows <= LM_SKINNY_ROWS_WIDE || row_invariant != 0u )
	{
		LmSkinnyArguments args;
		memset(&args, 0, sizeof(args));
		args.weight = (const uint8_t *)weight;
		args.activation = source;
		args.output_bf16 = destination;
		args.output_f32 = destination_f32;
		args.scale = LmScaleTensorNone();
		args.rows = rows;
		args.input_dimension = input_dimension;
		args.output_dimension = output_dimension;
		const int32_t status = LmSkinnyLaunch<LmBf16Format>(&args, stream, row_invariant);
		if ( status != LM_LAUNCH_ERR_SHAPE || rows > LM_SKINNY_ROWS_WIDE )
			return status;
		LmSlowPath("dsv41 linear: the skinny kernel refused a small wave", rows, input_dimension, output_dimension);
	}
	const int32_t status = LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows,
		input_dimension, output_dimension, 0u, 0u, multiprocessors, stream);
	if ( status != LM_LAUNCH_ERR_SHAPE )
		return status;
	LmSlowPath("dsv41 linear: the stream GEMM refused, naive linear", rows, input_dimension, output_dimension);
	D41NaiveLinearKernel<<<dim3((output_dimension + 7u) / 8u, rows), 256u, 0, stream>>>(weight, source, destination, destination_f32,
		input_dimension, output_dimension);
	return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static uint32_t D41MxReady(const D41Mx *mx, uint32_t rows, uint32_t input_dimension, uint32_t row_invariant);

static uint32_t D41RowsAttention(const D41ModelState *state)
{
	return D41_ROWS && state->own_rows != 0u && D41MxReady(&state->mx, state->own_rows, D41_QLORA, state->row_invariant) != 0u ? 1u : 0u;
}

static uint32_t D41MxReady(const D41Mx *mx, uint32_t rows, uint32_t input_dimension, uint32_t row_invariant)
{
	return mx != 0 && mx->codes != 0 && rows > LM_SKINNY_ROWS_WIDE && row_invariant == 0u && input_dimension % LM_MX_TILE_K == 0u &&
		input_dimension <= D41_MX_WIDTH ? 1u : 0u;
}

static int32_t D41LinearMx(const D41Fp8Linear *linear, const D41Mx *mx, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
{
	return LmMxGemmDense(mx->codes, mx->scales, linear->weight, linear->scale, destination, rows, input_dimension, output_dimension, 0u, 0u,
		multiprocessors, stream);
}

static int32_t D41Linear(const D41Fp8Linear *linear, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension, uint32_t multiprocessors, uint32_t row_invariant, const D41Mx *mx, cudaStream_t stream)
{
	const LmScaleTensor scale = LmScaleTensorBlockUe8m0(linear->scale, 1u, output_dimension, input_dimension, 1u, LM_MXFP8_GROUP);
	if ( D41MxReady(mx, rows, input_dimension, row_invariant) != 0u )
	{
		const int32_t status = LmMxQuantize(source, mx->codes, mx->scales, (uint64_t)rows * input_dimension, stream);
		return status != LM_LAUNCH_OK ? status : D41LinearMx(linear, mx, destination, rows, input_dimension, output_dimension, multiprocessors, stream);
	}
	if ( rows <= LM_SKINNY_ROWS_WIDE || row_invariant != 0u )
	{
		LmSkinnyArguments args;
		memset(&args, 0, sizeof(args));
		args.weight = linear->weight;
		args.activation = source;
		args.output_bf16 = destination;
		args.scale = scale;
		args.rows = rows;
		args.input_dimension = input_dimension;
		args.output_dimension = output_dimension;
		const int32_t status = LmSkinnyLaunch<LmMxfp8>(&args, stream, row_invariant);
		if ( status != LM_LAUNCH_ERR_SHAPE || rows > LM_SKINNY_ROWS_WIDE )
			return status;
		LmSlowPath("dsv41 fp8 linear: the skinny kernel refused a small wave", rows, input_dimension, output_dimension);
	}
	return LmStreamGemmDense<LmBf16Block32>(linear->wide, LmScaleTensorNone(), source, destination, (float *)0, rows, input_dimension,
		output_dimension, 0u, 0u, multiprocessors, stream);
}

static LmSkinnyDenseTarget D41Target(const D41Fp8Linear *linear, uint16_t *destination, uint32_t input_dimension, uint32_t output_dimension)
{
	LmSkinnyDenseTarget target;
	memset(&target, 0, sizeof(target));
	target.weight = linear->weight;
	target.output_bf16 = destination;
	target.output_dimension = output_dimension;
	target.scale = LmScaleTensorBlockUe8m0(linear->scale, 1u, output_dimension, input_dimension, 1u, LM_MXFP8_GROUP);
	return target;
}

static int32_t D41LinearPair(const D41Fp8Linear *first, const D41Fp8Linear *second, const uint16_t *source, uint16_t *first_destination,
	uint16_t *second_destination, uint32_t rows, uint32_t input_dimension, uint32_t first_dimension, uint32_t second_dimension,
	uint32_t multiprocessors, uint32_t row_invariant, const D41Mx *mx, cudaStream_t stream)
{
	if ( D41MxReady(mx, rows, input_dimension, row_invariant) != 0u )
	{
		int32_t status = LmMxQuantize(source, mx->codes, mx->scales, (uint64_t)rows * input_dimension, stream);
		if ( status == LM_LAUNCH_OK )
			status = D41LinearMx(first, mx, first_destination, rows, input_dimension, first_dimension, multiprocessors, stream);
		return status != LM_LAUNCH_OK ? status : D41LinearMx(second, mx, second_destination, rows, input_dimension, second_dimension, multiprocessors, stream);
	}
	if ( rows <= LM_SKINNY_ROWS_WIDE || row_invariant != 0u )
	{
		const LmSkinnyDenseTarget targets[2] = { D41Target(first, first_destination, input_dimension, first_dimension),
			D41Target(second, second_destination, input_dimension, second_dimension) };
		const int32_t status = LmSkinnyDenseMulti<LmMxfp8>(targets, 2u, source, rows, input_dimension, stream, row_invariant);
		if ( status != LM_LAUNCH_ERR_SHAPE || rows > LM_SKINNY_ROWS_WIDE )
			return status;
		LmSlowPath("dsv41 fp8 pair: the skinny kernel refused a small wave", rows, input_dimension, first_dimension);
	}
	const int32_t status = D41Linear(first, source, first_destination, rows, input_dimension, first_dimension, multiprocessors, 0u, (const D41Mx *)0,
		stream);
	return status != LM_LAUNCH_OK ? status :
		D41Linear(second, source, second_destination, rows, input_dimension, second_dimension, multiprocessors, 0u, (const D41Mx *)0, stream);
}

static void D41RmsNorm(const uint16_t *input, const uint16_t *weight, uint16_t *output, uint32_t rows, uint32_t width, cudaStream_t stream)
{
	LmFusedResidualRmsNormKernel<D41_THREADS,uint16_t><<<rows, D41_THREADS, (width + 8u) * sizeof(float), stream>>>(
		(uint16_t *)input, (const uint16_t *)0, weight, (uint16_t *)0, output, width, width, D41_EPS);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void D41RmsNormQdqKernel(const uint16_t *input, const uint16_t *__restrict__ streams, const float *__restrict__ pre,
	const uint16_t *__restrict__ weight, uint16_t *output, uint16_t *__restrict__ quantized, uint32_t dimension, float epsilon)
{
	extern __shared__ float d41_norm_shared[];
	float *row = d41_norm_shared;
	float *reduction = d41_norm_shared + dimension;
	const uint64_t base = (uint64_t)blockIdx.x * dimension;
	const uint32_t span = dimension / gridDim.y, begin = blockIdx.y * span;
	uint32_t index;
	float total = 0.0f, scale;
	for ( index = threadIdx.x; index < dimension; index += THREADS )
	{
		float value = LmBf16ToFloat(streams != 0 ? LmHcCollapseElement<D41K_HC>(streams, pre, blockIdx.x, dimension, index) : input[base + index]);
		row[index] = value;
		total += value * value;
	}
	total = LmBlockSum<THREADS>(total, reduction);
	scale = SparkMathRmsScale(total, (float)dimension, epsilon);
	for ( index = begin + threadIdx.x; index < begin + span; index += THREADS )
	{
		const uint16_t normed = LmFloatToBf16(row[index] * scale * LmScalarToFloat(weight[index]));
		if ( output != 0 )
			output[base + index] = normed;
		quantized[base + index] = SparkDsv41FlashFp8QdqLane(LmBf16ToFloat(normed));
	}
}

static void D41Fp8Qdq(uint16_t *data, uint64_t elements, cudaStream_t stream)
{
	(void)SparkDsv41FlashLaunchFp8Qdq(stream, data, elements);
}

static void D41Fp8QdqCopy(const uint16_t *source, uint16_t *data, uint64_t elements, cudaStream_t stream)
{
	(void)SparkDsv41FlashLaunchFp8QdqCopy(stream, source, data, elements);
}

static int32_t D41QdqLinear(const D41Fp8Linear *linear, uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension, uint32_t multiprocessors, uint32_t row_invariant, const D41Mx *mx, cudaStream_t stream)
{
	if ( D41MxReady(mx, rows, input_dimension, row_invariant) == 0u )
		D41Fp8Qdq(source, (uint64_t)rows * input_dimension, stream);
	return D41Linear(linear, source, destination, rows, input_dimension, output_dimension, multiprocessors, row_invariant, mx, stream);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void D41KvNormRopeQdqKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, const uint32_t *__restrict__ positions,
	const float *__restrict__ inverse_frequency, uint16_t *__restrict__ quantized, uint32_t dimension, uint32_t rope_offset, float epsilon)
{
	extern __shared__ float d41_kv_shared[];
	float *row = d41_kv_shared;
	float *reduction = d41_kv_shared + dimension;
	const uint64_t base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	float total = 0.0f, scale;
	for ( index = threadIdx.x; index < dimension; index += THREADS )
	{
		float value = LmBf16ToFloat(input[base + index]);
		row[index] = value;
		total += value * value;
	}
	total = LmBlockSum<THREADS>(total, reduction);
	scale = SparkMathRmsScale(total, (float)dimension, epsilon);
	for ( index = threadIdx.x; index < dimension; index += THREADS )
	{
		uint16_t value = LmFloatToBf16(row[index] * scale * LmScalarToFloat(weight[index]));
		const float own = LmBf16ToFloat(value), partner = __shfl_xor_sync(0xffffffffu, own, 1);
		if ( index >= rope_offset )
		{
			float sine, cosine;
			D41RopeTurn(positions[blockIdx.x], 0, inverse_frequency[(index - rope_offset) / 2u], 0u, &sine, &cosine);
			value = (index & 1u) == 0u ? D41RopeReal(own, partner, sine, cosine) : D41RopeImaginary(partner, own, sine, cosine);
		}
		quantized[base + index] = SparkDsv41FlashFp8QdqLane(LmBf16ToFloat(value));
	}
}

static uint32_t D41NormParts(uint32_t rows, uint32_t width)
{
	return rows <= D41_NORM_SPLIT_ROWS ? width / D41_THREADS : 1u;
}

static void D41RmsNormQdq(const uint16_t *input, const uint16_t *weight, uint16_t *output, uint16_t *quantized, uint32_t rows, uint32_t width,
	cudaStream_t stream)
{
	static_assert(D41_THREADS % D41_FP8_BLOCK == 0u, "a norm block holds whole fp8 groups");
	if ( width % D41_THREADS != 0u )
	{
		D41RmsNorm(input, weight, output != 0 ? output : quantized, rows, width, stream);
		D41Fp8QdqCopy(output != 0 ? output : quantized, quantized, (uint64_t)rows * width, stream);
		return;
	}
	D41RmsNormQdqKernel<D41_THREADS><<<dim3(rows, D41NormParts(rows, width)), D41_THREADS, (width + 8u) * sizeof(float), stream>>>(input, 0, 0, weight,
		output, quantized, width, D41_EPS);
}

static void D41CollapseNormQdq(const uint16_t *streams, const float *pre, const uint16_t *weight, uint16_t *output, uint16_t *quantized,
	uint32_t rows, cudaStream_t stream)
{
	static_assert(D41_HIDDEN % D41_THREADS == 0u, "the hidden row splits evenly over the norm block");
	D41RmsNormQdqKernel<D41_THREADS><<<dim3(rows, D41NormParts(rows, D41_HIDDEN)), D41_THREADS, (D41_HIDDEN + 8u) * sizeof(float), stream>>>(0, streams,
		pre, weight, output, quantized, D41_HIDDEN, D41_EPS);
}

static void D41Rope(uint16_t *data, uint32_t rows, uint32_t heads, uint32_t head_stride, uint32_t rope_offset,
	const uint32_t *positions, int32_t shift, const float *frequency, uint32_t inverse, const uint32_t *row_valid, cudaStream_t stream)
{
	D41RopeRowsKernel<<<dim3(rows, heads), D41K_ROPE_PAIRS, 0, stream>>>(data, heads, head_stride, rope_offset, positions, shift,
		frequency, inverse, row_valid);
}

static SparkStatus D41HcMixes(D41ModelState *state, const uint16_t *streams, const float *fn, const float *base, const float *scale,
	float *pre, float *post, float *comb, uint32_t rows, cudaStream_t stream)
{
	LmHcSinkhornMixes<D41K_HC>(streams, fn, scale, base, state->mixes, pre, post, comb, rows, D41_HIDDEN, D41_EPS, D41_HC_ITERATIONS, D41_HC_EPS, stream);
	return D41Launched();
}

static void CUDART_CB D41EngramFetch(void *argument)
{
	const D41EngramCall *call = (const D41EngramCall *)argument;
	D41ModelState *state = call->state;
	if ( SparkRowFileFetch(state->engram_file[call->which], state->engram_ids_host, (uint64_t)call->rows * D41_ENGRAM_COLUMNS,
		(int64_t)state->engram[call->which].first_row, state->engram_rows_host, D41_ENGRAM_HEAD_DIM) != SPARK_STATUS_OK )
		state->engram_failed = 1u;
}

static void *D41EngramAheadWork(void *argument)
{
	D41ModelState *state = (D41ModelState *)argument;
	D41EngramAhead *ahead = &state->engram_ahead;
	pthread_mutex_lock(&ahead->lock);
	for ( ;; )
	{
		while ( ahead->stop == 0u && ahead->pending == 0u )
			pthread_cond_wait(&ahead->wake, &ahead->lock);
		if ( ahead->stop != 0u )
			break;
		const uint32_t which = (ahead->pending & 1u) != 0u ? 0u : 1u, rows = ahead->rows;
		ahead->pending &= ~(1u << which);
		ahead->busy = 1u;
		pthread_mutex_unlock(&ahead->lock);
		const SparkStatus status = SparkRowFileFetch(state->engram_file[which], state->engram_ahead_ids_host[which],
			(uint64_t)rows * D41_ENGRAM_COLUMNS, (int64_t)state->engram[which].first_row, state->engram_ahead_rows_host[which], D41_ENGRAM_HEAD_DIM);
		pthread_mutex_lock(&ahead->lock);
		ahead->busy = 0u;
		ahead->finished |= 1u << which;
		if ( status != SPARK_STATUS_OK )
			ahead->failed |= 1u << which;
		pthread_cond_broadcast(&ahead->done);
	}
	pthread_mutex_unlock(&ahead->lock);
	return 0;
}

static void CUDART_CB D41EngramAheadStart(void *argument)
{
	const D41EngramCall *call = (const D41EngramCall *)argument;
	D41EngramAhead *ahead = &call->state->engram_ahead;
	pthread_mutex_lock(&ahead->lock);
	while ( ahead->busy != 0u )
		pthread_cond_wait(&ahead->done, &ahead->lock);
	ahead->rows = call->rows;
	ahead->pending = 3u;
	ahead->finished = 0u;
	ahead->failed = 0u;
	pthread_cond_signal(&ahead->wake);
	pthread_mutex_unlock(&ahead->lock);
}

static void CUDART_CB D41EngramAheadJoin(void *argument)
{
	const D41EngramCall *call = (const D41EngramCall *)argument;
	D41EngramAhead *ahead = &call->state->engram_ahead;
	pthread_mutex_lock(&ahead->lock);
	while ( (ahead->finished & (1u << call->which)) == 0u )
		pthread_cond_wait(&ahead->done, &ahead->lock);
	if ( (ahead->failed & (1u << call->which)) != 0u )
		call->state->engram_failed = 1u;
	pthread_mutex_unlock(&ahead->lock);
}

static void D41EngramHash(D41ModelState *state, uint32_t which, const SparkStageRunnerStep *step, int64_t *ids, cudaStream_t stream)
{
	const D41Engram *engram = &state->engram[which];
	LmNgramHashKernel<<<(step->rows + 63u) / 64u, 64u, 0, stream>>>(step->token_ids, state->token_map, step->positions,
		state->row_lane, state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index,
		(const int32_t *)(state->recurrent + state->ring_bytes + state->compressor_bytes), (uint32_t)(state->recurrent_bytes / sizeof(int32_t)),
		engram->multipliers, engram->primes, engram->offsets, ids, step->rows, D41_ENGRAM_ORDERS, D41_ENGRAM_HEADS, state->engram_pad, -1, 0u);
}

static D41EngramCall *D41EngramCallOf(D41ModelState *state, uint32_t which, uint32_t rows)
{
	D41EngramCall *call = &state->engram_calls[(uint64_t)which * (state->max_rows + 1u) + rows];
	call->state = state;
	call->which = which;
	call->rows = rows;
	return call;
}

static SparkStatus D41EngramAheadBegin(D41ModelState *state, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	state->engram_ahead_step = 0u;
	if ( state->engram_ahead.started == 0u || step->rows <= SPARK_STAGE_RUNNER_GRAPH_ROWS )
		return SPARK_STATUS_OK;
	for ( uint32_t which = 0u; which < 2u; ++which )
		D41EngramHash(state, which, step, state->engram_ahead_ids[which], stream);
	if ( cudaLaunchHostFunc(stream, D41EngramAheadStart, D41EngramCallOf(state, 0u, step->rows)) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->engram_ahead_step = step->rows;
	return D41Launched();
}

static void D41NgramHistory(D41ModelState *state, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	LmNgramHistoryKernel<<<(step->sequences + 63u) / 64u, 64u, 0, stream>>>(step->token_ids, state->token_map, step->sequence_row_begin,
		step->sequence_row_indices, step->recurrent_index,
		(int32_t *)(state->recurrent + state->ring_bytes + state->compressor_bytes), (uint32_t)(state->recurrent_bytes / sizeof(int32_t)),
		step->sequences, D41_ENGRAM_ORDERS, step->commit, (const uint32_t *)0);
}

static SparkStatus D41EngramPlanExchange(D41ModelState *state, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	uint32_t *counts_host = state->exchange_counts_host + D41_TP * D41_TP;
	for ( uint32_t which = 0u; which < 2u; ++which )
		D41EngramSlotsKernel<D41_TP><<<D41_TP, 1024u, 0, stream>>>(which == 0u ? state->engram_ids : state->engram_ids_next, step->rows,
			state->own_rows, D41_ENGRAM_COLUMNS, state->engram[which].part, state->engram[which].table_rows,
			state->engram_slot + (uint64_t)which * state->max_rows * D41_ENGRAM_COLUMNS, state->engram_counts + which * D41_TP * D41_TP);
	if ( D41Launched() != SPARK_STATUS_OK ||
		cudaMemcpyAsync(counts_host, state->engram_counts, 2u * D41_TP * D41_TP * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
		cudaStreamSynchronize(stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	for ( uint32_t which = 0u; which < 2u; ++which )
	{
		uint32_t capacity = 0u;
		for ( uint32_t index = 0u; index < D41_TP * D41_TP; ++index )
			capacity = counts_host[which * D41_TP * D41_TP + index] > capacity ? counts_host[which * D41_TP * D41_TP + index] : capacity;
		capacity = capacity != 0u ? (capacity + 7u) / 8u * 8u : 8u;
		if ( capacity > state->own_rows * D41_ENGRAM_COLUMNS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		state->engram_capacity[which] = capacity;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus D41EngramExchange(D41ModelState *state, uint32_t which, const int64_t *ids, const SparkStageRunnerStep *step,
	cudaStream_t stream)
{
	const D41Engram *engram = &state->engram[which];
	const uint32_t own = state->own_rows, capacity = state->engram_capacity[which];
	const uint32_t *slot = state->engram_slot + (uint64_t)which * state->max_rows * D41_ENGRAM_COLUMNS;
	SparkStatus status;
	D41EngramPackKernel<<<step->rows, D41_THREADS, 0, stream>>>(ids, slot, engram->payload, engram->scale, own, D41_ENGRAM_COLUMNS,
		engram->first_row, engram->local_rows, capacity, state->engram_send);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, capacity, D41K_ENGRAM_RECORD_BYTES / 2u,
			state->engram_send, state->engram_received);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41EngramUnpackKernel<<<dim3(own, D41_ENGRAM_COLUMNS), D41_THREADS, 0, stream>>>(ids, slot, state->engram_received, step->rows,
		state->own_first, D41_ENGRAM_COLUMNS, engram->part, engram->table_rows, capacity, state->engram_kv_local);
	if ( D41QdqLinear(&engram->wkv_whole, state->engram_kv_local, state->engram_kv, own, D41_ENGRAM_EMBED, D41_ENGRAM_KV_ROWS,
		step->multiprocessors, state->row_invariant, &state->mx, stream) != LM_LAUNCH_OK ||
		SparkDsv41FlashLaunchEngramGate(stream, state->streams + (uint64_t)state->own_first * D41K_HC * D41_HIDDEN, state->engram_kv,
		engram->q_weight, engram->k_weight, own, D41K_HC, D41_HIDDEN, D41_EPS) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return D41Launched();
}

static SparkStatus D41EngramApply(D41ModelState *state, uint32_t which, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t rows = step->rows;
	const D41Engram *engram = &state->engram[which];
	const uint32_t split = rows <= D41_ENGRAM_SPLIT_ROWS ? 1u : 0u;
	const uint32_t ar_rows = split != 0u ? 2u * rows : (rows * D41_ENGRAM_EMBED + D41_HIDDEN - 1u) / D41_HIDDEN;
	const uint32_t ar_width = split != 0u ? D41_ENGRAM_HALF : D41_HIDDEN;
	const uint32_t ahead = state->engram_ahead_step == rows ? 1u : 0u;
	const uint32_t own = D41_WHOLE && split == 0u && state->row_invariant == 0u ? state->own_rows : 0u;
	const int64_t *ids = !D41_ENGRAM_DISK && which != 0u ? state->engram_ids_next : state->engram_ids;
	SparkStatus status;
	if ( ahead == 0u && (D41_ENGRAM_DISK || which == 0u) )
		D41EngramHash(state, which, step, state->engram_ids, stream);
	if ( !D41_ENGRAM_DISK && which == 0u )
	{
		D41EngramHash(state, 1u, step, state->engram_ids_next, stream);
		D41NgramHistory(state, step, stream);
		if ( own != 0u && D41EngramPlanExchange(state, step, stream) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	if ( !D41_ENGRAM_DISK && own != 0u )
		return D41EngramExchange(state, which, ids, step, stream);
	cudaMemsetAsync(state->engram_embed, 0, own != 0u ? (uint64_t)own * D41_TP * D41_ENGRAM_EMBED * sizeof(uint16_t) :
		(uint64_t)ar_rows * ar_width * sizeof(uint16_t), stream);
	if ( ahead != 0u )
	{
		if ( cudaLaunchHostFunc(stream, D41EngramAheadJoin, D41EngramCallOf(state, which, rows)) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmNgramGatherFetchedE8m0Kernel<<<dim3(rows, D41_ENGRAM_COLUMNS), D41_THREADS, 0, stream>>>(state->engram_ahead_ids[which],
			state->engram_ahead_rows[which], engram->scale, state->engram_embed, engram->first_row, engram->local_rows, D41_ENGRAM_COLUMNS,
			D41_ENGRAM_HEAD_DIM);
	}
	else if ( D41_ENGRAM_DISK )
	{
		if ( cudaLaunchHostFunc(stream, D41EngramFetch, D41EngramCallOf(state, which, rows)) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmNgramGatherFetchedE8m0Kernel<<<dim3(rows, D41_ENGRAM_COLUMNS), D41_THREADS, 0, stream>>>(state->engram_ids, state->engram_rows, engram->scale,
			state->engram_embed, engram->first_row, engram->local_rows, D41_ENGRAM_COLUMNS, D41_ENGRAM_HEAD_DIM);
	}
	else
		LmNgramGatherE8m0Kernel<<<dim3(rows, D41_ENGRAM_COLUMNS), D41_THREADS, 0, stream>>>(ids, engram->payload, engram->scale,
			state->engram_embed, engram->first_row, engram->local_rows, D41_ENGRAM_COLUMNS, D41_ENGRAM_HEAD_DIM);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK && own != 0u )
	{
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_REDUCE_SCATTER_SUM_BF16, 2u * own * D41_TP, D41_ENGRAM_HALF,
			state->engram_embed, state->engram_kv_local);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( D41QdqLinear(&engram->wkv_whole, state->engram_kv_local, state->engram_kv, own, D41_ENGRAM_EMBED, D41_ENGRAM_KV_ROWS,
			step->multiprocessors, state->row_invariant, &state->mx, stream) != LM_LAUNCH_OK ||
			SparkDsv41FlashLaunchEngramGate(stream, state->streams + (uint64_t)state->own_first * D41K_HC * D41_HIDDEN, state->engram_kv,
			engram->q_weight, engram->k_weight, own, D41K_HC, D41_HIDDEN, D41_EPS) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return D41Launched();
	}
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, ar_rows, ar_width, state->engram_embed,
			state->engram_embed);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( D41QdqLinear(&engram->wkv, state->engram_embed, state->engram_kv_local, rows, D41_ENGRAM_EMBED, D41_ENGRAM_KV_LOCAL,
		step->multiprocessors, state->row_invariant, &state->mx, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows * D41_ENGRAM_KV_PARTS,
		D41_ENGRAM_KV_LOCAL / D41_ENGRAM_KV_PARTS, state->engram_kv_local, state->engram_kv_gathered);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41GroupInputKernel<<<dim3(D41Blocks(D41_ENGRAM_KV_ROWS), rows), D41_THREADS, 0, stream>>>(state->engram_kv_gathered, state->engram_kv,
		rows, D41_ENGRAM_KV_LOCAL, 0u, D41_TP);
	if ( state->own_pad != 0u )
		cudaMemsetAsync(state->engram_kv + (uint64_t)rows * D41_ENGRAM_KV_ROWS, 0, (uint64_t)state->own_pad * D41_ENGRAM_KV_ROWS * sizeof(uint16_t), stream);
	if ( SparkDsv41FlashLaunchEngramGate(stream, state->streams + (uint64_t)state->own_first * D41K_HC * D41_HIDDEN,
		state->engram_kv + (uint64_t)state->own_first * D41_ENGRAM_KV_ROWS, engram->q_weight, engram->k_weight,
		state->own_rows != 0u ? state->own_rows : rows, D41K_HC, D41_HIDDEN, D41_EPS) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return D41Launched();
}

template<class Geometry>
static void D41StoreCompressed(D41ModelState *state, uint32_t cache, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	D41StoreCompressedKernel<Geometry><<<step->rows, D41_THREADS, 0, stream>>>(state->latent_views[cache], state->c_latent_store,
		state->idx_key, state->emit_row, step->sequence_of_row, D41_INDEX_DIM);
}

static SparkStatus D41Compress(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step,
	cudaStream_t stream)
{
	const uint32_t rows = step->rows, ratio = D41Ratio(index), cache = D41CacheIndex(index), sms = step->multiprocessors;
	float *compressor_state = (float *)(state->recurrent + state->ring_bytes) + (uint64_t)cache * 2u * D41_HEAD_DIM;
	const uint32_t state_stride = (uint32_t)(state->recurrent_bytes / sizeof(float));
	if ( D41_ROWS && state->own_rows != 0u && state->x_complete == 0u )
	{
		const uint32_t own = state->own_rows;
		float *own_kv = state->own_c, *own_score = state->own_c + (uint64_t)own * D41_HEAD_DIM;
		SparkStatus status;
		if ( D41Project(layer->c_wkv, state->own_x, 0, own_kv, own, D41_HIDDEN, D41_HEAD_DIM, sms, state->row_invariant, stream) != LM_LAUNCH_OK ||
			(ratio > 1u && D41Project(layer->c_wgate, state->own_x, 0, own_score, own, D41_HIDDEN, D41_HEAD_DIM, sms, state->row_invariant, stream) !=
			LM_LAUNCH_OK) )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = D41Launched();
		if ( status == SPARK_STATUS_OK )
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, own, 2u * D41_HEAD_DIM, own_kv, state->c_kv);
		if ( status == SPARK_STATUS_OK && ratio > 1u )
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, own, 2u * D41_HEAD_DIM, own_score, state->c_score);
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	else if ( D41Project(layer->c_wkv, state->x, 0, state->c_kv, rows, D41_HIDDEN, D41_HEAD_DIM, sms, state->row_invariant, stream) != LM_LAUNCH_OK ||
		(ratio > 1u && D41Project(layer->c_wgate, state->x, 0, state->c_score, rows, D41_HIDDEN, D41_HEAD_DIM, sms, state->row_invariant, stream) != LM_LAUNCH_OK) )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41CompressorRowsKernel<<<rows, D41_THREADS, 0, stream>>>(state->c_kv, state->c_score, state->c_latent, state->emit_row,
		compressor_state, step->positions, state->row_lane, state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices,
		step->recurrent_index, ratio, state_stride, D41_HEAD_DIM);
	if ( ratio > 1u && step->commit == 0u )
	{
		if ( rows > D41_VERIFY_ROWS || cache >= D41_PAIR_CACHES )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		cudaMemcpyAsync(state->verify_c_kv + (uint64_t)cache * D41_VERIFY_ROWS * D41_HEAD_DIM, state->c_kv,
			(uint64_t)rows * D41_HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice, stream);
		cudaMemcpyAsync(state->verify_c_score + (uint64_t)cache * D41_VERIFY_ROWS * D41_HEAD_DIM, state->c_score,
			(uint64_t)rows * D41_HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice, stream);
	}
	if ( ratio > 1u )
		D41CompressorSaveKernel<<<rows, D41_THREADS, 0, stream>>>(state->c_kv, state->c_score, compressor_state, step->positions,
			state->row_lane, state->row_ordinal, step->sequence_row_begin, step->recurrent_index, rows, ratio, state_stride, D41_HEAD_DIM,
			step->commit, (const uint32_t *)0);
	D41RmsNorm(state->c_latent, layer->c_norm, state->c_latent, rows, D41_HEAD_DIM, stream);
	if ( D41Project(layer->idx_wk, state->c_latent, state->idx_key, 0, rows, D41_HEAD_DIM, D41_INDEX_DIM, sms, state->row_invariant, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41RmsNorm(state->idx_key, layer->idx_kn, state->idx_key, rows, D41_INDEX_DIM, stream);
	D41Rope(state->idx_key, rows, 1u, D41_INDEX_DIM, D41_INDEX_DIM - D41_ROPE, step->positions, 1 - (int32_t)ratio,
		state->compress_frequency, 0u, state->emit_row, stream);
	SparkDsv41FlashFp4Pow2QdqKernel<<<(rows * (D41_INDEX_DIM / 32u) + 255u) / 256u, 256u, 0, stream>>>(state->idx_key,
		rows * (D41_INDEX_DIM / 32u), 32u);
	cudaMemcpyAsync(state->c_latent_store, state->c_latent, (uint64_t)rows * D41_HEAD_DIM * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
	D41Rope(state->c_latent_store, rows, 1u, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 1 - (int32_t)ratio,
		state->compress_frequency, 0u, state->emit_row, stream);
	(void)SparkDsv41FlashLaunchKvFp4Qdq(stream, state->c_latent_store, D41_HEAD_DIM, rows, D41_HEAD_DIM);
	if ( cache < 3u )
		D41StoreCompressed<D41Ratio2Kv>(state, cache, step, stream);
	else
		D41StoreCompressed<D41Ratio1Kv>(state, cache, step, stream);
	return D41Launched();
}

static SparkStatus D41CandidateBlocks(D41ModelState *state, const SparkStageRunnerStep *step, const uint32_t *bound,
	const uint32_t *length, cudaStream_t stream)
{
	const uint32_t rows = step->rows, keep = state->local_block_keep;
	const SparkKvShard blocks = { D41_TP, state->shard.rank, 1u };
	SparkStatus status;
	if ( SparkDsv41FlashLaunchShardBlockMax(stream, state->local_scores, state->step_score_stride, state->shard, step->sequence_of_row,
			length, bound, state->block_scores, state->local_block_stride, rows) != cudaSuccess ||
		LmTopkExactLaunch<D41_THREADS>(state->block_scores, rows, state->local_block_stride, keep, LM_TOPK_EXACT_CHUNK,
			LM_TOPK_EXACT_CHUNKED_ROWS, state->topk_values, state->topk_positions, state->topk_entries, state->block_local_selected, stream) != cudaSuccess ||
		LmIndexShardCandidatePackLaunch<D41_THREADS>(blocks, state->block_scores, state->local_block_stride, state->block_local_selected,
			keep, rows, state->block_candidates, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for ( uint32_t first = 0u; first < rows; first += D41_SELECTION_CHUNK )
	{
		const uint32_t count = rows - first < D41_SELECTION_CHUNK ? rows - first : D41_SELECTION_CHUNK;
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, count * state->block_round_parts,
			keep * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)) / state->block_round_parts, state->block_candidates + (uint64_t)first * keep,
			state->block_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmIndexShardCandidateTopkLaunch<D41_THREADS,D41_CANDIDATE_BLOCKS>(state->block_gathered, (uint64_t)count * keep, D41_TP, keep, count,
				state->global_block_stride, state->global_block_keep, 0u, state->block_selected, stream) != cudaSuccess ||
			SparkDsv41FlashLaunchShardBlockMask(stream, state->global_block_stride, state->block_selected, state->global_block_keep,
				blocks, state->block_mask + (uint64_t)first * state->local_block_stride, state->local_block_stride, count) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	return D41Launched();
}

static SparkStatus D41IndexQuery(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step,
	uint16_t *packed, cudaStream_t stream)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors;
	const uint32_t local_width = D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM;
	if ( state->candidate_blocks != 0u && index >= D41_CANDIDATE_SOURCE && D41Ratio(index) != 1u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( D41Project(layer->idx_wp, state->x, state->idx_w_full, 0, rows, D41_HIDDEN, D41_LOCAL_INDEX_HEADS, sms, state->row_invariant, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41Rope(state->idx_pack_local, rows, D41_LOCAL_INDEX_HEADS, D41_INDEX_DIM, D41_INDEX_DIM - D41_ROPE, step->positions, 0,
		state->compress_frequency, 0u, 0, stream);
	SparkDsv41FlashFp4Pow2QdqKernel<<<(rows * (local_width / 32u) + 255u) / 256u, 256u, 0, stream>>>(state->idx_pack_local,
		rows * (local_width / 32u), 32u);
	D41ScaleRowsKernel<<<D41Blocks((uint64_t)rows * D41_LOCAL_INDEX_HEADS), D41_THREADS, 0, stream>>>(state->idx_w_full,
		(uint64_t)rows * D41_LOCAL_INDEX_HEADS, (float)(1.0 / sqrt((double)D41_INDEX_DIM) / sqrt((double)D41_INDEX_HEADS)));
	D41IndexPackKernel<<<rows, D41_THREADS, 0, stream>>>(state->idx_pack_local, state->idx_w_full, packed, local_width,
		D41_LOCAL_INDEX_HEADS, D41_INDEX_PACK);
	return D41Launched();
}

template<class Geometry>
static SparkStatus D41Select(D41ModelState *state, uint32_t index, const SparkStageRunnerStep *step, const uint32_t *bound,
	const uint32_t *length, const uint16_t *packed, uint64_t rank_stride, cudaStream_t stream)
{
	const uint32_t rows = step->rows, cache = D41CacheIndex(index);
	const uint32_t local_width = D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM;
	SparkStatus status;
	D41IndexUnpackKernel<<<dim3(rows, D41_TP), D41_THREADS, 0, stream>>>(packed, rank_stride, state->idx_query, state->idx_w_full,
		local_width, D41_LOCAL_INDEX_HEADS, D41_INDEX_PACK);
	if ( LmWeightedSparseScoreLaunch<Geometry,D41_INDEX_HEADS,D41_INDEX_DIM,LmKvShardView>(state->idx_query, state->idx_w_full,
		state->index_views[cache], step->sequence_of_row, length, bound, rows, state->step_score_stride, 1.0f, state->local_scores,
		stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->step_blocks != 0u && index == D41_CANDIDATE_SOURCE )
	{
		status = D41CandidateBlocks(state, step, bound, length, stream);
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	else if ( state->step_blocks != 0u && index > D41_CANDIDATE_SOURCE &&
		SparkDsv41FlashLaunchShardBlockApply(stream, state->local_scores, state->step_score_stride, D41_GRAIN, state->block_mask,
			state->local_block_stride, rows) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( LmTopkExactLaunch<D41_THREADS>(state->local_scores, rows, state->step_score_stride, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK,
			LM_TOPK_EXACT_CHUNKED_ROWS, state->topk_values, state->topk_positions, state->topk_entries, state->local_selected, stream) != cudaSuccess ||
		LmIndexShardCandidatePackLaunch<D41_THREADS>(state->index_views[cache].shard, state->local_scores, state->step_score_stride,
			state->local_selected, D41_INDEX_TOPK, rows, state->candidates, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows,
		D41_INDEX_TOPK * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)), state->candidates, state->candidates_gathered);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( LmIndexShardCandidateTopkLaunch<D41_THREADS,D41_INDEX_TOPK>(state->candidates_gathered, (uint64_t)rows * D41_INDEX_TOPK, D41_TP,
			D41_INDEX_TOPK, rows, state->global_score_stride, D41_INDEX_TOPK, 1u, state->selected, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return D41Launched();
}

template<class Geometry>
static SparkStatus D41SelectWhole(D41ModelState *state, uint32_t index, const SparkStageRunnerStep *step, const uint32_t *bound,
	const uint32_t *length, cudaStream_t stream)
{
	const uint32_t rows = step->rows, cache = D41CacheIndex(index), keep = state->local_block_keep;
	const uint32_t local_width = D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM;
	const uint32_t stride = D41Ratio(index) == 2u ? state->step_ratio2_stride : state->step_score_stride;
	const uint32_t own = state->own_rows, begin = state->own_first, end = own != 0u && begin + own < rows ? begin + own : rows;
	const SparkKvShard blocks = { 1u, 0u, 1u };
	const uint32_t unpacked = D41_ROWS && own != 0u ? own : rows, unpack_first = D41_ROWS && own != 0u ? begin : 0u;
	D41IndexUnpackKernel<<<dim3(unpacked, D41_ROWS ? 1u : D41_TP), D41_THREADS, 0, stream>>>(state->q_gathered, (uint64_t)unpacked * D41_INDEX_PACK,
		state->idx_query + (uint64_t)unpack_first * D41_INDEX_HEADS * D41_INDEX_DIM, state->idx_w_full + (uint64_t)unpack_first * D41_INDEX_HEADS, local_width,
		D41_LOCAL_INDEX_HEADS, D41_INDEX_PACK);
	const uint64_t fits = (uint64_t)state->select_rows * state->local_score_stride / stride;
	const uint32_t chunk = D41Ratio(index) != 2u ? state->select_rows : fits < (uint64_t)state->max_rows ? (uint32_t)fits : state->max_rows;
	for ( uint32_t first = begin; first < end; first += chunk )
	{
		const uint32_t count = end - first < chunk ? end - first : chunk;
		const uint32_t *sequences = step->sequence_of_row + first, *bounds = bound + first;
		uint32_t *chosen = state->local_selected + (uint64_t)first * D41_INDEX_TOPK;
		uint32_t *selected = own != 0u ? state->own_selected + (uint64_t)(first - begin) * D41_INDEX_TOPK : state->selected + (uint64_t)first * D41_INDEX_TOPK;
		uint2 *candidates = state->candidates + (uint64_t)first * D41_INDEX_TOPK;
		if ( LmWeightedSparseScoreLaunch<Geometry,D41_INDEX_HEADS,D41_INDEX_DIM,LmKvShardView>(
			state->idx_query + (uint64_t)first * D41_INDEX_HEADS * D41_INDEX_DIM, state->idx_w_full + (uint64_t)first * D41_INDEX_HEADS,
			state->index_views[cache], sequences, length, bounds, count, stride, 1.0f, state->local_scores, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( state->step_blocks != 0u && index == D41_CANDIDATE_SOURCE )
		{
			uint32_t *block_chosen = state->block_local_selected + (uint64_t)first * keep;
			uint2 *block_candidates = state->block_candidates + (uint64_t)first * keep;
			if ( SparkDsv41FlashLaunchShardBlockMax(stream, state->local_scores, stride, state->shard, sequences, length, bounds, state->block_scores,
					state->local_block_stride, count) != cudaSuccess ||
				LmTopkExactLaunch<D41_THREADS>(state->block_scores, count, state->local_block_stride, keep, LM_TOPK_EXACT_CHUNK,
					LM_TOPK_EXACT_CHUNKED_ROWS, state->topk_values, state->topk_positions, state->topk_entries, block_chosen, stream) != cudaSuccess ||
				LmIndexShardCandidatePackLaunch<D41_THREADS>(blocks, state->block_scores, state->local_block_stride, block_chosen, keep, count,
					block_candidates, stream) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			for ( uint32_t inner = 0u; inner < count; inner += D41_SELECTION_CHUNK )
			{
				const uint32_t part = count - inner < D41_SELECTION_CHUNK ? count - inner : D41_SELECTION_CHUNK;
				if ( LmIndexShardCandidateTopkLaunch<D41_THREADS,D41_CANDIDATE_BLOCKS>(block_candidates + (uint64_t)inner * keep, (uint64_t)part * keep, 1u,
						keep, part, state->global_block_stride, state->global_block_keep, 0u, state->block_selected, stream) != cudaSuccess ||
					SparkDsv41FlashLaunchShardBlockMask(stream, state->global_block_stride, state->block_selected, state->global_block_keep, blocks,
						state->block_mask + (uint64_t)(first + inner) * state->local_block_stride, state->local_block_stride, part) != cudaSuccess )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		else if ( state->step_blocks != 0u && index > D41_CANDIDATE_SOURCE &&
			SparkDsv41FlashLaunchShardBlockApply(stream, state->local_scores, stride, D41_GRAIN,
				state->block_mask + (uint64_t)first * state->local_block_stride, state->local_block_stride, count) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( LmTopkExactLaunch<D41_THREADS>(state->local_scores, count, stride, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS,
				state->topk_values, state->topk_positions, state->topk_entries, chosen, stream) != cudaSuccess ||
			LmIndexShardCandidatePackLaunch<D41_THREADS>(state->index_views[cache].shard, state->local_scores, stride, chosen, D41_INDEX_TOPK, count,
				candidates, stream) != cudaSuccess ||
			LmIndexShardCandidateTopkLaunch<D41_THREADS,D41_INDEX_TOPK>(candidates, (uint64_t)count * D41_INDEX_TOPK, 1u, D41_INDEX_TOPK, count,
				state->global_score_stride, D41_INDEX_TOPK, 1u, selected, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	if ( !D41_ROWS && own != 0u && D41Launched() == SPARK_STATUS_OK )
		return CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, own, D41_INDEX_TOPK * (uint32_t)(sizeof(uint32_t) / sizeof(uint16_t)),
			state->own_selected, state->selected);
	return D41Launched();
}

static uint64_t D41DestinationStride(uint32_t rows, uint32_t record_heads)
{
	return (uint64_t)rows * record_heads * D41K_RECORD;
}

static uint32_t D41PairRecords(const D41ModelState *state, uint32_t rows)
{
	return rows <= D41_PAIR_RECORD_CHOSEN_ROWS && 2u * rows <= state->max_rows ? 1u : 0u;
}

static uint32_t D41LandingClass(uint32_t cache)
{
	return cache < SPARK_DSV41_FLASH_STAGE_RATIO2_CACHES ? 0u : 1u;
}

template<class Geometry>
static SparkStatus D41LandCache(D41ModelState *state, uint32_t cache, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t landing = D41LandingClass(cache);
	uint8_t *send = (uint8_t *)state->q_gathered;
	D41GatherPlanKernel<<<1u, 1u, 0, stream>>>(state->gather_plan, state->slots, step->sequence_of_row,
		landing == 0u ? state->length_ratio2 : state->length_ratio1);
	if ( D41Launched() != SPARK_STATUS_OK ||
		LmKvShardGatherPackRangeLaunch<Geometry,D41_THREADS>(state->latent_views[cache], state->gather_plan, state->gather_plan + state->slots,
			state->gather_plan + 2u * state->slots, 1u, 0u, state->landing_keys[landing], send, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, state->landing_rows[landing], D41_HIDDEN, send, state->landing);
}

template<class Geometry>
static SparkStatus D41LocalPartials(D41ModelState *state, uint32_t index, const SparkStageRunnerStep *step, const uint32_t *bound,
	const uint32_t *length, cudaStream_t stream)
{
	const uint32_t rows = step->rows, cache = D41CacheIndex(index), landing = D41LandingClass(cache);
	const uint64_t stride = D41DestinationStride(rows, D41_LOCAL_HEADS);
	if ( LmLatentShardSplitLaunch(state->latent_views[cache].shard, state->selected, D41_INDEX_TOPK, bound, state->split_selected, state->split_offsets,
			rows, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for ( uint32_t source = 0u; source < D41_TP; ++source )
	{
		LmKvShardGatherView view;
		SparkKvShard shard = state->latent_views[cache].shard;
		shard.rank = source;
		if ( LmKvShardGatherViewInitialize<Geometry>(&view, state->latent_views[cache].pages, shard,
				state->landing + (uint64_t)source * state->landing_stride[landing], state->gather_plan + state->slots, state->gather_plan + 2u * state->slots) != 0 ||
			LmKvShardGatherViewWindow(&view, 0u, state->landing_keys[landing]) != 0 ||
			LmLatentShardAttendHeadsLaunch<Geometry,LmKvShardGatherView,D41_HEAD_DIM,0u>(view, state->q_local,
				(uint64_t)rows * D41_LOCAL_HEADS * D41_HEAD_DIM, D41_LOCAL_HEADS, D41_LOCAL_HEADS, step->sequence_of_row, length, bound, state->selected,
				D41_INDEX_TOPK, D41_INDEX_TOPK, 1.0f / sqrtf((float)D41_HEAD_DIM), state->records_received + (uint64_t)source * stride, stride,
				D41_LOCAL_HEADS, 1u, rows, stream, state->split_selected, state->split_offsets, 1u) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	return D41Launched();
}

template<class Geometry>
static SparkStatus D41Partials(D41ModelState *state, uint32_t index, const SparkStageRunnerStep *step, const uint32_t *bound,
	const uint32_t *length, uint64_t query_rank_stride, uint32_t record_heads, uint32_t record_copies, cudaStream_t stream)
{
	const uint32_t rows = step->rows, cache = D41CacheIndex(index);
	if ( LmLatentShardAttendHeadsLaunch<Geometry,LmKvShardView,D41_HEAD_DIM,0u>(state->latent_views[cache], state->q_gathered,
		query_rank_stride, D41_LOCAL_HEADS, D41_HEADS, step->sequence_of_row, length, bound, state->selected,
		D41_INDEX_TOPK, D41_INDEX_TOPK, 1.0f / sqrtf((float)D41_HEAD_DIM), state->records_send, D41DestinationStride(rows, record_heads), record_heads,
		record_copies, rows, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return D41Launched();
}

static SparkStatus D41OwnerOutputWeights(D41ModelState *state, cudaStream_t stream)
{
	const uint64_t slice = (uint64_t)D41_HIDDEN * D41_WO_A_ROWS, slice_scale = (uint64_t)D41_HIDDEN * (D41_WO_A_ROWS / D41_FP8_BLOCK);
	SparkStatus status = SPARK_STATUS_OK;
	for ( uint32_t index = 0u; index < D41_LAYERS && status == SPARK_STATUS_OK; ++index )
	{
		const D41Fp8Linear *o_b = &state->layers[index].o_b;
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, D41_HIDDEN, D41_WO_A_ROWS / 2u, o_b->weight,
			state->o_b_all + (uint64_t)index * D41_TP * slice);
		if ( status == SPARK_STATUS_OK )
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, D41_HIDDEN, D41_WO_A_ROWS / D41_FP8_BLOCK / 2u,
				o_b->scale, state->o_b_all_scale + (uint64_t)index * D41_TP * slice_scale);
	}
	if ( status == SPARK_STATUS_OK )
		state->o_b_ready = 1u;
	return status;
}

static SparkStatus D41OutputToOwners(D41ModelState *state, uint32_t index, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	const uint32_t own = state->own_rows;
	const uint64_t block = (uint64_t)own * D41_OWNER_OUTPUT_ELEMENTS * sizeof(uint16_t), codes_bytes = (uint64_t)own * D41_WO_A_ROWS;
	const uint64_t slice = (uint64_t)D41_HIDDEN * D41_WO_A_ROWS, slice_scale = (uint64_t)D41_HIDDEN * (D41_WO_A_ROWS / D41_FP8_BLOCK);
	uint8_t *send = (uint8_t *)state->moe_send;
	const uint8_t *received = (const uint8_t *)state->moe_received;
	SparkStatus status;
	if ( state->own_pad != 0u )
		cudaMemsetAsync(state->wo_a_out + (uint64_t)rows * D41_WO_A_ROWS, 0, (uint64_t)state->own_pad * D41_WO_A_ROWS * sizeof(uint16_t), stream);
	for ( uint32_t owner = 0u; owner < D41_TP; ++owner )
		if ( LmMxQuantize(state->wo_a_out + (uint64_t)owner * own * D41_WO_A_ROWS, send + owner * block, send + owner * block + codes_bytes, codes_bytes,
			stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, own, D41_OWNER_OUTPUT_ELEMENTS, state->moe_send,
			state->moe_received);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( uint32_t source = 0u; source < D41_TP; ++source )
	{
		const uint64_t at = (uint64_t)index * D41_TP + source;
		if ( LmMxGemmDense(received + source * block, received + source * block + codes_bytes, state->o_b_all + at * slice,
			state->o_b_all_scale + at * slice_scale, state->partial + (uint64_t)source * own * D41_HIDDEN, own, D41_WO_A_ROWS, D41_HIDDEN, 0u, 0u, sms,
			stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	D41SumSourcesKernel<<<(own * D41_HIDDEN / 2u + D41_THREADS - 1u) / D41_THREADS, D41_THREADS, 0, stream>>>(state->partial, state->own_out,
		own * D41_HIDDEN / 2u, (uint64_t)own * D41_HIDDEN / 2u, D41_TP);
	state->owner_reduced = 1u;
	return D41Launched();
}

#define D41_SPREAD_HEADS (D41_HEADS / D41_TP)
#define D41_SPREAD (D41_ROWS && D41_HEADS % D41_TP == 0u && (D41_HEADS / D41_OGROUPS) % D41_SPREAD_HEADS == 0u && D41_WO_A_ROWS % D41_TP == 0u && \
	D41_HIDDEN % D41_TP == 0u)

static uint32_t D41SpreadRow(const D41ModelState *state, const SparkStageRunnerStep *step)
{
	return D41_SPREAD && step->rows == 1u && step->commit != 0u && state->own_rows == 0u && state->dump == 0 && state->tap_slab == 0 ? 1u : 0u;
}

static D41Fp8Linear D41RowsOf(const D41Fp8Linear *linear, uint32_t first_row, uint32_t input_dimension)
{
	D41Fp8Linear part;
	part.weight = linear->weight + (uint64_t)first_row * input_dimension;
	part.scale = linear->scale + (uint64_t)first_row * (input_dimension / D41_FP8_BLOCK);
	part.wide = (const uint16_t *)0;
	return part;
}

static SparkStatus D41AttentionSpread(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step,
	cudaStream_t stream)
{
	const uint32_t sms = step->multiprocessors, ratio = D41Ratio(index), cache = D41CacheIndex(index);
	const uint32_t heads = D41_SPREAD_HEADS, head_first = state->tp_rank * heads, width = heads * D41_HEAD_DIM;
	const uint32_t group_heads = D41_HEADS / D41_OGROUPS, group = head_first / group_heads;
	const uint32_t o_a_rows = D41_WO_A_ROWS / D41_TP, o_b_rows = D41_HIDDEN / D41_TP;
	const uint32_t o_a_first = (head_first % group_heads) / heads * o_a_rows;
	const float *frequency = ratio > 0u ? state->compress_frequency : state->full_frequency;
	const uint32_t *bound = ratio == 2u ? state->bound_ratio2 : state->bound_ratio1;
	const uint32_t *length = ratio == 2u ? state->length_ratio2 : state->length_ratio1;
	const uint32_t indexed = ratio > 0u && D41IndexSource(index) ? 1u : 0u;
	const uint64_t stride = D41DestinationStride(1u, heads);
	const float qk_scale = 1.0f / sqrtf((float)D41_HEAD_DIM);
	uint16_t *ring = (uint16_t *)state->recurrent + (uint64_t)index * D41_RING_ROWS * D41_HEAD_DIM;
	uint16_t *packed = state->q_local + (uint64_t)D41_LOCAL_HEADS * D41_HEAD_DIM;
	uint16_t *group_input = state->group_input + (uint64_t)group * group_heads * D41_HEAD_DIM;
	const D41Fp8Linear q_part = D41RowsOf(&layer->q_b, head_first * D41_HEAD_DIM, D41_QLORA);
	const D41Fp8Linear o_a_part = D41RowsOf(&layer->o_a[group], o_a_first, D41_GROUP_WIDTH);
	const D41Fp8Linear o_b_part = D41RowsOf(&layer->o_b, state->tp_rank * o_b_rows, D41_WO_A_ROWS);
	SparkStatus status;
	static_assert(!D41_SPREAD || D41_LOCAL_HEADS == D41_HEADS, "a spread row starts from every head's weights on every rank");
	if ( D41LinearPair(&layer->q_a, &layer->kv_a, state->xq, state->q_latent, state->kv, 1u, D41_HIDDEN, D41_QLORA, D41_HEAD_DIM, sms,
		state->row_invariant, &state->mx, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41RmsNormQdq(state->q_latent, layer->q_norm, 0, state->qrq, 1u, D41_QLORA, stream);
	D41KvNormRopeQdqKernel<D41_THREADS><<<1u, D41_THREADS, (D41_HEAD_DIM + 8u) * sizeof(float), stream>>>(state->kv, layer->kv_norm,
		step->positions, frequency, state->window_kv, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, D41_EPS);
	if ( (indexed != 0u ? D41LinearPair(&q_part, &layer->idx_q_b, state->qrq, state->q_local, state->idx_pack_local, 1u, D41_QLORA, width,
		D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM, sms, state->row_invariant, &state->mx, stream) :
		D41Linear(&q_part, state->qrq, state->q_local, 1u, D41_QLORA, width, sms, state->row_invariant, &state->mx, stream)) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41Rope(state->q_local, 1u, heads, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 0u, 0, stream);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK && D41KvSource(index) && state->compress_done == 0u )
		status = D41Compress(state, layer, index, step, stream);
	if ( status == SPARK_STATUS_OK && indexed != 0u )
		status = D41IndexQuery(state, layer, index, step, packed, stream);
	if ( status == SPARK_STATUS_OK && indexed != 0u )
		status = cudaMemcpyAsync(state->q_gathered, packed, (uint64_t)D41_INDEX_PACK * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream) == cudaSuccess ?
			SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	if ( status == SPARK_STATUS_OK && indexed != 0u )
		status = cache < 3u ? D41SelectWhole<D41Ratio2Kv>(state, index, step, bound, length, stream) :
			D41SelectWhole<D41Ratio1Kv>(state, index, step, bound, length, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( ratio > 0u )
	{
		if ( (cache < 3u ? LmLatentShardAttendHeadsLaunch<D41Ratio2Kv,LmKvShardView,D41_HEAD_DIM,0u>(state->latent_views[cache], state->q_local,
				(uint64_t)width, heads, heads, step->sequence_of_row, length, bound, state->selected, D41_INDEX_TOPK, D41_INDEX_TOPK, qk_scale,
				state->records_received, stride, heads, 1u, 1u, stream) :
			LmLatentShardAttendHeadsLaunch<D41Ratio1Kv,LmKvShardView,D41_HEAD_DIM,0u>(state->latent_views[cache], state->q_local,
				(uint64_t)width, heads, heads, step->sequence_of_row, length, bound, state->selected, D41_INDEX_TOPK, D41_INDEX_TOPK, qk_scale,
				state->records_received, stride, heads, 1u, 1u, stream)) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	else
		D41InitRecords(state->records_received, stride, 1u, heads, 1u, stream);
	D41WindowFoldWholeKernel<<<dim3(1u, heads), D41K_THREADS, 0, stream>>>(state->records_received, heads, state->q_local, state->window_kv, ring,
		step->positions, state->row_lane, state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index,
		state->recurrent_bytes / sizeof(uint16_t), D41_WINDOW, qk_scale);
	D41WindowRingStoreKernel<<<1u, D41_THREADS, 0, stream>>>(state->window_kv, ring, step->positions, state->row_lane, state->row_ordinal,
		step->sequence_row_begin, step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_RING_ROWS, D41_WINDOW, 1u, 0u, 1u,
		step->commit, (const uint32_t *)0);
	D41MergeSinkKernel<<<dim3(1u, heads), D41_THREADS, 0, stream>>>(state->records_received, stride, 1u, heads, layer->sink + head_first,
		state->attn_merged);
	D41Rope(state->attn_merged, 1u, heads, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 1u, 0, stream);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, 1u, width, state->attn_merged, state->group_input);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( D41QdqLinear(&o_a_part, group_input, state->wo_a_group, 1u, D41_GROUP_WIDTH, o_a_rows, sms, state->row_invariant, &state->mx, stream) !=
		LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, 1u, o_a_rows, state->wo_a_group, state->wo_a_out);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( D41QdqLinear(&o_b_part, state->wo_a_out, state->wo_a_group, 1u, D41_WO_A_ROWS, o_b_rows, sms, state->row_invariant, &state->mx, stream) !=
		LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, 1u, o_b_rows, state->wo_a_group, state->partial);
	state->owner_reduced = 1u;
	return status;
}

static SparkStatus D41AttentionWhole(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step,
	cudaStream_t stream)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors, ratio = D41Ratio(index), cache = D41CacheIndex(index);
	const float *frequency = ratio > 0u ? state->compress_frequency : state->full_frequency;
	const uint32_t *bound = ratio == 2u ? state->bound_ratio2 : state->bound_ratio1;
	const uint32_t *length = ratio == 2u ? state->length_ratio2 : state->length_ratio1;
	const uint32_t local_width = D41_LOCAL_HEADS * D41_HEAD_DIM;
	const uint32_t indexed = ratio > 0u && D41IndexSource(index) ? 1u : 0u;
	const uint64_t stride = D41DestinationStride(rows, D41_LOCAL_HEADS), query_stride = (uint64_t)rows * local_width;
	const float qk_scale = 1.0f / sqrtf((float)D41_HEAD_DIM);
	const uint32_t own = state->own_rows, projected = own != 0u ? own : rows;
	uint16_t *ring = (uint16_t *)state->recurrent + (uint64_t)index * D41_RING_ROWS * D41_HEAD_DIM;
	uint16_t *packed = state->q_local + query_stride;
	uint16_t *own_qrq = state->own_q, *own_kv = state->own_q + (uint64_t)own * D41_QLORA;
	const uint32_t q_coded = own != 0u && D41MxReady(&state->mx, rows, D41_QLORA, state->row_invariant) != 0u ? 1u : 0u;
	SparkStatus status;
	if ( D41LinearPair(&layer->q_a, &layer->kv_a, own != 0u ? state->own_out : state->xq, state->q_latent, state->kv, projected, D41_HIDDEN, D41_QLORA,
		D41_HEAD_DIM, sms, state->row_invariant, &state->mx, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41RmsNormQdq(state->q_latent, layer->q_norm, 0, own != 0u ? own_qrq : state->qrq, projected, D41_QLORA, stream);
	D41KvNormRopeQdqKernel<D41_THREADS><<<projected, D41_THREADS, (D41_HEAD_DIM + 8u) * sizeof(float), stream>>>(state->kv, layer->kv_norm,
		step->positions + state->own_first, frequency, own != 0u ? own_kv : state->window_kv, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, D41_EPS);
	if ( own != 0u )
	{
		status = D41Launched();
		if ( status == SPARK_STATUS_OK && q_coded != 0u )
		{
			uint8_t *block = (uint8_t *)state->moe_send;
			if ( LmMxQuantize(own_qrq, block, block + (uint64_t)own * D41_QLORA, (uint64_t)own * D41_QLORA, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, own, D41_Q_CODE_ELEMENTS, state->moe_send,
				state->moe_received);
		}
		else if ( status == SPARK_STATUS_OK )
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, own, D41_QLORA, own_qrq, state->qrq);
		if ( status == SPARK_STATUS_OK )
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, own, D41_HEAD_DIM, own_kv, state->window_kv);
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	if ( q_coded != 0u )
	{
		const uint64_t segment = (uint64_t)own * D41_Q_CODE_ELEMENTS * sizeof(uint16_t);
		const uint32_t index_width = D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM;
		for ( uint32_t source = 0u; source < D41_TP; ++source )
		{
			const uint8_t *codes = (const uint8_t *)state->moe_received + source * segment, *scales = codes + (uint64_t)own * D41_QLORA;
			if ( LmMxGemmDense(codes, scales, layer->q_b.weight, layer->q_b.scale, state->q_local + (uint64_t)source * own * local_width, own, D41_QLORA,
				local_width, 0u, 0u, sms, stream) != LM_LAUNCH_OK ||
				(indexed != 0u && LmMxGemmDense(codes, scales, layer->idx_q_b.weight, layer->idx_q_b.scale,
				state->idx_pack_local + (uint64_t)source * own * index_width, own, D41_QLORA, index_width, 0u, 0u, sms, stream) != LM_LAUNCH_OK) )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
	}
	else if ( (indexed != 0u ? D41LinearPair(&layer->q_b, &layer->idx_q_b, state->qrq, state->q_local, state->idx_pack_local, rows, D41_QLORA,
		local_width, D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM, sms, state->row_invariant, &state->mx, stream) :
		D41Linear(&layer->q_b, state->qrq, state->q_local, rows, D41_QLORA, local_width, sms, state->row_invariant, &state->mx, stream)) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41Rope(state->q_local, rows, D41_LOCAL_HEADS, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 0u, 0, stream);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK && D41KvSource(index) && state->compress_done == 0u )
		status = D41Compress(state, layer, index, step, stream);
	if ( status == SPARK_STATUS_OK && indexed != 0u )
		status = D41IndexQuery(state, layer, index, step, packed, stream);
	if ( status == SPARK_STATUS_OK && indexed != 0u && D41_ROWS )
		status = cudaMemcpyAsync(state->q_gathered, packed, (uint64_t)rows * D41_INDEX_PACK * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream) ==
			cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	else if ( status == SPARK_STATUS_OK && indexed != 0u )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, D41_INDEX_PACK, packed, state->q_gathered);
	if ( status == SPARK_STATUS_OK && indexed != 0u )
		status = cache < 3u ? D41SelectWhole<D41Ratio2Kv>(state, index, step, bound, length, stream) :
			D41SelectWhole<D41Ratio1Kv>(state, index, step, bound, length, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	const uint32_t fused = rows > SPARK_STAGE_RUNNER_GRAPH_ROWS && D41_LOCAL_HEADS <= D41A_QUERIES ? 1u : 0u;
	if ( fused == 0u && rows > D41_RECORDED_ROWS(state) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	const uint32_t coded = fused != 0u && D41MxReady(&state->mx, rows, D41_GROUP_WIDTH, state->row_invariant) != 0u ? 1u : 0u;
	uint16_t *const merged = coded != 0u ? (uint16_t *)0 : state->attn_merged;
	uint8_t *const merged_codes = coded != 0u ? (uint8_t *)state->attn_merged : (uint8_t *)0;
	if ( fused != 0u )
	{
		const uint32_t attended = ratio > 0u ? 1u : 0u;
		const uint64_t ring_stride = state->recurrent_bytes / sizeof(uint16_t);
		if ( (cache < 3u ?
			D41AttendLaunch<D41Ratio2Kv,D41_HEAD_DIM>(state->latent_views[cache], attended, state->q_local, step->sequence_of_row, length,
				bound, state->selected, D41_INDEX_TOPK, state->window_kv, ring, ring_stride, D41_RING_ROWS, step->positions, state->row_lane,
				step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, layer->sink, qk_scale, merged, frequency, D41_HEAD_DIM - D41_ROPE,
				D41_GROUP_WIDTH / D41_HEAD_DIM, merged_codes, rows, rows, 0u, stream) :
			D41AttendLaunch<D41Ratio1Kv,D41_HEAD_DIM>(state->latent_views[cache], attended, state->q_local, step->sequence_of_row, length, bound,
				state->selected, D41_INDEX_TOPK, state->window_kv, ring, ring_stride, D41_RING_ROWS, step->positions, state->row_lane,
				step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, layer->sink, qk_scale, merged, frequency, D41_HEAD_DIM - D41_ROPE,
				D41_GROUP_WIDTH / D41_HEAD_DIM, merged_codes, rows, rows, 0u, stream)) !=
			cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	else if ( ratio > 0u )
	{
		if ( (cache < 3u ? LmLatentShardAttendHeadsLaunch<D41Ratio2Kv,LmKvShardView,D41_HEAD_DIM,0u>(state->latent_views[cache], state->q_local,
				query_stride, D41_LOCAL_HEADS, D41_LOCAL_HEADS, step->sequence_of_row, length, bound, state->selected, D41_INDEX_TOPK, D41_INDEX_TOPK,
				qk_scale, state->records_received, stride, D41_LOCAL_HEADS, 1u, rows, stream) :
			LmLatentShardAttendHeadsLaunch<D41Ratio1Kv,LmKvShardView,D41_HEAD_DIM,0u>(state->latent_views[cache], state->q_local,
				query_stride, D41_LOCAL_HEADS, D41_LOCAL_HEADS, step->sequence_of_row, length, bound, state->selected, D41_INDEX_TOPK, D41_INDEX_TOPK,
				qk_scale, state->records_received, stride, D41_LOCAL_HEADS, 1u, rows, stream)) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	else
		D41InitRecords(state->records_received, stride, rows, D41_LOCAL_HEADS, 1u, stream);
	if ( fused == 0u )
		D41WindowFoldWholeKernel<<<dim3(rows, D41_LOCAL_HEADS), D41K_THREADS, 0, stream>>>(state->records_received, D41_LOCAL_HEADS, state->q_local,
			state->window_kv, ring, step->positions, state->row_lane, state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices,
			step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_WINDOW, qk_scale);
	status = D41Launched();
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( step->commit == 0u )
	{
		if ( rows > D41_VERIFY_ROWS )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		cudaMemcpyAsync(state->verify_window + (uint64_t)index * D41_VERIFY_ROWS * D41_HEAD_DIM, state->window_kv,
			(uint64_t)rows * D41_HEAD_DIM * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
	}
	D41WindowRingStoreKernel<<<rows, D41_THREADS, 0, stream>>>(state->window_kv, ring, step->positions, state->row_lane, state->row_ordinal,
		step->sequence_row_begin, step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_RING_ROWS, D41_WINDOW, 1u, 0u, rows,
		step->commit, (const uint32_t *)0);
	if ( fused == 0u )
		D41MergeSinkKernel<<<dim3(rows, D41_LOCAL_HEADS), D41_THREADS, 0, stream>>>(state->records_received, stride, 1u, D41_LOCAL_HEADS, layer->sink,
			state->attn_merged);
	if ( coded == 0u )
		D41Rope(state->attn_merged, rows, D41_LOCAL_HEADS, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 1u, 0, stream);
	status = D41Launched();
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( uint32_t group = 0u; group < D41_LOCAL_GROUPS; ++group )
	{
		if ( coded != 0u )
		{
			const uint64_t codes_bytes = (uint64_t)rows * D41_GROUP_WIDTH;
			if ( LmMxGemmDense(merged_codes + group * codes_bytes, merged_codes + D41_LOCAL_GROUPS * codes_bytes + group * (codes_bytes / D41_FP8_BLOCK),
				layer->o_a[group].weight, layer->o_a[group].scale, state->wo_a_out, rows, D41_GROUP_WIDTH, D41_GROUP_ROWS, D41_WO_A_ROWS,
				group * D41_GROUP_ROWS, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			continue;
		}
		if ( D41MxReady(&state->mx, rows, D41_GROUP_WIDTH, state->row_invariant) != 0u )
		{
			if ( LmMxQuantizeRows(state->attn_merged + (uint64_t)group * D41_GROUP_WIDTH, local_width, state->mx.codes, state->mx.scales, rows, D41_GROUP_WIDTH,
				stream) != LM_LAUNCH_OK ||
				LmMxGemmDense(state->mx.codes, state->mx.scales, layer->o_a[group].weight, layer->o_a[group].scale, state->wo_a_out, rows, D41_GROUP_WIDTH,
				D41_GROUP_ROWS, D41_WO_A_ROWS, group * D41_GROUP_ROWS, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			continue;
		}
		D41ColumnsKernel<<<dim3(D41Blocks(D41_GROUP_WIDTH), rows), D41_THREADS, 0, stream>>>(state->attn_merged + (uint64_t)group * D41_GROUP_WIDTH,
			local_width, state->group_input, D41_GROUP_WIDTH, D41_GROUP_WIDTH);
		if ( D41QdqLinear(&layer->o_a[group], state->group_input, state->wo_a_group, rows, D41_GROUP_WIDTH, D41_GROUP_ROWS, sms, state->row_invariant,
			&state->mx, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		D41ColumnsKernel<<<dim3(D41Blocks(D41_GROUP_ROWS), rows), D41_THREADS, 0, stream>>>(state->wo_a_group, D41_GROUP_ROWS,
			state->wo_a_out + (uint64_t)group * D41_GROUP_ROWS, D41_WO_A_ROWS, D41_GROUP_ROWS);
	}
	if ( state->own_rows != 0u && state->o_b_ready != 0u && D41MxReady(&state->mx, state->own_rows, D41_WO_A_ROWS, state->row_invariant) != 0u )
		return D41OutputToOwners(state, index, rows, sms, stream);
	if ( D41QdqLinear(&layer->o_b, state->wo_a_out, state->partial, rows, D41_WO_A_ROWS, D41_HIDDEN, sms, state->row_invariant, &state->mx, stream) !=
		LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->owner_reduced = D41_ROWS ? 1u : 0u;
	return D41Launched();
}

static SparkStatus D41AttentionRows(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step,
	cudaStream_t stream)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors, ratio = D41Ratio(index), cache = D41CacheIndex(index);
	const uint32_t own = state->own_rows, first = state->own_first, parts = D41_HEADS / D41A_QUERIES;
	const uint32_t live = first < rows ? (rows - first < own ? rows - first : own) : 0u, edge = own < D41_WINDOW_EDGE_ROWS ? own : D41_WINDOW_EDGE_ROWS;
	const uint32_t part_width = D41A_QUERIES * D41_HEAD_DIM, index_width = D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM;
	const float *frequency = ratio > 0u ? state->compress_frequency : state->full_frequency;
	const uint32_t *bound = ratio == 2u ? state->bound_ratio2 : state->bound_ratio1;
	const uint32_t *length = ratio == 2u ? state->length_ratio2 : state->length_ratio1;
	const uint32_t indexed = ratio > 0u && D41IndexSource(index) ? 1u : 0u, attended = ratio > 0u ? 1u : 0u;
	const float qk_scale = 1.0f / sqrtf((float)D41_HEAD_DIM);
	const uint64_t ring_stride = state->recurrent_bytes / sizeof(uint16_t);
	const uint64_t part_codes = 2u * (uint64_t)own * D41_GROUP_WIDTH, part_bytes = part_codes + part_codes / D41_FP8_BLOCK;
	uint16_t *ring = (uint16_t *)state->recurrent + (uint64_t)index * D41_RING_ROWS * D41_HEAD_DIM;
	uint16_t *own_qrq = state->own_q, *own_kv = state->own_q + (uint64_t)own * D41_QLORA;
	uint8_t *qcodes = (uint8_t *)state->moe_send, *qscales = qcodes + (uint64_t)own * D41_QLORA, *codes = (uint8_t *)state->attn_merged;
	SparkStatus status;
	static_assert(D41_HEADS % D41A_QUERIES == 0u && D41A_QUERIES == 2u * (D41_GROUP_WIDTH / D41_HEAD_DIM), "a part of the heads covers two output groups");
	if ( D41LinearPair(&layer->q_a, &layer->kv_a, state->own_out, state->q_latent, state->kv, own, D41_HIDDEN, D41_QLORA, D41_HEAD_DIM, sms,
		state->row_invariant, &state->mx, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41RmsNormQdq(state->q_latent, layer->q_norm, 0, own_qrq, own, D41_QLORA, stream);
	D41KvNormRopeQdqKernel<D41_THREADS><<<own, D41_THREADS, (D41_HEAD_DIM + 8u) * sizeof(float), stream>>>(state->kv, layer->kv_norm,
		step->positions + first, frequency, own_kv, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, D41_EPS);
	if ( LmMxQuantize(own_qrq, qcodes, qscales, (uint64_t)own * D41_QLORA, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, edge, D41_HEAD_DIM,
			own_kv + (uint64_t)(own - edge) * D41_HEAD_DIM, state->window_edges);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41WindowAssembleKernel<<<own + D41_TP * edge, 64u, 0, stream>>>(own_kv, state->window_edges, state->window_kv, own, edge, first);
	for ( uint32_t part = 0u; part < parts; ++part )
	{
		uint16_t *query = state->q_local + (uint64_t)part * own * part_width;
		if ( LmMxGemmDense(qcodes, qscales, layer->q_b.weight + (uint64_t)part * part_width * D41_QLORA,
			layer->q_b.scale + (uint64_t)part * part_width * (D41_QLORA / D41_FP8_BLOCK), query, own, D41_QLORA, part_width, 0u, 0u,
			sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		D41Rope(query, own, D41A_QUERIES, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions + first, 0, frequency, 0u, 0, stream);
	}
	status = D41Launched();
	if ( status == SPARK_STATUS_OK && D41KvSource(index) && state->compress_done == 0u )
		status = D41Compress(state, layer, index, step, stream);
	if ( status == SPARK_STATUS_OK && indexed != 0u )
	{
		if ( LmMxGemmDense(qcodes, qscales, layer->idx_q_b.weight, layer->idx_q_b.scale, state->idx_pack_local, own, D41_QLORA, index_width, 0u, 0u, sms,
				stream) != LM_LAUNCH_OK ||
			D41Project(layer->idx_wp, state->x_complete != 0u ? state->x + (uint64_t)first * D41_HIDDEN : state->own_x, state->idx_w_full, 0, own,
				D41_HIDDEN, D41_LOCAL_INDEX_HEADS, sms, state->row_invariant, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		D41Rope(state->idx_pack_local, own, D41_LOCAL_INDEX_HEADS, D41_INDEX_DIM, D41_INDEX_DIM - D41_ROPE, step->positions + first, 0,
			state->compress_frequency, 0u, 0, stream);
		SparkDsv41FlashFp4Pow2QdqKernel<<<(own * (index_width / 32u) + 255u) / 256u, 256u, 0, stream>>>(state->idx_pack_local, own * (index_width / 32u),
			32u);
		D41ScaleRowsKernel<<<D41Blocks((uint64_t)own * D41_LOCAL_INDEX_HEADS), D41_THREADS, 0, stream>>>(state->idx_w_full,
			(uint64_t)own * D41_LOCAL_INDEX_HEADS, (float)(1.0 / sqrt((double)D41_INDEX_DIM) / sqrt((double)D41_INDEX_HEADS)));
		D41IndexPackKernel<<<own, D41_THREADS, 0, stream>>>(state->idx_pack_local, state->idx_w_full, state->q_gathered, index_width,
			D41_LOCAL_INDEX_HEADS, D41_INDEX_PACK);
		status = D41Launched();
		if ( status == SPARK_STATUS_OK )
			status = cache < 3u ? D41SelectWhole<D41Ratio2Kv>(state, index, step, bound, length, stream) :
				D41SelectWhole<D41Ratio1Kv>(state, index, step, bound, length, stream);
	}
	if ( status != SPARK_STATUS_OK )
		return status;
	static_assert(D41_HEADS % (D41_ATTEND_TILES * D41A_QUERIES) == 0u, "the heads split into whole passes of query tiles");
	for ( uint32_t part = 0u; part < parts; part += D41_ATTEND_TILES )
	{
		const uint16_t *query = state->q_local + (uint64_t)part * own * part_width;
		uint8_t *part_out = codes + (uint64_t)part * part_bytes;
		const float *sink = layer->sink + part * D41A_QUERIES;
		if ( (cache < 3u ?
			D41AttendLaunch<D41Ratio2Kv,D41_HEAD_DIM,D41_ATTEND_TILES>(state->latent_views[cache], attended, query, step->sequence_of_row + first, length,
				bound + first, state->own_selected, D41_INDEX_TOPK, state->window_kv, ring, ring_stride, D41_RING_ROWS, step->positions, state->row_lane + first,
				step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, sink, qk_scale, (uint16_t *)0, frequency, D41_HEAD_DIM - D41_ROPE,
				D41_GROUP_WIDTH / D41_HEAD_DIM, part_out, own, live, first, stream, (uint64_t)own * part_width, 0u, part_bytes) :
			D41AttendLaunch<D41Ratio1Kv,D41_HEAD_DIM,D41_ATTEND_TILES>(state->latent_views[cache], attended, query, step->sequence_of_row + first, length,
				bound + first, state->own_selected, D41_INDEX_TOPK, state->window_kv, ring, ring_stride, D41_RING_ROWS, step->positions, state->row_lane + first,
				step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, sink, qk_scale, (uint16_t *)0, frequency, D41_HEAD_DIM - D41_ROPE,
				D41_GROUP_WIDTH / D41_HEAD_DIM, part_out, own, live, first, stream, (uint64_t)own * part_width, 0u, part_bytes)) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	D41WindowRingStoreKernel<<<rows, D41_THREADS, 0, stream>>>(state->window_kv, ring, step->positions, state->row_lane, state->row_ordinal,
		step->sequence_row_begin, step->recurrent_index, ring_stride, D41_RING_ROWS, D41_WINDOW, 1u, 0u, rows, step->commit, (const uint32_t *)0);
	{
		const uint8_t *group_codes[D41_LOCAL_GROUPS], *group_scales[D41_LOCAL_GROUPS], *group_weights[D41_LOCAL_GROUPS], *group_weight_scales[D41_LOCAL_GROUPS];
		uint32_t group_columns[D41_LOCAL_GROUPS];
		static_assert(D41_LOCAL_GROUPS <= LM_MX_MAX_GROUPS, "one launch multiplies every output group");
		for ( uint32_t group = 0u; group < D41_LOCAL_GROUPS; ++group )
		{
			const uint8_t *part_out = codes + (uint64_t)(group / 2u) * part_bytes;
			const uint64_t sub = group % 2u;
			group_codes[group] = part_out + sub * own * D41_GROUP_WIDTH;
			group_scales[group] = part_out + part_codes + sub * own * (D41_GROUP_WIDTH / D41_FP8_BLOCK);
			group_weights[group] = layer->o_a[group].weight;
			group_weight_scales[group] = layer->o_a[group].scale;
			group_columns[group] = group * D41_GROUP_ROWS;
		}
		if ( LmMxGemmGrouped(group_codes, group_scales, group_weights, group_weight_scales, group_columns, D41_LOCAL_GROUPS, state->wo_a_out, own,
			D41_GROUP_WIDTH, D41_GROUP_ROWS, D41_WO_A_ROWS, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	if ( LmMxQuantize(state->wo_a_out, state->mx.codes, state->mx.scales, (uint64_t)own * D41_WO_A_ROWS, stream) != LM_LAUNCH_OK ||
		LmMxGemmDense(state->mx.codes, state->mx.scales, layer->o_b.weight, layer->o_b.scale, state->own_out, own, D41_WO_A_ROWS, D41_HIDDEN, 0u, 0u, sms,
			stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->owner_reduced = 1u;
	return D41Launched();
}

static SparkStatus D41AttentionSplit(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step, uint32_t own,
	cudaStream_t stream)
{
	const uint32_t rows = step->rows, first = state->tp_rank * own;
	const uint32_t live = first < rows ? (rows - first < own ? rows - first : own) : 0u;
	SparkStatus status = SPARK_STATUS_OK;
	if ( live != 0u && cudaMemcpyAsync(state->own_out, state->xq + (uint64_t)first * D41_HIDDEN, (uint64_t)live * D41_HIDDEN * sizeof(uint16_t),
		cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->own_rows = own;
	state->own_first = first;
	state->own_pad = own * D41_TP - rows;
	status = D41AttentionRows(state, layer, index, step, stream);
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, own, D41_HIDDEN, state->own_out, state->partial);
	state->own_rows = 0u;
	state->own_first = 0u;
	state->own_pad = 0u;
	state->owner_reduced = 1u;
	return status;
}

static SparkStatus D41Attention(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step,
	cudaStream_t stream)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors, ratio = D41Ratio(index);
	const float *frequency = ratio > 0u ? state->compress_frequency : state->full_frequency;
	const uint32_t pair = D41PairRecords(state, rows);
	const uint32_t local = (index < D41_CANDIDATE_SOURCE ? state->local_attention : state->local_upper) != 0u ? 1u : 0u;
	const uint32_t record_heads = pair != 0u ? 2u * D41_LOCAL_HEADS : D41_LOCAL_HEADS, record_copies = pair != 0u ? 2u : 1u;
	const uint64_t destination_stride = D41DestinationStride(rows, record_heads);
	const uint32_t *bound = ratio == 2u ? state->bound_ratio2 : state->bound_ratio1;
	const uint32_t *length = ratio == 2u ? state->length_ratio2 : state->length_ratio1;
	const uint32_t local_width = D41_LOCAL_HEADS * D41_HEAD_DIM;
	const uint32_t indexed = ratio > 0u && D41IndexSource(index) ? 1u : 0u;
	const uint32_t gather_width = local_width + (indexed != 0u ? D41_INDEX_PACK : 0u);
	const uint64_t query_rank_stride = (uint64_t)rows * gather_width;
	uint16_t *ring = (uint16_t *)state->recurrent + (uint64_t)index * D41_RING_ROWS * D41_HEAD_DIM;
	SparkStatus status;
	if ( D41_WHOLE )
	{
		const uint32_t split = (rows + D41_OWN_ROWS_STEP - 1u) / D41_OWN_ROWS_STEP * (D41_OWN_ROWS_STEP / D41_TP);
		if ( D41_ROWS && state->own_rows == 0u && step->commit != 0u && state->dump == 0 && state->tap_slab == 0 &&
			D41MxReady(&state->mx, split, D41_QLORA, state->row_invariant) != 0u )
			return D41AttentionSplit(state, layer, index, step, split, stream);
		if ( D41SpreadRow(state, step) != 0u )
			return D41AttentionSpread(state, layer, index, step, stream);
		return D41RowsAttention(state) != 0u ? D41AttentionRows(state, layer, index, step, stream) : D41AttentionWhole(state, layer, index, step, stream);
	}
	if ( D41LinearPair(&layer->q_a, &layer->kv_a, state->xq, state->q_latent, state->kv, rows, D41_HIDDEN, D41_QLORA, D41_HEAD_DIM, sms, state->row_invariant,
		&state->mx, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41RmsNormQdq(state->q_latent, layer->q_norm, 0, state->qrq, rows, D41_QLORA, stream);
	if ( (indexed != 0u ? D41LinearPair(&layer->q_b, &layer->idx_q_b, state->qrq, state->q_local, state->idx_pack_local, rows, D41_QLORA,
		local_width, D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM, sms, state->row_invariant, &state->mx, stream) :
		D41Linear(&layer->q_b, state->qrq, state->q_local, rows, D41_QLORA, local_width, sms, state->row_invariant, &state->mx, stream)) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41Rope(state->q_local, rows, D41_LOCAL_HEADS, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 0u, 0, stream);
	D41KvNormRopeQdqKernel<D41_THREADS><<<rows, D41_THREADS, (D41_HEAD_DIM + 8u) * sizeof(float), stream>>>(state->kv, layer->kv_norm,
		step->positions, frequency, state->window_kv, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, D41_EPS);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK && D41KvSource(index) && state->compress_done == 0u )
		status = D41Compress(state, layer, index, step, stream);
	if ( status == SPARK_STATUS_OK && local != 0u && D41KvSource(index) )
		status = D41CacheIndex(index) < 3u ? D41LandCache<D41Ratio2Kv>(state, D41CacheIndex(index), step, stream) :
			D41LandCache<D41Ratio1Kv>(state, D41CacheIndex(index), step, stream);
	if ( status == SPARK_STATUS_OK && indexed != 0u )
		status = D41IndexQuery(state, layer, index, step, state->q_local + (uint64_t)rows * local_width, stream);
	if ( status == SPARK_STATUS_OK && local == 0u )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, gather_width, state->q_local, state->q_gathered);
	else if ( status == SPARK_STATUS_OK && indexed != 0u )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, D41_INDEX_PACK,
			state->q_local + (uint64_t)rows * local_width, state->q_gathered);
	if ( status == SPARK_STATUS_OK )
		status = D41Launched();
	if ( status == SPARK_STATUS_OK && indexed != 0u )
	{
		const uint16_t *packed = local != 0u ? state->q_gathered : state->q_gathered + (uint64_t)rows * local_width;
		const uint64_t packed_stride = local != 0u ? (uint64_t)rows * D41_INDEX_PACK : query_rank_stride;
		status = D41CacheIndex(index) < 3u ? D41Select<D41Ratio2Kv>(state, index, step, bound, length, packed, packed_stride, stream) :
			D41Select<D41Ratio1Kv>(state, index, step, bound, length, packed, packed_stride, stream);
	}
	if ( status == SPARK_STATUS_OK && ratio > 0u && local != 0u )
		status = D41CacheIndex(index) < 3u ? D41LocalPartials<D41Ratio2Kv>(state, index, step, bound, length, stream) :
			D41LocalPartials<D41Ratio1Kv>(state, index, step, bound, length, stream);
	else if ( status == SPARK_STATUS_OK && ratio > 0u )
		status = D41CacheIndex(index) < 3u ?
			D41Partials<D41Ratio2Kv>(state, index, step, bound, length, query_rank_stride, record_heads, record_copies, stream) :
			D41Partials<D41Ratio1Kv>(state, index, step, bound, length, query_rank_stride, record_heads, record_copies, stream);
	else if ( status == SPARK_STATUS_OK )
	{
		D41InitRecords(local != 0u ? state->records_received : state->records_send, destination_stride, rows, record_heads, D41_TP, stream);
		status = D41Launched();
	}
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( local != 0u )
		for ( uint32_t source = 0u; source < D41_TP; ++source )
			D41WindowFoldKernel<<<dim3(rows, D41_LOCAL_HEADS), D41K_THREADS, 0, stream>>>(state->records_received + (uint64_t)source * destination_stride,
				destination_stride, D41_LOCAL_HEADS, state->q_local, (uint64_t)rows * local_width, state->window_kv,
				state->ring_landing + (uint64_t)source * D41_RING_LANDING_ROWS * D41_HIDDEN + (uint64_t)index * D41_RING_ELEMENTS, step->positions, state->row_lane,
				state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, 0u, D41_RING_ROWS, D41_WINDOW, D41_TP,
				source, 1.0f / sqrtf((float)D41_HEAD_DIM), D41_LOCAL_HEADS, 1u);
	else
		D41WindowFoldKernel<<<dim3(rows, D41_HEADS), D41K_THREADS, 0, stream>>>(state->records_send, destination_stride, D41_LOCAL_HEADS,
			state->q_gathered, query_rank_stride, state->window_kv, ring, step->positions, state->row_lane, state->row_ordinal,
			step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_RING_ROWS,
			D41_WINDOW, D41_TP, state->tp_rank,
			1.0f / sqrtf((float)D41_HEAD_DIM), record_heads, record_copies);
	status = D41Launched();
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( step->commit == 0u )
	{
		if ( rows > D41_VERIFY_ROWS )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		cudaMemcpyAsync(state->verify_window + (uint64_t)index * D41_VERIFY_ROWS * D41_HEAD_DIM, state->window_kv,
			(uint64_t)rows * D41_HEAD_DIM * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
	}
	D41WindowRingStoreKernel<<<rows, D41_THREADS, 0, stream>>>(state->window_kv, ring, step->positions, state->row_lane, state->row_ordinal,
		step->sequence_row_begin, step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_RING_ROWS, D41_WINDOW, D41_TP,
		state->tp_rank, rows, step->commit, (const uint32_t *)0);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK && local == 0u )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, record_copies * rows, D41_RECORD_ROW_ELEMENTS,
			state->records_send, state->records_received);
	if ( status != SPARK_STATUS_OK )
		return status;
	{
		uint16_t *merged = pair != 0u ? state->group_input : state->attn_merged;
		D41MergeSinkKernel<<<dim3(rows, record_heads), D41_THREADS, 0, stream>>>(state->records_received, destination_stride, D41_TP,
			record_heads, pair != 0u ? layer->group_sink : layer->sink, merged);
		D41Rope(merged, rows, record_heads, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 1u, 0, stream);
	}
	status = D41Launched();
	if ( status == SPARK_STATUS_OK && pair == 0u )
	{
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_PAIR_GATHER, rows, local_width, state->attn_merged,
			state->attn_gathered);
		if ( status == SPARK_STATUS_OK )
		{
			D41GroupInputKernel<<<dim3(D41Blocks(2u * local_width), rows), D41_THREADS, 0, stream>>>(state->attn_gathered, state->group_input,
				rows, local_width, (state->tp_rank / 2u) * 2u, 2u);
			status = D41Launched();
		}
	}
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( D41QdqLinear(&layer->o_a[0], state->group_input, state->wo_a_out, rows, 2u * local_width, D41_WO_A_ROWS, sms, state->row_invariant, &state->mx,
		stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( D41QdqLinear(&layer->o_b, state->wo_a_out, state->partial, rows, D41_WO_A_ROWS, D41_HIDDEN, sms, state->row_invariant, &state->mx, stream) !=
		LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return D41Launched();
}

static SparkStatus D41DumpRows(D41ModelState *state, uint32_t layer, uint32_t kind, const uint16_t *rows_bf16, uint32_t rows,
	uint32_t width, cudaStream_t stream);

static int32_t D41Experts(const uint8_t *cells, const uint16_t *activation, uint16_t *output, const D41ModelState *state, uint32_t rows,
	uint32_t routes, uint32_t packed, uint32_t input_dimension, uint32_t output_dimension, uint32_t groups, cudaStream_t stream)
{
	const uint32_t *source = packed != 0u ? (const uint32_t *)0 : state->route_source_token;
	if ( rows <= D41_SKINNY_EXPERT_ROWS || state->row_invariant != 0u )
		return LmSkinnyCellExperts(cells, activation, output, state->group_row_offset, source, groups, routes, rows, packed,
			input_dimension, output_dimension, LM_SKINNY_CELL_TILE_K, stream);
	return LmCellMmaExperts(cells, activation, output, state->group_row_offset, source, groups, routes, packed, input_dimension,
		output_dimension, LM_SKINNY_CELL_TILE_K, stream);
}

static void D41SharedSlice(uint32_t rank, uint32_t *first, uint32_t *width)
{
	const uint32_t parts = D41_ROWS ? 1u : D41_TP, part = D41_ROWS ? 0u : rank;
	const uint32_t blocks = D41_MOE_INTER / D41_SHARED_GRANULE, base = blocks / parts, extra = blocks % parts;
	*first = (part * base + (part < extra ? part : extra)) * D41_SHARED_GRANULE;
	*width = (base + (part < extra ? 1u : 0u)) * D41_SHARED_GRANULE;
}

static SparkStatus D41SharedPartials(D41ModelState *state, const D41Layer *layer, uint32_t rows, uint64_t stride, uint32_t sms, cudaStream_t stream)
{
	for ( uint32_t source = 0u; source < (D41_SHARED_ROUTED != 0u ? D41_TP : 0u); ++source )
	{
		uint32_t first, width;
		const uint8_t *codes = (const uint8_t *)(state->moe_received + (uint64_t)source * stride);
		uint16_t *partial = state->shared_out + (uint64_t)source * rows * D41_HIDDEN_SLICE;
		int32_t launched;
		D41SharedSlice(source, &first, &width);
		if ( D41MxReady(&state->mx, rows, width, state->row_invariant) != 0u )
			launched = LmMxGemmDense(codes, codes + (uint64_t)rows * width, layer->sh_w2_part[source].weight, layer->sh_w2_part[source].scale, partial,
				rows, width, D41_HIDDEN_SLICE, 0u, 0u, sms, stream);
		else
		{
			D41SharedDequantKernel<<<dim3(D41Blocks(width), rows), D41_THREADS, 0, stream>>>(codes, state->shared_act, rows, width);
			launched = D41Linear(&layer->sh_w2_part[source], state->shared_act, partial, rows, width, D41_HIDDEN_SLICE, sms, state->row_invariant,
				(const D41Mx *)0, stream);
		}
		if ( launched != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	return D41Launched();
}

static uint32_t D41Exchanges(const D41ModelState *state)
{
	return D41_ROWS && state->own_rows != 0u && state->dump == 0 && state->row_invariant == 0u && state->exchange_totals_host != 0 &&
		state->counts_ready != 0 && state->owner_host != 0 ? 1u : 0u;
}

static int32_t D41SharedUp(D41ModelState *state, const D41Layer *layer, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	const uint32_t own = state->own_rows;
	if ( own != 0u && D41MxReady(&state->mx, rows, D41_HIDDEN, state->row_invariant) != 0u )
	{
		const uint64_t segment = (uint64_t)own * D41_FFN_GATHER_ELEMENTS * sizeof(uint16_t);
		for ( uint32_t source = 0u; source < D41_TP; ++source )
		{
			const uint8_t *codes = (const uint8_t *)state->moe_received + source * segment;
			const uint64_t first = (uint64_t)source * own * state->shared_width;
			if ( LmMxGemmDense(codes, codes + (uint64_t)own * D41_HIDDEN, layer->sh_w1.weight, layer->sh_w1.scale, state->shared_gate + first, own,
				D41_HIDDEN, state->shared_width, 0u, 0u, sms, stream) != LM_LAUNCH_OK ||
				LmMxGemmDense(codes, codes + (uint64_t)own * D41_HIDDEN, layer->sh_w3.weight, layer->sh_w3.scale, state->shared_up + first, own,
				D41_HIDDEN, state->shared_width, 0u, 0u, sms, stream) != LM_LAUNCH_OK )
				return LM_LAUNCH_ERR_LAUNCH;
		}
		return LM_LAUNCH_OK;
	}
	return D41LinearPair(&layer->sh_w1, &layer->sh_w3, state->xq, state->shared_gate, state->shared_up, rows, D41_HIDDEN, state->shared_width,
		state->shared_width, sms, state->row_invariant, &state->mx, stream);
}

static SparkStatus D41ExchangeReturn(D41ModelState *state, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t own = state->own_rows, first = state->own_first;
	const uint32_t live = first < step->rows ? (step->rows - first < own ? step->rows - first : own) : 0u;
	SparkStatus status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, state->exchange_rows, D41_HIDDEN,
		state->moe_final, state->partial);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( state->moe_rows != 0u )
		D41ShuffleSumKernel<D41_TP><<<dim3(D41Blocks(D41_HIDDEN / 8u), state->moe_rows), D41_THREADS, 0, stream>>>(state->partial, state->exchange_slot,
			state->shuffle_received, state->shuffle_rows, D41_HIDDEN, state->moe_send);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, state->shuffle_rows, D41_HIDDEN, state->moe_send,
			state->moe_received);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( cudaStreamWaitEvent(stream, state->shared_join, 0u) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41ShuffleFinishKernel<<<dim3(D41Blocks(D41_HIDDEN / 8u), own), D41_THREADS, 0, stream>>>(state->moe_received, state->shared_out, live, first,
		D41_TP, state->shuffle_rows, D41_HIDDEN, state->own_out);
	return D41Launched();
}

static SparkStatus D41MoeExchanged(D41ModelState *state, const D41Layer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t index = (uint32_t)(layer - state->layers), held = index < D41_BACKUP_LAYERS && state->backup_cells != 0 ? 1u : 0u;
	const uint32_t groups = held != 0u ? D41_HELD_EXPERTS : D41_LOCAL_EXPERTS, rows = state->exchange_live, routes = rows * D41_TOPK;
	uint16_t *gate_up = state->expert_slice;
	uint8_t *codes = (uint8_t *)state->expert_act;
	if ( rows == 0u )
		return D41ExchangeReturn(state, step, stream);
	D41RouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global, routes,
		state->expert_group + (uint64_t)index * D41_EXPERTS, D41_LOCAL_EXPERTS, groups, D41_EXPERTS);
	if ( (held != 0u ?
		LmRouteBuild<D41_THREADS,D41_HELD_EXPERTS + 1u>(state->route_expert, rows, routes, D41_TOPK, state->group_row_offset, state->route_packed_row,
			state->route_source_token, 2u * D41_MOE_INTER, D41_HIDDEN, D41_GEMM_TILE_N, state->tile_prefix_up, state->tile_prefix_down, stream) :
		LmRouteBuild<D41_THREADS,D41_LOCAL_EXPERTS + 1u>(state->route_expert, rows, routes, D41_TOPK, state->group_row_offset, state->route_packed_row,
			state->route_source_token, 2u * D41_MOE_INTER, D41_HIDDEN, D41_GEMM_TILE_N, state->tile_prefix_up, state->tile_prefix_down, stream)) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
	D41FinalizeMapKernel<D41_TP><<<(rows + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_packed_row, state->exchange_received,
		state->exchange_rows, D41_TOPK, groups, rows, LM_CELL_MX_MAPPED, state->route_output_row);
	D41RecordPlaceKernel<D41_TP><<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_source_token, routes, state->exchange_received,
		state->exchange_rows);
	LmCellMxCodes received = LmCellMxPacked((const uint8_t *)state->moe_received, (const uint8_t *)state->moe_received + D41_HIDDEN, D41_HIDDEN);
	received.code_stride = (uint64_t)D41_FFN_GATHER_ELEMENTS * sizeof(uint16_t);
	received.scale_stride = received.code_stride;
	LmCellMxCodes down = LmCellMxPacked(codes, codes + (uint64_t)routes * D41_MOE_INTER, D41_MOE_INTER);
	down.output_rows = state->route_output_row;
	down.mapped_output = state->moe_final;
	if ( D41Launched() != SPARK_STATUS_OK ||
		LmCellMxExpertsSplit(layer->e_w13_cells, held != 0u ? layer->e_w13_backup : (const uint8_t *)0, held != 0u ? D41_LOCAL_EXPERTS : 0u,
		received, gate_up, state->group_row_offset, state->route_source_token, groups, routes, 0u, D41_HIDDEN, 2u * D41_MOE_INTER, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41ExpertActKernel<<<dim3(D41Blocks(D41_MOE_INTER), routes < D41_ACT_BLOCKS ? routes : D41_ACT_BLOCKS), D41_THREADS, 0, stream>>>(gate_up,
		state->packed_weight, state->group_row_offset, groups, codes, (uint16_t *)0, routes, D41_MOE_INTER, D41_SWIGLU_LIMIT);
	if ( D41Launched() != SPARK_STATUS_OK ||
		LmCellMxExpertsSplit(layer->e_w2_cells, held != 0u ? layer->e_w2_backup : (const uint8_t *)0, held != 0u ? D41_LOCAL_EXPERTS : 0u, down,
			state->expert_slice, state->group_row_offset, (const uint32_t *)0, groups, routes, 1u, D41_MOE_INTER, D41_HIDDEN, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41ExchangeFinalizeKernel<D41_TP><<<dim3(D41Blocks(D41_HIDDEN / 8u), rows), D41_THREADS, 0, stream>>>(state->expert_slice,
		state->route_packed_row, state->route_expert, state->exchange_received, state->exchange_rows, D41_TOPK, groups, D41_HIDDEN, 1u, state->moe_final);
	const SparkStatus status = D41Launched();
	return status != SPARK_STATUS_OK ? status : D41ExchangeReturn(state, step, stream);
}

static SparkStatus D41MoeWhole(D41ModelState *state, const D41Layer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	if ( D41Exchanges(state) != 0u )
		return D41MoeExchanged(state, layer, step, stream);
	const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * D41_TOPK;
	const uint32_t mx_down = state->dump == 0 && state->row_invariant == 0u && rows > D41_SKINNY_EXPERT_ROWS ? 1u : 0u;
	const uint32_t shared_block = D41_ROWS ? (rows + D41_TP - 1u) / D41_TP : rows, shared_first = D41_ROWS ? state->tp_rank * shared_block : 0u;
	const uint32_t shared_rows = shared_first < rows ? (rows - shared_first < shared_block ? rows - shared_first : shared_block) : 0u;
	uint16_t *gate_up = state->expert_slice;
	uint8_t *codes = (uint8_t *)state->expert_act;
	SparkStatus status;
	if ( D41_ROWS && cudaMemsetAsync(state->shared_out, 0, (uint64_t)rows * D41_HIDDEN * sizeof(uint16_t), stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->own_rows == 0u &&
		(D41Project(layer->router, state->x, 0, state->router_logits, rows, D41_HIDDEN, D41_EXPERTS, sms, state->row_invariant, stream) != LM_LAUNCH_OK ||
		LmTopkRouteLaunch<D41_THREADS,D41_TOPK,true,LM_TOPK_SCORE_SQRT_SOFTPLUS>(rows, state->router_logits, D41_EXPERTS, state->route_expert,
			state->route_weight, layer->router_bias, 0, D41_ROUTE_SCALE, step->route_fault, stream) != cudaSuccess) )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->dump != 0 &&
		(cudaStreamSynchronize(stream) != cudaSuccess ||
		cudaMemcpy(state->dump_routes, state->route_expert, (size_t)routes * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(state->dump_weights, state->route_weight, (size_t)routes * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	D41RouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global, routes,
		state->expert_group + (uint64_t)(layer - state->layers) * D41_EXPERTS, D41_LOCAL_EXPERTS, D41_LOCAL_EXPERTS, D41_EXPERTS);
	if ( LmRouteBuild<D41_THREADS,D41_LOCAL_EXPERTS + 1u>(state->route_expert, rows, routes, D41_TOPK, state->group_row_offset, state->route_packed_row,
		state->route_source_token, 2u * D41_MOE_INTER, D41_HIDDEN, D41_GEMM_TILE_N, state->tile_prefix_up, state->tile_prefix_down, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
	if ( D41Experts(layer->e_w13_cells, state->xq, gate_up, state, rows, routes, 0u, D41_HIDDEN, 2u * D41_MOE_INTER, D41_LOCAL_EXPERTS, stream) !=
		LM_LAUNCH_OK ||
		(D41_ROWS ? shared_rows != 0u && D41LinearPair(&layer->sh_w1, &layer->sh_w3, state->xq + (uint64_t)shared_first * D41_HIDDEN, state->shared_gate,
			state->shared_up, shared_rows, D41_HIDDEN, state->shared_width, state->shared_width, sms, state->row_invariant, &state->mx, stream) != LM_LAUNCH_OK :
			D41SharedUp(state, layer, rows, sms, stream) != LM_LAUNCH_OK) )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41ExpertActKernel<<<dim3(D41Blocks(D41_MOE_INTER), routes < D41_ACT_BLOCKS ? routes : D41_ACT_BLOCKS), D41_THREADS, 0, stream>>>(gate_up,
		state->packed_weight,
		state->group_row_offset, D41_LOCAL_EXPERTS, codes, mx_down != 0u ? (uint16_t *)0 : state->expert_act, routes, D41_MOE_INTER, D41_SWIGLU_LIMIT);
	if ( shared_rows != 0u )
		D41SwigluRowsKernel<<<dim3(D41Blocks(state->shared_width), shared_rows), D41_THREADS, 0, stream>>>(state->shared_gate, state->shared_up,
			state->shared_act, state->shared_width, D41_SWIGLU_LIMIT);
	status = D41Launched();
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( (shared_rows != 0u && D41QdqLinear(&layer->sh_w2, state->shared_act, state->shared_out + (uint64_t)shared_first * D41_HIDDEN, shared_rows,
		state->shared_width, D41_HIDDEN, sms, state->row_invariant, &state->mx, stream) != LM_LAUNCH_OK) ||
		(mx_down != 0u ?
		LmCellMxExperts(layer->e_w2_cells, codes, codes + (uint64_t)routes * D41_MOE_INTER, state->expert_slice, state->group_row_offset,
			(const uint32_t *)0, D41_LOCAL_EXPERTS, routes, 1u, D41_MOE_INTER, D41_HIDDEN, stream) :
		D41Experts(layer->e_w2_cells, state->expert_act, state->expert_slice, state, rows, routes, 1u, D41_MOE_INTER, D41_HIDDEN, D41_LOCAL_EXPERTS,
			stream)) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	LmMoeLocalFinalizeKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->expert_slice, state->route_packed_row,
		state->route_expert, state->shared_out, state->moe_final, D41_TOPK, D41_LOCAL_EXPERTS, D41_HIDDEN);
	status = D41Launched();
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( state->own_rows != 0u )
	{
		if ( state->own_pad != 0u )
			cudaMemsetAsync(state->moe_final + (uint64_t)rows * D41_HIDDEN, 0, (uint64_t)state->own_pad * D41_HIDDEN * sizeof(uint16_t), stream);
		return CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_REDUCE_SCATTER_SUM_BF16, state->own_rows * D41_TP, 0u,
			state->moe_final, state->own_out);
	}
	if ( rows > D41_OWN_ROWS_MIN )
		return CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->moe_final, state->partial);
	status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, D41_HIDDEN, state->moe_final, state->moe_gathered);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41SumSourcesKernel<<<D41Blocks(rows * D41_HIDDEN) < 1024u ? D41Blocks(rows * D41_HIDDEN) : 1024u, D41_THREADS, 0, stream>>>(state->moe_gathered,
		state->partial, (uint64_t)rows * D41_HIDDEN, D41_TP);
	return D41Launched();
}

static SparkStatus D41Moe(D41ModelState *state, const D41Layer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * D41_TOPK;
	const uint64_t stride = (uint64_t)rows * D41_MOE_ROW_ELEMENTS, shared_offset = (uint64_t)routes * D41_MOE_ROUTE_ELEMENTS;
	const uint64_t route_base = (uint64_t)rows * D41_MOE_SHARED_ELEMENTS;
	SparkStatus status;
	if ( state->moe_whole != 0u )
		return D41MoeWhole(state, layer, step, stream);
	if ( state->own_rows == 0u &&
		(D41Project(layer->router, state->x, 0, state->router_logits, rows, D41_HIDDEN, D41_EXPERTS, sms, state->row_invariant, stream) != LM_LAUNCH_OK ||
		LmTopkRouteLaunch<D41_THREADS,D41_TOPK,true,LM_TOPK_SCORE_SQRT_SOFTPLUS>(rows, state->router_logits, D41_EXPERTS, state->route_expert,
			state->route_weight, layer->router_bias, 0, D41_ROUTE_SCALE, step->route_fault, stream) != cudaSuccess) )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->dump != 0 &&
		(cudaStreamSynchronize(stream) != cudaSuccess ||
		cudaMemcpy(state->dump_routes, state->route_expert, (size_t)routes * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(state->dump_weights, state->route_weight, (size_t)routes * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( LmRouteBuild<D41_THREADS,D41_EXPERTS>(state->route_expert, rows, routes, D41_TOPK, state->group_row_offset, state->route_packed_row,
		state->route_source_token, 2u * D41_EXPERT_SLICE, D41_HIDDEN_SLICE, D41_GEMM_TILE_N, state->tile_prefix_up, state->tile_prefix_down,
		stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
	if ( D41Experts(layer->e_w13_cells, state->xq, state->expert_gate_up, state, rows, routes, 0u, D41_HIDDEN, 2u * D41_EXPERT_SLICE, D41_EXPERTS, stream) !=
		LM_LAUNCH_OK ||
		D41SharedUp(state, layer, rows, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41SwigluRowsKernel<<<dim3(D41Blocks(state->shared_width), rows), D41_THREADS, 0, stream>>>(state->shared_gate, state->shared_up,
		state->shared_act, state->shared_width, D41_SWIGLU_LIMIT);
	if ( D41_SHARED_ROUTED != 0u )
		D41SharedSendKernel<<<dim3(D41Blocks(state->shared_width), rows), D41_THREADS, 0, stream>>>(state->shared_act, state->moe_send, rows,
			state->shared_width, D41_TP, stride);
	else if ( D41QdqLinear(&layer->sh_w2, state->shared_act, state->shared_out, rows, state->shared_width, D41_HIDDEN, sms, state->row_invariant, &state->mx,
		stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41MoeSendKernel<<<D41ItemGrid(D41Blocks(D41_SHARED_ROUTED != 0u ? D41_EXPERT_SLICE : D41_HIDDEN), routes + (D41_SHARED_ROUTED != 0u ? 0u : rows)),
		D41_THREADS, 0, stream>>>(state->expert_gate_up, state->packed_weight, state->shared_out, state->moe_send + route_base, routes,
		D41_SHARED_ROUTED != 0u ? 0u : rows, D41_EXPERT_SLICE, D41_HIDDEN_SLICE, D41_TP, stride, D41_SWIGLU_LIMIT, D41_MOE_ROUTE_ELEMENTS, D41_MOE_CODES);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, rows, D41_MOE_ROW_ELEMENTS, state->moe_send,
			state->moe_received);
	if ( status != SPARK_STATUS_OK )
		return status;
	const uint32_t mx_down = D41_MOE_CODES != 0u && state->dump == 0 && state->row_invariant == 0u && rows > D41_SKINNY_EXPERT_ROWS ? 1u : 0u;
	const uint8_t *down_codes = (const uint8_t *)state->expert_act;
	if ( mx_down != 0u )
		D41ActAssembleCodesKernel<<<D41ItemGrid(D41Blocks(D41_MOE_INTER / 16u + D41_MOE_INTER / (2u * D41_FP8_BLOCK)), routes), D41_THREADS, 0, stream>>>(
			state->moe_received + route_base, (uint8_t *)state->expert_act, D41_EXPERT_SLICE, D41_MOE_INTER, stride, routes);
	else
		D41ActAssembleQdqKernel<<<D41ItemGrid(D41_MOE_INTER / D41_THREADS, routes), D41_THREADS, 0, stream>>>(state->moe_received + route_base,
			state->expert_act, D41_EXPERT_SLICE, D41_MOE_INTER, stride, routes, D41_MOE_CODES);
	if ( (mx_down != 0u ?
		LmCellMxExperts(layer->e_w2_cells, down_codes, down_codes + (uint64_t)routes * D41_MOE_INTER, state->expert_slice, state->group_row_offset,
			(const uint32_t *)0, D41_EXPERTS, routes, 1u, D41_MOE_INTER, D41_HIDDEN_SLICE, stream) :
		D41Experts(layer->e_w2_cells, state->expert_act, state->expert_slice, state, rows, routes, 1u, D41_MOE_INTER, D41_HIDDEN_SLICE, D41_EXPERTS, stream)) !=
		LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( D41_SHARED_ROUTED != 0u )
	{
		status = D41SharedPartials(state, layer, rows, stride, sms, stream);
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	D41FinalizeSliceKernel<<<dim3(D41Blocks(D41_HIDDEN_SLICE), rows), D41_THREADS, 0, stream>>>(state->expert_slice, state->moe_received,
		state->route_expert, state->route_packed_row, state->moe_final, D41_TOPK, D41_LOCAL_EXPERTS, D41_TP, D41_HIDDEN_SLICE, stride, shared_offset,
		D41_SHARED_ROUTED != 0u ? state->shared_out : (const uint16_t *)0);
	status = D41Launched();
	if ( state->dump != 0 && step->rows < 64u && status == SPARK_STATUS_OK )
	{
		status = D41DumpRows(state, 99u, 8u, state->expert_gate_up, routes, 2u * D41_EXPERT_SLICE, stream);
		if ( status == SPARK_STATUS_OK )
			status = D41DumpRows(state, 99u, 10u, state->expert_act, routes, D41_MOE_INTER, stream);
		if ( status == SPARK_STATUS_OK )
			status = D41DumpRows(state, 99u, 6u, state->expert_slice, routes, D41_HIDDEN_SLICE, stream);
		if ( status == SPARK_STATUS_OK )
			status = D41DumpRows(state, 99u, 11u, (const uint16_t *)state->route_packed_row, 1u, 2u * routes, stream);
		if ( status == SPARK_STATUS_OK )
			status = D41DumpRows(state, 99u, 12u, (const uint16_t *)state->route_expert, 1u, 2u * routes, stream);
		if ( status == SPARK_STATUS_OK )
			status = D41DumpRows(state, 99u, 13u, (const uint16_t *)state->packed_weight, 1u, 2u * routes, stream);
	}
	if ( status == SPARK_STATUS_OK && state->own_rows != 0u )
	{
		if ( state->own_pad != 0u )
			cudaMemsetAsync(state->moe_final + (uint64_t)rows * D41_HIDDEN_SLICE, 0, (uint64_t)state->own_pad * D41_HIDDEN_SLICE * sizeof(uint16_t), stream);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, state->own_rows, D41_HIDDEN_SLICE, state->moe_final,
			state->moe_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		D41SliceGatherKernel<<<dim3(D41Blocks(D41_HIDDEN), state->own_rows), D41_THREADS, 0, stream>>>(state->moe_gathered, state->own_out, state->own_rows,
			D41_HIDDEN_SLICE, D41_TP);
		return D41Launched();
	}
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, D41_HIDDEN_SLICE, state->moe_final,
			rows == 1u ? state->partial : state->moe_gathered);
	if ( status != SPARK_STATUS_OK || rows == 1u )
		return status;
	D41SliceGatherKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->moe_gathered, state->partial, rows, D41_HIDDEN_SLICE,
		D41_TP);
	return D41Launched();
}

static SparkStatus D41FfnGather(D41ModelState *state, const D41Layer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t own = state->own_rows, rows = step->rows, first = state->own_first;
	const uint32_t routed = first + own <= rows ? own : rows > first ? rows - first : 0u;
	uint8_t *block = (uint8_t *)state->moe_send;
	SparkStatus status;
	if ( routed != 0u &&
		(D41Project(layer->router, state->own_x, 0, state->router_logits, routed, D41_HIDDEN, D41_EXPERTS, step->multiprocessors, state->row_invariant,
		stream) != LM_LAUNCH_OK ||
		LmTopkRouteLaunch<D41_THREADS,D41_TOPK,true,LM_TOPK_SCORE_SQRT_SOFTPLUS>(routed, state->router_logits, D41_EXPERTS,
			state->route_expert + (uint64_t)first * D41_TOPK, state->route_weight + (uint64_t)first * D41_TOPK, layer->router_bias, 0, D41_ROUTE_SCALE,
			step->route_fault, stream) != cudaSuccess) )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( LmMxQuantize(state->own_x, block, block + (uint64_t)own * D41_HIDDEN, (uint64_t)own * D41_HIDDEN, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( routed != 0u )
		D41FfnPackKernel<<<(routed * D41_TOPK + D41_THREADS - 1u) / D41_THREADS, D41_THREADS, 0, stream>>>(state->route_expert + (uint64_t)first * D41_TOPK,
			state->route_weight + (uint64_t)first * D41_TOPK, block, own, routed, D41_HIDDEN, D41_TOPK);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, own, D41_FFN_GATHER_ELEMENTS, state->moe_send,
			state->moe_received);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41FfnUnpackKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>((const uint8_t *)state->moe_received, state->xq,
		state->route_expert, state->route_weight, own, D41_HIDDEN, D41_TOPK, (uint64_t)own * D41_FFN_GATHER_ELEMENTS * sizeof(uint16_t));
	return D41Launched();
}

static SparkStatus D41FillBackupPass(D41ModelState *state, const D41Layer *layer, uint32_t copy, uint32_t first_slot, uint32_t end_slot,
	cudaStream_t stream)
{
	const uint32_t index = (uint32_t)(layer - state->layers), me = state->tp_rank;
	const uint64_t g13 = state->backup_group13, g2 = state->backup_group2, row_bytes = (uint64_t)D41_HIDDEN * sizeof(uint16_t);
	const uint32_t expert_rows = (uint32_t)((g13 + g2 + row_bytes - 1u) / row_bytes), peer_rows = D41_BACKUP_PAIR * expert_rows;
	const uint64_t expert_bytes = (uint64_t)expert_rows * row_bytes, peer_bytes = (uint64_t)peer_rows * row_bytes;
	const uint8_t *owner = state->owner_host + (uint64_t)index * D41_EXPERTS;
	const uint8_t *backup = state->backup_host + ((uint64_t)index * D41_BACKUP_COPIES + copy) * D41_EXPERTS;
	const uint8_t *group = state->group_host + (uint64_t)index * D41_EXPERTS;
	const uint16_t *at = state->expert_at_host + (uint64_t)index * D41_TP * D41_LOCAL_EXPERTS;
	const uint32_t low = D41_LOCAL_EXPERTS + copy * D41_LOCAL_EXPERTS, high = low + D41_LOCAL_EXPERTS;
	uint8_t *send = (uint8_t *)state->moe_send, *received = (uint8_t *)state->moe_received;
	uint32_t sent[D41_TP] = {0}, got[D41_TP] = {0};
	SparkStatus status;
	if ( (uint64_t)D41_TP * peer_bytes > (uint64_t)D41_TP * state->max_rows * D41_MOE_ROW_ELEMENTS * sizeof(uint16_t) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for ( uint32_t slot = first_slot; slot < end_slot; ++slot )
	{
		const uint32_t expert = at[me * D41_LOCAL_EXPERTS + slot], target = backup[expert];
		uint8_t *record = send + target * peer_bytes + (uint64_t)sent[target] * expert_bytes;
		if ( target >= D41_TP || ++sent[target] > D41_BACKUP_PAIR ||
			cudaMemcpyAsync(record, layer->e_w13_cells + slot * g13, g13, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(record + g13, layer->e_w2_cells + slot * g2, g2, cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, peer_rows, D41_HIDDEN, state->moe_send, state->moe_received);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( uint32_t home = 0u; home < D41_TP; ++home )
		for ( uint32_t slot = first_slot; slot < end_slot; ++slot )
		{
			const uint32_t expert = at[home * D41_LOCAL_EXPERTS + slot];
			if ( owner[expert] != home || backup[expert] != me )
				continue;
			const uint8_t *record = received + home * peer_bytes + (uint64_t)got[home]++ * expert_bytes;
			const uint64_t place = (uint64_t)group[expert] - D41_LOCAL_EXPERTS;
			if ( group[expert] < low || group[expert] >= high ||
				cudaMemcpyAsync((uint8_t *)layer->e_w13_backup + place * g13, record, g13, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
				cudaMemcpyAsync((uint8_t *)layer->e_w2_backup + place * g2, record + g13, g2, cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
	return D41Launched();
}

static SparkStatus D41FillBackups(D41ModelState *state, const D41Layer *layer, cudaStream_t stream)
{
	const uint32_t index = (uint32_t)(layer - state->layers);
	for ( uint32_t copy = 0u; copy < D41_BACKUP_COPIES; ++copy )
		for ( uint32_t first = 0u; first < D41_LOCAL_EXPERTS; first += D41_BACKUP_PASS )
		{
			const SparkStatus status = D41FillBackupPass(state, layer, copy, first, first + D41_BACKUP_PASS < D41_LOCAL_EXPERTS ? first + D41_BACKUP_PASS :
				D41_LOCAL_EXPERTS, stream);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
	state->backup_filled |= 1ull << index;
	return SPARK_STATUS_OK;
}

static SparkStatus D41FfnExchange(D41ModelState *state, const D41Layer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t own = state->own_rows, rows = step->rows, first = state->own_first, sms = step->multiprocessors;
	const uint32_t routed = first + own <= rows ? own : rows > first ? rows - first : 0u, shuffle = (own + D41_TP - 1u) / D41_TP;
	const uint32_t index = (uint32_t)(layer - state->layers);
	const uint32_t held = index < D41_BACKUP_LAYERS && state->backup_cells != 0 && shuffle <= UINT8_MAX ? 1u : 0u;
	const uint64_t record_bytes = (uint64_t)D41_FFN_GATHER_ELEMENTS * sizeof(uint16_t);
	const uint64_t counts_bytes = held != 0u ? sizeof(D41Counts) : offsetof(D41Counts, routes);
	const uint8_t *owner = state->expert_owner + (uint64_t)index * D41_EXPERTS;
	uint8_t *codes = (uint8_t *)state->attn_merged, *scales = codes + (uint64_t)own * D41_HIDDEN;
	uint8_t *moe_codes = codes, *moe_scales = codes + (uint64_t)D41_TP * shuffle * D41_HIDDEN;
	uint32_t pair[D41_TP * D41_TP], capacity = 0u, offload = 0u;
	SparkStatus status = held != 0u && ((state->backup_filled >> index) & 1u) == 0u ? D41FillBackups(state, layer, stream) : SPARK_STATUS_OK;
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( routed != 0u &&
		(D41Project(layer->router, state->own_x, 0, state->router_logits, routed, D41_HIDDEN, D41_EXPERTS, sms, state->row_invariant, stream) != LM_LAUNCH_OK ||
		LmTopkRouteLaunch<D41_THREADS,D41_TOPK,true,LM_TOPK_SCORE_SQRT_SOFTPLUS>(routed, state->router_logits, D41_EXPERTS, state->route_expert,
			state->route_weight, layer->router_bias, 0, D41_ROUTE_SCALE, step->route_fault, stream) != cudaSuccess) )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( LmMxQuantize(state->own_x, codes, scales, (uint64_t)own * D41_HIDDEN, stream) != LM_LAUNCH_OK ||
		cudaEventRecord(state->shared_fork, stream) != cudaSuccess || cudaStreamWaitEvent(state->shared_stream, state->shared_fork, 0u) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( routed != 0u &&
		(D41LinearPair(&layer->sh_w1, &layer->sh_w3, state->own_x, state->shared_gate, state->shared_up, routed, D41_HIDDEN, state->shared_width,
			state->shared_width, sms, state->row_invariant, &state->mx, state->shared_stream) != LM_LAUNCH_OK) )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( routed != 0u )
		D41SwigluRowsKernel<<<dim3(D41Blocks(state->shared_width), routed), D41_THREADS, 0, state->shared_stream>>>(state->shared_gate, state->shared_up,
			state->shared_act, state->shared_width, D41_SWIGLU_LIMIT);
	if ( (routed != 0u && D41QdqLinear(&layer->sh_w2, state->shared_act, state->shared_out, routed, state->shared_width, D41_HIDDEN, sms,
		state->row_invariant, &state->mx, state->shared_stream) != LM_LAUNCH_OK) || cudaEventRecord(state->shared_join, state->shared_stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41ExchangeCountsKernel<D41_TP,D41_EXPERTS><<<1u, 1024u, 0, stream>>>(state->route_expert, routed, first, D41_TOPK, owner, held, state->exchange_count);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, 1u, (uint32_t)(counts_bytes / sizeof(uint16_t)),
			state->exchange_count, state->exchange_counts);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41ExchangeTotalsKernel<D41_TP,D41_EXPERTS><<<held != 0u ? 7u : 1u, 1024u, 0, stream>>>((const uint8_t *)state->exchange_counts, counts_bytes, held,
		state->exchange_totals);
	if ( D41Launched() != SPARK_STATUS_OK ||
		cudaMemcpyAsync(state->exchange_totals_host, state->exchange_totals, held != 0u ? sizeof(D41Totals) : sizeof(state->exchange_totals->pair),
			cudaMemcpyDeviceToHost, stream) != cudaSuccess || cudaEventRecord(state->counts_ready, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( routed != 0u )
		D41ShufflePackKernel<<<routed, 256u, 0, stream>>>(codes, scales, state->route_expert, state->route_weight, first, D41_TP, shuffle, D41_HIDDEN,
			D41_TOPK, record_bytes, (uint8_t *)state->moe_send);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, shuffle, D41_FFN_GATHER_ELEMENTS, state->moe_send,
			state->moe_received);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( cudaEventSynchronize(state->counts_ready) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( held != 0u )
		offload = D41BackupPlan<D41_TP,D41_EXPERTS,D41_LOCAL_EXPERTS,D41_BACKUP_COPIES>(state->exchange_totals_host,
			state->owner_host + (uint64_t)index * D41_EXPERTS, state->backup_host + (uint64_t)index * D41_BACKUP_COPIES * D41_EXPERTS,
			state->expert_at_host + (uint64_t)index * D41_TP * D41_LOCAL_EXPERTS, D41_TP * shuffle, state->exchange_quota_host, pair);
	else
		memcpy(pair, state->exchange_totals_host->pair, sizeof(pair));
	state->shuffle_rows = shuffle;
	state->shuffle_received.first[0] = 0u;
	state->exchange_received.first[0] = 0u;
	for ( uint32_t source = 0u; source < D41_TP; ++source )
	{
		state->shuffle_received.first[source + 1u] = state->shuffle_received.first[source] + D41ShuffleCount(source, state->tp_rank, rows, own, D41_TP);
		state->exchange_received.first[source + 1u] = state->exchange_received.first[source] + pair[source * D41_TP + state->tp_rank];
		for ( uint32_t target = 0u; target < D41_TP; ++target )
			capacity = pair[source * D41_TP + target] > capacity ? pair[source * D41_TP + target] : capacity;
	}
	capacity = capacity != 0u ? (capacity + 7u) / 8u * 8u : 8u;
	if ( capacity > D41_TP * shuffle )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->exchange_rows = capacity;
	state->moe_rows = state->shuffle_received.first[D41_TP];
	state->exchange_live = state->exchange_received.first[D41_TP];
	if ( offload != 0u && cudaMemcpy2DAsync(state->exchange_quota, D41_EXPERTS * sizeof(uint16_t),
		state->exchange_quota_host + (uint64_t)state->tp_rank * D41_EXPERTS, (uint64_t)D41_TP * D41_EXPERTS * sizeof(uint16_t), D41_EXPERTS * sizeof(uint16_t),
		D41_BACKUP_COPIES, cudaMemcpyHostToDevice, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( state->moe_rows != 0u )
		D41ShuffleUnpackKernel<D41_TP><<<dim3(2u, state->moe_rows), D41_THREADS, 0, stream>>>((const uint8_t *)state->moe_received, state->shuffle_received,
			shuffle, D41_HIDDEN, D41_TOPK, record_bytes, moe_codes, moe_scales, state->route_expert, state->route_weight);
	D41ExchangeSlotsKernel<D41_TP,D41_EXPERTS,D41_TOPK,D41_BACKUP_COPIES><<<1u, D41K_SLOT_WARPS * 32u, 0, stream>>>(state->route_expert, state->moe_rows,
		owner, state->expert_backup + (uint64_t)index * D41_BACKUP_COPIES * D41_EXPERTS, offload != 0u ? state->exchange_quota : (const uint16_t *)0, capacity,
		state->exchange_slot,
		state->exchange_slot_count);
	if ( state->moe_rows != 0u )
		D41ExchangePackKernel<<<dim3(state->moe_rows, D41_TP), 256u, 0, stream>>>(moe_codes, moe_scales, state->route_expert, state->route_weight,
			state->exchange_slot, owner, D41_TP, capacity, D41_HIDDEN, D41_TOPK, D41_EXPERTS, record_bytes, (uint8_t *)state->moe_send);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, capacity, D41_FFN_GATHER_ELEMENTS, state->moe_send,
			state->moe_received);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( state->exchange_live != 0u )
		D41ExchangeUnpackKernel<D41_TP><<<dim3(1u, state->exchange_live), 32u, 0, stream>>>((const uint8_t *)state->moe_received, state->exchange_received,
			capacity, D41_HIDDEN, D41_TOPK, record_bytes, (uint8_t *)0, (uint8_t *)0, state->route_expert, state->route_weight);
	return D41Launched();
}

static SparkStatus D41DumpRows(D41ModelState *state, uint32_t layer, uint32_t kind, const uint16_t *rows_bf16, uint32_t rows,
	uint32_t width, cudaStream_t stream)
{
	const uint32_t header[5] = { 0x58344c44u, layer, rows, kind, width };
	if ( state->dump == 0 )
		return SPARK_STATUS_OK;
	if ( cudaStreamSynchronize(stream) != cudaSuccess ||
		cudaMemcpy(state->dump_streams, rows_bf16, (size_t)rows * width * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	fwrite(header, sizeof(header), 1u, state->dump);
	fwrite(state->dump_streams, sizeof(uint16_t), (size_t)rows * width, state->dump);
	fflush(state->dump);
	return SPARK_STATUS_OK;
}

static SparkStatus D41DumpStep(D41ModelState *state, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t header[3] = { 0x53344c44u, step->rows, step->sequences };
	const uint64_t words = 3u * (uint64_t)step->rows + step->sequences + 1u + step->sequences;
	uint32_t *host = (uint32_t *)malloc((size_t)words * sizeof(uint32_t));
	uint32_t *cursor = host;
	cudaError_t error = host == 0 ? cudaErrorMemoryAllocation : cudaStreamSynchronize(stream);
	if ( error == cudaSuccess )
		error = cudaMemcpy(cursor, step->positions, (size_t)step->rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cursor += step->rows;
	if ( error == cudaSuccess )
		error = cudaMemcpy(cursor, step->sequence_of_row, (size_t)step->rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cursor += step->rows;
	if ( error == cudaSuccess )
		error = cudaMemcpy(cursor, step->sequence_row_indices, (size_t)step->rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cursor += step->rows;
	if ( error == cudaSuccess )
		error = cudaMemcpy(cursor, step->sequence_row_begin, (size_t)(step->sequences + 1u) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cursor += step->sequences + 1u;
	if ( error == cudaSuccess )
		error = cudaMemcpy(cursor, step->recurrent_index, (size_t)step->sequences * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess )
	{
		fwrite(header, sizeof(header), 1u, state->dump);
		fwrite(host, sizeof(uint32_t), (size_t)words, state->dump);
		fflush(state->dump);
	}
	free(host);
	return error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static uint32_t D41PlanLanding(D41ModelState *state, uint32_t landing, uint32_t ratio, uint32_t caches, uint64_t layer_rows, uint32_t reach)
{
	const uint64_t keys = SPARK_DSV41_FLASH_STAGE_LANDING_KEYS(reach, ratio), landing_rows = SPARK_DSV41_FLASH_STAGE_LANDING_ROWS(reach, ratio);
	const uint64_t row_bytes = (uint64_t)D41_HIDDEN * sizeof(uint16_t), peers = D41_TP - 1u;
	const uint64_t send_bytes = (uint64_t)D41_TP * state->max_rows * (D41_LOCAL_HEADS * D41_HEAD_DIM + D41_INDEX_PACK) * sizeof(uint16_t);
	const uint64_t gathered = peers * row_bytes * (caches * landing_rows + D41_RING_LANDING_ROWS);
	const uint64_t scattered = peers * layer_rows * ((uint64_t)D41_RECORD_ROW_ELEMENTS + D41_LOCAL_HEADS * D41_HEAD_DIM) * sizeof(uint16_t);
	if ( landing_rows > SPARK_DSV41_FLASH_STAGE_LANDING_ROWS(SPARK_DSV41_FLASH_STAGE_MAX_POSITIONS, 1u) ||
		landing_rows > SPARK_DSV41_FLASH_STAGE_LANDING_ROWS(state->max_positions, 1u) || keys * D41_SLOT_BYTES > send_bytes || gathered >= scattered )
		return 0u;
	state->landing_keys[landing] = (uint32_t)keys;
	state->landing_rows[landing] = (uint32_t)landing_rows;
	state->landing_stride[landing] = landing_rows * row_bytes;
	return 1u;
}

static uint32_t D41TailFirstRow(const D41ModelState *state, uint32_t index, uint32_t rows);

static void D41PlanLocalAttention(D41ModelState *state, uint32_t rows, uint32_t reach)
{
	uint64_t upper_rows = 0u;
	for ( uint32_t index = D41_CANDIDATE_SOURCE; index < D41_LAYERS; ++index )
		upper_rows += rows - D41TailFirstRow(state, index, rows);
	state->local_attention = D41PlanLanding(state, 0u, 2u, SPARK_DSV41_FLASH_STAGE_RATIO2_CACHES, (uint64_t)rows * D41_CANDIDATE_SOURCE, reach);
	state->local_upper = upper_rows != 0u ? D41PlanLanding(state, 1u, 1u, SPARK_DSV41_FLASH_STAGE_RATIO1_CACHES, upper_rows, reach) : 0u;
}

static SparkStatus D41StepReach(D41ModelState *state, const SparkStageRunnerStep *step)
{
	const uint32_t uncaptured = step->commit != 0u && step->rows > step->sequences && step->host_positions != 0 ? 1u : 0u;
	uint32_t reach = state->max_positions, row, contiguous = 1u;
	state->tail_target = 0u;
	state->local_attention = 0u;
	state->local_upper = 0u;
	if ( uncaptured != 0u )
		for ( row = 0u, reach = 0u; row < step->rows; ++row )
		{
			reach = step->host_positions[row] + 1u > reach ? step->host_positions[row] + 1u : reach;
			contiguous &= step->host_positions[row] == step->host_positions[0] + row ? 1u : 0u;
		}
	if ( reach > state->max_positions )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->step_score_stride = ((SparkKvShardGatherKeys(state->shard, reach) + 63u) / 64u) * 64u;
	state->step_score_stride = state->step_score_stride < state->local_score_stride ? state->step_score_stride : state->local_score_stride;
	state->step_ratio2_stride = ((SparkKvShardGatherKeys(state->shard, reach / 2u) + 63u) / 64u) * 64u;
	state->step_ratio2_stride = state->step_ratio2_stride > D41_INDEX_TOPK ? state->step_ratio2_stride : D41_INDEX_TOPK;
	state->step_ratio2_stride = state->step_ratio2_stride < state->step_score_stride ? state->step_ratio2_stride : state->step_score_stride;
	state->step_blocks = state->candidate_blocks != 0u && reach > D41_CANDIDATE_ROWS ? 1u : 0u;
	if ( uncaptured != 0u && contiguous != 0u && step->sequences == 1u && step->rows > D41_TAIL_MIN_ROWS && step->last_rows_only != 0u &&
		step->every_row_hidden == 0u && step->host_state_target != 0 && step->host_state_target[0] >= reach )
	{
		state->tail_target = step->host_state_target[0];
		state->tail_first_position = step->host_positions[0];
	}
	if ( !D41_WHOLE && step->host_positions != 0 && step->sequences == 1u && step->rows > SPARK_STAGE_RUNNER_GRAPH_ROWS &&
		step->rows > D41_PAIR_RECORD_ROWS )
	{
		uint32_t landing_reach = 0u;
		for ( row = 0u; row < step->rows; ++row )
			landing_reach = step->host_positions[row] + 1u > landing_reach ? step->host_positions[row] + 1u : landing_reach;
		if ( landing_reach <= state->max_positions )
			D41PlanLocalAttention(state, step->rows, landing_reach);
	}
	return SPARK_STATUS_OK;
}

static uint32_t D41TailReach(uint32_t index)
{
	return index >= D41_LAYERS ? 1u : D41_WINDOW + (D41_LAYERS - 1u - index) * (D41_WINDOW - 1u);
}

static uint32_t D41TailFirstRow(const D41ModelState *state, uint32_t index, uint32_t rows)
{
	const uint32_t reach = D41TailReach(index);
	const uint32_t first = state->tail_target > reach ? state->tail_target - reach : 0u;
	uint32_t skipped;
	if ( state->tail_target == 0u || index < D41_CANDIDATE_SOURCE || first <= state->tail_first_position )
		return 0u;
	skipped = first - state->tail_first_position < rows ? first - state->tail_first_position : rows;
	return skipped == rows || rows - skipped > D41_TAIL_MIN_ROWS ? skipped : rows - D41_TAIL_MIN_ROWS - 1u;
}

static void D41TailRebase(D41ModelState *state, uint32_t rows)
{
	state->streams += (uint64_t)rows * D41K_HC * D41_HIDDEN;
	state->pre_head += (uint64_t)rows * D41K_HC;
	state->selected += (uint64_t)rows * D41_INDEX_TOPK;
	if ( state->block_mask != 0 )
		state->block_mask += (uint64_t)rows * state->local_block_stride;
}

static SparkStatus D41TailStep(D41ModelState *state, const SparkStageRunnerStep *step, uint32_t first, SparkStageRunnerStep *tail,
	cudaStream_t stream)
{
	*tail = *step;
	tail->rows = step->rows - first;
	tail->hidden_bf16 = step->hidden_bf16 + (uint64_t)first * D41_HIDDEN;
	tail->positions = step->positions + first;
	tail->sequence_of_row = step->sequence_of_row + first;
	tail->sequence_row_begin = state->tail_row_begin;
	tail->token_ids = step->token_ids + first;
	tail->host_positions = step->host_positions + first;
	tail->host_sequence_of_row = step->host_sequence_of_row != 0 ? step->host_sequence_of_row + first : 0;
	if ( tail->rows == 0u )
		return SPARK_STATUS_OK;
	D41RowBeginKernel<<<1u, 1u, 0, stream>>>(state->tail_row_begin, tail->rows);
	D41CompressedEmitPositionKernel<<<(tail->rows + 255u) / 256u, 256u, 0, stream>>>(tail->positions, state->bound_ratio2, tail->rows, 2u);
	D41CompressedEmitPositionKernel<<<(tail->rows + 255u) / 256u, 256u, 0, stream>>>(tail->positions, state->bound_ratio1, tail->rows, 1u);
	LmRowLanesKernel<<<1u, 256u, 0, stream>>>(tail->sequence_row_begin, tail->sequence_row_indices, state->row_lane, state->row_ordinal);
	return D41Launched();
}

static uint32_t D41TailOwn(uint32_t tail_rows)
{
	return (tail_rows + D41_OWN_ROWS_STEP - 1u) / D41_OWN_ROWS_STEP * (D41_OWN_ROWS_STEP / D41_TP);
}

static uint32_t D41TailOwnable(const D41ModelState *state, const SparkStageRunnerStep *step, uint32_t index, uint32_t first)
{
	const uint32_t tail_own = first < step->rows ? D41TailOwn(step->rows - first) : 0u;
	return D41_ROWS && index == D41_CANDIDATE_SOURCE && state->tail_target != 0u && state->own_rows != 0u && first != 0u && first < step->rows &&
		tail_own > LM_SKINNY_ROWS_WIDE && tail_own * D41_TP <= state->max_rows && D41Exchanges(state) != 0u &&
		(uint64_t)D41_TP * tail_own * (D41K_HC * D41_HIDDEN + 8u) <= (uint64_t)D41_TP * state->max_rows * D41_MOE_ROW_ELEMENTS ? 1u : 0u;
}

static SparkStatus D41TailCompress(D41ModelState *state, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const D41Layer *layer = &state->layers[D41_CANDIDATE_SOURCE];
	LmHcCollapseKernel<D41K_HC><<<dim3(D41Blocks(D41_HIDDEN), step->rows), D41_THREADS, 0, stream>>>(state->streams, state->pre_head, state->x, D41_HIDDEN);
	D41RmsNorm(state->x, layer->attn_norm, state->x, step->rows, D41_HIDDEN, stream);
	return D41Compress(state, layer, D41_CANDIDATE_SOURCE, step, stream);
}

struct D41Cuda : CudaStage<D41Cuda>
{
	using State = D41ModelState;
	using Schedule = D41Schedule<D41Cuda>;
	using Kv = CudaStageKv<CudaStageKvViews<D41Ratio2Kv, &State::latent_views, 0u, 3u>, CudaStageKvViews<D41Ratio1Kv, &State::latent_views, 1u, 1u, 3u>,
		CudaStageKvViews<D41Ratio2Kv, &State::index_views, 0u, 3u, 0u, D41_HEAD_DIM * sizeof(uint16_t)>,
		CudaStageKvViews<D41Ratio1Kv, &State::index_views, 1u, 1u, 3u, D41_HEAD_DIM * sizeof(uint16_t)>>;

	static constexpr bool kGraphKey = true;
	static constexpr bool kKeepRows = true;
	static constexpr bool kTapPlan = true;
	static constexpr bool kCheck = true;
	static constexpr uint32_t kTpDegree = D41_TP;

	static const char *Tag(void)
	{
		return "sparkpipe_dsv41_flash";
	}

	static const char *ReportTag(void)
	{
		return "DSV41-STATE";
	}

	static const char *WeightdModel(void)
	{
		return "deepseek-v4.1-flash";
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration)
	{
		return configuration->kv_pages_per_sequence != 0u && configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_FP8_E4M3 ? 1u : 0u;
	}

	static SparkStatus Configure(D41ModelState *state, const SparkStageRunnerConfiguration *configuration, SparkStageRunnerModelGeometry *geometry);
	static uint64_t Plan(D41ModelState *state, uint8_t *base, uint32_t lane);
	static SparkStatus Bind(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry);
	static SparkStatus Upload(D41ModelState *state);
	static void Release(D41ModelState *state);
	static uint32_t GraphKey(void *model, const SparkStageRunnerStep *step);
	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream_pointer);
	static SparkStatus TapPlan(void *model, const SparkStageRunnerTapPlan *plan);

	static uint32_t RecurrentParts(const D41ModelState *state, CudaStageRecurrent *parts)
	{
		parts[0] = { state->recurrent, state->recurrent_bytes, 1u };
		return 1u;
	}

	static uint32_t ReportFields(const D41ModelState *state, CudaStageReportField *fields)
	{
		const CudaStageReportField report[] =
		{
			{ "arena_bytes", state->arena_bytes }, { "fp8_side_bytes", state->fp8_side_bytes }, { "slots", state->slots }, { "rows", state->max_rows },
			{ "positions", state->max_positions }, { "recurrent_bytes", state->recurrent_bytes }
		};
		return CudaStageReportOut(fields, report);
	}

	static SparkStatus CheckErrors(void *model)
	{
		const D41ModelState *state = (const D41ModelState *)model;
		const volatile LmKvAccessError *error = state->host_error;
		if ( state->engram_failed != 0u )
		{
			fprintf(stderr, "sparkpipe_dsv41_flash: a row of the engram tables could not be read from %s\n", state->engram_path);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		if ( error == 0 || error->error_code == 0u )
			return SPARK_STATUS_OK;
		fprintf(stderr, "sparkpipe_dsv41_flash: KV access error code=%u kind=%u row=%u sequence=%u position=%u page=%u\n", error->error_code,
			error->access_kind, error->row, error->sequence, error->position, error->page);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}

	struct Frame
	{
		const SparkStageRunnerStep *step;
		cudaStream_t stream;
		SparkStageRunnerStep tail;
		const SparkStageRunnerStep *current;
		uint32_t shifted;
		uint16_t *streams;
		float *pre_head;
		uint32_t *selected;
		uint8_t *block_mask;
	};

	static SparkStatus Begin(D41ModelState *state, const SparkStageRunnerStep *step, void *stream_void, Frame *frame)
	{
		cudaStream_t stream = (cudaStream_t)stream_void;
		const uint32_t rows = step->rows;
		SparkStatus status = SPARK_STATUS_OK;
		if ( rows == 0u || rows > state->max_rows || step->sequences > state->slots || state->kv_attached == 0u || step->token_ids == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( state->dump != 0 )
		{
			cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
			if ( cudaStreamIsCapturing(stream, &capturing) != cudaSuccess || capturing != cudaStreamCaptureStatusNone )
				return SPARK_STATUS_UNSUPPORTED;
			status = D41DumpStep(state, step, stream);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		state->row_invariant = step->commit == 0u ? 1u : 0u;
		status = D41StepReach(state, step);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( state->tap_first_row != 0 && state->tail_target != 0u && state->tap_reach < state->tail_target )
		{
			const uint32_t first = state->tail_target - state->tap_reach;
			*state->tap_first_row = first <= state->tail_first_position ? 0u :
				first - state->tail_first_position < rows ? first - state->tail_first_position : rows;
		}
		D41CompressedEmitPositionKernel<<<(rows + 255u) / 256u, 256u, 0, stream>>>(step->positions, state->bound_ratio2, rows, 2u);
		D41CompressedEmitPositionKernel<<<(rows + 255u) / 256u, 256u, 0, stream>>>(step->positions, state->bound_ratio1, rows, 1u);
		LmRowLanesKernel<<<step->sequences, 256u, 0, stream>>>(step->sequence_row_begin, step->sequence_row_indices, state->row_lane,
			state->row_ordinal);
		status = D41EngramAheadBegin(state, step, stream);
		if ( status != SPARK_STATUS_OK )
			return status;
		D41CompressedLengthKernel<<<(state->context_slots + 255u) / 256u, 256u, 0, stream>>>(step->context_length, state->length_ratio2,
			state->context_slots, 2u);
		D41CompressedLengthKernel<<<(state->context_slots + 255u) / 256u, 256u, 0, stream>>>(step->context_length, state->length_ratio1,
			state->context_slots, 1u);
		LmHcExpandStreamsKernel<D41_HIDDEN><<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(step->hidden_bf16, state->streams, rows);
		state->own_rows = (rows + D41_OWN_ROWS_STEP - 1u) / D41_OWN_ROWS_STEP * (D41_OWN_ROWS_STEP / D41_TP);
		if ( step->commit == 0u || rows < D41_OWN_ROWS_MIN || state->own_rows * D41_TP > state->max_rows || state->tap_slab != 0 || state->dump != 0 )
			state->own_rows = 0u;
		state->own_first = state->tp_rank * state->own_rows;
		state->own_pad = state->own_rows != 0u ? state->own_rows * D41_TP - rows : 0u;
		state->tail_owned = 0u;
		if ( state->own_rows != 0u && state->o_b_ready == 0u )
		{
			status = D41_ROWS ? SPARK_STATUS_OK : D41OwnerOutputWeights(state, stream);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		if ( state->own_pad != 0u )
			cudaMemsetAsync(state->streams + (uint64_t)rows * D41K_HC * D41_HIDDEN, 0, (uint64_t)state->own_pad * D41K_HC * D41_HIDDEN * sizeof(uint16_t), stream);
		if ( state->local_attention != 0u || state->local_upper != 0u )
		{
			D41RingPackKernel<<<D41_LAYERS, D41_THREADS, 0, stream>>>((const uint16_t *)state->recurrent, step->recurrent_index,
				state->recurrent_bytes / sizeof(uint16_t), D41_RING_ELEMENTS, state->ring_send);
			status = D41Launched();
			if ( status == SPARK_STATUS_OK )
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, D41_RING_LANDING_ROWS, D41_HIDDEN, state->ring_send,
					state->ring_landing);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		frame->step = step;
		frame->stream = stream;
		frame->streams = state->streams;
		frame->pre_head = state->pre_head;
		frame->selected = state->selected;
		frame->block_mask = state->block_mask;
		frame->tail = *step;
		frame->current = step;
		frame->shifted = 0u;
		return SPARK_STATUS_OK;
	}

	static SparkStatus WholeRows(D41ModelState *state, cudaStream_t stream)
	{
		const uint32_t rows = state->own_rows;
		SparkStatus status;
		if ( rows == 0u )
			return SPARK_STATUS_OK;
		status = D41Launched();
		if ( status == SPARK_STATUS_OK )
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows * D41K_HC, D41_HIDDEN,
				state->streams + (uint64_t)state->own_first * D41K_HC * D41_HIDDEN, state->streams_whole);
		if ( status == SPARK_STATUS_OK )
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, D41K_HC * 2u,
				state->pre_head + (uint64_t)state->own_first * D41K_HC, state->pre_whole);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->streams = state->streams_whole;
		state->pre_head = state->pre_whole;
		state->own_rows = 0u;
		state->own_first = 0u;
		state->own_pad = 0u;
		return SPARK_STATUS_OK;
	}

	static SparkStatus OwnTail(D41ModelState *state, Frame &frame, uint32_t index, uint32_t first)
	{
		const SparkStageRunnerStep *step = frame.step;
		const uint32_t rows = step->rows, own = state->own_rows, tail_rows = rows - first, tail_own = D41TailOwn(tail_rows);
		const uint32_t width = D41K_HC * D41_HIDDEN, pad = tail_own * D41_TP - tail_rows;
		uint32_t capacity = 0u, low;
		SparkStatus status;
		for ( uint32_t source = 0u; source < D41_TP; ++source )
			for ( uint32_t target = 0u; target < D41_TP; ++target )
			{
				const uint32_t count = D41TailSpan(source, target, rows, first, own, tail_own, &low);
				capacity = count > capacity ? count : capacity;
			}
		if ( capacity == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CollapseNorm(state, frame, index, D41_ATTENTION_PART);
		if ( status == SPARK_STATUS_OK )
			status = D41Compress(state, &state->layers[index], index, step, frame.stream);
		state->compress_done = 1u;
		if ( status != SPARK_STATUS_OK )
			return status;
		uint16_t *send_pre = state->moe_send + (uint64_t)D41_TP * capacity * width, *received_pre = state->moe_received + (uint64_t)D41_TP * capacity * width;
		D41TailPackKernel<<<dim3(8u, capacity, D41_TP), 256u, 0, frame.stream>>>(state->streams, state->pre_head, state->tp_rank, rows, first, own, tail_own,
			capacity, width, (uint4 *)state->moe_send, (uint4 *)send_pre);
		status = D41Launched();
		if ( status == SPARK_STATUS_OK )
			status = CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, capacity * D41K_HC, D41_HIDDEN, state->moe_send,
				state->moe_received);
		if ( status == SPARK_STATUS_OK )
			status = CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, capacity, 8u, send_pre, received_pre);
		if ( status != SPARK_STATUS_OK )
			return status;
		D41TailUnpackKernel<<<dim3(8u, capacity, D41_TP), 256u, 0, frame.stream>>>((const uint4 *)state->moe_received, (const uint4 *)received_pre,
			state->tp_rank, rows, first, own, tail_own, capacity, width, state->streams_whole, state->pre_whole);
		if ( pad != 0u && (cudaMemsetAsync(state->streams_whole + (uint64_t)tail_rows * width, 0, (uint64_t)pad * width * sizeof(uint16_t), frame.stream) !=
			cudaSuccess || cudaMemsetAsync(state->pre_whole + (uint64_t)tail_rows * D41K_HC, 0, (uint64_t)pad * D41K_HC * sizeof(float), frame.stream) !=
			cudaSuccess) )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		state->streams = state->streams_whole;
		state->pre_head = state->pre_whole;
		state->selected += (uint64_t)first * D41_INDEX_TOPK;
		if ( state->block_mask != 0 )
			state->block_mask += (uint64_t)first * state->local_block_stride;
		state->own_rows = tail_own;
		state->own_first = state->tp_rank * tail_own;
		state->own_pad = pad;
		state->tail_owned = 1u;
		frame.shifted = first;
		status = D41TailStep(state, step, first, &frame.tail, frame.stream);
		frame.current = &frame.tail;
		return status;
	}

	static SparkStatus LayerRows(D41ModelState *state, Frame &frame, uint32_t index)
	{
		const uint32_t first = D41TailFirstRow(state, index, frame.step->rows);
		if ( state->tail_owned != 0u )
			return SPARK_STATUS_OK;
		if ( D41TailOwnable(state, frame.step, index, first) != 0u )
			return OwnTail(state, frame, index, first);
		const uint32_t handoff = state->tail_target != 0u && index == D41_CANDIDATE_SOURCE && first == frame.step->rows && state->own_rows != 0u ? 1u : 0u;
		SparkStatus status = state->tail_target != 0u && index >= D41_CANDIDATE_SOURCE && handoff == 0u ? WholeRows(state, frame.stream) : SPARK_STATUS_OK;
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( handoff != 0u )
		{
			status = CollapseNorm(state, frame, index, D41_ATTENTION_PART);
			if ( status == SPARK_STATUS_OK )
				status = D41Compress(state, &state->layers[index], index, frame.step, frame.stream);
			state->compress_done = 1u;
			state->own_rows = 0u;
			state->own_first = 0u;
			state->own_pad = 0u;
		}
		else if ( index == D41_CANDIDATE_SOURCE && first != 0u )
		{
			status = D41TailCompress(state, frame.step, frame.stream);
			state->compress_done = 1u;
		}
		if ( status == SPARK_STATUS_OK && first != frame.shifted )
		{
			D41TailRebase(state, first - frame.shifted);
			frame.shifted = first;
			status = D41TailStep(state, frame.step, first, &frame.tail, frame.stream);
			frame.current = &frame.tail;
		}
		return status;
	}

	static bool HasRows(const Frame &frame)
	{
		return frame.current->rows != 0u;
	}

	static void LayerEnd(D41ModelState *state, Frame &frame, uint32_t index, SparkStatus status)
	{
		state->compress_done = 0u;
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr, "sparkpipe_dsv41_flash: layer %u failed status=%d rows=%u sequences=%u tail_rows=%u\n", index, (int)status, frame.step->rows,
				frame.step->sequences, frame.current->rows);
		else if ( state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, index);
	}

	static SparkStatus End(D41ModelState *state, Frame &frame, SparkStatus status)
	{
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		const uint32_t own = state->own_rows;
		if ( status == SPARK_STATUS_OK && own != 0u )
		{
			LmHcCollapseKernel<D41K_HC><<<dim3(D41Blocks(D41_HIDDEN), own), D41_THREADS, 0, stream>>>(
				state->streams + (uint64_t)state->own_first * D41K_HC * D41_HIDDEN, state->pre_head + (uint64_t)state->own_first * D41K_HC, state->own_x,
				D41_HIDDEN);
			status = D41Launched();
			if ( status == SPARK_STATUS_OK )
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, own, D41_HIDDEN, state->own_x, state->x);
			if ( status == SPARK_STATUS_OK && cudaMemcpyAsync(frame.current->hidden_bf16, state->x,
					(uint64_t)frame.current->rows * D41_HIDDEN * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
				status = SPARK_STATUS_INTERNAL_ERROR;
		}
		else if ( status == SPARK_STATUS_OK && frame.current->rows != 0u )
			LmHcCollapseKernel<D41K_HC><<<dim3(D41Blocks(D41_HIDDEN), frame.current->rows), D41_THREADS, 0, stream>>>(state->streams, state->pre_head,
				frame.current->hidden_bf16, D41_HIDDEN);
		state->own_rows = 0u;
		state->own_first = 0u;
		state->own_pad = 0u;
		state->tail_owned = 0u;
		state->streams = frame.streams;
		state->pre_head = frame.pre_head;
		state->selected = frame.selected;
		state->block_mask = frame.block_mask;
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( D41_ENGRAM_DISK )
			D41NgramHistory(state, step, stream);
		return D41Launched();
	}

	static SparkStatus Engram(D41ModelState *state, Frame &frame, uint32_t which)
	{
		return D41EngramApply(state, which, frame.current, frame.stream);
	}

	static SparkStatus HyperMix(D41ModelState *state, Frame &frame, uint32_t index, D41Part part)
	{
		const D41Layer *layer = &state->layers[index];
		const uint64_t first = state->own_first;
		const uint32_t rows = state->own_rows != 0u ? state->own_rows : frame.current->rows;
		const uint16_t *streams = state->streams + first * D41K_HC * D41_HIDDEN;
		if ( part == D41_FFN_PART )
			return D41HcMixes(state, streams, layer->hc_ffn_fn, layer->hc_ffn_base, layer->hc_ffn_scale, state->pre_head + first * D41K_HC,
				state->post_ffn + first * D41K_HC, state->comb_ffn + first * D41K_HC * D41K_HC, rows, frame.stream);
		return D41HcMixes(state, streams, layer->hc_attn_fn, layer->hc_attn_base, layer->hc_attn_scale, state->pre_attn + first * D41K_HC,
			state->post_attn + first * D41K_HC, state->comb_attn + first * D41K_HC * D41K_HC, rows, frame.stream);
	}

	static SparkStatus CollapseNorm(D41ModelState *state, Frame &frame, uint32_t index, D41Part part)
	{
		const D41Layer *layer = &state->layers[index];
		const uint32_t rows = frame.current->rows, own = state->own_rows;
		const uint64_t first = state->own_first;
		const float *pre = part == D41_FFN_PART ? state->pre_attn : index == 0u ? state->pre_init : state->pre_head;
		SparkStatus status;
		D41CollapseNormQdq(state->streams + first * D41K_HC * D41_HIDDEN, pre + first * D41K_HC, part == D41_FFN_PART ? layer->ffn_norm : layer->attn_norm,
			own != 0u ? state->own_x : state->x, own != 0u ? state->own_out : state->xq, own != 0u ? own : rows, frame.stream);
		const uint32_t gather_x = own != 0u && part != D41_FFN_PART && D41RowsAttention(state) == 0u &&
			((D41KvSource(index) && state->compress_done == 0u) || (D41Ratio(index) > 0u && D41IndexSource(index))) ? 1u : 0u;
		if ( part != D41_FFN_PART )
			state->x_complete = own == 0u || gather_x != 0u ? 1u : 0u;
		if ( own != 0u && (part == D41_FFN_PART || gather_x != 0u) )
		{
			status = D41Launched();
			if ( status == SPARK_STATUS_OK )
				status = part == D41_FFN_PART ? (D41Exchanges(state) != 0u ? D41FfnExchange(state, layer, frame.current, frame.stream) :
					D41FfnGather(state, layer, frame.current, frame.stream)) :
					CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, own, D41_HIDDEN, state->own_x, state->x);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		return index < 2u ? D41DumpRows(state, index, part == D41_FFN_PART ? 4u : 3u, part == D41_FFN_PART && own != 0u ? state->xq : state->x, rows,
			D41_HIDDEN, frame.stream) : SPARK_STATUS_OK;
	}

	static SparkStatus Attention(D41ModelState *state, Frame &frame, uint32_t index)
	{
		return D41Attention(state, &state->layers[index], index, frame.current, frame.stream);
	}

	static SparkStatus AttentionReduce(D41ModelState *state, Frame &frame, uint32_t index)
	{
		const uint32_t rows = frame.current->rows;
		if ( state->owner_reduced != 0u )
		{
			state->owner_reduced = 0u;
			return SPARK_STATUS_OK;
		}
		if ( state->own_pad != 0u )
			cudaMemsetAsync(state->partial + (uint64_t)rows * D41_HIDDEN, 0, (uint64_t)state->own_pad * D41_HIDDEN * sizeof(uint16_t), frame.stream);
		SparkStatus status = state->own_rows != 0u ?
			CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_REDUCE_SCATTER_SUM_BF16, state->own_rows * D41_TP, 0u, state->partial,
				state->own_out) :
			CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
		if ( status == SPARK_STATUS_OK && index < 2u )
			status = D41DumpRows(state, index, 1u, state->partial, rows, D41_HIDDEN, frame.stream);
		return status;
	}

	static SparkStatus HyperPost(D41ModelState *state, Frame &frame, D41Part part)
	{
		const bool ffn = part == D41_FFN_PART;
		const uint64_t first = state->own_first;
		const uint32_t own = state->own_rows;
		LmHcPostF32SourceMajorKernel<D41K_HC><<<dim3(D41Blocks(D41_HIDDEN), own != 0u ? own : frame.current->rows), D41_THREADS, 0, frame.stream>>>(
			own != 0u ? state->own_out : state->partial, (ffn ? state->post_ffn : state->post_attn) + first * D41K_HC,
			(ffn ? state->comb_ffn : state->comb_attn) + first * D41K_HC * D41K_HC, state->streams + first * D41K_HC * D41_HIDDEN, D41_HIDDEN);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Moe(D41ModelState *state, Frame &frame, uint32_t index)
	{
		const uint32_t rows = frame.current->rows;
		SparkStatus status = D41Moe(state, &state->layers[index], frame.current, frame.stream);
		if ( D41_SHARED_ROUTED == 0u && status == SPARK_STATUS_OK && index < 2u )
			status = D41DumpRows(state, index, 5u, state->shared_out, rows, D41_HIDDEN, frame.stream);
		if ( status == SPARK_STATUS_OK && index < 2u )
			status = D41DumpRows(state, index, 2u, state->partial, rows, D41_HIDDEN, frame.stream);
		return status;
	}

	static SparkStatus Tap(D41ModelState *state, Frame &frame, uint32_t index)
	{
		const uint32_t rows = frame.current->rows;
		if ( state->tap_slab != 0 && state->tap_output[index] >= 0 )
			LmStreamMeanRowsKernel<D41_THREADS><<<dim3((D41_HIDDEN + D41_THREADS - 1u) / D41_THREADS, rows), D41_THREADS, 0, frame.stream>>>(state->streams,
				state->tap_slab + (uint64_t)frame.shifted * state->tap_stride + (uint64_t)state->tap_output[index] * D41_HIDDEN, rows, D41K_HC, D41_HIDDEN,
				state->tap_stride);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Finish(D41ModelState *state, Frame &frame, uint32_t index)
	{
		const uint32_t rows = frame.current->rows;
		const uint64_t stream_bytes = (uint64_t)rows * D41K_HC * D41_HIDDEN * sizeof(uint16_t);
		if ( state->dump != 0 )
		{
			const uint32_t header[4] = { 0x44344c44u, index, rows, D41_TOPK };
			if ( cudaStreamSynchronize(frame.stream) != cudaSuccess ||
				cudaMemcpy(state->dump_streams, state->streams, stream_bytes, cudaMemcpyDeviceToHost) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			fwrite(header, sizeof(header), 1u, state->dump);
			fwrite(state->dump_streams, 1u, (size_t)stream_bytes, state->dump);
			fwrite(state->dump_routes, sizeof(uint32_t), (size_t)rows * D41_TOPK, state->dump);
			fwrite(state->dump_weights, sizeof(float), (size_t)rows * D41_TOPK, state->dump);
			fflush(state->dump);
		}
		return D41Launched();
	}
};

inline SparkStatus D41Cuda::KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream_pointer)
{
	D41ModelState *state = (D41ModelState *)model;
	cudaStream_t stream = (cudaStream_t)stream_pointer;
	const uint32_t rows = step->rows, state_stride = (uint32_t)(state->recurrent_bytes / sizeof(float));
	if ( state == 0 || kept == 0 || rows > D41_VERIFY_ROWS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint32_t index = 0u; index < D41_LAYERS; ++index )
		D41WindowRingStoreKernel<<<rows, D41_THREADS, 0, stream>>>(state->verify_window + (uint64_t)index * D41_VERIFY_ROWS * D41_HEAD_DIM,
			(uint16_t *)state->recurrent + (uint64_t)index * D41_RING_ROWS * D41_HEAD_DIM, step->positions, state->row_lane, state->row_ordinal,
			step->sequence_row_begin, step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_RING_ROWS, D41_WINDOW, D41_KV_TP,
			D41_KV_RANK(state), rows, 1u, kept);
	for ( uint32_t cache = 0u; cache < D41_PAIR_CACHES; ++cache )
		D41CompressorSaveKernel<<<rows, D41_THREADS, 0, stream>>>(state->verify_c_kv + (uint64_t)cache * D41_VERIFY_ROWS * D41_HEAD_DIM,
			state->verify_c_score + (uint64_t)cache * D41_VERIFY_ROWS * D41_HEAD_DIM,
			(float *)(state->recurrent + state->ring_bytes) + (uint64_t)cache * 2u * D41_HEAD_DIM, step->positions, state->row_lane,
			state->row_ordinal, step->sequence_row_begin, step->recurrent_index, rows, 2u, state_stride, D41_HEAD_DIM, 1u, kept);
	LmNgramHistoryKernel<<<(step->sequences + 63u) / 64u, 64u, 0, stream>>>(step->token_ids, state->token_map, step->sequence_row_begin,
		step->sequence_row_indices, step->recurrent_index,
		(int32_t *)(state->recurrent + state->ring_bytes + state->compressor_bytes), (uint32_t)(state->recurrent_bytes / sizeof(int32_t)),
		step->sequences, D41_ENGRAM_ORDERS, 1u, kept);
	return D41Launched();
}

static SparkStatus D41Frequencies(float *table)
{
	if ( D41LaunchFrequencies(table, D41_ROPE_THETA, D41_COMPRESS_THETA, D41_YARN_FACTOR, D41_YARN_ORIGINAL, D41_BETA_FAST,
		D41_BETA_SLOW) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static const D41PackEntry *D41Find(const D41ModelState *state, uint32_t kind, uint32_t layer)
{
	for ( uint32_t index = 0u; index < state->header.tensor_count; ++index )
		if ( state->entries[index].tensor_kind == kind && state->entries[index].layer_index == layer )
			return &state->entries[index];
	return 0;
}

static SparkStatus D41PackDirectoryCheck(const D41ModelState *state, const char *path)
{
	static const char *const reasons[] = { "", "has a kind or shape this stage does not read", "holds bytes or offsets that do not match its shape",
		"is a global tensor with a layer or a duplicate", "is not a tensor of its layer, or a duplicate", "overlaps another tensor",
		"overlaps its own scales" };
	uint32_t at = 0u;
	const int32_t check = SparkDsv41FlashStagePackDirectoryCheck(&state->header, state->entries, &at);
	if ( check == 0 )
		return SPARK_STATUS_OK;
	if ( check == SPARK_DSV41_FLASH_STAGEPACK_CHECK_GLOBALS )
		fprintf(stderr, "sparkpipe_dsv41_flash: %s lacks the embedding, final norm or head, or holds another global tensor\n", path);
	else if ( check == SPARK_DSV41_FLASH_STAGEPACK_CHECK_LAYER_SET )
		fprintf(stderr, "sparkpipe_dsv41_flash: %s layer %u does not hold exactly the tensor kinds the stage reads\n", path, at);
	else
	{
		const D41PackEntry *entry = &state->entries[at];
		fprintf(stderr, "sparkpipe_dsv41_flash: %s entry %u (kind %u layer %u, %u x %u x %u, %llu + %llu bytes) %s\n", path, at, entry->tensor_kind,
			entry->layer_index, entry->group_count, entry->rows, entry->columns, (unsigned long long)entry->payload_bytes,
			(unsigned long long)entry->scale_bytes, -check < (int32_t)(sizeof(reasons) / sizeof(reasons[0])) ? reasons[-check] : "fails the pack contract");
	}
	SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
}

inline uint64_t D41Cuda::Plan(D41ModelState *state, uint8_t *base, uint32_t lane)
{
	const uint64_t r = state->max_rows, s = state->slots, routes = r * D41_TOPK;
	const uint32_t recorded = D41_RECORDED_ROWS(state);
	const uint64_t destination = D41DestinationStride(recorded, D41_LOCAL_HEADS) * sizeof(float), scored = state->select_rows;
	const uint64_t exchanged = D41_WHOLE ? 0u : D41_TP;
	const uint64_t engram_rows = (r * D41_ENGRAM_EMBED + D41_HIDDEN - 1u) / D41_HIDDEN;
	(void)lane;
	const uint64_t sparse = D41_ROWS && !D41_ENGRAM_DISK && D41_OWN_ROWS_MIN <= r ? 1u : 0u;
	const uint64_t own_max = (r + D41_OWN_ROWS_STEP - 1u) / D41_OWN_ROWS_STEP * (D41_OWN_ROWS_STEP / D41_TP);
	const CudaStageSlice plan[] =
	{
		{ (void **)&state->streams, r * D41K_HC * D41_HIDDEN * 2u },
		{ (void **)&state->streams_whole, D41_OWN_ROWS_MIN <= r ? r * D41K_HC * D41_HIDDEN * 2u : 0u },
		{ (void **)&state->pre_whole, D41_OWN_ROWS_MIN <= r ? r * D41K_HC * 4u : 0u },
		{ (void **)&state->own_x, D41_OWN_ROWS_MIN <= r ? r / D41_TP * D41_HIDDEN * 2u : 0u },
		{ (void **)&state->own_out, D41_OWN_ROWS_MIN <= r ? r / D41_TP * D41_HIDDEN * 2u : 0u },
		{ (void **)&state->own_q, D41_OWN_ROWS_MIN <= r ? r / D41_TP * (D41_QLORA + D41_HEAD_DIM) * 2u : 0u },
		{ (void **)&state->window_edges, D41_ROWS && D41_OWN_ROWS_MIN <= r ? (uint64_t)D41_TP * D41_WINDOW_EDGE_ROWS * D41_HEAD_DIM * 2u : 0u },
		{ (void **)&state->own_c, D41_ROWS && D41_OWN_ROWS_MIN <= r ? r / D41_TP * D41_HEAD_DIM * 4u * 2u : 0u },
		{ (void **)&state->own_selected, D41_OWN_ROWS_MIN <= r ? r / D41_TP * D41_INDEX_TOPK * 4u : 0u },
		{ (void **)&state->exchange_slot, D41_ROWS && D41_OWN_ROWS_MIN <= r ? r * 4u : 0u },
		{ (void **)&state->exchange_slot_count, D41_ROWS ? D41_TP * 4u : 0u },
		{ (void **)&state->route_output_row, D41_ROWS && D41_OWN_ROWS_MIN <= r ? routes * 4u : 0u },
		{ (void **)&state->exchange_count, D41_ROWS ? sizeof(D41Counts) : 0u },
		{ (void **)&state->exchange_counts, D41_ROWS ? D41_TP * sizeof(D41Counts) : 0u },
		{ (void **)&state->exchange_totals, D41_ROWS ? sizeof(D41Totals) : 0u },
		{ (void **)&state->exchange_quota, D41_ROWS ? D41_BACKUP_COPIES * D41_EXPERTS * sizeof(uint16_t) : 0u },
		{ (void **)&state->o_b_all, !D41_ROWS && D41_OWN_ROWS_MIN <= r ? D41_LAYERS * D41_TP * D41_HIDDEN * D41_WO_A_ROWS : 0u },
		{ (void **)&state->o_b_all_scale, !D41_ROWS && D41_OWN_ROWS_MIN <= r ? D41_LAYERS * D41_TP * D41_HIDDEN * (D41_WO_A_ROWS / D41_FP8_BLOCK) : 0u },
		{ (void **)&state->mx.codes, LM_SKINNY_ROWS_WIDE < r ? r * D41_MX_WIDTH : 0u },
		{ (void **)&state->mx.scales, LM_SKINNY_ROWS_WIDE < r ? r * D41_MX_WIDTH / D41_FP8_BLOCK : 0u },
		{ (void **)&state->x, r * D41_HIDDEN * 2u }, { (void **)&state->xq, r * D41_HIDDEN * 2u },
		{ (void **)&state->q_latent, r * D41_QLORA * 2u }, { (void **)&state->qrq, r * D41_QLORA * 2u },
		{ (void **)&state->q_local, r * (D41_LOCAL_HEADS * D41_HEAD_DIM + D41_INDEX_PACK) * 2u },
		{ (void **)&state->q_gathered, D41_ROWS ? r * D41_INDEX_PACK * 2u : D41_TP * r * (D41_LOCAL_HEADS * D41_HEAD_DIM + D41_INDEX_PACK) * 2u },
		{ (void **)&state->kv, r * D41_HEAD_DIM * 2u }, { (void **)&state->window_kv, r * D41_HEAD_DIM * 2u },
		{ (void **)&state->mixes, r * LmHcSinkhornShape<D41K_HC>::kPartials * 4u },
		{ (void **)&state->pre_attn, r * D41K_HC * 4u }, { (void **)&state->post_attn, r * D41K_HC * 4u }, { (void **)&state->comb_attn, r * 16u * 4u },
		{ (void **)&state->post_ffn, r * D41K_HC * 4u }, { (void **)&state->comb_ffn, r * 16u * 4u },
		{ (void **)&state->pre_head, r * D41K_HC * 4u }, { (void **)&state->pre_init, r * D41K_HC * 4u },
		{ (void **)&state->c_kv, r * D41_HEAD_DIM * 4u }, { (void **)&state->c_score, r * D41_HEAD_DIM * 4u },
		{ (void **)&state->c_latent, r * D41_HEAD_DIM * 2u }, { (void **)&state->c_latent_store, r * D41_HEAD_DIM * 2u },
		{ (void **)&state->idx_key, r * D41_INDEX_DIM * 2u },
		{ (void **)&state->idx_pack_local, r * D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM * 2u },
		{ (void **)&state->idx_query, r * D41_INDEX_HEADS * D41_INDEX_DIM * 2u }, { (void **)&state->idx_w_full, r * D41_INDEX_HEADS * 2u },
		{ (void **)&state->emit_row, r * 4u }, { (void **)&state->bound_ratio2, r * 4u }, { (void **)&state->bound_ratio1, r * 4u },
		{ (void **)&state->length_ratio2, s * 4u }, { (void **)&state->length_ratio1, s * 4u },
		{ (void **)&state->row_lane, r * 4u }, { (void **)&state->row_ordinal, r * 4u }, { (void **)&state->tail_row_begin, 2u * 4u },
		{ (void **)&state->local_scores, scored * state->local_score_stride * 4u },
		{ (void **)&state->topk_values, state->topk_entries * 4u }, { (void **)&state->topk_positions, state->topk_entries * 4u },
		{ (void **)&state->local_selected, r * D41_INDEX_TOPK * 4u }, { (void **)&state->selected, r * D41_INDEX_TOPK * 4u },
		{ (void **)&state->candidates, r * D41_INDEX_TOPK * 8u }, { (void **)&state->candidates_gathered, D41_TP * r * D41_INDEX_TOPK * 8u },
		{ (void **)&state->block_scores, scored * state->local_block_stride * 4u },
		{ (void **)&state->block_local_selected, r * state->local_block_keep * 4u },
		{ (void **)&state->block_candidates, r * state->local_block_keep * 8u },
		{ (void **)&state->block_gathered, (uint64_t)D41_TP * D41_SELECTION_CHUNK * state->local_block_keep * 8u },
		{ (void **)&state->block_selected, (uint64_t)D41_SELECTION_CHUNK * state->global_block_keep * 4u },
		{ (void **)&state->block_mask, r * state->local_block_stride },
		{ (void **)&state->verify_window, (uint64_t)D41_LAYERS * D41_VERIFY_ROWS * D41_HEAD_DIM * 2u },
		{ (void **)&state->verify_c_kv, (uint64_t)D41_PAIR_CACHES * D41_VERIFY_ROWS * D41_HEAD_DIM * 4u },
		{ (void **)&state->verify_c_score, (uint64_t)D41_PAIR_CACHES * D41_VERIFY_ROWS * D41_HEAD_DIM * 4u },
		{ (void **)&state->records_send, exchanged * destination }, { (void **)&state->records_received, (D41_WHOLE ? 1u : D41_TP) * destination },
		{ (void **)&state->landing, exchanged * SPARK_DSV41_FLASH_STAGE_LANDING_ROWS(state->max_positions, 1u) * D41_HIDDEN * 2u },
		{ (void **)&state->ring_send, (D41_WHOLE ? 0u : 1u) * (uint64_t)D41_RING_LANDING_ROWS * D41_HIDDEN * 2u },
		{ (void **)&state->ring_landing, exchanged * D41_RING_LANDING_ROWS * D41_HIDDEN * 2u },
		{ (void **)&state->gather_plan, (3u * s + 1u) * 4u },
		{ (void **)&state->split_selected, r * D41_INDEX_TOPK * 4u }, { (void **)&state->split_offsets, r * (D41_TP + 1u) * 4u },
		{ (void **)&state->attn_merged, r * D41_LOCAL_HEADS * D41_HEAD_DIM * 2u },
		{ (void **)&state->attn_gathered, exchanged * r * D41_LOCAL_HEADS * D41_HEAD_DIM * 2u },
		{ (void **)&state->wo_a_group, (D41_WHOLE ? 1u : 0u) * r * D41_GROUP_ROWS * 2u },
		{ (void **)&state->group_input, r * 2u * D41_LOCAL_HEADS * D41_HEAD_DIM * 2u },
		{ (void **)&state->wo_a_out, r * D41_WO_A_ROWS * 2u }, { (void **)&state->partial, r * D41_HIDDEN * 2u },
		{ (void **)&state->router_logits, r * D41_EXPERTS * 4u }, { (void **)&state->route_weight, routes * 4u },
		{ (void **)&state->packed_weight, routes * 4u }, { (void **)&state->route_expert, routes * 4u },
		{ (void **)&state->route_global, state->moe_whole != 0u ? routes * 4u : 0u },
		{ (void **)&state->group_row_offset, (D41_EXPERTS + 1u) * 4u }, { (void **)&state->route_packed_row, routes * 4u },
		{ (void **)&state->route_source_token, routes * 4u }, { (void **)&state->tile_prefix_up, (D41_EXPERTS + 1u) * 4u },
		{ (void **)&state->tile_prefix_down, (D41_EXPERTS + 1u) * 4u },
		{ (void **)&state->expert_gate_up, state->moe_whole != 0u ? 0u : routes * 2u * D41_EXPERT_SLICE * 2u },
		{ (void **)&state->expert_act, routes * D41_MOE_INTER * 2u },
		{ (void **)&state->expert_slice, routes * (state->moe_whole != 0u ? D41_HIDDEN : D41_HIDDEN_SLICE) * 2u },
		{ (void **)&state->moe_send, D41_TP * r * D41_MOE_ROW_ELEMENTS * 2u }, { (void **)&state->moe_received, D41_TP * r * D41_MOE_ROW_ELEMENTS * 2u },
		{ (void **)&state->moe_final, r * (state->moe_whole != 0u ? D41_HIDDEN : D41_HIDDEN_SLICE) * 2u },
		{ (void **)&state->moe_gathered, state->moe_whole != 0u ? D41_TP * (r < D41_OWN_ROWS_MIN ? r : D41_OWN_ROWS_MIN) * D41_HIDDEN * 2u :
			D41_TP * r * D41_HIDDEN_SLICE * 2u },
		{ (void **)&state->shared_gate, r * D41_MOE_INTER * 2u }, { (void **)&state->shared_up, r * D41_MOE_INTER * 2u },
		{ (void **)&state->shared_act, r * D41_MOE_INTER * 2u }, { (void **)&state->shared_out, r * D41_HIDDEN * 2u },
		{ (void **)&state->engram_ids, (D41_ENGRAM_DISK ? 0u : 1u) * r * D41_ENGRAM_COLUMNS * 8u },
		{ (void **)&state->engram_ids_next, (D41_ENGRAM_DISK ? 0u : 1u) * r * D41_ENGRAM_COLUMNS * 8u }, { (void **)&state->engram_embed, engram_rows * D41_HIDDEN * 2u },
		{ (void **)&state->engram_slot, sparse * 2u * r * D41_ENGRAM_COLUMNS * 4u }, { (void **)&state->engram_counts, sparse * 2u * D41_TP * D41_TP * 4u },
		{ (void **)&state->engram_send, sparse * D41_TP * own_max * D41_ENGRAM_COLUMNS * D41K_ENGRAM_RECORD_BYTES },
		{ (void **)&state->engram_received, sparse * D41_TP * own_max * D41_ENGRAM_COLUMNS * D41K_ENGRAM_RECORD_BYTES },
		{ (void **)&state->engram_kv, r * D41_ENGRAM_KV_ROWS * 2u }, { (void **)&state->engram_kv_local, r * D41_ENGRAM_KV_LOCAL * 2u },
		{ (void **)&state->engram_kv_gathered, r * D41_ENGRAM_KV_ROWS * 2u },
		{ (void **)&state->recurrent, s * state->recurrent_bytes },
	};
	return CudaStagePlace(plan, base);
}

static SparkStatus D41ReadFile(const char *path, uint64_t offset, void *target, uint64_t bytes)
{
	FILE *file = fopen(path, "rb");
	SparkStatus status = SPARK_STATUS_OK;
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( fseeko(file, (off_t)offset, SEEK_SET) != 0 || fread(target, 1u, (size_t)bytes, file) != bytes )
		status = SPARK_STATUS_IO_ERROR;
	fclose(file);
	return status;
}

static uint64_t D41FileBytes(const char *path)
{
	FILE *file = fopen(path, "rb");
	uint64_t bytes = 0u;
	if ( file == 0 )
		return 0u;
	if ( fseeko(file, 0, SEEK_END) == 0 )
		bytes = (uint64_t)ftello(file);
	fclose(file);
	return bytes;
}

static void D41Sibling(const char *pack_path, const char *name, char *out, size_t capacity)
{
	const char *slash = strrchr(pack_path, '/');
	const int prefix = slash != 0 ? (int)(slash - pack_path) : 1;
	snprintf(out, capacity, "%.*s/%s", prefix, slash != 0 ? pack_path : ".", name);
}

static uint32_t D41BackupRank(uint32_t copy, uint32_t owner, uint32_t slot)
{
	return copy == 0u ? (owner + 1u + slot % (D41_TP > 1u ? D41_TP - 1u : 1u)) % D41_TP :
		(owner + D41_TP / 2u + slot % (D41_TP > 1u ? D41_TP / 2u : 1u)) % D41_TP;
}

static uint32_t D41BackupPlaces(D41ModelState *state, const uint8_t *owner, const uint8_t *slot, uint8_t *backup, uint8_t *group)
{
	const uint64_t table = (uint64_t)D41_LAYERS * D41_EXPERTS;
	const uint32_t me = state->header.tp_rank;
	state->owner_host = (uint8_t *)malloc((2u + D41_BACKUP_COPIES) * table);
	state->expert_at_host = (uint16_t *)malloc((uint64_t)D41_LAYERS * D41_TP * D41_LOCAL_EXPERTS * sizeof(uint16_t));
	if ( state->owner_host == 0 || state->expert_at_host == 0 )
		return 0u;
	state->backup_host = state->owner_host + table;
	state->group_host = state->owner_host + (1u + D41_BACKUP_COPIES) * table;
	for ( uint32_t layer = 0u; layer < D41_LAYERS; ++layer )
	{
		const uint64_t base = (uint64_t)layer * D41_EXPERTS;
		uint16_t *at = state->expert_at_host + (uint64_t)layer * D41_TP * D41_LOCAL_EXPERTS;
		uint8_t *layer_backup = backup + (uint64_t)layer * D41_BACKUP_COPIES * D41_EXPERTS;
		uint32_t placed[D41_BACKUP_COPIES][D41_TP], held[D41_EXPERTS];
		memset(placed, 0, sizeof(placed));
		for ( uint32_t expert = 0u; expert < D41_EXPERTS; ++expert )
		{
			at[owner[base + expert] * D41_LOCAL_EXPERTS + slot[base + expert]] = (uint16_t)expert;
			held[expert] = D41_HELD_EXPERTS;
			for ( uint32_t copy = 0u; copy < D41_BACKUP_COPIES; ++copy )
				layer_backup[copy * D41_EXPERTS + expert] = layer < D41_BACKUP_LAYERS ? (uint8_t)D41BackupRank(copy, owner[base + expert], slot[base + expert]) :
					(uint8_t)UINT8_MAX;
		}
		for ( uint32_t copy = 0u; copy < D41_BACKUP_COPIES && layer < D41_BACKUP_LAYERS; ++copy )
			for ( uint32_t home = 0u; home < D41_TP; ++home )
				for ( uint32_t index = 0u; index < D41_LOCAL_EXPERTS; ++index )
				{
					const uint32_t expert = at[home * D41_LOCAL_EXPERTS + index], holder = layer_backup[copy * D41_EXPERTS + expert];
					if ( holder >= D41_TP || holder == home || placed[copy][holder] >= D41_LOCAL_EXPERTS ||
						(copy != 0u && holder == layer_backup[expert]) )
						return 0u;
					if ( holder == me )
						held[expert] = D41_LOCAL_EXPERTS + copy * D41_LOCAL_EXPERTS + placed[copy][holder];
					placed[copy][holder]++;
				}
		for ( uint32_t expert = 0u; expert < D41_EXPERTS; ++expert )
			group[base + expert] = owner[base + expert] == me ? slot[base + expert] : (uint8_t)held[expert];
	}
	memcpy(state->owner_host, owner, table);
	memcpy(state->backup_host, backup, D41_BACKUP_COPIES * table);
	memcpy(state->group_host, group, table);
	return 1u;
}

static SparkStatus D41OpenExpertPlacement(D41ModelState *state, const char *pack_path)
{
	const uint64_t table = (uint64_t)D41_LAYERS * D41_EXPERTS;
	uint16_t *ids = (uint16_t *)malloc(table * sizeof(uint16_t));
	uint8_t *maps = (uint8_t *)calloc((3u + D41_BACKUP_COPIES) * table, 1u);
	uint32_t head[5], placed = (state->header.flags & SPARK_DSV41_FLASH_STAGEPACK_FLAG_PLACED_EXPERTS) != 0u ? 1u : 0u, valid = 1u;
	char path[1024];
	SparkStatus status = SPARK_STATUS_OK;
	if ( ids == 0 || maps == 0 )
	{
		free(ids);
		free(maps);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	for ( uint64_t index = 0u; index < table; ++index )
		ids[index] = (uint16_t)(index % D41_EXPERTS);
	D41Sibling(pack_path, "expert_placement.bin", path, sizeof(path));
	if ( placed != 0u && (D41ReadFile(path, 0u, head, sizeof(head)) != SPARK_STATUS_OK || head[0] != SPARK_DSV41_FLASH_EXPERT_PLACEMENT_MAGIC ||
		head[1] != 1u || head[2] != D41_LAYERS || head[3] != D41_TP || head[4] != D41_LOCAL_EXPERTS ||
		D41ReadFile(path, sizeof(head), ids, table * sizeof(uint16_t)) != SPARK_STATUS_OK) )
		valid = 0u;
	for ( uint32_t layer = 0u; layer < D41_LAYERS && valid != 0u; ++layer )
		for ( uint32_t index = 0u; index < D41_EXPERTS; ++index )
		{
			const uint32_t expert = ids[(uint64_t)layer * D41_EXPERTS + index];
			uint8_t *owner = maps + (uint64_t)layer * D41_EXPERTS, *slot = maps + table + (uint64_t)layer * D41_EXPERTS;
			if ( expert >= D41_EXPERTS || owner[expert] != 0u )
			{
				valid = 0u;
				break;
			}
			owner[expert] = (uint8_t)(index / D41_LOCAL_EXPERTS + 1u);
			slot[expert] = (uint8_t)(index % D41_LOCAL_EXPERTS);
		}
	for ( uint64_t index = 0u; index < table; ++index )
		maps[index] = (uint8_t)(maps[index] - 1u);
	free(ids);
	if ( valid != 0u )
		valid = D41BackupPlaces(state, maps, maps + table, maps + 2u * table, maps + (2u + D41_BACKUP_COPIES) * table);
	if ( valid == 0u )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: %s does not place every expert of every layer on one of %u ranks, %u a rank (tools/dsv41_flash_expert_placement.py)\n",
			path, D41_TP, D41_LOCAL_EXPERTS);
		free(maps);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	if ( cudaMalloc(&state->expert_owner, (3u + D41_BACKUP_COPIES) * table) != cudaSuccess ||
		cudaMemcpy(state->expert_owner, maps, (3u + D41_BACKUP_COPIES) * table, cudaMemcpyHostToDevice) != cudaSuccess )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	state->expert_slot = state->expert_owner + table;
	state->expert_backup = state->expert_owner + 2u * table;
	state->expert_group = state->expert_owner + (2u + D41_BACKUP_COPIES) * table;
	free(maps);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return SPARK_STATUS_OK;
}

static SparkStatus D41OpenEngramTables(D41ModelState *state, const char *pack_path)
{
	char path[1024];
	uint32_t head[5];
	uint64_t rows[2];
	int64_t numbers[8 + 48 + 48];
	D41Sibling(pack_path, "engram_tables.bin", path, sizeof(path));
	if ( D41ReadFile(path, 0u, head, sizeof(head)) != SPARK_STATUS_OK || head[0] != D41_ENGRAM_TABLES_MAGIC || head[1] != 1u ||
		head[2] != D41_VOCAB || head[3] != D41_ENGRAM_COLUMNS ||
		D41ReadFile(path, sizeof(head), rows, sizeof(rows)) != SPARK_STATUS_OK ||
		D41ReadFile(path, sizeof(head) + sizeof(rows), numbers, sizeof(numbers)) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: %s is not the engram hash table file (tools/dsv41_flash_engram_tables.py)\n", path);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	state->engram_pad = (int32_t)head[4];
	int32_t *map = (int32_t *)malloc(D41_VOCAB * sizeof(int32_t));
	if ( map == 0 || D41ReadFile(path, sizeof(head) + sizeof(rows) + sizeof(numbers), map, D41_VOCAB * sizeof(int32_t)) != SPARK_STATUS_OK )
	{
		free(map);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	if ( cudaMalloc(&state->token_map, D41_VOCAB * sizeof(int32_t)) != cudaSuccess ||
		cudaMemcpy(state->token_map, map, D41_VOCAB * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess )
	{
		free(map);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	free(map);
	for ( uint32_t which = 0u; which < 2u; ++which )
	{
		D41Engram *engram = &state->engram[which];
		const uint64_t part = (rows[which] + D41_TP - 1u) / D41_TP;
		const uint64_t first = (uint64_t)state->tp_rank * part;
		engram->part = part;
		engram->table_rows = rows[which];
		engram->first_row = first;
		engram->local_rows = first >= rows[which] ? 0u : (rows[which] - first < part ? rows[which] - first : part);
		if ( cudaMalloc(&engram->multipliers, D41_ENGRAM_MULTIPLIERS * sizeof(int64_t)) != cudaSuccess ||
			cudaMalloc(&engram->primes, D41_ENGRAM_COLUMNS * sizeof(int64_t)) != cudaSuccess ||
			cudaMalloc(&engram->offsets, D41_ENGRAM_COLUMNS * sizeof(int64_t)) != cudaSuccess ||
			cudaMemcpy(engram->multipliers, numbers + which * D41_ENGRAM_MULTIPLIERS, D41_ENGRAM_MULTIPLIERS * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(engram->primes, numbers + 8u + which * D41_ENGRAM_COLUMNS, D41_ENGRAM_COLUMNS * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(engram->offsets, numbers + 56u + which * D41_ENGRAM_COLUMNS, D41_ENGRAM_COLUMNS * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus D41OpenEngramShard(D41ModelState *state, const char *pack_path)
{
	char sha_path[1100];
	FILE *file;
	D41PackHeader header;
	D41Sibling(pack_path, "engram.spengram", state->engram_path, sizeof(state->engram_path));
	snprintf(sha_path, sizeof(sha_path), "%s.digest", state->engram_path);
	state->engram_bytes = D41FileBytes(state->engram_path);
	file = fopen(sha_path, "rb");
	memset(state->engram_sha256, 0, sizeof(state->engram_sha256));
	if ( file != 0 )
	{
		if ( fread(state->engram_sha256, 1u, SPARK_SHA256_HEX_BYTES - 1u, file) != SPARK_SHA256_HEX_BYTES - 1u )
			state->engram_sha256[0] = 0;
		fclose(file);
	}
	memset(&header, 0, sizeof(header));
	if ( state->engram_bytes == 0u || strlen(state->engram_sha256) != 64u ||
		D41ReadFile(state->engram_path, 0u, &header, sizeof(header)) != SPARK_STATUS_OK || header.magic != 0x31474544u )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: the engram shard %s (with its .digest) is missing or not an engram shard\n", state->engram_path);
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	const int32_t check = SparkDsv41FlashEngramHeaderCheck(&header, &state->header, state->tp_rank, state->engram_bytes);
	if ( check != 0 || D41ReadFile(state->engram_path, header.directory_offset, state->engram_entries, sizeof(state->engram_entries)) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: the engram shard %s is not this rank's (%u of %u) shard of the rank pack's checkpoint "
			"(check %d; shard: format %u, rank %u of %u, %u tensors)\n", state->engram_path, state->tp_rank, D41_TP, check, header.format_version, header.tp_rank,
			header.tp_degree, header.tensor_count);
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	}
	uint32_t at = 0u;
	if ( SparkDsv41FlashEngramDirectoryCheck(state->engram_entries, state->engram_bytes, &at) != 0 )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: engram shard %s entry %u has the wrong kind, layer, codec, shape or bytes\n", state->engram_path, at);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return SPARK_STATUS_OK;
}

inline void D41Cuda::Release(D41ModelState *state)
{
	if ( state->engram_pack != 0 )
		SparkWeightdLazyPackDestroy(state->engram_pack);
	if ( state->engram_ahead.started != 0u )
	{
		pthread_mutex_lock(&state->engram_ahead.lock);
		state->engram_ahead.stop = 1u;
		pthread_cond_signal(&state->engram_ahead.wake);
		pthread_mutex_unlock(&state->engram_ahead.lock);
		pthread_join(state->engram_ahead.thread, 0);
		pthread_cond_destroy(&state->engram_ahead.done);
		pthread_cond_destroy(&state->engram_ahead.wake);
		pthread_mutex_destroy(&state->engram_ahead.lock);
	}
	for ( uint32_t which = 0u; which < 2u; ++which )
	{
		SparkRowFileClose(state->engram_file[which]);
		if ( state->engram_ahead_ids_host[which] != 0 )
			(void)cudaFreeHost(state->engram_ahead_ids_host[which]);
		if ( state->engram_ahead_rows_host[which] != 0 )
			(void)cudaFreeHost(state->engram_ahead_rows_host[which]);
	}
	if ( state->engram_ids_host != 0 )
		(void)cudaFreeHost(state->engram_ids_host);
	if ( state->engram_rows_host != 0 )
		(void)cudaFreeHost(state->engram_rows_host);
	if ( state->engram_resident != 0 )
		(void)cudaFree(state->engram_resident);
	free(state->engram_calls);
	if ( state->host_error != 0 )
		(void)cudaFreeHost(state->host_error);
	if ( state->exchange_counts_host != 0 )
		(void)cudaFreeHost(state->exchange_counts_host);
	if ( state->exchange_totals_host != 0 )
		(void)cudaFreeHost(state->exchange_totals_host);
	if ( state->counts_ready != 0 )
		(void)cudaEventDestroy(state->counts_ready);
	if ( state->backup_cells != 0 )
		(void)cudaFree(state->backup_cells);
	free(state->owner_host);
	free(state->expert_at_host);
	if ( state->shared_join != 0 )
		(void)cudaEventDestroy(state->shared_join);
	if ( state->shared_fork != 0 )
		(void)cudaEventDestroy(state->shared_fork);
	if ( state->shared_stream != 0 )
		(void)cudaStreamDestroy(state->shared_stream);
	if ( state->expert_owner != 0 )
		(void)cudaFree(state->expert_owner);
	if ( state->fp8_side != 0 )
		(void)cudaFree(state->fp8_side);
	if ( state->cells != 0 )
		(void)cudaFree(state->cells);
	(void)cudaFree(state->token_map);
	(void)cudaFree(state->full_frequency);
	for ( uint32_t which = 0u; which < 2u; ++which )
	{
		(void)cudaFree(state->engram[which].multipliers);
		(void)cudaFree(state->engram[which].primes);
		(void)cudaFree(state->engram[which].offsets);
	}
	if ( state->dump != 0 )
		fclose(state->dump);
	free(state->dump_streams);
	free(state->dump_routes);
	free(state->dump_weights);
	free(state->entries);
}

inline SparkStatus D41Cuda::Configure(D41ModelState *state, const SparkStageRunnerConfiguration *configuration, SparkStageRunnerModelGeometry *geometry)
{
	uint64_t pack_bytes;
	SparkStatus status;
	state->configuration = configuration;
	state->context_slots = configuration->max_active_sequence_count;
	state->max_positions = configuration->kv_pages_per_sequence * D41_PAGE_TOKENS;
#ifdef DEBUG
	if ( getenv("SPARK_DSV41_FLASH_LAYER_DUMP") != 0 )
	{
		state->dump = fopen(getenv("SPARK_DSV41_FLASH_LAYER_DUMP"), "wb");
		state->dump_streams = (uint16_t *)malloc((size_t)state->max_rows * D41K_HC * D41_HIDDEN * sizeof(uint16_t));
		state->dump_routes = (uint32_t *)malloc((size_t)state->max_rows * D41_TOPK * sizeof(uint32_t));
		state->dump_weights = (float *)malloc((size_t)state->max_rows * D41_TOPK * sizeof(float));
		if ( state->dump == 0 || state->dump_streams == 0 || state->dump_routes == 0 || state->dump_weights == 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		fprintf(stderr, "sparkpipe_dsv41_flash: this rank dumps every layer's streams and routes to %s (debug; graphs off)\n",
			getenv("SPARK_DSV41_FLASH_LAYER_DUMP"));
	}
#endif
	pack_bytes = D41FileBytes(configuration->rank_pack_path);
	status = D41ReadFile(configuration->rank_pack_path, 0u, &state->header, sizeof(state->header));
	if ( status != SPARK_STATUS_OK || state->header.magic != SPARK_DSV41_FLASH_STAGEPACK_MAGIC ||
		state->header.format_version != SPARK_DSV41_FLASH_STAGEPACK_FORMAT_VERSION || state->header.tp_degree != D41_TP ||
		state->header.tp_rank != configuration->tp_rank || state->header.hidden_dimension != D41_HIDDEN || state->header.layer_count != D41_LAYERS ||
		state->header.vocab_count != D41_VOCAB || state->header.file_bytes != pack_bytes || state->header.tensor_count == 0u ||
		state->header.header_bytes != sizeof(state->header) || state->header.directory_entry_bytes != sizeof(D41PackEntry) ||
		state->header.codec_abi_version != SPARK_WEIGHT_CODEC_ABI_VERSION || (state->header.flags & ~SPARK_DSV41_FLASH_STAGEPACK_KNOWN_FLAGS) != 0u ||
		state->header.stage_count != 1u || state->header.stage_index != 0u || state->header.first_layer_index != 0u ||
		state->header.total_layer_count != D41_LAYERS || state->header.routed_expert_count != D41_EXPERTS ||
		state->header.linear_weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 || state->header.expert_weight_codec != SPARK_WEIGHT_CODEC_MXFP4_E2M1 ||
		state->header.directory_offset < state->header.header_bytes || state->header.directory_offset > pack_bytes ||
		state->header.tensor_count > (pack_bytes - state->header.directory_offset) / sizeof(D41PackEntry) )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: %s is not this rank's DSV4.1 stage pack for %u ranks of format %u (expert slices)\n",
			configuration->rank_pack_path, D41_TP, SPARK_DSV41_FLASH_STAGEPACK_FORMAT_VERSION);
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	}
	state->moe_whole = (state->header.flags & SPARK_DSV41_FLASH_STAGEPACK_FLAG_WHOLE_EXPERTS) != 0u ? 1u : 0u;
	if ( state->moe_whole != 0u && !D41_WHOLE )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: %s gives this rank whole experts; a %u-rank build that shards the context runs expert slices\n",
			configuration->rank_pack_path, D41_TP);
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	}
	if ( SparkDsv41FlashStagePackIdentityCheck(&state->header, DSV41_FLASH_CONTRACT_SHA256) != 0 )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: %s was not packed from this build's contract (%s) or carries no model revision or source identity (check %d)\n",
			configuration->rank_pack_path, DSV41_FLASH_CONTRACT_SHA256, SparkDsv41FlashStagePackIdentityCheck(&state->header, DSV41_FLASH_CONTRACT_SHA256));
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	}
	state->entries = (D41PackEntry *)malloc((size_t)state->header.tensor_count * sizeof(D41PackEntry));
	if ( state->entries == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = D41ReadFile(configuration->rank_pack_path, state->header.directory_offset, state->entries,
		(uint64_t)state->header.tensor_count * sizeof(D41PackEntry));
	if ( status == SPARK_STATUS_OK )
		status = D41PackDirectoryCheck(state, configuration->rank_pack_path);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = D41OpenEngramTables(state, configuration->rank_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = D41OpenEngramShard(state, configuration->rank_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = D41OpenExpertPlacement(state, configuration->rank_pack_path);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( cudaMalloc(&state->full_frequency, 2u * D41K_ROPE_PAIRS * sizeof(float)) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->compress_frequency = state->full_frequency + D41K_ROPE_PAIRS;
	status = D41Frequencies(state->full_frequency);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41SharedSlice(state->tp_rank, &state->shared_first, &state->shared_width);
	state->shard.degree = D41_KV_TP;
	state->shard.rank = D41_KV_RANK(state);
	state->shard.grain = D41_GRAIN;
	state->local_score_stride = ((SparkKvShardGatherKeys(state->shard, state->max_positions) + 63u) / 64u) * 64u;
	state->select_rows = state->max_rows;
	if ( D41_WHOLE && (uint64_t)state->select_rows * state->local_score_stride * sizeof(float) > D41_SELECT_BYTES )
		state->select_rows = (uint32_t)(D41_SELECT_BYTES / ((uint64_t)state->local_score_stride * sizeof(float)));
	if ( state->select_rows == 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->global_score_stride = ((state->max_positions + 63u) / 64u) * 64u;
	state->candidate_blocks = state->max_positions > D41_CANDIDATE_ROWS ? 1u : 0u;
	if ( state->candidate_blocks != 0u )
	{
		state->local_block_stride = ((state->local_score_stride / D41_GRAIN + 63u) / 64u) * 64u;
		state->global_block_stride = (((state->max_positions + D41_GRAIN - 1u) / D41_GRAIN + 63u) / 64u) * 64u;
		state->local_block_keep = state->local_block_stride < D41_CANDIDATE_BLOCKS ? state->local_block_stride : D41_CANDIDATE_BLOCKS;
		state->global_block_keep = state->global_block_stride < D41_CANDIDATE_BLOCKS ? state->global_block_stride : D41_CANDIDATE_BLOCKS;
		state->block_round_parts = 1u;
		while ( state->local_block_keep * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)) / state->block_round_parts > D41_HIDDEN )
			state->block_round_parts *= 2u;
	}
	{
		const uint64_t local = (uint64_t)state->select_rows * LmTopkExactCandidateEntries(state->local_score_stride, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK);
		const uint64_t local_blocks = state->candidate_blocks != 0u ?
			(uint64_t)state->select_rows * LmTopkExactCandidateEntries(state->local_block_stride, state->local_block_keep, LM_TOPK_EXACT_CHUNK) : 0u;
		state->topk_entries = 2u * (local > local_blocks ? local : local_blocks);
	}
	state->ring_bytes = (uint64_t)D41_LAYERS * D41_RING_ROWS * D41_HEAD_DIM * sizeof(uint16_t);
	state->compressor_bytes = 3u * 2u * D41_HEAD_DIM * sizeof(float);
	state->history_bytes = 4u * sizeof(int32_t);
	state->recurrent_bytes = (state->ring_bytes + state->compressor_bytes + state->history_bytes + 255u) & ~255ull;
	state->arena_bytes = Plan(state, 0, 0u);
	if ( state->arena_bytes > configuration->state_budget_bytes )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: rank state %llu bytes is over the %llu-byte budget (rows %u, slots %u, positions %u)\n",
			(unsigned long long)state->arena_bytes, (unsigned long long)configuration->state_budget_bytes, state->max_rows, state->slots,
			state->max_positions);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	{
		const D41PackEntry *embed = D41Find(state, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EMBEDDING, SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER);
		const D41PackEntry *norm = D41Find(state, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FINAL_NORM, SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER);
		const D41PackEntry *head = D41Find(state, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD, SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER);
		if ( embed == 0 || norm == 0 || head == 0 )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		geometry->hidden = D41_HIDDEN;
		geometry->vocab = D41_VOCAB;
		geometry->total_layers = D41_LAYERS;
		geometry->layer_count = D41_LAYERS;
		geometry->rms_epsilon = D41_EPS;
		geometry->pack_bytes = pack_bytes;
		geometry->embed_offset = embed->payload_offset;
		geometry->embed_bytes = embed->payload_bytes;
		geometry->embed_rows = embed->rows;
		geometry->embed_gather = D41_ROWS ? 1u : 0u;
		geometry->graph_ignores_gather = 1u;
		geometry->head_norm_offset = norm->payload_offset;
		geometry->head_norm_bytes = norm->payload_bytes;
		geometry->head_offset = head->payload_offset;
		geometry->head_bytes = head->payload_bytes;
		geometry->head_rows = head->rows;
		Kv::Geometry(geometry);
		geometry->kv_shard = state->shard;
		geometry->recurrent_bytes = state->recurrent_bytes;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus D41DenseManifest(const struct SparkWeightdManifest *manifest, void *context)
{
	(void)context;
	if ( manifest == 0 || manifest->range_count != 0u )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: the stage runner attaches the DSV4.1 packs whole (a zero-range .experts manifest)\n");
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus D41SliceEntry(SparkWeightdLazyPack *pack, const D41PackEntry *entry, uint32_t scale, const void **out)
{
	const uint64_t offset = scale != 0u ? entry->scale_offset : entry->payload_offset;
	const uint64_t bytes = scale != 0u ? entry->scale_bytes : entry->payload_bytes;
	if ( bytes == 0u || SparkWeightdLazyPackSlice(pack, offset, bytes, out) != SPARK_STATUS_OK || *out == 0 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus D41Fp8SliceAt(D41ModelState *state, const D41PackEntry *entry, const void *payload, const void *scale, uint32_t row_first,
	uint32_t rows, uint32_t column_first, uint32_t columns, uint32_t widen, uint64_t *cursor, D41Fp8Linear *out)
{
	const uint32_t whole_rows = column_first == 0u && entry != 0 && columns == entry->columns ? 1u : 0u;
	if ( entry == 0 || entry->weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 || entry->rows % D41_FP8_BLOCK != 0u ||
		entry->columns % D41_FP8_BLOCK != 0u || rows == 0u || columns == 0u || row_first % D41_FP8_BLOCK != 0u ||
		column_first % D41_FP8_BLOCK != 0u || columns % D41_FP8_BLOCK != 0u || row_first > entry->rows || rows > entry->rows - row_first ||
		column_first > entry->columns || columns > entry->columns - column_first ||
		(state->fp8_side != 0 && (payload == 0 || scale == 0 || ((uintptr_t)payload % 16u) != 0u)) )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	const uint64_t scale_bytes = (((uint64_t)rows * (columns / D41_FP8_BLOCK)) + 255u) & ~255ull;
	const uint64_t weight_bytes = whole_rows != 0u ? 0u : (((uint64_t)rows * columns) + 255u) & ~255ull;
	const uint64_t wide_bytes = widen != 0u ? (((uint64_t)rows * columns * sizeof(uint16_t)) + 255u) & ~255ull : 0u;
	if ( state->fp8_side != 0 )
	{
		uint8_t *row_scale = state->fp8_side + *cursor;
		uint8_t *weight = whole_rows != 0u ? (uint8_t *)0 : row_scale + scale_bytes;
		uint16_t *wide = widen != 0u ? (uint16_t *)(row_scale + scale_bytes + weight_bytes) : (uint16_t *)0;
		D41Fp8SliceKernel<<<rows, D41_THREADS>>>((const uint8_t *)payload, (const uint8_t *)scale, weight, row_scale, wide, entry->columns,
			row_first, column_first, columns);
		out->scale = row_scale;
		out->weight = whole_rows != 0u ? (const uint8_t *)payload + (uint64_t)row_first * entry->columns : weight;
		out->wide = wide;
	}
	*cursor += scale_bytes + weight_bytes + wide_bytes;
	return SPARK_STATUS_OK;
}

static SparkStatus D41Fp8Slice(D41ModelState *state, SparkWeightdLazyPack *pack, const D41PackEntry *entry, uint32_t row_first,
	uint32_t rows, uint32_t column_first, uint32_t columns, uint64_t *cursor, D41Fp8Linear *out)
{
	const void *payload = 0, *scale = 0;
	if ( entry == 0 || D41SliceEntry(pack, entry, 0u, &payload) != SPARK_STATUS_OK || D41SliceEntry(pack, entry, 1u, &scale) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return D41Fp8SliceAt(state, entry, payload, scale, row_first, rows, column_first, columns, 1u, cursor, out);
}

static SparkStatus D41Fp8(D41ModelState *state, SparkWeightdLazyPack *pack, const D41PackEntry *entry, uint64_t *cursor, D41Fp8Linear *out)
{
	if ( entry == 0 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return D41Fp8Slice(state, pack, entry, 0u, entry->rows, 0u, entry->columns, cursor, out);
}

#define D41_ENTRY(kind, layer) D41Find(state, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_##kind, layer)
#define D41_PAYLOAD(kind, layer, field) \
	do { const D41PackEntry *found = D41_ENTRY(kind, layer); const void *bound = 0; \
		if ( found == 0 || D41SliceEntry(lazy_pack, found, 0u, &bound) != SPARK_STATUS_OK ) { \
			fprintf(stderr, "sparkpipe_dsv41_flash: layer %u has no " #kind "\n", layer); return SPARK_STATUS_PARSE_ERROR; } \
		*(const void **)&(field) = bound; } while ( 0 )
#define D41_SCALE(kind, layer, field) \
	do { const D41PackEntry *found = D41_ENTRY(kind, layer); const void *bound = 0; \
		if ( found == 0 || D41SliceEntry(lazy_pack, found, 1u, &bound) != SPARK_STATUS_OK ) return SPARK_STATUS_PARSE_ERROR; \
		*(const void **)&(field) = bound; } while ( 0 )
#define D41_FP8(kind, layer, field) \
	do { if ( D41Fp8(state, lazy_pack, D41_ENTRY(kind, layer), &cursor, &(field)) != SPARK_STATUS_OK ) { \
			fprintf(stderr, "sparkpipe_dsv41_flash: layer %u " #kind " is not a 32x32-block fp8 tensor\n", layer); return SPARK_STATUS_PARSE_ERROR; } } while ( 0 )
#define D41_FP8_SLICE(kind, layer, field, first_row, row_count, first_column, column_count) \
	do { const D41PackEntry *found = D41_ENTRY(kind, layer); \
		if ( found == 0 || D41Fp8Slice(state, lazy_pack, found, first_row, row_count == 0u ? found->rows : row_count, first_column, \
			column_count == 0u ? found->columns : column_count, &cursor, &(field)) != SPARK_STATUS_OK ) { \
			fprintf(stderr, "sparkpipe_dsv41_flash: layer %u " #kind " has no 32x32-block fp8 slice for this rank\n", layer); \
			return SPARK_STATUS_PARSE_ERROR; } } while ( 0 )

static SparkStatus D41BindLayers(D41ModelState *state, SparkWeightdLazyPack *lazy_pack, uint64_t *side_bytes)
{
	uint64_t cursor = 0u;
	for ( uint32_t index = 0u; index < D41_LAYERS; ++index )
	{
		D41Layer *layer = &state->layers[index];
		D41_PAYLOAD(ATTN_NORM, index, layer->attn_norm);
		D41_PAYLOAD(FFN_NORM, index, layer->ffn_norm);
		D41_PAYLOAD(Q_NORM, index, layer->q_norm);
		D41_PAYLOAD(KV_NORM, index, layer->kv_norm);
		D41_PAYLOAD(ATTN_SINK, index, layer->sink);
		if ( D41_ENTRY(ATTN_SINK, index)->columns != D41_HEADS )
		{
			fprintf(stderr, "sparkpipe_dsv41_flash: layer %u carries %u attention sinks, not all %u\n", index, D41_ENTRY(ATTN_SINK, index)->columns,
				D41_HEADS);
			return SPARK_STATUS_SCHEMA_ERROR;
		}
		layer->group_sink = layer->sink + (uint64_t)(D41_ATTENTION_RANK(state) / 2u) * 2u * D41_LOCAL_HEADS;
		layer->sink += (uint64_t)D41_ATTENTION_RANK(state) * D41_LOCAL_HEADS;
		D41_FP8(Q_A, index, layer->q_a);
		D41_FP8(Q_B, index, layer->q_b);
		D41_FP8(KV_A, index, layer->kv_a);
		for ( uint32_t group = 0u; group < D41_LOCAL_GROUPS; ++group )
			D41_FP8_SLICE(O_A, index, layer->o_a[group], group * D41_GROUP_ROWS, D41_GROUP_ROWS, 0u, 0u);
		D41_FP8(O_B, index, layer->o_b);
		D41_PAYLOAD(HC_ATTN_FN, index, layer->hc_attn_fn);
		D41_PAYLOAD(HC_ATTN_BASE, index, layer->hc_attn_base);
		D41_PAYLOAD(HC_ATTN_SCALE, index, layer->hc_attn_scale);
		D41_PAYLOAD(HC_FFN_FN, index, layer->hc_ffn_fn);
		D41_PAYLOAD(HC_FFN_BASE, index, layer->hc_ffn_base);
		D41_PAYLOAD(HC_FFN_SCALE, index, layer->hc_ffn_scale);
		D41_PAYLOAD(ROUTER, index, layer->router);
		D41_PAYLOAD(ROUTER_BIAS, index, layer->router_bias);
		D41_PAYLOAD(EXPERT_W13_SLICE, index, layer->e_w13);
		D41_SCALE(EXPERT_W13_SLICE, index, layer->e_w13_scale);
		D41_PAYLOAD(EXPERT_W2_SLICE, index, layer->e_w2);
		D41_SCALE(EXPERT_W2_SLICE, index, layer->e_w2_scale);
		D41_FP8_SLICE(SHARED_W1, index, layer->sh_w1, state->shared_first, state->shared_width, 0u, 0u);
		if ( D41_SHARED_ROUTED != 0u && state->moe_whole == 0u )
			for ( uint32_t source = 0u; source < (D41_SHARED_ROUTED != 0u ? D41_TP : 0u); ++source )
			{
				uint32_t first, width;
				D41SharedSlice(source, &first, &width);
				D41_FP8_SLICE(SHARED_W2, index, layer->sh_w2_part[source], state->tp_rank * D41_HIDDEN_SLICE, D41_HIDDEN_SLICE, first, width);
			}
		else
			D41_FP8_SLICE(SHARED_W2, index, layer->sh_w2, 0u, 0u, state->shared_first, state->shared_width);
		D41_FP8_SLICE(SHARED_W3, index, layer->sh_w3, state->shared_first, state->shared_width, 0u, 0u);
		if ( D41KvSource(index) )
		{
			D41_PAYLOAD(COMPRESSOR_WKV, index, layer->c_wkv);
			D41_PAYLOAD(COMPRESSOR_NORM, index, layer->c_norm);
			D41_PAYLOAD(INDEXER_WK, index, layer->idx_wk);
			D41_PAYLOAD(INDEXER_K_NORM, index, layer->idx_kn);
			if ( D41Ratio(index) > 1u )
				D41_PAYLOAD(COMPRESSOR_WGATE, index, layer->c_wgate);
		}
		if ( D41Ratio(index) > 0u && D41IndexSource(index) )
		{
			D41_FP8(INDEXER_Q_B, index, layer->idx_q_b);
			D41_PAYLOAD(INDEXER_WEIGHTS_PROJ, index, layer->idx_wp);
		}
	}
	*side_bytes = cursor;
	return SPARK_STATUS_OK;
}

static SparkStatus D41EngramPlane(D41ModelState *state, const D41PackEntry *entry, uint32_t scale, uint64_t *resident, const void **out)
{
	const uint64_t offset = scale != 0u ? entry->scale_offset : entry->payload_offset;
	const uint64_t bytes = scale != 0u ? entry->scale_bytes : entry->payload_bytes;
	const uint64_t chunk = 32ull << 20;
	if ( !D41_ENGRAM_DISK )
		return D41SliceEntry(state->engram_pack, entry, scale, out);
	*out = 0;
	if ( bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	if ( state->engram_resident != 0 )
	{
		uint8_t *bounce = (uint8_t *)malloc((size_t)(bytes < chunk ? bytes : chunk));
		uint8_t *target = state->engram_resident + *resident;
		SparkStatus status = bounce != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED;
		for ( uint64_t done = 0u; done < bytes && status == SPARK_STATUS_OK; done += chunk )
		{
			const uint64_t piece = bytes - done < chunk ? bytes - done : chunk;
			status = D41ReadFile(state->engram_path, offset + done, bounce, piece);
			if ( status == SPARK_STATUS_OK && cudaMemcpy(target + done, bounce, (size_t)piece, cudaMemcpyHostToDevice) != cudaSuccess )
				status = SPARK_STATUS_IO_ERROR;
		}
		free(bounce);
		if ( status != SPARK_STATUS_OK )
			SPARK_FAIL(status);
		*out = target;
	}
	*resident += (bytes + 255u) & ~255ull;
	return SPARK_STATUS_OK;
}

static SparkStatus D41BindEngram(D41ModelState *state, uint64_t *cursor, uint64_t *resident, const SparkStageRunnerConfiguration *configuration)
{
	for ( uint32_t which = 0u; which < 2u; ++which )
	{
		const D41PackEntry *embed = &state->engram_entries[which * 4u], *wkv = &state->engram_entries[which * 4u + 1u];
		const D41PackEntry *q = &state->engram_entries[which * 4u + 2u], *k = &state->engram_entries[which * 4u + 3u];
		D41Engram *engram = &state->engram[which];
		const void *bound = 0, *wkv_payload = 0, *wkv_scale = 0;
		if ( embed->rows != engram->local_rows || embed->columns != D41_ENGRAM_HEAD_DIM || wkv->rows != D41_ENGRAM_KV_ROWS ||
			wkv->columns != D41_ENGRAM_EMBED || q->rows != D41K_HC || q->columns != D41_HIDDEN )
		{
			fprintf(stderr, "sparkpipe_dsv41_flash: engram shard layer %u holds %u rows of %u, expected %llu of %u\n", which == 0u ? 1u : 14u,
				embed->rows, embed->columns, (unsigned long long)engram->local_rows, D41_ENGRAM_HEAD_DIM);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( !D41_ENGRAM_DISK )
		{
			if ( D41SliceEntry(state->engram_pack, embed, 0u, &bound) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			engram->payload = (const uint8_t *)bound;
		}
		else if ( embed->payload_bytes != (uint64_t)embed->rows * D41_ENGRAM_HEAD_DIM )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( D41EngramPlane(state, embed, 1u, resident, &bound) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		engram->scale = (const uint8_t *)bound;
		if ( D41EngramPlane(state, q, 0u, resident, &bound) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		engram->q_weight = (const uint16_t *)bound;
		if ( D41EngramPlane(state, k, 0u, resident, &bound) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		engram->k_weight = (const uint16_t *)bound;
		if ( D41EngramPlane(state, wkv, 0u, resident, &wkv_payload) != SPARK_STATUS_OK ||
			D41EngramPlane(state, wkv, 1u, resident, &wkv_scale) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( D41Fp8SliceAt(state, wkv, wkv_payload, wkv_scale, state->tp_rank * D41_ENGRAM_KV_LOCAL, D41_ENGRAM_KV_LOCAL, 0u, D41_ENGRAM_EMBED, 1u,
			cursor, &engram->wkv) != SPARK_STATUS_OK ||
			(D41_WHOLE && D41Fp8SliceAt(state, wkv, wkv_payload, wkv_scale, 0u, D41_ENGRAM_KV_ROWS, 0u, D41_ENGRAM_EMBED, 0u, cursor,
			&engram->wkv_whole) != SPARK_STATUS_OK) )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	(void)configuration;
	return SPARK_STATUS_OK;
}

static SparkStatus D41OpenEngramRows(D41ModelState *state)
{
	for ( uint32_t which = 0u; which < 2u; ++which )
	{
		const D41PackEntry *embed = &state->engram_entries[which * 4u];
		const SparkRowFileOptions options = { embed->payload_offset, embed->rows, D41_ENGRAM_HEAD_DIM, D41_ENGRAM_CACHE_ROWS, D41_ENGRAM_READERS, 0u };
		const SparkStatus status = SparkRowFileOpen(state->engram_path, &options, &state->engram_file[which]);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr, "sparkpipe_dsv41_flash: the engram rows of layer %u in %s could not be opened for direct reads status=%d\n",
				which == 0u ? 1u : 14u, state->engram_path, (int)status);
			return status;
		}
	}
	return SPARK_STATUS_OK;
}

static SparkStatus D41EngramBuffers(D41ModelState *state)
{
	const uint64_t ids = (uint64_t)state->max_rows * D41_ENGRAM_COLUMNS * sizeof(int64_t);
	const uint64_t rows = (uint64_t)state->max_rows * D41_ENGRAM_COLUMNS * D41_ENGRAM_HEAD_DIM;
	state->engram_calls = (D41EngramCall *)calloc(2u * ((size_t)state->max_rows + 1u), sizeof(D41EngramCall));
	if ( state->engram_calls == 0 ||
		cudaHostAlloc((void **)&state->engram_ids_host, ids, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
		cudaHostAlloc((void **)&state->engram_rows_host, rows, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(state->engram_ids_host, 0, ids);
	memset(state->engram_rows_host, 0, rows);
	if ( cudaHostGetDevicePointer((void **)&state->engram_ids, state->engram_ids_host, 0u) != cudaSuccess ||
		cudaHostGetDevicePointer((void **)&state->engram_rows, state->engram_rows_host, 0u) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( state->max_rows <= SPARK_STAGE_RUNNER_GRAPH_ROWS )
		return SPARK_STATUS_OK;
	for ( uint32_t which = 0u; which < 2u; ++which )
	{
		if ( cudaHostAlloc((void **)&state->engram_ahead_ids_host[which], ids, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
			cudaHostAlloc((void **)&state->engram_ahead_rows_host[which], rows, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		memset(state->engram_ahead_ids_host[which], 0, ids);
		memset(state->engram_ahead_rows_host[which], 0, rows);
		if ( cudaHostGetDevicePointer((void **)&state->engram_ahead_ids[which], state->engram_ahead_ids_host[which], 0u) != cudaSuccess ||
			cudaHostGetDevicePointer((void **)&state->engram_ahead_rows[which], state->engram_ahead_rows_host[which], 0u) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	pthread_mutex_init(&state->engram_ahead.lock, 0);
	pthread_cond_init(&state->engram_ahead.wake, 0);
	pthread_cond_init(&state->engram_ahead.done, 0);
	if ( pthread_create(&state->engram_ahead.thread, 0, D41EngramAheadWork, state) != 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->engram_ahead.started = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus D41AttachEngram(D41ModelState *state, const SparkStageRunnerConfiguration *configuration)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	memset(&request, 0, sizeof(request));
	memset(&options, 0, sizeof(options));
	memcpy(request.identity.pack_sha256, state->engram_sha256, SPARK_SHA256_HEX_BYTES);
	snprintf(request.identity.model, sizeof(request.identity.model), "%s", "deepseek-v4.1-flash-engram");
	snprintf(request.identity.revision, sizeof(request.identity.revision), "%s", "official");
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = state->engram_bytes;
	request.identity.topology = D41_TP;
	if ( strlen(state->engram_path) >= sizeof(request.pack_path) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(request.pack_path, state->engram_path, strlen(state->engram_path) + 1u);
	options.timeout_nanoseconds = SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS;
	options.manifest_check = D41DenseManifest;
	options.manifest_check_context = state;
	return SparkWeightdLazyPackCreate(configuration->weights.socket_path, &request, &options, &state->engram_pack);
}

static SparkStatus D41PackCells(D41ModelState *state, SparkWeightdLazyPack *lazy_pack)
{
	const uint32_t kinds[2] = {SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W13_SLICE, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W2_SLICE};
	const uint32_t whole = state->moe_whole, groups = whole != 0u ? D41_LOCAL_EXPERTS : D41_EXPERTS;
	const uint32_t rows[2] = {2u * (whole != 0u ? D41_MOE_INTER : D41_EXPERT_SLICE), whole != 0u ? D41_HIDDEN : D41_HIDDEN_SLICE};
	const uint32_t columns[2] = {D41_HIDDEN, D41_MOE_INTER};
	const uint64_t bytes[2] = {LmCellPackBytes(groups, rows[0], columns[0]), LmCellPackBytes(groups, rows[1], columns[1])};
	uint8_t *place[D41_LAYERS][2];
	uint64_t separate = 0u, in_place = 0u;
	uint8_t *scratch = 0;
	for ( uint32_t index = 0u; index < D41_LAYERS; ++index )
		for ( uint32_t m = 0u; m < 2u; ++m )
		{
			const D41PackEntry *entry = D41Find(state, kinds[m], index);
			void *region = 0;
			place[index][m] = 0;
			if ( entry == 0 || entry->group_count != groups || entry->rows != rows[m] || entry->columns != columns[m] / 2u ||
				entry->weight_codec != SPARK_WEIGHT_CODEC_MXFP4_E2M1 )
			{
				fprintf(stderr, "sparkpipe_dsv41_flash: layer %u expert tensor %u is not %u experts x %u rows x %u MXFP4 columns\n", index, m,
					groups, rows[m], columns[m]);
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			}
			if ( entry->scale_offset == entry->payload_offset + entry->payload_bytes && entry->payload_bytes + entry->scale_bytes == bytes[m] &&
				SparkWeightdLazyPackPrivateSlice(lazy_pack, entry->payload_offset, bytes[m], &region) == SPARK_STATUS_OK )
			{
				place[index][m] = (uint8_t *)region;
				in_place += bytes[m];
			}
			else
				separate += bytes[m];
		}
	state->cells_bytes = separate;
	if ( (separate != 0u && cudaMalloc((void **)&state->cells, separate) != cudaSuccess) ||
		(in_place != 0u && cudaMalloc((void **)&scratch, bytes[0] > bytes[1] ? bytes[0] : bytes[1]) != cudaSuccess) )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: %llu bytes for the expert cells could not be allocated\n", (unsigned long long)separate);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	separate = 0u;
	for ( uint32_t index = 0u; index < D41_LAYERS; ++index )
	{
		D41Layer *layer = &state->layers[index];
		const uint8_t *payload[2] = {layer->e_w13, layer->e_w2}, *scale[2] = {layer->e_w13_scale, layer->e_w2_scale};
		const uint8_t **cells[2] = {&layer->e_w13_cells, &layer->e_w2_cells};
		for ( uint32_t m = 0u; m < 2u; ++m )
		{
			uint8_t *target = place[index][m];
			if ( target != 0 )
			{
				const uint64_t payload_bytes = (uint64_t)groups * rows[m] * columns[m] / 2u;
				if ( cudaMemcpy(scratch, target, bytes[m], cudaMemcpyDeviceToDevice) != cudaSuccess ||
					LmCellPackMxfp4(scratch, scratch + payload_bytes, target, groups, rows[m], columns[m], 0) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				target = state->cells + separate;
				separate += bytes[m];
				if ( LmCellPackMxfp4(payload[m], scale[m], target, groups, rows[m], columns[m], 0) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			*cells[m] = target;
		}
		layer->e_w13 = layer->e_w13_scale = layer->e_w2 = layer->e_w2_scale = 0;
	}
	if ( D41_BACKUP_LAYERS != 0u && whole != 0u )
	{
		const uint64_t g13 = bytes[0] / groups, g2 = bytes[1] / groups, layer_bytes = (uint64_t)D41_BACKUP_COPIES * D41_LOCAL_EXPERTS * (g13 + g2);
		if ( cudaMalloc((void **)&state->backup_cells, (uint64_t)D41_BACKUP_LAYERS * layer_bytes) != cudaSuccess )
		{
			state->backup_cells = 0;
			fprintf(stderr, "sparkpipe_dsv41_flash: %llu bytes for backup expert copies could not be allocated\n",
				(unsigned long long)((uint64_t)D41_BACKUP_LAYERS * layer_bytes));
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		for ( uint32_t index = 0u; index < D41_BACKUP_LAYERS; ++index )
		{
			state->layers[index].e_w13_backup = state->backup_cells + index * layer_bytes;
			state->layers[index].e_w2_backup = state->backup_cells + index * layer_bytes + D41_BACKUP_COPIES * D41_LOCAL_EXPERTS * g13;
		}
		state->backup_group13 = g13;
		state->backup_group2 = g2;
		state->backup_filled = 0u;
	}
	if ( cudaDeviceSynchronize() != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( scratch != 0 )
		(void)cudaFree(scratch);
	fprintf(stderr, "DSV41-CELLS rank=%u experts=%s in_place_bytes=%llu separate_bytes=%llu\n", state->tp_rank, whole != 0u ? "whole" : "slices",
		(unsigned long long)in_place, (unsigned long long)state->cells_bytes);
	return SPARK_STATUS_OK;
}

inline SparkStatus D41Cuda::Bind(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
{
	D41ModelState *state = (D41ModelState *)model;
	uint64_t bytes = 0u, resident = 0u;
	SparkStatus status;
	(void)geometry;
	status = D41_ENGRAM_DISK ? SPARK_STATUS_OK : D41AttachEngram(state, state->configuration);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: weightd refused the engram shard %s status=%d\n", state->engram_path, (int)status);
		return status;
	}
	status = D41BindLayers(state, lazy_pack, &bytes);
	if ( status == SPARK_STATUS_OK )
		status = D41BindEngram(state, &bytes, &resident, state->configuration);
	if ( status != SPARK_STATUS_OK )
		return status;
	state->fp8_side_bytes = bytes;
	state->engram_resident_bytes = resident;
	if ( resident != 0u && cudaMalloc((void **)&state->engram_resident, resident) != cudaSuccess )
	{
		state->engram_resident = 0;
		fprintf(stderr, "sparkpipe_dsv41_flash: %llu bytes for the engram scales and projections could not be allocated\n", (unsigned long long)resident);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( cudaMalloc((void **)&state->fp8_side, bytes) != cudaSuccess )
	{
		state->fp8_side = 0;
		fprintf(stderr, "sparkpipe_dsv41_flash: %llu bytes for the fp8 row scales, slices and bf16 widenings could not be allocated\n", (unsigned long long)bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	bytes = 0u;
	resident = 0u;
	status = D41BindLayers(state, lazy_pack, &bytes);
	if ( status == SPARK_STATUS_OK )
		status = D41BindEngram(state, &bytes, &resident, state->configuration);
	if ( status == SPARK_STATUS_OK && D41_ENGRAM_DISK )
		status = D41OpenEngramRows(state);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = D41PackCells(state, lazy_pack);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( cudaHostAlloc((void **)&state->host_error, sizeof(LmKvAccessError), cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess )
	{
		state->host_error = 0;
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	memset(state->host_error, 0, sizeof(LmKvAccessError));
	if ( D41_ROWS && cudaHostAlloc((void **)&state->exchange_counts_host, 3u * D41_TP * D41_TP * sizeof(uint32_t), cudaHostAllocPortable) != cudaSuccess )
	{
		state->exchange_counts_host = 0;
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( D41_ROWS && cudaHostAlloc((void **)&state->exchange_totals_host, sizeof(D41Totals) + (uint64_t)D41_BACKUP_COPIES * D41_TP * D41_EXPERTS * sizeof(uint16_t),
		cudaHostAllocPortable) != cudaSuccess )
	{
		state->exchange_totals_host = 0;
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	state->exchange_quota_host = D41_ROWS ? (uint16_t *)(state->exchange_totals_host + 1) : 0;
	if ( D41_ROWS && cudaEventCreateWithFlags(&state->counts_ready, cudaEventDisableTiming) != cudaSuccess )
	{
		state->counts_ready = 0;
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( D41_ROWS && (cudaStreamCreateWithFlags(&state->shared_stream, cudaStreamNonBlocking) != cudaSuccess ||
		cudaEventCreateWithFlags(&state->shared_fork, cudaEventDisableTiming) != cudaSuccess ||
		cudaEventCreateWithFlags(&state->shared_join, cudaEventDisableTiming) != cudaSuccess) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cudaHostGetDevicePointer((void **)&state->kv_error, state->host_error, 0u) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( cudaDeviceSynchronize() != cudaSuccess )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: the %llu-byte rank state could not be allocated\n", (unsigned long long)state->arena_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	status = CudaStage<D41Cuda>::Allocate(state);
	if ( status == SPARK_STATUS_OK && D41_ENGRAM_DISK )
		status = D41EngramBuffers(state);
	return status;
}

inline SparkStatus D41Cuda::Upload(D41ModelState *state)
{
	float *init = (float *)calloc((size_t)state->max_rows * D41K_HC, sizeof(float));
	if ( init == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for ( uint32_t row = 0u; row < state->max_rows; ++row )
		init[(uint64_t)row * D41K_HC] = 1.0f;
	const cudaError_t error = cudaMemcpy(state->pre_init, init, (size_t)state->max_rows * D41K_HC * sizeof(float), cudaMemcpyHostToDevice);
	free(init);
	if ( error != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

#undef D41_FP8_SLICE
#undef D41_FP8
#undef D41_SCALE
#undef D41_PAYLOAD
#undef D41_ENTRY

inline uint32_t D41Cuda::GraphKey(void *model, const SparkStageRunnerStep *step)
{
	(void)model;
	return step->commit == 0u && step->last_rows_only == 0u && step->sequences == 1u && step->rows > 1u && step->rows <= D41_VERIFY_ROWS ? 2u : 1u;
}

inline SparkStatus D41Cuda::TapPlan(void *model, const SparkStageRunnerTapPlan *plan)
{
	D41ModelState *state = (D41ModelState *)model;
	int16_t outputs[D41_LAYERS];
	uint32_t index, reach = UINT32_MAX;
	if ( state == 0 || plan == 0 || plan->first_row == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( index = 0u; index < D41_LAYERS; ++index )
		outputs[index] = -1;
	for ( index = 0u; index < plan->entry_count; ++index )
	{
		const uint32_t layer = plan->entries[index].layer;
		if ( plan->entries[index].form == SPARK_STAGE_RUNNER_TAP_FORM_FINAL )
			continue;
		if ( plan->entries[index].form != SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT || layer >= D41_LAYERS )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		outputs[layer] = (int16_t)index;
		if ( layer >= D41_CANDIDATE_SOURCE && D41TailReach(layer + 1u) < reach )
			reach = D41TailReach(layer + 1u);
	}
	memcpy(state->tap_output, outputs, sizeof(outputs));
	state->tap_slab = plan->capture != 0u ? plan->slab : 0;
	state->tap_stride = plan->row_stride_elements;
	state->tap_first_row = plan->first_row;
	state->tap_reach = reach;
	fprintf(stderr, "sparkpipe_dsv41_flash: taps entries=%u capture=%u; layer outputs are the mean of the %u streams, valid on the last %u positions of a tail wave\n",
		plan->entry_count, plan->capture, D41K_HC, reach);
	return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_CUDA_H
#define SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_CUDA_H

#include <cuda_runtime.h>
#include <stdint.h>

#include "sparkpipe/spark_k3_bind.h"
#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_k3_pool_sizing.h"
#include "inference/llms/kimi_k3/slice.cuh"

/*
 * CUDA dispatch for the K3 resident decode stage: the serving tier's bridge
 * between the CUDA-free pack/bind/pool modules and bind.cu's K3StageSlice ABI.
 *
 * Layout of ownership, top to bottom:
 *   spark_k3_pack_load   mmap + manifest name resolution (CUDA-free)
 *   spark_k3_bind        per-layer name tables -> bound entries (CUDA-free)
 *   spark_k3_pool_sizing per-rank slice pool sizes (CUDA-free)
 *   THIS FILE            device pools, weight table, K3StageSlice launches
 *
 * The three launch arguments K3StageSlice takes are all DEVICE objects:
 *   K3LayerWeights[layer_count]  weights indexed by slice offset; the pack
 *                                payload pointers are UVA (the mmap is
 *                                host-registered) so they read on device.
 *   K3SliceState                 one struct holding the recurrent pools; the
 *                                slice's K3BindLayerState strides them per
 *                                layer by K3_KDA_*_DIM * K3_KDA_CONV_KERNEL
 *                                exactly as tests/host_cuda/k3_slice_host.cu
 *                                allocates them.
 *   K3LayerBuffers               per-step scratch + inputs. The slice loop
 *                                rebinds the WEIGHT fields per layer via
 *                                K3BindLayer, so this module fills only the
 *                                stream/scratch pointers once per step; the
 *                                two model-level fields the loop never sets
 *                                (attnres_out_weight, router_bias) are filled
 *                                once at bind time.
 *
 * PACK V2 SCALE FIELDS: only the routed experts are quantised in the released
 * checkpoint (the spine is BF16), so every *_scale field in K3LayerWeights
 * stays NULL and expert_interleave is 1 on every MoE layer. K3LayerLatentMoe
 * currently REFUSES interleave=1 (the grouped GEMM has not learned the cell
 * yet - see the struct comment in layer.cuh), so an end-to-end run surfaces
 * that refusal as a launch error until the kernels wave lands. The dispatch
 * encodes the pack truth regardless.
 */

#define SPARK_K3_DISPATCH_OK 0
#define SPARK_K3_DISPATCH_ERR_ARGUMENT -1
#define SPARK_K3_DISPATCH_ERR_CUDA -2
#define SPARK_K3_DISPATCH_ERR_REGISTER -3
#define SPARK_K3_DISPATCH_ERR_BIND -4

/* Per-step device inputs the serving tier owns. Routing is packed on device
 * by the serving tier's router step; every pointer may be NULL when the
 * corresponding feature is not in use, EXCEPT the routing arrays on a slice
 * that contains MoE layers, which the MoE layer consumes directly. */
typedef struct SparkK3StepInput
{
	const uint16_t *hidden_in;            /* rows*K3_HIDDEN; consumed in place */
	const uint32_t *positions;            /* rows */
	const uint32_t *context_length;       /* rows */
	const uint32_t *sequence_of_row;      /* rows */
	const uint32_t *sequence_row_begin;   /* sequences+1; NULL = row i is sequence i */
	const uint32_t *kda_state_index;      /* sequences; per-sequence state slot */
	const uint32_t *route_expert;         /* packed_rows */
	const uint32_t *route_packed_row;     /* packed_rows */
	const uint32_t *route_source_token;   /* packed_rows */
	const float *route_weight;            /* packed_rows */
	const uint32_t *group_row_offset;     /* K3_EXPERTS + 1 */
	const uint32_t *group_tile_prefix_w1; /* K3_EXPERTS + 1 */
	const uint32_t *group_tile_prefix_w2; /* K3_EXPERTS + 1 */
	const uint32_t *dense_row_offset;     /* 2 */
	uint32_t *dense_tile_prefix;          /* 2, device-writable */
	float *head_candidate_score;          /* rows*K3_KDA_HEADS; head kernel out */
	uint32_t *head_candidate_token;       /* rows */
	uint32_t *output_token;               /* rows */
	float *output_score;                  /* rows */
} SparkK3StepInput;

typedef struct SparkK3Dispatch
{
	/* slice geometry */
	uint32_t first_layer;
	uint32_t layer_count;
	uint32_t kda_count;
	uint32_t mla_count;
	/* capacities */
	uint32_t sequences;
	uint32_t max_rows;
	uint32_t routes_capacity;
	uint32_t kv_pages_per_view;
	uint64_t kv_page_bytes;
	/* device objects */
	K3LayerWeights *weights;        /* [layer_count] */
	K3SliceState *slice_state;      /* one */
	K3LayerBuffers *buffers;        /* one */
	K3LayerBuffers *buffers_host;   /* host template; step edits then uploads */
	LmKvView *mla_cache;            /* [mla_count] */
	LmKvAccessError *access_error;  /* [mla_count] */
	uint8_t *kv_pool;               /* mla_count*pages*page_bytes */
	uint32_t *page_table;           /* mla_count*pages */
	uint8_t *kda_state_pool;        /* kda_count*sequences*K3_KDA_STATE_SLOT_BYTES */
	uint16_t *kda_q_window_pool;    /* kda_count*sequences*QK_DIM*CONV_KERNEL */
	uint16_t *kda_k_window_pool;    /* kda_count*sequences*QK_DIM*CONV_KERNEL */
	uint16_t *kda_v_window_pool;    /* kda_count*sequences*V_DIM*CONV_KERNEL */
	uint8_t *scratch;               /* one blob, carve table below */
	size_t scratch_bytes;
	int device;
} SparkK3Dispatch;

/* Allocate every device pool for one TP4xPP4 rank slice. */
int32_t SparkK3DispatchCreate(SparkK3Dispatch *d, const SparkK3PoolSizing *sizing,
	uint32_t sequences, uint32_t max_rows, uint32_t kv_pages_per_view,
	uint64_t kv_page_bytes, int device);
void SparkK3DispatchDestroy(SparkK3Dispatch *d);

/* Pin the pack mmap into UVA so kernel-visible weight pointers work. */
int32_t SparkK3DispatchRegisterPack(SparkK3Pack *pack);

/* Fill the device K3LayerWeights table from the binder output and set the two
 * model-level buffer fields. Validates that every layer's required names
 * resolved; returns SPARK_K3_DISPATCH_ERR_BIND on the first hole. */
int32_t SparkK3DispatchBindWeights(SparkK3Dispatch *d, SparkK3Pack *pack,
	SparkK3BoundLayer *bounds, uint32_t layer_count);

/* One slice launch. Returns K3StageSlice's raw int32_t; 0 is success. */
int32_t SparkK3DispatchStep(SparkK3Dispatch *d, const SparkK3StepInput *in,
	uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows,
	uint32_t context, uint32_t multiprocessors, cudaStream_t stream);

#endif

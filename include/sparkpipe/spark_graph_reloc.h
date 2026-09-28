#pragma once

#include <stdint.h>
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GRAPH_RELOC_SLOT_SHARED UINT32_MAX
#define SPARK_GRAPH_RELOC_NONE UINT32_MAX
#define SPARK_GRAPH_RELOC_KERNEL_PARAMS_MAX 256u
#define SPARK_GRAPH_RELOC_BLOB_ALIGN 16u
#define SPARK_GRAPH_RELOC_WORD_FLOOR UINT64_C(0x10000)

typedef enum SparkGraphRelocNodeKind
{
	SPARK_GRAPH_RELOC_NODE_EMPTY = 0,
	SPARK_GRAPH_RELOC_NODE_KERNEL,
	SPARK_GRAPH_RELOC_NODE_COPY,
	SPARK_GRAPH_RELOC_NODE_FILL,
	SPARK_GRAPH_RELOC_NODE_HOST,
	SPARK_GRAPH_RELOC_NODE_EVENT_RECORD,
	SPARK_GRAPH_RELOC_NODE_EVENT_WAIT
} SparkGraphRelocNodeKind;

typedef enum SparkGraphRelocWord
{
	SPARK_GRAPH_RELOC_WORD_CONSTANT = 0,
	SPARK_GRAPH_RELOC_WORD_REGION,
	SPARK_GRAPH_RELOC_WORD_UNKNOWN,
	SPARK_GRAPH_RELOC_WORD_AMBIGUOUS
} SparkGraphRelocWord;

typedef enum SparkGraphRelocReason
{
	SPARK_GRAPH_RELOC_REASON_NONE = 0,
	SPARK_GRAPH_RELOC_REASON_UNKNOWN,
	SPARK_GRAPH_RELOC_REASON_AMBIGUOUS,
	SPARK_GRAPH_RELOC_REASON_NODE_KIND,
	SPARK_GRAPH_RELOC_REASON_NODE_QUERY,
	SPARK_GRAPH_RELOC_REASON_CAPACITY,
	SPARK_GRAPH_RELOC_REASON_REGION,
	SPARK_GRAPH_RELOC_REASON_SHAPE,
	SPARK_GRAPH_RELOC_REASON_CONSTANT,
	SPARK_GRAPH_RELOC_REASON_SITE,
	SPARK_GRAPH_RELOC_REASON_SLOT,
	SPARK_GRAPH_RELOC_REASON_TARGET,
	SPARK_GRAPH_RELOC_REASON_APPLY
} SparkGraphRelocReason;

typedef uint32_t (*SparkGraphRelocProbe)(void *context,uint64_t word);

typedef struct SparkGraphRelocRegion
{
	uint64_t base;
	uint64_t bytes;
	uint32_t id;
	uint32_t region_class;
	uint32_t slot;
	uint32_t reserved;
} SparkGraphRelocRegion;

typedef struct SparkGraphRelocKey
{
	uint32_t id;
	uint32_t slot;
	uint32_t region;
} SparkGraphRelocKey;

typedef struct SparkGraphRelocRegistry
{
	SparkGraphRelocRegion *regions;
	SparkGraphRelocKey *keys;
	uint32_t capacity;
	uint32_t count;
	uint32_t sealed;
	uint32_t reserved;
	uint64_t window_margin;
	uint64_t window_low;
	uint64_t window_high;
	SparkGraphRelocProbe probe;
	void *probe_context;
} SparkGraphRelocRegistry;

typedef struct SparkGraphRelocMatch
{
	uint64_t offset;
	uint32_t region;
	uint32_t verdict;
} SparkGraphRelocMatch;

typedef struct SparkGraphRelocParam
{
	uint32_t offset;
	uint32_t bytes;
} SparkGraphRelocParam;

typedef struct SparkGraphRelocNode
{
	const void *function;
	uint32_t kind;
	uint32_t param_first;
	uint32_t param_count;
	uint32_t blob_offset;
	uint32_t blob_bytes;
	uint32_t site_first;
	uint32_t site_count;
	uint32_t reserved;
} SparkGraphRelocNode;

typedef struct SparkGraphRelocSite
{
	uint64_t offset;
	uint32_t node;
	uint32_t byte_offset;
	uint32_t region_id;
	uint32_t region_slot;
} SparkGraphRelocSite;

typedef struct SparkGraphRelocImage
{
	SparkGraphRelocNode *nodes;
	void **handles;
	SparkGraphRelocParam *params;
	SparkGraphRelocSite *sites;
	uint8_t *blob;
	uint64_t blob_capacity;
	uint64_t blob_bytes;
	uint64_t word_count;
	uint32_t node_capacity;
	uint32_t node_count;
	uint32_t param_capacity;
	uint32_t param_count;
	uint32_t site_capacity;
	uint32_t site_count;
	uint32_t slot;
	uint32_t kernel_count;
} SparkGraphRelocImage;

typedef struct SparkGraphRelocNodeInput
{
	void *handle;
	const void *function;
	const SparkGraphRelocParam *params;
	uint32_t kind;
	uint32_t param_count;
	uint32_t blob_bytes;
	uint32_t reserved;
} SparkGraphRelocNodeInput;

typedef struct SparkGraphRelocFault
{
	uint64_t word;
	uint32_t reason;
	uint32_t node;
	uint32_t byte_offset;
	uint32_t detail;
} SparkGraphRelocFault;

typedef struct SparkGraphRelocCapacity
{
	uint64_t blob_bytes;
	uint32_t regions;
	uint32_t nodes;
	uint32_t params;
	uint32_t sites;
} SparkGraphRelocCapacity;

typedef struct SparkGraphRelocWorkspace
{
	SparkGraphRelocRegistry registry;
	SparkGraphRelocImage images[2];
	uint8_t *held_blob[2];
	uint8_t *patch_blob;
	uint64_t reference_key;
	uint32_t reference_valid;
	uint32_t reference;
} SparkGraphRelocWorkspace;

const char *SparkGraphRelocReasonName(uint32_t reason);
void SparkGraphRelocRegistryInit(SparkGraphRelocRegistry *registry,SparkGraphRelocRegion *regions,SparkGraphRelocKey *keys,uint32_t capacity,uint64_t window_margin);
void SparkGraphRelocRegistryReset(SparkGraphRelocRegistry *registry);
SparkStatus SparkGraphRelocRegister(SparkGraphRelocRegistry *registry,uint32_t id,const void *base,uint64_t bytes,uint32_t region_class,uint32_t slot);
SparkStatus SparkGraphRelocSeal(SparkGraphRelocRegistry *registry,SparkGraphRelocFault *fault);
void SparkGraphRelocClassify(const SparkGraphRelocRegistry *registry,uint64_t word,SparkGraphRelocMatch *match);
uint32_t SparkGraphRelocFind(const SparkGraphRelocRegistry *registry,uint32_t id,uint32_t slot);
void SparkGraphRelocImageReset(SparkGraphRelocImage *image,uint32_t slot);
uint8_t *SparkGraphRelocImageTail(SparkGraphRelocImage *image,uint32_t bytes);
SparkStatus SparkGraphRelocImageCommit(SparkGraphRelocImage *image,const SparkGraphRelocRegistry *registry,const SparkGraphRelocNodeInput *input,SparkGraphRelocFault *fault);
SparkStatus SparkGraphRelocValidatePair(const SparkGraphRelocImage *first,const SparkGraphRelocImage *second,SparkGraphRelocFault *fault);
SparkStatus SparkGraphRelocRebase(const SparkGraphRelocImage *image,const SparkGraphRelocRegistry *target,uint32_t target_slot,uint8_t *blob,uint64_t blob_capacity,SparkGraphRelocFault *fault);
SparkStatus SparkGraphRelocHeldReset(const SparkGraphRelocImage *image,uint8_t *held,uint64_t held_capacity);
SparkStatus SparkGraphRelocWorkspaceCreate(const SparkGraphRelocCapacity *capacity,uint64_t window_margin,SparkGraphRelocWorkspace **out);
void SparkGraphRelocWorkspaceDestroy(SparkGraphRelocWorkspace *workspace);

uint32_t SparkGraphRelocCudaProbe(void *context,uint64_t word);
SparkStatus SparkGraphRelocCaptureCuda(void *graph,const SparkGraphRelocRegistry *registry,uint32_t slot,SparkGraphRelocImage *image,SparkGraphRelocFault *fault);
SparkStatus SparkGraphRelocApplyCuda(void *exec,const SparkGraphRelocImage *image,uint8_t *held,uint64_t held_capacity,const uint8_t *blob,uint32_t force,uint32_t *applied,SparkGraphRelocFault *fault);

#ifdef __cplusplus
}
#endif

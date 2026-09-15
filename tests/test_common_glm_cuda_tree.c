#include <stdio.h>
#include <stdint.h>
#include "common/common_glm_cuda_tree/spark_glm_cuda_config.h"
#include "common/common_glm_cuda_tree/spark_glm_cuda_launch_shape.h"

#ifndef VECTOR_ROUTED_LAYERS
#error "VECTOR_ROUTED_LAYERS must be injected from llm_defines.h"
#endif

static uint32_t failures;

#define CHECK_VECTOR(actual,vector,label) \
	do { \
		if ( (uint32_t)(actual) != (uint32_t)(vector) ) \
		{ \
			printf("  FAIL %s: config=%u vector=%u\n", \
				label,(unsigned)(actual),(unsigned)(vector)); \
			failures++; \
		} \
	} while (0)

int main(void)
{
	CHECK_VECTOR(GLM_ROUTED_LAYERS,VECTOR_ROUTED_LAYERS,"routed layers");
	CHECK_VECTOR(GLM_GATE_UP_DIM,VECTOR_GATE_UP_DIM,"gate+up dim");
	CHECK_VECTOR(GLM_LATENT_ROW,VECTOR_LATENT_ROW,"latent row elements");
	CHECK_VECTOR(GLM_WEIGHT_LAYERS,VECTOR_WEIGHT_LAYERS,"weight layers incl MTP");
	CHECK_VECTOR(GLM_KV_SLOT_BYTES,VECTOR_KV_SLOT_BYTES,"kv slot bytes");
	CHECK_VECTOR(GlmRowsPerExpert(1u),VECTOR_ROWS_PER_EXPERT_B1,"rows/expert at B1");
	CHECK_VECTOR(GlmRowsPerExpert(128u),VECTOR_ROWS_PER_EXPERT_B128,"rows/expert at B128");
	CHECK_VECTOR(GlmRowsPerExpert(1024u),VECTOR_ROWS_PER_EXPERT_B1024,"rows/expert at B1024");
	CHECK_VECTOR(GlmLayerHasFullIndexer(0u),VECTOR_FULL_INDEXER_L0,"full indexer layer 0");
	CHECK_VECTOR(GlmLayerHasFullIndexer(2u),VECTOR_FULL_INDEXER_L2,"full indexer layer 2");
	CHECK_VECTOR(GlmLayerHasFullIndexer(3u),VECTOR_FULL_INDEXER_L3,"full indexer layer 3");
	CHECK_VECTOR(GlmLayerHasFullIndexer(5u),VECTOR_FULL_INDEXER_L5,"full indexer layer 5");
	CHECK_VECTOR(GlmLayerHasFullIndexer(6u),VECTOR_FULL_INDEXER_L6,"full indexer layer 6");
	CHECK_VECTOR(GlmLayerHasFullIndexer(9u),VECTOR_FULL_INDEXER_L9,"full indexer layer 9");
	CHECK_VECTOR(GlmLayerHasFullIndexer(10u),VECTOR_FULL_INDEXER_L10,"full indexer layer 10");
	CHECK_VECTOR(GLM_LAYER_THREADS,VECTOR_LAYER_THREADS,"layer threads");
	CHECK_VECTOR(GLM_ATTN_THREADS,VECTOR_ATTN_THREADS,"attention threads");
	CHECK_VECTOR(GLM_DSA_QUERY_DIM,VECTOR_DSA_QUERY_DIM,"dsa query dim");
	CHECK_VECTOR(GLM_KV_PAGE_SLOTS,VECTOR_KV_PAGE_SLOTS,"kv page slots");
	CHECK_VECTOR(GLM_LAYER_KIND(0u) == LM_LAYER_LATENT,1,"layer kind is latent");
	if ( failures == 0u )
		printf("PASS common_glm_cuda_tree config algebra matches llm_defines vectors\n");
	return failures == 0u ? 0 : 1;
}

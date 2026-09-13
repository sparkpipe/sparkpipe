#include "sparkpipe/spark_dsv4_cache_plan.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static void SparkDsv4PrintPlan(SparkDsv4ModelVariant variant,const char *name)
{
	SparkDsv4CachePlanConfiguration configuration;
	SparkDsv4CachePlan plan;
	double exact_gib;
	double worst_gib;
	double saved_gib;
	double saved_percent;

	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_DSV4_CACHE_PLAN_ABI_VERSION;
	configuration.descriptor_bytes = sizeof(configuration);
	configuration.model_variant = variant;
	configuration.backbone_layer_count = variant == SPARK_DSV4_MODEL_VARIANT_FLASH
		? 43u
		: 61u;
	configuration.include_mtp_layer = 1u;
	configuration.active_sequence_capacity = 8u;
	configuration.maximum_context_tokens_per_sequence = 1048576u;
	configuration.aggregate_context_token_capacity = 1048576u;
	configuration.compressed_history_page_entries =
		SPARK_DSV4_CACHE_PLAN_DEFAULT_PAGE_ENTRIES;
	configuration.attention_content_element_bits = 16u;
	configuration.attention_rope_element_bits = 16u;
	configuration.indexer_element_bits = 8u;
	configuration.compressor_state_element_bits = 16u;
	configuration.allocation_alignment_bytes =
		SPARK_DSV4_CACHE_PLAN_DEFAULT_ALIGNMENT_BYTES;
	if ( SparkDsv4CachePlanBuild(&configuration,&plan) != SPARK_STATUS_OK )
	{
		printf("%s plan failed\n",name);
		return;
	}
	exact_gib = plan.total_arena_bytes / 1073741824.0;
	worst_gib = plan.worst_class_total_arena_bytes / 1073741824.0;
	saved_gib = worst_gib - exact_gib;
	saved_percent = worst_gib != 0.0 ? (saved_gib / worst_gib) * 100.0 : 0.0;
	printf("%s: layers=%u sliding=%u CSA=%u HCA=%u\n",
		name,
		plan.planned_layer_count,
		plan.sliding_layer_count,
		plan.compressed_sparse_layer_count,
		plan.heavily_compressed_layer_count);
	printf("  exact %.3f GiB, worst-class %.3f GiB, saved %.3f GiB (%.1f%%)\n",
		exact_gib,worst_gib,saved_gib,saved_percent);
	printf("  exact arenas: sliding %.3f GiB, history %.3f GiB, state %.3f GiB\n",
		plan.sliding_arena_bytes / 1073741824.0,
		plan.compressed_history_arena_bytes / 1073741824.0,
		plan.compressor_state_arena_bytes / 1073741824.0);
	printf("  exact bytes=%" PRIu64 ", worst bytes=%" PRIu64 "\n",
		plan.total_arena_bytes,plan.worst_class_total_arena_bytes);
}

int main(void)
{
	SparkDsv4PrintPlan(SPARK_DSV4_MODEL_VARIANT_FLASH,"DSV4 Flash");
	SparkDsv4PrintPlan(SPARK_DSV4_MODEL_VARIANT_PRO,"DSV4 Pro");
	return(0);
}

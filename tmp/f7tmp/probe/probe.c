#include <stdio.h>
#include "sparkpipe/spark_dsv4_cache_plan.h"
int main(void){
    SparkDsv4CachePlanConfiguration c; SparkDsv4CachePlan p;
    memset(&c,0,sizeof(c));
    c.abi_version=SPARK_DSV4_CACHE_PLAN_ABI_VERSION;
    c.descriptor_bytes=sizeof(c);
    c.model_variant=SPARK_DSV4_MODEL_VARIANT_FLASH;
    c.backbone_layer_count=43u; c.include_mtp_layer=0u;
    c.active_sequence_capacity=8u;
    c.maximum_context_tokens_per_sequence=1048576u;
    c.aggregate_context_token_capacity=1048576u;
    c.compressed_history_page_entries=512u;
    c.attention_content_element_bits=16u; c.attention_rope_element_bits=16u;
    c.indexer_element_bits=8u; c.compressor_state_element_bits=16u;
    c.allocation_alignment_bytes=256u;
    SparkStatus s=SparkDsv4CachePlanBuild(&c,&p);
    printf("flash build status=%d\n",(int)s);
    return 0;
}

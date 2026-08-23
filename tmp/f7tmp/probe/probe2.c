#include <stdio.h>
#include "sparkpipe/spark_dsv4_cache_plan.h"
static int build(int variant,int mtp){
    SparkDsv4CachePlanConfiguration c; SparkDsv4CachePlan p;
    memset(&c,0,sizeof(c));
    c.abi_version=SPARK_DSV4_CACHE_PLAN_ABI_VERSION; c.descriptor_bytes=sizeof(c);
    c.model_variant=(SparkDsv4ModelVariant)variant;
    c.backbone_layer_count=(variant==2)?61u:43u; c.include_mtp_layer=mtp;
    c.active_sequence_capacity=8u;
    c.maximum_context_tokens_per_sequence=1048576u;
    c.aggregate_context_token_capacity=1048576u;
    c.compressed_history_page_entries=512u;
    c.attention_content_element_bits=16u; c.attention_rope_element_bits=16u;
    c.indexer_element_bits=8u; c.compressor_state_element_bits=16u;
    c.allocation_alignment_bytes=256u;
    return (int)SparkDsv4CachePlanBuild(&c,&p);
}
int main(void){
    printf("flash mtp0=%d\n",build(1,0));
    printf("pro   mtp0=%d\n",build(2,0));
    printf("pro   mtp1=%d (test's failing case)\n",build(2,1));
    return 0;
}

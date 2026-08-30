/* Wire-layout probe for the qwen38_max stage pack header and directory
 * entry: prints one "kind field offset size" line per struct member so a
 * CPU-side audit can compare the C layout against the Python struct
 * layouts the packer and verifier use. Pure C, no CUDA, no GPU: the
 * stagepack format contract is checkable entirely on the build host.
 *
 * Usage: cc -I<family source dir> -I<firmware include dir> -I<model
 * family include dir> -I<repo include dir> -o probe <this file>
 * The probe adapts to the header's FORMAT_VERSION: v2 headers add the
 * TP fields, wherever that header's family chose to put them. */
#include <stddef.h>
#include <stdio.h>

#include "spark_qwen38_max_stagepack_format.h"

static void print_header_field(const char *name, size_t offset, size_t size)
{
	printf("header %s %zu %zu\n",name,offset,size);
}

int main(void)
{
	typedef SparkQwen38MaxStagePackHeader header_t;
	typedef SparkQwen38MaxStagePackEntry entry_t;
	printf("probe format_version %u header_bytes %u entry_bytes %u\n",
		SPARK_QWEN38_MAX_STAGEPACK_FORMAT_VERSION,
		SPARK_QWEN38_MAX_STAGEPACK_HEADER_BYTES,
		SPARK_QWEN38_MAX_STAGEPACK_ENTRY_BYTES);
	printf("probe sizeof_header %zu sizeof_entry %zu\n",
		sizeof(header_t),sizeof(entry_t));
	print_header_field("magic",offsetof(header_t,magic),sizeof(uint32_t));
	print_header_field("format_version",offsetof(header_t,format_version),sizeof(uint32_t));
	print_header_field("header_bytes",offsetof(header_t,header_bytes),sizeof(uint32_t));
	print_header_field("directory_entry_bytes",offsetof(header_t,directory_entry_bytes),sizeof(uint32_t));
	print_header_field("tensor_count",offsetof(header_t,tensor_count),sizeof(uint32_t));
	print_header_field("hidden_dimension",offsetof(header_t,hidden_dimension),sizeof(uint32_t));
	print_header_field("layer_count",offsetof(header_t,layer_count),sizeof(uint32_t));
	print_header_field("first_layer_index",offsetof(header_t,first_layer_index),sizeof(uint32_t));
	print_header_field("total_layer_count",offsetof(header_t,total_layer_count),sizeof(uint32_t));
	print_header_field("attention_period",offsetof(header_t,attention_period),sizeof(uint32_t));
	print_header_field("full_attention_phase",offsetof(header_t,full_attention_phase),sizeof(uint32_t));
	print_header_field("gdn_key_head_count",offsetof(header_t,gdn_key_head_count),sizeof(uint32_t));
	print_header_field("gdn_value_head_count",offsetof(header_t,gdn_value_head_count),sizeof(uint32_t));
	print_header_field("gdn_head_key_dimension",offsetof(header_t,gdn_head_key_dimension),sizeof(uint32_t));
	print_header_field("gdn_head_value_dimension",offsetof(header_t,gdn_head_value_dimension),sizeof(uint32_t));
	print_header_field("gdn_conv_kernel",offsetof(header_t,gdn_conv_kernel),sizeof(uint32_t));
	print_header_field("attn_query_head_count",offsetof(header_t,attn_query_head_count),sizeof(uint32_t));
	print_header_field("attn_kv_head_count",offsetof(header_t,attn_kv_head_count),sizeof(uint32_t));
	print_header_field("attn_head_dimension",offsetof(header_t,attn_head_dimension),sizeof(uint32_t));
	print_header_field("attn_rope_dimension",offsetof(header_t,attn_rope_dimension),sizeof(uint32_t));
	print_header_field("routed_expert_count",offsetof(header_t,routed_expert_count),sizeof(uint32_t));
	print_header_field("experts_per_token",offsetof(header_t,experts_per_token),sizeof(uint32_t));
	print_header_field("expert_intermediate_dimension",offsetof(header_t,expert_intermediate_dimension),sizeof(uint32_t));
	print_header_field("output_vocab_count",offsetof(header_t,output_vocab_count),sizeof(uint32_t));
	print_header_field("mxfp4_group_size",offsetof(header_t,mxfp4_group_size),sizeof(uint32_t));
	print_header_field("mtp_layer_count",offsetof(header_t,mtp_layer_count),sizeof(uint32_t));
#if SPARK_QWEN38_MAX_STAGEPACK_FORMAT_VERSION >= 2
	print_header_field("tp_degree",offsetof(header_t,tp_degree),sizeof(uint32_t));
	print_header_field("tp_rank",offsetof(header_t,tp_rank),sizeof(uint32_t));
#endif
	print_header_field("directory_offset",offsetof(header_t,directory_offset),sizeof(uint64_t));
	print_header_field("file_bytes",offsetof(header_t,file_bytes),sizeof(uint64_t));
	return(0);
}

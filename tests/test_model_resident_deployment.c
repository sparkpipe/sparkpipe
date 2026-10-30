#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_chat_template.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_resident_deployment.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_tokenizer_sidecar.h"

static void TestBuildDescriptor(
	SparkModelServingAdapterDescriptor *descriptor)
{
	memset(descriptor,0,sizeof(*descriptor));
	descriptor->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	descriptor->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	descriptor->capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH;
	descriptor->stage_count = 3u;
	descriptor->layer_count = 6u;
	descriptor->boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	descriptor->boundary_element_count = 64u;
	descriptor->boundary_element_bytes = 2u;
	descriptor->linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->expert_weight_codec = SPARK_WEIGHT_CODEC_INT8;
	descriptor->kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->max_inflight_submission_count = 4u;
	descriptor->max_active_sequence_count = 4u;
	descriptor->max_input_row_count = 8u;
	descriptor->max_resident_sequence_count = 16u;
	descriptor->max_output_token_count = 4u;
	descriptor->cache_block_token_count = 4u;
	descriptor->adapter_id = "test.adapter";
	descriptor->model_id = "test/model";
	descriptor->model_revision = "revision";
	descriptor->driver_program_name = "resident_decode";
	descriptor->artifact_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
	descriptor->stage_layer_counts[0] = 2u;
	descriptor->stage_layer_counts[1] = 2u;
	descriptor->stage_layer_counts[2] = 2u;
}

static void TestEosMetadata(const char *members,SparkStatus expected)
{
	SparkModelResidentDeployment deployment;
	char buffer[8192],path[256];
	FILE *file;
	uint32_t count;
	file = fopen("tests/fixtures/model_resident_deployment.json","rb");
	assert(file != 0);
	count = (uint32_t)fread(buffer,1,sizeof(buffer),file);
	assert(feof(file) != 0 && count > 1u && buffer[0] == '{');
	assert(fclose(file) == 0);
	assert(snprintf(path,sizeof(path),"/tmp/sparkpipe-eos-%ld.json",(long)getpid()) > 0);
	file = fopen(path,"wb");
	assert(file != 0);
	assert(fprintf(file,"{%s",members) > 0);
	assert(fwrite(buffer + 1u,1,count - 1u,file) == count - 1u);
	assert(fclose(file) == 0);
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad(path,&deployment) == expected);
	if ( expected == SPARK_STATUS_OK )
	{
		assert(deployment.eos_token_count == 2u);
		assert(deployment.eos_token_ids[0] == 0u);
		assert(deployment.eos_token_ids[1] == 154820u);
	}
	SparkModelResidentDeploymentDestroy(&deployment);
	assert(unlink(path) == 0);
}

static void TestPrefixReuseKey(const char *members,SparkStatus expected)
{
	SparkModelResidentDeployment deployment;
	char buffer[8192],path[256];
	FILE *file;
	uint32_t count;
	file = fopen("tests/fixtures/model_resident_deployment.json","rb");
	assert(file != 0);
	count = (uint32_t)fread(buffer,1,sizeof(buffer),file);
	assert(feof(file) != 0 && count > 1u && buffer[0] == '{');
	assert(fclose(file) == 0);
	assert(snprintf(path,sizeof(path),"/tmp/sparkpipe-prefix-%ld.json",(long)getpid()) > 0);
	file = fopen(path,"wb");
	assert(file != 0);
	assert(fprintf(file,"{%s",members) > 0);
	assert(fwrite(buffer + 1u,1,count - 1u,file) == count - 1u);
	assert(fclose(file) == 0);
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad(path,&deployment) == expected);
	SparkModelResidentDeploymentDestroy(&deployment);
	assert(unlink(path) == 0);
}

static void TestRuntimeMemberShared(const char *member,SparkStatus expected,uint32_t positions,uint32_t shared)
{
	SparkModelResidentDeployment deployment;
	char buffer[8192],path[256];
	const char *anchor;
	FILE *file;
	uint32_t count,prefix;
	file = fopen("tests/fixtures/model_resident_deployment.json","rb");
	assert(file != 0);
	count = (uint32_t)fread(buffer,1,sizeof(buffer) - 1u,file);
	assert(feof(file) != 0 && count > 1u);
	assert(fclose(file) == 0);
	buffer[count] = '\0';
	anchor = strstr(buffer,"\"runtime_limits\": {");
	assert(anchor != 0);
	prefix = (uint32_t)(anchor - buffer) + (uint32_t)strlen("\"runtime_limits\": {");
	assert(snprintf(path,sizeof(path),"/tmp/sparkpipe-runtime-%ld.json",(long)getpid()) > 0);
	file = fopen(path,"wb");
	assert(file != 0);
	assert(fwrite(buffer,1,prefix,file) == prefix);
	assert(fprintf(file,"%s,",member) > 0);
	assert(fwrite(buffer + prefix,1,count - prefix,file) == count - prefix);
	assert(fclose(file) == 0);
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad(path,&deployment) == expected);
	if ( expected == SPARK_STATUS_OK )
		assert(deployment.max_sequence_positions == positions && deployment.max_shared_input_row_count == shared);
	if ( expected == SPARK_STATUS_OK && strstr(member,"streamed_prefill") != 0 )
		assert(deployment.streamed_prefill_quantum == 4096u && deployment.streamed_prefill_min_tokens == 32768u);
	if ( expected == SPARK_STATUS_OK && strstr(member,"streamed_prefill") == 0 )
		assert(deployment.streamed_prefill_quantum == 0u && deployment.streamed_prefill_min_tokens == 0u);
	SparkModelResidentDeploymentDestroy(&deployment);
	assert(unlink(path) == 0);
}

static void TestRuntimeMember(const char *member,SparkStatus expected,uint32_t positions)
{
	TestRuntimeMemberShared(member,expected,positions,0u);
}

static void TestSharedInputRows(void)
{
	TestRuntimeMemberShared("\"max_shared_input_rows\": 2",SPARK_STATUS_OK,0u,2u);
	TestRuntimeMemberShared("\"max_shared_input_rows\": 4",SPARK_STATUS_OK,0u,4u);
	TestRuntimeMemberShared("\"max_shared_input_rows\": 5",SPARK_STATUS_SCHEMA_ERROR,0u,0u);
	TestRuntimeMemberShared("\"max_shared_input_rows\": 0",SPARK_STATUS_SCHEMA_ERROR,0u,0u);
	TestRuntimeMemberShared("\"streamed_prefill_min_tokens\": 32768, \"streamed_prefill_quantum\": 4096",SPARK_STATUS_OK,0u,0u);
	TestRuntimeMemberShared("\"streamed_prefill_quantum\": 4096",SPARK_STATUS_SCHEMA_ERROR,0u,0u);
	TestRuntimeMemberShared("\"streamed_prefill_min_tokens\": 2048, \"streamed_prefill_quantum\": 4096",SPARK_STATUS_SCHEMA_ERROR,0u,0u);
	TestRuntimeMemberShared("\"max_shared_input_row\": 2",SPARK_STATUS_SCHEMA_ERROR,0u,0u);
	puts("test_model_resident_deployment: a shared-wave row limit within max_input_rows loads; zero, larger or misspelled is refused");
}

static void TestDraftServerMember(const char *member,SparkStatus expected,SparkModelResidentDeployment *deployment)
{
	char buffer[8192],path[256];
	const char *anchor;
	FILE *file;
	uint32_t count,prefix;
	file = fopen("tests/fixtures/model_resident_deployment.json","rb");
	assert(file != 0);
	count = (uint32_t)fread(buffer,1,sizeof(buffer) - 1u,file);
	assert(feof(file) != 0 && count > 1u);
	assert(fclose(file) == 0);
	buffer[count] = '\0';
	anchor = strchr(buffer,'{');
	assert(anchor != 0);
	prefix = (uint32_t)(anchor - buffer) + 1u;
	assert(snprintf(path,sizeof(path),"/tmp/sparkpipe-draft-%ld.json",(long)getpid()) > 0);
	file = fopen(path,"wb");
	assert(file != 0);
	assert(fwrite(buffer,1,prefix,file) == prefix);
	assert(fprintf(file,"\"draft_server\": %s,",member) > 0);
	assert(fwrite(buffer + prefix,1,count - prefix,file) == count - prefix);
	assert(fclose(file) == 0);
	SparkModelResidentDeploymentReset(deployment);
	assert(SparkModelResidentDeploymentLoad(path,deployment) == expected);
	assert(unlink(path) == 0);
}

static void TestDraftServer(void)
{
	static const char *const refused[] =
	{
		"{\"host\":\"\",\"port\":7702,\"target_model\":\"kimi-k3\",\"depth\":7,\"wait_ms\":30,\"min_confidence\":0.35}",
		"{\"host\":\"127.0.0.1\",\"port\":0,\"target_model\":\"kimi-k3\",\"depth\":7,\"wait_ms\":30,\"min_confidence\":0.35}",
		"{\"host\":\"127.0.0.1\",\"port\":70000,\"target_model\":\"kimi-k3\",\"depth\":7,\"wait_ms\":30,\"min_confidence\":0.35}",
		"{\"host\":\"127.0.0.1\",\"port\":7702,\"target_model\":\"\",\"depth\":7,\"wait_ms\":30,\"min_confidence\":0.35}",
		"{\"host\":\"127.0.0.1\",\"port\":7702,\"target_model\":\"a-target-model-name-of-thirty-two\",\"depth\":7,\"wait_ms\":30,\"min_confidence\":0.35}",
		"{\"host\":\"127.0.0.1\",\"port\":7702,\"target_model\":\"kimi-k3\",\"depth\":0,\"wait_ms\":30,\"min_confidence\":0.35}",
		"{\"host\":\"127.0.0.1\",\"port\":7702,\"target_model\":\"kimi-k3\",\"depth\":9,\"wait_ms\":30,\"min_confidence\":0.35}",
		"{\"host\":\"127.0.0.1\",\"port\":7702,\"target_model\":\"kimi-k3\",\"depth\":7,\"wait_ms\":0,\"min_confidence\":0.35}",
		"{\"host\":\"127.0.0.1\",\"port\":7702,\"target_model\":\"kimi-k3\",\"depth\":7,\"wait_ms\":1001,\"min_confidence\":0.35}",
		"{\"host\":\"127.0.0.1\",\"port\":7702,\"target_model\":\"kimi-k3\",\"depth\":7,\"wait_ms\":30,\"min_confidence\":1.0}",
		"{\"host\":\"127.0.0.1\",\"port\":7702,\"target_model\":\"kimi-k3\",\"depth\":7,\"wait_ms\":30,\"min_confidence\":-0.1}",
		"{\"host\":\"127.0.0.1\",\"port\":7702,\"target_model\":\"kimi-k3\",\"depth\":7,\"wait_ms\":30}",
		"{\"host\":\"127.0.0.1\",\"port\":7702,\"target_model\":\"kimi-k3\",\"depth\":7,\"wait_ms\":30,\"min_confidence\":0.35,\"window\":8}",
		"[]"
	};
	SparkModelResidentDeployment deployment;
	uint32_t index;
	TestDraftServerMember("{\"host\":\"127.0.0.1\",\"port\":7702,\"target_model\":\"kimi-k3\",\"depth\":7,\"wait_ms\":30,\"min_confidence\":0.35}",
		SPARK_STATUS_OK,&deployment);
	assert(deployment.draft_server.host != 0 && strcmp(deployment.draft_server.host,"127.0.0.1") == 0 && deployment.draft_server.port == 7702u);
	assert(strcmp(deployment.draft_server.target_model,"kimi-k3") == 0 && deployment.draft_server.depth == 7u && deployment.draft_server.wait_ms == 30u);
	assert(deployment.draft_server.min_confidence > 0.349f && deployment.draft_server.min_confidence < 0.351f);
	SparkModelResidentDeploymentDestroy(&deployment);
	for (index=0u; index<sizeof(refused) / sizeof(refused[0]); index++)
	{
		TestDraftServerMember(refused[index],SPARK_STATUS_SCHEMA_ERROR,&deployment);
		SparkModelResidentDeploymentDestroy(&deployment);
	}
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment.json",&deployment) == SPARK_STATUS_OK);
	assert(deployment.draft_server.host == 0);
	SparkModelResidentDeploymentDestroy(&deployment);
	puts("test_model_resident_deployment: a draft server section loads its endpoint, target, depth, wait and confidence cut; empty, out-of-range, missing or unknown members are refused; no section means no draft server");
}

static void TestSequencePositions(void)
{
	SparkModelResidentDeployment deployment;
	TestRuntimeMember("\"max_sequence_positions\": 512",SPARK_STATUS_OK,512u);
	TestRuntimeMember("\"max_sequence_positions\": 0",SPARK_STATUS_SCHEMA_ERROR,0u);
	TestRuntimeMember("\"max_sequence_positions\": 1",SPARK_STATUS_SCHEMA_ERROR,0u);
	TestRuntimeMember("\"max_sequence_positions\": 512, \"max_sequence_positions\": 1024",SPARK_STATUS_SCHEMA_ERROR,0u);
	TestRuntimeMember("\"max_sequence_position\": 512",SPARK_STATUS_SCHEMA_ERROR,0u);
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment.json",&deployment) == SPARK_STATUS_OK);
	assert(deployment.max_sequence_positions == 0u);
	SparkModelResidentDeploymentDestroy(&deployment);
	assert(SparkModelResidentDeploymentLoad("deployment/glm5_next_tp16/model_resident.json",&deployment) == SPARK_STATUS_OK);
	assert(deployment.node_count == 16u);
	assert(deployment.max_sequence_positions == 131072u);
	assert(deployment.runtime_limits.max_input_row_count == 1024u);
	assert(deployment.runtime_limits.kv_logical_page_capacity == 524288u && deployment.runtime_limits.kv_physical_page_capacity == 32768u);
	SparkModelResidentDeploymentDestroy(&deployment);
}

static void TestGlm5NextDeploymentServesText(void)
{
	SparkModelResidentDeployment deployment;
	SparkTokenizerSidecarConfiguration configuration;
	SparkTokenizerSidecar sidecar;
	char actual_sha256[SPARK_SHA256_HEX_BYTES];
	const char *asset = "qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json";
	uint32_t index;
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad("deployment/glm5_next_tp16/model_resident.json",&deployment) == SPARK_STATUS_OK);
	assert(deployment.tokenizer_asset_path != 0);
	assert(strcmp(deployment.tokenizer_asset_path,"tokenizer/tokenizer.json") == 0);
	assert(deployment.tokenizer_asset_sha256 != 0);
	assert(SparkSha256File(asset,actual_sha256) == SPARK_STATUS_OK);
	assert(strcmp(actual_sha256,deployment.tokenizer_asset_sha256) == 0);
	SparkTokenizerSidecarReset(&sidecar);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TOKENIZER_SIDECAR_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_TOKENIZER_SIDECAR_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.asset_path = asset;
	configuration.format = SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO;
	assert(SparkTokenizerSidecarLoad(&sidecar,&configuration) == SPARK_STATUS_OK);
	assert((uint64_t)sidecar.tokenizer.maximum_token_id + 1u == deployment.tokenizer_vocabulary_size);
	assert(deployment.eos_token_count != 0u);
	for (index=0u; index<deployment.eos_token_count; index++)
		assert(deployment.eos_token_ids[index] < deployment.tokenizer_vocabulary_size);
	SparkTokenizerSidecarUnload(&sidecar);
	SparkModelResidentDeploymentDestroy(&deployment);
}

static char *TestReadText(const char *path)
{
	FILE *file;
	long size;
	char *text;
	file = fopen(path,"rb");
	assert(file != 0);
	assert(fseek(file,0,SEEK_END) == 0);
	size = ftell(file);
	assert(size > 0);
	assert(fseek(file,0,SEEK_SET) == 0);
	text = malloc((size_t)size + 1u);
	assert(text != 0);
	assert(fread(text,1u,(size_t)size,file) == (size_t)size);
	text[size] = '\0';
	assert(fclose(file) == 0);
	return(text);
}

static SparkStatus TestLoadWithChatTemplate(const char *chat_template_json,
	SparkModelResidentDeployment *deployment)
{
	char *base,path[256];
	FILE *file;
	SparkStatus status;
	base = TestReadText("tests/fixtures/model_resident_deployment.json");
	assert(base[0] == '{');
	assert(snprintf(path,sizeof(path),"/tmp/sparkpipe-chat-template-%ld.json",(long)getpid()) > 0);
	file = fopen(path,"wb");
	assert(file != 0);
	if ( chat_template_json != 0 )
		assert(fprintf(file,"{\"chat_template\":%s,%s",chat_template_json,base + 1) > 0);
	else
		assert(fputs(base,file) != EOF);
	assert(fclose(file) == 0);
	free(base);
	SparkModelResidentDeploymentReset(deployment);
	status = SparkModelResidentDeploymentLoad(path,deployment);
	assert(unlink(path) == 0);
	return(status);
}

static void TestRender(const SparkModelResidentDeployment *deployment, const char *body,
	bool thinking, SparkChatTemplateOutcome expected, const char *expected_text)
{
	SparkJsonDocument document;
	char *text = 0;
	uint32_t text_bytes = 0u;
	SparkChatTemplateOutcome outcome;
	SparkJsonDocumentReset(&document);
	assert(SparkJsonParseText(body,strlen(body),&document) == SPARK_STATUS_OK);
	outcome = SparkChatTemplateRender(&deployment->chat_template,&document,
		SparkJsonGetRootToken(&document),thinking,&text,&text_bytes);
	if ( outcome != expected || (expected_text != 0 && (text == 0 || strcmp(text,expected_text) != 0)) )
	{
		fprintf(stderr,"chat template render: outcome %d (want %d) text [%s] want [%s]\n",
			(int)outcome,(int)expected,text != 0 ? text : "",expected_text != 0 ? expected_text : "");
		assert(0);
	}
	if ( expected_text != 0 )
		assert(text_bytes == strlen(expected_text));
	else
		assert(text == 0 && text_bytes == 0u);
	free(text);
	SparkJsonDocumentDestroy(&document);
}

static void TestChatTemplateSchema(const char *chat_template_json, SparkStatus expected)
{
	SparkModelResidentDeployment deployment;
	SparkStatus status = TestLoadWithChatTemplate(chat_template_json,&deployment);
	if ( status != expected )
		fprintf(stderr,"chat_template schema: status %d want %d for %s\n",(int)status,(int)expected,chat_template_json);
	assert(status == expected);
	if ( expected == SPARK_STATUS_OK )
		SparkModelResidentDeploymentDestroy(&deployment);
}

static void TestOverCountArraysFailCleanly(void)
{
	static const char *base_template = "{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
		"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,"
		"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"U\"]";
	SparkModelResidentDeployment deployment;
	char json[1024],*base,*nodes,*open,*close,path[256];
	FILE *file;
	uint32_t depth = 0u,index;
	int quoted = 0;
	assert(snprintf(json,sizeof(json),"%s,\"control_markers\":[\"<a>\",\"<b>\"]}",base_template) > 0);
	TestChatTemplateSchema(json,SPARK_STATUS_OK);
	assert(snprintf(json,sizeof(json),"%s,\"control_markers\":[\"<a>\",\"<b>\",\"<c>\",\"<d>\",\"<e>\",\"<f>\",\"<g>\",\"<h>\",\"<i>\"]}",base_template) > 0);
	TestChatTemplateSchema(json,SPARK_STATUS_SCHEMA_ERROR);
	base = TestReadText("tests/fixtures/model_resident_deployment.json");
	nodes = strstr(base,"\"nodes\"");
	assert(nodes != 0);
	open = strchr(nodes,'[');
	assert(open != 0);
	for (close = open; *close != '\0'; close++)
	{
		if ( quoted != 0 )
		{
			if ( *close == '\\' && close[1] != '\0' )
				close++;
			else if ( *close == '"' )
				quoted = 0;
			continue;
		}
		if ( *close == '"' )
			quoted = 1;
		else if ( *close == '[' || *close == '{' )
			depth++;
		else if ( (*close == ']' || *close == '}') && --depth == 0u )
			break;
	}
	assert(*close == ']');
	assert(snprintf(path,sizeof(path),"/tmp/sparkpipe-node-count-%ld.json",(long)getpid()) > 0);
	file = fopen(path,"wb");
	assert(file != 0);
	assert(fwrite(base,1u,(size_t)(open - base) + 1u,file) == (size_t)(open - base) + 1u);
	for (index=0u; index<SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT + 1u; index++)
		assert(fputs(index == 0u ? "{}" : ",{}",file) != EOF);
	assert(fputs(close,file) != EOF);
	assert(fclose(file) == 0);
	free(base);
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad(path,&deployment) == SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkModelResidentDeploymentDestroy(&deployment);
	assert(unlink(path) == 0);
	printf("test_model_resident_deployment: over-count arrays OK (9 control markers and %u nodes fail cleanly instead of freeing past the arrays)\n",
		SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT + 1u);
}

static void TestChatTemplates(void)
{
	static const char user_hi[] = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
	static const char system_history[] =
		"{\"messages\":[{\"role\":\"system\",\"content\":\"Be terse.\"},"
		"{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"ok\"},"
		"{\"role\":\"user\",\"content\":\"bye\"}]}";
	static const char minimal[] =
		"{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
		"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,"
		"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"U\"]}";
	SparkModelResidentDeployment deployment;
	SparkTokenizerSpecialToken special[4];
	uint32_t stop_ids[SPARK_MODEL_RESIDENT_CHAT_TEMPLATE_MAX_STOP_MARKERS];
	uint32_t stop_count = 0u;
	const char *unresolved = 0;
	char *glm = TestReadText("model-families/glm5_next/chat_template.json");
	char *gemma = TestReadText("model-families/gemma4/chat_template.json");
	char *glm52 = TestReadText("model-families/glm52/chat_template.json");

	assert(TestLoadWithChatTemplate(0,&deployment) == SPARK_STATUS_OK);
	assert(deployment.chat_template.declared == 0u);
	TestRender(&deployment,user_hi,false,SPARK_CHAT_TEMPLATE_MISSING,0);
	TestRender(&deployment,"{\"prompt\":\"hi\"}",false,SPARK_CHAT_TEMPLATE_NOT_CHAT,0);
	SparkModelResidentDeploymentDestroy(&deployment);

	assert(TestLoadWithChatTemplate(glm,&deployment) == SPARK_STATUS_OK);
	assert(deployment.chat_template.declared == 1u && deployment.chat_template.stop_marker_count == 3u);
	TestRender(&deployment,user_hi,false,SPARK_CHAT_TEMPLATE_RENDERED,
		"[gMASK]<sop><|user|>\nhi<|assistant|>\n<think></think>\n");
	TestRender(&deployment,user_hi,true,SPARK_CHAT_TEMPLATE_RENDERED,
		"[gMASK]<sop><|user|>\nhi<|assistant|>\n<think>");
	TestRender(&deployment,system_history,true,SPARK_CHAT_TEMPLATE_RENDERED,
		"[gMASK]<sop><|system|>\nBe terse.<|user|>\nhi<|assistant|>\n<think>ok<|user|>\nbye<|assistant|>\n<think>");
	TestRender(&deployment,"{\"messages\":[{\"role\":\"observation\",\"content\":\"42\"}]}",false,
		SPARK_CHAT_TEMPLATE_RENDERED,"[gMASK]<sop><|observation|>\n42<|assistant|>\n<think></think>\n");
	TestRender(&deployment,"{\"messages\":[{\"role\":\"tool\",\"content\":\"42\"}]}",false,
		SPARK_CHAT_TEMPLATE_ROLE_UNSUPPORTED,0);
	TestRender(&deployment,"{\"messages\":[{\"role\":\"user\"}]}",false,SPARK_CHAT_TEMPLATE_INVALID_MESSAGES,0);
	TestRender(&deployment,"{\"messages\":[\"hi\"]}",false,SPARK_CHAT_TEMPLATE_INVALID_MESSAGES,0);
	TestRender(&deployment,"{\"messages\":{}}",false,SPARK_CHAT_TEMPLATE_INVALID_MESSAGES,0);
	TestRender(&deployment,"{\"messages\":[]}",false,SPARK_CHAT_TEMPLATE_INVALID_MESSAGES,0);
	memset(special,0,sizeof(special));
	special[0].text = (char *)"<|user|>"; special[0].text_bytes = 8u; special[0].token_id = 11u; special[0].is_special = 1u;
	special[1].text = (char *)"<|assistant|>"; special[1].text_bytes = 13u; special[1].token_id = 13u; special[1].is_special = 1u;
	special[2].text = (char *)"<|observation|>"; special[2].text_bytes = 15u; special[2].token_id = 17u; special[2].is_special = 1u;
	special[3].text = (char *)"<think>"; special[3].text_bytes = 7u; special[3].token_id = 19u; special[3].is_special = 0u;
	assert(SparkChatTemplateResolveStops(&deployment.chat_template,special,4u,stop_ids,&stop_count,&unresolved) == SPARK_STATUS_OK);
	assert(stop_count == 3u && stop_ids[0] == 11u && stop_ids[1] == 17u && stop_ids[2] == 13u && unresolved == 0);
	assert(SparkChatTemplateResolveStops(&deployment.chat_template,special,2u,stop_ids,&stop_count,&unresolved) == SPARK_STATUS_SCHEMA_ERROR);
	assert(unresolved != 0 && strcmp(unresolved,"<|observation|>") == 0 && stop_count == 0u);
	special[2].text = (char *)"<|user|>"; special[2].text_bytes = 8u;
	assert(SparkChatTemplateResolveStops(&deployment.chat_template,special,4u,stop_ids,&stop_count,&unresolved) == SPARK_STATUS_SCHEMA_ERROR);
	assert(unresolved != 0 && strcmp(unresolved,"<|user|>") == 0);
	SparkModelResidentDeploymentDestroy(&deployment);

	assert(TestLoadWithChatTemplate(gemma,&deployment) == SPARK_STATUS_OK);
	assert(deployment.chat_template.declared == 1u && deployment.chat_template.stop_marker_count == 2u);
	TestRender(&deployment,
		"{\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}]}",
		false,SPARK_CHAT_TEMPLATE_RENDERED,
		"<bos><|turn>user\nWhat is the capital of France? Answer in one word.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
	TestRender(&deployment,user_hi,true,SPARK_CHAT_TEMPLATE_RENDERED,
		"<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n");
	TestRender(&deployment,system_history,true,SPARK_CHAT_TEMPLATE_RENDERED,
		"<bos><|turn>system\n<|think|>\nBe terse.<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\nok<turn|>\n<|turn>user\nbye<turn|>\n<|turn>model\n");
	TestRender(&deployment,system_history,false,SPARK_CHAT_TEMPLATE_RENDERED,
		"<bos><|turn>system\nBe terse.<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\nok<turn|>\n<|turn>user\nbye<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
	TestRender(&deployment,"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"system\",\"content\":\"s\"}]}",
		true,SPARK_CHAT_TEMPLATE_RENDERED,
		"<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhi<turn|>\n<|turn>system\ns<turn|>\n<|turn>model\n");
	TestRender(&deployment,"{\"messages\":[{\"role\":\"observation\",\"content\":\"42\"}]}",false,
		SPARK_CHAT_TEMPLATE_ROLE_UNSUPPORTED,0);
	SparkModelResidentDeploymentDestroy(&deployment);

	{
		char *ling = TestReadText("model-families/ling/chat_template.json");
		char *hy4 = TestReadText("model-families/hy4/chat_template.json");
		char *muse = TestReadText("model-families/muse_glimmer/chat_template.json");
		assert(TestLoadWithChatTemplate(ling,&deployment) == SPARK_STATUS_OK);
		TestRender(&deployment,user_hi,false,SPARK_CHAT_TEMPLATE_RENDERED,
			"<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>hi<|role_end|><role>ASSISTANT</role>\n<think></think>");
		TestRender(&deployment,user_hi,true,SPARK_CHAT_TEMPLATE_RENDERED,
			"<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>hi<|role_end|><role>ASSISTANT</role>\n<think>");
		TestRender(&deployment,system_history,true,SPARK_CHAT_TEMPLATE_RENDERED,
			"<role>SYSTEM</role>Be terse.\ndetailed thinking on<|role_end|><role>HUMAN</role>hi<|role_end|>"
			"<role>ASSISTANT</role>\n<think></think>ok<|role_end|><role>HUMAN</role>bye<|role_end|><role>ASSISTANT</role>\n<think>");
		TestRender(&deployment,system_history,false,SPARK_CHAT_TEMPLATE_RENDERED,
			"<role>SYSTEM</role>Be terse.\ndetailed thinking off<|role_end|><role>HUMAN</role>hi<|role_end|>"
			"<role>ASSISTANT</role>\n<think></think>ok<|role_end|><role>HUMAN</role>bye<|role_end|><role>ASSISTANT</role>\n<think></think>");
		SparkModelResidentDeploymentDestroy(&deployment);
		assert(TestLoadWithChatTemplate(hy4,&deployment) == SPARK_STATUS_OK);
		TestRender(&deployment,user_hi,false,SPARK_CHAT_TEMPLATE_RENDERED,
			"<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>system<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>"
			"<\xef\xbd\x9creasoning_mode:opensource\xef\xbd\x9c>reasoning_effort:no_think<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"
			"<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>user<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>hi<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"
			"<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>assistant<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c><think:opensource></think:opensource>");
		TestRender(&deployment,system_history,true,SPARK_CHAT_TEMPLATE_RENDERED,
			"<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>system<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>Be terse."
			"<\xef\xbd\x9creasoning_mode:opensource\xef\xbd\x9c>reasoning_effort:high<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"
			"<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>user<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>hi<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"
			"<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>assistant<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c><think:opensource></think:opensource>ok<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"
			"<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>user<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>bye<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"
			"<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>assistant<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c><think:opensource>");
		SparkModelResidentDeploymentDestroy(&deployment);
		assert(TestLoadWithChatTemplate(muse,&deployment) == SPARK_STATUS_OK);
		TestRender(&deployment,user_hi,true,SPARK_CHAT_TEMPLATE_RENDERED,
			"<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\n\n"
			"Reasoning strength: high.\n\n# Valid recipients: \"self\", \"user\".<|eot|><|start|>user<|message|>hi<|eot|><|start|>assistant");
		TestRender(&deployment,system_history,false,SPARK_CHAT_TEMPLATE_RENDERED,
			"<|begin_of_text|><|start|>system<|message|>Be terse.\n\nReasoning strength: low.\n\n# Valid recipients: \"self\", \"user\".<|eot|>"
			"<|start|>user<|message|>hi<|eot|><|start|>assistant to=user<|message|>ok<|eot|><|start|>user<|message|>bye<|eot|><|start|>assistant");
		SparkModelResidentDeploymentDestroy(&deployment);
		free(ling);
		free(hy4);
		free(muse);
	}

	{
		char *mimo = TestReadText("model-families/mimo26/chat_template.json");
		assert(TestLoadWithChatTemplate(mimo,&deployment) == SPARK_STATUS_OK);
		TestRender(&deployment,user_hi,false,SPARK_CHAT_TEMPLATE_RENDERED,
			"<|im_start|>user\nhi<|im_end|><|im_start|>assistant\n<think></think>");
		TestRender(&deployment,user_hi,true,SPARK_CHAT_TEMPLATE_RENDERED,
			"<|im_start|>user\nhi<|im_end|><|im_start|>assistant\n");
		TestRender(&deployment,system_history,false,SPARK_CHAT_TEMPLATE_RENDERED,
			"<|im_start|>system\nBe terse.<|im_end|><|im_start|>user\nhi<|im_end|><|im_start|>assistant\n<think></think>ok<|im_end|>"
			"<|im_start|>user\nbye<|im_end|><|im_start|>assistant\n<think></think>");
		TestRender(&deployment,system_history,true,SPARK_CHAT_TEMPLATE_RENDERED,
			"<|im_start|>system\nBe terse.<|im_end|><|im_start|>user\nhi<|im_end|><|im_start|>assistant\n<think></think>ok<|im_end|>"
			"<|im_start|>user\nbye<|im_end|><|im_start|>assistant\n");
		SparkModelResidentDeploymentDestroy(&deployment);
		free(mimo);
	}

	{
		char *k3 = TestReadText("model-families/k3/chat_template.json");
		static const char suffixed_null[] =
			"{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
			"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,"
			"\"assistant_suffix\":null,\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"U\"]}";
		assert(TestLoadWithChatTemplate(k3,&deployment) == SPARK_STATUS_OK);
		assert(deployment.chat_template.declared == 1u && deployment.chat_template.stop_marker_count == 2u);
		TestRender(&deployment,
			"{\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}]}",
			false,SPARK_CHAT_TEMPLATE_RENDERED,
			"<|open|>message role=\"user\"<|sep|>What is the capital of France?<|close|>message<|sep|><|end_of_msg|>"
			"<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>");
		TestRender(&deployment,system_history,false,SPARK_CHAT_TEMPLATE_RENDERED,
			"<|open|>message role=\"system\"<|sep|>Be terse.<|close|>message<|sep|><|end_of_msg|>"
			"<|open|>message role=\"user\"<|sep|>hi<|close|>message<|sep|><|end_of_msg|>"
			"<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>ok<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
			"<|open|>message role=\"user\"<|sep|>bye<|close|>message<|sep|><|end_of_msg|>"
			"<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>");
		TestRender(&deployment,user_hi,true,SPARK_CHAT_TEMPLATE_THINKING_UNSUPPORTED,0);
		memset(special,0,sizeof(special));
		special[0].text = (char *)"<|end_of_msg|>"; special[0].text_bytes = 14u; special[0].token_id = 163586u; special[0].is_special = 1u;
		special[1].text = (char *)"<|close|>"; special[1].text_bytes = 9u; special[1].token_id = 163588u; special[1].is_special = 0u;
		assert(SparkChatTemplateResolveStops(&deployment.chat_template,special,2u,stop_ids,&stop_count,&unresolved) == SPARK_STATUS_OK);
		assert(stop_count == 2u && stop_ids[0] == 163588u && stop_ids[1] == 163586u && unresolved == 0);
		assert(SparkChatTemplateResolveStops(&deployment.chat_template,special,1u,stop_ids,&stop_count,&unresolved) == SPARK_STATUS_SCHEMA_ERROR);
		assert(unresolved != 0 && strcmp(unresolved,"<|close|>") == 0);
		deployment.chat_template.control_marker_count = 0u;
		assert(SparkChatTemplateResolveStops(&deployment.chat_template,special,2u,stop_ids,&stop_count,&unresolved) == SPARK_STATUS_SCHEMA_ERROR);
		assert(unresolved != 0 && strcmp(unresolved,"<|close|>") == 0);
		deployment.chat_template.control_marker_count = 3u;
		TestRender(&deployment,"{\"messages\":[{\"role\":\"observation\",\"content\":\"42\"}]}",false,
			SPARK_CHAT_TEMPLATE_ROLE_UNSUPPORTED,0);
		SparkModelResidentDeploymentDestroy(&deployment);
		TestChatTemplateSchema(suffixed_null,SPARK_STATUS_SCHEMA_ERROR);
		free(k3);
	}

	{
		char *laguna = TestReadText("model-families/laguna/chat_template.json");
		static const char eos[] = "\xe3\x80\x88|EOS|\xe3\x80\x89";
		static const char header[] = "\xe3\x80\x88|EOS|\xe3\x80\x89<system>You are a helpful, conversationally-fluent assistant made by Poolside. "
			"You are here to be helpful to users through natural language conversations.</system>\n";
		char expected[1024];
		assert(TestLoadWithChatTemplate(laguna,&deployment) == SPARK_STATUS_OK);
		assert(deployment.chat_template.declared == 1u && deployment.chat_template.stop_marker_count == 2u);
		snprintf(expected,sizeof(expected),"%s<user>hi</user>\n<assistant></think>",header);
		TestRender(&deployment,user_hi,false,SPARK_CHAT_TEMPLATE_RENDERED,expected);
		snprintf(expected,sizeof(expected),"%s<user>hi</user>\n<assistant><think>",header);
		TestRender(&deployment,user_hi,true,SPARK_CHAT_TEMPLATE_RENDERED,expected);
		snprintf(expected,sizeof(expected),"%s<system>Be terse.</system>\n<user>hi</user>\n<assistant></think>ok</assistant>\n<user>bye</user>\n<assistant></think>",eos);
		TestRender(&deployment,system_history,false,SPARK_CHAT_TEMPLATE_RENDERED,expected);
		snprintf(expected,sizeof(expected),"%s<system>Be terse.</system>\n<user>hi</user>\n<assistant><think></think>ok</assistant>\n<user>bye</user>\n<assistant><think>",eos);
		TestRender(&deployment,system_history,true,SPARK_CHAT_TEMPLATE_RENDERED,expected);
		snprintf(expected,sizeof(expected),"%s<user>hi</user>\n<tool_response>42</tool_response>\n<assistant></think>",header);
		TestRender(&deployment,"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"observation\",\"content\":\"42\"}]}",false,
			SPARK_CHAT_TEMPLATE_RENDERED,expected);
		snprintf(expected,sizeof(expected),"%s<user>hi</user>\n<system>s</system>\n<assistant><think>",header);
		TestRender(&deployment,"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"system\",\"content\":\"s\"}]}",true,
			SPARK_CHAT_TEMPLATE_RENDERED,expected);
		memset(special,0,sizeof(special));
		special[0].text = (char *)eos; special[0].text_bytes = (uint32_t)strlen(eos); special[0].token_id = 2u; special[0].is_special = 1u;
		special[1].text = (char *)"</assistant>"; special[1].text_bytes = 12u; special[1].token_id = 24u; special[1].is_special = 1u;
		special[2].text = (char *)"<assistant>"; special[2].text_bytes = 11u; special[2].token_id = 23u; special[2].is_special = 0u;
		assert(SparkChatTemplateResolveStops(&deployment.chat_template,special,3u,stop_ids,&stop_count,&unresolved) == SPARK_STATUS_OK);
		assert(stop_count == 2u && stop_ids[0] == 24u && stop_ids[1] == 2u && unresolved == 0);
		SparkModelResidentDeploymentDestroy(&deployment);
		TestChatTemplateSchema("{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
			"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,\"system_turn_suffix\":\"S\","
			"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"U\"]}",SPARK_STATUS_SCHEMA_ERROR);
		TestChatTemplateSchema("{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
			"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,\"observation_suffix\":\"O\","
			"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"U\"]}",SPARK_STATUS_SCHEMA_ERROR);
		TestChatTemplateSchema("{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
			"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,\"user_suffix\":null,"
			"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"U\"]}",SPARK_STATUS_SCHEMA_ERROR);
		free(laguna);
	}

	assert(TestLoadWithChatTemplate(glm52,&deployment) == SPARK_STATUS_OK);
	assert(deployment.chat_template.declared == 1u && deployment.chat_template.stop_marker_count == 3u);
	TestRender(&deployment,user_hi,false,SPARK_CHAT_TEMPLATE_RENDERED,
		"[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think></think>");
	TestRender(&deployment,user_hi,true,SPARK_CHAT_TEMPLATE_RENDERED,
		"[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>");
	TestRender(&deployment,system_history,true,SPARK_CHAT_TEMPLATE_RENDERED,
		"[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>Be terse.<|user|>hi<|assistant|><think></think>ok<|user|>bye<|assistant|><think>");
	TestRender(&deployment,system_history,false,SPARK_CHAT_TEMPLATE_RENDERED,
		"[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>Be terse.<|user|>hi<|assistant|><think></think>ok<|user|>bye<|assistant|><think></think>");
	TestRender(&deployment,"{\"messages\":[{\"role\":\"observation\",\"content\":\"42\"}]}",false,
		SPARK_CHAT_TEMPLATE_ROLE_UNSUPPORTED,0);
	SparkModelResidentDeploymentDestroy(&deployment);

	assert(TestLoadWithChatTemplate(minimal,&deployment) == SPARK_STATUS_OK);
	TestRender(&deployment,user_hi,false,SPARK_CHAT_TEMPLATE_RENDERED,"UhiG");
	TestRender(&deployment,user_hi,true,SPARK_CHAT_TEMPLATE_THINKING_UNSUPPORTED,0);
	TestRender(&deployment,"{\"messages\":[{\"role\":\"system\",\"content\":\"s\"}]}",false,
		SPARK_CHAT_TEMPLATE_ROLE_UNSUPPORTED,0);
	SparkModelResidentDeploymentDestroy(&deployment);

	TestChatTemplateSchema("{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
		"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,"
		"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null}",SPARK_STATUS_SCHEMA_ERROR);
	TestChatTemplateSchema("{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
		"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,"
		"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[]}",SPARK_STATUS_SCHEMA_ERROR);
	TestChatTemplateSchema("{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
		"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,"
		"\"turn_suffix\":\"\",\"generation\":\"\",\"generation_thinking\":null,\"stop_markers\":[\"U\"]}",SPARK_STATUS_SCHEMA_ERROR);
	TestChatTemplateSchema("{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":\"S\","
		"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,"
		"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"U\"]}",SPARK_STATUS_SCHEMA_ERROR);
	TestChatTemplateSchema("{\"prefix\":\"\",\"thinking_prefix\":\"T\",\"system\":null,\"system_thinking\":null,"
		"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,"
		"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"U\"]}",SPARK_STATUS_SCHEMA_ERROR);
	TestChatTemplateSchema("{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
		"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":\"AT\","
		"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"U\"]}",SPARK_STATUS_SCHEMA_ERROR);
	TestChatTemplateSchema("{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
		"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,"
		"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"U\"],\"tools\":\"\"}",SPARK_STATUS_SCHEMA_ERROR);
	TestChatTemplateSchema("{\"prefix\":null,\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
		"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,"
		"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"U\"]}",SPARK_STATUS_SCHEMA_ERROR);
	TestChatTemplateSchema("{\"prefix\":\"\",\"thinking_prefix\":\"\",\"system\":null,\"system_thinking\":null,"
		"\"user\":\"U\",\"observation\":null,\"assistant\":\"A\",\"assistant_thinking\":null,"
		"\"turn_suffix\":\"\",\"generation\":\"G\",\"generation_thinking\":null,\"stop_markers\":[\"\"]}",SPARK_STATUS_SCHEMA_ERROR);
	TestChatTemplateSchema("\"[gMASK]\"",SPARK_STATUS_SCHEMA_ERROR);
	free(glm);
	free(gemma);
	free(glm52);
	printf("test_model_resident_deployment: chat_template declarations OK (absent is refused, GLM, GLM-5.2, Gemma and K3 families render their publisher shapes (K3 closes the assistant response with its declared assistant_suffix, Laguna closes every role with its own suffix), undeclared roles and thinking are refused, stop markers resolve to exactly one special token or one declared control marker)\n");
}

int main(int argc,char **argv)
{
	SparkModelResidentDeployment deployment;
	SparkModelServingAdapterDescriptor descriptor;
	const SparkModelResidentDeploymentNode *node;
	char path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	if ( argc == 2 )
	{
		SparkStatus status;
		SparkModelResidentDeploymentReset(&deployment);
		status = SparkModelResidentDeploymentLoad(argv[1],&deployment);
		if ( status == SPARK_STATUS_OK )
		{
			if ( deployment.tokenizer_asset_path == 0 )
				puts("none");
			else
				printf("%s\n%u\n%s\n",deployment.tokenizer_asset_path,
				    deployment.tokenizer_vocabulary_size,deployment.tokenizer_asset_sha256);
			if ( deployment.chat_template.declared != 0u )
				printf("chat_template stop_markers=%u\n",deployment.chat_template.stop_marker_count);
		}
		SparkModelResidentDeploymentDestroy(&deployment);
		return(status == SPARK_STATUS_OK ? 0 : 1);
	}
	assert(argc == 1);
	TestEosMetadata("\"eos_token_ids\":[0,154820],",SPARK_STATUS_OK);
	TestEosMetadata("\"eos_token_ids\":[],",SPARK_STATUS_SCHEMA_ERROR);
	TestEosMetadata("\"eos_token_ids\":[1,1],",SPARK_STATUS_SCHEMA_ERROR);
	TestEosMetadata("\"eos_token_ids\":[1],\"eos_token_ids\":[2],",SPARK_STATUS_SCHEMA_ERROR);
	TestEosMetadata("\"eos_token_ids\":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],",SPARK_STATUS_SCHEMA_ERROR);
	TestOverCountArraysFailCleanly();
	TestPrefixReuseKey("",SPARK_STATUS_OK);
	TestPrefixReuseKey("\"prefix_reuse\":true,",SPARK_STATUS_SCHEMA_ERROR);
	TestPrefixReuseKey("\"prefix_reuse\":false,",SPARK_STATUS_SCHEMA_ERROR);
	TestSequencePositions();
	TestSharedInputRows();
	TestDraftServer();
	TestGlm5NextDeploymentServesText();
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment.json",&deployment) == SPARK_STATUS_OK);
	assert(deployment.node_count == 3u);
	assert(deployment.runtime_limits.resident_sequence_capacity == 8u);
	assert(strcmp(deployment.driver_program_name,"resident_decode") == 0);
	node = SparkModelResidentDeploymentFindRank(&deployment,1u);
	assert(node != 0);
	assert(node->stage_index == 2u);
	assert(strcmp(node->transport_host,"spark1") == 0);
	assert(strcmp(node->kv_backing_directory,"/tmp/test-runtime-1/kvcache") == 0);
	assert(node->kv_backing_maximum_bytes == 0u);
	assert(node->control_endpoint.kind == SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX);
	assert(strcmp(node->control_endpoint.unix_socket_path,"/tmp/test-model-resident-1.sock") == 0);
	assert(SparkModelResidentDeploymentResolvePath(node,deployment.driver_shared_object_path,path,sizeof(path)) == SPARK_STATUS_OK);
	assert(strcmp(path,"/tmp/test-runtime-1/build/test_modules/libdsv4_serving_driver_module.dylib") == 0);
	assert(SparkModelResidentDeploymentResolvePath(node,"../driver.so",path,sizeof(path)) == SPARK_STATUS_INVALID_ARGUMENT);
	node = SparkModelResidentDeploymentFindStage(&deployment,1u);
	assert(node != 0);
	assert(node->rank_index == 2u);
	TestBuildDescriptor(&descriptor);
	assert(SparkModelResidentDeploymentValidateForAdapter(&deployment,&descriptor) == SPARK_STATUS_OK);
	descriptor.driver_program_name = "other";
	assert(SparkModelResidentDeploymentValidateForAdapter(&deployment,&descriptor) == SPARK_STATUS_TARGET_MISMATCH);
	SparkModelResidentDeploymentDestroy(&deployment);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_unknown.json",&deployment) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_no_backing.json",&deployment) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_duplicate_rank.json",&deployment) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_tokenizer.json",&deployment) == SPARK_STATUS_OK);
	assert(deployment.tokenizer_asset_path != 0);
	assert(deployment.tokenizer_vocabulary_size == 129280u);
	assert(deployment.tokenizer_asset_sha256 != 0);
	assert(strlen(deployment.tokenizer_asset_sha256) == 64u);
	assert(deployment.tokenizer_text_prompt_prefix_count == 0u);
	SparkModelResidentDeploymentDestroy(&deployment);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_tokenizer_prefix.json",&deployment) == SPARK_STATUS_OK);
	assert(deployment.tokenizer_text_prompt_prefix_count == 1u && deployment.tokenizer_text_prompt_prefix_ids[0] == 0u);
	SparkModelResidentDeploymentDestroy(&deployment);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_tokenizer_prefix_empty.json",&deployment) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_tokenizer_prefix_out_of_range.json",&deployment) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_tokenizer_prefix_too_long.json",&deployment) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_tokenizer_path_only.json",&deployment) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_tokenizer_bad_sha.json",&deployment) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_tokenizer_zero_vocab.json",&deployment) == SPARK_STATUS_SCHEMA_ERROR);
	TestChatTemplates();
	return(0);
}

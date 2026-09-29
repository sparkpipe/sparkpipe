#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_quant_arm.h"

typedef struct TestArmFields
{
	const char *arm_id;
	const char *model;
	uint32_t tp;
	uint32_t pp;
	uint32_t kv_shard;
	const char *frame;
	uint32_t spine_count;
	char spine_fill;
	const char *expert_codec;
	const char *expert_label;
	const char *producer;
	const char *recipe;
	const char *latent;
	const char *index;
	const char *state;
	uint32_t group;
	const char *mode;
	const char *drafter_kind;
	const char *drafter_label;
	const char *drafter_codec;
	const char *drafter_head;
	uint32_t sidecar_count;
	uint32_t pack_count;
	const char *extra_root_member;
} TestArmFields;

static void TestHex(char fill,uint32_t rank,char output[65])
{
	uint32_t index;
	for (index=0u; index<64u; index++)
		output[index] = fill;
	output[62] = "0123456789abcdef"[(rank >> 4) & 15u];
	output[63] = "0123456789abcdef"[rank & 15u];
	output[64] = '\0';
}

static void TestHexArray(char *buffer,size_t buffer_bytes,char fill,uint32_t count)
{
	char hex[65];
	uint32_t index;
	size_t length = 0u;
	length += (size_t)snprintf(buffer + length,buffer_bytes - length,"[");
	for (index=0u; index<count; index++)
	{
		TestHex(fill,index,hex);
		length += (size_t)snprintf(buffer + length,buffer_bytes - length,"%s\"%s\"",index == 0u ? "" : ", ",hex);
	}
	(void)snprintf(buffer + length,buffer_bytes - length,"]");
}

static TestArmFields TestDefaultFields(void)
{
	TestArmFields fields;
	memset(&fields,0,sizeof(fields));
	fields.arm_id = "demo.S1.e-nvfp4nv.k-bf16/bf16/fp32.d-none";
	fields.model = "demo";
	fields.tp = 4u;
	fields.pp = 1u;
	fields.kv_shard = 1u;
	fields.frame = "S1";
	fields.spine_count = 4u;
	fields.spine_fill = 'a';
	fields.expert_codec = "nvfp4";
	fields.expert_label = "nvfp4nv";
	fields.producer = "community";
	fields.recipe = "null";
	fields.latent = "bf16";
	fields.index = "bf16";
	fields.state = "fp32";
	fields.group = 0u;
	fields.mode = "store";
	fields.drafter_kind = "none";
	fields.drafter_label = "none";
	fields.drafter_codec = "none";
	fields.drafter_head = "none";
	fields.sidecar_count = 0u;
	fields.pack_count = 4u;
	fields.extra_root_member = "";
	return(fields);
}

static char *TestRender(const TestArmFields *fields)
{
	char *text = (char *)malloc(65536u);
	char spine[16384];
	char sidecar[16384];
	char pack[16384];
	char artifact_a[65],artifact_b[65],artifact_c[65];
	assert(text != 0);
	TestHexArray(spine,sizeof(spine),fields->spine_fill,fields->spine_count);
	TestHexArray(sidecar,sizeof(sidecar),'c',fields->sidecar_count);
	TestHexArray(pack,sizeof(pack),'b',fields->pack_count);
	TestHex('d',1u,artifact_a);
	TestHex('e',2u,artifact_b);
	TestHex('f',3u,artifact_c);
	(void)snprintf(text,65536u,
		"{\n  \"format\": \"sparkpipe-quant-arm-v1\",\n  \"arm_id\": \"%s\",\n  \"model\": \"%s\",\n  \"revision\": \"org/Demo-BF16@f12e0fe1\",\n"
		"  \"topology\": {\"tp\": %u, \"pp\": %u, \"kv_shard\": %u},\n"
		"  \"spine\": {\"frame\": \"%s\", \"source\": \"org/Demo-BF16@f12e0fe1\", \"spine_digest\": %s},\n"
		"  \"expert\": {\"codec\": \"%s\", \"label\": \"%s\", \"producer\": \"%s\", \"source\": \"org/Demo-NVFP4@0badc0de\", \"recipe_sha256\": %s},\n"
		"  \"kv\": {\"latent\": \"%s\", \"index\": \"%s\", \"state\": \"%s\", \"group\": %u, \"mode\": \"%s\"},\n"
		"  \"drafter\": {\"kind\": \"%s\", \"label\": \"%s\", \"codec\": \"%s\", \"head\": \"%s\", \"sidecar_sha256\": %s},\n"
		"  \"pack_sha256\": %s,\n"
		"  \"artifacts\": {\"module_archive_sha256\": \"%s\", \"driver_sha256\": \"%s\", \"adapter_sha256\": \"%s\"}%s\n}\n",
		fields->arm_id,fields->model,fields->tp,fields->pp,fields->kv_shard,fields->frame,spine,
		fields->expert_codec,fields->expert_label,fields->producer,fields->recipe,
		fields->latent,fields->index,fields->state,fields->group,fields->mode,
		fields->drafter_kind,fields->drafter_label,fields->drafter_codec,fields->drafter_head,sidecar,
		pack,artifact_a,artifact_b,artifact_c,fields->extra_root_member);
	return(text);
}

static SparkStatus TestParse(const TestArmFields *fields,SparkQuantArm *arm,char *error,uint32_t error_bytes)
{
	char *text = TestRender(fields);
	SparkStatus status = SparkQuantArmParseText(text,strlen(text),arm,error,error_bytes);
	free(text);
	return(status);
}

static void TestRefused(const TestArmFields *fields,const char *needle)
{
	SparkQuantArm arm;
	char error[512];
	SparkStatus status = TestParse(fields,&arm,error,(uint32_t)sizeof(error));
	if ( status == SPARK_STATUS_OK || strstr(error,needle) == 0 )
	{
		fprintf(stderr,"expected refusal containing '%s', got status=%d error='%s'\n",needle,(int)status,error);
		abort();
	}
	assert(arm.arm_digest[0] == '\0');
}

static void TestCanonicalKnownAnswer(void)
{
	TestArmFields fields = TestDefaultFields();
	SparkQuantArm arm;
	char error[512];
	char canonical[SPARK_QUANT_ARM_CANONICAL_BYTES];
	char digest[SPARK_SHA256_HEX_BYTES];
	char spine[16384],pack[16384],expected[65536];
	char artifact_a[65],artifact_b[65],artifact_c[65];
	uint32_t bytes = 0u;
	fields.tp = 2u;
	fields.spine_count = 2u;
	fields.pack_count = 2u;
	assert(TestParse(&fields,&arm,error,(uint32_t)sizeof(error)) == SPARK_STATUS_OK);
	assert(SparkQuantArmCanonicalize(&arm,canonical,(uint32_t)sizeof(canonical),&bytes) == SPARK_STATUS_OK);
	TestHexArray(spine,sizeof(spine),'a',2u);
	TestHexArray(pack,sizeof(pack),'b',2u);
	{
		char *cursor;
		for (cursor=spine; (cursor=strstr(cursor,", "))!=0; )
			memmove(cursor + 1,cursor + 2,strlen(cursor + 2) + 1u);
		for (cursor=pack; (cursor=strstr(cursor,", "))!=0; )
			memmove(cursor + 1,cursor + 2,strlen(cursor + 2) + 1u);
	}
	TestHex('d',1u,artifact_a);
	TestHex('e',2u,artifact_b);
	TestHex('f',3u,artifact_c);
	(void)snprintf(expected,sizeof(expected),
		"{\"arm_id\":\"demo.S1.e-nvfp4nv.k-bf16/bf16/fp32.d-none\",\"artifacts\":{\"adapter_sha256\":\"%s\",\"driver_sha256\":\"%s\",\"module_archive_sha256\":\"%s\"},"
		"\"drafter\":{\"codec\":\"none\",\"head\":\"none\",\"kind\":\"none\",\"label\":\"none\",\"sidecar_sha256\":[]},"
		"\"expert\":{\"codec\":\"nvfp4\",\"label\":\"nvfp4nv\",\"producer\":\"community\",\"recipe_sha256\":null,\"source\":\"org/Demo-NVFP4@0badc0de\"},"
		"\"format\":\"sparkpipe-quant-arm-v1\",\"kv\":{\"group\":0,\"index\":\"bf16\",\"latent\":\"bf16\",\"mode\":\"store\",\"state\":\"fp32\"},"
		"\"model\":\"demo\",\"pack_sha256\":%s,\"revision\":\"org/Demo-BF16@f12e0fe1\","
		"\"spine\":{\"frame\":\"S1\",\"source\":\"org/Demo-BF16@f12e0fe1\",\"spine_digest\":%s},\"topology\":{\"kv_shard\":1,\"pp\":1,\"tp\":2}}",
		artifact_c,artifact_b,artifact_a,pack,spine);
	if ( strcmp(canonical,expected) != 0 )
	{
		fprintf(stderr,"canonical mismatch\n got %s\nwant %s\n",canonical,expected);
		abort();
	}
	assert(bytes == strlen(expected));
	assert(SparkSha256Bytes(expected,strlen(expected),digest) == SPARK_STATUS_OK);
	assert(strcmp(digest,arm.arm_digest) == 0);
	assert(SparkSha256Bytes(pack,strlen(pack),digest) == SPARK_STATUS_OK);
	assert(strcmp(digest,arm.pack_set_sha256) == 0);
	assert(strcmp(arm.kv_text,"bf16/bf16/fp32/0/store") == 0);
}

static void TestDigestSensitivity(void)
{
	TestArmFields fields = TestDefaultFields();
	SparkQuantArm first,second;
	char error[512];
	char *text;
	char *reordered;
	assert(TestParse(&fields,&first,error,(uint32_t)sizeof(error)) == SPARK_STATUS_OK);
	text = TestRender(&fields);
	reordered = (char *)malloc(strlen(text) + 64u);
	assert(reordered != 0);
	{
		const char *model = strstr(text,"  \"model\"");
		const char *revision = strstr(text,"  \"revision\"");
		size_t head = (size_t)(model - text);
		size_t model_bytes = (size_t)(revision - model);
		memcpy(reordered,text,head);
		memcpy(reordered + head,revision,strlen(revision) + 1u);
		{
			char *tail = strstr(reordered,"  \"artifacts\"");
			char saved[4096];
			(void)snprintf(saved,sizeof(saved),"%s",tail);
			memcpy(tail,model,model_bytes);
			memcpy(tail + model_bytes,saved,strlen(saved) + 1u);
		}
	}
	assert(SparkQuantArmParseText(reordered,strlen(reordered),&second,error,(uint32_t)sizeof(error)) == SPARK_STATUS_OK);
	assert(strcmp(first.arm_digest,second.arm_digest) == 0);
	free(reordered);
	free(text);
	fields.spine_fill = '9';
	assert(TestParse(&fields,&second,error,(uint32_t)sizeof(error)) == SPARK_STATUS_OK);
	assert(strcmp(first.arm_digest,second.arm_digest) != 0);
	assert(strcmp(first.pack_set_sha256,second.pack_set_sha256) == 0);
	fields = TestDefaultFields();
	fields.arm_id = "demo.S1.e-nvfp4nv.k-fp8g128/bf16/fp32.d-none";
	fields.latent = "fp8";
	fields.group = 128u;
	fields.mode = "sim";
	assert(TestParse(&fields,&second,error,(uint32_t)sizeof(error)) == SPARK_STATUS_OK);
	assert(strcmp(second.kv_text,"fp8/bf16/fp32/128/sim") == 0);
	assert(strcmp(first.arm_digest,second.arm_digest) != 0);
	fields = TestDefaultFields();
	fields.arm_id = "demo.S1.e-nvfp4nv.k-bf16/bf16/fp32.d-mtp8";
	fields.drafter_kind = "mtp";
	fields.drafter_label = "mtp8";
	fields.drafter_codec = "fp8";
	fields.drafter_head = "bf16";
	fields.sidecar_count = 4u;
	assert(TestParse(&fields,&second,error,(uint32_t)sizeof(error)) == SPARK_STATUS_OK);
	assert(second.drafter_sidecar_count == 4u);
}

static void TestRefusals(void)
{
	TestArmFields fields;
	SparkQuantArm arm;
	char error[512];
	fields = TestDefaultFields();
	fields.extra_root_member = ",\n  \"note\": \"x\"";
	TestRefused(&fields,"members are not exactly");
	fields = TestDefaultFields();
	fields.extra_root_member = ",\n  \"model\": \"demo\"";
	TestRefused(&fields,"members are not exactly");
	fields = TestDefaultFields();
	fields.arm_id = "demo.S1.e-fp8.k-bf16/bf16/fp32.d-none";
	TestRefused(&fields,"does not match its fields");
	fields = TestDefaultFields();
	fields.spine_count = 3u;
	TestRefused(&fields,"spine_digest has 3 entries, expected 4");
	fields = TestDefaultFields();
	fields.pack_count = 5u;
	TestRefused(&fields,"pack_sha256 has 5 entries");
	fields = TestDefaultFields();
	fields.spine_fill = 'A';
	TestRefused(&fields,"lowercase hex");
	fields = TestDefaultFields();
	fields.expert_codec = "fp4";
	TestRefused(&fields,"not a known token");
	fields = TestDefaultFields();
	fields.expert_label = "fp8nv";
	TestRefused(&fields,"start with the codec");
	fields = TestDefaultFields();
	fields.producer = "experiment";
	TestRefused(&fields,"recipe_sha256");
	fields = TestDefaultFields();
	fields.group = 64u;
	TestRefused(&fields,"group must be 0");
	fields = TestDefaultFields();
	fields.arm_id = "demo.S1.e-nvfp4nv.k-fp8g48/bf16/fp32.d-none";
	fields.latent = "fp8";
	fields.group = 48u;
	fields.mode = "sim";
	TestRefused(&fields,"must be 32, 64 or 128");
	fields = TestDefaultFields();
	fields.arm_id = "demo.S1.e-nvfp4nv.k-mxfp4g64/bf16/fp32.d-none";
	fields.latent = "mxfp4";
	fields.group = 64u;
	fields.mode = "sim";
	TestRefused(&fields,"mxfp4 latent needs group 32");
	fields = TestDefaultFields();
	fields.mode = "sim";
	TestRefused(&fields,"must use mode store");
	fields = TestDefaultFields();
	fields.drafter_codec = "fp8";
	TestRefused(&fields,"carries no weights");
	fields = TestDefaultFields();
	fields.arm_id = "demo.S1.e-nvfp4nv.k-bf16/bf16/fp32.d-mtp8";
	fields.drafter_kind = "mtp";
	fields.drafter_label = "mtp8";
	fields.drafter_codec = "fp8";
	fields.drafter_head = "bf16";
	fields.sidecar_count = 3u;
	TestRefused(&fields,"one sidecar sha256 per rank");
	fields = TestDefaultFields();
	fields.tp = 0u;
	TestRefused(&fields,"outside 1..64 ranks");
	fields = TestDefaultFields();
	fields.kv_shard = 2u;
	TestRefused(&fields,"kv_shard must be 0 or 1");
	fields = TestDefaultFields();
	fields.frame = "X1";
	fields.arm_id = "demo.X1.e-nvfp4nv.k-bf16/bf16/fp32.d-none";
	TestRefused(&fields,"frame must be S");
	fields = TestDefaultFields();
	fields.model = "Demo";
	fields.arm_id = "Demo.S1.e-nvfp4nv.k-bf16/bf16/fp32.d-none";
	TestRefused(&fields,"model must be lowercase");
	{
		static const char Bad[] = "{\"format\":\"sparkpipe-quant-arm-v0\"}";
		assert(SparkQuantArmParseText(Bad,strlen(Bad),&arm,error,(uint32_t)sizeof(error)) != SPARK_STATUS_OK);
	}
	{
		static const char *const Replacements[] = { "\"tp\": 4.0", "\"tp\": true", "\"tp\": \"4\"", "\"tp\": 1e1" };
		uint32_t replacement;
		for (replacement=0u; replacement<4u; replacement++)
		{
			char *text;
			char *cursor;
			char *edited = (char *)malloc(70000u);
			fields = TestDefaultFields();
			text = TestRender(&fields);
			cursor = strstr(text,"\"tp\": 4");
			assert(cursor != 0 && edited != 0);
			(void)snprintf(edited,70000u,"%.*s%s%s",(int)(cursor - text),text,Replacements[replacement],cursor + strlen("\"tp\": 4"));
			assert(SparkQuantArmParseText(edited,strlen(edited),&arm,error,(uint32_t)sizeof(error)) != SPARK_STATUS_OK);
			assert(strstr(error,"tp must be") != 0 || strstr(error,"not valid JSON") != 0);
			free(edited);
			free(text);
		}
	}
	assert(SparkQuantArmLoadFile("/nonexistent/arm.json",&arm,error,(uint32_t)sizeof(error)) == SPARK_STATUS_IO_ERROR);
}

static void TestLoadFile(const char *directory)
{
	TestArmFields fields = TestDefaultFields();
	SparkQuantArm from_text,from_file;
	char error[512];
	char path[4096];
	char *text = TestRender(&fields);
	FILE *file;
	(void)snprintf(path,sizeof(path),"%s/test_quant_arm.%d.json",directory,(int)getpid());
	file = fopen(path,"wb");
	assert(file != 0);
	assert(fwrite(text,1u,strlen(text),file) == strlen(text));
	assert(fclose(file) == 0);
	assert(SparkQuantArmParseText(text,strlen(text),&from_text,error,(uint32_t)sizeof(error)) == SPARK_STATUS_OK);
	assert(SparkQuantArmLoadFile(path,&from_file,error,(uint32_t)sizeof(error)) == SPARK_STATUS_OK);
	assert(strcmp(from_text.arm_digest,from_file.arm_digest) == 0);
	assert(remove(path) == 0);
	free(text);
}

int main(void)
{
	TestCanonicalKnownAnswer();
	TestDigestSensitivity();
	TestRefusals();
	TestLoadFile("build");
	printf("test_quant_arm: PASS\n");
	return(0);
}

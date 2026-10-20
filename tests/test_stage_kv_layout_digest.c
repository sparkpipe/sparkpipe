#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_weight_codec.h"

#define VARIANT_CAPACITY 32u

static uint8_t DIGESTS[VARIANT_CAPACITY][SPARK_SHA256_DIGEST_BYTES];
static uint32_t DIGEST_COUNT;

static void Base(SparkStageKvLayoutIdentity *identity)
{
	memset(identity,0,sizeof(*identity));
	identity->model_id = "model";
	identity->model_revision = "revision";
	memset(identity->pack_sha256,0x11,sizeof(identity->pack_sha256));
	memset(identity->contract_sha256,0x22,sizeof(identity->contract_sha256));
	memset(identity->driver_sha256,0x33,sizeof(identity->driver_sha256));
	identity->expert_codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	identity->kv_codec = SPARK_WEIGHT_CODEC_BF16;
	identity->context_shard.degree = 16u;
	identity->context_shard.rank = 3u;
	identity->context_shard.grain = 1u;
	identity->block_token_count = 64u;
	identity->region_count = 2u;
	identity->regions[0].layout = SPARK_STAGE_KV_REGION_PAGE_MAJOR;
	identity->regions[0].layer_count = 78u;
	identity->regions[0].layer_page_bytes = 4096u;
	identity->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	identity->regions[1].layer_count = 4u;
	identity->regions[1].layer_page_bytes = 1024u;
	identity->page_bytes = 78u * 4096u + 4u * 1024u;
	identity->state_page_bytes = 0u;
}

static void Record(const SparkStageKvLayoutIdentity *identity)
{
	uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
	const char *invalid = "unset";
	uint32_t index;
	assert(SparkStageKvLayoutDigest(identity,digest,&invalid) == SPARK_STATUS_OK && invalid == 0);
	for (index=0u; index<DIGEST_COUNT; index++)
		assert(memcmp(DIGESTS[index],digest,sizeof(digest)) != 0);
	assert(DIGEST_COUNT < VARIANT_CAPACITY);
	memcpy(DIGESTS[DIGEST_COUNT++],digest,sizeof(digest));
}

static void Refused(const SparkStageKvLayoutIdentity *identity,const char *expected)
{
	uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
	const char *invalid = 0;
	assert(SparkStageKvLayoutDigest(identity,digest,&invalid) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(invalid != 0 && strcmp(invalid,expected) == 0);
}

int main(void)
{
	SparkStageKvLayoutIdentity identity,again;
	uint8_t first[SPARK_SHA256_DIGEST_BYTES],second[SPARK_SHA256_DIGEST_BYTES];
	const char *invalid;
	Base(&identity);
	Base(&again);
	assert(SparkStageKvLayoutDigest(&identity,first,&invalid) == SPARK_STATUS_OK);
	assert(SparkStageKvLayoutDigest(&again,second,&invalid) == SPARK_STATUS_OK);
	assert(memcmp(first,second,sizeof(first)) == 0);
	Record(&identity);
	Base(&identity); identity.model_id = "model2"; Record(&identity);
	Base(&identity); identity.model_revision = "revision2"; Record(&identity);
	Base(&identity); identity.pack_sha256[5] ^= 1u; Record(&identity);
	Base(&identity); identity.contract_sha256[9] ^= 1u; Record(&identity);
	Base(&identity); identity.driver_sha256[31] ^= 1u; Record(&identity);
	Base(&identity); identity.expert_codec = SPARK_WEIGHT_CODEC_BF16; Record(&identity);
	Base(&identity); identity.kv_codec = SPARK_WEIGHT_CODEC_FP8_E4M3; Record(&identity);
	Base(&identity); identity.context_shard.degree = 8u; Record(&identity);
	Base(&identity); identity.context_shard.rank = 4u; Record(&identity);
	Base(&identity); identity.context_shard.grain = 2u; Record(&identity);
	Base(&identity); identity.context_shard.degree = 1u; Record(&identity);
	Base(&identity); identity.block_token_count = 32u; Record(&identity);
	Base(&identity); identity.region_count = 1u; Record(&identity);
	Base(&identity); identity.regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR; Record(&identity);
	Base(&identity); identity.regions[1].layer_count = 5u; Record(&identity);
	Base(&identity); identity.regions[1].layer_page_bytes = 2048u; Record(&identity);
	Base(&identity); identity.page_bytes += 64u; Record(&identity);
	Base(&identity); identity.state_page_bytes = 4096u; Record(&identity);
	Base(&identity); identity.model_id = "ab"; identity.model_revision = "c"; Record(&identity);
	Base(&identity); identity.model_id = "a"; identity.model_revision = "bc"; Record(&identity);
	Base(&identity); identity.context_shard.degree = 1u; identity.context_shard.rank = 0u; identity.context_shard.grain = 0u;
	assert(SparkStageKvLayoutDigest(&identity,first,&invalid) == SPARK_STATUS_OK);
	Base(&identity); identity.context_shard.degree = 0u; identity.context_shard.rank = 9u; identity.context_shard.grain = 7u;
	assert(SparkStageKvLayoutDigest(&identity,second,&invalid) == SPARK_STATUS_OK && memcmp(first,second,sizeof(first)) == 0);
	invalid = 0;
	assert(SparkStageKvLayoutDigest(0,first,&invalid) == SPARK_STATUS_INVALID_ARGUMENT && strcmp(invalid,"identity") == 0);
	Base(&identity); identity.model_id = 0; Refused(&identity,"model_id");
	Base(&identity); identity.model_revision = ""; Refused(&identity,"model_revision");
	Base(&identity); memset(identity.pack_sha256,0,sizeof(identity.pack_sha256)); Refused(&identity,"pack_sha256");
	Base(&identity); memset(identity.contract_sha256,0,sizeof(identity.contract_sha256)); Refused(&identity,"contract_sha256");
	Base(&identity); memset(identity.driver_sha256,0,sizeof(identity.driver_sha256)); Refused(&identity,"driver_sha256");
	Base(&identity); identity.expert_codec = 999u; Refused(&identity,"expert_codec");
	Base(&identity); identity.kv_codec = 999u; Refused(&identity,"kv_codec");
	Base(&identity); identity.context_shard.rank = 16u; Refused(&identity,"context_shard");
	Base(&identity); identity.block_token_count = 0u; Refused(&identity,"block_token_count");
	Base(&identity); identity.region_count = 0u; Refused(&identity,"region_count");
	Base(&identity); identity.region_count = SPARK_STAGE_KV_MAX_REGIONS + 1u; Refused(&identity,"region_count");
	Base(&identity); identity.regions[1].layer_count = 0u; Refused(&identity,"region");
	Base(&identity); identity.regions[0].layer_page_bytes = 0u; Refused(&identity,"region");
	Base(&identity); identity.page_bytes = 0u; Refused(&identity,"page_bytes");
	printf("PASS stage kv layout digest: %u distinct single-input variants, length-prefixed strings, unsharded identity normalized, refusals named\n",DIGEST_COUNT);
	return(0);
}

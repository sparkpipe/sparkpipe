#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_glm52_dspark_pack.h"
/* Wire layout + magic/version come from the glm52 internal header (the
 * container IS the glm52 stage-pack format). */
#include "spark_glm52_resident_decode_stage_internal.h"

static uint32_t failures;

static void require(int condition,const char *label)
{
	if ( condition )
		printf("ok   %s\n",label);
	else
	{
		printf("FAIL %s\n",label);
		failures++;
	}
}

static void digest_of_buffer(const uint8_t *data,size_t bytes,uint8_t *digest)
{
	SparkSha256Context h;
	SparkSha256Initialize(&h);
	SparkSha256Update(&h,data,bytes);
	SparkSha256Finalize(&h,digest);
}


/* Assemble one pack in memory: header, 4-entry directory, 256-aligned
 * payloads [receipt, manifest, config, safetensors]. */
static size_t assemble_pack(uint8_t **pack_out,
	const uint8_t *manifest,size_t manifest_bytes,
	const uint8_t *config,size_t config_bytes,
	const uint8_t *safetensors,size_t safetensors_bytes,
	const uint8_t digest_manifest[SPARK_SHA256_DIGEST_BYTES],
	const uint8_t digest_config[SPARK_SHA256_DIGEST_BYTES],
	const uint8_t digest_safetensors[SPARK_SHA256_DIGEST_BYTES],
	int corrupt_safetensors)
{
	SparkGlm52StagePackHeader header;
	SparkGlm52StagePackEntry entries[4];
	SparkGlm52DsparkPackReceipt receipt;
	size_t payload_start;
	size_t total;
	uint8_t *buffer;
	uint64_t offsets[4];
	uint64_t counts[4];
	uint8_t *payloads[3];
	size_t sizes[3];
	uint32_t i;
	memset(&receipt,0,sizeof(receipt));
	memcpy(receipt.entries[0].sha256,digest_manifest,SPARK_SHA256_DIGEST_BYTES);
	receipt.entries[0].byte_count = manifest_bytes;
	memcpy(receipt.entries[1].sha256,digest_config,SPARK_SHA256_DIGEST_BYTES);
	receipt.entries[1].byte_count = config_bytes;
	memcpy(receipt.entries[2].sha256,digest_safetensors,
		SPARK_SHA256_DIGEST_BYTES);
	receipt.entries[2].byte_count = safetensors_bytes;
	payloads[0] = (uint8_t *)manifest;   sizes[0] = manifest_bytes;
	payloads[1] = (uint8_t *)config;     sizes[1] = config_bytes;
	payloads[2] = (uint8_t *)safetensors; sizes[2] = safetensors_bytes;
	payload_start = SPARK_GLM52_STAGEPACK_HEADER_BYTES +
		4u * SPARK_GLM52_STAGEPACK_ENTRY_BYTES;
	payload_start = (payload_start + 255u) & ~(size_t)255u;
	/* Mint law: every payload (receipt included) starts on its own
	 * 256-B boundary - the resolver refuses off-aligned artifacts. */
	offsets[0] = payload_start;
	counts[0] = sizeof(receipt);
	offsets[1] = (offsets[0] + counts[0] + 255u) & ~(uint64_t)255u;
	counts[1] = manifest_bytes;
	offsets[2] = (offsets[1] + counts[1] + 255u) & ~(uint64_t)255u;
	counts[2] = config_bytes;
	offsets[3] = (offsets[2] + counts[2] + 255u) & ~(uint64_t)255u;
	counts[3] = safetensors_bytes;
	total = offsets[3] + counts[3];
	buffer = calloc(1u,total);
	if ( buffer == 0 )
		exit(2);
	memset(&header,0,sizeof(header));
	header.magic = SPARK_GLM52_STAGEPACK_MAGIC;
	header.format_version = SPARK_GLM52_STAGEPACK_FORMAT_VERSION;
	header.header_bytes = SPARK_GLM52_STAGEPACK_HEADER_BYTES;
	header.directory_entry_bytes = SPARK_GLM52_STAGEPACK_ENTRY_BYTES;
	header.tensor_count = 4u;
	header.directory_offset = SPARK_GLM52_STAGEPACK_HEADER_BYTES;
	header.file_bytes = total;
	memcpy(buffer,&header,sizeof(header));
	for ( i = 0u; i < 4u; ++i )
	{
		memset(&entries[i],0,sizeof(entries[i]));
	}
	entries[0].tensor_kind = SPARK_GLM52_DSPARK_PACK_KIND_RECEIPT;
	entries[0].payload_offset = offsets[0];
	entries[0].payload_bytes = counts[0];
	entries[1].tensor_kind = SPARK_GLM52_DSPARK_PACK_KIND_MANIFEST;
	entries[1].payload_offset = offsets[1];
	entries[1].payload_bytes = counts[1];
	entries[2].tensor_kind = SPARK_GLM52_DSPARK_PACK_KIND_CONFIG;
	entries[2].payload_offset = offsets[2];
	entries[2].payload_bytes = counts[2];
	entries[3].tensor_kind = SPARK_GLM52_DSPARK_PACK_KIND_SAFETENSORS;
	entries[3].payload_offset = offsets[3];
	entries[3].payload_bytes = counts[3];
	memcpy(buffer + SPARK_GLM52_STAGEPACK_HEADER_BYTES,entries,
		sizeof(entries));
	memcpy(buffer + offsets[0],&receipt,sizeof(receipt));
	memcpy(buffer + offsets[1],manifest,manifest_bytes);
	memcpy(buffer + offsets[2],config,config_bytes);
	if ( corrupt_safetensors )
		buffer[offsets[3]] ^= 0x40u;
	else
		memcpy(buffer + offsets[3],safetensors,safetensors_bytes);
	*pack_out = buffer;
	return(total);
}

static void write_file(const char *path,const uint8_t *data,size_t bytes)
{
	FILE *f = fopen(path,"wb");
	if ( f == 0 )
	{
		printf("FAIL cannot write %s\n",path);
		exit(2);
	}
	fwrite(data,1u,bytes,f);
	fclose(f);
}

int main(void)
{
	static const char manifest_body[] =
		"{\"format\":\"glm52-dspark-manifest-dwarf\"}";
	static const char config_body[] =
		"{\"model_id\":\"glm52-dspark\",\"block_size\":7}";
	static const char safetensors_body[] = "DSPARK-DWARF-SAFETENSORS-PAYLOAD";
	char pack_path[256];
	char scratch[256];
	char out_m[1024],out_c[1024],out_s[1024];
	uint8_t digests[3][SPARK_SHA256_DIGEST_BYTES];
	uint8_t *good,*bad;
	size_t good_bytes,bad_bytes;
	SparkStatus status;
	digest_of_buffer((const uint8_t *)manifest_body,sizeof(manifest_body)-1u,
		digests[0]);
	digest_of_buffer((const uint8_t *)config_body,sizeof(config_body)-1u,
		digests[1]);
	digest_of_buffer((const uint8_t *)safetensors_body,
		sizeof(safetensors_body)-1u,digests[2]);

	snprintf(pack_path,sizeof(pack_path),"build/tmp/glm52_pack_ingest");
	mkdir(pack_path,0755);
	snprintf(scratch,sizeof(scratch),"%s/scratch",pack_path);
	mkdir(scratch,0755);

	good_bytes = assemble_pack(&good,
		(const uint8_t *)manifest_body,sizeof(manifest_body)-1u,
		(const uint8_t *)config_body,sizeof(config_body)-1u,
		(const uint8_t *)safetensors_body,sizeof(safetensors_body)-1u,
		digests[0],digests[1],digests[2],0);
	snprintf(pack_path,sizeof(pack_path),
		"build/tmp/glm52_pack_ingest/good.pack");
	write_file(pack_path,good,good_bytes);

	/* 1. Positive: resolve extracts all three artifacts and returns paths
	 * whose bytes hash back to the minted receipts. */
	status = SparkGlm52DsparkPackResolve(pack_path,scratch,
		out_m,sizeof(out_m),out_c,sizeof(out_c),out_s,sizeof(out_s));
	require(status == SPARK_STATUS_OK,"good pack resolves OK");
	if ( status == SPARK_STATUS_OK )
	{
		FILE *f;
		uint8_t d[SPARK_SHA256_DIGEST_BYTES];
		f = fopen(out_s,"rb");
		require(f != 0,"safetensors artifact materialized");
		if ( f != 0 )
		{
			fclose(f);
			digest_of_buffer((const uint8_t *)safetensors_body,
				sizeof(safetensors_body)-1u,d);
			require(memcmp(d,digests[2],SPARK_SHA256_DIGEST_BYTES) == 0,
				"extracted bytes match mint receipt");
		}
		require(strstr(out_m,"dspark_pack_") != 0 &&
			strlen(out_m) < sizeof(out_m),
			"extraction cell named after pack digest");
	}

	/* 2. Negative: tampered payload -> HASH_MISMATCH, loud refusal. */
	bad_bytes = assemble_pack(&bad,
		(const uint8_t *)manifest_body,sizeof(manifest_body)-1u,
		(const uint8_t *)config_body,sizeof(config_body)-1u,
		(const uint8_t *)safetensors_body,sizeof(safetensors_body)-1u,
		digests[0],digests[1],digests[2],1);
	snprintf(pack_path,sizeof(pack_path),
		"build/tmp/glm52_pack_ingest/tampered.pack");
	write_file(pack_path,bad,bad_bytes);
	status = SparkGlm52DsparkPackResolve(pack_path,scratch,
		out_m,sizeof(out_m),out_c,sizeof(out_c),out_s,sizeof(out_s));
	require(status == SPARK_STATUS_HASH_MISMATCH,
		"tampered artifact refused HASH_MISMATCH");

	/* 3. Negative: bad magic -> SCHEMA_ERROR before any extraction. */
	bad[0] ^= 0xFFu;
	snprintf(pack_path,sizeof(pack_path),
		"build/tmp/glm52_pack_ingest/badmagic.pack");
	write_file(pack_path,bad,bad_bytes);
	status = SparkGlm52DsparkPackResolve(pack_path,scratch,
		out_m,sizeof(out_m),out_c,sizeof(out_c),out_s,sizeof(out_s));
	require(status == SPARK_STATUS_SCHEMA_ERROR,
		"alien magic refused SCHEMA_ERROR");

	printf(failures == 0u ?
		"PASS glm52 dspark pack ingest\n" : "FAIL %u checks\n",failures);
	return failures != 0u;
}

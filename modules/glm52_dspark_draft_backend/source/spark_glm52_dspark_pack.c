#include <cuda_runtime.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "spark_glm52_resident_decode_stage_internal.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_glm52_dspark_pack.h"

#define SPARK_GLM52_DSPARK_PACK_ALIGNMENT 256u

_Static_assert(sizeof(SparkGlm52DsparkPackReceipt) ==
	3u * (SPARK_SHA256_DIGEST_BYTES + 8u),
	"receipt layout must stay fixed for minted packs");

static SparkStatus SparkGlm52DsparkPackReadExact(
	FILE *file,void *buffer,uint64_t bytes)
{
	if ( bytes > SIZE_MAX || fread(buffer,1u,(size_t)bytes,file) != bytes )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

static int SparkGlm52DsparkPackFindKind(
	const SparkGlm52StagePackEntry *entries,uint32_t entry_count,
	uint32_t kind,uint64_t *offset_out,uint64_t *bytes_out)
{
	uint32_t index;
	int found = 0;
	for ( index = 0u; index < entry_count; ++index )
	{
		if ( entries[index].tensor_kind != kind )
			continue;
		if ( found != 0 )
			return(-1); /* duplicated artifact entry */
		found = 1;
		*offset_out = entries[index].payload_offset;
		*bytes_out = entries[index].payload_bytes;
	}
	return(found == 0 ? -2 : 0);
}

static void SparkGlm52DsparkPackDigestToHex(
	const uint8_t *digest,char *out,size_t out_bytes)
{
	static const char hex[] = "0123456789abcdef";
	uint32_t index;
	if ( out_bytes < SPARK_SHA256_DIGEST_BYTES * 2u + 1u )
		return;
	for ( index = 0u; index < SPARK_SHA256_DIGEST_BYTES; ++index )
	{
		out[index * 2u] = hex[digest[index] >> 4];
		out[index * 2u + 1u] = hex[digest[index] & 0x0fu];
	}
	out[SPARK_SHA256_DIGEST_BYTES * 2u] = '\0';
}

static SparkStatus SparkGlm52DsparkPackExtractAndVerify(
	FILE *file,const char *pack_path,uint64_t payload_offset,
	uint64_t payload_bytes,const SparkGlm52DsparkPackReceiptEntry *receipt,
	const char *destination)
{
	FILE *out;
	SparkSha256Context hash;
	uint8_t buffer[4096];
	uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
	uint64_t remaining = payload_bytes;
	if ( fseek(file,(long)payload_offset,SEEK_SET) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	out = fopen(destination,"wb");
	if ( out == 0 )
		return(SPARK_STATUS_IO_ERROR);
	SparkSha256Initialize(&hash);
	while ( remaining != 0u )
	{
		uint64_t chunk = remaining < (uint64_t)sizeof(buffer) ?
			remaining : (uint64_t)sizeof(buffer);
		if ( fread(buffer,1u,(size_t)chunk,file) != chunk )
		{
			fclose(out);
			remove(destination);
			(void)pack_path;
			return(SPARK_STATUS_IO_ERROR);
		}
		if ( fwrite(buffer,1u,(size_t)chunk,out) != chunk )
		{
			fclose(out);
			remove(destination);
			return(SPARK_STATUS_IO_ERROR);
		}
		SparkSha256Update(&hash,buffer,(size_t)chunk);
		remaining -= chunk;
	}
	if ( fclose(out) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	SparkSha256Finalize(&hash,digest);
	if ( memcmp(digest,receipt->sha256,SPARK_SHA256_DIGEST_BYTES) != 0 ||
		payload_bytes != receipt->byte_count )
	{
		remove(destination);
		return(SPARK_STATUS_HASH_MISMATCH);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52DsparkPackResolve(
	const char *pack_path,
	const char *scratch_directory,
	char *manifest_path_out,
	size_t manifest_path_bytes,
	char *config_path_out,
	size_t config_path_bytes,
	char *safetensors_path_out,
	size_t safetensors_path_bytes)
{
	FILE *file;
	SparkGlm52StagePackHeader header;
	SparkGlm52StagePackEntry *entries = 0;
	SparkGlm52DsparkPackReceipt receipt;
	char work[512];
	char digest_hex[SPARK_SHA256_DIGEST_BYTES * 2u + 1u];
	uint8_t pack_digest[SPARK_SHA256_DIGEST_BYTES];
	const char *artifact_names[3] = {
		"manifest.json","config.json","model.safetensors" };
	uint64_t offsets[3];
	uint64_t byte_counts[3];
	const uint32_t kinds[3] = {
		SPARK_GLM52_DSPARK_PACK_KIND_MANIFEST,
		SPARK_GLM52_DSPARK_PACK_KIND_CONFIG,
		SPARK_GLM52_DSPARK_PACK_KIND_SAFETENSORS };
	long file_size;
	int status;
	uint32_t index,entry_index;
	if ( pack_path == 0 || scratch_directory == 0 || manifest_path_out == 0 ||
		config_path_out == 0 || safetensors_path_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	file = fopen(pack_path,"rb");
	if ( file == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	status = fseek(file,0L,SEEK_END);
	if ( status != 0 || (file_size = ftell(file)) < 0L )
		{ fclose(file); return(SPARK_STATUS_IO_ERROR); }
	rewind(file);
	status = SparkGlm52DsparkPackReadExact(file,&header,
		SPARK_GLM52_STAGEPACK_HEADER_BYTES);
	if ( status != SPARK_STATUS_OK )
		{ fclose(file); return(status); }
	if ( header.magic != SPARK_GLM52_STAGEPACK_MAGIC ||
		header.format_version != SPARK_GLM52_STAGEPACK_FORMAT_VERSION ||
		header.header_bytes != SPARK_GLM52_STAGEPACK_HEADER_BYTES ||
		header.directory_entry_bytes != SPARK_GLM52_STAGEPACK_ENTRY_BYTES )
		{ fclose(file); return(SPARK_STATUS_SCHEMA_ERROR); }
	if ( (uint64_t)file_size != header.file_bytes ||
		header.directory_offset > header.file_bytes ||
		header.tensor_count > (header.file_bytes -
			header.directory_offset) / SPARK_GLM52_STAGEPACK_ENTRY_BYTES )
		{ fclose(file); return(SPARK_STATUS_SCHEMA_ERROR); }
	entries = (SparkGlm52StagePackEntry *)malloc(
		(size_t)header.tensor_count * SPARK_GLM52_STAGEPACK_ENTRY_BYTES);
	if ( entries == 0 )
		{ fclose(file); return(SPARK_STATUS_CAPACITY_EXCEEDED); }
	if ( fseek(file,(long)header.directory_offset,SEEK_SET) != 0 ||
		SparkGlm52DsparkPackReadExact(file,entries,
			(uint64_t)header.tensor_count *
			SPARK_GLM52_STAGEPACK_ENTRY_BYTES) != SPARK_STATUS_OK )
	{
		free(entries);
		fclose(file);
		return(SPARK_STATUS_IO_ERROR);
	}
	/* Receipt first: it pins every artifact's bytes+sha256 at mint time. */
	if ( SparkGlm52DsparkPackFindKind(entries,header.tensor_count,
		SPARK_GLM52_DSPARK_PACK_KIND_RECEIPT,&offsets[0],&byte_counts[0]) != 0 ||
		byte_counts[0] != sizeof(receipt) ||
		offsets[0] % SPARK_GLM52_DSPARK_PACK_ALIGNMENT != 0u ||
		offsets[0] + byte_counts[0] > header.file_bytes ||
		fseek(file,(long)offsets[0],SEEK_SET) != 0 ||
		SparkGlm52DsparkPackReadExact(file,&receipt,sizeof(receipt)) !=
			SPARK_STATUS_OK )
	{
		free(entries);
		fclose(file);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	for ( index = 0u; index < 3u; ++index )
	{
		status = SparkGlm52DsparkPackFindKind(entries,header.tensor_count,
			kinds[index],&offsets[index],&byte_counts[index]);
		if ( status != 0 ||
			offsets[index] % SPARK_GLM52_DSPARK_PACK_ALIGNMENT != 0u ||
			offsets[index] + byte_counts[index] > header.file_bytes )
		{
			free(entries);
			fclose(file);
			return(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	free(entries);
	entries = 0;
	/* Scratch cell named after the PACK digest so two packs never share an
	 * extraction directory. */
	{
		SparkSha256Context whole;
		rewind(file);
		SparkSha256Initialize(&whole);
		for ( ;; )
		{
			size_t chunk = fread(work,1u,sizeof(work),file);
			if ( chunk == 0u )
				break;
			SparkSha256Update(&whole,work,chunk);
		}
		SparkSha256Finalize(&whole,pack_digest);
	}
	SparkGlm52DsparkPackDigestToHex(pack_digest,digest_hex,sizeof(digest_hex));
	status = snprintf(work,sizeof(work),"%s/dspark_pack_%c%c%c%c%c%c%c%c",
		scratch_directory,digest_hex[0],digest_hex[1],digest_hex[2],
		digest_hex[3],digest_hex[4],digest_hex[5],digest_hex[6],
		digest_hex[7]);
	if ( status <= 0 || (size_t)status >= sizeof(work) )
		{ fclose(file); return(SPARK_STATUS_INVALID_ARGUMENT); }
	if ( mkdir(work,0755) != 0 && errno != EEXIST )
		{ fclose(file); return(SPARK_STATUS_IO_ERROR); }
	for ( index = 0u; index < 3u; ++index )
	{
		char destination[sizeof(work) + 64u];
		status = snprintf(destination,sizeof(destination),"%s/%s",work,
			artifact_names[index]);
		if ( status <= 0 || (size_t)status >= sizeof(destination) )
			{ fclose(file); return(SPARK_STATUS_INVALID_ARGUMENT); }
		status = SparkGlm52DsparkPackExtractAndVerify(file,pack_path,
			offsets[index],byte_counts[index],&receipt.entries[index],
			destination);
		if ( status != SPARK_STATUS_OK )
			{ fclose(file); return((SparkStatus)status); }
		if ( index == 0u )
		{
			if ( snprintf(manifest_path_out,manifest_path_bytes,"%s",
				destination) >= (int)manifest_path_bytes )
				{ fclose(file); return(SPARK_STATUS_CAPACITY_EXCEEDED); }
		}
		else if ( index == 1u )
		{
			if ( snprintf(config_path_out,config_path_bytes,"%s",
				destination) >= (int)config_path_bytes )
				{ fclose(file); return(SPARK_STATUS_CAPACITY_EXCEEDED); }
		}
		else
		{
			if ( snprintf(safetensors_path_out,safetensors_path_bytes,"%s",
				destination) >= (int)safetensors_path_bytes )
				{ fclose(file); return(SPARK_STATUS_CAPACITY_EXCEEDED); }
		}
	}
	(void)entry_index;
	fclose(file);
	return(SPARK_STATUS_OK);
}

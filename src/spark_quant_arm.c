#include "sparkpipe/spark_quant_arm.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_json.h"

typedef struct SparkQuantArmWriter
{
	char *buffer;
	uint32_t capacity;
	uint32_t length;
	uint32_t overflow;
} SparkQuantArmWriter;

static const char *const SparkQuantArmRootMembers[] = { "format","arm_id","model","revision","topology","spine","expert","kv","drafter","pack_sha256","artifacts" };
static const char *const SparkQuantArmTopologyMembers[] = { "tp","pp","kv_shard" };
static const char *const SparkQuantArmSpineMembers[] = { "frame","source","spine_digest" };
static const char *const SparkQuantArmExpertMembers[] = { "codec","label","producer","source","recipe_sha256" };
static const char *const SparkQuantArmKvMembers[] = { "latent","index","state","group","mode" };
static const char *const SparkQuantArmDrafterMembers[] = { "kind","label","codec","head","sidecar_sha256" };
static const char *const SparkQuantArmArtifactMembers[] = { "module_archive_sha256","driver_sha256","adapter_sha256" };
static const char *const SparkQuantArmExpertCodecs[] = { "bf16","fp8","nvfp4","mxfp4","int8","int7","int6" };
static const char *const SparkQuantArmProducers[] = { "publisher","community","experiment" };
static const char *const SparkQuantArmKvLatentCodecs[] = { "bf16","fp8","mxfp4" };
static const char *const SparkQuantArmKvIndexCodecs[] = { "bf16","fp8" };
static const char *const SparkQuantArmKvStateCodecs[] = { "fp32","bf16" };
static const char *const SparkQuantArmKvModes[] = { "sim","store" };
static const char *const SparkQuantArmDrafterKinds[] = { "none","lookup","mtp","dflash","oracle","adversary","random" };
static const char *const SparkQuantArmDrafterHeads[] = { "bf16","fp8" };

#define SPARK_QUANT_ARM_COUNT(values) ((uint32_t)(sizeof(values) / sizeof((values)[0])))

static SparkStatus SparkQuantArmRefuse(char *error_buffer, uint32_t error_buffer_bytes, SparkStatus status, const char *format, ...)
{
	va_list arguments;
	if ( error_buffer != 0 && error_buffer_bytes != 0u )
	{
		va_start(arguments,format);
		(void)vsnprintf(error_buffer,error_buffer_bytes,format,arguments);
		va_end(arguments);
	}
	SPARK_FAIL(status);
}

static int32_t SparkQuantArmTextIsPlain(const char *text)
{
	const uint8_t *cursor;
	for (cursor=(const uint8_t *)text; *cursor!=0u; cursor++)
	{
		if ( *cursor < 0x20u || *cursor > 0x7eu || *cursor == '"' || *cursor == '\\' )
			return(0);
	}
	return(1);
}

static int32_t SparkQuantArmTextIsLabel(const char *text,uint32_t allow_dash)
{
	const char *cursor;
	if ( text[0] == '\0' )
		return(0);
	for (cursor=text; *cursor!='\0'; cursor++)
	{
		if ( (*cursor >= 'a' && *cursor <= 'z') || (*cursor >= '0' && *cursor <= '9') )
			continue;
		if ( allow_dash != 0u && (*cursor == '-' || *cursor == '_') )
			continue;
		return(0);
	}
	return(1);
}

static int32_t SparkQuantArmTextIsHex(const char *text)
{
	uint32_t index;
	for (index=0u; index<64u; index++)
	{
		if ( !((text[index] >= '0' && text[index] <= '9') || (text[index] >= 'a' && text[index] <= 'f')) )
			return(0);
	}
	return(text[64] == '\0');
}

static int32_t SparkQuantArmTextIsSource(const char *text)
{
	const char *at = strchr(text,'@');
	return(at != 0 && at != text && at[1] != '\0' && strchr(at + 1,'@') == 0);
}

static int32_t SparkQuantArmTokenIn(const char *text,const char *const *values,uint32_t count)
{
	uint32_t index;
	for (index=0u; index<count; index++)
	{
		if ( strcmp(text,values[index]) == 0 )
			return(1);
	}
	return(0);
}

static SparkStatus SparkQuantArmObject(const SparkJsonDocument *document,int32_t parent,const char *name,const char *const *members,uint32_t member_count,int32_t *object,char *error_buffer,uint32_t error_buffer_bytes)
{
	*object = parent < 0 ? SparkJsonGetRootToken(document) : SparkJsonFindObjectMember(document,parent,name);
	if ( *object < 0 || !SparkJsonTokenIsType(document,*object,SPARK_JSON_TOKEN_OBJECT) )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: %s must be an object",name));
	if ( SparkJsonValidateObjectMembersExact(document,*object,members,member_count) != SPARK_STATUS_OK )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: %s members are not exactly the %s set",name,SPARK_QUANT_ARM_FORMAT));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQuantArmCopyTokenText(const SparkJsonDocument *document,int32_t token,const char *name,char *output,uint32_t output_bytes,char *error_buffer,uint32_t error_buffer_bytes)
{
	char *text = 0;
	size_t length;
	if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_STRING) )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: %s must be a string",name));
	if ( SparkJsonCopyString(document,token,&text) != SPARK_STATUS_OK )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_PARSE_ERROR,"arm: %s is not a valid string",name));
	length = strlen(text);
	if ( length == 0u || length >= output_bytes || SparkQuantArmTextIsPlain(text) == 0 )
	{
		free(text);
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: %s must be 1-%u printable ASCII characters without quote or backslash",name,output_bytes - 1u));
	}
	memcpy(output,text,length + 1u);
	free(text);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQuantArmString(const SparkJsonDocument *document,int32_t object,const char *name,char *output,uint32_t output_bytes,char *error_buffer,uint32_t error_buffer_bytes)
{
	return(SparkQuantArmCopyTokenText(document,SparkJsonFindObjectMember(document,object,name),name,output,output_bytes,error_buffer,error_buffer_bytes));
}

static SparkStatus SparkQuantArmEnum(const SparkJsonDocument *document,int32_t object,const char *name,const char *const *values,uint32_t count,char *output,char *error_buffer,uint32_t error_buffer_bytes)
{
	SparkStatus status = SparkQuantArmString(document,object,name,output,SPARK_QUANT_ARM_TOKEN_BYTES,error_buffer,error_buffer_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( SparkQuantArmTokenIn(output,values,count) == 0 )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: %s value %s is not a known token",name,output));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQuantArmHex(const SparkJsonDocument *document,int32_t token,const char *name,char *output,char *error_buffer,uint32_t error_buffer_bytes)
{
	SparkStatus status = SparkQuantArmCopyTokenText(document,token,name,output,SPARK_SHA256_HEX_BYTES,error_buffer,error_buffer_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( SparkQuantArmTextIsHex(output) == 0 )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: %s must be 64 lowercase hex digits",name));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQuantArmUnsigned(const SparkJsonDocument *document,int32_t object,const char *name,uint32_t *value,char *error_buffer,uint32_t error_buffer_bytes)
{
	int32_t token = SparkJsonFindObjectMember(document,object,name);
	char *raw = 0;
	uint32_t raw_bytes = 0u;
	uint32_t index;
	if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_PRIMITIVE) || SparkJsonCopyRawValue(document,token,&raw,&raw_bytes) != SPARK_STATUS_OK )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: %s must be an unsigned integer",name));
	for (index=0u; index<raw_bytes; index++)
	{
		if ( raw[index] < '0' || raw[index] > '9' || (index == 0u && raw[index] == '0' && raw_bytes > 1u) )
		{
			free(raw);
			return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: %s must be a canonical unsigned integer",name));
		}
	}
	free(raw);
	if ( raw_bytes == 0u || raw_bytes > 9u || SparkJsonGetUInt32(document,token,value) != SPARK_STATUS_OK )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: %s is out of range",name));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQuantArmHexArray(const SparkJsonDocument *document,int32_t object,const char *name,uint32_t expected_count,char output[][SPARK_SHA256_HEX_BYTES],char *error_buffer,uint32_t error_buffer_bytes)
{
	int32_t array = SparkJsonFindObjectMember(document,object,name);
	uint32_t index;
	SparkStatus status;
	if ( array < 0 || !SparkJsonTokenIsType(document,array,SPARK_JSON_TOKEN_ARRAY) )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: %s must be an array",name));
	if ( SparkJsonGetArrayElementCount(document,array) != expected_count )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: %s has %u entries, expected %u",name,SparkJsonGetArrayElementCount(document,array),expected_count));
	for (index=0u; index<expected_count; index++)
	{
		status = SparkQuantArmHex(document,SparkJsonGetArrayElement(document,array,index),name,output[index],error_buffer,error_buffer_bytes);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

static int32_t SparkQuantArmTokenIsNull(const SparkJsonDocument *document,int32_t token)
{
	char *raw = 0;
	uint32_t raw_bytes = 0u;
	int32_t result;
	if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_PRIMITIVE) || SparkJsonCopyRawValue(document,token,&raw,&raw_bytes) != SPARK_STATUS_OK )
		return(0);
	result = raw_bytes == 4u && memcmp(raw,"null",4u) == 0 ? 1 : 0;
	free(raw);
	return(result);
}

static void SparkQuantArmAppend(SparkQuantArmWriter *writer,const char *format,...)
{
	va_list arguments;
	int written;
	uint32_t room;
	if ( writer->overflow != 0u )
		return;
	room = writer->capacity - writer->length;
	va_start(arguments,format);
	written = vsnprintf(writer->buffer + writer->length,room,format,arguments);
	va_end(arguments);
	if ( written < 0 || (uint32_t)written >= room )
	{
		writer->overflow = 1u;
		return;
	}
	writer->length += (uint32_t)written;
}

static void SparkQuantArmAppendHexArray(SparkQuantArmWriter *writer,const char values[][SPARK_SHA256_HEX_BYTES],uint32_t count)
{
	uint32_t index;
	SparkQuantArmAppend(writer,"[");
	for (index=0u; index<count; index++)
		SparkQuantArmAppend(writer,"%s\"%s\"",index == 0u ? "" : ",",values[index]);
	SparkQuantArmAppend(writer,"]");
}

static void SparkQuantArmKvToken(const SparkQuantArm *arm,const char *codec,char *output,uint32_t output_bytes)
{
	if ( strcmp(codec,"bf16") == 0 )
		(void)snprintf(output,output_bytes,"%s",codec);
	else
		(void)snprintf(output,output_bytes,"%sg%u",codec,arm->kv_group);
}

SparkStatus SparkQuantArmExpectedId(const SparkQuantArm *arm,char *buffer,uint32_t buffer_bytes)
{
	char latent[SPARK_QUANT_ARM_TOKEN_BYTES + 16u];
	char index[SPARK_QUANT_ARM_TOKEN_BYTES + 16u];
	int written;
	if ( arm == 0 || buffer == 0 || buffer_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkQuantArmKvToken(arm,arm->kv_latent,latent,(uint32_t)sizeof(latent));
	SparkQuantArmKvToken(arm,arm->kv_index,index,(uint32_t)sizeof(index));
	written = snprintf(buffer,buffer_bytes,"%s.%s.e-%s.k-%s/%s/%s.d-%s",arm->model,arm->spine_frame,arm->expert_label,latent,index,arm->kv_state,arm->drafter_label);
	if ( written < 0 || (uint32_t)written >= buffer_bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkQuantArmCanonicalize(const SparkQuantArm *arm,char *buffer,uint32_t buffer_bytes,uint32_t *written_bytes)
{
	SparkQuantArmWriter writer;
	if ( arm == 0 || buffer == 0 || buffer_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	writer.buffer = buffer;
	writer.capacity = buffer_bytes;
	writer.length = 0u;
	writer.overflow = 0u;
	buffer[0] = '\0';
	SparkQuantArmAppend(&writer,"{\"arm_id\":\"%s\",\"artifacts\":{\"adapter_sha256\":\"%s\",\"driver_sha256\":\"%s\",\"module_archive_sha256\":\"%s\"}",arm->arm_id,arm->adapter_sha256,arm->driver_sha256,arm->module_archive_sha256);
	SparkQuantArmAppend(&writer,",\"drafter\":{\"codec\":\"%s\",\"head\":\"%s\",\"kind\":\"%s\",\"label\":\"%s\",\"sidecar_sha256\":",arm->drafter_codec,arm->drafter_head,arm->drafter_kind,arm->drafter_label);
	SparkQuantArmAppendHexArray(&writer,(const char (*)[SPARK_SHA256_HEX_BYTES])arm->drafter_sidecar_sha256,arm->drafter_sidecar_count);
	SparkQuantArmAppend(&writer,"},\"expert\":{\"codec\":\"%s\",\"label\":\"%s\",\"producer\":\"%s\",\"recipe_sha256\":",arm->expert_codec,arm->expert_label,arm->expert_producer);
	if ( arm->expert_recipe_present != 0u )
		SparkQuantArmAppend(&writer,"\"%s\"",arm->expert_recipe_sha256);
	else
		SparkQuantArmAppend(&writer,"null");
	SparkQuantArmAppend(&writer,",\"source\":\"%s\"},\"format\":\"%s\"",arm->expert_source,SPARK_QUANT_ARM_FORMAT);
	SparkQuantArmAppend(&writer,",\"kv\":{\"group\":%u,\"index\":\"%s\",\"latent\":\"%s\",\"mode\":\"%s\",\"state\":\"%s\"}",arm->kv_group,arm->kv_index,arm->kv_latent,arm->kv_mode,arm->kv_state);
	SparkQuantArmAppend(&writer,",\"model\":\"%s\",\"pack_sha256\":",arm->model);
	SparkQuantArmAppendHexArray(&writer,(const char (*)[SPARK_SHA256_HEX_BYTES])arm->pack_sha256,arm->rank_count);
	SparkQuantArmAppend(&writer,",\"revision\":\"%s\",\"spine\":{\"frame\":\"%s\",\"source\":\"%s\",\"spine_digest\":",arm->revision,arm->spine_frame,arm->spine_source);
	SparkQuantArmAppendHexArray(&writer,(const char (*)[SPARK_SHA256_HEX_BYTES])arm->spine_digest,arm->rank_count);
	SparkQuantArmAppend(&writer,"},\"topology\":{\"kv_shard\":%u,\"pp\":%u,\"tp\":%u}}",arm->kv_shard,arm->pipeline_parallel,arm->tensor_parallel);
	if ( writer.overflow != 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( written_bytes != 0 )
		*written_bytes = writer.length;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQuantArmValidateAxes(SparkQuantArm *arm,char *error_buffer,uint32_t error_buffer_bytes)
{
	uint32_t bf16_kv = strcmp(arm->kv_latent,"bf16") == 0 && strcmp(arm->kv_index,"bf16") == 0;
	uint32_t weightless;
	if ( arm->kv_shard > 1u )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: topology kv_shard must be 0 or 1"));
	if ( SparkQuantArmTextIsLabel(arm->model,1u) == 0 )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: model must be lowercase letters, digits, dash or underscore"));
	if ( arm->spine_frame[0] != 'S' || arm->spine_frame[1] == '\0' || strspn(arm->spine_frame + 1,"0123456789") != strlen(arm->spine_frame + 1) )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: spine frame must be S followed by digits"));
	if ( SparkQuantArmTextIsSource(arm->spine_source) == 0 || SparkQuantArmTextIsSource(arm->expert_source) == 0 )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: spine and expert sources must be repo@revision"));
	if ( SparkQuantArmTextIsLabel(arm->expert_label,0u) == 0 || strncmp(arm->expert_label,arm->expert_codec,strlen(arm->expert_codec)) != 0 )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: expert label %s must be lowercase alphanumeric and start with the codec %s",arm->expert_label,arm->expert_codec));
	if ( strcmp(arm->expert_producer,"experiment") == 0 && arm->expert_recipe_present == 0u )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: an experiment expert codec needs its recipe_sha256"));
	if ( bf16_kv != 0u && arm->kv_group != 0u )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: kv group must be 0 when latent and index are bf16"));
	if ( bf16_kv == 0u && arm->kv_group != 32u && arm->kv_group != 64u && arm->kv_group != 128u )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: kv group %u must be 32, 64 or 128 for a quantized latent or index",arm->kv_group));
	if ( strcmp(arm->kv_latent,"mxfp4") == 0 && arm->kv_group != 32u )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: an mxfp4 latent needs group 32"));
	if ( bf16_kv != 0u && strcmp(arm->kv_state,"fp32") == 0 && strcmp(arm->kv_mode,"store") != 0 )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: the unquantized kv reference must use mode store"));
	weightless = strcmp(arm->drafter_kind,"mtp") != 0 && strcmp(arm->drafter_kind,"dflash") != 0;
	if ( weightless != 0u && (strcmp(arm->drafter_codec,"none") != 0 || strcmp(arm->drafter_head,"none") != 0 || arm->drafter_sidecar_count != 0u) )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: drafter %s carries no weights, so codec and head must be none and sidecar_sha256 empty",arm->drafter_kind));
	if ( weightless == 0u && (SparkQuantArmTokenIn(arm->drafter_codec,SparkQuantArmExpertCodecs,SPARK_QUANT_ARM_COUNT(SparkQuantArmExpertCodecs)) == 0 || SparkQuantArmTokenIn(arm->drafter_head,SparkQuantArmDrafterHeads,SPARK_QUANT_ARM_COUNT(SparkQuantArmDrafterHeads)) == 0 || arm->drafter_sidecar_count != arm->rank_count) )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: drafter %s needs a weight codec, a bf16 or fp8 head and one sidecar sha256 per rank",arm->drafter_kind));
	if ( SparkQuantArmTextIsLabel(arm->drafter_label,0u) == 0 || strncmp(arm->drafter_label,arm->drafter_kind,strlen(arm->drafter_kind)) != 0 )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: drafter label %s must be lowercase alphanumeric and start with the kind %s",arm->drafter_label,arm->drafter_kind));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQuantArmParseDrafter(const SparkJsonDocument *document,int32_t root,SparkQuantArm *arm,char *error_buffer,uint32_t error_buffer_bytes)
{
	int32_t object,array;
	uint32_t count,index;
	SparkStatus status = SparkQuantArmObject(document,root,"drafter",SparkQuantArmDrafterMembers,SPARK_QUANT_ARM_COUNT(SparkQuantArmDrafterMembers),&object,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmEnum(document,object,"kind",SparkQuantArmDrafterKinds,SPARK_QUANT_ARM_COUNT(SparkQuantArmDrafterKinds),arm->drafter_kind,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmString(document,object,"label",arm->drafter_label,SPARK_QUANT_ARM_TOKEN_BYTES,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmString(document,object,"codec",arm->drafter_codec,SPARK_QUANT_ARM_TOKEN_BYTES,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmString(document,object,"head",arm->drafter_head,SPARK_QUANT_ARM_TOKEN_BYTES,error_buffer,error_buffer_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	array = SparkJsonFindObjectMember(document,object,"sidecar_sha256");
	if ( array < 0 || !SparkJsonTokenIsType(document,array,SPARK_JSON_TOKEN_ARRAY) )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: sidecar_sha256 must be an array"));
	count = SparkJsonGetArrayElementCount(document,array);
	if ( count > SPARK_QUANT_ARM_MAX_RANKS )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: sidecar_sha256 has more than %u entries",SPARK_QUANT_ARM_MAX_RANKS));
	for (index=0u; status==SPARK_STATUS_OK && index<count; index++)
		status = SparkQuantArmHex(document,SparkJsonGetArrayElement(document,array,index),"sidecar_sha256",arm->drafter_sidecar_sha256[index],error_buffer,error_buffer_bytes);
	arm->drafter_sidecar_count = count;
	return(status);
}

static SparkStatus SparkQuantArmParseDocument(const SparkJsonDocument *document,SparkQuantArm *arm,char *error_buffer,uint32_t error_buffer_bytes)
{
	int32_t root,object,token;
	char format[SPARK_QUANT_ARM_TOKEN_BYTES];
	SparkStatus status = SparkQuantArmObject(document,-1,"root",SparkQuantArmRootMembers,SPARK_QUANT_ARM_COUNT(SparkQuantArmRootMembers),&root,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmString(document,root,"format",format,(uint32_t)sizeof(format),error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK && strcmp(format,SPARK_QUANT_ARM_FORMAT) != 0 )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_ABI_MISMATCH,"arm: format %s is not %s",format,SPARK_QUANT_ARM_FORMAT));
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmString(document,root,"arm_id",arm->arm_id,SPARK_QUANT_ARM_ID_BYTES,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmString(document,root,"model",arm->model,SPARK_QUANT_ARM_TOKEN_BYTES,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmString(document,root,"revision",arm->revision,SPARK_QUANT_ARM_TEXT_BYTES,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmObject(document,root,"topology",SparkQuantArmTopologyMembers,SPARK_QUANT_ARM_COUNT(SparkQuantArmTopologyMembers),&object,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmUnsigned(document,object,"tp",&arm->tensor_parallel,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmUnsigned(document,object,"pp",&arm->pipeline_parallel,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmUnsigned(document,object,"kv_shard",&arm->kv_shard,error_buffer,error_buffer_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( arm->tensor_parallel == 0u || arm->pipeline_parallel == 0u || arm->tensor_parallel > SPARK_QUANT_ARM_MAX_RANKS || arm->tensor_parallel * arm->pipeline_parallel > SPARK_QUANT_ARM_MAX_RANKS )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: topology tp=%u pp=%u is outside 1..%u ranks",arm->tensor_parallel,arm->pipeline_parallel,SPARK_QUANT_ARM_MAX_RANKS));
	arm->rank_count = arm->tensor_parallel * arm->pipeline_parallel;
	status = SparkQuantArmObject(document,root,"spine",SparkQuantArmSpineMembers,SPARK_QUANT_ARM_COUNT(SparkQuantArmSpineMembers),&object,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmString(document,object,"frame",arm->spine_frame,SPARK_QUANT_ARM_TOKEN_BYTES,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmString(document,object,"source",arm->spine_source,SPARK_QUANT_ARM_TEXT_BYTES,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmHexArray(document,object,"spine_digest",arm->rank_count,arm->spine_digest,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmObject(document,root,"expert",SparkQuantArmExpertMembers,SPARK_QUANT_ARM_COUNT(SparkQuantArmExpertMembers),&object,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmEnum(document,object,"codec",SparkQuantArmExpertCodecs,SPARK_QUANT_ARM_COUNT(SparkQuantArmExpertCodecs),arm->expert_codec,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmString(document,object,"label",arm->expert_label,SPARK_QUANT_ARM_TOKEN_BYTES,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmEnum(document,object,"producer",SparkQuantArmProducers,SPARK_QUANT_ARM_COUNT(SparkQuantArmProducers),arm->expert_producer,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmString(document,object,"source",arm->expert_source,SPARK_QUANT_ARM_TEXT_BYTES,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
	{
		token = SparkJsonFindObjectMember(document,object,"recipe_sha256");
		arm->expert_recipe_present = SparkQuantArmTokenIsNull(document,token) != 0 ? 0u : 1u;
		if ( arm->expert_recipe_present != 0u )
			status = SparkQuantArmHex(document,token,"recipe_sha256",arm->expert_recipe_sha256,error_buffer,error_buffer_bytes);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmObject(document,root,"kv",SparkQuantArmKvMembers,SPARK_QUANT_ARM_COUNT(SparkQuantArmKvMembers),&object,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmEnum(document,object,"latent",SparkQuantArmKvLatentCodecs,SPARK_QUANT_ARM_COUNT(SparkQuantArmKvLatentCodecs),arm->kv_latent,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmEnum(document,object,"index",SparkQuantArmKvIndexCodecs,SPARK_QUANT_ARM_COUNT(SparkQuantArmKvIndexCodecs),arm->kv_index,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmEnum(document,object,"state",SparkQuantArmKvStateCodecs,SPARK_QUANT_ARM_COUNT(SparkQuantArmKvStateCodecs),arm->kv_state,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmUnsigned(document,object,"group",&arm->kv_group,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmEnum(document,object,"mode",SparkQuantArmKvModes,SPARK_QUANT_ARM_COUNT(SparkQuantArmKvModes),arm->kv_mode,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmParseDrafter(document,root,arm,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmHexArray(document,root,"pack_sha256",arm->rank_count,arm->pack_sha256,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmObject(document,root,"artifacts",SparkQuantArmArtifactMembers,SPARK_QUANT_ARM_COUNT(SparkQuantArmArtifactMembers),&object,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmHex(document,SparkJsonFindObjectMember(document,object,"module_archive_sha256"),"module_archive_sha256",arm->module_archive_sha256,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmHex(document,SparkJsonFindObjectMember(document,object,"driver_sha256"),"driver_sha256",arm->driver_sha256,error_buffer,error_buffer_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmHex(document,SparkJsonFindObjectMember(document,object,"adapter_sha256"),"adapter_sha256",arm->adapter_sha256,error_buffer,error_buffer_bytes);
	return(status);
}

void SparkQuantArmReset(SparkQuantArm *arm)
{
	if ( arm != 0 )
		memset(arm,0,sizeof(*arm));
}

static SparkStatus SparkQuantArmFinish(SparkQuantArm *arm,char *error_buffer,uint32_t error_buffer_bytes)
{
	char expected[SPARK_QUANT_ARM_ID_BYTES];
	char *canonical;
	SparkQuantArmWriter writer;
	uint32_t canonical_bytes = 0u;
	SparkStatus status = SparkQuantArmValidateAxes(arm,error_buffer,error_buffer_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( SparkQuantArmExpectedId(arm,expected,(uint32_t)sizeof(expected)) != SPARK_STATUS_OK || strcmp(expected,arm->arm_id) != 0 )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_SCHEMA_ERROR,"arm: arm_id %s does not match its fields (expected %s)",arm->arm_id,expected));
	canonical = (char *)malloc(SPARK_QUANT_ARM_CANONICAL_BYTES);
	if ( canonical == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkQuantArmCanonicalize(arm,canonical,SPARK_QUANT_ARM_CANONICAL_BYTES,&canonical_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkSha256Bytes(canonical,canonical_bytes,arm->arm_digest);
	if ( status == SPARK_STATUS_OK )
	{
		writer.buffer = canonical;
		writer.capacity = SPARK_QUANT_ARM_CANONICAL_BYTES;
		writer.length = 0u;
		writer.overflow = 0u;
		SparkQuantArmAppendHexArray(&writer,(const char (*)[SPARK_SHA256_HEX_BYTES])arm->pack_sha256,arm->rank_count);
		status = writer.overflow != 0u ? SPARK_STATUS_CAPACITY_EXCEEDED : SparkSha256Bytes(canonical,writer.length,arm->pack_set_sha256);
	}
	free(canonical);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	(void)snprintf(arm->kv_text,sizeof(arm->kv_text),"%s/%s/%s/%u/%s",arm->kv_latent,arm->kv_index,arm->kv_state,arm->kv_group,arm->kv_mode);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkQuantArmParseText(const char *text,size_t text_bytes,SparkQuantArm *arm,char *error_buffer,uint32_t error_buffer_bytes)
{
	SparkJsonDocument document;
	SparkStatus status;
	if ( error_buffer != 0 && error_buffer_bytes != 0u )
		error_buffer[0] = '\0';
	if ( text == 0 || arm == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkQuantArmReset(arm);
	if ( text_bytes > SPARK_QUANT_ARM_MAX_FILE_BYTES )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_CAPACITY_EXCEEDED,"arm: %zu bytes exceeds the %u byte limit",text_bytes,SPARK_QUANT_ARM_MAX_FILE_BYTES));
	SparkJsonDocumentReset(&document);
	if ( SparkJsonParseText(text,text_bytes,&document) != SPARK_STATUS_OK )
	{
		SparkJsonDocumentDestroy(&document);
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_PARSE_ERROR,"arm: not valid JSON (duplicate members are refused)"));
	}
	status = SparkQuantArmParseDocument(&document,arm,error_buffer,error_buffer_bytes);
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkQuantArmFinish(arm,error_buffer,error_buffer_bytes);
	if ( status != SPARK_STATUS_OK )
		SparkQuantArmReset(arm);
	return(status);
}

SparkStatus SparkQuantArmLoadFile(const char *path,SparkQuantArm *arm,char *error_buffer,uint32_t error_buffer_bytes)
{
	FILE *file;
	char *text;
	long length;
	size_t read_bytes;
	SparkStatus status;
	if ( path == 0 || arm == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	file = fopen(path,"rb");
	if ( file == 0 )
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_IO_ERROR,"arm: cannot open %s",path));
	if ( fseek(file,0,SEEK_END) != 0 || (length = ftell(file)) < 0 || fseek(file,0,SEEK_SET) != 0 )
	{
		fclose(file);
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_IO_ERROR,"arm: cannot size %s",path));
	}
	if ( (unsigned long)length > SPARK_QUANT_ARM_MAX_FILE_BYTES )
	{
		fclose(file);
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_CAPACITY_EXCEEDED,"arm: %s exceeds the %u byte limit",path,SPARK_QUANT_ARM_MAX_FILE_BYTES));
	}
	text = (char *)malloc((size_t)length + 1u);
	if ( text == 0 )
	{
		fclose(file);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	read_bytes = fread(text,1u,(size_t)length,file);
	fclose(file);
	if ( read_bytes != (size_t)length )
	{
		free(text);
		return(SparkQuantArmRefuse(error_buffer,error_buffer_bytes,SPARK_STATUS_IO_ERROR,"arm: short read of %s",path));
	}
	text[length] = '\0';
	status = SparkQuantArmParseText(text,(size_t)length,arm,error_buffer,error_buffer_bytes);
	free(text);
	return(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_sha256.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_manifest.h"

#define MANIFEST_LINE_BYTES (SPARK_DEVICE_KERNEL_NAME_LIMIT + 1024u)
#define MANIFEST_TABLE 16u
#define MANIFEST_AXES 3u

#define MANIFEST_TRY(call) \
	do { \
		const SparkStatus manifest_status = (call); \
		if ( manifest_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(manifest_status); \
	} while (0)

static SparkStatus ManifestRefuse(SparkStatus code, const char *owner, const char *line, const char *what)
{
	char text[128];
	snprintf(text,sizeof(text),"kernel manifest line \"%.40s\": %s",line,what);
	{
		const SparkStatus status = SparkDeviceRecordError(code,"library_load",owner,0,text);
		SPARK_FAIL(status);
	}
}

static SparkStatus ManifestGrow(void **table, uint32_t *capacity, uint32_t count, size_t element, const char *owner)
{
	void *grown;
	uint32_t wanted;
	if ( count < *capacity )
		return(SPARK_STATUS_OK);
	wanted = *capacity != 0u ? *capacity * 2u : MANIFEST_TABLE;
	grown = realloc(*table,(size_t)wanted * element);
	if ( grown == 0 )
		return(ManifestRefuse(SPARK_STATUS_CAPACITY_EXCEEDED,owner,"","out of host memory for the kernel manifest"));
	*table = grown;
	*capacity = wanted;
	return(SPARK_STATUS_OK);
}

static SparkStatus ManifestName(const char *line, const char *keyword, const char *owner, const char **name, size_t *length)
{
	const char *at = line + strlen(keyword);
	const char *end = strchr(at,' ');
	if ( end == 0 || end == at )
		return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"the line needs a kernel name"));
	if ( (size_t)(end - at) > SPARK_DEVICE_KERNEL_NAME_LIMIT )
		return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"the kernel name is longer than a device kernel name can be"));
	*name = at;
	*length = (size_t)(end - at);
	return(SPARK_STATUS_OK);
}

static SparkStatus ManifestKernel(SparkNvrmManifest *manifest, const char *line, const char *owner)
{
	SparkNvrmManifestKernel *kernel;
	const char *name = 0;
	char *copy;
	size_t length = 0u;
	unsigned registers,barriers,stack,stores,loads,shared;
	MANIFEST_TRY(ManifestName(line,"kernel ",owner,&name,&length));
	if ( sscanf(name + length," registers %u barriers %u stack %u spill_stores %u spill_loads %u smem %u",&registers,&barriers,&stack,&stores,&loads,&shared) != 6 )
		return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"a kernel line needs its name and the compiler's seven numbers"));
	if ( (copy = (char *)malloc(length + 1u)) == 0 )
		return(ManifestRefuse(SPARK_STATUS_CAPACITY_EXCEEDED,owner,line,"out of host memory for the kernel name"));
	memcpy(copy,name,length);
	copy[length] = '\0';
	if ( SparkNvrmManifestKernelNamed(manifest,copy) != 0 )
	{
		free(copy);
		return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"the kernel is listed twice"));
	}
	{
		const SparkStatus status = ManifestGrow((void **)&manifest->kernels,&manifest->kernel_capacity,manifest->kernel_count,sizeof(*manifest->kernels),owner);
		if ( status != SPARK_STATUS_OK )
		{
			free(copy);
			SPARK_FAIL(status);
		}
	}
	kernel = &manifest->kernels[manifest->kernel_count++];
	memset(kernel,0,sizeof(*kernel));
	kernel->name = copy;
	kernel->registers = registers;
	kernel->barriers = barriers;
	kernel->stack_bytes = stack;
	kernel->shared_bytes = shared;
	return(SPARK_STATUS_OK);
}

static SparkNvrmManifestKernel *ManifestCurrent(SparkNvrmManifest *manifest, const char *name, size_t length)
{
	SparkNvrmManifestKernel *kernel = manifest->kernel_count != 0u ? &manifest->kernels[manifest->kernel_count - 1u] : 0;
	return(kernel != 0 && strlen(kernel->name) == length && memcmp(kernel->name,name,length) == 0 ? kernel : 0);
}

static SparkStatus ManifestParameter(SparkNvrmManifest *manifest, const char *line, const char *owner)
{
	SparkNvrmManifestKernel *kernel;
	SparkNvrmParameter *parameter;
	const char *name = 0;
	char kind[16];
	size_t length = 0u;
	unsigned index,align,bytes,offset;
	MANIFEST_TRY(ManifestName(line,"param ",owner,&name,&length));
	if ( sscanf(name + length," %u %15s align %u bytes %u offset %u",&index,kind,&align,&bytes,&offset) != 5 )
		return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"a parameter line needs its kernel, index, type, alignment, size and offset"));
	kernel = ManifestCurrent(manifest,name,length);
	if ( kernel == 0 || index != kernel->parameter_count )
		return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"parameters follow their kernel's line in declaration order"));
	if ( align == 0u || (align & (align - 1u)) != 0u || offset % align != 0u || bytes == 0u || offset < SparkNvrmManifestParameterEnd(kernel) )
		return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"a parameter needs a power-of-two alignment, a size and an aligned offset past the one before"));
	MANIFEST_TRY(ManifestGrow((void **)&kernel->parameters,&kernel->parameter_capacity,kernel->parameter_count,sizeof(*kernel->parameters),owner));
	parameter = &kernel->parameters[kernel->parameter_count++];
	parameter->align = align;
	parameter->bytes = bytes;
	parameter->offset = offset;
	return(SPARK_STATUS_OK);
}

static SparkStatus ManifestBound(SparkNvrmManifest *manifest, const char *line, const char *owner)
{
	SparkNvrmManifestKernel *kernel;
	const char *name = 0;
	char directive[32];
	size_t length = 0u;
	unsigned values[MANIFEST_AXES] = {1u, 1u, 1u};
	int consumed = 0;
	MANIFEST_TRY(ManifestName(line,"bound ",owner,&name,&length));
	if ( sscanf(name + length," %31s%n",directive,&consumed) != 1 || (kernel = ManifestCurrent(manifest,name,length)) == 0 )
		return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"a bound line follows its kernel's line"));
	if ( strcmp(directive,"maxntid") == 0 || strcmp(directive,"reqntid") == 0 )
	{
		if ( sscanf(name + length + consumed,"%u %u %u",&values[0],&values[1],&values[2]) < 1 || values[0] == 0u || values[1] == 0u || values[2] == 0u )
			return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"a thread bound needs one to three nonzero sizes"));
		kernel->max_threads = values[0] * values[1] * values[2];
	}
	else if ( strcmp(directive,"reqnctapercluster") == 0 || strcmp(directive,"explicitcluster") == 0 || strcmp(directive,"maxclusterrank") == 0 )
		kernel->clustered = 1u;
	else if ( strcmp(directive,"minnctapersm") != 0 && strcmp(directive,"maxnreg") != 0 )
		return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"the bound is not a PTX launch-bound or cluster directive"));
	return(SPARK_STATUS_OK);
}

static uint32_t ManifestStarts(const char *line, const char *prefix)
{
	return(strncmp(line,prefix,strlen(prefix)) == 0 ? 1u : 0u);
}

static SparkStatus ManifestLine(SparkNvrmManifest *manifest, const char *line, const char *owner)
{
	unsigned long long cubin_bytes = 0u;
	if ( line[0] == '\0' || ManifestStarts(line,"nvcc ") || ManifestStarts(line,"flags ") || ManifestStarts(line,"source ") )
		return(SPARK_STATUS_OK);
	if ( ManifestStarts(line,"cubin ") )
	{
		if ( sscanf(line,"cubin %64s %llu",manifest->cubin_digest,&cubin_bytes) != 2 || strlen(manifest->cubin_digest) != SPARK_NVRM_MANIFEST_DIGEST_BYTES - 1u )
			return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"the cubin line needs its sha256 and size"));
		manifest->cubin_bytes = cubin_bytes;
		return(SPARK_STATUS_OK);
	}
	if ( ManifestStarts(line,"kernel ") )
		return(ManifestKernel(manifest,line,owner));
	if ( ManifestStarts(line,"param ") )
		return(ManifestParameter(manifest,line,owner));
	if ( ManifestStarts(line,"bound ") )
		return(ManifestBound(manifest,line,owner));
	return(ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,line,"the line is not part of a kernel manifest"));
}

SparkStatus SparkNvrmManifestParse(const char *text, uint64_t bytes, const char *owner, SparkNvrmManifest *manifest)
{
	char line[MANIFEST_LINE_BYTES];
	uint64_t at = 0u;
	memset(manifest,0,sizeof(*manifest));
	while ( at < bytes )
	{
		uint64_t end = at;
		SparkStatus status;
		while ( end < bytes && text[end] != '\n' )
			end++;
		if ( end - at >= sizeof(line) )
			status = ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,text + at,"the line is longer than a kernel manifest line can be");
		else
		{
			memcpy(line,text + at,(size_t)(end - at));
			line[end - at] = '\0';
			status = ManifestLine(manifest,line,owner);
		}
		if ( status != SPARK_STATUS_OK )
		{
			SparkNvrmManifestRelease(manifest);
			SPARK_FAIL(status);
		}
		at = end + 1u;
	}
	if ( manifest->cubin_digest[0] == '\0' || manifest->kernel_count == 0u )
	{
		const SparkStatus status = ManifestRefuse(SPARK_STATUS_PARSE_ERROR,owner,"","a kernel manifest names its cubin and at least one kernel");
		SparkNvrmManifestRelease(manifest);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

const SparkNvrmManifestKernel *SparkNvrmManifestKernelNamed(const SparkNvrmManifest *manifest, const char *name)
{
	uint32_t index;
	for (index = 0u; index < manifest->kernel_count; index++)
		if ( strcmp(manifest->kernels[index].name,name) == 0 )
			return(&manifest->kernels[index]);
	return(0);
}

uint32_t SparkNvrmManifestParameterEnd(const SparkNvrmManifestKernel *kernel)
{
	const SparkNvrmParameter *last = kernel->parameter_count != 0u ? &kernel->parameters[kernel->parameter_count - 1u] : 0;
	return(last != 0 ? last->offset + last->bytes : 0u);
}

SparkStatus SparkNvrmManifestCheck(const SparkNvrmManifest *manifest, const uint8_t *blob, uint64_t bytes, const SparkNvrmCubin *cubin, const char *owner)
{
	char digest[SPARK_SHA256_HEX_BYTES];
	uint32_t index;
	if ( SparkSha256Bytes(blob,(size_t)bytes,digest) != SPARK_STATUS_OK || bytes != manifest->cubin_bytes || strcmp(digest,manifest->cubin_digest) != 0 )
		return(ManifestRefuse(SPARK_STATUS_TARGET_MISMATCH,owner,"","the manifest describes another cubin than the one loaded"));
	if ( cubin->kernel_count != manifest->kernel_count )
		return(ManifestRefuse(SPARK_STATUS_TARGET_MISMATCH,owner,"","the manifest and the cubin list different kernels"));
	for (index = 0u; index < cubin->kernel_count; index++)
	{
		const SparkNvrmCubinKernel *kernel = &cubin->kernels[index];
		const SparkNvrmManifestKernel *reported = SparkNvrmManifestKernelNamed(manifest,kernel->name);
		if ( reported == 0 )
			return(ManifestRefuse(SPARK_STATUS_TARGET_MISMATCH,owner,kernel->name,"the cubin has a kernel the manifest does not list"));
		if ( reported->registers != kernel->registers || reported->stack_bytes != kernel->stack_bytes ||
			SparkNvrmManifestParameterEnd(reported) != kernel->parameter_bytes )
			return(ManifestRefuse(SPARK_STATUS_TARGET_MISMATCH,owner,kernel->name,
				"the compiler's registers, stack or parameter table disagree with the cubin"));
	}
	return(SPARK_STATUS_OK);
}

void SparkNvrmManifestRelease(SparkNvrmManifest *manifest)
{
	uint32_t index;
	for (index = 0u; index < manifest->kernel_count; index++)
	{
		free(manifest->kernels[index].name);
		free(manifest->kernels[index].parameters);
	}
	free(manifest->kernels);
	memset(manifest,0,sizeof(*manifest));
}

#include <dirent.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__linux__)
#include <link.h>
#endif

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_launch.h"
#include "runtime/device/nvrm/nvrm_manifest.h"
#include "runtime/device/nvrm/nvrm_names.h"
#include "runtime/device/nvrm/nvrm_register.h"
#include "runtime/device/nvrm/spark_gpu_nvrm.h"

#define REGISTER_PATH_BYTES 4096u
#define REGISTER_TABLE_START 1024u
#define REGISTER_GROW_START 16u
#define REGISTER_CUBIN_SUFFIX ".cubin"
#define REGISTER_MANIFEST_SUFFIX ".manifest"
#define REGISTER_OWNER "registered kernels"
#define REGISTER_PROVIDER_ENV "SPARKPIPE_KERNEL_PROVIDER"
#define REGISTER_OVERLAY_DIR_ENV "SPARKPIPE_SKL_CUBIN_DIR"
#define REGISTER_OVERLAY_STATS_ENV "SPARKPIPE_SKL_OVERLAY_STATS"

typedef struct RegisterCubin
{
	char *path;
	SparkDeviceLibrary *library;
} RegisterCubin;

typedef struct RegisterName
{
	char name[SPARK_NVRM_SHORT_NAME_BYTES];
	uint32_t cubin;
	uint32_t overlay_cubin;
	uint32_t overlay_index;
	uint32_t overlaid;
	const char *file;
} RegisterName;

typedef struct RegisterOverlayName
{
	char name[SPARK_NVRM_SHORT_NAME_BYTES];
	uint32_t cubin;
	const char *file;
	SparkNvrmParameter *parameters;
	uint32_t parameter_count;
	uint32_t parameter_bytes;
	uint32_t max_threads;
	uint32_t clustered;
	uint32_t cluster[3];
	uint32_t cluster_limit;
	uint64_t direct_launches;
	uint64_t program_launches;
} RegisterOverlayName;

typedef struct RegisterOverlay
{
	uint32_t loaded;
	uint32_t active;
	uint32_t stats_enabled;
	RegisterCubin *cubins;
	uint32_t cubin_count;
	uint32_t cubin_capacity;
	RegisterOverlayName *names;
	uint32_t name_count;
	uint32_t name_capacity;
	SparkStatus status;
	char failure[SPARK_NVRM_REGISTER_FAILURE_BYTES];
} RegisterOverlay;

struct SparkNvrmModule
{
	char *path;
	RegisterCubin *cubins;
	uint32_t cubin_count;
	uint32_t cubin_capacity;
	RegisterName *names;
	uint32_t name_count;
	uint32_t name_capacity;
	uint32_t binaries;
	SparkStatus status;
	char failure[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	SparkNvrmModule *next;
};

typedef struct RegisterEntry
{
	const void *host;
	SparkNvrmModule *module;
	char name[SPARK_NVRM_SHORT_NAME_BYTES];
	uint32_t cubin;
	uint32_t overlay_index;
	uint32_t overlaid;
	uint32_t variable;
	SparkDeviceKernel *kernel;
	SparkStatus status;
	char failure[SPARK_NVRM_REGISTER_FAILURE_BYTES];
} RegisterEntry;

typedef struct RegisterBinary
{
	SparkNvrmModule *module;
} RegisterBinary;

static pthread_mutex_t register_mutex = PTHREAD_MUTEX_INITIALIZER;
static SparkNvrmModule *register_modules;
static RegisterEntry **register_table;
static uint32_t register_capacity;
static uint32_t register_count;
static RegisterOverlay register_overlay;
static uint32_t register_overlay_atexit;

static SparkStatus RegisterRefuse(SparkStatus code, const char *operation, const char *owner, const char *text)
{
	const SparkStatus status = SparkDeviceRecordError(code,operation,owner,0,text);
	SPARK_FAIL(status);
}

static SparkStatus ModuleFail(SparkNvrmModule *module, SparkStatus code, const char *text)
{
	module->status = code;
	snprintf(module->failure,sizeof(module->failure),"%s",text);
	fprintf(stderr,"nvrm: kernel module %s: %s\n",module->path,text);
	return(RegisterRefuse(code,"kernel_register",REGISTER_OWNER,text));
}

static SparkStatus RegisterGrow(void **table, uint32_t *capacity, uint32_t count, size_t element)
{
	void *grown;
	uint32_t wanted;
	if ( count < *capacity )
		return(SPARK_STATUS_OK);
	wanted = *capacity != 0u ? *capacity * 2u : REGISTER_GROW_START;
	if ( (grown = realloc(*table,(size_t)wanted * element)) == 0 )
		return(RegisterRefuse(SPARK_STATUS_CAPACITY_EXCEEDED,"kernel_register",REGISTER_OWNER,"out of host memory for the kernel registry"));
	*table = grown;
	*capacity = wanted;
	return(SPARK_STATUS_OK);
}

static uint32_t RegisterEnds(const char *text, const char *suffix)
{
	const size_t length = strlen(text),tail = strlen(suffix);
	return(length > tail && strcmp(text + length - tail,suffix) == 0 ? 1u : 0u);
}

static int RegisterNameOrder(const void *left, const void *right)
{
	return(strcmp(((const RegisterName *)left)->name,((const RegisterName *)right)->name));
}

static int RegisterNameFileOrder(const void *left, const void *right)
{
	const int name = RegisterNameOrder(left,right);
	return(name != 0 ? name : strcmp(((const RegisterName *)left)->file,((const RegisterName *)right)->file));
}

static int RegisterOverlayNameOrder(const void *left, const void *right)
{
	return(strcmp(((const RegisterOverlayName *)left)->name,((const RegisterOverlayName *)right)->name));
}

static int RegisterOverlayNameFileOrder(const void *left, const void *right)
{
	const int name = RegisterOverlayNameOrder(left,right);
	return(name != 0 ? name : strcmp(((const RegisterOverlayName *)left)->file,((const RegisterOverlayName *)right)->file));
}

static SparkStatus RegisterOverlayFail(SparkStatus code, const char *text)
{
	register_overlay.status = code;
	snprintf(register_overlay.failure,sizeof(register_overlay.failure),"%s",text);
	fprintf(stderr,"nvrm: skl-overlay: %s\n",text);
	return(RegisterRefuse(code,"kernel_register",REGISTER_OWNER,text));
}

static const RegisterOverlayName *RegisterOverlayFind(const char *short_name)
{
	RegisterOverlayName key;
	if ( register_overlay.active == 0u || register_overlay.name_count == 0u )
		return(0);
	memcpy(key.name,short_name,sizeof(key.name));
	return((const RegisterOverlayName *)bsearch(&key,register_overlay.names,register_overlay.name_count,sizeof(*register_overlay.names),
		RegisterOverlayNameOrder));
}

static int RegisterOverlayAbiCheck(const SparkNvrmManifestKernel *base, const RegisterOverlayName *overlay, char *text, size_t capacity)
{
	const uint32_t base_bytes = SparkNvrmManifestParameterEnd(base);
	uint32_t index;
	if ( base->parameter_count != overlay->parameter_count || base_bytes != overlay->parameter_bytes )
	{
		snprintf(text,capacity,"skl-overlay kernel %.48s ABI mismatch: overlay has %u params in %u bytes, base expects %u in %u",base->name,
			overlay->parameter_count,overlay->parameter_bytes,base->parameter_count,base_bytes);
		return(0);
	}
	for (index = 0u; index < base->parameter_count; index++)
		if ( base->parameters[index].offset != overlay->parameters[index].offset ||
			base->parameters[index].bytes != overlay->parameters[index].bytes ||
			base->parameters[index].align != overlay->parameters[index].align )
		{
			snprintf(text,capacity,"skl-overlay kernel %.48s param %u ABI mismatch: overlay %uB@%u(a%u), base %uB@%u(a%u)",base->name,index,
				overlay->parameters[index].bytes,overlay->parameters[index].offset,overlay->parameters[index].align,
				base->parameters[index].bytes,base->parameters[index].offset,base->parameters[index].align);
			return(0);
		}
	if ( base->clustered != 0u && (overlay->clustered == 0u || memcmp(base->cluster,overlay->cluster,sizeof(base->cluster)) != 0 ||
		(base->cluster_limit != 0u && (overlay->cluster_limit == 0u || overlay->cluster_limit > base->cluster_limit))) )
	{
		snprintf(text,capacity,"skl-overlay kernel %.48s cluster ABI mismatch with base kernel",base->name);
		return(0);
	}
	return(1);
}

static SparkStatus RegisterCubinAdd(SparkNvrmModule *module, const char *directory, const char *file)
{
	char path[REGISTER_PATH_BYTES],manifest_path[REGISTER_PATH_BYTES],text[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	const size_t stem = strlen(file) - (sizeof(REGISTER_CUBIN_SUFFIX) - 1u);
	SparkNvrmManifest manifest;
	uint64_t bytes = 0u;
	uint32_t index;
	char *contents;
	if ( snprintf(path,sizeof(path),"%s/%s",directory,file) >= (int)sizeof(path) ||
		snprintf(manifest_path,sizeof(manifest_path),"%s/%.*s%s",directory,(int)stem,file,REGISTER_MANIFEST_SUFFIX) >= (int)sizeof(manifest_path) )
		return(ModuleFail(module,SPARK_STATUS_INVALID_ARGUMENT,"a kernel path is longer than 4096 bytes"));
	if ( (contents = (char *)SparkNvrmReadFile(manifest_path,&bytes)) == 0 )
	{
		snprintf(text,sizeof(text),"the cubin %.80s has no manifest beside it",file);
		return(ModuleFail(module,SPARK_STATUS_NOT_FOUND,text));
	}
	if ( SparkNvrmManifestParse(contents,bytes,REGISTER_OWNER,&manifest) != SPARK_STATUS_OK )
	{
		free(contents);
		snprintf(text,sizeof(text),"%.60s: %.90s",file,SparkDeviceLastError()->text);
		return(ModuleFail(module,SPARK_STATUS_PARSE_ERROR,text));
	}
	free(contents);
	if ( RegisterGrow((void **)&module->cubins,&module->cubin_capacity,module->cubin_count,sizeof(*module->cubins)) != SPARK_STATUS_OK ||
		(module->cubins[module->cubin_count].path = strdup(path)) == 0 )
	{
		SparkNvrmManifestRelease(&manifest);
		return(ModuleFail(module,SPARK_STATUS_CAPACITY_EXCEEDED,"out of host memory for the kernel registry"));
	}
	module->cubins[module->cubin_count].library = 0;
	for (index = 0u; index < manifest.kernel_count; index++)
	{
		RegisterName *slot;
		const RegisterOverlayName *overlay = RegisterOverlayFind(manifest.kernels[index].name);
		if ( overlay != 0 && RegisterOverlayAbiCheck(&manifest.kernels[index],overlay,text,sizeof(text)) == 0 )
		{
			SparkNvrmManifestRelease(&manifest);
			return(ModuleFail(module,SPARK_STATUS_TARGET_MISMATCH,text));
		}
		if ( RegisterGrow((void **)&module->names,&module->name_capacity,module->name_count,sizeof(*module->names)) != SPARK_STATUS_OK )
		{
			SparkNvrmManifestRelease(&manifest);
			return(ModuleFail(module,SPARK_STATUS_CAPACITY_EXCEEDED,"out of host memory for the kernel registry"));
		}
		slot = &module->names[module->name_count++];
		memcpy(slot->name,manifest.kernels[index].name,sizeof(slot->name));
		slot->cubin = module->cubin_count;
		slot->overlay_cubin = overlay != 0 ? overlay->cubin : 0u;
		slot->overlay_index = overlay != 0 ? (uint32_t)(overlay - register_overlay.names) : 0u;
		slot->overlaid = overlay != 0 ? 1u : 0u;
		slot->file = module->cubins[module->cubin_count].path;
		if ( overlay != 0 && register_overlay.stats_enabled != 0u )
			fprintf(stderr,"nvrm: skl-overlay: bound %.64s -> %s\n",slot->name,overlay->file);
	}
	module->cubin_count++;
	SparkNvrmManifestRelease(&manifest);
	return(SPARK_STATUS_OK);
}

static SparkStatus RegisterCubinRead(const char *path, SparkNvrmCubin *cubin)
{
	uint64_t bytes = 0u;
	uint8_t *blob = SparkNvrmReadFile(path,&bytes);
	SparkStatus status;
	memset(cubin,0,sizeof(*cubin));
	if ( blob == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	status = SparkNvrmCubinParse(blob,bytes,REGISTER_OWNER,cubin);
	free(blob);
	return(status);
}

static uint32_t RegisterWithin(uint64_t offset, uint64_t start, uint64_t bytes)
{
	return(offset >= start && offset - start < bytes ? 1u : 0u);
}

static uint32_t RegisterRelocated(const SparkNvrmCubin *cubin, uint64_t start, uint64_t bytes)
{
	uint32_t index;
	for (index = 0u; index < cubin->relocation_count; index++)
		if ( RegisterWithin(cubin->relocations[index].image_offset,start,bytes) != 0u )
			return(1u);
	return(0u);
}

static const char *RegisterCopiesDiffer(const SparkNvrmCubin *first, const SparkNvrmCubinKernel *a, const SparkNvrmCubin *second,
	const SparkNvrmCubinKernel *b)
{
	uint32_t bank;
	if ( a == 0 || b == 0 )
		return("one cubin does not hold it");
	if ( a->code_bytes != b->code_bytes || memcmp(first->image + a->code_offset,second->image + b->code_offset,(size_t)a->code_bytes) != 0 )
		return("the code differs");
	if ( a->registers != b->registers || a->stack_bytes != b->stack_bytes || a->frame_bytes != b->frame_bytes || a->shared_bytes != b->shared_bytes ||
		a->parameter_offset != b->parameter_offset || a->parameter_bytes != b->parameter_bytes || a->barriers != b->barriers ||
		a->clustered != b->clustered || memcmp(a->cluster,b->cluster,sizeof(a->cluster)) != 0 || a->cluster_limit != b->cluster_limit )
		return("the attributes differ");
	if ( a->bank[0].present != b->bank[0].present || a->bank[0].bytes != b->bank[0].bytes )
		return("the parameter banks differ");
	if ( RegisterRelocated(first,a->code_offset,a->code_bytes) != 0u || RegisterRelocated(second,b->code_offset,b->code_bytes) != 0u )
		return("the code is relocated against its own cubin's data");
	for (bank = 1u; bank < SPARK_NVRM_CUBIN_BANKS; bank++)
	{
		const SparkNvrmCubinBank *mine = &a->bank[bank],*theirs = &b->bank[bank];
		if ( mine->present == 0u || theirs->present == 0u )
			continue;
		if ( mine->bytes != theirs->bytes || memcmp(first->image + mine->image_offset,second->image + theirs->image_offset,(size_t)mine->bytes) != 0 )
			return("a constant bank both copies have differs");
		if ( RegisterRelocated(first,mine->image_offset,mine->bytes) != 0u || RegisterRelocated(second,theirs->image_offset,theirs->bytes) != 0u )
			return("a constant bank both copies have is relocated");
	}
	return(0);
}

static const char *RegisterBase(const char *path)
{
	const char *slash = strrchr(path,'/');
	return(slash != 0 ? slash + 1 : path);
}

typedef struct RegisterParsedCubin
{
	uint32_t state;
	SparkNvrmCubin cubin;
} RegisterParsedCubin;

static const SparkNvrmCubin *RegisterCachedCubin(const RegisterCubin *cubins, RegisterParsedCubin *cache, uint32_t index)
{
	if ( cache[index].state == 0u )
		cache[index].state = RegisterCubinRead(cubins[index].path,&cache[index].cubin) == SPARK_STATUS_OK ? 1u : 2u;
	return(cache[index].state == 1u ? &cache[index].cubin : 0);
}

static void RegisterCachedCubinsRelease(RegisterParsedCubin *cache, uint32_t count)
{
	uint32_t index;
	if ( cache == 0 )
		return;
	for (index = 0u; index < count; index++)
		if ( cache[index].state == 1u )
			SparkNvrmCubinRelease(&cache[index].cubin);
	free(cache);
}

static const char *RegisterSameCubinPair(const RegisterCubin *cubins, RegisterParsedCubin *cache, uint32_t first_idx, uint32_t second_idx,
	const char *name)
{
	const SparkNvrmCubin *first = RegisterCachedCubin(cubins,cache,first_idx);
	const SparkNvrmCubin *second = RegisterCachedCubin(cubins,cache,second_idx);
	if ( first == 0 || second == 0 )
		return("a cubin does not parse");
	return(RegisterCopiesDiffer(first,SparkNvrmCubinKernelNamed(first,name),second,SparkNvrmCubinKernelNamed(second,name)));
}

static SparkStatus RegisterSameKernel(SparkNvrmModule *module, RegisterParsedCubin *cache, const RegisterName *kept, const RegisterName *again)
{
	char text[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	const char *differ = RegisterSameCubinPair(module->cubins,cache,kept->cubin,again->cubin,kept->name);
	if ( differ == 0 )
		return(SPARK_STATUS_OK);
	snprintf(text,sizeof(text),"kernel %.64s is in two cubins of the module, %.32s and %.32s, and %s",kept->name,
		RegisterBase(module->cubins[kept->cubin].path),RegisterBase(module->cubins[again->cubin].path),differ);
	return(ModuleFail(module,SPARK_STATUS_DUPLICATE,text));
}

static SparkStatus RegisterOverlayCubinAdd(const char *directory, const char *file)
{
	char path[REGISTER_PATH_BYTES],manifest_path[REGISTER_PATH_BYTES],text[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	const size_t stem = strlen(file) - (sizeof(REGISTER_CUBIN_SUFFIX) - 1u);
	SparkNvrmManifest manifest;
	uint64_t bytes = 0u;
	uint32_t index;
	char *contents;
	if ( snprintf(path,sizeof(path),"%s/%s",directory,file) >= (int)sizeof(path) ||
		snprintf(manifest_path,sizeof(manifest_path),"%s/%.*s%s",directory,(int)stem,file,REGISTER_MANIFEST_SUFFIX) >= (int)sizeof(manifest_path) )
		return(RegisterOverlayFail(SPARK_STATUS_INVALID_ARGUMENT,"an overlay kernel path is longer than 4096 bytes"));
	if ( (contents = (char *)SparkNvrmReadFile(manifest_path,&bytes)) == 0 )
	{
		snprintf(text,sizeof(text),"the overlay cubin %.80s has no manifest beside it",file);
		return(RegisterOverlayFail(SPARK_STATUS_NOT_FOUND,text));
	}
	if ( SparkNvrmManifestParse(contents,bytes,REGISTER_OWNER,&manifest) != SPARK_STATUS_OK )
	{
		free(contents);
		snprintf(text,sizeof(text),"%.60s: %.90s",file,SparkDeviceLastError()->text);
		return(RegisterOverlayFail(SPARK_STATUS_PARSE_ERROR,text));
	}
	free(contents);
	if ( RegisterGrow((void **)&register_overlay.cubins,&register_overlay.cubin_capacity,register_overlay.cubin_count,
		sizeof(*register_overlay.cubins)) != SPARK_STATUS_OK ||
		(register_overlay.cubins[register_overlay.cubin_count].path = strdup(path)) == 0 )
	{
		SparkNvrmManifestRelease(&manifest);
		return(RegisterOverlayFail(SPARK_STATUS_CAPACITY_EXCEEDED,"out of host memory for the overlay registry"));
	}
	register_overlay.cubins[register_overlay.cubin_count].library = 0;
	for (index = 0u; index < manifest.kernel_count; index++)
	{
		const SparkNvrmManifestKernel *src = &manifest.kernels[index];
		RegisterOverlayName *dst;
		if ( RegisterGrow((void **)&register_overlay.names,&register_overlay.name_capacity,register_overlay.name_count,
			sizeof(*register_overlay.names)) != SPARK_STATUS_OK )
		{
			SparkNvrmManifestRelease(&manifest);
			return(RegisterOverlayFail(SPARK_STATUS_CAPACITY_EXCEEDED,"out of host memory for the overlay registry"));
		}
		dst = &register_overlay.names[register_overlay.name_count];
		memset(dst,0,sizeof(*dst));
		if ( src->parameter_count != 0u )
		{
			dst->parameters = (SparkNvrmParameter *)malloc((size_t)src->parameter_count * sizeof(*dst->parameters));
			if ( dst->parameters == 0 )
			{
				SparkNvrmManifestRelease(&manifest);
				return(RegisterOverlayFail(SPARK_STATUS_CAPACITY_EXCEEDED,"out of host memory for the overlay registry"));
			}
			memcpy(dst->parameters,src->parameters,(size_t)src->parameter_count * sizeof(*dst->parameters));
		}
		memcpy(dst->name,src->name,sizeof(dst->name));
		dst->cubin = register_overlay.cubin_count;
		dst->file = register_overlay.cubins[register_overlay.cubin_count].path;
		dst->parameter_count = src->parameter_count;
		dst->parameter_bytes = SparkNvrmManifestParameterEnd(src);
		dst->max_threads = src->max_threads;
		dst->clustered = src->clustered;
		memcpy(dst->cluster,src->cluster,sizeof(dst->cluster));
		dst->cluster_limit = src->cluster_limit;
		register_overlay.name_count++;
	}
	register_overlay.cubin_count++;
	SparkNvrmManifestRelease(&manifest);
	return(SPARK_STATUS_OK);
}

static void RegisterOverlayClear(void)
{
	uint32_t index;
	for (index = 0u; index < register_overlay.name_count; index++)
		free(register_overlay.names[index].parameters);
	free(register_overlay.names);
	for (index = 0u; index < register_overlay.cubin_count; index++)
	{
		if ( register_overlay.cubins[index].library != 0 )
			(void)SparkDeviceLibraryUnload(register_overlay.cubins[index].library);
		free(register_overlay.cubins[index].path);
	}
	free(register_overlay.cubins);
	memset(&register_overlay,0,sizeof(register_overlay));
}

static SparkStatus RegisterOverlayScan(const char *directory)
{
	char path[REGISTER_PATH_BYTES],text[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	SparkStatus status = SPARK_STATUS_OK;
	RegisterParsedCubin *cache = 0;
	struct dirent *entry;
	uint32_t index,kept;
	DIR *listing;
	if ( (listing = opendir(directory)) == 0 )
	{
		snprintf(text,sizeof(text),"overlay directory %.100s cannot be opened",directory);
		return(RegisterOverlayFail(SPARK_STATUS_NOT_FOUND,text));
	}
	while ( status == SPARK_STATUS_OK && (entry = readdir(listing)) != 0 )
	{
		if ( RegisterEnds(entry->d_name,REGISTER_CUBIN_SUFFIX) != 0u )
			status = RegisterOverlayCubinAdd(directory,entry->d_name);
		else if ( RegisterEnds(entry->d_name,REGISTER_MANIFEST_SUFFIX) != 0u )
		{
			const size_t stem = strlen(entry->d_name) - (sizeof(REGISTER_MANIFEST_SUFFIX) - 1u);
			if ( snprintf(path,sizeof(path),"%s/%.*s%s",directory,(int)stem,entry->d_name,REGISTER_CUBIN_SUFFIX) >= (int)sizeof(path) )
				status = RegisterOverlayFail(SPARK_STATUS_INVALID_ARGUMENT,"an overlay kernel path is longer than 4096 bytes");
			else if ( access(path,R_OK) != 0 )
			{
				snprintf(text,sizeof(text),"the overlay manifest %.80s has no cubin beside it",entry->d_name);
				status = RegisterOverlayFail(SPARK_STATUS_NOT_FOUND,text);
			}
		}
	}
	closedir(listing);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	if ( register_overlay.cubin_count == 0u )
	{
		snprintf(text,sizeof(text),"overlay directory %.100s holds no cubin",directory);
		return(RegisterOverlayFail(SPARK_STATUS_NOT_FOUND,text));
	}
	qsort(register_overlay.names,register_overlay.name_count,sizeof(*register_overlay.names),RegisterOverlayNameFileOrder);
	cache = (RegisterParsedCubin *)calloc(register_overlay.cubin_count,sizeof(*cache));
	if ( cache == 0 )
		return(RegisterOverlayFail(SPARK_STATUS_CAPACITY_EXCEEDED,"out of host memory for the overlay registry"));
	for (index = 0u, kept = 0u; index < register_overlay.name_count; index++)
	{
		if ( kept != 0u && strcmp(register_overlay.names[kept - 1u].name,register_overlay.names[index].name) == 0 )
		{
			const char *differ = RegisterSameCubinPair(register_overlay.cubins,cache,register_overlay.names[kept - 1u].cubin,
				register_overlay.names[index].cubin,register_overlay.names[kept - 1u].name);
			if ( differ != 0 )
			{
				snprintf(text,sizeof(text),"overlay kernel %.64s is in two cubins, %.32s and %.32s, and %s",
					register_overlay.names[kept - 1u].name,RegisterBase(register_overlay.cubins[register_overlay.names[kept - 1u].cubin].path),
					RegisterBase(register_overlay.cubins[register_overlay.names[index].cubin].path),differ);
				RegisterCachedCubinsRelease(cache,register_overlay.cubin_count);
				return(RegisterOverlayFail(SPARK_STATUS_DUPLICATE,text));
			}
			free(register_overlay.names[index].parameters);
			continue;
		}
		register_overlay.names[kept++] = register_overlay.names[index];
	}
	RegisterCachedCubinsRelease(cache,register_overlay.cubin_count);
	register_overlay.name_count = kept;
	register_overlay.active = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus RegisterOverlayEnsure(void)
{
	const char *provider,*directory,*stats;
	SparkStatus status = SPARK_STATUS_OK;
	if ( register_overlay.loaded != 0u )
		return(register_overlay.status);
	register_overlay.loaded = 1u;
	register_overlay.status = SPARK_STATUS_OK;
	provider = getenv(REGISTER_PROVIDER_ENV);
	directory = getenv(REGISTER_OVERLAY_DIR_ENV);
	stats = getenv(REGISTER_OVERLAY_STATS_ENV);
	if ( provider != 0 && provider[0] != '\0' )
	{
		if ( strcmp(provider,"default") == 0 )
			return(SPARK_STATUS_OK);
		if ( strcmp(provider,"skl-overlay") != 0 )
			return(RegisterOverlayFail(SPARK_STATUS_INVALID_ARGUMENT,"unknown SPARKPIPE_KERNEL_PROVIDER; expected default or skl-overlay"));
		if ( directory == 0 || directory[0] == '\0' )
			return(RegisterOverlayFail(SPARK_STATUS_INVALID_ARGUMENT,"SPARKPIPE_KERNEL_PROVIDER=skl-overlay requires SPARKPIPE_SKL_CUBIN_DIR"));
		status = RegisterOverlayScan(directory);
	}
	else if ( directory != 0 && directory[0] != '\0' )
		status = RegisterOverlayScan(directory);
	if ( status == SPARK_STATUS_OK && register_overlay.active != 0u && stats != 0 && stats[0] != '\0' && strcmp(stats,"0") != 0 )
	{
		__atomic_store_n(&register_overlay.stats_enabled,1u,__ATOMIC_RELEASE);
		if ( register_overlay_atexit == 0u )
		{
			register_overlay_atexit = 1u;
			(void)atexit(SparkNvrmOverlayStatsDump);
		}
	}
	return(status);
}

static SparkStatus RegisterScan(SparkNvrmModule *module)
{
	char directory[REGISTER_PATH_BYTES],path[REGISTER_PATH_BYTES],text[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	SparkStatus status = SPARK_STATUS_OK;
	RegisterParsedCubin *cache = 0;
	struct dirent *entry;
	uint32_t index,kept;
	DIR *listing;
	if ( RegisterOverlayEnsure() != SPARK_STATUS_OK )
		return(ModuleFail(module,register_overlay.status,register_overlay.failure));
	if ( snprintf(directory,sizeof(directory),"%s%s",module->path,SPARK_NVRM_KERNEL_DIRECTORY_SUFFIX) >= (int)sizeof(directory) )
		return(ModuleFail(module,SPARK_STATUS_INVALID_ARGUMENT,"the module path is longer than 4096 bytes"));
	if ( (listing = opendir(directory)) == 0 )
	{
		snprintf(text,sizeof(text),"has no kernel directory %.100s",directory);
		return(ModuleFail(module,SPARK_STATUS_NOT_FOUND,text));
	}
	while ( status == SPARK_STATUS_OK && (entry = readdir(listing)) != 0 )
	{
		if ( RegisterEnds(entry->d_name,REGISTER_CUBIN_SUFFIX) != 0u )
			status = RegisterCubinAdd(module,directory,entry->d_name);
		else if ( RegisterEnds(entry->d_name,REGISTER_MANIFEST_SUFFIX) != 0u )
		{
			const size_t stem = strlen(entry->d_name) - (sizeof(REGISTER_MANIFEST_SUFFIX) - 1u);
			if ( snprintf(path,sizeof(path),"%s/%.*s%s",directory,(int)stem,entry->d_name,REGISTER_CUBIN_SUFFIX) >= (int)sizeof(path) )
				status = ModuleFail(module,SPARK_STATUS_INVALID_ARGUMENT,"a kernel path is longer than 4096 bytes");
			else if ( access(path,R_OK) != 0 )
			{
				snprintf(text,sizeof(text),"the manifest %.80s has no cubin beside it",entry->d_name);
				status = ModuleFail(module,SPARK_STATUS_NOT_FOUND,text);
			}
		}
	}
	closedir(listing);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	if ( module->cubin_count == 0u )
	{
		snprintf(text,sizeof(text),"the kernel directory %.100s holds no cubin",directory);
		return(ModuleFail(module,SPARK_STATUS_NOT_FOUND,text));
	}
	qsort(module->names,module->name_count,sizeof(*module->names),RegisterNameFileOrder);
	for (index = 0u, kept = 0u; index < module->name_count; index++)
	{
		if ( kept != 0u && strcmp(module->names[kept - 1u].name,module->names[index].name) == 0 )
		{
			SparkStatus same;
			if ( cache == 0 && (cache = (RegisterParsedCubin *)calloc(module->cubin_count,sizeof(*cache))) == 0 )
				return(ModuleFail(module,SPARK_STATUS_CAPACITY_EXCEEDED,"out of host memory for the kernel registry"));
			same = RegisterSameKernel(module,cache,&module->names[kept - 1u],&module->names[index]);
			if ( same != SPARK_STATUS_OK )
			{
				RegisterCachedCubinsRelease(cache,module->cubin_count);
				return(same);
			}
			continue;
		}
		module->names[kept++] = module->names[index];
	}
	RegisterCachedCubinsRelease(cache,module->cubin_count);
	module->name_count = kept;
	return(SPARK_STATUS_OK);
}

static SparkNvrmModule *RegisterModuleNamed(const char *path)
{
	SparkNvrmModule *module;
	for (module = register_modules; module != 0; module = module->next)
		if ( strcmp(module->path,path) == 0 )
			return(module);
	return(0);
}

SparkStatus SparkNvrmRegisterModule(const char *path, SparkNvrmModule **module)
{
	SparkNvrmModule *found;
	SparkStatus status;
	if ( path == 0 || path[0] == '\0' || module == 0 )
		return(RegisterRefuse(SPARK_STATUS_INVALID_ARGUMENT,"kernel_register",REGISTER_OWNER,"a kernel module needs its path"));
	pthread_mutex_lock(&register_mutex);
	if ( (found = RegisterModuleNamed(path)) == 0 )
	{
		if ( (found = (SparkNvrmModule *)calloc(1u,sizeof(*found))) == 0 || (found->path = strdup(path)) == 0 )
		{
			free(found);
			pthread_mutex_unlock(&register_mutex);
			return(RegisterRefuse(SPARK_STATUS_CAPACITY_EXCEEDED,"kernel_register",REGISTER_OWNER,"out of host memory for the kernel registry"));
		}
		(void)RegisterScan(found);
		found->next = register_modules;
		register_modules = found;
	}
	*module = found;
	status = found->status;
	pthread_mutex_unlock(&register_mutex);
	if ( status != SPARK_STATUS_OK )
		return(RegisterRefuse(status,"kernel_register",REGISTER_OWNER,found->failure));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmModuleStatus(const SparkNvrmModule *module, const char **failure)
{
	*failure = module->failure;
	return(module->status);
}

uint32_t SparkNvrmModuleKernelCount(const SparkNvrmModule *module)
{
	return(module->name_count);
}

static uint32_t RegisterSlot(const void *host, uint32_t capacity)
{
	return((uint32_t)((((uintptr_t)host >> 4) * 0x9e3779b97f4a7c15ull) >> 32) & (capacity - 1u));
}

static RegisterEntry *RegisterFind(const void *host)
{
	uint32_t slot;
	if ( register_capacity == 0u )
		return(0);
	for (slot = RegisterSlot(host,register_capacity); register_table[slot] != 0; slot = (slot + 1u) & (register_capacity - 1u))
		if ( register_table[slot]->host == host )
			return(register_table[slot]);
	return(0);
}

static void RegisterPlace(RegisterEntry **table, uint32_t capacity, RegisterEntry *entry)
{
	uint32_t slot;
	for (slot = RegisterSlot(entry->host,capacity); table[slot] != 0; slot = (slot + 1u) & (capacity - 1u))
		;
	table[slot] = entry;
}

static SparkStatus RegisterInsert(RegisterEntry *entry)
{
	if ( (register_count + 1u) * 2u > register_capacity )
	{
		const uint32_t capacity = register_capacity != 0u ? register_capacity * 2u : REGISTER_TABLE_START;
		RegisterEntry **table = (RegisterEntry **)calloc(capacity,sizeof(*table));
		uint32_t slot;
		if ( table == 0 )
			return(RegisterRefuse(SPARK_STATUS_CAPACITY_EXCEEDED,"kernel_register",REGISTER_OWNER,"out of host memory for the kernel registry"));
		for (slot = 0u; slot < register_capacity; slot++)
			if ( register_table[slot] != 0 )
				RegisterPlace(table,capacity,register_table[slot]);
		free(register_table);
		register_table = table;
		register_capacity = capacity;
	}
	RegisterPlace(register_table,register_capacity,entry);
	register_count++;
	return(SPARK_STATUS_OK);
}

static void RegisterEntryFail(RegisterEntry *entry, SparkStatus code, const char *text)
{
	entry->status = code;
	snprintf(entry->failure,sizeof(entry->failure),"%s",text);
	fprintf(stderr,"nvrm: %s\n",text);
}

static SparkStatus RegisterEntryAdd(SparkNvrmModule *module, const void *host, const char *name, uint32_t variable, RegisterEntry **added)
{
	RegisterEntry *entry;
	char text[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	if ( module == 0 || host == 0 || name == 0 || name[0] == '\0' )
		return(RegisterRefuse(SPARK_STATUS_INVALID_ARGUMENT,"kernel_register",REGISTER_OWNER,"a registration needs its module, host address and name"));
	if ( RegisterFind(host) != 0 )
	{
		snprintf(text,sizeof(text),"the host address of %.64s is registered twice",name);
		return(RegisterRefuse(SPARK_STATUS_DUPLICATE,"kernel_register",REGISTER_OWNER,text));
	}
	if ( (entry = (RegisterEntry *)calloc(1u,sizeof(*entry))) == 0 )
		return(RegisterRefuse(SPARK_STATUS_CAPACITY_EXCEEDED,"kernel_register",REGISTER_OWNER,"out of host memory for the kernel registry"));
	SparkNvrmShortName(name,entry->name);
	entry->host = host;
	entry->module = module;
	entry->variable = variable;
	if ( RegisterInsert(entry) != SPARK_STATUS_OK )
	{
		free(entry);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	*added = entry;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmRegisterFunction(SparkNvrmModule *module, const void *host, const char *name)
{
	RegisterEntry *entry = 0;
	RegisterName key;
	const RegisterName *found;
	char text[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	SparkStatus status;
	pthread_mutex_lock(&register_mutex);
	status = RegisterEntryAdd(module,host,name,0u,&entry);
	if ( status == SPARK_STATUS_OK && module->status == SPARK_STATUS_OK )
	{
		SparkNvrmShortName(name,key.name);
		key.cubin = 0u;
		found = (const RegisterName *)bsearch(&key,module->names,module->name_count,sizeof(*module->names),RegisterNameOrder);
		if ( found != 0 )
		{
			entry->overlaid = found->overlaid;
			entry->overlay_index = found->overlay_index;
			entry->cubin = found->overlaid != 0u ? found->overlay_cubin : found->cubin;
		}
		else
		{
			snprintf(text,sizeof(text),"kernel %.64s is registered but no cubin of %.60s lists it",name,module->path);
			RegisterEntryFail(entry,SPARK_STATUS_NOT_FOUND,text);
			status = RegisterRefuse(SPARK_STATUS_NOT_FOUND,"kernel_register",REGISTER_OWNER,text);
		}
	}
	pthread_mutex_unlock(&register_mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmRegisterVariable(SparkNvrmModule *module, const void *host, const char *name, uint64_t bytes, uint32_t managed)
{
	RegisterEntry *entry = 0;
	char text[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	SparkStatus status;
	(void)bytes;
	pthread_mutex_lock(&register_mutex);
	status = RegisterEntryAdd(module,host,name,1u,&entry);
	if ( status == SPARK_STATUS_OK && managed != 0u )
	{
		snprintf(text,sizeof(text),"managed variable %.64s needs managed memory, which the hosted runtime does not provide",name);
		RegisterEntryFail(entry,SPARK_STATUS_UNSUPPORTED,text);
		status = ModuleFail(module,SPARK_STATUS_UNSUPPORTED,text);
	}
	pthread_mutex_unlock(&register_mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmRegisteredName(const void *host, const char **name)
{
	const RegisterEntry *entry;
	pthread_mutex_lock(&register_mutex);
	entry = RegisterFind(host);
	*name = entry != 0 ? entry->name : 0;
	pthread_mutex_unlock(&register_mutex);
	if ( entry == 0 )
		return(RegisterRefuse(SPARK_STATUS_NOT_FOUND,"kernel_launch",REGISTER_OWNER,"no module registered a kernel or variable at that host address"));
	return(SPARK_STATUS_OK);
}

static SparkStatus RegisterResolve(RegisterEntry *entry)
{
	char text[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	RegisterCubin *cubin;
	if ( entry->variable != 0u )
	{
		snprintf(text,sizeof(text),"%.64s is a device variable, not a kernel",entry->name);
		return(RegisterRefuse(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",REGISTER_OWNER,text));
	}
	if ( entry->module->status != SPARK_STATUS_OK )
	{
		snprintf(text,sizeof(text),"%.40s: %.110s",entry->name,entry->module->failure);
		return(RegisterRefuse(entry->module->status,"kernel_launch",REGISTER_OWNER,text));
	}
	if ( entry->status != SPARK_STATUS_OK )
		return(RegisterRefuse(entry->status,"kernel_launch",REGISTER_OWNER,entry->failure));
	cubin = entry->overlaid != 0u ? &register_overlay.cubins[entry->cubin] : &entry->module->cubins[entry->cubin];
	if ( cubin->library == 0 && SparkDeviceLibraryLoad(&cubin->library,cubin->path,REGISTER_OWNER) != SPARK_STATUS_OK )
		SPARK_FAIL(SparkDeviceLastError()->status);
	if ( SparkDeviceKernelCreate(&entry->kernel,cubin->library,entry->name) != SPARK_STATUS_OK )
		SPARK_FAIL(SparkDeviceLastError()->status);
	if ( entry->overlaid != 0u )
		SparkNvrmKernelSetOverlayIndex(entry->kernel,(int32_t)entry->overlay_index);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmRegisteredKernel(const void *host, SparkDeviceKernel **kernel)
{
	RegisterEntry *entry;
	SparkStatus status = SPARK_STATUS_OK;
	pthread_mutex_lock(&register_mutex);
	entry = RegisterFind(host);
	if ( entry == 0 )
		status = RegisterRefuse(SPARK_STATUS_NOT_FOUND,"kernel_launch",REGISTER_OWNER,"a launch names a function that no module registered as a kernel");
	else if ( entry->kernel == 0 )
		status = RegisterResolve(entry);
	*kernel = status == SPARK_STATUS_OK ? entry->kernel : 0;
	pthread_mutex_unlock(&register_mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmRegisteredResolve(const void *host, const char **name, SparkDeviceKernel **kernel)
{
	RegisterEntry *entry;
	SparkStatus status = SPARK_STATUS_OK;
	pthread_mutex_lock(&register_mutex);
	entry = RegisterFind(host);
	if ( entry == 0 )
	{
		*name = 0;
		*kernel = 0;
		status = RegisterRefuse(SPARK_STATUS_NOT_FOUND,"kernel_launch",REGISTER_OWNER,"no module registered a kernel or variable at that host address");
	}
	else
	{
		*name = entry->name;
		if ( entry->kernel == 0 )
			status = RegisterResolve(entry);
		*kernel = status == SPARK_STATUS_OK ? entry->kernel : 0;
	}
	pthread_mutex_unlock(&register_mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmRegisteredOverlay(const void *host, uint32_t *overlaid)
{
	const RegisterEntry *entry;
	if ( overlaid == 0 )
		return(RegisterRefuse(SPARK_STATUS_INVALID_ARGUMENT,"kernel_register",REGISTER_OWNER,"an overlay query needs an output"));
	pthread_mutex_lock(&register_mutex);
	entry = RegisterFind(host);
	*overlaid = entry != 0 ? entry->overlaid : 0u;
	pthread_mutex_unlock(&register_mutex);
	if ( entry == 0 )
		return(RegisterRefuse(SPARK_STATUS_NOT_FOUND,"kernel_launch",REGISTER_OWNER,"no module registered a kernel or variable at that host address"));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmOverlayReload(void)
{
	SparkStatus status;
	uint32_t slot;
	pthread_mutex_lock(&register_mutex);
	for (slot = 0u; slot < register_capacity; slot++)
		if ( register_table[slot] != 0 && register_table[slot]->overlaid != 0u && register_table[slot]->kernel != 0 )
		{
			(void)SparkDeviceKernelDestroy(register_table[slot]->kernel);
			register_table[slot]->kernel = 0;
		}
	RegisterOverlayClear();
	status = RegisterOverlayEnsure();
	pthread_mutex_unlock(&register_mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

uint32_t SparkNvrmOverlayStatsEnabled(void)
{
	return(__atomic_load_n(&register_overlay.stats_enabled,__ATOMIC_ACQUIRE));
}

void SparkNvrmOverlayRecord(uint32_t overlay_index, uint64_t count, uint32_t in_program)
{
	if ( __atomic_load_n(&register_overlay.stats_enabled,__ATOMIC_ACQUIRE) == 0u || count == 0u )
		return;
	if ( overlay_index >= register_overlay.name_count )
		return;
	if ( in_program != 0u )
		(void)__atomic_add_fetch(&register_overlay.names[overlay_index].program_launches,count,__ATOMIC_RELAXED);
	else
		(void)__atomic_add_fetch(&register_overlay.names[overlay_index].direct_launches,count,__ATOMIC_RELAXED);
}

SparkStatus SparkNvrmOverlayStats(const char *name, uint64_t *direct_launches, uint64_t *program_launches)
{
	char short_name[SPARK_NVRM_SHORT_NAME_BYTES];
	const RegisterOverlayName *found;
	if ( name == 0 || name[0] == '\0' )
		return(RegisterRefuse(SPARK_STATUS_INVALID_ARGUMENT,"kernel_register",REGISTER_OWNER,"an overlay stats query needs a kernel name"));
	SparkNvrmShortName(name,short_name);
	pthread_mutex_lock(&register_mutex);
	found = RegisterOverlayFind(short_name);
	if ( found != 0 )
	{
		if ( direct_launches != 0 )
			*direct_launches = __atomic_load_n(&found->direct_launches,__ATOMIC_ACQUIRE);
		if ( program_launches != 0 )
			*program_launches = __atomic_load_n(&found->program_launches,__ATOMIC_ACQUIRE);
	}
	pthread_mutex_unlock(&register_mutex);
	if ( found == 0 )
		return(RegisterRefuse(SPARK_STATUS_NOT_FOUND,"kernel_register",REGISTER_OWNER,"no overlaid kernel with that name"));
	return(SPARK_STATUS_OK);
}

void SparkNvrmOverlayStatsDump(void)
{
	uint32_t index;
	if ( __atomic_load_n(&register_overlay.stats_enabled,__ATOMIC_ACQUIRE) == 0u )
		return;
	pthread_mutex_lock(&register_mutex);
	for (index = 0u; index < register_overlay.name_count; index++)
	{
		const RegisterOverlayName *item = &register_overlay.names[index];
		const uint64_t direct = __atomic_load_n(&item->direct_launches,__ATOMIC_ACQUIRE);
		const uint64_t program = __atomic_load_n(&item->program_launches,__ATOMIC_ACQUIRE);
		fprintf(stderr,"nvrm: skl-overlay stats: kernel=%s direct=%llu program=%llu total=%llu\n",item->name,
			(unsigned long long)direct,(unsigned long long)program,(unsigned long long)(direct + program));
	}
	fflush(stderr);
	pthread_mutex_unlock(&register_mutex);
}

SparkStatus SparkNvrmRegisteredVariable(const void *host)
{
	const RegisterEntry *entry;
	char text[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	pthread_mutex_lock(&register_mutex);
	entry = RegisterFind(host);
	if ( entry != 0 && entry->variable != 0u )
		snprintf(text,sizeof(text),"host access to device variable %.64s is not supported by the hosted runtime",entry->name);
	pthread_mutex_unlock(&register_mutex);
	if ( entry == 0 || entry->variable == 0u )
		return(RegisterRefuse(SPARK_STATUS_NOT_FOUND,"symbol_access",REGISTER_OWNER,"no module registered a device variable at that host address"));
	return(RegisterRefuse(SPARK_STATUS_UNSUPPORTED,"symbol_access",REGISTER_OWNER,text));
}

static SparkStatus RegisterReleaseModule(SparkNvrmModule *module)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t slot,index;
	for (slot = 0u; slot < register_capacity; slot++)
		if ( register_table[slot] != 0 && register_table[slot]->module == module && register_table[slot]->kernel != 0 )
		{
			if ( SparkDeviceKernelDestroy(register_table[slot]->kernel) != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
				status = SparkDeviceLastError()->status;
			else
				register_table[slot]->kernel = 0;
		}
	for (index = 0u; index < module->cubin_count; index++)
		if ( module->cubins[index].library != 0 )
		{
			if ( SparkDeviceLibraryUnload(module->cubins[index].library) != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
				status = SparkDeviceLastError()->status;
			else
				module->cubins[index].library = 0;
		}
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmRegisteredRelease(void)
{
	SparkStatus status = SPARK_STATUS_OK;
	SparkNvrmModule *module;
	uint32_t index;
	pthread_mutex_lock(&register_mutex);
	for (module = register_modules; module != 0; module = module->next)
	{
		const SparkStatus released = RegisterReleaseModule(module);
		status = status == SPARK_STATUS_OK ? released : status;
	}
	for (index = 0u; index < register_overlay.cubin_count; index++)
		if ( register_overlay.cubins[index].library != 0 )
		{
			if ( SparkDeviceLibraryUnload(register_overlay.cubins[index].library) != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
				status = SparkDeviceLastError()->status;
			else
				register_overlay.cubins[index].library = 0;
		}
	pthread_mutex_unlock(&register_mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static void RegisterForget(SparkNvrmModule *module)
{
	RegisterEntry **kept = register_table;
	const uint32_t capacity = register_capacity;
	uint32_t slot;
	register_table = 0;
	register_capacity = 0u;
	register_count = 0u;
	for (slot = 0u; slot < capacity; slot++)
		if ( kept[slot] != 0 )
		{
			if ( kept[slot]->module == module )
				free(kept[slot]);
			else if ( RegisterInsert(kept[slot]) != SPARK_STATUS_OK )
				fprintf(stderr,"nvrm: out of host memory keeping the kernel registry while a module unloads\n");
		}
	free(kept);
}

#if defined(__linux__)
typedef struct RegisterSearch
{
	const void *address;
	char *path;
	size_t bytes;
	uint32_t found;
} RegisterSearch;

static int RegisterObject(struct dl_phdr_info *info, size_t size, void *context)
{
	RegisterSearch *search = (RegisterSearch *)context;
	uint32_t index;
	(void)size;
	for (index = 0u; index < info->dlpi_phnum; index++)
	{
		const ElfW(Phdr) *header = &info->dlpi_phdr[index];
		const uintptr_t start = (uintptr_t)info->dlpi_addr + header->p_vaddr;
		if ( header->p_type != PT_LOAD || (uintptr_t)search->address < start || (uintptr_t)search->address >= start + header->p_memsz )
			continue;
		if ( info->dlpi_name != 0 && info->dlpi_name[0] != '\0' )
			search->found = snprintf(search->path,search->bytes,"%s",info->dlpi_name) < (int)search->bytes ? 1u : 0u;
		else
		{
			const ssize_t length = readlink("/proc/self/exe",search->path,search->bytes - 1u);
			search->found = length > 0 ? 1u : 0u;
			search->path[length > 0 ? length : 0] = '\0';
		}
		return(1);
	}
	return(0);
}

static uint32_t RegisterModulePath(const void *address, char *path, size_t bytes)
{
	RegisterSearch search = {address, path, bytes, 0u};
	dl_iterate_phdr(RegisterObject,&search);
	return(search.found);
}
#else
static uint32_t RegisterModulePath(const void *address, char *path, size_t bytes)
{
	Dl_info info;
	if ( dladdr(address,&info) == 0 || info.dli_fname == 0 || info.dli_fname[0] == '\0' )
		return(0u);
	return(snprintf(path,bytes,"%s",info.dli_fname) < (int)bytes ? 1u : 0u);
}
#endif

static void RegisterAbort(const char *what, const void *function)
{
	const RegisterEntry *entry;
	pthread_mutex_lock(&register_mutex);
	entry = function != 0 ? RegisterFind(function) : 0;
	fprintf(stderr,"FATAL nvrm: %s%s%.200s; the hosted runtime launches kernels only through LM_LAUNCH\n",what,entry != 0 ? " for " : "",
		entry != 0 ? entry->name : "");
	pthread_mutex_unlock(&register_mutex);
	abort();
}

static __attribute__((noinline)) void **RegisterBinaryFrom(const void *caller)
{
	char path[REGISTER_PATH_BYTES];
	SparkNvrmModule *module = 0;
	RegisterBinary *binary = (RegisterBinary *)calloc(1u,sizeof(*binary));
	if ( binary == 0 )
		RegisterAbort("out of host memory registering a kernel module",0);
	if ( RegisterModulePath(caller,path,sizeof(path)) == 0u )
		snprintf(path,sizeof(path),"%s","(no loaded object contains the registering code)");
	(void)SparkNvrmRegisterModule(path,&module);
	if ( module != 0 )
	{
		pthread_mutex_lock(&register_mutex);
		module->binaries++;
		pthread_mutex_unlock(&register_mutex);
	}
	binary->module = module;
	return((void **)binary);
}

void **__cudaRegisterFatBinary(void *fat_binary)
{
	(void)fat_binary;
	return(RegisterBinaryFrom(__builtin_return_address(0)));
}

void __cudaRegisterFatBinaryEnd(void **handle)
{
	(void)handle;
}

void __cudaUnregisterFatBinary(void **handle)
{
	RegisterBinary *binary = (RegisterBinary *)(void *)handle;
	SparkNvrmModule *module;
	if ( binary == 0 )
		return;
	module = binary->module;
	free(binary);
	if ( module == 0 )
		return;
	pthread_mutex_lock(&register_mutex);
	if ( --module->binaries == 0u )
	{
		(void)RegisterReleaseModule(module);
		RegisterForget(module);
	}
	pthread_mutex_unlock(&register_mutex);
}

void __cudaRegisterFunction(void **handle, const char *host, char *device, const char *name, int thread_limit, void *thread, void *group,
	void *block, void *grid, int *warp)
{
	const RegisterBinary *binary = (const RegisterBinary *)(const void *)handle;
	(void)device; (void)thread_limit; (void)thread; (void)group; (void)block; (void)grid; (void)warp;
	(void)SparkNvrmRegisterFunction(binary != 0 ? binary->module : 0,host,name);
}

void __cudaRegisterVar(void **handle, char *host, char *device, const char *name, int external, size_t bytes, int constant, int global)
{
	const RegisterBinary *binary = (const RegisterBinary *)(const void *)handle;
	(void)device; (void)external; (void)constant; (void)global;
	(void)SparkNvrmRegisterVariable(binary != 0 ? binary->module : 0,host,name,bytes,0u);
}

void __cudaRegisterManagedVar(void **handle, void **host, char *device, const char *name, int external, size_t bytes, int constant, int global)
{
	const RegisterBinary *binary = (const RegisterBinary *)(const void *)handle;
	(void)device; (void)external; (void)constant; (void)global;
	(void)SparkNvrmRegisterVariable(binary != 0 ? binary->module : 0,host,name,bytes,1u);
}

char __cudaInitModule(void **handle)
{
	(void)handle;
	RegisterAbort("a managed-variable module initialisation reached the hosted runtime",0);
	return(0);
}

static _Thread_local SparkNvrmArgumentCapture *register_capture;

SparkNvrmArgumentCapture *SparkNvrmCaptureSet(SparkNvrmArgumentCapture *capture)
{
	SparkNvrmArgumentCapture *previous = register_capture;
	register_capture = capture;
	return(previous);
}

static int RegisterCaptureArgs(const void *function, void **arguments)
{
	SparkNvrmArgumentCapture *capture = register_capture;
	uint32_t index;
	if ( capture == 0 )
		return(0);
	if ( capture->layout == 0 )
	{
		capture->error = SparkNvrmLaunchPointers(function,capture->grid,capture->block,capture->dependent,capture->shared,capture->stream,arguments);
		capture->captured = 1u;
		return(1);
	}
	if ( capture->count != 0u && arguments == 0 )
		return(0);
	for (index = 0u; index < capture->count; index++)
		memcpy(capture->packed + capture->layout[index].offset,arguments[index],capture->layout[index].bytes);
	capture->captured = 1u;
	return(1);
}

unsigned __cudaPushCallConfiguration(SparkNvrmDim grid, SparkNvrmDim block, size_t shared, void *stream)
{
	(void)grid; (void)block; (void)shared; (void)stream;
	if ( register_capture != 0 )
		return(0u);
	RegisterAbort("a <<<>>> launch configuration reached the hosted runtime",0);
	return(1u);
}

int __cudaPopCallConfiguration(SparkNvrmDim *grid, SparkNvrmDim *block, size_t *shared, void *stream)
{
	if ( register_capture != 0 )
	{
		const SparkNvrmDim one = {1u, 1u, 1u};
		if ( grid != 0 ) *grid = one;
		if ( block != 0 ) *block = one;
		if ( shared != 0 ) *shared = 0u;
		if ( stream != 0 ) *(void **)stream = 0;
		return(0);
	}
	(void)grid; (void)block; (void)shared; (void)stream;
	RegisterAbort("a <<<>>> launch reached the hosted runtime",0);
	return(1);
}

int __cudaGetKernel(void **kernel, const void *function)
{
	if ( register_capture != 0 )
	{
		if ( kernel != 0 ) *kernel = (void *)(uintptr_t)function;
		return(0);
	}
	(void)kernel;
	RegisterAbort("a <<<>>> kernel lookup reached the hosted runtime",function);
	return(1);
}

int __cudaLaunchKernel(void *kernel, SparkNvrmDim grid, SparkNvrmDim block, void **arguments, size_t shared, void *stream)
{
	(void)grid; (void)block; (void)shared; (void)stream;
	if ( RegisterCaptureArgs(kernel,arguments) )
		return(0);
	RegisterAbort("a <<<>>> kernel launch reached the hosted runtime",kernel);
	return(1);
}

int __cudaLaunchKernel_ptsz(void *kernel, SparkNvrmDim grid, SparkNvrmDim block, void **arguments, size_t shared, void *stream)
{
	return(__cudaLaunchKernel(kernel,grid,block,arguments,shared,stream));
}

int cudaLaunchKernel(const void *function, SparkNvrmDim grid, SparkNvrmDim block, void **arguments, size_t shared, void *stream)
{
	(void)grid; (void)block; (void)shared; (void)stream;
	if ( RegisterCaptureArgs(function,arguments) )
		return(0);
	RegisterAbort("a runtime-API kernel launch reached the hosted runtime",function);
	return(1);
}

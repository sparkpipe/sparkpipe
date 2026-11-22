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

#define REGISTER_PATH_BYTES 4096u
#define REGISTER_TABLE_START 1024u
#define REGISTER_GROW_START 16u
#define REGISTER_CUBIN_SUFFIX ".cubin"
#define REGISTER_MANIFEST_SUFFIX ".manifest"
#define REGISTER_OWNER "registered kernels"

typedef struct RegisterCubin
{
	char *path;
	SparkDeviceLibrary *library;
} RegisterCubin;

typedef struct RegisterName
{
	char name[SPARK_NVRM_SHORT_NAME_BYTES];
	uint32_t cubin;
} RegisterName;

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
		if ( RegisterGrow((void **)&module->names,&module->name_capacity,module->name_count,sizeof(*module->names)) != SPARK_STATUS_OK )
		{
			SparkNvrmManifestRelease(&manifest);
			return(ModuleFail(module,SPARK_STATUS_CAPACITY_EXCEEDED,"out of host memory for the kernel registry"));
		}
		memcpy(module->names[module->name_count].name,manifest.kernels[index].name,sizeof(module->names[module->name_count].name));
		module->names[module->name_count].cubin = module->cubin_count;
		module->name_count++;
	}
	module->cubin_count++;
	SparkNvrmManifestRelease(&manifest);
	return(SPARK_STATUS_OK);
}

static SparkStatus RegisterScan(SparkNvrmModule *module)
{
	char directory[REGISTER_PATH_BYTES],path[REGISTER_PATH_BYTES],text[SPARK_NVRM_REGISTER_FAILURE_BYTES];
	SparkStatus status = SPARK_STATUS_OK;
	struct dirent *entry;
	uint32_t index;
	DIR *listing;
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
	qsort(module->names,module->name_count,sizeof(*module->names),RegisterNameOrder);
	for (index = 1u; index < module->name_count; index++)
		if ( strcmp(module->names[index - 1u].name,module->names[index].name) == 0 )
		{
			snprintf(text,sizeof(text),"kernel %.64s is in two cubins of the module",module->names[index].name);
			return(ModuleFail(module,SPARK_STATUS_DUPLICATE,text));
		}
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
			entry->cubin = found->cubin;
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
	cubin = &entry->module->cubins[entry->cubin];
	if ( cubin->library == 0 && SparkDeviceLibraryLoad(&cubin->library,cubin->path,REGISTER_OWNER) != SPARK_STATUS_OK )
		SPARK_FAIL(SparkDeviceLastError()->status);
	if ( SparkDeviceKernelCreate(&entry->kernel,cubin->library,entry->name) != SPARK_STATUS_OK )
		SPARK_FAIL(SparkDeviceLastError()->status);
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
	pthread_mutex_lock(&register_mutex);
	for (module = register_modules; module != 0; module = module->next)
	{
		const SparkStatus released = RegisterReleaseModule(module);
		status = status == SPARK_STATUS_OK ? released : status;
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
	(void)device;
	(void)thread_limit;
	(void)thread;
	(void)group;
	(void)block;
	(void)grid;
	(void)warp;
	(void)SparkNvrmRegisterFunction(binary != 0 ? binary->module : 0,host,name);
}

void __cudaRegisterVar(void **handle, char *host, char *device, const char *name, int external, size_t bytes, int constant, int global)
{
	const RegisterBinary *binary = (const RegisterBinary *)(const void *)handle;
	(void)device;
	(void)external;
	(void)constant;
	(void)global;
	(void)SparkNvrmRegisterVariable(binary != 0 ? binary->module : 0,host,name,bytes,0u);
}

void __cudaRegisterManagedVar(void **handle, void **host, char *device, const char *name, int external, size_t bytes, int constant, int global)
{
	const RegisterBinary *binary = (const RegisterBinary *)(const void *)handle;
	(void)device;
	(void)external;
	(void)constant;
	(void)global;
	(void)SparkNvrmRegisterVariable(binary != 0 ? binary->module : 0,host,name,bytes,1u);
}

char __cudaInitModule(void **handle)
{
	(void)handle;
	RegisterAbort("a managed-variable module initialisation reached the hosted runtime",0);
	return(0);
}

unsigned __cudaPushCallConfiguration(SparkNvrmDim grid, SparkNvrmDim block, size_t shared, void *stream)
{
	(void)grid;
	(void)block;
	(void)shared;
	(void)stream;
	RegisterAbort("a <<<>>> launch configuration reached the hosted runtime",0);
	return(1u);
}

int __cudaPopCallConfiguration(SparkNvrmDim *grid, SparkNvrmDim *block, size_t *shared, void *stream)
{
	(void)grid;
	(void)block;
	(void)shared;
	(void)stream;
	RegisterAbort("a <<<>>> launch reached the hosted runtime",0);
	return(1);
}

int __cudaGetKernel(void **kernel, const void *function)
{
	(void)kernel;
	RegisterAbort("a <<<>>> kernel lookup reached the hosted runtime",function);
	return(1);
}

int __cudaLaunchKernel(void *kernel, SparkNvrmDim grid, SparkNvrmDim block, void **arguments, size_t shared, void *stream)
{
	(void)grid;
	(void)block;
	(void)arguments;
	(void)shared;
	(void)stream;
	RegisterAbort("a <<<>>> kernel launch reached the hosted runtime",kernel);
	return(1);
}

int __cudaLaunchKernel_ptsz(void *kernel, SparkNvrmDim grid, SparkNvrmDim block, void **arguments, size_t shared, void *stream)
{
	return(__cudaLaunchKernel(kernel,grid,block,arguments,shared,stream));
}

int cudaLaunchKernel(const void *function, SparkNvrmDim grid, SparkNvrmDim block, void **arguments, size_t shared, void *stream)
{
	(void)grid;
	(void)block;
	(void)arguments;
	(void)shared;
	(void)stream;
	RegisterAbort("a runtime-API kernel launch reached the hosted runtime",function);
	return(1);
}

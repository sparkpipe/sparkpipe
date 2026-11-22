#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_launch.h"
#include "runtime/device/nvrm/nvrm_register.h"
#include "runtime/device/nvrm/nvrm_names.h"
#include "tests/nvrm/fixture_kernels.h"

#ifndef SPARK_NVRM_FIXTURE_CUBIN
#define SPARK_NVRM_FIXTURE_CUBIN "tests/fixtures/nvrm/fixture_kernels.sm_121a.cubin"
#endif
#ifndef SPARK_NVRM_FIXTURE_MANIFEST
#define SPARK_NVRM_FIXTURE_MANIFEST "tests/fixtures/nvrm/fixture_kernels.sm_121a.manifest"
#endif

#define REGISTER_PATH 4096
#define REGISTER_FILE (2 * REGISTER_PATH)
#define REGISTER_THREADS 128u

static int failures;
static const char hosts[8];
static uint32_t variable_shadow;
static uint32_t managed_shadow;
static char made_files[32][REGISTER_FILE];
static char made_directories[16][REGISTER_FILE];
static int made_file_count, made_directory_count;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

static int Says(const char *part)
{
	return(strstr(SparkDeviceLastError()->text,part) != 0);
}

static void MakeDirectory(const char *path)
{
	if ( mkdir(path,0755) == 0 && made_directory_count < 16 )
		snprintf(made_directories[made_directory_count++],REGISTER_FILE,"%s",path);
}

static void PutFile(const char *directory, const char *name, const char *source)
{
	char path[REGISTER_FILE];
	uint64_t bytes = 0u;
	uint8_t *data = source != 0 ? SparkNvrmReadFile(source,&bytes) : (uint8_t *)calloc(1u,1u);
	FILE *file;
	snprintf(path,sizeof(path),"%s/%s",directory,name);
	file = fopen(path,"wb");
	CHECK(data != 0 && file != 0 && fwrite(data,1u,(size_t)bytes,file) == (size_t)bytes,"write a kernel directory file");
	if ( file != 0 )
		fclose(file);
	free(data);
	if ( made_file_count < 32 )
		snprintf(made_files[made_file_count++],REGISTER_FILE,"%s",path);
}

static void PutPair(const char *directory, const char *stem)
{
	char name[64];
	snprintf(name,sizeof(name),"%s.cubin",stem);
	PutFile(directory,name,SPARK_NVRM_FIXTURE_CUBIN);
	snprintf(name,sizeof(name),"%s.manifest",stem);
	PutFile(directory,name,SPARK_NVRM_FIXTURE_MANIFEST);
}

static void ModulePaths(const char *root, const char *name, char *module, char *directory)
{
	snprintf(module,REGISTER_PATH,"%s/%s",root,name);
	snprintf(directory,REGISTER_PATH,"%s%s",module,SPARK_NVRM_KERNEL_DIRECTORY_SUFFIX);
	MakeDirectory(directory);
}

static void Cleanup(void)
{
	int index;
	for (index = made_file_count - 1; index >= 0; index--)
		unlink(made_files[index]);
	for (index = made_directory_count - 1; index >= 0; index--)
		rmdir(made_directories[index]);
}

static uint32_t FixtureKernels(void)
{
	uint64_t bytes = 0u;
	char *text = (char *)SparkNvrmReadFile(SPARK_NVRM_FIXTURE_MANIFEST,&bytes);
	SparkNvrmManifest manifest;
	uint32_t count = 0u;
	if ( text != 0 && SparkNvrmManifestParse(text,bytes,"register",&manifest) == SPARK_STATUS_OK )
	{
		count = manifest.kernel_count;
		SparkNvrmManifestRelease(&manifest);
	}
	free(text);
	return(count);
}

static void TestRefusals(const char *root)
{
	char module_path[REGISTER_PATH],directory[REGISTER_PATH];
	SparkNvrmModule *module = 0;
	SparkDeviceKernel *kernel = (SparkDeviceKernel *)&kernel;
	const char *failure = 0;
	snprintf(module_path,sizeof(module_path),"%s/absent.so",root);
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_NOT_FOUND && module != 0 && Says("has no kernel directory"),
		"a module without its kernel directory is refused, naming the directory");
	CHECK(module != 0 && SparkNvrmModuleStatus(module,&failure) == SPARK_STATUS_NOT_FOUND && strstr(failure,"absent.so.kernels") != 0,
		"the module keeps its failure");
	CHECK(module != 0 && SparkNvrmRegisterFunction(module,&hosts[5],"SparkNvrmFirstLight") == SPARK_STATUS_OK,
		"a kernel of a failed module is still recorded");
	CHECK(SparkNvrmRegisteredKernel(&hosts[5],&kernel) == SPARK_STATUS_NOT_FOUND && kernel == 0 && Says("has no kernel directory"),
		"a launch through a failed module fails with the module's failure");
	ModulePaths(root,"twice.so",module_path,directory);
	PutPair(directory,"a");
	PutPair(directory,"b");
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_DUPLICATE && Says("is in two cubins"),"a kernel name in two cubins of one module is refused");
	ModulePaths(root,"stray.so",module_path,directory);
	PutPair(directory,"a");
	PutFile(directory,"stray.manifest",SPARK_NVRM_FIXTURE_MANIFEST);
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_NOT_FOUND && Says("has no cubin beside it"),"a manifest without its cubin is refused");
	ModulePaths(root,"lone.so",module_path,directory);
	PutFile(directory,"lone.cubin",SPARK_NVRM_FIXTURE_CUBIN);
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_NOT_FOUND && Says("has no manifest"),"a cubin without its manifest is refused");
	ModulePaths(root,"empty.so",module_path,directory);
	PutFile(directory,"empty.ptx",0);
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_NOT_FOUND && Says("holds no cubin"),"a kernel directory without a cubin is refused");
	ModulePaths(root,"broken.so",module_path,directory);
	PutFile(directory,"broken.cubin",SPARK_NVRM_FIXTURE_CUBIN);
	PutFile(directory,"broken.manifest",0);
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_PARSE_ERROR,"a manifest that does not parse is refused");
	ModulePaths(root,"managed.so",module_path,directory);
	PutPair(directory,"a");
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_OK,"register the managed-variable module");
	CHECK(SparkNvrmRegisterFunction(module,&hosts[6],"SparkNvrmFirstLight") == SPARK_STATUS_OK &&
		SparkNvrmRegisterVariable(module,&managed_shadow,"spark_managed_word",4u,1u) == SPARK_STATUS_UNSUPPORTED && Says("managed memory"),
		"a managed variable is refused");
	CHECK(SparkNvrmRegisteredKernel(&hosts[6],&kernel) == SPARK_STATUS_UNSUPPORTED && kernel == 0 && Says("spark_managed_word"),
		"a module with a managed variable launches nothing, naming the variable");
}

static void TestModule(const char *root)
{
	char module_path[REGISTER_PATH],directory[REGISTER_PATH],removed[REGISTER_FILE];
	SparkNvrmModule *module = 0,*again = 0;
	SparkDeviceKernel *kernel = (SparkDeviceKernel *)&kernel;
	const char *name = 0;
	ModulePaths(root,"module.so",module_path,directory);
	PutPair(directory,"fixture");
	PutFile(directory,"fixture.ptx",0);
	PutFile(directory,"fixture.cudafe1.cpp",0);
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_OK && module != 0,"register a module whose kernel directory holds a cubin, its manifest and compiler leftovers");
	CHECK(module != 0 && SparkNvrmModuleKernelCount(module) == FixtureKernels() && FixtureKernels() != 0u,"the module lists every kernel of its manifests");
	snprintf(removed,sizeof(removed),"%s/fixture.ptx",directory);
	unlink(removed);
	CHECK(SparkNvrmRegisterModule(module_path,&again) == SPARK_STATUS_OK && again == module,"a module's later fat binaries reuse its first scan");
	CHECK(SparkNvrmRegisterFunction(module,&hosts[0],"SparkNvrmFirstLight") == SPARK_STATUS_OK &&
		SparkNvrmRegisterFunction(module,&hosts[1],SPARK_NVRM_FIXTURE_LONG_TEXT) == SPARK_STATUS_OK,"register kernels by their names");
	CHECK(SparkNvrmRegisteredName(&hosts[0],&name) == SPARK_STATUS_OK && strcmp(name,"SparkNvrmFirstLight") == 0,"a host function maps to its kernel name");
	{
		char expected[SPARK_NVRM_SHORT_NAME_BYTES];
		SparkNvrmShortName(SPARK_NVRM_FIXTURE_LONG_TEXT,expected);
		CHECK(SparkNvrmRegisteredName(&hosts[1],&name) == SPARK_STATUS_OK && strcmp(name,expected) == 0 && strlen(name) <= SPARK_DEVICE_KERNEL_NAME_LIMIT,
			"a 152-character mangled name nvcc registers is kept only as its short name");
	}
	CHECK(SparkNvrmRegisterFunction(module,&hosts[0],"SparkNvrmStreamFill") == SPARK_STATUS_DUPLICATE,"a host function registered twice is refused");
	CHECK(SparkNvrmRegisterFunction(module,&hosts[2],"SparkNvrmAbsentKernel") == SPARK_STATUS_NOT_FOUND && Says("SparkNvrmAbsentKernel"),
		"a kernel no cubin lists is refused at registration, naming it");
	CHECK(SparkNvrmRegisteredKernel(&hosts[2],&kernel) == SPARK_STATUS_NOT_FOUND && kernel == 0 && Says("SparkNvrmAbsentKernel"),
		"a launch of that kernel fails, naming it");
	CHECK(SparkNvrmRegisteredKernel(&hosts[7],&kernel) == SPARK_STATUS_NOT_FOUND && Says("no module registered"),"an unregistered function is refused");
	CHECK(SparkNvrmRegisterVariable(module,&variable_shadow,"spark_nvrm_fixture_word",4u,0u) == SPARK_STATUS_OK,"register a device variable");
	CHECK(SparkNvrmRegisteredVariable(&variable_shadow) == SPARK_STATUS_UNSUPPORTED && Says("spark_nvrm_fixture_word"),
		"host access to a device variable is refused, naming the symbol");
	CHECK(SparkNvrmRegisteredKernel(&variable_shadow,&kernel) == SPARK_STATUS_INVALID_ARGUMENT && Says("device variable"),"a device variable is not launched");
	CHECK(SparkNvrmRegisteredVariable(&hosts[0]) == SPARK_STATUS_NOT_FOUND,"a kernel is not a device variable");
}

static void TestFatBinary(const char *program)
{
	char module_path[PATH_MAX],directory[REGISTER_FILE];
	const char *name = 0,*failure = 0;
	void **handle;
	if ( realpath(program,module_path) == 0 )
	{
		CHECK(0,"resolve the test program's path");
		return;
	}
	snprintf(directory,sizeof(directory),"%s%s",module_path,SPARK_NVRM_KERNEL_DIRECTORY_SUFFIX);
	MakeDirectory(directory);
	PutPair(directory,"program");
	handle = __cudaRegisterFatBinary(0);
	__cudaRegisterFunction(handle,&hosts[3],0,"SparkNvrmStreamFill",-1,0,0,0,0,0);
	__cudaRegisterVar(handle,(char *)&hosts[4],0,"spark_nvrm_fixture_table",0,16u,1,0);
	__cudaRegisterFatBinaryEnd(handle);
	CHECK(handle != 0 && *(SparkNvrmModule **)(void *)handle != 0 && SparkNvrmModuleStatus(*(SparkNvrmModule **)(void *)handle,&failure) == SPARK_STATUS_OK,
		"a fat binary registered from the program finds the program's kernel directory");
	CHECK(SparkNvrmRegisteredName(&hosts[3],&name) == SPARK_STATUS_OK && strcmp(name,"SparkNvrmStreamFill") == 0,"and records its kernels");
	CHECK(SparkNvrmRegisteredVariable(&hosts[4]) == SPARK_STATUS_UNSUPPORTED && Says("spark_nvrm_fixture_table"),"and its device variables");
	__cudaUnregisterFatBinary(handle);
	CHECK(SparkNvrmRegisteredName(&hosts[3],&name) == SPARK_STATUS_NOT_FOUND && SparkNvrmRegisteredVariable(&hosts[4]) == SPARK_STATUS_NOT_FOUND,
		"unregistering a module's last fat binary forgets its kernels and variables");
}

static int Aborts(int which)
{
	const SparkNvrmDim one = {1u, 1u, 1u};
	SparkNvrmDim grid = one,block = one;
	size_t shared = 0u;
	void *kernel = 0;
	int status = 0;
	const pid_t child = fork();
	if ( child == 0 )
	{
		if ( freopen("/dev/null","w",stderr) == 0 )
			_exit(2);
		if ( which == 0 )
			(void)__cudaPushCallConfiguration(one,one,0u,0);
		else if ( which == 1 )
			(void)__cudaPopCallConfiguration(&grid,&block,&shared,0);
		else if ( which == 2 )
			(void)__cudaGetKernel(&kernel,&hosts[0]);
		else if ( which == 3 )
			(void)__cudaLaunchKernel(0,one,one,0,0u,0);
		else if ( which == 4 )
			(void)cudaLaunchKernel(&hosts[0],one,one,0,0u,0);
		else
			(void)__cudaInitModule(0);
		_exit(0);
	}
	return(child > 0 && waitpid(child,&status,0) == child && WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

static void TestLaunchEntries(void)
{
	int which,aborted = 0;
	for (which = 0; which < 6; which++)
		aborted += Aborts(which);
	CHECK(aborted == 6,"every <<<>>> launch entry, the runtime-API launch and the managed module initialisation abort");
}

#ifdef SPARK_NVRM_REGISTER_GPU
typedef struct FirstLightBlock
{
	uint64_t output;
	uint32_t value;
} FirstLightBlock;

static void TestGpu(void)
{
	SparkDeviceKernel *kernel = 0,*cached = 0,*fresh = 0;
	SparkDeviceQueue *queue = 0;
	SparkDeviceBuffer output;
	SparkDeviceLaunch launch;
	FirstLightBlock block;
	uint32_t index,wrong = 0u;
	if ( access("/dev/nvidiactl",F_OK) != 0 )
	{
		printf("SKIP the registered-kernel launch (no open kernel module on this host)\n");
		return;
	}
	CHECK(SparkDeviceInitialize(0) == SPARK_STATUS_OK && SparkDeviceQueueCreate(&queue,"register") == SPARK_STATUS_OK,"initialize the device");
	CHECK(SparkNvrmRegisteredKernel(&hosts[0],&kernel) == SPARK_STATUS_OK && kernel != 0,"a registered function resolves to a kernel on first use");
	CHECK(SparkNvrmRegisteredKernel(&hosts[0],&cached) == SPARK_STATUS_OK && cached == kernel,"the kernel is cached per process");
	CHECK(SparkDeviceAllocate(&output,SPARK_DEVICE_SPACE_SHARED_COHERENT,REGISTER_THREADS * sizeof(uint32_t),"register output") == SPARK_STATUS_OK,"allocate");
	memset(&block,0,sizeof(block));
	block.output = output.device;
	block.value = 0x77000000u;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = REGISTER_THREADS;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &block;
	launch.constant_bytes = (uint32_t)(sizeof(uint64_t) + sizeof(uint32_t));
	CHECK(kernel != 0 && SparkDeviceKernelLaunch(queue,kernel,&launch) == SPARK_STATUS_OK && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
		"launch the registered kernel");
	for (index = 0u; index < REGISTER_THREADS; index++)
		wrong += ((const uint32_t *)output.host)[index] != (block.value ^ index);
	CHECK(wrong == 0u,"the registered kernel is the one that ran");
	CHECK(SparkNvrmRegisteredKernel(&hosts[1],&fresh) == SPARK_STATUS_OK && fresh != 0,"the 152-character mangled kernel resolves");
	CHECK(SparkNvrmRegisteredRelease() == SPARK_STATUS_OK && SparkNvrmRegisteredKernel(&hosts[0],&fresh) == SPARK_STATUS_OK && fresh != 0,
		"releasing the registered kernels lets the next launch create them again");
	CHECK(SparkDeviceFree(&output) == SPARK_STATUS_OK && SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK,"free the output and the queue");
	CHECK(SparkNvrmRegisteredRelease() == SPARK_STATUS_OK && SparkDeviceShutdown() == SPARK_STATUS_OK,"after the release the device shuts down with no live library");
}
#endif

int main(int argc, char **argv)
{
	char root[] = "/tmp/spark-nvrm-register-XXXXXX";
	(void)argc;
	if ( mkdtemp(root) == 0 )
	{
		printf("FAIL make a scratch directory\n");
		return(1);
	}
	TestRefusals(root);
	TestModule(root);
	TestFatBinary(argv[0]);
	TestLaunchEntries();
#ifdef SPARK_NVRM_REGISTER_GPU
	TestGpu();
#endif
	Cleanup();
	rmdir(root);
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS kernel modules register from their kernel directories, refuse what does not pair or repeats, and launch registered kernels lazily\n");
	return(0);
}

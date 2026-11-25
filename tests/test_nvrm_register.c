#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_cubin.h"
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
static const char hosts[16];
static uint32_t variable_shadow;
static uint32_t managed_shadow;
static char made_files[96][REGISTER_FILE];
static char made_directories[32][REGISTER_FILE];
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
	if ( mkdir(path,0755) == 0 && made_directory_count < 32 )
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
	if ( made_file_count < 96 )
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

static void PutBytes(const char *directory, const char *name, const void *data, uint64_t bytes)
{
	char path[REGISTER_FILE];
	FILE *file;
	snprintf(path,sizeof(path),"%s/%s",directory,name);
	file = fopen(path,"wb");
	CHECK(data != 0 && file != 0 && fwrite(data,1u,(size_t)bytes,file) == (size_t)bytes,"write a kernel directory file");
	if ( file != 0 )
		fclose(file);
	if ( made_file_count < 96 )
		snprintf(made_files[made_file_count++],REGISTER_FILE,"%s",path);
}

static int ManifestLineKept(const char *line, size_t length, const char *kernel)
{
	static const char *const heads[] = {"nvcc ","flags ","source ","cubin "};
	const char *space = memchr(line,' ',length);
	const size_t name = strlen(kernel);
	size_t head;
	for ( head = 0u; head < sizeof(heads) / sizeof(heads[0]); head++ )
		if ( strncmp(line,heads[head],strlen(heads[head])) == 0 )
			return(1);
	if ( space == 0 )
		return(0);
	space++;
	return((size_t)(line + length - space) > name && strncmp(space,kernel,name) == 0 && (space[name] == ' ' || space[name] == '\n'));
}

static void PutManifestOf(const char *directory, const char *name, const char *kernel)
{
	uint64_t bytes = 0u,used = 0u;
	char *text = (char *)SparkNvrmReadFile(SPARK_NVRM_FIXTURE_MANIFEST,&bytes),*kept = (char *)calloc(1u,(size_t)bytes + 1u);
	const char *line = text;
	while ( text != 0 && kept != 0 && line < text + bytes )
	{
		const char *end = memchr(line,'\n',(size_t)(text + bytes - line));
		const size_t length = end != 0 ? (size_t)(end - line) + 1u : (size_t)(text + bytes - line);
		if ( ManifestLineKept(line,length,kernel) != 0 )
		{
			memcpy(kept + used,line,length);
			used += length;
		}
		line += length;
	}
	PutBytes(directory,name,kept,used);
	free(text);
	free(kept);
}

static void PutChangedCubin(const char *directory, const char *name, const char *kernel)
{
	uint64_t bytes = 0u,length = 0u;
	uint8_t *blob = SparkNvrmReadFile(SPARK_NVRM_FIXTURE_CUBIN,&bytes),*found = 0;
	SparkNvrmCubin cubin;
	const SparkNvrmCubinKernel *code;
	memset(&cubin,0,sizeof(cubin));
	if ( blob != 0 && SparkNvrmCubinParse(blob,bytes,"register",&cubin) == SPARK_STATUS_OK && (code = SparkNvrmCubinKernelNamed(&cubin,kernel)) != 0 )
	{
		length = code->code_bytes;
		found = (uint8_t *)memmem(blob,(size_t)bytes,cubin.image + code->code_offset,(size_t)length);
	}
	CHECK(found != 0 && length >= 32u,"find a kernel's code in the fixture cubin");
	if ( found != 0 )
		found[length / 2u] ^= 0x01u;
	PutBytes(directory,name,blob,bytes);
	SparkNvrmCubinRelease(&cubin);
	free(blob);
}

static uint64_t ElfWord(const uint8_t *at, uint32_t bytes)
{
	uint64_t value = 0u;
	uint32_t index;
	for (index = 0u; index < bytes; index++)
		value |= (uint64_t)at[index] << (8u * index);
	return(value);
}

static void PutUnrelocatedCubin(const char *directory, const char *name)
{
	uint64_t bytes = 0u,section,emptied = 0u;
	uint8_t *blob = SparkNvrmReadFile(SPARK_NVRM_FIXTURE_CUBIN,&bytes);
	const uint64_t table = blob != 0 && bytes > 64u ? ElfWord(blob + 0x28u,8u) : 0u;
	const uint64_t entry = blob != 0 && bytes > 64u ? ElfWord(blob + 0x3au,2u) : 0u,count = blob != 0 && bytes > 64u ? ElfWord(blob + 0x3cu,2u) : 0u;
	for (section = 0u; table != 0u && entry >= 64u && table + (section + 1u) * entry <= bytes && section < count; section++)
	{
		uint8_t *header = blob + table + section * entry;
		const uint64_t type = ElfWord(header + 4u,4u);
		if ( type == 4u || type == 9u )
		{
			memset(header + 32u,0,8u);
			emptied++;
		}
	}
	CHECK(emptied != 0u,"empty the fixture cubin's relocation sections");
	PutBytes(directory,name,blob,bytes);
	free(blob);
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
	PutUnrelocatedCubin(directory,"a.cubin");
	PutFile(directory,"a.manifest",SPARK_NVRM_FIXTURE_MANIFEST);
	PutUnrelocatedCubin(directory,"b.cubin");
	PutManifestOf(directory,"b.manifest","SparkNvrmFirstLight");
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_OK && SparkNvrmModuleStatus(module,&failure) == SPARK_STATUS_OK &&
		SparkNvrmModuleKernelCount(module) == FixtureKernels(),"a kernel in two cubins with the same code and attributes is kept once");
	ModulePaths(root,"changed.so",module_path,directory);
	PutPair(directory,"a");
	PutChangedCubin(directory,"b.cubin","SparkNvrmFirstLight");
	PutManifestOf(directory,"b.manifest","SparkNvrmFirstLight");
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_DUPLICATE &&
		Says("kernel SparkNvrmFirstLight is in two cubins of the module, a.cubin and b.cubin, and the code differs"),
		"a kernel in two cubins whose code differs is refused, naming both cubins");
	ModulePaths(root,"relocated.so",module_path,directory);
	PutPair(directory,"a");
	PutFile(directory,"b.cubin",SPARK_NVRM_FIXTURE_CUBIN);
	PutManifestOf(directory,"b.manifest","SparkNvrmGlobalSwap");
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_DUPLICATE && Says("and a constant bank both copies have is relocated"),
		"a kernel in two cubins whose shared constant bank holds relocated addresses is refused: each copy would reach other storage");
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

static void TestOverlay(const char *root)
{
	char overlay_dir[REGISTER_PATH],bad_abi_dir[REGISTER_PATH],empty_dir[REGISTER_PATH];
	char module_path[REGISTER_PATH],directory[REGISTER_PATH];
	SparkNvrmModule *module = 0;
	uint32_t overlaid = 99u;
	static const char bad_abi_manifest[] =
		"nvcc sklc-4.0\n"
		"flags --arch sm_121a\n"
		"cubin 0000000000000000000000000000000000000000000000000000000000000000 64\n"
		"kernel SparkNvrmFirstLight short SparkNvrmFirstLight registers 16 barriers 0 stack 0 spill_stores 0 spill_loads 0 smem 0\n"
		"param SparkNvrmFirstLight 0 ptr align 8 bytes 8 offset 0\n"
		"param SparkNvrmFirstLight 1 u32 align 4 bytes 4 offset 8\n"
		"param SparkNvrmFirstLight 2 u32 align 4 bytes 4 offset 12\n";
	snprintf(overlay_dir,sizeof(overlay_dir),"%s/skl_overlay",root);
	snprintf(bad_abi_dir,sizeof(bad_abi_dir),"%s/skl_bad_abi",root);
	snprintf(empty_dir,sizeof(empty_dir),"%s/skl_empty",root);
	MakeDirectory(overlay_dir);
	MakeDirectory(bad_abi_dir);
	MakeDirectory(empty_dir);
	PutFile(overlay_dir,"first.cubin",SPARK_NVRM_FIXTURE_CUBIN);
	PutManifestOf(overlay_dir,"first.manifest","SparkNvrmFirstLight");
	PutFile(bad_abi_dir,"bad.cubin",SPARK_NVRM_FIXTURE_CUBIN);
	PutBytes(bad_abi_dir,"bad.manifest",bad_abi_manifest,sizeof(bad_abi_manifest) - 1u);
	unsetenv("SPARKPIPE_KERNEL_PROVIDER");
	unsetenv("SPARKPIPE_SKL_CUBIN_DIR");
	CHECK(SparkNvrmOverlayReload() == SPARK_STATUS_OK,"overlay reload with no env is inactive and OK");
	CHECK(SparkNvrmRegisteredOverlay(&hosts[0],&overlaid) == SPARK_STATUS_OK && overlaid == 0u,"base registration is not overlaid when unset");
	setenv("SPARKPIPE_KERNEL_PROVIDER","bogus",1);
	CHECK(SparkNvrmOverlayReload() == SPARK_STATUS_INVALID_ARGUMENT && Says("unknown SPARKPIPE_KERNEL_PROVIDER"),"unknown provider fails closed");
	setenv("SPARKPIPE_KERNEL_PROVIDER","skl-overlay",1);
	unsetenv("SPARKPIPE_SKL_CUBIN_DIR");
	CHECK(SparkNvrmOverlayReload() == SPARK_STATUS_INVALID_ARGUMENT && Says("requires SPARKPIPE_SKL_CUBIN_DIR"),
		"skl-overlay without SPARKPIPE_SKL_CUBIN_DIR fails closed");
	setenv("SPARKPIPE_SKL_CUBIN_DIR",empty_dir,1);
	CHECK(SparkNvrmOverlayReload() == SPARK_STATUS_NOT_FOUND && Says("holds no cubin"),"empty overlay directory fails closed");
	setenv("SPARKPIPE_KERNEL_PROVIDER","default",1);
	setenv("SPARKPIPE_SKL_CUBIN_DIR",bad_abi_dir,1);
	CHECK(SparkNvrmOverlayReload() == SPARK_STATUS_OK,"default provider explicitly disables overlay");
	setenv("SPARKPIPE_KERNEL_PROVIDER","skl-overlay",1);
	setenv("SPARKPIPE_SKL_CUBIN_DIR",bad_abi_dir,1);
	CHECK(SparkNvrmOverlayReload() == SPARK_STATUS_OK,"bad-abi overlay parses its own manifest");
	ModulePaths(root,"overlay_bad.so",module_path,directory);
	PutPair(directory,"a");
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_TARGET_MISMATCH && Says("ABI mismatch"),
		"overlay kernel with mismatched parameter count fails closed on module scan");
	setenv("SPARKPIPE_KERNEL_PROVIDER","skl-overlay",1);
	setenv("SPARKPIPE_SKL_CUBIN_DIR",overlay_dir,1);
	CHECK(SparkNvrmOverlayReload() == SPARK_STATUS_OK,"valid skl-overlay directory loads cleanly");
	ModulePaths(root,"overlay_ok.so",module_path,directory);
	PutPair(directory,"a");
	CHECK(SparkNvrmRegisterModule(module_path,&module) == SPARK_STATUS_OK,"module registers cleanly with matching skl-overlay");
	CHECK(SparkNvrmRegisterFunction(module,&hosts[8],"SparkNvrmFirstLight") == SPARK_STATUS_OK &&
		SparkNvrmRegisterFunction(module,&hosts[9],"SparkNvrmStreamFill") == SPARK_STATUS_OK,"register overlaid and base functions");
	CHECK(SparkNvrmRegisteredOverlay(&hosts[8],&overlaid) == SPARK_STATUS_OK && overlaid == 1u,"matching kernel resolves to skl-overlay");
	CHECK(SparkNvrmRegisteredOverlay(&hosts[9],&overlaid) == SPARK_STATUS_OK && overlaid == 0u,"non-matching kernel stays on base cubin");
	unsetenv("SPARKPIPE_KERNEL_PROVIDER");
	unsetenv("SPARKPIPE_SKL_CUBIN_DIR");
	CHECK(SparkNvrmOverlayReload() == SPARK_STATUS_OK,"reset overlay to default after overlay tests");
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
	TestOverlay(root);
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

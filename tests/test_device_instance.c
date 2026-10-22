#include <dlfcn.h>
#include <libgen.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_device.h"

typedef SparkStatus (*InstanceBuffer)(SparkDeviceBuffer *buffer);
typedef const SparkDeviceProperties *(*InstanceProperties)(void);

static int failures;

#define CHECK(condition, what) do { if ( !(condition) ) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, what); failures++; } } while ( 0 )

static void *Open(const char *directory, const char *name)
{
	char path[4096];
	void *module;
	snprintf(path,sizeof(path),"%s/%s.%s",directory,name,SPARK_TEST_MODULE_EXT);
	module = dlopen(path,RTLD_NOW | RTLD_LOCAL);
	if ( module == 0 )
		printf("FAIL cannot load %s: %s\n",path,dlerror());
	return(module);
}

int main(int argc, char **argv)
{
	char self[4096];
	const char *directory;
	void *first, *second;
	InstanceBuffer allocate, release;
	InstanceProperties properties_a, properties_b;
	SparkDeviceBuffer buffer;
	SparkDeviceRange range;
	(void)argc;
	snprintf(self,sizeof(self),"%s",argv[0]);
	directory = dirname(self);
	CHECK(SparkDeviceInitialize(0) == SPARK_STATUS_OK,"initialize in the executable");
	first = Open(directory,"test_device_instance_a");
	second = Open(directory,"test_device_instance_b");
	if ( first == 0 || second == 0 )
		return 1;
	allocate = (InstanceBuffer)dlsym(first,"DeviceInstanceAllocate");
	release = (InstanceBuffer)dlsym(second,"DeviceInstanceFree");
	properties_a = (InstanceProperties)dlsym(first,"DeviceInstanceProperties");
	properties_b = (InstanceProperties)dlsym(second,"DeviceInstanceProperties");
	CHECK(allocate != 0 && release != 0 && properties_a != 0 && properties_b != 0,"both modules export their entry points");
	if ( failures != 0 )
		return 1;
	CHECK(properties_a() == SparkDeviceGetProperties() && properties_b() == SparkDeviceGetProperties(),
		"both modules see the executable's initialized device");
	CHECK(allocate(&buffer) == SPARK_STATUS_OK,"the first module allocates on the shared device");
	CHECK(SparkDeviceResolve(buffer.device + 10u,1u,&range) == SPARK_STATUS_OK && range.buffer->handle == buffer.handle,
		"the executable resolves the first module's buffer");
	CHECK(release(&buffer) == SPARK_STATUS_OK,"the second module frees the first module's buffer");
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_OK,"no buffer is left live in another copy of the device layer");
	dlclose(second);
	dlclose(first);
	if ( failures != 0 )
	{
		printf("device instance: %d failure(s)\n",failures);
		return 1;
	}
	printf("PASS device instance: one device layer across the executable and two modules\n");
	return 0;
}

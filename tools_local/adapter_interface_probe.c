#include <dlfcn.h>
#include <stdio.h>
#include "sparkpipe/spark_model_serving_adapter.h"

int main(int argc, char **argv)
{
    void *handle;
    SparkModelServingAdapterGetInterfaceFunction get_interface;
    SparkModelServingAdapterInterface *interface;

    (void)argc;
    handle = dlopen(argv[1], RTLD_NOW);
    if (handle == 0)
    {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }
    get_interface = (SparkModelServingAdapterGetInterfaceFunction)dlsym(handle,
        "SparkModelServingAdapterGetInterface");
    if (get_interface == 0)
    {
        printf("no symbol\n");
        return 1;
    }
    interface = get_interface();
    printf("so:      abi=%u interface_bytes=%u\n",
        interface->abi_version, interface->interface_bytes);
    printf("local:   abi=%u interface_bytes=%u\n",
        SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
        (unsigned)SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES);
    printf("sizeof(SparkModelServingAdapterInterface)=%zu\n",
        sizeof(SparkModelServingAdapterInterface));
    return 0;
}

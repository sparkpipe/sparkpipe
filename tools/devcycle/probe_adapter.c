// Probe a model serving adapter .so: print its interface descriptor and
// function-pointer nulls, mimicking SparkModelServingAdapterValidateInterface.
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>

typedef uint32_t SparkStatus;
typedef struct SparkModelServingAdapterInterface SparkModelServingAdapterInterface;
typedef struct SparkModelServingAdapterDescriptor SparkModelServingAdapterDescriptor;
typedef const SparkModelServingAdapterInterface *(*GetInterface)(void);

#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL 0x1u
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE 0x2u
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH 0x1000u
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV 0x2000u

struct SparkModelServingAdapterInterface {
    uint32_t abi_version;
    uint32_t interface_bytes;
    const SparkModelServingAdapterDescriptor *descriptor;
    void *initialize;
    void *destroy;
    void *validate_submission;
    void *submit;
    void *progress;
    void *quiesce;
    void *snapshot;
    void *prefetch;
    void *resolve_prefetch;
    void *reset;
};

struct SparkModelServingAdapterDescriptor {
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t capability_flags;
    uint32_t stage_count;
    uint32_t layer_count;
    uint32_t boundary_format;
    uint32_t boundary_element_count;
    uint32_t boundary_element_bytes;
    uint32_t linear_weight_codec;
    uint32_t expert_weight_codec;
    uint32_t kv_cache_codec;
    uint32_t max_inflight_submission_count;
    uint32_t max_active_sequence_count;
    uint32_t max_input_row_count;
    uint32_t max_resident_sequence_count;
    uint32_t max_output_token_count;
    uint32_t max_speculative_token_count;
    uint32_t resident_sequence_slot_reuse;
    const char *adapter_id;
    const char *model_id;
    const char *model_revision;
    const char *driver_program_name;
    const char *artifact_sha256;
};

int main(int argc, char **argv)
{
    void *h;
    GetInterface get_interface;
    const SparkModelServingAdapterInterface *iface;
    const SparkModelServingAdapterDescriptor *d;
    if (argc != 2) { fprintf(stderr, "usage: %s ADAPTER_SO\n", argv[0]); return 2; }
    h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    get_interface = (GetInterface)dlsym(h, "SparkModelServingAdapterGetInterface");
    if (!get_interface) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }
    iface = get_interface();
    if (!iface) { fprintf(stderr, "interface null\n"); return 1; }
    d = iface->descriptor;
    printf("abi_version=%u interface_bytes=%u\n", iface->abi_version, iface->interface_bytes);
    if (!d) { fprintf(stderr, "descriptor null\n"); return 1; }
    printf("descriptor abi=%u bytes=%u stage_count=%u layer_count=%u\n",
           d->abi_version, d->descriptor_bytes, d->stage_count, d->layer_count);
    printf("capability_flags=0x%08x prefill=%d decode=%d prefetch=%d jit_kv=%d\n",
           d->capability_flags,
           !!(d->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL),
           !!(d->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE),
           !!(d->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH),
           !!(d->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV));
    printf("adapter_id=%s\n", d->adapter_id ? d->adapter_id : "(null)");
    printf("model_id=%s\n", d->model_id ? d->model_id : "(null)");
    printf("program=%s\n", d->driver_program_name ? d->driver_program_name : "(null)");
    printf("artifact=%s\n", d->artifact_sha256 ? d->artifact_sha256 : "(null)");
    printf("fn initialize=%d destroy=%d validate=%d submit=%d progress=%d quiesce=%d snapshot=%d prefetch=%d resolve_prefetch=%d reset=%d\n",
           iface->initialize != 0, iface->destroy != 0,
           iface->validate_submission != 0, iface->submit != 0,
           iface->progress != 0, iface->quiesce != 0, iface->snapshot != 0,
           iface->prefetch != 0, iface->resolve_prefetch != 0,
           iface->reset != 0);
    return 0;
}

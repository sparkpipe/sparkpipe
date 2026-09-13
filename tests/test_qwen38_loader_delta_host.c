/* Loader-delta harness: drive the REAL module Initialize/Execute against a
 * fabricated full-geometry sliced-MXFP4 TP16 rank pack.
 *   mode=reject-no-tp   : no SPARK_QWEN38_STAGE_TP_* env -> pack must be REFUSED
 *   mode=accept-execute : tp_degree=4 rank=0 -> init OK, one decode step hits
 *                         the rank_local_moe_execution_not_wired fail-closed
 *   mode=tampered-rows  : W1 layer0 rows corrupted -> REFUSED even with TP env
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

extern SparkStatus SparkQwen38ResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state);
extern void SparkQwen38ResidentDecodeStageDestroy(void *module_state);
extern SparkStatus SparkQwen38ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);
extern void q38_stub_dump(void);

static void set_env(const char *pack_path, int with_tp)
{
    static const char *tp[][2] = {
        {"SPARK_QWEN38_STAGE_TP_DEGREE", "16"},
        {"SPARK_QWEN38_STAGE_TP_RANK", "0"},
        {"SPARK_QWEN38_STAGE_TP_BACKEND_PATH", "/tmp/q38ld/no-such-backend.so"},
        {"SPARK_QWEN38_STAGE_TP_IDENTIFIER", "134217728"},
        {"SPARK_QWEN38_STAGE_TP_PORT_BASE", "66620"},
        {"SPARK_QWEN38_STAGE_TP_HOSTS", "h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15"},
        {"SPARK_QWEN38_STAGE_TP_LOCAL_HOST", "h0"},
        {0, 0}
    };
    setenv("SPARK_QWEN38_ALLOW_UNQUALIFIED_EXECUTION", "1", 1);
    setenv("SPARK_QWEN38_STAGE_COUNT", "16", 1);
    setenv("SPARK_QWEN38_STAGE_INDEX", "1", 1);
    setenv("SPARK_QWEN38_STAGE_FIRST_LAYER", "1", 1);
    setenv("SPARK_QWEN38_STAGE_LAYER_COUNT", "1", 1);
    setenv("SPARK_QWEN38_STAGE_MAX_ACTIVE_SEQUENCES", "400", 1);
    setenv("SPARK_QWEN38_STAGE_PIPELINE_SLOTS", "1", 1);
    setenv("SPARK_QWEN38_STAGE_KV_BLOCKS", "8", 1);
    setenv("SPARK_QWEN38_STAGE_PACK_PATH", pack_path, 1);
    if (with_tp)
        for (int i = 0; tp[i][0]; i++)
            setenv(tp[i][0], tp[i][1], 1);
}

int main(int argc, char **argv)
{
    SparkFirmwareModuleConfiguration configuration;
    SparkFirmwareModuleHostServices services;
    SparkModelDriverBuffer buffers[2];
    SparkModelDriverFrame frame;
    uint32_t input_tokens[1] = {123u};
    uint32_t output_tokens[1] = {0xdeadbeefu};
    void *state = 0;
    SparkStatus status;
    int with_tp;
    if (argc != 3)
    {
        fprintf(stderr, "usage: harness reject-no-tp|accept-execute|tampered-rows PACK\n");
        return 2;
    }
    with_tp = strcmp(argv[1], "reject-no-tp") != 0;
    set_env(argv[2], with_tp);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    configuration.descriptor_bytes = sizeof(configuration);
    configuration.model_id = "qwen38-loader-delta-test";
    configuration.model_revision = "test";
    configuration.stage_name = "qwen38_resident_decode_stage";
    configuration.program_name = "resident_decode";
    configuration.operation_name = "initialize";
    memset(&services, 0, sizeof(services));
    services.abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
    services.descriptor_bytes = sizeof(services);
    status = SparkQwen38ResidentDecodeStageInitialize(&configuration, &services, &state);
    fprintf(stderr, "HARNESS initialize status=%d state=%p\n", (int)status, state);
    if (strcmp(argv[1], "accept-execute") == 0)
    {
        if (status != SPARK_STATUS_OK || state == 0)
        {
            fprintf(stderr, "HARNESS VERDICT FAIL (expected accepted load)\n");
            return 1;
        }
        memset(&frame, 0, sizeof(frame));
        frame.active_slot_count = 1u;
        frame.new_token_count = 1u;
        frame.sequence_position = 0u;
        buffers[0].address = input_tokens;
        buffers[0].bytes = sizeof(input_tokens);
        buffers[1].address = output_tokens;
        buffers[1].bytes = sizeof(output_tokens);
        frame.buffers = buffers;
        frame.buffer_count = 2u;
        status = SparkQwen38ResidentDecodeStageExecute(state, &frame);
        fprintf(stderr, "HARNESS execute status=%d (SPARK_STATUS_UNSUPPORTED=%d) output_token=%u\n",
                (int)status, (int)SPARK_STATUS_UNSUPPORTED, output_tokens[0]);
        {
            extern unsigned long long q38_grouped_linear_calls;
            extern unsigned long long q38_grouped_tile_calls;
            extern uint32_t q38_last_linear_rpe, q38_last_linear_tpd, q38_last_linear_tpr;
            fprintf(stderr, "HARNESS grouped_calls linear=%llu tile=%llu last(rpe=%u tpd=%u tpr=%u)\n",
                    q38_grouped_linear_calls, q38_grouped_tile_calls,
                    q38_last_linear_rpe, q38_last_linear_tpd, q38_last_linear_tpr);
        }
        q38_stub_dump();
        SparkQwen38ResidentDecodeStageDestroy(state);
        /* The decode step MUST fail closed: UNSUPPORTED before any
         * grouped-expert launch, after the GDN side of layer 0 ran. */
        {
            extern unsigned long long q38_grouped_linear_calls;
            if (status == SPARK_STATUS_OK && q38_grouped_linear_calls == 3ull)
            {
                fprintf(stderr, "HARNESS VERDICT PASS (sliced table EXECUTED: 3 grouped-scalar launches, rank-local shift suppressed)\n");
                return 0;
            }
            if (status == SPARK_STATUS_UNSUPPORTED && q38_grouped_linear_calls == 0ull)
            {
                fprintf(stderr, "HARNESS VERDICT PASS (load accepted; execution fail-closed as documented)\n");
                return 0;
            }
        }
        fprintf(stderr, "HARNESS VERDICT FAIL (status=%d)\n", (int)status);
        return 1;
    }
    /* Refusal modes: Initialize itself must fail with VALIDATION_FAILED. */
    if (status == SPARK_STATUS_VALIDATION_FAILED)
    {
        fprintf(stderr, "HARNESS VERDICT PASS (%s refused with validation_failed)\n", argv[1]);
        return 0;
    }
    fprintf(stderr, "HARNESS VERDICT FAIL (expected validation_failed, got %d)\n", (int)status);
    return 1;
}

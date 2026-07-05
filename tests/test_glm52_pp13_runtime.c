#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "sparkpipe/spark_glm52_pp13_runtime.h"

static void SparkTestWriteFile(const char *path)
{
    FILE *file;

    file = fopen(path,"wb");
    assert(file != 0);
    assert(fputs("x\n",file) >= 0);
    assert(fclose(file) == 0);
}

static void SparkTestGlm52Pp13RuntimeRankPlan(void)
{
    SparkGlm52Pp13RuntimeRankPlan rank_plan;
    char error_buffer[256];
    char host_name[16];

    assert(SparkGlm52Pp13RuntimeRankHostName(
        0u,host_name,sizeof(host_name)) == SPARK_STATUS_OK);
    assert(strcmp(host_name,"spark0") == 0);
    assert(SparkGlm52Pp13RuntimeRankHostName(
        10u,host_name,sizeof(host_name)) == SPARK_STATUS_OK);
    assert(strcmp(host_name,"sparka") == 0);
    assert(SparkGlm52Pp13RuntimeRankHostName(
        12u,host_name,sizeof(host_name)) == SPARK_STATUS_OK);
    assert(strcmp(host_name,"sparkc") == 0);
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        0u,1024u,52100u,&rank_plan,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_OK);
    assert(rank_plan.first_layer_index == 0u);
    assert(rank_plan.layer_count == 6u);
    assert(rank_plan.listen_port == 52100u);
    assert(rank_plan.next_port == 52101u);
    assert((rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_DENSE_PREFIX) != 0u);
    assert((rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) == 0u);
    assert(strcmp(rank_plan.next_host_name,"spark1") == 0);
    assert(strcmp(rank_plan.output_route_name,"spark0_to_spark1_hidden") == 0);
    assert(rank_plan.quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT);
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        12u,1024u,52100u,&rank_plan,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_OK);
    assert(rank_plan.first_layer_index == 72u);
    assert(rank_plan.layer_count == 6u);
    assert(rank_plan.listen_port == 52112u);
    assert(rank_plan.next_port == 0u);
    assert((rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u);
    assert((rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u);
    assert(strcmp(rank_plan.previous_host_name,"sparkb") == 0);
    assert(strcmp(rank_plan.input_route_name,"sparkb_to_sparkc_hidden") == 0);
}

static void SparkTestGlm52Pp13RuntimeFp8Packs(void)
{
    SparkGlm52Pp13RuntimeRankPlan rank_plan;
    char error_buffer[256];
    char pack_path[512];
    const char *pack_root;

    pack_root = "build/test_glm52_pp13_runtime_packs";
    (void)mkdir(pack_root,0775);
    (void)remove("build/test_glm52_pp13_runtime_packs/fp8_moe_pack_manifest.json");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0003_fp8_moe.spfp8");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0004_fp8_moe.spfp8");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0005_fp8_moe.spfp8");
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        0u,1024u,52100u,&rank_plan,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeValidateStageFp8PackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    SparkTestWriteFile("build/test_glm52_pp13_runtime_packs/fp8_moe_pack_manifest.json");
    assert(SparkGlm52Pp13RuntimeBuildFp8PackPath(
        pack_root,3u,pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    assert(strcmp(pack_path,
        "build/test_glm52_pp13_runtime_packs/glm52_layer_0003_fp8_moe.spfp8") == 0);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeBuildFp8PackPath(
        pack_root,4u,pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeValidateStageFp8PackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    assert(SparkGlm52Pp13RuntimeBuildFp8PackPath(
        pack_root,5u,pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeValidateStageFp8PackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_OK);
}

static void SparkTestGlm52Pp13RuntimeFinalEventRoute(void)
{
    SparkGlm52Pp13RuntimeFinalEventRoute route;
    char error_buffer[256];

    assert(SparkGlm52Pp13RuntimeBuildFinalEventRoute(
        52100u,&route,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(route.source_rank_index == 12u);
    assert(route.sink_rank_index == 0u);
    assert(route.listen_port == 52300u);
    assert(route.connect_port == 52300u);
    assert(strcmp(route.source_host_name,"sparkc") == 0);
    assert(strcmp(route.sink_host_name,"spark0") == 0);
    assert(strcmp(route.route_name,"sparkc_to_spark0_final_events") == 0);
    assert(SparkGlm52Pp13RuntimeValidateFinalEventRoute(
        &route,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    route.sink_rank_index = 1u;
    assert(SparkGlm52Pp13RuntimeValidateFinalEventRoute(
        &route,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    SparkTestGlm52Pp13RuntimeRankPlan();
    SparkTestGlm52Pp13RuntimeFp8Packs();
    SparkTestGlm52Pp13RuntimeFinalEventRoute();
    return 0;
}

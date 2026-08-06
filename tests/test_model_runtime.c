#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_runtime_contract.h"
#include "sparkpipe/spark_k3_runtime_contract.h"
#include "sparkpipe/spark_model_runtime.h"
#include "sparkpipe/spark_qwen36_runtime_contract.h"

#define TEST_OPERATION_CAPACITY 256u

typedef struct TestModelProviderContext
{
    uint32_t begin_count;
    uint32_t invoke_count;
    uint32_t commit_count;
    uint32_t cancel_count;
    uint32_t fail_operation_ordinal;
    uint64_t observed_operations[TEST_OPERATION_CAPACITY];
    SparkStatus last_cancel_reason;
} TestModelProviderContext;

typedef SparkModelRuntimeContract (*TestContractFactory)(void);

typedef struct TestContractCase
{
    const char *model_id;
    TestContractFactory factory;
    SparkModelRuntimeContractValidationFunction validator;
} TestContractCase;

static int TestExpect(
    int condition,
    const char *message)
{
    if (!condition)
    {
        fprintf(stderr,"FAIL: %s\n",message);
        return 0;
    }
    return 1;
}

static SparkStatus TestModelBegin(
    void *provider_context,
    const SparkRuntimeTransactionRequest *request)
{
    TestModelProviderContext *context;

    context = (TestModelProviderContext *)provider_context;
    if (context == 0 || request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    context->begin_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus TestModelInvoke(
    void *provider_context,
    const SparkRuntimeTransactionRequest *request,
    uint64_t operation,
    uint32_t operation_ordinal)
{
    TestModelProviderContext *context;

    context = (TestModelProviderContext *)provider_context;
    if (context == 0 || request == 0 ||
        operation_ordinal >= TEST_OPERATION_CAPACITY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    context->observed_operations[operation_ordinal] = operation;
    context->invoke_count += 1u;
    if (context->fail_operation_ordinal == operation_ordinal + 1u)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static void TestModelCommit(
    void *provider_context,
    const SparkRuntimeTransactionRequest *request)
{
    TestModelProviderContext *context;

    context = (TestModelProviderContext *)provider_context;
    if (context != 0 && request != 0)
    {
        context->commit_count += 1u;
    }
}

static void TestModelCancel(
    void *provider_context,
    const SparkRuntimeTransactionRequest *request,
    SparkStatus reason)
{
    TestModelProviderContext *context;

    context = (TestModelProviderContext *)provider_context;
    if (context != 0 && request != 0)
    {
        context->cancel_count += 1u;
        context->last_cancel_reason = reason;
    }
}

static uint32_t TestBuildOperationSequence(
    uint64_t required_operation_mask,
    uint64_t operations[TEST_OPERATION_CAPACITY])
{
    uint64_t operation;
    uint32_t operation_count;

    operation_count = 0u;
    for (operation = 1u;
         operation != 0u &&
            operation <= SPARK_MODEL_RUNTIME_OPERATION_SHARED_MOE;
         operation <<= 1u)
    {
        if ((required_operation_mask & operation) != 0u &&
            operation != SPARK_MODEL_RUNTIME_OPERATION_OUTPUT_HEAD)
        {
            operations[operation_count] = operation;
            operation_count += 1u;
        }
    }
    operations[operation_count] = SPARK_MODEL_RUNTIME_OPERATION_OUTPUT_HEAD;
    operation_count += 1u;
    return operation_count;
}

static void TestInitializeRequest(
    SparkRuntimeTransactionRequest *request)
{
    static const uint8_t payload[] = {1u,3u,3u,7u};

    memset(request,0,sizeof(*request));
    request->descriptor_bytes = SPARK_RUNTIME_TRANSACTION_REQUEST_BYTES;
    request->participant_count = 1u;
    request->identity.control_generation = 1u;
    request->identity.transaction_id = 2u;
    request->identity.dispatch_generation = 3u;
    request->identity.request_generation = 4u;
    request->identity.step_generation = 5u;
    request->identity.step_chunk_count = 1u;
    request->identity.phase = SPARK_WORK_TRANSACTION_PHASE_DECODE;
    request->request_id = 6u;
    request->sequence_id = 7u;
    request->payload = payload;
    request->payload_bytes = (uint32_t)sizeof(payload);
}

static void TestInitializeProvider(
    SparkModelRuntimeProvider *provider,
    TestModelProviderContext *context,
    const TestContractCase *contract_case,
    SparkModelRuntimeContract *contract,
    uint64_t operations[TEST_OPERATION_CAPACITY],
    uint32_t *operation_count_out)
{
    static const char artifact_sha256[] =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    memset(context,0,sizeof(*context));
    memset(provider,0,sizeof(*provider));
    *contract = contract_case->factory();
    *operation_count_out = TestBuildOperationSequence(
        contract->required_operation_mask,
        operations);
    provider->abi_version = SPARK_MODEL_RUNTIME_ABI_VERSION;
    provider->descriptor_bytes = SPARK_MODEL_RUNTIME_PROVIDER_BYTES;
    provider->flags = SPARK_MODEL_RUNTIME_PROVIDER_REQUIRED_FLAGS;
    provider->model_id = contract_case->model_id;
    provider->model_revision = "test-revision";
    provider->artifact_sha256 = artifact_sha256;
    provider->contract = contract;
    provider->supported_operation_mask = contract->required_operation_mask;
    provider->operation_sequence = operations;
    provider->operation_count = *operation_count_out;
    provider->validate_contract = contract_case->validator;
    provider->provider_context = context;
    provider->begin = TestModelBegin;
    provider->invoke = TestModelInvoke;
    provider->commit = TestModelCommit;
    provider->cancel = TestModelCancel;
}

static int TestContractExecution(
    const TestContractCase *contract_case)
{
    uint64_t operations[TEST_OPERATION_CAPACITY];
    TestModelProviderContext context;
    SparkModelRuntimeProvider provider;
    SparkModelRuntimeContract contract;
    SparkRuntimeTransactionRequest request;
    uint32_t operation_count;
    SparkStatus status;

    TestInitializeProvider(
        &provider,
        &context,
        contract_case,
        &contract,
        operations,
        &operation_count);
    TestInitializeRequest(&request);
    status = SparkModelRuntimeExecute(&provider,&request);
    return TestExpect(status == SPARK_STATUS_OK,"model execution status") &&
        TestExpect(context.begin_count == 1u,"model begin") &&
        TestExpect(context.invoke_count == operation_count,
            "all model operations invoked") &&
        TestExpect(context.commit_count == 1u,"model committed") &&
        TestExpect(context.cancel_count == 0u,"model not cancelled") &&
        TestExpect(operations[operation_count - 1u] ==
            SPARK_MODEL_RUNTIME_OPERATION_OUTPUT_HEAD,
            "output head is the terminal operation");
}

static int TestFailureCancels(
    const TestContractCase *contract_case)
{
    uint64_t operations[TEST_OPERATION_CAPACITY];
    TestModelProviderContext context;
    SparkModelRuntimeProvider provider;
    SparkModelRuntimeContract contract;
    SparkRuntimeTransactionRequest request;
    uint32_t operation_count;
    SparkStatus status;

    TestInitializeProvider(
        &provider,
        &context,
        contract_case,
        &contract,
        operations,
        &operation_count);
    context.fail_operation_ordinal = 3u;
    TestInitializeRequest(&request);
    status = SparkModelRuntimeExecute(&provider,&request);
    return TestExpect(status == SPARK_STATUS_IO_ERROR,
            "operation failure returned") &&
        TestExpect(context.commit_count == 0u,"failed model not committed") &&
        TestExpect(context.cancel_count == 1u,"failed model cancelled") &&
        TestExpect(context.last_cancel_reason == SPARK_STATUS_IO_ERROR,
            "cancel reason retained") &&
        TestExpect(operation_count >= context.invoke_count,
            "failure stopped later operations");
}

static int TestRejectsModifiedContract(
    const TestContractCase *contract_case)
{
    uint64_t operations[TEST_OPERATION_CAPACITY];
    TestModelProviderContext context;
    SparkModelRuntimeProvider provider;
    SparkModelRuntimeContract contract;
    uint32_t operation_count;
    SparkStatus status;

    TestInitializeProvider(
        &provider,
        &context,
        contract_case,
        &contract,
        operations,
        &operation_count);
    contract.precision_policy.nonexpert_weight_precision =
        SPARK_MODEL_RUNTIME_PRECISION_FP8_E4M3;
    status = SparkModelRuntimeValidateProvider(&provider);
    return TestExpect(status == SPARK_STATUS_VALIDATION_FAILED,
        "family validator rejects a modified precision contract");
}

static int TestDsv4GaBaselineExcludesDspark(void)
{
    SparkModelRuntimeContract contract;

    contract = SparkDsv4FlashRuntimeContract();
    return TestExpect(strcmp(contract.contract_id,
            "deepseek-v4-flash-0731-ga-baseline-mixed-v2") == 0,
            "DSV4 GA baseline contract identity") &&
        TestExpect((contract.required_operation_mask &
            SPARK_MODEL_RUNTIME_OPERATION_MULTI_TOKEN_PREDICTION) == 0u,
            "DSV4 GA baseline excludes DSpark");
}

int main(void)
{
    const TestContractCase contract_cases[] =
    {
        {"kimi-k3",SparkK3RuntimeContract,SparkK3RuntimeValidateContract},
        {"qwen-3.6-27b",SparkQwen36RuntimeContract,
            SparkQwen36RuntimeValidateContract},
        {"deepseek-v4-flash",SparkDsv4FlashRuntimeContract,
            SparkDsv4FlashRuntimeValidateContract},
        {"deepseek-v4-pro",SparkDsv4ProRuntimeContract,
            SparkDsv4ProRuntimeValidateContract}
    };
    uint32_t case_index;

    for (case_index = 0u;
         case_index < (uint32_t)(sizeof(contract_cases) /
            sizeof(contract_cases[0u]));
         ++case_index)
    {
        if (!TestContractExecution(&contract_cases[case_index]))
        {
            return 1;
        }
    }
    if (!TestFailureCancels(&contract_cases[0u]) ||
        !TestRejectsModifiedContract(&contract_cases[1u]) ||
        !TestDsv4GaBaselineExcludesDspark())
    {
        return 1;
    }
    printf("PASS model runtime operation and precision contracts\n");
    return 0;
}

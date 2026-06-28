#include "sparkpipe/spark_orchestrator.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"


typedef struct SparkOrchestratorNode
{
    bool active;
    char node_id[SPARK_ORCHESTRATOR_NAME_BYTES];
    char target[SPARK_ORCHESTRATOR_TARGET_BYTES];
    void *node_context;
} SparkOrchestratorNode;

typedef struct SparkOrchestratorDriverCompletionContext
{
    struct SparkOrchestrator *orchestrator;
    SparkOrchestratorDriverHandle driver_handle;
} SparkOrchestratorDriverCompletionContext;

typedef struct SparkOrchestratorDriver
{
    bool active;
    SparkOrchestratorNodeHandle node_handle;
    SparkLoadedModelDriver loaded_driver;
    void *driver_instance;
    atomic_uint_fast64_t outstanding;
    atomic_uint_fast64_t *program_outstanding;
    SparkOrchestratorDriverCompletionContext completion_context;
} SparkOrchestratorDriver;

typedef struct SparkOrchestratorRouteEndpoint
{
    SparkOrchestratorDriverHandle driver_handle;
    uint32_t program_index;
    const SparkModelDriverProgramDescriptor *program;
} SparkOrchestratorRouteEndpoint;

typedef struct SparkOrchestratorRoute
{
    bool active;
    char model_id[SPARK_ORCHESTRATOR_NAME_BYTES];
    char model_revision[SPARK_ORCHESTRATOR_NAME_BYTES];
    char stage_name[SPARK_ORCHESTRATOR_NAME_BYTES];
    char program_name[SPARK_ORCHESTRATOR_NAME_BYTES];
    uint32_t first_endpoint;
    uint32_t endpoint_count;
    atomic_uint_fast32_t next_endpoint;
} SparkOrchestratorRoute;

struct SparkOrchestrator
{
    SparkOrchestratorConfiguration configuration;
    SparkOrchestratorNode *nodes;
    SparkOrchestratorDriver *drivers;
    SparkOrchestratorRoute *routes;
    SparkOrchestratorRouteEndpoint *route_endpoints;
    uint32_t node_count;
    uint32_t driver_count;
    uint32_t route_count;
    uint32_t route_endpoint_count;
};

static void SparkOrchestratorDecrementIfNonzero(atomic_uint_fast64_t *counter)
{
    uint_fast64_t current_value;

    current_value = atomic_load_explicit(counter, memory_order_relaxed);
    while (current_value != 0u &&
           !atomic_compare_exchange_weak_explicit(
               counter,
               &current_value,
               current_value - 1u,
               memory_order_release,
               memory_order_relaxed))
    {
    }
}

static void SparkOrchestratorDriverCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkOrchestratorDriverCompletionContext *driver_completion_context;
    SparkOrchestrator *orchestrator;
    SparkOrchestratorDriver *driver;
    const SparkModelDriverDescriptor *descriptor;
    uint32_t program_index;

    driver_completion_context = (SparkOrchestratorDriverCompletionContext *)completion_context;
    if (driver_completion_context == 0 || completion == 0)
    {
        return;
    }
    orchestrator = driver_completion_context->orchestrator;
    if (orchestrator == 0 || driver_completion_context->driver_handle >= orchestrator->driver_count)
    {
        return;
    }
    driver = &orchestrator->drivers[driver_completion_context->driver_handle];
    descriptor = driver->loaded_driver.interface->descriptor;
    for (program_index = 0u; program_index < descriptor->program_count; ++program_index)
    {
        if (descriptor->programs[program_index].program_id == completion->program_id)
        {
            SparkOrchestratorDecrementIfNonzero(&driver->program_outstanding[program_index]);
            SparkOrchestratorDecrementIfNonzero(&driver->outstanding);
            break;
        }
    }
    if (orchestrator->configuration.completion_function != 0)
    {
        orchestrator->configuration.completion_function(orchestrator->configuration.completion_context, completion);
    }
}

SparkStatus SparkCreateOrchestrator(
    const SparkOrchestratorConfiguration *configuration,
    SparkOrchestrator **orchestrator)
{
    SparkOrchestrator *created_orchestrator;

    if (configuration == 0 || orchestrator == 0 || configuration->node_capacity == 0u ||
        configuration->driver_capacity == 0u || configuration->route_capacity == 0u ||
        configuration->route_endpoint_capacity == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *orchestrator = 0;
    created_orchestrator = (SparkOrchestrator *)calloc(1u, sizeof(*created_orchestrator));
    if (created_orchestrator == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    created_orchestrator->nodes = (SparkOrchestratorNode *)calloc(configuration->node_capacity, sizeof(*created_orchestrator->nodes));
    created_orchestrator->drivers = (SparkOrchestratorDriver *)calloc(configuration->driver_capacity, sizeof(*created_orchestrator->drivers));
    created_orchestrator->routes = (SparkOrchestratorRoute *)calloc(configuration->route_capacity, sizeof(*created_orchestrator->routes));
    created_orchestrator->route_endpoints = (SparkOrchestratorRouteEndpoint *)calloc(configuration->route_endpoint_capacity, sizeof(*created_orchestrator->route_endpoints));
    if (created_orchestrator->nodes == 0 || created_orchestrator->drivers == 0 ||
        created_orchestrator->routes == 0 || created_orchestrator->route_endpoints == 0)
    {
        SparkDestroyOrchestrator(created_orchestrator);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    created_orchestrator->configuration = *configuration;
    *orchestrator = created_orchestrator;
    return SPARK_STATUS_OK;
}

void SparkDestroyOrchestrator(SparkOrchestrator *orchestrator)
{
    uint32_t driver_index;

    if (orchestrator == 0)
    {
        return;
    }
    for (driver_index = orchestrator->driver_count; driver_index != 0u; --driver_index)
    {
        SparkOrchestratorDriver *driver;

        driver = &orchestrator->drivers[driver_index - 1u];
        if (driver->active && driver->loaded_driver.interface != 0)
        {
            driver->loaded_driver.interface->destroy(driver->driver_instance);
            driver->driver_instance = 0;
            free(driver->program_outstanding);
            driver->program_outstanding = 0;
            SparkUnloadModelDriver(&driver->loaded_driver);
            driver->active = false;
        }
    }
    free(orchestrator->route_endpoints);
    free(orchestrator->routes);
    free(orchestrator->drivers);
    free(orchestrator->nodes);
    free(orchestrator);
}

SparkStatus SparkOrchestratorAddNode(
    SparkOrchestrator *orchestrator,
    const char *node_id,
    const char *target,
    void *node_context,
    SparkOrchestratorNodeHandle *node_handle)
{
    uint32_t node_index;
    SparkOrchestratorNode *node;
    SparkStatus status;

    if (orchestrator == 0 || node_id == 0 || target == 0 || node_handle == 0 || node_id[0] == '\0' || target[0] == '\0')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (orchestrator->route_count != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    if (orchestrator->node_count >= orchestrator->configuration.node_capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    for (node_index = 0u; node_index < orchestrator->node_count; ++node_index)
    {
        if (strcmp(orchestrator->nodes[node_index].node_id, node_id) == 0)
        {
            return SPARK_STATUS_DUPLICATE;
        }
    }

    node = &orchestrator->nodes[orchestrator->node_count];
    status = SparkCopyString(node->node_id, sizeof(node->node_id), node_id);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkCopyString(node->target, sizeof(node->target), target);
    if (status != SPARK_STATUS_OK)
    {
        memset(node, 0, sizeof(*node));
        return status;
    }
    node->node_context = node_context;
    node->active = true;
    *node_handle = orchestrator->node_count;
    orchestrator->node_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkOrchestratorAttachDriver(
    SparkOrchestrator *orchestrator,
    SparkOrchestratorNodeHandle node_handle,
    const char *driver_path,
    SparkOrchestratorDriverHandle *driver_handle,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkOrchestratorNode *node;
    SparkOrchestratorDriver *driver;
    SparkModelDriverCreateRequest create_request;
    SparkStatus status;

    if (orchestrator == 0 || driver_path == 0 || driver_handle == 0 || node_handle >= orchestrator->node_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (orchestrator->route_count != 0u)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "drivers must be attached before routes are resolved");
        return SPARK_STATUS_BUSY;
    }
    if (orchestrator->driver_count >= orchestrator->configuration.driver_capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    node = &orchestrator->nodes[node_handle];
    if (!node->active)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    driver = &orchestrator->drivers[orchestrator->driver_count];
    SparkLoadedModelDriverReset(&driver->loaded_driver);
    status = SparkLoadModelDriver(driver_path, node->target, &driver->loaded_driver, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    driver->program_outstanding = (atomic_uint_fast64_t *)calloc(
        driver->loaded_driver.interface->descriptor->program_count,
        sizeof(*driver->program_outstanding));
    if (driver->program_outstanding == 0)
    {
        SparkUnloadModelDriver(&driver->loaded_driver);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    {
        uint32_t program_index;

        for (program_index = 0u; program_index < driver->loaded_driver.interface->descriptor->program_count; ++program_index)
        {
            atomic_init(&driver->program_outstanding[program_index], 0u);
        }
    }
    driver->completion_context.orchestrator = orchestrator;
    driver->completion_context.driver_handle = orchestrator->driver_count;
    memset(&create_request, 0, sizeof(create_request));
    create_request.node_id = node->node_id;
    create_request.node_target = node->target;
    create_request.node_context = node->node_context;
    create_request.completion_function = SparkOrchestratorDriverCompletion;
    create_request.completion_context = &driver->completion_context;
    status = driver->loaded_driver.interface->create(&create_request, &driver->driver_instance);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "driver instance creation failed with %s", SparkStatusToString(status));
        free(driver->program_outstanding);
        driver->program_outstanding = 0;
        SparkUnloadModelDriver(&driver->loaded_driver);
        memset(driver, 0, sizeof(*driver));
        return status;
    }
    driver->node_handle = node_handle;
    atomic_init(&driver->outstanding, 0u);
    driver->active = true;
    *driver_handle = orchestrator->driver_count;
    orchestrator->driver_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkOrchestratorRoute *SparkFindOrchestratorRoute(
    SparkOrchestrator *orchestrator,
    const char *model_id,
    const char *model_revision,
    const char *stage_name,
    const char *program_name,
    SparkOrchestratorRouteHandle *route_handle)
{
    uint32_t route_index;

    for (route_index = 0u; route_index < orchestrator->route_count; ++route_index)
    {
        SparkOrchestratorRoute *route;

        route = &orchestrator->routes[route_index];
        if (route->active && strcmp(route->model_id, model_id) == 0 &&
            strcmp(route->model_revision, model_revision) == 0 &&
            strcmp(route->stage_name, stage_name) == 0 && strcmp(route->program_name, program_name) == 0)
        {
            if (route_handle != 0)
            {
                *route_handle = route_index;
            }
            return route;
        }
    }
    return 0;
}

SparkStatus SparkOrchestratorResolveRoute(
    SparkOrchestrator *orchestrator,
    const char *model_id,
    const char *model_revision,
    const char *stage_name,
    const char *program_name,
    SparkOrchestratorRouteHandle *route_handle)
{
    SparkOrchestratorRoute *route;
    const char *model_description_sha256;
    const char *compiled_program_sha256;
    uint32_t driver_index;
    uint32_t endpoint_count;
    SparkStatus status;

    if (orchestrator == 0 || model_id == 0 || model_revision == 0 || stage_name == 0 ||
        program_name == 0 || route_handle == 0 || model_id[0] == '\0' || model_revision[0] == '\0' ||
        stage_name[0] == '\0' || program_name[0] == '\0')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    route = SparkFindOrchestratorRoute(orchestrator, model_id, model_revision, stage_name, program_name, route_handle);
    if (route != 0)
    {
        return SPARK_STATUS_OK;
    }
    if (orchestrator->route_count >= orchestrator->configuration.route_capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    endpoint_count = 0u;
    model_description_sha256 = 0;
    compiled_program_sha256 = 0;
    for (driver_index = 0u; driver_index < orchestrator->driver_count; ++driver_index)
    {
        SparkOrchestratorDriver *driver;
        const SparkModelDriverDescriptor *descriptor;
        const SparkModelDriverProgramDescriptor *program;

        driver = &orchestrator->drivers[driver_index];
        if (!driver->active)
        {
            continue;
        }
        descriptor = driver->loaded_driver.interface->descriptor;
        if (strcmp(descriptor->model_id, model_id) != 0 ||
            strcmp(descriptor->model_revision, model_revision) != 0 ||
            strcmp(descriptor->stage_name, stage_name) != 0)
        {
            continue;
        }
        program = SparkFindLoadedModelDriverProgram(&driver->loaded_driver, program_name);
        if (program != 0)
        {
            if (model_description_sha256 == 0)
            {
                model_description_sha256 = descriptor->model_description_sha256;
                compiled_program_sha256 = descriptor->compiled_program_sha256;
            }
            else if (strcmp(model_description_sha256, descriptor->model_description_sha256) != 0 ||
                     strcmp(compiled_program_sha256, descriptor->compiled_program_sha256) != 0)
            {
                return SPARK_STATUS_HASH_MISMATCH;
            }
            endpoint_count += 1u;
        }
    }
    if (endpoint_count == 0u)
    {
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    if (endpoint_count > orchestrator->configuration.route_endpoint_capacity - orchestrator->route_endpoint_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    route = &orchestrator->routes[orchestrator->route_count];
    status = SparkCopyString(route->model_id, sizeof(route->model_id), model_id);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkCopyString(route->model_revision, sizeof(route->model_revision), model_revision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkCopyString(route->stage_name, sizeof(route->stage_name), stage_name);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkCopyString(route->program_name, sizeof(route->program_name), program_name);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    route->first_endpoint = orchestrator->route_endpoint_count;
    route->endpoint_count = 0u;
    atomic_init(&route->next_endpoint, 0u);

    for (driver_index = 0u; driver_index < orchestrator->driver_count; ++driver_index)
    {
        SparkOrchestratorDriver *driver;
        const SparkModelDriverDescriptor *descriptor;
        const SparkModelDriverProgramDescriptor *program;

        driver = &orchestrator->drivers[driver_index];
        if (!driver->active)
        {
            continue;
        }
        descriptor = driver->loaded_driver.interface->descriptor;
        if (strcmp(descriptor->model_id, model_id) != 0 ||
            strcmp(descriptor->model_revision, model_revision) != 0 ||
            strcmp(descriptor->stage_name, stage_name) != 0 ||
            strcmp(descriptor->model_description_sha256, model_description_sha256) != 0 ||
            strcmp(descriptor->compiled_program_sha256, compiled_program_sha256) != 0)
        {
            continue;
        }
        program = SparkFindLoadedModelDriverProgram(&driver->loaded_driver, program_name);
        if (program != 0)
        {
            SparkOrchestratorRouteEndpoint *endpoint;

            endpoint = &orchestrator->route_endpoints[orchestrator->route_endpoint_count++];
            endpoint->driver_handle = driver_index;
            endpoint->program_index = (uint32_t)(program - descriptor->programs);
            endpoint->program = program;
            route->endpoint_count += 1u;
        }
    }
    route->active = true;
    *route_handle = orchestrator->route_count;
    orchestrator->route_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkBuildDriverAdmissionRequest(
    const SparkOrchestratorRouteEndpoint *endpoint,
    const SparkModelDriverFrame *frame,
    SparkModelDriverAdmissionRequest *request)
{
    memset(request, 0, sizeof(*request));
    request->descriptor_bytes = sizeof(*request);
    request->program_id = endpoint->program->program_id;
    request->request_id = frame->request_id;
    request->sequence_id = frame->sequence_id;
    request->sequence_position = frame->sequence_position;
    request->deadline_time_ns = frame->deadline_time_ns;
    request->active_slot_count = frame->active_slot_count;
    request->new_token_count = frame->new_token_count != 0u ? frame->new_token_count : 1u;
    request->priority = frame->priority;
    request->frame_flags = frame->flags;
    request->residency = frame->residency;
}

static uint64_t SparkDriverAdmissionDecisionCost(
    const SparkModelDriverAdmissionDecision *decision)
{
    uint64_t cost;

    cost = decision->endpoint_cost;
    cost += decision->estimated_queue_delay_ns;
    cost += decision->estimated_service_time_ns;
    cost += ((uint64_t)decision->private_queue_pressure) << 20u;
    cost += decision->host_staging_bytes << 2u;
    cost += decision->device_memcpy_bytes;
    if (decision->residency_match_score > cost)
    {
        return 0u;
    }
    return cost - decision->residency_match_score;
}

static bool SparkTryReserveRouteEndpointOutstanding(
    SparkOrchestrator *orchestrator,
    const SparkOrchestratorRouteEndpoint *endpoint)
{
    SparkOrchestratorDriver *driver;
    atomic_uint_fast64_t *program_outstanding;
    uint_fast64_t outstanding;

    driver = &orchestrator->drivers[endpoint->driver_handle];
    program_outstanding = &driver->program_outstanding[endpoint->program_index];
    outstanding = atomic_load_explicit(program_outstanding, memory_order_relaxed);
    while (outstanding < endpoint->program->max_inflight)
    {
        if (atomic_compare_exchange_weak_explicit(
                program_outstanding,
                &outstanding,
                outstanding + 1u,
                memory_order_acquire,
                memory_order_relaxed))
        {
            atomic_fetch_add_explicit(&driver->outstanding, 1u, memory_order_relaxed);
            return true;
        }
    }
    return false;
}

static SparkStatus SparkReserveRouteEndpoint(
    SparkOrchestrator *orchestrator,
    SparkOrchestratorRoute *route,
    const SparkModelDriverFrame *frame,
    SparkOrchestratorRouteEndpoint **reserved_endpoint,
    SparkModelDriverAdmissionDecision *reserved_decision)
{
    uint_fast32_t starting_endpoint;
    uint32_t attempt_index;
    uint32_t selected_endpoint_index;
    bool selected_endpoint_valid;
    uint64_t selected_cost;

    starting_endpoint = atomic_fetch_add_explicit(&route->next_endpoint, 1u, memory_order_relaxed);
    selected_endpoint_index = 0u;
    selected_endpoint_valid = false;
    selected_cost = UINT64_MAX;
    memset(reserved_decision, 0, sizeof(*reserved_decision));

    for (attempt_index = 0u; attempt_index < route->endpoint_count; ++attempt_index)
    {
        uint32_t route_endpoint_index;
        SparkOrchestratorRouteEndpoint *endpoint;
        SparkOrchestratorDriver *driver;
        SparkModelDriverAdmissionRequest admission_request;
        SparkModelDriverAdmissionDecision admission_decision;
        SparkStatus admission_status;
        uint64_t endpoint_cost;

        route_endpoint_index = (uint32_t)((starting_endpoint + attempt_index) % route->endpoint_count);
        endpoint = &orchestrator->route_endpoints[route->first_endpoint + route_endpoint_index];
        driver = &orchestrator->drivers[endpoint->driver_handle];
        if (atomic_load_explicit(&driver->program_outstanding[endpoint->program_index], memory_order_relaxed) >=
            endpoint->program->max_inflight)
        {
            continue;
        }
        SparkBuildDriverAdmissionRequest(endpoint, frame, &admission_request);
        memset(&admission_decision, 0, sizeof(admission_decision));
        admission_status = driver->loaded_driver.interface->admit(
            driver->driver_instance,
            &admission_request,
            &admission_decision);
        if (admission_status != SPARK_STATUS_OK ||
            admission_decision.descriptor_bytes < sizeof(admission_decision) ||
            admission_decision.accepted == 0u)
        {
            continue;
        }
        endpoint_cost = SparkDriverAdmissionDecisionCost(&admission_decision);
        if (!selected_endpoint_valid || endpoint_cost < selected_cost)
        {
            selected_endpoint_index = route_endpoint_index;
            selected_endpoint_valid = true;
            selected_cost = endpoint_cost;
            *reserved_decision = admission_decision;
        }
    }

    if (!selected_endpoint_valid)
    {
        return SPARK_STATUS_BUSY;
    }
    *reserved_endpoint = &orchestrator->route_endpoints[route->first_endpoint + selected_endpoint_index];
    if (!SparkTryReserveRouteEndpointOutstanding(orchestrator, *reserved_endpoint))
    {
        *reserved_endpoint = 0;
        return SPARK_STATUS_BUSY;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkOrchestratorSubmit(
    SparkOrchestrator *orchestrator,
    SparkOrchestratorRouteHandle route_handle,
    SparkModelDriverFrame *frame)
{
    SparkOrchestratorRoute *route;
    uint32_t retry_index;
    uint32_t retry_limit;
    uint32_t base_frame_flags;
    uint32_t base_dispatch_slot;
    uint64_t base_dispatch_generation;
    uint64_t base_dispatch_cookie0;
    uint64_t base_dispatch_cookie1;

    if (orchestrator == 0 || frame == 0 || route_handle >= orchestrator->route_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    route = &orchestrator->routes[route_handle];
    if (!route->active || route->endpoint_count == 0u)
    {
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }

    base_frame_flags =
        frame->flags & ~SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID;
    base_dispatch_slot = frame->driver_dispatch_slot;
    base_dispatch_generation = frame->driver_dispatch_generation;
    base_dispatch_cookie0 = frame->driver_dispatch_cookie0;
    base_dispatch_cookie1 = frame->driver_dispatch_cookie1;
    retry_limit = route->endpoint_count + 1u;
    for (retry_index = 0u; retry_index < retry_limit; ++retry_index)
    {
        SparkOrchestratorRouteEndpoint *endpoint;
        SparkOrchestratorDriver *driver;
        SparkModelDriverAdmissionDecision admission_decision;
        SparkStatus status;

        endpoint = 0;
        frame->flags = base_frame_flags;
        frame->driver_dispatch_slot = base_dispatch_slot;
        frame->driver_dispatch_generation = base_dispatch_generation;
        frame->driver_dispatch_cookie0 = base_dispatch_cookie0;
        frame->driver_dispatch_cookie1 = base_dispatch_cookie1;
        status = SparkReserveRouteEndpoint(
            orchestrator,
            route,
            frame,
            &endpoint,
            &admission_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        driver = &orchestrator->drivers[endpoint->driver_handle];
        if (admission_decision.driver_dispatch_slot !=
            SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT)
        {
            frame->driver_dispatch_slot = admission_decision.driver_dispatch_slot;
            frame->driver_dispatch_generation =
                admission_decision.driver_dispatch_generation;
            frame->driver_dispatch_cookie0 = admission_decision.driver_dispatch_cookie0;
            frame->driver_dispatch_cookie1 = admission_decision.driver_dispatch_cookie1;
            frame->flags |= SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID;
        }
        status = endpoint->program->submit(driver->driver_instance, frame);
        if (status == SPARK_STATUS_OK)
        {
            return SPARK_STATUS_OK;
        }
        SparkOrchestratorDecrementIfNonzero(
            &driver->program_outstanding[endpoint->program_index]);
        SparkOrchestratorDecrementIfNonzero(&driver->outstanding);
        if (status != SPARK_STATUS_BUSY)
        {
            return status;
        }
    }
    return SPARK_STATUS_BUSY;
}

uint64_t SparkOrchestratorGetDriverOutstanding(
    const SparkOrchestrator *orchestrator,
    SparkOrchestratorDriverHandle driver_handle)
{
    if (orchestrator == 0 || driver_handle >= orchestrator->driver_count || !orchestrator->drivers[driver_handle].active)
    {
        return UINT64_MAX;
    }
    return atomic_load_explicit(&orchestrator->drivers[driver_handle].outstanding, memory_order_acquire);
}


SparkStatus SparkOrchestratorGetDriverProgramSnapshot(
    SparkOrchestrator *orchestrator,
    SparkOrchestratorDriverHandle driver_handle,
    const char *program_name,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
    SparkOrchestratorDriver *driver;
    const SparkModelDriverProgramDescriptor *program;

    if (orchestrator == 0 || program_name == 0 || snapshot == 0 ||
        driver_handle >= orchestrator->driver_count ||
        !orchestrator->drivers[driver_handle].active)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    driver = &orchestrator->drivers[driver_handle];
    program = SparkFindLoadedModelDriverProgram(&driver->loaded_driver, program_name);
    if (program == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    return driver->loaded_driver.interface->snapshot(
        driver->driver_instance,
        program->program_id,
        snapshot);
}

#include "sparkpipe/spark_glm52_service.h"

#include <limits.h>
#include <string.h>

static uint32_t SparkGlm52ServiceNormalizeFlags(
    uint32_t flags)
{
    if (flags == 0u)
    {
        return SPARK_GLM52_SERVICE_CONFIGURATION_DEFAULT_FLAGS;
    }
    return flags;
}

static uint32_t SparkGlm52ServiceNormalizePumpDispatchSteps(
    uint32_t default_pump_dispatch_steps)
{
    if (default_pump_dispatch_steps == 0u)
    {
        return SPARK_GLM52_SERVICE_DEFAULT_PUMP_DISPATCH_STEPS;
    }
    return default_pump_dispatch_steps;
}

static uint64_t SparkGlm52ServiceNormalizeRequestIdBase(
    uint64_t request_id_base)
{
    if (request_id_base == 0u)
    {
        return SPARK_GLM52_SERVICE_DEFAULT_REQUEST_ID_BASE;
    }
    return request_id_base;
}

static void SparkGlm52ServiceInitializeClientSession(
    SparkGlm52ServiceClientSession *client_session)
{
    memset(client_session, 0, sizeof(*client_session));
    client_session->abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
    client_session->descriptor_bytes =
        SPARK_GLM52_SERVICE_CLIENT_SESSION_DESCRIPTOR_BYTES;
    client_session->state = SPARK_GLM52_SERVICE_CLIENT_STATE_FREE;
    client_session->client_hash_next = SPARK_GLM52_SERVICE_NO_HASH_SLOT;
}

static void SparkGlm52ServiceInitializeRequestMap(
    SparkGlm52ServiceRequestMap *request_map)
{
    memset(request_map, 0, sizeof(*request_map));
    request_map->abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
    request_map->descriptor_bytes =
        SPARK_GLM52_SERVICE_REQUEST_MAP_DESCRIPTOR_BYTES;
    request_map->state = SPARK_GLM52_SERVICE_REQUEST_STATE_FREE;
    request_map->client_request_hash_next =
        SPARK_GLM52_SERVICE_NO_HASH_SLOT;
    request_map->serving_handle_hash_next =
        SPARK_GLM52_SERVICE_NO_HASH_SLOT;
}

static uint32_t SparkGlm52ServiceHash64(
    uint64_t value,
    uint32_t slot_count)
{
    uint64_t hash;

    hash = value;
    hash ^= (hash >> 33u);
    hash *= 0xff51afd7ed558ccdull;
    hash ^= (hash >> 33u);
    return (uint32_t)(hash % slot_count);
}

static uint32_t SparkGlm52ServiceHashClientRequest(
    SparkGlm52ServiceClientId client_id,
    SparkGlm52ServiceRequestId client_request_id)
{
    uint64_t hash;

    hash = client_id ^ (client_request_id + 0x9e3779b97f4a7c15ull +
        (client_id << 6u) + (client_id >> 2u));
    return SparkGlm52ServiceHash64(
        hash,
        SPARK_GLM52_SERVICE_REQUEST_MAP_HASH_SLOTS);
}

static uint32_t SparkGlm52ServiceClientIndex(
    const SparkGlm52ServiceRuntime *service,
    const SparkGlm52ServiceClientSession *client_session)
{
    uint64_t byte_offset;
    uint64_t client_index;

    if (service == 0 || client_session == 0 ||
        service->client_sessions == 0 ||
        client_session < service->client_sessions ||
        client_session >=
            &service->client_sessions[service->client_session_capacity])
    {
        return SPARK_GLM52_SERVICE_NO_HASH_SLOT;
    }
    byte_offset = (uint64_t)((uintptr_t)client_session -
        (uintptr_t)service->client_sessions);
    client_index = byte_offset / (uint64_t)sizeof(*client_session);
    if (client_index >= service->client_session_capacity)
    {
        return SPARK_GLM52_SERVICE_NO_HASH_SLOT;
    }
    return (uint32_t)client_index;
}

static uint32_t SparkGlm52ServiceRequestMapIndex(
    const SparkGlm52ServiceRuntime *service,
    const SparkGlm52ServiceRequestMap *request_map)
{
    uint64_t byte_offset;
    uint64_t request_index;

    if (service == 0 || request_map == 0 || service->request_maps == 0 ||
        request_map < service->request_maps ||
        request_map >= &service->request_maps[service->request_map_capacity])
    {
        return SPARK_GLM52_SERVICE_NO_HASH_SLOT;
    }
    byte_offset = (uint64_t)((uintptr_t)request_map -
        (uintptr_t)service->request_maps);
    request_index = byte_offset / (uint64_t)sizeof(*request_map);
    if (request_index >= service->request_map_capacity)
    {
        return SPARK_GLM52_SERVICE_NO_HASH_SLOT;
    }
    return (uint32_t)request_index;
}

static void SparkGlm52ServiceInitializeEvent(
    SparkGlm52ServiceEvent *event)
{
    memset(event, 0, sizeof(*event));
    event->abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
    event->descriptor_bytes = SPARK_GLM52_SERVICE_EVENT_DESCRIPTOR_BYTES;
}

static SparkStatus SparkGlm52ServiceValidateRuntime(
    SparkGlm52ServiceRuntime *service)
{
    if (service == 0 ||
        service->abi_version != SPARK_GLM52_SERVICE_ABI_VERSION ||
        service->descriptor_bytes != SPARK_GLM52_SERVICE_RUNTIME_DESCRIPTOR_BYTES ||
        service->serving_engine == 0 ||
        service->client_sessions == 0 ||
        service->request_maps == 0 ||
        service->event_ring == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52ServiceInsertClientHash(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientSession *client_session)
{
    uint32_t client_index;
    uint32_t hash_slot;

    client_index = SparkGlm52ServiceClientIndex(service, client_session);
    if (client_index == SPARK_GLM52_SERVICE_NO_HASH_SLOT ||
        client_session->client_id == 0u)
    {
        return;
    }
    hash_slot = SparkGlm52ServiceHash64(
        client_session->client_id,
        SPARK_GLM52_SERVICE_CLIENT_HASH_SLOTS);
    client_session->client_hash_next = service->client_hash_heads[hash_slot];
    service->client_hash_heads[hash_slot] = client_index;
}

static void SparkGlm52ServiceRemoveClientHash(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientSession *client_session)
{
    uint32_t client_index;
    uint32_t hash_slot;
    uint32_t current_index;
    uint32_t previous_index;

    client_index = SparkGlm52ServiceClientIndex(service, client_session);
    if (client_index == SPARK_GLM52_SERVICE_NO_HASH_SLOT ||
        client_session->client_id == 0u)
    {
        return;
    }
    hash_slot = SparkGlm52ServiceHash64(
        client_session->client_id,
        SPARK_GLM52_SERVICE_CLIENT_HASH_SLOTS);
    current_index = service->client_hash_heads[hash_slot];
    previous_index = SPARK_GLM52_SERVICE_NO_HASH_SLOT;
    while (current_index != SPARK_GLM52_SERVICE_NO_HASH_SLOT)
    {
        if (current_index == client_index)
        {
            if (previous_index == SPARK_GLM52_SERVICE_NO_HASH_SLOT)
            {
                service->client_hash_heads[hash_slot] =
                    service->client_sessions[current_index].client_hash_next;
            }
            else
            {
                service->client_sessions[previous_index].client_hash_next =
                    service->client_sessions[current_index].client_hash_next;
            }
            service->client_sessions[current_index].client_hash_next =
                SPARK_GLM52_SERVICE_NO_HASH_SLOT;
            return;
        }
        previous_index = current_index;
        current_index = service->client_sessions[current_index].client_hash_next;
    }
}

static void SparkGlm52ServiceInsertRequestMapHash(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceRequestMap *request_map)
{
    uint32_t request_index;
    uint32_t hash_slot;

    request_index = SparkGlm52ServiceRequestMapIndex(service, request_map);
    if (request_index == SPARK_GLM52_SERVICE_NO_HASH_SLOT)
    {
        return;
    }
    hash_slot = SparkGlm52ServiceHashClientRequest(
        request_map->client_id,
        request_map->client_request_id);
    request_map->client_request_hash_next =
        service->client_request_hash_heads[hash_slot];
    service->client_request_hash_heads[hash_slot] = request_index;
    hash_slot = SparkGlm52ServiceHash64(
        request_map->serving_request_handle,
        SPARK_GLM52_SERVICE_REQUEST_MAP_HASH_SLOTS);
    request_map->serving_handle_hash_next =
        service->serving_handle_hash_heads[hash_slot];
    service->serving_handle_hash_heads[hash_slot] = request_index;
}

static void SparkGlm52ServiceRemoveRequestMapHash(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceRequestMap *request_map)
{
    uint32_t request_index;
    uint32_t hash_slot;
    uint32_t current_index;
    uint32_t previous_index;

    request_index = SparkGlm52ServiceRequestMapIndex(service, request_map);
    if (request_index == SPARK_GLM52_SERVICE_NO_HASH_SLOT)
    {
        return;
    }
    hash_slot = SparkGlm52ServiceHashClientRequest(
        request_map->client_id,
        request_map->client_request_id);
    current_index = service->client_request_hash_heads[hash_slot];
    previous_index = SPARK_GLM52_SERVICE_NO_HASH_SLOT;
    while (current_index != SPARK_GLM52_SERVICE_NO_HASH_SLOT)
    {
        if (current_index == request_index)
        {
            if (previous_index == SPARK_GLM52_SERVICE_NO_HASH_SLOT)
                service->client_request_hash_heads[hash_slot] =
                    service->request_maps[current_index].client_request_hash_next;
            else
                service->request_maps[previous_index].client_request_hash_next =
                    service->request_maps[current_index].client_request_hash_next;
            break;
        }
        previous_index = current_index;
        current_index =
            service->request_maps[current_index].client_request_hash_next;
    }
    hash_slot = SparkGlm52ServiceHash64(
        request_map->serving_request_handle,
        SPARK_GLM52_SERVICE_REQUEST_MAP_HASH_SLOTS);
    current_index = service->serving_handle_hash_heads[hash_slot];
    previous_index = SPARK_GLM52_SERVICE_NO_HASH_SLOT;
    while (current_index != SPARK_GLM52_SERVICE_NO_HASH_SLOT)
    {
        if (current_index == request_index)
        {
            if (previous_index == SPARK_GLM52_SERVICE_NO_HASH_SLOT)
                service->serving_handle_hash_heads[hash_slot] =
                    service->request_maps[current_index].serving_handle_hash_next;
            else
                service->request_maps[previous_index].serving_handle_hash_next =
                    service->request_maps[current_index].serving_handle_hash_next;
            break;
        }
        previous_index = current_index;
        current_index =
            service->request_maps[current_index].serving_handle_hash_next;
    }
    request_map->client_request_hash_next = SPARK_GLM52_SERVICE_NO_HASH_SLOT;
    request_map->serving_handle_hash_next = SPARK_GLM52_SERVICE_NO_HASH_SLOT;
}

static SparkGlm52ServiceClientSession *SparkGlm52ServiceFindClient(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id)
{
    uint32_t hash_slot;
    uint32_t client_index;

    if (client_id == 0u)
    {
        return 0;
    }
    hash_slot = SparkGlm52ServiceHash64(
        client_id,
        SPARK_GLM52_SERVICE_CLIENT_HASH_SLOTS);
    client_index = service->client_hash_heads[hash_slot];
    while (client_index != SPARK_GLM52_SERVICE_NO_HASH_SLOT &&
           client_index < service->client_session_capacity)
    {
        SparkGlm52ServiceClientSession *client_session;

        client_session = &service->client_sessions[client_index];
        if (client_session->state != SPARK_GLM52_SERVICE_CLIENT_STATE_FREE &&
            client_session->client_id == client_id)
        {
            return client_session;
        }
        client_index = client_session->client_hash_next;
    }
    return 0;
}

static SparkGlm52ServiceClientSession *SparkGlm52ServiceFindFreeClient(
    SparkGlm52ServiceRuntime *service)
{
    uint32_t client_index;

    for (client_index = 0u;
         client_index < service->client_session_capacity;
         ++client_index)
    {
        if (service->client_sessions[client_index].state ==
            SPARK_GLM52_SERVICE_CLIENT_STATE_FREE)
        {
            return &service->client_sessions[client_index];
        }
    }
    return 0;
}

static SparkGlm52ServiceRequestMap *SparkGlm52ServiceFindFreeRequestMap(
    SparkGlm52ServiceRuntime *service)
{
    uint32_t request_index;

    for (request_index = 0u;
         request_index < service->request_map_capacity;
         ++request_index)
    {
        if (service->request_maps[request_index].state ==
            SPARK_GLM52_SERVICE_REQUEST_STATE_FREE)
        {
            return &service->request_maps[request_index];
        }
    }
    return 0;
}

static SparkGlm52ServiceRequestMap *SparkGlm52ServiceFindRequestMapByClientRequest(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id,
    SparkGlm52ServiceRequestId client_request_id)
{
    uint32_t hash_slot;
    uint32_t request_index;

    if (client_id == 0u || client_request_id == 0u)
    {
        return 0;
    }
    hash_slot = SparkGlm52ServiceHashClientRequest(client_id, client_request_id);
    request_index = service->client_request_hash_heads[hash_slot];
    while (request_index != SPARK_GLM52_SERVICE_NO_HASH_SLOT &&
           request_index < service->request_map_capacity)
    {
        SparkGlm52ServiceRequestMap *request_map;

        request_map = &service->request_maps[request_index];
        if (request_map->state != SPARK_GLM52_SERVICE_REQUEST_STATE_FREE &&
            request_map->client_id == client_id &&
            request_map->client_request_id == client_request_id)
        {
            return request_map;
        }
        request_index = request_map->client_request_hash_next;
    }
    return 0;
}

static SparkGlm52ServiceRequestMap *SparkGlm52ServiceFindRequestMapByServingHandle(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServingRequestHandle serving_request_handle)
{
    uint32_t hash_slot;
    uint32_t request_index;

    if (serving_request_handle == 0u)
    {
        return 0;
    }
    hash_slot = SparkGlm52ServiceHash64(
        serving_request_handle,
        SPARK_GLM52_SERVICE_REQUEST_MAP_HASH_SLOTS);
    request_index = service->serving_handle_hash_heads[hash_slot];
    while (request_index != SPARK_GLM52_SERVICE_NO_HASH_SLOT &&
           request_index < service->request_map_capacity)
    {
        SparkGlm52ServiceRequestMap *request_map;

        request_map = &service->request_maps[request_index];
        if (request_map->state != SPARK_GLM52_SERVICE_REQUEST_STATE_FREE &&
            request_map->serving_request_handle == serving_request_handle)
        {
            return request_map;
        }
        request_index = request_map->serving_handle_hash_next;
    }
    return 0;
}

static uint32_t SparkGlm52ServiceEventRingFreeCount(
    const SparkGlm52ServiceRuntime *service)
{
    if (service == 0 || service->event_ring_capacity < service->event_count)
    {
        return 0u;
    }
    return service->event_ring_capacity - service->event_count;
}

static SparkStatus SparkGlm52ServicePushEvent(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52ServiceEvent *event)
{
    if (service->event_count == service->event_ring_capacity)
    {
        service->dropped_event_count += 1u;
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    service->event_ring[service->event_write_index] = *event;
    service->event_write_index += 1u;
    if (service->event_write_index == service->event_ring_capacity)
    {
        service->event_write_index = 0u;
    }
    service->event_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServicePushClientEvent(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id,
    uint32_t event_kind,
    SparkStatus status)
{
    SparkGlm52ServiceEvent event;

    SparkGlm52ServiceInitializeEvent(&event);
    event.kind = event_kind;
    event.status = status;
    event.client_id = client_id;
    return SparkGlm52ServicePushEvent(service, &event);
}

static uint32_t SparkGlm52ServiceMapServingEventKind(
    uint32_t serving_event_kind)
{
    switch (serving_event_kind)
    {
        case SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_ACCEPTED:
        {
            return SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_ACCEPTED;
        }
        case SPARK_GLM52_SERVING_EVENT_KIND_PREFILL_PROGRESS:
        {
            return SPARK_GLM52_SERVICE_EVENT_KIND_PREFILL_PROGRESS;
        }
        case SPARK_GLM52_SERVING_EVENT_KIND_TOKEN:
        {
            return SPARK_GLM52_SERVICE_EVENT_KIND_TOKEN;
        }
        case SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_COMPLETED:
        {
            return SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_COMPLETED;
        }
        case SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_CANCELLED:
        {
            return SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_CANCELLED;
        }
        case SPARK_GLM52_SERVING_EVENT_KIND_ERROR:
        {
            return SPARK_GLM52_SERVICE_EVENT_KIND_ERROR;
        }
        case SPARK_GLM52_SERVING_EVENT_KIND_BACKPRESSURE:
        {
            return SPARK_GLM52_SERVICE_EVENT_KIND_BACKPRESSURE;
        }
        default:
        {
            return SPARK_GLM52_SERVICE_EVENT_KIND_ERROR;
        }
    }
}

static SparkStatus SparkGlm52ServiceForwardServingEvent(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52ServingEvent *serving_event)
{
    SparkGlm52ServiceRequestMap *request_map;
    SparkGlm52ServiceEvent service_event;

    request_map = SparkGlm52ServiceFindRequestMapByServingHandle(
        service,
        serving_event->request_handle);
    if (request_map == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    SparkGlm52ServiceInitializeEvent(&service_event);
    service_event.kind = SparkGlm52ServiceMapServingEventKind(serving_event->kind);
    service_event.flags = serving_event->flags;
    service_event.status = serving_event->status;
    service_event.token_id = serving_event->token_id;
    service_event.token_index = serving_event->token_index;
    service_event.token_count = serving_event->token_count;
    service_event.prompt_token_offset = serving_event->prompt_token_offset;
    service_event.prompt_token_count = serving_event->prompt_token_count;
    service_event.dispatch_kind = serving_event->dispatch_kind;
    service_event.dispatch_flags = serving_event->dispatch_flags;
    service_event.client_id = request_map->client_id;
    service_event.client_request_id = request_map->client_request_id;
    service_event.serving_request_id = request_map->serving_request_id;
    service_event.sequence_id = request_map->sequence_id;
    service_event.serving_request_handle = request_map->serving_request_handle;

    if (service_event.kind == SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_COMPLETED)
    {
        SparkGlm52ServiceClientSession *client_session;

        request_map->state = SPARK_GLM52_SERVICE_REQUEST_STATE_COMPLETED;
        client_session = SparkGlm52ServiceFindClient(service, request_map->client_id);
        if (client_session != 0)
        {
            client_session->completed_request_count += 1u;
        }
    }
    else if (service_event.kind == SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_CANCELLED)
    {
        request_map->state = SPARK_GLM52_SERVICE_REQUEST_STATE_CANCELLED;
    }

    return SparkGlm52ServicePushEvent(service, &service_event);
}

static void SparkGlm52ServiceReleaseCompletedMappingsIfRequested(
    SparkGlm52ServiceRuntime *service)
{
    uint32_t request_index;

    if ((service->flags &
            SPARK_GLM52_SERVICE_CONFIGURATION_FLAG_AUTO_RELEASE_COMPLETED_MAPPINGS) == 0u)
    {
        return;
    }
    for (request_index = 0u;
         request_index < service->request_map_capacity;
         ++request_index)
    {
        SparkGlm52ServiceRequestMap *request_map;

        request_map = &service->request_maps[request_index];
        if (request_map->state == SPARK_GLM52_SERVICE_REQUEST_STATE_COMPLETED ||
            request_map->state == SPARK_GLM52_SERVICE_REQUEST_STATE_CANCELLED)
        {
            SparkGlm52ServiceRemoveRequestMapHash(service, request_map);
            SparkGlm52ServiceInitializeRequestMap(request_map);
        }
    }
}

static SparkStatus SparkGlm52ServiceDrainServingEvents(
    SparkGlm52ServiceRuntime *service)
{
    SparkStatus final_status;

    final_status = SPARK_STATUS_OK;
    for (;;)
    {
        SparkGlm52ServingEvent serving_event;
        SparkStatus status;

        if (SparkGlm52ServiceEventRingFreeCount(service) == 0u)
        {
            final_status = SPARK_STATUS_BUSY;
            break;
        }
        status = SparkGlm52ServingEnginePopEvent(
            service->serving_engine,
            &serving_event);
        if (status == SPARK_STATUS_NOT_FOUND)
        {
            break;
        }
        if (status != SPARK_STATUS_OK)
        {
            final_status = status;
            break;
        }
        status = SparkGlm52ServiceForwardServingEvent(service, &serving_event);
        if (status == SPARK_STATUS_NOT_FOUND)
        {
            continue;
        }
        if (status != SPARK_STATUS_OK)
        {
            final_status = status;
            break;
        }
        service->stats.forwarded_event_count += 1u;
    }
    SparkGlm52ServiceReleaseCompletedMappingsIfRequested(service);
    return final_status;
}

static void SparkGlm52ServiceRefreshStats(
    SparkGlm52ServiceRuntime *service)
{
    uint32_t client_index;
    uint32_t request_index;
    SparkGlm52ServiceStats *stats;

    stats = &service->stats;
    stats->abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
    stats->descriptor_bytes = SPARK_GLM52_SERVICE_STATS_DESCRIPTOR_BYTES;
    stats->connected_client_count = 0u;
    stats->live_request_count = 0u;
    stats->completed_request_mapping_count = 0u;
    stats->event_count = service->event_count;
    stats->event_capacity = service->event_ring_capacity;
    stats->dropped_event_count = service->dropped_event_count;

    for (client_index = 0u;
         client_index < service->client_session_capacity;
         ++client_index)
    {
        if (service->client_sessions[client_index].state !=
            SPARK_GLM52_SERVICE_CLIENT_STATE_FREE)
        {
            stats->connected_client_count += 1u;
        }
    }
    for (request_index = 0u;
         request_index < service->request_map_capacity;
         ++request_index)
    {
        if (service->request_maps[request_index].state ==
            SPARK_GLM52_SERVICE_REQUEST_STATE_LIVE)
        {
            stats->live_request_count += 1u;
        }
        else if (service->request_maps[request_index].state ==
            SPARK_GLM52_SERVICE_REQUEST_STATE_COMPLETED)
        {
            stats->completed_request_mapping_count += 1u;
        }
    }
    (void)SparkGlm52ServingEngineGetStats(
        service->serving_engine,
        &stats->serving_stats);
}

static SparkStatus SparkGlm52ServiceValidateConfiguration(
    const SparkGlm52ServiceConfiguration *configuration)
{
    uint32_t flags;

    if (configuration == 0 ||
        configuration->abi_version != SPARK_GLM52_SERVICE_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_GLM52_SERVICE_CONFIGURATION_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    flags = SparkGlm52ServiceNormalizeFlags(configuration->flags);
    if ((flags & ~SPARK_GLM52_SERVICE_CONFIGURATION_KNOWN_FLAGS) != 0u ||
        configuration->serving_engine == 0 ||
        configuration->serving_engine->abi_version !=
            SPARK_GLM52_SERVING_ENGINE_ABI_VERSION ||
        configuration->client_sessions == 0 ||
        configuration->client_session_capacity == 0u ||
        configuration->request_maps == 0 ||
        configuration->request_map_capacity == 0u ||
        configuration->event_ring == 0 ||
        configuration->event_ring_capacity == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

void SparkGlm52ServiceInitializeSubmitTextRequest(
    SparkGlm52ServiceSubmitTextRequest *request)
{
    if (request == 0)
    {
        return;
    }
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_SERVICE_SUBMIT_TEXT_DESCRIPTOR_BYTES;
}

void SparkGlm52ServiceInitializeSubmitTokenIdsRequest(
    SparkGlm52ServiceSubmitTokenIdsRequest *request)
{
    if (request == 0)
    {
        return;
    }
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_SERVICE_SUBMIT_TOKENS_DESCRIPTOR_BYTES;
}

void SparkGlm52ServiceInitializeFrameHeader(
    SparkGlm52ServiceFrameHeader *frame_header,
    uint32_t frame_kind)
{
    if (frame_header == 0)
    {
        return;
    }
    memset(frame_header, 0, sizeof(*frame_header));
    frame_header->magic = SPARK_GLM52_SERVICE_FRAME_MAGIC;
    frame_header->abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
    frame_header->descriptor_bytes =
        SPARK_GLM52_SERVICE_FRAME_HEADER_DESCRIPTOR_BYTES;
    frame_header->kind = frame_kind;
}

SparkStatus SparkGlm52ServiceValidateFrameHeader(
    const SparkGlm52ServiceFrameHeader *frame_header,
    uint32_t maximum_body_bytes)
{
    uint32_t body_limit;

    if (frame_header == 0 ||
        frame_header->magic != SPARK_GLM52_SERVICE_FRAME_MAGIC ||
        frame_header->abi_version != SPARK_GLM52_SERVICE_ABI_VERSION ||
        frame_header->descriptor_bytes !=
            SPARK_GLM52_SERVICE_FRAME_HEADER_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    body_limit = maximum_body_bytes != 0u
        ? maximum_body_bytes
        : SPARK_GLM52_SERVICE_MAX_FRAME_BODY_BYTES;
    if (frame_header->body_bytes > body_limit)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    switch (frame_header->kind)
    {
        case SPARK_GLM52_SERVICE_FRAME_KIND_SUBMIT_TEXT:
        case SPARK_GLM52_SERVICE_FRAME_KIND_SUBMIT_TOKEN_IDS:
        case SPARK_GLM52_SERVICE_FRAME_KIND_CANCEL_REQUEST:
        case SPARK_GLM52_SERVICE_FRAME_KIND_PING:
        case SPARK_GLM52_SERVICE_FRAME_KIND_EVENT:
        case SPARK_GLM52_SERVICE_FRAME_KIND_SUBMIT_ACK:
        case SPARK_GLM52_SERVICE_FRAME_KIND_ERROR:
        case SPARK_GLM52_SERVICE_FRAME_KIND_PONG:
        {
            return SPARK_STATUS_OK;
        }
        default:
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
}

SparkStatus SparkGlm52ServiceInitialize(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52ServiceConfiguration *configuration)
{
    uint32_t client_index;
    uint32_t request_index;
    uint32_t event_index;
    uint32_t hash_index;
    SparkStatus status;

    if (service == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52ServiceValidateConfiguration(configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(service, 0, sizeof(*service));
    service->abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
    service->descriptor_bytes = SPARK_GLM52_SERVICE_RUNTIME_DESCRIPTOR_BYTES;
    service->flags = SparkGlm52ServiceNormalizeFlags(configuration->flags);
    service->default_pump_dispatch_steps =
        SparkGlm52ServiceNormalizePumpDispatchSteps(
            configuration->default_pump_dispatch_steps);
    service->next_generated_request_id = SparkGlm52ServiceNormalizeRequestIdBase(
        configuration->request_id_base);
    service->next_generated_client_id = 1u;
    service->serving_engine = configuration->serving_engine;
    service->client_sessions = configuration->client_sessions;
    service->client_session_capacity = configuration->client_session_capacity;
    service->request_maps = configuration->request_maps;
    service->request_map_capacity = configuration->request_map_capacity;
    service->event_ring = configuration->event_ring;
    service->event_ring_capacity = configuration->event_ring_capacity;

    for (hash_index = 0u;
         hash_index < SPARK_GLM52_SERVICE_CLIENT_HASH_SLOTS;
         ++hash_index)
    {
        service->client_hash_heads[hash_index] =
            SPARK_GLM52_SERVICE_NO_HASH_SLOT;
    }
    for (hash_index = 0u;
         hash_index < SPARK_GLM52_SERVICE_REQUEST_MAP_HASH_SLOTS;
         ++hash_index)
    {
        service->client_request_hash_heads[hash_index] =
            SPARK_GLM52_SERVICE_NO_HASH_SLOT;
        service->serving_handle_hash_heads[hash_index] =
            SPARK_GLM52_SERVICE_NO_HASH_SLOT;
    }
    for (client_index = 0u;
         client_index < service->client_session_capacity;
         ++client_index)
    {
        SparkGlm52ServiceInitializeClientSession(
            &service->client_sessions[client_index]);
    }
    for (request_index = 0u;
         request_index < service->request_map_capacity;
         ++request_index)
    {
        SparkGlm52ServiceInitializeRequestMap(&service->request_maps[request_index]);
    }
    for (event_index = 0u;
         event_index < service->event_ring_capacity;
         ++event_index)
    {
        SparkGlm52ServiceInitializeEvent(&service->event_ring[event_index]);
    }
    SparkGlm52ServiceRefreshStats(service);
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ServiceRegisterClient(
    SparkGlm52ServiceRuntime *service,
    uint64_t user_cookie,
    SparkGlm52ServiceClientId *client_id_out)
{
    SparkGlm52ServiceClientSession *client_session;
    SparkStatus status;

    status = SparkGlm52ServiceValidateRuntime(service);
    if (status != SPARK_STATUS_OK || client_id_out == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    client_session = SparkGlm52ServiceFindFreeClient(service);
    if (client_session == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    client_session->state = SPARK_GLM52_SERVICE_CLIENT_STATE_CONNECTED;
    client_session->client_id = service->next_generated_client_id++;
    if (service->next_generated_client_id == 0u)
    {
        service->next_generated_client_id = 1u;
    }
    client_session->user_cookie = user_cookie;
    SparkGlm52ServiceInsertClientHash(service, client_session);
    *client_id_out = client_session->client_id;
    status = SparkGlm52ServicePushClientEvent(
        service,
        client_session->client_id,
        SPARK_GLM52_SERVICE_EVENT_KIND_CLIENT_CONNECTED,
        SPARK_STATUS_OK);
    SparkGlm52ServiceRefreshStats(service);
    return status;
}

SparkStatus SparkGlm52ServiceDisconnectClient(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id)
{
    SparkGlm52ServiceClientSession *client_session;
    uint32_t request_index;
    SparkStatus status;

    status = SparkGlm52ServiceValidateRuntime(service);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    client_session = SparkGlm52ServiceFindClient(service, client_id);
    if (client_session == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    for (request_index = 0u;
         request_index < service->request_map_capacity;
         ++request_index)
    {
        SparkGlm52ServiceRequestMap *request_map;

        request_map = &service->request_maps[request_index];
        if (request_map->state == SPARK_GLM52_SERVICE_REQUEST_STATE_LIVE &&
            request_map->client_id == client_id)
        {
            (void)SparkGlm52ServingEngineCancelRequest(
                service->serving_engine,
                request_map->serving_request_handle);
            request_map->state = SPARK_GLM52_SERVICE_REQUEST_STATE_CANCELLED;
        }
    }
    SparkGlm52ServiceRemoveClientHash(service, client_session);
    SparkGlm52ServiceInitializeClientSession(client_session);
    status = SparkGlm52ServicePushClientEvent(
        service,
        client_id,
        SPARK_GLM52_SERVICE_EVENT_KIND_CLIENT_DISCONNECTED,
        SPARK_STATUS_OK);
    SparkGlm52ServiceRefreshStats(service);
    return status;
}

static SparkStatus SparkGlm52ServiceReserveRequestMap(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id,
    SparkGlm52ServiceRequestId client_request_id,
    SparkGlm52ServiceRequestMap **request_map_out)
{
    SparkGlm52ServiceRequestMap *request_map;

    if (SparkGlm52ServiceFindClient(service, client_id) == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (client_request_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ServiceFindRequestMapByClientRequest(
            service,
            client_id,
            client_request_id) != 0)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    request_map = SparkGlm52ServiceFindFreeRequestMap(service);
    if (request_map == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *request_map_out = request_map;
    return SPARK_STATUS_OK;
}

static void SparkGlm52ServiceFillSubmitResult(
    const SparkGlm52ServiceRequestMap *request_map,
    const SparkGlm52ServingSubmitResult *serving_result,
    SparkGlm52ServiceSubmitResult *result)
{
    if (result == 0)
    {
        return;
    }
    memset(result, 0, sizeof(*result));
    result->abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
    result->descriptor_bytes = SPARK_GLM52_SERVICE_SUBMIT_RESULT_DESCRIPTOR_BYTES;
    result->prompt_token_count = serving_result->prompt_token_count;
    result->required_token_capacity = serving_result->required_token_capacity;
    result->thinking_token_budget = serving_result->thinking_token_budget;
    result->output_token_budget = serving_result->output_token_budget;
    result->client_id = request_map->client_id;
    result->client_request_id = request_map->client_request_id;
    result->serving_request_id = request_map->serving_request_id;
    result->sequence_id = request_map->sequence_id;
    result->serving_request_handle = request_map->serving_request_handle;
}

SparkStatus SparkGlm52ServiceSubmitTokenIds(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52ServiceSubmitTokenIdsRequest *request,
    SparkGlm52ServiceSubmitResult *result)
{
    SparkGlm52ServiceRequestMap *request_map;
    SparkGlm52ServingSubmitTokenIdsRequest serving_request;
    SparkGlm52ServingSubmitResult serving_result;
    SparkGlm52ServiceClientSession *client_session;
    SparkStatus status;

    status = SparkGlm52ServiceValidateRuntime(service);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (request == 0 ||
        request->abi_version != SPARK_GLM52_SERVICE_ABI_VERSION ||
        request->descriptor_bytes != SPARK_GLM52_SERVICE_SUBMIT_TOKENS_DESCRIPTOR_BYTES ||
        (request->flags & ~SPARK_GLM52_SERVICE_FRAME_KNOWN_SUBMIT_FLAGS) != 0u ||
        request->token_count == 0u ||
        request->token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52ServiceReserveRequestMap(
        service,
        request->client_id,
        request->client_request_id,
        &request_map);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    SparkGlm52ServingInitializeSubmitTokenIdsRequest(&serving_request);
    serving_request.flags = request->flags;
    serving_request.priority = request->priority;
    serving_request.token_count = request->token_count;
    serving_request.thinking_token_budget = request->thinking_token_budget;
    serving_request.output_token_budget = request->output_token_budget;
    serving_request.max_prefill_tokens_per_step =
        request->max_prefill_tokens_per_step;
    serving_request.request_id = service->next_generated_request_id++;
    serving_request.sequence_id = request->sequence_id;
    serving_request.token_ids = request->token_ids;
    status = SparkGlm52ServingEngineSubmitTokenIds(
        service->serving_engine,
        &serving_request,
        &serving_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    request_map->state = SPARK_GLM52_SERVICE_REQUEST_STATE_LIVE;
    request_map->client_id = request->client_id;
    request_map->client_request_id = request->client_request_id;
    request_map->serving_request_id = serving_result.request_id;
    request_map->sequence_id = serving_result.sequence_id;
    request_map->serving_request_handle = serving_result.request_handle;
    SparkGlm52ServiceInsertRequestMapHash(service, request_map);
    client_session = SparkGlm52ServiceFindClient(service, request->client_id);
    if (client_session != 0)
    {
        client_session->accepted_request_count += 1u;
    }
    service->stats.submitted_request_count += 1u;
    service->stats.accepted_request_count += 1u;
    SparkGlm52ServiceFillSubmitResult(request_map, &serving_result, result);
    SparkGlm52ServiceRefreshStats(service);
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ServiceSubmitText(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52ServiceSubmitTextRequest *request,
    SparkGlm52ServiceSubmitResult *result)
{
    SparkGlm52ServiceRequestMap *request_map;
    SparkGlm52ServingSubmitTextRequest serving_request;
    SparkGlm52ServingSubmitResult serving_result;
    SparkGlm52ServiceClientSession *client_session;
    SparkStatus status;

    status = SparkGlm52ServiceValidateRuntime(service);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (request == 0 ||
        request->abi_version != SPARK_GLM52_SERVICE_ABI_VERSION ||
        request->descriptor_bytes != SPARK_GLM52_SERVICE_SUBMIT_TEXT_DESCRIPTOR_BYTES ||
        (request->flags & ~SPARK_GLM52_SERVICE_FRAME_KNOWN_SUBMIT_FLAGS) != 0u ||
        request->text == 0 ||
        request->text_bytes == 0u ||
        request->text_bytes > SPARK_GLM52_SERVICE_MAX_TEXT_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52ServiceReserveRequestMap(
        service,
        request->client_id,
        request->client_request_id,
        &request_map);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    SparkGlm52ServingInitializeSubmitTextRequest(&serving_request);
    serving_request.flags = request->flags;
    serving_request.priority = request->priority;
    serving_request.thinking_token_budget = request->thinking_token_budget;
    serving_request.output_token_budget = request->output_token_budget;
    serving_request.max_prefill_tokens_per_step =
        request->max_prefill_tokens_per_step;
    serving_request.tokenizer_encode_flags = request->tokenizer_encode_flags;
    serving_request.request_id = service->next_generated_request_id++;
    serving_request.sequence_id = request->sequence_id;
    serving_request.text = request->text;
    serving_request.text_bytes = request->text_bytes;
    status = SparkGlm52ServingEngineSubmitText(
        service->serving_engine,
        &serving_request,
        &serving_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    request_map->state = SPARK_GLM52_SERVICE_REQUEST_STATE_LIVE;
    request_map->client_id = request->client_id;
    request_map->client_request_id = request->client_request_id;
    request_map->serving_request_id = serving_result.request_id;
    request_map->sequence_id = serving_result.sequence_id;
    request_map->serving_request_handle = serving_result.request_handle;
    SparkGlm52ServiceInsertRequestMapHash(service, request_map);
    client_session = SparkGlm52ServiceFindClient(service, request->client_id);
    if (client_session != 0)
    {
        client_session->accepted_request_count += 1u;
    }
    service->stats.submitted_request_count += 1u;
    service->stats.accepted_request_count += 1u;
    SparkGlm52ServiceFillSubmitResult(request_map, &serving_result, result);
    SparkGlm52ServiceRefreshStats(service);
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ServiceHandleSubmitTokenIdsFrame(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id,
    const SparkGlm52ServiceFrameHeader *frame_header,
    const void *body,
    uint32_t body_bytes,
    SparkGlm52ServiceSubmitResult *result)
{
    const SparkGlm52ServiceSubmitTokenIdsFrameBody *frame_body;
    SparkGlm52ServiceSubmitTokenIdsRequest request;
    const uint32_t *token_ids;
    uint32_t required_body_bytes;
    SparkStatus status;

    status = SparkGlm52ServiceValidateFrameHeader(
        frame_header,
        SPARK_GLM52_SERVICE_MAX_FRAME_BODY_BYTES);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (frame_header->kind != SPARK_GLM52_SERVICE_FRAME_KIND_SUBMIT_TOKEN_IDS ||
        body == 0 ||
        body_bytes < SPARK_GLM52_SERVICE_FRAME_SUBMIT_TOKENS_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    frame_body = (const SparkGlm52ServiceSubmitTokenIdsFrameBody *)body;
    if (frame_body->abi_version != SPARK_GLM52_SERVICE_ABI_VERSION ||
        frame_body->descriptor_bytes !=
            SPARK_GLM52_SERVICE_FRAME_SUBMIT_TOKENS_DESCRIPTOR_BYTES ||
        frame_body->token_count == 0u ||
        frame_body->token_count > SPARK_GLM52_SERVICE_MAX_TOKEN_FRAME_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (frame_body->token_count >
        (UINT32_MAX - SPARK_GLM52_SERVICE_FRAME_SUBMIT_TOKENS_DESCRIPTOR_BYTES) /
            (uint32_t)sizeof(uint32_t))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    required_body_bytes = SPARK_GLM52_SERVICE_FRAME_SUBMIT_TOKENS_DESCRIPTOR_BYTES +
        frame_body->token_count * (uint32_t)sizeof(uint32_t);
    if (body_bytes != required_body_bytes || frame_header->body_bytes != body_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    token_ids = (const uint32_t *)((const unsigned char *)body +
        SPARK_GLM52_SERVICE_FRAME_SUBMIT_TOKENS_DESCRIPTOR_BYTES);

    SparkGlm52ServiceInitializeSubmitTokenIdsRequest(&request);
    request.flags = frame_header->flags;
    request.priority = frame_body->priority;
    request.thinking_token_budget = frame_body->thinking_token_budget;
    request.output_token_budget = frame_body->output_token_budget;
    request.max_prefill_tokens_per_step = frame_body->max_prefill_tokens_per_step;
    request.token_count = frame_body->token_count;
    request.client_id = client_id;
    request.client_request_id = frame_header->client_request_id;
    request.sequence_id = frame_body->sequence_id;
    request.token_ids = token_ids;
    return SparkGlm52ServiceSubmitTokenIds(service, &request, result);
}

SparkStatus SparkGlm52ServiceHandleSubmitTextFrame(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id,
    const SparkGlm52ServiceFrameHeader *frame_header,
    const void *body,
    uint32_t body_bytes,
    SparkGlm52ServiceSubmitResult *result)
{
    const SparkGlm52ServiceSubmitTextFrameBody *frame_body;
    SparkGlm52ServiceSubmitTextRequest request;
    const char *text;
    uint32_t required_body_bytes;
    SparkStatus status;

    status = SparkGlm52ServiceValidateFrameHeader(
        frame_header,
        SPARK_GLM52_SERVICE_MAX_FRAME_BODY_BYTES);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (frame_header->kind != SPARK_GLM52_SERVICE_FRAME_KIND_SUBMIT_TEXT ||
        body == 0 ||
        body_bytes < SPARK_GLM52_SERVICE_FRAME_SUBMIT_TEXT_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    frame_body = (const SparkGlm52ServiceSubmitTextFrameBody *)body;
    if (frame_body->abi_version != SPARK_GLM52_SERVICE_ABI_VERSION ||
        frame_body->descriptor_bytes !=
            SPARK_GLM52_SERVICE_FRAME_SUBMIT_TEXT_DESCRIPTOR_BYTES ||
        frame_body->text_bytes == 0u ||
        frame_body->text_bytes > SPARK_GLM52_SERVICE_MAX_TEXT_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (frame_body->text_bytes >
        UINT32_MAX - SPARK_GLM52_SERVICE_FRAME_SUBMIT_TEXT_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    required_body_bytes = SPARK_GLM52_SERVICE_FRAME_SUBMIT_TEXT_DESCRIPTOR_BYTES +
        frame_body->text_bytes;
    if (body_bytes != required_body_bytes || frame_header->body_bytes != body_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    text = (const char *)((const unsigned char *)body +
        SPARK_GLM52_SERVICE_FRAME_SUBMIT_TEXT_DESCRIPTOR_BYTES);

    SparkGlm52ServiceInitializeSubmitTextRequest(&request);
    request.flags = frame_header->flags;
    request.priority = frame_body->priority;
    request.thinking_token_budget = frame_body->thinking_token_budget;
    request.output_token_budget = frame_body->output_token_budget;
    request.max_prefill_tokens_per_step = frame_body->max_prefill_tokens_per_step;
    request.tokenizer_encode_flags = frame_body->tokenizer_encode_flags;
    request.client_id = client_id;
    request.client_request_id = frame_header->client_request_id;
    request.sequence_id = frame_body->sequence_id;
    request.text = text;
    request.text_bytes = frame_body->text_bytes;
    return SparkGlm52ServiceSubmitText(service, &request, result);
}


SparkStatus SparkGlm52ServiceHandleCancelFrame(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id,
    const SparkGlm52ServiceFrameHeader *frame_header,
    const void *body,
    uint32_t body_bytes)
{
    const SparkGlm52ServiceCancelFrameBody *frame_body;
    SparkGlm52ServiceRequestId client_request_id;
    SparkStatus status;

    status = SparkGlm52ServiceValidateFrameHeader(
        frame_header,
        SPARK_GLM52_SERVICE_MAX_FRAME_BODY_BYTES);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (frame_header->kind != SPARK_GLM52_SERVICE_FRAME_KIND_CANCEL_REQUEST ||
        body == 0 ||
        body_bytes != SPARK_GLM52_SERVICE_FRAME_CANCEL_DESCRIPTOR_BYTES ||
        frame_header->body_bytes != body_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    frame_body = (const SparkGlm52ServiceCancelFrameBody *)body;
    if (frame_body->abi_version != SPARK_GLM52_SERVICE_ABI_VERSION ||
        frame_body->descriptor_bytes !=
            SPARK_GLM52_SERVICE_FRAME_CANCEL_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    client_request_id = frame_body->client_request_id != 0u
        ? frame_body->client_request_id
        : frame_header->client_request_id;
    if (client_request_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52ServiceCancelRequest(service, client_id, client_request_id);
}

SparkStatus SparkGlm52ServiceBuildEventFrame(
    const SparkGlm52ServiceEvent *event,
    SparkGlm52ServiceFrameHeader *frame_header,
    SparkGlm52ServiceEvent *frame_body)
{
    if (event == 0 || frame_header == 0 || frame_body == 0 ||
        event->abi_version != SPARK_GLM52_SERVICE_ABI_VERSION ||
        event->descriptor_bytes != SPARK_GLM52_SERVICE_EVENT_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkGlm52ServiceInitializeFrameHeader(
        frame_header,
        SPARK_GLM52_SERVICE_FRAME_KIND_EVENT);
    frame_header->body_bytes = SPARK_GLM52_SERVICE_EVENT_DESCRIPTOR_BYTES;
    frame_header->client_id = event->client_id;
    frame_header->client_request_id = event->client_request_id;
    *frame_body = *event;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ServiceCancelRequest(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id,
    SparkGlm52ServiceRequestId client_request_id)
{
    SparkGlm52ServiceRequestMap *request_map;
    SparkStatus status;

    status = SparkGlm52ServiceValidateRuntime(service);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    request_map = SparkGlm52ServiceFindRequestMapByClientRequest(
        service,
        client_id,
        client_request_id);
    if (request_map == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = SparkGlm52ServingEngineCancelRequest(
        service->serving_engine,
        request_map->serving_request_handle);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    request_map->state = SPARK_GLM52_SERVICE_REQUEST_STATE_CANCELLED;
    SparkGlm52ServiceRefreshStats(service);
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ServicePump(
    SparkGlm52ServiceRuntime *service,
    uint32_t max_dispatch_steps,
    SparkGlm52ServiceStats *stats_out)
{
    uint32_t dispatch_steps;
    SparkStatus status;
    SparkStatus drain_status;
    SparkGlm52ServingStats serving_stats;

    status = SparkGlm52ServiceValidateRuntime(service);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if ((service->flags &
            SPARK_GLM52_SERVICE_CONFIGURATION_FLAG_DRAIN_ENGINE_EVENTS_BEFORE_PUMP) != 0u)
    {
        drain_status = SparkGlm52ServiceDrainServingEvents(service);
        if (drain_status != SPARK_STATUS_OK && drain_status != SPARK_STATUS_BUSY)
        {
            service->stats.last_status = drain_status;
            SparkGlm52ServiceRefreshStats(service);
            if (stats_out != 0)
            {
                *stats_out = service->stats;
            }
            return drain_status;
        }
    }

    dispatch_steps = max_dispatch_steps != 0u
        ? max_dispatch_steps
        : service->default_pump_dispatch_steps;
    status = SparkGlm52ServingEnginePump(
        service->serving_engine,
        0u,
        dispatch_steps,
        &serving_stats);
    service->stats.engine_pump_count += 1u;
    drain_status = SparkGlm52ServiceDrainServingEvents(service);
    if (drain_status != SPARK_STATUS_OK && drain_status != SPARK_STATUS_BUSY)
    {
        status = drain_status;
    }
    if (status == SPARK_STATUS_NOT_FOUND || status == SPARK_STATUS_BUSY ||
        status == SPARK_STATUS_PENDING)
    {
        status = SPARK_STATUS_OK;
    }
    service->stats.last_status = status;
    SparkGlm52ServiceRefreshStats(service);
    if (stats_out != 0)
    {
        *stats_out = service->stats;
    }
    return status;
}

SparkStatus SparkGlm52ServicePopEvent(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceEvent *event_out)
{
    SparkStatus status;

    status = SparkGlm52ServiceValidateRuntime(service);
    if (status != SPARK_STATUS_OK || event_out == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    if (service->event_count == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    *event_out = service->event_ring[service->event_read_index];
    SparkGlm52ServiceInitializeEvent(&service->event_ring[service->event_read_index]);
    service->event_read_index += 1u;
    if (service->event_read_index == service->event_ring_capacity)
    {
        service->event_read_index = 0u;
    }
    service->event_count -= 1u;
    SparkGlm52ServiceRefreshStats(service);
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ServiceGetStats(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceStats *stats_out)
{
    SparkStatus status;

    status = SparkGlm52ServiceValidateRuntime(service);
    if (status != SPARK_STATUS_OK || stats_out == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    SparkGlm52ServiceRefreshStats(service);
    *stats_out = service->stats;
    return SPARK_STATUS_OK;
}

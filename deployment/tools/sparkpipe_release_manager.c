#define _POSIX_C_SOURCE 200809L

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_release.h"
#include "spark_filesystem.h"
#include "runtime/net.h"

#define SPARK_RELEASE_MANAGER_BUFFER_BYTES 65536u
#define SPARK_RELEASE_MANAGER_DEFAULT_PORT 55420u
#define SPARK_RELEASE_MANAGER_MAX_URL_BYTES 4096u
#define SPARK_RELEASE_MANAGER_HTTP_HEADER_BYTES 4096u
#define SPARK_RELEASE_MANAGER_COMMAND_LINE_BYTES 8192u
#define SPARK_RELEASE_MANAGER_AGENT_IDLE_SLEEP_MS 1000u

static const char SparkReleaseManagerHttpHeaderEnd[] = "\r\n\r\n";
static const char SparkReleaseManagerHttp10Ok[] = "HTTP/1.0 200";
static const char SparkReleaseManagerHttp11Ok[] = "HTTP/1.1 200";
static const char SparkReleaseManagerHttpScheme[] = "http://";

typedef struct SparkReleaseManagerUrl
{
    char host[512];
    char path[2048];
    uint32_t port;
} SparkReleaseManagerUrl;

typedef struct SparkReleaseManagerAgentConfig
{
    const char *release_directory;
    const char *release_url;
    const char *staging_directory;
    const char *install_directory;
    const char *state_directory;
    const char *role_name;
    const char *host;
    uint32_t rank;
    uint32_t rank_is_set;
    uint32_t once;
    uint32_t poll_interval_ms;
    uint32_t have_active_manifest;
    char active_manifest_sha256[SPARK_SHA256_HEX_BYTES];
} SparkReleaseManagerAgentConfig;

static void SparkReleaseManagerPrintUsage(void)
{
    fputs(
        "usage:\n"
        "  sparkpipe_release_manager example --output PATH\n"
        "  sparkpipe_release_manager validate --manifest PATH\n"
        "  sparkpipe_release_manager plan --manifest PATH --host HOST --rank N [--role NAME]\n"
        "  sparkpipe_release_manager sync --manifest PATH --release-dir DIR [--install-dir DIR]\n"
        "  sparkpipe_release_manager agent (--release-dir DIR | --release-url http://HOST:PORT[/sparkpipe.json]) --state-dir DIR --host HOST --rank N --role NAME [--install-dir DIR] [--once]\n"
        "  sparkpipe_release_manager serve --release-dir DIR [--bind ADDRESS] [--port PORT]\n",
        stderr);
}

static uint64_t SparkReleaseManagerMonotonicMilliseconds(void)
{
    struct timespec timestamp;

    if (clock_gettime(CLOCK_MONOTONIC,&timestamp) != 0)
    {
        return 0u;
    }
    return ((uint64_t)timestamp.tv_sec * 1000ull) + ((uint64_t)timestamp.tv_nsec / 1000000ull);
}

static void SparkReleaseManagerSleepMilliseconds(uint32_t milliseconds)
{
    struct timespec request;

    request.tv_sec = (time_t)(milliseconds / 1000u);
    request.tv_nsec = (long)(milliseconds % 1000u) * 1000000l;
    while (nanosleep(&request,&request) != 0 && errno == EINTR)
    {
    }
}

static SparkStatus SparkReleaseManagerParseUrl(
    const char *url,
    SparkReleaseManagerUrl *parsed_url)
{
    const char *cursor;
    const char *host_start;
    const char *host_end;
    const char *path_start;
    const char *colon;
    uint32_t host_bytes;
    uint32_t path_bytes;

    if (url == 0 || parsed_url == 0 ||
        strncmp(
            url,
            SparkReleaseManagerHttpScheme,
            sizeof(SparkReleaseManagerHttpScheme) - 1u) != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(parsed_url,0,sizeof(*parsed_url));
    parsed_url->port = SPARK_RELEASE_MANAGER_DEFAULT_PORT;
    cursor = url + sizeof(SparkReleaseManagerHttpScheme) - 1u;
    host_start = cursor;
    path_start = strchr(cursor,'/');
    if (path_start == 0)
    {
        host_end = url + strlen(url);
        SparkCopyString(parsed_url->path,sizeof(parsed_url->path),"/");
    }
    else
    {
        host_end = path_start;
        path_bytes = (uint32_t)strlen(path_start);
        if (path_bytes >= sizeof(parsed_url->path))
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        memcpy(parsed_url->path,path_start,path_bytes + 1u);
    }
    colon = memchr(host_start,':',(size_t)(host_end - host_start));
    if (colon != 0)
    {
        uint32_t port;
        char port_text[32];
        uint32_t port_text_bytes;

        host_bytes = (uint32_t)(colon - host_start);
        port_text_bytes = (uint32_t)(host_end - colon - 1);
        if (port_text_bytes == 0u || port_text_bytes >= sizeof(port_text))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        memcpy(port_text,colon + 1u,port_text_bytes);
        port_text[port_text_bytes] = '\0';
        if (SparkNetParseU32(port_text,&port) != 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        parsed_url->port = port;
    }
    else
    {
        host_bytes = (uint32_t)(host_end - host_start);
    }
    if (host_bytes == 0u || host_bytes >= sizeof(parsed_url->host))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    memcpy(parsed_url->host,host_start,host_bytes);
    parsed_url->host[host_bytes] = '\0';
    return SPARK_STATUS_OK;
}

static uint32_t SparkReleaseManagerUrlPathIsManifest(const char *path)
{
    const char *manifest_name;
    size_t path_length;
    size_t manifest_bytes;

    if (path == 0)
    {
        return 0u;
    }
    manifest_name = "sparkpipe.json";
    path_length = strlen(path);
    manifest_bytes = strlen(manifest_name);
    if (path_length < manifest_bytes)
    {
        return 0u;
    }
    if (strcmp(path + path_length - manifest_bytes,manifest_name) != 0)
    {
        return 0u;
    }
    return path_length == manifest_bytes || path[path_length - manifest_bytes - 1u] == '/' ? 1u : 0u;
}

static SparkStatus SparkReleaseManagerBuildUrlPath(
    const SparkReleaseManagerUrl *base_url,
    const char *relative_path,
    char *path,
    uint32_t path_bytes)
{
    const char *base_path;

    if (base_url == 0 || relative_path == 0 || path == 0 || path_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    base_path = base_url->path[0] == '\0' ? "/" : base_url->path;
    if (SparkReleaseManagerUrlPathIsManifest(base_path) != 0u)
    {
        const char *slash;

        if (strcmp(relative_path,"sparkpipe.json") == 0)
        {
            if (snprintf(path,path_bytes,"%s",base_path) >= (int)path_bytes)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            return SPARK_STATUS_OK;
        }
        slash = strrchr(base_path,'/');
        if (slash == 0 || slash == base_path)
        {
            if (snprintf(path,path_bytes,"/%s",relative_path) >= (int)path_bytes)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            return SPARK_STATUS_OK;
        }
        if (snprintf(
                path,
                path_bytes,
                "%.*s/%s",
                (int)(slash - base_path),
                base_path,
                relative_path) >= (int)path_bytes)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
    }
    else if (base_path[strlen(base_path) - 1u] == '/')
    {
        if (snprintf(path,path_bytes,"%s%s",base_path,relative_path) >= (int)path_bytes)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
    }
    else
    {
        if (snprintf(path,path_bytes,"%s/%s",base_path,relative_path) >= (int)path_bytes)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
    }
    return SPARK_STATUS_OK;
}

static int SparkReleaseManagerConnectTcp(const char *host,uint32_t port)
{
    struct addrinfo hints;
    struct addrinfo *results;
    struct addrinfo *candidate;
    char service[32];
    int socket_fd;

    memset(&hints,0,sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    snprintf(service,sizeof(service),"%u",port);
    if (getaddrinfo(host,service,&hints,&results) != 0)
    {
        return -1;
    }
    socket_fd = -1;
    for (candidate = results; candidate != 0; candidate = candidate->ai_next)
    {
        socket_fd = socket(candidate->ai_family,candidate->ai_socktype,candidate->ai_protocol);
        if (socket_fd < 0)
        {
            continue;
        }
        if (connect(socket_fd,candidate->ai_addr,candidate->ai_addrlen) == 0)
        {
            break;
        }
        close(socket_fd);
        socket_fd = -1;
    }
    freeaddrinfo(results);
    return socket_fd;
}

static SparkStatus SparkReleaseManagerWriteAll(int socket_fd,const void *data,size_t data_bytes)
{
    const uint8_t *cursor;
    size_t remaining;

    cursor = (const uint8_t *)data;
    remaining = data_bytes;
    while (remaining != 0u)
    {
        ssize_t bytes_written;

        bytes_written = send(socket_fd,cursor,remaining,0);
        if (bytes_written <= 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return SPARK_STATUS_IO_ERROR;
        }
        cursor += (size_t)bytes_written;
        remaining -= (size_t)bytes_written;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReleaseManagerReadHttpResponseToFile(
    int socket_fd,
    const char *output_path)
{
    FILE *output;
    char header[SPARK_RELEASE_MANAGER_HTTP_HEADER_BYTES];
    uint32_t header_bytes;
    uint32_t header_done;
    uint32_t status_ok;
    uint8_t buffer[SPARK_RELEASE_MANAGER_BUFFER_BYTES];

    output = fopen(output_path,"wb");
    if (output == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    header_bytes = 0u;
    header_done = 0u;
    status_ok = 0u;
    while (1)
    {
        ssize_t bytes_read;
        uint32_t offset;

        bytes_read = recv(socket_fd,buffer,sizeof(buffer),0);
        if (bytes_read < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            fclose(output);
            unlink(output_path);
            return SPARK_STATUS_IO_ERROR;
        }
        if (bytes_read == 0)
        {
            break;
        }
        offset = 0u;
        if (header_done == 0u)
        {
            while (offset < (uint32_t)bytes_read && header_done == 0u)
            {
                if (header_bytes + 1u >= sizeof(header))
                {
                    fclose(output);
                    unlink(output_path);
                    return SPARK_STATUS_CAPACITY_EXCEEDED;
                }
                header[header_bytes++] = (char)buffer[offset++];
                header[header_bytes] = '\0';
                if (header_bytes >= sizeof(SparkReleaseManagerHttpHeaderEnd) - 1u &&
                    memcmp(
                        header + header_bytes -
                            (sizeof(SparkReleaseManagerHttpHeaderEnd) - 1u),
                        SparkReleaseManagerHttpHeaderEnd,
                        sizeof(SparkReleaseManagerHttpHeaderEnd) - 1u) == 0)
                {
                    header_done = 1u;
                    status_ok =
                        strncmp(
                            header,
                            SparkReleaseManagerHttp10Ok,
                            sizeof(SparkReleaseManagerHttp10Ok) - 1u) == 0 ||
                        strncmp(
                            header,
                            SparkReleaseManagerHttp11Ok,
                            sizeof(SparkReleaseManagerHttp11Ok) - 1u) == 0;
                    if (status_ok == 0u)
                    {
                        fclose(output);
                        unlink(output_path);
                        return SPARK_STATUS_NOT_FOUND;
                    }
                }
            }
        }
        if (header_done != 0u && offset < (uint32_t)bytes_read)
        {
            if (fwrite(buffer + offset,1u,(uint32_t)bytes_read - offset,output) != (uint32_t)bytes_read - offset)
            {
                fclose(output);
                unlink(output_path);
                return SPARK_STATUS_IO_ERROR;
            }
        }
    }
    if (header_done == 0u || status_ok == 0u)
    {
        fclose(output);
        unlink(output_path);
        return SPARK_STATUS_IO_ERROR;
    }
    if (fflush(output) != 0 || fclose(output) != 0)
    {
        unlink(output_path);
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReleaseManagerHttpGetToFile(
    const SparkReleaseManagerUrl *base_url,
    const char *relative_path,
    const char *output_path)
{
    char path[SPARK_RELEASE_MANAGER_MAX_URL_BYTES];
    char request[SPARK_RELEASE_MANAGER_MAX_URL_BYTES + 512u];
    int socket_fd;
    SparkStatus status;

    status = SparkReleaseManagerBuildUrlPath(base_url,relative_path,path,sizeof(path));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    socket_fd = SparkReleaseManagerConnectTcp(base_url->host,base_url->port);
    if (socket_fd < 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (snprintf(request,sizeof(request),"GET %s HTTP/1.0\r\nHost: %s\r\nConnection: close\r\n\r\n",path,base_url->host) >= (int)sizeof(request))
    {
        close(socket_fd);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkReleaseManagerWriteAll(socket_fd,request,strlen(request));
    if (status == SPARK_STATUS_OK)
    {
        status = SparkReleaseManagerReadHttpResponseToFile(socket_fd,output_path);
    }
    close(socket_fd);
    return status;
}

static SparkStatus SparkReleaseManagerDownloadReleaseToStaging(
    const char *release_url,
    const SparkReleaseManifest *manifest,
    const char *staging_directory)
{
    SparkReleaseManagerUrl base_url;
    char destination_path[SPARK_RELEASE_MAX_PATH_BYTES];
    uint32_t file_index;
    SparkStatus status;

    if (release_url == 0 || manifest == 0 || staging_directory == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkReleaseManagerParseUrl(release_url,&base_url);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkCreateDirectories(staging_directory);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (file_index = 0u; file_index < manifest->file_count; ++file_index)
    {
        if (snprintf(destination_path,sizeof(destination_path),"%s/%s",staging_directory,manifest->files[file_index].path) >= (int)sizeof(destination_path))
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        status = SparkCreateDirectories(staging_directory);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        {
            char parent_path[SPARK_RELEASE_MAX_PATH_BYTES];
            char *slash;

            strcpy(parent_path,destination_path);
            slash = strrchr(parent_path,'/');
            if (slash != 0)
            {
                *slash = '\0';
                status = SparkCreateDirectories(parent_path);
                if (status != SPARK_STATUS_OK)
                {
                    return status;
                }
            }
        }
        status = SparkReleaseManagerHttpGetToFile(&base_url,manifest->files[file_index].path,destination_path);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReleaseManagerLoadManifestFromSource(
    const SparkReleaseManagerAgentConfig *configuration,
    const char *manifest_path,
    SparkReleaseManifest *manifest)
{
    SparkStatus status;

    if (configuration->release_url != 0)
    {
        SparkReleaseManagerUrl base_url;

        status = SparkReleaseManagerParseUrl(configuration->release_url,&base_url);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkCreateDirectories(configuration->staging_directory);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkReleaseManagerHttpGetToFile(&base_url,"sparkpipe.json",manifest_path);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkReleaseManifestLoadFile(manifest_path,manifest);
    }
    if (configuration->release_directory == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkReleaseManifestLoadFile(manifest_path,manifest);
}

static SparkStatus SparkReleaseManagerPrepareSourceDirectory(
    const SparkReleaseManagerAgentConfig *configuration,
    const SparkReleaseManifest *manifest,
    const char **source_directory_out)
{
    SparkStatus status;

    if (configuration->release_url != 0)
    {
        status = SparkReleaseManagerDownloadReleaseToStaging(configuration->release_url,manifest,configuration->staging_directory);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        *source_directory_out = configuration->staging_directory;
        return SPARK_STATUS_OK;
    }
    *source_directory_out = configuration->release_directory;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReleaseManagerReadPidFile(const char *pid_file,pid_t *pid_out)
{
    char *text;
    size_t text_bytes;
    long value;
    char *end;
    SparkStatus status;

    if (pid_file == 0 || pid_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *pid_out = 0;
    status = SparkReadEntireFile(pid_file,&text,&text_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    (void)text_bytes;
    errno = 0;
    end = 0;
    value = strtol(text,&end,10);
    free(text);
    if (errno != 0 || end == 0 || value <= 0)
    {
        return SPARK_STATUS_PARSE_ERROR;
    }
    *pid_out = (pid_t)value;
    return SPARK_STATUS_OK;
}

static uint32_t SparkReleaseManagerProcessIsAlive(pid_t pid)
{
    if (pid <= 0)
    {
        return 0u;
    }
    if (kill(pid,0) == 0)
    {
        return 1u;
    }
    return errno == EPERM ? 1u : 0u;
}

static SparkStatus SparkReleaseManagerStopRole(const SparkReleaseResolvedRole *resolved_role,uint32_t stop_grace_ms)
{
    pid_t pid;
    uint64_t deadline;

    if (resolved_role == 0 || resolved_role->pid_file[0] == '\0')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkReleaseManagerReadPidFile(resolved_role->pid_file,&pid) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_OK;
    }
    if (SparkReleaseManagerProcessIsAlive(pid) == 0u)
    {
        unlink(resolved_role->pid_file);
        return SPARK_STATUS_OK;
    }
    kill(pid,SIGTERM);
    deadline = SparkReleaseManagerMonotonicMilliseconds() + stop_grace_ms;
    while (SparkReleaseManagerMonotonicMilliseconds() < deadline)
    {
        if (SparkReleaseManagerProcessIsAlive(pid) == 0u)
        {
            unlink(resolved_role->pid_file);
            return SPARK_STATUS_OK;
        }
        SparkReleaseManagerSleepMilliseconds(50u);
    }
    kill(pid,SIGKILL);
    SparkReleaseManagerSleepMilliseconds(100u);
    unlink(resolved_role->pid_file);
    return SPARK_STATUS_OK;
}

static uint32_t SparkReleaseManagerIsSha256Hex(const char *text)
{
    uint32_t index;

    if (text == 0)
    {
        return 0u;
    }
    for (index = 0u; index < SPARK_SHA256_HEX_BYTES - 1u; ++index)
    {
        if (!((text[index] >= '0' && text[index] <= '9') ||
              (text[index] >= 'a' && text[index] <= 'f')))
        {
            return 0u;
        }
    }
    return text[SPARK_SHA256_HEX_BYTES - 1u] == '\0' ? 1u : 0u;
}

static uint32_t SparkReleaseManagerBuildRoleManifestStatePath(
    const SparkReleaseManagerAgentConfig *configuration,
    const SparkReleaseRole *role,
    char *path,
    uint32_t path_bytes)
{
    if (configuration == 0 || role == 0 || path == 0 || path_bytes == 0u ||
        configuration->state_directory == 0)
    {
        return 0u;
    }
    return snprintf(path,path_bytes,"%s/run/%s.manifest.sha256",
        configuration->state_directory,role->name) < (int)path_bytes ? 1u : 0u;
}

static void SparkReleaseManagerLoadRoleManifestState(
    SparkReleaseManagerAgentConfig *configuration,
    const SparkReleaseRole *role)
{
    char path[SPARK_RELEASE_MAX_PATH_BYTES];
    char *data;
    size_t data_bytes;

    if (SparkReleaseManagerBuildRoleManifestStatePath(
            configuration,role,path,sizeof(path)) == 0u ||
        SparkReadEntireFile(path,&data,&data_bytes) != SPARK_STATUS_OK)
    {
        return;
    }
    if (data_bytes == SPARK_SHA256_HEX_BYTES &&
        data[SPARK_SHA256_HEX_BYTES - 1u] == '\n')
    {
        data[SPARK_SHA256_HEX_BYTES - 1u] = '\0';
    }
    if (SparkReleaseManagerIsSha256Hex(data) != 0u)
    {
        SparkCopyString(configuration->active_manifest_sha256,
            sizeof(configuration->active_manifest_sha256),data);
        configuration->have_active_manifest = 1u;
    }
    free(data);
}

static SparkStatus SparkReleaseManagerRememberRoleManifest(
    SparkReleaseManagerAgentConfig *configuration,
    const SparkReleaseRole *role,
    const char *manifest_sha256)
{
    char path[SPARK_RELEASE_MAX_PATH_BYTES];
    char state_text[SPARK_SHA256_HEX_BYTES + 1u];
    SparkStatus status;

    SparkCopyString(configuration->active_manifest_sha256,
        sizeof(configuration->active_manifest_sha256),manifest_sha256);
    configuration->have_active_manifest = 1u;
    if (SparkReleaseManagerBuildRoleManifestStatePath(
            configuration,role,path,sizeof(path)) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    status = SparkReleaseEnsureParentDirectory(path);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    snprintf(state_text,sizeof(state_text),"%s\n",manifest_sha256);
    return SparkWriteEntireFileAtomically(path,state_text,strlen(state_text));
}

static SparkStatus SparkReleaseManagerOpenRoleDescriptors(
    const SparkReleaseResolvedRole *resolved_role,
    int *null_fd_out,
    int *log_fd_out)
{
    char log_path[SPARK_RELEASE_MAX_PATH_BYTES];
    int written_bytes;

    if (resolved_role == 0 || null_fd_out == 0 || log_fd_out == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    written_bytes = snprintf(
        log_path,
        sizeof(log_path),
        "%s.log",
        resolved_role->pid_file);
    if (written_bytes < 0 || (uint32_t)written_bytes >= sizeof(log_path))
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    *null_fd_out = open("/dev/null",O_RDONLY);
    if (*null_fd_out < 0)
        return SPARK_STATUS_IO_ERROR;
    *log_fd_out = open(log_path,O_WRONLY | O_CREAT | O_APPEND,0644);
    if (*log_fd_out < 0)
    {
        close(*null_fd_out);
        *null_fd_out = -1;
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReleaseManagerStartRole(const SparkReleaseResolvedRole *resolved_role)
{
    char *arguments[SPARK_RELEASE_MAX_ARGUMENTS + 2u];
    uint32_t argument_index;
    uint32_t environment_index;
    int log_fd;
    int null_fd;
    pid_t pid;
    SparkStatus status;

    if (resolved_role == 0 || resolved_role->command[0] == '\0')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkReleaseEnsureParentDirectory(resolved_role->pid_file);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    null_fd = -1;
    log_fd = -1;
    status = SparkReleaseManagerOpenRoleDescriptors(
        resolved_role,
        &null_fd,
        &log_fd);
    if (status != SPARK_STATUS_OK)
        return status;
    pid = fork();
    if (pid < 0)
    {
        close(null_fd);
        close(log_fd);
        return SPARK_STATUS_IO_ERROR;
    }
    if (pid == 0)
    {
        if (setsid() < 0 ||
            dup2(null_fd,STDIN_FILENO) < 0 ||
            dup2(log_fd,STDOUT_FILENO) < 0 ||
            dup2(log_fd,STDERR_FILENO) < 0)
        {
            _exit(126);
        }
        close(null_fd);
        close(log_fd);
        for (environment_index = 0u; environment_index < resolved_role->environment_count; ++environment_index)
        {
            char *equals;
            char environment_copy[SPARK_RELEASE_MAX_PATH_BYTES];

            SparkCopyString(environment_copy,sizeof(environment_copy),resolved_role->environment[environment_index]);
            equals = strchr(environment_copy,'=');
            if (equals != 0)
            {
                *equals = '\0';
                setenv(environment_copy,equals + 1,1);
            }
        }
        arguments[0] = (char *)resolved_role->command;
        for (argument_index = 0u; argument_index < resolved_role->argument_count; ++argument_index)
        {
            arguments[argument_index + 1u] = (char *)resolved_role->arguments[argument_index];
        }
        arguments[resolved_role->argument_count + 1u] = 0;
        execv(resolved_role->command,arguments);
        _exit(127);
    }
    close(null_fd);
    close(log_fd);
    {
        char pid_text[64];

        snprintf(pid_text,sizeof(pid_text),"%ld\n",(long)pid);
        status = SparkWriteEntireFileAtomically(resolved_role->pid_file,pid_text,strlen(pid_text));
        if (status != SPARK_STATUS_OK)
        {
            kill(pid,SIGTERM);
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReleaseManagerRunAgentOnce(SparkReleaseManagerAgentConfig *configuration)
{
    SparkReleaseManifest manifest;
    SparkReleaseNodeIdentity identity;
    SparkReleaseResolvedRole resolved_role;
    const SparkReleaseRole *role;
    const char *source_directory;
    SparkReleaseSyncResult sync_result;
    char manifest_path[SPARK_RELEASE_MAX_PATH_BYTES];
    char command_line[SPARK_RELEASE_MANAGER_COMMAND_LINE_BYTES];
    char resolved_install_directory[SPARK_RELEASE_MAX_PATH_BYTES];
    const char *install_directory;
    SparkStatus status;
    char manifest_sha256[SPARK_SHA256_HEX_BYTES];
    uint32_t role_alive;

    if (configuration->staging_directory == 0)
    {
        configuration->staging_directory = "/tmp/sparkpipe_release_staging";
    }
    if (configuration->release_url != 0)
    {
        snprintf(manifest_path,sizeof(manifest_path),"%s/sparkpipe.json",configuration->staging_directory);
    }
    else
    {
        snprintf(manifest_path,sizeof(manifest_path),"%s/sparkpipe.json",configuration->release_directory);
    }
    status = SparkReleaseManagerLoadManifestFromSource(configuration,manifest_path,&manifest);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkSha256File(manifest_path,manifest_sha256);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkReleaseNodeIdentityInitialize(&identity);
    SparkCopyString(identity.host,sizeof(identity.host),configuration->host == 0 ? "spark0" : configuration->host);
    identity.rank = configuration->rank;
    identity.rank_is_set = configuration->rank_is_set;
    identity.rank_count = manifest.rank_count;
    status = SparkReleaseManifestFindRoleForNode(&manifest,&identity,configuration->role_name,&role);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkReleaseResolveRole(&manifest,&identity,role,&resolved_role);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkReleaseManagerLoadRoleManifestState(configuration,role);
    role_alive = 0u;
    {
        pid_t existing_pid;

        if (SparkReleaseManagerReadPidFile(resolved_role.pid_file,&existing_pid) == SPARK_STATUS_OK &&
            SparkReleaseManagerProcessIsAlive(existing_pid) != 0u)
        {
            role_alive = 1u;
        }
    }
    if (configuration->have_active_manifest != 0u &&
        strcmp(configuration->active_manifest_sha256,manifest_sha256) == 0 &&
        role_alive != 0u)
    {
        return SPARK_STATUS_OK;
    }
    status = SparkReleaseManagerPrepareSourceDirectory(configuration,&manifest,&source_directory);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    install_directory = configuration->install_directory;
    if (install_directory == 0)
    {
        status = SparkReleaseResolveInstallRoot(
            &manifest,
            &identity,
            resolved_install_directory,
            sizeof(resolved_install_directory));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        install_directory = resolved_install_directory;
    }
    status = SparkReleaseSyncFilesFromDirectory(
        &manifest,
        source_directory,
        install_directory,
        &sync_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (role_alive != 0u &&
        (role->flags & (SPARK_RELEASE_ROLE_FLAG_RESTART_ON_UPDATE |
                        SPARK_RELEASE_ROLE_FLAG_REQUIRE_EXACT_GENERATION)) == 0u)
    {
        status = SparkReleaseManagerRememberRoleManifest(
            configuration,role,manifest_sha256);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        printf("sparkpipe_release_manager synced without restart role=%s generation=%llu files=%u changed=%u\n",
            role->name,(unsigned long long)manifest.generation,sync_result.verified_file_count,sync_result.changed_file_count);
        return SPARK_STATUS_OK;
    }
    if (role_alive != 0u)
    {
        (void)SparkReleaseManagerStopRole(&resolved_role,manifest.stop_grace_ms);
    }
    status = SparkReleaseFormatResolvedCommandLine(&resolved_role,command_line,sizeof(command_line));
    if (status == SPARK_STATUS_OK)
    {
        printf("sparkpipe_release_manager starting role=%s generation=%llu command=%s\n",
            role->name,(unsigned long long)manifest.generation,command_line);
    }
    status = SparkReleaseManagerStartRole(&resolved_role);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkReleaseManagerRememberRoleManifest(
            configuration,role,manifest_sha256);
    }
    if (status == SPARK_STATUS_OK)
    {
        printf("sparkpipe_release_manager synced files=%u changed=%u cuda_packs=%u pid_file=%s\n",
            sync_result.verified_file_count,sync_result.changed_file_count,sync_result.cuda_pack_file_count,resolved_role.pid_file);
    }
    return status;
}

static int SparkReleaseManagerRunAgent(int argc,char **argv)
{
    SparkReleaseManagerAgentConfig configuration;
    char default_staging_directory[SPARK_RELEASE_MAX_PATH_BYTES];
    int index;
    SparkStatus status;

    memset(&configuration,0,sizeof(configuration));
    memset(default_staging_directory,0,sizeof(default_staging_directory));
    configuration.poll_interval_ms = SPARK_RELEASE_MANAGER_AGENT_IDLE_SLEEP_MS;
    for (index = 2; index < argc; ++index)
    {
        if (strcmp(argv[index],"--release-dir") == 0 && index + 1 < argc)
        {
            configuration.release_directory = argv[++index];
        }
        else if (strcmp(argv[index],"--release-url") == 0 && index + 1 < argc)
        {
            configuration.release_url = argv[++index];
        }
        else if (strcmp(argv[index],"--staging-dir") == 0 && index + 1 < argc)
        {
            configuration.staging_directory = argv[++index];
        }
        else if (strcmp(argv[index],"--install-dir") == 0 && index + 1 < argc)
        {
            configuration.install_directory = argv[++index];
        }
        else if (strcmp(argv[index],"--state-dir") == 0 && index + 1 < argc)
        {
            configuration.state_directory = argv[++index];
        }
        else if (strcmp(argv[index],"--role") == 0 && index + 1 < argc)
        {
            configuration.role_name = argv[++index];
        }
        else if (strcmp(argv[index],"--host") == 0 && index + 1 < argc)
        {
            configuration.host = argv[++index];
        }
        else if (strcmp(argv[index],"--rank") == 0 && index + 1 < argc)
        {
            if (SparkNetParseU32(argv[++index],&configuration.rank) != 0)
            {
                return 2;
            }
            configuration.rank_is_set = 1u;
        }
        else if (strcmp(argv[index],"--poll-ms") == 0 && index + 1 < argc)
        {
            if (SparkNetParseU32(argv[++index],&configuration.poll_interval_ms) != 0)
            {
                return 2;
            }
        }
        else if (strcmp(argv[index],"--once") == 0)
        {
            configuration.once = 1u;
        }
        else
        {
            SparkReleaseManagerPrintUsage();
            return 2;
        }
    }
    if ((configuration.release_directory == 0 && configuration.release_url == 0) ||
        (configuration.release_directory != 0 && configuration.release_url != 0) ||
        configuration.host == 0 || configuration.role_name == 0)
    {
        SparkReleaseManagerPrintUsage();
        return 2;
    }
    if (configuration.staging_directory == 0)
    {
        if (configuration.state_directory != 0)
        {
            if (snprintf(default_staging_directory,sizeof(default_staging_directory),"%s/release_staging",configuration.state_directory) >= (int)sizeof(default_staging_directory))
            {
                return 2;
            }
            configuration.staging_directory = default_staging_directory;
        }
        else
        {
            configuration.staging_directory = "/tmp/sparkpipe_release_staging";
        }
    }
    do
    {
        status = SparkReleaseManagerRunAgentOnce(&configuration);
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr,"sparkpipe_release_manager agent status=%d\n",(int)status);
            if (configuration.once != 0u)
            {
                return 1;
            }
        }
        if (configuration.once != 0u)
        {
            return status == SPARK_STATUS_OK ? 0 : 1;
        }
        SparkReleaseManagerSleepMilliseconds(configuration.poll_interval_ms);
    }
    while (1);
}

static int SparkReleaseManagerServe(const char *release_directory,const char *bind_address,uint32_t port)
{
    int listen_fd;
    struct sockaddr_in address;
    int one;

    (void)bind_address;
    listen_fd = socket(AF_INET,SOCK_STREAM,0);
    if (listen_fd < 0)
    {
        return 1;
    }
    one = 1;
    setsockopt(listen_fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
    memset(&address,0,sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(listen_fd,(struct sockaddr *)&address,sizeof(address)) != 0 || listen(listen_fd,64) != 0)
    {
        close(listen_fd);
        return 1;
    }
    printf("sparkpipe_release_manager serving %s on port %u\n",release_directory,port);
    while (1)
    {
        int client_fd;
        char request[2048];
        ssize_t bytes_read;
        char method[16];
        char path[1024];
        char local_path[SPARK_RELEASE_MAX_PATH_BYTES];
        FILE *file;
        struct stat file_status;
        char header[256];
        uint8_t buffer[SPARK_RELEASE_MANAGER_BUFFER_BYTES];

        client_fd = accept(listen_fd,0,0);
        if (client_fd < 0)
        {
            continue;
        }
        bytes_read = recv(client_fd,request,sizeof(request) - 1u,0);
        if (bytes_read <= 0)
        {
            close(client_fd);
            continue;
        }
        request[bytes_read] = '\0';
        method[0] = '\0';
        path[0] = '\0';
        sscanf(request,"%15s %1023s",method,path);
        if (strcmp(method,"GET") != 0 || path[0] != '/' || strstr(path,"..") != 0)
        {
            SparkReleaseManagerWriteAll(client_fd,"HTTP/1.0 400 Bad Request\r\nConnection: close\r\n\r\n",47u);
            close(client_fd);
            continue;
        }
        if (snprintf(local_path,sizeof(local_path),"%s/%s",release_directory,path + 1u) >= (int)sizeof(local_path))
        {
            SparkReleaseManagerWriteAll(client_fd,"HTTP/1.0 414 URI Too Long\r\nConnection: close\r\n\r\n",51u);
            close(client_fd);
            continue;
        }
        file = fopen(local_path,"rb");
        if (file == 0 || stat(local_path,&file_status) != 0 || !S_ISREG(file_status.st_mode))
        {
            if (file != 0)
            {
                fclose(file);
            }
            SparkReleaseManagerWriteAll(client_fd,"HTTP/1.0 404 Not Found\r\nConnection: close\r\n\r\n",45u);
            close(client_fd);
            continue;
        }
        snprintf(header,sizeof(header),"HTTP/1.0 200 OK\r\nContent-Length: %llu\r\nConnection: close\r\n\r\n",(unsigned long long)file_status.st_size);
        SparkReleaseManagerWriteAll(client_fd,header,strlen(header));
        while ((bytes_read = fread(buffer,1u,sizeof(buffer),file)) > 0)
        {
            if (SparkReleaseManagerWriteAll(client_fd,buffer,(size_t)bytes_read) != SPARK_STATUS_OK)
            {
                break;
            }
        }
        fclose(file);
        close(client_fd);
    }
}

int main(int argc,char **argv)
{
    if (argc < 2)
    {
        SparkReleaseManagerPrintUsage();
        return 2;
    }
    if (strcmp(argv[1],"example") == 0)
    {
        const char *output_path;

        output_path = 0;
        if (argc == 4 && strcmp(argv[2],"--output") == 0)
        {
            output_path = argv[3];
        }
        if (output_path == 0)
        {
            SparkReleaseManagerPrintUsage();
            return 2;
        }
        return SparkReleaseWriteExampleManifest(output_path) == SPARK_STATUS_OK ? 0 : 1;
    }
    if (strcmp(argv[1],"validate") == 0)
    {
        SparkReleaseManifest manifest;
        const char *manifest_path;
        SparkStatus status;

        manifest_path = 0;
        if (argc == 4 && strcmp(argv[2],"--manifest") == 0)
        {
            manifest_path = argv[3];
        }
        if (manifest_path == 0)
        {
            SparkReleaseManagerPrintUsage();
            return 2;
        }
        status = SparkReleaseManifestLoadFile(manifest_path,&manifest);
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr,"validate failed status=%d\n",(int)status);
            return 1;
        }
        printf("valid release_id=%s generation=%llu files=%u roles=%u\n",
            manifest.release_id,(unsigned long long)manifest.generation,manifest.file_count,manifest.role_count);
        return 0;
    }
    if (strcmp(argv[1],"plan") == 0)
    {
        SparkReleaseManifest manifest;
        SparkReleaseNodeIdentity identity;
        SparkReleaseResolvedRole resolved_role;
        const SparkReleaseRole *role;
        const char *manifest_path;
        const char *role_name;
        char command_line[SPARK_RELEASE_MANAGER_COMMAND_LINE_BYTES];
        int index;
        SparkStatus status;

        manifest_path = 0;
        role_name = 0;
        SparkReleaseNodeIdentityInitialize(&identity);
        for (index = 2; index < argc; ++index)
        {
            if (strcmp(argv[index],"--manifest") == 0 && index + 1 < argc)
            {
                manifest_path = argv[++index];
            }
            else if (strcmp(argv[index],"--host") == 0 && index + 1 < argc)
            {
                SparkCopyString(identity.host,sizeof(identity.host),argv[++index]);
            }
            else if (strcmp(argv[index],"--rank") == 0 && index + 1 < argc)
            {
                if (SparkNetParseU32(argv[++index],&identity.rank) != 0)
                {
                    return 2;
                }
                identity.rank_is_set = 1u;
            }
            else if (strcmp(argv[index],"--role") == 0 && index + 1 < argc)
            {
                role_name = argv[++index];
            }
            else
            {
                SparkReleaseManagerPrintUsage();
                return 2;
            }
        }
        if (manifest_path == 0 || identity.host[0] == '\0')
        {
            SparkReleaseManagerPrintUsage();
            return 2;
        }
        status = SparkReleaseManifestLoadFile(manifest_path,&manifest);
        if (status == SPARK_STATUS_OK)
        {
            identity.rank_count = manifest.rank_count;
            status = SparkReleaseManifestFindRoleForNode(&manifest,&identity,role_name,&role);
        }
        if (status == SPARK_STATUS_OK)
        {
            status = SparkReleaseResolveRole(&manifest,&identity,role,&resolved_role);
        }
        if (status == SPARK_STATUS_OK)
        {
            status = SparkReleaseFormatResolvedCommandLine(&resolved_role,command_line,sizeof(command_line));
        }
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr,"plan failed status=%d\n",(int)status);
            return 1;
        }
        printf("role=%s\ncommand=%s\npid_file=%s\n",role->name,command_line,resolved_role.pid_file);
        return 0;
    }
    if (strcmp(argv[1],"sync") == 0)
    {
        SparkReleaseManifest manifest;
        SparkReleaseSyncResult result;
        const char *manifest_path;
        const char *release_directory;
        const char *install_directory;
        int index;
        SparkStatus status;

        manifest_path = 0;
        release_directory = 0;
        install_directory = 0;
        for (index = 2; index < argc; ++index)
        {
            if (strcmp(argv[index],"--manifest") == 0 && index + 1 < argc)
            {
                manifest_path = argv[++index];
            }
            else if (strcmp(argv[index],"--release-dir") == 0 && index + 1 < argc)
            {
                release_directory = argv[++index];
            }
            else if (strcmp(argv[index],"--install-dir") == 0 && index + 1 < argc)
            {
                install_directory = argv[++index];
            }
            else
            {
                SparkReleaseManagerPrintUsage();
                return 2;
            }
        }
        if (manifest_path == 0 || release_directory == 0)
        {
            SparkReleaseManagerPrintUsage();
            return 2;
        }
        status = SparkReleaseManifestLoadFile(manifest_path,&manifest);
        if (status == SPARK_STATUS_OK)
        {
            status = SparkReleaseSyncFilesFromDirectory(&manifest,release_directory,install_directory,&result);
        }
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr,"sync failed status=%d\n",(int)status);
            return 1;
        }
        printf("synced verified=%u changed=%u copied=%u executable=%u cuda_packs=%u flags=0x%08x\n",
            result.verified_file_count,result.changed_file_count,result.copied_file_count,
            result.executable_file_count,result.cuda_pack_file_count,result.action_flags);
        return 0;
    }
    if (strcmp(argv[1],"agent") == 0)
    {
        return SparkReleaseManagerRunAgent(argc,argv);
    }
    if (strcmp(argv[1],"serve") == 0)
    {
        const char *release_directory;
        const char *bind_address;
        uint32_t port;
        int index;

        release_directory = 0;
        bind_address = "0.0.0.0";
        port = SPARK_RELEASE_MANAGER_DEFAULT_PORT;
        for (index = 2; index < argc; ++index)
        {
            if (strcmp(argv[index],"--release-dir") == 0 && index + 1 < argc)
            {
                release_directory = argv[++index];
            }
            else if (strcmp(argv[index],"--bind") == 0 && index + 1 < argc)
            {
                bind_address = argv[++index];
            }
            else if (strcmp(argv[index],"--port") == 0 && index + 1 < argc)
            {
                if (SparkNetParseU32(argv[++index],&port) != 0)
                {
                    return 2;
                }
            }
            else
            {
                SparkReleaseManagerPrintUsage();
                return 2;
            }
        }
        if (release_directory == 0)
        {
            SparkReleaseManagerPrintUsage();
            return 2;
        }
        return SparkReleaseManagerServe(release_directory,bind_address,port);
    }
    SparkReleaseManagerPrintUsage();
    return 2;
}


#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <fcntl.h>
#include <dirent.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/stat.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"

SparkStatus SparkWeightdMeshInit(uint32_t rank, const char *interface_name,
    uint32_t sgid_index, const char *mesh_dir);
uint32_t SparkWeightdMeshReady(void);
void SparkWeightdMeshDoorbellLoop(void);

typedef struct SparkWeightdMeshLaunch
{
    uint32_t rank;
    const char *interface_name;
    uint32_t sgid_index;
    const char *mesh_dir;
} SparkWeightdMeshLaunch;

static SparkWeightdMeshLaunch weightd_mesh_launch;

static void *SparkWeightdMeshThread(void *argument)
{
    SparkWeightdMeshLaunch *launch = (SparkWeightdMeshLaunch *)argument;
    SparkStatus status;
    status = SparkWeightdMeshInit(launch->rank,launch->interface_name,
        launch->sgid_index,launch->mesh_dir);
    if (status == SPARK_STATUS_BUSY)
        SparkWeightdMeshDoorbellLoop();
    else if (status != SPARK_STATUS_OK)
        fprintf(stderr, "weightd-mesh init=%s (serving degraded)\n",
            SparkStatusToString(status));
    return 0;
}

static volatile sig_atomic_t SparkWeightdStop;

static void SparkWeightdSignal(int signal_number)
{
    (void)signal_number;
    SparkWeightdStop = 1;
}

static void SparkWeightdUsage(const char *program)
{
    fprintf(stderr,
        "usage: %s --socket <path> [--device-bytes-max <bytes>]\n"
        "  --socket <path>          unix listen path "
            "(env SPARK_WEIGHTD_SOCKET, default /tmp/spark_weightd.sock)\n"
        "  --device-bytes-max <n>   arena ceiling in bytes "
            "(env SPARK_WEIGHTD_DEVICE_BYTES_MAX, default %llu — the "
            "operator 110 GiB device law; lower it when the node is "
            "shared, never raise it)\n"
        "  --mesh-rank <n>          mesh rank 0..%u; with "
            "--mesh-interface and --mesh-sgid-index, state all three "
            "or none\n"
        "  --mesh-interface <name>  verbs device name to bind\n"
        "  --mesh-sgid-index <n>    source GID index 0..255\n"
        "  --mesh-dir <path>        record exchange dir (env SPARK_WEIGHTD_MESH_DIR, default /tmp/weightd-mesh; use a per-deployment dir when two weightd-line daemons share the host)\n",
        program,
        (unsigned long long)SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT,
        (unsigned)SPARK_WEIGHTD_MESH_RANKS - 1u);
}


static int SparkWeightdLatchProbe(uint16_t port)
{
    struct sockaddr_in address;
    int probe_fd = socket(AF_INET, SOCK_STREAM, 0);
    if ( probe_fd < 0 )
        return -1;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if ( connect(probe_fd, (struct sockaddr *)&address, sizeof(address)) == 0 )
    {
        close(probe_fd);
        return 1;
    }
    close(probe_fd);
    return 0;
}

static pid_t SparkWeightdLatchHolder(uint16_t port)
{
    uint32_t want = (uint32_t)port << 16;
    char line[512];
    FILE *table = fopen("/proc/net/tcp", "r");
    if ( table == 0 )
        return -1;
    while ( fgets(line, sizeof(line), table) != 0 )
    {
        unsigned int local, state_hex, inode = 0;
        if ( sscanf(line, "%*s %x %*s %x %*s %*s %*s %*s %*s %u",
                &local, &state_hex, &inode) == 3 &&
            (local & 0xffffu) == port && (local >> 16) == 0x0100007fu &&
            inode != 0 )
        {
            char pattern[64];
            DIR *processes;
            struct dirent *entry;
            (void)fclose(table);
            table = 0;
            (void)snprintf(pattern, sizeof(pattern), "socket:[%u]", inode);
            processes = opendir("/proc");
            if ( processes == 0 )
                return -1;
            while ( (entry = readdir(processes)) != 0 )
            {
                char fd_path[64];
                DIR *fds;
                struct dirent *fd_entry;
                if ( entry->d_name[0] < '0' || entry->d_name[0] > '9' )
                    continue;
                (void)snprintf(fd_path, sizeof(fd_path), "/proc/%s/fd",
                    entry->d_name);
                fds = opendir(fd_path);
                if ( fds == 0 )
                    continue;
                while ( (fd_entry = readdir(fds)) != 0 )
                {
                    char link_path[128];
                    char target[96];
                    ssize_t length;
                    (void)snprintf(link_path, sizeof(link_path), "%s/%s",
                        fd_path, fd_entry->d_name);
                    length = readlink(link_path, target, sizeof(target) - 1);
                    if ( length > 0 )
                    {
                        target[length] = '\0';
                        if ( strcmp(target, pattern) == 0 )
                        {
                            pid_t found = (pid_t)atoi(entry->d_name);
                            (void)closedir(fds);
                            (void)closedir(processes);
                            return found;
                        }
                    }
                }
                (void)closedir(fds);
            }
            (void)closedir(processes);
            return -1;
        }
    }
    if ( table != 0 )
        (void)fclose(table);
    return -1;
}

static int SparkWeightdLatchAcquire(uint16_t port)
{
    struct sockaddr_in address;
    int latch_fd, attempt, on = 1;
    latch_fd = socket(AF_INET, SOCK_STREAM, 0);
    if ( latch_fd < 0 )
        return -1;
    (void)setsockopt(latch_fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    for ( attempt = 0; attempt < 12; attempt++ )
    {
        if ( bind(latch_fd, (struct sockaddr *)&address, sizeof(address)) == 0 )
        {
            if ( listen(latch_fd, 1) == 0 )
                return latch_fd;
            close(latch_fd);
            return -1;
        }
        if ( SparkWeightdLatchProbe(port) == 1 )
        {
            fprintf(stderr,
                "weightd: an instance already serves latch port %u; nothing to do\n",
                (unsigned)port);
            close(latch_fd);
            return 0;
        }
        if ( attempt == 0 )
        {
            pid_t holder = SparkWeightdLatchHolder(port);
            if ( holder > 0 )
            {
                fprintf(stderr,
                    "weightd: pid %d holds latch port %u without serving; taking over\n",
                    (int)holder, (unsigned)port);
                (void)kill(holder, SIGKILL);
            }
        }
        sleep(1);
    }
    fprintf(stderr, "weightd: latch port %u stayed occupied; exiting\n",
        (unsigned)port);
    close(latch_fd);
    return -1;
}

int main(int argument_count, char **arguments)
{
    const char *socket_path = "/tmp/spark_weightd.sock";
    uint16_t latch_port = SPARK_WEIGHTD_LATCH_PORT_DEFAULT;
    uint64_t device_bytes_max = SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT;
    uint64_t kv_reserve_bytes = 0ull;
    uint32_t mesh_fields = 0u;
    int ceiling_set_by_flag = 0;
    SparkWeightdServerConfig config;
    SparkWeightdServer *server = 0;
    uint32_t arena_count;
    uint64_t resident_bytes;
    SparkStatus status;
    int index;

    for (index = 1; index < argument_count; index++)
    {
        if (strcmp(arguments[index], "--socket") == 0 &&
            index + 1 < argument_count)
        {
            socket_path = arguments[++index];
        }
        else if (strcmp(arguments[index], "--latch-port") == 0 &&
            index + 1 < argument_count)
        {
            char *parse_end = 0;
            latch_port = (uint16_t)strtoul(arguments[index + 1], &parse_end, 10);
            if ( parse_end == arguments[index + 1] || *parse_end != '\0' ||
                latch_port == 0u )
            {
                fprintf(stderr, "weightd: bad --latch-port '%s'\n",
                    arguments[index + 1]);
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
            index++;
        }
        else if (strcmp(arguments[index], "--device-bytes-max") == 0 &&
            index + 1 < argument_count)
        {
            char *parse_end = 0;
            device_bytes_max = strtoull(arguments[index + 1], &parse_end, 10);
            if (parse_end == arguments[index + 1] || *parse_end != '\0' ||
                device_bytes_max == 0ull)
            {
                fprintf(stderr, "weightd: bad --device-bytes-max '%s'\n",
                    arguments[index + 1]);
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
            ceiling_set_by_flag = 1;
            index++;
        }
        else if (strcmp(arguments[index], "--kv-reserve-bytes") == 0 &&
            index + 1 < argument_count)
        {
            char *parse_end = 0;
            kv_reserve_bytes = strtoull(arguments[index + 1], &parse_end, 10);
            if (parse_end == arguments[index + 1] || *parse_end != '\0')
            {
                fprintf(stderr, "weightd: bad --kv-reserve-bytes '%s'\n",
                    arguments[index + 1]);
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
            index++;
        }
        else if (strcmp(arguments[index], "--mesh-rank") == 0 &&
            index + 1 < argument_count)
        {
            char *parse_end = 0;
            unsigned long parsed = strtoul(arguments[index + 1],
                &parse_end, 10);
            if (parse_end == arguments[index + 1] || *parse_end != '\0' ||
                parsed >= SPARK_WEIGHTD_MESH_RANKS)
            {
                fprintf(stderr,
                    "weightd: bad --mesh-rank '%s' (need 0..%u)\n",
                    arguments[index + 1],
                    (unsigned)SPARK_WEIGHTD_MESH_RANKS - 1u);
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
            weightd_mesh_launch.rank = (uint32_t)parsed;
            mesh_fields++;
            index++;
        }
        else if (strcmp(arguments[index], "--mesh-interface") == 0 &&
            index + 1 < argument_count)
        {
            weightd_mesh_launch.interface_name = arguments[++index];
            if (weightd_mesh_launch.interface_name[0] == '\0')
            {
                fprintf(stderr, "weightd: bad --mesh-interface ''\n");
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
            mesh_fields++;
        }
        else if (strcmp(arguments[index], "--mesh-sgid-index") == 0 &&
            index + 1 < argument_count)
        {
            char *parse_end = 0;
            unsigned long parsed = strtoul(arguments[index + 1],
                &parse_end, 10);
            if (parse_end == arguments[index + 1] || *parse_end != '\0' ||
                parsed > 255ul)
            {
                fprintf(stderr,
                    "weightd: bad --mesh-sgid-index '%s' (need 0..255)\n",
                    arguments[index + 1]);
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
            weightd_mesh_launch.sgid_index = (uint32_t)parsed;
            mesh_fields++;
            index++;
        }
        else if (strcmp(arguments[index], "--mesh-dir") == 0 &&
            index + 1 < argument_count)
        {
            weightd_mesh_launch.mesh_dir = arguments[++index];
            if (weightd_mesh_launch.mesh_dir[0] == '\0')
            {
                fprintf(stderr, "weightd: bad --mesh-dir ''\n");
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
        }
        else if (strcmp(arguments[index], "--help") == 0)
        {
            SparkWeightdUsage(arguments[0]);
            return 0;
        }
        else
        {
            fprintf(stderr, "weightd: unknown argument '%s'\n", arguments[index]);
            SparkWeightdUsage(arguments[0]);
            return 2;
        }
    }
    if (mesh_fields != 0u && mesh_fields != 3u)
    {
        fprintf(stderr,
            "weightd: mesh identity partially stated (%u of 3: "
            "--mesh-rank --mesh-interface --mesh-sgid-index); "
            "state all three or none\n", mesh_fields);
        return 2;
    }
    {
        const char *env_socket = getenv("SPARK_WEIGHTD_SOCKET");
        const char *env_ceiling = getenv("SPARK_WEIGHTD_DEVICE_BYTES_MAX");
        const char *env_reserve = getenv("SPARK_WEIGHTD_KV_RESERVE_BYTES");
        const char *env_mesh_dir = getenv("SPARK_WEIGHTD_MESH_DIR");
        if (weightd_mesh_launch.mesh_dir == 0 && env_mesh_dir != 0 &&
            env_mesh_dir[0] != '\0')
            weightd_mesh_launch.mesh_dir = env_mesh_dir;
        if (env_socket != 0 && env_socket[0] != '\0')
        {
            socket_path = env_socket;
        }
        if (env_ceiling != 0 && env_ceiling[0] != '\0')
        {
            char *parse_end = 0;
            uint64_t parsed = strtoull(env_ceiling, &parse_end, 10);
            if (parse_end == env_ceiling || *parse_end != '\0' ||
                parsed == 0ull)
            {
                fprintf(stderr, "weightd: bad SPARK_WEIGHTD_DEVICE_BYTES_MAX '%s'\n",
                    env_ceiling);
                return 2;
            }
            if (!ceiling_set_by_flag)
            {
                device_bytes_max = parsed;
            }
        }
        if (env_reserve != 0 && env_reserve[0] != '\0')
        {
            char *parse_end = 0;
            uint64_t parsed = strtoull(env_reserve, &parse_end, 10);
            if (parse_end == env_reserve || *parse_end != '\0')
            {
                fprintf(stderr, "weightd: bad SPARK_WEIGHTD_KV_RESERVE_BYTES '%s'\n",
                    env_reserve);
                return 2;
            }
            kv_reserve_bytes = parsed;
        }
    }

    if (kv_reserve_bytes >= device_bytes_max)
    {
        fprintf(stderr,
            "weightd: kv reserve %llu leaves no arena room under ceiling %llu\n",
            (unsigned long long)kv_reserve_bytes,
            (unsigned long long)device_bytes_max);
        return 2;
    }

    memset(&config, 0, sizeof(config));
    config.socket_path = socket_path;
    config.device_bytes_max = device_bytes_max;
    config.kv_reserve_bytes = kv_reserve_bytes;

    signal(SIGINT, SparkWeightdSignal);
    signal(SIGTERM, SparkWeightdSignal);

    {
        const char *latch_env = getenv("SPARK_WEIGHTD_LATCH_PORT");
        int latch_fd;
        if ( latch_env != 0 && latch_env[0] != '\0' )
        {
            char *parse_end = 0;
            unsigned long parsed = strtoul(latch_env, &parse_end, 10);
            if ( parse_end != latch_env && *parse_end == '\0' && parsed > 0ul &&
                parsed <= 65535ul )
                latch_port = (uint16_t)parsed;
        }
        latch_fd = SparkWeightdLatchAcquire(latch_port);
        if ( latch_fd == 0 )
            return 0;
        if ( latch_fd < 0 )
            return 1;
        (void)unlink(socket_path);
    }

    status = SparkWeightdServerCreate(&config, &server);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightd create=%s socket=%s\n",
            SparkStatusToString(status), socket_path);
        return 1;
    }
    printf("spark_weightd ready unix=%s ceiling=%llu\n",
        socket_path, (unsigned long long)device_bytes_max);
    fflush(stdout);

    if (mesh_fields == 3u)
    {
        static pthread_t mesh_thread;
        if (pthread_create(&mesh_thread,0,SparkWeightdMeshThread,
            &weightd_mesh_launch) != 0)
            fprintf(stderr, "weightd-mesh: thread create failed\n");
    }
    else
    {
        fprintf(stderr,
            "weightd-mesh: identity not stated; mesh disabled\n");
    }

    status = SparkWeightdServerRun(server, &SparkWeightdStop);

    arena_count = SparkWeightdServerArenaCount(server);
    resident_bytes = SparkWeightdServerResidentBytes(server);
    SparkWeightdServerDestroy(server);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightd run=%s\n", SparkStatusToString(status));
        return 1;
    }
    fprintf(stderr, "spark_weightd stopped arenas=%u bytes=%llu\n",
        arena_count, (unsigned long long)resident_bytes);
    return 0;
}


#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"

SparkStatus SparkWeightdMeshInit(uint32_t rank, const char *interface_name,
    uint32_t sgid_index, const char *mesh_dir, uint32_t rank_mask);
uint32_t SparkWeightdMeshReady(void);
void SparkWeightdMeshDoorbellLoop(void);

typedef struct SparkWeightdMeshLaunch
{
    uint32_t rank;
    uint32_t rank_mask;
    const char *interface_name;
    uint32_t sgid_index;
    const char *mesh_dir;
} SparkWeightdMeshLaunch;

static SparkWeightdMeshLaunch weightd_mesh_launch;

static void *SparkWeightdMeshThread(void *argument)
{
    (void)argument;
    SparkWeightdMeshDoorbellLoop();
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
            "--mesh-interface, --mesh-sgid-index and --mesh-rank-mask; state all four "
            "or none\n"
        "  --mesh-rank-mask <mask>  exact participant ranks including self (e.g. 0xf for TP4)\n"
        "  --mesh-interface <name>  verbs device name to bind\n"
        "  --mesh-sgid-index <n>    source GID index 0..255\n"
        "  --mesh-dir <path>        record exchange dir (env SPARK_WEIGHTD_MESH_DIR, default /tmp/weightd-mesh; use a per-deployment dir when two weightd-line daemons share the host)\n",
        program,
        (unsigned long long)SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT,
        (unsigned)SPARK_WEIGHTD_MESH_RANKS - 1u);
}

#define SPARK_WEIGHTD_LATCH_PORT_DEFAULT 61900u

static int spark_weightd_latch_fd = -1;

static int SparkWeightdLatchBind(uint16_t port)
{
    struct sockaddr_in address;
    int fd = socket(AF_INET,SOCK_STREAM,0);
    int on = 1;
    if ( fd < 0 )
        return(-1);
    (void)setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&on,sizeof(on));
    memset(&address,0,sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    if ( bind(fd,(const struct sockaddr *)&address,sizeof(address)) != 0 ||
         listen(fd,1) != 0 )
    {
        (void)close(fd);
        return(-1);
    }
    return(fd);
}

static int SparkWeightdLatchAcquire(uint16_t port)
{
    if ( port == 0u )
    {
        fprintf(stderr,"weightd latch: port must be nonzero\n");
        return(-1);
    }
    spark_weightd_latch_fd = SparkWeightdLatchBind(port);
    if ( spark_weightd_latch_fd < 0 )
    {
        fprintf(stderr,"weightd latch: cannot own port %u: %s; existing owner untouched\n",
            (unsigned)port,strerror(errno));
        return(-1);
    }
    fprintf(stderr,"weightd latch: acquired port %u\n",(unsigned)port);
    return(1);
}

int main(int argument_count, char **arguments)
{
    const char *socket_path = "/tmp/spark_weightd.sock";
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
            mesh_fields |= 1u;
            index++;
        }
        else if (strcmp(arguments[index], "--mesh-rank-mask") == 0 &&
            index + 1 < argument_count)
        {
            char *parse_end = 0;
            unsigned long parsed;
            errno = 0;
            parsed = strtoul(arguments[index + 1],&parse_end,0);
            if ( errno != 0 || parse_end == arguments[index + 1] ||
                 *parse_end != '\0' || parsed >= (1ul << SPARK_WEIGHTD_MESH_RANKS) ||
                 __builtin_popcount((uint32_t)parsed) < 2 )
            {
                fprintf(stderr,"weightd: bad --mesh-rank-mask '%s'\n",arguments[index + 1]);
                return 2;
            }
            weightd_mesh_launch.rank_mask = (uint32_t)parsed;
            mesh_fields |= 8u;
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
            mesh_fields |= 2u;
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
            mesh_fields |= 4u;
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
    if (mesh_fields != 0u && (mesh_fields != 15u ||
        (weightd_mesh_launch.rank_mask & (1u << weightd_mesh_launch.rank)) == 0u))
    {
        fprintf(stderr,
            "weightd: mesh identity incomplete or rank outside group (fields=%x: "
            "--mesh-rank --mesh-interface --mesh-sgid-index --mesh-rank-mask); "
            "state all four or none\n", mesh_fields);
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
        uint16_t latch_port = SPARK_WEIGHTD_LATCH_PORT_DEFAULT;
        const char *latch_env = getenv("SPARK_WEIGHTD_LATCH_PORT");
        int latch;
        if ( latch_env != 0 )
        {
            char *end;
            unsigned long value;
            errno = 0;
            value = strtoul(latch_env,&end,10);
            if ( errno != 0 || end == latch_env || *end != '\0' ||
                 latch_env[0] == '-' || value == 0ul || value > UINT16_MAX )
            {
                fprintf(stderr,"weightd: invalid SPARK_WEIGHTD_LATCH_PORT\n");
                return 2;
            }
            latch_port = (uint16_t)value;
        }
        latch = SparkWeightdLatchAcquire(latch_port);
        if ( latch < 0 )
            return 1;
    }

    status = SparkWeightdServerCreate(&config, &server);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightd create=%s socket=%s\n",
            SparkStatusToString(status), socket_path);
        return 1;
    }
    if (mesh_fields == 15u)
    {
        static pthread_t mesh_thread;
        status = SparkWeightdMeshInit(weightd_mesh_launch.rank,
            weightd_mesh_launch.interface_name,weightd_mesh_launch.sgid_index,
            weightd_mesh_launch.mesh_dir,weightd_mesh_launch.rank_mask);
        if ( status != SPARK_STATUS_BUSY && status != SPARK_STATUS_OK )
        {
            fprintf(stderr,"weightd-mesh init=%s; startup failed\n",SparkStatusToString(status));
            SparkWeightdServerDestroy(server);
            return 1;
        }
        if (pthread_create(&mesh_thread,0,SparkWeightdMeshThread,0) != 0)
        {
            fprintf(stderr,"weightd-mesh: thread create failed; startup failed\n");
            SparkWeightdServerDestroy(server);
            return 1;
        }
    }
    else
    {
        fprintf(stderr,
            "weightd-mesh: identity not stated; mesh disabled\n");
    }

    printf("spark_weightd ready unix=%s ceiling=%llu\n",
        socket_path, (unsigned long long)device_bytes_max);
    fflush(stdout);

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

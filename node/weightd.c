/* spark_weightd — the weight-residency daemon process
 * (docs/WEIGHTD_DESIGN.md W2a skeleton).
 *
 * One daemon per node owns the weight arenas; serving processes attach by
 * content identity and never load pack bytes themselves. The process is
 * deliberately thin: argument parse, signal flags, the shared server's
 * Run loop, and a clean teardown (close every connection — refcounts drop
 * — free every arena, unlink the socket). TERM is THE shutdown path: the
 * loop's poll quantum bounds how late the stop flag is seen, so the fleet
 * TERM-first protocol applies to this daemon by construction.
 *
 *   spark_weightd --socket /tmp/spark_weightd.sock \
 *       [--device-bytes-max <bytes>]   # default: the 110 GiB device law
 *
 * The ready line (stdout, flushed) is the launch gate:
 *   spark_weightd ready unix=<path> ceiling=<bytes>
 * TERM completion prints:
 *   spark_weightd stopped arenas=<n> bytes=<n>
 * to stderr and exits 0. */

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"

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
            "shared, never raise it)\n",
        program,
        (unsigned long long)SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT);
}

int main(int argument_count, char **arguments)
{
    const char *socket_path = "/tmp/spark_weightd.sock";
    uint64_t device_bytes_max = SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT;
    uint64_t kv_reserve_bytes = 0ull;
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
    {
        const char *env_socket = getenv("SPARK_WEIGHTD_SOCKET");
        const char *env_ceiling = getenv("SPARK_WEIGHTD_DEVICE_BYTES_MAX");
        const char *env_reserve = getenv("SPARK_WEIGHTD_KV_RESERVE_BYTES");
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
            /* the flag wins over the environment; environment wins over the
             * law's default — the law itself is never raised by either */
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

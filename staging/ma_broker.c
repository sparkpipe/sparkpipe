#include <nng/nng.h>
#include <nng/protocol/survey0/survey.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ENTRY_MAX 1024
#define TABLE_MAX (16 * ENTRY_MAX)

static int entry_bytes = 128;

int main(int argc, char **argv)
{
    nng_socket sock;
    int degree = 16;
    int port = 58399;
    uint8_t table[TABLE_MAX];
    int have[16];
    int i;
    int total = 0;
    int rv;
    char url[48];
    setvbuf(stdout, 0, _IOLBF, 0);
    if (argc >= 3)
    {
        degree = atoi(argv[1]);
        port = atoi(argv[2]);
    }
    if (argc >= 4)
        entry_bytes = atoi(argv[3]);
    if (entry_bytes != 128 && entry_bytes != 1024)
        return 2;
    if (degree != 4 && degree != 8 && degree != 16)
        return 2;
    memset(have, 0, sizeof(have));
    snprintf(url, sizeof(url), "tcp://0.0.0.0:%d", port);
    if ((rv = nng_surveyor0_open(&sock)) != 0)
    {
        fprintf(stderr, "broker open: %s\n", nng_strerror(rv));
        return 1;
    }
    nng_socket_set_ms(sock, NNG_OPT_RECVTIMEO, 8000);
    if ((rv = nng_listen(sock, url, NULL, 0)) != 0)
    {
        fprintf(stderr, "broker listen: %s\n", nng_strerror(rv));
        return 1;
    }
    printf("BROKER nng surveyor degree=%d port=%d listening\n", degree, port);
    while (total < degree)
    {
        uint8_t *msg = 0;
        size_t sz = 0;
        int32_t rank;
        if ((rv = nng_send(sock, "JOIN", 4, 0)) != 0)
        {
            fprintf(stderr, "broker survey send: %s\n", nng_strerror(rv));
            return 1;
        }
        while ((rv = nng_recv(sock, &msg, &sz, NNG_FLAG_ALLOC)) == 0)
        {
            if (sz == (size_t)(4 + entry_bytes) && msg != 0)
            {
                memcpy(&rank, msg, sizeof(rank));
                if (rank >= 0 && rank < degree && have[rank] == 0)
                {
                    memcpy(table + (size_t)rank * (size_t)entry_bytes,
                        msg + 4, (size_t)entry_bytes);
                    have[rank] = 1;
                    ++total;
                    printf("BROKER rank %d joined (%d/%d)\n", rank, total,
                        degree);
                }
            }
            nng_free(msg, sz);
            msg = 0;
            if (total == degree)
                break;
        }
        if (total < degree)
            nng_msleep(1000);
    }
    for (i = 0; i < 3; ++i)
    {
        if ((rv = nng_send(sock, table,
                (size_t)degree * (size_t)entry_bytes, 0)) != 0)
        {
            fprintf(stderr, "broker table send: %s\n", nng_strerror(rv));
            return 1;
        }
        nng_msleep(1500);
    }
    printf("BROKER table surveys sent\n");
    nng_close(sock);
    return 0;
}

#include <nng/nng.h>
#include <nng/protocol/survey0/survey.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define DEGREE 16
#define ENTRY_BYTES 128
#define JOIN_BYTES (4 + ENTRY_BYTES)
#define TABLE_BYTES (DEGREE * ENTRY_BYTES)

int main(void)
{
    nng_socket sock;
    uint8_t table[TABLE_BYTES];
    int have[DEGREE];
    int i;
    int total = 0;
    int rv;
    setvbuf(stdout, 0, _IOLBF, 0);
    memset(have, 0, sizeof(have));
    if ((rv = nng_surveyor0_open(&sock)) != 0)
    {
        fprintf(stderr, "broker open: %s\n", nng_strerror(rv));
        return 1;
    }
    nng_socket_set_ms(sock, NNG_OPT_RECVTIMEO, 8000);
    nng_socket_set_ms(sock, NNG_OPT_SURVEYTIME, 9000);
    if ((rv = nng_listen(sock, "tcp://0.0.0.0:58399", NULL, 0)) != 0)
    {
        fprintf(stderr, "broker listen: %s\n", nng_strerror(rv));
        return 1;
    }
    printf("BROKER nng surveyor listening\n");
    while (total < DEGREE)
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
            if (sz == JOIN_BYTES && msg != 0)
            {
                memcpy(&rank, msg, sizeof(rank));
                if (rank >= 0 && rank < DEGREE && have[rank] == 0)
                {
                    memcpy(table + (size_t)rank * ENTRY_BYTES, msg + 4,
                        ENTRY_BYTES);
                    have[rank] = 1;
                    ++total;
                    printf("BROKER rank %d joined (%d/16)\n", rank, total);
                }
            }
            nng_free(msg, sz);
            msg = 0;
            if (total == DEGREE)
                break;
        }
        if (total < DEGREE)
            nng_msleep(1000);
    }
    for (i = 0; i < 3; ++i)
    {
        if ((rv = nng_send(sock, table, TABLE_BYTES, 0)) != 0)
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

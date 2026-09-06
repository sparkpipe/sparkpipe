#include <nng/nng.h>
#include <nng/protocol/survey0/respond.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    nng_socket sock;
    uint8_t *msg = 0;
    size_t sz = 0;
    char url[64];
    uint8_t join[4 + 128];
    int32_t rank32;
    int rv;
    int reps;
    if (argc < 3)
        return 2;
    snprintf(url, sizeof(url), "tcp://10.10.100.19:%s", argv[1]);
    rank32 = (int32_t)atoi(argv[2]);
    if ((rv = nng_respondent0_open(&sock)) != 0)
        return 1;
    nng_socket_set_ms(sock, NNG_OPT_RECVTIMEO, 60000);
    if ((rv = nng_dial(sock, url, NULL, NNG_FLAG_NONBLOCK)) != 0)
        return 1;
    for (reps = 0; reps < 8; ++reps)
    {
        rv = nng_recv(sock, &msg, &sz, NNG_FLAG_ALLOC);
        if (rv != 0)
        {
            fprintf(stderr, "mini: recv %s\n", nng_strerror(rv));
            return 1;
        }
        fprintf(stderr, "mini: got sz=%zu\n", sz);
        nng_free(msg, sz);
        memset(join, 0x41, sizeof(join));
        memcpy(join, &rank32, sizeof(rank32));
        rv = nng_send(sock, join, sizeof(join), 0);
        fprintf(stderr, "mini: sent join rv=%d\n", rv);
        if (rv != 0)
            return 1;
    }
    return 0;
}

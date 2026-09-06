#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

int main(void)
{
    struct sockaddr_in a;
    int fd;
    int attempt;
    for (attempt = 0; attempt < 5; ++attempt)
    {
        memset(&a, 0, sizeof(a));
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl((10u << 24) | (10u << 16) | (100u << 8) |
            11u);
        a.sin_port = htons(58301);
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (connect(fd, (struct sockaddr *)&a, sizeof(a)) == 0)
        {
            printf("probe attempt %d: CONNECTED\n", attempt);
            close(fd);
            return 0;
        }
        printf("probe attempt %d: errno=%d (%s)\n", attempt, errno,
            strerror(errno));
        close(fd);
        sleep(2);
    }
    return 1;
}

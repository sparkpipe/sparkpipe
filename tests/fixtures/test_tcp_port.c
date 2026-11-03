#include "test_tcp_port.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static uint32_t test_tcp_port_state;
static uint32_t test_tcp_port_handed[TEST_TCP_PORT_HANDED_MAX];
static uint32_t test_tcp_port_handed_count;

static uint32_t TestTcpPortNext(void)
{
	if ( test_tcp_port_state == 0u )
		test_tcp_port_state = ((uint32_t)getpid() * 2654435761u) ^ (uint32_t)time(0) ^ 1u;
	test_tcp_port_state ^= test_tcp_port_state << 13u;
	test_tcp_port_state ^= test_tcp_port_state >> 17u;
	test_tcp_port_state ^= test_tcp_port_state << 5u;
	return(test_tcp_port_state);
}

static int TestTcpPortHanded(uint32_t port)
{
	uint32_t index;
	for (index=0u; index<test_tcp_port_handed_count; index++)
		if ( test_tcp_port_handed[index] == port )
			return(1);
	return(0);
}

static int TestTcpPortBindable(uint32_t port)
{
	struct sockaddr_in address;
	int bound;
	int fd = socket(AF_INET,SOCK_STREAM,0);
	if ( fd < 0 )
		return(0);
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	address.sin_port = htons((uint16_t)port);
	bound = bind(fd,(const struct sockaddr *)&address,(socklen_t)sizeof(address)) == 0;
	close(fd);
	return(bound);
}

uint32_t TestTcpPortRange(uint32_t count)
{
	uint32_t attempt,base,offset;
	if ( count == 0u || count >= TEST_TCP_PORT_END - TEST_TCP_PORT_FIRST || test_tcp_port_handed_count + count > TEST_TCP_PORT_HANDED_MAX )
		return(0u);
	for (attempt=0u; attempt<TEST_TCP_PORT_ATTEMPTS; attempt++)
	{
		base = TEST_TCP_PORT_FIRST + TestTcpPortNext() % (TEST_TCP_PORT_END - TEST_TCP_PORT_FIRST - count);
		for (offset=0u; offset<count && TestTcpPortHanded(base + offset) == 0 && TestTcpPortBindable(base + offset) != 0; offset++)
			;
		if ( offset == count )
		{
			for (offset=0u; offset<count; offset++)
				test_tcp_port_handed[test_tcp_port_handed_count++] = base + offset;
			return(base);
		}
	}
	return(0u);
}

#!/usr/bin/env python3
"""The fleet weightd every shared-socket lane attaches to.

fleet_node_agent.sh starts sparkpipe_weightd inside the fleet-agent unit on
each Spark with this socket, which is also weightd's built-in default. Lane
generators require it so a deployment can only name the daemon that serves
the node.
"""

FLEET_WEIGHTD_SOCKET = "/tmp/spark_weightd.sock"


def fleet_weightd_socket_error(path: str) -> str:
    if path == FLEET_WEIGHTD_SOCKET:
        return ""
    return f"weightd socket {path} is not the fleet weightd {FLEET_WEIGHTD_SOCKET}"


if __name__ == "__main__":
    print(FLEET_WEIGHTD_SOCKET)

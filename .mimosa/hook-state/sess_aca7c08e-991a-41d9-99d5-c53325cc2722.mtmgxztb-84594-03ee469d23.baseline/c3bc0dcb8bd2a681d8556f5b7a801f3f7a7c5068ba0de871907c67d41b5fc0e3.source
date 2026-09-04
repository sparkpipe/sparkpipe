#!/usr/bin/env python3
"""Open an SSH byte stream over the best available Spark management path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import ipaddress
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import threading
import time
from typing import BinaryIO


PROFILE_FORMAT = "ds4-spark-ssh-failover-v1"
SSH_PREFACE_LIMIT = 8192
COPY_BYTES = 1024 * 1024


class RouteFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class EmergencyRoute:
    user: str
    port: int
    known_hosts: Path


@dataclass(frozen=True)
class NodeRoute:
    node_id: str
    user: str
    management_ip: str
    fabric_ip: str
    wifi_hosts: tuple[str,...]
    direct_enabled: bool


@dataclass(frozen=True)
class RouteProfile:
    ssh_binary: str
    direct_timeout_s: float
    ring_timeout_s: float
    wifi_timeout_s: float
    route_log: Path
    spark_known_hosts: Path
    emergency: EmergencyRoute
    ring_bastions: tuple[str,...]
    wifi_bastions: tuple[str,...]
    nodes: dict[str,NodeRoute]


@dataclass
class OpenChannel:
    route: str
    endpoint: str
    preface: bytes
    sock: socket.socket | None = None
    process: subprocess.Popen[bytes] | None = None


def _required_text(value: object, label: str) -> str:
    if not isinstance(value,str) or value == "":
        raise RouteFailure(f"{label} must be a nonempty string")
    return(value)


def _required_number(value: object, label: str) -> float:
    if isinstance(value,bool) or not isinstance(value,(int,float)) or value <= 0:
        raise RouteFailure(f"{label} must be positive")
    return(float(value))


def _required_port(value: object, label: str) -> int:
    if isinstance(value,bool) or not isinstance(value,int) or value < 1 or value > 65535:
        raise RouteFailure(f"{label} must be in 1..65535")
    return(value)


def _expanded_path(value: object, label: str) -> Path:
    return(Path(_required_text(value,label)).expanduser())


def _validate_ip(value: object, label: str) -> str:
    text = _required_text(value,label)
    try:
        parsed = ipaddress.ip_address(text)
    except ValueError as error:
        raise RouteFailure(f"{label} is not an IP address: {text}") from error
    if parsed.version != 4:
        raise RouteFailure(f"{label} must be IPv4")
    return(text)


def _parse_node(value: object, index: int) -> NodeRoute:
    if not isinstance(value,dict):
        raise RouteFailure(f"nodes[{index}] must be an object")
    wifi_value = value.get("wifi_hosts",[])
    if not isinstance(wifi_value,list):
        raise RouteFailure(f"nodes[{index}].wifi_hosts must be an array")
    wifi_hosts = tuple(_required_text(item,f"nodes[{index}].wifi_hosts") for item in wifi_value)
    direct_enabled = value.get("direct_enabled",True)
    if not isinstance(direct_enabled,bool):
        raise RouteFailure(f"nodes[{index}].direct_enabled must be boolean")
    return(NodeRoute(
        node_id=_required_text(value.get("node_id"),f"nodes[{index}].node_id"),
        user=_required_text(value.get("user"),f"nodes[{index}].user"),
        management_ip=_validate_ip(value.get("management_ip"),f"nodes[{index}].management_ip"),
        fabric_ip=_validate_ip(value.get("fabric_ip"),f"nodes[{index}].fabric_ip"),
        wifi_hosts=wifi_hosts,
        direct_enabled=direct_enabled,
    ))


def load_profile(path: Path) -> RouteProfile:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError,UnicodeDecodeError,json.JSONDecodeError) as error:
        raise RouteFailure(f"could not read route profile {path}: {error}") from error
    if not isinstance(raw,dict) or raw.get("format") != PROFILE_FORMAT:
        raise RouteFailure(f"route profile format must be {PROFILE_FORMAT}")
    emergency_value = raw.get("emergency")
    if not isinstance(emergency_value,dict):
        raise RouteFailure("emergency must be an object")
    nodes_value = raw.get("nodes")
    if not isinstance(nodes_value,list) or len(nodes_value) == 0:
        raise RouteFailure("nodes must be a nonempty array")
    nodes: dict[str,NodeRoute] = {}
    for index,value in enumerate(nodes_value):
        node = _parse_node(value,index)
        if node.node_id in nodes:
            raise RouteFailure(f"duplicate node_id {node.node_id}")
        nodes[node.node_id] = node
    ring_bastions = _parse_bastions(raw.get("ring_bastions"),"ring_bastions",nodes)
    wifi_bastions = _parse_bastions(raw.get("wifi_bastions"),"wifi_bastions",nodes)
    return(RouteProfile(
        ssh_binary=_required_text(raw.get("ssh_binary"),"ssh_binary"),
        direct_timeout_s=_required_number(raw.get("direct_timeout_s"),"direct_timeout_s"),
        ring_timeout_s=_required_number(raw.get("ring_timeout_s"),"ring_timeout_s"),
        wifi_timeout_s=_required_number(raw.get("wifi_timeout_s"),"wifi_timeout_s"),
        route_log=_expanded_path(raw.get("route_log"),"route_log"),
        spark_known_hosts=_expanded_path(raw.get("spark_known_hosts"),"spark_known_hosts"),
        emergency=EmergencyRoute(
            user=_required_text(emergency_value.get("user"),"emergency.user"),
            port=_required_port(emergency_value.get("port"),"emergency.port"),
            known_hosts=_expanded_path(emergency_value.get("known_hosts"),"emergency.known_hosts"),
        ),
        ring_bastions=ring_bastions,
        wifi_bastions=wifi_bastions,
        nodes=nodes,
    ))


def _parse_bastions(value: object, label: str, nodes: dict[str,NodeRoute]) -> tuple[str,...]:
    if not isinstance(value,list) or len(value) == 0:
        raise RouteFailure(f"{label} must be a nonempty array")
    out = tuple(_required_text(item,label) for item in value)
    if len(set(out)) != len(out):
        raise RouteFailure(f"{label} contains duplicates")
    missing = [item for item in out if item not in nodes]
    if missing:
        raise RouteFailure(f"{label} references unknown nodes: {','.join(missing)}")
    return(out)


def _contains_ssh_banner(data: bytes) -> bool:
    return(any(line.startswith(b"SSH-") for line in data.splitlines()))


def _read_ssh_preface(fd: int, timeout_s: float) -> bytes:
    deadline = time.monotonic() + timeout_s
    data = bytearray()
    while len(data) < SSH_PREFACE_LIMIT:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        readable,_,_ = select.select([fd],[],[],remaining)
        if not readable:
            break
        chunk = os.read(fd,min(1024,SSH_PREFACE_LIMIT - len(data)))
        if chunk == b"":
            break
        data.extend(chunk)
        if _contains_ssh_banner(bytes(data)):
            return(bytes(data))
    raise RouteFailure("endpoint did not provide an SSH banner")


def _ring_command(profile: RouteProfile, node: NodeRoute, bastion: str, port: int) -> list[str]:
    return([
        profile.ssh_binary,
        "-T",
        "-o", "BatchMode=yes",
        "-o", "ConnectionAttempts=1",
        "-o", f"ConnectTimeout={max(1,int(profile.ring_timeout_s))}",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "LogLevel=ERROR",
        "-W", f"{node.fabric_ip}:{port}",
        f"{bastion}-10g",
    ])


def _wifi_command(
    profile: RouteProfile,
    wifi_host: str,
    host_key_alias: str,
    destination: str,
    port: int,
) -> list[str]:
    emergency = profile.emergency
    command = [
        profile.ssh_binary,
        "-F", "/dev/null",
        "-T",
        "-o", "BatchMode=yes",
        "-o", "ConnectionAttempts=1",
        "-o", f"ConnectTimeout={max(1,int(profile.wifi_timeout_s))}",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={emergency.known_hosts}",
        "-o", f"HostKeyAlias={host_key_alias}",
        "-o", "LogLevel=ERROR",
        "-p", str(emergency.port),
    ]
    if ":" in wifi_host:
        command.append("-6")
    command.extend([
        f"{emergency.user}@{wifi_host}",
        "exec", "/usr/bin/nc", destination, str(port),
    ])
    return(command)


def _open_tcp_channel(profile: RouteProfile, node: NodeRoute, port: int) -> OpenChannel:
    try:
        sock = socket.create_connection(
            (node.management_ip,port),
            timeout=profile.direct_timeout_s,
        )
        preface = _read_ssh_preface(sock.fileno(),profile.direct_timeout_s)
        sock.settimeout(None)
        return(OpenChannel(
            route="10g",
            endpoint=f"{node.management_ip}:{port}",
            preface=preface,
            sock=sock,
        ))
    except (OSError,RouteFailure) as error:
        try:
            sock.close()
        except UnboundLocalError:
            pass
        raise RouteFailure(str(error)) from error


def _open_process_channel(
    command: list[str],
    route: str,
    endpoint: str,
    timeout_s: float,
) -> OpenChannel:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise RouteFailure(str(error)) from error
    assert process.stdout is not None
    try:
        preface = _read_ssh_preface(process.stdout.fileno(),timeout_s)
    except RouteFailure:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    return(OpenChannel(route=route,endpoint=endpoint,preface=preface,process=process))


def _ring_channels(profile: RouteProfile, node: NodeRoute, port: int):
    for bastion in profile.ring_bastions:
        endpoint = f"{bastion}->{node.fabric_ip}:{port}"
        yield(lambda b=bastion,e=endpoint: _open_process_channel(
            _ring_command(profile,node,b,port),
            "200g",
            e,
            profile.ring_timeout_s,
        ))


def _wifi_channels(profile: RouteProfile, node: NodeRoute, port: int):
    for wifi_host in node.wifi_hosts:
        endpoint = f"{wifi_host}->127.0.0.1:{port}"
        yield(lambda h=wifi_host,e=endpoint: _open_process_channel(
            _wifi_command(profile,h,f"{node.node_id}-emergency","127.0.0.1",port),
            "wifi",
            e,
            profile.wifi_timeout_s,
        ))
    for bastion_id in profile.wifi_bastions:
        bastion = profile.nodes[bastion_id]
        for wifi_host in bastion.wifi_hosts:
            endpoint = f"{wifi_host}->{node.fabric_ip}:{port}"
            yield(lambda h=wifi_host,b=bastion_id,e=endpoint: _open_process_channel(
                _wifi_command(profile,h,f"{b}-emergency",node.fabric_ip,port),
                "wifi",
                e,
                profile.wifi_timeout_s,
            ))


def open_channel(
    profile: RouteProfile,
    node: NodeRoute,
    route: str,
    port: int,
) -> tuple[OpenChannel,list[str]]:
    attempts = []
    factories = []
    if route in ("auto","10g") and node.direct_enabled:
        factories.append(lambda: _open_tcp_channel(profile,node,port))
    if route in ("auto","200g"):
        factories.extend(_ring_channels(profile,node,port))
    if route in ("auto","wifi"):
        factories.extend(_wifi_channels(profile,node,port))
    for factory in factories:
        try:
            return(factory(),attempts)
        except RouteFailure as error:
            attempts.append(str(error))
    raise RouteFailure("; ".join(attempts) or f"no route candidates for {route}")


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        offset += os.write(fd,data[offset:])


def _copy_stdin_to_socket(sock: socket.socket) -> None:
    try:
        while True:
            data = os.read(0,COPY_BYTES)
            if data == b"":
                break
            sock.sendall(data)
        sock.shutdown(socket.SHUT_WR)
    except OSError:
        pass


def _copy_stdin_to_file(stream: BinaryIO) -> None:
    try:
        while True:
            data = os.read(0,COPY_BYTES)
            if data == b"":
                break
            stream.write(data)
            stream.flush()
    except (BrokenPipeError,OSError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def relay_channel(channel: OpenChannel) -> int:
    _write_all(1,channel.preface)
    if channel.sock is not None:
        thread = threading.Thread(target=_copy_stdin_to_socket,args=(channel.sock,),daemon=True)
        thread.start()
        try:
            while True:
                data = channel.sock.recv(COPY_BYTES)
                if data == b"":
                    break
                _write_all(1,data)
        finally:
            channel.sock.close()
        thread.join(timeout=1)
        return(0)
    if channel.process is None or channel.process.stdin is None or channel.process.stdout is None:
        raise RouteFailure("opened channel has no transport")
    thread = threading.Thread(target=_copy_stdin_to_file,args=(channel.process.stdin,),daemon=True)
    thread.start()
    while True:
        data = os.read(channel.process.stdout.fileno(),COPY_BYTES)
        if data == b"":
            break
        _write_all(1,data)
    code = channel.process.wait()
    thread.join(timeout=1)
    return(code)


def close_channel(channel: OpenChannel) -> None:
    if channel.sock is not None:
        channel.sock.close()
    if channel.process is not None:
        channel.process.terminate()
        try:
            channel.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            channel.process.kill()
            channel.process.wait()


def log_selection(profile: RouteProfile, node: NodeRoute, channel: OpenChannel) -> None:
    try:
        profile.route_log.parent.mkdir(parents=True,exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).isoformat()
        with profile.route_log.open("a",encoding="utf-8") as handle:
            handle.write(f"{stamp} node={node.node_id} route={channel.route} endpoint={channel.endpoint}\n")
    except OSError:
        pass


def _default_profile_path() -> Path:
    configured = os.environ.get("DS4_SPARK_SSH_PROFILE")
    if configured:
        return(Path(configured).expanduser())
    return(Path(__file__).with_name("spark_ssh_failover.json"))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=_default_profile_path())
    parser.add_argument("--node", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--route", choices=("auto","10g","200g","wifi"), default="auto")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    return(parser.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        if arguments.port < 1 or arguments.port > 65535:
            raise RouteFailure("--port must be in 1..65535")
        profile = load_profile(arguments.profile)
        node = profile.nodes.get(arguments.node)
        if node is None:
            raise RouteFailure(f"unknown Spark node {arguments.node}")
        channel,attempts = open_channel(profile,node,arguments.route,arguments.port)
        if not arguments.no_log:
            log_selection(profile,node,channel)
        if arguments.probe:
            print(json.dumps({
                "format": "ds4-spark-ssh-route-probe-v1",
                "status": "ok",
                "node": node.node_id,
                "route": channel.route,
                "endpoint": channel.endpoint,
                "failed_attempts": len(attempts),
            },sort_keys=True))
            close_channel(channel)
            return(0)
        return(relay_channel(channel))
    except RouteFailure as error:
        print(f"spark_ssh_proxy: {error}",file=sys.stderr)
        return(1)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render and verify the RTX 5090 fleet-hub node contract."""

from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import sys


FORMAT = "ds4-auxiliary-node-v2"
SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.:@%/-]+")


class SpecNodeError(RuntimeError):
    """Raised when the auxiliary-node contract is invalid or not live."""


def _string(payload: dict[str,object],key: str) -> str:
    value = payload.get(key)
    if not isinstance(value,str) or SAFE_TOKEN.fullmatch(value) is None:
        raise SpecNodeError(f"invalid {key}")
    return(value)


def _text(payload: dict[str,object],key: str) -> str:
    value = payload.get(key)
    if not isinstance(value,str) or value == "":
        raise SpecNodeError(f"invalid {key}")
    return(value)


def load_profile(path: Path) -> dict[str,object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as error:
        raise SpecNodeError(f"cannot read profile {path}: {error}") from error
    if not isinstance(payload,dict) or payload.get("format") != FORMAT:
        raise SpecNodeError(f"unsupported profile format: {path}")
    validate_profile(payload)
    return(payload)


def _mapping(payload: dict[str,object],key: str) -> dict[str,object]:
    value = payload.get(key)
    if not isinstance(value,dict):
        raise SpecNodeError(f"invalid {key}")
    return(value)


def _port(payload: dict[str,object],key: str) -> int:
    value = payload.get(key)
    if not isinstance(value,int) or isinstance(value,bool) or value < 1 or value > 65535:
        raise SpecNodeError(f"{key} must be in 1..65535")
    return(value)


def validate_profile(profile: dict[str,object]) -> None:
    if profile.get("role") != "hub" or profile.get("spark_rank") is not None:
        raise SpecNodeError("the hub node must not be assigned a Spark rank")
    operator = _mapping(profile,"operator")
    recovery = _mapping(profile,"recovery_ssh")
    link = _mapping(profile,"fleet_link")
    rtx = _mapping(link,"rtx5090")
    sparkf = _mapping(link,"sparkf")
    fabric = _mapping(link,"fabric")
    hub = _mapping(profile,"hub")
    runtime = _mapping(profile,"runtime")
    management = _mapping(profile,"management")
    wired = _mapping(management,"wired")
    wifi = _mapping(management,"wifi")
    storage = _mapping(profile,"storage")
    aliases = _mapping(profile,"hostname_aliases")
    values = (
        (profile,"hostname"),(operator,"user"),(operator,"management_host"),
        (operator,"tailscale_host"),(recovery,"user"),(rtx,"interface"),
        (sparkf,"interface"),(sparkf,"connection"),(sparkf,"ssh_target"),(sparkf,"ssh_bastion"),
        (fabric,"interface"),(fabric,"fabric_unit"),(fabric,"route_unit"),(fabric,"probe_ssh_target"),
        (hub,"release_directory"),(hub,"release_probe_path"),(hub,"heartbeat_directory"),(hub,"api_unit"),
    )
    for payload,key in values:
        _string(payload,key)
    for payload,key in ((wired,"interface"),(wired,"address_mode"),(wifi,"interface"),(wifi,"ssid"),(wifi,"address_mode"),(storage,"drafters_path"),(storage,"filesystem"),(aliases,"lan_address"),(aliases,"sparkf_address")):
        _string(payload,key)
    rtx_interface = ipaddress.ip_interface(_string(rtx,"address"))
    sparkf_interface = ipaddress.ip_interface(_string(sparkf,"address"))
    if rtx_interface.network != sparkf_interface.network:
        raise SpecNodeError("fleet-link addresses are not in the same network")
    if rtx_interface.network.prefixlen != 30 or rtx_interface.ip == sparkf_interface.ip:
        raise SpecNodeError("fleet link must contain two distinct addresses in a /30")
    fabric_network = ipaddress.ip_network(_string(fabric,"network"))
    if ipaddress.ip_address(_string(fabric,"sparkf_address")) not in fabric_network:
        raise SpecNodeError("sparkf fabric address is outside the fabric network")
    if fabric_network.overlaps(rtx_interface.network):
        raise SpecNodeError("fabric network must not overlap the fleet link")
    if link.get("mtu") != 9000:
        raise SpecNodeError("fleet link MTU must be 9000")
    if link.get("speed_mbps") != 10000:
        raise SpecNodeError("fleet link must be 10 GbE")
    _port(recovery,"port")
    if len({_port(hub,"release_http_port"),_port(hub,"api_port"),recovery["port"]}) != 3:
        raise SpecNodeError("hub release, API and recovery ports must differ")
    if runtime.get("state") != "infrastructure_only":
        raise SpecNodeError("profile must not claim an unverified speculation runtime")
    if runtime.get("transport") != "not_implemented":
        raise SpecNodeError("profile must expose the missing remote draft transport")
    if wired.get("address_mode") != "dhcp" or wifi.get("address_mode") != "dhcp":
        raise SpecNodeError("management interfaces must use DHCP")
    if wired.get("route_metric") != 100 or wifi.get("route_metric") != 300:
        raise SpecNodeError("management route metrics must prefer wired Ethernet")
    if storage.get("drafters_path") != "/srv/drafters" or storage.get("filesystem") != "ext4":
        raise SpecNodeError("drafter storage contract changed")
    if not isinstance(storage.get("minimum_available_gib"),int):
        raise SpecNodeError("invalid drafter storage capacity gate")
    if aliases.get("lan_address") != _string(operator,"management_host"):
        raise SpecNodeError("LAN hostname alias must match the management host")
    if aliases.get("sparkf_address") != str(rtx_interface.ip):
        raise SpecNodeError("sparkf hostname alias must use the fleet link")
    nodes = aliases.get("spark_nodes")
    if not isinstance(nodes,list) or len(nodes) != 16 or len(set(nodes)) != 16 or any(not isinstance(node,str) or SAFE_TOKEN.fullmatch(node) is None for node in nodes):
        raise SpecNodeError("hostname alias inventory must cover spark0 through sparkf")


def _link_addresses(profile: dict[str,object]) -> tuple[str,str]:
    link = _mapping(profile,"fleet_link")
    rtx = str(ipaddress.ip_interface(_string(_mapping(link,"rtx5090"),"address")).ip)
    sparkf = str(ipaddress.ip_interface(_string(_mapping(link,"sparkf"),"address")).ip)
    return(rtx,sparkf)


def render_netplan(profile: dict[str,object]) -> str:
    link = _mapping(profile,"fleet_link")
    endpoint = _mapping(link,"rtx5090")
    fabric = _mapping(link,"fabric")
    _,sparkf_ip = _link_addresses(profile)
    return(
        "network:\n"
        "  version: 2\n"
        "  ethernets:\n"
        f"    {_string(endpoint,'interface')}:\n"
        "      dhcp4: false\n"
        "      dhcp6: false\n"
        "      link-local: []\n"
        f"      addresses: [{_string(endpoint,'address')}]\n"
        f"      mtu: {link['mtu']}\n"
        "      routes:\n"
        f"        - to: {_string(fabric,'network')}\n"
        f"          via: {sparkf_ip}\n"
        "      optional: true\n"
    )


def render_emergency_sshd(profile: dict[str,object]) -> str:
    recovery = _mapping(profile,"recovery_ssh")
    user = _string(recovery,"user")
    root_policy = "prohibit-password" if user == "root" else "no"
    return(
        f"Port {recovery['port']}\n"
        "ListenAddress 0.0.0.0\n"
        "ListenAddress ::\n"
        "Protocol 2\n"
        "HostKey /etc/ssh/ssh_host_ed25519_key\n"
        "HostKey /etc/ssh/ssh_host_rsa_key\n"
        f"PermitRootLogin {root_policy}\n"
        "PasswordAuthentication no\n"
        "KbdInteractiveAuthentication no\n"
        "PubkeyAuthentication yes\n"
        "UsePAM yes\n"
        f"AllowUsers {user}\n"
        "AuthorizedKeysFile .ssh/authorized_keys\n"
        "Subsystem sftp internal-sftp\n"
    )


def render_emergency_unit() -> str:
    return(
        "[Unit]\n"
        "Description=DS4 emergency SSH\n"
        "After=network.target\n\n"
        "[Service]\n"
        "ExecStart=/usr/sbin/sshd -D -f /etc/ssh/sshd_config_ds4_emergency\n"
        "ExecReload=/bin/kill -HUP $MAINPID\n"
        "KillMode=process\n"
        "Restart=on-failure\n\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def render_sparkf_nmcli(profile: dict[str,object]) -> list[str]:
    link = _mapping(profile,"fleet_link")
    endpoint = _mapping(link,"sparkf")
    return([
        "sudo","nmcli","connection","modify",_string(endpoint,"connection"),
        "connection.interface-name",_string(endpoint,"interface"),
        "connection.autoconnect","yes",
        "connection.autoconnect-priority","300",
        "802-3-ethernet.auto-negotiate","yes",
        "802-3-ethernet.mtu",str(link["mtu"]),
        "ipv4.method","manual",
        "ipv4.addresses",_string(endpoint,"address"),
        "ipv4.gateway","",
        "ipv4.dns","",
        "ipv4.ignore-auto-routes","yes",
        "ipv4.ignore-auto-dns","yes",
        "ipv4.never-default","yes",
        "ipv6.method","disabled",
    ])


def render_sparkf_forwarding() -> str:
    return("net.ipv4.ip_forward = 1\n")


def render_hub_route_unit(profile: dict[str,object]) -> str:
    link = _mapping(profile,"fleet_link")
    fabric = _mapping(link,"fabric")
    network = str(ipaddress.ip_interface(_string(_mapping(link,"rtx5090"),"address")).network)
    route = f"{network} via {_string(fabric,'sparkf_address')} dev {_string(fabric,'interface')}"
    return(
        "[Unit]\n"
        f"Description=SparkPipe route to the rtx5090 hub ({network}) through sparkf on the switched fabric\n"
        f"After={_string(fabric,'fabric_unit')} network-online.target\n"
        "Wants=network-online.target\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        "RemainAfterExit=yes\n"
        f"ExecStart=/usr/sbin/ip route replace {route}\n"
        f"ExecStop=/usr/sbin/ip route del {route}\n\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def _run(argv: list[str]) -> str:
    result = subprocess.run(argv,capture_output=True,text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SpecNodeError(f"command failed ({result.returncode}): {detail}")
    return(result.stdout.strip())


def _ssh(
    target: str,
    command: str,
    bastion: str = "",
    host_key_alias: str = "",
) -> str:
    argv = ["ssh","-o","BatchMode=yes"]
    if bastion != "":
        argv.extend(["-J",bastion])
    if host_key_alias != "":
        argv.extend(["-o",f"HostKeyAlias={host_key_alias}"])
    argv.extend([target,command])
    return(_run(argv))


def _require_contains(results: dict[str,str],key: str,values: tuple[str,...]) -> None:
    missing = [value for value in values if value not in results[key]]
    if missing:
        raise SpecNodeError(f"{key} missing expected values: {', '.join(missing)}")


def verify_live(profile: dict[str,object]) -> dict[str,str]:
    operator = _mapping(profile,"operator")
    link = _mapping(profile,"fleet_link")
    rtx = _mapping(link,"rtx5090")
    sparkf = _mapping(link,"sparkf")
    fabric = _mapping(link,"fabric")
    hub = _mapping(profile,"hub")
    recovery = _mapping(profile,"recovery_ssh")
    management = _mapping(profile,"management")
    wired = _mapping(management,"wired")
    wifi = _mapping(management,"wifi")
    storage = _mapping(profile,"storage")
    hostname = _string(profile,"hostname")
    management_host = _string(operator,"management_host")
    user = _string(operator,"user")
    target = f"{user}@{management_host}"
    rtx_if = _string(rtx,"interface")
    sparkf_if = _string(sparkf,"interface")
    rtx_ip,sparkf_ip = _link_addresses(profile)
    sparkf_target = _string(sparkf,"ssh_target")
    probe_target = _string(fabric,"probe_ssh_target")
    bastion = _string(sparkf,"ssh_bastion")
    release_url = f"http://{rtx_ip}:{hub['release_http_port']}/{_string(hub,'release_probe_path')}"
    results = {
        "rtx_hostname": _ssh(target,"hostname"),
        "rtx_gpu": _ssh(target,"nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
        "rtx_driver_license": _ssh(target,"modinfo -F license nvidia"),
        "rtx_secure_boot": _ssh(target,"mokutil --sb-state"),
        "rtx_management": _ssh(target,f"ip -4 -o address show dev {_string(wired,'interface')}; ip -4 route show default"),
        "rtx_wifi": _ssh(target,f"iw dev {_string(wifi,'interface')} link; ip -4 -o address show dev {_string(wifi,'interface')}"),
        "rtx_storage": _ssh(target,f"findmnt -T {_string(storage,'drafters_path')} -n -o TARGET,FSTYPE; df -BG --output=avail {_string(storage,'drafters_path')} | tail -n 1; stat -c %U:%G {_string(storage,'drafters_path')}"),
        "rtx_link": _ssh(target,f"ip -4 -o address show dev {rtx_if}; cat /sys/class/net/{rtx_if}/speed"),
        "rtx_fabric_route": _ssh(target,f"ip -4 route show {_string(fabric,'network')}"),
        "rtx_tailscale": _ssh(target,"systemctl is-enabled tailscaled; systemctl is-active tailscaled"),
        "hub_api": _ssh(target,f"systemctl --user is-active {_string(hub,'api_unit')}"),
        "hub_heartbeats": _ssh(target,f"ls {_string(hub,'heartbeat_directory')}"),
        "tailscale_ssh": _ssh(
            f"{user}@{_string(operator,'tailscale_host')}","hostname",
            host_key_alias=management_host,
        ),
        "rtx_peer": _ssh(target,f"ping -c 2 -W 2 -M do -s 8972 {sparkf_ip}"),
        "sparkf_link": _ssh(sparkf_target,f"ip -4 -o address show dev {sparkf_if}; cat /sys/class/net/{sparkf_if}/speed",bastion),
        "sparkf_peer": _ssh(sparkf_target,f"ping -c 2 -W 2 -M do -s 8972 {rtx_ip}",bastion),
        "sparkf_forwarding": _ssh(sparkf_target,"sysctl -n net.ipv4.ip_forward",bastion),
        "spark_hub_route": _ssh(probe_target,f"systemctl is-active {_string(fabric,'route_unit')}; ip -4 route get {rtx_ip}",bastion),
        "spark_release": _ssh(probe_target,f"curl -sf --max-time 5 -o /dev/null -w '%{{http_code}}' {release_url}",bastion),
        "recovery_ssh": _run([
            "ssh","-o","BatchMode=yes","-o","ConnectTimeout=5",
            "-o",f"HostKeyAlias={management_host}",
            "-p",str(recovery["port"]),
            f"{_string(recovery,'user')}@{management_host}",
            "hostname",
        ]),
    }
    if results["rtx_hostname"] != hostname:
        raise SpecNodeError(f"unexpected RTX hostname: {results['rtx_hostname']}")
    if results["tailscale_ssh"] != hostname or results["recovery_ssh"] != hostname:
        raise SpecNodeError("management paths did not reach the expected host")
    _require_contains(results,"rtx_gpu",(_text(_mapping(profile,"gpu"),"model"),))
    _require_contains(results,"rtx_driver_license",("Dual MIT/GPL",))
    _require_contains(results,"rtx_secure_boot",("SecureBoot enabled",))
    _require_contains(results,"rtx_management",(_string(wired,"interface"),f"metric {wired['route_metric']}",_string(wifi,"interface"),f"metric {wifi['route_metric']}"))
    _require_contains(results,"rtx_wifi",(f"SSID: {_string(wifi,'ssid')}",_string(wifi,"interface")))
    _require_contains(results,"rtx_storage",(_string(storage,"drafters_path"),_string(storage,"filesystem"),"spec:spec"))
    available = [line.strip() for line in results["rtx_storage"].splitlines() if line.strip().endswith("G")]
    if len(available) != 1 or int(available[0][:-1]) < storage["minimum_available_gib"]:
        raise SpecNodeError("drafter storage below the configured capacity gate")
    speed = str(link["speed_mbps"])
    if results["rtx_link"].splitlines()[-1:] != [speed]:
        raise SpecNodeError(f"rtx_link speed is not {speed} Mb/s")
    _require_contains(results,"rtx_link",(_string(rtx,"address"),))
    _require_contains(results,"rtx_fabric_route",(f"via {sparkf_ip}",))
    if results["sparkf_link"].splitlines()[-1:] != [speed]:
        raise SpecNodeError(f"sparkf_link speed is not {speed} Mb/s")
    _require_contains(results,"sparkf_link",(_string(sparkf,"address"),))
    if results["sparkf_forwarding"] != "1":
        raise SpecNodeError("sparkf does not forward IPv4 between the fabric and the fleet link")
    _require_contains(results,"spark_hub_route",("active",f"via {_string(fabric,'sparkf_address')} dev {_string(fabric,'interface')}"))
    if results["spark_release"] != "200":
        raise SpecNodeError(f"hub release endpoint {release_url} answered {results['spark_release']!r} over the fabric route")
    if results["hub_api"] != "active":
        raise SpecNodeError(f"hub API unit {_string(hub,'api_unit')} is {results['hub_api']!r}")
    beats = set(results["hub_heartbeats"].split())
    missing = [node for node in _mapping(profile,"hostname_aliases")["spark_nodes"] if f"{node}.json" not in beats]
    if missing:
        raise SpecNodeError(f"hub heartbeats missing for {', '.join(missing)}")
    _require_contains(results,"rtx_tailscale",("enabled","active"))
    return(results)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",type=Path,
        default=repo / "deployment/rtx5090_speculation/node.local.json",
    )
    parser.add_argument("command",choices=("plan","render-netplan","render-emergency","verify"))
    return(parser.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
        profile = load_profile(arguments.profile)
        if arguments.command == "plan":
            print(json.dumps({
                "netplan": render_netplan(profile),
                "emergency_sshd": render_emergency_sshd(profile),
                "emergency_unit": render_emergency_unit(),
                "sparkf_nmcli": render_sparkf_nmcli(profile),
                "sparkf_forwarding": render_sparkf_forwarding(),
                "spark_hub_route_unit": render_hub_route_unit(profile),
            },indent=2,sort_keys=True))
        elif arguments.command == "render-netplan":
            print(render_netplan(profile),end="")
        elif arguments.command == "render-emergency":
            print(render_emergency_sshd(profile),end="")
        else:
            print(json.dumps(verify_live(profile),indent=2,sort_keys=True))
        return(0)
    except (OSError,SpecNodeError,ValueError) as error:
        print(f"rtx5090_spec_node: {error}",file=sys.stderr)
        return(1)


if __name__ == "__main__":
    raise SystemExit(main())

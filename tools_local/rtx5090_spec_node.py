#!/usr/bin/env python3
"""Render and verify the RTX 5090 speculation-node infrastructure contract."""

from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import sys


FORMAT = "ds4-auxiliary-node-v1"
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


def validate_profile(profile: dict[str,object]) -> None:
    if profile.get("role") != "speculation" or profile.get("spark_rank") is not None:
        raise SpecNodeError("speculation node must not be assigned a Spark rank")
    operator = _mapping(profile,"operator")
    recovery = _mapping(profile,"recovery_ssh")
    private_link = _mapping(profile,"private_link")
    rtx = _mapping(private_link,"rtx5090")
    sparkf = _mapping(private_link,"sparkf")
    runtime = _mapping(profile,"runtime")
    management = _mapping(profile,"management")
    wired = _mapping(management,"wired")
    wifi = _mapping(management,"wifi")
    storage = _mapping(profile,"storage")
    aliases = _mapping(profile,"hostname_aliases")
    values = (
        (profile,"hostname"),(operator,"user"),(operator,"management_host"),
        (operator,"tailscale_host"),(recovery,"user"),(rtx,"interface"),
        (sparkf,"interface"),(sparkf,"ssh_target"),(sparkf,"ssh_bastion"),
    )
    for payload,key in values:
        _string(payload,key)
    for payload,key in ((wired,"interface"),(wired,"address_mode"),(wifi,"interface"),(wifi,"ssid"),(wifi,"address_mode"),(storage,"drafters_path"),(storage,"filesystem"),(aliases,"lan_address"),(aliases,"sparkf_address")):
        _string(payload,key)
    rtx_interface = ipaddress.ip_interface(_string(rtx,"address"))
    sparkf_interface = ipaddress.ip_interface(_string(sparkf,"address"))
    if rtx_interface.network != sparkf_interface.network:
        raise SpecNodeError("private-link addresses are not in the same network")
    if rtx_interface.network.prefixlen != 30 or rtx_interface.ip == sparkf_interface.ip:
        raise SpecNodeError("private link must contain two distinct addresses in a /30")
    if private_link.get("mtu") != 9000:
        raise SpecNodeError("private link MTU must be 9000")
    port = recovery.get("port")
    if not isinstance(port,int) or isinstance(port,bool) or port < 1 or port > 65535:
        raise SpecNodeError("recovery SSH port must be in 1..65535")
    if runtime.get("state") != "infrastructure_only":
        raise SpecNodeError("profile must not claim an unverified speculation runtime")
    if runtime.get("transport") != "not_implemented":
        raise SpecNodeError("profile must expose the missing remote transport")
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
        raise SpecNodeError("sparkf hostname alias must use the private link")
    nodes = aliases.get("spark_nodes")
    if not isinstance(nodes,list) or len(nodes) != 16 or any(not isinstance(node,str) for node in nodes):
        raise SpecNodeError("hostname alias inventory must cover spark0 through sparkf")


def render_netplan(profile: dict[str,object]) -> str:
    private_link = _mapping(profile,"private_link")
    endpoint = _mapping(private_link,"rtx5090")
    interface = _string(endpoint,"interface")
    address = _string(endpoint,"address")
    mtu = private_link["mtu"]
    return(
        "network:\n"
        "  version: 2\n"
        "  ethernets:\n"
        f"    {interface}:\n"
        "      dhcp4: false\n"
        "      dhcp6: false\n"
        "      link-local: []\n"
        f"      addresses: [{address}]\n"
        f"      mtu: {mtu}\n"
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
    private_link = _mapping(profile,"private_link")
    endpoint = _mapping(private_link,"sparkf")
    return([
        "sudo","nmcli","connection","modify","ds4-uplink-wired",
        "connection.id","ds4-speculation-link",
        "connection.interface-name",_string(endpoint,"interface"),
        "connection.autoconnect","yes",
        "connection.autoconnect-priority","300",
        "802-3-ethernet.auto-negotiate","yes",
        "802-3-ethernet.mtu",str(private_link["mtu"]),
        "ipv4.method","manual",
        "ipv4.addresses",_string(endpoint,"address"),
        "ipv4.gateway","",
        "ipv4.dns","",
        "ipv4.ignore-auto-routes","yes",
        "ipv4.ignore-auto-dns","yes",
        "ipv4.never-default","yes",
        "ipv6.method","disabled",
    ])


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
    private_link = _mapping(profile,"private_link")
    rtx = _mapping(private_link,"rtx5090")
    sparkf = _mapping(private_link,"sparkf")
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
    rtx_peer = str(ipaddress.ip_interface(_string(sparkf,"address")).ip)
    sparkf_peer = str(ipaddress.ip_interface(_string(rtx,"address")).ip)
    sparkf_target = _string(sparkf,"ssh_target")
    bastion = _string(sparkf,"ssh_bastion")
    results = {
        "rtx_hostname": _ssh(target,"hostname"),
        "rtx_gpu": _ssh(target,"nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
        "rtx_driver_license": _ssh(target,"modinfo -F license nvidia"),
        "rtx_secure_boot": _ssh(target,"mokutil --sb-state"),
        "rtx_management": _ssh(target,f"ip -4 -o address show dev {_string(wired,'interface')}; ip -4 route show default"),
        "rtx_wifi": _ssh(target,f"iw dev {_string(wifi,'interface')} link; ip -4 -o address show dev {_string(wifi,'interface')}"),
        "rtx_storage": _ssh(target,f"findmnt -T {_string(storage,'drafters_path')} -n -o TARGET,FSTYPE; df -BG --output=avail {_string(storage,'drafters_path')} | tail -n 1; stat -c %U:%G {_string(storage,'drafters_path')}"),
        "rtx_link": _ssh(target,f"ip -4 -o address show dev {rtx_if}; cat /sys/class/net/{rtx_if}/speed"),
        "rtx_tailscale": _ssh(target,"systemctl is-enabled tailscaled; systemctl is-active tailscaled"),
        "tailscale_ssh": _ssh(
            f"{user}@{_string(operator,'tailscale_host')}","hostname",
            host_key_alias=management_host,
        ),
        "rtx_peer": _ssh(target,f"ping -c 2 -W 2 -M do -s 8972 {rtx_peer}"),
        "sparkf_link": _ssh(sparkf_target,f"ip -4 -o address show dev {sparkf_if}; cat /sys/class/net/{sparkf_if}/speed",bastion),
        "sparkf_peer": _ssh(sparkf_target,f"ping -c 2 -W 2 -M do -s 8972 {sparkf_peer}",bastion),
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
    _require_contains(results,"rtx_link",(_string(rtx,"address").split("/",1)[0],"10000"))
    _require_contains(results,"sparkf_link",(_string(sparkf,"address").split("/",1)[0],"10000"))
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

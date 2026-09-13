#!/usr/bin/env python3
"""Install and control the generic DS4 ARM64 PXE login rescue service."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SCRIPT = "tools/devcycle/ds4_parallel_pxe_rescue.py"
DEFAULT_SERVER = "spark0"
DEFAULT_INTERFACE = "enP7s7"
DEFAULT_SERVER_IP = "192.168.50.128"
DEFAULT_ROOT_DEVICE = "/dev/nvme0n1p2"
DEFAULT_RECOVERY_IDENTITY = Path.home() / ".ssh" / "sparkpipe_fleet_root"
DEFAULT_PROBE_NODES = ("spark0","spark2","spark3","spark4","spark5","spark6","spark7")
REMOTE_STAGE = "/tmp/ds4_parallel_pxe_rescue.py"
INSTALLED_SCRIPT = Path("/usr/local/sbin/ds4-parallel-pxe-rescue")
CONFIG_PATH = Path("/etc/ds4-pxe-rescue/config.json")
STATE_DIR = Path("/var/lib/ds4-pxe-rescue")
SERVICE_PATH = Path("/etc/systemd/system/ds4-parallel-pxe-rescue.service")
SERVICE_NAME = SERVICE_PATH.name
FIREWALL_FAMILY = "inet"
FIREWALL_TABLE = "spark_guard"
FIREWALL_CHAIN = "input"
FIREWALL_COMMENT = "ds4-parallel-pxe-rescue"
ARM64_UEFI_ARCH = 11
SHIM_SOURCE = Path("/usr/lib/shim/shimaa64.efi.signed.latest")
GRUB_SOURCE = Path("/usr/lib/grub/arm64-efi-signed/grubnetaa64.efi.signed")
MOK_SOURCE = Path("/usr/lib/shim/mmaa64.efi")
GRUB_CONFIG_NAMES = ("grub.cfg","grub/grub.cfg")
EFFECTIVE_GRUB_CONFIG = STATE_DIR / "grub" / "grub.cfg"
RECOVERY_KEY_SOURCE = Path("/etc/ds4-pxe-rescue/recovery_key.pub")
RECOVERY_LOCAL_BOTTOM_SOURCE = Path("/etc/ds4-pxe-rescue/ds4-recovery-key")
CONFIG_FORMAT = "ds4-parallel-pxe-rescue-v2"
MANIFEST_FORMAT = "ds4-parallel-pxe-rescue-manifest-v2"
RECOVERY_INITRAMFS_HOOK = """#!/bin/sh
PREREQ=""

prereqs()
{
    echo "$PREREQ"
}

case "$1" in
prereqs)
    prereqs
    exit 0
    ;;
esac

mkdir -p "${DESTDIR}/etc/ds4-rescue" "${DESTDIR}/scripts/local-bottom"
cp /etc/ds4-pxe-rescue/recovery_key.pub "${DESTDIR}/etc/ds4-rescue/recovery_key.pub"
cp /etc/ds4-pxe-rescue/ds4-recovery-key "${DESTDIR}/scripts/local-bottom/ds4-recovery-key"
chmod 0600 "${DESTDIR}/etc/ds4-rescue/recovery_key.pub"
chmod 0755 "${DESTDIR}/scripts/local-bottom/ds4-recovery-key"
"""
RECOVERY_LOCAL_BOTTOM = """#!/bin/sh
PREREQ=""

prereqs()
{
    echo "$PREREQ"
}

case "$1" in
prereqs)
    prereqs
    exit 0
    ;;
esac

key_file=/etc/ds4-rescue/recovery_key.pub
target=/root/root/.ssh/authorized_keys
if [ ! -f "$key_file" ] || [ ! -d /root ]; then
    exit 0
fi
mount -o remount,rw /root
mkdir -p /root/root/.ssh
chmod 0700 /root/root/.ssh
key="$(cat "$key_file")"
if [ -f "$target" ] && grep -Fqx "$key" "$target"; then
    exit 0
fi
temporary="${target}.ds4-rescue.$$"
cat "$key_file" > "$temporary"
printf '\n' >> "$temporary"
if [ -f "$target" ]; then
    cat "$target" >> "$temporary"
fi
chmod 0600 "$temporary"
mv "$temporary" "$target"
"""


class PxeRescueError(RuntimeError):
    pass


def run(
    argv: list[str],
    *,
    input_bytes: bytes | None = None,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(argv,input=input_bytes,capture_output=True,timeout=timeout)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8",errors="replace").strip()
        raise PxeRescueError(f"command failed ({result.returncode}): {' '.join(argv)}: {detail}")
    return(result)


def command(argv: list[str],timeout: int = 120,check: bool = True) -> str:
    return(run(argv,timeout=timeout,check=check).stdout.decode("utf-8",errors="replace").strip())


def atomic_write_bytes(path: Path,data: bytes,mode: int) -> bool:
    if path.exists() and path.read_bytes() == data and (path.stat().st_mode & 0o777) == mode:
        return(False)
    path.parent.mkdir(parents=True,exist_ok=True)
    handle,temp_name = tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(handle,"wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temp_name,mode)
        os.replace(temp_name,path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return(True)


def atomic_write(path: Path,text: str,mode: int) -> bool:
    return(atomic_write_bytes(path,text.encode("utf-8"),mode))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return(digest.hexdigest())


def atomic_copy(source: Path,target: Path,mode: int) -> bool:
    source_digest = sha256_file(source)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        if sha256_file(target) == source_digest and (target.stat().st_mode & 0o777) == mode:
            return(False)
    target.parent.mkdir(parents=True,exist_ok=True)
    handle,temp_name = tempfile.mkstemp(prefix=f".{target.name}.",dir=target.parent)
    try:
        with source.open("rb") as input_file,os.fdopen(handle,"wb") as output_file:
            shutil.copyfileobj(input_file,output_file,8 * 1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.chmod(temp_name,mode)
        os.replace(temp_name,target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return(True)


def validate_interface(interface: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.:-]{1,32}",interface) is None:
        raise PxeRescueError(f"invalid PXE interface: {interface}")
    return(interface)


def validate_server_ip(server_ip: str) -> str:
    try:
        address = ipaddress.ip_address(server_ip)
    except ValueError as error:
        raise PxeRescueError(f"invalid PXE server address: {server_ip}") from error
    if address.version != 4:
        raise PxeRescueError("PXE server address must be IPv4")
    return(str(address))


def validate_root_device(root_device: str) -> str:
    if re.fullmatch(r"/dev/[A-Za-z0-9._/-]+",root_device) is None or ".." in root_device:
        raise PxeRescueError(f"invalid rescue root device: {root_device}")
    return(root_device)


def validate_recovery_public_key(public_key: str) -> str:
    key = public_key.strip()
    if re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/]+={0,3}(?: [^\r\n]+)?",key) is None:
        raise PxeRescueError("invalid fleet recovery public key")
    return(key)


def validated_config(payload: dict[str,object]) -> dict[str,str]:
    if payload.get("format") != CONFIG_FORMAT:
        raise PxeRescueError("unsupported PXE rescue config format")
    return({
        "format":CONFIG_FORMAT,
        "interface":validate_interface(str(payload.get("interface",""))),
        "recovery_public_key":validate_recovery_public_key(str(payload.get("recovery_public_key",""))),
        "root_device":validate_root_device(str(payload.get("root_device",""))),
        "server_ip":validate_server_ip(str(payload.get("server_ip",""))),
        "source_commit":str(payload.get("source_commit","unknown")),
    })


def dnsmasq_config(config: dict[str,str]) -> str:
    interface = config["interface"]
    server_ip = config["server_ip"]
    return(f"""port=0
log-dhcp
interface={interface}
bind-dynamic
dhcp-range=192.168.50.0,proxy,255.255.255.0
dhcp-match=set:efi-arm64,option:client-arch,{ARM64_UEFI_ARCH}
dhcp-boot=tag:efi-arm64,shimaa64.efi,,{server_ip}
pxe-service=tag:efi-arm64,ARM64_EFI,"DS4 Spark login rescue",shimaa64.efi,{server_ip}
pxe-prompt=tag:efi-arm64,"DS4 Spark login rescue",0
enable-tftp
tftp-root={STATE_DIR}
""")


def grub_config(config: dict[str,str]) -> str:
    interface = config["interface"]
    root_device = config["root_device"]
    return(f"""set timeout=0
set default=0

menuentry 'DS4 Spark login rescue' {{
    linux /vmlinuz root={root_device} rw fsck.mode=skip fsck.repair=no systemd.mask=ds4-switched-fabric.service systemd.mask=ds4-direct-pair-fabric.service systemd.unit=multi-user.target console=tty0 console=ttyS0,921600 ip=:::::{interface}:dhcp
    initrd /initrd.img
}}
""")


def service_config() -> str:
    return(f"""[Unit]
Description=DS4 parallel PXE login rescue
After=network.target
Wants=network.target

[Service]
Type=simple
ExecStartPre={INSTALLED_SCRIPT} --remote-preflight
ExecStartPre={INSTALLED_SCRIPT} --remote-firewall-open
ExecStart=/usr/sbin/dnsmasq --keep-in-foreground --log-facility=- --conf-file={STATE_DIR}/dnsmasq.conf
ExecStopPost={INSTALLED_SCRIPT} --remote-firewall-close
Restart=on-failure
RestartSec=2s
TimeoutStartSec=30
TimeoutStopSec=15
""")


def require_root() -> None:
    if os.geteuid() != 0:
        raise PxeRescueError("remote PXE operation must run as root")


def load_config() -> dict[str,str]:
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as error:
        raise PxeRescueError(f"cannot read {CONFIG_PATH}: {error}") from error
    if not isinstance(payload,dict):
        raise PxeRescueError("PXE rescue config is not an object")
    return(validated_config(payload))


def interface_has_address(interface: str,address: str) -> bool:
    output = command(["ip","-j","address","show","dev",interface])
    try:
        documents = json.loads(output)
    except json.JSONDecodeError as error:
        raise PxeRescueError("ip produced invalid JSON") from error
    return(any(
        item.get("local") == address
        for document in documents
        for item in document.get("addr_info",[])
        if isinstance(item,dict)
    ))


def resolved_boot_assets() -> dict[str,Path]:
    sources = {
        "shimaa64.efi":SHIM_SOURCE,
        "grubaa64.efi":GRUB_SOURCE,
        "mmaa64.efi":MOK_SOURCE,
        "vmlinuz":Path(command(["readlink","-f","/boot/vmlinuz"])),
        "initrd.img":Path(command(["readlink","-f","/boot/initrd.img"])),
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise PxeRescueError(f"missing PXE boot assets: {missing}")
    return(sources)


def recovery_public_key(identity: Path) -> str:
    if not identity.is_file():
        raise PxeRescueError(f"missing recovery identity: {identity}")
    return(validate_recovery_public_key(command(["ssh-keygen","-y","-f",str(identity)])))


def install_rescue_initrd(config: dict[str,str],target: Path) -> bool:
    kernel_release = command(["uname","-r"])
    atomic_write(RECOVERY_KEY_SOURCE,config["recovery_public_key"] + "\n",0o600)
    atomic_write(RECOVERY_LOCAL_BOTTOM_SOURCE,RECOVERY_LOCAL_BOTTOM,0o755)
    with tempfile.TemporaryDirectory(prefix="ds4-pxe-initramfs.") as directory:
        work = Path(directory)
        config_dir = work / "config"
        shutil.copytree("/etc/initramfs-tools",config_dir,symlinks=True)
        atomic_write(config_dir / "hooks" / "ds4-recovery-key",RECOVERY_INITRAMFS_HOOK,0o755)
        output = work / "initrd.img"
        run(["mkinitramfs","-d",str(config_dir),"-o",str(output),kernel_release],timeout=600)
        listing = command(["lsinitramfs",str(output)],timeout=180)
        for name in ("etc/ds4-rescue/recovery_key.pub","scripts/local-bottom/ds4-recovery-key"):
            if name not in listing.splitlines():
                raise PxeRescueError(f"generated rescue initramfs is missing {name}")
        return(atomic_copy(output,target,0o644))


def build_manifest(config: dict[str,str],sources: dict[str,Path]) -> dict[str,object]:
    files = {}
    for name,source in sources.items():
        target = STATE_DIR / name
        source_name = f"generated:mkinitramfs:{os.uname().release}" if name == "initrd.img" else str(source)
        files[name] = {
            "bytes":target.stat().st_size,
            "sha256":sha256_file(target),
            "source":source_name,
        }
    for name in ("dnsmasq.conf",*GRUB_CONFIG_NAMES):
        target = STATE_DIR / name
        files[name] = {"bytes":target.stat().st_size,"sha256":sha256_file(target)}
    return({
        "config":config,
        "files":files,
        "format":MANIFEST_FORMAT,
        "generated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })


def remote_install(config: dict[str,str]) -> dict[str,object]:
    require_root()
    config = validated_config(config)
    if not interface_has_address(config["interface"],config["server_ip"]):
        raise PxeRescueError(f"{config['server_ip']} is not assigned to {config['interface']}")
    for executable in ("dnsmasq","nft","systemctl","mkinitramfs","lsinitramfs"):
        if shutil.which(executable) is None:
            raise PxeRescueError(f"required executable is missing: {executable}")
    sources = resolved_boot_assets()
    changed = []
    if atomic_write_bytes(INSTALLED_SCRIPT,Path(__file__).read_bytes(),0o755):
        changed.append(str(INSTALLED_SCRIPT))
    if atomic_write(CONFIG_PATH,json.dumps(config,indent=2,sort_keys=True) + "\n",0o600):
        changed.append(str(CONFIG_PATH))
    for name,source in sources.items():
        if name == "initrd.img":
            continue
        if atomic_copy(source,STATE_DIR / name,0o644):
            changed.append(str(STATE_DIR / name))
    if install_rescue_initrd(config,STATE_DIR / "initrd.img"):
        changed.append(str(STATE_DIR / "initrd.img"))
    if atomic_write(STATE_DIR / "dnsmasq.conf",dnsmasq_config(config),0o644):
        changed.append(str(STATE_DIR / "dnsmasq.conf"))
    for name in GRUB_CONFIG_NAMES:
        path = STATE_DIR / name
        if atomic_write(path,grub_config(config),0o644):
            changed.append(str(path))
    manifest = build_manifest(config,sources)
    if atomic_write(STATE_DIR / "manifest.json",json.dumps(manifest,indent=2,sort_keys=True) + "\n",0o644):
        changed.append(str(STATE_DIR / "manifest.json"))
    if atomic_write(SERVICE_PATH,service_config(),0o644):
        changed.append(str(SERVICE_PATH))
    command(["dnsmasq","--test",f"--conf-file={STATE_DIR / 'dnsmasq.conf'}"])
    command(["systemd-analyze","verify",str(SERVICE_PATH)])
    command(["systemctl","daemon-reload"])
    scrub_legacy_dhcp_rules(config["interface"])
    return({"changed":changed,"config":config,"manifest":manifest})


def nft_chain() -> list[dict[str,object]]:
    output = command(["nft","-j","-a","list","chain",FIREWALL_FAMILY,FIREWALL_TABLE,FIREWALL_CHAIN])
    try:
        document = json.loads(output)
    except json.JSONDecodeError as error:
        raise PxeRescueError("nft produced invalid JSON") from error
    entries = document.get("nftables")
    if not isinstance(entries,list):
        raise PxeRescueError("nft chain document is invalid")
    return([entry["rule"] for entry in entries if isinstance(entry,dict) and isinstance(entry.get("rule"),dict)])


def match_field(expression: object,protocol: str,field: str,value: object) -> bool:
    if not isinstance(expression,dict) or not isinstance(expression.get("match"),dict):
        return(False)
    match = expression["match"]
    left = match.get("left")
    if not isinstance(left,dict):
        return(False)
    payload = left.get("payload")
    return(isinstance(payload,dict) and payload.get("protocol") == protocol and payload.get("field") == field and match.get("right") == value)


def match_interface(expression: object,interface: str) -> bool:
    if not isinstance(expression,dict) or not isinstance(expression.get("match"),dict):
        return(False)
    match = expression["match"]
    return(match.get("left") == {"meta":{"key":"iifname"}} and match.get("right") == interface)


def invalid_drop_handle(rules: list[dict[str,object]]) -> int:
    for rule in rules:
        expressions = rule.get("expr",[])
        if not isinstance(expressions,list):
            continue
        has_invalid = any(
            isinstance(item,dict)
            and isinstance(item.get("match"),dict)
            and item["match"].get("left") == {"ct":{"key":"state"}}
            and item["match"].get("right") == "invalid"
            for item in expressions
        )
        if has_invalid and any(isinstance(item,dict) and "drop" in item for item in expressions):
            return(int(rule["handle"]))
    raise PxeRescueError("spark_guard input has no invalid-state drop rule")


def legacy_exact_dhcp_rule(rule: dict[str,object],interface: str) -> bool:
    expressions = rule.get("expr",[])
    if not isinstance(expressions,list):
        return(False)
    has_source_mac = any(
        isinstance(item,dict)
        and isinstance(item.get("match"),dict)
        and item["match"].get("left") == {"payload":{"protocol":"ether","field":"saddr"}}
        for item in expressions
    )
    return(
        any(match_interface(item,interface) for item in expressions)
        and has_source_mac
        and any(match_field(item,"udp","sport",68) for item in expressions)
        and any(match_field(item,"udp","dport",67) for item in expressions)
        and any(isinstance(item,dict) and "accept" in item for item in expressions)
    )


def delete_rule(handle: int) -> None:
    command(["nft","delete","rule",FIREWALL_FAMILY,FIREWALL_TABLE,FIREWALL_CHAIN,"handle",str(handle)])


def rescue_firewall_handles(rules: list[dict[str,object]]) -> list[int]:
    return([int(rule["handle"]) for rule in rules if rule.get("comment") == FIREWALL_COMMENT])


def scrub_legacy_dhcp_rules(interface: str) -> None:
    for rule in nft_chain():
        if legacy_exact_dhcp_rule(rule,interface):
            delete_rule(int(rule["handle"]))


def remote_firewall_close() -> dict[str,object]:
    require_root()
    removed = rescue_firewall_handles(nft_chain())
    for handle in removed:
        delete_rule(handle)
    return({"removed_handles":removed})


def remote_firewall_open() -> dict[str,object]:
    require_root()
    config = load_config()
    remote_firewall_close()
    rules = nft_chain()
    position = invalid_drop_handle(rules)
    command([
        "nft","insert","rule",FIREWALL_FAMILY,FIREWALL_TABLE,FIREWALL_CHAIN,
        "position",str(position),"iifname",config["interface"],"udp","sport","68",
        "udp","dport","67","counter","accept","comment",FIREWALL_COMMENT,
    ])
    handles = rescue_firewall_handles(nft_chain())
    if len(handles) != 1:
        raise PxeRescueError(f"expected one PXE firewall rule, found {len(handles)}")
    return({"firewall_handles":handles})


def verify_manifest() -> list[str]:
    failures = []
    try:
        manifest = json.loads((STATE_DIR / "manifest.json").read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as error:
        return([f"manifest:{error}"])
    if manifest.get("format") != MANIFEST_FORMAT:
        failures.append("manifest-format")
    files = manifest.get("files")
    if not isinstance(files,dict):
        return([*failures,"manifest-files"])
    for name,specification in files.items():
        path = STATE_DIR / str(name)
        if not path.is_file():
            failures.append(f"missing:{name}")
            continue
        if not isinstance(specification,dict):
            failures.append(f"manifest-entry:{name}")
            continue
        if path.stat().st_size != specification.get("bytes") or sha256_file(path) != specification.get("sha256"):
            failures.append(f"hash:{name}")
    return(failures)


def remote_status(require_active: bool = False) -> dict[str,object]:
    require_root()
    config = load_config()
    failures = verify_manifest()
    dnsmasq = (STATE_DIR / "dnsmasq.conf").read_text(encoding="utf-8")
    grub_files = [(STATE_DIR / name).read_text(encoding="utf-8") for name in GRUB_CONFIG_NAMES]
    grub = grub_files[0]
    if any(text != grub for text in grub_files[1:]):
        failures.append("grub-config-drift")
    if "dhcp-host=" in dnsmasq or "dhcp-ignore=" in dnsmasq:
        failures.append("identity-restricted-dnsmasq")
    if f"option:client-arch,{ARM64_UEFI_ARCH}" not in dnsmasq:
        failures.append("arm64-match")
    recovery_key = RECOVERY_KEY_SOURCE.read_text(encoding="utf-8").strip() if RECOVERY_KEY_SOURCE.is_file() else ""
    if recovery_key != config["recovery_public_key"]:
        failures.append("recovery-public-key")
    initrd_listing = command(["lsinitramfs",str(STATE_DIR / "initrd.img")],timeout=180,check=False)
    for name in ("etc/ds4-rescue/recovery_key.pub","scripts/local-bottom/ds4-recovery-key"):
        if name not in initrd_listing.splitlines():
            failures.append(f"initrd:{name}")
    for token in (
        f"root={config['root_device']}",
        "systemd.mask=ds4-switched-fabric.service",
        "systemd.mask=ds4-direct-pair-fabric.service",
        f"ip=:::::{config['interface']}:dhcp",
    ):
        if token not in grub:
            failures.append(f"grub:{token}")
    if not interface_has_address(config["interface"],config["server_ip"]):
        failures.append("server-address")
    active = command(["systemctl","is-active",SERVICE_NAME],check=False)
    enabled = command(["systemctl","is-enabled",SERVICE_NAME],check=False)
    handles = rescue_firewall_handles(nft_chain())
    listeners = command(["ss","-lunp"],check=False)
    if require_active and active != "active":
        failures.append(f"service:{active}")
    if active == "active":
        if len(handles) != 1:
            failures.append(f"firewall-rules:{len(handles)}")
        for port in (67,69):
            if re.search(rf":{port}\b",listeners) is None:
                failures.append(f"listener:{port}")
    elif handles:
        failures.append("inactive-firewall-rule")
    if enabled not in ("disabled","static"):
        failures.append(f"unexpected-enable-state:{enabled}")
    return({
        "active":active,
        "config":config,
        "enabled":enabled,
        "failures":failures,
        "firewall_handles":handles,
        "listeners":listeners,
    })


def remote_preflight() -> dict[str,object]:
    require_root()
    config = load_config()
    command(["dnsmasq","--test",f"--conf-file={STATE_DIR / 'dnsmasq.conf'}"])
    status = remote_status(require_active=False)
    allowed = {"service:activating","service:inactive"}
    failures = [failure for failure in status["failures"] if failure not in allowed]
    if failures:
        raise PxeRescueError(f"PXE preflight failed: {failures}")
    return({"config":config,"failures":[]})


def ssh(server: str,*argv: str,input_bytes: bytes | None = None,timeout: int = 120) -> subprocess.CompletedProcess[bytes]:
    return(run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=8",server,*argv],input_bytes=input_bytes,timeout=timeout))


def source_commit(ref: str) -> str:
    return(command(["git","-C",str(ROOT),"rev-parse","--verify",ref]))


def committed_script(ref: str) -> bytes:
    return(run(["git","-C",str(ROOT),"show",f"{ref}:{REPOSITORY_SCRIPT}"]).stdout)


def stage_controller(server: str,ref: str) -> str:
    commit = source_commit(ref)
    ssh(server,"tee",REMOTE_STAGE,input_bytes=committed_script(ref))
    ssh(server,"chmod","0700",REMOTE_STAGE)
    return(commit)


def remote_json(server: str,*argv: str,timeout: int = 120) -> dict[str,object]:
    result = ssh(server,"sudo","-n","python3",str(INSTALLED_SCRIPT),*argv,timeout=timeout)
    try:
        document = json.loads(result.stdout.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise PxeRescueError(f"invalid remote JSON from {server}") from error
    if not isinstance(document,dict):
        raise PxeRescueError(f"invalid remote document from {server}")
    return(document)


def parse_nodes(value: str) -> tuple[str,...]:
    nodes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not nodes:
        raise PxeRescueError("at least one probe node is required")
    return(nodes)


def probe_client(node: str,server_ip: str,expected_sha256: str) -> dict[str,object]:
    result = ssh(node,"curl","--silent","--show-error",f"tftp://{server_ip}/grub/grub.cfg",timeout=30)
    actual_sha256 = hashlib.sha256(result.stdout).hexdigest()
    return({
        "bytes":len(result.stdout),
        "expected_sha256":expected_sha256,
        "node":node,
        "sha256":actual_sha256,
        "status":"PASS" if actual_sha256 == expected_sha256 else "FAIL",
    })


def parallel_probe(server: str,nodes: tuple[str,...],jobs: int) -> dict[str,object]:
    status = remote_json(server,"--remote-status","--require-active")
    if status.get("failures"):
        raise PxeRescueError(f"PXE server audit failed: {status['failures']}")
    config = status.get("config")
    if not isinstance(config,dict):
        raise PxeRescueError("PXE server status has no config")
    validated = validated_config(config)
    expected_sha256 = hashlib.sha256(grub_config(validated).encode("utf-8")).hexdigest()
    documents = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(jobs,1),len(nodes))) as executor:
        futures = {executor.submit(probe_client,node,validated["server_ip"],expected_sha256):node for node in nodes}
        for future in concurrent.futures.as_completed(futures):
            node = futures[future]
            try:
                document = future.result()
            except (PxeRescueError,subprocess.TimeoutExpired,OSError) as error:
                document = {"node":node,"status":"FAIL","error":str(error)}
            documents.append(document)
            if document["status"] != "PASS":
                failures.append(node)
    documents.sort(key=lambda item:str(item["node"]))
    return({"failures":failures,"nodes":documents,"server":status})


def write_receipt(action: str,server: str,document: dict[str,object]) -> Path:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(tempfile.gettempdir()) / f"ds4_parallel_pxe_rescue_{action}_{timestamp}.json"
    path.write_text(json.dumps({"action":action,"server":server,"result":document},indent=2,sort_keys=True) + "\n")
    return(path)


def controller_action(args: argparse.Namespace) -> int:
    if args.action == "deploy":
        commit = stage_controller(args.server,args.source_ref)
        config = {
            "format":CONFIG_FORMAT,
            "interface":args.interface,
            "recovery_public_key":recovery_public_key(args.recovery_identity),
            "root_device":args.root_device,
            "server_ip":args.server_ip,
            "source_commit":commit,
        }
        payload = (json.dumps(config,sort_keys=True) + "\n").encode("utf-8")
        result = ssh(args.server,"sudo","-n","python3",REMOTE_STAGE,"--remote-install","-",input_bytes=payload,timeout=1200)
        document = json.loads(result.stdout.decode("utf-8"))
        ssh(args.server,"rm","-f",REMOTE_STAGE)
    elif args.action == "start":
        ssh(args.server,"sudo","-n","systemctl","start",SERVICE_NAME)
        document = remote_json(args.server,"--remote-status","--require-active")
    elif args.action == "stop":
        ssh(args.server,"sudo","-n","systemctl","stop",SERVICE_NAME)
        document = remote_json(args.server,"--remote-status")
    elif args.action == "probe":
        document = parallel_probe(args.server,parse_nodes(args.nodes),args.jobs)
    else:
        remote_args = ["--remote-status"]
        if args.require_active:
            remote_args.append("--require-active")
        document = remote_json(args.server,*remote_args)
    receipt = write_receipt(args.action,args.server,document)
    print(json.dumps(document,indent=2,sort_keys=True))
    print(f"receipt={receipt}")
    return(1 if document.get("failures") else 0)


def parse_remote_install(path_text: str) -> dict[str,str]:
    text = sys.stdin.read() if path_text == "-" else Path(path_text).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload,dict):
        raise PxeRescueError("remote install payload is not an object")
    return(validated_config(payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("deploy","start","stop","status","probe"),default="status")
    parser.add_argument("--server",default=DEFAULT_SERVER)
    parser.add_argument("--interface",default=DEFAULT_INTERFACE)
    parser.add_argument("--server-ip",default=DEFAULT_SERVER_IP)
    parser.add_argument("--root-device",default=DEFAULT_ROOT_DEVICE)
    parser.add_argument("--recovery-identity",type=Path,default=DEFAULT_RECOVERY_IDENTITY)
    parser.add_argument("--source-ref",default="HEAD")
    parser.add_argument("--nodes",default=",".join(DEFAULT_PROBE_NODES))
    parser.add_argument("--jobs",type=int,default=len(DEFAULT_PROBE_NODES))
    parser.add_argument("--require-active",action="store_true")
    parser.add_argument("--remote-install",help=argparse.SUPPRESS)
    parser.add_argument("--remote-preflight",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--remote-firewall-open",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--remote-firewall-close",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--remote-status",action="store_true",help=argparse.SUPPRESS)
    return(parser.parse_args())


def main() -> int:
    args = parse_args()
    if args.remote_install is not None:
        print(json.dumps(remote_install(parse_remote_install(args.remote_install)),sort_keys=True))
        return(0)
    if args.remote_preflight:
        print(json.dumps(remote_preflight(),sort_keys=True))
        return(0)
    if args.remote_firewall_open:
        print(json.dumps(remote_firewall_open(),sort_keys=True))
        return(0)
    if args.remote_firewall_close:
        print(json.dumps(remote_firewall_close(),sort_keys=True))
        return(0)
    if args.remote_status:
        print(json.dumps(remote_status(args.require_active),sort_keys=True))
        return(0)
    return(controller_action(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PxeRescueError,subprocess.TimeoutExpired,json.JSONDecodeError,OSError) as error:
        print(f"ds4_parallel_pxe_rescue: {error}",file=sys.stderr)
        raise SystemExit(1)

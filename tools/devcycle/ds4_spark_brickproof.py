#!/usr/bin/env python3
"""Apply and audit the canonical Spark anti-brick boot and OOM policy."""

from __future__ import annotations

import argparse
import base64
import binascii
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


SELF_SOURCE = "tools/devcycle/ds4_spark_brickproof.py"
DEFAULT_NODES = tuple(f"spark{rank:x}" for rank in range(16))
DEFAULT_SPARKPIPE_REPO = Path.home() / "sparkpipe"
DEFAULT_RECOVERY_IDENTITY = Path.home() / ".ssh" / "sparkpipe_fleet_root"
REMOTE_SCRIPT = "/tmp/ds4-spark-brickproof.py"
REMOTE_PAYLOAD = "/tmp/ds4-spark-brickproof.json"
MANAGEMENT_ADDRESS_BASE = 10
MANAGEMENT_GATEWAY = "10.20.0.1"
MANAGEMENT_INTERFACE = "enP7s7"
MANAGEMENT_NETMASK = "255.255.255.0"
MANAGEMENT_PREFIX = "10.20.0"
MAX_SPARK_RANK = 15
NODE_RANK_PATH = Path("/etc/ds4-node-rank")
SWITCHED_ADDRESS_BASE = 10
SWITCHED_PREFIX = "10.10.100"
HOSTS_BLOCK_BEGIN = "# SparkPipe fleet aliases BEGIN"
HOSTS_BLOCK_END = "# SparkPipe fleet aliases END"
CEPH_WARM_STORAGE_MARKER = Path("/etc/sparkpipe/enable-ceph-warm-storage")
LEGACY_HOSTS_BLOCKS = {
    "# DS4 Spark aliases BEGIN":"# DS4 Spark aliases END",
    "# DS4 switched Spark aliases BEGIN":"# DS4 switched Spark aliases END",
    HOSTS_BLOCK_BEGIN:HOSTS_BLOCK_END,
}
SWAP_FILES = (("/swap.img",16), ("/swap-extra-16g.img",16), ("/swap-extra-32g.img",32))
REQUIRED_KERNEL_MODULES = ("sch_fq_codel",)
KERNEL_MODULES_LOAD = "".join(f"{module}\n" for module in REQUIRED_KERNEL_MODULES)
ASSET_SOURCES = {
    "/etc/systemd/system/sparkpipe_model_residentd.service": ("tools/devcycle/sparkpipe_model_residentd.service",0o644),
    "/usr/local/bin/sparkpipe_fsck_health.sh": ("tools/devcycle/sparkpipe_fsck_health.sh",0o755),
    "/etc/systemd/system/sparkpipe-fsck-health.service": ("tools/devcycle/sparkpipe-fsck-health.service",0o644),
}
BOOT_TIMEOUT_SOURCES = {
    "ceph-b52b3459-74b2-428d-b944-1bb691b263c7@.service": "ceph-b52b3459-74b2-428d-b944-1bb691b263c7@.service.conf",
    "rbdmap.service": "rbdmap.service.conf",
    "nvmf-autoconnect.service": "nvmf-autoconnect.service.conf",
    "open-iscsi.service": "open-iscsi.service.conf",
    "pollinate.service": "pollinate.service.conf",
}
ASSET_SOURCES.update({
    "/etc/logrotate.d/00-sparkpipe-cephadm": ("tools/devcycle/fleet/00-sparkpipe-cephadm",0o644),
    "/etc/systemd/system/ds4-switched-fabric.service": ("tools/devcycle/fleet/ds4-switched-fabric.service",0o644),
    "/etc/systemd/system/ds4-switched-fabric.timer": ("tools/devcycle/fleet/ds4-switched-fabric.timer",0o644),
    "/etc/systemd/system/ds4-direct-pair-fabric.service": ("tools/devcycle/fleet/ds4-direct-pair-fabric.service",0o644),
    "/etc/systemd/system/ds4-direct-pair-fabric.timer": ("tools/devcycle/fleet/ds4-direct-pair-fabric.timer",0o644),
    "/usr/local/sbin/ds4-switched-fabric-apply": ("tools/devcycle/fleet/ds4_switched_fabric_apply.sh",0o755),
    "/usr/local/sbin/ds4-direct-pair-fabric-apply": ("tools/devcycle/fleet/ds4_direct_pair_fabric_apply.sh",0o755),
    "/etc/systemd/system/ds4-optional-storage.service": ("tools/devcycle/fleet/ds4-optional-storage.service",0o644),
    "/etc/systemd/system/ds4-optional-storage.timer": ("tools/devcycle/fleet/ds4-optional-storage.timer",0o644),
    "/etc/systemd/system/sparkpipe-fsck-health.timer": ("tools/devcycle/fleet/sparkpipe-fsck-health.timer",0o644),
    "/etc/systemd/system/spark-firewall.service.d/10-ds4-timeout.conf": ("tools/devcycle/fleet/spark-firewall-timeout.conf",0o644),
})
USER_SLICE = """[Slice]
MemoryHigh=100G
MemoryMax=108G
MemorySwapMax=0
"""
EARLYOOM_DEFAULT = 'EARLYOOM_ARGS="-m 7 -s 25 -r 5 --avoid sshd|systemd|init"\n'
NO_THRASH_SYSCTL = "vm.swappiness=10\n"
RECOVERY_SYSCTL = """kernel.panic=60
kernel.softlockup_panic=1
kernel.hung_task_panic=1
"""
SYSTEMD_RECOVERY = """[Manager]
RuntimeWatchdogSec=10min
RebootWatchdogSec=5min
"""
EMERGENCY_SSH_CONFIG = """Port 2222
Protocol 2
ListenAddress 0.0.0.0
UsePAM no
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
AllowUsers root
PidFile /run/sshd-spark-emergency.pid
PrintMotd no
UseDNS no
"""
EMERGENCY_SSH_SERVICE = """[Unit]
Description=Spark emergency public-key SSH on port 2222
After=network.target
Wants=network.target

[Service]
Type=simple
ExecStart=/usr/sbin/sshd -D -e -f /etc/ssh/sshd_config_spark_emergency
Restart=always
RestartSec=2s

[Install]
WantedBy=multi-user.target
"""


class BrickproofError(RuntimeError):
    pass


def run(argv: list[str],input_bytes: bytes | None = None,timeout: int = 120,check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(argv,input=input_bytes,capture_output=True,timeout=timeout)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8",errors="replace").strip()
        raise BrickproofError(f"command failed ({result.returncode}): {' '.join(argv)}: {detail}")
    return(result)


def command(argv: list[str],timeout: int = 120,check: bool = True) -> str:
    return(run(argv,timeout=timeout,check=check).stdout.decode("utf-8",errors="replace").strip())


def atomic_write(path: Path,data: str,mode: int) -> bool:
    encoded = data.encode("utf-8")
    if path.exists() and path.read_bytes() == encoded and (path.stat().st_mode & 0o777) == mode:
        return(False)
    path.parent.mkdir(parents=True,exist_ok=True)
    handle,temp_name = tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(handle,"wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temp_name,mode)
        os.replace(temp_name,path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return(True)


def shell_assignment(text: str,name: str) -> str:
    match = re.search(rf"^[ \t]*{re.escape(name)}[ \t]*=(.*)$",text,re.MULTILINE)
    if match is None:
        return("")
    values = shlex.split(match.group(1),comments=False,posix=True)
    return(values[0] if values else "")


def set_shell_assignment(text: str,name: str,value: str) -> str:
    line = f"{name}={shlex.quote(value)}"
    pattern = re.compile(rf"^[ \t]*{re.escape(name)}[ \t]*=.*$",re.MULTILINE)
    if pattern.search(text) is not None:
        return(pattern.sub(line,text,count=1))
    suffix = "" if text.endswith("\n") else "\n"
    return(f"{text}{suffix}{line}\n")


def public_key_material_is_valid(key_type: str,blob: str) -> bool:
    if not key_type.startswith(("ssh-","ecdsa-","sk-")):
        return(False)
    try:
        decoded = base64.b64decode(blob,validate=True)
    except (binascii.Error,ValueError):
        return(False)
    if len(decoded) < 4:
        return(False)
    name_length = int.from_bytes(decoded[:4],"big")
    if name_length == 0 or (4 + name_length) > len(decoded):
        return(False)
    return(decoded[4:4 + name_length].decode("ascii",errors="replace") == key_type)


def canonical_authorized_keys(text: str,required_key: str) -> str:
    required_tokens = required_key.split()
    if len(required_tokens) < 2 or not public_key_material_is_valid(required_tokens[0],required_tokens[1]):
        raise BrickproofError("fleet recovery public key is invalid")
    required_material = (required_tokens[0],required_tokens[1])
    repaired = re.sub(r"(?<=\S)n(?=ssh-(?:ed25519|rsa|dss|ecdsa-|sk-))","\n",text)
    rendered: list[str] = []
    seen: set[tuple[str,str]] = set()
    pending_type = ""
    required_present = False
    for raw_line in repaired.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            fields = shlex.split(line)
        except ValueError:
            fields = line.split()
        key_index = next((index for index,field in enumerate(fields) if field.startswith(("ssh-","ecdsa-","sk-"))),-1)
        if key_index >= 0:
            pending_type = fields[key_index] if key_index == 0 and len(fields) == 1 else ""
            if (key_index + 1) >= len(fields):
                continue
            material = (fields[key_index],fields[key_index + 1])
            if not public_key_material_is_valid(*material) or material in seen:
                continue
            if material == required_material:
                if key_index > 0:
                    continue
                required_present = True
            rendered.append(line)
            seen.add(material)
            continue
        if pending_type and fields and public_key_material_is_valid(pending_type,fields[0]):
            material = (pending_type,fields[0])
            if material != required_material and material not in seen:
                rendered.append(f"{material[0]} {material[1]}")
                seen.add(material)
        pending_type = ""
    if not required_present:
        rendered.append(required_key.strip())
    return("\n".join(rendered) + "\n")


def recovery_ip_token(recovery_network: object) -> str:
    if not isinstance(recovery_network,dict):
        raise BrickproofError("recovery network is invalid")
    names = ("address","gateway","netmask","node_id","interface")
    values = [str(recovery_network.get(name,"")) for name in names]
    if any(not value for value in values):
        raise BrickproofError("recovery network is incomplete")
    return(f"ip={values[0]}::{values[1]}:{values[2]}:{values[3]}:{values[4]}:none")


def canonical_grub(text: str,recovery_network: object) -> str:
    cmdline = shell_assignment(text,"GRUB_CMDLINE_LINUX_DEFAULT")
    tokens = shlex.split(cmdline)
    tokens = [token for token in tokens if not token.startswith(("ip=","fsck.mode=","fsck.repair=","console=ttyS0"))]
    if "console=tty0" not in tokens:
        tokens.append("console=tty0")
    tokens.extend(("console=ttyS0,921600",recovery_ip_token(recovery_network),"fsck.mode=skip","fsck.repair=no"))
    text = set_shell_assignment(text,"GRUB_CMDLINE_LINUX_DEFAULT",shlex.join(tokens))
    return(set_shell_assignment(text,"GRUB_DEFAULT","0"))


def parse_efi_boot_entries(text: str) -> tuple[list[str],dict[str,str]]:
    order_match = re.search(r"^BootOrder:\s*(.*)$",text,re.MULTILINE)
    if order_match is None:
        raise BrickproofError("efibootmgr output has no BootOrder")
    order = [item.strip().upper() for item in order_match.group(1).split(",") if item.strip()]
    entries = {}
    for match in re.finditer(r"^Boot([0-9A-Fa-f]{4})\*?\s+([^\r\n]+)$",text,re.MULTILINE):
        entries[match.group(1).upper()] = match.group(2).strip()
    if not order or any(item not in entries for item in order):
        raise BrickproofError("efibootmgr BootOrder references a missing entry")
    return((order,entries))


def linux_efi_boot_entry(entries: dict[str,str]) -> str:
    candidates = []
    for entry,label in entries.items():
        normalized = label.strip().casefold()
        if any(normalized == name or normalized.startswith(f"{name} ") or normalized.startswith(f"{name}\t") for name in ("ubuntu","dgx os")):
            candidates.append(entry)
    if len(candidates) != 1:
        raise BrickproofError(f"expected one Linux EFI boot entry, found {len(candidates)}")
    return(candidates[0])


def pxe_ipv4_efi_boot_entries(order: list[str],entries: dict[str,str]) -> list[str]:
    candidates = [entry for entry in order if "pxe ipv4" in entries[entry].casefold()]
    if not candidates:
        raise BrickproofError("expected at least one PXE IPv4 EFI boot entry")
    return(candidates)


def desired_efi_boot_order(text: str) -> list[str]:
    order,entries = parse_efi_boot_entries(text)
    linux_entry = linux_efi_boot_entry(entries)
    pxe_entries = pxe_ipv4_efi_boot_entries(order,entries)
    leading = [*pxe_entries,linux_entry]
    return([*leading,*[entry for entry in order if entry not in leading]])


def package_installed(name: str) -> bool:
    result = run(["dpkg-query","-W","-f=${Status}",name],check=False)
    return(result.returncode == 0 and b"install ok installed" in result.stdout)


def kernel_release_from_vmlinuz(path: Path) -> str:
    try:
        target = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise BrickproofError(f"missing boot kernel link: {path}") from error
    prefix = "vmlinuz-"
    if not target.name.startswith(prefix) or len(target.name) == len(prefix):
        raise BrickproofError(f"cannot derive boot kernel release from {target}")
    return(target.name[len(prefix):])


def boot_kernel_release() -> str:
    return(kernel_release_from_vmlinuz(Path("/boot/vmlinuz")))


def kernel_releases() -> tuple[str,...]:
    return(tuple(dict.fromkeys((os.uname().release,boot_kernel_release()))))


def kernel_packages(kernel_release: str) -> tuple[str,...]:
    return(f"linux-image-{kernel_release}",f"linux-modules-{kernel_release}")


def required_packages(kernel_release: str) -> tuple[str,...]:
    return("earlyoom","dropbear-initramfs","efibootmgr",*kernel_packages(kernel_release))


def package_verification_lines(name: str) -> tuple[str,...]:
    result = run(["dpkg","-V",name],check=False)
    text = (result.stdout + result.stderr).decode("utf-8",errors="replace")
    return(tuple(line for line in text.splitlines() if line.strip()))


def install_packages() -> None:
    packages = list(required_packages(os.uname().release))
    for release in kernel_releases():
        packages.extend(kernel_packages(release))
    packages = list(dict.fromkeys(packages))
    missing = [name for name in packages if not package_installed(name)]
    if missing:
        run(["apt-get","install","-y",*missing],timeout=600)
    damaged = [name for release in kernel_releases() for name in kernel_packages(release) if package_verification_lines(name)]
    damaged = list(dict.fromkeys(damaged))
    if damaged:
        run(["apt-get","install","--reinstall","-y",*damaged],timeout=600)


def rebuild_initramfs() -> None:
    for release in kernel_releases():
        mode = "-u" if Path(f"/boot/initrd.img-{release}").exists() else "-c"
        run(["update-initramfs",mode,"-k",release],timeout=600)


def install_required_kernel_modules() -> None:
    atomic_write(Path("/etc/modules-load.d/90-ds4-network.conf"),KERNEL_MODULES_LOAD,0o644)
    for module in REQUIRED_KERNEL_MODULES:
        run(["modprobe",module])


def fleet_alias_lines() -> tuple[str,...]:
    lines = []
    for rank in range(MAX_SPARK_RANK + 1):
        node = f"spark{rank:x}"
        lines.append(f"{SWITCHED_PREFIX}.{SWITCHED_ADDRESS_BASE + rank} {node} {node}-fabric")
        lines.append(f"{MANAGEMENT_PREFIX}.{MANAGEMENT_ADDRESS_BASE + rank} {node}-mgmt")
    return(tuple(lines))


def fleet_aliases() -> set[str]:
    return({alias for line in fleet_alias_lines() for alias in line.split()[1:]})


def canonical_hosts(text: str,removed_aliases: set[str]) -> str:
    output = []
    managed_end = ""
    aliases_to_replace = fleet_aliases() | removed_aliases
    for line in text.splitlines():
        stripped = line.strip()
        if managed_end:
            if stripped == managed_end:
                managed_end = ""
            continue
        if stripped in LEGACY_HOSTS_BLOCKS:
            managed_end = LEGACY_HOSTS_BLOCKS[stripped]
            continue
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            aliases = [alias for alias in fields[1:] if alias not in aliases_to_replace]
            if aliases:
                output.append(f"{fields[0]}\t{' '.join(aliases)}")
            elif len(fields) == 1:
                output.append(line)
            continue
        output.append(line)
    if managed_end:
        raise BrickproofError(f"unterminated hosts block: expected {managed_end}")
    rendered = "\n".join(output).rstrip()
    block = "\n".join((HOSTS_BLOCK_BEGIN,*fleet_alias_lines(),HOSTS_BLOCK_END))
    return(f"{rendered}\n\n{block}\n" if rendered else f"{block}\n")


def spark_rank(node: str) -> int:
    match = re.fullmatch(r"spark([0-9a-f])",node.casefold())
    if match is None:
        raise BrickproofError(f"invalid Spark node id: {node}")
    rank = int(match.group(1),16)
    if rank > MAX_SPARK_RANK:
        raise BrickproofError(f"Spark rank {rank} exceeds the fleet")
    return(rank)


def install_node_identity(payload: dict[str,object]) -> bool:
    hostname = str(payload["recovery_network"]["node_id"])
    rank = spark_rank(hostname)
    previous_hostname = command(["hostname","-s"],check=False)
    changed = previous_hostname != hostname
    if changed:
        run(["hostnamectl","set-hostname",hostname])
    atomic_write(Path("/etc/hostname"),f"{hostname}\n",0o644)
    rank_changed = atomic_write(NODE_RANK_PATH,f"{rank}\n",0o644)
    hosts = Path("/etc/hosts")
    hosts_changed = atomic_write(hosts,canonical_hosts(hosts.read_text(),{hostname,previous_hostname}),0o644)
    return(changed or hosts_changed or rank_changed)


def ensure_swap_file(path_text: str,size_gib: int) -> None:
    path = Path(path_text)
    expected = size_gib * 1024 * 1024 * 1024
    active = {line.split()[0] for line in Path("/proc/swaps").read_text().splitlines()[1:]}
    if path.exists() and path.stat().st_size != expected:
        raise BrickproofError(f"{path} has {path.stat().st_size} bytes, expected {expected}")
    if not path.exists():
        run(["fallocate","-l",f"{size_gib}G",path_text],timeout=600)
        os.chmod(path,0o600)
        run(["mkswap",path_text],timeout=120)
    elif path_text not in active:
        os.chmod(path,0o600)
        run(["mkswap",path_text],timeout=120)
    if path_text not in active:
        run(["swapon",path_text],timeout=120)


def ensure_swap() -> None:
    for path,size_gib in SWAP_FILES:
        ensure_swap_file(path,size_gib)
    fstab_path = Path("/etc/fstab")
    fstab = fstab_path.read_text()
    fields = {line.split()[0] for line in fstab.splitlines() if line.strip() and not line.lstrip().startswith("#")}
    additions = [f"{path} none swap sw 0 0" for path,_ in SWAP_FILES if path not in fields]
    if additions:
        suffix = "" if fstab.endswith("\n") else "\n"
        atomic_write(fstab_path,f"{fstab}{suffix}" + "\n".join(additions) + "\n",0o644)


def install_sshd_config() -> None:
    target = Path("/etc/ssh/sshd_config_spark_emergency")
    handle,temp_name = tempfile.mkstemp(prefix="sshd-spark-emergency.",dir="/etc/ssh")
    try:
        with os.fdopen(handle,"w",encoding="utf-8") as output:
            output.write(EMERGENCY_SSH_CONFIG)
        os.chmod(temp_name,0o644)
        run(["/usr/sbin/sshd","-t","-f",temp_name])
        os.replace(temp_name,target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def install_recovery_keys(public_key: str) -> bool:
    changed = False
    root_keys = Path("/root/.ssh/authorized_keys")
    root_keys.parent.mkdir(parents=True,exist_ok=True)
    root_text = root_keys.read_text() if root_keys.exists() else ""
    changed |= atomic_write(root_keys,canonical_authorized_keys(root_text,public_key),0o600)
    os.chmod(root_keys.parent,0o700)
    dropbear_keys = Path("/etc/dropbear/initramfs/authorized_keys")
    dropbear_keys.parent.mkdir(parents=True,exist_ok=True)
    dropbear_text = dropbear_keys.read_text() if dropbear_keys.exists() else ""
    changed |= atomic_write(dropbear_keys,canonical_authorized_keys(dropbear_text,public_key),0o600)
    return(changed)


def install_assets(payload: dict[str,object]) -> bool:
    changed = False
    assets = payload["assets"]
    if not isinstance(assets,dict):
        raise BrickproofError("payload assets are invalid")
    for path_text,specification in assets.items():
        if not isinstance(specification,dict):
            raise BrickproofError(f"invalid asset specification for {path_text}")
        changed |= atomic_write(Path(path_text),str(specification["text"]),int(specification["mode"]))
    return(changed)


def install_grub_policy(recovery_network: object) -> bool:
    path = Path("/etc/default/grub")
    before = path.read_text()
    changed = atomic_write(path,canonical_grub(before,recovery_network),0o644)
    legacy = Path("/etc/grub.d/40_ds4_fastboot")
    if legacy.exists():
        legacy.unlink()
        changed = True
    if changed:
        run(["update-grub"],timeout=300)
    return(changed)


def install_efi_boot_order() -> bool:
    before = command(["efibootmgr","-v"])
    order,_ = parse_efi_boot_entries(before)
    desired = desired_efi_boot_order(before)
    if order == desired:
        return(False)
    run(["efibootmgr","-o",",".join(desired)])
    after = command(["efibootmgr","-v"])
    if desired_efi_boot_order(after) != desired or parse_efi_boot_entries(after)[0] != desired:
        raise BrickproofError("EFI boot order verification failed")
    return(True)


def persistent_unit_links(unit: str,root: Path = Path("/etc/systemd/system")) -> list[Path]:
    return(sorted(
        path for path in root.rglob(unit)
        if path.is_symlink() and path.parent.name.endswith((".wants",".requires"))
    ))


def disable_persistent_unit(unit: str) -> None:
    run(["systemctl","disable",unit],check=False)
    for path in persistent_unit_links(unit):
        path.unlink()


def enable_policy_services() -> None:
    run(["systemctl","daemon-reload"])
    run(["sysctl","--system"],timeout=180)
    run(["systemctl","enable","earlyoom.service"])
    run(["systemctl","restart","earlyoom.service"])
    run(["systemctl","enable","ssh-emergency.service"])
    run(["systemctl","restart","ssh-emergency.service"])
    disable_persistent_unit("sparkpipe-fsck-health.service")
    run(["systemctl","enable","--now","sparkpipe-fsck-health.timer"])
    run(["systemctl","enable","serial-getty@ttyS0.service"])
    disable_persistent_unit("ds4-switched-fabric.service")
    disable_persistent_unit("ds4-direct-pair-fabric.service")
    run(["systemctl","enable","--now","ds4-switched-fabric.timer"])
    run(["systemctl","enable","--now","ds4-direct-pair-fabric.timer"])
    disable_persistent_unit("sparkpipe_model_residentd.service")
    run(["systemctl","reset-failed","sparkpipe_model_residentd.service"],check=False)
    run(["systemctl","enable","--now","ds4-optional-storage.timer"])
    run(["systemctl","unmask","fstrim.service"])
    run(["systemctl","enable","--now","fstrim.timer"])
    run(["systemctl","reset-failed","fstrim.service","fstrim.timer"],check=False)
    run(["systemctl","reset-failed","ceph*.service","ceph*.target"],check=False)
    run(["systemctl","set-default","multi-user.target"])
    run(["systemctl","set-property","--runtime","user-1000.slice","MemoryHigh=100G","MemoryMax=108G","MemorySwapMax=0"])


def has_configured_remote_storage() -> list[str]:
    configured = []
    rbdmap = read_optional(Path("/etc/ceph/rbdmap"))
    if any(line.strip() and not line.lstrip().startswith("#") for line in rbdmap.splitlines()):
        configured.append("rbdmap")
    fstab = read_optional(Path("/etc/fstab"))
    for line in fstab.splitlines():
        fields = line.split()
        if line.strip() and not line.lstrip().startswith("#") and len(fields) >= 4:
            if fields[2] in ("nfs","nfs4","cifs","ceph") or "_netdev" in fields[3].split(","):
                configured.append("remote-fstab")
    iscsi = Path("/etc/iscsi/nodes")
    if iscsi.exists() and any(path.is_file() for path in iscsi.rglob("*")):
        configured.append("iscsi")
    discovery = read_optional(Path("/etc/nvme/discovery.conf"))
    if any(line.strip() and not line.lstrip().startswith("#") for line in discovery.splitlines()):
        configured.append("nvme-of")
    return(sorted(set(configured)))


def decouple_optional_boot_work() -> None:
    configured = has_configured_remote_storage()
    if configured:
        raise BrickproofError(f"remote storage configuration exists; refusing to mask initiators: {configured}")
    ceph_targets = command(["systemctl","list-unit-files","--type=target","--no-legend","ceph*.target"],check=False)
    for row in ceph_targets.splitlines():
        unit = row.split()[0] if row.split() else ""
        if unit:
            run(["systemctl","disable",unit],check=False)
    for unit in ("rbdmap.service","nvmf-autoconnect.service","open-iscsi.service","iscsid.service","srp_daemon.service"):
        run(["systemctl","disable",unit],check=False)
        run(["systemctl","mask",unit],check=False)
    for unit in ("cloud-init.service","cloud-init-local.service","cloud-config.service","cloud-final.service","pollinate.service","nvidia-spark-run-apt-upgrade-once.service","systemd-networkd-wait-online.service"):
        run(["systemctl","disable",unit],check=False)
        run(["systemctl","mask",unit],check=False)
    atomic_write(Path("/etc/cloud/cloud-init.disabled"),"commissioned by ds4_spark_brickproof\n",0o644)


def remote_apply(payload_path: Path) -> dict[str,object]:
    if os.geteuid() != 0:
        raise BrickproofError("remote apply must run as root")
    payload = json.loads(payload_path.read_text())
    memory_current = int(command(["systemctl","show","user-1000.slice","-p","MemoryCurrent","--value"]) or "0")
    if memory_current >= 108 * 1024 * 1024 * 1024:
        raise BrickproofError(f"user-1000.slice already uses {memory_current} bytes; refusing a 108G cap")
    install_packages()
    install_required_kernel_modules()
    identity_changed = install_node_identity(payload)
    ensure_swap()
    install_assets(payload)
    atomic_write(Path("/etc/systemd/system/user-1000.slice.d/20-ds4-brickproof.conf"),USER_SLICE,0o644)
    atomic_write(Path("/etc/default/earlyoom"),EARLYOOM_DEFAULT,0o644)
    atomic_write(Path("/etc/sysctl.d/90-sparkpipe-no-thrash.conf"),NO_THRASH_SYSCTL,0o644)
    atomic_write(Path("/etc/sysctl.d/99-spark-recovery.conf"),RECOVERY_SYSCTL,0o644)
    atomic_write(Path("/etc/systemd/system.conf.d/90-ds4-recovery.conf"),SYSTEMD_RECOVERY,0o644)
    install_sshd_config()
    atomic_write(Path("/etc/systemd/system/ssh-emergency.service"),EMERGENCY_SSH_SERVICE,0o644)
    obsolete = Path("/etc/systemd/system/sparkpipe_model_residentd.service.d/10-oom-guardrails.conf")
    if obsolete.exists():
        obsolete.unlink()
    for unit in ("ds4-switched-fabric.service","ds4-direct-pair-fabric.service"):
        legacy_timeout = Path(f"/etc/systemd/system/{unit}.d/10-boot-timeout.conf")
        if legacy_timeout.exists():
            legacy_timeout.unlink()
    keys_changed = install_recovery_keys(str(payload["fleet_public_key"]))
    grub_changed = install_grub_policy(payload["recovery_network"])
    efi_boot_order_changed = install_efi_boot_order()
    rebuild_initramfs()
    decouple_optional_boot_work()
    enable_policy_services()
    run(["systemctl","start","logrotate.service"])
    return({"efi_boot_order_changed":efi_boot_order_changed,"grub_changed":grub_changed,"identity_changed":identity_changed,"keys_changed":keys_changed,"memory_current_before":memory_current,"source_commit":payload["source_commit"]})


def read_optional(path: Path) -> str:
    try:
        return(path.read_text())
    except (FileNotFoundError,PermissionError):
        return("")


def service_value(unit: str,property_name: str) -> str:
    return(command(["systemctl","show",unit,"-p",property_name,"--value"],check=False))


def is_enabled(unit: str) -> str:
    return(command(["systemctl","is-enabled",unit],check=False))


def remote_audit(payload_path: Path) -> dict[str,object]:
    payload = json.loads(payload_path.read_text())
    public_key = str(payload["fleet_public_key"])
    failures = []
    observations: dict[str,object] = {}
    required_module_state: dict[str,object] = {}
    swaps = Path("/proc/swaps").read_text()
    swap_bytes = sum(int(line.split()[2]) * 1024 for line in swaps.splitlines()[1:] if len(line.split()) >= 3)
    observations["swap_bytes"] = swap_bytes
    if swap_bytes < 63 * 1024 * 1024 * 1024:
        failures.append("swap<63GiB")
    observations["swappiness"] = command(["sysctl","-n","vm.swappiness"],check=False)
    observations["panic"] = command(["sysctl","-n","kernel.panic"],check=False)
    running_kernel = os.uname().release
    try:
        boot_kernel = boot_kernel_release()
    except BrickproofError as error:
        boot_kernel = ""
        failures.append(str(error))
    observations["running_kernel"] = running_kernel
    observations["boot_kernel"] = boot_kernel
    kernel_state: dict[str,object] = {}
    for release in dict.fromkeys((running_kernel,boot_kernel)):
        if not release:
            continue
        release_state: dict[str,object] = {"module_tree":Path(f"/lib/modules/{release}").is_dir()}
        if not release_state["module_tree"]:
            failures.append(f"missing-module-tree:{release}")
        for package in kernel_packages(release):
            installed = package_installed(package)
            verification_lines = package_verification_lines(package) if installed else ()
            release_state[package] = {"installed":installed,"verification_lines":len(verification_lines)}
            if not installed:
                failures.append(f"missing-package:{package}")
            elif verification_lines:
                failures.append(f"damaged-package:{package}:{len(verification_lines)}")
        kernel_state[release] = release_state
    observations["kernel_releases"] = kernel_state
    modules_load = read_optional(Path("/etc/modules-load.d/90-ds4-network.conf"))
    if modules_load != KERNEL_MODULES_LOAD:
        failures.append("kernel-modules-load-config")
    for module in REQUIRED_KERNEL_MODULES:
        available = command(["modinfo","-F","filename",module],check=False)
        loaded = (Path("/sys/module") / module).is_dir()
        required_module_state[module] = {"available":available,"loaded":loaded}
        if not available:
            failures.append(f"kernel-module-unavailable:{module}")
        if not loaded:
            failures.append(f"kernel-module-not-loaded:{module}")
    observations["required_kernel_modules"] = required_module_state
    observations["default_qdisc"] = command(["sysctl","-n","net.core.default_qdisc"],check=False)
    if observations["default_qdisc"] != "fq_codel":
        failures.append(f"default-qdisc:{observations['default_qdisc']}")
    observations["earlyoom_process"] = command(["pgrep","-af","earlyoom"],check=False)
    if "-m 7 -s 25 -r 5" not in str(observations["earlyoom_process"]):
        failures.append("earlyoom-runtime-args")
    for unit in ("earlyoom.service","ssh-emergency.service"):
        state = service_value(unit,"ActiveState")
        observations[f"{unit}.active"] = state
        if state != "active":
            failures.append(f"{unit}:{state}")
    for unit in ("serial-getty@ttyS0.service","sparkpipe-fsck-health.timer","ds4-switched-fabric.timer","ds4-direct-pair-fabric.timer","ds4-optional-storage.timer"):
        state = is_enabled(unit)
        observations[f"{unit}.enabled"] = state
        if state != "enabled":
            failures.append(f"{unit}:not-enabled")
    fstrim_service = is_enabled("fstrim.service")
    fstrim_timer = is_enabled("fstrim.timer")
    observations["fstrim.service.enabled"] = fstrim_service
    observations["fstrim.timer.enabled"] = fstrim_timer
    if fstrim_service != "static":
        failures.append(f"fstrim.service:{fstrim_service}")
    if fstrim_timer != "enabled":
        failures.append(f"fstrim.timer:{fstrim_timer}")
    cmdline_tokens = shlex.split(read_optional(Path("/proc/cmdline")))
    runtime_masks = {token.split("=",1)[1] for token in cmdline_tokens if token.startswith("systemd.mask=")}
    for unit in ("ds4-switched-fabric.service","ds4-direct-pair-fabric.service","sparkpipe-fsck-health.service","sparkpipe_model_residentd.service"):
        state = is_enabled(unit)
        links = [str(path) for path in persistent_unit_links(unit)]
        observations[f"{unit}.enabled"] = state
        observations[f"{unit}.persistent_links"] = links
        rescue_masked = state == "masked-runtime" and unit in runtime_masks
        if state not in ("disabled","static") and not rescue_masked:
            failures.append(f"boot-critical-link:{unit}={state}")
        if links:
            failures.append(f"boot-critical-links:{unit}")
    for unit in ("rbdmap.service","nvmf-autoconnect.service","open-iscsi.service","iscsid.service","srp_daemon.service","cloud-init.service","cloud-init-local.service","cloud-config.service","cloud-final.service","pollinate.service","nvidia-spark-run-apt-upgrade-once.service","systemd-networkd-wait-online.service"):
        state = is_enabled(unit)
        observations[f"{unit}.enabled"] = state
        if state != "masked":
            failures.append(f"optional-boot-unit:{unit}={state}")
    for unit in ("ds4-switched-fabric.service","ds4-direct-pair-fabric.service"):
        before = service_value(unit,"Before")
        timeout = service_value(unit,"TimeoutStartUSec")
        installed = read_optional(Path("/etc/systemd/system") / unit)
        rescue_masked = unit in runtime_masks
        observations[f"{unit}.before"] = before
        observations[f"{unit}.timeout"] = timeout
        observations[f"{unit}.rescue_masked"] = rescue_masked
        if "network-online.target" in installed or "multi-user.target" in installed:
            failures.append(f"fabric-boot-order:{unit}")
        if "TimeoutStartSec=60" not in installed:
            failures.append(f"fabric-timeout:{unit}={timeout}")
    firewall_timeout = service_value("spark-firewall.service","TimeoutStartUSec")
    observations["spark-firewall.timeout"] = firewall_timeout
    if firewall_timeout not in ("15s","15sec"):
        failures.append(f"spark-firewall-timeout={firewall_timeout}")
    expected_limits = {"MemoryHigh":str(100 * 1024**3),"MemoryMax":str(108 * 1024**3),"MemorySwapMax":"0"}
    for unit in ("user-1000.slice","sparkpipe_model_residentd.service"):
        for property_name,expected in expected_limits.items():
            value = service_value(unit,property_name)
            observations[f"{unit}.{property_name}"] = value
            if value != expected:
                failures.append(f"{unit}.{property_name}={value}")
    watchdog = service_value("sparkpipe_model_residentd.service","WatchdogUSec")
    observations["resident_watchdog"] = watchdog
    if watchdog not in ("0","0us","infinity"):
        failures.append(f"resident-watchdog={watchdog}")
    if Path("/etc/systemd/system/sparkpipe_model_residentd.service.d/10-oom-guardrails.conf").exists():
        failures.append("obsolete-resident-dropin")
    key_id = " ".join(public_key.split()[:2])
    for path in (Path("/root/.ssh/authorized_keys"),Path("/etc/dropbear/initramfs/authorized_keys")):
        key_text = read_optional(path)
        present = any(" ".join(line.split()[:2]) == key_id for line in key_text.splitlines())
        observations[f"key:{path}"] = present
        if not present:
            failures.append(f"missing-key:{path}")
        if key_text != canonical_authorized_keys(key_text,public_key):
            failures.append(f"noncanonical-key:{path}")
    initrd = command(["readlink","-f","/boot/initrd.img"],check=False)
    initrd_files = command(["lsinitramfs",initrd],timeout=180,check=False) if initrd else ""
    observations["initramfs_recovery_key"] = any(line.endswith("/.ssh/authorized_keys") for line in initrd_files.splitlines())
    if not observations["initramfs_recovery_key"]:
        failures.append("missing-key:initramfs")
    grub = read_optional(Path("/etc/default/grub"))
    cmdline = shell_assignment(grub,"GRUB_CMDLINE_LINUX_DEFAULT")
    observations["grub_default"] = shell_assignment(grub,"GRUB_DEFAULT")
    observations["grub_cmdline"] = cmdline
    if shell_assignment(grub,"GRUB_DEFAULT") != "0":
        failures.append("grub-default")
    for token in ("fsck.mode=skip","fsck.repair=no"):
        if token not in shlex.split(cmdline):
            failures.append(f"grub-missing:{token}")
    expected_recovery_ip = recovery_ip_token(payload["recovery_network"])
    if expected_recovery_ip not in shlex.split(cmdline):
        failures.append("grub-recovery-ip")
    if not any(token.startswith("console=ttyS0") for token in shlex.split(cmdline)):
        failures.append("grub-missing:serial")
    if Path("/etc/grub.d/40_ds4_fastboot").exists():
        failures.append("legacy-fastboot-entry")
    efi_boot = command(["efibootmgr","-v"],check=False)
    observations["efi_boot"] = efi_boot
    try:
        efi_order,efi_entries = parse_efi_boot_entries(efi_boot)
        desired_efi = desired_efi_boot_order(efi_boot)
        if efi_order != desired_efi:
            failures.append("efi-pxe-not-first")
    except BrickproofError as error:
        failures.append(f"efi-boot:{error}")
    listeners = command(["ss","-ltn"],check=False)
    if not re.search(r"(?:\*|0\.0\.0\.0):2222\b",listeners):
        failures.append("emergency-port-not-listening")
    root_options = command(["findmnt","-n","-o","OPTIONS","/"],check=False)
    observations["root_options"] = root_options
    if "rw" not in root_options.split(","):
        failures.append("root-read-only")
    observations["hostname"] = command(["hostname","-s"],check=False)
    if observations["hostname"] != payload["recovery_network"]["node_id"]:
        failures.append(f"hostname:{observations['hostname']}")
    expected_rank = str(spark_rank(str(payload["recovery_network"]["node_id"])))
    observations["node_rank"] = read_optional(NODE_RANK_PATH).strip()
    if observations["node_rank"] != expected_rank:
        failures.append(f"node-rank:{observations['node_rank']}")
    hosts = read_optional(Path("/etc/hosts"))
    if hosts != canonical_hosts(hosts,{str(payload["recovery_network"]["node_id"])}):
        failures.append("hosts-loopback-alias")
    observations["kernel"] = command(["uname","-r"],check=False)
    observations["system_state"] = command(["systemctl","is-system-running"],check=False)
    if observations["system_state"] != "running":
        failures.append(f"system-state:{observations['system_state']}")
    failed_units = command(["systemctl","--failed","--no-legend","--plain"],check=False)
    observations["failed_units"] = failed_units
    for line in failed_units.splitlines():
        fields = line.split()
        if fields:
            failures.append(f"failed-systemd-unit:{fields[0]}")
    logrotate = run(["logrotate","--debug","/etc/logrotate.conf"],timeout=180,check=False)
    observations["logrotate_config_rc"] = logrotate.returncode
    if logrotate.returncode != 0:
        failures.append("logrotate-config")
    observations["ceph_warm_storage_selected"] = CEPH_WARM_STORAGE_MARKER.exists()
    observations["ceph_target_active"] = service_value("ceph.target","ActiveState")
    if (not observations["ceph_warm_storage_selected"] and
            observations["ceph_target_active"] == "active"):
        failures.append("unselected-ceph-active")
    health_path = Path("/var/lib/sparkpipe/fsck-health/last.json")
    if health_path.exists():
        try:
            observations["fsck_health"] = json.loads(health_path.read_text()).get("classification","unknown")
        except json.JSONDecodeError:
            failures.append("fsck-health-json")
    else:
        observations["fsck_health"] = "pending-next-boot"
    return({"failures":failures,"observations":observations,"source_commit":payload["source_commit"]})


def git_show(repo: Path,ref: str,path: str) -> str:
    result = run(["git","-C",str(repo),"show",f"{ref}:{path}"])
    return(result.stdout.decode("utf-8"))


def build_payload(repo: Path,ref: str) -> dict[str,object]:
    commit = command(["git","-C",str(repo),"rev-parse","--verify",ref])
    assets: dict[str,dict[str,object]] = {}
    for destination,(source,mode) in ASSET_SOURCES.items():
        assets[destination] = {"mode":mode,"text":git_show(repo,ref,source)}
        if not str(assets[destination]["text"]).endswith("\n"):
            assets[destination]["text"] = str(assets[destination]["text"]) + "\n"
    for unit,source_name in BOOT_TIMEOUT_SOURCES.items():
        destination = f"/etc/systemd/system/{unit}.d/10-boot-timeout.conf"
        source = f"tools/devcycle/boot-unblock/{source_name}"
        text = git_show(repo,ref,source)
        assets[destination] = {"mode":0o644,"text":text if text.endswith("\n") else text + "\n"}
    public_key = git_show(repo,ref,"tools/devcycle/sparkpipe_fleet_root.pub").strip()
    if not public_key.startswith("ssh-ed25519 "):
        raise BrickproofError("fleet recovery key is not an Ed25519 public key")
    remote_script = git_show(repo,ref,SELF_SOURCE)
    if Path(__file__).read_text(encoding="utf-8") != remote_script:
        raise BrickproofError(f"controller does not match {ref}:{SELF_SOURCE}")
    return({"assets":assets,"fleet_public_key":public_key,"remote_script":remote_script,"source_commit":commit})


def recovery_network_for_node(node: str) -> dict[str,str]:
    rank = spark_rank(node)
    return({
        "address":f"{MANAGEMENT_PREFIX}.{MANAGEMENT_ADDRESS_BASE + rank}",
        "gateway":MANAGEMENT_GATEWAY,
        "interface":MANAGEMENT_INTERFACE,
        "netmask":MANAGEMENT_NETMASK,
        "node_id":node.casefold(),
    })


def ssh(host: str,*argv: str,input_bytes: bytes | None = None,timeout: int = 120,check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return(run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=8",host,*argv],input_bytes=input_bytes,timeout=timeout,check=check))


def stage_remote(host: str,payload: dict[str,object]) -> None:
    staged_payload = dict(payload)
    script_data = str(staged_payload.pop("remote_script")).encode("utf-8")
    payload_data = (json.dumps(staged_payload,sort_keys=True) + "\n").encode("utf-8")
    ssh(host,"tee",REMOTE_SCRIPT,input_bytes=script_data)
    ssh(host,"tee",REMOTE_PAYLOAD,input_bytes=payload_data)
    ssh(host,"chmod","0700",REMOTE_SCRIPT)
    ssh(host,"chmod","0600",REMOTE_PAYLOAD)


def remote_action(host: str,action: str,payload: dict[str,object]) -> dict[str,object]:
    node_payload = dict(payload)
    node_payload["recovery_network"] = recovery_network_for_node(host)
    stage_remote(host,node_payload)
    try:
        result = ssh(host,"sudo","-n","python3",REMOTE_SCRIPT,f"--remote-{action}",REMOTE_PAYLOAD,timeout=1200)
        document = json.loads(result.stdout.decode("utf-8"))
        document["node"] = host
        return(document)
    finally:
        ssh(host,"rm","-f",REMOTE_SCRIPT,REMOTE_PAYLOAD,check=False)


def emergency_probe(host: str,identity: Path) -> str:
    result = run(["ssh","-T","-p","2222","-i",str(identity),"-o","BatchMode=yes","-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=accept-new","-o","ConnectTimeout=8",f"root@{host}","true"],timeout=20,check=False)
    return("ok" if result.returncode == 0 else (result.stderr or result.stdout).decode("utf-8",errors="replace").strip())


def apply_one(host: str,payload: dict[str,object],identity: Path) -> dict[str,object]:
    applied = remote_action(host,"apply",payload)
    audited = remote_action(host,"audit",payload)
    audited["apply"] = applied
    audited["emergency_login"] = emergency_probe(host,identity)
    if audited["emergency_login"] != "ok":
        audited["failures"].append("controller-emergency-login")
    if audited["failures"]:
        raise BrickproofError(f"{host} failed post-apply audit: {json.dumps(audited,sort_keys=True)}")
    return(audited)


def audit_one(host: str,payload: dict[str,object],identity: Path) -> dict[str,object]:
    audited = remote_action(host,"audit",payload)
    audited["emergency_login"] = emergency_probe(host,identity)
    if audited["emergency_login"] != "ok":
        audited["failures"].append("controller-emergency-login")
    return(audited)


def write_receipt(action: str,documents: list[dict[str,object]]) -> Path:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(tempfile.gettempdir()) / f"ds4_spark_brickproof_{action}_{timestamp}.json"
    path.write_text(json.dumps({"action":action,"nodes":documents},indent=2,sort_keys=True) + "\n")
    return(path)


def parse_nodes(value: str) -> tuple[str,...]:
    nodes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not nodes:
        raise BrickproofError("at least one node is required")
    return(nodes)


def controller_main(args: argparse.Namespace) -> int:
    nodes = parse_nodes(args.nodes)
    payload = build_payload(args.sparkpipe_repo,args.source_ref)
    documents = []
    if not args.recovery_identity.is_file():
        raise BrickproofError(f"missing recovery identity: {args.recovery_identity}")
    if args.action == "apply":
        canary = args.canary if args.canary in nodes else nodes[0]
        print(f"canary={canary}",flush=True)
        documents.append(apply_one(canary,payload,args.recovery_identity))
        remaining = [node for node in nodes if node != canary]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs,len(remaining) or 1)) as executor:
            futures = {executor.submit(apply_one,node,payload,args.recovery_identity):node for node in remaining}
            for future in concurrent.futures.as_completed(futures):
                document = future.result()
                documents.append(document)
                print(f"{document['node']}: PASS",flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs,len(nodes))) as executor:
            futures = {executor.submit(audit_one,node,payload,args.recovery_identity):node for node in nodes}
            for future in concurrent.futures.as_completed(futures):
                document = future.result()
                documents.append(document)
                print(f"{document['node']}: {'PASS' if not document['failures'] else 'FAIL'}",flush=True)
    documents.sort(key=lambda item:str(item["node"]))
    receipt = write_receipt(args.action,documents)
    print(f"receipt={receipt}")
    return(1 if any(document.get("failures") for document in documents) else 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("audit","apply"),default="audit")
    parser.add_argument("--nodes",default=",".join(DEFAULT_NODES))
    parser.add_argument("--canary",default="spark3")
    parser.add_argument("--jobs",type=int,default=3)
    parser.add_argument("--sparkpipe-repo",type=Path,default=DEFAULT_SPARKPIPE_REPO)
    parser.add_argument("--source-ref",default="origin/main")
    parser.add_argument("--recovery-identity",type=Path,default=DEFAULT_RECOVERY_IDENTITY)
    parser.add_argument("--remote-apply",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--remote-audit",type=Path,help=argparse.SUPPRESS)
    return(parser.parse_args())


def main() -> int:
    args = parse_args()
    if args.remote_apply is not None:
        print(json.dumps(remote_apply(args.remote_apply),sort_keys=True))
        return(0)
    if args.remote_audit is not None:
        print(json.dumps(remote_audit(args.remote_audit),sort_keys=True))
        return(0)
    return(controller_main(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BrickproofError,subprocess.TimeoutExpired,json.JSONDecodeError) as error:
        print(f"ds4_spark_brickproof: {error}",file=sys.stderr)
        raise SystemExit(1)

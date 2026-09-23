#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'b690c3a57005981fec1a2f611d95ec946aa3ddf7'
ASSETS = {
    'sparkpipe-shared-serving-linux-arm64-cuda13.tar.gz': '097165339f969bd57d0409092684ffd8f3237af96dc617534cdb7639772194ca',
    'shared-serving-qualification.tar.gz': 'a184ff86961628581624f8092eae2d895927dc9919d4ff1d3989e4bf21d3c238',
}
WSET = 'fbd8e8a4bff8ec99840b9bd1be5039d7ca73d305bb92ee0bafd5552bbbfcd06b'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def prepare(assets, output, remote, replicas):
    for name, expected in ASSETS.items():
        if digest(assets / name) != expected:
            raise ValueError('release asset checksum mismatch: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(assets / 'shared-serving-qualification.tar.gz') as archive:
        archive.extractall(output, members=[m for m in archive.getmembers()
                           if m.name.startswith('inference-inputs/')], filter='data')
    inputs = output / 'inference-inputs'
    campaign = json.loads((inputs / 'campaign-plan.json').read_text())
    if campaign['source_commit'] != SOURCE:
        raise ValueError('qualification source differs from released source')
    old = campaign['remote_root']
    for path in inputs.rglob('*'):
        if path.is_file():
            path.write_text(path.read_text().replace(old, remote))
    deployment = inputs / 'common/deployment.json'
    reference = inputs / f'common/reference-{replicas}.json'
    value = json.loads(reference.read_text())
    value['deployment_sha256'] = digest(deployment)
    reference.write_text(json.dumps(value, indent=2) + '\n')
    for host in campaign['pack_hash_inventory']:
        stage = inputs / host['host'] / 'stage-assets.sh'
        lines = stage.read_text().splitlines()
        if not lines[-1].startswith('cp ') or '.wset' not in lines[-1]:
            raise ValueError('unexpected archived working-set preparation')
        lines[-1] = f'cp {shlex.quote(remote + "/qualified-smoke.wset")} {shlex.quote(remote + "/inputs/runtime/packs/stage.sp.wset")}'
        stage.write_text('\n'.join(lines) + '\n')
    return inputs


NODE_STAGE = r'''
import hashlib,json,pathlib,subprocess,sys,tarfile,shutil
root=pathlib.Path(sys.argv[1]);host=sys.argv[2];job=sys.argv[3];sha=sys.argv[4];bundle_sha=sys.argv[5];wset_sha=sys.argv[6]
source=pathlib.Path('/home')/host/'srcdata/sparkqueue'/job/sha
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==sha
subprocess.run(['git','-C',str(source),'diff','--exit-code','HEAD'],check=True,stdout=subprocess.DEVNULL)
assert hashlib.sha256((root/'bundle.tar.gz').read_bytes()).hexdigest()==bundle_sha
with tarfile.open(root/'bundle.tar.gz') as archive:archive.extractall(root/'artifacts',filter='data')
subprocess.run(['sha256sum','--quiet','--check','SHA256SUMS'],cwd=root/'artifacts',check=True)
assert (root/'artifacts/SOURCE_COMMIT').read_text().strip()==sha
wset=root/'qualified-smoke.wset'
assert hashlib.sha256(wset.read_bytes()).hexdigest()==wset_sha
(root/'source').symlink_to(source,target_is_directory=True)
subprocess.run(['bash',str(root/'stage-assets.sh')],check=True)
shutil.copyfile(root/'stage.json',root/'inputs/runtime/config/stage.json')
print(json.dumps(dict(host=host,source=sha,bundle_sha256=bundle_sha)),flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--replicas', type=int, choices=(2, 3, 4, 8), default=8)
    parser.add_argument('--host-memory-mib', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'tools'))
    import spark_queue as queue
    queue.valid_name(args.id)
    device = 28 * 1024 + 512 + args.replicas * 4096
    total = device + args.host_memory_mib
    if args.host_memory_mib < 64 or total + queue.HOST_HEADROOM_MIB > queue.NODE_MEMORY_MIB_MAX:
        parser.error('host plus device reservation and fleet headroom exceed the node budget')
    hosts = ['spark' + format(rank, 'x') for rank in range(16)]
    remote = '/tmp/sparkpipe-replay-' + args.id
    output = queue.STATE / ('replay-' + args.id)
    inputs = prepare(args.assets.resolve(), output, remote, args.replicas)
    wset = ROOT / 'model-families/glm5_next/glm53flash.fp8.tp16.smoke.wset'
    if digest(wset) != WSET:
        parser.error('qualified input working set checksum differs')
    run([str(ROOT / 'tools/sparkpipe_github_pat.sh'), 'git', 'fetch', '--depth=1', 'origin', SOURCE], cwd=ROOT)
    cli = [sys.executable, str(ROOT / 'tools/spark_queue.py')]
    run(cli + ['sync', '--id', args.id, '--nodes', ','.join(hosts), '--ref', SOURCE])

    def stage(host):
        run(['ssh', host, 'mkdir ' + shlex.quote(remote) + ' && mkdir ' + shlex.quote(remote + '/inputs')])
        run(['scp', '-q', str(args.assets / next(iter(ASSETS))), host + ':' + remote + '/bundle.tar.gz'])
        run(['scp', '-q', *map(str, (inputs / 'common').glob('*.json')), host + ':' + remote + '/inputs/'])
        run(['scp', '-q', str(wset), host + ':' + remote + '/qualified-smoke.wset'])
        run(['scp', '-q', str(inputs / host / 'stage-assets.sh'), str(inputs / host / 'config/stage.json'), host + ':' + remote + '/'])
        run(['ssh', host, 'python3 - ' + shlex.join([remote, host, args.id, SOURCE, next(iter(ASSETS.values())), WSET])], input=NODE_STAGE, text=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(stage, hosts))
    command = output / 'run.cmd'
    command.write_text('exec python3 tools/inference_smoke.py --spec ' + shlex.quote(remote + f'/inputs/spec-{args.replicas}.json') + '\n')
    run(cli + ['add', '--id', args.id, '--nodes', ','.join(hosts), '--per-node', '--cwd', remote + '/source',
               '--resources', 'gpu-shared', '--memory-mib', str(total), '--device-memory-mib', str(device),
               '--ports', f'20000:{20000 + args.replicas * 1000 - 1}', '--ttl-min', '15', '--cmd-file', str(command)])
    print('Submitted exact release replay; inspect: ' + shlex.join(cli + ['status', '--id', args.id]))
    print('Queue success alone is insufficient: collect receipt.json from all 16 attempt directories and verify PASS, tokens, and owned PID absence.')


if __name__ == '__main__':
    main()

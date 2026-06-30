# GLM52 B12x Pack Worker Protocol

The B12x resident pack worker is a control-plane protocol for building
`.spb12x` resident expert packs on Spark nodes.

It does not transfer pack payloads over SSH.  SSH may be used only for tiny
control messages such as submit, queue depth, and status.  Multi-GB pack
artifacts must stay on node-local NVMe, a fast shared mount, or a dedicated
200 Gbps artifact path.

## Commands

Queue depth:

```sh
python3 tools/glm52_b12x_pack_worker.py queue-depth \
    --queue-dir /tmp/sparkpipe_b12x_pack_queue \
    --max-jobs 2
```

Submit:

```sh
python3 tools/glm52_b12x_pack_worker.py submit \
    --queue-dir /tmp/sparkpipe_b12x_pack_queue \
    --job-id stage-0011-0018 \
    --model-dir /mnt/mac/16tb0/models/hf/nvidia/GLM-5.2-NVFP4 \
    --aot-manifest build/glm52_b12x_aot/generated/aot_manifest.json \
    --layers 11,12,13,14,15,16,17,18 \
    --output-dir build/glm52_b12x_resident_moe_0011_0018 \
    --local-jobs 2
```

Serve:

```sh
python3 tools/glm52_b12x_pack_worker.py serve \
    --queue-dir /tmp/sparkpipe_b12x_pack_queue \
    --max-jobs 2
```

Status:

```sh
python3 tools/glm52_b12x_pack_worker.py status \
    --queue-dir /tmp/sparkpipe_b12x_pack_queue \
    --job-id stage-0011-0018
```

## Concurrency Rule

Live Spark2 evidence showed 16 local pack writers were I/O-bound, with CPU well
below saturation.  The default local worker count is therefore conservative.

To use the 13-Spark fleet efficiently:

```text
good:
    distribute stage pack jobs across idle Sparks
    keep local-jobs low enough that each node remains I/O healthy
    use 200 Gbps/shared storage for artifact placement

bad:
    run all pack jobs on one Spark
    move .spb12x payloads over ssh/scp
    assume 16 local writers helps without measuring CPU, IO wait, and disk bandwidth
```

The protocol reports:

```text
artifact_transfer = none_control_plane_only
artifact_manifest = <node-local or shared path>/resident_moe_pack_manifest.json
```

The coordinator must treat the artifact path as a location on the fast artifact
plane, not as data to copy through the control channel.

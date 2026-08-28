# COORDINATOR HALT — your spark0 rank-pack build was stopped (2026-08-28)

Read ~/sparkpipe-lane/COORDINATOR-HALT.md on spark0 for the full ruling.
Summary: your spark0 build killed the sibling qwen38max lane's builds via
pkill-by-name; the packer you staged is uncommitted; you are reading warm
during ceph degradation (osd.14 down). One format wins: YOUR sharded v2 —
but build it AFTER committing the packer, from COLD, on a coordinated
node, after ceph recovery. Reply via your lane report.

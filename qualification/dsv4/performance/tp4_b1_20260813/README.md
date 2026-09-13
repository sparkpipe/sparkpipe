# DeepSeek V4 Flash TP4 B1 evidence

This directory retains the raw benchmark receipts and exact input batch for
the 2026-08-13 cached-prefill B1 milestone documented in
[`../../../../PERFORMANCE_STATUS.md`](../../../../PERFORMANCE_STATUS.md).
The JSON files are byte-for-byte copies of the measured artifacts; no fields
were removed or rewritten. A repository secret and private-address scan found
no credentials, user home paths, or private IP addresses before ingestion.

| File | SHA-256 |
| --- | --- |
| `dsv4-combine-relay-run1.json` | `ba792c90f5f484bb49f6a92e95ef807d1e9efcd30d0dad06fb96b76481be2321` |
| `dsv4-combine-relay-run2.json` | `5e7f8307bd29d42cea0aead9fd09d8edd63c02e459ce4e49a89cb75bbe9d32fc` |
| `dsv4-combine-relay-run3.json` | `ff130bc6d751543b073e340772132457eff501b160588883fb198e18f19feaa8` |
| `dsv4-combine-relay-run4.json` | `6e0e4ea9afdbd4c40eebea1c6d235062761c1811241b7fce28a69a771d2767ec` |
| `dsv4-combine-relay-control2.json` | `ff9530527512f6ef1d128ae3e51dc17a1a481e430666742701482258be2d91b7` |
| `dsv4-tp4-pp4-b1-compsec076-o128.json` | `e498f1fc88854044eafa64c41ce308b73d54f0a351fe156a513d7ff7ca630ead` |

The receipt `command` arrays preserve the measured scratch paths. Those paths
identify the original run but are not required repository layout. Use the
reproduction command in the performance ledger with a current zero-drift
runtime instead.

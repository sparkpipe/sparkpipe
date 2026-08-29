# Tokenizer assets for the ds4-eval ground truth

These are the tokenizer assets the 92-case quality fixtures were built
against, staged beside the fixtures they ground-truth so the host gates can
prove the text/token-id edge against real vocabularies offline.

| asset | source | upstream commit | sha256 | bytes |
|---|---|---|---|---|
| `glm-5.3-flash-tokenizer.json` | `huggingface.co/zai-org/GLM-5.3-Flash` (`resolve/main/tokenizer.json`) | `04c4e9e95c5da8862dced7e5056455116f83a7e0` | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` | 20,217,442 |
| `kimi-k3-tiktoken.model` | `huggingface.co/moonshotai/Kimi-K3` (`resolve/main/tiktoken.model`) | `a590ce090cb049c93a33dfe8c208ec652aa20503` | `b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103` | 2,795,286 |

Fetched 2026-08-29 by the tokenizer-sidecar lane. The glm5.3-flash fixture
(`../quality-fixtures-glm5.3-flash.json`, key `"tokenizer"`:
`"glm-5.3-flash/tokenizer.json"`) carries the pre-tokenized `prompt_token_ids`
for all 92 cases, generated with real tiktoken on spark1; the exact rendered
prompt texts live in `../runs/kimi-k3-api-20260728/cases.json`
(`rendered_prompt`, joined case-id for case-id, 92/92). `build/
test_tokenizer_sidecar` asserts encode(text)==ids and decode(ids)==text for
every case through the committed GLM asset, and round trips the committed
tiktoken ranks asset (the Kimi-K3 format) through the same sidecar.

This directory is excluded from the source package and the authored-code
ratchet (same classification as the rest of `qualification/ds4_eval/`): the
assets are measurement ground truth, not implementation source.

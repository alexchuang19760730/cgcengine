# GDN (#7): from "ABSENT" to "real but never entered" — 2026-09-22

Answer first, evidence second.

| claim | verdict |
|---|---|
| "there is no chunked GatedDeltaNet operator in this tree ⇒ the knob has nowhere to push" | **wrong**. The fused operator exists and covers both token-count shapes (§1) |
| the knob can make the engine do something different | **true, proven at runtime** (§2): `fused_ar=0 req_ar=0 ablated=1` |
| so GDN is a lever for 25 t/s | **no**. In the delivered workload `build_delta_net` is **never reached** (§3), so the axis costs nothing and gains nothing |

## 1. The operator exists

- `ggml_gated_delta_net` has a Metal pipeline named at `ggml-metal-device.cpp:638`
  (`kernel_gated_delta_net_<type>_<nsg>`).
- `ne20 / ne30 / K` are **function constants**, and `K` is the number of state snapshots — so the
  same kernel serves the `T == 1` recurrence (K=1) and the rolled-up `T > 1` case. The
  "we need a chunked operator first" premise was false; it came from grepping the enum member
  `LLM_FUSED_OP_GDN_CH` and finding no reference to it.
- Three implementations, dispatched at `models/delta-net-base.cpp:430-451`:
  `build_delta_net_fused` / `build_delta_net_autoregressive` (T=1) / `build_delta_net_chunking`
  (T>1), selected by `cparams.fused_gdn_ar` and `cparams.fused_gdn_ch`.
- The model is genuinely hybrid: `general.architecture = qwen35moe`,
  `full_attention_interval = 4`, `block_count = 41`, `ssm.state_size = 128` ⇒ by the loader's own
  rule ~30 recurrent layers and 10 full-attention layers.

## 2. The knob does flip something (runtime proof)

`CGC_SHAPE_GDN_AR=0` / `CGC_SHAPE_GDN_CH=0` are wired into the context constructor
(`llama-context.cpp:253-254`) and read back by `cgc_shape_note_gdn`:

```
# default                 phase=gdn fused_ar=1 fused_ch=1 req_ar=-1 req_ch=-1 ablated=0 bitident=yes
# CGC_SHAPE_GDN_AR=0      phase=gdn fused_ar=0 fused_ch=1 req_ar=0  req_ch=-1 ablated=1 bitident=NO
# both =0                 phase=gdn fused_ar=0 fused_ch=0 req_ar=0  req_ch=0  ablated=1 bitident=NO
```

Two fixes landed along the way:

1. **The line was breaking its own contract.** `note=` was printed as prose containing spaces *and*
   the substring `K=1`, so `shape_knob_search.py:parse_shape_line` invented a phantom key `K` with
   value `1)` and dumped the rest into `_unparsed`. Values are now one token with no `=`:
   `note=fused_metal_default_k_snapshot` / `note=fusion_off_ablation_not_comparable`. The parser
   self-test keeps the broken sentence as a fixture and asserts it is caught (it must go red if the
   prose ever comes back).
2. **`gdn_ablated` moved onto the `phase=final` line**, the row everything downstream reads, so an
   ablation cannot pass as a normal measurement. The harness now reports such rows as `ABLATED`
   rather than as "slower".

## 3. And yet nothing changed — because the operator never runs

Turning the fusion off left the oracle dump **byte-for-byte identical** (`cmp` clean, 9 records,
248320 logits each). Before believing that, either run could have been explained away, so the
builder itself now reports which branch it took (`cgc_shape_count_gdn`, called from
`build_delta_net`), printed as `gdn_saw_fused` / `gdn_saw_manual`:

| run | `gdn_saw_fused` | `gdn_saw_manual` | gate result |
|---|---|---|---|
| default (fused on) | **0** | **0** | PASS M1/M2/M3 9/9 |
| `AR=0` | 0 | 0 | PASS 9/9 (M1 identical) |
| `AR=0 CH=0` | 0 | 0 | INVALID COMPARISON (different numerics config) |

`build_recurrent_attn` calls `build_delta_net` unconditionally, so **no layer built a delta-net
node in any of these runs** — with fusion on or off. Hence:

- The equality of the dumps is *not* evidence that the two implementations agree numerically.
  There was nothing to disagree about.
- **Any GDN measurement made through this server path has been measuring zero.** This retires the
  axis honourably rather than leaving it "to be tried later".
- Note the trap that made the arithmetic-only ablation look like a null result: under MTP the
  target verifies several tokens at once, so `n_seq_tokens > 1` and the **chunking** branch applies.
  Ablating `AR` alone can be invisible for that reason even when the axis is live. Always set both.

**Not yet pinned down:** why the recurrent branch is not entered for this GGUF even though its
metadata declares 30 recurrent layers. Next step is one counter at the top of
`build_recurrent_attn` to separate "layer function not called" from "called but the SSM pieces are
missing". Until then the honest reading is the observation itself: not called.

## 4. Consequence for the 25 t/s plan

The GDN axis is not a source of time in the delivered decode cell. Nothing further should be spent
on it until a workload is shown where `gdn_saw_*` is non-zero; the counters are permanent, so that
is now a one-look check on any future row.

## Files changed

| file | what |
|---|---|
| `src/llama.cpp/src/llama-shape-knob.{h,cpp}` | `note=` as one token; `gdn_ablated` on the final line; `cgc_shape_count_gdn()` |
| `src/llama.cpp/src/models/delta-net-base.cpp` | calls `cgc_shape_count_gdn(fused)` on every dispatch |
| `scripts/check/shape_knob_search.py` | harvests the `phase=gdn` row; `ABLATED` verdict; dead "axis ABSENT ⇒ VOID" rule removed; docstring corrected |
| `scripts/check/decode_window_harness.py` | anchor re-set to the build the gate passed (`971ade0f947d23f8`) |

Evidence: `Backup/m123_oracle_gate/summary_gdn_probe_{default,ablate}_20260922.json`, logs under
`Backup/cgc_logs/llama_server_20260922_22*.log`.

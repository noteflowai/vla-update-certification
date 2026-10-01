# Prospective update primitives, before main policy outcomes

The native inference producer remains separate. `update_variants.py` supplies
checked configuration and weight-update primitives; its seven controls use
small CPU tensors and fixture configurations, not robot policy outcomes.

| Pinned family | Consumed flow field | Baseline flow steps | Predicted / executed actions |
|---|---|---:|---:|
| pi05 | `num_inference_steps` | 10 | 50 / 50 |
| xvla | `num_denoising_steps` | 10 | 30 / 30 |

The fields are confirmed against cached revision configs and actual installed
modeling sources in `analysis/pinned-update-configuration-001.json`.
`configure_variant` deep-copies the config, requires the pinned cadence and
flow defaults, and changes only the family-consumed field for `steps2`.
The control called `baseline_reload` still requires the producer to instantiate
and independently reload the identical pipeline. Configuration alone does not
perform or validate that control.

W4/W3 round every actual `nn.Linear` weight to a symmetric grid in float32,
grouped in 128 input columns, then store dequantized values in the original
FP32/FP16/BF16 dtype. A final short group is handled separately even for widths
above 128. Do not fall back to whole-row scales on non-divisible widths.
Biases are preserved. Shared Parameter objects are changed once and reported;
the same weights may also be used by modules other than `nn.Linear`.

The fixed default row chunk is 128. Hashing, finite-value validation and
rounding process row blocks instead of materializing whole large matrices on
CPU or allocating multiple full-size temporary matrices. Different row chunk
sizes produce identical stored tensor values/hashes in the CPU control.
This establishes numerical equivalence in that control, not GPU memory or
hardware-performance measurements.

All selected weights are validated before any mutation. A later runtime
failure invalidates the policy instance: preserve the attempt evidence and
discard/reload it rather than continue with partially updated weights.
Every layer's before/after hash, dtype, shape, group remainder and squared
error are recorded; actual runtime hook coverage still requires native trials.

The checkpoint header contains 27 pi05 rank-two weight candidates with widths
above 128 and nonzero remainder, and zero such xvla candidates. Headers include
weights outside `nn.Linear`; these are not claimed as actual forward coverage.
The original first-paper evaluator and its historical outcomes are unchanged.

Freeze snapshot `update-variants-002` before any main update outcome. Require
the corrected family health/clean-exit gates and full reset audit before native
collection. This is floating round/dequantize, not integer-kernel acceleration.

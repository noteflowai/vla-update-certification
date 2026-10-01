# Health-check infrastructure amendment, before new outcomes

September 30, 2026. Earlier runs and their zero completed episodes remain
archived. This changes infrastructure only; it supplies no update efficacy.

- Keep the pinned PI05 checkpoint, official selective BF16/FP32 architecture,
  official processors, ten flow steps, 50-action chunk and 520-step horizon.
- Keep the same five predeclared task/state pairs, seeds, controls and 3/5
  success threshold. No outcome-based substitution.
- Replace transient CPU random initialization with direct-CUDA construction
  inside Transformers' no-initialization context. Use the official remapping.
  Fail on missing, unexpected, nonfinite or wrong-shape tensors. Compare all
  loaded tensors exactly to stored values after the official dtype conversion.
  A loading exception cannot return a random-weight policy.
  Rebuild derived nonpersistent rotary/position/embedding-scale buffers from
  CPU references, retaining official buffer precision; reject unknown buffers.
- Use the existing host/GPU headroom guards, low process priority, no compilation
  and at most half of total CUDA memory for this process.
- Cap the run at 1,800 wall-clock seconds, including imports via the outer
  supervisor. Save initialization, reset, action-request and rollout events.
  Persist partial traces. An interrupted episode has null success and is an
  infrastructure result, never a failed policy episode.
- Five completed, error-free episodes determine passed/failed health. Otherwise
  report not assessed, including when only some cases finish.
- Shared-host duration is diagnostic elapsed time, not dedicated inference
  latency or hardware acceleration evidence. Do not interfere with other jobs.

The strict-copy controls use a tiny real torch model and check dtype conversion,
missing keys, shape/nonfinite rejection and a deliberately broken silent loader.
The actual model must additionally pass complete tensor verification before
any health outcome can be interpreted.

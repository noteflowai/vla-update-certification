# Native paired collection, frozen before its first model outcome

`native_collection.py`, `native_rollout.py` and `run_native_collection.py`
implement the previously missing real LeRobot/LIBERO producer. Implementation
and CPU fixture tests are not validation of actual native collection.

The entry point requires the family's complete fixed five-state health cohort,
its clean worker exit, exact reset checks and pinned checkpoint. It checks all
health rows, summary/lifecycle hashes, health sources and the source-matched
45-case reset/scoring audit. PI05 additionally requires its strict loaded-tensor
audit. It never pools interrupted health attempts.

Before any native outcome, freeze the twenty-state inventory, model/config/
processor bytes, policy and processor implementation files, evaluator source,
package versions, variants, cadence and the six methods' 1,000-pair cap.
Separate policy-seed streams for the eight comparisons are:
`2026100100 + 4*family_index + update_index`, using family order PI05, X-VLA and
update order baseline reload, W4, W3, two flow steps. The original scene seeds
remain fixed. This newly specified seed rule precedes main update outcomes.

The default `first_batch` mode uses a three-hour initialization/collection
allowance and collects exactly the first uniform method's
requested ten pairs at state index zero. It is an operational pilot with no
full-method or update-efficacy conclusion. A subsequent `full` invocation can
reuse these committed pairs while retaining the original 1,000-pair method
horizon. Changing the mode does not change context, sample identities or cap.
The unresolved-attempt guard still applies; restarting after an error requires
explicit evidence reconciliation.

Load one policy at a time, run the requested old sides, release that policy,
independently reload the new side and run the same identities. Checkpoint bytes
are fully hashed once per unchanged process-local file signature; a changed
signature forces rehashing. PI05's native load retains per-tensor strict
verification; X-VLA retains its official strict-key loading implementation.
Rounding is floating round/dequantize, with no hardware acceleration claim.

Each attempted side gets a durable claim before environment construction.
Save both reset snapshots, complete input/state hashes, every full raw
pre-action observation, selected action before stepping, checker/goal fields,
termination, terminal observation and raw evidence digests. Failed or
interrupted episodes remain infrastructure attempts, not binary failures.
Model provisioning, releases, episode attempts and errors are fsynced in a
separate physical event journal. The producer rejects raw side reuse without
reconciliation, validates all returned pairs and keeps old/new reset inputs
identical. Native cache reuse also verifies side commits, saved side results
and a state reset-input binding. The actual episode kernel freezes each
state's first verified simulator/input digests before prediction. Every later
repeat and side must match; agreement inside each individual old/new pair
alone cannot conceal a scene that drifts between repeats.
It also verifies raw evidence digests. Process-local file signatures avoid repeatedly
rehashing unchanged raw files; a changed signature forces revalidation.
Full raw observation recording may be expensive; its cost is part
of collection and must not be advertised as inference latency.

Supervise the new process with the existing parent-death and signal cleanup
controls. A typed native summary reader keeps collection status separate from
episode-health verdict. Native errors or unclean exits do not become successful
collection. All eight cohorts share a conservative twelve-hour wall-time
reservation ledger; an unresolved launch keeps its full reservation. A cohort
lock blocks simultaneous collectors on the same cache. Native launch
preflight also rejects every unresolved, failed or changed previous launch.
A later clean process that merely reads the cache cannot promote an earlier
unclean native run to operational success; investigate and explicitly reconcile
that incident before resume. Health
preflight attempts precede this main-collection ledger and must be reported
separately as feasibility overhead.

The native evaluator binds physical GPU zero, a hashed device UUID and the
driver version. Switching a device/driver on resume requires another cohort,
instead of silently mixing outcome streams. Runtime reports also retain the
actual Torch build, CUDA, cuDNN and thread settings. This does not retrospectively
establish missing historical hardware metadata or guarantee deterministic
kernels under the declared warn-only setting.

Example, **only after the named receipt actually passes**:

```bash
python3 experiments/run_native_collection.py \
  --family pi05 --update baseline_reload \
  --health-receipt analysis/pi05-health-20261001-003-supervisor.json \
  --output runs/pi05-baseline-native-001 --prepare-only
```

Remove `--prepare-only` for the fixed first batch. The command fails before
model imports if health evidence is incomplete. Require a successful native
operational pilot before scheduling full cohorts; do not treat compilation,
synthetic fixture controls or a prepared manifest as collected policy evidence.

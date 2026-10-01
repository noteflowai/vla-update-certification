# Durable requested batches before native collection

`PairCache.get_many(keys, produce_many)` adds a batch contract while preserving
the existing one-pair interface. `collect_methods` uses it when the producer
implements `produce_many`; otherwise it retains the original behavior.

The native producer is still pending. These changes and their controls do not
constitute robot outcomes, model-health evidence or a measured speedup.

## Why batch at the producer boundary

A native producer can execute the requested old episodes together, reload the
new pipeline once, then execute their paired new episodes. It need not reload
both large policies for each requested pair or keep both on the shared GPU.
The implementation can retain one policy at a time and persist raw side records
between stages. Model switching and every actual attempted episode must remain
in the physical cost record.

The batch interface does not prefetch future requests. A method receives exactly
the batch its existing allocation rule requested, in that order. Cached outcomes
remain unavailable to other methods until those methods request them. Fixture
controls require byte-identical consumed-batch journals and equal decision
snapshots between single and batch collection.

## Durable and scientific boundaries

- Preflight all requested identities and cached evidence before starting new
  attempts. An unresolved member blocks the whole requested batch.
- Persist every missing attempt before invoking the native producer. The
  producer must additionally save its raw episode/trace evidence as it runs.
- Pass copies of expected randomization identities. A producer that mutates its
  argument cannot change the validator's task, state, scene or policy seed.
- Require the complete ordered result inventory and validate every returned
  pair before committing any pair from that produced batch.
- Preserve successful commits if a later disk write fails. Unresolved records
  block blind retry; interruption never becomes a policy failure or an
  accepted missing outcome.
- Reuse only committed pairs with matching evidence digests. Keep the requesting
  method's existing consumed-prefix journal unchanged.

## Verification

Nine new controls cover durable claims, cache reuse, mixed cached/new requests,
invalid inventories, unresolved attempts, short/reordered/mismatched results,
interrupted generators, mutation of producer identities, partial commit failure,
and integration with all six methods and replay. Some controls exercise several
failure cases.

```bash
PYTHONPATH=experiments /tmp/vlareg/.venv/bin/python -m unittest \
  test_paired_batches test_paired_collection
```

The CPU fixtures are explicitly synthetic. Before collecting a real cohort,
freeze this source version in its evaluator identity, implement the native
producer, require both family health and clean-exit gates, and retain the
source-matched 45-case reset/scoring audit. Do not append these fixtures to either
paper's empirical data.

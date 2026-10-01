"""Durable consumed-prefix controller; a validated native producer is separate.

No robot inference is implemented here. Producer attempts must persist their own
raw episode evidence. Unresolved attempts block blind retries. Cached outcomes
are delivered only through the requesting method's predeclared prefix.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
from closedloop_decisions import METHODS, PrefixDecision


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class PairCache:
    def __init__(self, folder, context):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.context = json.loads(json.dumps(context, sort_keys=True, allow_nan=False))
        for key in ("protocol_sha256", "evaluator_sha256",
                    "old_pipeline_sha256", "new_pipeline_sha256"):
            value = self.context.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"Missing frozen collection provenance: {key}")
            int(value, 16)
        marker = self.folder/"context.json"
        if marker.exists():
            if json.loads(marker.read_text()) != self.context:
                raise ValueError("Collection context differs; use a new cohort")
        else:
            atomic_json(marker, self.context)
        self.identity_hash = sha256(json.dumps(self.context, sort_keys=True).encode()).hexdigest()

    def identity(self, state, repeat):
        if not 0 <= state < len(self.context["states"]) or repeat < 0:
            raise ValueError("Invalid state/repeat")
        case = self.context["states"][state]
        seed = int(np.random.SeedSequence(
            [self.context["seed"], state, repeat]).generate_state(1)[0])
        return {"context_sha256": self.identity_hash, "state_index": state,
                "repeat": repeat, "task": case["task"], "init_state": case["init_state"],
                "scene_seed": case["scene_seed"], "policy_seed": seed,
                **{key: self.context[key] for key in (
                    "protocol_sha256", "evaluator_sha256",
                    "old_pipeline_sha256", "new_pipeline_sha256")}}

    def path(self, state, repeat):
        return self.folder/f"pair-{state}-{repeat}.json"

    @staticmethod
    def validate(record, identity):
        if record.get("identity") != identity:
            raise ValueError("Pair identity or seed mismatch")
        states = []
        inputs = []
        values = []
        for side in ("old", "new"):
            episode = record[side]
            if (type(episode.get("success")) is not bool
                    or episode.get("infrastructure_error") is not None
                    or episode.get("reset_state_exact") is not True
                    or episode.get("reset_inputs_exact") is not True
                    or type(episode.get("steps")) is not int
                    or not 1 <= episode["steps"] <= 520):
                raise ValueError("Incomplete, erroneous or invalid policy episode")
            if any(episode.get(key) != identity[key] for key in (
                    "task", "init_state", "scene_seed", "policy_seed")):
                raise ValueError("Episode is not the requested paired randomization")
            if (episode.get("pipeline_sha256") != identity[f"{side}_pipeline_sha256"]
                    or episode.get("evaluator_sha256") != identity["evaluator_sha256"]):
                raise ValueError("Wrong old/new pipeline or evaluator provenance")
            initial = episode.get("initial_state_sha256")
            if not isinstance(initial, str) or len(initial) != 64:
                raise ValueError("Missing simulator-state digest")
            int(initial, 16)
            states.append(initial)
            initial_input = episode.get("initial_input_sha256")
            if not isinstance(initial_input, str) or len(initial_input) != 64:
                raise ValueError("Missing complete reset-input digest")
            int(initial_input, 16)
            inputs.append(initial_input)
            values.append(episode["success"])
        if states[0] != states[1]:
            raise ValueError("Old/new simulator initial states differ")
        if inputs[0] != inputs[1]:
            raise ValueError("Old/new complete reset inputs differ")
        return tuple(values)

    def get(self, state, repeat, producer):
        identity = self.identity(state, repeat)
        path = self.path(state, repeat)
        attempt = self.folder/f"attempt-{state}-{repeat}.json"
        if path.exists():
            commit = json.loads(attempt.read_text()) if attempt.exists() else {}
            if (commit.get("status") != "committed"
                    or commit.get("pair_sha256") != sha256(path.read_bytes()).hexdigest()):
                raise ValueError("Pair commit missing or evidence changed; reconcile before use")
            return self.validate(json.loads(path.read_text()), identity)
        if attempt.exists():
            raise RuntimeError("Unresolved producer attempt; reconcile raw evidence before retry")
        atomic_json(attempt, {"identity": identity, "status": "started",
                              "at": datetime.now(timezone.utc).isoformat()})
        try:
            record = producer(dict(identity))
            values = self.validate(record, identity)
            atomic_json(path, record)
        except BaseException as error:
            atomic_json(attempt, {"identity": identity, "status": "error",
                                  "error": type(error).__name__+": "+str(error)})
            raise
        atomic_json(attempt, {"identity": identity, "status": "committed",
                              "pair_sha256": sha256(path.read_bytes()).hexdigest()})
        return values

    def get_many(self, keys, produce_many):
        """Claim a requested batch before native work; validate all before commit.

        A native producer can run every old episode and then every new episode
        with one model reload per side. It must durably save raw attempts itself.
        This interface neither prefetches future requests nor exposes other
        methods' cached outcomes to the requesting method.
        """
        keys = [tuple(key) for key in keys]
        if any(len(key) != 2 for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Batch needs distinct state/repeat identities")
        identities = [self.identity(*key) for key in keys]
        cached, missing = {}, []
        # Preflight the entire request before starting any new physical work.
        for key, identity in zip(keys, identities, strict=True):
            path = self.path(*key)
            attempt = self.folder/f"attempt-{key[0]}-{key[1]}.json"
            if path.exists():
                cached[key] = self.get(*key, producer=None)
            elif attempt.exists():
                raise RuntimeError("Unresolved producer attempt; reconcile raw evidence before retry")
            else:
                missing.append((key, identity, path, attempt))
        if not missing:
            return [cached[key] for key in keys]
        started, committed = [], set()
        try:
            for item in missing:
                key, identity, path, attempt = item
                atomic_json(attempt, {"identity": identity, "status": "started",
                                     "at": datetime.now(timezone.utc).isoformat()})
                started.append(item)
            records = list(produce_many([dict(item[1]) for item in missing]))
            if len(records) != len(missing):
                raise ValueError("Producer batch does not cover the complete requested inventory")
            values = [self.validate(record, item[1])
                      for record, item in zip(records, missing, strict=True)]
            for record, value, item in zip(records, values, missing, strict=True):
                key, identity, path, attempt = item
                atomic_json(path, record)
                atomic_json(attempt, {"identity": identity, "status": "committed",
                                     "pair_sha256": sha256(path.read_bytes()).hexdigest()})
                committed.add(key)
                cached[key] = value
        except BaseException as error:
            for key, identity, path, attempt in started:
                if key not in committed:
                    atomic_json(attempt, {"identity": identity, "status": "error",
                                         "error": type(error).__name__+": "+str(error)})
            raise
        return [cached[key] for key in keys]


def collect_methods(folder, context, producer, cap=1000, cache_class=PairCache):
    """Resume exact allocation histories; do not pass other methods' data to a gate."""
    folder = Path(folder)
    cache = cache_class(folder/"pairs", context)
    engines = [PrefixDecision(method, allocation, context["seed"]+index,
                              states=len(context["states"]), cap=cap)
               for index, (method, allocation) in enumerate(METHODS)]
    config = folder/"method-config.json"
    expected = {"methods": [list(item) for item in METHODS], "cap": cap,
                "alpha": .05/(8*6), "states": len(context["states"])}
    if config.exists():
        if json.loads(config.read_text()) != expected:
            raise ValueError("Method config changed during collection")
    else:
        atomic_json(config, expected)
    journal = folder/"consumed-batches.jsonl"
    if journal.exists():
        data = journal.read_bytes()
        if data and not data.endswith(b"\n"):
            raise ValueError("Incomplete journal tail; preserve and reconcile before resume")
        for line in data.splitlines():
            item = json.loads(line)
            engine = engines[item["method_index"]]
            requested = engine.request()
            if [list(key) for key in requested] != item["requested"]:
                raise ValueError("Resumed allocation differs from the journal")
            values = []
            for key, recorded_sha in zip(requested, item["pair_sha256"], strict=True):
                path = cache.path(*key)
                if sha256(path.read_bytes()).hexdigest() != recorded_sha:
                    raise ValueError("Consumed pair evidence changed")
                values.append(cache.validate(json.loads(path.read_text()), cache.identity(*key)))
            engine.consume(requested, values)
    error = None
    try:
        with journal.open("a") as output:
            while not all(engine.finished for engine in engines):
                for index, engine in enumerate(engines):
                    if engine.finished:
                        continue
                    requested = engine.request()
                    values = (cache.get_many(requested, producer.produce_many)
                              if hasattr(producer, "produce_many") else
                              [cache.get(*key, producer) for key in requested])
                    engine.consume(requested, values)
                    item = {"method_index": index, "requested": requested,
                            "pair_sha256": [sha256(cache.path(*key).read_bytes()).hexdigest()
                                            for key in requested]}
                    output.write(json.dumps(item)+"\n")
                    output.flush()
                    os.fsync(output.fileno())
    except BaseException as caught:
        error = type(caught).__name__+": "+str(caught)
        raise
    finally:
        pairs = len(list((folder/"pairs").glob("pair-*.json")))
        atomic_json(folder/"summary.json", {
            "status": ("producer_error" if error else "completed"
                       if all(engine.finished for engine in engines) else "interrupted"),
            "error": error, "methods": [engine.snapshot() for engine in engines],
            "committed_physical_pairs": pairs, "committed_pair_policy_episodes": 2*pairs,
            "producer_attempt_count": len(list((folder/"pairs").glob("attempt-*.json"))),
            "scope": "Producer raw attempt evidence is required separately; committed pair "
                     "episodes do not include incomplete/error attempts. No reference-cohort labels."})
    return engines

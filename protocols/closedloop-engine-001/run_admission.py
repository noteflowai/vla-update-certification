"""Collect frozen synthetic screening and descriptive historical VLA replay."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import time
import numpy as np
from scipy.stats import beta
from admission import Certificate

ROOT = Path("/home/dcvuser/work")
HERE = Path(__file__).resolve().parent


def scenarios(states=20):
    def make(name, plus, minus):
        return {"name": name, "plus": np.broadcast_to(plus, (states,)).copy(),
                "minus": np.broadcast_to(minus, (states,)).copy()}
    cases = [
        make("unchanged_low_disagreement", .002, .002),
        make("improved", .005, .06),
        make("diffuse_harm", .08, .005),
        make("just_inside_budget", .044, .005),
        make("just_outside_budget", .046, .005),
    ]
    for name, harm_states, positive, background in [
        ("sparse_harm", 2, .55, .002),
        ("heterogeneous_harm", 3, .4, .06),
    ]:
        p = np.full(states, background)
        m = np.full(states, background)
        p[:harm_states], m[:harm_states] = positive, .01
        cases.append(make(name, p, m))
    return cases


def fixed_bounds(n, plus, minus, alpha, delta):
    # One terminal look only, not an anytime-valid baseline.
    a = alpha / (2 * len(n))
    def interval(k, count):
        if not count:
            return 0., 1.
        lo = 0. if k == 0 else float(beta.ppf(a/2, k, count-k+1))
        hi = 1. if k == count else float(beta.ppf(1-a/2, k+1, count-k))
        return lo, hi
    low, high = [], []
    for count, p, m in zip(n, plus, minus):
        lp, up = interval(p, count)
        lm, um = interval(m, count)
        low.append(max(lp-um-delta, 0))
        high.append(max(up-lm-delta, 0))
    return float(np.mean(low)), float(np.mean(high))


def paired_keys(row):
    required = ("task", "init_state", "scene_seed", "repeat", "seed")
    if any(k not in row for k in required) or type(row.get("success")) is not bool:
        raise ValueError("Historical row lacks a complete paired identity or boolean outcome")
    return tuple(row[k] for k in required)


def replay():
    folder = ROOT / "vla-regressions/results/pilot"
    def load(name):
        rows = [json.loads(line) for line in (folder/name).read_text().splitlines() if line.strip()]
        indexed = {paired_keys(r): r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError("Duplicate historical pair identity")
        return indexed
    old = load("libero_10-fp32.jsonl")
    output = []
    for variant in ("bf16", "w4", "w3", "steps2"):
        new = load(f"libero_10-{variant}.jsonl")
        if set(old) != set(new):
            raise ValueError("Historical pair identities differ: " + variant)
        states = sorted({k[:3] for k in old})
        index = {s: i for i, s in enumerate(states)}
        certificate = Certificate(len(states))
        for key in sorted(old):
            certificate.add(index[key[:3]], [int(old[key]["success"])-int(new[key]["success"])])
        lower, upper = certificate.loss_bounds()
        output.append({
            "variant": variant, "states": len(states), "paired_observations": len(old),
            "estimated_loss": float(np.mean(np.maximum(
                (certificate.plus-certificate.minus)/certificate.n - certificate.delta, 0))),
            "loss_lower": lower, "loss_upper": upper, "decision": certificate.decision(),
            "minimum_repeats": int(certificate.n.min()), "maximum_repeats": int(certificate.n.max()),
            "scope": "descriptive replay; not new closed-loop validation",
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--permute-states", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.budget < 20 or args.budget % 20:
        raise ValueError("Positive repetitions and a budget divisible by 20 are required")
    args.output.mkdir(parents=True, exist_ok=False)
    files = ["PROTOCOL.md", "admission.py", "run_admission.py", "test_admission.py"]
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repetitions": args.repetitions, "budget_paired_samples": args.budget,
        "batch": 10, "states": 20, "alpha": .05, "delta": .02, "epsilon": .02,
        "seed": args.seed, "permuted_states": args.permute_states,
        "sources": {f: sha256((HERE/f).read_bytes()).hexdigest() for f in files},
        "data_kind": "synthetic simulation plus separately identified historical replay",
    }
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for f in files:
        (args.output/f).write_bytes((HERE/f).read_bytes())
    observations, traces = [], []
    started = time.monotonic()
    for scenario_index, case in enumerate(scenarios()):
        true_loss = float(np.mean(np.maximum(case["plus"]-case["minus"]-.02, 0)))
        for rep in range(args.repetitions):
            # Per-state pre-generated streams give common inputs across allocations.
            rng = np.random.default_rng(args.seed + 10000*scenario_index + rep)
            permutation = rng.permutation(20) if args.permute_states else np.arange(20)
            positive, negative = case["plus"][permutation], case["minus"][permutation]
            streams = rng.random((20, args.budget))
            differences = np.where(streams < positive[:, None], 1, np.where(
                streams < (positive+negative)[:, None], -1, 0)).astype(np.int8)
            for strategy in ["uniform", "adaptive"]:
                cert = Certificate(20)
                for step in range(0, args.budget, 10):
                    state = cert.choose(strategy, step//10)
                    count = min(10, args.budget-step)
                    offset = cert.n[state]
                    cert.add(state, differences[state, offset:offset+count])
                    decision = cert.decision()
                    if rep == 0 and (step % 200 == 0 or decision != "uncertain"):
                        lo, hi = cert.loss_bounds()
                        traces.append({"scenario": case["name"], "strategy": strategy,
                                       "pairs": step+count, "lower": lo, "upper": hi,
                                       "true_loss": true_loss, "epsilon": .02})
                    if decision != "uncertain":
                        break
                lo, hi = cert.loss_bounds()
                observations.append({
                    "scenario": case["name"], "repetition": rep, "strategy": strategy,
                    "true_loss": true_loss, "decision": decision,
                    "pairs": int(cert.n.sum()), "policy_episodes": int(2*cert.n.sum()),
                    "lower": lo, "upper": hi,
                    "false_accept": decision == "accept" and true_loss > .02,
                    "false_reject": decision == "reject" and true_loss <= .02,
                    "counts": cert.n.tolist(),
                })
            # Fixed-budget reference uses uniform counts and a single terminal look.
            n = np.full(20, args.budget//20, dtype=int)
            fixed_data = differences[:, :args.budget//20]
            lo, hi = fixed_bounds(n, (fixed_data == 1).sum(1), (fixed_data == -1).sum(1), .05, .02)
            decision = "accept" if hi <= .02 else "reject" if lo > .02 else "uncertain"
            observations.append({
                "scenario": case["name"], "repetition": rep, "strategy": "fixed_exact",
                "true_loss": true_loss, "decision": decision, "pairs": int(n.sum()),
                "policy_episodes": int(2*n.sum()), "lower": lo, "upper": hi,
                "false_accept": decision == "accept" and true_loss > .02,
                "false_reject": decision == "reject" and true_loss <= .02,
            })
        print(case["name"], "completed", args.repetitions, "repetitions; elapsed",
              round(time.monotonic()-started, 1), "seconds", flush=True)
    (args.output/"observations.jsonl").write_text(
        "".join(json.dumps(row)+"\n" for row in observations))
    (args.output/"traces.json").write_text(json.dumps(traces, indent=2)+"\n")
    summary = []
    for case in scenarios():
        for strategy in ["uniform", "adaptive", "fixed_exact"]:
            rows = [r for r in observations if r["scenario"] == case["name"] and r["strategy"] == strategy]
            summary.append({
                "scenario": case["name"], "strategy": strategy,
                "true_loss": rows[0]["true_loss"], "repetitions": len(rows),
                "accept": sum(r["decision"] == "accept" for r in rows),
                "reject": sum(r["decision"] == "reject" for r in rows),
                "uncertain": sum(r["decision"] == "uncertain" for r in rows),
                "false_accept": sum(r["false_accept"] for r in rows),
                "false_reject": sum(r["false_reject"] for r in rows),
                "median_pairs": float(np.median([r["pairs"] for r in rows])),
            })
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"historical-replay.json").write_text(json.dumps(replay(), indent=2)+"\n")
    print("Completed screening; no new robot inference performed.", flush=True)


if __name__ == "__main__":
    main()

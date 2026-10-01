"""Frozen small cohort comparing marginal and joint standard mixture bounds."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import time
import numpy as np
from admission import Certificate
from joint_admission import JointCertificate, joint_cs
from run_admission import scenarios, fixed_bounds

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = ["JOINT_PROTOCOL.md", "admission.py", "joint_admission.py",
             "run_admission.py", "run_joint_comparison.py", "test_joint_admission.py"]
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "small development bound-comparison cohort",
        "repetitions": 30, "seed": 161803, "states": 20, "budget_pairs": 5000,
        "alpha": .05, "delta": .02, "epsilon": .02, "permuted_states": True,
        "sources": {name: sha256((HERE/name).read_bytes()).hexdigest() for name in files},
    }
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for name in files:
        (args.output/name).write_bytes((HERE/name).read_bytes())
    rows = []
    started = time.monotonic()
    for index, case in enumerate(scenarios()):
        true_loss = float(np.mean(np.maximum(case["plus"]-case["minus"]-.02, 0)))
        for repetition in range(30):
            rng = np.random.default_rng(161803+10000*index+repetition)
            perm = rng.permutation(20)
            p, m = case["plus"][perm], case["minus"][perm]
            streams = rng.random((20, 5000))
            differences = np.where(streams < p[:, None], 1,
                                   np.where(streams < (p+m)[:, None], -1, 0)).astype(np.int8)
            for method, cls in (("marginal", Certificate), ("joint", JointCertificate)):
                for allocation in ("uniform", "adaptive"):
                    cert = cls(20)
                    for step in range(0, 5000, 10):
                        state = cert.choose(allocation, step//10)
                        offset = cert.n[state]
                        cert.add(state, differences[state, offset:offset+10])
                        if cert.decision() != "uncertain":
                            break
                    lower, upper = cert.loss_bounds()
                    decision = cert.decision()
                    rows.append({
                        "scenario": case["name"], "repetition": repetition,
                        "method": method, "allocation": allocation,
                        "true_loss": true_loss, "decision": decision,
                        "lower": lower, "upper": upper, "pairs": int(cert.n.sum()),
                        "false_accept": decision == "accept" and true_loss > .02,
                        "false_reject": decision == "reject" and true_loss <= .02,
                        "counts": cert.n.tolist(), "positive": cert.plus.tolist(),
                        "negative": cert.minus.tolist(),
                    })
            data = differences[:, :250]
            lower, upper = fixed_bounds(np.full(20, 250), (data == 1).sum(1),
                                        (data == -1).sum(1), .05, .02)
            decision = "accept" if upper <= .02 else "reject" if lower > .02 else "uncertain"
            rows.append({
                "scenario": case["name"], "repetition": repetition,
                "method": "fixed_exact", "allocation": "uniform", "true_loss": true_loss,
                "decision": decision, "lower": lower, "upper": upper, "pairs": 5000,
                "false_accept": decision == "accept" and true_loss > .02,
                "false_reject": decision == "reject" and true_loss <= .02})
        print(case["name"], "completed;", round(time.monotonic()-started, 1), "seconds", flush=True)
    (args.output/"observations.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    summary = []
    for case in scenarios():
        for method, allocation in [("marginal", "uniform"), ("marginal", "adaptive"),
                                   ("joint", "uniform"), ("joint", "adaptive"),
                                   ("fixed_exact", "uniform")]:
            selected = [r for r in rows if r["scenario"] == case["name"]
                        and r["method"] == method and r["allocation"] == allocation]
            summary.append({
                "scenario": case["name"], "method": method, "allocation": allocation,
                "repetitions": len(selected),
                **{d: sum(r["decision"] == d for r in selected)
                   for d in ("accept", "reject", "uncertain")},
                "false_accept": sum(r["false_accept"] for r in selected),
                "false_reject": sum(r["false_reject"] for r in selected),
                "median_pairs": float(np.median([r["pairs"] for r in selected])),
            })
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    cost = []
    for states in (20, 100):
        n = 1
        while max(joint_cs(n, 0, 0, .05/states)[1]-.02, 0) > .02:
            n += 1
        cost.append({"states": states, "equal_repeats": n, "pairs": states*n,
                     "scope": "Specific joint bound with zero observed discordance"})
    (args.output/"zero-discordance-cost.json").write_text(json.dumps(cost, indent=2)+"\n")
    print("Bound comparison completed; no robot inference in this run.", flush=True)


if __name__ == "__main__":
    main()

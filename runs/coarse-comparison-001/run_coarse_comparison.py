"""Collect a fresh comparison with the pooled-discordance sufficient gate."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from admission import Certificate
from joint_admission import JointCertificate
from coarse_admission import coarse_bounds
from run_admission import scenarios, fixed_bounds

HERE = Path(__file__).resolve().parent


def cases():
    result = scenarios()
    result.append({"name": "compensated_harm",
                   "plus": np.r_[np.full(5, .35), np.zeros(15)],
                   "minus": np.r_[np.zeros(5), np.full(15, .13)]})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    names = ("COARSE_PROTOCOL.md", "admission.py", "joint_admission.py",
             "coarse_admission.py", "run_admission.py", "run_coarse_comparison.py",
             "test_coarse_admission.py")
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(),
                "scope": "synthetic development only", "seed": 424243,
                "repetitions": 30, "states": 20, "cap_pairs": 5000,
                "alpha": .05, "delta": .02, "epsilon": .02,
                "sources": {n: sha256((HERE/n).read_bytes()).hexdigest() for n in names}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for name in names:
        (args.output/name).write_bytes((HERE/name).read_bytes())
    rows = []
    for index, case in enumerate(cases()):
        risk = float(np.maximum(case["plus"]-case["minus"]-.02, 0).mean())
        for rep in range(30):
            rng = np.random.default_rng(424243+10000*index+rep)
            perm = rng.permutation(20)
            p, m = case["plus"][perm], case["minus"][perm]
            random = rng.random((20, 5000))
            stream = np.where(random < p[:, None], 1,
                              np.where(random < (p+m)[:, None], -1, 0)).astype(np.int8)
            outcomes = []
            for method, cls in (("marginal", Certificate), ("joint", JointCertificate)):
                for allocation in ("uniform", "adaptive"):
                    cert = cls(20)
                    for step in range(0, 5000, 10):
                        state = cert.choose(allocation, step//10)
                        offset = cert.n[state]
                        cert.add(state, stream[state, offset:offset+10])
                        if cert.decision() != "uncertain":
                            break
                    lo, hi = cert.loss_bounds()
                    outcomes.append((method, allocation, lo, hi, cert.n, cert.plus, cert.minus))
            n = np.full(20, 250)
            hp, gp = (stream[:, :250] == 1).sum(1), (stream[:, :250] == -1).sum(1)
            lo, hi = fixed_bounds(n, hp, gp, .05, .02)
            outcomes.append(("fixed_exact", "uniform", lo, hi, n, hp, gp))
            selected = np.random.default_rng(104729+10000*index+rep).integers(0, 20, 5000)
            n, hp, gp = [np.zeros(20, dtype=int) for _ in range(3)]
            for offset in range(0, 5000, 10):
                for state in selected[offset:offset+10]:
                    d = stream[state, n[state]]
                    n[state] += 1
                    hp[state] += d == 1
                    gp[state] += d == -1
                lo, hi = coarse_bounds(int(n.sum()), int(hp.sum()), int(gp.sum()))
                if hi <= .02 or lo > .02:
                    break
            outcomes.append(("coarse", "iid_states", lo, hi, n, hp, gp))
            for method, allocation, lo, hi, n, hp, gp in outcomes:
                decision = "accept" if hi <= .02 else "reject" if lo > .02 else "uncertain"
                rows.append({"scenario": case["name"], "repetition": rep,
                             "method": method, "allocation": allocation, "true_loss": risk,
                             "aggregate_loss": float((p-m).mean()),
                             "decision": decision, "lower": lo, "upper": hi,
                             "pairs": int(n.sum()), "counts": n.tolist(),
                             "positive": hp.tolist(), "negative": gp.tolist(),
                             "false_accept": decision == "accept" and risk > .02,
                             "false_reject": decision == "reject" and risk <= .02})
        print(case["name"], "complete", flush=True)
    (args.output/"observations.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    summary = []
    for case in cases():
        for method, allocation in (("marginal", "uniform"), ("marginal", "adaptive"),
                                   ("joint", "uniform"), ("joint", "adaptive"),
                                   ("fixed_exact", "uniform"), ("coarse", "iid_states")):
            selected = [r for r in rows if r["scenario"] == case["name"]
                        and r["method"] == method and r["allocation"] == allocation]
            summary.append({"scenario": case["name"], "method": method, "allocation": allocation,
                            "repetitions": len(selected),
                            **{d: sum(r["decision"] == d for r in selected)
                               for d in ("accept", "reject", "uncertain")},
                            "median_pairs": float(np.median([r["pairs"] for r in selected])),
                            "false_accept": sum(r["false_accept"] for r in selected),
                            "false_reject": sum(r["false_reject"] for r in selected)})
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()

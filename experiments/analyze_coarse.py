"""Audit the frozen six-method cohort, including IID allocation and first stops."""
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
import math
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln

BASE = Path(__file__).resolve().parent.parent
RUN = BASE/"runs/coarse-comparison-001"
sys.path.insert(0, str(RUN))
from admission import bernoulli_cs
from joint_admission import joint_cs
from run_admission import fixed_bounds
from run_coarse_comparison import cases


@lru_cache(maxsize=200000)
def independent_beta_interval(n, k, alpha):
    """SciPy root inversion, independent of the collector's lgamma/bisection."""
    if n == 0:
        return 0., 1.
    constant = float(betaln(k+.5, n-k+.5)-betaln(.5, .5)+math.log(alpha))
    if k == 0:
        return 0., -math.expm1(constant/n)
    if k == n:
        return math.exp(constant/n), 1.
    f = lambda p: constant-k*math.log(p)-(n-k)*math.log1p(-p)
    return (brentq(f, 1e-300, k/n, xtol=1e-14),
            brentq(f, k/n, 1-1e-15, xtol=1e-14))


def verdict(lower, upper):
    return "accept" if upper <= .02 else "reject" if lower > .02 else "uncertain"


def main():
    manifest = json.loads((RUN/"manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        assert sha256((RUN/name).read_bytes()).hexdigest() == digest, name
    rows = [json.loads(line) for line in (RUN/"observations.jsonl").read_text().splitlines()]
    assert len(rows) == 8*30*6
    assert len({(r["scenario"], r["repetition"], r["method"], r["allocation"])
                for r in rows}) == len(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["repetition"]), []).append(row)
    for index, case in enumerate(cases()):
        risk = float(np.maximum(case["plus"]-case["minus"]-.02, 0).mean())
        for rep in range(30):
            rng = np.random.default_rng(424243+10000*index+rep)
            perm = rng.permutation(20)
            p, m = case["plus"][perm], case["minus"][perm]
            random = rng.random((20, 5000))
            stream = np.where(random < p[:, None], 1,
                              np.where(random < (p+m)[:, None], -1, 0))
            selected = grouped[(case["name"], rep)]
            assert len(selected) == 6
            for row in selected:
                n = np.array(row["counts"])
                assert all(type(c) is int and 0 <= c <= 5000 for c in row["counts"])
                assert len(n) == 20 and n.sum() == row["pairs"]
                assert 0 < row["pairs"] <= 5000 and row["pairs"] % 10 == 0
                hp = np.array([(stream[s, :c] == 1).sum() for s, c in enumerate(n)])
                gp = np.array([(stream[s, :c] == -1).sum() for s, c in enumerate(n)])
                assert hp.tolist() == row["positive"] and gp.tolist() == row["negative"]
                assert abs(row["true_loss"]-risk) < 1e-12
                assert abs(row["aggregate_loss"]-(p-m).mean()) < 1e-12
                if row["method"] == "fixed_exact":
                    assert (n == 250).all()
                    lower, upper = fixed_bounds(n, hp, gp, .05, .02)
                elif row["method"] == "coarse":
                    iid = np.random.default_rng(104729+10000*index+rep).integers(0, 20, 5000)
                    assert np.bincount(iid[:row["pairs"]], minlength=20).tolist() == n.tolist()
                    used = np.zeros(20, dtype=int)
                    harmful = beneficial = 0
                    for count, state in enumerate(iid[:row["pairs"]], start=1):
                        d = stream[state, used[state]]
                        used[state] += 1
                        harmful += d == 1
                        beneficial += d == -1
                        if count % 10:
                            continue
                        lp, up = independent_beta_interval(count, int(harmful), .025)
                        _, um = independent_beta_interval(count, int(beneficial), .025)
                        lower, upper = max(lp-um-.02, 0.), .98*up
                        if count < row["pairs"]:
                            assert verdict(lower, upper) == "uncertain"
                    # Compare independently inverted roots with archived collector endpoints.
                else:
                    intervals = []
                    for c, h, g in zip(n, hp, gp):
                        if row["method"] == "joint":
                            interval = joint_cs(int(c), int(h), int(g), .05/20)
                        else:
                            lp, up = independent_beta_interval(int(c), int(h), .05/40)
                            lm, um = independent_beta_interval(int(c), int(g), .05/40)
                            interval = lp-um, up-lm
                        intervals.append(interval)
                    lower, upper = [float(np.maximum(np.array(intervals)[:, j]-.02, 0).mean())
                                    for j in (0, 1)]
                assert abs(lower-row["lower"]) < 1e-9
                assert abs(upper-row["upper"]) < 1e-9
                decision = verdict(lower, upper)
                assert decision == row["decision"]
                assert row["false_accept"] == (decision == "accept" and risk > .02)
                assert row["false_reject"] == (decision == "reject" and risk <= .02)
    summaries = json.loads((RUN/"summary.json").read_text())
    assert len(summaries) == 48
    for summary in summaries:
        selected = [r for r in rows if all(r[k] == summary[k]
                    for k in ("scenario", "method", "allocation"))]
        assert len(selected) == summary["repetitions"] == 30
        decisions = Counter(r["decision"] for r in selected)
        for d in ("accept", "reject", "uncertain"):
            assert summary[d] == decisions[d]
        for d in ("false_accept", "false_reject"):
            assert summary[d] == sum(r[d] for r in selected)
        assert summary["median_pairs"] == float(np.median([r["pairs"] for r in selected]))
    zero_pairs = next(n for n in range(1, 5000)
                      if .98*independent_beta_interval(n, 0, .025)[1] <= .02)
    result = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source_hashes_verified": True, "consumed_outcomes_reconstructed": True,
        "iid_state_prefix_counts_verified": True, "coarse_first_inspection_stop_verified": True,
        "bernoulli_roots_independently_verified": True,
        "strategy_rows": len(rows), "independent_scenario_repetitions": 240,
        "paired_strategies_per_repetition": 6,
        "false_accept": sum(r["false_accept"] for r in rows),
        "false_reject": sum(r["false_reject"] for r in rows),
        "zero_errors_out_of_30_two_sided_exact_95_upper": 1-.025**(1/30),
        "pooled_all_zero_discordance_pairs_first_integer": zero_pairs,
        "pooled_all_zero_discordance_pairs_first_ten_pair_inspection":
            10*math.ceil(zero_pairs/10),
        "run_sha256": {name: sha256((RUN/name).read_bytes()).hexdigest()
                       for name in ("manifest.json", "observations.jsonl", "summary.json")},
        "scope": "Synthetic development; no operational time or independent pooled strategy inference.",
    }
    (BASE/"analysis/coarse-verified-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    print("scenario: method/allocation A/R/U median pairs")
    for s in summaries:
        print(s["scenario"], s["method"]+"/"+s["allocation"],
              f'{s["accept"]}/{s["reject"]}/{s["uncertain"]}', s["median_pairs"])


if __name__ == "__main__":
    main()

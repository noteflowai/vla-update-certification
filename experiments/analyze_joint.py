"""Audit the archived paired cohort and summarize bound-specific tradeoffs."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RUN = BASE/"runs/joint-comparison-001"
# Use the collected source snapshot, not subsequently modified collectors.
sys.path.insert(0, str(RUN))
from admission import bernoulli_cs
from joint_admission import joint_cs
from run_admission import scenarios, fixed_bounds


def main():
    manifest = json.loads((RUN/"manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        assert sha256((RUN/name).read_bytes()).hexdigest() == digest, name
    rows = [json.loads(line) for line in (RUN/"observations.jsonl").read_text().splitlines()]
    assert len(rows) == 7*30*5
    assert len({(r["scenario"], r["repetition"], r["method"], r["allocation"])
                for r in rows}) == len(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["repetition"]), []).append(row)
    for index, case in enumerate(scenarios()):
        true_loss = float(np.maximum(case["plus"]-case["minus"]-.02, 0).mean())
        for rep in range(30):
            rng = np.random.default_rng(161803+10000*index+rep)
            perm = rng.permutation(20)
            p, m = case["plus"][perm], case["minus"][perm]
            random = rng.random((20, 5000))
            stream = np.where(random < p[:, None], 1,
                              np.where(random < (p+m)[:, None], -1, 0))
            selected = grouped[(case["name"], rep)]
            assert len(selected) == 5
            for row in selected:
                assert abs(row["true_loss"]-true_loss) < 1e-12
                counts = row.get("counts", [250]*20)
                assert len(counts) == 20 and sum(counts) == row["pairs"]
                assert all(type(n) is int and 0 <= n <= 5000 for n in counts)
                assert 0 < row["pairs"] <= 5000
                plus = np.array([int((stream[s, :n] == 1).sum())
                                 for s, n in enumerate(counts)])
                minus = np.array([int((stream[s, :n] == -1).sum())
                                  for s, n in enumerate(counts)])
                if row["method"] == "fixed_exact":
                    lower, upper = fixed_bounds(np.array(counts), plus, minus, .05, .02)
                else:
                    assert plus.tolist() == row["positive"]
                    assert minus.tolist() == row["negative"]
                    intervals = []
                    for n, kp, km in zip(counts, plus, minus):
                        if row["method"] == "joint":
                            interval = joint_cs(int(n), int(kp), int(km), .05/20)
                        else:
                            lp, up = bernoulli_cs(int(n), int(kp), .05/40)
                            lm, um = bernoulli_cs(int(n), int(km), .05/40)
                            interval = (lp-um, up-lm)
                        intervals.append(interval)
                    lower, upper = [float(np.maximum(np.array(intervals)[:, j]-.02, 0).mean())
                                    for j in (0, 1)]
                assert abs(lower-row["lower"]) < 1e-10
                assert abs(upper-row["upper"]) < 1e-10
                verdict = "accept" if upper <= .02 else "reject" if lower > .02 else "uncertain"
                assert verdict == row["decision"]
                assert row["false_accept"] == (verdict == "accept" and true_loss > .02)
                assert row["false_reject"] == (verdict == "reject" and true_loss <= .02)
    summaries = json.loads((RUN/"summary.json").read_text())
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
    savings = []
    for scenario in ("sparse_harm", "heterogeneous_harm"):
        for method in ("marginal", "joint"):
            matched = []
            for rep in range(30):
                selected = {r["allocation"]: r for r in grouped[(scenario, rep)]
                            if r["method"] == method}
                if all(r["decision"] == "reject" for r in selected.values()):
                    matched.append(1-selected["adaptive"]["pairs"]/selected["uniform"]["pairs"])
            rng = np.random.default_rng(20260930)
            draws = np.median(rng.choice(matched, size=(10000, len(matched))), axis=1)
            savings.append({"scenario": scenario, "method": method,
                            "matched_correct_rejections": len(matched),
                            "median_paired_fraction_saved": float(np.median(matched)),
                            "exploratory_percentile_bootstrap_95": np.quantile(draws, [.025, .975]).tolist()})
    cost = json.loads((RUN/"zero-discordance-cost.json").read_text())
    for row in cost:
        n, states = row["equal_repeats"], row["states"]
        # Closed-form joint endpoint for all-zero discordances is independent
        # of the profile/bisection code used to collect the cohort.
        exact_upper = lambda k: 1-(.05/states/(2*k+1))**(1/k)
        assert exact_upper(n) <= .04 < exact_upper(n-1)
        assert row["pairs"] == n*states
    result = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source_hashes_verified": True, "consumed_outcomes_reconstructed": True,
        "strategy_rows": len(rows), "independent_scenario_repetitions": 210,
        "paired_strategies_per_repetition": 5,
        "false_accept": sum(r["false_accept"] for r in rows),
        "false_reject": sum(r["false_reject"] for r in rows),
        "zero_errors_out_of_30_two_sided_exact_95_upper": 1-.025**(1/30),
        "paired_savings": savings, "zero_discordance_cost": cost,
        "run_sha256": {name: sha256((RUN/name).read_bytes()).hexdigest()
                       for name in ("manifest.json", "observations.jsonl", "summary.json")},
        "scope": "Synthetic development; paired strategies are not independent trials.",
    }
    (BASE/"analysis/joint-verified-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    methods = [("marginal", "uniform"), ("marginal", "adaptive"),
               ("joint", "uniform"), ("joint", "adaptive"), ("fixed_exact", "uniform")]
    labels = ["Marginal\nuniform", "Marginal\nadaptive", "Joint\nuniform",
              "Joint\nadaptive", "Terminal\nexact"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ("#476b9e", "#2a826d")
    for i, scenario in enumerate(("sparse_harm", "heterogeneous_harm")):
        values = [next(s["median_pairs"] for s in summaries
                       if (s["method"], s["allocation"]) == pair
                       and s["scenario"] == scenario) for pair in methods]
        bars = ax.bar(np.arange(5)+(i-.5)*.38, values, width=.36,
                      color=colors[i], label=scenario.replace("_", " ").capitalize())
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(np.arange(5), labels)
    ax.set_ylim(0, 5600)
    ax.set_ylabel("Median paired samples (two policy episodes per pair)")
    ax.set_title("Synthetic correct-rejection cost: 30/30 rejects in each cell")
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(BASE/f"analysis/joint-rejection-cost.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Verified 1,050 paired strategy rows, 210 independent scenario repetitions.")


if __name__ == "__main__":
    main()

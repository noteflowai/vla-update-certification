"""Verify archived observations and render an honest research-screen report."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import platform
import sys
import numpy as np
import scipy
from scipy.stats import beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from admission import bernoulli_cs
from run_admission import scenarios

BASE = Path(__file__).resolve().parent.parent


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_admission(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        assert sha256((folder/name).read_bytes()).hexdigest() == expected, name
    rows = load_rows(folder/"observations.jsonl")
    assert len(rows) == 7*3*manifest["repetitions"]
    identities = {(r["scenario"], r["strategy"], r["repetition"]) for r in rows}
    assert len(identities) == len(rows)
    for row in rows:
        lower, upper = row["lower"], row["upper"]
        assert 0 <= lower <= upper <= 1
        expected = "accept" if upper <= .02 else "reject" if lower > .02 else "uncertain"
        assert row["decision"] == expected
        assert row["false_accept"] == (expected == "accept" and row["true_loss"] > .02)
        assert row["false_reject"] == (expected == "reject" and row["true_loss"] <= .02)
        assert 0 < row["pairs"] <= manifest["budget_paired_samples"]
        assert row["policy_episodes"] == 2*row["pairs"]
        if "counts" in row:
            assert sum(row["counts"]) == row["pairs"]
            assert len(row["counts"]) == manifest["states"]
    # Reconstruct final consumed outcomes from the recorded seed and counts.
    # This checks the saved bounds against inputs rather than just their summary.
    for scenario_index, case in enumerate(scenarios()):
        for repetition in range(manifest["repetitions"]):
            rng = np.random.default_rng(manifest.get("seed", 271828)
                                        + 10000*scenario_index + repetition)
            perm = rng.permutation(20) if manifest.get("permuted_states", False) else np.arange(20)
            positive, negative = case["plus"][perm], case["minus"][perm]
            streams = rng.random((20, manifest["budget_paired_samples"]))
            differences = np.where(streams < positive[:, None], 1, np.where(
                streams < (positive+negative)[:, None], -1, 0))
            selected = [r for r in rows if r["scenario"] == case["name"]
                        and r["repetition"] == repetition]
            for row in selected:
                counts = row.get("counts", [manifest["budget_paired_samples"]//20]*20)
                lower, upper = [], []
                for state, n in enumerate(counts):
                    values = differences[state, :n]
                    intervals = []
                    for indicator in (1, -1):
                        k = int((values == indicator).sum())
                        a = manifest["alpha"]/(2*20)
                        if row["strategy"] == "fixed_exact":
                            interval = ((0. if k == 0 else float(beta.ppf(a/2, k, n-k+1))),
                                        (1. if k == n else float(beta.ppf(1-a/2, k+1, n-k))))
                        else:
                            interval = bernoulli_cs(n, k, a)
                        intervals.append(interval)
                    lower.append(max(intervals[0][0]-intervals[1][1]-.02, 0))
                    upper.append(max(intervals[0][1]-intervals[1][0]-.02, 0))
                assert abs(np.mean(lower)-row["lower"]) < 1e-10
                assert abs(np.mean(upper)-row["upper"]) < 1e-10
    for summary in json.loads((folder/"summary.json").read_text()):
        selected = [r for r in rows if r["scenario"] == summary["scenario"]
                    and r["strategy"] == summary["strategy"]]
        counts = Counter(r["decision"] for r in selected)
        assert len(selected) == summary["repetitions"]
        for decision in ("accept", "reject", "uncertain"):
            assert counts[decision] == summary[decision]
        assert float(np.median([r["pairs"] for r in selected])) == summary["median_pairs"]
        for error in ("false_accept", "false_reject"):
            assert sum(r[error] for r in selected) == summary[error]
    return rows


def verify_skills(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        assert sha256((folder/name).read_bytes()).hexdigest() == expected
    assert manifest["model_trials"] == manifest["native_client_trials"] == 0
    tasks = json.loads((folder/"tasks.json").read_text())
    rows = load_rows(folder/"observations.jsonl")
    assert len(rows) == 42
    assert len({(r["skill"], r["task"], r["condition"]) for r in rows}) == 42
    for row in rows:
        task = next(t for t in tasks[row["skill"]] if t["name"] == row["task"])
        execution = row["execution"]
        if row["skill"].startswith("cloudflare"):
            passed = execution["returncode"] == 0 and task["expected"] in execution["stdout"]
        else:
            try:
                parsed = json.loads(execution["stdout"])
            except json.JSONDecodeError:
                parsed = None
            passed = execution["returncode"] == 0 and parsed == task["expected"]
        assert passed == row["actual_outcome_pass"]
    for summary in json.loads((folder/"summary.json").read_text()):
        selected = [r for r in rows if r["condition"] == summary["condition"]]
        assert len(selected) == summary["controls"]
        assert sum(r["actual_outcome_pass"] for r in selected) == summary["passed"]
    return rows


def zero_discordance_cost():
    rows = []
    for states in (10, 20, 100, 1000):
        alpha = .05/(2*states)
        for method in ("anytime_mixture", "fixed_exact"):
            def upper(n):
                return (bernoulli_cs(n, 0, alpha)[1] if method == "anytime_mixture"
                        else float(beta.ppf(1-alpha/2, 1, n)))
            lo, hi = 0, 1
            while upper(hi) > .04:
                hi *= 2
            while hi-lo > 1:
                mid = (lo+hi)//2
                if upper(mid) <= .04:
                    hi = mid
                else:
                    lo = mid
            rows.append({"states": states, "method": method,
                         "minimum_equal_repeats_all_discordances_zero": hi,
                         "paired_observations": states*hi, "policy_episodes": 2*states*hi,
                         "loss_upper": max(upper(hi)-.02, 0)})
    return {
        "scope": "Balanced zero-observed-discordance certificate cost for these "
                 "specific bounds; not an information-theoretic lower bound or expected cost.",
        "alpha": .05, "delta": .02, "epsilon": .02, "rows": rows}


def paired_savings(rows, seed):
    results = []
    for scenario in ("sparse_harm", "heterogeneous_harm"):
        matched = {}
        for row in rows:
            if row["scenario"] == scenario and row["strategy"] in ("uniform", "adaptive"):
                matched.setdefault(row["repetition"], {})[row["strategy"]] = row
        pairs = [pair for pair in matched.values()
                 if all(pair[s]["decision"] == "reject" for s in ("uniform", "adaptive"))]
        ratios = np.array([1-p["adaptive"]["pairs"]/p["uniform"]["pairs"] for p in pairs])
        rng = np.random.default_rng(seed)
        bootstrap = np.median(rng.choice(ratios, size=(10000, len(ratios))), axis=1)
        results.append({
            "scenario": scenario, "matched_correct_decisions": len(pairs),
            "uniform_median_pairs": float(np.median([p["uniform"]["pairs"] for p in pairs])),
            "adaptive_median_pairs": float(np.median([p["adaptive"]["pairs"] for p in pairs])),
            "median_paired_fraction_saved": float(np.median(ratios)),
            "exploratory_percentile_bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
        })
    return results


def figure(rows, output):
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    strategies = ("uniform", "adaptive", "fixed_exact")
    colors = {"accept": "#2b8c65", "reject": "#c85648", "uncertain": "#c6cbd1"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, strategy in zip(axes, strategies):
        left = np.zeros(len(scenarios))
        for decision in colors:
            values = [sum(r["decision"] == decision for r in rows
                          if r["scenario"] == case and r["strategy"] == strategy)
                      for case in scenarios]
            ax.barh(range(len(scenarios)), values, left=left, color=colors[decision],
                    label=decision, height=.7)
            left += values
        ax.set_title(strategy.replace("_", " "))
        ax.set_xlabel("Runs out of 100")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(range(len(scenarios)), [s.replace("_", " ") for s in scenarios])
    axes[0].invert_yaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Randomized state-order development cohort — decisions at ≤5,000 pairs")
    fig.tight_layout(rect=(0, .07, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(output/f"decisions.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    scenarios = ("sparse_harm", "heterogeneous_harm")
    for i, strategy in enumerate(strategies):
        values = [np.median([r["pairs"] for r in rows if r["scenario"] == s
                            and r["strategy"] == strategy]) for s in scenarios]
        ax.bar(np.arange(2)+(i-1)*.24, values, width=.22, label=strategy.replace("_", " "))
    ax.set_xticks(np.arange(2), ["Sparse severe harm", "Heterogeneous harm"])
    ax.set_ylabel("Median paired samples (2 policy episodes per pair)")
    ax.set_title("Correct rejection cost; 100/100 decisions for each method")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output/f"rejection-cost.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    output = BASE/"analysis"
    output.mkdir(exist_ok=True)
    cohorts = {}
    for name in ("admission-001", "admission-002"):
        cohorts[name] = verify_admission(BASE/"runs"/name)
    skills = verify_skills(BASE/"runs/skill-controls-001")
    zero = zero_discordance_cost()
    (BASE/"runs/zero-discordance-cost.json").write_text(json.dumps(zero, indent=2)+"\n")
    stats = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__, "scipy": scipy.__version__,
                    "matplotlib": matplotlib.__version__},
        "independent_scenario_repetitions_per_cohort": 700,
        "strategy_observations_per_cohort": 2100,
        "note": "Three strategies share each repetition's outcome streams. "
                "Strategy rows are not independent experiments.",
        "cohorts": {name: {"rows": len(rows), "false_accept": sum(r["false_accept"] for r in rows),
                          "false_reject": sum(r["false_reject"] for r in rows),
                          "paired_savings": paired_savings(rows, 20260930)}
                    for name, rows in cohorts.items()},
        "skills": {"controls": len(skills), "passed": sum(r["actual_outcome_pass"] for r in skills),
                   "model_trials": 0, "native_client_trials": 0},
        "zero_errors_out_of_100_two_sided_exact_95_upper": float(beta.ppf(.975, 1, 100)),
        "historical_data_hashes_recorded_after_screen": {
            path.name: sha256(path.read_bytes()).hexdigest() for path in
            Path("/home/dcvuser/work/vla-regressions/results/pilot").glob("*.jsonl")},
    }
    (output/"verified-summary.json").write_text(json.dumps(stats, indent=2)+"\n")
    figure(cohorts["admission-002"], output)
    print(json.dumps(stats["cohorts"], indent=2))
    print("Archived hashes, observation decisions and summary denominators verified.")


if __name__ == "__main__":
    main()

"""Exact constructed distributions, not synthetic trials or collected outcomes."""
from fractions import Fraction as F
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Columns: both succeed, old only, new only, both fail.
    stable = [[F(45, 100), F(5, 100), F(5, 100), F(45, 100)] for _ in range(20)]
    mixed = ([[F(45, 100), F(10, 100), F(0), F(45, 100)] for _ in range(10)] +
             [[F(45, 100), F(0), F(10, 100), F(45, 100)] for _ in range(10)])
    delta, epsilon = F(2, 100), F(2, 100)
    facts = {}
    for name, rows in (("stable", stable), ("mixed", mixed)):
        assert all(sum(row) == 1 for row in rows)
        pooled = [sum(row[k] for row in rows) / 20 for k in range(4)]
        differences = [row[1] - row[2] for row in rows]
        risk = sum(max(d - delta, F(0)) for d in differences) / 20
        facts[name] = {
            "joint_probabilities_by_state": [[str(v) for v in row] for row in rows],
            "pooled_joint_probabilities": [str(v) for v in pooled],
            "aggregate_old_success": str(pooled[0] + pooled[1]),
            "aggregate_new_success": str(pooled[0] + pooled[2]),
            "statewise_differences": [str(v) for v in differences],
            "statewise_regression_risk": str(risk)}
    assert facts["stable"]["pooled_joint_probabilities"] == facts["mixed"]["pooled_joint_probabilities"]
    assert facts["stable"]["statewise_regression_risk"] == "0"
    assert facts["mixed"]["statewise_regression_risk"] == "1/25"
    facts.update(scope="Exact constructed probability distributions; no model outcomes "
                 "or sampled trials. An explanatory identifiability example, not new statistical theory.",
                 states=20, delta=str(delta), epsilon=str(epsilon))
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.0),
                             gridspec_kw={"width_ratios": [1.3, 1.3, .8]})
    orange, blue, gray = "#c96d2b", "#287ca4", "#5f6670"
    states = np.arange(1, 21)
    differences = np.array([float(F(v)) for v in facts["mixed"]["statewise_differences"]])
    axes[0].bar(states, differences, color=[orange] * 10 + [blue] * 10, width=.8)
    axes[0].plot(states, [0] * 20, color=gray, marker=".", label="Stable: all differences zero")
    axes[0].axhline(float(delta), color=gray, linestyle="--", linewidth=.8, label=r"Tolerance $\delta=.02$")
    axes[0].set(xlabel="State index", ylabel="Old − new success probability",
                title="(a) Different statewise effects", xticks=[1, 5, 10, 15, 20], ylim=(-.125, .145))
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    pooled = np.array([float(F(v)) for v in facts["stable"]["pooled_joint_probabilities"]])
    labels = ["Both\nsucceed", "Old\nonly", "New\nonly", "Both\nfail"]
    x = np.arange(4)
    axes[1].bar(x - .18, pooled, .34, color=gray, label="Stable")
    axes[1].bar(x + .18, pooled, .34, color=orange, label="Mixed")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Paired-outcome probability",
                title="(b) Identical pooled outcomes", ylim=(0, .56))
    axes[1].legend(fontsize=8, frameon=False)
    risks = [float(F(facts[n]["statewise_regression_risk"])) for n in ("stable", "mixed")]
    axes[2].bar([0, 1], risks, color=[gray, orange], width=.5)
    axes[2].axhline(float(epsilon), color=blue, linestyle="--", linewidth=1)
    axes[2].text(-.35, .021, r"Budget $\epsilon=.02$", color=blue, fontsize=8)
    axes[2].set(xticks=[0, 1], xticklabels=["Stable", "Mixed"], ylim=(0, .055),
                ylabel=r"Statewise risk $R_\delta$", title="(c) Opposite budget status")
    for i, risk in enumerate(risks):
        axes[2].text(i, risk + .002, f"{risk:.2f}", ha="center", fontsize=8)
    fig.subplots_adjust(left=.065, right=.985, bottom=.23, top=.85, wspace=.45)
    fig.text(.5, .035, "Constructed distributions: both policies have 50% aggregate success in both cases; no collected outcomes.",
             ha="center", fontsize=8, color=gray)
    folder = ROOT / "analysis"
    fig.savefig(folder / "aggregate-identifiability.svg")
    fig.savefig(folder / "aggregate-identifiability.png", dpi=180)
    plt.close(fig)
    (folder / "aggregate-identifiability.json").write_text(json.dumps(facts, indent=2) + "\n")


if __name__ == "__main__":
    main()

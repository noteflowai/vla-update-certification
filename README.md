# Statewise regression budgets for VLA updates — development materials

Public development study for **The Cost of Certifying VLA Updates under Statewise Regression Budgets**. The manuscript is an anonymous methods-review draft, not an archival submission, acceptance, or completed native efficacy study.

The six-method development comparison contains 1,440 strategy records from 240 independent synthetic case repetitions (30 per scenario), with paired methods within each repetition. Method rankings depend on the scenario; all methods remain uncertain near the budget boundary. Zero observed errors in 30 repetitions still permits an 11.57% one-sided 95% upper bound. These observations do not establish a generally superior method or a zero-error guarantee.

The analytic figure uses exact constructed probabilities: both distributions have 50% aggregate success and the same pooled paired outcomes, but statewise risk is 0 versus 0.04 at tolerance 0.02 and budget 0.02. This illustrates loss of state information; it does not claim new statistical theory or a sample-complexity lower bound.

## Contents and verification

- `paper/anonymous-development-review.pdf`: six pages (five body, one references), one vector figure and two tables. The immutable source archive and original verification receipt accompany it.
- `runs/`: frozen synthetic development cohorts and collector source snapshots. `analysis/`: independently checked summaries and figures.
- `experiments/`: methods, validators, synthetic engineering controls and native collection implementation. The frozen native study plan is in `protocols/`. Model health checks and orchestration fixtures are not update-efficacy results. The public synthetic health fixture mocks the installed-factory digest so CPU checks do not require a local LeRobot installation; frozen source snapshots remain byte-identical.

CPU checks, without loading a model:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-analysis.txt
.venv/bin/python scripts/verify_release.py
cd experiments
../.venv/bin/python -m unittest test_admission test_joint_admission test_coarse_admission test_closedloop_decisions test_paired_collection test_paired_batches test_native_collection test_health_supervisor test_health_lifecycle
```

Native collection requires the separately pinned LeRobot/LIBERO environments, model revisions, reset inputs, source witnesses, device identity and shared physical budget ledger. Historical absolute paths are preserved in source snapshots and need an explicit fresh environment configuration; they are not a portable, bit-exact simulator replay. Do not run collection commands against a new or reset budget ledger to bypass spent resources. Floating weight round/dequantization does not establish integer-kernel acceleration.

No incomplete native update cohort or private account material is included in this release. No completed native policy-update efficacy comparison is claimed. The ongoing baseline-reload control must complete both sides, clean worker teardown and raw-evidence audit before an engineering pass can be reported. A new release will be needed for verified native results.

Prior paired-outcome preprint: https://github.com/noteflowai/vla-regressions

Licensing: [NOTICE.md](NOTICE.md).

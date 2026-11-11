# House Prices — Regression From First Principles

A one-week Kaggle project built around a single decision: **use the simplest
possible model (ordinary least squares — one straight line in 250+ dimensions)
and actually learn it**, instead of reaching for ensembles and tool-churn.

> **Deliberately out of scope:** boosting, bagging, ensembles, regularization,
> serving APIs, Docker, CI, experiment-tracking UIs. Each is a fine tool;
> none teaches what this week set out to teach.

## The problem

[Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques):
predict sale prices for 1,459 Ames, Iowa homes, trained on 1,460 homes with
known prices (79 features each). Scored by **RMSLE** — error on the log scale.

## The week, as commits

| Day | Stage | What happened | What it teaches |
|---|---|---|---|
| 1 | 🔍 Explore | EDA: skew → log-target, absence ≠ unknown, collinearity pairs | look before you touch |
| 2 | 🧹 Prepare | Feature pipeline, TDD (12 tests); **caught pandas eating `"None"` as NaN** | reading data is a decision |
| 3 | 🏋️ Fit | First OLS fit: CV RMSLE **0.1545**; experiment log (`results/runs.csv`) | honest evaluation via CV |
| 4 | 🔎 Diagnose | Residuals, Q–Q, leverage = **1.000** traced to one-house categories | reading a fit for lies |
| 5 | ⚔️ Spec war | 4 challengers, one arena; **S1 crowned** (0.1517), 2 hypotheses rejected | iterate with reason |
| 6 | 📦 Ship | Validated submission generator + this README | reproducibility |

## Results (5-fold CV, out-of-fold, identical protocol for every spec)

| spec | CV RMSE (usd) | CV RMSLE (log) | verdict |
|---|---|---|---|
| v1_full_pipeline | $77,885 | 0.15453 | baseline |
| **S1_collapse_rare (champion)** | **$71,996** | **0.15171** | adopted |
| S2_no_outliers | $84,338 | 0.14818 | sensitivity probe only — *not adoptable* (deleting honest houses) |
| S3_drop_twins | $78,525 | 0.15491 | hypothesis rejected |
| S4_collapse_and_drop_twins | $73,695 | 0.15317 | kitchen sink dilutes S1 |

**Why S1 won:** Thursday's diagnostics exposed one-house categories
(`RoofMatl_ClyTile` appears exactly once) acting as private levers — leverage
1.0, a −2.34 coefficient claiming a 90% price cut from a single house.
Collapsing categories with <10 houses into `Other` removed the memorization,
improved *both* error rulers, and kept every honest house in the data.
The champion's loudest coefficients finally tell a realtor's story:
commercial zoning negative, upscale neighborhoods positive.

**Field lessons** (the real deliverables):
- A chart that disagrees with the data dictionary means the *reading* is
  wrong — pandas' default NA list silently converts the string `"None"` to
  NaN; 864 real "no veneer" houses were being corrupted until the missingness
  table said 59.7% for a column the dictionary says is ~0.5%.
- The best-looking number can be inadmissible: S2's RMSLE beat everyone by
  deleting two influential houses. We measured it, wrote it down, and
  rejected it — fixing the cause (S1) beats fixing the score.
- A diagnosable pathology isn't always a performance problem: collinear
  twins were real but cost almost nothing out-of-fold.

## Reproduce

```bash
uv sync                      # environment
uv run pytest -q             # 29 tests guard every decision
uv run python -m house_prices.train        # refit + log runs
uv run python -m house_prices.submission   # validated submission file
# notebooks: 01_eda, 02_diagnostics, 03_spec_comparison (run top to bottom)
```

Data: download `train.csv` + `test.csv` from the competition page into
`data/raw/` (see `data/README.md`; not committed).

## Layout

```
src/house_prices/    data.py · features.py · train.py · submission.py
tests/               29 tests, shared fixtures in conftest.py
notebooks/           01 EDA · 02 diagnostics · 03 spec comparison
results/runs.csv     the experiment log (every run has a written reason)
progress.html        the week as a one-page visual build sheet
models/              champion.joblib (gitignored, regenerate via train/submission)
```

## What I'd try next (honestly out of scope)

Regularized linear models (Ridge/Lasso — the natural sequel once OLS's
failure modes are understood), gradient boosting for the accuracy ceiling,
and prediction intervals instead of point estimates. Named, not attempted —
that restraint *is* the project.

# FIFA World Cup 2026 Forecast Engine

> **Matchday 2 snapshot — frozen 2026-06-23 23:59 UTC**
>
> This is a reproducible, time-stamped forecast, not a live-updating system.
> Every input, feature, and model weight is locked to data that existed before
> Matchday 3 began. The engine predicts Matchday 3 and the entire knockout
> bracket as forward-looking outputs.

---

## Portfolio Summary

This project demonstrates end-to-end applied data science under real-world constraints:
heterogeneous data sources, confirmed synthetic data that had to be identified and excluded,
name standardisation across seven different naming conventions, a known tactical data gap
for four teams, and a bracket structure that required decoding seeding logic from raw match
labels rather than an explicit lookup table.

The technical choices — Bayesian updating, chronological cross-validation, dual rating
systems, and explicit uncertainty intervals — each reflect a product decision, not a
textbook exercise. The result is a forecast engine that is fully reproducible from the
frozen datasets, auditable at every pipeline stage, and honest about what it does not know.

**Primary output:** 48-team tournament win probabilities with 90% confidence intervals,
derived from 10,000 Monte Carlo simulations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [June 23 Freeze Methodology](#2-june-23-freeze-methodology)
3. [Dataset Architecture](#3-dataset-architecture)
4. [Leakage Prevention](#4-leakage-prevention)
5. [Feature Engineering](#5-feature-engineering)
6. [Modeling Approach](#6-modeling-approach)
7. [Monte Carlo Simulation](#7-monte-carlo-simulation)
8. [How to Run](#8-how-to-run)
9. [Expected Outputs](#9-expected-outputs)
10. [Limitations](#10-limitations)

---

## 1. Project Overview

The engine answers a single question: given the state of the tournament after 48 matches
(Matchday 1 and Matchday 2 complete), what is the probability that each of the 48 teams
wins the 2026 FIFA World Cup?

It does this by combining:

- **Pre-tournament team strength** — Elo ratings, FIFA points, and WC historical records
- **In-tournament form** — MD1 and MD2 results, tactical statistics where available
- **Structural simulation** — the exact 2026 bracket seeding rules including Annex C
  third-place assignment logic

The full pipeline runs in six Jupyter notebooks backed by a `src/` package that contains
all reusable logic. The notebooks orchestrate; the source code computes.

---

## 2. June 23 Freeze Methodology

**Why freeze at Matchday 2?**

Matchday 3 results were available on June 23, 2026 — the same day this engine was built.
Incorporating them would turn a forecast into a retrodiction. The value of the project is
demonstrating that a structured, auditable model can produce competitive predictions using
only pre-MD3 information. The freeze creates a clear accountability boundary: when MD3
results come in, the predictions can be evaluated honestly.

**What "frozen" means in practice:**

| Element | Frozen value |
|---------|-------------|
| Match results used | 48 completed matches (MD1 + MD2) |
| Elo snapshot | `2026-05-27` — 15 days before tournament start |
| FIFA rankings | `2026-06-08` — 3 days before tournament start |
| Form window | `2024-01-01` to `2026-06-10` (last 10 competitive matches) |
| Hard cutoff | `2026-06-23 23:59 UTC` — no data after this timestamp |

**The four June 23 matches:**

Four matches completed on June 23 are included in the standings but have a known data gap:
Portugal 5–0 Uzbekistan, Colombia 1–0 Congo DR, England 0–0 Ghana, Croatia 1–0 Panama.
These four results are included as scores only — no tactical statistics (possession, shots,
formations) are available because the tactical feed was frozen before those stats were
published. The engine flags this explicitly via a `has_full_tactical_md2` feature rather
than imputing missing values.

---

## 3. Dataset Architecture

Thirteen files across six archives constitute the complete input layer. Each is assigned
a trust tier.

### Tier 1 — Ground truth (used directly for targets and features)

| ID | File | Rows | Content |
|----|------|------|---------|
| DS1 | `matches.csv` (Arc3) | 44 | MD1 + MD2 results with full tactical statistics |
| DS1-ext | June 23 results (injected) | 4 | Scores only for the four June 23 matches |
| DS2 | `elo_ratings_wc2026.csv` | 4,683 | Elo snapshot — filter to `snapshot_date = 2026-05-27` |
| DS4 | `results.csv` (Arc6) | 49,477 | International results 1872–June 18, 2026 |
| DS6 | `shootouts.csv` (Arc6) | 678 | Penalty shootout records |
| DS8 | `matches_1930_2022.csv` | 964 | All WC matches 1930–2022 |

### Tier 2 — Reference and structural (joins and context only)

| ID | File | Rows | Purpose |
|----|------|------|---------|
| DS9 | `schedule_2026.csv` | 72 | Canonical name authority for 2026 team spellings |
| DS10 | `fifa_ranking_2026-06-08.csv` | 211 | FIFA points and rank, June 8 2026 |
| DS11 | `fifa_ranking_2022-10-06.csv` | 211 | FIFA points at 2022 WC (4-year trajectory only) |
| DS16 | `matches.csv` (Arc_base) | 104 | Full 2026 bracket structure |
| DS17 | `teams.csv` (Arc_base) | 48 | Team ID to name/group mapping |
| DS18 | `host_cities.csv` (Arc_base) | 16 | Venue and city data |

### Tier 3 — Excluded (see Leakage Prevention)

| ID | File | Reason |
|----|------|--------|
| DS3 | `player_performance.csv` | Confirmed synthetic — fabricated match outcomes mixed with real player metadata |
| DS13 | `train (1).csv` | Unknown provenance, non-2026 teams, synthetic features |
| DS14 | `test (2).csv` | Same provenance concerns as DS13 |

### Name standardisation

DS9 (`schedule_2026.csv`) is the canonical name authority. All other datasets are mapped
to these spellings before any join. Known non-trivial mappings:

| Raw name (DS2/DS4/DS8) | Canonical |
|------------------------|-----------|
| South Korea | Korea Republic |
| Turkey | Türkiye |
| Ivory Coast | Côte d'Ivoire |
| Iran | IR Iran |
| DR Congo | Congo DR |
| Cabo Verde | Cape Verde |
| Bosnia and Herzegovina | Bosnia-Herzegovina |
| Czech Republic | Czechia |
| USA | United States |

All translations are registered in `src/name_map.py` and validated by `tests/test_name_map.py`.

---

## 4. Leakage Prevention

Seven rules govern the entire project. They are enforced programmatically by
`src/leakage_guard.py`, which raises `LeakageError` and halts execution if any rule
is violated.

| Rule | Description |
|------|-------------|
| **L1 — Freeze date** | No data with date after `2026-06-23` enters any model, feature, or simulation. DS4 contains null-score rows for MD3 matches — these are never imputed or treated as zero-zero draws. |
| **L2 — Elo snapshot** | DS2 must be filtered to exactly `snapshot_date == '2026-05-27'`. For historical backtesting, use `snapshot_date == '{year-1}-12-31'` for each match year. |
| **L3 — FIFA rankings** | DS10 (June 8, 2026) is safe — no WC match outcomes embedded. DS11 (Oct 2022) is used only for 4-year trajectory features. |
| **L4 — DS8 primacy** | DS8 is the primary WC training set (1930–2022, zero null scores). DS4 WC 2026 rows with null scores are never treated as completed matches. |
| **L5 — Tactical gap** | The four June 23 matches provide scores only. The `has_full_tactical_md2` boolean flags this structural absence. Imputing MD2 tactical stats from population means is prohibited. |
| **L6 — DS3 exclusion** | DS3 is fully excluded. No column from DS3 may be used, including columns that appear benign (age, position). The contamination risk is at the file level. |
| **L7 — Target derivation** | Win/draw/loss outcomes must be computed only from rows where both `home_score` and `away_score` are populated integers from verified completed matches. |

`leakage_guard.run_all_checks()` is called at the start of every notebook that uses
features or training data. A clean run produces no output. A failed run raises a named
`LeakageError` with the specific rule violated.

---

## 5. Feature Engineering

Eight feature groups constitute the complete feature space. All logic is in `src/features.py`.

### Group 1 — Pre-tournament Elo strength
*Source: DS2, filtered to `snapshot_date = 2026-05-27`*

| Feature | Definition |
|---------|-----------|
| `elo_rating` | Absolute Elo rating at pre-tournament snapshot. Range: 1,425 (Qatar) to 2,165 (Spain). |
| `elo_win_expectancy` | `1 / (1 + 10^((opp_elo − team_elo) / 400))` — calibrated probability baseline. Validated at 66.7% directional accuracy on modern WC backtesting. |
| `elo_rating_delta` | Team Elo minus opponent Elo — signed matchup signal. |
| `elo_rating_career_peak` | Highest Elo ever achieved — encodes institutional quality ceiling. |
| `elo_rating_career_avg` | Career average Elo — mean-reversion signal for teams above or below their historical level. |
| `elo_rank` | Global rank at pre-tournament snapshot. |
| `elo_is_host` | Binary: 1 for USA, Canada, Mexico; 0 otherwise. |

### Group 2 — FIFA points and momentum
*Source: DS10 (current, June 8 2026), DS11 (2022 baseline)*

| Feature | Definition |
|---------|-----------|
| `fifa_points` | Official FIFA points, June 8 2026. Range: 722.9 to 1,876.1. |
| `fifa_points_delta` | Points minus previous period — short-term momentum. |
| `fifa_points_4yr_change` | DS10 minus DS11 points — four-year program trajectory. |
| `elo_fifa_rank_disagreement` | Absolute difference between Elo rank and FIFA rank — uncertainty signal. |

### Group 3 — World Cup historical performance
*Source: DS8, filtered to 1998–2022 for modern-era features*

| Feature | Definition |
|---------|-----------|
| `wc_win_rate_modern` | Wins / matches played, 1998–2022 WC only. |
| `wc_win_rate_knockout_modern` | Wins / matches in knockout rounds, 1998–2022. |
| `wc_gd_per_game_modern` | Average goal difference per WC match, modern era. |
| `wc_group_vs_knockout_uplift` | Knockout win rate minus group-stage win rate — tournament temperament. |
| `wc_debut_modern_flag` | 1 for teams with zero WC appearances before 2026. |

Teams with zero WC appearances — Bosnia-Herzegovina, Curaçao, Cape Verde, Congo DR,
Jordan, Uzbekistan — receive a missing-class encoding for WC historical features, not a
zero. Zero would imply competitive history with no wins; missing correctly represents no data.

### Group 4 — Recent competitive form
*Source: DS4, filtered to non-friendly matches between 2024-01-01 and 2026-06-10, last 10 matches*

| Feature | Definition |
|---------|-----------|
| `form_win_rate_last10` | Win rate in last 10 competitive matches. |
| `form_gd_last10` | Cumulative goal difference in last 10 competitive matches. |

### Group 5 — 2026 in-tournament form (MD1 + MD2)
*Source: DS1 (44 matches, full tactical data) + DS1-ext (4 June 23 results, scores only)*

| Feature | Coverage |
|---------|---------|
| `tourn_pts_md2`, `tourn_gf_md2`, `tourn_gd_md2` | All 48 teams |
| `tourn_avg_possession`, `tourn_avg_sot` | 44 teams — DS1 tactical |
| `has_full_tactical_md2` | 1 for 44 teams, 0 for Portugal / Colombia / England / Croatia |

### Groups 6–8 — Venue context, matchup deltas, group context (MD3 only)

Venue cluster (East / Central / West), host-nation alignment, and group-context flags
(`must_win_flag`, `already_qualified`, `already_eliminated`) are computed at prediction
time from DS16, DS17, DS18, and the frozen standings.

### Penalty shootout features
*Source: DS6 (historical shootouts), DS8 (WC appearances)*

Bayesian shrinkage formula: `shrunk_rate = (wins + 4) / (appearances + 8)`, where `k = 8`
encodes a 0.5 prior. Teams with fewer than 2 WC shootout appearances use `w1 = 0.55`,
`w2 = 0.45`, `w3 = 0.00` (no WC-specific rate component).

---

## 6. Modeling Approach

The engine uses a **three-layer architecture**. All model logic is in `src/models.py`.

### Layer 1 — Pre-tournament baseline

**Training data:** DS8 (448 WC matches, 1998–2022, both perspectives = 896 rows),
optionally augmented with DS4 Tier-1 competitive matches (2014–2026, ~1,070 matches).

**Model:** XGBoost (`multi:softprob`, 3 classes) as primary; Logistic Regression as
interpretable baseline. The two are ensembled at 70% WC-only / 30% augmented by default,
with the weight CV-tuned in `[0.60, 0.90]`.

**Cross-validation:** 6-fold chronological leave-one-tournament-out (1998, 2002, 2006,
2010, 2014, 2018). The 2022 fold is reserved exclusively for calibration and never used
for hyperparameter search.

**Calibration:** Platt scaling → isotonic regression → temperature scaling (fallback
chain), fitted on the 2022 holdout. The raw XGBoost output is known to be poorly
calibrated at extreme probabilities; calibration is mandatory before simulation.

**Class weights:** `{WIN: 1.2, DRAW: 1.5, LOSS: 1.1}` — up-weights the minority draw
class which accounts for ~18.5% of WC match outcomes (1998–2022).

### Layer 2 — Bayesian tournament update

Takes Layer 1 probability vectors as priors and updates them with MD1 + MD2 evidence.

```
P_posterior = (1 − α) × P_prior + α × P_likelihood
```

`α = 0.17` — a conservative shrinkage parameter tuned on historical WC data (using
MD1 + MD2 stats from past tournaments to predict MD3-onward performance). This is
deliberately conservative: two matchdays is a small sample. The four June 23 tactical-gap
teams (Portugal, Colombia, England, Croatia) receive goals and points updates only; their
tactical feature adjustments are derived from MD1 alone.

### Layer 3 — Stage-conditional knockout models

- **Stage Group A (R32, R16):** XGBoost (`max_depth=2`) + Logistic Regression ensemble
- **Stage Group B (QF, SF, Final):** Logistic Regression primary + XGBoost secondary,
  equal weight — justified by the small (~49-match) training set at this stage

**Penalty shootout model:** `P(A wins) = w1 × shrunk_historical_rate + w2 × Elo_WE + w3 × WC_specific_rate`.
Applied conditionally only when the knockout model predicts a draw at 90 minutes
(draw rate in modern WC knockout matches: 24/112 = 21.4%).

---

## 7. Monte Carlo Simulation

The simulation engine is in `src/simulation.py`. Each of 10,000 runs executes:

1. **Simulate all 24 MD3 matches simultaneously** — outcomes drawn from the Layer 1 + Layer 2
   combined probability distribution. Group-context features (must-win, already-qualified)
   apply soft adjustments reflecting historical lineup rotation patterns.

2. **Apply tiebreakers deterministically** — head-to-head points, head-to-head GD,
   head-to-head GF, overall GD, overall GF, FIFA rank as a final differentiator.

3. **Rank all 12 third-placed teams** — by the standard 2026 WC tiebreaker sequence;
   select the eight best.

4. **Assign to R32 bracket** — using DS16 match labels and Annex C (495 rows, C(12,8)
   combinations) to determine which third-placed teams fill which R32 slots.

5. **Simulate R32 through Final** — using Layer 3 stage-conditional models. Draw at 90
   minutes triggers the penalty shootout model.

6. **Record champion and full bracket path** for this run.

7. **After 10,000 runs:** aggregate win probability per team, average round reached,
   most common final matchup, most common semifinal combinations. Report 90% confidence
   intervals alongside every point estimate.

**Seed:** `seed=2026` by default. Two runs with the same seed produce bit-identical outputs.

---

## 8. How to Run

### Prerequisites

```bash
# Python 3.9+
pip install -r requirements.txt
```

Place the six source archives in the project root directory:

| Archive | Contents |
|---------|---------|
| `archive.zip` | Arc_base: DS16, DS17, DS18 |
| `archive (2).zip` | Arc2: DS4, DS10, DS11 |
| `archive (3).zip` | Arc3: DS8 |
| `archive (4).zip` | Arc4: DS2 (Elo) |
| `archive (6).zip` | Arc6: DS6 |

### Run the test suite

```bash
# From the project root
python -m pytest tests/ -v
```

All 280 tests should pass. The suite validates:
- Name mapping completeness (48 canonical teams, all variant translations)
- Standings computation and tiebreaker logic against known frozen scorelines
- All 7 leakage rules (adversarial inputs that should raise `LeakageError`)
- Feature construction invariants (formula correctness, debutant flags, shrinkage rates)
- Model layer invariants (probability sums, draw rate, Bayesian bounds)
- Simulation invariants (reproducibility, bracket structure, probability invariants)

### Run the notebooks (in order)

```bash
jupyter notebook
```

| Notebook | Purpose | Key outputs |
|----------|---------|-------------|
| `01_data_audit.ipynb` | Verify frozen dataset state | `group_standings_freeze.csv` |
| `02_feature_engineering.ipynb` | Build all 8 feature groups | `team_features_freeze.parquet`, `training_rows.parquet` |
| `03_model_training.ipynb` | Train and calibrate models | `group_stage_model.joblib`, `knockout_model.joblib` |
| `04_simulation.ipynb` | Bayesian update + 10,000 MC runs | `win_probabilities.csv`, `simulation_log.parquet` |
| `05_results_visualization.ipynb` | Publication-ready charts | All `outputs/charts/*.png` |
| `06_reproducibility.ipynb` | Full pipeline audit | Pass/fail report — no new files |

Run notebooks in order. Each notebook reads from `outputs/` files produced by previous steps.
Notebook 06 is read-only and asserts correctness of all previous outputs.

### Run a quick smoke test

```bash
# Verify all src imports are clean
python -c "
import src.name_map, src.standings, src.leakage_guard
import src.features, src.models, src.simulation
print('All imports successful')
"
```

---

## 9. Expected Outputs

After running all notebooks:

```
outputs/
├── group_standings_freeze.csv          — 48-row verified MD2 group standings
├── third_place_ranking_freeze.csv      — 12-row third-place ranking at freeze
├── team_features_freeze.parquet        — 48-row team feature table (~55 features)
├── training_rows.parquet               — 896-row training dataset with outcomes
├── feature_audit_report.csv            — per-feature null counts, ranges, flags
├── group_stage_model.joblib            — trained + calibrated GroupStageModel
├── knockout_model.joblib               — trained + calibrated KnockoutModel
├── penalty_model.joblib                — penalty shootout model
├── feature_importance_group.csv        — top features by XGBoost gain
├── feature_importance_knockout.csv
├── cv_results.csv                      — per-fold accuracy, RPS, Brier score
├── bayesian_update_table.csv           — 48-row before/after probability table
├── win_probabilities.csv               — 48-row win probs with 90% CI
├── simulation_log.parquet              — 10,000-row detailed simulation log
├── bracket_simulation_summary.csv      — most common matchups per round
├── final_summary_table.csv             — human-readable summary
└── charts/
    ├── calibration_curve.png           — reliability diagram (2022 holdout)
    ├── update_shifts.png               — Bayesian update shift bar chart
    ├── win_probability_chart.png       — 48-team win probability chart (300 DPI)
    ├── group_standings_heatmap.png     — 12-group colour-coded standings
    ├── feature_importance_chart.png    — top 15 features bar chart
    └── bayesian_shifts_chart.png       — diverging shift chart (before/after)
```

---

## 10. Limitations

**Stated explicitly per the project design:**

1. **Small 2026 in-tournament sample.** The Bayesian update (Layer 2) is based on two
   matchdays — 48 matches for 48 teams. The α = 0.17 shrinkage is deliberately conservative
   to avoid over-weighting a small sample. Win probabilities are meaningfully influenced
   by pre-tournament Elo and WC history, not just recent form.

2. **No player-level modeling.** There is no verified, non-synthetic player data in the
   frozen dataset collection. Injury states, squad depth, and individual player form are
   not modeled. This is a team-level engine.

3. **Four-team tactical data gap.** Portugal, Colombia, England, and Croatia have complete
   tactical data for MD1 only. Their MD2 tactical features are structurally absent and
   flagged — not imputed. Model predictions for these teams depend more heavily on
   historical and Elo signals than for the 44 teams with full data.

4. **DS3 excluded.** The `player_performance.csv` file exists in the data archive but is
   entirely excluded. It was confirmed to contain fabricated match outcomes and synthetic
   features encoded in a way that could contaminate any join. See Rule L6 in Leakage
   Prevention.

5. **Confidence intervals are simulation-based, not analytic.** The 90% CI on win
   probabilities is computed from the empirical distribution across 10,000 runs using a
   Wilson interval. It captures simulation variance but not model uncertainty — the
   intervals would be wider if epistemic uncertainty in the model weights were fully
   propagated.

6. **Validation is prospective.** The model has not been validated against MD3 results
   because those results occurred after the freeze. Backtesting on 2022 WC (the
   calibration holdout) provides the primary accuracy estimate. Claims about accuracy
   against the actual 2026 tournament cannot be made without breaking the freeze.

7. **Tiebreakers use FIFA rank as a final differentiator.** The actual 2026 WC rules use
   a drawing of lots when all statistical criteria are equal. The simulation substitutes
   FIFA rank to remain deterministic and avoid simulation-within-simulation complexity.
   This affects a small number of edge cases.

---

## Repository Structure

```
fifa-wc2026-forecast/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── name_map.py          — canonical name translation for all 48 teams
│   ├── standings.py         — group standings, tiebreakers, third-place ranking
│   ├── features.py          — feature group construction (Groups 1–8)
│   ├── models.py            — Layer 1/2/3 model classes + BayesianTournamentUpdater
│   ├── leakage_guard.py     — 7-rule leakage enforcement, run_all_checks()
│   └── simulation.py        — Monte Carlo engine, Annex C seeding, bracket logic
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_simulation.ipynb
│   ├── 05_results_visualization.ipynb
│   └── 06_reproducibility.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_name_map.py
│   ├── test_standings.py
│   ├── test_leakage.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_simulation.py
│
├── outputs/                 — generated by running notebooks (not committed)
│   └── charts/
│
└── third_place_annex_c.csv  — 495-row Annex C third-place seeding table
```

---

## Data Sources

| Archive | Original source |
|---------|----------------|
| `archive.zip` (Arc_base) | 2026 WC bracket and team data |
| `archive (2).zip` | FIFA rankings and international results |
| `archive (3).zip` | WC match archive 1930–2022 |
| `archive (4).zip` | Elo ratings snapshot |
| `archive (6).zip` | International results and penalty shootouts |

Archive files are not committed to this repository (they are large binary files and are
listed in `.gitignore`). Obtain them from the original Kaggle sources and place them in
the project root before running the pipeline.

---

*Matchday 2 snapshot — frozen 2026-06-23 23:59 UTC. Built as a reproducible research artifact.*

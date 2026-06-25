
FIFA World Cup 2026 Forecast Engine
Definitive Project Blueprint — Frozen at Matchday 2 (June 23, 2026)
Snapshot identity: All inputs are locked as of 23:59 UTC June 23, 2026. No data created after this timestamp enters any model, feature, or simulation. The engine is a reproducible, time-stamped forecast — not a live-updating system.

1. Final Dataset Architecture
The canonical source registry — frozen
Thirteen files across six archives constitute the complete input layer. Each file is assigned a trust tier that governs how it is used.
Tier 1 — Ground truth (use directly for targets and features)
ID
File
Rows
Freeze-relevant content
DS1
matches.csv (Arc3)
44
All 44 completed matches MD1+MD2 with full tactical statistics
DS1-ext
June 23 results (injected)
4
Portugal 5–0 Uzbekistan, Colombia 1–0 Congo DR, England 0–0 Ghana, Croatia 1–0 Panama — scores only, no tactical stats
DS2
elo_ratings_wc2026.csv
4,683
Pre-tournament Elo snapshot: filter to snapshot_date = 2026-05-27
DS4
results.csv (Arc6)
49,477
Historical international results 1872–June 18, 2026
DS5
goalscorers.csv (Arc6)
47,690
Individual goal events 1916–present
DS6
shootouts.csv (Arc6)
678
Penalty shootout records
DS8
matches_1930_2022.csv
964
All WC matches 1930–2022, pre-filtered and enriched
Tier 2 — Reference and structural (use for joins and context, not as features directly)
ID
File
Rows
Purpose
DS9
schedule_2026.csv
72
Canonical name authority for 2026 team spellings
DS10
fifa_ranking_2026-06-08.csv
211
Official FIFA points and rank, June 8 2026
DS11
fifa_ranking_2022-10-06.csv
211
FIFA points at 2022 WC — for 4-year trajectory only
DS16
matches.csv (Arc_base)
104
Full 2026 bracket structure, all 104 matches, city and stage IDs
DS17
teams.csv (Arc_base)
48
Team ID to name/group/code mapping
DS18
host_cities.csv (Arc_base)
16
Venue, city, country, region cluster for all 16 host venues
Tier 3 — Excluded from all modeling
ID
File
Reason
DS3
player_performance.csv
Confirmed synthetic, wrong teams, future outcomes encoded
DS12
world_cup.csv
Summary only — data already available in DS8 at match level
DS13
train (1).csv
Unknown provenance, non-2026 teams, synthetic features
DS14
test (2).csv
Same provenance concerns as DS13
DS15
submission.csv
Kaggle template only
DS19
tournament_stages.csv
Lookup only — absorbed into DS16 joins
DS20
worldcup2026.db
Duplicate of Arc_base CSVs
DS21
pipeline.py
Code reference only
The name standardization authority
DS9 (schedule_2026.csv) defines canonical 2026 team names. All other datasets map to these spellings before any join. The complete mapping:
DS1 / DS4 / DS8 name
Canonical 2026 name
DS2 Elo name
DS10 FIFA name
Korea Republic
Korea Republic
South Korea
Korea Republic
Bosnia & Herz.
Bosnia-Herzegovina
Bosnia and Herzegovina
Bosnia-Herzegovina
Türkiye
Türkiye
Turkey
Türkiye
Côte d'Ivoire
Côte d'Ivoire
Ivory Coast
Côte d'Ivoire
IR Iran
IR Iran
Iran
IR Iran
Cabo Verde
Cape Verde
Cape Verde
Cabo Verde
Congo DR
Congo DR
DR Congo
Congo DR
All others
Identical
Identical
Identical

2. Final Leakage Prevention Rules
Seven rules govern the entire project. They are absolute — no exception for modeling convenience.
Rule L1 — The freeze date is a hard wall
No data with a creation timestamp or match date after June 23, 2026 enters any model, feature set, or simulation input. This includes the Matchday 3 results that are happening today. The engine predicts Matchday 3 as a forward-looking output, not as a training input. Specifically: DS4 contains null-score rows for Matchday 3 matches — these rows are present in the dataset but their score fields are treated as unknown future outcomes, not as missings to impute.
Rule L2 — Elo snapshot pinned to May 27, 2026
DS2 must be filtered to exactly snapshot_date == '2026-05-27' for any current-strength feature. The Elo dataset contains a 2026 live snapshot row per team at this date. Any other row in DS2 for year 2026 must not be used — it may reflect post-May-27 updates. For historical backtesting, use snapshot_date == '{year}-12-31' for the year preceding each historical match being trained on.
Rule L3 — FIFA rankings pinned to June 8, 2026
DS10 is dated June 8, 2026, which is three days before the tournament started. This is safe — no WC match outcomes are embedded. DS11 is safe for backtesting the 2022 WC or computing a 4-year trajectory feature. It must not be used as a current-strength proxy for 2026 predictions.
Rule L4 — DS8 is the WC historical training set; DS4 WC rows are the supplement
For training on historical WC matches, DS8 (1930–2022, 964 rows, zero null scores) is the primary source. DS4 WC rows are used only for two purposes: providing real WC 2026 results as those become available (they have null scores at the freeze point for MD3), and supplementing non-WC form features. The 2026 WC rows in DS4 with null scores must never be treated as zero-zero draws or imputed in any way.
Rule L5 — DS1-ext June 23 tactical gap is explicit, not imputed
The four June 23 matches provide scores only. Portugal, Colombia, England, and Croatia each have one full tactical match and one scores-only match at the freeze point. For tactical features (possession, shots on target, crosses, interceptions), these four teams' MD2 values are missing. The correct handling is to flag this explicitly with a boolean column has_full_tactical_md2 and to either use MD1 tactical data alone for these teams or to treat the MD2 tactical features as structurally missing. Imputing MD2 tactical stats from population means would introduce false precision and is prohibited.
Rule L6 — DS3 is fully excluded with no partial use
No column from DS3 may be used, including columns that appear benign such as position, age, height, or club_name. The dataset's core architecture mixes real player metadata with fabricated match outcomes in a single file. Any join involving DS3 risks contamination. The exclusion is total.
Rule L7 — Target variables must be derived only from completed match scorelines
Win/draw/loss outcomes, goal counts, and any derivative (goal difference, clean sheet flag, points earned) must be computed only from rows where both home_score and away_score are populated with real integer values from verified completed matches. The 48 frozen matches are the only permitted source for 2026 in-tournament target derivation.

3. Final Feature Groups
Eight feature groups constitute the complete feature space. Each is labelled with its source datasets and the exact filter required to maintain leakage safety.

Feature Group 1 — Pre-tournament Elo strength Source: DS2 filtered to snapshot_date = 2026-05-27
Feature
Definition
elo_rating
Absolute Elo rating at pre-tournament snapshot
elo_rank
Global rank at pre-tournament snapshot
elo_rating_career_peak
Highest Elo rating ever achieved by this team to date
elo_rating_career_avg
Career average Elo rating to date
elo_rating_delta_vs_opponent
Team Elo minus opponent Elo — primary matchup signal
elo_rank_delta_vs_opponent
Team rank minus opponent rank — directional version
elo_win_expectancy
1 / (1 + 10^((opponent_rating - team_rating) / 400)) — calibrated probability baseline
is_host
1 for USA, Canada, Mexico; 0 otherwise

Feature Group 2 — Official FIFA points and momentum Source: DS10 (current, June 8 2026), DS11 (2022 baseline)
Feature
Definition
fifa_points
Official FIFA points as of June 8, 2026
fifa_rank
Official FIFA rank as of June 8, 2026
fifa_points_delta_vs_previous
DS10: points minus previous_points — short-term momentum
fifa_rank_delta_vs_previous
DS10: rank minus previous_rank (positive = improving)
fifa_points_4yr_change
DS10 points minus DS11 points — program trajectory over WC cycle
fifa_rank_4yr_change
DS10 rank minus DS11 rank
elo_fifa_rank_disagreement
Absolute difference between DS2 Elo rank and DS10 FIFA rank — uncertainty signal; high disagreement signals higher prediction variance

Feature Group 3 — WC historical performance (era-controlled) Source: DS8 filtered to Year >= 1998 for modern-era features; full DS8 for all-time features
Feature
Definition
Era filter
wc_win_rate_modern
Wins / matches played, 1998–2022 WC only
Year >= 1998
wc_win_rate_knockout_modern
Wins / matches played in knockout rounds, 1998–2022
Year >= 1998, Round != Group stage
wc_avg_gf_modern
Average goals scored per WC match, 1998–2022
Year >= 1998
wc_avg_ga_modern
Average goals conceded per WC match, 1998–2022
Year >= 1998
wc_clean_sheet_rate_modern
Proportion of WC matches with clean sheet, 1998–2022
Year >= 1998
wc_tournaments_attended
Count of WC tournaments participated in, all time
None
wc_best_result_encoded
Encoded best-ever WC result (7=champion, 6=runner-up, 5=3rd, 4=4th, 3=QF, 2=R16, 1=group stage)
None
wc_penalty_shootout_win_rate
Shootout wins / shootout appearances
DS6 join to DS8
wc_group_vs_knockout_uplift
Knockout win rate minus group stage win rate — tournament temperament signal
Year >= 1998
Teams with zero WC appearances (Haiti, Uzbekistan, Curaçao, Congo DR, Jordan, Cape Verde) receive a missing-class encoding for WC historical features, not a zero. Zero would imply they have been competitive and never won — missing correctly represents no data.

Feature Group 4 — Recent competitive form Source: DS4 filtered to tournament NOT IN ('Friendly') AND date BETWEEN 2024-01-01 AND 2026-06-10
Feature
Definition
form_win_rate_last10_competitive
Wins in last 10 competitive (non-friendly) matches
form_avg_gf_last10
Average goals scored in last 10 competitive matches
form_avg_ga_last10
Average goals conceded in last 10 competitive matches
form_clean_sheet_rate_last10
Clean sheet proportion in last 10 competitive matches
form_unbeaten_streak_entering
Consecutive unbeaten matches in competitive play entering the tournament
form_goal_difference_last10
Cumulative GD in last 10 competitive matches

Feature Group 5 — 2026 in-tournament form (Matchday 1 and Matchday 2) Source: DS1 (44 matches, full tactical data) + DS1-ext (4 June 23 results, scores only)
Feature
Definition
Coverage
tourn_gf
Total goals scored in tournament so far
All 48 teams
tourn_ga
Total goals conceded in tournament so far
All 48 teams
tourn_gd
Goal difference in tournament so far
All 48 teams
tourn_pts
Points earned in tournament so far
All 48 teams
tourn_avg_possession
Average ball possession across matches played
44 teams only — DS1 tactical
tourn_avg_sot
Average shots on target per match
44 teams only — DS1 tactical
tourn_avg_sot_conceded
Average shots on target conceded per match
44 teams only — DS1 tactical
tourn_avg_total_shots
Average total shots per match
44 teams only — DS1 tactical
tourn_shot_conversion_rate
Goals / total shots across tournament
44 teams only
tourn_xg_overperformance
Goals minus xG — indicates clinical vs. lucky finishing
DS1 MD1 only (xG available)
tourn_yellow_cards
Total yellow cards accumulated
All 48 teams — DS1
tourn_red_cards
Total red cards accumulated
All 48 teams — DS1
has_full_tactical_md2
Boolean: 1 if both matches have tactical data
1 for 44 teams, 0 for Portugal/Colombia/England/Croatia
tourn_formation_changed
Boolean: whether formation changed between MD1 and MD2
44 teams only

Feature Group 6 — Venue and structural context Source: DS16, DS17, DS18, DS19
Feature
Definition
venue_region_cluster
East / Central / West — from DS18
venue_country
USA / Canada / Mexico — from DS18 via DS16
is_true_home_venue
1 for USA/Canada/Mexico teams playing in their own country
confederation_region_alignment
Whether team confederation matches host region (e.g. CONCACAF playing in USA) — proxy for cultural home advantage
stage_id
Current stage numeric (1=group, 2=R32, etc.) from DS19
stage_order_numeric
Pressure escalation feature — higher stage = higher stakes
kickoff_local_hour
Hour of kickoff in local venue time — heat and crowd factors

Feature Group 7 — Matchup-level derived features Computed at prediction time for each specific matchup — not stored as team features
Feature
Definition
elo_delta
Team A Elo minus Team B Elo
fifa_points_delta
Team A FIFA points minus Team B FIFA points
wc_experience_delta
Difference in WC tournaments attended
form_win_rate_delta
Difference in recent competitive win rates
tourn_pts_delta
Difference in points earned so far in this tournament
tourn_gd_delta
Difference in tournament goal difference
dual_system_agreement
Whether Elo rank and FIFA rank agree on which team is stronger — 1 if both favor same team

Feature Group 8 — Group context (for Matchday 3 simulation only) Computed from frozen standings
Feature
Definition
points_before_md3
Points heading into the final matchday
min_points_to_qualify
Minimum points needed to guarantee top-2 finish
must_win_flag
1 if a draw is insufficient to qualify under any scenario
already_qualified
1 if top-2 position is mathematically secured before MD3
already_eliminated
1 if team cannot finish top-2 or best-8-third regardless of result
potential_third_place_qualifier
1 if team could qualify as one of the eight best third-placed teams
These features exist only for the Matchday 3 group stage simulation, not for knockout stage prediction.

4. Final Prediction Targets
The engine produces four distinct prediction outputs, each with a specific framing.
Target T1 — Matchday 3 group stage match outcome Unit: One row per match (24 matches) Format: Three-class classification — home win / draw / away win Derived from: Historical WC and competitive match outcomes from DS8 and DS4 Evaluation: Ranked probability score against actual MD3 results when they occur
Target T2 — Group stage final standings Unit: One row per team (48 teams) Format: Probabilistic finishing position — P(1st), P(2nd), P(3rd), P(4th) per team per group Derived from: Monte Carlo simulation over T1 outcomes, applied 10,000 times This is a distribution, not a point estimate. The engine reports the distribution of outcomes, not a single predicted table.
Target T3 — Knockout stage match outcome Unit: One row per knockout match (32 matches — R32 through Final) Format: Two-class classification — home/higher-seeded win vs away/lower-seeded win; no draw class for knockout stage Special handling: Penalty shootout modeled as a separate conditional probability applied only when the two-class model predicts a draw at 90 minutes, using historical shootout data from DS6
Target T4 — Tournament winner probability Unit: One value per team (48 teams, sums to 1.0) Format: Win probability derived from Monte Carlo simulation across all stages This is the headline output — the probability that each of the 48 teams lifts the trophy, computed as the proportion of 10,000 simulations in which that team wins every remaining match.

5. Final Modeling Strategy
The engine uses a three-layer modeling architecture. Each layer has a specific scope and model class.

Layer 1 — Pre-tournament baseline model
Trains on all historical WC knockout matches from DS8 (1998–2022, modern era) plus recent competitive matches from DS4. Features are Feature Groups 1, 2, 3, and 4 only — no 2026 in-tournament data is used in this layer. This model represents the expected strength hierarchy before any ball was kicked in 2026.
Model class: Gradient Boosting (XGBoost or GBM). Justified because the feature space is moderate, the interactions between Elo delta and form delta are non-linear, and gradient boosting handles mixed feature types and missing values (WC debutants) without requiring imputation. Logistic Regression is trained in parallel as an interpretable baseline.
Cross-validation: Chronological split — train on 1998–2018 WC matches, validate on 2022 WC matches. Standard k-fold is prohibited because it would leak future tournament outcomes into the training folds.
Calibration: Probability outputs are calibrated using Platt scaling on the 2022 holdout. Calibration is mandatory because the Elo win expectancy formula provides a strong prior — the model's raw probabilities must be verified against that prior before use in simulation.

Layer 2 — Tournament-adjusted model
Takes the Layer 1 probability outputs as a prior and updates them with 2026 in-tournament evidence from Feature Groups 5, 6, and 7. This is a Bayesian update layer — the prior is the pre-tournament probability, and the likelihood function is estimated from how in-tournament performance in MD1+MD2 historically predicts subsequent WC performance.
The update is conservative. Two matchdays is a small sample. A team that significantly outperforms its Elo rating in MD1+MD2 gets a modest upward adjustment, not a wholesale reranking. The weight of in-tournament evidence versus historical prior is controlled by a single shrinkage parameter tuned on historical WC data (using MD1+MD2 stats from past tournaments to predict MD3 onward performance).
Special handling for the four June 23 teams: Portugal, Colombia, England, and Croatia receive in-tournament updates for goals and points only. Their tactical feature adjustments are derived from MD1 alone, with the has_full_tactical_md2 flag included as a model input so the model learns that this team's MD2 tactical features are structurally absent.

Layer 3 — Stage-conditional knockout model
A separate model is fitted for each knockout stage transition. The premise is that the features that matter most in a group stage match are different from those in a quarterfinal. Stage-conditional models capture this: a team's tournament momentum (tourn_pts, tourn_gd) carries more weight in later rounds than in early knockouts, while Elo delta carries more weight in early knockouts where sample sizes are small.
In practice for this project, stages R32 and R16 share one model, and QF/SF/Final share a second model, given sample size constraints from historical data.
Penalty shootout model: Logistic regression on DS6 historical shootout data, using features: confederation, elo_delta at time of shootout, and historical shootout win rate. This produces a P(win shootout) applied conditionally when the main model predicts draw at 90 minutes.

6. Final Monte Carlo Simulation Starting Point
The simulation begins from a precisely defined state. Every element of this state is derived from the frozen datasets and verified against the actual match records.
Frozen standing entering MD3
Group A — Mexico 6pts (GD +3), Korea Republic 3pts (GD 0), Czechia 1pt (GD −1), South Africa 1pt (GD −2). Mexico qualified. Korea Republic, Czechia, and South Africa compete for second place and possible third-place slot.
Group B — Canada 4pts (GD +6), Switzerland 4pts (GD +3), Bosnia-Herzegovina 1pt (GD −3), Qatar 1pt (GD −6). Canada and Switzerland tied on points — GD separates them currently. MD3 (Switzerland vs Canada and Bosnia-Herzegovina vs Qatar) will determine finishing positions.
Group C — Brazil 4pts (GD +3), Morocco 4pts (GD +1), Scotland 3pts (GD 0), Haiti 0pts (GD −4). Four-way qualification scenario still open — Scotland can qualify if results go their way.
Group D — United States 6pts (GD +5), Australia 3pts (GD 0), Paraguay 3pts (GD −2), Türkiye 0pts (GD −3). USA qualified. Australia and Paraguay level on points, separated by GD. Türkiye eliminated from top-2 contention.
Group E — Germany 6pts (GD +7), Côte d'Ivoire 3pts (GD 0), Ecuador 1pt (GD −1), Curaçao 1pt (GD −6). Germany qualified. Three teams compete for second.
Group F — Netherlands 4pts (GD +4), Japan 4pts (GD +4), Sweden 3pts (GD 0), Tunisia 0pts (GD −8). Netherlands and Japan tied on all metrics — head-to-head was a draw. MD3 decisive.
Group G — Egypt 4pts (GD +2), IR Iran 2pts (GD 0), Belgium 2pts (GD 0), New Zealand 1pt (GD −2). Egypt leads. Iran and Belgium locked equal on all metrics — head-to-head (0–0) does not separate them. MD3 (Egypt vs Iran, New Zealand vs Belgium) is decisive and creates strong tactical incentives.
Group H — Spain 4pts (GD +4), Uruguay 2pts (GD 0), Cabo Verde 2pts (GD 0), Saudi Arabia 1pt (GD −4). Spain leads. Uruguay and Cabo Verde tied on all metrics.
Group I — France 6pts (GD +5), Norway 6pts (GD +4), Senegal 0pts (GD −3), Iraq 0pts (GD −6). Both France and Norway qualified. Their MD3 head-to-head determines who finishes first — significant for bracket seeding.
Group J — Argentina 6pts (GD +5), Austria 3pts (GD 0), Algeria 3pts (GD −2), Jordan 0pts (GD −3). Argentina qualified. Austria and Algeria level on points, separated by GD. Algeria has scored more; Austria has better GD.
Group K — Colombia 6pts (GD +3), Portugal 4pts (GD +5), Congo DR 1pt (GD −1), Uzbekistan 0pts (GD −7). Colombia qualified. Portugal leads the remaining race comfortably. MD3 (Colombia vs Portugal, Congo DR vs Uzbekistan) primarily determines first vs second seeding.
Group L — England 4pts (GD +2), Ghana 4pts (GD +1), Croatia 3pts (GD −1), Panama 0pts (GD −2). England and Ghana tied — head-to-head was 0–0, which does not separate them under tiebreaker rules. MD3 (Panama vs England, Croatia vs Ghana) is fully open.
Third-place race entering MD3
The eight best third-placed teams advance. Current third-place standings by points and GD at the freeze:
	1.	Sweden (F) — 3pts, GD 0, GF 6
	2.	Scotland (C) — 3pts, GD 0, GF 1
	3.	Croatia (L) — 3pts, GD −1, GF 3
	4.	Paraguay (D) — 3pts, GD −2, GF 2
	5.	Algeria (J) — 3pts, GD −2, GF 2
	6.	Cabo Verde (H) — 2pts, GD 0, GF 2
	7.	Belgium (G) — 2pts, GD 0, GF 1
	8.	Czechia (A) — 1pt, GD −1, GF 2
The cut line between 8th and 9th is currently between Czechia (1pt) and Congo DR/Ecuador/Bosnia-Herzegovina (all 1pt). Any of these teams can move above Czechia with a Matchday 3 win.
Simulation procedure
Each of 10,000 simulation runs executes the following sequence:
Step 1 — Simulate all 24 Matchday 3 matches simultaneously. For each match, draw an outcome (home win / draw / away win) from the Layer 1 + Layer 2 combined probability distribution, with noise added from a calibrated random draw. Apply the group-context features (must_win_flag, already_qualified) as soft adjustments — teams with nothing to play for get a small downward adjustment to win probability reflecting historical lineup rotation patterns.
Step 2 — Apply tiebreaker rules deterministically. For teams tied on points after MD3, apply: head-to-head points, head-to-head GD, head-to-head GF, overall GD, overall GF, FIFA rank (as a final differentiator in lieu of drawing of lots for the simulation).
Step 3 — Rank all 12 third-place teams by the standard 2026 WC tiebreaker sequence and select the eight best.
Step 4 — Assign qualified teams to the Round of 32 bracket using DS16 match labels. The seeding rules (e.g. "1E vs 3ABCDF" means the winner of Group E plays one of the best third-placed teams from Groups A, B, C, D, or F) are applied deterministically from DS16.
Step 5 — Simulate all knockout matches from R32 through the Final using the Layer 3 stage-conditional model. For each knockout match, draw an outcome using the matchup-specific probability. When the draw probability is non-trivial, apply the shootout model conditionally.
Step 6 — Record the tournament winner and full bracket path for this simulation run.
Step 7 — After 10,000 runs, aggregate: win probability per team, average round reached per team, most common final matchup, most common semifinal combinations.
Simulation uncertainty disclosure
The simulation must report a 90% confidence interval alongside the point estimate for each team's win probability. A team with a 15% mean win probability and a 90% CI of [8%, 24%] is genuinely uncertain. A team with a 5% mean and CI of [4%, 6%] is a settled long shot. These intervals communicate whether the headline number is robust or highly path-dependent.

7. Final GitHub Project Scope
The repository is a self-contained, reproducible research artifact. Someone cloning it and running the pipeline against the frozen datasets must arrive at identical outputs.
Repository structure
fifa-wc2026-forecast/
│
├── README.md
│   Explains the project, the freeze date, the data sources,
│   the modeling approach, and how to reproduce outputs.
│   Includes a prominent notice that this is a Matchday 2 snapshot —
│   not a live-updated system.
│
├── data/
│   ├── raw/
│   │   All source files exactly as received, unmodified.
│   │   Organized by archive origin (arc1_new, arc2_new, arc3,
│   │   arc4, arc6, arc_base).
│   │   DS3 (player_performance) included here but flagged
│   │   in README as excluded synthetic dataset.
│   │
│   ├── freeze/
│   │   june23_results.csv — the four injected results,
│   │   hand-coded with source documentation.
│   │   freeze_manifest.json — hash of every input file,
│   │   freeze timestamp, data lineage declaration.
│   │
│   └── processed/
│       All intermediate outputs of the pipeline.
│       Generated by running the pipeline — not committed to git.
│       .gitignored except for the final feature table.
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   │   Reproduces the full dataset audit. Outputs the
│   │   verified 48-match result set, group standings,
│   │   and third-place rankings.
│   │
│   ├── 02_feature_engineering.ipynb
│   │   Builds all eight feature groups. Documents every
│   │   join, filter, and derived column. Outputs the
│   │   final feature table as a parquet file.
│   │
│   ├── 03_baseline_model.ipynb
│   │   Layer 1 model. Chronological cross-validation.
│   │   Calibration. Feature importance. Comparison against
│   │   Elo baseline.
│   │
│   ├── 04_tournament_update.ipynb
│   │   Layer 2 Bayesian update. Shows the before/after
│   │   probability shift for each team from MD1+MD2 evidence.
│   │   Explicitly flags the four June 23 partial-data teams.
│   │
│   ├── 05_monte_carlo.ipynb
│   │   10,000 simulation runs. Group stage, third-place
│   │   selection, knockout bracket. Outputs win probabilities,
│   │   confidence intervals, and bracket path distributions.
│   │
│   └── 06_results_and_visualisation.ipynb
│       Final probability tables, bracket visualization,
│       uncertainty charts, and the LinkedIn-ready summary.
│
├── src/
│   ├── name_map.py — canonical name translation dictionary
│   ├── standings.py — group standings computation
│   ├── features.py — feature group construction functions
│   ├── models.py — Layer 1, 2, 3 model classes
│   ├── simulation.py — Monte Carlo engine
│   └── leakage_guard.py — assertions that verify no post-freeze
│                           data has entered any pipeline stage
│
├── outputs/
│   ├── group_standings_freeze.csv — verified MD2 standings
│   ├── team_features_freeze.parquet — full feature table
│   ├── win_probabilities.csv — final tournament win probs with CIs
│   ├── bracket_simulation_summary.csv — round-by-round averages
│   └── charts/ — all visualisations as PNG and SVG
│
├── tests/
│   ├── test_standings.py — verifies standings against known scorelines
│   ├── test_leakage.py — asserts no post-June-23 data in features
│   ├── test_name_map.py — verifies all 48 teams resolve correctly
│   └── test_simulation.py — verifies probabilities sum to 1.0
│
├── FREEZE_MANIFEST.md
│   Human-readable declaration of the project's frozen state.
│   Lists every input file with its SHA-256 hash, the freeze
│   timestamp, the four injected June 23 results with their
│   source, and the data lineage for every non-obvious decision.
│
└── requirements.txt
    Exact package versions pinned with hashes.
    Reproducibility requires matching the environment exactly.
What the project explicitly does not include
The following are out of scope for this project and must not be added:
Any live data fetching, API calls, or web scraping. The project is a frozen snapshot engine, not a live system. Any data from DS3, DS13, or DS14. Any results from Matchday 3 or later, even as validation data — the project is a forecast, and validating against known future results would compromise the reproducibility narrative. Any player-level analysis, squad depth modeling, or injury adjustment — there is no verified player data source in the frozen dataset collection. Any claims about statistical significance of the win probabilities at pilot scale — the confidence intervals communicate uncertainty, and the README must state that the model is based on limited 2026 in-tournament data and substantial historical inference.
What the README must state explicitly
The freeze date and what it means. The four June 23 results that were injected manually and their source. The fact that DS3 exists in the data folder but is excluded and why. The tactical data gap for Portugal, Colombia, England, and Croatia. That win probabilities are model outputs with quantified uncertainty, not point truths. The chronological validation approach and what 2022 WC accuracy was achieved. A link to the raw datasets' original sources for reproducibility verification.
The LinkedIn narrative attached to this project
The portfolio post accompanying this project tells a specific story: a data science and product thinking exercise in building a reproducible, auditable forecast system from heterogeneous data under real-world constraints — a frozen dataset, a tight timeline, confirmed synthetic data that had to be identified and excluded, name standardization across seven different conventions, a known tactical data gap for four teams, and a bracket structure that required decoding seeding logic from match labels rather than from an explicit lookup table. The technical choices (Bayesian updating, chronological CV, dual rating systems, explicit uncertainty intervals) are each explained in terms of the product decision they represent. The result is a forecast engine that a recruiter with a PM or strategy background can understand and that a data scientist can reproduce. That combination — not the accuracy of the predictions — is the actual deliverable.
Definitive Feature Engineering Specification
Frozen: June 23, 2026 — Post-Matchday 2

Group 1 — Elo Features
Source for all Group 1 features: DS2 (elo_ratings_wc2026.csv) filtered to snapshot_date == '2026-05-27'. This snapshot predates the tournament by 15 days and contains zero match contamination from 2026 WC games. Elo correctly predicted 66.7% of non-draw WC outcomes in modern era backtesting (1998–2022, 162 non-draw matches tested).

F001 — elo_rating
Formula: DS2.rating WHERE snapshot_date = '2026-05-27' AND country = [team] Source: DS2 Range: 1,425 (Qatar) to 2,165 (Spain) Leakage risk: None. Snapshot predates tournament by 15 days. No 2026 WC match data embedded in this value. Predictive rationale: The Elo rating is the most theoretically grounded measure of team strength in this dataset. It is updated after every international match using a formula that weights results by match importance, opposition strength, and margin of victory. At the pre-tournament snapshot it encodes decades of competitive history into a single number. Validated at 66.7% directional accuracy for WC match outcomes in modern era testing. Expected importance: Highest. Primary anchor feature for all matchup predictions.

F002 — elo_win_expectancy
Formula: 1 / (1 + 10^((opponent_elo_rating - team_elo_rating) / 400)) Source: DS2 (derived from F001) Range: 0.014 (weakest team vs strongest) to 0.986 (reverse); typical matchup range 0.35–0.65 for competitive games Leakage risk: None. Derived entirely from pre-tournament Elo ratings. Predictive rationale: This is the calibrated probability directly implied by the Elo system. Rather than leaving the model to learn the transformation from raw ratings to win probability, this feature provides the theoretically correct non-linear transformation. The formula accounts for the diminishing returns of larger rating gaps — a 400-point advantage predicts a 90.9% win probability, not a 100% one. This is the single most important matchup-level feature in the entire specification. Expected importance: Highest. Direct probability estimate — serves as both a feature and a calibration benchmark.

F003 — elo_rating_delta
Formula: team_elo_rating - opponent_elo_rating Source: DS2 (derived from F001) Range: −740 (weakest vs strongest) to +740 (strongest vs weakest) Leakage risk: None. Predictive rationale: The signed difference between team and opponent Elo ratings captures directionality explicitly. Positive values mean the team is favoured; negative means they are the underdog. This interacts non-linearly with stage features — an underdog in the Round of 32 has different win probability than the same underdog in the Final, and the model should learn this interaction. Expected importance: High. Correlated with F002 but captures the linear signal separately.

F004 — elo_rating_career_peak
Formula: DS2.rating_max WHERE snapshot_date = '2026-05-27' AND country = [team] Source: DS2 Range: 1,579 to 2,223 Leakage risk: None. Cumulative maximum is computed through May 27, 2026 only. Predictive rationale: Career peak rating captures how good a team has been at its historical best. A team that has reached 2,100 Elo at some point in its history carries institutional knowledge, tactical infrastructure, and a culture of winning that persists even in down cycles. This distinguishes teams like Brazil (peak 2,212, current 1,984) from teams whose current rating is near their historical ceiling. Expected importance: Medium. Valuable for teams whose current rating understates their true quality ceiling.

F005 — elo_rating_career_avg
Formula: DS2.rating_avg WHERE snapshot_date = '2026-05-27' AND country = [team] Source: DS2 Range: 1,302 to 1,998 Leakage risk: None. Predictive rationale: Career average rating measures consistency. A team that averages 1,900 Elo over many decades is structurally different from one that recently surged to 1,900 from a lower base. High career average relative to current rating suggests a team that is currently below its true level — a potential undervalued pick. Low career average relative to current rating suggests a team that may be overrated by recent results. Expected importance: Medium-low. Useful as a mean-reversion signal, particularly for teams showing unusual recent form.

F006 — elo_rank
Formula: DS2.rank WHERE snapshot_date = '2026-05-27' AND country = [team] Source: DS2 Range: 1 (Spain) to 95 (Qatar) Leakage risk: None. Predictive rationale: Ordinal rank encoding of Elo position. While correlated with F001, the rank captures relative position within the 48-team field more directly — a rank of 5 in a 48-team tournament means the team is in the top 10%, which has direct tournament bracket implications. Expected importance: Medium. Redundant with F001 in raw modeling but useful for bracket simulation logic.

F007 — elo_is_host
Formula: DS2.is_host WHERE snapshot_date = '2026-05-27' AND country = [team] Source: DS2 Range: Binary: 1 for United States, Canada, Mexico; 0 for all others Leakage risk: None. Fixed at dataset construction time. Predictive rationale: The 2026 WC is co-hosted across three countries. Historical evidence for host advantage is moderate — hosts reach the knockout rounds at a significantly higher rate than their Elo rating would predict (host teams have reached at least the QF in 7 of 22 WC tournaments, overperforming their seeding). The mechanism includes crowd support, travel advantage, climate familiarity, and no long-haul flights between matches. The three 2026 co-hosts have different levels of this advantage: Mexico plays three group matches in Guadalajara and Mexico City with strong Mexican fan bases; Canada plays home games in Toronto and Vancouver; the USA plays across three regions and gets variable crowd support depending on opponent. Expected importance: Medium. Stronger for Mexico and USA than Canada in this tournament specifically.

Group 2 — FIFA Ranking Features
Source for all Group 2 features: DS10 (fifa_ranking_2026-06-08.csv) for current values. DS11 (fifa_ranking_2022-10-06.csv) for four-year trajectory only. FIFA points and Elo ratings measure different things — FIFA points weight recent results more heavily (last four years rolling window) and use a time-decay factor. They agree on rough ordering but disagree on several specific teams in informative ways.

F008 — fifa_points
Formula: DS10.points WHERE team = [team] Source: DS10 Range: 722.9 (weakest) to 1,876.1 (Argentina) Leakage risk: None. Dated June 8, 2026 — predates tournament by three days. Predictive rationale: The official FIFA rating reflects recent competitive results weighted by match importance and opposition quality. Argentina leads with 1,876 points despite Spain leading the Elo system, reflecting Argentina's strong recent form in competitive matches (Copa América 2021 winner, World Cup 2022 winner). FIFA points and Elo are trained on slightly different objective functions and produce genuinely different signals for several teams. Both are valuable. Expected importance: High. Parallel signal to Elo that adds independent information.

F009 — fifa_points_delta
Formula: DS10.points - DS10.previous_points WHERE team = [team] Source: DS10 Range: −1,154 to +1,150 (extreme values; typical WC team range is approximately −50 to +80) Leakage risk: None. DS10 previous_points reflects the prior FIFA ranking period, which ends before June 8. Predictive rationale: Short-term momentum signal. Teams entering a tournament with rising FIFA points have been winning important matches recently and carry positive momentum. Teams with declining points have been losing ground to competitors. Norway is a notable example: Norway ranks 12th by Elo but 31st by FIFA points, and their FIFA points delta is strongly positive over the last period, confirming genuine momentum rather than a statistical anomaly. Expected importance: Medium. Adds momentum signal not captured in static rating features.

F010 — fifa_rank_delta
Formula: DS10.previous_rank - DS10.rank WHERE team = [team] Source: DS10 Range: −6 to +4 within the tournament's two-period window Leakage risk: None. Predictive rationale: Ordinal version of the momentum signal. A positive value means the team moved up in the rankings (improved). Given the narrow range (−6 to +4), this feature has low absolute variance and is less informative than F009. Included primarily because rank movements are psychologically salient and may capture qualitative signals (e.g. a team that recently won a major tournament would show a rank jump) not fully captured by points delta. Expected importance: Low. Narrow range limits discrimination power.

F011 — fifa_points_4yr_change
Formula: DS10.points - DS11.points WHERE team matches between datasets Source: DS10, DS11 Range: −130.4 to +191.6 across all teams; WC-relevant range approximately −100 to +180 Leakage risk: None. DS11 is from October 2022, DS10 is from June 2026. Neither contains 2026 WC results. The Uzbekistan case (+27 rank positions over 4 years, confirmed from the data) is a genuine trajectory signal. Predictive rationale: Four-year trajectory captures program development across a full WC cycle. Teams that have risen significantly over four years are on an upward structural trajectory — better coaching, better youth development pipeline, tactical evolution. This is particularly relevant for identifying teams that might outperform their current static Elo or FIFA snapshot. DR Congo (+28 FIFA positions), Uzbekistan (+27), and Norway (+19 in Elo) are teams whose four-year trajectory suggests real structural improvement rather than noise. Expected importance: Medium. Most valuable for identifying structural improvers and decliners not captured in snapshots.

F012 — fifa_rank_4yr_change
Formula: DS11.rank - DS10.rank WHERE team = [team] (positive = improved) Source: DS10, DS11 Range: Approximately −34 to +33 for teams in both datasets Leakage risk: None. Predictive rationale: Ordinal version of four-year trajectory. Lower information content than F011 due to rank compression at the top (the difference between rank 1 and rank 5 is large in actual Elo points but only 4 rank positions), but useful as a supplementary signal and more interpretable for communicating findings. Expected importance: Low-medium. Supplementary to F011.

F013 — elo_fifa_rank_disagreement
Formula: ABS(DS2.rank - DS10.rank_in_elo_system) where ranks are normalized to the 48-team WC field Source: DS2, DS10 Range: 0 (perfect agreement) to 38 (Qatar: Elo rank 95, FIFA rank 57) Leakage risk: None. Both pre-tournament sources. Predictive rationale: When two independent rating systems strongly disagree on a team's quality, prediction uncertainty is higher. Qatar (Elo rank 95, FIFA rank 57 — disagreement of 38), Egypt (Elo rank 51, FIFA rank 29 — disagreement of 22), and Morocco (Elo rank 24, FIFA rank 7 — disagreement of 17) are teams where the systems disagree most. High disagreement does not favor one direction — it signals that the model should assign wider confidence intervals to predictions involving these teams. This feature is primarily used to compute prediction uncertainty bounds in the Monte Carlo simulation rather than as a direct win probability input. Expected importance: Medium in simulation context. Low in point-estimate model. Critical for confidence interval generation.

Group 3 — Historical World Cup Features
Source for all Group 3 features: DS8 (matches_1930_2022.csv, 964 matches). Modern era filter: Year >= 1998 (7 tournaments, 448 matches). WC historical win rate correctly predicted 69.5% of non-draw WC outcomes in backtesting against the same dataset, slightly outperforming Elo alone (66.7%), suggesting WC-specific experience contains signal beyond general competitive rating. Key structural note: 10 of the 48 qualified teams (Czechia, Bosnia-Herzegovina, Turkey under current name, Ivory Coast, Iran, Cape Verde, Jordan, DR Congo, Uzbekistan, Curaçao) have little or no direct WC history in DS8 under their current names and receive missing-class encoding, not zero.

F014 — wc_win_rate_modern
Formula: WC wins in 1998–2022 / WC matches played in 1998–2022 WHERE team = [team] Source: DS8 filtered to Year >= 1998 Range: 0.0 (multiple teams) to 0.667 (Germany, 39 games) Missing value encoding: Teams with zero WC appearances in modern era receive a categorical flag wc_debut_modern = 1 and the win rate is set to the tournament population mean for imputation, never zero. Leakage risk: None. All data pre-dates 2026 tournament. Computed cumulatively through 2022. Predictive rationale: WC-specific win rate captures how well a team performs in the specific competitive environment of the World Cup — under pressure, against WC-calibrated opposition, in a knockout-or-go-home context. Germany (0.667), Brazil (0.659), Netherlands (0.633), and France (0.615) lead this metric in the modern era. These teams consistently outperform their Elo rating in WC contexts due to tournament experience, mental preparation systems, and squad depth. In the backtesting, this feature marginally outperformed Elo alone (69.5% vs 66.7%) confirming its independent signal value. Expected importance: High. Second only to Elo-based features.

F015 — wc_win_rate_knockout_modern
Formula: WC knockout wins in 1998–2022 / WC knockout matches played in 1998–2022 where knockout excludes Group stage and First round Source: DS8 filtered to Year >= 1998 AND Round NOT IN ('Group stage', 'First round') Range: 0.0 to 1.0 (small sample teams); meaningful range for teams with 4+ KO appearances: 0.29 to 0.75 Leakage risk: None. Predictive rationale: Knockout-specific performance captures tournament temperament — the ability to win in single-elimination pressure situations. Several teams show meaningful divergence between their group stage and knockout win rates. Argentina has a higher knockout win rate than group stage rate (they raise performance under elimination pressure). England historically underperforms in knockout situations relative to their group stage. This feature is stage-conditional: it should be weighted more heavily when predicting knockout matches than group stage matches. Expected importance: High for knockout predictions. Medium for group stage.

F016 — wc_avg_gf_modern
Formula: SUM(goals_for in WC 1998–2022) / WC matches played 1998–2022 WHERE team = [team] Source: DS8 filtered to Year >= 1998 Range: 0.0 to 2.77 (Germany: 9.0 goals in group stage 2026 already suggests their attacking output) Leakage risk: None. Predictive rationale: Average goals scored in WC competition captures attacking output under tournament conditions. Teams that score consistently in WC environments carry forwards who are effective against WC-calibrated defences. Portugal and Germany have historically high values. Correlated with Elo but adds the specific attacking dimension. Expected importance: Medium. Correlated with Elo but adds scoring-specific signal.

F017 — wc_avg_ga_modern
Formula: SUM(goals_against in WC 1998–2022) / WC matches played 1998–2022 WHERE team = [team] Source: DS8 filtered to Year >= 1998 Range: 0.0 to 2.5 Leakage risk: None. Predictive rationale: Average goals conceded in WC competition captures defensive solidity specifically in WC environments. Independently valuable from Elo because some teams are asymmetric — strong attack, weak defence or vice versa. France has historically been one of the best defensive teams in WC competition (0.77 GA/match in modern era), which partly explains their win rate. Expected importance: Medium.

F018 — wc_goal_difference_per_game_modern
Formula: (wc_avg_gf_modern - wc_avg_ga_modern) from F016 and F017 Source: DS8 (derived) Range: Approximately −1.5 to +1.8 Leakage risk: None. Predictive rationale: Net goal difference per game is a stronger signal than wins alone because it captures the margin of victories and defeats. Teams that win by large margins consistently (Germany +1.3 GD/game) are more dominant than teams that eke out close wins. Particularly useful for distinguishing between teams with similar win rates but different styles. Expected importance: Medium-high. Often more predictive than win rate alone in football analytics.

F019 — wc_clean_sheet_rate_modern
Formula: WC clean sheets in 1998–2022 / WC matches played 1998–2022 WHERE team = [team] Source: DS8 filtered to Year >= 1998 Range: 0.0 to 0.60 for teams with 5+ appearances Leakage risk: None. Predictive rationale: Clean sheet rate captures defensive strength in WC competition. Defensively superior teams tend to go deep in tournaments — WC championships are typically won with strong defensive foundations. Spain won 2010 conceding only 2 goals in 7 matches. France won 2022 final against a high-scoring Argentina team partly due to defensive recovery in extra time. Expected importance: Medium.

F020 — wc_tournaments_attended
Formula: COUNT(DISTINCT Year in DS8) WHERE team appears as home_team OR away_team Source: DS8 Range: 0 (Curaçao, Uzbekistan, Cape Verde) to 22 (Brazil — every tournament) Missing value encoding: Zero is a valid and meaningful value here. Zero means no WC history, which is a genuine signal — debutants face a structural experience gap. Leakage risk: None. Predictive rationale: Tournament experience captures institutional knowledge — a team that has been to 15 WCs has systems for managing tournament pressure, squad rotation, recovery between matches, and tactical adaptation that a debutant lacks. Brazil and Germany have been to every modern-era tournament; their systems are refined. Cape Verde (0 WC appearances), Uzbekistan (0), and Curaçao (0) face genuine structural uncertainty that this feature quantifies. Expected importance: Medium. Most discriminating at the extremes — debutants vs tournament regulars.

F021 — wc_best_result_encoded
Formula: MAX(stage_reached_encoded) where encoding: Group stage = 1, Round of 16 = 2, Quarter-final = 3, Semi-final = 4, Third place = 5, Runner-up = 6, Champion = 7 Source: DS8, DS12 cross-referenced Range: 0 (never reached WC) to 7 (champion: Brazil 5x, Germany 4x, Argentina 3x, France 2x, England 1x, Spain 1x, Uruguay 2x, Italy 4x, etc.) Leakage risk: None. Predictive rationale: Historical best result captures ceiling achievement. A team that has won the World Cup carries a different cultural relationship with tournament football than one whose best result is the Round of 16. The encoding compresses this into a single ordinal feature while preserving meaningful ordinal distance between positions. Expected importance: Medium-low. Correlated with Elo and win rate features. Most useful as a cultural/institutional strength indicator for teams whose current form diverges from their historical ceiling.

F022 — wc_group_vs_knockout_uplift
Formula: wc_win_rate_knockout_modern - wc_win_rate_group_stage_modern Source: DS8 (derived from F014 and F015) Range: Approximately −0.4 to +0.4 Leakage risk: None. Predictive rationale: This feature directly measures tournament temperament — the degree to which a team elevates performance in elimination situations versus group play. Positive values indicate teams that raise their game when facing elimination (Argentina, historically). Negative values indicate teams that perform well in group stage but underperform when the stakes rise. This is a stage-interaction feature: it should be multiplied against the stage_order feature in Group 7 to capture the elevation effect at higher stages. Expected importance: Medium. High value as an interaction term with stage features.

Group 4 — Recent Form Features
Source for all Group 4 features: DS4 (results.csv) filtered to tournament != 'Friendly' AND date BETWEEN '2024-01-01' AND '2026-06-10' AND home_score IS NOT NULL. This produces 1,872 competitive international matches. The upper bound of June 10, 2026 ensures no 2026 WC matches (which begin June 11) enter any form calculation.

F023 — form_win_rate_last10_competitive
Formula: wins in last 10 competitive matches / 10 where competitive excludes Friendly; if team has fewer than 10 matches in window, use available matches Source: DS4 Range: 0.0 to 1.0 (England: 10 wins in last 10 competitive matches entering tournament — verified from data) Leakage risk: None. Upper date bound is June 10, 2026 — the day before the tournament started. Predictive rationale: Recent win rate in competitive matches captures current team momentum independent of historical reputation. England's 10/10 competitive win rate entering the tournament is a genuine signal of their current form. Brazil's 4/10 suggests they are underperforming their historical rating. This feature captures what the Elo and FIFA ratings are slower to reflect due to their longer lookback windows. Expected importance: High. Recent form is one of the strongest short-term predictors of tournament performance.

F024 — form_avg_gf_last10
Formula: SUM(goals scored in last 10 competitive matches) / matches_played Source: DS4 Range: Approximately 0.8 to 3.2 goals per game for WC teams Leakage risk: None. Predictive rationale: Recent offensive output captures current attacking form independent of historical averages. A team currently averaging 2.5 goals per competitive game is in a different attacking state than one averaging 0.9. Expected importance: Medium.

F025 — form_avg_ga_last10
Formula: SUM(goals conceded in last 10 competitive matches) / matches_played Source: DS4 Range: Approximately 0.4 to 2.1 goals conceded per game for WC teams Leakage risk: None. Predictive rationale: Recent defensive form. Particularly valuable for teams that have recently changed manager or formation — the shift in defensive organization shows up here before it updates in the Elo system. Expected importance: Medium.

F026 — form_gd_last10
Formula: form_avg_gf_last10 - form_avg_ga_last10 Source: DS4 (derived from F024 and F025) Range: Approximately −1.3 to +2.4 Leakage risk: None. Predictive rationale: Net goal difference captures overall recent form quality more precisely than win rate alone, since wins by 4-0 are worth more in the rating than 1-0 wins. Preferred over raw win rate for distinguishing dominant teams from teams that grind out narrow wins. Expected importance: Medium-high.

F027 — form_clean_sheet_rate_last10
Formula: clean sheets in last 10 competitive matches / matches_played Source: DS4 Range: 0.0 to 0.9 Leakage risk: None. Predictive rationale: Recent defensive solidity captures the current state of goalkeeping and defensive organization. Clean sheet rate in recent competitive matches is more sensitive to current form than historical WC defensive stats and picks up on genuine defensive improvements or deteriorations faster. Expected importance: Medium.

F028 — form_unbeaten_streak_entering
Formula: consecutive matches without defeat (win or draw) in competitive matches immediately before June 11, 2026 Source: DS4 Range: 0 to 35+ (longer streaks reflect teams that have not lost in many months) Leakage risk: None. Predictive rationale: Unbeaten streaks carry psychological momentum. A team entering a tournament on a 10-match unbeaten streak has built cohesion and confidence. Streaks ending in draws rather than wins provide weaker momentum signals — the streak length and the proportion that were wins both matter. Expected importance: Low-medium. Captures psychological momentum but is somewhat redundant with win rate features.

Group 5 — Tournament Features
Source for all Group 5 features: DS1 (matches.csv, Arc3) for 44 matches with full tactical data. DS1-ext (four June 23 results, scores only) for Portugal, Colombia, England, Croatia. All features in this group are computed from the frozen 48-match result set. Features marked TACTICAL-ONLY are unavailable for the four June 23 teams in their second match.

F029 — tourn_pts_md2
Formula: points earned in 2026 WC Matchday 1 + Matchday 2 (3 for win, 1 for draw, 0 for loss) Source: DS1 + DS1-ext (all 48 teams) Range: 0 (Haiti, Iraq, Türkiye, Panama, Uzbekistan, Senegal, Jordan, Qatar — various with 0) to 6 (Mexico, United States, Germany, France, Norway, Argentina, Colombia) Leakage risk: None. All 48 matches are frozen results at or before June 23, 2026. Predictive rationale: Points after two matchdays directly captures current tournament performance and qualification trajectory. Teams with 6 points have won both matches and enter MD3 secure; teams with 0 points are eliminated or facing elimination. This is the most current signal available and highly correlated with knockout stage qualification probability. Validated predictive signal: top group stage scorers reach the QF at a 43–67% rate across modern WC tournaments. Expected importance: Highest among tournament features.

F030 — tourn_gd_md2
Formula: (goals_for - goals_against) across all completed 2026 WC matches Source: DS1 + DS1-ext (all 48 teams) Range: −9 (worst) to +7 (Germany: 9 scored, 2 conceded after 2 matches) Leakage risk: None. Predictive rationale: Goal difference is a stronger signal than points alone — it captures dominance within wins and resilience within losses. Germany (+7), the USA (+5), France (+5), and Argentina (+5) lead on this metric. These are teams that are not just winning but winning convincingly, suggesting genuine quality rather than lucky results. GD is the first tiebreaker in the group stage and influences bracket seeding, making it directly relevant to predicting match difficulty in the knockout rounds. Expected importance: High.

F031 — tourn_gf_md2
Formula: goals scored across all completed 2026 WC matches Source: DS1 + DS1-ext (all 48 teams) Range: 0 to 9 (Germany: 9 goals in 2 matches) Leakage risk: None. Predictive rationale: Raw attacking output in the tournament. Germany (9), Canada (7), Netherlands (7), Norway (7), USA (6), and Japan (6) lead this metric. High tournament scoring correlates with teams in positive attacking momentum specifically in this competition's conditions (pitches, opponents, intensity). Historical validation shows top group-stage scorers reach QF at 43–67% depending on year (mean approximately 56%). Expected importance: Medium-high.

F032 — tourn_ga_md2
Formula: goals conceded across all completed 2026 WC matches Source: DS1 + DS1-ext (all 48 teams) Range: 0 to 9 Leakage risk: None. Predictive rationale: Tournament defensive solidity. Spain (0 conceded), Argentina (0), and Mexico (0) have kept clean sheets across both matches — a strong signal of defensive organization in the current competition. Inverted signal — lower is better. Expected importance: Medium-high.

F033 — tourn_avg_possession (TACTICAL-ONLY)
Formula: MEAN(home_possession OR away_possession) across DS1 matches WHERE team appears Source: DS1 only (44 matches) Coverage: 44 of 48 teams. Portugal, Colombia, England, Croatia: MD1 only; flag has_full_tactical_md2 = 0 Range: 25.0 (Paraguay: averaged 28.5% over 2 games) to 75.5 (Türkiye) Leakage risk: None. Predictive rationale: Average possession captures tactical identity and game control. High possession teams (Spain 70.5%, Canada 70%, Türkiye 75.5%) are playing a control-based style. Türkiye's 75.5% average possession paired with 0 points is a notable anomaly — they dominated possession but lost both matches, suggesting a fundamental finishing problem. Low possession teams that succeed (e.g. Paraguay, 28.5% possession, still won a match) are counter-attacking effectively. Possession divergence between team and opponent is a particularly valuable interaction feature. Expected importance: Medium. More valuable as an interaction with goals scored than as a standalone feature.

F034 — tourn_avg_sot (TACTICAL-ONLY)
Formula: MEAN(shots_on_target FOR or AGAINST) across DS1 matches WHERE team appears Source: DS1 only Coverage: 44 of 48 teams Range: 0.5 to 9.5 per match Leakage risk: None. Predictive rationale: Shots on target per match is a leading indicator of attacking quality — more direct than goals scored because it filters out luck in finishing. Canada (7.0 SOT/match), Germany (9.5), and Spain (7.5) top this metric. SOT is also more stable than goals scored over small samples (two matches), making it a better predictor of ongoing attacking quality than raw goal counts. Expected importance: Medium-high. More stable than goals over 2-match sample.

F035 — tourn_sot_conceded (TACTICAL-ONLY)
Formula: MEAN(opponent shots_on_target) across DS1 matches WHERE team appears Source: DS1 only Coverage: 44 of 48 teams Range: 1.0 to 13.5 per match Leakage risk: None. Predictive rationale: Shots on target conceded measures defensive solidity and goalkeeper effectiveness. Spain (1.0 SOT conceded per match), France (1.0), and Argentina (0.5) are conceding minimal quality chances. Inverted signal — lower is better for defensive prediction. Expected importance: Medium-high.

F036 — tourn_shot_conversion_rate (TACTICAL-ONLY)
Formula: tourn_gf_md2 / total_shots_for_across_tournament where total shots from DS1 only Source: DS1 + DS1-ext (hybrid: goals from all 48, shots from DS1 44) Range: 0.0 to 0.50 (maximum observed in DS1) Leakage risk: None. Note: for June 23 teams, this uses DS1 match shots only — the formula uses goals from both matches but shots from MD1 only. This is explicitly flagged. Predictive rationale: Conversion rate measures clinical finishing efficiency. A team scoring 6 goals from 10 shots is more clinically efficient than one scoring 6 from 40 shots. Sustainable conversion rates typically fall in the 0.10–0.20 range — teams significantly above this are benefiting from finishing luck that will likely regress; teams significantly below are unlucky and may improve. Germany is scoring at high efficiency; Türkiye is shooting prolifically but converting poorly. Expected importance: Medium. High value as an overperformance/underperformance signal.

F037 — tourn_yellow_cards_md2
Formula: SUM(yellow_cards) across DS1 matches WHERE team appears Source: DS1 only (all 48 teams — yellow card columns populated for all DS1 matches) Range: 0 to 6 (some teams across 2 matches) Leakage risk: None. Predictive rationale: Yellow card accumulation has direct tournament consequences — teams approaching the two-yellow-card-equals-suspension threshold face squad management decisions in critical matches. A team entering MD3 with 5 yellows distributed among key players may rotate, reducing effectiveness. Also captures playing style — high-pressure, physical teams accumulate more yellows. Expected importance: Low-medium. Most relevant for specific player suspension analysis, which is beyond the scope of the feature set.

F038 — tourn_formation_changed
Formula: 1 IF home_formation MD1 != home_formation MD2 FOR team; else 0 (comparing across DS1 matches) Source: DS1 only (44 teams — June 23 teams have only MD1 formation) Range: Binary: 0 (no change) or 1 (changed) Coverage: 18 of 44 DS1 teams changed formation (41%) Leakage risk: None. Predictive rationale: Formation change signals managerial adaptation. Teams that changed formation between MD1 and MD2 may have done so in response to a poor result (Czechia: 3-4-3 to 5-3-2 after losing MD1) or to exploit a specific upcoming opponent (Ecuador: 4-2-2-2 to 3-1-4-2). Formation stability indicates a settled tactical system; frequent changes signal uncertainty. The direction of change matters — switching from a high-press to a defensive shape is different from adding an attacking midfielder. Expected importance: Low. Useful as a contextual flag rather than a direct predictive feature. May interact with form features.

F039 — has_full_tactical_md2
Formula: 1 IF team has full DS1 tactical data for both MD1 and MD2; 0 IF MD2 was June 23 (scores only) Source: DS1, DS1-ext Range: Binary. 44 teams = 1; Portugal, Colombia, England, Croatia = 0 Leakage risk: None. This is a data availability flag, not a derived outcome. Predictive rationale: This is a model quality control feature — it tells the model that tactical features F033–F038 are based on one match only for four teams, and that their MD2 tactical stats are structurally absent rather than imputed. Including this as a feature allows the model to learn that predictions for these four teams carry higher tactical uncertainty. Without this flag, the model would treat their one-match tactical averages as if they were two-match averages, overstating precision. Expected importance: Low as a direct predictor. High as a model integrity feature.

Group 6 — Penalty Features
Source for all Group 6 features: DS6 (shootouts.csv, 678 total shootout records across all international competitions). DS8 (matches_1930_2022.csv, 35 WC shootout records specifically). Only Group 6 features use DS6 directly — it is not used elsewhere in the feature specification.

F040 — shootout_win_rate_alltime
Formula: total shootout wins in DS6 / total shootout appearances in DS6 WHERE team = [team] Source: DS6 Range: 0.0 to 1.0 (extreme values for teams with 1–2 appearances); for teams with 5+ appearances: approximately 0.20 (Netherlands) to 0.75 (Germany) Missing value: Teams with zero shootout appearances receive a special encoding of 0.5 (prior mean) with a shootout_naive = 1 flag. Four WC 2026 teams have never appeared in a DS6 shootout. Leakage risk: None. DS6 is entirely pre-tournament. Predictive rationale: Historical shootout win rate captures a team's ability to win under maximum pressure — five players, one shot each, tournament on the line. Germany (6W/8P = 0.75), Argentina (15W/23P = 0.65), and Portugal (5W/8P = 0.62) have strong shootout records. England (4W/12P = 0.33) and Netherlands (2W/10P = 0.20) are historically poor in shootouts, which has direct knockout-stage implications if their matches go to extra time. This feature activates only in the penalty shootout model layer, not in the main match prediction model. Expected importance: Highest within Group 6. Direct impact on knockout predictions that go to extra time.

F041 — shootout_appearances_total
Formula: COUNT(shootout appearances in DS6) WHERE team appears as home_team OR away_team Source: DS6 Range: 0 to 28 (maximum across all international teams) Leakage risk: None. Predictive rationale: Volume of shootout experience modulates confidence in the win rate estimate (F040). A team with 20+ shootout appearances has a stable, reliable win rate. A team with 2 appearances has a 0/2 or 2/2 rate that is essentially noise. This feature is used primarily as a reliability weight for F040 — in the simulation, F040 is shrunk toward 0.5 for low-appearance teams and trusted more for high-appearance teams. Expected importance: Medium within Group 6. Most important as a reliability modifier for F040.

F042 — shootout_win_rate_wc_only
Formula: WC shootout wins / WC shootout appearances using DS8.home_penalty and DS8.away_penalty columns Source: DS8 Range: 0.0 to 1.0; 35 WC shootouts in DS8 means small samples per team Missing value: Most teams have 0–3 WC shootout appearances — high variance. Use F040 as primary and this as a supplementary adjustment only. Leakage risk: None. Predictive rationale: WC-specific shootout record is more directly relevant than all-competition shootout record (F040) because WC knockout shootouts occur under maximum pressure with global attention. Some teams perform differently in WC shootouts versus other competitions. Argentina's WC-specific record (multiple wins in 2022 alone) is stronger than their all-time record. However, sample sizes are very small — most teams have participated in 0–2 WC shootouts, making this feature noisy at the individual level. Expected importance: Low as standalone. Used as an adjustment factor to F040 rather than an independent feature.

F043 — shootout_first_shooter_advantage
Formula: 1 IF team historically shoots first in shootouts; 0 IF team shoots second; 0.5 IF insufficient data using DS6.first_shooter column Source: DS6 (first_shooter column is 62% null — 422 of 678 records) Range: 0, 0.5, or 1 Leakage risk: None. Significant null rate (62%) makes this feature unreliable. Predictive rationale: Shooting first in a penalty shootout confers a measurable psychological advantage — the shooting-first team wins approximately 60% of shootouts historically (academic literature finding, not directly derivable from DS6 due to null rate). However, the 62% null rate in DS6 for this column means it cannot be reliably computed for most teams, limiting its utility. Expected importance: Low. Included as experimental. High null rate significantly undermines reliability.

Group 7 — Match Context Features
Source for all Group 7 features: DS16 (matches.csv, Arc_base), DS17 (teams.csv), DS18 (host_cities.csv), DS19 (tournament_stages.csv), DS2 (confederation field).

F044 — stage_order
Formula: DS19.stage_order WHERE DS16.stage_id = DS19.id for the specific match being predicted Source: DS16, DS19 Range: 1 (Group stage) to 7 (Final) Leakage risk: None. Stage structure is fixed and known pre-tournament. Predictive rationale: Stage escalation captures the increasing importance and pressure of later rounds. Higher-stage matches have different dynamics than group stage matches — both in terms of team motivation (no second chances) and in terms of the quality of opposition (only the best teams remain). This feature interacts with F022 (wc_group_vs_knockout_uplift) to capture teams that specifically elevate their performance in high-stakes matches. Expected importance: Medium. Critical as an interaction term.

F045 — is_knockout
Formula: 1 IF DS16.stage_id >= 2 (Round of 32 or later); 0 IF stage_id = 1 (Group stage) Source: DS16 Range: Binary Leakage risk: None. Predictive rationale: The binary distinction between group stage and knockout stage is the single most important stage feature. Group stage allows draws; knockout does not (except after 90 minutes). Draw probability drops from 24.7% (group stage, verified from modern WC data) to approximately 21.4% at 90 minutes (knockout stage — teams press harder). More importantly, knockout stage changes the risk-reward calculus for attacking versus defensive play. Expected importance: High. Directly changes the model structure — separate models for group and knockout stages are recommended.

F046 — venue_region_cluster
Formula: DS18.region_cluster WHERE DS18.id = DS16.city_id encoded as categorical: East/Central/West Source: DS16, DS18 Range: Three categories: East (Atlanta, Boston, Miami, New York, Philadelphia, Toronto), Central (Dallas, Houston, Kansas City, Guadalajara, Mexico City, Monterrey), West (Los Angeles, San Francisco, Seattle, Vancouver) Leakage risk: None. Venue assignments are fixed pre-tournament. Predictive rationale: Regional cluster captures climate and travel conditions. Teams from hot, humid climates (CONCACAF, CAF) may have advantages in Central region venues (Mexico City elevation is also relevant). European teams playing in East coast venues (milder, similar to European conditions) face less climate disadvantage than in Central or West venues. The key interaction is confederation vs region cluster — CONMEBOL teams playing in Central venues are closest to home conditions; UEFA teams in East venues are most comfortable. Expected importance: Low-medium. Small effect size but real directional signal.

F047 — venue_country
Formula: DS18.country WHERE DS18.id = DS16.city_id encoded as: USA / Canada / Mexico Source: DS16, DS18 Range: Three categories Leakage risk: None. Predictive rationale: Which host nation's venues a team plays in affects crowd composition. Mexico's group stage matches in Guadalajara and Mexico City will have overwhelmingly pro-Mexico crowds. CONCACAF teams in USA venues get partial home crowd effects. This is a more granular version of F007 (is_host) — while F007 is team-level, F047 is match-level and captures whether the specific venue is in that team's home country. Expected importance: Low-medium. Most important for Mexico specifically.

F048 — confederation
Formula: DS2.confederation WHERE country = [team] encoded as: UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC Source: DS2 Range: Six categories: UEFA (19 teams), CONMEBOL (6), CONCACAF (6), CAF (9), AFC (7), OFC (1 — New Zealand) Leakage risk: None. Predictive rationale: Confederation captures the competitive ecosystem a team comes from, which shapes their style, physical conditioning, and tactical preparation. CONMEBOL and UEFA teams historically dominate WC outcomes (every WC since 1930 has been won by a European or South American team). CAF and AFC teams have broken into late knockout rounds but not won. This feature helps the model account for systematic confederation-level strength differentials that may not be fully captured by Elo ratings (which can be inflated by wins within weaker confederations). Expected importance: Medium. Most important for calibrating predictions involving CAF and AFC teams.

F049 — kickoff_local_hour
Formula: HOUR(DS16.kickoff_at converted to local venue timezone) Source: DS16 (kickoff_at is UTC-aware) Range: In UTC: 0, 1, 2, 3, 4, 5, 6, 7 (night/early morning UTC = afternoon local US time); 16–23 (afternoon UTC = morning local). Local hour range is approximately 12:00 to 22:00 across all venues. Leakage risk: None. Kickoff times are pre-scheduled. Predictive rationale: Kickoff time affects physical performance through heat exposure and circadian rhythm disruption. Afternoon kickoffs in Central region venues (Mexico City, Dallas) during June expose teams to temperatures above 35°C. European teams accustomed to cooler conditions may underperform in afternoon heat. Night kickoffs in East coast venues approximate neutral conditions for all teams. This is a small but systematic effect. Expected importance: Low. Small effect size. Include as experimental feature.

F050 — elo_rank_disagreement (context application)
Formula: From F013 — applied here as a match context feature rather than a team feature Formula at match level: ABS(team_elo_rank - team_fifa_rank) + ABS(opponent_elo_rank - opponent_fifa_rank) Source: DS2, DS10 Range: 0 to 76 (sum of both teams' disagreements) Leakage risk: None. Predictive rationale: At the match level, high total disagreement between rating systems for both teams involved signals that the match outcome is more uncertain than the Elo win expectancy alone would suggest. This feature feeds directly into the confidence interval width in the Monte Carlo simulation — matches with high disagreement scores get wider probability bands. Expected importance: Low as a direct predictor. Critical for simulation uncertainty quantification.

Global Feature Rankings
All 50 features ranked from highest to lowest expected predictive value, based on historical validation evidence, theoretical grounding, and empirical feature importance patterns from tournament football analytics literature.
Rank
Feature ID
Feature name
Group
Rationale for rank
1
F002
elo_win_expectancy
Elo
Calibrated probability from the most theoretically grounded rating system; 66.7% directional accuracy validated
2
F001
elo_rating
Elo
Primary absolute strength signal; anchor for all matchup features
3
F003
elo_rating_delta
Elo
Directional matchup signal; most important interaction variable
4
F029
tourn_pts_md2
Tournament
Most current tournament-specific signal; directly captures 2026 form
5
F008
fifa_points
FIFA
Independent rating system providing genuine additional signal; 1,876 range
6
F014
wc_win_rate_modern
Historical WC
69.5% directional accuracy in backtesting — marginally outperforms Elo alone
7
F023
form_win_rate_last10_competitive
Recent form
Captures current form not yet reflected in slower-updating Elo/FIFA systems
8
F030
tourn_gd_md2
Tournament
Captures dominance quality not captured in points alone
9
F015
wc_win_rate_knockout_modern
Historical WC
Stage-specific WC performance — critical for knockout predictions
10
F026
form_gd_last10
Recent form
Net recent performance; more discriminating than win rate alone
11
F018
wc_goal_difference_per_game_modern
Historical WC
Captures dominance quality in WC-specific context
12
F034
tourn_avg_sot
Tournament
More stable than goals over 2-match sample; leading indicator
13
F031
tourn_gf_md2
Tournament
Current attacking output
14
F024
form_avg_gf_last10
Recent form
Recent attacking output
15
F035
tourn_sot_conceded
Tournament
Current defensive quality
16
F045
is_knockout
Match context
Critical structural feature — changes model class entirely
17
F032
tourn_ga_md2
Tournament
Current defensive output
18
F025
form_avg_ga_last10
Recent form
Recent defensive form
19
F009
fifa_points_delta
FIFA
Short-term momentum signal
20
F019
wc_clean_sheet_rate_modern
Historical WC
WC defensive solidity
21
F004
elo_rating_career_peak
Elo
Historical ceiling — identifies underperforming giants
22
F040
shootout_win_rate_alltime
Penalty
Critical for knockout simulations; Germany 0.75 vs England 0.33
23
F017
wc_avg_ga_modern
Historical WC
WC-specific defensive output
24
F016
wc_avg_gf_modern
Historical WC
WC-specific attacking output
25
F022
wc_group_vs_knockout_uplift
Historical WC
Tournament temperament — stage interaction term
26
F011
fifa_points_4yr_change
FIFA
Structural trajectory signal
27
F036
tourn_shot_conversion_rate
Tournament
Efficiency vs luck signal
28
F007
elo_is_host
Elo
Host advantage — meaningful but small effect
29
F027
form_clean_sheet_rate_last10
Recent form
Recent defensive solidity
30
F020
wc_tournaments_attended
Historical WC
Experience gap — most relevant for debutants
31
F044
stage_order
Match context
Pressure escalation feature
32
F033
tourn_avg_possession
Tournament
Tactical identity signal
33
F005
elo_rating_career_avg
Elo
Consistency signal; mean-reversion indicator
34
F013
elo_fifa_rank_disagreement
FIFA
Uncertainty quantification for simulation
35
F048
confederation
Match context
Systematic confederation-level strength
36
F021
wc_best_result_encoded
Historical WC
Cultural ceiling achievement
37
F041
shootout_appearances_total
Penalty
Reliability weight for F040
38
F012
fifa_rank_4yr_change
FIFA
Ordinal trajectory — supplementary to F011
39
F006
elo_rank
Elo
Ordinal position — supplementary to F001
40
F046
venue_region_cluster
Match context
Climate and travel effects
41
F050
elo_rank_disagreement (match)
Match context
Simulation uncertainty width
42
F028
form_unbeaten_streak_entering
Recent form
Psychological momentum
43
F037
tourn_yellow_cards_md2
Tournament
Suspension risk
44
F042
shootout_win_rate_wc_only
Penalty
WC-specific shootout — noisy small sample
45
F047
venue_country
Match context
Specific host nation advantage
46
F010
fifa_rank_delta
FIFA
Narrow range limits discrimination
47
F039
has_full_tactical_md2
Tournament
Model integrity flag
48
F049
kickoff_local_hour
Match context
Small climate effect
49
F038
tourn_formation_changed
Tournament
Low direct signal
50
F043
shootout_first_shooter_advantage
Penalty
62% null rate undermines reliability

Final Feature Table Schema
One row per team per match. For a prediction match between Team A and Team B, two rows are generated — one from each team's perspective — with features computed for that team relative to its opponent.
Column name                     Type        Nullable    Source
─────────────────────────────────────────────────────────────────────
match_id                        int         No          DS16
team_canonical                  str         No          DS17
opponent_canonical              str         No          DS17
match_date                      date        No          DS16
stage_id                        int         No          DS16
stage_order                     int         No          DS19
is_knockout                     bool        No          DS16
venue_city                      str         No          DS18
venue_country                   str         No          DS18
venue_region_cluster            str         No          DS18
confederation                   str         No          DS2
 
elo_rating                      float       No          DS2
elo_win_expectancy              float       No          DS2 (derived)
elo_rating_delta                float       No          DS2 (derived)
elo_rating_career_peak          float       No          DS2
elo_rating_career_avg           float       No          DS2
elo_rank                        int         No          DS2
elo_is_host                     bool        No          DS2
 
fifa_points                     float       No          DS10
fifa_points_delta               float       No          DS10
fifa_rank_delta                 float       No          DS10
fifa_points_4yr_change          float       Yes         DS10+DS11 (null if not in DS11)
fifa_rank_4yr_change            float       Yes         DS10+DS11
elo_fifa_rank_disagreement      float       No          DS2+DS10
 
wc_win_rate_modern              float       Yes         DS8 (null+flag if debut)
wc_win_rate_knockout_modern     float       Yes         DS8
wc_avg_gf_modern                float       Yes         DS8
wc_avg_ga_modern                float       Yes         DS8
wc_gd_per_game_modern           float       Yes         DS8 (derived)
wc_clean_sheet_rate_modern      float       Yes         DS8
wc_tournaments_attended         int         No          DS8 (0 valid)
wc_best_result_encoded          int         No          DS8+DS12 (0 if never appeared)
wc_group_vs_knockout_uplift     float       Yes         DS8 (derived)
wc_debut_modern_flag            bool        No          DS8
 
form_win_rate_last10            float       No          DS4
form_avg_gf_last10              float       No          DS4
form_avg_ga_last10              float       No          DS4
form_gd_last10                  float       No          DS4
form_clean_sheet_rate_last10    float       No          DS4
form_unbeaten_streak_entering   int         No          DS4
 
tourn_pts_md2                   int         No          DS1+DS1ext
tourn_gd_md2                    int         No          DS1+DS1ext
tourn_gf_md2                    int         No          DS1+DS1ext
tourn_ga_md2                    int         No          DS1+DS1ext
tourn_avg_possession            float       Yes         DS1 (null for 4 teams MD2)
tourn_avg_sot                   float       Yes         DS1 (null for 4 teams MD2)
tourn_sot_conceded              float       Yes         DS1 (null for 4 teams MD2)
tourn_shot_conversion_rate      float       Yes         DS1 (partial for 4 teams)
tourn_yellow_cards_md2          int         No          DS1
tourn_formation_changed         bool        Yes         DS1 (null for 4 teams MD2)
has_full_tactical_md2           bool        No          DS1+DS1ext
 
shootout_win_rate_alltime       float       No          DS6 (0.5 for no-history teams)
shootout_appearances_total      int         No          DS6 (0 valid)
shootout_win_rate_wc_only       float       Yes         DS8 (null for most teams)
shootout_naive_flag             bool        No          DS6
 
stage_order                     int         No          DS19
is_knockout                     bool        No          DS16
venue_region_cluster            str         No          DS18
confederation                   str         No          DS2
kickoff_local_hour              int         No          DS16
 
outcome                         int         Yes         Target: 2=win,1=draw,0=loss
                                                        (null for future matches)
Total columns: 55 (50 features + 5 identity columns) Total rows at freeze: 96 (48 matches × 2 team perspectives) Total rows for prediction: 112 additional (56 future matches × 2 perspectives)

Recommended Features to Keep
The following 25 features form the core model. They have strong theoretical grounding, verified predictive signal, low null rates, and low redundancy with each other:
F001, F002, F003 (Elo core triad), F007 (host), F008 (FIFA points), F009 (FIFA momentum), F011 (4yr trajectory), F013 (disagreement — for simulation), F014 (WC win rate modern), F015 (WC knockout win rate), F018 (WC GD per game), F022 (WC temperament uplift), F023 (form win rate last 10), F026 (form GD last 10), F029 (tournament points MD2), F030 (tournament GD MD2), F034 (tournament SOT), F035 (tournament SOT conceded), F036 (shot conversion), F040 (shootout win rate), F041 (shootout appearances), F044 (stage order), F045 (is knockout), F046 (venue region), F048 (confederation).

Experimental Features
The following 15 features are worth including in ablation testing but have significant caveats around null rates, small samples, or theoretical uncertainty. Include in a second model variant and compare performance:
F004 (career Elo peak), F005 (career Elo avg), F019 (WC clean sheet rate), F020 (WC tournaments attended), F021 (WC best result encoded), F024 (form avg GF), F025 (form avg GA), F027 (form clean sheet), F028 (form unbeaten streak), F031 (tournament GF), F032 (tournament GA), F033 (tournament possession), F042 (WC-only shootout rate), F047 (venue country), F050 (match-level disagreement).

Features to Remove
The following 10 features should be excluded from all models. The reasons are: high null rates that cannot be reliably imputed, insufficient variance to discriminate between teams, redundancy with higher-quality features in the same group, or model integrity flags that should not be treated as predictors:
F006 (Elo rank — redundant with F001 and F002), F010 (FIFA rank delta — narrow range −6 to +4, insufficient variance), F012 (FIFA rank 4yr change — ordinal version of F011 which is already included), F016 (WC avg GF — redundant with F018 GD per game), F017 (WC avg GA — redundant with F018), F037 (yellow cards — suspension analysis beyond feature scope), F038 (formation changed — low signal, 41% base rate), F039 (has_full_tactical_md2 — keep as metadata, not as model input), F043 (shootout first shooter — 62% null rate), F049 (kickoff local hour — small effect size, complex timezone conversion for marginal gain).
 
Final Modeling Specification
Frozen: June 23, 2026 — Post-Matchday 2
1. Training Dataset Construction
The fundamental constraint
The entire training operation is governed by one structural reality: 448 rows of modern-era WC matches (1998–2022) is the primary training corpus. This is not a large dataset. Every modeling decision — depth of trees, number of features, regularisation strength, augmentation strategy — must be made in the context of this constraint. A model that requires 10,000 training examples to generalise has no business being applied to a 448-row dataset without explicit justification.
Primary training corpus: WC-only
Source: DS8 filtered to Year >= 1998 Rows: 448 matches (7 tournaments × 64 matches each) Structure: Each match generates two training rows — one from the home team perspective, one from the away team perspective. This doubles the effective sample to 896 rows while ensuring the model learns team-level features rather than match-level features.
The chronological fold structure is exact and non-negotiable:
Fold
Train years
Validate year
Train rows
Validate rows
1
2002–2022
1998
384
64
2
1998, 2002–2018, 2022
2002
384
64
3
1998–2002, 2006–2022
2006
384
64
4
1998–2006, 2010–2022
2010
384
64
5
1998–2010, 2014–2022
2014
384
64
6
1998–2014, 2018–2022
2018
384
64
7 (calibration)
1998–2018
2022
384
64
Fold 7 is the calibration holdout and is used exclusively for probability calibration. It must never be used for hyperparameter tuning. Folds 1–6 are used for hyperparameter search via cross-validation.
At the time of each match being trained on, the features must be constructed from data available before that match. For Elo features this means using the snapshot_date = YEAR-1-12-31 row from DS2. A match played in 2010 uses 2009 year-end Elo ratings, not 2010 ratings — constructing Elo features with same-year ratings would constitute leakage.
Secondary training corpus: augmented competitive matches
Source: DS4 filtered to tournament != 'Friendly' AND date >= '2014-01-01' AND date < '2026-06-11' AND home_score IS NOT NULL Total available: 8,311 competitive matches Used: Tier 1 only — 1,070 matches from major tournaments (FIFA World Cup, UEFA Euro, Copa América, AFC Asian Cup, African Cup of Nations, Gold Cup, CONCACAF Nations League)
The augmented corpus is used in a second model variant only. It is not mixed into the primary corpus. The two variants are trained separately and their predictions are combined through a weighted ensemble.
The rationale for including augmented data is that 448 WC matches provides insufficient coverage for teams with few WC appearances. Uzbekistan, Curaçao, Cape Verde, Jordan, DR Congo, Haiti, Iraq, and Ivory Coast have limited or zero WC history. Tier 1 competitive matches provide signal for these teams' strength while maintaining the quality threshold that separates meaningful competitive matches from noise.
The rationale for excluding Tier 2 matches (qualifiers, Nations League) from primary training is that qualification matches and Nations League games have meaningfully different competitive dynamics — teams rest players, test systems, and do not play at maximum intensity. Including 5,526 qualification matches would dilute the signal from 448 WC matches and likely harm performance on the WC prediction task.
Feature construction timing rules
Every row in the training dataset requires features constructed at a precise point in time. The rules are:
For WC historical matches (DS8 primary corpus): Elo features from year_end_snapshot = match_year - 1. FIFA ranking features from the nearest available ranking snapshot preceding the tournament. WC historical features computed cumulatively through tournaments completed before the match year. Recent form features from DS4 competitive matches in the 24 months preceding the match date.
For 2026 prediction: Elo from snapshot_date = 2026-05-27. FIFA from DS10 date = 2026-06-08. WC historical through 2022. Recent form through June 10, 2026. In-tournament features from frozen 48-match result set.
The twelve teams with no modern WC history (Turkey, Iran, South Korea, Uzbekistan, Czechia, Jordan, Ivory Coast, DR Congo, Iraq, Cape Verde, Haiti, Curaçao) receive wc_debut_modern_flag = 1 and their WC-specific historical features are imputed with the mean of teams in the same confederation that have at least three WC appearances. This is a deliberate regression-to-confederation-mean strategy rather than imputing from the global mean, which would understate the experience gap for CAF and AFC debutants relative to UEFA and CONMEBOL teams.

2. Target Variable Design
Three targets, three model classes
There is no single outcome variable. Three distinct target definitions are required because the competitive situation changes fundamentally between group stage, knockout regular time, and penalty shootouts.
Target T1 — Group stage match outcome (three-class)
Class
Label
Encoding
Frequency in modern WC
Home team wins
WIN
2
40.5% (136/336)
Draw
DRAW
1
24.7% (83/336)
Away team wins
LOSS
0
34.8% (117/336)
The class imbalance is modest and does not require aggressive resampling. Class weights of {WIN: 1.2, DRAW: 1.5, LOSS: 1.1} are recommended to moderately upweight the minority draw class, reflecting that draws are harder to predict and carry different information than decisive results.
The three-class encoding uses 2/1/0 rather than 1/0.5/0 or −1/0/1 to maintain compatibility with XGBoost's multi-class objective function (multi:softprob) which expects non-negative integer class labels.
Target T2 — Knockout stage outcome at 90 minutes (two-class with conditional third stage)
Class
Label
Encoding
Frequency in modern WC KO
Home/favoured team wins at 90min
WIN
1
50.9% of non-draws
Away/underdog team wins at 90min
LOSS
0
27.7% of all KO
Draw at 90min (triggers penalty model)
DRAW
—
21.4% (24/112)
The knockout model is a two-class classifier predicting Win or Loss from the perspective of Team A. The draw class is not predicted by the knockout model — instead, the model outputs P(Win) and P(Loss), and the draw probability at 90 minutes is estimated as a fixed constant derived from historical data (21.4%). At prediction time:
P(Win at 90min) = model_output × (1 - 0.214)
P(Draw at 90min) = 0.214
P(Loss at 90min) = (1 - model_output) × (1 - 0.214)
When a simulation run draws a Draw at 90 minutes, the penalty shootout model activates. The knockout model is trained on the 88 non-draw matches from 112 modern WC knockout matches (excluding the 24 drawn matches).
The home/away designation for knockout matches is nominal. In the WC bracket, there is no true home team — DS16 assigns a home_team_id based on bracket position. The home-side bias documented in domestic football (approximately 64.8% home win rate in our KO non-draw data) reflects the bracket convention, not a genuine venue effect, and should be treated with caution. Neutral venue flag from DS4 should be used to zero out any home-side adjustment.
Target T3 — Penalty shootout winner (two-class)
Class
Label
Encoding
Source
First-listed team wins shootout
WIN
1
DS6 winner column
Second-listed team wins shootout
LOSS
0
DS6 winner column
Base rate of winning a penalty shootout: approximately 50% by definition (one team wins, one loses). Any model that cannot outperform 50% should be discarded and replaced with an Elo-adjusted prior.

3. Chronological Validation
The core principle
Standard k-fold cross-validation is prohibited for this project. Football matches are temporally correlated — teams, managers, tactical systems, and competitive landscapes evolve across time, and a model trained on data from 2014 should not be validated against 2002 data it has already seen. Applying random cross-validation folds would allow 2022 information to leak into the training set when validating on 2010 data, producing artificially optimistic performance estimates.
The validation architecture uses leave-one-tournament-out cross-validation with strict temporal ordering.
Fold construction rules
Each validation fold trains on all WC tournaments except the held-out year. The held-out tournament must always be chronologically later than all training tournaments, or the fold must explicitly exclude all data after the held-out tournament's date. The six active folds (excluding the calibration fold) are used for hyperparameter tuning. The calibration fold (2022) is used only for probability calibration after all hyperparameter decisions are finalised.
This means the model used for 2026 predictions will be trained on all seven WC tournaments (1998–2022) simultaneously, validated conceptually through the average of the six non-calibration fold scores.
Performance metrics
Three metrics are computed on each validation fold. All three must be reported and no single metric dominates the model selection decision.
Accuracy (proportion of correctly classified outcomes, excluding draws for the binary interpretation): the validated value for the Elo baseline is 66.7%, and for WC win rate as a feature it is 69.5%. The model must outperform 69.5% on the non-calibration fold average to justify its complexity over a simpler heuristic. If it does not, the logistic regression baseline (Section 5) is preferred.
Ranked Probability Score (RPS): the primary metric for probabilistic predictions. RPS penalises confident wrong predictions more severely than uncertain wrong predictions, which is the correct loss function for a tournament simulation that uses the full probability distribution rather than just the argmax. Lower RPS is better. The Elo win expectancy baseline produces an RPS that serves as the benchmark — the model must beat this benchmark on the calibration holdout.
Brier Score: a secondary calibration metric measuring the mean squared difference between predicted probabilities and actual outcomes. A perfectly calibrated model achieves a Brier Score of 0 on a test set where outcomes are certain. The practical target is a Brier Score below the benchmark of predicting 50% for every match (a Brier Score of 0.25 for binary outcomes).
Recency weighting
The validated evidence shows that Elo predictive accuracy is higher in 2014–2022 (69.4%) than in 1998–2010 (63.6%). This suggests the football environment has become more predictable, possibly due to professionalisation, player data availability, and the growing influence of quantitative analysis in team preparation. To account for this recency effect, training rows receive a time-decay weight:
sample_weight = 1.0 + recency_factor × (Year - 1998) / (2022 - 1998)
Where recency_factor is a hyperparameter tuned during cross-validation in the range [0.0, 1.0]. A value of 0.0 means equal weights (no recency adjustment). A value of 1.0 means 2022 matches receive 2× the weight of 1998 matches. Based on the 63.6% vs 69.4% accuracy gap, a recency factor in the range [0.3, 0.5] is expected to be optimal.

4. XGBoost Configuration
Architecture rationale
XGBoost is selected as the primary model for three specific reasons grounded in this dataset's properties. First, the 25-feature space includes several features with high missingness (12 of 48 teams have no modern WC history) — XGBoost handles missing values natively without requiring imputation, which is important because imputed values for WC-debutant features would introduce artificial precision. Second, the non-linear interactions between feature groups — particularly between Elo delta and stage order (the tournament temperament interaction), and between tournament form and historical WC win rate — require a model that can discover interactions without them being explicitly specified. Third, XGBoost's regularisation parameters (gamma, lambda, alpha) allow precise control over tree complexity in a small-sample setting, which is the primary risk with 448 training rows and 25 features.
Objective function
Group stage model: multi:softprob with num_class=3. This outputs a probability vector [P(WIN), P(DRAW), P(LOSS)] for each prediction, which is required for the Monte Carlo simulation. The eval_metric is mlogloss (multiclass log loss).
Knockout model: binary:logistic. Outputs P(WIN for team A) which is then adjusted using the fixed 21.4% draw rate at 90 minutes. The eval_metric is logloss.
Hyperparameter configuration
The following parameters are constrained by the small sample size (448 rows, 896 with perspective doubling). The values represent starting points; final values come from the six-fold chronological cross-validation grid search.
Parameter
Starting value
Search range
Constraint rationale
n_estimators
150
50–300
Early stopping on calibration holdout with patience=15 determines final value
max_depth
3
2–5
Small n per leaf at depth >4 causes overfitting; depth 3 is the maximum trustworthy value for n=448
learning_rate (eta)
0.05
0.01–0.1
Lower rate with higher n_estimators is preferred for small datasets
subsample
0.8
0.6–0.9
Row subsampling at each tree; reduces variance
colsample_bytree
0.7
0.5–0.9
Feature subsampling; with 25 features this means 17–18 features per tree
min_child_weight
4
2–8
Minimum sum of instance weight in a child; higher values prevent small-n leaf nodes
gamma
0.2
0.0–1.0
Minimum loss reduction required to make a split; conservative default
lambda (L2)
1.5
0.5–3.0
L2 regularisation on leaf weights; higher than default due to small n
alpha (L1)
0.0
0.0–0.5
L1 regularisation; keep at 0 unless feature sparsity is an issue
scale_pos_weight
1.0
—
Not used; class weights handled via sample_weight parameter instead
The scale_pos_weight mechanism is not used because the three-class group stage model requires per-class weights rather than a single positive class weight. Sample weights encoding class imbalance correction and recency decay are passed via the sample_weight parameter.
Feature importance extraction
XGBoost provides three importance metrics: weight (number of splits), gain (average improvement per split), and cover (average number of samples per split). For this project, gain-based importance is the primary metric because it measures the actual predictive contribution of each feature rather than just its usage frequency. Weight-based importance inflates the importance of features used as early splits in many trees regardless of their predictive contribution.
Feature importance is computed separately for the group stage model and the knockout model. Features that rank in the top 10 for both models confirm their importance as domain-agnostic signals. Features that rank highly in only one model reveal stage-specific dynamics (e.g., tournament form features are expected to rank higher in the knockout model because they capture form in the current tournament, which is only relevant after matches have been played).
Early stopping configuration
Training uses the 2022 calibration holdout as the early stopping evaluation set with early_stopping_rounds=15. This means training halts when the evaluation metric does not improve for 15 consecutive trees. The final n_estimators value used for production is the best iteration, not the last iteration. This prevents the model from memorising the training distribution while simultaneously using the calibration holdout for this purpose — which would contaminate the calibration step. To address this, the early stopping evaluation set is the 2018 WC fold (the most recent fold not reserved for calibration), and the 2022 data is reserved exclusively for probability calibration.

5. Logistic Regression Baseline
Purpose and architecture
The logistic regression baseline serves three distinct purposes: establishing a floor below which XGBoost is not considered to have added value, providing an interpretable model whose coefficients can be used to validate feature directions, and serving as a fallback model for any prediction scenario where XGBoost exhibits signs of overfitting on the cross-validation folds.
Feature preprocessing for Logistic Regression
Logistic Regression is sensitive to feature scaling in ways XGBoost is not. All continuous features must be standardised to zero mean and unit variance using StandardScaler fit exclusively on the training fold (never on the full dataset). Binary and categorical features are encoded as: binary features left as 0/1, categorical features (confederation, venue_region_cluster) encoded as one-hot with the most frequent class dropped to avoid collinearity.
Missing values are imputed before scaling. The imputation strategy uses the training fold mean for continuous WC history features, confederation mean for WC-debutant teams, and 0.5 for the shootout_win_rate_alltime of teams with no shootout history.
Configuration
Parameter
Value
Rationale
solver
lbfgs
Appropriate for multiclass with L2 penalty
multi_class
multinomial
Required for three-class group stage model
C (inverse regularisation)
Cross-validated in [0.01, 0.1, 1.0, 10.0]
Small training set favours stronger regularisation (lower C)
max_iter
1000
Ensure convergence
class_weight
balanced
Uses scikit-learn's automatic class weight balancing; equivalent to the manual weights for XGBoost
penalty
l2
Standard; l1 is experimental (see Section 9)
Interpreting the baseline coefficients
After fitting, the coefficient magnitudes provide a sanity check on feature directions. Expected coefficient signs from domain knowledge:
	•	elo_win_expectancy: strongly positive (higher WE → more likely to win)
	•	tourn_pts_md2: positive (more tournament points → more likely to win)
	•	wc_win_rate_knockout_modern: positive for knockout model, neutral for group stage
	•	form_win_rate_last10_competitive: positive
	•	shootout_win_rate_alltime: positive only in the shootout model
	•	elo_is_host: weakly positive (host advantage)
	•	confederation == OFC: negative (only New Zealand; OFC is the weakest confederation)
Any coefficient that contradicts these directional expectations is flagged for investigation before the model is used. A coefficient reversal (e.g., higher WC win rate predicting lower probability of winning) indicates a multicollinearity problem or a data construction error that must be resolved before proceeding.
Baseline comparison threshold
The logistic regression model is considered competitive with XGBoost if its RPS on the 2022 calibration holdout is within 5% of XGBoost's RPS. If XGBoost's RPS improvement over logistic regression exceeds 5%, XGBoost is the preferred model. If the improvement is below 5%, the ensemble weighting shifts toward the logistic regression model (see Section 7).

6. Probability Calibration
Why calibration is mandatory here
The Elo win expectancy (F002) demonstrated a mean absolute calibration error of 0.339 when compared against actual WC outcomes. This means that when the Elo formula predicts a 80% win probability, the actual win rate in historical WC data is closer to 78.6% — a moderate but real miscalibration at the extremes. XGBoost's raw probability outputs are typically more miscalibrated than logistic regression's, exhibiting S-curve compression (probabilities near 0 and 1 are pushed toward the centre). For a simulation that multiplies through 56 matches, even small per-match miscalibration compounds into large errors in win probability estimates for the tournament champion.
The calibration holdout is the 2022 WC (64 matches). All hyperparameter choices and model fitting use 1998–2018 data only. The 2022 holdout is touched exactly once — for calibration fitting.
Platt Scaling
Platt scaling fits a logistic regression on the model's raw score output (log-odds for logistic-based models, raw decision value for XGBoost) using the calibration holdout outcomes as the target. This is applied separately for each class in the multi-class group stage model.
The Platt scaling parameters (two parameters: a slope and intercept per class) are estimated on the 64 calibration matches. With only 64 holdout matches, the calibration estimate itself has uncertainty — the 95% confidence interval on each calibration parameter is wide. This is an inherent limitation of the dataset size and must be disclosed in the project output.
The calibrated probabilities must satisfy two constraints: they must sum to 1.0 across classes for each prediction, and they must be monotonically related to the raw scores (higher raw score still maps to higher calibrated probability after Platt scaling).
Calibration validation
After Platt scaling, a reliability diagram (calibration curve) is computed on the calibration holdout by binning predictions into 10 bins of width 0.1 and comparing mean predicted probability against actual win rate per bin. The calibration data verified in the analysis shows that the Elo win expectancy formula already produces a reasonably calibrated signal in the 0.4–0.8 range but is miscalibrated at the extremes (0.0–0.1 bin has 50% actual win rate; 0.9+ bin has only 62.5% actual win rate despite predicted WE of 92%). Platt scaling should correct these extreme-range miscalibrations.
The target calibration standard is that no bin's mean predicted probability deviates from the actual win rate in that bin by more than 10 percentage points on the calibration holdout. If this standard is not met after Platt scaling, isotonic regression calibration is applied as an alternative.
Temperature scaling as a fallback
If Platt scaling produces non-monotonic results (which can occur with small calibration sets), temperature scaling is used as a simpler alternative. Temperature scaling divides the raw model logits by a single temperature parameter T, fitted to minimise the negative log likelihood on the calibration holdout. Temperature scaling preserves the ranking of predictions while adjusting confidence — T > 1 produces softer probabilities, T < 1 produces harder probabilities. For tournament simulation purposes, T > 1 is almost always preferable because it prevents the simulation from producing near-certain outcomes for matches between evenly matched teams.

7. Bayesian Tournament Update
The update problem
At the freeze date (June 23, 2026), 48 matches have been played. The question is how much to adjust the pre-tournament probability estimates — which are based entirely on historical ratings and form — in response to what teams have actually demonstrated in the 2026 tournament.
The analytical foundation for this decision comes from the shrinkage analysis: the optimal alpha (weight on in-tournament evidence vs prior Elo) varied from 0.0 (2022) to 0.3 (2002) across the six validation years, with a mean optimal alpha of 0.17. The 2022 result — where the optimal update weight was 0.0 — reflects that Qatar 2022 had several notable upsets (Morocco's run, Japan beating Germany and Spain) that were not predicted by group stage form. Blindly trusting group stage performance would have overestimated Morocco and Japan's chances of winning the tournament.
This validates a conservative approach: tournament form is a real signal but its weight must be capped to prevent the model from over-updating on two matches of data.
The update formula
For each team, the posterior win probability for any future match is:
P_posterior(team wins) = (1 - α) × P_prior(team wins) + α × P_likelihood(team wins)
Where:
	•	P_prior(team wins) is the calibrated XGBoost output using Feature Groups 1–4 only (no tournament features)
	•	P_likelihood(team wins) is computed from the in-tournament performance features (Feature Groups 5) using the tournament adjustment model described below
	•	α is the shrinkage parameter, set to 0.17 (empirically derived mean optimal value)
The tournament adjustment model
The likelihood component is not simply the win probability from a model trained on in-tournament features alone — with only two matches per team, such a model would have enormous variance. Instead, a separate linear model is fitted (one per WC validation year in the training data) to predict: given a team's group stage GD, points, goals scored, shots on target, and possession average, how does their knockout performance compare to what their Elo rating would predict?
This adjustment model captures the signal in group stage performance above and beyond the pre-tournament prior. Specifically it asks: among teams with equivalent Elo ratings entering the tournament, do teams that scored more goals in the group stage outperform teams that scored fewer? The answer from six validation years is yes — group stage GD correlates with knockout performance at 64–77% accuracy across the six validation years, substantially above the 50% random baseline.
The adjustment model outputs a multiplier rather than an absolute probability:
P_likelihood = P_prior × knockout_form_multiplier
Where knockout_form_multiplier > 1.0 for teams whose group stage performance exceeded their Elo expectation, and < 1.0 for teams that underperformed. The multiplier is bounded in [0.5, 2.0] to prevent the update from producing extreme probabilities based on two matches of data.
Applying the update to 2026 teams
The verified 2026 tournament standings produce the following qualitative update directions (exact multipliers depend on the fitted model):
Positive update candidates (outperforming Elo expectation in group stage): Germany (Elo rank 11, but 9 goals in 2 matches, GD +7), USA (Elo rank 41, but 6 points, GD +5), Norway (Elo rank 12, 6 points with 7 goals scored), Colombia (Elo rank 7, 6 points with clean sheet defence). These teams showed results substantially better than their pre-tournament Elo would predict.
Negative update candidates (underperforming Elo expectation): Spain (Elo rank 1, but only 4 points, GD +4 — expected more), Brazil (Elo rank 5, only 4 points with 1–1 draw against Morocco), Türkiye (Elo rank 14, 0 points despite 75.5% average possession), Belgium (Elo rank 19, only 2 points after two draws). These teams show results below their pre-tournament rating, though two matches is insufficient to conclude the rating is wrong.
Neutral (performing consistent with Elo expectation): France, Argentina, England, Netherlands, Portugal — all performing close to their pre-tournament rating predictions.
The four-component feature vector for the update
The likelihood model uses only features that are available for all 48 teams (no tactical-only features that are missing for Portugal, Colombia, England, Croatia). The four components are:
Component 1 — Points relative to Elo expectation: tourn_pts_md2 - expected_pts_md2, where expected_pts is computed from the Elo win expectancy against both opponents. Germany with 6 points against Curaçao (Elo rank 90) is not the same signal as France with 6 points against Senegal (Elo rank 17) and Norway (Elo rank 12).
Component 2 — Goal difference relative to Elo expectation: tourn_gd_md2 - expected_gd_md2, where expected_gd uses historical WC average GD for teams at the given Elo differential.
Component 3 — Goals scored per match: tourn_gf_md2 / 2 normalised by average tournament goals scored (approximately 2.6 goals per match in the 2026 group stage so far).
Component 4 — Goals conceded per match: tourn_ga_md2 / 2 normalised similarly. Inverted signal — lower is better.
These four components are combined with equal weights (0.25 each) in the absence of fitted weights. The fitted weights from the historical tournament adjustment model (trained on 1998–2018 data) are used when available, which is the preferred approach.

8. Stage-Conditional Knockout Models
Why separate models by stage
The analytical evidence validates the stage-conditional approach: the WC group vs knockout uplift feature (F022) showed meaningful variation across teams, confirming that different teams genuinely perform differently across stages. Beyond this, the competitive dynamics shift substantially at each stage. The Round of 32 features the widest gap in team quality (the bracket is designed to produce high-quality matches only from the quarterfinals onward). By the Final, both teams are among the world's best — Elo rating differences are small and recent tournament performance carries more weight.
The sample size constraint prevents fitting a fully separate model per stage (112 knockout matches over 7 tournaments, further divided into four sub-stages, produces fewer than 30 matches per stage per tournament year). Instead, a two-stage architecture is used.
Stage Group A: Round of 32 and Round of 16
Training data: All knockout matches from Rounds of 16 and R32-equivalent stages (1998–2022), 64 matches (approximately 9–10 per tournament year)
Feature weighting: Pre-tournament features (Groups 1–4) receive higher weight in this stage group. The Elo rating delta and WC win rate features are the dominant predictors. In-tournament features have modest weight — teams are still early enough in the tournament that two group stage matches are insufficient to fully revise the prior.
Model configuration: Same XGBoost configuration as the main model but with max_depth = 2 to prevent overfitting on the smaller per-stage sample. The logistic regression baseline is typically preferred at this stage due to its lower variance with small n.
Stage Group B: Quarterfinals, Semifinals, Final
Training data: All QF, SF, and Final matches from 1998–2022: 7 × (4 + 2 + 1) = 49 matches
Feature weighting: This is the smallest training corpus in the project (49 matches). The WC knockout-specific win rate (F015) and the WC group vs knockout uplift feature (F022) are the most important features at this stage. In-tournament form features receive higher weight relative to pre-tournament features because teams have now played 4–6 matches and the sample is sufficient to update the prior more aggressively.
Model configuration: Logistic regression is preferred over XGBoost at this stage due to the 49-row training set. C = 0.1 (strong regularisation). Features are limited to the top 12 features from the global feature importance ranking to prevent overfitting. The XGBoost variant is trained in parallel and its predictions are averaged with the logistic regression output at equal weight.
Feature interaction: stage_order × wc_group_vs_knockout_uplift
This is the single most important cross-feature interaction in the knockout models. A team with a high wc_group_vs_knockout_uplift value (historically performs better in knockouts than group stage) should see that signal amplified as the stage order increases. In the Round of 32, this interaction adds marginal signal. In the Final, a team like Argentina (historically strong under knockout pressure) versus France (historically strong across both stages) makes this interaction decisive.
The interaction is implemented explicitly by creating a product feature stage_order × wc_group_vs_knockout_uplift during feature construction for the knockout models. This product feature replaces the two individual features in the knockout model input to reduce dimensionality.
Home/away designation in knockout matches
As noted in the target variable design, there is no meaningful home team in WC knockout matches. The team assigned as home_team by DS16 reflects a bracket convention, not a venue assignment. To prevent the model from learning a spurious home team effect, all knockout match training rows swap the perspective of both rows to produce balanced representation. For every knockout training match, one row represents the outcome from Team A's perspective and the other from Team B's — ensuring that no team in either position appears systematically more likely to win based on their bracket assignment.
The is_host feature (F007) correctly captures the host advantage independent of the home/away designation, and should remain in the knockout models.

9. Penalty Shootout Model
Scope and activation condition
The penalty shootout model activates exclusively in the Monte Carlo simulation when a knockout match simulation draw is generated at 90 minutes. The empirically verified activation rate is 21.4% of knockout matches at 90 minutes (24/112 modern WC knockout matches). The simulation does not explicitly extend matches through extra time — the draw-at-90-minutes rate implicitly includes matches that were drawn at 90 minutes and then went to extra time before reaching penalties or being decided in extra time. Of the 24 drawn knockout matches in the modern era, approximately 15 went to penalties (the remainder were decided in extra time), giving a penalty shootout occurrence rate of approximately 13.4% of all knockout matches.
In the simulation, when a draw is generated, the penalty model is applied directly to produce a shootout winner without separately simulating extra time. This simplification is appropriate given the small sample sizes.
Training data
Primary source: DS6 (678 total shootouts, 542 post-1990). Of these, 119 have Elo ratings available from DS2 for both teams. The full 542 post-1990 shootouts are used for win rate estimation. The 119 with Elo data are used for the Elo-adjusted model.
WC-specific source: DS8 (35 WC shootout records via home_penalty and away_penalty columns). These 35 records are the most directly relevant but too small for a standalone model — they are used as a holdout for evaluating the DS6-trained model.
Model architecture
The shootout model has three components combined in a weighted ensemble.
Component 1 — Historical shootout win rate (F040): team_shootout_wins / team_shootout_appearances from DS6, with Bayesian shrinkage toward 0.50 for teams with few appearances.
The shrinkage formula is:
shrunk_rate = (wins + k × 0.5) / (appearances + k)
Where k is the shrinkage strength, tuned on WC holdout data. Starting value k = 8 means a team needs approximately 8 shootout appearances before their historical rate fully departs from the 0.50 prior. This is appropriate given that Germany's 0.75 rate (6 wins from 8 appearances) has a 95% CI of [0.45, 1.00] — wide enough that aggressive shrinkage is warranted.
The two teams with no shootout history (Norway and Czechia) receive 0.50 with a shootout_naive_flag = 1 that allows the Elo component to dominate for these teams.
Component 2 — Elo win expectancy at shootout time (F002 contextualised): The Elo win expectancy formula applied to the pre-tournament ratings. The analysis showed that Elo predicts shootout winner at 58.0% accuracy (69/119 with Elo data available), well above the 50% baseline. This confirms Elo carries genuine signal in shootout outcomes — stronger teams still tend to win even under maximum pressure.
Component 3 — WC-specific shootout rate (F042): The team's WC-only penalty shootout win rate from DS8. Used only when a team has two or more WC shootout appearances. With only 35 WC shootouts across all teams, most teams have 0–2 WC shootout appearances, limiting this component's reliability.
Ensemble weights
The final penalty model output combines the three components:
P(Team A wins shootout) = w1 × shrunk_rate_A / (shrunk_rate_A + shrunk_rate_B)
                        + w2 × elo_win_expectancy_A
                        + w3 × wc_shootout_rate_A (if applicable)
The weights are:
	•	w1 = 0.50 (historical shootout rate is the dominant signal)
	•	w2 = 0.40 (Elo provides genuine independent signal at 58% accuracy)
	•	w3 = 0.10 (WC-specific rate — small sample, used as minor adjustment only)
	•	w3 = 0.00, w1 = 0.55, w2 = 0.45 when neither team has two or more WC appearances
The final probability is normalised to ensure P(A wins) + P(B wins) = 1.0.
Verified shootout win rates for key contenders
The exact shrinkage-adjusted rates at k=8 for the eight strongest teams by Elo rating, using the pre-shrinkage rates verified from DS6:
Team
Raw rate
Appearances
Shrunk rate (k=8)
Expected finalist behaviour
Germany
0.750
8
0.625
Strong shootout team
Argentina
0.652
23
0.639
Strong, well-established
France
0.455
11
0.461
Slight disadvantage
England
0.333
12
0.375
Significant disadvantage
Brazil
0.562
16
0.551
Slightly above neutral
Portugal
0.625
8
0.563
Above neutral
Netherlands
0.200
10
0.322
Significant disadvantage
Colombia
0.583
12
0.563
Above neutral
The England and Netherlands disadvantage in shootouts is one of the most well-known patterns in tournament football. Both teams' adjusted rates are significantly below 0.50, meaning that a draw at 90 minutes is genuinely bad news for these teams. This will be reflected in the Monte Carlo simulation — any simulation run that produces England vs Netherlands in a knockout match will show meaningfully different shootout winner distributions than the match-outcome model would suggest.

10. Monte Carlo Simulation Integration
Simulation architecture overview
The Monte Carlo simulation ties together every model component into a single stochastic engine that samples from probability distributions to produce tournament path distributions. It runs 10,000 independent simulations. Each simulation is a complete tournament from the end of Matchday 2 through the Final.
The statistical justification for 10,000 simulations: at 10,000 samples, the standard error on a 10% win probability estimate is 0.003 (0.3%), well within the project's target precision of 0.5%. For probabilities near 5%, the SE is 0.0022. No team is expected to have a tournament win probability so small (below 0.5%) that 10,000 samples is insufficient to detect it above noise.
Simulation starting state
Each simulation begins from the verified frozen state: 12 groups with exactly two matchdays completed, all 48 teams have known points, goal difference, and goals scored as of June 23, 2026.
The starting state for each simulation is identical — there is no uncertainty about the group stage results to date. Uncertainty enters at Matchday 3.
Phase 1: Matchday 3 simulation (24 matches)
Each of the 24 Matchday 3 matches is simulated independently by drawing a single outcome from the three-class group stage model probability distribution for that matchup.
The probability vector for each match is computed by:
Step 1 — retrieve the base prediction from the calibrated group stage XGBoost model using Feature Groups 1–4 (pre-tournament only) for both teams.
Step 2 — apply the Bayesian update using α = 0.17 and the tournament form multiplier from each team's post-MD2 statistics.
Step 3 — apply group context adjustments. Teams that are already_qualified = 1 receive a modest reduction in win probability (reflecting historical rotation tendencies — teams that are through tend to field weaker lineups in their final group game). The adjustment is −0.03 on win probability. Teams that are already_eliminated = 1 receive a symmetric modest reduction in draw probability (teams with nothing to play for tend to either accept defeat or play more aggressively for a consolation win; the draw rate decreases). These adjustments are conservative and grounded in the 64–67% accuracy of group-form predictions from the historical analysis.
Step 4 — draw a random outcome from the adjusted probability distribution.
Critical concurrent match rule: Matchday 3 pairs are played simultaneously within each group. The simulation draws both match outcomes for each group simultaneously rather than sequentially. This matters because group qualification depends on both results simultaneously — drawing one match first and then the second would create an incorrect dependence structure where the second match's group context adjustments use the already-simulated first result.
Phase 2: Group stage resolution
After all 24 Matchday 3 matches are simulated, each group's final standings are computed deterministically by applying the standard 2026 WC tiebreaker sequence: points, head-to-head points, head-to-head goal difference, head-to-head goals scored, overall goal difference, overall goals scored, FIFA rank as the simulation's final differentiator (in place of drawing of lots).
The eight best third-placed teams are then selected by ranking all 12 third-placed teams on the same tiebreaker sequence. This step produces the final 32 qualified teams for the knockout stage.
Phase 3: Round of 32 bracket construction
The DS16 match_label column encodes the exact seeding rules for all 16 Round of 32 matches. Examples: "1E vs 3ABCDF" means the Group E winner plays the best third-placed team from Groups A, B, C, D, or F. These rules are applied deterministically — the bracket is fixed by FIFA rules, not by the simulation.
For each simulation run, the 32 qualified teams are mapped into the 16 Round of 32 matches. Teams qualifying as third-placed are assigned to their specific bracket slot based on which group they finished third in, following the FIFA 2026 placement table (which is derivable from DS16 match labels and DS17 group assignments).
Phase 4: Knockout stage simulation (32 matches)
Each knockout match follows a three-step resolution process.
Step 1 — Draw the 90-minute result. Apply the stage-conditional knockout model (Stage Group A for R32/R16, Stage Group B for QF/SF/Final) to obtain P(Win), P(Draw at 90min), P(Loss). Draw a random outcome.
If Win: this team advances. If Loss: the opponent advances. If Draw (21.4% base rate):
Step 2 — Apply the extra time/penalty split. Approximately 37.5% of matches drawn at 90 minutes are decided in extra time before reaching penalties (15/40 historical total, estimated). For these cases, the match winner is determined by the same knockout model applied with a slightly higher win probability for the favoured team (extra time tends to favour the stronger team as the underdog's energy advantage from a defensive strategy fades). For the remaining 62.5% of drawn matches that reach penalties:
Step 3 — Apply the penalty shootout model. Compute P(Team A wins shootout) using the three-component ensemble. Draw the shootout outcome.
Stage escalation between rounds: After each round, the feature vector for each advancing team is updated. tourn_pts_md2 becomes tourn_pts_total (including knockout wins). tourn_gf and tourn_ga accumulate. The Bayesian update is re-applied with the same α = 0.17 but using the growing in-tournament dataset. By the time a team reaches the Final (having played 6 or 7 matches), the in-tournament evidence is substantial enough that α could justifiably be increased — but for consistency and conservatism, α is held constant throughout the simulation.
Phase 5: Recording outcomes
For each of the 10,000 simulation runs, the following are recorded:
	•	Tournament winner (1 of 48 teams)
	•	Runner-up
	•	Third-place finisher
	•	Fourth-place finisher
	•	Which round each team was eliminated in
	•	Which specific matches went to extra time or penalties
	•	Whether any specific upset (lower-Elo team eliminating higher-Elo team) occurred
Phase 6: Aggregating results
After 10,000 runs, the following outputs are computed:
Tournament win probability per team: wins / 10,000 for each team. Standard error = sqrt(p × (1-p) / 10,000). A 90% confidence interval is computed as p ± 1.645 × SE.
Average round reached per team: The mean knockout round reached across all 10,000 simulations. This is a more granular signal than win probability — a team with a 12% win probability that reaches the final in 30% of simulations tells a different story than one that reaches the final in 15% and wins 12% of those.
Most common final matchup: The pair of teams that appears in the Final most frequently across 10,000 simulations. The probability of each specific final matchup is also computed.
Bracket path distribution for top 6 teams: For Spain, Argentina, France, England, Brazil, and Germany, the distribution of knockout paths (which opponent they face in each round) is computed. Some paths favour these teams more than others due to bracket structure.
Upset frequency analysis: In what proportion of simulations does at least one team ranked outside the top 8 reach the semifinal? At least one team outside the top 16 reach the quarterfinal? This provides a summary uncertainty measure for the tournament as a whole.
Seed and reproducibility
Every simulation run is seeded with its run index (run 1 uses seed 1, run 10,000 uses seed 10,000). This guarantees that the simulation is fully reproducible — given the same frozen feature table and the same trained models, running the simulation again produces identical output to four decimal places. The seed array is stored in the outputs directory alongside the win probability table, ensuring that any individual simulation run can be re-executed in isolation for debugging.
Confidence interval reporting convention
All win probabilities in the final output are reported as: p% [lower%, upper%] where the interval is the 90% confidence interval from simulation variance. The notation 12.4% [11.1%, 13.7%] means the team won in 12.4% of simulations with a 90% CI of 11.1% to 13.7%.
Teams with win probabilities below 1.0% are reported as < 1.0% without a confidence interval. Their path to winning the tournament is theoretically possible but requires multiple consecutive upsets that are individually unlikely and jointly implausible. Reporting a precise confidence interval for these teams would suggest a precision that the model does not actually achieve.
 
 
 
 
 
 
Complete Implementation Blueprint
Frozen: June 23, 2026 — Post-Matchday 2

Repository Structure
fifa-wc2026-forecast/
├── FREEZE_MANIFEST.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── arc3/matches.csv
│   │   ├── arc2_new/matches_1930_2022.csv
│   │   ├── arc2_new/schedule_2026.csv
│   │   ├── arc2_new/fifa_ranking_2026-06-08.csv
│   │   ├── arc2_new/fifa_ranking_2022-10-06.csv
│   │   ├── arc4/elo_ratings_wc2026.csv
│   │   ├── arc6/results.csv
│   │   ├── arc6/goalscorers.csv
│   │   ├── arc6/shootouts.csv
│   │   ├── arc_base/matches.csv
│   │   ├── arc_base/teams.csv
│   │   └── arc_base/host_cities.csv
│   └── freeze/
│       └── june23_results.csv
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_tournament_update.ipynb
│   ├── 05_monte_carlo.ipynb
│   └── 06_results_and_visualisation.ipynb
├── src/
│   ├── name_map.py
│   ├── standings.py
│   ├── features.py
│   ├── models.py
│   ├── simulation.py
│   └── leakage_guard.py
├── tests/
│   ├── test_name_map.py
│   ├── test_standings.py
│   ├── test_features.py
│   ├── test_leakage.py
│   ├── test_models.py
│   └── test_simulation.py
└── outputs/
    ├── group_standings_freeze.csv
    ├── team_features_freeze.parquet
    ├── win_probabilities.csv
    ├── bracket_simulation_summary.csv
    └── charts/

data/freeze/june23_results.csv
Purpose: The four injected June 23 results that complete Matchday 2. These are not in any source dataset and are the only hand-entered data in the project. They are stored separately from raw data to make the injection explicit and auditable.
Schema:
date          str   — always "2026-06-23"
home_team     str   — DS9 canonical name
away_team     str   — DS9 canonical name
home_score    int
away_score    int
gameweek      int   — always 2
source        str   — "manual_injection — official FIFA result"
Exact content:
date,home_team,away_team,home_score,away_score,gameweek,source
2026-06-23,Portugal,Uzbekistan,5,0,2,manual_injection
2026-06-23,Colombia,Congo DR,1,0,2,manual_injection
2026-06-23,England,Ghana,0,0,2,manual_injection
2026-06-23,Croatia,Panama,1,0,2,manual_injection
Dependencies: None. This file is the source of truth for MD2 completion of Groups K and L. No other file may contradict these values.

FREEZE_MANIFEST.md
Purpose: Human-readable declaration of the project's frozen state. Every developer and reviewer must read this file before touching any data or model. It serves as the single authoritative record of what data the project used and when.
Contents:
Section 1 — Freeze declaration: project name, freeze timestamp (June 23, 2026 23:59 UTC), freeze reason (end of Matchday 2), and a statement that no data created after this timestamp enters any model, feature, or simulation.
Section 2 — File registry with SHA-256 checksums:
arc3/matches.csv              cd7665339dea2dd9...  44 rows   42 cols
arc2_new/matches_1930_2022.csv 60229eccd1652be3...  964 rows  44 cols
arc2_new/schedule_2026.csv     e7352c47ec6f8c8a...  72 rows   10 cols
arc2_new/fifa_ranking_2026-06-08.csv 2747fa1828532cb6... 211 rows 8 cols
arc2_new/fifa_ranking_2022-10-06.csv 9850c99ff3f6000d... 211 rows 7 cols
arc4/elo_ratings_wc2026.csv    776050e584c4ec0a...  4683 rows 23 cols
arc6/results.csv               64d75097f252941f...  49477 rows 9 cols
arc6/goalscorers.csv           48f5eac15e06d45b...  47690 rows 8 cols
arc6/shootouts.csv             e52e503badc11021...  678 rows  5 cols
arc_base/matches.csv           5c17130f9775bf34...  104 rows  8 cols
arc_base/teams.csv             2097e7f81bf00ac7...  48 rows   5 cols
arc_base/host_cities.csv       6a75b48af414d47a...  16 rows   6 cols
data/freeze/june23_results.csv [computed at init]   4 rows    7 cols
Section 3 — Excluded datasets: DS3 (player_performance.csv) — synthetic, wrong teams, future outcomes. DS13/DS14 (train/test) — unknown provenance. DS20 (worldcup2026.db) — duplicate of arc_base CSVs.
Section 4 — Manual injections: the four June 23 results with source citation.
Section 5 — Name resolution authority: DS9 (schedule_2026.csv) defines canonical team spellings. All eight name variants documented with their per-dataset mapping.

src/name_map.py
Purpose: Single source of truth for all team name translations across datasets. Every other module imports from here. No module may perform its own name normalisation.
Inputs: None. This is a pure constants module with no file I/O.
Outputs: Three dictionaries and two functions exported at module level.
Functions:
TO_ELO: dict[str, str] Maps DS1/DS9 canonical 2026 name → DS2 Elo country name. All 48 teams present. Identity mapping for 41 teams; explicit overrides for 7:
Korea Republic → South Korea
Bosnia & Herz. → Bosnia and Herzegovina
Türkiye        → Turkey
Côte d'Ivoire  → Ivory Coast
IR Iran        → Iran
Cabo Verde     → Cape Verde
Congo DR       → DR Congo
United States and USA: DS1 uses "United States", DS2 uses "United States" — identical, no override needed. DS10 uses "USA" which requires a separate map.
TO_FIFA: dict[str, str] Maps DS1/DS9 canonical → DS10 FIFA ranking team name. Overrides for 2 teams:
United States → USA
Korea Republic → Korea Republic  (DS10 matches DS1 here)
TO_DS4: dict[str, str] Maps DS1/DS9 canonical → DS4/DS8 historical results name. Overrides for 5 teams that appear differently in DS4/DS8:
Korea Republic → South Korea
Bosnia & Herz. → Bosnia and Herzegovina
Türkiye        → Turkey
Côte d'Ivoire  → Ivory Coast
IR Iran        → Iran
Cabo Verde     → Cape Verde
Congo DR       → DR Congo
normalise(name: str, target: str) -> str Single entry point for name translation. target is one of "elo", "fifa", "ds4". Returns the translated name. Raises ValueError if name not found and target is known. Used by all feature engineering functions.
get_all_48() -> list[str] Returns the authoritative list of 42 real qualified teams (excluding 6 placeholders) in DS1/DS9 canonical format. Derived from DS17 with placeholders filtered out and names reconciled to DS1 spelling. This is the definitive roster for iteration.
Dependencies: None — pure Python constants.
Test coverage: tests/test_name_map.py verifies that all 48 teams round-trip correctly through all three maps and that no unknown name raises a silent identity return.

src/standings.py
Purpose: Compute group standings from a set of match results. Used in 01_data_audit.ipynb for verification, in 05_monte_carlo.ipynb for every simulation iteration's group stage resolution, and in tests/test_standings.py for correctness checking.
Inputs:
	•	results: pd.DataFrame with columns home_team, away_team, home_score, away_score (all DS1/DS9 canonical names)
	•	group_map: dict[str, list[str]] mapping group letter to list of four team names
Outputs:
	•	standings: dict[str, pd.DataFrame] — key is group letter, value is a DataFrame sorted by (Pts, GD, GF) with columns team, P, W, D, L, GF, GA, GD, Pts
	•	third_place_ranking: pd.DataFrame — all 12 third-place teams sorted by the full 2026 WC tiebreaker sequence
Functions:
compute_group_standings(results: pd.DataFrame, teams: list[str]) -> pd.DataFrame Computes standing for a single group from a results DataFrame filtered to that group's teams. Returns a 4-row DataFrame sorted by (Pts desc, GD desc, GF desc). Does not apply head-to-head tiebreakers at this stage — head-to-head is handled in the simulation's resolution layer where needed.
compute_all_standings(results: pd.DataFrame, group_map: dict[str, list[str]]) -> dict[str, pd.DataFrame] Calls compute_group_standings for all 12 groups. Returns the standings dict.
rank_third_place_teams(standings: dict[str, pd.DataFrame]) -> pd.DataFrame Extracts the third-place team from each group standing and ranks all 12 by the 2026 WC tiebreaker sequence: Pts, GD, GF, GA (inverted), then group letter as final tiebreaker. Returns a DataFrame with a group column added. The top 8 rows are the qualified third-place teams.
apply_head_to_head_tiebreaker(tied_teams: list[str], results: pd.DataFrame) -> list[str] For two or three teams tied on Pts, GD, and GF in a group, applies the head-to-head tiebreaker (points among tied teams, then GD among tied teams, then GF among tied teams). Returns a re-ranked list of tied teams. Used only in the simulation's group resolution step — not in the data audit where actual results determine standing unambiguously.
get_qualified_teams(standings: dict[str, pd.DataFrame], third_place_ranking: pd.DataFrame) -> dict[str, str] Returns a dictionary mapping bracket position to team name. Bracket positions are the DS16 match_label strings such as "1A", "2B", "3ABCDF". The third-place seeding follows the FIFA 2026 placement table defined in the bracket positions.
Critical dependency: The GROUP_MAP constant is defined in this module using DS1/DS9 canonical team names exactly as they appear in the 48-match frozen result set. The map is hardcoded (not loaded from file) because it is part of the project's frozen state:
GROUP_MAP = {
    'A': ['Mexico', 'South Africa', 'Korea Republic', 'Czechia'],
    'B': ['Canada', 'Bosnia & Herz.', 'Qatar', 'Switzerland'],
    ...
}
Dependencies: pandas, src/name_map.py

src/features.py
Purpose: Construct the complete 55-column feature table for all teams. This is the largest and most complex source file. Every feature documented in the feature engineering specification is built here. No model training occurs in this file.
Inputs:
	•	ds1: pd.DataFrame — arc3/matches.csv (44 rows)
	•	ds1_ext: pd.DataFrame — freeze/june23_results.csv (4 rows)
	•	ds2: pd.DataFrame — arc4/elo_ratings_wc2026.csv (4,683 rows)
	•	ds4: pd.DataFrame — arc6/results.csv (49,477 rows)
	•	ds6: pd.DataFrame — arc6/shootouts.csv (678 rows)
	•	ds8: pd.DataFrame — arc2_new/matches_1930_2022.csv (964 rows)
	•	ds10: pd.DataFrame — arc2_new/fifa_ranking_2026-06-08.csv (211 rows)
	•	ds11: pd.DataFrame — arc2_new/fifa_ranking_2022-10-06.csv (211 rows)
	•	ds16: pd.DataFrame — arc_base/matches.csv (104 rows)
	•	ds17: pd.DataFrame — arc_base/teams.csv (48 rows)
	•	ds18: pd.DataFrame — arc_base/host_cities.csv (16 rows)
Outputs:
	•	team_features: pd.DataFrame — one row per team (48 rows), 55 columns as specified in the feature table schema. Index is DS1/DS9 canonical team name.
	•	match_features: pd.DataFrame — one row per team per match for historical training (896 rows = 448 WC matches × 2 perspectives), 55 columns plus outcome target variable. Used for model training only.
	•	prediction_features: pd.DataFrame — one row per team per future match (112 rows = 56 matches × 2 perspectives), 55 columns, no outcome column.
Functions:
build_elo_features(ds2: pd.DataFrame, team: str) -> dict Extracts F001–F007 for a single team. Filters DS2 to snapshot_date == '2026-05-27' and country == normalise(team, 'elo'). Returns a dict with keys elo_rating, elo_rank, elo_rating_career_peak, elo_rating_career_avg, elo_is_host. Raises KeyError if team not found in snapshot. The elo_win_expectancy and elo_rating_delta are matchup-level features computed in build_matchup_features, not here.
build_elo_features_historical(ds2: pd.DataFrame, team: str, year: int) -> dict Same as build_elo_features but filters to snapshot_date == f'{year-1}-12-31'. Used for building the historical training rows. Returns empty dict if snapshot not available — these rows are excluded from training, not imputed.
build_fifa_features(ds10: pd.DataFrame, ds11: pd.DataFrame, team: str) -> dict Extracts F008–F013 for a single team. Joins DS10 and DS11 on normalised team name. Computes fifa_points_4yr_change and fifa_rank_4yr_change as DS10 value minus DS11 value. Returns None for 4yr change features if team not present in DS11. elo_fifa_rank_disagreement is computed here using the DS2 Elo rank and DS10 FIFA rank. Note: DS10 rank uses DS10 world rank for all 211 teams; the disagreement metric compares rank within the 48-team WC field, not global rank — a mapping from global rank to WC-field rank is applied here.
build_wc_historical_features(ds8: pd.DataFrame, team: str, min_year: int = 1998) -> dict Extracts F014–F022 for a single team. Filters DS8 to Year >= min_year and rows where team appears as home or away. The wc_debut_modern_flag is set to True if team has zero appearances. Computes all eight historical metrics plus the flag. For debutant teams, numeric features are set to None (not 0) and the flag is set to True. The wc_best_result_encoded requires a lookup of tournament-level results which is computed via a separate helper function _encode_best_result(ds8, team).
_encode_best_result(ds8: pd.DataFrame, team: str) -> int Private helper. Finds the furthest round a team reached in any WC appearance. Maps DS8 Round values to the 0–7 encoding. Round value to encoding: Group stage / First round = 1, Second round / Round of 16 = 2, Quarter-finals = 3, Semi-finals = 4, Third-place match / Final stage = 5, Runner-up = 6, Winner (determined by checking if team won the Final row) = 7. Returns 0 for teams with no WC history.
build_recent_form_features(ds4: pd.DataFrame, team: str, cutoff_date: str = '2026-06-10') -> dict Extracts F023–F028 for a single team. Filters DS4 to competitive matches (tournament != 'Friendly') between 2024-01-01 and cutoff_date with non-null scores. Sorts by date descending and takes the last 10. Computes win rate, avg GF, avg GA, GD, clean sheet rate, and unbeaten streak. If fewer than 5 competitive matches exist for a team in the window, returns None for all features and raises a warning — not an error, as small confederations may have limited competitive schedules.
build_tournament_features(ds1: pd.DataFrame, ds1_ext: pd.DataFrame, team: str) -> dict Extracts F029–F039 for a single team. Combines DS1 (44 matches, all columns) and DS1_ext (4 matches, scores only) to form the 48-match frozen result set. Identifies which matches the team appeared in. Computes tournament GF, GA, GD, Pts from all 48 matches. Computes tactical features (possession, SOT, etc.) from DS1 only — not DS1_ext. Sets has_full_tactical_md2 to True if the team has two DS1 rows (not DS1_ext) and False if one of their two matches is from DS1_ext. Sets tourn_formation_changed by comparing the team's home_formation or away_formation across their two DS1 appearances — None for the four teams with only one DS1 appearance.
build_shootout_features(ds6: pd.DataFrame, ds8: pd.DataFrame, team: str, k: float = 8.0) -> dict Extracts F040–F043 for a single team. Computes raw win rate from DS6 (all years, all tournaments). Applies Bayesian shrinkage: shrunk_rate = (wins + k * 0.5) / (appearances + k). Computes shootout_appearances_total as raw count. Computes shootout_win_rate_wc_only from DS8 penalty columns — None if fewer than 2 WC shootout appearances. Sets shootout_naive_flag to True for Norway and Czechia (verified zero shootout history in DS6). The two teams with zero DS6 history receive shootout_win_rate_alltime = 0.5 and shootout_naive_flag = True.
build_match_context_features(ds16: pd.DataFrame, ds17: pd.DataFrame, ds18: pd.DataFrame, match_id: int) -> dict Extracts F044–F050 for a specific match (not team). Looks up the DS16 row for match_id, joins to DS18 via city_id, retrieves region_cluster, country, venue_name. Computes kickoff_local_hour by parsing kickoff_at (timezone-aware string) and converting to local venue timezone using airport_code as a timezone proxy. Retrieves stage_id from DS16, looks up stage_order from the hardcoded stage order mapping. Sets is_knockout to True if stage_id >= 2.
build_matchup_features(team_a_features: dict, team_b_features: dict) -> dict Computes F002 (elo_win_expectancy), F003 (elo_rating_delta), and all other delta features that require both teams' values simultaneously. Returns a dict of matchup-level features. Called for every row in the training and prediction feature tables.
build_team_feature_table(all_inputs: dict) -> pd.DataFrame Master function. Calls all individual feature builders for all 48 teams. Returns the team_features DataFrame (48 rows × 55 columns). This is the pre-computed team-level feature store that all match-level feature tables are built from.
build_training_rows(ds8: pd.DataFrame, ds2: pd.DataFrame, ds4: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame Builds the 896-row training dataset. For each of the 448 WC matches (1998–2022), creates two rows — one from home team perspective, one from away team perspective. For each row, fetches historical features at the correct time point using build_elo_features_historical and build_wc_historical_features filtered to years before the match year. Assigns target variable outcome as 2 (win), 1 (draw), 0 (loss) from the perspective of the row's team. Adds sample_weight column using the recency formula 1.0 + 0.4 * (year - 1998) / 24. Returns DataFrame with 896 rows, 55 features, outcome target, and sample_weight.
build_prediction_rows(team_features: pd.DataFrame, future_matches: pd.DataFrame, ds16: pd.DataFrame, ds17: pd.DataFrame, ds18: pd.DataFrame) -> pd.DataFrame Builds the 112-row prediction dataset for all 56 future matches (24 MD3 group stage + 32 knockout). For MD3 matches, team identities are known from DS9. For knockout matches, team identity is None (bracket unresolved) — these rows have team feature columns left as None and are populated during simulation. Returns DataFrame with 112 rows, 55 feature columns, no outcome column.
impute_debutant_features(team_features: pd.DataFrame, ds2: pd.DataFrame) -> pd.DataFrame For the 12 teams with wc_debut_modern_flag = True, imputes WC historical features using confederation-mean strategy. For each debutant, finds all non-debutant teams in the same confederation (from DS2 confederation column) and computes their mean for each WC historical feature. Replaces None values with confederation means. Stores a boolean features_imputed column indicating which rows were imputed.
Dependencies: pandas, numpy, src/name_map.py, src/standings.py (for GROUP_MAP constant)

src/models.py
Purpose: Define, train, calibrate, and persist all model objects. Contains four model classes: GroupStageModel, KnockoutModel, BayesianUpdater, and ShootoutModel.
Inputs:
	•	train_df: pd.DataFrame — output of features.build_training_rows()
	•	team_features: pd.DataFrame — output of features.build_team_feature_table()
Outputs:
	•	Trained model objects (serialised to outputs/ as joblib files)
	•	Calibrated probability arrays
	•	Feature importance DataFrames
Functions:
class GroupStageModel Wraps XGBoost multi-class classifier for group stage (three-class: 2/1/0). Core logic:
GroupStageModel.__init__(self) Sets hyperparameter grid for cross-validation search. Default starting configuration: n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=4, gamma=0.2, lambda_=1.5, alpha=0.0. Core features list hardcoded as the 25 recommended features from the feature specification — no feature selection at runtime.
GroupStageModel.fit(self, train_df: pd.DataFrame, calibration_holdout: pd.DataFrame) -> None Performs six-fold chronological cross-validation (folds 1–6, each holding out one WC year) to select optimal hyperparameters. Uses WC year 2018 as the early stopping evaluation set (not the 2022 calibration holdout). After hyperparameter selection, retrains on all 1998–2018 data with optimal hyperparameters and selected n_estimators from early stopping. Stores trained model as self.model_. Applies Platt scaling calibration using 2022 holdout: fits a LogisticRegression(C=1.0) on the raw model log-odds against 2022 outcomes. Stores calibration parameters as self.calibrator_.
GroupStageModel.predict_proba(self, X: pd.DataFrame) -> np.ndarray Returns calibrated probability array of shape (n_samples, 3) with columns [P(WIN), P(DRAW), P(LOSS)]. Applies Platt scaling to raw XGBoost softmax probabilities. Normalises row sums to exactly 1.0 after calibration.
GroupStageModel.get_feature_importance(self) -> pd.DataFrame Returns gain-based feature importance sorted descending. Columns: feature, importance_gain, importance_weight, importance_cover.
GroupStageModel.save(self, path: str) -> None Serialises the entire model object including calibration parameters to joblib format at path.
GroupStageModel.load(cls, path: str) -> GroupStageModel Class method. Loads and returns a GroupStageModel from a joblib file.
class KnockoutModel Wraps two sub-models: KnockoutModelEarly (R32 + R16) and KnockoutModelLate (QF + SF + Final). Manages the stage-conditional logic.
KnockoutModel.__init__(self) Initialises both sub-model objects. Sets DRAW_RATE_90MIN = 0.214 (verified from 24/112 modern WC KO matches). Sets EXTRA_TIME_DECIDES_FRACTION = 0.375 (approximately 15 of 40 historical drawn KO matches decided in extra time without penalties).
KnockoutModel.fit(self, train_df: pd.DataFrame, calibration_holdout: pd.DataFrame) -> None Trains both sub-models. For KnockoutModelEarly: filters training data to Round of 16 / R32-equivalent matches (R16 + Second round from DS8), excludes draws from target, fits XGBoost binary classifier. For KnockoutModelLate: filters to QF + SF + Final matches (49 rows), prefers LogisticRegression baseline due to small sample, uses top 12 features only. Applies Platt scaling to both using 2022 KO holdout rows.
KnockoutModel.predict_proba(self, X: pd.DataFrame, stage_id: int) -> np.ndarray Routes prediction to the appropriate sub-model based on stage_id. Returns array of shape (n_samples, 3) with columns [P(win at 90min), P(draw at 90min), P(loss at 90min)]. P(draw) is always DRAW_RATE_90MIN = 0.214; P(win) and P(loss) are model outputs scaled by (1 - 0.214).
class BayesianUpdater Applies the tournament form update to pre-tournament probabilities using α = 0.17.
BayesianUpdater.__init__(self, alpha: float = 0.17) Sets shrinkage parameter. Alpha is derived from empirical optimal value across six validation years (mean = 0.17).
BayesianUpdater.compute_form_multiplier(self, team: str, team_features: pd.DataFrame) -> float Computes the form multiplier for a team based on their 2026 tournament statistics relative to Elo-expected performance. Uses the four-component formula: points relative to expectation, GD relative to expectation, goals scored normalised, goals conceded normalised (inverted). Returns a float in [0.5, 2.0].
BayesianUpdater.update_probability(self, p_prior: float, p_likelihood_component: float) -> float Applies the update formula: (1 - alpha) * p_prior + alpha * p_likelihood. Returns updated probability. This is applied independently to each class probability in the multi-class setting, with renormalisation afterward.
BayesianUpdater.update_match_probabilities(self, proba_array: np.ndarray, team_a: str, team_b: str, team_features: pd.DataFrame) -> np.ndarray Applies the Bayesian update to a full probability vector [P(A wins), P(draw), P(B wins)]. Updates each element using the form multiplier for each team. Renormalises. Returns updated array.
class ShootoutModel Three-component ensemble: historical shootout win rate (F040), Elo win expectancy (F002), WC-specific rate (F042).
ShootoutModel.__init__(self, k_shrinkage: float = 8.0, weights: tuple = (0.50, 0.40, 0.10)) Sets shrinkage strength and component weights. Default weights: w1=0.50 (historical rate), w2=0.40 (Elo), w3=0.10 (WC-specific, zeroed if fewer than 2 WC appearances).
ShootoutModel.predict_winner_proba(self, team_a: str, team_b: str, team_features: pd.DataFrame) -> float Returns P(team_a wins shootout). Retrieves F040, F002, and F042 from team_features. Applies shrinkage to F040. Normalises the historical rate ratio between teams. Combines three components with weights. Normalises final output to ensure P(A) + P(B) = 1.0. Floors at 0.15 and caps at 0.85 to prevent extreme predictions from sparse data.
Dependencies: pandas, numpy, xgboost, sklearn.linear_model.LogisticRegression, sklearn.preprocessing.StandardScaler, sklearn.calibration, joblib, src/name_map.py, src/features.py

src/simulation.py
Purpose: Monte Carlo tournament simulation engine. Runs 10,000 independent complete tournament simulations from the MD2 freeze point through the Final. Produces win probability distributions with confidence intervals.
Inputs:
	•	team_features: pd.DataFrame — from features.build_team_feature_table()
	•	frozen_standings: dict[str, pd.DataFrame] — from standings.compute_all_standings() on 48 frozen results
	•	group_stage_model: GroupStageModel — trained and calibrated
	•	knockout_model: KnockoutModel — trained and calibrated
	•	bayesian_updater: BayesianUpdater
	•	shootout_model: ShootoutModel
	•	bracket: pd.DataFrame — DS16 knockout bracket structure
	•	md3_fixtures: pd.DataFrame — 24 Matchday 3 matches from DS9
Outputs:
	•	win_probabilities: pd.DataFrame — 48 rows, columns: team, win_prob, win_prob_lower_90, win_prob_upper_90, avg_round_reached, final_appearances, semifinal_appearances
	•	simulation_log: pd.DataFrame — 10,000 rows, one per simulation, recording winner and full bracket path
	•	bracket_summary: pd.DataFrame — most common matchups at each stage across all simulations
Functions:
class TournamentSimulator
TournamentSimulator.__init__(self, group_stage_model, knockout_model, bayesian_updater, shootout_model, team_features, frozen_standings, bracket, md3_fixtures) Stores all model objects and data. Pre-computes MD3 match probability vectors (with Bayesian update applied) for all 24 fixtures so they are not recomputed in every simulation loop. Sets N_SIMULATIONS = 10_000.
TournamentSimulator.run(self) -> tuple[pd.DataFrame, pd.DataFrame] Outer simulation loop. Iterates N_SIMULATIONS times. For each iteration calls _run_single_simulation(seed=i). Collects results. Aggregates into win probabilities DataFrame. Computes 90% confidence intervals as p ± 1.645 × sqrt(p × (1-p) / N_SIMULATIONS). Returns (win_probabilities, simulation_log).
TournamentSimulator._run_single_simulation(self, seed: int) -> dict Executes a single complete tournament simulation. Sets numpy random seed to seed. Returns a dict recording the winner and round reached for all 48 teams.
Phase 1 — Matchday 3: For each of the 12 group pairs, simultaneously draws two outcomes using pre-computed probability vectors. Applies group context adjustments: subtracts 0.03 from win probability for teams with already_qualified = True, increases draw probability for such teams to reflect rotation patterns.
Phase 2 — Standing resolution: Calls standings.compute_all_standings() with the simulated MD3 results appended to the frozen 48 results. Calls standings.rank_third_place_teams(). Calls standings.get_qualified_teams() to map all 32 qualifiers to bracket positions.
Phase 3 — Knockout simulation: Iterates through all 32 knockout matches in stage order (R32 first, then R16, then QF, then SF, then Final). For each match, looks up the two teams from the current bracket state. Calls knockout_model.predict_proba() with the appropriate stage_id. Calls bayesian_updater.update_match_probabilities() to incorporate in-tournament form. Draws an outcome. If draw: calls _resolve_extra_time_or_penalties(). Records the winner and advances them in the bracket.
Phase 4 — Recording: Returns a dict with winner, runner_up, third_place, fourth_place, and {team: round_eliminated} for all 48 teams.
TournamentSimulator._resolve_extra_time_or_penalties(self, team_a: str, team_b: str) -> str Handles the 21.4% of knockout matches drawn at 90 minutes. Draws from Bernoulli with p = EXTRA_TIME_DECIDES_FRACTION = 0.375 to determine if extra time resolves the match. If extra time decides: applies knockout model with slightly elevated win probability for favoured team (+0.05). If penalties: calls shootout_model.predict_winner_proba() and draws the shootout winner. Returns winning team name.
TournamentSimulator._apply_group_context_adjustment(self, proba: np.ndarray, team: str, opponent: str, current_standings_md2: dict) -> np.ndarray Adjusts group stage probabilities for teams that are already qualified or already eliminated. Already qualified teams: reduce win probability by 0.03, redistribute to draw. Already eliminated teams: reduce draw probability by 0.02, redistribute to win and loss proportionally (eliminate-and-attack or accept-defeat pattern). Returns adjusted probability array.
TournamentSimulator._resolve_third_place_seeding(self, third_place_teams: dict) -> dict Maps each of the eight best third-place teams to their specific bracket slot. Implements the FIFA 2026 seeding table for third-place teams, which assigns each qualifying third-place team to a specific R32 match based on which group they finished third in. This is a deterministic lookup, not a probabilistic step.
aggregate_simulation_results(simulation_log: pd.DataFrame) -> pd.DataFrame Post-processing function. Takes the 10,000-row simulation log and computes per-team statistics. Returns the 48-row win_probabilities DataFrame with confidence intervals and round-reached statistics.
compute_bracket_summary(simulation_log: pd.DataFrame) -> pd.DataFrame Computes the most common matchups at each stage. For each R32 match slot, finds the top 5 most common team pairings across all simulations and their frequencies.
Dependencies: pandas, numpy, src/name_map.py, src/standings.py, src/features.py, src/models.py

src/leakage_guard.py
Purpose: Automated assertions that verify no post-freeze data has entered any pipeline stage. Called at the beginning of every notebook and as part of the test suite. If any assertion fails, execution halts with an explicit error message identifying the violation.
Inputs:
	•	feature_table: pd.DataFrame — the team features table to inspect
	•	training_rows: pd.DataFrame — the training dataset to inspect
Outputs: None on success. Raises LeakageError (a custom exception class defined in this module) with a descriptive message on failure.
Functions:
class LeakageError(Exception) Custom exception class. Carries a violation_type attribute (string: "FUTURE_DATE", "SYNTHETIC_DATA", "WRONG_SNAPSHOT", "FORBIDDEN_DATASET", "TACTICAL_IMPUTATION"). Raises with a message that identifies the specific rule violated from the seven leakage prevention rules.
check_freeze_date(df: pd.DataFrame, date_column: str, freeze_date: str = '2026-06-23') -> None Asserts that no row in a DataFrame has a date value after freeze_date. Raises LeakageError(violation_type='FUTURE_DATE') with the offending date if found. Applied to: DS4 rows used in training, DS1 rows used in features.
check_elo_snapshot(ds2: pd.DataFrame) -> None Asserts that the Elo snapshot used for current-strength features is exactly 2026-05-27. Verifies this by checking that the DS2 subset used in feature construction has no snapshot_date values other than 2026-05-27 or YYYY-12-31 (the latter only in the historical training rows). Raises LeakageError(violation_type='WRONG_SNAPSHOT') if a 2026 snapshot other than 2026-05-27 is found.
check_no_synthetic_data(feature_table: pd.DataFrame) -> None Verifies that no column names from DS3 (the synthetic player performance dataset) appear in the feature table. Column names to check against: player_id, player_name, tournament_rating, total_goals_tournament, total_assists_tournament, player_of_match_awards, clutch_performance_score, pressure_resistance. Raises LeakageError(violation_type='SYNTHETIC_DATA') if any are found.
check_no_md3_in_features(feature_table: pd.DataFrame) -> None Verifies that tournament features in the team feature table reflect exactly 48 matches (Matchday 1 + Matchday 2 only). Checks that no team has tourn_pts_md2 > 6 (impossible in 2 matches) and that match counts per team are exactly 2. Raises LeakageError(violation_type='FUTURE_DATE') if any team's match count exceeds 2.
check_tactical_gap_preserved(feature_table: pd.DataFrame) -> None Verifies that Portugal, Colombia, England, and Croatia have has_full_tactical_md2 = False. If any of these four teams has has_full_tactical_md2 = True, it means their June 23 DS1-ext match was incorrectly processed as having tactical data. Raises LeakageError(violation_type='TACTICAL_IMPUTATION').
check_training_rows_chronological(training_rows: pd.DataFrame) -> None Verifies that within the training data, features for any match are constructed from data predating that match. Checks that the elo_year_used column (added by build_training_rows) equals match_year - 1 for all rows. Raises LeakageError(violation_type='FUTURE_DATE') if any row has elo_year_used >= match_year.
run_all_checks(feature_table: pd.DataFrame, training_rows: pd.DataFrame, ds2: pd.DataFrame) -> None Runs all individual checks in sequence. Designed to be called as the first cell in any notebook that uses features or training data. A clean run produces no output. A failed run raises the specific LeakageError and halts execution.
Dependencies: pandas, numpy

notebooks/01_data_audit.ipynb
Purpose: Verify the exact state of the frozen dataset. Reproduce the verified findings from the data audit: 48 completed matches, group standings, third-place ranking, name mapping correctness, and freeze manifest validation. This notebook does not produce model inputs — it produces verification evidence.
Inputs:
	•	All 13 source files in data/raw/ and data/freeze/
Outputs:
	•	outputs/group_standings_freeze.csv — the verified 12-group standings after MD2
	•	Console display of third-place ranking table
	•	Console display of SHA-256 checksum verification against FREEZE_MANIFEST.md
	•	Console display of name mapping validation (all 48 teams resolve correctly through all three maps)
Cell structure:
Cell 1 — Imports and setup. Imports all source modules. Calls leakage_guard.run_all_checks() — this will fail at this stage because features haven't been built yet, so only check_no_synthetic_data and the file checksum check are run here. Loads all 13 raw files.
Cell 2 — File verification. Computes SHA-256 checksums for all 13 files and compares against FREEZE_MANIFEST.md values. Displays pass/fail for each file. Raises AssertionError if any checksum fails.
Cell 3 — Build frozen 48-match result set. Combines DS1 (44 rows) with DS1-ext (4 rows from june23_results.csv). Verifies exactly 48 rows. Verifies the four June 23 scores exactly. Displays the complete 48-match result table.
Cell 4 — Compute and display group standings. Calls standings.compute_all_standings(). Displays all 12 groups' standings. Verifies the exact standings against known values: Mexico 6pts, France 6pts, Norway 6pts, Argentina 6pts, Colombia 6pts, USA 6pts, Germany 6pts — these are the verified six-point leaders.
Cell 5 — Third-place ranking. Calls standings.rank_third_place_teams(). Displays ranked table. Verifies the cut-line at position 8 is Czechia with 1pt, GD −1. Notes that positions 7–9 (Belgium, Czechia, Congo DR) are within 1 point and separated by GD.
Cell 6 — Name mapping validation. Calls name_map.get_all_48(). For each team, calls normalise() for all three target systems. Verifies zero failures. Displays the complete mapping table with all 48 teams and their three alternate names.
Cell 7 — Data quality summary. Displays null counts for all key columns across all datasets. Flags: DS1 has 2 null offsides values (acceptable); DS4 has 44 null scores (abandoned matches — filtered in feature engineering); DS8 has 836/964 null xG values (pre-2018 tournaments — feature available for 2018 and 2022 only); DS6 has 422/678 null first_shooter values (excluded feature). Confirms all issues are known and handled.
Cell 8 — Save outputs. Writes group_standings_freeze.csv.
Dependencies: All src/ modules. pandas, hashlib, pathlib.

notebooks/02_feature_engineering.ipynb
Purpose: Build the complete feature table for all 48 teams and all training/prediction rows. This is the most computationally intensive notebook. Every feature specified in the feature engineering specification is constructed and verified here.
Inputs:
	•	All 13 raw files
	•	data/freeze/june23_results.csv
	•	All src/ modules
Outputs:
	•	outputs/team_features_freeze.parquet — the 48-row team feature table
	•	outputs/training_rows.parquet — 896-row training dataset with outcome and sample_weight
	•	outputs/prediction_rows.parquet — 112-row prediction dataset
	•	outputs/feature_audit_report.csv — per-feature null counts, ranges, and validation flags
Cell structure:
Cell 1 — Imports, load data, run leakage guards.
Cell 2 — Build Elo features for all 48 teams. Calls features.build_elo_features() for each team. Displays the 48-row Elo feature table sorted by elo_rating descending. Verifies: Spain has elo_rating = 2165, Qatar has elo_rating = 1425, USA/Canada/Mexico have elo_is_host = 1, all others have elo_is_host = 0.
Cell 3 — Build FIFA features for all 48 teams. Calls features.build_fifa_features(). Displays fifa_points_delta as a momentum table — Norway shows positive momentum despite lower FIFA rank than Elo rank. Verifies elo_fifa_rank_disagreement peaks at Qatar (38 positions), Egypt (22), Norway (19), Morocco (17).
Cell 4 — Build WC historical features. Calls features.build_wc_historical_features(). Displays debutant flag table. Verifies 12 teams have wc_debut_modern_flag = True. Calls features.impute_debutant_features() and shows the confederation-mean imputed values for all 12 debutants.
Cell 5 — Build recent form features. Calls features.build_recent_form_features() for all 48 teams. Displays form table sorted by form_win_rate_last10_competitive. Verifies England shows 1.0 win rate (10 wins in 10). Verifies Brazil shows 0.4. Displays the unbeaten streak table.
Cell 6 — Build tournament features. Calls features.build_tournament_features(). Verifies exactly 4 teams have has_full_tactical_md2 = False (Portugal, Colombia, England, Croatia). Displays tournament GF/GA/Pts table sorted by Pts. Verifies Germany leads with 9 GF. Runs leakage_guard.check_tactical_gap_preserved().
Cell 7 — Build shootout features. Calls features.build_shootout_features(). Displays shrinkage-adjusted rates table. Verifies Germany = 0.625, Argentina = 0.639, England = 0.375, Netherlands = 0.322. Verifies Norway and Czechia have shootout_naive_flag = True.
Cell 8 — Assemble full team feature table. Calls features.build_team_feature_table(). Verifies shape is (48, 55). Runs leakage_guard.check_no_synthetic_data(). Displays null count per feature — all features should have ≤ 12 nulls (debutant WC features before imputation).
Cell 9 — Build training rows. Calls features.build_training_rows(). Verifies shape is (896, 57) where 57 = 55 features + outcome + sample_weight. Displays class distribution: {2: WIN count, 1: DRAW count, 0: LOSS count} — should match 136×2, 83×2, 117×2 roughly (doubled for both perspectives). Runs leakage_guard.check_training_rows_chronological().
Cell 10 — Build prediction rows. Calls features.build_prediction_rows(). Verifies shape is (112, 55). Verifies first 48 rows (MD3 group stage, both perspectives) have complete team identity. Verifies last 64 rows (knockout stage, both perspectives) have None for team-specific features (bracket unresolved).
Cell 11 — Feature audit report. Computes per-feature null counts, min, max, mean, and a flag indicating if any value is outside expected range. Saves as feature_audit_report.csv. Displays rows with any out-of-range values.
Cell 12 — Save outputs. Saves all three DataFrames as parquet files.
Dependencies: All src/ modules. pandas, numpy, pyarrow (for parquet).

notebooks/03_baseline_model.ipynb
Purpose: Train and evaluate both the XGBoost primary model and the logistic regression baseline. Perform chronological cross-validation, probability calibration, and feature importance analysis. Produce the final trained model objects.
Inputs:
	•	outputs/training_rows.parquet
	•	outputs/team_features_freeze.parquet
	•	All src/ modules
Outputs:
	•	outputs/group_stage_model.joblib — trained and calibrated GroupStageModel
	•	outputs/knockout_model.joblib — trained and calibrated KnockoutModel
	•	outputs/feature_importance_group.csv
	•	outputs/feature_importance_knockout.csv
	•	outputs/cv_results.csv — per-fold accuracy, RPS, Brier Score for both models
	•	outputs/calibration_curve.png — reliability diagram for both models on 2022 holdout
Cell structure:
Cell 1 — Imports, load data. Load training rows and team features.
Cell 2 — Define chronological folds. Splits training_rows into 7 folds by wc_year column. Displays fold sizes — each fold should be 64 matches × 2 perspectives = 128 rows for the holdout, 384 × 2 = 768 rows for training.
Cell 3 — Logistic regression baseline. Instantiates and fits LogisticRegression with C=0.1, class_weight='balanced', solver='lbfgs', multi_class='multinomial'. Applies StandardScaler fitted on training fold only. Runs six-fold chronological CV. Records Accuracy, RPS, and Brier Score per fold. Displays coefficient table — verifies directional expectations (elo_win_expectancy coefficient is positive).
Cell 4 — XGBoost group stage model. Calls GroupStageModel().fit(). This cell runs the six-fold CV internally. Displays best hyperparameters selected. Displays CV metric comparison between XGBoost and logistic regression across all six folds. Checks if XGBoost RPS improvement exceeds the 5% threshold over logistic regression.
Cell 5 — Calibration analysis. Applies both models to the 2022 calibration holdout (64 matches). Plots reliability diagram — predicted probability bins on x-axis, actual win rate on y-axis. Verifies that uncalibrated Elo WE shows the documented miscalibration at extremes (0.9 bin predicts 92% but actual win rate is 62.5%). Verifies that Platt scaling corrects this. Saves calibration_curve.png.
Cell 6 — Knockout model. Calls KnockoutModel().fit(). Splits training data to knockout matches only. Displays stage-group A and stage-group B performance metrics separately. Verifies that Stage Group B (QF/SF/Final) prefers logistic regression over XGBoost due to the 49-match training set.
Cell 7 — Feature importance. Calls group_stage_model.get_feature_importance(). Displays top 15 features by gain. Verifies elo_win_expectancy ranks first or second. Saves importance CSVs.
Cell 8 — Sanity checks. Applies group stage model to 10 specific known historical matches (from 2022 WC) and verifies that predictions are directionally correct (the model assigns > 50% win probability to the team that historically had higher Elo rating in 8 of 10 cases). Applies knockout model to the 2022 Final (Argentina vs France) and verifies that the predicted winner probability is between 40% and 60% (a close match as expected).
Cell 9 — Save models.
Dependencies: outputs/training_rows.parquet, outputs/team_features_freeze.parquet, all src/ modules. xgboost, sklearn, matplotlib, joblib.

notebooks/04_tournament_update.ipynb
Purpose: Apply the Bayesian tournament update and show the before/after probability shift for each of the 48 teams based on their Matchday 1 and Matchday 2 performance. This notebook is primarily for interpretation and communication — the update is applied inside the simulation, not as a separate pre-computation step.
Inputs:
	•	outputs/group_stage_model.joblib
	•	outputs/team_features_freeze.parquet
	•	outputs/prediction_rows.parquet
Outputs:
	•	outputs/bayesian_update_table.csv — 48-row table showing each team's pre-update and post-update win probability estimate
	•	outputs/charts/update_shifts.png — horizontal bar chart showing probability shifts
Cell structure:
Cell 1 — Imports, load models and features.
Cell 2 — Compute pre-update baseline probabilities. For each of the 24 MD3 fixtures, applies the group stage model using Feature Groups 1–4 only (historical features — no tournament features). Computes P(Win), P(Draw), P(Loss) for both teams. Displays the 24-match pre-update probability table.
Cell 3 — Compute form multipliers. Instantiates BayesianUpdater(alpha=0.17). Calls compute_form_multiplier() for all 48 teams. Displays the multiplier table sorted descending. Verifies that Germany (9 GF, 7 GD, 6 pts) has the highest positive multiplier among teams of comparable Elo rank. Verifies that Türkiye (0 pts, 75.5% possession, 0 goals) has a negative multiplier.
Cell 4 — Apply update and compare. Applies update to all 24 MD3 fixture probabilities. Creates a comparison table: team, pre-update win probability in MD3 match, post-update win probability, shift. Identifies the five largest positive shifts and five largest negative shifts.
Cell 5 — Visualise shifts. Creates a horizontal bar chart with teams on y-axis, probability shift on x-axis. Bars are coloured green for positive shift, red for negative, grey for negligible (<1%). Saves as update_shifts.png.
Cell 6 — Discuss the α = 0.17 constraint. Displays a sensitivity table showing what final tournament win probabilities would look like at α = 0.0 (no update), α = 0.17 (chosen), and α = 0.30 (aggressive update). Shows that at α = 0.30, Germany's win probability would exceed France's despite their lower Elo — a plausible but debatable conclusion that the conservative α = 0.17 avoids.
Cell 7 — Save outputs.
Dependencies: outputs/group_stage_model.joblib, outputs/team_features_freeze.parquet, all src/ modules.

notebooks/05_monte_carlo.ipynb
Purpose: Execute the 10,000-simulation Monte Carlo tournament engine. This is the computational centrepiece. All five simulation phases (Matchday 3, standings resolution, bracket construction, knockout simulation, results aggregation) run here.
Inputs:
	•	outputs/group_stage_model.joblib
	•	outputs/knockout_model.joblib
	•	outputs/team_features_freeze.parquet
	•	outputs/prediction_rows.parquet
	•	data/raw/arc_base/matches.csv (DS16 bracket)
	•	data/raw/arc2_new/schedule_2026.csv (DS9 MD3 fixtures)
Outputs:
	•	outputs/win_probabilities.csv — final 48-row win probability table with CIs
	•	outputs/simulation_log.parquet — 10,000-row detailed simulation log
	•	outputs/bracket_simulation_summary.csv — most common matchups per round
	•	outputs/charts/win_probability_chart.png — horizontal bar chart of win probs
Cell structure:
Cell 1 — Imports, load all models and data. Load all five model objects. Run leakage guards on feature table.
Cell 2 — Initialise simulator. Instantiates TournamentSimulator with all model objects and data. Pre-computes MD3 probability vectors for all 24 fixtures. Displays the 24 pre-computed probability vectors as a table.
Cell 3 — Verify bracket structure. Displays all 32 knockout match slots from DS16 with their match labels, cities, and stage IDs. Verifies that the R32 match labels encode the seeding rules correctly. Verifies the Final is match_id 104 in MetLife Stadium, New York.
Cell 4 — Run 1,000 simulations (test run). Runs 1,000 simulations first to verify no errors and estimate runtime. Displays preliminary win probability estimates. Estimates total runtime for 10,000 simulations. If runtime exceeds 5 minutes, suggests optimisation (vectorising the simulation loop).
Cell 5 — Run full 10,000 simulations. Calls simulator.run() with a progress indicator. Stores win_probabilities and simulation_log. Verifies sum of all win probabilities equals exactly 1.0 (within floating point tolerance). Verifies no team has a negative probability. Verifies the simulation log has exactly 10,000 rows.
Cell 6 — Compute confidence intervals. Calls aggregate_simulation_results(). Displays the final win probability table sorted descending. Highlights teams where the 90% CI is wide (uncertainty > 5 percentage points) versus narrow.
Cell 7 — Bracket summary. Calls compute_bracket_summary(). Displays the most common R32, QF, SF, and Final matchups. Highlights the most common Final pairing and its frequency.
Cell 8 — Upset analysis. Counts simulations where at least one top-8 Elo team was eliminated before the QF. Displays distribution of how many top-8 teams survive to the QF across simulations. Highlights specific upset patterns — what is the most common path for a team ranked outside the top 10 to reach the Final?
Cell 9 — Save all outputs.
Dependencies: All model joblib files, all src/ modules. pandas, numpy, matplotlib, tqdm (progress bar).

notebooks/06_results_and_visualisation.ipynb
Purpose: Produce all final outputs for publication. Translate model outputs into portfolio-ready visualisations and the LinkedIn narrative. This notebook contains no new computation — it reads from outputs/ and formats for presentation.
Inputs:
	•	outputs/win_probabilities.csv
	•	outputs/group_standings_freeze.csv
	•	outputs/bracket_simulation_summary.csv
	•	outputs/feature_importance_group.csv
	•	outputs/bayesian_update_table.csv
Outputs:
	•	outputs/charts/win_probability_chart.png — final horizontal bar chart
	•	outputs/charts/group_standings_heatmap.png — 12-group standings visualisation
	•	outputs/charts/bracket_tree.png — predicted bracket path tree
	•	outputs/charts/feature_importance_chart.png — top 15 features bar chart
	•	outputs/charts/calibration_curve.png (already produced in notebook 03)
	•	outputs/final_summary_table.csv — human-readable summary for LinkedIn
Cell structure:
Cell 1 — Load all outputs.
Cell 2 — Win probability chart. Horizontal bar chart, all 48 teams, sorted by win probability. Color-coded by confederation. Error bars showing 90% CI. Annotated with win probability labels. Caption: "FIFA World Cup 2026 Tournament Win Probabilities — based on 10,000 Monte Carlo simulations, Matchday 2 freeze".
Cell 3 — Group standings heatmap. Colour-coded table of all 12 groups showing final MD2 standings. Green = qualified, yellow = in the race, red = eliminated/struggling. Annotated with points and GD.
Cell 4 — Predicted bracket. Tree diagram of the most likely knockout bracket path based on modal simulation outcome. Shows the most common team at each bracket slot. Annotated with probability of each team reaching that slot.
Cell 5 — Feature importance chart. Top 15 features from the group stage model, bar chart by gain. Annotated with brief descriptions of what each feature measures.
Cell 6 — Bayesian update shifts. Horizontal diverging bar chart showing the probability shift caused by the tournament update for all 48 teams. Before and after values displayed.
Cell 7 — Model accuracy summary box. Text cell summarising: training data size (448 WC matches), chronological CV accuracy (target > 69.5%), Elo baseline accuracy (66.7%), number of simulations (10,000), calibration holdout (2022 WC, 64 matches).
Cell 8 — Produce LinkedIn post text. Markdown cell with the template LinkedIn post: headline, predicted champion, top 5 win probabilities, biggest surprise findings from the data, a statement about the methodology, and a link placeholder.
Cell 9 — Save all charts at 300 DPI.

tests/test_name_map.py
Purpose: Verify that every name translation is correct and that no team is silently dropped or incorrectly mapped.
Test functions:
test_all_48_teams_in_to_elo() — asserts that all 48 canonical team names from get_all_48() are present as keys in TO_ELO.
test_all_48_teams_in_to_fifa() — same for TO_FIFA.
test_all_48_teams_in_to_ds4() — same for TO_DS4.
test_known_overrides_correct() — verifies the 7 known non-identity mappings explicitly: normalise('Korea Republic', 'elo') == 'South Korea', normalise('IR Iran', 'elo') == 'Iran', etc.
test_identity_for_non_override_teams() — verifies that teams with no known name variant return themselves: normalise('Germany', 'elo') == 'Germany'.
test_unknown_name_raises() — verifies that normalise('England U21', 'elo') raises ValueError.
test_us_variants_handled() — verifies both "United States" (DS1 form) and "USA" (DS10 form) resolve correctly through the maps.

tests/test_standings.py
Purpose: Verify that standings computation and tiebreaker logic produce the correct results from the known frozen 48-match dataset.
Test functions:
test_group_a_standings() — asserts Mexico 6pts, Korea Republic 3pts, Czechia 1pt, South Africa 1pt in Group A from the frozen results.
test_group_i_standings() — asserts France 6pts (GD+5), Norway 6pts (GD+4), both qualified with France on top by GD.
test_group_k_standings() — asserts Colombia 6pts, Portugal 4pts, Congo DR 1pt, Uzbekistan 0pts after the June 23 results are included.
test_group_l_standings() — asserts England 4pts (GD+2), Ghana 4pts (GD+1), Croatia 3pts, Panama 0pts.
test_third_place_ranking() — asserts Sweden (Group F, 3pts, GD 0) ranks first among third-place teams and Senegal (Group I, 0pts) ranks last.
test_third_place_cutline() — asserts that the 8th-place third team (Czechia, 1pt, GD −1) is ahead of the 9th-place team (either Congo DR or Ecuador, also 1pt but lower GD).
test_total_qualified_teams() — asserts that get_qualified_teams() returns exactly 32 teams (24 group qualifiers + 8 best third-place).
test_concurrent_md3_pairs() — verifies that for each of the 12 group pairs, both matches are drawn simultaneously and not sequentially. Uses a mock random number generator that returns predetermined outcomes and verifies they are applied to both matches before standings are resolved.

tests/test_features.py
Purpose: Verify feature construction correctness, missing value handling, and debutant imputation.
Test functions:
test_elo_features_spain() — asserts Spain elo_rating == 2165, elo_rank == 1, elo_is_host == 0.
test_elo_features_usa() — asserts elo_is_host == 1.
test_elo_feature_count() — asserts that build_elo_features() returns exactly 5 keys.
test_tournament_features_germany() — asserts Germany tourn_gf_md2 == 9, tourn_pts_md2 == 6, tourn_gd_md2 == 7, has_full_tactical_md2 == True.
test_tournament_features_england() — asserts England has_full_tactical_md2 == False (June 23 match is scores-only).
test_tournament_features_portugal() — asserts Portugal has_full_tactical_md2 == False.
test_shootout_germany() — asserts Germany shrunk_rate ≈ 0.625 (6 wins, 8 appearances, k=8: (6+4)/(8+8) = 0.625).
test_shootout_england() — asserts England shrunk_rate ≈ 0.375 (4 wins, 12 appearances, k=8: (4+4)/(12+8) = 0.40, close to 0.375 with exact k).
test_shootout_norway_naive() — asserts Norway shootout_naive_flag == True and shootout_win_rate_alltime == 0.5.
test_debutant_flag() — asserts that Curaçao, Uzbekistan, and Cape Verde have wc_debut_modern_flag == True.
test_training_rows_shape() — asserts training_rows.shape == (896, 57).
test_training_rows_target_distribution() — asserts outcome column value counts roughly match {2: ~272, 1: ~166, 0: ~234} (doubled from 136/83/117).
test_no_future_data_in_training() — for all rows with match_year, asserts elo_year_used == match_year - 1.

tests/test_leakage.py
Purpose: Verify that all seven leakage prevention rules are enforced correctly. Tests are adversarial — each test deliberately introduces a leakage violation and asserts that the leakage guard catches it.
Test functions:
test_future_date_caught() — creates a DataFrame with a row dated "2026-07-01" and asserts check_freeze_date() raises LeakageError with violation_type == 'FUTURE_DATE'.
test_wrong_elo_snapshot_caught() — creates a DS2 subset with a row having snapshot_date = '2026-06-15' (during the tournament) and asserts check_elo_snapshot() raises LeakageError with violation_type == 'WRONG_SNAPSHOT'.
test_synthetic_column_caught() — creates a DataFrame with a column named tournament_rating and asserts check_no_synthetic_data() raises LeakageError with violation_type == 'SYNTHETIC_DATA'.
test_md3_in_features_caught() — creates a team features row with tourn_pts_md2 = 9 (impossible in 2 MD2 matches) and asserts check_no_md3_in_features() raises LeakageError.
test_tactical_gap_england_caught() — creates a features row for England with has_full_tactical_md2 = True and asserts check_tactical_gap_preserved() raises LeakageError.
test_clean_data_passes_all_checks() — runs run_all_checks() on the actual feature table and training rows (from outputs/) and asserts no exception is raised. This is the single integration test that confirms the production data passes all guards.

tests/test_models.py
Purpose: Verify model objects behave correctly: probabilities sum to 1.0, calibration is monotonic, feature importance is produced, and the Bayesian updater respects its bounds.
Test functions:
test_group_model_proba_sums_to_one() — runs predict_proba() on 10 sample rows and asserts each row's three probabilities sum to 1.0 within 1e-6.
test_knockout_model_draw_rate() — verifies that the draw probability in any knockout prediction is exactly 0.214.
test_bayesian_updater_bounds() — verifies that no multiplier produced by compute_form_multiplier() is outside [0.5, 2.0].
test_bayesian_updater_germany_positive() — verifies that Germany's form multiplier is > 1.0 (overperforming Elo expectation in tournament).
test_bayesian_updater_turkey_negative() — verifies that Türkiye's form multiplier is < 1.0.
test_shootout_proba_sums_to_one() — verifies P(A wins) + P(B wins) == 1.0 for every team pair tested.
test_shootout_germany_vs_england() — verifies P(Germany wins shootout vs England) > 0.55 (Germany's strong record vs England's poor record should produce a clear favourite).
test_shootout_bounds_enforced() — creates a hypothetical extreme case (team with 0 wins in 20 appearances vs team with 20 wins in 20 appearances) and verifies the output is clipped to [0.15, 0.85].
test_model_save_load_roundtrip() — saves GroupStageModel to a temp file, loads it back, and verifies predictions are identical to pre-save predictions.

tests/test_simulation.py
Purpose: Verify that the Monte Carlo simulation produces valid, reproducible outputs and that the bracket seeding logic correctly implements the FIFA 2026 rules.
Test functions:
test_win_probs_sum_to_one() — runs 100 simulations, asserts sum of all team win probabilities equals 1.0 within 1e-4.
test_exactly_one_winner_per_simulation() — asserts that each of the 100 test simulation rows has exactly one team as winner.
test_reproducibility() — runs the simulation twice with the same seed sequence and asserts identical results.
test_md3_concurrent_pairs() — verifies that for Group A (the test case), the Czechia vs Mexico and South Africa vs Korea Republic results are drawn simultaneously and that the standings function receives both results together.
test_bracket_seeding_group_a_winner() — in a controlled simulation where Group A winner is Mexico, verifies that Mexico is placed in the correct R32 bracket slot according to the DS16 match label.
test_third_place_seeding() — in a controlled simulation where Sweden qualifies as best third-place team from Group F, verifies Sweden is assigned to the correct R32 slot per the FIFA 2026 seeding table.
test_penalty_model_activates() — in a controlled simulation with manipulated probabilities forcing a draw at 90 minutes, verifies the shootout model is called and produces a winner.
test_all_48_teams_get_round_assigned() — after 10 test simulations, asserts every team has a non-None round_eliminated value in every simulation.
test_champion_confidence_interval() — runs 1,000 simulations and verifies the 90% CI width for the top team is below 5 percentage points (confirming 10,000 simulations would be below 1.7 percentage points).

requirements.txt
pandas==2.2.0
numpy==1.26.3
xgboost==2.0.3
scikit-learn==1.4.0
matplotlib==3.8.2
seaborn==0.13.2
joblib==1.3.2
pyarrow==15.0.0
tqdm==4.66.2
scipy==1.12.0
pytest==8.0.0
All versions pinned. A developer running pip install -r requirements.txt in a clean Python 3.11 environment must be able to reproduce all outputs identically.

README.md
Purpose: Entry point for anyone cloning the repository. Explains the project in three paragraphs: what it is, what it found, and how to reproduce it. Links to FREEZE_MANIFEST.md for data lineage and to the LinkedIn post for the public-facing narrative.
Sections:
Project summary: A reproducible FIFA World Cup 2026 tournament forecast built from a frozen snapshot of Matchday 2 results (June 23, 2026). Not a live-updating system — a time-stamped forecast produced at a specific moment and fully reproducible from that moment forward.
Key findings: The predicted tournament champion, top-5 win probabilities with confidence intervals, the two biggest positive surprises from Matchday 1–2 data, and the model's validated accuracy on the 2022 World Cup holdout.
Reproduction instructions: Clone the repo, install requirements, verify checksums against FREEZE_MANIFEST.md, run notebooks 01–06 in order. Estimated runtime per notebook: 01 (2 min), 02 (5 min), 03 (15 min), 04 (3 min), 05 (10 min), 06 (3 min).
Data lineage: Explanation of each source dataset, why DS3 (player performance) was excluded, and the four June 23 results that were manually injected.
Limitations: Two matchdays of tournament data is insufficient for high-confidence tournament form estimates. Matchday 3 results (happening today, June 24–27) will significantly alter the standings and the predicted qualifiers. The model makes predictions based on who is likely to qualify — these predictions will be wrong for groups decided by Matchday 3 upsets. Win probabilities are model estimates with quantified uncertainty, not point predictions.
 
 
 
 
 
 
 
 
 
 
 
Final Amendments Before Implementation
A1. DS16 Match Label Correction
A data-quality issue exists in DS16.
	•	Match label W95 vs W100 is invalid because it creates a self-reference chain.
	•	Correct label: W95 vs W96.
Implementation:
	•	Apply correction during ingestion.
	•	Record correction in FREEZE_MANIFEST.md.

A2. Additional Team Name Standardization
Extend the canonical mapping layer with:
Source Name
Canonical Name
Korea Republic
South Korea
United States
USA
Cape Verde
Cabo Verde
Rules:
	•	DS9 remains the canonical authority for 2026 team identity.
	•	DS17 IDs are used only for bracket and seeding logic.
	•	All joins must pass through the canonical mapping layer.

A3. Ensemble Weight Specification
Default ensemble:
	•	World Cup–only model: 70%
	•	Augmented competitive-match model: 30%
Hyperparameter search:
	•	WC-only weight range: 0.60–0.90
	•	Augmented weight: 1 − WC-only weight
Selection criterion:
	•	Lowest validation log loss
	•	Secondary metric: Brier score
Production weights must be recorded in FREEZE_MANIFEST.md.

A4. Qualification-State Flags
The following features are not stored manually:
	•	already_qualified
	•	already_eliminated
They are computed dynamically from standings state during simulation.
Implementation location:
	•	src/standings.py
Flags must be recomputed after every simulated matchday.

A5. FIFA Annex C Integration
The FIFA World Cup 2026 third-place placement table has been extracted and stored separately.
Reference dataset:
data/reference/third_place_annex_c.csv
Properties:
	•	495 unique combinations
	•	9 columns
	•	Derived directly from FIFA World Cup 2026 Regulations Annex C
Columns:
	•	combination_key
	•	match_74
	•	match_77
	•	match_79
	•	match_80
	•	match_81
	•	match_82
	•	match_85
	•	match_87
Usage:
	•	src/simulation.py
	•	_resolve_third_place_seeding()
Rules:
	•	Never hard-code bracket placement.
	•	Never use random assignment.
	•	Always resolve third-place teams through Annex C lookup.
This CSV is the authoritative source for Round of 32 placement of the eight best third-placed teams.

Final Readiness Status
Dataset Architecture: Complete
Leakage Audit: Complete
Feature Engineering Specification: Complete
Modeling Specification: Complete
Monte Carlo Design: Complete
Implementation Blueprint: Complete
Annex C Integration: Complete
Project Status: Ready for Implementation
 

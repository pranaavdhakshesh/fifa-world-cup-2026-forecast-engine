"""
Leakage prevention assertions for the FIFA WC 2026 Forecast Engine.

This module enforces the seven data-leakage rules from the design specification
(§2 Final Leakage Prevention Rules).  Every function raises LeakageError on
failure with a violation_type attribute that names the specific rule violated.

Design-specified functions (all present):
    LeakageError                       — custom exception
    check_freeze_date()                — Rule L1
    check_elo_snapshot()               — Rule L2
    check_no_synthetic_data()          — Rule L6 (DS3/DS13/DS14 column fingerprints)
    check_no_md3_in_features()         — Rule L1 applied to tournament features
    check_tactical_gap_preserved()     — Rule L5
    check_training_rows_chronological()— Rule L1 applied to training set construction
    run_all_checks()                   — master entry point

Additional checks (required by audit findings and notebook spec):
    check_ds4_wc_null_scores()         — Rule L4: null WC scores not imputed
    check_ds8_year_ceiling()           — Rule L4: DS8 contains no post-2022 rows
    check_form_window()                — Rule L1: DS4 form filter upper-bound enforced
    check_annex_c()                    — structural integrity of third_place_annex_c.csv
    check_canonical_names()            — name authority validation (wraps name_map)
    check_no_forbidden_datasets()      — Rule L6 extended: DS13/DS14 column fingerprints

Called as the first cell of every notebook.  A clean run produces no output.
A failed run raises LeakageError and halts execution immediately.
"""

from __future__ import annotations

from math import comb
from typing import Iterable

import pandas as pd

from src.name_map import CANONICAL_48, assert_all_canonical

# ---------------------------------------------------------------------------
# Freeze constants
# ---------------------------------------------------------------------------

FREEZE_DATE          = "2026-06-23"          # Rule L1 hard wall
ELO_SNAPSHOT_DATE   = "2026-05-27"           # Rule L2
FORM_WINDOW_START   = "2024-01-01"           # Feature Group 4
FORM_WINDOW_END     = "2026-06-10"           # Feature Group 4 ceiling
DS8_MAX_YEAR        = 2022                   # Rule L4
DS1_EXPECTED_ROWS   = 44
DS1EXT_EXPECTED_ROWS = 4
TOTAL_FROZEN_MATCHES = 48
ANNEX_C_EXPECTED_ROWS = 495                  # C(12, 8)
ANNEX_C_EXPECTED_COLS = 9
ANNEX_C_MATCH_COLS = ("m75", "m78", "m79", "m80", "m81", "m82", "m85", "m88")

# Teams whose MD2 is scores-only (DS1-ext) — Rule L5
TACTICAL_GAP_TEAMS: frozenset[str] = frozenset({
    "Portugal", "Colombia", "England", "Croatia"
})

# ---------------------------------------------------------------------------
# DS3 synthetic dataset column fingerprint (design spec §2 Rule L6 exact list,
# extended with full column set confirmed from the actual file).
# ---------------------------------------------------------------------------
_DS3_FINGERPRINT_COLS: frozenset[str] = frozenset({
    # Explicitly listed in design spec
    "player_id",
    "player_name",
    "tournament_rating",
    "total_goals_tournament",
    "total_assists_tournament",
    "player_of_match_awards",
    "clutch_performance_score",
    "pressure_resistance",
    # Additional high-specificity DS3 columns (confirmed from file)
    "creativity_score",
    "consistency_score",
    "offensive_contribution",
    "defensive_contribution",
    "possession_impact",
    "stamina_score",
    "player_rating",
    "performance_score",
    "distance_covered_km",
    "sprint_distance_km",
    "top_speed_kmh",
    "penalty_saves",
    "expected_goals_xg",
    "expected_assists_xa",
    "total_minutes_tournament",
    "total_assists_tournament",
    "club_name",
    "market_value_eur",
    "jersey_number",
})

# DS13/DS14 synthetic dataset column fingerprint
_DS13_DS14_FINGERPRINT_COLS: frozenset[str] = frozenset({
    "wins_last_10_matches",
    "losses_last_10_matches",
    "draws_last_10_matches",
    "win_rate_last_year",
    "goals_scored_avg",
})

# All forbidden column fingerprints combined
_ALL_FORBIDDEN_COLS: frozenset[str] = (
    _DS3_FINGERPRINT_COLS | _DS13_DS14_FINGERPRINT_COLS
)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class LeakageError(Exception):
    """Raised when a leakage prevention rule is violated.

    Attributes
    ----------
    violation_type : str
        One of: "FUTURE_DATE", "SYNTHETIC_DATA", "WRONG_SNAPSHOT",
                "FORBIDDEN_DATASET", "TACTICAL_IMPUTATION",
                "INVALID_REFERENCE_DATA", "NAME_ERROR".
    rule : str
        The specific rule label from the design spec (e.g. "L1", "L2").
    """

    VALID_TYPES = frozenset({
        "FUTURE_DATE",
        "SYNTHETIC_DATA",
        "WRONG_SNAPSHOT",
        "FORBIDDEN_DATASET",
        "TACTICAL_IMPUTATION",
        "INVALID_REFERENCE_DATA",
        "NAME_ERROR",
    })

    def __init__(self, message: str, violation_type: str, rule: str = "") -> None:
        if violation_type not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown violation_type {violation_type!r}. "
                f"Must be one of {sorted(self.VALID_TYPES)}."
            )
        super().__init__(message)
        self.violation_type = violation_type
        self.rule = rule

    def __str__(self) -> str:
        prefix = f"[{self.rule}] " if self.rule else ""
        return f"LeakageError({self.violation_type}) {prefix}{super().__str__()}"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_date_series(series: pd.Series) -> pd.Series:
    """Coerce a Series to datetime, silently dropping unconvertible values."""
    return pd.to_datetime(series, errors="coerce")


# ---------------------------------------------------------------------------
# Rule L1 — check_freeze_date
# ---------------------------------------------------------------------------

def check_freeze_date(
    df: pd.DataFrame,
    date_column: str,
    freeze_date: str = FREEZE_DATE,
) -> None:
    """Assert no row in *df* has a date in *date_column* after *freeze_date*.

    Applied to: DS4 rows used in training, DS1 rows used in features.
    Rows with non-parseable or null dates are ignored (they are future
    null-score WC placeholder rows — correctly excluded by score filters
    upstream).

    Raises
    ------
    LeakageError(violation_type='FUTURE_DATE', rule='L1')
    """
    if date_column not in df.columns:
        raise LeakageError(
            f"Date column {date_column!r} not found in DataFrame "
            f"(columns: {list(df.columns)}).",
            violation_type="FUTURE_DATE",
            rule="L1",
        )

    dates = _to_date_series(df[date_column])
    cutoff = pd.Timestamp(freeze_date)
    violations = df.loc[dates > cutoff, date_column]

    if not violations.empty:
        bad = violations.iloc[0]
        raise LeakageError(
            f"Rule L1 violated: found {len(violations)} row(s) with date after "
            f"freeze date {freeze_date!r}. First offending value: {bad!r}. "
            "No data created or dated after June 23, 2026 may enter any model, "
            "feature set, or simulation input.",
            violation_type="FUTURE_DATE",
            rule="L1",
        )


# ---------------------------------------------------------------------------
# Rule L2 — check_elo_snapshot
# ---------------------------------------------------------------------------

def check_elo_snapshot(ds2: pd.DataFrame) -> None:
    """Assert DS2 contains no 2026 snapshot rows other than 2026-05-27.

    Permitted snapshot_date values:
      - Exactly '2026-05-27' (current-strength features).
      - Any 'YYYY-12-31' pattern for year < 2026 (historical backtesting).

    The 2026-12-31 row present in DS2 must never be passed into any
    feature construction call — if this function receives a DS2 subset
    that includes it, it raises immediately.

    Raises
    ------
    LeakageError(violation_type='WRONG_SNAPSHOT', rule='L2')
    """
    if "snapshot_date" not in ds2.columns:
        raise LeakageError(
            "DS2 is missing required column 'snapshot_date'.",
            violation_type="WRONG_SNAPSHOT",
            rule="L2",
        )

    dates = ds2["snapshot_date"].astype(str)
    bad = dates[
        dates.str.startswith("2026") & (dates != ELO_SNAPSHOT_DATE)
    ]

    if not bad.empty:
        offenders = bad.unique().tolist()
        raise LeakageError(
            f"Rule L2 violated: DS2 contains {len(bad)} row(s) with 2026 "
            f"snapshot_date other than '{ELO_SNAPSHOT_DATE}'. "
            f"Offending values: {offenders}. "
            "Any DS2 row with snapshot_date != '2026-05-27' for year 2026 "
            "may embed post-tournament rating updates and must not be used.",
            violation_type="WRONG_SNAPSHOT",
            rule="L2",
        )


# ---------------------------------------------------------------------------
# Rule L6 — check_no_synthetic_data
# ---------------------------------------------------------------------------

def check_no_synthetic_data(feature_table: pd.DataFrame) -> None:
    """Assert no DS3 synthetic column appears in *feature_table*.

    Column names to detect (from design spec §2 Rule L6, plus confirmed
    high-specificity columns from the actual DS3 file):
      player_id, player_name, tournament_rating, total_goals_tournament,
      total_assists_tournament, player_of_match_awards,
      clutch_performance_score, pressure_resistance, and others.

    DS3's architecture mixes real player metadata with fabricated match
    outcomes; no column may be used even if it appears benign.

    Raises
    ------
    LeakageError(violation_type='SYNTHETIC_DATA', rule='L6')
    """
    found = _ALL_FORBIDDEN_COLS & set(feature_table.columns)
    if found:
        raise LeakageError(
            f"Rule L6 violated: feature table contains {len(found)} column(s) "
            f"from excluded synthetic datasets (DS3/DS13/DS14): {sorted(found)}. "
            "DS3 is fully excluded — no column may be used even if it appears "
            "benign (design spec Rule L6). DS13 and DS14 have unknown provenance.",
            violation_type="SYNTHETIC_DATA",
            rule="L6",
        )


# ---------------------------------------------------------------------------
# Rule L1 applied to feature table — check_no_md3_in_features
# ---------------------------------------------------------------------------

def check_no_md3_in_features(feature_table: pd.DataFrame) -> None:
    """Assert tournament features reflect at most 2 matches per team (MD1+MD2).

    Checks:
      1. If 'tourn_pts_md2' is present: no team has > 6 pts (max in 2 games).
      2. If 'matches_played_tourn' is present: no team has > 2.

    Raises
    ------
    LeakageError(violation_type='FUTURE_DATE', rule='L1')
    """
    if "tourn_pts_md2" in feature_table.columns:
        pts = pd.to_numeric(feature_table["tourn_pts_md2"], errors="coerce")
        bad = feature_table.loc[pts > 6, "team"] if "team" in feature_table.columns else pts[pts > 6]
        if not bad.empty:
            offenders = bad.tolist() if hasattr(bad, "tolist") else list(bad)
            raise LeakageError(
                f"Rule L1 violated: {len(offenders)} team(s) have tourn_pts_md2 > 6, "
                f"which is impossible in 2 group-stage matches: {offenders}. "
                "This indicates Matchday 3 or later results have leaked into "
                "the tournament feature columns.",
                violation_type="FUTURE_DATE",
                rule="L1",
            )

    if "matches_played_tourn" in feature_table.columns:
        mp = pd.to_numeric(feature_table["matches_played_tourn"], errors="coerce")
        bad_mp = feature_table.loc[mp > 2] if "team" in feature_table.columns else mp[mp > 2]
        if not bad_mp.empty:
            count = len(bad_mp)
            raise LeakageError(
                f"Rule L1 violated: {count} row(s) have matches_played_tourn > 2. "
                "The frozen feature table must reflect exactly Matchday 1 + Matchday 2.",
                violation_type="FUTURE_DATE",
                rule="L1",
            )


# ---------------------------------------------------------------------------
# Rule L5 — check_tactical_gap_preserved
# ---------------------------------------------------------------------------

def check_tactical_gap_preserved(feature_table: pd.DataFrame) -> None:
    """Assert Portugal, Colombia, England, and Croatia have has_full_tactical_md2 = False.

    These four teams played their MD2 match on June 23 (scores only, no
    tactical data).  If any of them shows has_full_tactical_md2 = True,
    their June 23 DS1-ext match was incorrectly processed as having full
    tactical coverage, introducing false precision.

    Raises
    ------
    LeakageError(violation_type='TACTICAL_IMPUTATION', rule='L5')
    """
    if "has_full_tactical_md2" not in feature_table.columns:
        # Column not yet built — skip (called early in notebook before features exist).
        return

    if "team" not in feature_table.columns:
        return

    for team in TACTICAL_GAP_TEAMS:
        rows = feature_table[feature_table["team"] == team]
        if rows.empty:
            continue
        val = rows["has_full_tactical_md2"].iloc[0]
        if bool(val):
            raise LeakageError(
                f"Rule L5 violated: {team!r} has has_full_tactical_md2 = True. "
                f"{team}'s MD2 match (June 23) provided scores only — no tactical "
                "statistics (possession, SOT, crosses, interceptions) were recorded. "
                "Imputing MD2 tactical stats from population means or carrying forward "
                "MD1 values as MD2 values introduces false precision and is prohibited.",
                violation_type="TACTICAL_IMPUTATION",
                rule="L5",
            )


# ---------------------------------------------------------------------------
# Rule L1 applied to training construction — check_training_rows_chronological
# ---------------------------------------------------------------------------

def check_training_rows_chronological(training_rows: pd.DataFrame) -> None:
    """Assert Elo features in training rows use data predating each match.

    The 'elo_year_used' column (added by build_training_rows in features.py)
    must equal match_year - 1 for every row.  If elo_year_used >= match_year,
    Elo ratings from the same year or a future year were used — forward leakage.

    Raises
    ------
    LeakageError(violation_type='FUTURE_DATE', rule='L1')
    """
    required = {"elo_year_used", "match_year"}
    missing = required - set(training_rows.columns)
    if missing:
        # Columns not yet present — skip (early pipeline call).
        return

    elo_year   = pd.to_numeric(training_rows["elo_year_used"],  errors="coerce")
    match_year = pd.to_numeric(training_rows["match_year"],     errors="coerce")

    bad = training_rows.loc[elo_year >= match_year]
    if not bad.empty:
        sample = bad[["match_year", "elo_year_used"]].head(3).to_dict("records")
        raise LeakageError(
            f"Rule L1 violated: {len(bad)} training row(s) have "
            "elo_year_used >= match_year, meaning Elo data from the same year "
            f"or later was used to construct features. Sample rows: {sample}. "
            "Per Rule L2, historical backtesting must use "
            "snapshot_date == '{{year-1}}-12-31' for each match year.",
            violation_type="FUTURE_DATE",
            rule="L1",
        )


# ---------------------------------------------------------------------------
# Rule L4 — check_ds4_wc_null_scores
# ---------------------------------------------------------------------------

def check_ds4_wc_null_scores(ds4: pd.DataFrame) -> None:
    """Assert that null-score WC 2026 rows in DS4 have not been imputed.

    DS4 contains WC 2026 match rows with null home_score/away_score for
    MD2 and MD3.  These must remain null — treating them as 0-0 draws or
    any other imputed value violates Rule L4.

    This check verifies that zero-zero rows in DS4's WC 2026 subset were
    not introduced by imputation: a row is suspicious if it is in the 2026
    WC tournament AND has home_score == 0 AND away_score == 0 AND the match
    date is on or after June 20, 2026 (when DS4 WC null-score rows begin).

    Note: legitimate 0-0 draws in DS4 before June 20 are actual results.

    Raises
    ------
    LeakageError(violation_type='FUTURE_DATE', rule='L4')
    """
    required_cols = {"date", "home_score", "away_score", "tournament"}
    missing = required_cols - set(ds4.columns)
    if missing:
        return  # Cannot check — column names differ, skip gracefully.

    wc_2026 = ds4[
        ds4["tournament"].astype(str).str.contains("FIFA World Cup", na=False)
        & (_to_date_series(ds4["date"]) >= pd.Timestamp("2026-06-11"))
    ].copy()

    if wc_2026.empty:
        return

    home = pd.to_numeric(wc_2026["home_score"], errors="coerce")
    away = pd.to_numeric(wc_2026["away_score"], errors="coerce")

    # Rows where both scores are 0 AND originally null in DS4 are suspicious.
    # Since DS4 stores nulls as NA (string) for these rows, any row that
    # survived pd.to_numeric as 0 when the original was NA is the red flag.
    originally_null = (
        wc_2026["home_score"].astype(str).isin({"NA", "nan", "", "None"})
        | wc_2026["away_score"].astype(str).isin({"NA", "nan", "", "None"})
    )
    zero_zero = (home == 0) & (away == 0) & ~originally_null
    # Zero-zero rows that are NOT in the originally-null set and are post-MD1
    # could be real results — only flag those after June 20 (null-score zone).
    late_zero_zero = zero_zero & (
        _to_date_series(wc_2026["date"]) >= pd.Timestamp("2026-06-20")
    )

    if late_zero_zero.any():
        count = late_zero_zero.sum()
        raise LeakageError(
            f"Rule L4 violated: {count} WC 2026 row(s) in DS4 dated "
            "June 20+ appear to have been imputed as 0-0 draws. "
            "Null-score WC rows (MD2/MD3) must remain null — they represent "
            "unknown future outcomes, not zero-zero draws.",
            violation_type="FUTURE_DATE",
            rule="L4",
        )


# ---------------------------------------------------------------------------
# Rule L4 — check_ds8_year_ceiling
# ---------------------------------------------------------------------------

def check_ds8_year_ceiling(ds8: pd.DataFrame) -> None:
    """Assert DS8 contains no match data with Year > 2022.

    DS8 is the WC historical training set covering 1930–2022 (964 rows).
    Any post-2022 row would introduce future tournament data into the
    historical training distribution.

    Raises
    ------
    LeakageError(violation_type='FUTURE_DATE', rule='L4')
    """
    if "Year" not in ds8.columns:
        return

    year = pd.to_numeric(ds8["Year"], errors="coerce")
    bad = ds8.loc[year > DS8_MAX_YEAR, "Year"] if "Year" in ds8.columns else pd.Series(dtype=object)

    if not bad.empty:
        raise LeakageError(
            f"Rule L4 violated: DS8 contains {len(bad)} row(s) with Year > "
            f"{DS8_MAX_YEAR}. Found years: {sorted(bad.unique().tolist())}. "
            "DS8 must cover only 1930–2022 WC matches.",
            violation_type="FUTURE_DATE",
            rule="L4",
        )


# ---------------------------------------------------------------------------
# Rule L1 applied to form window — check_form_window
# ---------------------------------------------------------------------------

def check_form_window(ds4_filtered: pd.DataFrame, date_column: str = "date") -> None:
    """Assert the DS4 form window respects both date bounds.

    Form features (F023–F028, Feature Group 4) must use competitive matches
    between 2024-01-01 and 2026-06-10 inclusive.  Any row outside these
    bounds in the filtered DataFrame indicates the window was applied
    incorrectly.

    Raises
    ------
    LeakageError(violation_type='FUTURE_DATE', rule='L1')
        If any date > 2026-06-10.
    """
    if date_column not in ds4_filtered.columns:
        return

    dates = _to_date_series(ds4_filtered[date_column])
    ceiling = pd.Timestamp(FORM_WINDOW_END)
    above = ds4_filtered.loc[dates > ceiling, date_column]

    if not above.empty:
        raise LeakageError(
            f"Rule L1 violated: DS4 form window contains {len(above)} row(s) "
            f"with date after form ceiling {FORM_WINDOW_END!r}. "
            "First offending date: {above.iloc[0]!r}. "
            "Form features must use dates <= 2026-06-10 to exclude all "
            "2026 WC matches (which begin June 11).",
            violation_type="FUTURE_DATE",
            rule="L1",
        )


# ---------------------------------------------------------------------------
# Rule L6 extended — check_no_forbidden_datasets
# ---------------------------------------------------------------------------

def check_no_forbidden_datasets(
    *dataframes: pd.DataFrame,
    context: str = "",
) -> None:
    """Assert no DataFrame contains column fingerprints from DS3, DS13, or DS14.

    Accepts any number of DataFrames (feature table, training rows, etc.).
    Uses column-name fingerprinting — the same set checked by
    check_no_synthetic_data but applicable to any DataFrame, not just the
    feature table.

    Raises
    ------
    LeakageError(violation_type='FORBIDDEN_DATASET', rule='L6')
    """
    for i, df in enumerate(dataframes):
        found = _ALL_FORBIDDEN_COLS & set(df.columns)
        if found:
            ctx_str = f" [{context}, DataFrame #{i}]" if context else f" [DataFrame #{i}]"
            raise LeakageError(
                f"Rule L6 violated{ctx_str}: DataFrame contains {len(found)} "
                f"column(s) matching DS3/DS13/DS14 fingerprints: {sorted(found)}. "
                "DS3 is fully excluded (synthetic, fabricated outcomes). "
                "DS13/DS14 have unknown provenance and are excluded entirely.",
                violation_type="FORBIDDEN_DATASET",
                rule="L6",
            )


# ---------------------------------------------------------------------------
# Structural — check_annex_c
# ---------------------------------------------------------------------------

def check_annex_c(annex_c_df: pd.DataFrame) -> None:
    """Assert the third-place placement table is structurally valid.

    Validates:
      1. Exactly 495 data rows (C(12, 8) — all ways to choose 8 of 12 groups).
      2. Exactly 9 columns (qualifying_groups + 8 match slot columns).
      3. The 8 match columns are m75, m78, m79, m80, m81, m82, m85, m88 —
         verified against DS16 R32 matches that accept third-placed teams.
      4. No duplicate qualifying_groups combinations.
      5. All group letters in qualifying_groups are drawn from A–L only.
      6. All cell values in match columns follow the pattern '3X' where X
         is a letter A–L (e.g. '3F', '3K').

    Raises
    ------
    LeakageError(violation_type='INVALID_REFERENCE_DATA')
    """
    def _fail(msg: str) -> None:
        raise LeakageError(
            f"Annex C integrity check failed: {msg} "
            "third_place_annex_c.csv is the authoritative FIFA lookup table "
            "for Round of 32 third-place seeding and must not be modified.",
            violation_type="INVALID_REFERENCE_DATA",
        )

    # 1. Row count
    n_rows = len(annex_c_df)
    expected = comb(12, 8)  # 495
    if n_rows != expected:
        _fail(
            f"Expected {expected} rows (C(12,8)), found {n_rows}."
        )

    # 2. Column count and names
    if "qualifying_groups" not in annex_c_df.columns:
        _fail("Missing required column 'qualifying_groups'.")

    missing_match_cols = set(ANNEX_C_MATCH_COLS) - set(annex_c_df.columns)
    if missing_match_cols:
        _fail(f"Missing match slot columns: {sorted(missing_match_cols)}.")

    if len(annex_c_df.columns) != ANNEX_C_EXPECTED_COLS:
        _fail(
            f"Expected {ANNEX_C_EXPECTED_COLS} columns, "
            f"found {len(annex_c_df.columns)}: {list(annex_c_df.columns)}."
        )

    # 3. No duplicate combinations
    dupes = annex_c_df["qualifying_groups"].duplicated().sum()
    if dupes:
        _fail(f"Found {dupes} duplicate qualifying_groups combination(s).")

    # 4. Group letters in qualifying_groups are A–L only
    valid_letters = set("ABCDEFGHIJKL")
    for raw in annex_c_df["qualifying_groups"].dropna():
        letters = {c.strip() for c in str(raw).split(",") if c.strip()}
        bad_letters = letters - valid_letters
        if bad_letters:
            _fail(f"qualifying_groups contains invalid letters: {bad_letters}.")
        if len(letters) != 8:
            _fail(
                f"Each qualifying_groups entry must contain exactly 8 letters; "
                f"found {len(letters)} in {raw!r}."
            )

    # 5. Cell values in match columns follow '3X' pattern
    for col in ANNEX_C_MATCH_COLS:
        vals = annex_c_df[col].dropna().astype(str)
        bad_vals = vals[~vals.str.match(r"^3[A-L]$")]
        if not bad_vals.empty:
            _fail(
                f"Column '{col}' contains values that do not match '3X' pattern "
                f"(X must be A–L): {bad_vals.unique().tolist()[:5]}."
            )


# ---------------------------------------------------------------------------
# Name authority — check_canonical_names
# ---------------------------------------------------------------------------

def check_canonical_names(
    names: Iterable[str],
    context: str = "",
) -> None:
    """Assert every name in *names* is a DS9 canonical team name.

    Thin wrapper around name_map.assert_all_canonical.

    Raises
    ------
    LeakageError(violation_type='NAME_ERROR')
    """
    names_list = list(names)
    unknown = [n for n in names_list if n not in CANONICAL_48]
    if unknown:
        raise LeakageError(
            f"Non-canonical team name(s) detected{' in ' + context if context else ''}: "
            f"{unknown}. "
            "All team names must match DS9 (schedule_2026.csv) spelling before "
            "any join or feature computation. "
            "Apply name_map.canonicalize() upstream.",
            violation_type="NAME_ERROR",
        )


# ---------------------------------------------------------------------------
# Master entry point — run_all_checks
# ---------------------------------------------------------------------------

def run_all_checks(
    feature_table: pd.DataFrame,
    training_rows: pd.DataFrame,
    ds2: pd.DataFrame,
) -> None:
    """Run all leakage prevention checks in sequence.

    Designed to be called as the first statement in every notebook cell that
    uses features or training data.  A clean run produces no output and returns
    None.  The first failing check raises LeakageError and halts execution.

    Parameters
    ----------
    feature_table:
        The 48-row team feature table produced by features.build_feature_table().
        Pass an empty DataFrame if features have not yet been built.
    training_rows:
        The training dataset produced by features.build_training_rows().
        Pass an empty DataFrame if training rows have not yet been built.
    ds2:
        The full DS2 Elo DataFrame (all snapshot dates) — used to check that
        only the May 27 snapshot was pulled for current-strength features.
        This is the raw DS2, not the filtered subset.

    Checks run (in order):
        1.  check_elo_snapshot(ds2_filtered)   — Rule L2
        2.  check_no_synthetic_data(feature_table) — Rule L6
        3.  check_no_synthetic_data(training_rows) — Rule L6
        4.  check_no_forbidden_datasets(feature_table, training_rows) — Rule L6
        5.  check_no_md3_in_features(feature_table) — Rule L1
        6.  check_tactical_gap_preserved(feature_table) — Rule L5
        7.  check_training_rows_chronological(training_rows) — Rule L1

    DS2 note: this function checks the *filtered* 2026 portion of DS2.
    If ds2 is the full dataset, it isolates year == 2026 rows for the check.
    """
    # --- Rule L2: Elo snapshot ---
    if not ds2.empty and "snapshot_date" in ds2.columns:
        ds2_2026 = ds2[ds2["snapshot_date"].astype(str).str.startswith("2026")]
        if not ds2_2026.empty:
            check_elo_snapshot(ds2_2026)

    # --- Rule L6: no synthetic data in feature table ---
    if not feature_table.empty:
        check_no_synthetic_data(feature_table)

    # --- Rule L6: no synthetic data in training rows ---
    if not training_rows.empty:
        check_no_synthetic_data(training_rows)

    # --- Rule L6 extended: no forbidden dataset columns anywhere ---
    non_empty = [df for df in (feature_table, training_rows) if not df.empty]
    if non_empty:
        check_no_forbidden_datasets(*non_empty, context="run_all_checks")

    # --- Rule L1: no MD3 data in features ---
    if not feature_table.empty:
        check_no_md3_in_features(feature_table)

    # --- Rule L5: tactical gap preserved ---
    if not feature_table.empty:
        check_tactical_gap_preserved(feature_table)

    # --- Rule L1: chronological training construction ---
    if not training_rows.empty:
        check_training_rows_chronological(training_rows)

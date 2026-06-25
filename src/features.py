"""
Feature engineering for the FIFA WC 2026 Forecast Engine.

Frozen snapshot: June 23, 2026 — Post-Matchday 2
All features are computed from datasets locked at or before this date.

Feature groups implemented:
  Group 1  F001–F007   Pre-tournament Elo strength              (DS2 @ 2026-05-27)
  Group 2  F008–F013   FIFA ranking and momentum                (DS10, DS11)
  Group 3  F014–F022   WC historical performance (era-filtered) (DS8, DS6)
  Group 4  F023–F028   Recent competitive form                  (DS4 2024-01-01—2026-06-10)
  Group 5  F029–F039   2026 in-tournament MD1+MD2              (DS1, DS1-ext)
  Group 6  F040–F043   Penalty shootout history                 (DS6, DS8)
  Group 7  F044–F050   Match context                            (DS16–DS19, DS2)
  Matchup               Opponent-relative delta features        (derived at match level)

Public API:
  load_ds*(path)                  — raw dataset loaders with name canonicalization
  build_elo_features(ds2)         — Group 1 team-level table
  build_fifa_features(ds10, ds11, ds2_elo_ranks)  — Group 2
  build_wc_historical_features(ds8, ds6)           — Group 3
  build_form_features(ds4)        — Group 4
  build_tournament_features(ds1, ds1ext)            — Group 5
  build_penalty_features(ds6, ds8)                  — Group 6
  build_team_features(...)        — merge Groups 1–6 into 48-row team table
  build_feature_table(...)        — full 96-row (48 matches × 2 perspectives) table
  build_training_rows(...)        — historical WC match rows for Layer 1 training
"""

from __future__ import annotations

import io
import zipfile
from math import log10
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.name_map import (
    CANONICAL_48,
    WC_DEBUTANTS,
    assert_all_canonical,
    canonicalize,
    canonicalize_id,
    apply_to_series,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELO_SNAPSHOT         = "2026-05-27"
FIFA_SNAPSHOT        = "2026-06-08"
FORM_START           = "2024-01-01"
FORM_END             = "2026-06-10"       # day before tournament
TOURNAMENT_START     = "2026-06-11"
FREEZE_DATE          = "2026-06-23"
WC_MODERN_START      = 1998
FORM_WINDOW_MATCHES  = 10
SHOOTOUT_SHRINKAGE_K = 8                  # Bayesian shrinkage parameter
HOST_TEAMS: frozenset[str] = frozenset({"United States", "Canada", "Mexico"})

# DS8 Round → numeric stage level (excluding special Finals/3rd-place)
_DS8_ROUND_LEVEL: dict[str, int] = {
    "Group stage":           1,
    "First round":           1,
    "First group stage":     1,
    "Second group stage":    1,
    "Group stage play-off":  1,
    "Final stage":           1,
    "Second round":          2,
    "Round of 16":           2,
    "Quarter-finals":        3,
    "Semi-finals":           4,
}
_DS8_KNOCKOUT_ROUNDS: frozenset[str] = frozenset({
    "Round of 16", "Second round", "Quarter-finals",
    "Semi-finals", "Third-place match", "Final",
})
_DS8_GROUP_ROUNDS: frozenset[str] = frozenset({
    "Group stage", "First round", "First group stage",
    "Second group stage", "Group stage play-off", "Final stage",
})

# Outcome encoding used throughout
WIN, DRAW, LOSS = 2, 1, 0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _read_csv_from_zip(zip_path: str | Path, filename: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(filename) as f:
            return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))


def _match_outcome(home_score: int | float, away_score: int | float) -> int | None:
    """Return WIN/DRAW/LOSS from home team perspective; None if scores invalid."""
    try:
        h, a = int(home_score), int(away_score)
    except (ValueError, TypeError):
        return None
    if h > a:
        return WIN
    if h < a:
        return LOSS
    return DRAW


def _team_outcome(team_is_home: bool, home_score, away_score) -> int | None:
    r = _match_outcome(home_score, away_score)
    if r is None:
        return None
    if team_is_home:
        return r
    return {WIN: LOSS, DRAW: DRAW, LOSS: WIN}[r]


def _elo_win_expectancy(team_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - team_elo) / 400.0))


# ---------------------------------------------------------------------------
# Data loaders — each returns a DataFrame with team name columns canonicalized
# ---------------------------------------------------------------------------

def load_ds2(zip_path: str | Path) -> pd.DataFrame:
    """DS2 — Elo ratings. Filter to snapshot_date = 2026-05-27 for current use.
    Returns full dataset; caller filters by snapshot_date as needed."""
    df = _read_csv_from_zip(zip_path, "elo_ratings_wc2026.csv")
    df["country"] = apply_to_series(df["country"])
    df["is_host"] = _to_numeric(df["is_host"]).fillna(0).astype(int)
    df["rating"]     = _to_numeric(df["rating"])
    df["rating_max"] = _to_numeric(df["rating_max"])
    df["rating_avg"] = _to_numeric(df["rating_avg"])
    df["rank"]       = _to_numeric(df["rank"])
    return df


def load_ds4(zip_path: str | Path) -> pd.DataFrame:
    """DS4 — International match results 1872-present."""
    df = _read_csv_from_zip(zip_path, "results.csv")
    df["home_team"] = apply_to_series(df["home_team"])
    df["away_team"] = apply_to_series(df["away_team"])
    df["home_score"] = _to_numeric(df["home_score"])
    df["away_score"] = _to_numeric(df["away_score"])
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["neutral"]    = df["neutral"].astype(str).str.upper() == "TRUE"
    return df


def load_ds6(zip_path: str | Path) -> pd.DataFrame:
    """DS6 — Penalty shootout records."""
    df = _read_csv_from_zip(zip_path, "shootouts.csv")
    df["home_team"] = apply_to_series(df["home_team"])
    df["away_team"] = apply_to_series(df["away_team"])
    df["winner"]    = apply_to_series(df["winner"])
    df["date"]      = pd.to_datetime(df["date"], errors="coerce")
    df["first_shooter"] = df["first_shooter"].where(df["first_shooter"].notna() & (df["first_shooter"] != ""), other=pd.NA)
    return df


def load_ds8(zip_path: str | Path) -> pd.DataFrame:
    """DS8 — WC historical matches 1930-2022."""
    df = _read_csv_from_zip(zip_path, "matches_1930_2022.csv")
    df["home_canonical"] = apply_to_series(df["home_team"])
    df["away_canonical"] = apply_to_series(df["away_team"])
    df["home_score"]     = _to_numeric(df["home_score"])
    df["away_score"]     = _to_numeric(df["away_score"])
    df["home_penalty"]   = _to_numeric(df["home_penalty"])
    df["away_penalty"]   = _to_numeric(df["away_penalty"])
    df["Year"]           = _to_numeric(df["Year"])
    df["Date"]           = pd.to_datetime(df["Date"], errors="coerce")
    return df


def load_ds10(zip_path: str | Path) -> pd.DataFrame:
    """DS10 — FIFA ranking June 8, 2026."""
    df = _read_csv_from_zip(zip_path, "fifa_ranking_2026-06-08.csv")
    df["team_canonical"] = apply_to_series(df["team"])
    df["points"]         = _to_numeric(df["points"])
    df["previous_points"]= _to_numeric(df["previous_points"])
    df["rank"]           = _to_numeric(df["rank"])
    df["previous_rank"]  = _to_numeric(df["previous_rank"])
    return df


def load_ds11(zip_path: str | Path) -> pd.DataFrame:
    """DS11 — FIFA ranking October 6, 2022."""
    df = _read_csv_from_zip(zip_path, "fifa_ranking_2022-10-06.csv")
    df["team_canonical"] = apply_to_series(df["team"])
    df["points"]         = _to_numeric(df["points"])
    df["rank"]           = _to_numeric(df["rank"])
    return df


def load_ds16(zip_path: str | Path) -> pd.DataFrame:
    """DS16 — Full 2026 bracket structure (Arc_base matches.csv)."""
    df = _read_csv_from_zip(zip_path, "matches.csv")
    df["id"]            = _to_numeric(df["id"])
    df["home_team_id"]  = _to_numeric(df["home_team_id"])
    df["away_team_id"]  = _to_numeric(df["away_team_id"])
    df["city_id"]       = _to_numeric(df["city_id"])
    df["stage_id"]      = _to_numeric(df["stage_id"])
    # Parse kickoff; the timestamp includes UTC offset — .hour gives local hour
    df["kickoff_at"]    = pd.to_datetime(df["kickoff_at"], errors="coerce", utc=False)
    return df


def load_ds17(zip_path: str | Path) -> pd.DataFrame:
    """DS17 — Team ID mapping with placeholder resolution."""
    df = _read_csv_from_zip(zip_path, "teams.csv")
    df["id"]            = _to_numeric(df["id"])
    df["is_placeholder"]= df["is_placeholder"].astype(str).str.lower() == "true"
    # Resolve canonical names: placeholders via ID, real teams via name map
    df["team_canonical"] = df.apply(
        lambda r: canonicalize_id(int(r["id"]), r["team_name"]), axis=1
    )
    return df


def load_ds18(zip_path: str | Path) -> pd.DataFrame:
    """DS18 — Host cities."""
    df = _read_csv_from_zip(zip_path, "host_cities.csv")
    df["id"] = _to_numeric(df["id"])
    return df


def load_ds19(zip_path: str | Path) -> pd.DataFrame:
    """DS19 — Tournament stages."""
    df = _read_csv_from_zip(zip_path, "tournament_stages.csv")
    df["id"]          = _to_numeric(df["id"])
    df["stage_order"] = _to_numeric(df["stage_order"])
    return df


def load_ds1(zip_path: str | Path) -> pd.DataFrame:
    """DS1 — Arc3 matches.csv: 44 WC 2026 matches MD1+MD2 with tactical data."""
    df = _read_csv_from_zip(zip_path, "matches.csv")
    df["home_team"] = apply_to_series(df["home_team"])
    df["away_team"] = apply_to_series(df["away_team"])
    df["home_score"]      = _to_numeric(df["home_score"])
    df["away_score"]      = _to_numeric(df["away_score"])
    df["home_possession"] = _to_numeric(df["home_possession"])
    df["away_possession"] = _to_numeric(df["away_possession"])
    df["home_sot"]        = _to_numeric(df["home_sot"])
    df["away_sot"]        = _to_numeric(df["away_sot"])
    df["home_total_shots"]= _to_numeric(df["home_total_shots"])
    df["away_total_shots"]= _to_numeric(df["away_total_shots"])
    df["home_cards_yellow"]= _to_numeric(df["home_cards_yellow"])
    df["away_cards_yellow"]= _to_numeric(df["away_cards_yellow"])
    df["home_cards_red"]  = _to_numeric(df["home_cards_red"])
    df["away_cards_red"]  = _to_numeric(df["away_cards_red"])
    df["date"]            = pd.to_datetime(df["date"], errors="coerce")
    df["gameweek"]        = _to_numeric(df["gameweek"])
    # Verify freeze: DS1 must not have any June 23 matches
    if (df["date"] > pd.Timestamp(FREEZE_DATE)).any():
        raise ValueError("DS1 contains rows after freeze date. Check data source.")
    return df


def load_ds1ext(csv_path: str | Path) -> pd.DataFrame:
    """DS1-ext — june23_results.csv: 4 June 23 scores, no tactical data."""
    df = pd.read_csv(csv_path)
    df["home_team"] = apply_to_series(df["home_team"])
    df["away_team"] = apply_to_series(df["away_team"])
    df["home_score"] = _to_numeric(df["home_score"])
    df["away_score"] = _to_numeric(df["away_score"])
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["gameweek"]   = _to_numeric(df["gameweek"])
    if len(df) != 4:
        raise ValueError(f"DS1-ext must have exactly 4 rows, found {len(df)}.")
    return df


# ---------------------------------------------------------------------------
# Group 1 — Pre-tournament Elo features
# Features: F001–F007
# Source: DS2 @ snapshot_date == '2026-05-27'
# ---------------------------------------------------------------------------

def build_elo_features(ds2: pd.DataFrame) -> pd.DataFrame:
    """Return 48-row DataFrame with Elo features (F001–F007).

    Index: team_canonical (DS9 authority).
    """
    snap = ds2[ds2["snapshot_date"] == ELO_SNAPSHOT].copy()
    # Restrict to the 48 WC teams
    snap = snap[snap["country"].isin(CANONICAL_48)].copy()
    snap = snap.set_index("country")

    if len(snap) != 48:
        missing = CANONICAL_48 - set(snap.index)
        raise ValueError(f"DS2 @ {ELO_SNAPSHOT} missing {len(missing)} teams: {sorted(missing)}")

    # Compute WC-field Elo ranks (rank within 48 teams, 1 = highest rating)
    snap["elo_rank_in_wc_field"] = snap["rating"].rank(ascending=False, method="min").astype(int)

    out = pd.DataFrame(index=snap.index)
    out.index.name = "team_canonical"
    out["elo_rating"]            = snap["rating"]                  # F001
    out["elo_rank"]              = snap["rank"]                    # F006 (global rank)
    out["elo_rating_career_peak"]= snap["rating_max"]              # F004
    out["elo_rating_career_avg"] = snap["rating_avg"]              # F005
    out["elo_is_host"]           = snap["is_host"].astype(int)     # F007
    out["confederation"]         = snap["confederation"]           # used in Group 2 & 7
    out["elo_rank_in_wc_field"]  = snap["elo_rank_in_wc_field"]   # needed for F013

    # F002 and F003 are matchup-level (opponent-relative); added later.
    return out.copy()


# ---------------------------------------------------------------------------
# Group 2 — FIFA ranking features
# Features: F008–F013
# Source: DS10 (current June 8), DS11 (2022 baseline), DS2 (Elo ranks for F013)
# ---------------------------------------------------------------------------

def build_fifa_features(
    ds10: pd.DataFrame,
    ds11: pd.DataFrame,
    elo_features: pd.DataFrame,
) -> pd.DataFrame:
    """Return 48-row DataFrame with FIFA features (F008–F013).

    Parameters
    ----------
    elo_features : output of build_elo_features — provides Elo ranks for F013.
    """
    # Filter to WC 48 teams
    ds10_wc = ds10[ds10["team_canonical"].isin(CANONICAL_48)].copy()
    ds10_wc = ds10_wc.set_index("team_canonical")

    if len(ds10_wc) != 48:
        missing = CANONICAL_48 - set(ds10_wc.index)
        raise ValueError(f"DS10 missing {len(missing)} WC teams: {sorted(missing)}")

    # DS11 indexed by canonical name (may not contain all 48 if some are new federations)
    ds11_idx = ds11[ds11["team_canonical"].isin(CANONICAL_48)].set_index("team_canonical")

    # Compute WC-field FIFA rank (rank within 48, for F013)
    ds10_wc["fifa_rank_in_wc_field"] = ds10_wc["points"].rank(ascending=False, method="min").astype(int)

    out = pd.DataFrame(index=ds10_wc.index)
    out.index.name = "team_canonical"
    out["fifa_points"]       = ds10_wc["points"]                   # F008
    out["fifa_points_delta"] = ds10_wc["points"] - ds10_wc["previous_points"]  # F009
    out["fifa_rank_delta"]   = ds10_wc["previous_rank"] - ds10_wc["rank"]      # F010 (pos = improving)
    out["fifa_rank_in_wc_field"] = ds10_wc["fifa_rank_in_wc_field"]

    # F011, F012 — four-year trajectory (null if team not in DS11)
    out["fifa_points_4yr_change"] = (
        ds10_wc["points"] - ds11_idx["points"].reindex(ds10_wc.index)
    )
    out["fifa_rank_4yr_change"] = (
        ds11_idx["rank"].reindex(ds10_wc.index) - ds10_wc["rank"]
    )

    # F013 — disagreement between Elo WC rank and FIFA WC rank (within 48-team field)
    elo_rank_wc = elo_features["elo_rank_in_wc_field"].reindex(out.index)
    out["elo_fifa_rank_disagreement"] = (
        (elo_rank_wc - out["fifa_rank_in_wc_field"]).abs()
    )

    return out.copy()


# ---------------------------------------------------------------------------
# Group 3 — WC historical features (era-filtered)
# Features: F014–F022  (+ wc_debut_modern_flag)
# Source: DS8 (1930–2022) filtered to Year >= 1998 for modern-era metrics
# ---------------------------------------------------------------------------

def _wc_best_result_for_team(team: str, team_matches: pd.DataFrame) -> int:
    """Return the best WC finishing position encoded 1–7 for a single team."""
    best = 0
    for year, yr in team_matches.groupby("Year"):
        score = _wc_year_result(team, yr)
        if score > best:
            best = score
    return best


def _wc_year_result(team: str, yr_matches: pd.DataFrame) -> int:
    """Return the encoded finishing position for one team in one WC year."""
    # Final → 7 (winner) or 6 (runner-up)
    finals = yr_matches[yr_matches["Round"] == "Final"]
    if not finals.empty:
        m = finals.iloc[0]
        if m["home_canonical"] == team:
            return WIN if _match_outcome(m["home_score"], m["away_score"]) == WIN else 6
        else:
            return WIN if _match_outcome(m["home_score"], m["away_score"]) == LOSS else 6

    # Third-place match → 5 (winner) or 4 (loser)
    third = yr_matches[yr_matches["Round"] == "Third-place match"]
    if not third.empty:
        m = third.iloc[0]
        if m["home_canonical"] == team:
            return 5 if _match_outcome(m["home_score"], m["away_score"]) == WIN else 4
        else:
            return 5 if _match_outcome(m["home_score"], m["away_score"]) == LOSS else 4

    # Highest round reached
    best = 0
    for r in yr_matches["Round"].unique():
        lvl = _DS8_ROUND_LEVEL.get(r, 0)
        if lvl > best:
            best = lvl
    return best


def build_wc_historical_features(
    ds8: pd.DataFrame,
    ds6: pd.DataFrame,
    through_year: int = 2022,
) -> pd.DataFrame:
    """Return 48-row DataFrame with WC historical features (F014–F022).

    Parameters
    ----------
    through_year : include DS8 matches with Year <= through_year.
                   Default 2022 for 2026 prediction. For training, set to year-1.
    """
    # Restrict to years up through the cutoff
    ds8 = ds8[ds8["Year"] <= through_year].copy()

    out_rows: list[dict] = []
    for team in sorted(CANONICAL_48):
        team_mask = (ds8["home_canonical"] == team) | (ds8["away_canonical"] == team)
        all_matches = ds8[team_mask].copy()

        # WC tournaments attended (any year, any round)
        years_attended = all_matches["Year"].dropna().unique()
        wc_tournaments_attended = len(years_attended)

        # Modern era: 1998+
        modern = all_matches[all_matches["Year"] >= WC_MODERN_START].copy()
        has_modern = len(modern) > 0

        # Flags for group vs knockout rows
        modern_grp = modern[modern["Round"].isin(_DS8_GROUP_ROUNDS)]
        modern_ko  = modern[modern["Round"].isin(_DS8_KNOCKOUT_ROUNDS)]

        def _team_stats(match_df: pd.DataFrame) -> dict:
            """Compute W/D/L, GF, GA, CS for this team from a set of matches."""
            w = d = l = gf = ga = cs = 0
            for _, m in match_df.iterrows():
                is_home = m["home_canonical"] == team
                h, a = m["home_score"], m["away_score"]
                try:
                    h, a = int(h), int(a)
                except (ValueError, TypeError):
                    continue
                team_goals  = h if is_home else a
                opp_goals   = a if is_home else h
                gf += team_goals; ga += opp_goals
                if team_goals > opp_goals:   w += 1
                elif team_goals == opp_goals: d += 1
                else:                         l += 1
                if opp_goals == 0:            cs += 1
            return {"w": w, "d": d, "l": l, "gf": gf, "ga": ga, "cs": cs}

        total = _team_stats(modern)
        total_mp = total["w"] + total["d"] + total["l"]
        ko_s = _team_stats(modern_ko)
        ko_mp = ko_s["w"] + ko_s["d"] + ko_s["l"]
        grp_s = _team_stats(modern_grp)
        grp_mp = grp_s["w"] + grp_s["d"] + grp_s["l"]

        if total_mp > 0:
            wc_win_rate_modern         = total["w"] / total_mp
            wc_avg_gf_modern           = total["gf"] / total_mp
            wc_avg_ga_modern           = total["ga"] / total_mp
            wc_clean_sheet_rate_modern = total["cs"] / total_mp
            wc_gd_per_game_modern      = (total["gf"] - total["ga"]) / total_mp
        else:
            # WC debutant — all None (missing-class encoding)
            wc_win_rate_modern = wc_avg_gf_modern = wc_avg_ga_modern = None
            wc_clean_sheet_rate_modern = wc_gd_per_game_modern = None

        wc_win_rate_knockout_modern = (
            ko_s["w"] / ko_mp if ko_mp > 0 else None
        )

        grp_win_rate = (grp_s["w"] / grp_mp) if grp_mp > 0 else None
        ko_win_rate  = wc_win_rate_knockout_modern
        if grp_win_rate is not None and ko_win_rate is not None:
            wc_group_vs_knockout_uplift = ko_win_rate - grp_win_rate
        else:
            wc_group_vs_knockout_uplift = None

        # Best result encoded
        wc_best_result_encoded = _wc_best_result_for_team(team, all_matches)

        out_rows.append({
            "team_canonical":              team,
            "wc_tournaments_attended":     wc_tournaments_attended,   # F020
            "wc_win_rate_modern":          wc_win_rate_modern,        # F014
            "wc_win_rate_knockout_modern": wc_win_rate_knockout_modern,# F015
            "wc_avg_gf_modern":            wc_avg_gf_modern,          # F016
            "wc_avg_ga_modern":            wc_avg_ga_modern,          # F017
            "wc_gd_per_game_modern":       wc_gd_per_game_modern,     # F018
            "wc_clean_sheet_rate_modern":  wc_clean_sheet_rate_modern,# F019
            "wc_best_result_encoded":      wc_best_result_encoded,    # F021
            "wc_group_vs_knockout_uplift": wc_group_vs_knockout_uplift,# F022
            "wc_debut_modern_flag":        int(not has_modern),       # flag
        })

    out = pd.DataFrame(out_rows).set_index("team_canonical")
    out.index.name = "team_canonical"
    return out


# ---------------------------------------------------------------------------
# Group 4 — Recent competitive form
# Features: F023–F028
# Source: DS4 filtered to non-Friendly AND 2024-01-01 ≤ date ≤ 2026-06-10
# ---------------------------------------------------------------------------

def build_form_features(ds4: pd.DataFrame) -> pd.DataFrame:
    """Return 48-row DataFrame with recent competitive form features (F023–F028)."""
    form_start = pd.Timestamp(FORM_START)
    form_end   = pd.Timestamp(FORM_END)

    # Filter: non-friendly, within window, scores available
    form_df = ds4[
        (ds4["tournament"].str.strip() != "Friendly")
        & (ds4["date"] >= form_start)
        & (ds4["date"] <= form_end)
        & ds4["home_score"].notna()
        & ds4["away_score"].notna()
    ].copy()

    # For unbeaten streak: extend lookback to all competitive history (any year)
    streak_df = ds4[
        (ds4["tournament"].str.strip() != "Friendly")
        & (ds4["date"] < pd.Timestamp(TOURNAMENT_START))
        & ds4["home_score"].notna()
        & ds4["away_score"].notna()
    ].copy()

    out_rows: list[dict] = []
    for team in sorted(CANONICAL_48):
        # Build per-match rows from team's perspective within the form window
        home_m = form_df[form_df["home_team"] == team][
            ["date", "home_score", "away_score"]
        ].rename(columns={"home_score": "gf", "away_score": "ga"}).assign(is_home=True)
        away_m = form_df[form_df["away_team"] == team][
            ["date", "home_score", "away_score"]
        ].rename(columns={"away_score": "gf", "home_score": "ga"}).assign(is_home=False)

        matches = pd.concat([home_m, away_m], ignore_index=True).sort_values("date")
        # Last 10 (or fewer) competitive matches
        last10 = matches.tail(FORM_WINDOW_MATCHES)

        if len(last10) == 0:
            out_rows.append({
                "team_canonical":              team,
                "form_win_rate_last10":        0.0,
                "form_avg_gf_last10":          0.0,
                "form_avg_ga_last10":          0.0,
                "form_gd_last10":              0.0,
                "form_clean_sheet_rate_last10":0.0,
                "form_unbeaten_streak_entering":0,
            })
            continue

        n = len(last10)
        gf = last10["gf"].astype(float)
        ga = last10["ga"].astype(float)
        wins   = ((gf - ga) > 0).sum()
        draws  = ((gf - ga) == 0).sum()
        cs     = (ga == 0).sum()

        # Unbeaten streak: consecutive non-losses ending at June 10, 2026
        streak_home = streak_df[streak_df["home_team"] == team][
            ["date", "home_score", "away_score"]
        ].rename(columns={"home_score": "gf", "away_score": "ga"})
        streak_away = streak_df[streak_df["away_team"] == team][
            ["date", "home_score", "away_score"]
        ].rename(columns={"away_score": "gf", "home_score": "ga"})
        streak_all = pd.concat([streak_home, streak_away]).sort_values("date", ascending=False)
        streak_count = 0
        for _, row in streak_all.iterrows():
            try:
                diff = float(row["gf"]) - float(row["ga"])
            except (ValueError, TypeError):
                break
            if diff < 0:
                break
            streak_count += 1

        out_rows.append({
            "team_canonical":              team,
            "form_win_rate_last10":        float(wins) / n,           # F023
            "form_avg_gf_last10":          float(gf.sum()) / n,       # F024
            "form_avg_ga_last10":          float(ga.sum()) / n,       # F025
            "form_gd_last10":              float((gf - ga).sum()) / n,# F026
            "form_clean_sheet_rate_last10":float(cs) / n,             # F027
            "form_unbeaten_streak_entering":int(streak_count),        # F028
        })

    out = pd.DataFrame(out_rows).set_index("team_canonical")
    out.index.name = "team_canonical"
    return out


# ---------------------------------------------------------------------------
# Group 5 — 2026 in-tournament features (MD1 + MD2)
# Features: F029–F039
# Source: DS1 (44 matches with full tactical data) + DS1-ext (4 June 23 scores)
# ---------------------------------------------------------------------------

_JUNE23_TEAMS: frozenset[str] = frozenset({
    "Portugal", "Colombia", "England", "Croatia"
})


def build_tournament_features(
    ds1: pd.DataFrame,
    ds1ext: pd.DataFrame,
) -> pd.DataFrame:
    """Return 48-row DataFrame with in-tournament features (F029–F039).

    DS1  — 44 matches through June 22 with full tactical stats.
    DS1-ext — 4 June 23 matches (scores only, tactical columns absent).
    """
    # Combine DS1 and DS1-ext, aligning on shared columns
    shared_cols = ["date", "home_team", "away_team", "home_score", "away_score",
                   "home_cards_yellow", "away_cards_yellow", "gameweek"]
    tactical_cols = ["home_possession", "away_possession", "home_sot", "away_sot",
                     "home_total_shots", "away_total_shots", "home_formation",
                     "away_formation"]

    ds1_shared = ds1[shared_cols + tactical_cols].copy()
    # DS1-ext has no tactical columns → set them to NaN
    ds1ext_ext = ds1ext[shared_cols].copy()
    for col in tactical_cols:
        ds1ext_ext[col] = np.nan

    # yellow cards may not be in DS1-ext
    if "home_cards_yellow" not in ds1ext.columns:
        ds1ext_ext["home_cards_yellow"] = np.nan
        ds1ext_ext["away_cards_yellow"] = np.nan

    all_matches = pd.concat([ds1_shared, ds1ext_ext], ignore_index=True)

    out_rows: list[dict] = []
    for team in sorted(CANONICAL_48):
        home_m = all_matches[all_matches["home_team"] == team]
        away_m = all_matches[all_matches["away_team"] == team]

        # Scores
        home_gf = home_m["home_score"].dropna().astype(int).sum()
        home_ga = home_m["away_score"].dropna().astype(int).sum()
        away_gf = away_m["away_score"].dropna().astype(int).sum()
        away_ga = away_m["home_score"].dropna().astype(int).sum()
        tourn_gf = int(home_gf + away_gf)
        tourn_ga = int(home_ga + away_ga)

        # Points
        tourn_pts = 0
        for _, m in home_m.iterrows():
            r = _match_outcome(m["home_score"], m["away_score"])
            if r == WIN:   tourn_pts += 3
            elif r == DRAW:tourn_pts += 1
        for _, m in away_m.iterrows():
            r = _match_outcome(m["home_score"], m["away_score"])
            if r == LOSS:  tourn_pts += 3  # away team won
            elif r == DRAW:tourn_pts += 1

        # Yellow cards (DS1 only — may be missing for DS1-ext rows)
        home_yc = home_m["home_cards_yellow"].dropna().astype(int).sum()
        away_yc = away_m["away_cards_yellow"].dropna().astype(int).sum()

        # Tactical: possession, SOT — only from DS1 (where not NaN)
        def _col_avg(df_home, df_away, home_col, away_col):
            h_vals = df_home[home_col].dropna().astype(float).tolist()
            a_vals = df_away[away_col].dropna().astype(float).tolist()
            vals = h_vals + a_vals
            return float(np.mean(vals)) if vals else np.nan

        tourn_avg_possession = _col_avg(
            home_m, away_m, "home_possession", "away_possession")
        tourn_avg_sot        = _col_avg(
            home_m, away_m, "home_sot", "away_sot")
        tourn_sot_conceded   = _col_avg(
            home_m, away_m, "away_sot", "home_sot")

        # Shot conversion rate: goals / total shots (shots from DS1 only)
        total_shots_home = home_m["home_total_shots"].dropna().astype(float).sum()
        total_shots_away = away_m["away_total_shots"].dropna().astype(float).sum()
        total_shots = total_shots_home + total_shots_away
        tourn_shot_conversion = float(tourn_gf / total_shots) if total_shots > 0 else np.nan

        # Formation change: compare MD1 vs MD2 formations (DS1 only, 44 teams)
        tourn_formation_changed = np.nan  # default for 4 June 23 teams
        if team not in _JUNE23_TEAMS:
            gw1_home = home_m[home_m["gameweek"] == 1]["home_formation"].tolist()
            gw1_away = away_m[away_m["gameweek"] == 1]["away_formation"].tolist()
            gw2_home = home_m[home_m["gameweek"] == 2]["home_formation"].tolist()
            gw2_away = away_m[away_m["gameweek"] == 2]["away_formation"].tolist()
            f1 = (gw1_home + gw1_away + [None])[0]
            f2 = (gw2_home + gw2_away + [None])[0]
            if f1 is not None and f2 is not None:
                tourn_formation_changed = int(str(f1).strip() != str(f2).strip())
            else:
                tourn_formation_changed = np.nan

        has_full_tactical_md2 = int(team not in _JUNE23_TEAMS)

        out_rows.append({
            "team_canonical":            team,
            "tourn_pts_md2":             int(tourn_pts),             # F029
            "tourn_gd_md2":              int(tourn_gf - tourn_ga),   # F030
            "tourn_gf_md2":              int(tourn_gf),              # F031
            "tourn_ga_md2":              int(tourn_ga),              # F032
            "tourn_avg_possession":      tourn_avg_possession,       # F033
            "tourn_avg_sot":             tourn_avg_sot,              # F034
            "tourn_sot_conceded":        tourn_sot_conceded,         # F035
            "tourn_shot_conversion_rate":tourn_shot_conversion,      # F036
            "tourn_yellow_cards_md2":    int(home_yc + away_yc),    # F037
            "tourn_formation_changed":   tourn_formation_changed,    # F038
            "has_full_tactical_md2":     has_full_tactical_md2,      # F039
        })

    out = pd.DataFrame(out_rows).set_index("team_canonical")
    out.index.name = "team_canonical"
    return out


# ---------------------------------------------------------------------------
# Group 6 — Penalty shootout features
# Features: F040–F043 + shootout_naive_flag
# Source: DS6 (all competitions), DS8 (WC-only subset)
# ---------------------------------------------------------------------------

def build_penalty_features(
    ds6: pd.DataFrame,
    ds8: pd.DataFrame,
    shrinkage_k: int = SHOOTOUT_SHRINKAGE_K,
) -> pd.DataFrame:
    """Return 48-row DataFrame with penalty shootout features (F040–F043)."""

    # Build shootout win/loss record from DS6 for each team
    shootout_stats: dict[str, dict] = {t: {"wins": 0, "apps": 0} for t in CANONICAL_48}

    for _, m in ds6.iterrows():
        ht = m["home_team"] if m["home_team"] in CANONICAL_48 else None
        at = m["away_team"] if m["away_team"] in CANONICAL_48 else None
        wt = m["winner"]    if m["winner"]    in CANONICAL_48 else None
        if ht and ht in shootout_stats:
            shootout_stats[ht]["apps"] += 1
            if ht == wt:
                shootout_stats[ht]["wins"] += 1
        if at and at in shootout_stats:
            shootout_stats[at]["apps"] += 1
            if at == wt:
                shootout_stats[at]["wins"] += 1

    # WC-specific shootout record from DS8
    ds8_shootouts = ds8[
        ds8["home_penalty"].notna() & ds8["away_penalty"].notna()
    ].copy()
    wc_shootout_stats: dict[str, dict] = {t: {"wins": 0, "apps": 0} for t in CANONICAL_48}

    for _, m in ds8_shootouts.iterrows():
        ht = m["home_canonical"] if m["home_canonical"] in CANONICAL_48 else None
        at = m["away_canonical"] if m["away_canonical"] in CANONICAL_48 else None
        try:
            hp, ap = int(m["home_penalty"]), int(m["away_penalty"])
        except (ValueError, TypeError):
            continue
        if ht and ht in wc_shootout_stats:
            wc_shootout_stats[ht]["apps"] += 1
            if hp > ap:
                wc_shootout_stats[ht]["wins"] += 1
        if at and at in wc_shootout_stats:
            wc_shootout_stats[at]["apps"] += 1
            if ap > hp:
                wc_shootout_stats[at]["wins"] += 1

    # First-shooter advantage from DS6 (62% null rate)
    first_shooter_stats: dict[str, dict] = {
        t: {"shot_first": 0, "total": 0} for t in CANONICAL_48
    }
    for _, m in ds6.iterrows():
        if pd.isna(m.get("first_shooter")):
            continue
        fs = canonicalize(str(m["first_shooter"]).strip())
        if fs in first_shooter_stats:
            first_shooter_stats[fs]["shot_first"] += 1
            first_shooter_stats[fs]["total"] += 1
        # The non-first team
        ht = m["home_team"] if m["home_team"] in CANONICAL_48 else None
        at = m["away_team"] if m["away_team"] in CANONICAL_48 else None
        for t in [ht, at]:
            if t and t != fs and t in first_shooter_stats:
                first_shooter_stats[t]["total"] += 1

    out_rows: list[dict] = []
    for team in sorted(CANONICAL_48):
        s = shootout_stats[team]
        apps, wins = s["apps"], s["wins"]
        naive_flag = int(apps == 0)

        # Shrinkage-adjusted win rate (k = shrinkage_k)
        shrunk_rate = (wins + shrinkage_k * 0.5) / (apps + shrinkage_k)

        # WC-only rate
        wcs = wc_shootout_stats[team]
        wc_rate = (wcs["wins"] / wcs["apps"]) if wcs["apps"] >= 2 else np.nan

        # First-shooter advantage
        fs = first_shooter_stats[team]
        if fs["total"] > 0:
            first_shooter_adv = fs["shot_first"] / fs["total"]
        else:
            first_shooter_adv = 0.5  # unknown

        out_rows.append({
            "team_canonical":               team,
            "shootout_win_rate_alltime":    shrunk_rate,          # F040 (shrunk)
            "shootout_appearances_total":   apps,                 # F041
            "shootout_win_rate_wc_only":    wc_rate,             # F042
            "shootout_first_shooter_advantage": first_shooter_adv,# F043
            "shootout_naive_flag":          naive_flag,
        })

    out = pd.DataFrame(out_rows).set_index("team_canonical")
    out.index.name = "team_canonical"
    return out


# ---------------------------------------------------------------------------
# Team feature assembler — merge Groups 1–6 into one 48-row table
# ---------------------------------------------------------------------------

def build_team_features(
    ds2: pd.DataFrame,
    ds10: pd.DataFrame,
    ds11: pd.DataFrame,
    ds8: pd.DataFrame,
    ds6: pd.DataFrame,
    ds4: pd.DataFrame,
    ds1: pd.DataFrame,
    ds1ext: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all team-level feature groups into a single 48-row DataFrame.

    The returned DataFrame is indexed by team_canonical (48 rows). It does NOT
    include matchup-relative features (F002, F003, delta features) — those are
    added by build_feature_table() for each specific match perspective.
    """
    elo  = build_elo_features(ds2)
    fifa = build_fifa_features(ds10, ds11, elo)
    hist = build_wc_historical_features(ds8, ds6)
    form = build_form_features(ds4)
    tourn= build_tournament_features(ds1, ds1ext)
    pen  = build_penalty_features(ds6, ds8)

    teams = (
        elo
        .join(fifa,  how="left", rsuffix="_fifa")
        .join(hist,  how="left")
        .join(form,  how="left")
        .join(tourn, how="left")
        .join(pen,   how="left")
    )

    # Drop duplicate columns from joins
    dup_cols = [c for c in teams.columns if c.endswith("_fifa")]
    teams = teams.drop(columns=dup_cols)

    if len(teams) != 48:
        raise ValueError(f"Team feature table has {len(teams)} rows, expected 48.")

    return teams


# ---------------------------------------------------------------------------
# Match context features (Group 7)
# ---------------------------------------------------------------------------

def _build_match_context(
    ds16: pd.DataFrame,
    ds17: pd.DataFrame,
    ds18: pd.DataFrame,
    ds19: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per DS16 match with venue, stage, and kickoff context."""
    # Build lookup tables
    stage_lookup = ds19.set_index("id")[["stage_order", "stage_name"]]
    city_lookup  = ds18.set_index("id")[["city_name", "country", "region_cluster"]]
    team_lookup  = ds17.set_index("id")["team_canonical"]

    ctx = ds16.copy()
    ctx["stage_order"]       = ctx["stage_id"].map(stage_lookup["stage_order"])
    ctx["stage_name"]        = ctx["stage_id"].map(stage_lookup["stage_name"])
    ctx["is_knockout"]       = ctx["stage_order"] > 1
    ctx["venue_city"]        = ctx["city_id"].map(city_lookup["city_name"])
    ctx["venue_country"]     = ctx["city_id"].map(city_lookup["country"])
    ctx["venue_region_cluster"] = ctx["city_id"].map(city_lookup["region_cluster"])
    ctx["home_team_canonical"] = ctx["home_team_id"].map(team_lookup)
    ctx["away_team_canonical"] = ctx["away_team_id"].map(team_lookup)

    # Kickoff local hour (timestamp already in local time with UTC offset)
    ctx["kickoff_local_hour"] = ctx["kickoff_at"].dt.hour

    return ctx[[
        "id", "home_team_canonical", "away_team_canonical",
        "match_label", "stage_id", "stage_order", "stage_name",
        "is_knockout", "venue_city", "venue_country", "venue_region_cluster",
        "kickoff_local_hour",
    ]].rename(columns={"id": "match_id"})


# ---------------------------------------------------------------------------
# Full feature table — 96 rows (48 frozen matches × 2 team perspectives)
# ---------------------------------------------------------------------------

def build_feature_table(
    team_features: pd.DataFrame,
    ds16: pd.DataFrame,
    ds17: pd.DataFrame,
    ds18: pd.DataFrame,
    ds19: pd.DataFrame,
    ds1: Optional[pd.DataFrame] = None,
    ds1ext: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the complete 96-row feature table (48 frozen matches × 2 perspectives).

    Parameters
    ----------
    team_features : output of build_team_features() — 48-row team-level table.
    ds16, ds17, ds18, ds19 : bracket structure datasets.
    ds1, ds1ext : passed only when the target outcome column is desired
                  (derive outcome from the frozen match results).

    Returns
    -------
    DataFrame with one row per (match, team_perspective). Columns:
      - 5 identity columns (match_id, team_canonical, opponent_canonical, ...)
      - 50 feature columns (F001–F050 where applicable)
      - matchup delta features (elo_delta, etc.)
      - outcome (int or NaN for future matches)
    """
    ctx = _build_match_context(ds16, ds17, ds18, ds19)

    # Limit to the 48 Group Stage frozen matches (stage_id == 1, with real results)
    # Future knockout matches are included with outcome = NaN
    rows: list[dict] = []

    for _, match in ctx.iterrows():
        home = match["home_team_canonical"]
        away = match["away_team_canonical"]

        if home not in team_features.index or away not in team_features.index:
            # Skip matches for teams not in team features (e.g. placeholder slots)
            continue

        ht = team_features.loc[home]
        at = team_features.loc[away]

        # Derive match outcome from DS1+DS1ext if provided
        home_score = away_score = np.nan
        if ds1 is not None and ds1ext is not None:
            all_results = pd.concat([ds1, ds1ext], ignore_index=True) if ds1ext is not None else ds1
            match_row = all_results[
                (all_results["home_team"] == home) & (all_results["away_team"] == away)
            ]
            if not match_row.empty:
                home_score = match_row.iloc[0]["home_score"]
                away_score = match_row.iloc[0]["away_score"]

        def _perspective_row(
            team: str,
            opp: str,
            tf,       # team features Series
            of,       # opponent features Series
            is_home: bool,
        ) -> dict:
            """Build one row from a team's perspective."""
            team_elo = float(tf["elo_rating"])  if pd.notna(tf.get("elo_rating")) else np.nan
            opp_elo  = float(of["elo_rating"])  if pd.notna(of.get("elo_rating")) else np.nan

            elo_delta     = (team_elo - opp_elo) if (not np.isnan(team_elo) and not np.isnan(opp_elo)) else np.nan
            elo_win_exp   = _elo_win_expectancy(team_elo, opp_elo) if not np.isnan(elo_delta) else np.nan

            # F013 at match level: sum of both teams' rank disagreements
            td = tf.get("elo_fifa_rank_disagreement", np.nan)
            od = of.get("elo_fifa_rank_disagreement", np.nan)
            match_rank_disagreement = (
                float(td) + float(od)
                if (pd.notna(td) and pd.notna(od)) else np.nan
            )

            # Outcome (from team's perspective)
            if pd.notna(home_score) and pd.notna(away_score):
                outcome = _team_outcome(is_home, home_score, away_score)
            else:
                outcome = None

            return {
                # Identity
                "match_id":            match["match_id"],
                "team_canonical":      team,
                "opponent_canonical":  opp,
                "is_home_team":        int(is_home),
                "stage_id":            match["stage_id"],
                # Match context (F044–F050)
                "stage_order":         match["stage_order"],
                "is_knockout":         int(match["is_knockout"]),
                "venue_city":          match["venue_city"],
                "venue_country":       match["venue_country"],
                "venue_region_cluster":match["venue_region_cluster"],
                "kickoff_local_hour":  match["kickoff_local_hour"],
                "elo_rank_disagreement_match": match_rank_disagreement,  # F050
                # Group 1 (F001–F007) — team-level
                "elo_rating":            tf.get("elo_rating"),
                "elo_win_expectancy":    elo_win_exp,                    # F002
                "elo_rating_delta":      elo_delta,                      # F003
                "elo_rating_career_peak":tf.get("elo_rating_career_peak"),
                "elo_rating_career_avg": tf.get("elo_rating_career_avg"),
                "elo_rank":              tf.get("elo_rank"),
                "elo_is_host":           tf.get("elo_is_host"),
                "confederation":         tf.get("confederation"),        # F048
                # Group 2 (F008–F013) — team-level
                "fifa_points":           tf.get("fifa_points"),
                "fifa_points_delta":     tf.get("fifa_points_delta"),
                "fifa_rank_delta":       tf.get("fifa_rank_delta"),
                "fifa_points_4yr_change":tf.get("fifa_points_4yr_change"),
                "fifa_rank_4yr_change":  tf.get("fifa_rank_4yr_change"),
                "elo_fifa_rank_disagreement": tf.get("elo_fifa_rank_disagreement"),
                # Group 3 (F014–F022)
                "wc_win_rate_modern":          tf.get("wc_win_rate_modern"),
                "wc_win_rate_knockout_modern": tf.get("wc_win_rate_knockout_modern"),
                "wc_avg_gf_modern":            tf.get("wc_avg_gf_modern"),
                "wc_avg_ga_modern":            tf.get("wc_avg_ga_modern"),
                "wc_gd_per_game_modern":       tf.get("wc_gd_per_game_modern"),
                "wc_clean_sheet_rate_modern":  tf.get("wc_clean_sheet_rate_modern"),
                "wc_tournaments_attended":     tf.get("wc_tournaments_attended"),
                "wc_best_result_encoded":      tf.get("wc_best_result_encoded"),
                "wc_group_vs_knockout_uplift": tf.get("wc_group_vs_knockout_uplift"),
                "wc_debut_modern_flag":        tf.get("wc_debut_modern_flag"),
                # Group 4 (F023–F028)
                "form_win_rate_last10":          tf.get("form_win_rate_last10"),
                "form_avg_gf_last10":            tf.get("form_avg_gf_last10"),
                "form_avg_ga_last10":            tf.get("form_avg_ga_last10"),
                "form_gd_last10":                tf.get("form_gd_last10"),
                "form_clean_sheet_rate_last10":  tf.get("form_clean_sheet_rate_last10"),
                "form_unbeaten_streak_entering": tf.get("form_unbeaten_streak_entering"),
                # Group 5 (F029–F039)
                "tourn_pts_md2":             tf.get("tourn_pts_md2"),
                "tourn_gd_md2":              tf.get("tourn_gd_md2"),
                "tourn_gf_md2":              tf.get("tourn_gf_md2"),
                "tourn_ga_md2":              tf.get("tourn_ga_md2"),
                "tourn_avg_possession":      tf.get("tourn_avg_possession"),
                "tourn_avg_sot":             tf.get("tourn_avg_sot"),
                "tourn_sot_conceded":        tf.get("tourn_sot_conceded"),
                "tourn_shot_conversion_rate":tf.get("tourn_shot_conversion_rate"),
                "tourn_yellow_cards_md2":    tf.get("tourn_yellow_cards_md2"),
                "tourn_formation_changed":   tf.get("tourn_formation_changed"),
                "has_full_tactical_md2":     tf.get("has_full_tactical_md2"),
                # Group 6 (F040–F043)
                "shootout_win_rate_alltime":       tf.get("shootout_win_rate_alltime"),
                "shootout_appearances_total":      tf.get("shootout_appearances_total"),
                "shootout_win_rate_wc_only":       tf.get("shootout_win_rate_wc_only"),
                "shootout_first_shooter_advantage":tf.get("shootout_first_shooter_advantage"),
                "shootout_naive_flag":             tf.get("shootout_naive_flag"),
                # Matchup deltas (Group 7 matchup-level)
                "fifa_points_delta_vs_opp":  (
                    (float(tf.get("fifa_points", np.nan) or np.nan)
                     - float(of.get("fifa_points", np.nan) or np.nan))
                    if pd.notna(tf.get("fifa_points")) and pd.notna(of.get("fifa_points"))
                    else np.nan
                ),
                "wc_experience_delta": (
                    int(tf.get("wc_tournaments_attended", 0) or 0)
                    - int(of.get("wc_tournaments_attended", 0) or 0)
                ),
                "form_win_rate_delta": (
                    (float(tf.get("form_win_rate_last10", np.nan) or np.nan)
                     - float(of.get("form_win_rate_last10", np.nan) or np.nan))
                    if pd.notna(tf.get("form_win_rate_last10")) and pd.notna(of.get("form_win_rate_last10"))
                    else np.nan
                ),
                "tourn_pts_delta": (
                    int(tf.get("tourn_pts_md2", 0) or 0)
                    - int(of.get("tourn_pts_md2", 0) or 0)
                ),
                "tourn_gd_delta": (
                    int(tf.get("tourn_gd_md2", 0) or 0)
                    - int(of.get("tourn_gd_md2", 0) or 0)
                ),
                "dual_system_agreement": int(
                    (elo_delta is not None and not np.isnan(elo_delta))
                    and (
                        (elo_delta > 0 and pd.notna(tf.get("fifa_points")) and pd.notna(of.get("fifa_points"))
                         and float(tf.get("fifa_points")) > float(of.get("fifa_points")))
                        or (elo_delta < 0 and pd.notna(tf.get("fifa_points")) and pd.notna(of.get("fifa_points"))
                            and float(tf.get("fifa_points")) < float(of.get("fifa_points")))
                        or (elo_delta == 0)
                    )
                ),
                # Target
                "outcome": outcome,
            }

        rows.append(_perspective_row(home, away, ht, at, is_home=True))
        rows.append(_perspective_row(away, home, at, ht, is_home=False))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Training rows — historical WC matches for Layer 1 model
# Source: DS8 (1998–2022 primary corpus), DS4 (form), DS2 (Elo snapshots)
# ---------------------------------------------------------------------------

def build_training_rows(
    ds8: pd.DataFrame,
    ds4: pd.DataFrame,
    ds2: pd.DataFrame,
    ds6: pd.DataFrame,
    ds10: pd.DataFrame,
    ds11: pd.DataFrame,
    year_range: tuple[int, int] = (WC_MODERN_START, 2022),
) -> pd.DataFrame:
    """Build training rows for the Layer 1 historical model.

    Each WC match in DS8 from year_range produces two rows (home/away perspective).
    Features use only data available before each match (no leakage):
      - Elo from DS2 snapshot_date == f"{match_year-1}-12-31"
      - WC historical through year < match_year
      - Form from DS4: competitive matches in 24 months before match date
      - No in-tournament features (Groups 5–6 not applicable for training)

    Returns a DataFrame with match_year and elo_year_used columns for
    leakage_guard.check_training_rows_chronological().
    """
    min_year, max_year = year_range
    train_ds8 = ds8[
        (ds8["Year"] >= min_year) & (ds8["Year"] <= max_year)
        & ds8["home_score"].notna() & ds8["away_score"].notna()
    ].copy()

    rows: list[dict] = []

    for year in sorted(train_ds8["Year"].unique()):
        year = int(year)
        year_matches = train_ds8[train_ds8["Year"] == year]
        elo_snap_date = f"{year - 1}-12-31"

        # Year-end Elo snapshot for this training year
        elo_snap = ds2[ds2["snapshot_date"] == elo_snap_date].copy()
        elo_snap = elo_snap.set_index("country")

        # WC historical features through previous tournaments
        hist = build_wc_historical_features(ds8, ds6, through_year=year - 1)

        # Penalty features (all-time through the dataset, no leakage concern since DS6 is pre-tournament)
        pen = build_penalty_features(ds6, ds8)

        # FIFA features — only available for 2022 (DS11) and no earlier snapshots
        # For 2022, use DS11; for earlier years, FIFA features are null
        if year == 2022:
            # Create a mock ds10-like frame from DS11 for the 2022 training year
            ds11_as_10 = ds11.copy()
            ds11_as_10["previous_points"] = np.nan
            ds11_as_10["previous_rank"]   = np.nan
            fifa_feats = build_fifa_features(ds11_as_10, pd.DataFrame(columns=ds11.columns), hist)
        else:
            fifa_feats = None

        for _, match in year_matches.iterrows():
            home = match["home_canonical"]
            away = match["away_canonical"]

            if home not in CANONICAL_48 or away not in CANONICAL_48:
                continue  # Skip non-WC-2026-team historical matches

            match_date = match["Date"]
            home_score = int(match["home_score"])
            away_score = int(match["away_score"])

            # Form: DS4 competitive matches 24 months before this match date
            form_start_dt = match_date - pd.Timedelta(days=730)
            form_df = ds4[
                (ds4["tournament"].str.strip() != "Friendly")
                & (ds4["date"] >= form_start_dt)
                & (ds4["date"] < match_date)
                & ds4["home_score"].notna()
                & ds4["away_score"].notna()
            ].copy()

            def _form(team: str) -> dict:
                home_m = form_df[form_df["home_team"] == team][
                    ["home_score", "away_score"]
                ].rename(columns={"home_score": "gf", "away_score": "ga"})
                away_m = form_df[form_df["away_team"] == team][
                    ["home_score", "away_score"]
                ].rename(columns={"away_score": "gf", "home_score": "ga"})
                m10 = pd.concat([home_m, away_m], ignore_index=True).tail(FORM_WINDOW_MATCHES)
                n = len(m10)
                if n == 0:
                    return {
                        "form_win_rate_last10": np.nan,
                        "form_avg_gf_last10":   np.nan,
                        "form_avg_ga_last10":   np.nan,
                        "form_gd_last10":       np.nan,
                    }
                gf = m10["gf"].astype(float)
                ga = m10["ga"].astype(float)
                return {
                    "form_win_rate_last10": float((gf > ga).sum()) / n,
                    "form_avg_gf_last10":   float(gf.mean()),
                    "form_avg_ga_last10":   float(ga.mean()),
                    "form_gd_last10":       float((gf - ga).mean()),
                }

            def _elo_vals(team: str) -> dict:
                if team not in elo_snap.index:
                    return {"elo_rating": np.nan, "elo_rank": np.nan}
                row = elo_snap.loc[team]
                return {
                    "elo_rating":            float(row["rating"]),
                    "elo_rank":              float(row["rank"]),
                    "elo_rating_career_peak":float(row["rating_max"]),
                    "elo_rating_career_avg": float(row["rating_avg"]),
                    "elo_is_host":           0,  # host advantage not applicable historically
                }

            def _hist_vals(team: str) -> dict:
                if team not in hist.index:
                    return {}
                return hist.loc[team].to_dict()

            def _pen_vals(team: str) -> dict:
                if team not in pen.index:
                    return {}
                return pen.loc[team].to_dict()

            def _fifa_vals(team: str) -> dict:
                if fifa_feats is None or team not in fifa_feats.index:
                    return {}
                return fifa_feats.loc[team].to_dict()

            def _build_row(team: str, opp: str, is_home: bool) -> dict:
                t_elo   = _elo_vals(team)
                o_elo   = _elo_vals(opp)
                t_hist  = _hist_vals(team)
                t_form  = _form(team)
                t_pen   = _pen_vals(team)
                t_fifa  = _fifa_vals(team)

                elo_delta   = (t_elo.get("elo_rating", np.nan) - o_elo.get("elo_rating", np.nan)
                               if pd.notna(t_elo.get("elo_rating")) and pd.notna(o_elo.get("elo_rating"))
                               else np.nan)
                elo_win_exp = (
                    _elo_win_expectancy(t_elo["elo_rating"], o_elo["elo_rating"])
                    if pd.notna(elo_delta) else np.nan
                )
                outcome = _team_outcome(is_home, home_score, away_score)

                r = {
                    # Metadata for chronological check
                    "match_year":     year,
                    "elo_year_used":  year - 1,
                    "match_date":     match_date,
                    "team_canonical": team,
                    "opponent_canonical": opp,
                    "is_home_team":   int(is_home),
                    # Elo
                    "elo_rating":            t_elo.get("elo_rating"),
                    "elo_win_expectancy":    elo_win_exp,
                    "elo_rating_delta":      elo_delta,
                    "elo_rating_career_peak":t_elo.get("elo_rating_career_peak"),
                    "elo_rating_career_avg": t_elo.get("elo_rating_career_avg"),
                    "elo_rank":              t_elo.get("elo_rank"),
                    "elo_is_host":           t_elo.get("elo_is_host", 0),
                    # FIFA (null for years without snapshot)
                    **{f"fifa_{k}": v for k, v in t_fifa.items() if k.startswith("fifa_")
                       or k in ("elo_fifa_rank_disagreement",)},
                    # WC historical
                    **t_hist,
                    # Form
                    **t_form,
                    # Penalty
                    "shootout_win_rate_alltime":  t_pen.get("shootout_win_rate_alltime"),
                    "shootout_appearances_total": t_pen.get("shootout_appearances_total"),
                    # Target
                    "outcome": outcome,
                }
                return r

            rows.append(_build_row(home, away, is_home=True))
            rows.append(_build_row(away, home, is_home=False))

    df = pd.DataFrame(rows)
    # Reorder: metadata first, then features, then target
    meta = ["match_year", "elo_year_used", "match_date",
            "team_canonical", "opponent_canonical", "is_home_team"]
    feat  = [c for c in df.columns if c not in meta + ["outcome"]]
    return df[meta + feat + ["outcome"]].reset_index(drop=True)

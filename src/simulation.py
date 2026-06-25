"""
Monte Carlo tournament engine for the FIFA WC 2026 Forecast Engine.

Frozen snapshot: June 23, 2026 — Post-Matchday 2

Architecture
------------
Each simulation run proceeds in six deterministic-plus-stochastic steps:

  Step 1  Simulate all 24 Matchday 3 matches simultaneously.
          Probabilities come from the Layer 1 + Layer 2 combined output stored in
          md3_proba (shape n_matches × 3: [P(home_win), P(draw), P(away_win)]).
          Group-context soft adjustments apply before sampling.

  Step 2  Apply tiebreakers deterministically within each group.
          Sequence: H2H pts → H2H GD → H2H GF → overall GD → overall GF → FIFA rank.

  Step 3  Rank all 12 third-place finishers; select the eight best.
          Sequence: points → GD → GF → FIFA rank (proxy for drawing of lots).

  Step 4  Resolve bracket using Annex C (third_place_annex_c.csv).
          The 8 qualifying groups determine which group's third-placed team
          occupies each third-place slot in the R32 bracket.

  Step 5  Simulate all knockout matches from R32 through the Final.
          Uses WCForecastEnsemble.predict_knockout() with stage_order context.
          Draws at 90 min trigger a conditional penalty shootout via PenaltyModel.

  Step 6  Record the tournament winner and each team's stage reached.

  Step 7  Aggregate over N runs: win probability, round-reach distribution,
          90% confidence intervals, most common final matchup.

Bracket reference
-----------------
Source: DS16 matches.csv / DS19 tournament_stages.csv

  Stage 2 (R32):      matches 73–88  (16 matches)
  Stage 3 (R16):      matches 89–96  ( 8 matches)
  Stage 4 (QF):       matches 97–100 ( 4 matches)   ← DS16 match-100 data error corrected
  Stage 5 (SF):       matches 101–102 (2 matches)
  Stage 6 (3rd-place):match 103
  Stage 7 (Final):    match 104

DS16 data error: match_number=100 label is "W95 vs W100" — the correct opponent is
W96 (the only remaining R16 winner not yet paired). All internal references use W96.

Third-place Annex C
-------------------
File: third_place_annex_c.csv
Columns: qualifying_groups (sorted comma-separated group letters), m75, m78, m79,
         m80, m81, m82, m85, m88.
Each cell value is a group designator like "3F" — the third-placed team from that
group occupies that R32 slot. The qualifying_groups key is constructed from the set
of 8 groups that produced third-placed qualifiers after MD3 resolution.

Public API
----------
  load_annex_c(path)                           → AnnexCLookup
  build_md3_schedule(ds16, ds17)               → list[MD3Match]
  MonteCarloEngine(...)                        — main simulation class
  MonteCarloEngine.run(n, seed)                → SimulationResults
  SimulationResults.to_dataframe()             → pd.DataFrame
  SimulationResults.to_csv(path)               → None
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.leakage_guard import check_freeze_date, LeakageError
from src.name_map import CANONICAL_48, WC_DEBUTANTS, canonicalize, canonicalize_id
from src.standings import build_standings_from_raw, compute_group_standings, rank_third_placed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROUPS: list[str] = list("ABCDEFGHIJKL")

# Stage order values (from DS19)
STAGE_GROUP    = 1
STAGE_R32      = 2
STAGE_R16      = 3
STAGE_QF       = 4
STAGE_SF       = 5
STAGE_TPM      = 6   # Third-Place Match
STAGE_FINAL    = 7

# Probability of a draw at 90 min in knockout matches
DRAW_RATE_90MIN = 0.214

# Soft adjustments for group-context flags (per design spec §6 Step 1)
QUALIFIED_WIN_ADJ  = -0.03   # subtract from P(win) when already qualified
ELIMINATED_DRAW_ADJ = -0.03  # subtract from P(draw) when already eliminated

# Annex C column names (verified against file on disk)
ANNEX_C_KEY_COL  = "qualifying_groups"
ANNEX_C_SLOT_COLS: list[str] = ["m75", "m78", "m79", "m80", "m81", "m82", "m85", "m88"]

# R32 structure: match_number → (home_slot, away_slot)
# Format: "1X" = winner of group X, "2X" = runner-up of group X,
#         None = third-place team assigned from Annex C
R32_STRUCTURE: dict[int, tuple[str, Optional[str]]] = {
    73: ("2A", "2B"),
    74: ("1C", "2F"),
    75: ("1E", None),     # third-place from Annex C
    76: ("1F", "2C"),
    77: ("2E", "2I"),
    78: ("1I", None),
    79: ("1A", None),
    80: ("1L", None),
    81: ("1G", None),
    82: ("1D", None),
    83: ("1H", "2J"),
    84: ("2K", "2L"),
    85: ("1B", None),
    86: ("2D", "2G"),
    87: ("1J", "2H"),
    88: ("1K", None),
}

# R32 matches that host a third-place qualifier in the away slot
R32_THIRD_PLACE_SLOTS: frozenset[int] = frozenset({75, 78, 79, 80, 81, 82, 85, 88})

# Bracket progression: each match → (parent_match_a, parent_match_b)
# Teams are the WINNERS of the parent matches. Runner-ups used for third-place match.
R16_BRACKET: dict[int, tuple[int, int]] = {
    89: (73, 75),
    90: (74, 77),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
}

QF_BRACKET: dict[int, tuple[int, int]] = {
    97:  (89, 90),
    98:  (93, 94),
    99:  (91, 92),
    100: (95, 96),   # DS16 data error: label says "W95 vs W100", correct is W95 vs W96
}

SF_BRACKET: dict[int, tuple[int, int]] = {
    101: (97,  98),
    102: (99, 100),
}

THIRD_PLACE_MATCH = 103   # runner-ups of SF101 vs SF102
FINAL_MATCH       = 104   # winners of SF101 vs SF102


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MD3Match:
    """One Matchday 3 match with canonical team names."""
    match_id: int        # DS16 match_number
    home_team: str
    away_team: str
    group: str


@dataclass
class GroupEntry:
    """One team's group-stage record after MD3 (one simulation run)."""
    team:  str
    group: str
    pts:   int = 0
    gd:    int = 0
    gf:    int = 0
    ga:    int = 0
    # Head-to-head records keyed by opponent name
    h2h:   dict = field(default_factory=dict)  # {opp: (pts, gd, gf)}
    position: int = 0   # 1–4 within group, set after tiebreakers

    @property
    def rank_key(self) -> tuple:
        """Primary sort key (descending): pts, gd, gf (lower=worse)."""
        return (-self.pts, -self.gd, -self.gf)


@dataclass
class SimRun:
    """Complete state of one Monte Carlo simulation run."""
    # group → ordered list of GroupEntry (sorted 1st→4th)
    final_standings: dict[str, list[GroupEntry]] = field(default_factory=dict)
    # 8 teams that qualify as best third-placed (in selection order)
    third_place_qualifiers: list[str] = field(default_factory=list)
    # match_id → winning team (canonical name)
    bracket_winners: dict[int, str] = field(default_factory=dict)
    # match_id → losing team (canonical name); needed for third-place match
    bracket_runners: dict[int, str] = field(default_factory=dict)
    tournament_winner: Optional[str] = None
    # team → highest stage reached (as STAGE_* constant)
    stage_reached: dict[str, int] = field(default_factory=dict)


@dataclass
class SimulationResults:
    """Aggregated results from N Monte Carlo simulation runs.

    Attributes
    ----------
    n_simulations : number of completed runs.
    win_prob      : {team: fraction of runs where team won the tournament}.
    stage_probs   : {team: {stage_int: probability of reaching that stage}}.
    ci_90         : {team: (lower_5th, upper_95th) of win_prob bootstrap CI}.
    most_common_final : list of (team_a, team_b, count) top 5 final matchups.
    """
    n_simulations: int = 0
    win_prob:      dict[str, float] = field(default_factory=dict)
    stage_probs:   dict[str, dict[int, float]] = field(default_factory=dict)
    ci_90:         dict[str, tuple[float, float]] = field(default_factory=dict)
    most_common_final: list[tuple[str, str, int]] = field(default_factory=list)
    final_pair_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-team win probabilities with 90% CI as a DataFrame."""
        rows = []
        for team in sorted(self.win_prob):
            lo, hi = self.ci_90.get(team, (np.nan, np.nan))
            rows.append({
                "team":       team,
                "win_prob":   round(self.win_prob[team], 4),
                "ci_90_lo":   round(lo, 4) if not np.isnan(lo) else np.nan,
                "ci_90_hi":   round(hi, 4) if not np.isnan(hi) else np.nan,
                **{
                    f"p_stage_{s}": round(
                        self.stage_probs.get(team, {}).get(s, 0.0), 4
                    )
                    for s in range(1, 8)
                },
            })
        return pd.DataFrame(rows).sort_values("win_prob", ascending=False).reset_index(drop=True)

    def to_csv(self, path: str) -> None:
        """Write win probability table to CSV."""
        self.to_dataframe().to_csv(path, index=False)
        print(f"Results written to {path} ({self.n_simulations} simulations)")


# ---------------------------------------------------------------------------
# Annex C loader and resolver
# ---------------------------------------------------------------------------

class AnnexCLookup:
    """Lookup table for third-place bracket placement.

    The Annex C CSV has 495 rows = C(12, 8). The key is a frozenset of the 8
    group letters whose third-placed teams qualified. The value maps each R32
    match slot (m75, m78, …) to the group letter whose third-placed team fills it.

    Usage
    -----
        resolver = AnnexCLookup.from_csv("third_place_annex_c.csv")
        assignment = resolver.resolve(frozenset("ABCDEFGH"))
        # → {75: "A", 78: "D", 79: "B", ...}  group letter for each match slot
    """

    def __init__(self, lookup: dict[frozenset, dict[int, str]]) -> None:
        self._lookup = lookup

    @classmethod
    def from_csv(cls, path: str) -> "AnnexCLookup":
        """Load from the third_place_annex_c.csv file.

        Verifies:
          - Expected 495 rows (C(12,8)).
          - Columns: qualifying_groups, m75, m78, m79, m80, m81, m82, m85, m88.
          - Cell values are "3X" patterns (e.g. "3F").
        """
        df = pd.read_csv(path)

        # Verify schema
        expected_cols = {ANNEX_C_KEY_COL} | set(ANNEX_C_SLOT_COLS)
        actual_cols   = set(df.columns)
        if not expected_cols.issubset(actual_cols):
            raise ValueError(
                f"Annex C schema mismatch. Expected columns: {sorted(expected_cols)}, "
                f"got: {sorted(actual_cols)}"
            )
        if len(df) != 495:
            raise ValueError(
                f"Annex C expected 495 rows (C(12,8)), got {len(df)}."
            )

        # Build lookup dict
        slot_to_match = {col: int(col[1:]) for col in ANNEX_C_SLOT_COLS}
        lookup: dict[frozenset, dict[int, str]] = {}

        for _, row in df.iterrows():
            key_str = str(row[ANNEX_C_KEY_COL])
            groups  = frozenset(g.strip() for g in key_str.split(","))
            assignment: dict[int, str] = {}
            for slot_col, match_id in slot_to_match.items():
                cell = str(row[slot_col]).strip()
                if cell.startswith("3") and len(cell) == 2:
                    assignment[match_id] = cell[1]   # e.g. "3F" → "F"
                else:
                    raise ValueError(
                        f"Unexpected Annex C cell value {cell!r} "
                        f"in column {slot_col} for qualifying_groups={key_str!r}"
                    )
            lookup[groups] = assignment

        return cls(lookup)

    def resolve(self, qualifying_groups: frozenset[str]) -> dict[int, str]:
        """Return {match_id: group_letter} for the 8 third-place slots.

        Parameters
        ----------
        qualifying_groups : frozenset of 8 group letters whose third-placed
                            teams qualified (e.g. frozenset("ABCDEFHK")).

        Returns
        -------
        dict mapping R32 match number → group letter of the third-placed team
        assigned to that slot.

        Raises
        ------
        KeyError if the combination is not in the Annex C table (should never
        happen if qualifying_groups has exactly 8 elements from A–L).
        """
        if len(qualifying_groups) != 8:
            raise ValueError(
                f"resolve() expects exactly 8 qualifying groups, "
                f"got {len(qualifying_groups)}: {sorted(qualifying_groups)}"
            )
        if qualifying_groups not in self._lookup:
            raise KeyError(
                f"Qualifying groups combination {sorted(qualifying_groups)} "
                "not found in Annex C table. Verify the 8 qualifying groups."
            )
        return self._lookup[qualifying_groups]


# ---------------------------------------------------------------------------
# MD3 schedule builder
# ---------------------------------------------------------------------------

def build_md3_schedule(
    ds16: pd.DataFrame,
    ds17: pd.DataFrame,
) -> list[MD3Match]:
    """Extract Matchday 3 matches from DS16 with canonical team names.

    Matchday 3 = group stage matches with match_number in 49–72 (verified
    against DS16 structure: 72 group stage matches total across 12 groups,
    24 per matchday).

    Parameters
    ----------
    ds16 : DataFrame from archive.zip matches.csv (DS16).
    ds17 : DataFrame from archive.zip teams.csv (DS17) with canonical names
           already applied (placeholder IDs resolved).

    Returns
    -------
    List of 24 MD3Match objects, two per group, with canonical team names.
    """
    # Build DS17 id → canonical_name lookup (resolve placeholders)
    id_to_name: dict[int, str] = {}
    for _, row in ds17.iterrows():
        tid  = int(row["id"])
        name = canonicalize_id(tid, str(row.get("team_name", "")))
        id_to_name[tid] = name

    # Build DS17 id → group_letter lookup
    id_to_group: dict[int, str] = {}
    for _, row in ds17.iterrows():
        tid = int(row["id"])
        id_to_group[tid] = str(row.get("group_letter", "")).upper()

    # MD3 matches are 49–72
    md3_rows = ds16[
        (ds16["match_number"] >= 49) & (ds16["match_number"] <= 72)
    ].copy()

    if len(md3_rows) != 24:
        raise ValueError(
            f"Expected 24 MD3 matches (match numbers 49–72), "
            f"found {len(md3_rows)}."
        )

    md3_matches: list[MD3Match] = []
    for _, row in md3_rows.iterrows():
        h_id = int(row["home_team_id"])
        a_id = int(row["away_team_id"])
        home_name = id_to_name.get(h_id, "")
        away_name = id_to_name.get(a_id, "")
        group     = id_to_group.get(h_id, "")

        if home_name not in CANONICAL_48:
            raise ValueError(
                f"MD3 match {row['match_number']}: home team ID {h_id} "
                f"resolved to unknown name {home_name!r}."
            )
        if away_name not in CANONICAL_48:
            raise ValueError(
                f"MD3 match {row['match_number']}: away team ID {a_id} "
                f"resolved to unknown name {away_name!r}."
            )

        md3_matches.append(MD3Match(
            match_id  = int(row["match_number"]),
            home_team = home_name,
            away_team = away_name,
            group     = group,
        ))

    return md3_matches


# ---------------------------------------------------------------------------
# Group-stage helpers
# ---------------------------------------------------------------------------

def _compute_group_context(
    frozen_standings: dict[str, list[dict]],
) -> dict[str, dict[str, bool]]:
    """Determine which teams are already qualified or already eliminated.

    Returns
    -------
    {team_name: {"already_qualified": bool, "already_eliminated": bool}}

    Logic
    -----
    already_qualified  : team currently has 6 pts (won both MD1+MD2 games).
                         With only 3 pts available in MD3, no other team can
                         overtake them unless they also have 6 pts.
                         Conservative: apply only when pts_md2 == 6.

    already_eliminated : team has 0 pts AND the 3rd-placed best-third threshold
                         requires typically ≥ 4 pts to qualify. A team on 0 pts
                         can gain at most 3 pts in MD3, which historically is
                         below the safe threshold. Apply when pts_md2 == 0 AND
                         the team is currently 4th in their group.
    """
    context: dict[str, dict[str, bool]] = {}

    for group, entries in frozen_standings.items():
        # entries is a list of dicts sorted by rank (1st to 4th)
        for rank, entry in enumerate(entries, start=1):
            team = entry["team"]
            pts  = entry.get("pts", entry.get("points", 0))
            context[team] = {
                "already_qualified":  int(pts) == 6,
                "already_eliminated": int(pts) == 0 and rank == 4,
            }

    return context


def _adjust_proba_for_context(
    proba: np.ndarray,
    home_context: dict[str, bool],
    away_context: dict[str, bool],
) -> np.ndarray:
    """Apply group-context soft adjustments to a (3,) probability vector.

    Parameters
    ----------
    proba        : [P(home_win), P(draw), P(away_win)] from model.
    home_context : {"already_qualified": bool, "already_eliminated": bool}
    away_context : same for away team.

    Returns
    -------
    Adjusted (3,) array, re-normalised to sum to 1.
    """
    p = proba.copy().astype(float)

    # Already-qualified teams: lower win probability (rotation tendencies)
    if home_context.get("already_qualified", False):
        adj = min(abs(QUALIFIED_WIN_ADJ), p[0] * 0.5)
        p[0] -= adj
        p[2] += adj   # redistribute to away win (away also may rotate)
    if away_context.get("already_qualified", False):
        adj = min(abs(QUALIFIED_WIN_ADJ), p[2] * 0.5)
        p[2] -= adj
        p[0] += adj

    # Already-eliminated teams: lower draw probability (higher variance)
    if home_context.get("already_eliminated", False):
        adj = min(abs(ELIMINATED_DRAW_ADJ), p[1] * 0.5)
        p[1] -= adj
        p[0] += adj   # if eliminated, more likely to press and either win or lose
    if away_context.get("already_eliminated", False):
        adj = min(abs(ELIMINATED_DRAW_ADJ), p[1] * 0.5)
        p[1] -= adj
        p[2] += adj

    # Normalise
    s = p.sum()
    if s > 0:
        p /= s
    return p


def _apply_tiebreakers(
    teams: list[GroupEntry],
    fifa_ranks: dict[str, float],
) -> list[GroupEntry]:
    """Sort a group's teams by the 2026 WC tiebreaker sequence.

    Sequence (per design spec §6 Step 2):
      1. Most points
      2. Head-to-head points (among tied teams)
      3. Head-to-head GD (among tied teams)
      4. Head-to-head GF (among tied teams)
      5. Overall GD
      6. Overall GF
      7. FIFA rank (proxy for drawing of lots)

    Parameters
    ----------
    teams      : list of 4 GroupEntry objects from one group.
    fifa_ranks : {team: FIFA rank (float)} — lower is better.

    Returns
    -------
    Sorted list 1st → 4th, with `.position` assigned.
    """
    def sort_key(e: GroupEntry) -> tuple:
        # Primary: most pts (negate for descending sort)
        return (
            -e.pts,
            -e.gd,
            -e.gf,
            float(fifa_ranks.get(e.team, 200.0)),
        )

    # Two-pass: first check if H2H matters for tied teams
    sorted_teams = sorted(teams, key=sort_key)

    # Check for ties (same pts) and apply H2H tiebreaker
    # Group by points
    by_pts: dict[int, list[GroupEntry]] = {}
    for t in sorted_teams:
        by_pts.setdefault(t.pts, []).append(t)

    final_order: list[GroupEntry] = []
    for pts_val in sorted({t.pts for t in sorted_teams}, reverse=True):
        tied = by_pts[pts_val]
        if len(tied) == 1:
            final_order.extend(tied)
        else:
            # Try H2H tiebreaker among tied teams
            h2h_sorted = _sort_by_h2h(tied, fifa_ranks)
            final_order.extend(h2h_sorted)

    for i, t in enumerate(final_order, start=1):
        t.position = i
    return final_order


def _sort_by_h2h(
    tied: list[GroupEntry],
    fifa_ranks: dict[str, float],
) -> list[GroupEntry]:
    """Apply H2H tiebreaker among teams tied on points.

    Among the tied teams only, compute:
      H2H points → H2H GD → H2H GF → overall GD → overall GF → FIFA rank.
    """
    tied_names = {t.team for t in tied}

    def h2h_key(e: GroupEntry) -> tuple:
        h2h_pts = sum(
            e.h2h[opp][0] for opp in tied_names if opp != e.team and opp in e.h2h
        )
        h2h_gd = sum(
            e.h2h[opp][1] for opp in tied_names if opp != e.team and opp in e.h2h
        )
        h2h_gf = sum(
            e.h2h[opp][2] for opp in tied_names if opp != e.team and opp in e.h2h
        )
        return (
            -h2h_pts,
            -h2h_gd,
            -h2h_gf,
            -e.gd,
            -e.gf,
            float(fifa_ranks.get(e.team, 200.0)),
        )

    return sorted(tied, key=h2h_key)


def _rank_third_placed(
    third_place_entries: list[GroupEntry],
    fifa_ranks: dict[str, float],
) -> list[GroupEntry]:
    """Rank all 12 third-placed teams and return the 8 best.

    Tiebreaker: pts → GD → GF → FIFA rank (proxy for drawing of lots).
    Since teams are from different groups, H2H does not apply.
    """
    def key(e: GroupEntry) -> tuple:
        return (
            -e.pts,
            -e.gd,
            -e.gf,
            float(fifa_ranks.get(e.team, 200.0)),
        )

    ranked = sorted(third_place_entries, key=key)
    return ranked[:8]   # best 8


# ---------------------------------------------------------------------------
# Matchup feature builder for knockout predictions
# ---------------------------------------------------------------------------

def _build_ko_feature_row(
    team: str,
    opponent: str,
    team_features_df: pd.DataFrame,
    simulated_tourn_state: dict[str, dict],
    stage_order: int,
) -> pd.DataFrame:
    """Build a 1-row feature DataFrame for a knockout matchup prediction.

    Combines pre-tournament features from team_features_df with in-tournament
    state (simulated group stage points/GD) to produce the input required by
    WCForecastEnsemble.predict_knockout().

    Parameters
    ----------
    team              : canonical name of the team to predict for.
    opponent          : canonical name of the opposing team.
    team_features_df  : pre-tournament feature table indexed by canonical name.
    simulated_tourn_state : {team: {"pts": int, "gd": int, "gf": int, "ga": int}}
                            from the current simulation run's group stage outcomes.
    stage_order       : current stage (STAGE_R32=2 … STAGE_FINAL=7).

    Returns
    -------
    pd.DataFrame with 1 row, columns matching PRETOURNAMENT_FEATURES +
    TOURNAMENT_FEATURES + KO_CONTEXT_FEATURES.
    """
    idx = team_features_df.index
    t_row = (
        team_features_df.loc[team].copy()
        if team in idx
        else pd.Series(dtype=float)
    )
    o_row = (
        team_features_df.loc[opponent].copy()
        if opponent in idx
        else pd.Series(dtype=float)
    )

    # Matchup-specific Elo features
    elo_t = float(t_row.get("elo_rating", 1500.0))
    elo_o = float(o_row.get("elo_rating", 1500.0))
    delta = elo_t - elo_o
    win_exp = 1.0 / (1.0 + 10.0 ** (-delta / 400.0))

    # In-tournament state (simulated)
    ts = simulated_tourn_state.get(team, {})
    os = simulated_tourn_state.get(opponent, {})

    # Tournament feature override: use simulated group-stage totals
    tourn_pts    = float(ts.get("pts", t_row.get("tourn_pts_md2", 0)))
    tourn_gd     = float(ts.get("gd",  t_row.get("tourn_gd_md2", 0)))
    tourn_gf     = float(ts.get("gf",  t_row.get("tourn_gf_md2", 0)))
    tourn_ga     = float(ts.get("ga",  t_row.get("tourn_ga_md2", 0)))
    opp_pts      = float(os.get("pts", o_row.get("tourn_pts_md2", 0)))
    opp_gd       = float(os.get("gd",  o_row.get("tourn_gd_md2", 0)))

    # Stage interaction feature
    uplift = float(t_row.get("wc_group_vs_knockout_uplift", 0.0))
    ko_interaction = stage_order * uplift

    row = {
        # Group 1 — Elo
        "elo_rating":              elo_t,
        "elo_win_expectancy":      win_exp,
        "elo_rating_delta":        delta,
        "elo_rating_career_peak":  float(t_row.get("elo_rating_career_peak", elo_t)),
        "elo_rating_career_avg":   float(t_row.get("elo_rating_career_avg",  elo_t)),
        "elo_rank":                float(t_row.get("elo_rank", 50)),
        "elo_is_host":             float(t_row.get("elo_is_host", 0)),
        # Group 2 — FIFA
        "fifa_points":             float(t_row.get("fifa_points", 1000)),
        "fifa_points_delta":       float(t_row.get("fifa_points", 1000))
                                   - float(o_row.get("fifa_points", 1000)),
        "fifa_rank_delta":         float(o_row.get("elo_rank", 50))
                                   - float(t_row.get("elo_rank", 50)),
        "fifa_points_4yr_change":  float(t_row.get("fifa_points_4yr_change", 0)),
        "fifa_rank_4yr_change":    float(t_row.get("fifa_rank_4yr_change", 0)),
        "elo_fifa_rank_disagreement": float(t_row.get("elo_fifa_rank_disagreement", 0)),
        # Group 3 — WC historical
        "wc_win_rate_modern":          float(t_row.get("wc_win_rate_modern", 0)),
        "wc_win_rate_knockout_modern": float(t_row.get("wc_win_rate_knockout_modern", 0)),
        "wc_avg_gf_modern":            float(t_row.get("wc_avg_gf_modern", 0)),
        "wc_avg_ga_modern":            float(t_row.get("wc_avg_ga_modern", 0)),
        "wc_gd_per_game_modern":       float(t_row.get("wc_gd_per_game_modern", 0)),
        "wc_clean_sheet_rate_modern":  float(t_row.get("wc_clean_sheet_rate_modern", 0)),
        "wc_tournaments_attended":     float(t_row.get("wc_tournaments_attended", 0)),
        "wc_best_result_encoded":      float(t_row.get("wc_best_result_encoded", 0)),
        "wc_group_vs_knockout_uplift": uplift,
        "wc_debut_modern_flag":        float(t_row.get("wc_debut_modern_flag", 0)),
        # Group 4 — form
        "form_win_rate_last10":     float(t_row.get("form_win_rate_last10", 0.5)),
        "form_avg_gf_last10":       float(t_row.get("form_avg_gf_last10", 1.0)),
        "form_avg_ga_last10":       float(t_row.get("form_avg_ga_last10", 1.0)),
        "form_gd_last10":           float(t_row.get("form_gd_last10", 0)),
        "form_clean_sheet_rate_last10": float(t_row.get("form_clean_sheet_rate_last10", 0)),
        "form_unbeaten_streak_entering": float(t_row.get("form_unbeaten_streak_entering", 0)),
        # Group 5 — tournament (simulated)
        "tourn_pts_md2":                tourn_pts,
        "tourn_gd_md2":                 tourn_gd,
        "tourn_gf_md2":                 tourn_gf,
        "tourn_ga_md2":                 tourn_ga,
        "tourn_avg_possession":         float(t_row.get("tourn_avg_possession", 50)),
        "tourn_avg_sot":                float(t_row.get("tourn_avg_sot", 4)),
        "tourn_sot_conceded":           float(t_row.get("tourn_sot_conceded", 4)),
        "tourn_shot_conversion_rate":   float(t_row.get("tourn_shot_conversion_rate", 0.1)),
        "tourn_yellow_cards_md2":       float(t_row.get("tourn_yellow_cards_md2", 2)),
        "tourn_formation_changed":      float(t_row.get("tourn_formation_changed", 0)),
        "has_full_tactical_md2":        float(t_row.get("has_full_tactical_md2", 1)),
        # Group 6 — shootout
        "shootout_win_rate_alltime":    float(t_row.get("shootout_win_rate_alltime", 0.5)),
        "shootout_appearances_total":   float(t_row.get("shootout_appearances_total", 0)),
        "shootout_win_rate_wc_only":    float(t_row.get("shootout_win_rate_wc_only", np.nan)),
        "shootout_first_shooter_advantage": float(t_row.get("shootout_first_shooter_advantage", 0.5)),
        "shootout_naive_flag":          float(t_row.get("shootout_naive_flag", 0)),
        # Group 7 — stage context
        "stage_order":              stage_order,
        "is_knockout":              1.0,
        "ko_temperament_interaction": ko_interaction,
        # Matchup deltas
        "tourn_pts_delta":         tourn_pts - opp_pts,
        "tourn_gd_delta":          tourn_gd  - opp_gd,
        "confederation":           str(t_row.get("confederation", "")),
    }

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

class _SingleRun:
    """Executes one complete tournament simulation.

    This class is intended to be called from MonteCarloEngine and is not part
    of the public API. It mutates its SimRun object in-place.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        md3_matches: list[MD3Match],
        md3_proba: dict[int, np.ndarray],
        frozen_standings: dict[str, list[dict]],
        group_context: dict[str, dict[str, bool]],
        ensemble,
        team_features_df: pd.DataFrame,
        annex_c: AnnexCLookup,
        fifa_ranks: dict[str, float],
    ) -> None:
        self.rng              = rng
        self.md3_matches      = md3_matches
        self.md3_proba        = md3_proba
        self.frozen_standings = frozen_standings
        self.group_context    = group_context
        self.ensemble         = ensemble
        self.team_features_df = team_features_df
        self.annex_c          = annex_c
        self.fifa_ranks       = fifa_ranks

    def execute(self) -> SimRun:
        result = SimRun()

        # Step 1 + 2: Simulate MD3 and compute final group standings
        group_entries = self._simulate_md3(result)

        # Step 3: Rank third-placed teams, select 8 best
        third_place_entries = [
            standings[2]   # index 2 = 3rd place (0-indexed)
            for standings in group_entries.values()
        ]
        best_8 = _rank_third_placed(third_place_entries, self.fifa_ranks)
        result.third_place_qualifiers = [e.team for e in best_8]
        qualifying_groups = frozenset(e.group for e in best_8)

        # Record stage reached: group stage (everyone starts here)
        for group, standings in group_entries.items():
            for entry in standings:
                result.stage_reached[entry.team] = STAGE_GROUP

        # Step 4: Resolve Annex C bracket
        third_place_group_assignment = self.annex_c.resolve(qualifying_groups)
        # third_place_group_assignment: {match_id: group_letter}
        # Map group_letter → team (from best_8)
        group_to_third_team = {e.group: e.team for e in best_8}

        # Step 5: Build bracket and simulate knockouts
        self._simulate_bracket(
            result, group_entries, third_place_group_assignment, group_to_third_team
        )

        result.tournament_winner = result.bracket_winners.get(FINAL_MATCH)
        return result

    # ------------------------------------------------------------------
    # MD3 simulation
    # ------------------------------------------------------------------

    def _simulate_md3(self, result: SimRun) -> dict[str, list[GroupEntry]]:
        """Simulate all 24 MD3 matches and return final group standings."""

        # Initialise GroupEntry objects from frozen MD2 standings
        group_entries: dict[str, dict[str, GroupEntry]] = {}

        for group, row_list in self.frozen_standings.items():
            group_entries[group] = {}
            for entry in row_list:
                team = entry["team"]
                pts  = int(entry.get("pts", entry.get("points", 0)))
                gd   = int(entry.get("gd",  entry.get("goal_diff", 0)))
                gf   = int(entry.get("gf",  entry.get("goals_for", 0)))
                ga   = int(entry.get("ga",  entry.get("goals_against", 0)))
                # Head-to-head: {opp: (pts, gd, gf)} from completed MD1+MD2
                h2h_raw = entry.get("h2h", {})
                h2h: dict[str, tuple[int, int, int]] = {}
                for opp, data in h2h_raw.items():
                    if isinstance(data, (list, tuple)) and len(data) >= 3:
                        h2h[opp] = (int(data[0]), int(data[1]), int(data[2]))
                    elif isinstance(data, dict):
                        h2h[opp] = (
                            int(data.get("pts", 0)),
                            int(data.get("gd",  0)),
                            int(data.get("gf",  0)),
                        )
                    else:
                        h2h[opp] = (0, 0, 0)

                group_entries[group][team] = GroupEntry(
                    team=team, group=group,
                    pts=pts, gd=gd, gf=gf, ga=ga, h2h=h2h,
                )

        # Simulate MD3 matches (simultaneously within each group)
        for match in self.md3_matches:
            p = self.md3_proba.get(match.match_id)
            if p is None:
                raise RuntimeError(
                    f"No probability vector for MD3 match {match.match_id} "
                    f"({match.home_team} vs {match.away_team})."
                )

            # Apply group context adjustment
            home_ctx = self.group_context.get(match.home_team, {})
            away_ctx = self.group_context.get(match.away_team, {})
            p_adj = _adjust_proba_for_context(p, home_ctx, away_ctx)

            # Sample outcome: 0=home_win, 1=draw, 2=away_win
            outcome = int(self.rng.choice(3, p=p_adj))

            grp = match.group
            home_e = group_entries[grp][match.home_team]
            away_e = group_entries[grp][match.away_team]

            # Sample a plausible scoreline (for GD/GF/GA tracking)
            gf_h, gf_a = _sample_scoreline(self.rng, outcome)

            # Update standings
            if outcome == 0:    # home win
                home_e.pts += 3
                home_e.h2h.setdefault(match.away_team, (0, 0, 0))
                away_pts_h2h = 0
            elif outcome == 1:  # draw
                home_e.pts += 1
                away_e.pts += 1
                away_pts_h2h = 1
                home_e.h2h.setdefault(match.away_team, (0, 0, 0))
            else:               # away win
                away_e.pts += 3
                home_e.h2h.setdefault(match.away_team, (0, 0, 0))
                away_pts_h2h = 3

            # Home H2H update
            prev_h2h = home_e.h2h.get(match.away_team, (0, 0, 0))
            home_pts_h2h = {0: 3, 1: 1, 2: 0}[outcome]
            home_e.h2h[match.away_team] = (
                prev_h2h[0] + home_pts_h2h,
                prev_h2h[1] + (gf_h - gf_a),
                prev_h2h[2] + gf_h,
            )
            # Away H2H update
            prev_a = away_e.h2h.get(match.home_team, (0, 0, 0))
            away_e.h2h[match.home_team] = (
                prev_a[0] + away_pts_h2h,
                prev_a[1] + (gf_a - gf_h),
                prev_a[2] + gf_a,
            )

            # Update GD/GF/GA
            home_e.gf += gf_h; home_e.ga += gf_a; home_e.gd = home_e.gf - home_e.ga
            away_e.gf += gf_a; away_e.ga += gf_h; away_e.gd = away_e.gf - away_e.ga

        # Apply tiebreakers and return sorted lists
        sorted_standings: dict[str, list[GroupEntry]] = {}
        for group, team_map in group_entries.items():
            sorted_list = _apply_tiebreakers(list(team_map.values()), self.fifa_ranks)
            sorted_standings[group] = sorted_list

        result.final_standings = sorted_standings
        return sorted_standings

    # ------------------------------------------------------------------
    # Knockout bracket simulation
    # ------------------------------------------------------------------

    def _simulate_bracket(
        self,
        result: SimRun,
        group_entries: dict[str, list[GroupEntry]],
        third_place_group_assignment: dict[int, str],
        group_to_third_team: dict[str, str],
    ) -> None:
        """Simulate R32 → Final and populate result.bracket_winners/runners."""

        # Build lookup: group + position → team
        def team_from_slot(slot: str) -> str:
            """Resolve "1X" / "2X" slot to a team name from group X."""
            pos   = int(slot[0])    # 1 or 2
            group = slot[1]
            return group_entries[group][pos - 1].team

        # Build simulated tournament state for feature construction
        tourn_state: dict[str, dict] = {}
        for group, standings in group_entries.items():
            for entry in standings:
                tourn_state[entry.team] = {
                    "pts": entry.pts,
                    "gd":  entry.gd,
                    "gf":  entry.gf,
                    "ga":  entry.ga,
                }

        # Simulate R32 (matches 73–88)
        for match_id, (home_slot, away_slot) in R32_STRUCTURE.items():
            home_team = team_from_slot(home_slot)

            if away_slot is None:
                # Third-place team from Annex C
                group_letter = third_place_group_assignment.get(match_id)
                if group_letter is None:
                    raise RuntimeError(
                        f"Annex C returned no group for R32 match {match_id}."
                    )
                away_team = group_to_third_team.get(group_letter)
                if away_team is None:
                    raise RuntimeError(
                        f"No third-place team found for group {group_letter!r} "
                        f"in this simulation run."
                    )
            else:
                away_team = team_from_slot(away_slot)

            winner, loser = self._play_knockout_match(
                home_team, away_team, STAGE_R32, tourn_state, result
            )
            result.bracket_winners[match_id] = winner
            result.bracket_runners[match_id] = loser
            result.stage_reached[winner] = max(
                result.stage_reached.get(winner, STAGE_GROUP), STAGE_R32
            )

        # Simulate R16 (matches 89–96)
        for match_id, (parent_a, parent_b) in R16_BRACKET.items():
            home_team = result.bracket_winners[parent_a]
            away_team = result.bracket_winners[parent_b]
            winner, loser = self._play_knockout_match(
                home_team, away_team, STAGE_R16, tourn_state, result
            )
            result.bracket_winners[match_id] = winner
            result.bracket_runners[match_id] = loser
            result.stage_reached[winner] = max(
                result.stage_reached.get(winner, STAGE_GROUP), STAGE_R16
            )

        # Simulate QF (matches 97–100)
        for match_id, (parent_a, parent_b) in QF_BRACKET.items():
            home_team = result.bracket_winners[parent_a]
            away_team = result.bracket_winners[parent_b]
            winner, loser = self._play_knockout_match(
                home_team, away_team, STAGE_QF, tourn_state, result
            )
            result.bracket_winners[match_id] = winner
            result.bracket_runners[match_id] = loser
            result.stage_reached[winner] = max(
                result.stage_reached.get(winner, STAGE_GROUP), STAGE_QF
            )

        # Simulate SF (matches 101–102)
        for match_id, (parent_a, parent_b) in SF_BRACKET.items():
            home_team = result.bracket_winners[parent_a]
            away_team = result.bracket_winners[parent_b]
            winner, loser = self._play_knockout_match(
                home_team, away_team, STAGE_SF, tourn_state, result
            )
            result.bracket_winners[match_id] = winner
            result.bracket_runners[match_id] = loser
            result.stage_reached[winner] = max(
                result.stage_reached.get(winner, STAGE_GROUP), STAGE_SF
            )

        # Third-place match (103): runner-ups of SF101 and SF102
        tpm_home = result.bracket_runners[101]
        tpm_away = result.bracket_runners[102]
        tpm_w, tpm_l = self._play_knockout_match(
            tpm_home, tpm_away, STAGE_TPM, tourn_state, result
        )
        result.bracket_winners[THIRD_PLACE_MATCH] = tpm_w
        result.bracket_runners[THIRD_PLACE_MATCH] = tpm_l
        result.stage_reached[tpm_w] = max(
            result.stage_reached.get(tpm_w, STAGE_GROUP), STAGE_TPM
        )
        result.stage_reached[tpm_l] = max(
            result.stage_reached.get(tpm_l, STAGE_GROUP), STAGE_TPM
        )

        # Final (104)
        final_home = result.bracket_winners[101]
        final_away = result.bracket_winners[102]
        final_w, final_l = self._play_knockout_match(
            final_home, final_away, STAGE_FINAL, tourn_state, result
        )
        result.bracket_winners[FINAL_MATCH] = final_w
        result.bracket_runners[FINAL_MATCH] = final_l
        result.stage_reached[final_w] = STAGE_FINAL
        result.stage_reached[final_l] = max(
            result.stage_reached.get(final_l, STAGE_GROUP), STAGE_FINAL
        )

    def _play_knockout_match(
        self,
        home_team: str,
        away_team: str,
        stage_order: int,
        tourn_state: dict[str, dict],
        result: SimRun,
    ) -> tuple[str, str]:
        """Simulate one knockout match and return (winner, loser).

        Process:
          1. Get knockout probabilities from ensemble.
          2. Sample: home_win / draw@90min / away_win.
          3. If draw: sample extra time (EXTRA_TIME_DECIDES = 37.5% chance one
             side wins in ET). If ET doesn't decide, go to shootout.
          4. Return winner and loser.
        """
        X = _build_ko_feature_row(
            home_team, away_team, self.team_features_df, tourn_state, stage_order
        )

        try:
            p = self.ensemble.predict_knockout(X, stage_order=stage_order)[0]
            # p = [P(win@90), P(draw@90), P(loss@90)]
        except Exception as e:
            warnings.warn(
                f"Knockout model failed for {home_team} vs {away_team}: {e}. "
                "Falling back to Elo win expectancy.",
                stacklevel=3,
            )
            p = _elo_fallback_proba(home_team, away_team, self.team_features_df)

        # Sample 90-minute outcome
        p_norm = np.clip(p, 0, 1)
        p_norm = p_norm / p_norm.sum()
        outcome_90 = int(self.rng.choice(3, p=p_norm))

        if outcome_90 == 0:      # home wins at 90 min
            return home_team, away_team
        elif outcome_90 == 2:    # away wins at 90 min
            return away_team, home_team
        else:
            # Draw at 90 min → Extra time + possible penalties
            return self._resolve_extra_time(home_team, away_team, result)

    def _resolve_extra_time(
        self,
        home_team: str,
        away_team: str,
        result: SimRun,
    ) -> tuple[str, str]:
        """Resolve a match that is drawn at 90 minutes.

        37.5% of ET draws are decided in extra time (EXTRA_TIME_DECIDES).
        The rest go to penalties. Winner in ET is modelled as a fair 50/50
        (insufficient data to distinguish ET winning probability from 90-min).
        For penalties, use PenaltyModel.
        """
        # Extra time resolution
        if self.rng.random() < 0.375:
            # ET decides: simple 50/50 (equal chance of scoring first)
            if self.rng.random() < 0.5:
                return home_team, away_team
            else:
                return away_team, home_team

        # Penalties
        home_feats = (
            self.team_features_df.loc[home_team]
            if home_team in self.team_features_df.index
            else pd.Series(dtype=float)
        )
        away_feats = (
            self.team_features_df.loc[away_team]
            if away_team in self.team_features_df.index
            else pd.Series(dtype=float)
        )
        p_home_wins = self.ensemble.predict_shootout(home_feats, away_feats)
        p_home_wins = float(np.clip(p_home_wins, 0.01, 0.99))

        if self.rng.random() < p_home_wins:
            return home_team, away_team
        else:
            return away_team, home_team


def _elo_fallback_proba(
    home_team: str,
    away_team: str,
    team_features_df: pd.DataFrame,
) -> np.ndarray:
    """Compute basic knockout probabilities using only Elo win expectancy."""
    def elo(team: str) -> float:
        if team in team_features_df.index:
            v = team_features_df.loc[team].get("elo_rating", 1500.0)
            return float(v) if not pd.isna(v) else 1500.0
        return 1500.0

    delta = elo(home_team) - elo(away_team)
    p_win = 1.0 / (1.0 + 10.0 ** (-delta / 400.0))
    # [P(win@90), P(draw@90), P(loss@90)]
    p_draw = DRAW_RATE_90MIN
    p_win_adj  = p_win  * (1 - p_draw)
    p_loss_adj = (1 - p_win) * (1 - p_draw)
    return np.array([p_win_adj, p_draw, p_loss_adj])


def _sample_scoreline(rng: np.random.Generator, outcome: int) -> tuple[int, int]:
    """Sample a plausible goals (home, away) pair given a match outcome.

    Uses historical WC average of ~2.6 goals per game (2026 group stage).
    This is only used for GD/GF/GA tracking — not for qualification logic.

    Parameters
    ----------
    outcome : 0 = home win, 1 = draw, 2 = away win.

    Returns
    -------
    (home_goals, away_goals) — non-negative integers consistent with outcome.
    """
    # Sample total goals from a Poisson distribution (mean ~2.5)
    total = int(rng.poisson(2.5))
    total = max(total, 1 if outcome != 1 else 2)   # ensure at least 1 goal in non-draws

    if outcome == 1:  # draw
        half = total // 2
        return half, half
    elif outcome == 0:  # home win
        if total < 2:
            return 1, 0
        away = int(rng.integers(0, total))
        home = total - away
        if home <= away:
            home, away = away + 1, max(0, away - 1)
        return home, away
    else:  # away win
        if total < 2:
            return 0, 1
        home = int(rng.integers(0, total))
        away = total - home
        if away <= home:
            away, home = home + 1, max(0, home - 1)
        return home, away


# ---------------------------------------------------------------------------
# Monte Carlo Engine — public API
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """Run N complete tournament simulations and aggregate results.

    Parameters
    ----------
    ensemble          : Fitted WCForecastEnsemble from models.py.
    team_features_df  : Pre-tournament feature table indexed by canonical team name.
                        Produced by features.build_team_features() or
                        features.build_feature_table().
    md3_matches       : List of 24 MD3Match objects (from build_md3_schedule()).
    md3_proba         : {match_id: np.ndarray shape (3,)} — [P(home_win), P(draw), P(away_win)]
                        Pre-computed from model for each MD3 match. The caller is
                        responsible for applying the Bayesian Layer 2 update before
                        passing these probabilities.
    frozen_standings  : {group: list[dict]} — each dict has keys team, pts, gd, gf, ga, h2h.
                        Represents the verified MD2 snapshot (the freeze-point truth).
    annex_c           : AnnexCLookup loaded from third_place_annex_c.csv.
    fifa_ranks        : {team: float} — FIFA rank for tiebreaker use. Lower = better.
                        Obtained from the DS10/DS11 feature table.

    Usage
    -----
        engine = MonteCarloEngine(...)
        results = engine.run(n=10_000, seed=42)
        results.to_csv("outputs/win_probabilities.csv")
        print(results.to_dataframe().head(10))
    """

    def __init__(
        self,
        ensemble,
        team_features_df: pd.DataFrame,
        md3_matches: list[MD3Match],
        md3_proba: dict[int, np.ndarray],
        frozen_standings: dict[str, list[dict]],
        annex_c: AnnexCLookup,
        fifa_ranks: dict[str, float],
        n_simulations: int = 10_000,
    ) -> None:
        self.ensemble         = ensemble
        self.team_features_df = team_features_df
        self.md3_matches      = md3_matches
        self.md3_proba        = md3_proba
        self.frozen_standings = frozen_standings
        self.annex_c          = annex_c
        self.fifa_ranks       = fifa_ranks
        self.n_simulations    = n_simulations

        # Validate inputs
        self._validate()

        # Pre-compute group context flags (do NOT change between runs)
        self.group_context = _compute_group_context(frozen_standings)

    def _validate(self) -> None:
        if len(self.md3_matches) != 24:
            raise ValueError(f"Expected 24 MD3 matches, got {len(self.md3_matches)}.")
        if len(self.md3_proba) != 24:
            raise ValueError(f"Expected 24 MD3 probability vectors, got {len(self.md3_proba)}.")
        for match in self.md3_matches:
            mid = match.match_id
            if mid not in self.md3_proba:
                raise ValueError(f"No probability vector for MD3 match {mid}.")
            p = self.md3_proba[mid]
            if not np.isclose(p.sum(), 1.0, atol=1e-3):
                raise ValueError(
                    f"MD3 match {mid} probability vector does not sum to 1.0: {p}"
                )
        if len(self.frozen_standings) != 12:
            raise ValueError(
                f"Expected 12 groups in frozen_standings, got {len(self.frozen_standings)}."
            )
        if not isinstance(self.team_features_df.index, pd.Index):
            raise TypeError("team_features_df must have a named index (canonical team name).")

    def run(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> "SimulationResults":
        """Execute N complete simulation runs.

        Parameters
        ----------
        n    : number of simulations (defaults to self.n_simulations).
        seed : random seed for full reproducibility. Each run uses a
               derived sub-seed to ensure independence.

        Returns
        -------
        SimulationResults with win probabilities, stage reach distributions,
        90% confidence intervals, and top-5 final matchup counts.

        Notes
        -----
        - Each run uses a fresh np.random.Generator spawned from the root SeedSequence.
          This guarantees reproducibility regardless of run order.
        - The total computation is O(N × 31 × model_predict). With N = 10,000 and
          a fast XGBoost model (~0.5 ms per batch), total time ≈ 5–10 minutes.
        """
        n = n or self.n_simulations
        seed_sequence = np.random.SeedSequence(seed)
        child_seeds   = seed_sequence.spawn(n)

        # Accumulators
        win_counts:   dict[str, int] = {t: 0 for t in CANONICAL_48}
        stage_counts: dict[str, dict[int, int]] = {
            t: {s: 0 for s in range(1, 8)} for t in CANONICAL_48
        }
        win_per_run:  dict[str, list[float]] = {t: [] for t in CANONICAL_48}
        final_pairs:  dict[tuple[str, str], int] = {}

        print(f"Starting {n:,} Monte Carlo simulations (seed={seed})...")

        for i, child_seed in enumerate(child_seeds):
            rng = np.random.default_rng(child_seed)
            run = _SingleRun(
                rng              = rng,
                md3_matches      = self.md3_matches,
                md3_proba        = self.md3_proba,
                frozen_standings = self.frozen_standings,
                group_context    = self.group_context,
                ensemble         = self.ensemble,
                team_features_df = self.team_features_df,
                annex_c          = self.annex_c,
                fifa_ranks       = self.fifa_ranks,
            )
            try:
                sim_result = run.execute()
            except Exception as e:
                warnings.warn(f"Simulation run {i} failed: {e}", stacklevel=2)
                continue

            # Record win
            winner = sim_result.tournament_winner
            if winner:
                win_counts[winner] = win_counts.get(winner, 0) + 1

            # Record stage reached
            for team, stage in sim_result.stage_reached.items():
                if team in stage_counts:
                    stage_counts[team][stage] = stage_counts[team].get(stage, 0) + 1

            # Record per-run win indicator (for CI bootstrap)
            for team in CANONICAL_48:
                win_per_run[team].append(1.0 if team == winner else 0.0)

            # Record final matchup
            if FINAL_MATCH in sim_result.bracket_winners and FINAL_MATCH in sim_result.bracket_runners:
                f_w = sim_result.bracket_winners[FINAL_MATCH]
                f_l = sim_result.bracket_runners[FINAL_MATCH]
                pair = tuple(sorted([f_w, f_l]))
                final_pairs[pair] = final_pairs.get(pair, 0) + 1

            if (i + 1) % 1000 == 0:
                print(f"  Completed {i + 1:,} / {n:,} runs...")

        print(f"Simulation complete. {n:,} runs finished.")
        return self._aggregate(n, win_counts, stage_counts, win_per_run, final_pairs)

    def _aggregate(
        self,
        n: int,
        win_counts:   dict[str, int],
        stage_counts: dict[str, dict[int, int]],
        win_per_run:  dict[str, list[float]],
        final_pairs:  dict[tuple[str, str], int],
    ) -> SimulationResults:
        """Compute final statistics from raw run accumulators."""
        results = SimulationResults(n_simulations=n)

        for team in CANONICAL_48:
            results.win_prob[team] = win_counts.get(team, 0) / n

            results.stage_probs[team] = {
                s: stage_counts.get(team, {}).get(s, 0) / n
                for s in range(1, 8)
            }

            # 90% CI via bootstrap (5th and 95th percentile of win_per_run means)
            run_wins = np.array(win_per_run.get(team, []))
            if len(run_wins) >= 100:
                ci_lo, ci_hi = _bootstrap_ci(run_wins, n_boot=1000, ci=0.90)
                results.ci_90[team] = (float(ci_lo), float(ci_hi))
            else:
                p = results.win_prob[team]
                std = np.sqrt(p * (1 - p) / max(n, 1))
                results.ci_90[team] = (
                    max(0.0, p - 1.645 * std),
                    min(1.0, p + 1.645 * std),
                )

        # Top-5 most common final matchups
        results.final_pair_counts = final_pairs
        results.most_common_final = sorted(
            [(a, b, cnt) for (a, b), cnt in final_pairs.items()],
            key=lambda x: -x[2],
        )[:5]

        return results


def _bootstrap_ci(
    x: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.90,
    seed: int = 0,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean of x."""
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(x, size=len(x), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return lo, hi


# ---------------------------------------------------------------------------
# MD3 probability computation
# ---------------------------------------------------------------------------

def compute_md3_proba(
    md3_matches: list[MD3Match],
    team_features_df: pd.DataFrame,
    ensemble,
    updater=None,
) -> dict[int, np.ndarray]:
    """Compute Layer 1 + Layer 2 outcome probabilities for all 24 MD3 matches.

    For each MD3 match, produces a (3,) vector [P(home_win), P(draw), P(away_win)]
    using the fitted WCForecastEnsemble and (optionally) the Bayesian tournament
    updater (Layer 2).

    The Bayesian update uses in-tournament features (tourn_pts_md2, tourn_gd_md2)
    to shift the pre-tournament prior based on MD1+MD2 performance.

    Parameters
    ----------
    md3_matches       : List of 24 MD3Match objects.
    team_features_df  : Pre-tournament feature table indexed by team name.
    ensemble          : Fitted WCForecastEnsemble (provides Layer 1 + Layer 3).
    updater           : Optional BayesianTournamentUpdater (Layer 2). If None,
                        the raw Layer 1 probabilities are used.

    Returns
    -------
    {match_id: np.ndarray([P(home_win), P(draw), P(away_win)])}

    Notes
    -----
    The group stage model outputs [P(LOSS), P(DRAW), P(WIN)] from the home team's
    perspective. This function converts to [P(home_win), P(draw), P(away_win)].
    """
    proba: dict[int, np.ndarray] = {}

    for match in md3_matches:
        home = match.home_team
        away = match.away_team

        # Build feature vector for the home team (opponent = away team)
        X = _build_ko_feature_row(
            team=home, opponent=away,
            team_features_df=team_features_df,
            simulated_tourn_state={},   # pre-MD3: use frozen features only
            stage_order=STAGE_GROUP,
        )
        # Override is_knockout for group stage
        X["is_knockout"] = 0.0

        try:
            raw_p = ensemble.predict_group_stage(X)[0]
            # raw_p = [P(LOSS), P(DRAW), P(WIN)] from home perspective
        except Exception as e:
            warnings.warn(
                f"Group stage model failed for {home} vs {away}: {e}. "
                "Falling back to uniform.",
                stacklevel=2,
            )
            raw_p = np.array([0.333, 0.334, 0.333])

        # Apply Bayesian Layer 2 update if updater provided
        if updater is not None:
            home_feats = (
                team_features_df.loc[home]
                if home in team_features_df.index
                else pd.Series(dtype=float)
            )
            # Elo expected pts: 3 × P(win at Elo) + 1 × P(draw at Elo)
            elo_win_exp = float(home_feats.get("elo_win_expectancy", 0.5))
            elo_exp_pts = 3.0 * elo_win_exp + 1.0 * 0.25  # rough draw rate
            elo_exp_gd  = (elo_win_exp - (1 - elo_win_exp)) * 1.2  # ~1.2 goals margin at home
            raw_p = updater.update_proba(
                raw_p, home_feats, elo_expected_pts=elo_exp_pts * 2,  # over 2 games
                elo_expected_gd=elo_exp_gd * 2,
            )

        # Convert: [P(LOSS), P(DRAW), P(WIN)] → [P(home_win), P(draw), P(away_win)]
        p_home_win = float(raw_p[2])
        p_draw     = float(raw_p[1])
        p_away_win = float(raw_p[0])

        p_vec = np.array([p_home_win, p_draw, p_away_win])
        p_vec = np.clip(p_vec, 0, 1)
        s = p_vec.sum()
        if s > 0:
            p_vec /= s

        proba[match.match_id] = p_vec

    return proba


# ---------------------------------------------------------------------------
# Convenience — build frozen_standings dict from standings.py output
# ---------------------------------------------------------------------------

def standings_to_simulation_input(
    computed_standings: dict,
) -> tuple[dict[str, list[dict]], dict[str, float]]:
    """Convert standings.py output into the format expected by MonteCarloEngine.

    Parameters
    ----------
    computed_standings : output of standings.build_standings_from_raw() or
                         standings.compute_group_standings(). Expected structure:
                         {group: [{"team": str, "pts": int, "gd": int,
                                    "gf": int, "ga": int, "h2h": dict}, ...]}

    Returns
    -------
    frozen_standings : {group: [dict]} ready for MonteCarloEngine.
    group_to_teams   : {group: [team1, team2, team3, team4]} for reference.
    """
    frozen: dict[str, list[dict]] = {}
    group_to_teams: dict[str, list[str]] = {}

    for group, entries in computed_standings.items():
        group_upper = str(group).upper()
        rows = []
        teams = []
        for e in entries:
            # Normalise to expected keys
            row = {
                "team": str(e.get("team", e.get("team_canonical", ""))),
                "pts":  int(e.get("pts",  e.get("points", 0))),
                "gd":   int(e.get("gd",   e.get("goal_diff", 0))),
                "gf":   int(e.get("gf",   e.get("goals_for", 0))),
                "ga":   int(e.get("ga",   e.get("goals_against", 0))),
                "h2h":  dict(e.get("h2h", {})),
            }
            rows.append(row)
            teams.append(row["team"])
        frozen[group_upper] = rows
        group_to_teams[group_upper] = teams

    return frozen, group_to_teams


# ---------------------------------------------------------------------------
# Freeze-date guard wrapper
# ---------------------------------------------------------------------------

def assert_simulation_inputs_frozen(
    team_features_df: pd.DataFrame,
    md3_proba: dict[int, np.ndarray],
) -> None:
    """Assert that simulation inputs do not contain post-freeze data.

    Checks:
      1. team_features_df has no 'match_date' column with dates > freeze date.
      2. md3_proba probabilities are all in [0, 1] and sum to 1.
      3. team_features_df index contains only canonical team names.

    Raises
    ------
    LeakageError  if any post-freeze data is detected.
    ValueError    if probability vectors are malformed.
    """
    FREEZE_DATE = "2026-06-23"

    # Check team names
    for team in team_features_df.index:
        if team not in CANONICAL_48:
            raise LeakageError(
                f"Non-canonical team name {team!r} found in team_features_df index. "
                "Run canonicalize() on the index before passing to the simulator.",
                violation_type="NAME_ERROR",
                rule="L0",
            )

    # Check for date columns
    for col in ["match_date", "date"]:
        if col in team_features_df.columns:
            check_freeze_date(team_features_df, col, FREEZE_DATE)

    # Validate probability vectors
    for match_id, p in md3_proba.items():
        p_arr = np.asarray(p, dtype=float)
        if len(p_arr) != 3:
            raise ValueError(
                f"MD3 match {match_id}: probability vector must have length 3, "
                f"got {len(p_arr)}."
            )
        if not np.isclose(p_arr.sum(), 1.0, atol=1e-3):
            raise ValueError(
                f"MD3 match {match_id}: probabilities sum to {p_arr.sum():.4f}, "
                "expected 1.0."
            )
        if (p_arr < 0).any() or (p_arr > 1).any():
            raise ValueError(
                f"MD3 match {match_id}: probabilities out of [0, 1] range: {p_arr}."
            )


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def run_tournament_simulation(
    ensemble,
    team_features_df: pd.DataFrame,
    ds16: pd.DataFrame,
    ds17: pd.DataFrame,
    frozen_standings: dict[str, list[dict]],
    annex_c_path: str,
    fifa_ranks: dict[str, float],
    updater=None,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> SimulationResults:
    """Run the complete FIFA WC 2026 Monte Carlo simulation from frozen inputs.

    This is the single entry point for end-to-end simulation. It:
      1. Loads the Annex C lookup.
      2. Builds the MD3 match schedule from DS16/DS17.
      3. Computes MD3 outcome probabilities (Layer 1 + optional Layer 2).
      4. Validates all inputs for freeze-date compliance.
      5. Runs N Monte Carlo simulations.
      6. Returns aggregated SimulationResults.

    Parameters
    ----------
    ensemble          : Fitted WCForecastEnsemble.
    team_features_df  : Feature table indexed by canonical team name.
    ds16              : DS16 matches DataFrame (from archive.zip matches.csv).
    ds17              : DS17 teams DataFrame (from archive.zip teams.csv).
    frozen_standings  : MD2 group standings (from standings_to_simulation_input()).
    annex_c_path      : Path to third_place_annex_c.csv.
    fifa_ranks        : {team: FIFA rank float}.
    updater           : Optional BayesianTournamentUpdater for Layer 2.
    n_simulations     : Number of Monte Carlo runs (default 10,000).
    seed              : Global random seed.

    Returns
    -------
    SimulationResults — call .to_dataframe() or .to_csv(path) for output.
    """
    print(f"Loading Annex C from {annex_c_path} ...")
    annex_c = AnnexCLookup.from_csv(annex_c_path)

    print("Building MD3 schedule ...")
    md3_matches = build_md3_schedule(ds16, ds17)

    print("Computing MD3 probabilities (Layer 1 + Layer 2) ...")
    md3_proba = compute_md3_proba(md3_matches, team_features_df, ensemble, updater)

    print("Validating freeze-date compliance ...")
    assert_simulation_inputs_frozen(team_features_df, md3_proba)

    print("Initialising Monte Carlo engine ...")
    engine = MonteCarloEngine(
        ensemble         = ensemble,
        team_features_df = team_features_df,
        md3_matches      = md3_matches,
        md3_proba        = md3_proba,
        frozen_standings = frozen_standings,
        annex_c          = annex_c,
        fifa_ranks       = fifa_ranks,
        n_simulations    = n_simulations,
    )

    return engine.run(n=n_simulations, seed=seed)

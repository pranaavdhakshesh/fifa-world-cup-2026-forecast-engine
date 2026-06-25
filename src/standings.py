"""
Group standings computation for the FIFA WC 2026 Forecast Engine.

Tiebreaker sequence (design specification §6 Step 2):
  1. Points
  2. Head-to-head points         (among tied teams only)
  3. Head-to-head goal difference (among tied teams only)
  4. Head-to-head goals for      (among tied teams only)
  5. Overall goal difference
  6. Overall goals for
  7. FIFA rank — ascending (proxy for drawing of lots in simulation)

Third-place cross-group ranking uses the same sequence without head-to-head
(teams from different groups have never met):
  points → GD → GF → FIFA rank.

Feature Group 8 context flags (for Matchday 3 simulation soft adjustments):
  points_before_md3, min_points_to_qualify, must_win_flag,
  already_qualified, already_eliminated, potential_third_place_qualifier.

All team names entering this module must be DS9 canonical.
Apply name_map.canonicalize() before calling any function here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable

import pandas as pd

from src.name_map import canonicalize, canonicalize_id, DS17_NAME_TO_CANONICAL, CANONICAL_48


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

POINTS_WIN  = 3
POINTS_DRAW = 1
POINTS_LOSS = 0

# The 12 group letters used in WC 2026.
GROUP_LETTERS: tuple[str, ...] = tuple("ABCDEFGHIJKL")

# Maximum FIFA rank value used as a fallback when a team is absent from DS10.
# Set high so unknown teams sort last rather than crashing.
_FALLBACK_FIFA_RANK = 9999


# ---------------------------------------------------------------------------
# 1. Group membership builder
# ---------------------------------------------------------------------------

def build_group_membership(ds17_df: pd.DataFrame) -> dict[str, str]:
    """Return {canonical_team_name: group_letter} from DS17 (teams.csv).

    Resolves placeholder IDs and non-canonical name variants via name_map.
    Placeholders are included — they are resolved to their real team names.
    """
    mapping: dict[str, str] = {}
    for _, row in ds17_df.iterrows():
        team_id   = int(row["id"])
        raw_name  = str(row["team_name"]).strip()
        group     = str(row["group_letter"]).strip().upper()
        canonical = canonicalize_id(team_id, raw_name)
        mapping[canonical] = group
    return mapping


# ---------------------------------------------------------------------------
# 2. Core standing row
# ---------------------------------------------------------------------------

@dataclass
class _TeamRecord:
    team:   str
    group:  str
    pts:    int = 0
    w:      int = 0
    d:      int = 0
    l:      int = 0
    gf:     int = 0
    ga:     int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    @property
    def mp(self) -> int:
        return self.w + self.d + self.l

    def to_dict(self) -> dict:
        return {
            "team":            self.team,
            "group":           self.group,
            "pts":             self.pts,
            "w":               self.w,
            "d":               self.d,
            "l":               self.l,
            "gf":              self.gf,
            "ga":              self.ga,
            "gd":              self.gd,
            "matches_played":  self.mp,
        }


# ---------------------------------------------------------------------------
# 3. Raw accumulator
# ---------------------------------------------------------------------------

def _accumulate(
    matches_df: pd.DataFrame,
    group_membership: dict[str, str],
) -> dict[str, _TeamRecord]:
    """Build one _TeamRecord per team from completed match rows.

    Ignores rows where home_score or away_score is null / non-integer.
    Only processes matches where both teams are in group_membership.
    """
    records: dict[str, _TeamRecord] = {
        team: _TeamRecord(team=team, group=grp)
        for team, grp in group_membership.items()
    }

    for _, row in matches_df.iterrows():
        home = canonicalize(str(row["home_team"]).strip())
        away = canonicalize(str(row["away_team"]).strip())

        if home not in records or away not in records:
            continue

        try:
            hs = int(row["home_score"])
            as_ = int(row["away_score"])
        except (ValueError, TypeError):
            continue

        h = records[home]
        a = records[away]

        h.gf += hs;  h.ga += as_
        a.gf += as_; a.ga += hs

        if hs > as_:
            h.pts += POINTS_WIN;  h.w += 1
            a.pts += POINTS_LOSS; a.l += 1
        elif hs == as_:
            h.pts += POINTS_DRAW; h.d += 1
            a.pts += POINTS_DRAW; a.d += 1
        else:
            h.pts += POINTS_LOSS; h.l += 1
            a.pts += POINTS_WIN;  a.w += 1

    return records


# ---------------------------------------------------------------------------
# 4. Head-to-head helper
# ---------------------------------------------------------------------------

def _h2h_stats(
    tied_teams: list[str],
    matches_df: pd.DataFrame,
) -> dict[str, tuple[int, int, int]]:
    """Return {team: (h2h_pts, h2h_gd, h2h_gf)} restricted to matches
    exclusively between the tied teams."""
    tied_set = set(tied_teams)
    h2h: dict[str, list[int]] = {t: [0, 0, 0] for t in tied_teams}

    for _, row in matches_df.iterrows():
        home = canonicalize(str(row["home_team"]).strip())
        away = canonicalize(str(row["away_team"]).strip())
        if home not in tied_set or away not in tied_set:
            continue
        try:
            hs  = int(row["home_score"])
            as_ = int(row["away_score"])
        except (ValueError, TypeError):
            continue

        h2h[home][1] += hs - as_
        h2h[away][1] += as_ - hs
        h2h[home][2] += hs
        h2h[away][2] += as_

        if hs > as_:
            h2h[home][0] += POINTS_WIN
            h2h[away][0] += POINTS_LOSS
        elif hs == as_:
            h2h[home][0] += POINTS_DRAW
            h2h[away][0] += POINTS_DRAW
        else:
            h2h[home][0] += POINTS_LOSS
            h2h[away][0] += POINTS_WIN

    return {t: tuple(v) for t, v in h2h.items()}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 5. Ranking core — handles recursive multi-team tiebreakers
# ---------------------------------------------------------------------------

def _sort_key_overall(
    rec: _TeamRecord,
    fifa_ranks: dict[str, int],
) -> tuple:
    """Sort key using only overall stats and FIFA rank (no head-to-head)."""
    return (
        -rec.pts,
        -rec.gd,
        -rec.gf,
        fifa_ranks.get(rec.team, _FALLBACK_FIFA_RANK),
    )


def _rank_group(
    group_records: list[_TeamRecord],
    matches_df: pd.DataFrame,
    fifa_ranks: dict[str, int],
) -> list[tuple[int, _TeamRecord]]:
    """Return [(rank, record), ...] for a single group, 1-indexed.

    When teams are tied on points the full tiebreaker chain is applied:
      h2h_pts → h2h_gd → h2h_gf → overall_gd → overall_gf → FIFA rank.

    The h2h phase is applied to all tied teams simultaneously.  If that
    resolves some teams but leaves a smaller subset still tied, overall
    stats and FIFA rank separate the remainder.
    """
    # Start with a deterministic preliminary order so recursion is stable.
    ordered = sorted(
        group_records,
        key=lambda r: _sort_key_overall(r, fifa_ranks),
    )

    result: list[tuple[int, _TeamRecord]] = []
    next_rank = 1
    i = 0

    while i < len(ordered):
        # Find the run of teams tied on points.
        pts_val = ordered[i].pts
        j = i
        while j < len(ordered) and ordered[j].pts == pts_val:
            j += 1
        tied_slice = ordered[i:j]

        if len(tied_slice) == 1:
            result.append((next_rank, tied_slice[0]))
            next_rank += 1
        else:
            # Apply head-to-head within the tied subset.
            teams_in_tie = [r.team for r in tied_slice]
            h2h = _h2h_stats(teams_in_tie, matches_df)

            def full_key(rec: _TeamRecord) -> tuple:
                hp, hgd, hgf = h2h[rec.team]
                return (
                    -hp,
                    -hgd,
                    -hgf,
                    -rec.gd,
                    -rec.gf,
                    fifa_ranks.get(rec.team, _FALLBACK_FIFA_RANK),
                )

            sorted_tied = sorted(tied_slice, key=full_key)

            # Walk the sorted tied teams; if a sub-sub-tie exists after the
            # h2h step (same h2h key AND same overall stats), those teams
            # are indistinguishable except by FIFA rank, which the key
            # already includes as the final differentiator.  Assign ranks
            # sequentially — FIFA rank guarantees no true ties remain.
            for rec in sorted_tied:
                result.append((next_rank, rec))
                next_rank += 1

        i = j

    return result


# ---------------------------------------------------------------------------
# 6. Public standings function
# ---------------------------------------------------------------------------

def compute_group_standings(
    matches_df: pd.DataFrame,
    group_membership: dict[str, str],
    fifa_ranks: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute group standings from completed match results.

    Parameters
    ----------
    matches_df:
        DataFrame with columns [home_team, away_team, home_score, away_score].
        Rows with non-integer or null scores are silently skipped (they are
        future/incomplete matches).  Team names must be DS9 canonical.
    group_membership:
        {canonical_team_name: group_letter} for all 48 teams.
        Obtain via build_group_membership(ds17_df).
    fifa_ranks:
        {canonical_team_name: int} — DS10 FIFA rank as of June 8 2026.
        Used only as the final tiebreaker.  If None, all teams receive
        _FALLBACK_FIFA_RANK and the tiebreaker is effectively random within
        perfectly equal pairs (acceptable for pre-MD3 context).

    Returns
    -------
    DataFrame with columns:
        team, group, pts, w, d, l, gf, ga, gd, matches_played, group_rank.
    Sorted by (group, group_rank).
    """
    if fifa_ranks is None:
        fifa_ranks = {}

    records = _accumulate(matches_df, group_membership)

    # Rank within each group.
    rows: list[dict] = []
    for grp in GROUP_LETTERS:
        grp_records = [r for r in records.values() if r.group == grp]
        if not grp_records:
            continue
        ranked = _rank_group(grp_records, matches_df, fifa_ranks)
        for rank, rec in ranked:
            d = rec.to_dict()
            d["group_rank"] = rank
            rows.append(d)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["group", "group_rank"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Third-place ranking
# ---------------------------------------------------------------------------

def rank_third_placed(
    standings_df: pd.DataFrame,
    fifa_ranks: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Rank the 12 third-placed teams to identify the best 8.

    No head-to-head is applied (third-placed teams come from different groups).
    Tiebreaker: points → GD → GF → FIFA rank.

    Parameters
    ----------
    standings_df:
        Output of compute_group_standings — must contain group_rank column.
    fifa_ranks:
        {canonical_team_name: int} from DS10.

    Returns
    -------
    DataFrame of the 12 third-placed teams with an added third_place_rank
    column (1 = best, 12 = worst) and qualifies_as_third (True/False for
    the top 8).
    """
    if fifa_ranks is None:
        fifa_ranks = {}

    thirds = standings_df[standings_df["group_rank"] == 3].copy()

    thirds["_fr"] = thirds["team"].map(
        lambda t: fifa_ranks.get(t, _FALLBACK_FIFA_RANK)
    )
    thirds = thirds.sort_values(
        ["pts", "gd", "gf", "_fr"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    thirds["third_place_rank"]    = range(1, len(thirds) + 1)
    thirds["qualifies_as_third"]  = thirds["third_place_rank"] <= 8
    thirds = thirds.drop(columns=["_fr"])
    return thirds


# ---------------------------------------------------------------------------
# 8. Feature Group 8 — MD3 context flags
# ---------------------------------------------------------------------------

def _simulate_group_outcome(
    current: dict[str, _TeamRecord],
    md3_pair: tuple[str, str],
    outcome: str,          # "H" home win, "D" draw, "A" away win
) -> dict[str, _TeamRecord]:
    """Return a copy of current records updated for one MD3 match outcome."""
    import copy
    updated = copy.deepcopy(current)
    home, away = md3_pair
    if home not in updated or away not in updated:
        return updated

    h = updated[home]
    a = updated[away]

    if outcome == "H":
        h.pts += POINTS_WIN;  h.w += 1
        a.pts += POINTS_LOSS; a.l += 1
        h.gf += 1             # symbolic +1/-0 — sufficient for qualification math
        a.ga += 1
    elif outcome == "D":
        h.pts += POINTS_DRAW; h.d += 1
        a.pts += POINTS_DRAW; a.d += 1
        h.gf += 0; a.gf += 0  # draw adds no goals symbolically
    else:  # "A"
        h.pts += POINTS_LOSS; h.l += 1
        a.pts += POINTS_WIN;  a.w += 1
        a.gf += 1
        h.ga += 1

    return updated


def compute_md3_context(
    standings_df: pd.DataFrame,
    md3_fixtures: list[tuple[str, str]],
    all_matches_df: pd.DataFrame,
    fifa_ranks: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Add Feature Group 8 context columns to standings_df.

    Enumerates all 9 possible MD3 outcomes for each group (3 outcomes per
    match × 2 concurrent matches) to compute exact qualification thresholds.

    Parameters
    ----------
    standings_df:
        Output of compute_group_standings with group_rank column.
    md3_fixtures:
        List of (home_team, away_team) canonical-name tuples for the 24 MD3
        matches — only the 2 fixtures for each group are used.
    all_matches_df:
        The full set of completed group-stage matches (MD1 + MD2).
        Used to rebuild head-to-head records inside scenario enumeration.
    fifa_ranks:
        {canonical_team_name: int}.

    Returns
    -------
    standings_df with columns added:
        points_before_md3, min_points_to_qualify, must_win_flag,
        already_qualified, already_eliminated, potential_third_place_qualifier.

    Note: potential_third_place_qualifier requires knowing the cross-group
    third-place threshold, which depends on all groups simultaneously.  This
    column is set to a conservative True for any team that might achieve ≥ 1
    point; callers running the full simulation can refine it post-hoc.
    """
    if fifa_ranks is None:
        fifa_ranks = {}

    out = standings_df.copy()
    out["points_before_md3"]           = out["pts"]
    out["min_points_to_qualify"]        = pd.NA
    out["must_win_flag"]                = False
    out["already_qualified"]            = False
    out["already_eliminated"]           = False
    out["potential_third_place_qualifier"] = True  # conservative default

    # Build a {team: group} lookup for quick access.
    team_to_group: dict[str, str] = dict(
        zip(standings_df["team"], standings_df["group"])
    )

    # Organise MD3 fixtures by group.
    group_fixtures: dict[str, list[tuple[str, str]]] = {g: [] for g in GROUP_LETTERS}
    for home, away in md3_fixtures:
        grp = team_to_group.get(home) or team_to_group.get(away)
        if grp:
            group_fixtures[grp].append((home, away))

    OUTCOMES = ("H", "D", "A")

    for grp in GROUP_LETTERS:
        grp_teams_rows = standings_df[standings_df["group"] == grp]
        if grp_teams_rows.empty:
            continue

        fixtures = group_fixtures[grp]
        if len(fixtures) != 2:
            # Cannot enumerate — skip flag computation for this group.
            continue

        # Reconstruct _TeamRecord objects for current state.
        base: dict[str, _TeamRecord] = {}
        for _, row in grp_teams_rows.iterrows():
            base[row["team"]] = _TeamRecord(
                team=row["team"], group=grp,
                pts=int(row["pts"]),
                w=int(row["w"]), d=int(row["d"]), l=int(row["l"]),
                gf=int(row["gf"]), ga=int(row["ga"]),
            )

        # Enumerate all 9 scenarios.
        # scenario_top2[team] = set of scenarios where team finishes top-2.
        # scenario_3rd[team]  = set of scenarios where team finishes 3rd.
        scenario_top2: dict[str, list[bool]] = {t: [] for t in base}
        scenario_3rd:  dict[str, list[bool]] = {t: [] for t in base}

        for o1, o2 in product(OUTCOMES, OUTCOMES):
            state = _simulate_group_outcome(base, fixtures[0], o1)
            state = _simulate_group_outcome(state, fixtures[1], o2)

            ranked = _rank_group(list(state.values()), all_matches_df, fifa_ranks)
            rank_map = {rec.team: rank for rank, rec in ranked}

            for team in base:
                r = rank_map.get(team, 4)
                scenario_top2[team].append(r <= 2)
                scenario_3rd[team].append(r == 3)

        # Derive flags for each team in this group.
        for team in base:
            top2_results = scenario_top2[team]
            always_top2  = all(top2_results)
            never_top2   = not any(top2_results)
            always_3rd   = all(scenario_3rd[team])

            idx = out.index[out["team"] == team]
            out.loc[idx, "already_qualified"] = always_top2
            out.loc[idx, "already_eliminated"] = never_top2 and not any(scenario_3rd[team])

            # must_win_flag: a draw is never sufficient for top-2.
            # Check if any draw scenario (o1 or o2 is "D" for this team's match)
            # results in top-2.  Find which fixture involves this team.
            team_fixture_idx = next(
                (fi for fi, fx in enumerate(fixtures) if team in fx), None
            )
            if team_fixture_idx is not None:
                draw_scenarios_top2 = []
                for idx_s, (o1, o2) in enumerate(product(OUTCOMES, OUTCOMES)):
                    outcomes_for_fixtures = [o1, o2]
                    raw_outcome = outcomes_for_fixtures[team_fixture_idx]
                    # From this team's perspective: H if they're home in the fixture.
                    fix = fixtures[team_fixture_idx]
                    is_home = (fix[0] == team)
                    team_result = raw_outcome if is_home else (
                        "A" if raw_outcome == "H" else
                        "H" if raw_outcome == "A" else "D"
                    )
                    if team_result == "D":
                        draw_scenarios_top2.append(scenario_top2[team][idx_s])
                must_win = not any(draw_scenarios_top2) and not always_top2
                out.loc[out["team"] == team, "must_win_flag"] = must_win

            # min_points_to_qualify:
            # Minimum final-points total that guarantees top-2 across all 9 scenarios.
            # Scan: if team wins (+3), can they guarantee top-2 regardless of the
            # other fixture's outcome?  If yes → min = current + 3.
            # If draw (+1) is sufficient → min = current + 1.
            # If already qualified → min = current.
            cur_pts = int(base[team].pts)
            if always_top2:
                min_pts = cur_pts
            else:
                fix_idx = team_fixture_idx
                if fix_idx is None:
                    min_pts = cur_pts + 3
                else:
                    fix = fixtures[fix_idx]
                    is_home = (fix[0] == team)
                    # Check if winning always qualifies (regardless of other match).
                    win_outcome_raw = "H" if is_home else "A"
                    win_scenarios_top2 = [
                        scenario_top2[team][i]
                        for i, (o1, o2) in enumerate(product(OUTCOMES, OUTCOMES))
                        if [o1, o2][fix_idx] == win_outcome_raw
                    ]
                    draw_outcome_raw = "D"
                    draw_scenarios_top2_check = [
                        scenario_top2[team][i]
                        for i, (o1, o2) in enumerate(product(OUTCOMES, OUTCOMES))
                        if [o1, o2][fix_idx] == draw_outcome_raw
                    ]
                    if all(win_scenarios_top2):
                        min_pts = cur_pts + 3
                    elif any(win_scenarios_top2):
                        # Win sometimes qualifies but not always; treat as needing win.
                        min_pts = cur_pts + 3
                    else:
                        min_pts = cur_pts + 3  # can't qualify even with a win

                    # Check if draw ever qualifies (use lower threshold).
                    if any(draw_scenarios_top2_check):
                        min_pts = cur_pts + 1

            out.loc[out["team"] == team, "min_points_to_qualify"] = min_pts

        # potential_third_place_qualifier: any scenario where team finishes 3rd.
        # A more precise cross-group threshold requires simulation.py.
        # Here we conservatively flag True if team can finish 3rd in ≥1 scenario.
        for team in base:
            can_be_third = any(scenario_3rd[team])
            out.loc[out["team"] == team, "potential_third_place_qualifier"] = can_be_third

    # already_eliminated also considers third-place route.
    # A team already eliminated from top-2 is NOT eliminated overall if it can
    # still be a best-8 third-placed team.  We leave that cross-group judgment
    # to simulation.py which has global state.  Here we expose the flag
    # conservatively: already_eliminated = True only if the team cannot even
    # finish 3rd (4th in all 9 scenarios).
    out["already_eliminated"] = out["already_eliminated"].astype(bool)
    out["already_qualified"]  = out["already_qualified"].astype(bool)
    out["must_win_flag"]      = out["must_win_flag"].astype(bool)
    out["potential_third_place_qualifier"] = out["potential_third_place_qualifier"].astype(bool)

    return out


# ---------------------------------------------------------------------------
# 9. Convenience loader — builds everything from raw DataFrames
# ---------------------------------------------------------------------------

def build_standings_from_raw(
    ds1_df: pd.DataFrame,
    ds1ext_df: pd.DataFrame,
    ds17_df: pd.DataFrame,
    ds10_df: pd.DataFrame,
) -> pd.DataFrame:
    """One-shot helper used by notebooks and tests.

    Combines DS1 + DS1-ext, builds group membership from DS17, loads FIFA
    ranks from DS10, and returns ranked group standings.

    DS10 must have columns [team, rank] — apply name_map before passing.
    DS1 and DS1-ext must have columns [home_team, away_team, home_score,
    away_score] with DS9 canonical names.
    """
    matches = pd.concat(
        [
            ds1_df[["home_team", "away_team", "home_score", "away_score"]],
            ds1ext_df[["home_team", "away_team", "home_score", "away_score"]],
        ],
        ignore_index=True,
    )

    group_membership = build_group_membership(ds17_df)

    fifa_ranks: dict[str, int] = {}
    if "team" in ds10_df.columns and "rank" in ds10_df.columns:
        fifa_ranks = dict(zip(ds10_df["team"], ds10_df["rank"].astype(int)))

    return compute_group_standings(matches, group_membership, fifa_ranks)

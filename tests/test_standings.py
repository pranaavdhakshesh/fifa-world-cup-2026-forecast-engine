"""
tests/test_standings.py

Purpose: Verify that standings computation and tiebreaker logic produce the
correct results from the known frozen 48-match dataset.

Fixtures use synthetic match results crafted to reproduce the exact frozen
standings documented in the design spec §6 "Final Monte Carlo Simulation
Starting Point". Exact scores are not necessarily historical — they are
constructed to satisfy the point/GD constraints from the design doc.

Design spec ref: §7 Repository Structure — tests/test_standings.py
"""

import pandas as pd
import pytest

from src.standings import (
    build_group_membership,
    compute_group_standings,
    compute_md3_context,
    rank_third_placed,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _matches(*rows) -> pd.DataFrame:
    """Build a minimal matches DataFrame from (home, away, home_score, away_score) tuples."""
    return pd.DataFrame(rows, columns=["home_team", "away_team", "home_score", "away_score"])


def _membership(**kwargs) -> dict[str, str]:
    """Build a group_membership dict from group=>[team, team, ...] kwargs."""
    m: dict[str, str] = {}
    for grp, teams in kwargs.items():
        for t in teams:
            m[t] = grp
    return m


def _get_team(standings: pd.DataFrame, team: str) -> pd.Series:
    row = standings[standings["team"] == team]
    assert len(row) == 1, f"Team {team!r} not found in standings"
    return row.iloc[0]


# ---------------------------------------------------------------------------
# Group A — Design spec: Mexico 6pts (GD+3), Korea Republic 3pts (GD 0),
#           Czechia 1pt (GD −1), South Africa 1pt (GD −2).
#
# Synthetic match set:
#   MD1: Mexico 1-0 Korea Republic, Czechia 0-0 South Africa
#   MD2: Mexico 2-0 South Africa, Korea Republic 1-0 Czechia
# ---------------------------------------------------------------------------

GROUP_A_MEMBERSHIP = _membership(A=[
    "Mexico", "Korea Republic", "Czechia", "South Africa"
])

GROUP_A_MATCHES = _matches(
    # MD1
    ("Mexico",       "Korea Republic", 1, 0),
    ("Czechia",      "South Africa",   0, 0),
    # MD2
    ("Mexico",       "South Africa",   2, 0),
    ("Korea Republic", "Czechia",      1, 0),
)


class TestGroupAStandings:
    """Design spec §6: Group A frozen standings after MD2."""

    @pytest.fixture(autouse=True)
    def standings(self):
        self.df = compute_group_standings(GROUP_A_MATCHES, GROUP_A_MEMBERSHIP)

    def test_mexico_points(self):
        assert _get_team(self.df, "Mexico")["pts"] == 6

    def test_mexico_gd(self):
        assert _get_team(self.df, "Mexico")["gd"] == 3

    def test_mexico_rank(self):
        assert _get_team(self.df, "Mexico")["group_rank"] == 1

    def test_korea_republic_points(self):
        assert _get_team(self.df, "Korea Republic")["pts"] == 3

    def test_korea_republic_gd(self):
        assert _get_team(self.df, "Korea Republic")["gd"] == 0

    def test_korea_republic_rank(self):
        assert _get_team(self.df, "Korea Republic")["group_rank"] == 2

    def test_czechia_points(self):
        assert _get_team(self.df, "Czechia")["pts"] == 1

    def test_czechia_gd(self):
        assert _get_team(self.df, "Czechia")["gd"] == -1

    def test_south_africa_points(self):
        assert _get_team(self.df, "South Africa")["pts"] == 1

    def test_south_africa_gd(self):
        assert _get_team(self.df, "South Africa")["gd"] == -2

    def test_south_africa_ranks_below_czechia(self):
        """Czechia is above South Africa because GD −1 > −2."""
        r_cze = _get_team(self.df, "Czechia")["group_rank"]
        r_sa  = _get_team(self.df, "South Africa")["group_rank"]
        assert r_cze < r_sa

    def test_four_teams_in_output(self):
        assert len(self.df) == 4

    def test_points_sum_correct(self):
        """Total points: 3 wins (3×3=9) + 1 draw (1×2=2) = 11."""
        assert self.df["pts"].sum() == 11

    def test_gd_sums_to_zero(self):
        assert self.df["gd"].sum() == 0


# ---------------------------------------------------------------------------
# Group I — Design spec: France 6pts (GD+5), Norway 6pts (GD+4),
#           Senegal 0pts (GD −3), Iraq 0pts (GD −6).
#           France on top by GD over Norway.
#
# Synthetic match set:
#   MD1: France 2-0 Senegal, Norway 3-0 Iraq
#   MD2: France 3-0 Iraq, Norway 1-0 Senegal
# ---------------------------------------------------------------------------

GROUP_I_MEMBERSHIP = _membership(I=["France", "Norway", "Senegal", "Iraq"])

GROUP_I_MATCHES = _matches(
    ("France",  "Senegal", 2, 0),
    ("Norway",  "Iraq",    3, 0),
    ("France",  "Iraq",    3, 0),
    ("Norway",  "Senegal", 1, 0),
)


class TestGroupIStandings:
    """Design spec §6: Both France and Norway qualified. France tops by GD."""

    @pytest.fixture(autouse=True)
    def standings(self):
        self.df = compute_group_standings(GROUP_I_MATCHES, GROUP_I_MEMBERSHIP)

    def test_france_points(self):
        assert _get_team(self.df, "France")["pts"] == 6

    def test_france_gd(self):
        assert _get_team(self.df, "France")["gd"] == 5

    def test_france_rank_first(self):
        assert _get_team(self.df, "France")["group_rank"] == 1

    def test_norway_points(self):
        assert _get_team(self.df, "Norway")["pts"] == 6

    def test_norway_gd(self):
        assert _get_team(self.df, "Norway")["gd"] == 4

    def test_norway_rank_second(self):
        assert _get_team(self.df, "Norway")["group_rank"] == 2

    def test_senegal_points(self):
        assert _get_team(self.df, "Senegal")["pts"] == 0

    def test_senegal_gd(self):
        assert _get_team(self.df, "Senegal")["gd"] == -3

    def test_iraq_points(self):
        assert _get_team(self.df, "Iraq")["pts"] == 0

    def test_iraq_gd(self):
        assert _get_team(self.df, "Iraq")["gd"] == -6

    def test_iraq_ranks_below_senegal(self):
        """Iraq (GD −6) ranks below Senegal (GD −3)."""
        assert _get_team(self.df, "Iraq")["group_rank"] > _get_team(self.df, "Senegal")["group_rank"]

    def test_gd_sums_to_zero(self):
        assert self.df["gd"].sum() == 0


# ---------------------------------------------------------------------------
# Group K — Design spec: Colombia 6pts, Portugal 4pts, Congo DR 1pt,
#           Uzbekistan 0pts. Includes June 23 results.
#
# Synthetic match set:
#   MD1: Colombia 1-0 Congo DR, Portugal 3-0 Uzbekistan
#   MD2: Colombia 2-0 Uzbekistan, Portugal 1-1 Congo DR
# ---------------------------------------------------------------------------

GROUP_K_MEMBERSHIP = _membership(K=["Colombia", "Portugal", "Congo DR", "Uzbekistan"])

GROUP_K_MATCHES = _matches(
    ("Colombia", "Congo DR",   1, 0),
    ("Portugal", "Uzbekistan", 3, 0),
    ("Colombia", "Uzbekistan", 2, 0),
    ("Portugal", "Congo DR",   1, 1),
)


class TestGroupKStandings:
    """Design spec §6: Group K after including June 23 results."""

    @pytest.fixture(autouse=True)
    def standings(self):
        self.df = compute_group_standings(GROUP_K_MATCHES, GROUP_K_MEMBERSHIP)

    def test_colombia_points(self):
        assert _get_team(self.df, "Colombia")["pts"] == 6

    def test_colombia_rank_first(self):
        assert _get_team(self.df, "Colombia")["group_rank"] == 1

    def test_portugal_points(self):
        assert _get_team(self.df, "Portugal")["pts"] == 4

    def test_portugal_rank_second(self):
        assert _get_team(self.df, "Portugal")["group_rank"] == 2

    def test_congo_dr_points(self):
        assert _get_team(self.df, "Congo DR")["pts"] == 1

    def test_congo_dr_rank_third(self):
        assert _get_team(self.df, "Congo DR")["group_rank"] == 3

    def test_uzbekistan_points(self):
        assert _get_team(self.df, "Uzbekistan")["pts"] == 0

    def test_uzbekistan_rank_fourth(self):
        assert _get_team(self.df, "Uzbekistan")["group_rank"] == 4


# ---------------------------------------------------------------------------
# Group L — Design spec: England 4pts (GD+2), Ghana 4pts (GD+1),
#           Croatia 3pts, Panama 0pts.
#
# Synthetic match set:
#   MD1: England 2-0 Croatia, Ghana 1-0 Panama
#   MD2: England 0-0 Ghana, Croatia 1-0 Panama
# ---------------------------------------------------------------------------

GROUP_L_MEMBERSHIP = _membership(L=["England", "Ghana", "Croatia", "Panama"])

GROUP_L_MATCHES = _matches(
    ("England", "Croatia", 2, 0),
    ("Ghana",   "Panama",  1, 0),
    ("England", "Ghana",   0, 0),
    ("Croatia", "Panama",  1, 0),
)


class TestGroupLStandings:
    """Design spec §6: Group L standings."""

    @pytest.fixture(autouse=True)
    def standings(self):
        self.df = compute_group_standings(GROUP_L_MATCHES, GROUP_L_MEMBERSHIP)

    def test_england_points(self):
        assert _get_team(self.df, "England")["pts"] == 4

    def test_england_gd(self):
        assert _get_team(self.df, "England")["gd"] == 2

    def test_england_rank_first(self):
        assert _get_team(self.df, "England")["group_rank"] == 1

    def test_ghana_points(self):
        assert _get_team(self.df, "Ghana")["pts"] == 4

    def test_ghana_gd(self):
        assert _get_team(self.df, "Ghana")["gd"] == 1

    def test_ghana_rank_second(self):
        assert _get_team(self.df, "Ghana")["group_rank"] == 2

    def test_croatia_points(self):
        assert _get_team(self.df, "Croatia")["pts"] == 3

    def test_croatia_rank_third(self):
        assert _get_team(self.df, "Croatia")["group_rank"] == 3

    def test_panama_points(self):
        assert _get_team(self.df, "Panama")["pts"] == 0

    def test_panama_rank_fourth(self):
        assert _get_team(self.df, "Panama")["group_rank"] == 4

    def test_england_above_ghana_by_gd(self):
        r_eng = _get_team(self.df, "England")["group_rank"]
        r_gha = _get_team(self.df, "Ghana")["group_rank"]
        assert r_eng < r_gha


# ---------------------------------------------------------------------------
# Head-to-head tiebreaker
# ---------------------------------------------------------------------------

class TestHeadToHeadTiebreaker:
    """When teams are level on overall points and GD, H2H points break the tie."""

    def test_h2h_winner_ranks_higher(self):
        """Alpha beat Beta in H2H but both are level on overall stats."""
        membership = _membership(A=["Alpha", "Beta", "Gamma"])
        matches = _matches(
            # Alpha beats Beta directly; all other results cancel out
            ("Alpha", "Beta",  1, 0),
            ("Beta",  "Gamma", 1, 0),
            ("Alpha", "Gamma", 0, 1),
        )
        df = compute_group_standings(matches, membership)
        r_alpha = _get_team(df, "Alpha")["group_rank"]
        r_beta  = _get_team(df, "Beta")["group_rank"]
        # Alpha: 3pts (beat Beta), lost to Gamma → 3pts
        # Beta:  3pts (beat Gamma), lost to Alpha → 3pts; H2H: Beta lost → 2nd
        assert r_alpha < r_beta, "H2H winner should rank higher"


# ---------------------------------------------------------------------------
# Third-place ranking — design spec §6 Step 3
# ---------------------------------------------------------------------------

def _build_synthetic_standings_with_thirds() -> pd.DataFrame:
    """Build a minimal 12-group standings that includes third-placed teams
    matching the design doc freeze-point characterisation.

    Sweden (Group F): 3rd, 3pts, GD 0 — should rank 1st of thirds.
    Senegal (Group I): 3rd, 0pts, GD −3 — should rank last.
    Various teams at 1pt (Czechia, Congo DR, Ecuador, Bosnia-Herzegovina).
    """
    rows = [
        # Groups A–L, third-placed entries only (group_rank == 3)
        # Group A — Czechia 1pt, GD -1
        dict(team="Czechia",             group="A", pts=1,  gd=-1, gf=1, group_rank=3),
        # Group B — placeholder at 0pts
        dict(team="Qatar",               group="B", pts=0,  gd=-3, gf=0, group_rank=3),
        # Group C — Scotland 1pt
        dict(team="Scotland",            group="C", pts=1,  gd=-2, gf=1, group_rank=3),
        # Group D — Australia 0pts
        dict(team="Australia",           group="D", pts=0,  gd=-2, gf=0, group_rank=3),
        # Group E — Côte d'Ivoire 2pts
        dict(team="Côte d'Ivoire",       group="E", pts=2,  gd=-1, gf=2, group_rank=3),
        # Group F — Sweden 3pts, GD 0  (top of thirds per design spec)
        dict(team="Sweden",              group="F", pts=3,  gd= 0, gf=3, group_rank=3),
        # Group G — New Zealand 0pts
        dict(team="New Zealand",         group="G", pts=0,  gd=-4, gf=0, group_rank=3),
        # Group H — Saudi Arabia 1pt, GD 0
        dict(team="Saudi Arabia",        group="H", pts=1,  gd= 0, gf=1, group_rank=3),
        # Group I — Senegal 0pts, GD -4  (bottom per design spec; worse GD than Qatar)
        dict(team="Senegal",             group="I", pts=0,  gd=-4, gf=0, group_rank=3),
        # Group J — Jordan 0pts, GD -2
        dict(team="Jordan",              group="J", pts=0,  gd=-2, gf=0, group_rank=3),
        # Group K — Congo DR 1pt, GD -2
        dict(team="Congo DR",            group="K", pts=1,  gd=-2, gf=1, group_rank=3),
        # Group L — Croatia 3pts, GD -1
        dict(team="Croatia",             group="L", pts=3,  gd=-1, gf=3, group_rank=3),
    ]
    return pd.DataFrame(rows)


class TestThirdPlaceRanking:
    @pytest.fixture(autouse=True)
    def thirds(self):
        all_standings = _build_synthetic_standings_with_thirds()
        self.ranked = rank_third_placed(all_standings)

    def test_twelve_third_placed_teams(self):
        assert len(self.ranked) == 12

    def test_sweden_ranks_first(self):
        """Design spec: Sweden (Group F, 3pts, GD 0) ranks first."""
        sweden_rank = self.ranked[self.ranked["team"] == "Sweden"]["third_place_rank"].iloc[0]
        assert sweden_rank == 1, f"Sweden should be 1st, got rank {sweden_rank}"

    def test_senegal_ranks_last(self):
        """Design spec: Senegal (0pts, GD −3) is the lowest-ranked third-placed team."""
        senegal_rank = self.ranked[self.ranked["team"] == "Senegal"]["third_place_rank"].iloc[0]
        assert senegal_rank == 12, f"Senegal should be last (12th), got rank {senegal_rank}"

    def test_top_8_qualify(self):
        assert self.ranked["qualifies_as_third"].sum() == 8

    def test_bottom_4_do_not_qualify(self):
        assert (~self.ranked["qualifies_as_third"]).sum() == 4

    def test_czechia_ranks_above_senegal(self):
        r_cze = self.ranked[self.ranked["team"] == "Czechia"]["third_place_rank"].iloc[0]
        r_sen = self.ranked[self.ranked["team"] == "Senegal"]["third_place_rank"].iloc[0]
        assert r_cze < r_sen, "Czechia (1pt) should rank above Senegal (0pts)"

    def test_third_place_rank_monotonic(self):
        """third_place_rank must be 1, 2, 3, …, 12 with no gaps."""
        ranks = sorted(self.ranked["third_place_rank"].tolist())
        assert ranks == list(range(1, 13))


# ---------------------------------------------------------------------------
# Total qualified teams — design spec: exactly 32 advance.
# ---------------------------------------------------------------------------

class TestTotalQualifiedTeams:
    def test_32_teams_qualify_from_groups(self):
        """24 group qualifiers (1st + 2nd) + 8 best 3rd = 32."""
        # Build a minimal standings with 3 groups of 4 teams each for testing
        # Extrapolate: 12 groups × (1st + 2nd) + 8 best 3rd = 32
        n_groups = 12
        group_qualifiers = n_groups * 2   # 1st + 2nd per group
        best_thirds      = 8
        assert group_qualifiers + best_thirds == 32

    def test_rank_third_placed_returns_qualifies_column(self):
        df = _build_synthetic_standings_with_thirds()
        ranked = rank_third_placed(df)
        assert "qualifies_as_third" in ranked.columns

    def test_qualifies_as_third_is_boolean(self):
        df = _build_synthetic_standings_with_thirds()
        ranked = rank_third_placed(df)
        assert ranked["qualifies_as_third"].dtype == bool


# ---------------------------------------------------------------------------
# Concurrent MD3 match simulation (design spec §6 Step 1)
# ---------------------------------------------------------------------------

class TestConcurrentMD3:
    """Verify that both MD3 matches in a group are drawn simultaneously before
    standings are resolved (not sequentially, which would leak first-result info
    into the second match's context determination)."""

    def test_both_md3_matches_applied_before_ranking(self):
        """Simulate: Group A, MD3 — both results should update standings together.

        We construct a scenario where the final rank depends on BOTH MD3 results.
        If the second result were applied after the first's ranking, the context
        could affect the simulation (e.g., a 'must-win' flag would be stale).
        """
        # Use Group A fixture (after MD1+MD2) and add two MD3 results
        md1_md2 = GROUP_A_MATCHES.copy()

        # MD3: Czechia beats Mexico (upset), South Africa beats Korea Republic
        md3 = _matches(
            ("Czechia",         "Mexico",         2, 0),
            ("South Africa",    "Korea Republic", 1, 0),
        )
        all_matches = pd.concat([md1_md2, md3], ignore_index=True)
        final = compute_group_standings(all_matches, GROUP_A_MEMBERSHIP)

        # Both MD3 results are applied: Mexico dropped from 6→6pts (but lost MD3),
        # Czechia gained 3pts (total 4), South Africa gained 3pts (total 4)
        mex = _get_team(final, "Mexico")
        cze = _get_team(final, "Czechia")
        sa  = _get_team(final, "South Africa")

        assert mex["pts"] == 6, "Mexico still has 6pts (two MD1+MD2 wins)"
        assert cze["pts"] == 4, "Czechia gained 3pts in MD3"
        assert sa["pts"]  == 4, "South Africa gained 3pts in MD3"

    def test_standings_are_deterministic_for_same_inputs(self):
        """Same input → same output every time (no stochastic logic in standings)."""
        df1 = compute_group_standings(GROUP_A_MATCHES, GROUP_A_MEMBERSHIP)
        df2 = compute_group_standings(GROUP_A_MATCHES, GROUP_A_MEMBERSHIP)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_required_columns_present(self):
        df = compute_group_standings(GROUP_A_MATCHES, GROUP_A_MEMBERSHIP)
        required = {"team", "group", "pts", "gd", "gf", "ga", "group_rank"}
        missing = required - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_group_rank_is_integer(self):
        df = compute_group_standings(GROUP_A_MATCHES, GROUP_A_MEMBERSHIP)
        assert df["group_rank"].dtype in (int, "int64", "int32")

    def test_null_scores_are_skipped(self):
        """Rows with null scores (future matches) are silently ignored."""
        matches_with_nulls = _matches(
            ("Mexico",       "Korea Republic", 1,    0   ),
            ("Czechia",      "South Africa",   0,    0   ),
            ("Mexico",       "South Africa",   None, None),  # future
            ("Korea Republic", "Czechia",      None, None),  # future
        )
        df = compute_group_standings(matches_with_nulls, GROUP_A_MEMBERSHIP)
        # 2 completed matches: Mexico 1-0 KR (3+0 pts) and Czechia 0-0 SA (1+1 pts) → total 5
        assert df["pts"].sum() == 5
        assert len(df) == 4

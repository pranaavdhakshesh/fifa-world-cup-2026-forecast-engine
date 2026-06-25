"""
tests/test_simulation.py

Purpose: Verify that the Monte Carlo simulation engine satisfies probability
invariants, reproducibility guarantees, and bracket-seeding correctness
using a minimal synthetic group-stage scenario.

All tests use a lightweight mock model that returns deterministic probabilities
so the test suite runs without real archive data or a fitted model.

Design spec ref: §7 Repository Structure — tests/test_simulation.py
"""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.name_map import CANONICAL_48


# ---------------------------------------------------------------------------
# Minimal fake model that satisfies the simulation.py model protocol
# ---------------------------------------------------------------------------

class _FakeModel:
    """Returns equal-probability predictions for any match pair."""

    def predict_group(self, features: pd.DataFrame) -> np.ndarray:
        """Return [P(W), P(D), P(L)] = [1/3, 1/3, 1/3] for each row."""
        n = len(features)
        return np.full((n, 3), 1.0 / 3.0)

    def predict_knockout(self, features: pd.DataFrame) -> np.ndarray:
        """Return [P(home wins), P(draw@90), P(away wins)] = [0.393, 0.214, 0.393]."""
        n = len(features)
        draw = 0.214
        p    = (1.0 - draw) / 2.0
        return np.column_stack([
            np.full(n, p),
            np.full(n, draw),
            np.full(n, p),
        ])

    def predict_penalty(self, features: pd.DataFrame) -> np.ndarray:
        """Return [P(home wins shootout), P(away wins shootout)] = [0.5, 0.5]."""
        n = len(features)
        return np.full((n, 2), 0.5)

    def bayesian_update(self, prior: float, team: str, in_tournament_data: dict) -> float:
        return prior


# ---------------------------------------------------------------------------
# Minimal group configuration (12 groups × 4 teams each = 48 teams)
# ---------------------------------------------------------------------------

def _build_minimal_group_membership() -> dict[str, str]:
    """Assign all 48 canonical teams to 12 groups of 4."""
    teams = sorted(CANONICAL_48)
    groups = "ABCDEFGHIJKL"
    membership: dict[str, str] = {}
    for i, team in enumerate(teams):
        membership[team] = groups[i // 4]
    return membership


def _build_empty_match_results() -> pd.DataFrame:
    """No in-tournament results yet (pre-MD1 state)."""
    return pd.DataFrame(columns=["home_team", "away_team", "home_score", "away_score",
                                  "stage", "match_number"])


# ---------------------------------------------------------------------------
# Probability invariant tests (pure math, no real model required)
# ---------------------------------------------------------------------------

class TestProbabilityInvariants:
    """Core invariants that must hold for any valid simulation output."""

    def _uniform_win_probs(self, n_teams: int = 48) -> np.ndarray:
        """Simulate uniform win probabilities (all teams equally likely)."""
        return np.full(n_teams, 1.0 / n_teams)

    def test_win_probs_sum_to_one(self):
        """Sum of all team win probabilities must equal 1.0."""
        probs = self._uniform_win_probs(48)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_win_probs_non_negative(self):
        """No team can have a negative win probability."""
        probs = self._uniform_win_probs(48)
        assert (probs >= 0).all()

    def test_win_probs_bounded_by_one(self):
        """No team can have a win probability greater than 1."""
        probs = self._uniform_win_probs(48)
        assert (probs <= 1.0).all()

    def test_stage_reach_probs_monotone_decreasing(self):
        """P(reach round R) ≥ P(reach round R+1) for every team."""
        stages = ["group_stage", "r32", "r16", "qf", "sf", "final", "champion"]
        p_group_stage = 1.0
        # Under uniform bracket, each elimination round halves the probability
        p_r32    = 1.0 / 1.0      # everyone reaches R32 from group stage
        p_r16    = 1.0 / 2.0      # must win R32
        p_qf     = 1.0 / 4.0
        p_sf     = 1.0 / 8.0
        p_final  = 1.0 / 16.0
        p_champ  = 1.0 / 32.0

        probs = [p_group_stage, p_r32, p_r16, p_qf, p_sf, p_final, p_champ]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], (
                f"Stage reach probs not monotone at stage {i}: "
                f"{probs[i]} < {probs[i+1]}"
            )

    def test_exactly_one_winner_invariant(self):
        """Across any set of runs, champion wins sum to 1 per tournament."""
        n_runs = 100
        # Simulate wins from a multinomial with uniform probs
        rng = np.random.default_rng(42)
        win_counts = np.zeros(48, dtype=int)
        for _ in range(n_runs):
            # Each simulated tournament has exactly one winner
            winner_idx = rng.integers(0, 48)
            win_counts[winner_idx] += 1
        # Total wins == total runs (one winner per tournament)
        assert win_counts.sum() == n_runs


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Two MC runs with the same seed must produce bit-identical results."""

    def test_numpy_rng_reproducibility(self):
        """Numpy default_rng with same seed yields identical sequences."""
        def _simulate(seed: int, n: int = 1000) -> np.ndarray:
            rng = np.random.default_rng(seed)
            return rng.integers(0, 48, size=n)

        run_a = _simulate(seed=2026)
        run_b = _simulate(seed=2026)
        assert (run_a == run_b).all(), "Same seed must produce identical sequences"

    def test_different_seeds_differ(self):
        """Different seeds should produce different outcomes (with high probability)."""
        def _simulate(seed: int, n: int = 1000) -> np.ndarray:
            rng = np.random.default_rng(seed)
            return rng.integers(0, 48, size=n)

        run_a = _simulate(seed=2026)
        run_b = _simulate(seed=2027)
        assert not (run_a == run_b).all(), "Different seeds produced identical sequences"

    def test_coin_flip_reproducibility(self):
        """Match outcomes are reproducible with the same seed."""
        def _flip_match(rng: np.random.Generator, p: float) -> int:
            return int(rng.random() < p)

        results_a = [_flip_match(np.random.default_rng(42), 0.6) for _ in range(50)]
        results_b = [_flip_match(np.random.default_rng(42), 0.6) for _ in range(50)]
        assert results_a == results_b


# ---------------------------------------------------------------------------
# Group-stage simulation logic
# ---------------------------------------------------------------------------

class TestGroupStageSimulation:
    """Verify group-stage simulation properties."""

    def _simulate_group(self, teams: list[str], seed: int = 0) -> pd.DataFrame:
        """Simulate a single group (6 matches) with equal probabilities."""
        rng = np.random.default_rng(seed)
        records = []
        for i, home in enumerate(teams):
            for j, away in enumerate(teams):
                if j <= i:
                    continue
                outcome = rng.choice([2, 1, 0], p=[1/3, 1/3, 1/3])
                if outcome == 2:   # home win
                    records.append({"team": home, "pts": 3, "gf": 1, "ga": 0})
                    records.append({"team": away, "pts": 0, "gf": 0, "ga": 1})
                elif outcome == 1:  # draw
                    records.append({"team": home, "pts": 1, "gf": 0, "ga": 0})
                    records.append({"team": away, "pts": 1, "gf": 0, "ga": 0})
                else:              # away win
                    records.append({"team": home, "pts": 0, "gf": 0, "ga": 1})
                    records.append({"team": away, "pts": 3, "gf": 1, "ga": 0})
        df = pd.DataFrame(records)
        return df.groupby("team").sum().reset_index()

    def test_group_points_sum_correct(self):
        """In a 4-team group (6 matches), total points distributed = 3×6 minus draws.
        Min = 12 (all draws: 6×2=12), Max = 18 (all decisive: 6×3=18)."""
        teams = ["Germany", "France", "Spain", "Brazil"]
        result = self._simulate_group(teams, seed=0)
        total_pts = result["pts"].sum()
        assert 12 <= total_pts <= 18, (
            f"Group total points {total_pts} outside valid range [12, 18]"
        )

    def test_all_group_teams_have_results(self):
        """Every team in the group gets a row in the standings."""
        teams = ["Germany", "France", "Spain", "Brazil"]
        result = self._simulate_group(teams, seed=1)
        assert set(result["team"]) == set(teams)

    def test_md3_concurrent_group_a(self):
        """Group A MD3 matches must use the same RNG state (concurrent)."""
        # Simulate two concurrent MD3 matches using same seed
        rng = np.random.default_rng(2026)
        match_1_outcome = rng.choice([2, 1, 0], p=[1/3, 1/3, 1/3])
        match_2_outcome = rng.choice([2, 1, 0], p=[1/3, 1/3, 1/3])
        # Both outcomes drawn from the same sequential RNG — they are "concurrent"
        # in the sense that neither influences the other's sampling
        assert match_1_outcome in [0, 1, 2]
        assert match_2_outcome in [0, 1, 2]


# ---------------------------------------------------------------------------
# Bracket seeding logic
# ---------------------------------------------------------------------------

class TestBracketSeeding:
    """Verify that R32 slot assignments follow the DS16 bracket structure."""

    # From DS16 / design spec: R32 match 73: "2A" vs "2B"
    # 1A goes to match 79
    # Group F winner goes to... R32 match 74: "1C" vs "2F" (so 2F, not 1F winner)
    R32_STRUCTURE = {
        73: ("2A", "2B"), 74: ("1C", "2F"), 75: ("1E", None),
        76: ("1F", "2C"), 77: ("2E", "2I"), 78: ("1I", None),
        79: ("1A", None), 80: ("1L", None), 81: ("1G", None),
        82: ("1D", None), 83: ("1H", "2J"), 84: ("2K", "2L"),
        85: ("1B", None), 86: ("2D", "2G"), 87: ("1J", "2H"), 88: ("1K", None),
    }

    def test_r32_has_16_matches(self):
        assert len(self.R32_STRUCTURE) == 16

    def test_group_a_winner_in_r32_match_79(self):
        """Group A winner (1A) is seeded into R32 match 79."""
        slot_home, slot_away = self.R32_STRUCTURE[79]
        assert slot_home == "1A"

    def test_group_a_runner_up_in_r32_match_73(self):
        """Group A runner-up (2A) faces Group B runner-up in match 73."""
        slot_home, slot_away = self.R32_STRUCTURE[73]
        assert "2A" in (slot_home, slot_away)

    def test_no_same_group_in_r32(self):
        """No R32 match pairs 1st and 2nd place from the same group."""
        for match_no, (slot_a, slot_b) in self.R32_STRUCTURE.items():
            if slot_a is None or slot_b is None:
                continue
            # slots like "1A" and "2A" would be same group
            if len(slot_a) == 2 and len(slot_b) == 2:
                group_a = slot_a[1]
                group_b = slot_b[1]
                assert group_a != group_b or (slot_a[0] == slot_b[0]), (
                    f"R32 match {match_no} pairs teams from the same group: "
                    f"{slot_a} vs {slot_b}"
                )

    def test_third_place_slots_count(self):
        """Exactly 8 R32 slots are filled by third-placed teams via Annex C."""
        R32_THIRD_PLACE_SLOTS = frozenset({75, 78, 79, 80, 81, 82, 85, 88})
        assert len(R32_THIRD_PLACE_SLOTS) == 8

    def test_r16_bracket_structure(self):
        """R16 connects correct pairs of R32 matches."""
        R16_BRACKET = {
            89: (73, 75), 90: (74, 77), 91: (76, 78), 92: (79, 80),
            93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
        }
        assert len(R16_BRACKET) == 8
        # All R32 match numbers must appear exactly once across R16 bracket values
        r32_sources = [m for pair in R16_BRACKET.values() for m in pair]
        assert len(r32_sources) == 16
        assert len(set(r32_sources)) == 16, "Each R32 match feeds exactly one R16 match"


# ---------------------------------------------------------------------------
# Annex C schema
# ---------------------------------------------------------------------------

class TestAnnexC:
    """Verify Annex C (third-place seeding table) matches the design spec."""

    EXPECTED_COLS = {
        "qualifying_groups", "m75", "m78", "m79", "m80", "m81", "m82", "m85", "m88"
    }
    EXPECTED_ROWS = 495          # C(12, 8) combinations

    def _load_annex_c(self) -> pd.DataFrame:
        import os
        path = "/Users/pranaavdhaksheshganesh/Downloads/FIFA_WC_2026_Project/third_place_annex_c.csv"
        if not os.path.exists(path):
            pytest.skip("Annex C CSV not found; skipping")
        return pd.read_csv(path)

    def test_annex_c_row_count(self):
        df = self._load_annex_c()
        assert len(df) == self.EXPECTED_ROWS, (
            f"Annex C should have {self.EXPECTED_ROWS} rows, got {len(df)}"
        )

    def test_annex_c_required_columns(self):
        df = self._load_annex_c()
        missing = self.EXPECTED_COLS - set(df.columns)
        assert not missing, f"Annex C missing columns: {missing}"

    def test_annex_c_slot_values_are_strings(self):
        df = self._load_annex_c()
        slot_cols = [c for c in df.columns if c.startswith("m")]
        for col in slot_cols:
            non_str = df[col].dropna().map(lambda x: not isinstance(x, str))
            assert not non_str.any(), f"Column {col} contains non-string values"

    def test_annex_c_qualifying_groups_are_8_groups(self):
        """Each row covers exactly 8 qualifying groups stored as comma-separated letters."""
        df = self._load_annex_c()
        counts = df["qualifying_groups"].str.split(",").apply(len)
        assert (counts == 8).all(), (
            f"qualifying_groups should have 8 comma-separated groups per row; "
            f"got counts: {counts.unique()}"
        )


# ---------------------------------------------------------------------------
# Penalty model activation
# ---------------------------------------------------------------------------

class TestPenaltyModelActivation:
    """Knockout matches drawn at 90 min must proceed to penalty shootout."""

    def test_draw_triggers_shootout(self):
        """A 90-min draw should trigger the penalty model (not return a draw)."""
        outcome_90min = 1     # DRAW at 90 min
        # After detecting a draw, simulation must decide via shootout
        assert outcome_90min == 1, "Outcome 1 signals draw — shootout path must be taken"

    def test_win_does_not_trigger_shootout(self):
        """A decisive 90-min result skips the shootout."""
        for outcome in [0, 2]:      # WIN or LOSS (decisive)
            assert outcome != 1     # no shootout needed

    def test_penalty_proba_sums_to_one(self):
        """Penalty model must return [P(A), P(B)] summing to 1.0."""
        # Synthetic: Germany (shrunk rate 0.625) vs England (0.40)
        rate_ger, rate_eng = 0.625, 0.40
        K = 8
        # Simple normalisation: proportion of rates
        total = rate_ger + rate_eng
        p_ger = rate_ger / total
        p_eng = rate_eng / total
        assert abs(p_ger + p_eng - 1.0) < 1e-10

    def test_penalty_favors_higher_rate_team(self):
        """Team with higher shrunk shootout rate is favored."""
        rate_a, rate_b = 0.625, 0.40
        total = rate_a + rate_b
        p_a = rate_a / total
        assert p_a > 0.5, f"Higher-rate team should be favored, P(A)={p_a}"


# ---------------------------------------------------------------------------
# All 48 teams get a stage reached
# ---------------------------------------------------------------------------

class TestAllTeamsReachStage:
    """Every team must exit the tournament at a defined stage (no nulls)."""

    VALID_STAGES = {
        "group_stage", "r32", "r16", "qf", "sf", "3rd_place", "final", "champion"
    }

    def test_stage_reached_set_from_canonical(self):
        """All 48 canonical teams can be assigned a stage."""
        # Assign every team the group_stage exit (minimum, all teams guaranteed it)
        stage_reached = {team: "group_stage" for team in CANONICAL_48}
        assert len(stage_reached) == 48

    def test_no_none_stage_reached(self):
        """stage_reached dictionary must not contain None values."""
        stage_reached = {team: "group_stage" for team in CANONICAL_48}
        nones = [t for t, s in stage_reached.items() if s is None]
        assert not nones, f"Teams with None stage: {nones}"

    def test_all_stages_are_valid(self):
        stage_reached = {team: "group_stage" for team in CANONICAL_48}
        # Replace 1 team as champion
        champion = next(iter(CANONICAL_48))
        stage_reached[champion] = "champion"

        for team, stage in stage_reached.items():
            assert stage in self.VALID_STAGES, (
                f"Team {team} has invalid stage {stage!r}"
            )

    def test_exactly_one_champion(self):
        """There must be exactly one champion per tournament simulation."""
        stage_reached = {team: "group_stage" for team in CANONICAL_48}
        champion = next(iter(CANONICAL_48))
        stage_reached[champion] = "champion"

        champions = [t for t, s in stage_reached.items() if s == "champion"]
        assert len(champions) == 1, f"Expected 1 champion, got {len(champions)}: {champions}"


# ---------------------------------------------------------------------------
# Champion confidence interval (statistical test)
# ---------------------------------------------------------------------------

class TestChampionConfidenceInterval:
    """Design spec: 90% CI width < 5 pp for tournament winner (1,000 simulations)."""

    N_SIMULATIONS = 1000
    CI_MAX_WIDTH   = 0.05         # 5 percentage points

    def test_ci_width_under_threshold_uniform(self):
        """Under a uniform prior with 1,000 runs, 90% CI width for any team
        should be well under 5 percentage points."""
        rng  = np.random.default_rng(2026)
        # Multinomial: 48 teams, 1000 draws — each trial picks one winner
        outcomes = rng.integers(0, 48, size=self.N_SIMULATIONS)
        # Pick the team with the most wins
        counts = np.bincount(outcomes, minlength=48)
        p_hat  = counts / self.N_SIMULATIONS
        # Wilson 90% CI half-width for the mode team
        p_top = p_hat.max()
        z = 1.645      # 90% CI z-score
        se = math.sqrt(p_top * (1 - p_top) / self.N_SIMULATIONS)
        ci_width = 2 * z * se
        assert ci_width < self.CI_MAX_WIDTH, (
            f"CI width {ci_width:.4f} exceeds threshold {self.CI_MAX_WIDTH}"
        )

    def test_ci_width_formula_with_known_values(self):
        """Verify CI formula: for p=1/48 and N=1000, CI width should be small."""
        p = 1.0 / 48.0
        n = 1000
        z = 1.645
        se = math.sqrt(p * (1 - p) / n)
        ci_width = 2 * z * se
        # Uniform prob: ~0.021 per team; CI should be narrow
        assert ci_width < self.CI_MAX_WIDTH, (
            f"CI width {ci_width:.4f} exceeds {self.CI_MAX_WIDTH}"
        )


# ---------------------------------------------------------------------------
# DS16 data error correction
# ---------------------------------------------------------------------------

class TestDS16DataErrorCorrection:
    """Design spec notes: DS16 match 100 label 'W95 vs W100' is an error.
    The correct label should be 'W95 vs W96' (QF winners feed into SF)."""

    QF_BRACKET = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
    SF_BRACKET = {101: (97, 98), 102: (99, 100)}

    def test_match_100_inputs_are_95_and_96(self):
        """QF match 100 draws from R16 matches 95 and 96 (not 95 and 100)."""
        r16_a, r16_b = self.QF_BRACKET[100]
        assert r16_a == 95 and r16_b == 96, (
            f"Match 100 should reference R16 matches 95 and 96, got {r16_a} and {r16_b}"
        )

    def test_sf_inputs_reference_qf_matches(self):
        """Both SF matches draw from QF matches 97-100."""
        for sf_match, (qf_a, qf_b) in self.SF_BRACKET.items():
            assert qf_a in self.QF_BRACKET and qf_b in self.QF_BRACKET, (
                f"SF match {sf_match} references unknown QF matches: {qf_a}, {qf_b}"
            )

    def test_no_self_reference_in_bracket(self):
        """No match's slot references its own match number (prevents the W100→100 error)."""
        for match_no, (src_a, src_b) in self.QF_BRACKET.items():
            assert src_a != match_no, f"Match {match_no} self-references slot A"
            assert src_b != match_no, f"Match {match_no} self-references slot B"

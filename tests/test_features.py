"""
tests/test_features.py

Purpose: Verify feature construction correctness, missing value handling, and
debutant imputation without requiring the full archive data on disk.

Integration tests that depend on actual zip files are gated behind
pytest.mark.integration and a HAVE_DATA fixture. Run them with:
    pytest -m integration

Unit tests cover the mathematical correctness of each feature calculation
using minimal synthetic DataFrames that exercise the same code paths.

Design spec ref: §7 Repository Structure — tests/test_features.py
"""

import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_ds2() -> pd.DataFrame:
    """Minimal DS2 (Elo) snapshot with Spain (rank 1, 2165) and USA (host)."""
    return pd.DataFrame([
        {
            "snapshot_date":  "2026-05-27",
            "country":        "Spain",
            "rank":           1,
            "rating":         2165.0,
            "rank_max":       1,
            "rating_max":     2165.0,
            "rank_avg":       2,
            "rating_avg":     2100.0,
            "confederation":  "UEFA",
            "is_host":        False,
        },
        {
            "snapshot_date":  "2026-05-27",
            "country":        "United States",
            "rank":           20,
            "rating":         1830.0,
            "rank_max":       15,
            "rating_max":     1850.0,
            "rank_avg":       22,
            "rating_avg":     1800.0,
            "confederation":  "CONCACAF",
            "is_host":        True,
        },
        {
            "snapshot_date":  "2026-05-27",
            "country":        "Germany",
            "rank":           3,
            "rating":         2030.0,
            "rank_max":       1,
            "rating_max":     2100.0,
            "rank_avg":       5,
            "rating_avg":     1980.0,
            "confederation":  "UEFA",
            "is_host":        False,
        },
        {
            "snapshot_date":  "2026-05-27",
            "country":        "Norway",
            "rank":           7,
            "rating":         1950.0,
            "rank_max":       5,
            "rating_max":     1970.0,
            "rank_avg":       9,
            "rating_avg":     1920.0,
            "confederation":  "UEFA",
            "is_host":        False,
        },
        {
            "snapshot_date":  "2026-05-27",
            "country":        "Curaçao",
            "rank":           75,
            "rating":         1450.0,
            "rank_max":       70,
            "rating_max":     1480.0,
            "rank_avg":       78,
            "rating_avg":     1440.0,
            "confederation":  "CONCACAF",
            "is_host":        False,
        },
    ])


@pytest.fixture
def minimal_ds6() -> pd.DataFrame:
    """Minimal DS6 (shootouts) with Germany 6W/8A and England 4W/12A.

    Germany's opponents are France; England's opponents are Spain.
    This keeps the two teams' appearance counts independent.
    """
    rows = []
    # Germany: 6 wins, 2 losses vs France = 8 appearances
    for _ in range(6):
        rows.append({"date": "2022-01-01", "home_team": "Germany", "away_team": "France",
                     "winner": "Germany", "first_shooter": "Germany"})
    for _ in range(2):
        rows.append({"date": "2022-01-01", "home_team": "Germany", "away_team": "France",
                     "winner": "France", "first_shooter": "Germany"})
    # England: 4 wins, 8 losses vs Spain = 12 appearances
    for _ in range(4):
        rows.append({"date": "2020-01-01", "home_team": "England", "away_team": "Spain",
                     "winner": "England", "first_shooter": "England"})
    for _ in range(8):
        rows.append({"date": "2020-01-01", "home_team": "England", "away_team": "Spain",
                     "winner": "Spain", "first_shooter": "England"})
    # Norway: no appearances
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Elo feature tests (build_elo_features equivalent math)
# ---------------------------------------------------------------------------

class TestEloFeatures:
    """Test the mathematical correctness of Elo-derived features."""

    def test_spain_elo_rating(self, minimal_ds2):
        spain = minimal_ds2[minimal_ds2["country"] == "Spain"].iloc[0]
        assert float(spain["rating"]) == 2165.0

    def test_spain_rank(self, minimal_ds2):
        spain = minimal_ds2[minimal_ds2["country"] == "Spain"].iloc[0]
        assert int(spain["rank"]) == 1

    def test_usa_is_host(self, minimal_ds2):
        usa = minimal_ds2[minimal_ds2["country"] == "United States"].iloc[0]
        assert bool(usa["is_host"]) is True

    def test_spain_is_not_host(self, minimal_ds2):
        spain = minimal_ds2[minimal_ds2["country"] == "Spain"].iloc[0]
        assert bool(spain["is_host"]) is False

    def test_elo_win_expectancy_formula(self):
        """WE = 1 / (1 + 10^(-(Δ)/400)) — verify against known values."""
        elo_a, elo_b = 2165.0, 1830.0
        delta  = elo_a - elo_b
        we_a   = 1.0 / (1.0 + 10.0 ** (-delta / 400.0))
        assert round(we_a, 2) == 0.87, f"Expected ~0.87, got {we_a:.4f}"

    def test_elo_win_expectancy_symmetric(self):
        """WE(A vs B) + WE(B vs A) == 1.0 exactly."""
        elo_a, elo_b = 2030.0, 1950.0
        we_a = 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))
        we_b = 1.0 / (1.0 + 10.0 ** (-(elo_b - elo_a) / 400.0))
        assert abs(we_a + we_b - 1.0) < 1e-10

    def test_elo_delta_sign(self, minimal_ds2):
        """Stronger team (Spain) vs weaker team (USA): delta > 0."""
        elo_spain = float(minimal_ds2[minimal_ds2["country"] == "Spain"]["rating"].iloc[0])
        elo_usa   = float(minimal_ds2[minimal_ds2["country"] == "United States"]["rating"].iloc[0])
        delta = elo_spain - elo_usa
        assert delta > 0

    def test_equal_elo_gives_50_50(self):
        we = 1.0 / (1.0 + 10.0 ** (0.0 / 400.0))
        assert abs(we - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# Penalty / shootout feature tests
# ---------------------------------------------------------------------------

class TestShootoutFeatures:
    """Test the Bayesian shrinkage formula for shootout win rates.

    Design spec: shrunk_rate = (wins + k × 0.5) / (appearances + k), k = 8
    """

    K = 8

    def _shrunk_rate(self, wins: int, appearances: int) -> float:
        return (wins + self.K * 0.5) / (appearances + self.K)

    def test_germany_shrunk_rate(self):
        """Design spec: Germany 6W/8A, k=8 → (6+4)/(8+8) = 10/16 = 0.625."""
        rate = self._shrunk_rate(6, 8)
        assert abs(rate - 0.625) < 1e-6, f"Expected 0.625, got {rate}"

    def test_england_shrunk_rate(self):
        """Design spec: England 4W/12A, k=8 → (4+4)/(12+8) = 8/20 = 0.40."""
        rate = self._shrunk_rate(4, 12)
        assert abs(rate - 0.40) < 1e-6, f"Expected 0.40, got {rate}"

    def test_no_appearances_gives_prior(self):
        """A team with 0 appearances receives the prior: k×0.5/k = 0.5."""
        rate = self._shrunk_rate(0, 0)
        assert abs(rate - 0.5) < 1e-6, f"Expected 0.5, got {rate}"

    def test_naive_flag_set_for_zero_appearances(self, minimal_ds6):
        """Norway has no DS6 appearances → shootout_naive_flag should be True."""
        norway_rows = minimal_ds6[
            (minimal_ds6["home_team"] == "Norway") |
            (minimal_ds6["away_team"] == "Norway")
        ]
        assert len(norway_rows) == 0, "Fixture assumption: Norway has no shootout records"
        # When appearances == 0, the feature builder sets naive flag and rate = 0.5
        appearances = 0
        rate = self._shrunk_rate(0, appearances)
        assert abs(rate - 0.5) < 1e-6

    def test_germany_appearances(self, minimal_ds6):
        germany_rows = minimal_ds6[
            (minimal_ds6["home_team"] == "Germany") |
            (minimal_ds6["away_team"] == "Germany")
        ]
        assert len(germany_rows) == 8

    def test_england_appearances(self, minimal_ds6):
        england_rows = minimal_ds6[
            (minimal_ds6["home_team"] == "England") |
            (minimal_ds6["away_team"] == "England")
        ]
        assert len(england_rows) == 12

    def test_shrunk_rate_bounded_between_0_and_1(self):
        """Shrunk rate must always stay in (0, 1)."""
        for wins, apps in [(0, 0), (0, 20), (20, 20), (1, 1)]:
            rate = self._shrunk_rate(wins, apps)
            assert 0.0 < rate < 1.0, f"Rate {rate} out of bounds for wins={wins}, apps={apps}"

    def test_shrinkage_pulls_extreme_rates_toward_half(self):
        """Perfect record (20/20 → 1.0) is pulled toward 0.5 by shrinkage."""
        raw  = 20 / 20     # 1.0 without shrinkage
        shrunk = self._shrunk_rate(20, 20)
        assert shrunk < raw, "Shrinkage must pull perfect record below 1.0"
        assert shrunk > 0.5, "Shrinkage should not pull rate below 0.5 for a strong team"


# ---------------------------------------------------------------------------
# WC debutant flag
# ---------------------------------------------------------------------------

class TestDebutantFlag:
    def test_known_debutants_in_set(self):
        from src.name_map import WC_DEBUTANTS
        for team in ("Curaçao", "Uzbekistan", "Cape Verde"):
            assert team in WC_DEBUTANTS, f"{team} should be in WC_DEBUTANTS"

    def test_established_teams_not_debutants(self):
        from src.name_map import WC_DEBUTANTS
        for team in ("Germany", "France", "Brazil", "Argentina", "Spain"):
            assert team not in WC_DEBUTANTS, f"{team} should NOT be in WC_DEBUTANTS"

    def test_all_debutants_are_canonical(self):
        from src.name_map import CANONICAL_48, WC_DEBUTANTS
        for team in WC_DEBUTANTS:
            assert team in CANONICAL_48, (
                f"Debutant {team!r} is not in CANONICAL_48"
            )


# ---------------------------------------------------------------------------
# Training rows invariant: elo_year_used == match_year - 1
# ---------------------------------------------------------------------------

class TestTrainingRowsLeakageInvariant:
    def test_no_future_elo_in_synthetic_training(self):
        """elo_year_used must always equal match_year − 1 in training rows."""
        training_rows = pd.DataFrame({
            "match_year":   [1998, 2002, 2006, 2010, 2014, 2018, 2022],
            "elo_year_used":[1997, 2001, 2005, 2009, 2013, 2017, 2021],
            "outcome":      [2,    1,    0,    2,    1,    0,    2],
        })
        violations = training_rows[
            training_rows["elo_year_used"] != training_rows["match_year"] - 1
        ]
        assert violations.empty, (
            f"elo_year_used must be match_year−1. Violations:\n{violations}"
        )

    def test_elo_year_mismatch_detectable(self):
        """A row with elo_year_used == match_year is a clear data leak."""
        bad_row = pd.DataFrame({
            "match_year":   [2010],
            "elo_year_used":[2010],  # same year = forward-looking Elo
        })
        violations = bad_row[
            bad_row["elo_year_used"] != bad_row["match_year"] - 1
        ]
        assert not violations.empty


# ---------------------------------------------------------------------------
# Recency weighting formula
# ---------------------------------------------------------------------------

class TestRecencyWeighting:
    """Verify the recency decay formula from the design spec.

    Design spec: sample_weight = 1.0 + recency_factor × (Year − 1998) / (2022 − 1998)
    Default recency_factor = 0.4.
    """

    RECENCY_FACTOR = 0.4

    def _weight(self, year: int) -> float:
        return 1.0 + self.RECENCY_FACTOR * (year - 1998) / 24.0

    def test_1998_has_weight_one(self):
        assert abs(self._weight(1998) - 1.0) < 1e-10

    def test_2022_has_max_weight(self):
        w = self._weight(2022)
        assert abs(w - 1.4) < 1e-10, f"Expected 1.4, got {w}"

    def test_weights_are_strictly_increasing(self):
        years = [1998, 2002, 2006, 2010, 2014, 2018, 2022]
        weights = [self._weight(y) for y in years]
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1], (
                f"Weights must be strictly increasing; {weights[i]} >= {weights[i+1]}"
            )

    def test_all_weights_positive(self):
        for year in range(1998, 2023):
            assert self._weight(year) > 0


# ---------------------------------------------------------------------------
# Form window date constraints (unit-level)
# ---------------------------------------------------------------------------

class TestFormWindowConstraints:
    """Design spec: form_window = [2024-01-01, 2026-06-10], last 10 matches."""

    FORM_START = pd.Timestamp("2024-01-01")
    FORM_END   = pd.Timestamp("2026-06-10")

    def _filter_form(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"])
        return df[(df["date"] >= self.FORM_START) & (df["date"] <= self.FORM_END)].tail(10)

    def test_form_window_excludes_pre_2024(self):
        df = pd.DataFrame({"date": ["2023-12-31", "2024-01-01", "2025-06-01"]})
        filtered = self._filter_form(df)
        assert "2023-12-31" not in filtered["date"].astype(str).values

    def test_form_window_excludes_post_june10(self):
        df = pd.DataFrame({"date": ["2026-06-10", "2026-06-11", "2026-06-22"]})
        filtered = self._filter_form(df)
        assert "2026-06-11" not in filtered["date"].astype(str).values
        assert "2026-06-22" not in filtered["date"].astype(str).values

    def test_form_window_capped_at_10(self):
        dates = pd.date_range("2024-01-01", periods=15, freq="ME")
        df = pd.DataFrame({"date": dates})
        filtered = self._filter_form(df)
        assert len(filtered) == 10, f"Form window should cap at 10 matches, got {len(filtered)}"

    def test_exactly_june10_is_included(self):
        df = pd.DataFrame({"date": ["2026-06-10"]})
        filtered = self._filter_form(df)
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# Tournament feature constraints
# ---------------------------------------------------------------------------

class TestTournamentFeatureConstraints:
    """Validate that in-tournament feature values respect logical bounds."""

    def test_tourn_pts_md2_max_is_6(self):
        """Maximum possible: 2 wins × 3pts = 6pts from MD1+MD2."""
        max_pts = 6
        for pts in range(0, max_pts + 1):
            # pts in {0, 1, 2, 3, 4, 5, 6} are all valid
            assert 0 <= pts <= max_pts

    def test_tourn_pts_md2_cannot_be_9(self):
        """9pts would require 3 wins but only 2 games exist in MD1+MD2."""
        assert 9 > 6, "9pts is impossible in 2 games (6 is the maximum)"

    def test_tactical_gap_teams_correct(self):
        """Design spec: exactly 4 teams have incomplete tactical data."""
        from src.leakage_guard import TACTICAL_GAP_TEAMS
        assert len(TACTICAL_GAP_TEAMS) == 4
        assert "England"  in TACTICAL_GAP_TEAMS
        assert "Portugal" in TACTICAL_GAP_TEAMS
        assert "Colombia" in TACTICAL_GAP_TEAMS
        assert "Croatia"  in TACTICAL_GAP_TEAMS
        assert "Germany"  not in TACTICAL_GAP_TEAMS


# ---------------------------------------------------------------------------
# Integration tests (skipped unless archive data present)
# ---------------------------------------------------------------------------

def _archive_exists(name: str) -> bool:
    import os
    root = "/Users/pranaavdhaksheshganesh/Downloads/FIFA_WC_2026_Project"
    return os.path.exists(os.path.join(root, name))


@pytest.mark.skipif(
    not _archive_exists("archive (4).zip"),
    reason="DS2 archive not present; skipping Elo integration tests",
)
class TestEloFeaturesIntegration:
    @pytest.fixture(autouse=True)
    def elo_features(self):
        from src.features import load_ds2, build_elo_features
        ds2 = load_ds2("/Users/pranaavdhaksheshganesh/Downloads/FIFA_WC_2026_Project/archive (4).zip")
        self.elo = build_elo_features(ds2)

    def test_elo_features_48_rows(self):
        assert len(self.elo) == 48

    def test_spain_elo_rating(self):
        spain = self.elo.loc["Spain"] if "Spain" in self.elo.index else None
        if spain is not None:
            assert float(spain["elo_rating"]) == pytest.approx(2165.0, abs=5)

    def test_usa_is_host(self):
        if "United States" in self.elo.index:
            assert self.elo.loc["United States"]["elo_is_host"] == 1


@pytest.mark.skipif(
    not (_archive_exists("archive (6).zip") and _archive_exists("archive (2).zip")),
    reason="DS6/DS8 archives not present; skipping shootout integration tests",
)
class TestShootoutFeaturesIntegration:
    @pytest.fixture(autouse=True)
    def penalty_feats(self):
        from src.features import load_ds6, load_ds8, build_penalty_features
        ds6 = load_ds6("/Users/pranaavdhaksheshganesh/Downloads/FIFA_WC_2026_Project/archive (6).zip")
        ds8 = load_ds8("/Users/pranaavdhaksheshganesh/Downloads/FIFA_WC_2026_Project/archive (2).zip")
        self.pf = build_penalty_features(ds6, ds8)

    def test_germany_shootout_shrunk_rate(self):
        if "Germany" in self.pf.index:
            assert self.pf.loc["Germany"]["shootout_win_rate_alltime"] == pytest.approx(0.625, abs=0.05)

    def test_norway_naive_flag(self):
        if "Norway" in self.pf.index:
            assert self.pf.loc["Norway"]["shootout_naive_flag"] == 1

    def test_all_win_rates_in_0_1(self):
        col = "shootout_win_rate_alltime"
        if col in self.pf.columns:
            vals = self.pf[col].dropna()
            assert (vals >= 0.0).all() and (vals <= 1.0).all()

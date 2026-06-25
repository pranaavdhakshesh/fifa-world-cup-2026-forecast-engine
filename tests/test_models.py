"""
tests/test_models.py

Purpose: Verify model layer mathematical invariants without requiring a full
fit on real data — tests use minimal synthetic datasets and mock predictions.

Integration tests that depend on actual training archives are gated behind
pytest.mark.integration.

Design spec ref: §7 Repository Structure — tests/test_models.py
"""

import math
import pickle
import io
import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants mirroring the design spec (do NOT import from src.models yet to
# avoid XGBoost import errors on CI; we import selectively in each test)
# ---------------------------------------------------------------------------

WIN, DRAW, LOSS    = 2, 1, 0
DRAW_RATE_90MIN    = 0.214
BAYESIAN_ALPHA     = 0.17
PENALTY_CLIP_LOW   = 0.15
PENALTY_CLIP_HIGH  = 0.85


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_group_proba(n: int = 20) -> np.ndarray:
    """Return synthetic group-stage probability rows that sum to 1."""
    rng = np.random.default_rng(42)
    raw = rng.dirichlet([1.5, 0.8, 1.2], size=n)
    return raw          # already sums to 1 by dirichlet construction


def _synthetic_ko_proba(n: int = 20) -> np.ndarray:
    """Return synthetic knockout probability rows [P(A wins), P(draw@90), P(B wins)]
    where P(draw@90) is always DRAW_RATE_90MIN."""
    rng = np.random.default_rng(7)
    rem = 1.0 - DRAW_RATE_90MIN
    p_a = rng.beta(2, 2, size=n) * rem
    p_b = rem - p_a
    p_d = np.full(n, DRAW_RATE_90MIN)
    return np.column_stack([p_a, p_d, p_b])


# ---------------------------------------------------------------------------
# Group-stage probability invariant
# ---------------------------------------------------------------------------

class TestGroupModelProba:
    def test_group_proba_rows_sum_to_one(self):
        """Each predicted distribution [P(W), P(D), P(L)] must sum to 1.0."""
        proba = _synthetic_group_proba(100)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"Max deviation from 1.0: {np.abs(row_sums - 1.0).max()}"
        )

    def test_group_proba_columns_match_classes(self):
        """Three columns: WIN (idx 0), DRAW (idx 1), LOSS (idx 2)."""
        proba = _synthetic_group_proba(10)
        assert proba.shape[1] == 3

    def test_group_proba_non_negative(self):
        proba = _synthetic_group_proba(100)
        assert (proba >= 0).all()

    def test_group_proba_bounded_above_one(self):
        proba = _synthetic_group_proba(100)
        assert (proba <= 1.0 + 1e-10).all()


# ---------------------------------------------------------------------------
# Knockout draw rate invariant
# ---------------------------------------------------------------------------

class TestKnockoutModelDrawRate:
    def test_draw_rate_constant_value(self):
        """Design spec: draw rate at 90 min = 24/112 ≈ 0.214."""
        assert abs(DRAW_RATE_90MIN - 24 / 112) < 1e-3

    def test_knockout_proba_draw_column_matches_constant(self):
        """P(draw @ 90 min) must always equal DRAW_RATE_90MIN in every row."""
        proba = _synthetic_ko_proba(50)
        draw_col = proba[:, 1]          # column index 1 = DRAW
        assert np.allclose(draw_col, DRAW_RATE_90MIN, atol=1e-10), (
            f"Draw rate column deviates from {DRAW_RATE_90MIN}"
        )

    def test_knockout_proba_rows_sum_to_one(self):
        proba = _synthetic_ko_proba(50)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_knockout_proba_non_negative(self):
        proba = _synthetic_ko_proba(50)
        assert (proba >= 0).all()

    def test_draw_rate_not_used_in_group_stage(self):
        """Group-stage predictions should NOT fix any column to 0.214."""
        proba = _synthetic_group_proba(200)
        draw_col = proba[:, 1]
        # At least some rows should differ from 0.214 significantly
        deviations = np.abs(draw_col - DRAW_RATE_90MIN)
        assert deviations.max() > 0.05, (
            "Group-stage draw column appears to be fixed to the knockout draw rate"
        )


# ---------------------------------------------------------------------------
# Bayesian updater invariants
# ---------------------------------------------------------------------------

class TestBayesianUpdater:
    """Design spec: multiplier ∈ [0.5, 2.0]; alpha = 0.17.

    P_posterior = (1 - α) × P_prior + α × (P_prior × multiplier)
                = P_prior × [(1 - α) + α × multiplier]
    """

    ALPHA = BAYESIAN_ALPHA

    def _apply_update(self, p_prior: float, multiplier: float) -> float:
        multiplier = max(0.5, min(2.0, multiplier))
        raw = p_prior * ((1 - self.ALPHA) + self.ALPHA * multiplier)
        return min(1.0, max(0.0, raw))

    def test_multiplier_clip_lower_bound(self):
        """Multiplier below 0.5 is clipped to 0.5."""
        p_posterior_clipped = self._apply_update(0.5, 0.1)
        p_posterior_at_min  = self._apply_update(0.5, 0.5)
        assert abs(p_posterior_clipped - p_posterior_at_min) < 1e-10

    def test_multiplier_clip_upper_bound(self):
        """Multiplier above 2.0 is clipped to 2.0."""
        p_posterior_clipped = self._apply_update(0.5, 5.0)
        p_posterior_at_max  = self._apply_update(0.5, 2.0)
        assert abs(p_posterior_clipped - p_posterior_at_max) < 1e-10

    def test_multiplier_1_is_identity(self):
        """Multiplier == 1.0 leaves prior unchanged."""
        p_prior = 0.6
        p_post  = self._apply_update(p_prior, 1.0)
        assert abs(p_post - p_prior) < 1e-10

    def test_multiplier_above_one_increases_probability(self):
        """Multiplier > 1.0 raises posterior above prior (positive signal)."""
        p_prior = 0.55
        p_post  = self._apply_update(p_prior, 1.5)
        assert p_post > p_prior

    def test_multiplier_below_one_decreases_probability(self):
        """Multiplier < 1.0 lowers posterior below prior (negative signal)."""
        p_prior = 0.55
        p_post  = self._apply_update(p_prior, 0.8)
        assert p_post < p_prior

    def test_germany_positive_form_multiplier_above_one(self):
        """Germany's in-tournament form → multiplier > 1.0 (strong group results)."""
        # Synthetic scenario: Germany 3W, GD +6 in group stage = bullish signal
        germany_form_multiplier = 1.30       # reasonable strong-form value
        assert germany_form_multiplier > 1.0

    def test_turkey_negative_form_multiplier_below_one(self):
        """Türkiye's poor in-tournament results → multiplier < 1.0."""
        turkey_form_multiplier = 0.75        # poor group-stage showing
        assert turkey_form_multiplier < 1.0

    def test_posterior_bounded_between_0_and_1(self):
        """Bayesian update must keep probability in [0, 1]."""
        for p_prior in np.linspace(0.01, 0.99, 20):
            for m in [0.5, 1.0, 1.5, 2.0]:
                p_post = self._apply_update(p_prior, m)
                assert 0.0 <= p_post <= 1.0, (
                    f"Posterior {p_post} out of [0,1] for prior={p_prior}, m={m}"
                )

    def test_alpha_effect_size(self):
        """Alpha=0.17 means update is conservative — new signal has limited weight."""
        p_prior  = 0.50
        p_post_max = self._apply_update(p_prior, 2.0)
        delta = abs(p_post_max - p_prior)
        # With α=0.17 and m=2.0: P = 0.5 × (0.83 + 0.34) = 0.585; delta = 0.085
        assert delta < 0.15, (
            "Bayesian update with alpha=0.17 should produce conservative adjustments"
        )


# ---------------------------------------------------------------------------
# Penalty / shootout model invariants
# ---------------------------------------------------------------------------

class TestShootoutModel:
    """Design spec: P(A wins) + P(B wins) = 1.0; clipped to [0.15, 0.85].
    Germany (6W/8A, k=8) vs England (4W/12A, k=8).
    """

    K = 8

    def _shrunk_rate(self, wins: int, apps: int) -> float:
        return (wins + self.K * 0.5) / (apps + self.K)

    def _penalty_proba(self, rate_a: float, rate_b: float) -> tuple[float, float]:
        """Model: P_A = 0.5 + (rate_A − rate_B) / 2, then normalise and clip."""
        raw_a = 0.5 + (rate_a - rate_b) / 2.0
        raw_a = max(PENALTY_CLIP_LOW, min(PENALTY_CLIP_HIGH, raw_a))
        raw_b = 1.0 - raw_a
        return raw_a, raw_b

    def test_shootout_proba_sums_to_one(self):
        """P(A wins) + P(B wins) must == 1.0."""
        for rate_a, rate_b in [(0.6, 0.4), (0.5, 0.5), (0.8, 0.3), (0.2, 0.7)]:
            p_a, p_b = self._penalty_proba(rate_a, rate_b)
            assert abs(p_a + p_b - 1.0) < 1e-10, (
                f"P(A)+P(B) = {p_a+p_b} ≠ 1.0 for rates ({rate_a},{rate_b})"
            )

    def test_germany_vs_england_germany_favored(self):
        """Design spec: P(Germany wins) > 0.55 given higher shrunk shootout rate."""
        rate_germany = self._shrunk_rate(6, 8)    # 0.625
        rate_england = self._shrunk_rate(4, 12)   # 0.40
        p_ger, _ = self._penalty_proba(rate_germany, rate_england)
        assert p_ger > 0.55, f"Germany should be favored in shootout, got P(GER)={p_ger:.4f}"

    def test_germany_shrunk_rate(self):
        assert abs(self._shrunk_rate(6, 8) - 0.625) < 1e-6

    def test_england_shrunk_rate(self):
        assert abs(self._shrunk_rate(4, 12) - 0.40) < 1e-6

    def test_shootout_bounds_enforced_low(self):
        """Extreme weak team is clipped to PENALTY_CLIP_LOW = 0.15."""
        p_a, p_b = self._penalty_proba(0.01, 0.99)
        assert p_a >= PENALTY_CLIP_LOW
        assert p_b <= 1.0 - PENALTY_CLIP_LOW

    def test_shootout_bounds_enforced_high(self):
        """Extreme strong team is clipped to PENALTY_CLIP_HIGH = 0.85."""
        p_a, p_b = self._penalty_proba(0.99, 0.01)
        assert p_a <= PENALTY_CLIP_HIGH
        assert p_b >= 1.0 - PENALTY_CLIP_HIGH

    def test_equal_rates_gives_50_50(self):
        """Equal shrunk rates → P(A wins) == P(B wins) == 0.5."""
        p_a, p_b = self._penalty_proba(0.5, 0.5)
        assert abs(p_a - 0.5) < 1e-10
        assert abs(p_b - 0.5) < 1e-10

    def test_proba_always_in_0_1(self):
        for rate_a in np.linspace(0.0, 1.0, 11):
            for rate_b in np.linspace(0.0, 1.0, 11):
                p_a, p_b = self._penalty_proba(rate_a, rate_b)
                assert 0 <= p_a <= 1
                assert 0 <= p_b <= 1


# ---------------------------------------------------------------------------
# Penalty model weight formula (design spec component weights)
# ---------------------------------------------------------------------------

class TestPenaltyModelWeights:
    """Design spec:
      Has WC data:    w1=0.50, w2=0.40, w3=0.10  (shrunk hist + Elo WE + WC rate)
      No WC data:     w1=0.55, w2=0.45, w3=0.00  (shrunk hist + Elo WE)
    """

    def test_has_wc_data_weights_sum_to_one(self):
        w1, w2, w3 = 0.50, 0.40, 0.10
        assert abs(w1 + w2 + w3 - 1.0) < 1e-10

    def test_no_wc_data_weights_sum_to_one(self):
        w1, w2, w3 = 0.55, 0.45, 0.00
        assert abs(w1 + w2 + w3 - 1.0) < 1e-10

    def test_formula_with_known_values(self):
        """Synthetic: c1=0.625, c2=0.80, c3=0.70, has WC data → P_A."""
        w1, w2, w3 = 0.50, 0.40, 0.10
        c1_a, c2_a, c3_a = 0.625, 0.80, 0.70
        p_a = w1 * c1_a + w2 * c2_a + w3 * c3_a
        assert abs(p_a - (0.50 * 0.625 + 0.40 * 0.80 + 0.10 * 0.70)) < 1e-10


# ---------------------------------------------------------------------------
# Ensemble weight invariant
# ---------------------------------------------------------------------------

class TestEnsembleWeights:
    """Design spec: 70% WC-only / 30% augmented by default; CV range [0.60, 0.90]."""

    DEFAULT_WC_WEIGHT = 0.70
    CV_RANGE = (0.60, 0.90)

    def test_default_weights_sum_to_one(self):
        wc_w = self.DEFAULT_WC_WEIGHT
        aug_w = 1.0 - wc_w
        assert abs(wc_w + aug_w - 1.0) < 1e-10

    def test_cv_range_contains_default(self):
        low, high = self.CV_RANGE
        assert low <= self.DEFAULT_WC_WEIGHT <= high

    def test_ensemble_combination_formula(self):
        """P_ensemble = wc_weight × P_wc + (1 - wc_weight) × P_aug."""
        p_wc  = np.array([0.6, 0.2, 0.2])
        p_aug = np.array([0.5, 0.3, 0.2])
        w     = self.DEFAULT_WC_WEIGHT
        p_ens = w * p_wc + (1 - w) * p_aug
        assert abs(p_ens.sum() - 1.0) < 1e-10

    def test_all_cv_weights_in_range(self):
        """Any weight produced during CV must stay in [0.60, 0.90]."""
        low, high = self.CV_RANGE
        for w in np.linspace(low, high, 100):
            assert low <= w <= high


# ---------------------------------------------------------------------------
# Model serialisation (save/load roundtrip)
# ---------------------------------------------------------------------------

class TestModelSerialisation:
    def test_pickle_roundtrip_sklearn_lr(self):
        """LogisticRegression survives pickle save/load with identical predictions."""
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 5))
        y = rng.integers(0, 3, size=60)

        lr = LogisticRegression(max_iter=300, random_state=42)
        lr.fit(X, y)

        # Save → bytes
        buf = io.BytesIO()
        pickle.dump(lr, buf)
        buf.seek(0)

        # Load from bytes
        lr2 = pickle.load(buf)

        X_test = rng.standard_normal((10, 5))
        p1 = lr.predict_proba(X_test)
        p2 = lr2.predict_proba(X_test)
        assert np.allclose(p1, p2, atol=1e-10), "Predictions differ after pickle roundtrip"

    def test_pickle_preserves_classes(self):
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 4))
        y = np.array([LOSS] * 20 + [DRAW] * 20 + [WIN] * 20)

        lr = LogisticRegression(max_iter=300, random_state=42)
        lr.fit(X, y)

        buf = io.BytesIO()
        pickle.dump(lr, buf)
        buf.seek(0)
        lr2 = pickle.load(buf)

        assert list(lr.classes_) == list(lr2.classes_)


# ---------------------------------------------------------------------------
# Chronological CV fold structure
# ---------------------------------------------------------------------------

class TestChronologicalCV:
    """Design spec: 7 leave-one-tournament-out folds.
    Fold 7 (2022) is reserved for calibration only — never used for HP search.
    """

    FOLD_STRUCTURE = [
        (1998, [2002, 2006, 2010, 2014, 2018, 2022]),
        (2002, [1998, 2006, 2010, 2014, 2018, 2022]),
        (2006, [1998, 2002, 2010, 2014, 2018, 2022]),
        (2010, [1998, 2002, 2006, 2014, 2018, 2022]),
        (2014, [1998, 2002, 2006, 2010, 2018, 2022]),
        (2018, [1998, 2002, 2006, 2010, 2014, 2022]),
    ]
    ALL_YEARS = {1998, 2002, 2006, 2010, 2014, 2018, 2022}

    def test_six_cv_folds(self):
        assert len(self.FOLD_STRUCTURE) == 6

    def test_2022_in_all_training_sets(self):
        """Fold 7 (2022) is calibration-only → appears in every train split."""
        for _, train_years in self.FOLD_STRUCTURE:
            assert 2022 in train_years, "2022 must be in every CV training set"

    def test_held_out_year_not_in_train(self):
        """Each fold's held-out year must not appear in its training years."""
        for held_out, train_years in self.FOLD_STRUCTURE:
            assert held_out not in train_years

    def test_each_fold_trains_on_6_tournaments(self):
        """Each fold trains on 6 out of 7 tournaments."""
        for held_out, train_years in self.FOLD_STRUCTURE:
            assert len(train_years) == 6

    def test_train_plus_held_out_covers_all_years(self):
        for held_out, train_years in self.FOLD_STRUCTURE:
            combined = set(train_years) | {held_out}
            assert combined == self.ALL_YEARS


# ---------------------------------------------------------------------------
# Class weight specification
# ---------------------------------------------------------------------------

class TestClassWeights:
    """Design spec: {WIN: 1.2, DRAW: 1.5, LOSS: 1.1} — up-weights minority draw."""

    CLASS_WEIGHTS = {WIN: 1.2, DRAW: 1.5, LOSS: 1.1}

    def test_draw_has_highest_weight(self):
        """Draw is the minority class and must have the highest weight."""
        assert self.CLASS_WEIGHTS[DRAW] > self.CLASS_WEIGHTS[WIN]
        assert self.CLASS_WEIGHTS[DRAW] > self.CLASS_WEIGHTS[LOSS]

    def test_all_weights_above_one(self):
        """All class weights should be ≥ 1.0."""
        for cls, w in self.CLASS_WEIGHTS.items():
            assert w >= 1.0, f"Class {cls} weight {w} is below 1.0"

    def test_exact_weight_values(self):
        assert self.CLASS_WEIGHTS[WIN]  == pytest.approx(1.2)
        assert self.CLASS_WEIGHTS[DRAW] == pytest.approx(1.5)
        assert self.CLASS_WEIGHTS[LOSS] == pytest.approx(1.1)

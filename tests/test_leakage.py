"""
tests/test_leakage.py

Purpose: Verify that all seven leakage prevention rules are enforced correctly.
Tests are adversarial — each test deliberately introduces a leakage violation
and asserts that the leakage guard catches it.

Design spec ref: §7 Repository Structure — tests/test_leakage.py
"Each test deliberately introduces a leakage violation and asserts that the
leakage guard catches it."
"""

import pandas as pd
import pytest

from src.leakage_guard import (
    FREEZE_DATE,
    TACTICAL_GAP_TEAMS,
    LeakageError,
    check_annex_c,
    check_canonical_names,
    check_ds4_wc_null_scores,
    check_ds8_year_ceiling,
    check_elo_snapshot,
    check_form_window,
    check_freeze_date,
    check_no_forbidden_datasets,
    check_no_md3_in_features,
    check_no_synthetic_data,
    check_tactical_gap_preserved,
    check_training_rows_chronological,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _future_df(date_str: str = "2026-07-01") -> pd.DataFrame:
    return pd.DataFrame({"date": [date_str], "value": [1.0]})


def _feature_table(**kwargs) -> pd.DataFrame:
    """Build a minimal feature table row."""
    defaults = {
        "elo_rating":       1800.0,
        "tourn_pts_md2":    3,
        "tourn_gd_md2":     1,
        "has_full_tactical_md2": True,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Rule L1 — check_freeze_date
# ---------------------------------------------------------------------------

class TestCheckFreezeDate:
    def test_future_date_caught(self):
        """A row dated 2026-07-01 is after the freeze date and must be caught."""
        df = _future_df("2026-07-01")
        with pytest.raises(LeakageError) as exc_info:
            check_freeze_date(df, "date")
        assert exc_info.value.violation_type == "FUTURE_DATE"
        assert exc_info.value.rule == "L1"

    def test_exactly_freeze_date_passes(self):
        """A row dated exactly 2026-06-23 (the freeze date) is permitted."""
        df = pd.DataFrame({"date": ["2026-06-23"], "value": [1]})
        check_freeze_date(df, "date")  # must not raise

    def test_pre_freeze_date_passes(self):
        df = pd.DataFrame({"date": ["2026-06-22", "2025-01-15"], "value": [1, 2]})
        check_freeze_date(df, "date")  # must not raise

    def test_missing_date_column_raises(self):
        df = pd.DataFrame({"score": [1]})
        with pytest.raises(LeakageError) as exc_info:
            check_freeze_date(df, "date")
        assert exc_info.value.violation_type == "FUTURE_DATE"

    def test_null_dates_are_ignored(self):
        """Null dates (future placeholder rows) must be silently skipped."""
        df = pd.DataFrame({"date": [None, "2026-06-20", "2026-06-23"]})
        check_freeze_date(df, "date")  # must not raise

    def test_violation_type_is_future_date(self):
        df = _future_df("2026-12-31")
        with pytest.raises(LeakageError) as exc_info:
            check_freeze_date(df, "date")
        assert exc_info.value.violation_type == "FUTURE_DATE"


# ---------------------------------------------------------------------------
# Rule L2 — check_elo_snapshot
# ---------------------------------------------------------------------------

class TestCheckEloSnapshot:
    def _make_ds2(self, snapshot_dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "snapshot_date": snapshot_dates,
            "country":       ["Germany"] * len(snapshot_dates),
            "rating":        [1800.0] * len(snapshot_dates),
        })

    def test_wrong_elo_snapshot_caught(self):
        """A 2026-06-15 snapshot (during the tournament) must raise."""
        ds2 = self._make_ds2(["2026-06-15"])
        with pytest.raises(LeakageError) as exc_info:
            check_elo_snapshot(ds2)
        assert exc_info.value.violation_type == "WRONG_SNAPSHOT"

    def test_correct_elo_snapshot_passes(self):
        """The approved 2026-05-27 snapshot must pass."""
        ds2 = self._make_ds2(["2026-05-27"])
        check_elo_snapshot(ds2)  # must not raise

    def test_historical_year_end_snapshots_pass(self):
        """YYYY-12-31 snapshots for years < 2026 are used in backtesting — allowed."""
        ds2 = self._make_ds2(["2020-12-31", "2021-12-31", "2022-12-31"])
        check_elo_snapshot(ds2)  # must not raise

    def test_2026_12_31_snapshot_caught(self):
        """The 2026-12-31 row in DS2 must never enter feature construction."""
        ds2 = self._make_ds2(["2026-12-31"])
        with pytest.raises(LeakageError) as exc_info:
            check_elo_snapshot(ds2)
        assert exc_info.value.violation_type == "WRONG_SNAPSHOT"

    def test_mixed_valid_and_invalid_caught(self):
        """Even one invalid snapshot in a batch must raise."""
        ds2 = self._make_ds2(["2026-05-27", "2026-06-15"])
        with pytest.raises(LeakageError):
            check_elo_snapshot(ds2)


# ---------------------------------------------------------------------------
# Rule L6 — check_no_synthetic_data (DS3 column fingerprint)
# ---------------------------------------------------------------------------

class TestCheckNoSyntheticData:
    def test_tournament_rating_column_caught(self):
        """DS3's 'tournament_rating' column is a high-specificity synthetic signal."""
        df = _feature_table(tournament_rating=5.5)
        with pytest.raises(LeakageError) as exc_info:
            check_no_synthetic_data(df)
        assert exc_info.value.violation_type == "SYNTHETIC_DATA"

    def test_player_id_column_caught(self):
        df = _feature_table(player_id=12345)
        with pytest.raises(LeakageError) as exc_info:
            check_no_synthetic_data(df)
        assert exc_info.value.violation_type == "SYNTHETIC_DATA"

    def test_clutch_performance_score_caught(self):
        df = _feature_table(clutch_performance_score=7.2)
        with pytest.raises(LeakageError) as exc_info:
            check_no_synthetic_data(df)
        assert exc_info.value.violation_type == "SYNTHETIC_DATA"

    def test_clean_feature_table_passes(self):
        """A feature table with no DS3 columns must pass."""
        df = _feature_table()
        check_no_synthetic_data(df)  # must not raise

    def test_market_value_caught(self):
        df = _feature_table(market_value_eur=50_000_000)
        with pytest.raises(LeakageError) as exc_info:
            check_no_synthetic_data(df)
        assert exc_info.value.violation_type == "SYNTHETIC_DATA"


# ---------------------------------------------------------------------------
# Rule L6 extended — check_no_forbidden_datasets (DS13/DS14)
# ---------------------------------------------------------------------------

class TestCheckNoForbiddenDatasets:
    def test_wins_last_10_matches_caught(self):
        df = pd.DataFrame({"wins_last_10_matches": [7], "other_col": [1]})
        with pytest.raises(LeakageError) as exc_info:
            check_no_forbidden_datasets(df)
        assert exc_info.value.violation_type in ("FORBIDDEN_DATASET", "SYNTHETIC_DATA")

    def test_win_rate_last_year_caught(self):
        df = pd.DataFrame({"win_rate_last_year": [0.6], "other": [1]})
        with pytest.raises(LeakageError) as exc_info:
            check_no_forbidden_datasets(df)
        assert exc_info.value.violation_type in ("FORBIDDEN_DATASET", "SYNTHETIC_DATA")

    def test_clean_dataframe_passes(self):
        df = pd.DataFrame({"elo_rating": [1800], "form_win_rate_last10": [0.7]})
        check_no_forbidden_datasets(df)  # must not raise

    def test_multiple_dataframes_checked(self):
        """Passing multiple DataFrames — any violation raises."""
        clean = pd.DataFrame({"elo_rating": [1800]})
        dirty = pd.DataFrame({"wins_last_10_matches": [7]})
        with pytest.raises(LeakageError):
            check_no_forbidden_datasets(clean, dirty)


# ---------------------------------------------------------------------------
# Rule L1 on tournament features — check_no_md3_in_features
# ---------------------------------------------------------------------------

class TestCheckNoMd3InFeatures:
    def test_max_pts_md3_caught(self):
        """tourn_pts_md2 = 9 is impossible in 2 games; signals MD3 data leaked in."""
        df = _feature_table(tourn_pts_md2=9)
        with pytest.raises(LeakageError) as exc_info:
            check_no_md3_in_features(df)
        assert exc_info.value.violation_type == "FUTURE_DATE"

    def test_legal_pts_passes(self):
        """Maximum legitimate value is 6pts (2 wins from 2 MD1+MD2 games)."""
        for pts in (0, 1, 2, 3, 4, 5, 6):
            df = _feature_table(tourn_pts_md2=pts)
            check_no_md3_in_features(df)  # must not raise

    def test_column_missing_skipped(self):
        """If tourn_pts_md2 is absent, the check is skipped without raising."""
        df = pd.DataFrame({"elo_rating": [1800]})
        check_no_md3_in_features(df)  # must not raise


# ---------------------------------------------------------------------------
# Rule L5 — check_tactical_gap_preserved
# ---------------------------------------------------------------------------

class TestCheckTacticalGapPreserved:
    def _england_row(self, tactical_flag: bool) -> pd.DataFrame:
        return pd.DataFrame([{
            "team":                  "England",
            "has_full_tactical_md2": tactical_flag,
            "tourn_pts_md2":         3,
        }])

    def test_england_true_tactical_caught(self):
        """England's MD2 was scores-only; has_full_tactical_md2 must be False."""
        df = self._england_row(True)
        with pytest.raises(LeakageError) as exc_info:
            check_tactical_gap_preserved(df)
        assert exc_info.value.violation_type == "TACTICAL_IMPUTATION"

    def test_england_false_tactical_passes(self):
        df = self._england_row(False)
        check_tactical_gap_preserved(df)  # must not raise

    @pytest.mark.parametrize("team", ["Portugal", "Colombia", "Croatia"])
    def test_other_tactical_gap_teams_caught(self, team):
        df = pd.DataFrame([{
            "team":                  team,
            "has_full_tactical_md2": True,
            "tourn_pts_md2":         3,
        }])
        with pytest.raises(LeakageError) as exc_info:
            check_tactical_gap_preserved(df)
        assert exc_info.value.violation_type == "TACTICAL_IMPUTATION"

    def test_germany_full_tactical_passes(self):
        """Germany's MD2 data is complete; has_full_tactical_md2 = True is correct."""
        df = pd.DataFrame([{
            "team":                  "Germany",
            "has_full_tactical_md2": True,
            "tourn_pts_md2":         6,
        }])
        check_tactical_gap_preserved(df)  # must not raise

    def test_tactical_gap_teams_constant_matches_design_spec(self):
        """The four affected teams are exactly as specified in design doc §2 Rule L5."""
        expected = frozenset({"Portugal", "Colombia", "England", "Croatia"})
        assert TACTICAL_GAP_TEAMS == expected


# ---------------------------------------------------------------------------
# Rule L1 on training — check_training_rows_chronological
# ---------------------------------------------------------------------------

class TestCheckTrainingRowsChronological:
    def test_future_training_year_caught(self):
        """A training row where elo_year_used == match_year violates Rule L1."""
        df = pd.DataFrame({
            "match_year":    [1998, 2002, 2006, 2026],
            "elo_year_used": [1997, 2001, 2006, 2025],  # 2006 row: elo_year == match_year
        })
        with pytest.raises(LeakageError) as exc_info:
            check_training_rows_chronological(df)
        assert exc_info.value.violation_type == "FUTURE_DATE"

    def test_valid_years_pass(self):
        df = pd.DataFrame({"match_year": [1998, 2002, 2006, 2010, 2014, 2018, 2022]})
        check_training_rows_chronological(df)  # must not raise

    def test_elo_year_used_must_be_before_match_year(self):
        """elo_year_used must be match_year − 1 (no future Elo data in training)."""
        df = pd.DataFrame({
            "match_year":   [2006, 2010],
            "elo_year_used":[2006, 2010],  # same year — violation
        })
        with pytest.raises(LeakageError):
            check_training_rows_chronological(df)

    def test_correct_elo_year_passes(self):
        df = pd.DataFrame({
            "match_year":   [2006, 2010, 2014, 2018, 2022],
            "elo_year_used":[2005, 2009, 2013, 2017, 2021],
        })
        check_training_rows_chronological(df)  # must not raise


# ---------------------------------------------------------------------------
# Rule L4 — check_ds8_year_ceiling
# ---------------------------------------------------------------------------

class TestCheckDs8YearCeiling:
    def _make_ds8(self, years: list[int]) -> pd.DataFrame:
        return pd.DataFrame({
            "Year":       years,
            "home_team":  ["Germany"] * len(years),
            "away_team":  ["France"]  * len(years),
            "home_score": [1]         * len(years),
            "away_score": [0]         * len(years),
        })

    def test_2024_year_caught(self):
        """DS8 is an archive through 2022; a 2024 row signals contamination."""
        ds8 = self._make_ds8([2018, 2022, 2024])
        with pytest.raises(LeakageError) as exc_info:
            check_ds8_year_ceiling(ds8)
        assert exc_info.value.violation_type in ("FUTURE_DATE", "WRONG_SNAPSHOT")

    def test_max_year_2022_passes(self):
        ds8 = self._make_ds8([1998, 2002, 2006, 2010, 2014, 2018, 2022])
        check_ds8_year_ceiling(ds8)  # must not raise


# ---------------------------------------------------------------------------
# Rule L1 on DS4 form window — check_form_window
# ---------------------------------------------------------------------------

class TestCheckFormWindow:
    def test_post_window_date_caught(self):
        """DS4 form data must not include matches after 2026-06-10."""
        df = pd.DataFrame({"date": ["2026-06-11", "2026-06-09"]})
        with pytest.raises(LeakageError) as exc_info:
            check_form_window(df, date_column="date")
        assert exc_info.value.violation_type == "FUTURE_DATE"

    def test_exactly_form_window_end_passes(self):
        df = pd.DataFrame({"date": ["2026-06-10"]})
        check_form_window(df, date_column="date")  # must not raise

    def test_pre_window_dates_pass(self):
        df = pd.DataFrame({"date": ["2024-01-15", "2025-03-20", "2026-06-09"]})
        check_form_window(df, date_column="date")  # must not raise


# ---------------------------------------------------------------------------
# Annex C structural check — check_annex_c
# ---------------------------------------------------------------------------

class TestCheckAnnexC:
    def _valid_annex_c(self) -> pd.DataFrame:
        from itertools import combinations
        groups = list("ABCDEFGHIJKL")
        rows = []
        for combo in combinations(groups, 8):
            row = {"qualifying_groups": ",".join(sorted(combo))}
            for col in ["m75", "m78", "m79", "m80", "m81", "m82", "m85", "m88"]:
                row[col] = f"3{combo[0]}"  # placeholder value
            rows.append(row)
        return pd.DataFrame(rows)

    def test_correct_annex_c_passes(self):
        df = self._valid_annex_c()
        check_annex_c(df)  # must not raise

    def test_wrong_row_count_caught(self):
        """Fewer than 495 rows is a sign of file corruption."""
        df = self._valid_annex_c().head(10)
        with pytest.raises(LeakageError):
            check_annex_c(df)

    def test_missing_column_caught(self):
        df = self._valid_annex_c().drop(columns=["m75"])
        with pytest.raises(LeakageError):
            check_annex_c(df)


# ---------------------------------------------------------------------------
# Canonical name check — check_canonical_names
# ---------------------------------------------------------------------------

class TestCheckCanonicalNames:
    def test_valid_names_pass(self):
        check_canonical_names(["Germany", "France", "Brazil"], context="test")

    def test_invalid_name_caught(self):
        with pytest.raises(LeakageError) as exc_info:
            check_canonical_names(["Germany", "England U21"], context="test")
        assert exc_info.value.violation_type == "NAME_ERROR"

    def test_all_48_pass(self):
        from src.name_map import CANONICAL_48
        check_canonical_names(list(CANONICAL_48), context="all_48")


# ---------------------------------------------------------------------------
# LeakageError contract
# ---------------------------------------------------------------------------

class TestLeakageErrorContract:
    def test_valid_violation_types(self):
        """Each known violation type can be instantiated."""
        for vt in ["FUTURE_DATE", "SYNTHETIC_DATA", "WRONG_SNAPSHOT",
                   "FORBIDDEN_DATASET", "TACTICAL_IMPUTATION",
                   "INVALID_REFERENCE_DATA", "NAME_ERROR"]:
            err = LeakageError("test", violation_type=vt, rule="L0")
            assert err.violation_type == vt

    def test_invalid_violation_type_raises_value_error(self):
        with pytest.raises(ValueError):
            LeakageError("test", violation_type="MADE_UP_TYPE")

    def test_rule_attribute_set(self):
        err = LeakageError("test", violation_type="FUTURE_DATE", rule="L1")
        assert err.rule == "L1"

    def test_str_includes_violation_type(self):
        err = LeakageError("something bad", violation_type="FUTURE_DATE", rule="L1")
        s = str(err)
        assert "FUTURE_DATE" in s

    def test_is_exception(self):
        err = LeakageError("test", violation_type="FUTURE_DATE")
        assert isinstance(err, Exception)

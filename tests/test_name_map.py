"""
tests/test_name_map.py

Purpose: Verify that every name translation is correct, that no team is silently
dropped or incorrectly mapped, and that the canonical name set is complete.

Design spec ref: §7 Repository Structure — tests/test_name_map.py
All 48 teams must round-trip correctly through all three maps, and no unknown name
may raise a silent identity return.
"""

import pytest

from src.name_map import (
    CANONICAL_48,
    DS17_NAME_TO_CANONICAL,
    DS17_PLACEHOLDER_ID_TO_CANONICAL,
    DS17_PLACEHOLDER_NAME_TO_CANONICAL,
    TO_CANONICAL,
    WC_DEBUTANTS,
    apply_to_series,
    assert_all_canonical,
    canonicalize,
    canonicalize_id,
)


# ---------------------------------------------------------------------------
# CANONICAL_48 integrity
# ---------------------------------------------------------------------------

class TestCanonical48:
    def test_exactly_48_teams(self):
        assert len(CANONICAL_48) == 48, (
            f"CANONICAL_48 must contain exactly 48 teams, got {len(CANONICAL_48)}"
        )

    def test_no_empty_strings(self):
        assert "" not in CANONICAL_48

    def test_no_whitespace_names(self):
        for name in CANONICAL_48:
            assert name == name.strip(), (
                f"Team name {name!r} has leading/trailing whitespace"
            )

    def test_known_teams_present(self):
        expected_sample = {
            "Spain", "France", "Brazil", "Argentina", "Germany",
            "England", "Portugal", "Netherlands", "United States",
            "Mexico", "Canada", "Japan", "Korea Republic", "IR Iran",
            "Türkiye", "Côte d'Ivoire", "Bosnia-Herzegovina", "Congo DR",
            "Cape Verde", "Curaçao", "Uzbekistan", "Jordan",
        }
        missing = expected_sample - CANONICAL_48
        assert not missing, f"Missing from CANONICAL_48: {sorted(missing)}"


# ---------------------------------------------------------------------------
# TO_CANONICAL variant → canonical mappings
# ---------------------------------------------------------------------------

class TestToCanonical:
    """Each key in TO_CANONICAL must map to a valid CANONICAL_48 member."""

    def test_all_values_are_canonical(self):
        for variant, canonical in TO_CANONICAL.items():
            assert canonical in CANONICAL_48, (
                f"TO_CANONICAL[{variant!r}] = {canonical!r} is not in CANONICAL_48"
            )

    # DS2 (Elo) variants
    def test_bosnia_and_herzegovina_ds2(self):
        assert TO_CANONICAL["Bosnia and Herzegovina"] == "Bosnia-Herzegovina"

    def test_dr_congo_ds2(self):
        assert TO_CANONICAL["DR Congo"] == "Congo DR"

    def test_iran_ds2(self):
        assert TO_CANONICAL["Iran"] == "IR Iran"

    def test_ivory_coast_ds2(self):
        assert TO_CANONICAL["Ivory Coast"] == "Côte d'Ivoire"

    def test_south_korea_ds2(self):
        assert TO_CANONICAL["South Korea"] == "Korea Republic"

    def test_turkey_ds2(self):
        assert TO_CANONICAL["Turkey"] == "Türkiye"

    # DS1 (Arc3) variants
    def test_bosnia_ds1_abbreviated(self):
        assert TO_CANONICAL["Bosnia & Herz."] == "Bosnia-Herzegovina"

    def test_cabo_verde_ds1(self):
        assert TO_CANONICAL["Cabo Verde"] == "Cape Verde"

    # DS6 variant
    def test_congo_ds6(self):
        assert TO_CANONICAL["Congo"] == "Congo DR"

    # DS8 variant
    def test_czech_republic_ds8(self):
        assert TO_CANONICAL["Czech Republic"] == "Czechia"

    # DS10 variant
    def test_usa_ds10(self):
        assert TO_CANONICAL["USA"] == "United States"


# ---------------------------------------------------------------------------
# canonicalize() function
# ---------------------------------------------------------------------------

class TestCanonicalize:
    """canonicalize() must apply all three maps in order and return the input
    unchanged for already-canonical names."""

    def test_identity_for_canonical_names(self):
        """Teams already in canonical form are returned unchanged."""
        identity_teams = [
            "Germany", "France", "Brazil", "Spain", "Argentina",
            "England", "Netherlands", "Portugal", "Mexico", "Japan",
        ]
        for team in identity_teams:
            assert canonicalize(team) == team, (
                f"canonicalize({team!r}) should be identity, got {canonicalize(team)!r}"
            )

    def test_south_korea_to_canonical(self):
        assert canonicalize("South Korea") == "Korea Republic"

    def test_iran_to_canonical(self):
        assert canonicalize("Iran") == "IR Iran"

    def test_ivory_coast_to_canonical(self):
        assert canonicalize("Ivory Coast") == "Côte d'Ivoire"

    def test_dr_congo_to_canonical(self):
        assert canonicalize("DR Congo") == "Congo DR"

    def test_turkey_to_canonical(self):
        assert canonicalize("Turkey") == "Türkiye"

    def test_usa_to_canonical(self):
        assert canonicalize("USA") == "United States"

    def test_cabo_verde_ds1_variant(self):
        assert canonicalize("Cabo Verde") == "Cape Verde"

    def test_bosnia_ds1_abbreviated(self):
        assert canonicalize("Bosnia & Herz.") == "Bosnia-Herzegovina"

    def test_czech_republic_ds8_variant(self):
        assert canonicalize("Czech Republic") == "Czechia"

    def test_congo_ds6_variant(self):
        assert canonicalize("Congo") == "Congo DR"

    def test_ds17_real_team_variants(self):
        """DS17_NAME_TO_CANONICAL variants are resolved by canonicalize()."""
        assert canonicalize("South Korea") == "Korea Republic"
        assert canonicalize("USA") == "United States"
        # Cabo Verde handled by TO_CANONICAL
        assert canonicalize("Cabo Verde") == "Cape Verde"

    def test_placeholder_name_resolution(self):
        """DS17 placeholder names are resolved by canonicalize()."""
        assert canonicalize("Winner UEFA Playoff D") == "Czechia"
        assert canonicalize("Winner UEFA Playoff A") == "Bosnia-Herzegovina"
        assert canonicalize("Winner UEFA Playoff C") == "Türkiye"
        assert canonicalize("Winner UEFA Playoff B") == "Sweden"
        assert canonicalize("Winner FIFA Playoff 2") == "Iraq"
        assert canonicalize("Winner FIFA Playoff 1") == "Congo DR"

    def test_strips_whitespace(self):
        assert canonicalize("  Germany  ") == "Germany"

    def test_unknown_name_returns_itself(self):
        """Unknown names are returned unchanged (no exception)."""
        unknown = "England U21"
        assert canonicalize(unknown) == unknown


# ---------------------------------------------------------------------------
# canonicalize_id() function
# ---------------------------------------------------------------------------

class TestCanonicalizeId:
    """Placeholder team IDs must resolve to their real canonical names.
    Real team IDs should fall through to canonicalize()."""

    @pytest.mark.parametrize("team_id,expected", [
        (4,  "Czechia"),
        (6,  "Bosnia-Herzegovina"),
        (16, "Türkiye"),
        (23, "Sweden"),
        (35, "Iraq"),
        (42, "Congo DR"),
    ])
    def test_placeholder_ids_resolve(self, team_id, expected):
        result = canonicalize_id(team_id, "Winner placeholder")
        assert result == expected, (
            f"canonicalize_id({team_id}, ...) should be {expected!r}, got {result!r}"
        )

    def test_real_team_id_delegates_to_canonicalize(self):
        """A non-placeholder ID falls through to canonicalize(current_name)."""
        result = canonicalize_id(1, "South Korea")
        assert result == "Korea Republic"

    def test_real_team_canonical_name_unchanged(self):
        result = canonicalize_id(17, "Germany")
        assert result == "Germany"

    def test_all_placeholder_ids_in_canonical_48(self):
        for team_id, name in DS17_PLACEHOLDER_ID_TO_CANONICAL.items():
            assert name in CANONICAL_48, (
                f"Placeholder ID {team_id} resolves to {name!r} which is not in CANONICAL_48"
            )


# ---------------------------------------------------------------------------
# apply_to_series()
# ---------------------------------------------------------------------------

class TestApplyToSeries:
    def test_vectorised_canonicalize(self):
        import pandas as pd

        raw = pd.Series(["South Korea", "Iran", "Germany", "USA", "DR Congo"])
        result = apply_to_series(raw)
        assert list(result) == [
            "Korea Republic", "IR Iran", "Germany", "United States", "Congo DR"
        ]

    def test_non_string_values_pass_through(self):
        import pandas as pd

        raw = pd.Series(["Germany", None, float("nan"), "France"])
        result = apply_to_series(raw)
        assert result.iloc[0] == "Germany"
        assert result.iloc[3] == "France"


# ---------------------------------------------------------------------------
# assert_all_canonical()
# ---------------------------------------------------------------------------

class TestAssertAllCanonical:
    def test_all_48_pass(self):
        assert_all_canonical(CANONICAL_48, context="test")

    def test_single_valid_name_passes(self):
        assert_all_canonical(["Germany"], context="test")

    def test_invalid_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Non-canonical"):
            assert_all_canonical(["England U21"], context="test")

    def test_mix_valid_and_invalid_raises(self):
        with pytest.raises(ValueError):
            assert_all_canonical(["Germany", "Not A Team"], context="test")

    def test_empty_list_passes(self):
        assert_all_canonical([], context="test")


# ---------------------------------------------------------------------------
# WC_DEBUTANTS
# ---------------------------------------------------------------------------

class TestWCDebutants:
    def test_debutants_subset_of_canonical_48(self):
        non_canonical = WC_DEBUTANTS - CANONICAL_48
        assert not non_canonical, (
            f"WC_DEBUTANTS contains non-canonical names: {non_canonical}"
        )

    def test_known_debutants_present(self):
        expected = {"Cape Verde", "Congo DR", "Curaçao",
                    "Haiti", "Iraq", "Jordan", "Uzbekistan"}
        assert expected == WC_DEBUTANTS, (
            f"WC_DEBUTANTS mismatch. Expected {expected}, got {WC_DEBUTANTS}"
        )

    def test_established_teams_not_in_debutants(self):
        established = {"Germany", "Brazil", "France", "Argentina", "Spain",
                       "England", "Mexico", "Japan", "Korea Republic"}
        overlap = established & WC_DEBUTANTS
        assert not overlap, (
            f"Established teams found in WC_DEBUTANTS: {overlap}"
        )


# ---------------------------------------------------------------------------
# DS17 placeholder maps
# ---------------------------------------------------------------------------

class TestPlaceholderMaps:
    def test_exactly_six_placeholder_ids(self):
        assert len(DS17_PLACEHOLDER_ID_TO_CANONICAL) == 6

    def test_exactly_six_placeholder_names(self):
        assert len(DS17_PLACEHOLDER_NAME_TO_CANONICAL) == 6

    def test_placeholder_names_consistent_with_ids(self):
        id_values   = set(DS17_PLACEHOLDER_ID_TO_CANONICAL.values())
        name_values = set(DS17_PLACEHOLDER_NAME_TO_CANONICAL.values())
        assert id_values == name_values, (
            f"Placeholder ID map values {id_values} != name map values {name_values}"
        )

    def test_all_placeholder_resolutions_in_canonical_48(self):
        for name, canonical in DS17_PLACEHOLDER_NAME_TO_CANONICAL.items():
            assert canonical in CANONICAL_48, (
                f"{name!r} → {canonical!r} but {canonical!r} is not in CANONICAL_48"
            )

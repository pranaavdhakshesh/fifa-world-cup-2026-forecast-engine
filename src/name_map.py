"""
Canonical name translation for the FIFA WC 2026 Forecast Engine.

Authority: DS9 (schedule_2026.csv) defines the 48 canonical team names.
All other datasets must be mapped to these spellings before any join.

Verified against actual dataset contents:
  DS2  (elo_ratings_wc2026.csv)    — 6 variants
  DS4  (results.csv)               — 7 variants
  DS6  (shootouts.csv)             — 3 variants
  DS8  (matches_1930_2022.csv)     — 2 variants + debutants absent entirely
  DS10 (fifa_ranking_2026-06-08.csv) — 3 variants
  DS17 (teams.csv)                 — 3 real-name variants + 6 placeholder IDs
"""

# ---------------------------------------------------------------------------
# 1. Variant → canonical.
#    Key  : name as it appears in a source dataset.
#    Value: DS9 canonical spelling.
#
#    Sources that need no mapping (exact match on all 48 teams) are omitted.
#    DS1 (Arc3 matches.csv) uses DS9 canonical spellings already — no mapping.
#    DS1-ext (june23_results.csv) uses DS9 canonical spellings — no mapping.
# ---------------------------------------------------------------------------
TO_CANONICAL: dict[str, str] = {
    # ---------- DS2 (Elo) ----------
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "DR Congo":               "Congo DR",
    "Iran":                   "IR Iran",
    "Ivory Coast":            "Côte d'Ivoire",
    "South Korea":            "Korea Republic",
    "Turkey":                 "Türkiye",

    # ---------- DS4 (results.csv) ----------
    # "Bosnia and Herzegovina", "DR Congo", "Iran", "Ivory Coast",
    # "South Korea", "Turkey" — already covered above.
    # Cape Verde appears in DS4 as "Cape Verde" which IS the canonical name;
    # DS10 uses "Cabo Verde" — that mapping is handled below under DS10.

    # ---------- DS1 (Arc3 matches.csv) ----------
    # DS1 uses two abbreviated / alternate spellings.
    "Bosnia & Herz.":         "Bosnia-Herzegovina",
    "Cabo Verde":             "Cape Verde",

    # ---------- DS6 (shootouts.csv) ----------
    # "DR Congo", "Iran", "South Korea" — already covered above.
    # "Congo" appears in DS6 as a separate entry (older name); map it too
    # so historical shootout lookups for Congo DR don't silently miss rows.
    "Congo":                  "Congo DR",

    # ---------- DS8 (matches_1930_2022.csv) ----------
    # "Bosnia and Herzegovina", "Côte d'Ivoire", "IR Iran", "Korea Republic"
    # are already exact matches or covered above.
    # Czech Republic is the name used in DS8 for all historical WC appearances
    # prior to the 2016 rebranding.  Czechia debuted under that name in 2022
    # qualification but did not qualify; their DS8 WC history is as Czech Republic.
    "Czech Republic":         "Czechia",

    # ---------- DS10 (FIFA ranking 2026-06-08) ----------
    "USA":                    "United States",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",   # duplicate key — same value
    "Cabo Verde":             "Cape Verde",
}

# ---------------------------------------------------------------------------
# 2. DS17 real-team name variants → canonical.
#    DS17 contains 42 real teams; three use non-canonical spellings.
# ---------------------------------------------------------------------------
DS17_NAME_TO_CANONICAL: dict[str, str] = {
    "South Korea": "Korea Republic",
    "USA":         "United States",
    "Cabo Verde":  "Cape Verde",
}

# ---------------------------------------------------------------------------
# 3. DS17 placeholder resolution.
#    Six rows in DS17 are placeholders for late-qualifying teams not confirmed
#    when Arc_base was assembled (January 2026).  DS16 match records reference
#    the placeholder IDs; these must be resolved before any bracket join.
#
#    Resolution verified by cross-referencing DS17 group letters against the
#    confirmed 2026 group compositions in the design specification (§6).
# ---------------------------------------------------------------------------
DS17_PLACEHOLDER_ID_TO_CANONICAL: dict[int, str] = {
    4:  "Czechia",            # Group A — Winner UEFA Playoff D
    6:  "Bosnia-Herzegovina", # Group B — Winner UEFA Playoff A
    16: "Türkiye",            # Group D — Winner UEFA Playoff C
    23: "Sweden",             # Group F — Winner UEFA Playoff B
    35: "Iraq",               # Group I — Winner FIFA Playoff 2
    42: "Congo DR",           # Group K — Winner FIFA Playoff 1
}

DS17_PLACEHOLDER_NAME_TO_CANONICAL: dict[str, str] = {
    "Winner UEFA Playoff D": "Czechia",
    "Winner UEFA Playoff A": "Bosnia-Herzegovina",
    "Winner UEFA Playoff C": "Türkiye",
    "Winner UEFA Playoff B": "Sweden",
    "Winner FIFA Playoff 2": "Iraq",
    "Winner FIFA Playoff 1": "Congo DR",
}

# ---------------------------------------------------------------------------
# 4. Teams with zero WC appearances in DS8.
#    These teams receive missing-class encoding for all Group 3 (WC historical)
#    features — never zero.  Listed here as documentation; features.py reads
#    this set to apply the missing-class flag.
# ---------------------------------------------------------------------------
WC_DEBUTANTS: frozenset[str] = frozenset({
    "Bosnia-Herzegovina",
    "Cape Verde",
    "Congo DR",
    "Curaçao",
    "Jordan",
    "Uzbekistan",
})

# ---------------------------------------------------------------------------
# 5. The 48 canonical team names (DS9 authority).
#    Used in validation assertions throughout the pipeline.
# ---------------------------------------------------------------------------
CANONICAL_48: frozenset[str] = frozenset({
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia-Herzegovina", "Brazil", "Canada", "Cape Verde", "Colombia",
    "Congo DR", "Croatia", "Curaçao", "Czechia", "Côte d'Ivoire",
    "Ecuador", "Egypt", "England", "France", "Germany",
    "Ghana", "Haiti", "IR Iran", "Iraq", "Japan",
    "Jordan", "Korea Republic", "Mexico", "Morocco", "Netherlands",
    "New Zealand", "Norway", "Panama", "Paraguay", "Portugal",
    "Qatar", "Saudi Arabia", "Scotland", "Senegal", "South Africa",
    "Spain", "Sweden", "Switzerland", "Tunisia", "Türkiye",
    "United States", "Uruguay", "Uzbekistan",
})


# ---------------------------------------------------------------------------
# 6. Public helpers.
# ---------------------------------------------------------------------------

def canonicalize(name: str) -> str:
    """Return the DS9 canonical spelling for *name*, or *name* unchanged.

    Applies TO_CANONICAL first, then DS17_NAME_TO_CANONICAL, then
    DS17_PLACEHOLDER_NAME_TO_CANONICAL.  Returns the input string unmodified
    if no mapping exists (the name is either already canonical or genuinely
    unknown — callers should validate against CANONICAL_48 if strictness is
    required).
    """
    name = name.strip()
    return (
        TO_CANONICAL.get(name)
        or DS17_NAME_TO_CANONICAL.get(name)
        or DS17_PLACEHOLDER_NAME_TO_CANONICAL.get(name)
        or name
    )


def canonicalize_id(ds17_id: int, current_name: str) -> str:
    """Resolve a DS17 team_id to a canonical name.

    For placeholder IDs (4, 6, 16, 23, 35, 42) returns the resolved real team.
    For real team IDs delegates to canonicalize(current_name).
    """
    if ds17_id in DS17_PLACEHOLDER_ID_TO_CANONICAL:
        return DS17_PLACEHOLDER_ID_TO_CANONICAL[ds17_id]
    return canonicalize(current_name)


def apply_to_series(series):
    """Vectorised helper: apply canonicalize() to a pandas Series of names."""
    return series.map(lambda x: canonicalize(x) if isinstance(x, str) else x)


def assert_all_canonical(names, context: str = "") -> None:
    """Raise ValueError if any name in *names* is not in CANONICAL_48."""
    unknown = [n for n in names if n not in CANONICAL_48]
    if unknown:
        prefix = f"[{context}] " if context else ""
        raise ValueError(
            f"{prefix}Non-canonical team names detected: {unknown}\n"
            f"Apply name_map.canonicalize() before this assertion."
        )

"""Turn raw EPC fields into clean modelling features.

EPC descriptions are free-ish text ("Cavity wall, filled cavity", "Pitched, 270 mm
loft insulation"). These functions map them to a small number of consistent groups
that a housing professional would recognise.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .load import compact, concat_compact


# ---------------------------------------------------------------------------
# Field parsers (each takes a Series of raw strings and returns a clean Series)
# ---------------------------------------------------------------------------
def to_dates(s: pd.Series) -> pd.Series:
    """Parse dates. Text first, because a categorical column can come back categorical rather than as dates."""
    return pd.to_datetime(s.astype("string"), errors="coerce").astype("datetime64[ns]")


def _lower(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def parse_construction_year(s: pd.Series) -> pd.Series:
    """Convert age bands to a representative build year.

    'England and Wales: 1930-1949' -> 1940 (midpoint)
    'England and Wales: before 1900' -> 1890
    'England and Wales: 2012 onwards' -> 2015
    '2019' -> 2019
    """

    def one(value):
        if pd.isna(value):
            return np.nan
        text = str(value).lower()
        years = [int(y) for y in re.findall(r"(1[89]\d{2}|20\d{2})", text)]
        if not years:
            return np.nan
        if "before" in text:
            return years[0] - 10
        if "onwards" in text:
            return years[0] + 3
        if len(years) >= 2:
            return (years[0] + years[1]) / 2
        return years[0]

    return s.map(one).astype(float)


def group_fuel(s: pd.Series) -> pd.Series:
    t = _lower(s)
    out = pd.Series("other", index=s.index, dtype="object")
    community = t.str.contains("community", na=False) & ~t.str.contains("not community", na=False)
    rules = [
        (t.str.contains("electric", na=False), "electricity"),
        (t.str.contains("oil", na=False), "oil"),
        (t.str.contains("lpg|bottled|gas: lpg", na=False), "lpg"),
        (t.str.contains("coal|anthracite|smokeless|wood|biomass|dual fuel", na=False), "solid_fuel"),
        (t.str.contains("mains gas", na=False), "mains_gas"),
        (community | t.str.contains("heat network", na=False), "communal"),
    ]
    for mask, label in rules:
        out[mask] = label
    # A placeholder label used when there is no heating system or data comes from a heat network
    out[t.str.contains("to be used only when", na=False)] = np.nan
    out[t.isna()] = np.nan
    return out


def group_wall_type(s: pd.Series) -> pd.Series:
    t = _lower(s)
    out = pd.Series("other", index=s.index, dtype="object")
    rules = [
        (t.str.contains("thermal transmittance", na=False), "new_build_spec"),
        (t.str.contains("timber", na=False), "timber_frame"),
        (t.str.contains("system built", na=False), "system_built"),
        (t.str.contains("solid brick|sandstone|limestone|granite|whinstone|stone", na=False), "solid"),
        (t.str.contains("cavity", na=False), "cavity"),
    ]
    for mask, label in rules:
        out[mask] = label
    out[t.isna()] = np.nan
    return out


def group_wall_insulation(s: pd.Series) -> pd.Series:
    t = _lower(s)
    out = pd.Series("unknown", index=s.index, dtype="object")
    insulated = t.str.contains(r"filled cavity|with internal insulation|with external insulation|"
                               r"insulated \(assumed\)|as built, insulated|thermal transmittance", na=False)
    partial = t.str.contains("partial insulation", na=False)
    none = t.str.contains(r"no insulation|uninsulated", na=False)
    out[insulated] = "insulated"
    out[partial] = "partial"
    out[none] = "uninsulated"
    out[t.isna()] = np.nan
    return out


def group_roof_type(s: pd.Series) -> pd.Series:
    t = _lower(s)
    out = pd.Series("other", index=s.index, dtype="object")
    rules = [
        (t.str.contains("pitched", na=False), "pitched"),
        (t.str.contains("flat", na=False), "flat"),
        (t.str.contains("roof room", na=False), "roof_room"),
        (t.str.contains("thatched", na=False), "thatched"),
        (t.str.contains("another dwelling above|other premises above", na=False), "dwelling_above"),
    ]
    for mask, label in rules:
        out[mask] = label
    out[t.isna()] = np.nan
    return out


def parse_loft_insulation_mm(s: pd.Series) -> pd.Series:
    """'Pitched, 270 mm loft insulation' -> 270; 'Pitched, no insulation' -> 0; otherwise NaN."""
    t = _lower(s)
    mm = t.str.extract(r"(\d{2,3})\+?\s*mm", expand=False).astype(float)
    mm[t.str.contains("no insulation", na=False)] = 0.0
    return mm


def group_heating(s: pd.Series) -> pd.Series:
    t = _lower(s)
    out = pd.Series("other", index=s.index, dtype="object")
    rules = [
        (t.str.contains("room heaters", na=False), "room_heaters"),
        (t.str.contains("warm air", na=False), "warm_air"),
        (t.str.contains("storage heaters", na=False), "storage_heaters"),
        (t.str.contains("boiler", na=False), "boiler"),
        (t.str.contains("heat pump", na=False), "heat_pump"),
        (t.str.contains("community scheme|heat network", na=False), "communal"),
    ]
    for mask, label in rules:
        out[mask] = label
    out[t.isna()] = np.nan
    return out


MISSING_LABELS = {"not recorded", "unknown", "no data", "n/a"}


def clean_category(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.strip()
    t = t.where((t.str.len() > 0) & ~t.str.lower().isin(MISSING_LABELS))
    return t.str.capitalize().astype("object")


# ---------------------------------------------------------------------------
# Record-level steps
# ---------------------------------------------------------------------------
def keep_latest_certificate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per home: keep the most recent certificate.

    Homes are identified by the anonymous home_id of a --history extract, else by
    UPRN where available, otherwise by first address line plus postcode. Records
    with none of these are kept as they are.
    """
    df = df.copy()
    sort_col = "lodgement_datetime" if "lodgement_datetime" in df else "lodgement_date"
    df["_ts"] = to_dates(df[sort_col])

    key = pd.Series(pd.NA, index=df.index, dtype="string")
    if "home_id" in df:
        key = df["home_id"].astype("string")
    elif "uprn" in df:
        key = df["uprn"].astype("string").str.strip().replace("", pd.NA)
    if {"address1", "postcode"} <= set(df.columns):
        fallback = (df["address1"].astype("string").str.upper().str.strip() + "|"
                    + df["postcode"].astype("string").str.upper().str.replace(" ", "", regex=False))
        key = key.fillna(fallback)
    df["_key"] = key

    with_key = df[df["_key"].notna()].sort_values("_ts").drop_duplicates("_key", keep="last")
    without_key = df[df["_key"].isna()]
    out = pd.concat([with_key, without_key]).drop(columns=["_ts", "_key"])
    return out.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the modelling table: target, features, and grouping column."""
    rating = df["current_energy_rating"].astype("string").str.strip().str.upper()
    df = df.loc[rating.isin(config.VALID_RATINGS)].copy()
    rating = rating.loc[df.index]

    out = pd.DataFrame(index=df.index)
    out["current_energy_rating"] = rating
    out["below_c"] = rating.isin(config.BELOW_C_RATINGS).astype(int)

    out["property_type"] = clean_category(df["property_type"])
    out["built_form"] = clean_category(df["built_form"])
    out["construction_year"] = parse_construction_year(df["construction_age_band"])
    area = pd.to_numeric(df["total_floor_area"], errors="coerce")
    out["total_floor_area"] = area.where(area.between(10, 500))
    out["fuel_group"] = group_fuel(df["main_fuel"])

    if "mains_gas_flag" in df:
        flag = df["mains_gas_flag"].astype("string").str.strip().str.upper()
        out["mains_gas"] = flag.map({"Y": 1.0, "N": 0.0}).astype(float)
    else:
        out["mains_gas"] = np.nan

    optional = {
        "walls_description": [("wall_type", group_wall_type), ("wall_insulation", group_wall_insulation)],
        "roof_description": [("roof_type", group_roof_type), ("loft_insulation_mm", parse_loft_insulation_mm)],
        "mainheat_description": [("heating_system", group_heating)],
    }
    for source, targets in optional.items():
        for name, fn in targets:
            out[name] = fn(df[source]) if source in df else np.nan

    if "multi_glaze_proportion" in df:
        glaze = pd.to_numeric(df["multi_glaze_proportion"], errors="coerce")
        out["glazing_pct"] = glaze.where(glaze.between(0, 100))
    else:
        out["glazing_pct"] = np.nan

    out["local_authority"] = df["local_authority"].astype("string")
    if "local_authority_label" in df:
        out["local_authority_label"] = df["local_authority_label"].astype("string")
    lodged = to_dates(df["lodgement_date"])
    out["lodgement_date"] = lodged.dt.strftime("%Y-%m-%d")
    out["lodgement_year"] = lodged.dt.year
    out["rdsap10"] = (lodged >= config.RDSAP10_START).astype(float).where(lodged.notna())

    leaked = set(config.LEAKAGE_COLUMNS) & set(out.columns)
    assert not leaked, f"Leakage columns slipped into the feature table: {leaked}"
    return out.reset_index(drop=True)


def build_features_chunked(df: pd.DataFrame, chunk_rows: int = 500_000) -> pd.DataFrame:
    """build_features a slice at a time, storing text as categories. Same result, far less memory."""
    if len(df) <= chunk_rows:
        return build_features(df)
    parts = [compact(build_features(df.iloc[i:i + chunk_rows])) for i in range(0, len(df), chunk_rows)]
    return concat_compact(parts)


def load_region_lookup(path=None) -> dict[str, str]:
    """Local authority code -> region name, from the ONS lookup if it is present."""
    path = Path(path or config.REGION_LOOKUP)
    if not path.exists():
        return {}
    lu = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    code = next(c for c in lu.columns if re.fullmatch(r"LAD\d{2}CD", c, re.I))
    name = next(c for c in lu.columns if re.fullmatch(r"RGN\d{2}NM", c, re.I))
    return dict(zip(lu[code].str.strip(), lu[name].str.strip()))


# The EPC register and ONS lookups use different codes for these two areas
LA_CODE_ALIASES = {"E08000038": "E08000016", "E08000039": "E08000019"}  # Barnsley, Sheffield


def add_region(df: pd.DataFrame, lookup: dict[str, str] | None = None) -> pd.DataFrame:
    """Add a `region` column from the local authority code.

    Welsh authorities become 'Wales'; anything else not in the lookup becomes 'Unknown'.
    """
    lookup = load_region_lookup() if lookup is None else lookup
    if not lookup:
        return df
    codes = df["local_authority"].astype("string").str.strip().replace(LA_CODE_ALIASES)
    region = codes.map(lookup)
    region = region.where(region.notna() | ~codes.str.startswith("W", na=False), "Wales")
    df["region"] = region.fillna("Unknown").astype("category")
    return df


def regions_add_detail(df: pd.DataFrame) -> bool:
    """True when regions group several local authorities, so regional results say something new.

    With only a few local authorities (each its own region) a regional table would just
    repeat the local authority one, so the five development areas are reported by area.
    """
    if "region" not in df:
        return False
    regions = df["region"].astype("string").dropna()
    regions = regions[~regions.isin(["Unknown", "Wales"])].nunique()
    return regions >= 3 and df["local_authority"].nunique() > regions


def england_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep English local authorities (codes starting E). The 2030 standard is for England."""
    keep = df["local_authority"].astype("string").str.startswith("E", na=False).to_numpy()
    return df.loc[keep].reset_index(drop=True)


def age_band_label(year: pd.Series) -> pd.Series:
    """Readable build-era bands for reporting."""
    # Edges follow the EPC age bands (before 1900 and 1900-1929, 1930-1949, 1950-1966,
    # 1967-1975 and 1976-1982, ...), so each era is made of whole bands.
    bins = [0, 1930, 1950, 1967, 1983, 2003, 3000]
    labels = ["pre-1930", "1930-49", "1950-66", "1967-82", "1983-2002", "2003+"]
    return pd.cut(year, bins=bins, labels=labels, right=False).astype("object")

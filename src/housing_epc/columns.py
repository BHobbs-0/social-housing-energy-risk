"""Column-name handling.

The EPC data service moved to a new GOV.UK platform in 2026 and column naming may
differ between downloads (UPPER_CASE, lower-case, hyphenated API names). Every
column is normalised to lower snake_case, then known alternative names are mapped
to the canonical names used in this project.
"""

from __future__ import annotations

import re

# alternative name -> canonical name. Add entries here if a download uses a new name.
ALIASES: dict[str, str] = {
    "lmk": "lmk_key",
    "certificate_number": "lmk_key",
    "rating": "current_energy_rating",
    "energy_rating_current": "current_energy_rating",
    "local_authority_code": "local_authority",
    "la_code": "local_authority",
    "registration_date": "lodgement_date",
    "age_band": "construction_age_band",
    "floor_area": "total_floor_area",
    "tenure_type": "tenure",
}

# Columns the pipeline reads. The rest of the (wide) file is skipped to save memory.
REQUIRED = [
    "current_energy_rating",
    "tenure",
    "property_type",
    "built_form",
    "construction_age_band",
    "total_floor_area",
    "main_fuel",
    "lodgement_date",
    "local_authority",
]
OPTIONAL = [
    "home_id",  # anonymous id written by scripts/extract_subset.py --history
    "lmk_key",
    "uprn",
    "postcode",
    "address1",
    "local_authority_label",
    "mains_gas_flag",
    "walls_description",
    "roof_description",
    "mainheat_description",
    "multi_glaze_proportion",
    "lodgement_datetime",
]


def normalise_name(name: str) -> str:
    """'CURRENT-ENERGY-RATING ' -> 'current_energy_rating'."""
    name = name.strip().lower()
    name = re.sub(r"[^0-9a-z]+", "_", name)
    return name.strip("_")


def build_rename_map(columns: list[str]) -> dict[str, str]:
    """Map original column names to canonical names."""
    mapping = {}
    for col in columns:
        norm = normalise_name(col)
        mapping[col] = ALIASES.get(norm, norm)
    return mapping


def check_required(canonical_columns: list[str], source: str = "file") -> None:
    """Raise a helpful error if the file is missing columns the pipeline needs."""
    missing = [c for c in REQUIRED if c not in canonical_columns]
    if missing:
        raise KeyError(
            f"{source} is missing required columns {missing}.\n"
            f"Columns found (normalised): {sorted(canonical_columns)}\n"
            "If a column has simply been renamed, add it to ALIASES in "
            "src/housing_epc/columns.py."
        )

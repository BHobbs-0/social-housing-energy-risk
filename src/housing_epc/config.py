"""Project-wide settings: which records to keep, which features to use, and what to exclude."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Record selection
# ---------------------------------------------------------------------------
# Tenure values have varied over the years ("rental (social)", "Rented (social)").
# Matching on the word "social" catches all of them.
TENURE_SOCIAL_PATTERN = "social"

# Older certificates used earlier versions of the assessment method, and many
# homes have been improved since. Keep only reasonably recent certificates.
MIN_LODGEMENT_YEAR = 2015

# The assessment method for existing homes changed to RdSAP 10 on 15 June 2025,
# which shifted ratings for otherwise identical homes.
RDSAP10_START = "2025-06-15"

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
# Social rented homes must reach EPC band C by April 2030, so the target is
# "currently below C" (bands D to G).
BELOW_C_RATINGS = {"D", "E", "F", "G"}
VALID_RATINGS = {"A", "B", "C", "D", "E", "F", "G"}

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
# CONTROLS: when the certificate was lodged. Ratings drift over time (homes get improved,
# the method changes), so both feature sets include these to stop the stock features
# absorbing measurement effects. To score a home "as if assessed today", set them to now.
CONTROL_NUMERIC = ["lodgement_year", "rdsap10"]

# CORE: what a landlord's asset register usually holds without a survey.
CORE_CATEGORICAL = ["property_type", "built_form", "fuel_group"]
CORE_NUMERIC = ["construction_year", "total_floor_area", "mains_gas"] + CONTROL_NUMERIC

# EXTENDED: fabric and heating knowledge that typically needs a stock survey.
EXTENDED_CATEGORICAL = CORE_CATEGORICAL + ["wall_type", "wall_insulation", "roof_type", "heating_system"]
EXTENDED_NUMERIC = CORE_NUMERIC + ["loft_insulation_mm", "glazing_pct"]

FEATURE_SETS = {
    "core": {"categorical": CORE_CATEGORICAL, "numeric": CORE_NUMERIC},
    "extended": {"categorical": EXTENDED_CATEGORICAL, "numeric": EXTENDED_NUMERIC},
}

GROUP_COLUMN = "local_authority"  # used for leave-areas-out validation, never as a feature

# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------
# These columns are outputs of (or near-duplicates of) the SAP calculation that
# produces the rating. Using them would let the model reverse-engineer the
# formula instead of learning anything useful. They are never used as features.
LEAKAGE_COLUMNS = [
    "current_energy_efficiency", "potential_energy_efficiency", "potential_energy_rating",
    "environment_impact_current", "environment_impact_potential",
    "energy_consumption_current", "energy_consumption_potential",
    "co2_emissions_current", "co2_emiss_curr_per_floor_area", "co2_emissions_potential",
    "lighting_cost_current", "lighting_cost_potential",
    "heating_cost_current", "heating_cost_potential",
    "hot_water_cost_current", "hot_water_cost_potential",
    "hot_water_energy_eff", "floor_energy_eff", "windows_energy_eff", "walls_energy_eff",
    "sheating_energy_eff", "roof_energy_eff", "mainheat_energy_eff", "mainheatc_energy_eff",
    "lighting_energy_eff",
]

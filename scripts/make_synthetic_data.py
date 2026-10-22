"""Generate a SYNTHETIC EPC-style file so the pipeline can be run without real data.

The numbers this produces are invented. They exist only to test that the code
runs end to end. Never publish results from synthetic data as findings.

    python scripts/make_synthetic_data.py --rows 30000
    python -m housing_epc.pipeline --raw data/synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

AGE_BANDS = [
    ("England and Wales: before 1900", 1890), ("England and Wales: 1900-1929", 1915),
    ("England and Wales: 1930-1949", 1940), ("England and Wales: 1950-1966", 1958),
    ("England and Wales: 1967-1975", 1971), ("England and Wales: 1976-1982", 1979),
    ("England and Wales: 1983-1990", 1987), ("England and Wales: 1991-1995", 1993),
    ("England and Wales: 1996-2002", 1999), ("England and Wales: 2003-2006", 2005),
    ("England and Wales: 2007-2011", 2009), ("England and Wales: 2012 onwards", 2015),
]
SAP_BANDS = [(92, "A"), (81, "B"), (69, "C"), (55, "D"), (39, "E"), (21, "F"), (0, "G")]


def sap_to_band(sap: np.ndarray) -> list[str]:
    return [next(b for t, b in SAP_BANDS if s >= t) for s in sap]


def make(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    las = [(f"E0600{i:04d}", f"Synthetic Borough {i}") for i in range(1, 13)]
    la_effect = rng.normal(0, 3, len(las))
    la_idx = rng.integers(0, len(las), rows)

    age_idx = rng.choice(len(AGE_BANDS), rows, p=np.array([4, 7, 12, 14, 14, 9, 8, 6, 8, 6, 6, 6]) / 100)
    year = np.array([AGE_BANDS[i][1] for i in age_idx])
    ptype = rng.choice(["Flat", "House", "Maisonette", "Bungalow"], rows, p=[0.45, 0.4, 0.07, 0.08])
    form = np.where(ptype == "Flat", "Mid-Terrace",
                    rng.choice(["Mid-Terrace", "End-Terrace", "Semi-Detached", "Detached"], rows,
                               p=[0.35, 0.25, 0.32, 0.08]))
    area = np.clip(rng.normal(np.where(ptype == "Flat", 55, 80), 15), 25, 200).round(1)

    fuel = rng.choice(["mains gas (not community)", "electricity (not community)", "mains gas (community)",
                       "oil (not community)", "LPG (not community)"], rows, p=[0.72, 0.15, 0.08, 0.03, 0.02])
    heating = np.select(
        [np.char.startswith(fuel.astype(str), "electricity"), np.char.find(fuel.astype(str), "(community)") > 0],
        [rng.choice(["Electric storage heaters", "Air source heat pump, radiators, electric",
                     "Room heaters, electric"], rows, p=[0.7, 0.15, 0.15]),
         np.full(rows, "Community scheme")],
        default="Boiler and radiators, mains gas",
    )
    solid = year < 1930
    wall_type = np.where(solid, "Solid brick", np.where(rng.random(rows) < 0.08, "System built", "Cavity wall"))
    insulated = rng.random(rows) < np.where(solid, 0.25, np.where(year > 1995, 0.97, 0.75))
    walls = np.where(insulated,
                     np.where(wall_type == "Cavity wall", wall_type + ", filled cavity",
                              wall_type + ", with external insulation"),
                     wall_type + ", as built, no insulation (assumed)")
    loft = rng.choice([0, 100, 150, 200, 270, 300], rows, p=[0.05, 0.1, 0.15, 0.2, 0.35, 0.15])
    dwelling_above = (ptype == "Flat") & (rng.random(rows) < 0.6)
    roof = np.where(dwelling_above, "(another dwelling above)",
                    np.where(loft == 0, "Pitched, no insulation", [f"Pitched, {m} mm loft insulation" for m in loft]))
    glazing = np.clip(rng.normal(92, 15, rows), 0, 100).round()

    sap = (
        72 - 0.9 * np.clip(2000 - year, 0, 120) / 10 + 4 * (year > 2002)
        - 9 * (~insulated) - 6 * solid
        + np.where(ptype == "Flat", 5, 0) - np.where(form == "Detached", 4, 0)
        - 12 * (heating == "Electric storage heaters") - 15 * (heating == "Room heaters, electric")
        + 6 * (heating == "Air source heat pump, radiators, electric")
        - 8 * np.char.startswith(fuel.astype(str), "oil") - 9 * np.char.startswith(fuel.astype(str), "LPG")
        - 6 * ((loft < 150) & ~dwelling_above) + 0.05 * (glazing - 90) - 0.03 * (area - 60)
        + la_effect[la_idx] + rng.normal(0, 4, rows)
    )
    sap = np.clip(sap, 1, 100).round()

    dates = pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 365 * 11, rows), unit="D")
    df = pd.DataFrame({
        "LMK_KEY": [f"SYN{i:09d}" for i in range(rows)],
        "ADDRESS1": [f"{rng.integers(1, 200)} Example Street" for _ in range(rows)],
        "POSTCODE": [f"ZZ{rng.integers(1, 99)} {rng.integers(1, 9)}AA" for _ in range(rows)],
        "UPRN": rng.integers(10**9, 10**9 + rows * 3, rows).astype(str),
        "CURRENT_ENERGY_RATING": sap_to_band(sap),
        "CURRENT_ENERGY_EFFICIENCY": sap.astype(int),  # leakage column: must be ignored by the pipeline
        "PROPERTY_TYPE": ptype,
        "BUILT_FORM": form,
        "CONSTRUCTION_AGE_BAND": [AGE_BANDS[i][0] for i in age_idx],
        "TOTAL_FLOOR_AREA": area,
        "MAIN_FUEL": fuel,
        "MAINS_GAS_FLAG": np.where(np.char.find(fuel.astype(str), "mains gas") >= 0, "Y", "N"),
        "WALLS_DESCRIPTION": walls,
        "ROOF_DESCRIPTION": roof,
        "MAINHEAT_DESCRIPTION": heating,
        "MULTI_GLAZE_PROPORTION": glazing,
        "LODGEMENT_DATE": dates.strftime("%Y-%m-%d"),
        "LOCAL_AUTHORITY": [las[i][0] for i in la_idx],
        "LOCAL_AUTHORITY_LABEL": [las[i][1] for i in la_idx],
        "TENURE": rng.choice(["rental (social)", "Rented (social)", "owner-occupied", "rental (private)"],
                             rows, p=[0.55, 0.25, 0.12, 0.08]),
    })
    # A few deliberately messy values, as in the real data
    df.loc[df.sample(frac=0.01, random_state=1).index, "CONSTRUCTION_AGE_BAND"] = "NO DATA!"
    df.loc[df.sample(frac=0.005, random_state=2).index, "TOTAL_FLOOR_AREA"] = 0
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=30_000)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = make(args.rows)
    path = args.out / "certificates.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} SYNTHETIC rows to {path}")

"""Shrink large EPC downloads to a small, address-free file of social rented homes.

Reads one or more CSV files, zip files or folders (any mix), keeps only:
  * social rented certificates lodged since 2015
  * the chosen local authorities (default: eight areas, see DEFAULT_AREAS)
  * the latest certificate per home
  * the ~20 columns the model uses (addresses and identifiers are dropped)
and writes a compressed CSV, typically a few MB.

With --history it keeps EVERY social rented certificate for each home instead of
just the latest, plus the SAP score and the reason the certificate was lodged.
Homes are linked by an anonymous home_id: a salted hash of the UPRN (or address
and postcode) whose salt is random for each run and never saved, so the id cannot
be traced back to an address. This file feeds the 2030 forecast.

Needs only pandas:  pip install pandas

Examples
    python scripts/extract_subset.py "C:/Users/me/Downloads/epc_download.csv"
    python scripts/extract_subset.py Downloads/epc.zip Downloads/more_epcs/
    python scripts/extract_subset.py big.csv --areas E08000025 E09000028
    python scripts/extract_subset.py big.csv --all-areas
    python scripts/extract_subset.py Downloads/epc.zip --history     # every certificate, for forecasting
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import secrets
import sys
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_AREAS = {
    "E08000025": "Birmingham", "E08000035": "Leeds", "E08000003": "Manchester",
    "E08000019": "Sheffield", "E08000021": "Newcastle upon Tyne", "E09000028": "Southwark",
    "E09000030": "Tower Hamlets", "E06000052": "Cornwall",
}
MIN_YEAR = 2015
HISTORY_MIN_YEAR = 2008  # earlier certificates add more before/after pairs for the forecast

ALIASES = {
    "lmk": "lmk_key", "certificate_number": "lmk_key", "rating": "current_energy_rating",
    "energy_rating_current": "current_energy_rating", "local_authority_code": "local_authority",
    "la_code": "local_authority", "registration_date": "lodgement_date",
    "age_band": "construction_age_band", "floor_area": "total_floor_area", "tenure_type": "tenure",
}
KEEP = [
    "current_energy_rating", "tenure", "property_type", "built_form", "construction_age_band",
    "total_floor_area", "main_fuel", "mains_gas_flag", "walls_description", "roof_description",
    "mainheat_description", "multi_glaze_proportion", "lodgement_date", "lodgement_datetime",
    "local_authority", "local_authority_label",
]
HISTORY_EXTRA = [  # kept only with --history; the SAP score is never used as a model feature
    "current_energy_efficiency", "potential_energy_efficiency", "potential_energy_rating",
    "transaction_type", "inspection_date",
]
DEDUPE_ONLY = ["uprn", "address1", "postcode"]  # used to find duplicates, then dropped
NA = ["", "NO DATA!", "INVALID!", "N/A", "NODATA!", "nan"]


def norm(name: str) -> str:
    n = re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")
    return ALIASES.get(n, n)


def sources(paths: list[str]):
    """Yield (label, opener) for every CSV inside the given files, zips and folders."""
    for p in map(Path, paths):
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() == ".csv":
                    yield str(f), (lambda f=f: open(f, "rb"))
                elif f.suffix.lower() == ".zip":
                    yield from zip_sources(f)
        elif p.suffix.lower() == ".zip":
            yield from zip_sources(p)
        elif p.suffix.lower() == ".csv":
            yield str(p), (lambda p=p: open(p, "rb"))
        else:
            print(f"  skipping {p} (not a CSV, zip or folder)")


def zip_sources(path: Path):
    zf = zipfile.ZipFile(path)
    for member in zf.namelist():
        if member.lower().endswith(".csv"):
            yield f"{path.name}:{member}", (lambda m=member: zf.open(m))


def read_filtered(label, opener, codes, names, all_areas, chunksize, min_year=MIN_YEAR, keep=KEEP):
    with opener() as fh:
        header = pd.read_csv(fh, nrows=0).columns.tolist()
    rename = {c: norm(c) for c in header}
    canon = set(rename.values())
    if "current_energy_rating" not in canon:
        return None  # recommendations file or something else
    missing = [c for c in ("tenure", "lodgement_date", "local_authority") if c not in canon]
    if missing:
        print(f"  {label}: missing {missing}; columns found: {sorted(canon)}")
        return None
    usecols = [c for c, n in rename.items() if n in set(keep + DEDUPE_ONLY)]

    kept, rows_read, areas_seen = [], 0, {}
    with opener() as fh:
        reader = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), usecols=usecols,
                             dtype=str, chunksize=chunksize, keep_default_na=False, na_values=NA)
        for chunk in reader:
            rows_read += len(chunk)
            chunk = chunk.rename(columns=rename)
            for code, n in chunk["local_authority"].value_counts().head(50).items():
                areas_seen[code] = areas_seen.get(code, 0) + n
            mask = chunk["tenure"].str.lower().str.contains("social", na=False)
            mask &= pd.to_datetime(chunk["lodgement_date"], errors="coerce").dt.year >= min_year
            if not all_areas:
                area = chunk["local_authority"].str.upper().isin(codes)
                if "local_authority_label" in chunk:
                    area |= chunk["local_authority_label"].str.lower().isin(names)
                mask &= area
            if mask.any():
                kept.append(chunk.loc[mask])
            print(f"\r  {label}: {rows_read:,} rows read", end="", flush=True)
    print()
    if not kept:
        top = sorted(areas_seen.items(), key=lambda kv: -kv[1])[:10]
        print(f"  no matching rows. Most common local authority values in this file: {top}")
        return None
    return pd.concat(kept, ignore_index=True)


def latest_per_home(df: pd.DataFrame) -> pd.DataFrame:
    sort_col = "lodgement_datetime" if "lodgement_datetime" in df else "lodgement_date"
    df = df.assign(_ts=pd.to_datetime(df[sort_col], errors="coerce")).sort_values("_ts")
    key = df["uprn"].where(df["uprn"].notna()) if "uprn" in df else pd.Series(pd.NA, index=df.index)
    if {"address1", "postcode"} <= set(df.columns):
        alt = df["address1"].str.upper().str.strip() + "|" + df["postcode"].str.upper().str.replace(" ", "")
        key = key.fillna(alt)
    with_key = df[key.notna()].assign(_k=key[key.notna()]).drop_duplicates("_k", keep="last")
    out = pd.concat([with_key, df[key.isna()]])
    return out.drop(columns=[c for c in ["_ts", "_k", *DEDUPE_ONLY] if c in out.columns])


def home_key(df: pd.DataFrame) -> pd.Series:
    """UPRN where present, otherwise first address line plus postcode."""
    key = (df["uprn"].astype("string").str.strip().replace("", pd.NA) if "uprn" in df
           else pd.Series(pd.NA, index=df.index, dtype="string"))
    if {"address1", "postcode"} <= set(df.columns):
        alt = (df["address1"].astype("string").str.upper().str.strip() + "|"
               + df["postcode"].astype("string").str.upper().str.replace(" ", "", regex=False))
        key = key.fillna(alt)
    return key


def all_certificates(df: pd.DataFrame) -> pd.DataFrame:
    """Every certificate, linked by an anonymous home_id. Homes with no UPRN or address are dropped."""
    salt = secrets.token_bytes(16)  # random per run and never written anywhere
    key = home_key(df)
    df = df.loc[key.notna()].copy()
    df["home_id"] = [hashlib.sha256(salt + k.encode()).hexdigest()[:16] for k in key.loc[df.index]]
    sort_col = "lodgement_datetime" if "lodgement_datetime" in df else "lodgement_date"
    df["_ts"] = pd.to_datetime(df[sort_col], errors="coerce")
    # the same certificate can appear in more than one download
    df = df.sort_values(["home_id", "_ts"]).drop_duplicates(["home_id", "_ts", "current_energy_rating"])
    df = df.drop(columns=["_ts", *[c for c in DEDUPE_ONLY if c in df.columns]])
    return df[["home_id", *[c for c in df.columns if c != "home_id"]]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="CSV files, zip files or folders")
    ap.add_argument("--areas", nargs="+", help="local authority codes or names (default: eight areas)")
    ap.add_argument("--all-areas", action="store_true", help="keep every local authority in the input")
    ap.add_argument("--history", action="store_true",
                    help="keep every certificate per home (anonymous home_id), for the 2030 forecast")
    ap.add_argument("--min-year", type=int, default=None,
                    help=f"earliest lodgement year (default {MIN_YEAR}, or {HISTORY_MIN_YEAR} with --history)")
    ap.add_argument("--out", default=None, help="output file (default epc_social_subset.csv.gz, "
                    "or epc_social_history.csv.gz with --history)")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    wanted = args.areas or list(DEFAULT_AREAS) + list(DEFAULT_AREAS.values())
    codes = {a.upper() for a in wanted if re.fullmatch(r"[EW]\d{8}", a.upper())}
    names = {a.lower() for a in wanted if not re.fullmatch(r"[EW]\d{8}", a.upper())}
    if args.areas is None:
        codes |= set(DEFAULT_AREAS)

    min_year = args.min_year or (HISTORY_MIN_YEAR if args.history else MIN_YEAR)
    keep = KEEP + HISTORY_EXTRA if args.history else KEEP
    out = args.out or ("epc_social_history.csv.gz" if args.history else "epc_social_subset.csv.gz")

    frames = []
    for label, opener in sources(args.inputs):
        df = read_filtered(label, opener, codes, names, args.all_areas, args.chunksize, min_year, keep)
        if df is not None:
            frames.append(df)
    if not frames:
        sys.exit("Nothing matched. Check the area codes printed above and rerun with --areas.")

    combined = pd.concat(frames, ignore_index=True)
    data = all_certificates(combined) if args.history else latest_per_home(combined)
    data.to_csv(out, index=False, compression="gzip")
    size_mb = Path(out).stat().st_size / 1e6
    if args.history:
        counts = data["home_id"].value_counts()
        print(f"\nWrote {len(data):,} certificates for {len(counts):,} social rented homes to {out} "
              f"({size_mb:.1f} MB); {(counts > 1).sum():,} homes have more than one certificate")
    else:
        print(f"\nWrote {len(data):,} social rented homes to {out} ({size_mb:.1f} MB)")
    label_col = "local_authority_label" if "local_authority_label" in data else "local_authority"
    print(data[label_col].value_counts().to_string())


if __name__ == "__main__":
    main()

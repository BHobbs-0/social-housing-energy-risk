"""Load EPC certificate CSVs, keeping only social rented homes.

Works with either the single national bulk file (several GB) or a folder of
per-local-authority downloads. Files are read in chunks and filtered as they
stream in, so memory use stays modest even for the full national file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import union_categoricals

from . import columns, config


def find_certificate_files(raw_dir: Path) -> list[Path]:
    """Find every CSV under raw_dir that looks like a domestic certificates file.

    Recommendations files (which share the download zip) are skipped because they
    do not contain a current energy rating column.
    """
    files = []
    candidates = list(Path(raw_dir).rglob("*.csv")) + list(Path(raw_dir).rglob("*.csv.gz"))
    for path in sorted(candidates):
        header = pd.read_csv(path, nrows=0).columns.tolist()
        canonical = set(columns.build_rename_map(header).values())
        if "current_energy_rating" in canonical:
            files.append(path)
    return files


def _read_one(path: Path, chunksize: int, min_year: int) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    rename = columns.build_rename_map(header)
    canonical = list(rename.values())
    columns.check_required(canonical, source=str(path))

    wanted = set(columns.REQUIRED) | set(columns.OPTIONAL)
    usecols = [orig for orig, canon in rename.items() if canon in wanted]

    kept = []
    reader = pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,  # parse types ourselves; EPC files mix types within columns
        chunksize=chunksize,
        keep_default_na=False,
        na_values=["", "NO DATA!", "INVALID!", "N/A", "NODATA!", "nan"],
    )
    for chunk in reader:
        chunk = chunk.rename(columns=rename)
        is_social = chunk["tenure"].str.lower().str.contains(config.TENURE_SOCIAL_PATTERN, na=False)
        chunk = chunk.loc[is_social]
        dates = pd.to_datetime(chunk["lodgement_date"], errors="coerce")
        chunk = chunk.loc[dates.dt.year >= min_year]
        if len(chunk):
            kept.append(compact(chunk))

    if not kept:
        return pd.DataFrame(columns=[c for c in canonical if c in wanted])
    return concat_compact(kept)


# Identifiers stay as text; every other column repeats a small set of values, so it is
# stored as a category, which needs about an eighth of the memory at national scale.
TEXT_ID_COLUMNS = {"home_id", "lmk_key", "uprn", "address1", "postcode"}


def compact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        dtype = df[c].dtype
        is_text = dtype == object or (pd.api.types.is_string_dtype(dtype) and not isinstance(dtype, pd.CategoricalDtype))
        if c not in TEXT_ID_COLUMNS and is_text:
            df[c] = df[c].astype("category")
    return df


def concat_compact(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames from `compact`, merging category sets so columns stay categorical."""
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    cols = list(dict.fromkeys(c for f in frames for c in f.columns))
    out = {}
    for c in cols:
        parts = [f[c] if c in f else pd.Series(pd.NA, index=f.index, dtype="object") for f in frames]
        if all(isinstance(p.dtype, pd.CategoricalDtype) for p in parts):
            out[c] = union_categoricals(parts, ignore_order=True)
        else:
            out[c] = pd.concat([p.astype(object) if isinstance(p.dtype, pd.CategoricalDtype) else p
                                for p in parts], ignore_index=True)
    n = sum(len(f) for f in frames)
    return pd.DataFrame({c: (pd.Series(v) if not isinstance(v, pd.Series) else v.reset_index(drop=True))
                         for c, v in out.items()}, index=pd.RangeIndex(n))


def load_social_certificates(
    raw_dir: Path = config.RAW_DIR,
    chunksize: int = 200_000,
    min_year: int = config.MIN_LODGEMENT_YEAR,
) -> pd.DataFrame:
    """Read all certificate files under raw_dir and return social rented records."""
    files = find_certificate_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No EPC certificate CSVs found under {raw_dir}. "
            "See the README section 'Getting the data'."
        )
    frames = []
    for path in files:
        df = _read_one(path, chunksize=chunksize, min_year=min_year)
        print(f"  {path.name}: {len(df):,} social rented certificates since {min_year}")
        frames.append(df)
    return concat_compact(frames)

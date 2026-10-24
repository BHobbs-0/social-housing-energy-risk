"""Estimate how many social homes are below band C now, and project to 1 April 2030.

The classifier in this project is a snapshot. To say anything about 2030 we need
to know how ratings change, and the direct evidence is homes that have been
assessed more than once. This module learns from consecutive certificates on the
same home.

What the repeat certificates show
---------------------------------
Three things, each of which shaped the method:

1. A single certificate is a noisy reading. Low scores tend to rise and high
   scores tend to fall at the next certificate whatever the gap: partly assessor
   variation, partly "before" certificates lodged ahead of retrofit work. So the
   model starts from a home's SAP SCORE, not its band: a D at 68 is a near miss,
   a D at 56 is not.
2. Time between certificates barely matters. The chance of crossing the C line is
   much the same after six months as after ten years.
3. WHEN the next certificate is lodged matters a lot. Homes just below C were
   about 60% likely to still be below C at their next certificate up to 2022, and
   about 30% from 2023. Something shifted across the stock: retrofit programmes,
   changes in assessment practice, or both.

An earlier version modelled steady improvement over time (a two-state Markov
chain). It failed its backtest for exactly these reasons, which is why the model
below looks the way it does.

Model
-----
For each pair of consecutive certificates on the same home, a logistic regression
predicts whether the LATER certificate is below C from:

* the earlier certificate's SAP score (a spline, so the effect can bend near the line),
* the gap between them (a spline, mostly flat in practice),
* whether the earlier certificate was lodged because of works (often a "before" survey),
* property type, build era, main fuel and local authority,
* the PERIOD the later certificate was lodged in: each year, with certificates
  under RdSAP 10 (from 15 June 2025) as a period of their own.

Pairs whose later certificate was itself triggered by works (ECO, grant schemes)
are left out of the fit, so the model describes a routine reassessment rather than
one commissioned to record works.

Projection
----------
Every home is scored from its LATEST certificate as if reassessed under current
conditions (the RdSAP 10 period, which is also the method 2030 certificates use).

* "Today": reassessed now.
* "2030, conditions as now": reassessed in April 2030 with the period effect held
  at its current level. Because the gap barely matters, this is close to today.
* "2030, recent trend continues": the period effect keeps falling at its
  2019-2025 pace until April 2030.

The data cannot say which scenario is more likely; they bracket the outcome.

Usage
-----
    python -m housing_epc.forecast --history path/to/epc_social_history.csv.gz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from . import columns, config, explain, features
from .load import compact, concat_compact

TARGET_DATE = pd.Timestamp("2030-04-01")
RDSAP10 = pd.Timestamp(config.RDSAP10_START)
RDSAP10_PERIOD = "RdSAP 10"
C_THRESHOLD = 69              # SAP 69 and above is band C
MIN_GAP_YEARS = 0.25          # shorter gaps are mostly corrections or re-lodgements
MAX_FLOOR_AREA_RATIO = 1.25   # bigger changes between certificates suggest a different dwelling
FIT_FROM = pd.Timestamp("2013-01-01")   # pairs whose later certificate is from here
TREND_FROM_YEAR = 2019                  # period effects from here set the "recent trend"
NEAR_LINE = (60, 68)          # "just below C" for the headline chart
BOOTSTRAP_DRAWS = 30

# Certificates lodged because works were done, rather than for a letting or survey
WORKS_PATTERN = r"eco|green deal|grant|rhi|renewable heat|feed.?in|fit application|warm homes|retrofit"

CATEGORICAL = ["property_type", "build_era", "fuel_group", "local_authority"]
NA_VALUES = ["", "NO DATA!", "INVALID!", "N/A", "NODATA!", "nan"]
HISTORY_COLUMNS = {  # what the forecast reads from a --history extract
    "home_id", "current_energy_rating", "current_energy_efficiency", "transaction_type", "lodgement_date",
    "property_type", "built_form", "construction_age_band", "total_floor_area", "main_fuel", "mains_gas_flag",
    "walls_description", "roof_description", "mainheat_description", "multi_glaze_proportion",
    "local_authority", "local_authority_label",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_history(path: Path) -> pd.DataFrame:
    """Every certificate per home, cleaned with the same feature code as the classifier."""
    path = Path(path)
    files = sorted(p for p in path.rglob("*.csv*")) if path.is_dir() else [path]
    parts = []
    for f in files:
        header = pd.read_csv(f, nrows=0).columns
        rename = columns.build_rename_map(list(header))
        if "home_id" not in rename.values():
            raise KeyError(f"{f} has no home_id column. Create it with scripts/extract_subset.py --history.")
        use = [c for c in header if rename[c] in HISTORY_COLUMNS]
        for chunk in pd.read_csv(f, dtype=str, keep_default_na=False, na_values=NA_VALUES, usecols=use,
                                 chunksize=500_000):
            parts.append(_certificates_from(chunk.rename(columns=rename)))
    certs = concat_compact(parts)
    certs = (certs.dropna(subset=["date", "sap"]).sort_values(["home_id", "date"])
             .drop_duplicates(["home_id", "date"], keep="last").reset_index(drop=True))
    return features.add_region(certs)


def _certificates_from(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean one chunk of a --history extract with the same feature code as the classifier."""
    rating = raw["current_energy_rating"].astype("string").str.strip().str.upper()
    raw = raw.loc[rating.isin(config.VALID_RATINGS).to_numpy()].reset_index(drop=True)
    certs = features.build_features(raw)  # row order preserved
    certs["home_id"] = raw["home_id"].to_numpy()
    certs["sap"] = pd.to_numeric(raw.get("current_energy_efficiency"), errors="coerce").to_numpy()
    trans = raw["transaction_type"] if "transaction_type" in raw else pd.Series(pd.NA, index=raw.index)
    certs["transaction_type"] = trans.astype("string").str.strip().str.lower().to_numpy()
    certs["works_triggered"] = certs["transaction_type"].str.contains(WORKS_PATTERN, regex=True, na=False)
    certs["date"] = pd.to_datetime(certs["lodgement_date"], errors="coerce")
    certs["build_era"] = features.age_band_label(certs["construction_year"])
    for c in CATEGORICAL:
        certs[c] = certs[c].fillna("unknown").astype(str)
    return compact(certs.drop(columns=["lodgement_date"]))


def latest_per_home(certs: pd.DataFrame) -> pd.DataFrame:
    return certs.sort_values("date").groupby("home_id", sort=False).tail(1).reset_index(drop=True)


def period_of(dates: pd.Series) -> pd.Series:
    """Calendar year of a certificate, with RdSAP 10 certificates as their own period."""
    out = dates.dt.year.astype("Int64").astype(str)
    return out.where(dates < RDSAP10, RDSAP10_PERIOD)


def make_pairs(certs: pd.DataFrame) -> pd.DataFrame:
    """Consecutive certificates on the same home: where it started, where it ended, and when."""
    c = certs.sort_values(["home_id", "date"]).reset_index(drop=True)
    nxt = c.groupby("home_id").shift(-1)
    keep = nxt["date"].notna().to_numpy()
    start, end = c.loc[keep].reset_index(drop=True), nxt.loc[keep].reset_index(drop=True)
    pairs = start[["home_id", "sap", *CATEGORICAL]].copy()
    pairs["start_below"] = start["below_c"].astype(int)
    pairs["start_works"] = start["works_triggered"].astype(int)
    pairs["start_date"], pairs["end_date"] = start["date"], end["date"]
    pairs["gap"] = (end["date"] - start["date"]).dt.days / 365.25
    pairs["period"] = period_of(end["date"])
    pairs["end_sap"] = end["sap"]
    pairs["end_below"] = end["below_c"].astype(int)
    pairs["end_works"] = end["works_triggered"].astype(bool)

    # Some ids cover more than one dwelling (a UPRN or address shared across a building).
    # A different property type or a big change in floor area means the pair is not the same home.
    a0, a1 = start["total_floor_area"].to_numpy(float), end["total_floor_area"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.maximum(a0, a1) / np.minimum(a0, a1)
    area_ok = np.isnan(ratio) | (ratio <= MAX_FLOOR_AREA_RATIO)
    same = (start["property_type"].to_numpy() == end["property_type"].to_numpy()) & area_ok
    return pairs[(pairs["gap"] >= MIN_GAP_YEARS) & same].reset_index(drop=True)


def routine(pairs: pd.DataFrame, start=FIT_FROM, end=None) -> pd.DataFrame:
    """Pairs whose later certificate was not triggered by works, lodged in [start, end)."""
    p = pairs[~pairs["end_works"] & (pairs["end_date"] >= start)]
    return p if end is None else p[p["end_date"] < end]


def period_midpoint(pairs: pd.DataFrame) -> pd.Series:
    """Average decimal date of the later certificates in each period."""
    d = pairs["end_date"]
    dec = d.dt.year + (d.dt.dayofyear - 0.5) / 365.25
    return dec.groupby(pairs["period"]).mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class NextCertificateModel:
    """Chance the next certificate on a home is below C, given its last one and when it is lodged."""

    NUMERIC = ["sap", "gap", "start_works"]

    def __init__(self, C: float = 1.0):
        self.C = C
        self.pipe: Pipeline | None = None

    def _pipeline(self) -> Pipeline:
        prep = ColumnTransformer([
            ("sap", SplineTransformer(n_knots=7, degree=3, extrapolation="constant"), ["sap"]),
            ("gap", SplineTransformer(n_knots=4, degree=3, extrapolation="constant"), ["gap"]),
            ("works", "passthrough", ["start_works"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL + ["period"]),
        ])
        return Pipeline([("prep", prep), ("logit", LogisticRegression(C=self.C, max_iter=5000))])

    def fit(self, pairs: pd.DataFrame, weights=None) -> "NextCertificateModel":
        self.pipe = self._pipeline()
        self.pipe.fit(pairs[self.NUMERIC + CATEGORICAL + ["period"]], pairs["end_below"],
                      logit__sample_weight=weights)
        return self

    def logit(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.decision_function(df[self.NUMERIC + CATEGORICAL + ["period"]])

    def predict(self, df: pd.DataFrame, shift: float = 0.0) -> np.ndarray:
        """Probability of being below C; `shift` moves every home's log-odds (negative = improvement)."""
        return 1 / (1 + np.exp(-(self.logit(df) + shift)))

    def period_effects(self) -> pd.Series:
        enc = self.pipe.named_steps["prep"].named_transformers_["cat"]
        names = enc.get_feature_names_out(CATEGORICAL + ["period"])
        coef = self.pipe.named_steps["logit"].coef_[0][-len(names):]
        eff = pd.Series(coef, index=names)
        eff = eff[eff.index.str.startswith("period_")]
        return eff.rename(lambda s: s.removeprefix("period_"))


def trend_per_year(effects: pd.Series, midpoints: pd.Series, weights: pd.Series) -> float:
    """Weighted straight-line slope of the yearly period effects from TREND_FROM_YEAR, before RdSAP 10."""
    years = [p for p in effects.index if p != RDSAP10_PERIOD and p.isdigit() and int(p) >= TREND_FROM_YEAR]
    if len(years) < 3:
        return 0.0
    x, y, w = midpoints[years].to_numpy(), effects[years].to_numpy(), weights[years].to_numpy(float)
    return float(np.polyfit(x, y, 1, w=np.sqrt(w))[0])


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def as_reassessed(homes: pd.DataFrame, when: pd.Timestamp, period: str = RDSAP10_PERIOD) -> pd.DataFrame:
    """Each home's latest certificate as the start of a pair whose next certificate is lodged at `when`."""
    df = homes[["sap", *CATEGORICAL]].copy()
    df["start_works"] = homes["works_triggered"].astype(int).to_numpy()
    df["gap"] = ((when - homes["date"]).dt.days / 365.25).clip(lower=MIN_GAP_YEARS).to_numpy()
    df["period"] = period
    return df


def scenarios(model: NextCertificateModel, homes: pd.DataFrame, today: pd.Timestamp, slope: float,
              rd_mid: float) -> dict[str, np.ndarray]:
    years_ahead = TARGET_DATE.year + (TARGET_DATE.dayofyear - 0.5) / 365.25 - rd_mid
    return {
        "today": model.predict(as_reassessed(homes, today)),
        "conditions_as_now": model.predict(as_reassessed(homes, TARGET_DATE)),
        "trend_continues": model.predict(as_reassessed(homes, TARGET_DATE), shift=slope * years_ahead),
    }


def fit_all(pairs: pd.DataFrame, weights=None):
    fit_pairs = routine(pairs)
    w = None if weights is None else weights.loc[fit_pairs.index].to_numpy()
    model = NextCertificateModel().fit(fit_pairs, w)
    eff = model.period_effects()
    mids = period_midpoint(fit_pairs)
    counts = fit_pairs["period"].value_counts()
    slope = trend_per_year(eff, mids, counts)
    return model, fit_pairs, eff, mids, counts, slope


def bootstrap(pairs: pd.DataFrame, homes: pd.DataFrame, today: pd.Timestamp, draws: int) -> dict:
    """Refit with Poisson weights per home (resampling homes, not pairs) for 90% intervals."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    ids, inverse = np.unique(pairs["home_id"].to_numpy(), return_inverse=True)
    totals = {k: [] for k in ("today", "conditions_as_now", "trend_continues")}
    for _ in range(draws):
        w = pd.Series(rng.poisson(1.0, len(ids))[inverse], index=pairs.index)
        model, fit_pairs, _, mids, _, slope = fit_all(pairs, w)
        for k, p in scenarios(model, homes, today, slope, mids[RDSAP10_PERIOD]).items():
            totals[k].append(p.sum())
    return {k: [float(np.percentile(v, 5)), float(np.percentile(v, 95))] for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def brier(y, p) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def temporal_backtest(pairs: pd.DataFrame, train_end: str, test_end: str, label: str) -> dict:
    """Fit on routine pairs before train_end, predict those in [train_end, test_end) with the last period's effect."""
    train = routine(pairs, end=pd.Timestamp(train_end))
    test = routine(pairs, start=pd.Timestamp(train_end), end=pd.Timestamp(test_end))
    if len(train) < 500 or len(test) < 200:
        return {}
    last = str(pd.Timestamp(train_end).year - 1)
    model = NextCertificateModel().fit(train)
    p = model.predict(test.assign(period=last))
    y = test["end_below"].to_numpy()
    started_below = test["start_below"].to_numpy() == 1
    return {
        "label": label, "train_pairs": len(train), "test_pairs": len(test), "period_assumed": last,
        "observed_below": float(y.mean()), "predicted_below": float(p.mean()),
        "observed_still_below_if_started_below": float(y[started_below].mean()),
        "predicted_still_below_if_started_below": float(p[started_below].mean()),
        "brier": brier(y, p), "brier_assume_no_change": brier(y, test["start_below"]),
    }


def recent_cross_validation(pairs: pd.DataFrame, since: str = "2023-01-01", folds: int = 5):
    """How well does the model rank and calibrate recent reassessments for homes it has not seen?"""
    fit_pairs = routine(pairs)
    recent = fit_pairs["end_date"] >= pd.Timestamp(since)
    out_p = pd.Series(np.nan, index=fit_pairs.index)
    for train_idx, test_idx in GroupKFold(folds).split(fit_pairs, groups=fit_pairs["home_id"]):
        tr, te = fit_pairs.iloc[train_idx], fit_pairs.iloc[test_idx]
        te = te[recent.iloc[test_idx].to_numpy()]
        if len(te):
            out_p.loc[te.index] = NextCertificateModel().fit(tr).predict(te)
    held = fit_pairs[recent]
    y, p = held["end_below"].to_numpy(), out_p[recent].to_numpy()
    sap_only = (held["sap"] < C_THRESHOLD).astype(float).to_numpy()
    return {"pairs": int(len(held)), "brier": brier(y, p), "brier_assume_no_change": brier(y, sap_only),
            "observed_below": float(y.mean()), "predicted_below": float(p.mean())}, (y, p)


def near_line_by_period(pairs: pd.DataFrame, min_pairs: int = 100) -> pd.DataFrame:
    """Observed share still below C at the next routine certificate, for homes starting just below C."""
    p = routine(pairs, start=pd.Timestamp("2010-01-01"))
    p = p[p["sap"].between(*NEAR_LINE)]
    mids = period_midpoint(p)
    out = (p.groupby("period").agg(pairs=("end_below", "size"), still_below=("end_below", "mean"),
                                   mean_gap=("gap", "mean"))
           .assign(midpoint=mids).reset_index().sort_values("midpoint"))
    return out[out["pairs"] >= min_pairs].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def breakdown(homes: pd.DataFrame, by: list[str], min_homes: int) -> pd.DataFrame:
    grp = (homes.groupby(by, dropna=False, observed=True)
           .agg(homes=("below_c", "size"), below_c_latest_certificate=("below_c", "sum"),
                expected_below_c_today=("p_today", "sum"),
                expected_below_c_2030_conditions_as_now=("p_flat", "sum"),
                expected_below_c_2030_trend_continues=("p_trend", "sum"))
           .reset_index())
    grp["share_today"] = grp["expected_below_c_today"] / grp["homes"]
    grp["share_2030_trend_continues"] = grp["expected_below_c_2030_trend_continues"] / grp["homes"]
    grp = grp[grp["homes"] >= min_homes].sort_values("expected_below_c_today", ascending=False)
    return grp.round({c: 0 for c in grp.columns if c.startswith("expected")} | {
        "share_today": 3, "share_2030_trend_continues": 3})


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_near_line(table: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    pre = table[table["period"] != RDSAP10_PERIOD]
    ax.plot(pre["midpoint"], pre["still_below"], color=explain.BLUE, linewidth=2, marker="o", markersize=5)
    rd = table[table["period"] == RDSAP10_PERIOD]
    if len(rd):
        ax.plot(rd["midpoint"], rd["still_below"], color=explain.ORANGE, marker="o", markersize=7, linestyle="none")
        ax.annotate("RdSAP 10", (rd["midpoint"].iloc[0], rd["still_below"].iloc[0]), xytext=(6, 8),
                    textcoords="offset points", color=explain.ORANGE, fontsize=9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    lo, hi = NEAR_LINE
    explain._style(ax, f"Homes rated just below C (SAP {lo}-{hi}): still below C at the next certificate",
                   xlabel="Year the next certificate was lodged", ylabel="Still below C")
    ax.grid(axis="y", color=explain.GRID, linewidth=0.8)
    explain._save(fig, path)


def plot_by_region(table: pd.DataFrame, path: Path):
    """Estimated share below C today (bars) and in 2030 if the recent trend continues (markers)."""
    t = table.sort_values("share_today")
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(t) + 1.4))
    y = np.arange(len(t))
    ax.barh(y, t["share_today"], color=explain.BLUE, height=0.62, label="Estimated today")
    ax.scatter(t["share_2030_trend_continues"], y, color=explain.ORANGE, zorder=3, s=36,
               label="2030, recent trend continues")
    for yi, v, e in zip(y, t["share_today"], t["share_2030_trend_continues"]):
        ax.text(max(v, e) + 0.008, yi, f"{v:.0%}", va="center", fontsize=9, color=explain.INK)
    ax.set_yticks(y, t["region"].astype(str))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, t["share_today"].max() * 1.25)
    explain._style(ax, "Social homes likely to be below C, by region", xlabel="Below C")
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=12, color=explain.INK, pad=30)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(lbl) for lbl in sorted(labels, key=lambda lbl: lbl.startswith("2030") or lbl.startswith("Expected"))]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=9, ncol=2,
              labelcolor=explain.INK, loc="lower left", bbox_to_anchor=(-0.01, 1.0), borderaxespad=0.2, handlelength=1.2)
    explain._save(fig, path)


def plot_scenarios(results: dict, path: Path):
    s = results["scenarios"]
    rows = [("Latest certificates", "latest_certificates", explain.INK_2),
            ("Estimated today", "today", explain.BLUE),
            ("2030, conditions as now", "conditions_as_now", explain.ORANGE),
            ("2030, recent trend continues", "trend_continues", explain.ORANGE)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(rows))[::-1]
    vals = [s[k]["share"] for _, k, _ in rows]
    ax.barh(y, vals, color=[c for *_, c in rows], height=0.6)
    for yi, (_, k, _) in zip(y, rows):
        v, ci = s[k]["share"], s[k].get("share_90pc")
        txt = f"{v:.0%}" if not ci else f"{v:.0%}  ({ci[0]:.0%} to {ci[1]:.0%})"
        ax.text(v + 0.006, yi, txt, va="center", fontsize=9, color=explain.INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, max(vals) * 1.5)
    explain._style(ax, "Share of social homes below band C", xlabel="Below C")
    explain._save(fig, path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_summary(path: Path, r: dict):
    s = r["scenarios"]
    pct = lambda v: f"{v:.1%}"  # noqa: E731
    names = {"latest_certificates": "Latest certificates, taken at face value",
             "today": f"Estimated today ({r['today']})",
             "conditions_as_now": "1 April 2030, conditions as now",
             "trend_continues": "1 April 2030, recent trend continues"}
    lines = [
        "# 2030 forecast",
        "",
        f"- Homes: **{r['homes']:,}** social rented homes, each scored from its latest certificate",
        f"- Consecutive certificate pairs used to fit the model: **{r['fit_pairs']:,}** "
        f"(later certificate from {FIT_FROM:%Y}, not triggered by works)",
        f"- Homes whose latest SAP score is within 3 points of the C line (66 to 71): "
        f"**{r['near_line_share']:.0%}**",
        "",
        "| Position | Expected homes below C | Share | 90% interval |",
        "|---|---|---|---|",
    ]
    for key, label in names.items():
        sc = s[key]
        ci = sc.get("count_90pc")
        lines.append(f"| {label} | {sc['count']:,.0f} | {pct(sc['share'])} | "
                     f"{f'{ci[0]:,.0f} to {ci[1]:,.0f}' if ci else ''} |")
    pace = r["pace"]
    lines += [
        "",
        f"Getting every home to C by the deadline means about **{pace['needed_per_year']:,.0f} fewer below-C "
        f"homes each year** over the {pace['years_left']:.1f} years left. If the recent trend continues the "
        f"fall is about **{pace['trend_per_year']:,.0f} a year**.",
        "",
        "The intervals cover uncertainty in the fitted model only. Which scenario holds is a bigger "
        "uncertainty than either interval.",
        "",
        "## Checks",
        "",
    ]
    for bt in r["backtests"]:
        if not bt:
            continue
        lines.append(
            f"- **{bt['label']}:** fitted on {bt['train_pairs']:,} pairs, tested on {bt['test_pairs']:,}, with "
            f"the {bt['period_assumed']} period effect carried forward. Observed {pct(bt['observed_below'])} below "
            f"C, predicted {pct(bt['predicted_below'])}. Brier score {bt['brier']:.3f} against "
            f"{bt['brier_assume_no_change']:.3f} for assuming each home keeps its band (lower is better).")
    cv = r["recent_cv"]
    lines += [
        f"- **Recent reassessments, homes not seen in training** (5-fold, split by home): "
        f"{cv['pairs']:,} pairs, observed {pct(cv['observed_below'])}, predicted {pct(cv['predicted_below'])}. "
        f"Brier {cv['brier']:.3f} against {cv['brier_assume_no_change']:.3f} for assuming the band stays put.",
        "",
        "## Period effects (log-odds of being below C at the next certificate)",
        "",
        "| Period | Pairs | Effect |",
        "|---|---|---|",
    ]
    for row in r["period_effects"]:
        lines.append(f"| {row['period']} | {row['pairs']:,} | {row['effect']:+.2f} |")
    lines += ["", f"Recent trend in the period effect: {r['trend_per_year']:+.3f} log-odds a year "
              f"({TREND_FROM_YEAR} to the start of RdSAP 10).", "",
              "Tables: `forecast_by_archetype.csv`, `forecast_by_area.csv`"
              + (", `forecast_by_region.csv`" if r.get("has_regions") else "")
              + ". Everything else: `forecast_2030.json`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved {path}")


def run(history: Path, reports: Path, figures: Path, today: pd.Timestamp, draws: int,
        england: bool = False) -> dict:
    t0 = time.time()
    print("1. Loading certificate history")
    certs = load_history(history)
    if england:
        certs = features.england_only(certs)
    homes = latest_per_home(certs)
    pairs = make_pairs(certs)
    print(f"   {len(certs):,} certificates, {len(homes):,} homes, {len(pairs):,} consecutive pairs")

    print("2. Fitting")
    model, fit_pairs, effects, mids, counts, slope = fit_all(pairs)
    if RDSAP10_PERIOD not in effects.index:
        raise ValueError("No RdSAP 10 certificates in the fitting pairs, so current conditions are unknown.")
    print(f"   {len(fit_pairs):,} routine pairs; recent trend {slope:+.3f} log-odds a year")

    print("3. Checking")
    backtests = [
        temporal_backtest(pairs, "2021-01-01", "2023-01-01", "Stable years: fit to 2020, test 2021-22"),
        temporal_backtest(pairs, "2023-01-01", RDSAP10.strftime("%Y-%m-%d"),
                          "Across the 2023 shift: fit to 2022, test 2023 to RdSAP 10"),
    ]
    cv, calib = recent_cross_validation(pairs)
    near = near_line_by_period(pairs)

    print("4. Projecting")
    sc = scenarios(model, homes, today, slope, mids[RDSAP10_PERIOD])
    homes["p_today"], homes["p_flat"], homes["p_trend"] = sc["today"], sc["conditions_as_now"], sc["trend_continues"]
    ci = bootstrap(pairs, homes, today, draws) if draws else {}
    n = len(homes)

    def summarise(values, key=None):
        out = {"count": float(np.sum(values)), "share": float(np.mean(values))}
        if key in ci:
            out["count_90pc"], out["share_90pc"] = ci[key], [c / n for c in ci[key]]
        return out

    years_left = (TARGET_DATE - today).days / 365.25
    results = {
        "homes": n, "certificates": len(certs), "pairs": len(pairs), "fit_pairs": len(fit_pairs),
        "today": f"{today:%Y-%m-%d}", "target_date": f"{TARGET_DATE:%Y-%m-%d}",
        "homes_with_more_than_one_certificate": int(certs["home_id"].value_counts().gt(1).sum()),
        "works_triggered_share": float(certs["works_triggered"].mean()),
        "near_line_share": float(homes["sap"].between(66, 71).mean()),
        "scenarios": {"latest_certificates": summarise(homes["below_c"]),
                      **{k: summarise(v, k) for k, v in sc.items()}},
        "pace": {"years_left": years_left, "needed_per_year": float(sc["today"].sum() / years_left),
                 "trend_per_year": float((sc["today"].sum() - sc["trend_continues"].sum()) / years_left)},
        "trend_per_year": slope,
        "period_effects": [{"period": p, "pairs": int(counts.get(p, 0)), "midpoint": float(mids[p]),
                            "effect": float(effects[p])} for p in mids.sort_values().index if p in effects],
        "near_line_by_period": near.round(4).to_dict(orient="records"),
        "backtests": backtests, "recent_cv": cv,
    }

    print("5. Writing outputs")
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (reports / "forecast_2030.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    breakdown(homes, ["property_type", "build_era", "fuel_group"], 100).to_csv(
        reports / "forecast_by_archetype.csv", index=False)
    area = "local_authority_label" if "local_authority_label" in homes else "local_authority"
    breakdown(homes, [area], 50).to_csv(reports / "forecast_by_area.csv", index=False)
    if "region" in homes and homes["region"].nunique() > 1:
        by_region = breakdown(homes, ["region"], 50)
        by_region = by_region[by_region["region"].astype(str) != "Unknown"]
        by_region.to_csv(reports / "forecast_by_region.csv", index=False)
        plot_by_region(by_region, figures / "11_forecast_by_region.png")
    results["has_regions"] = bool("region" in homes and homes["region"].nunique() > 1)
    write_summary(reports / "forecast_summary.md", results)
    if len(near):
        plot_near_line(near, figures / "07_near_line_by_year.png")
    plot_scenarios(results, figures / "08_forecast_2030.png")
    explain.plot_calibration(*calib, figures / "09_forecast_calibration.png",
                             "Calibration: recent reassessments, unseen homes")
    print(f"Done in {time.time() - t0:.0f}s")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", type=Path, required=True,
                   help="file or folder made by scripts/extract_subset.py --history")
    p.add_argument("--today", default=None, help="date for the 'estimated today' figure (default: today)")
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DRAWS, help="refits for intervals (0 to skip)")
    p.add_argument("--reports", type=Path, default=config.REPORTS_DIR)
    p.add_argument("--theme", choices=["dark", "light"], default="dark")
    p.add_argument("--england-only", action="store_true", help="drop Welsh local authorities")
    args = p.parse_args(argv)
    explain.set_theme(args.theme)
    today = pd.Timestamp(args.today) if args.today else pd.Timestamp.today().normalize()
    run(args.history, args.reports, args.reports / "figures", today, args.bootstrap, args.england_only)


if __name__ == "__main__":
    main()

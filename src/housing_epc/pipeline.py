"""End-to-end run: load -> clean -> train -> evaluate -> explain -> report.

Usage (from the project root):
    python -m housing_epc.pipeline                 # uses data/raw
    python -m housing_epc.pipeline --sample 200000 # subsample for a faster run
    python -m housing_epc.pipeline --raw path/to/folder --no-shap
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from . import config, explain, features, load, model


def build_dataset(raw_dir: Path, min_year: int, england: bool = False) -> pd.DataFrame:
    print("1. Loading social rented certificates")
    raw = load.load_social_certificates(raw_dir, min_year=min_year)
    print(f"   {len(raw):,} certificates loaded")
    if england:
        raw = features.england_only(raw)
    latest = features.keep_latest_certificate(raw)
    del raw
    print(f"   {len(latest):,} homes after keeping the latest certificate per home")
    table = features.add_region(features.build_features_chunked(latest))
    print(f"   {len(table):,} homes with a valid rating; {table['below_c'].mean():.1%} below C")
    return table


def write_summary(path: Path, table: pd.DataFrame, metrics: dict, cv: dict, top_features: pd.DataFrame,
                  archetypes: pd.DataFrame):
    core, ext = metrics["core"]["gradient_boosting"], metrics["extended"]["gradient_boosting"]
    lines = [
        "# Results summary",
        "",
        f"- Homes modelled: **{len(table):,}** social rented homes, latest EPC per home",
        f"- Share below band C: **{table['below_c'].mean():.1%}**",
        f"- Local authorities covered: **{table['local_authority'].nunique():,}**",
        "",
        "## How well can we predict which homes are below C?",
        "",
        "| Feature set | Model | ROC AUC | PR AUC | Brier |",
        "|---|---|---|---|---|",
    ]
    for fs in metrics:
        for m in ("logistic_regression", "gradient_boosting"):
            s = metrics[fs][m]
            lines.append(f"| {fs} | {m.replace('_', ' ')} | {s['roc_auc']:.3f} | {s['pr_auc']:.3f} | {s['brier']:.3f} |")
    lines += [
        "",
        f"Adding survey-level fabric and heating detail moves gradient boosting ROC AUC from "
        f"{core['roc_auc']:.3f} (asset-register data only) to {ext['roc_auc']:.3f}.",
        "",
    ]
    if cv:
        lines += ["## Does it transfer to unseen areas?", ""]
        for fs, res in cv.items():
            if res:
                lines.append(f"- {fs}: mean ROC AUC {res['mean_roc_auc']:.3f} (± {res['std_roc_auc']:.3f}) "
                             f"across {len(res['fold_roc_auc'])} leave-areas-out folds")
                for areas, auc in zip(res.get("held_out_areas", []), res["fold_roc_auc"]):
                    lines.append(f"  - trained without {', '.join(areas)}, tested on it: {auc:.3f}")
        lines.append("")
    lines += ["## Most important drivers (extended model, permutation importance)", ""]
    for _, r in top_features.head(6).iterrows():
        lines.append(f"- {explain.pretty(r['feature'])}: ROC AUC drops by {r['auc_drop_mean']:.3f} when shuffled")
    lines += ["", "## Highest-risk archetypes (core model)", "",
              "Homes and the share below C count every home. Predicted is the model's average risk for the "
              "archetype's homes in the held-out test set.", "",
              "| Property type | Build era | Main fuel | Homes | Below C | Share below C | Predicted |",
              "|---|---|---|---|---|---|---|"]
    for _, r in archetypes.head(10).iterrows():
        lines.append(f"| {r['property_type']} | {r['build_era']} | {r['fuel_group']} | {r['homes']:,} | "
                     f"{r['homes_below_c']:,} | {r['actual_below_c']:.0%} | {r['predicted_risk']:.0%} |")
    where = path.parent.relative_to(config.PROJECT_ROOT) if path.parent.is_relative_to(config.PROJECT_ROOT) else path.parent
    lines += ["", f"Figures are in `{where.as_posix()}/figures/`. Archetype and area tables are in `{where.as_posix()}/`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved {path.relative_to(config.PROJECT_ROOT) if path.is_relative_to(config.PROJECT_ROOT) else path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=config.RAW_DIR, help="folder containing EPC certificate CSVs")
    p.add_argument("--min-year", type=int, default=config.MIN_LODGEMENT_YEAR)
    p.add_argument("--sample", type=int, default=None, help="randomly subsample this many homes")
    p.add_argument("--no-shap", action="store_true", help="skip SHAP even if installed")
    p.add_argument("--no-cv", action="store_true", help="skip leave-areas-out validation")
    p.add_argument("--theme", choices=["dark", "light"], default="dark",
                   help="chart colour scheme (dark matches GitHub's dark mode)")
    p.add_argument("--england-only", action="store_true", help="drop Welsh local authorities")
    p.add_argument("--reports", type=Path, default=None,
                   help="write reports and figures here instead of reports/ (e.g. reports/england)")
    args = p.parse_args(argv)
    if args.reports:
        config.REPORTS_DIR = args.reports.resolve()
        config.FIGURES_DIR = config.REPORTS_DIR / "figures"
    start = time.time()
    explain.set_theme(args.theme)

    for d in (config.PROCESSED_DIR, config.FIGURES_DIR, config.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    table = build_dataset(args.raw, args.min_year, args.england_only)
    if args.sample and args.sample < len(table):
        table = table.sample(args.sample, random_state=config.RANDOM_STATE).reset_index(drop=True)
        print(f"   subsampled to {len(table):,} homes")
    table.to_csv(config.PROCESSED_DIR / "social_homes_features.csv.gz", index=False)

    print("2. Training models")
    runs, metrics, cv = {}, {}, {}
    for fs in config.FEATURE_SETS:
        print(f"  feature set: {fs}")
        runs[fs] = model.train_feature_set(table, fs)
        metrics[fs] = runs[fs].metrics
        gb = metrics[fs]["gradient_boosting"]
        print(f"   gradient boosting ROC AUC {gb['roc_auc']:.3f} | "
              f"logistic ROC AUC {metrics[fs]['logistic_regression']['roc_auc']:.3f}")
        joblib.dump({"encoder": runs[fs].encoder, "model": runs[fs].booster},
                    config.MODELS_DIR / f"booster_{fs}.joblib")
        if not args.no_cv:
            if features.regions_add_detail(table):  # national data: hold out one whole region at a time
                regions = [r for r in table["region"].astype("string").dropna().unique() if r != "Unknown"]
                cv[fs] = model.leave_areas_out_cv(table, fs, n_splits=len(regions), group_col="region")
            else:
                cv[fs] = model.leave_areas_out_cv(table, fs)

    print("3. Explaining")
    fig = config.FIGURES_DIR
    explain.plot_feature_set_comparison(metrics, fig / "01_feature_set_comparison.png")
    explain.plot_rate_over_time(table, fig / "00_below_c_over_time.png")
    explain.plot_calibration(runs["extended"].y_test, runs["extended"].prob_test,
                             fig / "02_calibration_extended.png", "Calibration: extended model")

    importances = {}
    for fs, run in runs.items():
        importances[fs] = explain.permutation_importances(run)
        importances[fs].to_csv(config.REPORTS_DIR / f"permutation_importance_{fs}.csv", index=False)
        explain.plot_importance(importances[fs], fig / f"03_importance_{fs}.png",
                                f"What drives the prediction? ({fs} features)")

    pd_features = {"core": ["construction_year", "property_type", "fuel_group", "total_floor_area"],
                   "extended": ["wall_insulation", "heating_system", "loft_insulation_mm"]}
    for fs, feats in pd_features.items():
        run = runs[fs]
        for f in feats:
            if f in run.encoder.feature_names:
                df = explain.partial_dependence(run, f)
                explain.plot_partial_dependence(df, f, f in run.encoder.categorical,
                                                fig / f"04_pdp_{fs}_{f}.png")

    if not args.no_shap:
        run = runs["extended"]
        out = explain.shap_values(run)
        if out is not None:
            values, X = out
            sh = explain.shap_importance(values, run.encoder.feature_names)
            sh.to_csv(config.REPORTS_DIR / "shap_importance_extended.csv", index=False)
            explain.plot_importance(sh, fig / "05_shap_importance_extended.png",
                                    "SHAP: average impact on predicted risk", value_col="mean_abs_shap",
                                    xlabel="Mean |SHAP value|")
            for f in sh["feature"].head(4):
                explain.plot_shap_dependence(run, values, X, f, fig / f"06_shap_{f}.png")

    # at least 30 test homes, or 0.1% of them for national-scale data, so tiny groups don't top the list
    # counts over all homes; at least 150 homes, or 0.1% of them at national scale, so tiny groups don't top the list
    archetypes = explain.archetype_table(runs["core"], population=table, min_homes=max(150, len(table) // 1000))
    archetypes.to_csv(config.REPORTS_DIR / "archetype_risk.csv", index=False)
    explain.local_authority_table(runs["core"]).to_csv(config.REPORTS_DIR / "local_authority_gap.csv", index=False)
    if features.regions_add_detail(table):
        regions = explain.local_authority_table(runs["core"], by="region")
        regions = regions[regions["region"].astype(str) != "Unknown"]
        regions.to_csv(config.REPORTS_DIR / "region_gap.csv", index=False)
        explain.plot_area_gap(regions, "region", fig / "10_region_gap.png",
                              "Share below C by region, against the stock mix")

    with open(config.REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"metrics": metrics, "leave_areas_out_cv": cv, "n_homes": len(table)}, fh, indent=2)
    write_summary(config.REPORTS_DIR / "summary.md", table, metrics, cv, importances["extended"], archetypes)
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()

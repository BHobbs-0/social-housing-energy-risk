"""Explaining the model: what drives the risk of a social home being below EPC C?

Three layers, from overall to individual:
  1. Permutation importance: which features matter most for ranking homes.
  2. Partial dependence: how risk changes as one feature changes, others held as observed.
  3. SHAP values (optional, needs `pip install shap`): each feature's contribution
     to each individual home's prediction.

Plus two stakeholder tables: risk by housing archetype and by local authority.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from . import config
from .features import age_band_label
from .model import RunResult

# Chart styling: one accent colour, recessive axes, readable labels.
# "dark" matches GitHub's dark theme so figures sit seamlessly in the README.
THEMES = {
    "dark": {"BLUE": "#3987e5", "ORANGE": "#d95926", "INK": "#e6edf3", "INK_2": "#9198a1",
             "GRID": "#30363d", "SURFACE": "#0d1117"},
    "light": {"BLUE": "#2a78d6", "ORANGE": "#eb6834", "INK": "#0b0b0b", "INK_2": "#52514e",
              "GRID": "#e4e3df", "SURFACE": "#ffffff"},
}
BLUE = ORANGE = INK = INK_2 = GRID = SURFACE = ""


def set_theme(name: str = "dark") -> None:
    """Switch every chart in this module between the dark and light palettes."""
    if name not in THEMES:
        raise ValueError(f"theme must be one of {list(THEMES)}")
    globals().update(THEMES[name])


set_theme("dark")

PRETTY = {
    "property_type": "Property type", "built_form": "Built form", "fuel_group": "Main fuel",
    "construction_year": "Build year", "total_floor_area": "Floor area (m²)",
    "mains_gas": "Mains gas available", "wall_type": "Wall type",
    "wall_insulation": "Wall insulation", "roof_type": "Roof type",
    "heating_system": "Heating system", "loft_insulation_mm": "Loft insulation (mm)",
    "glazing_pct": "Multiple glazing (%)", "lodgement_year": "Year assessed",
    "rdsap10": "Assessed under RdSAP 10",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " ").capitalize())


def pretty_level(value) -> str:
    """'storage_heaters' -> 'Storage heaters'; leaves EPC labels like 'Mid-Terrace' alone."""
    text = str(value)
    return text.replace("_", " ").capitalize() if "_" in text or text.islower() else text


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved {path.relative_to(config.PROJECT_ROOT) if path.is_relative_to(config.PROJECT_ROOT) else path}")


# ---------------------------------------------------------------------------
# Model-level charts
# ---------------------------------------------------------------------------
def plot_feature_set_comparison(metrics: dict, path: Path):
    """Does knowing wall, roof and heating details beat basic asset-register data?"""
    rows = []
    for fs, m in metrics.items():
        for model_name in ("logistic_regression", "gradient_boosting"):
            rows.append((fs, model_name, m[model_name]["roc_auc"]))
    df = pd.DataFrame(rows, columns=["feature_set", "model", "roc_auc"])

    fig, ax = plt.subplots(figsize=(7, 3.8))
    sets = list(metrics)
    y = np.arange(len(sets))
    h = 0.36
    for i, (model_name, colour, label) in enumerate([
        ("logistic_regression", ORANGE, "Logistic regression"),
        ("gradient_boosting", BLUE, "Gradient boosting"),
    ]):
        vals = df[df["model"] == model_name].set_index("feature_set").loc[sets, "roc_auc"].to_numpy()
        bars = ax.barh(y + (i - 0.5) * h, vals, height=h - 0.04, color=colour, label=label)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [f"{s.capitalize()} features" for s in sets])
    ax.set_xlim(0.5, 1.0)
    _style(ax, "Test-set ROC AUC by feature set and model", xlabel="ROC AUC (0.5 = random)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=9, ncol=2, labelcolor=INK,
              loc="upper center", bbox_to_anchor=(0.5, -0.25))
    _save(fig, path)


def plot_calibration(y_true, prob, path: Path, title: str):
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, color=BLUE, linewidth=2, marker="o", markersize=6, label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax, title, xlabel="Predicted probability of being below C", ylabel="Observed share below C")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _save(fig, path)


def plot_rate_over_time(table: pd.DataFrame, path: Path, min_homes: int = 200):
    """Observed share below C by quarter of assessment, with the RdSAP 10 change marked."""
    df = table.dropna(subset=["lodgement_date"]).copy()
    df["quarter"] = pd.to_datetime(df["lodgement_date"], errors="coerce").dt.to_period("Q").dt.start_time
    q = df.groupby("quarter")["below_c"].agg(["mean", "size"])
    q = q[q["size"] >= min_homes]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(q.index, q["mean"], color=BLUE, linewidth=2)
    change = pd.Timestamp(config.RDSAP10_START)
    ax.axvline(change, color=ORANGE, linewidth=1.2, linestyle="--")
    ax.text(change, q["mean"].max() * 1.05, "  RdSAP 10 introduced",
            color=ORANGE, fontsize=9, va="bottom")
    ax.set_ylim(0, max(0.6, q["mean"].max() * 1.2))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax, "Share of assessed social homes below band C, by quarter",
           xlabel="Quarter the EPC was lodged", ylabel="Below C")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 1. Permutation importance
# ---------------------------------------------------------------------------
def permutation_importances(run: RunResult, max_rows: int = 20_000) -> pd.DataFrame:
    n = min(max_rows, len(run.y_test))
    idx = np.random.default_rng(config.RANDOM_STATE).choice(len(run.y_test), size=n, replace=False)
    result = permutation_importance(
        run.booster, run.X_test[idx], run.y_test[idx], scoring="roc_auc",
        n_repeats=5, random_state=config.RANDOM_STATE,
    )
    return (pd.DataFrame({
        "feature": run.encoder.feature_names,
        "auc_drop_mean": result.importances_mean,
        "auc_drop_std": result.importances_std,
    }).sort_values("auc_drop_mean", ascending=False).reset_index(drop=True))


def plot_importance(imp: pd.DataFrame, path: Path, title: str, value_col: str = "auc_drop_mean",
                    xlabel: str = "Drop in ROC AUC when the feature is shuffled"):
    imp = imp.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(imp) + 1.2))
    ax.barh([pretty(f) for f in imp["feature"]], imp[value_col], color=BLUE, height=0.65)
    _style(ax, title, xlabel=xlabel)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 2. Partial dependence (model-agnostic, computed directly)
# ---------------------------------------------------------------------------
def partial_dependence(run: RunResult, feature: str, max_rows: int = 3000) -> pd.DataFrame:
    """Average predicted risk when every home is assigned each value of `feature`."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    n = min(max_rows, len(run.X_test))
    X = run.X_test[rng.choice(len(run.X_test), size=n, replace=False)].copy()
    j = run.encoder.feature_names.index(feature)

    if feature in run.encoder.categorical:
        grid = np.arange(len(run.encoder.categories[feature]), dtype=float)
        labels = run.encoder.categories[feature]
        counts = pd.Series(run.X_test[:, j]).value_counts()
        share = [counts.get(g, 0) / len(run.X_test) for g in grid]
    else:
        col = run.X_test[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            return pd.DataFrame(columns=["value", "avg_risk", "share"])
        grid = np.unique(np.quantile(col, np.linspace(0.02, 0.98, 25)))
        labels = grid
        share = [np.nan] * len(grid)

    avg = []
    for g in grid:
        X[:, j] = g
        avg.append(run.booster.predict_proba(X)[:, 1].mean())
    return pd.DataFrame({"value": labels, "avg_risk": avg, "share": share})


def plot_partial_dependence(pd_df: pd.DataFrame, feature: str, categorical: bool, path: Path):
    if pd_df.empty:
        return
    if categorical:
        pd_df = pd_df[pd_df["share"] >= 0.005].sort_values("avg_risk")
        fig, ax = plt.subplots(figsize=(7, 0.42 * len(pd_df) + 1.2))
        ax.barh(pd_df["value"].map(pretty_level), pd_df["avg_risk"], color=BLUE, height=0.65)
        for i, (v, s) in enumerate(zip(pd_df["avg_risk"], pd_df["share"])):
            ax.text(v + 0.01, i, f"{v:.0%}  ({s:.0%} of homes)", va="center", fontsize=8.5, color=INK_2)
        ax.set_xlim(0, min(1.0, pd_df["avg_risk"].max() * 1.35))
        _style(ax, f"Predicted risk of being below C, by {pretty(feature).lower()}",
               xlabel="Average predicted probability")
    else:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(pd_df["value"], pd_df["avg_risk"], color=BLUE, linewidth=2)
        _style(ax, f"Predicted risk of being below C, by {pretty(feature).lower()}",
               xlabel=pretty(feature), ylabel="Average predicted probability")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_ylim(0, 1)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 3. SHAP (optional)
# ---------------------------------------------------------------------------
def shap_values(run: RunResult, n_explain: int = 1000, n_background: int = 100):
    """Return (values, X_explained) or None if shap is not installed.

    Uses the model-agnostic permutation explainer on the encoded matrix, which works
    with scikit-learn's native categorical handling and missing values.
    """
    try:
        import shap
    except ImportError:
        print("  shap not installed: skipping SHAP (pip install shap to enable)")
        return None
    rng = np.random.default_rng(config.RANDOM_STATE)
    X = run.X_test
    background = X[rng.choice(len(X), size=min(n_background, len(X)), replace=False)]
    explain_rows = X[rng.choice(len(X), size=min(n_explain, len(X)), replace=False)]

    def predict(data):
        return run.booster.predict_proba(data)[:, 1]

    explainer = shap.Explainer(predict, shap.maskers.Independent(background),
                               feature_names=run.encoder.feature_names, algorithm="permutation")
    sv = explainer(explain_rows)
    return np.asarray(sv.values), explain_rows


def shap_importance(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    return (pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def shap_by_category(run: RunResult, values: np.ndarray, X: np.ndarray, feature: str) -> pd.DataFrame:
    j = run.encoder.feature_names.index(feature)
    labels = run.encoder.decode(feature, X[:, j])
    return (pd.DataFrame({"value": labels, "shap": values[:, j]})
            .groupby("value")["shap"].agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "mean_shap"}).sort_values("mean_shap"))


def plot_shap_dependence(run: RunResult, values: np.ndarray, X: np.ndarray, feature: str, path: Path):
    j = run.encoder.feature_names.index(feature)
    if feature in run.encoder.categorical:
        df = shap_by_category(run, values, X, feature)
        df = df[df["count"] >= 10]
        fig, ax = plt.subplots(figsize=(7, 0.42 * len(df) + 1.2))
        colours = [ORANGE if v > 0 else BLUE for v in df["mean_shap"]]
        ax.barh(df["value"].map(pretty_level), df["mean_shap"], color=colours, height=0.65)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        _style(ax, f"SHAP: how {pretty(feature).lower()} shifts each home's risk",
               xlabel="Mean SHAP value (positive = pushes towards below C)")
    else:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.scatter(X[:, j], values[:, j], s=9, alpha=0.35, color=BLUE, linewidths=0)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        _style(ax, f"SHAP: how {pretty(feature).lower()} shifts each home's risk",
               xlabel=pretty(feature), ylabel="SHAP value")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    _save(fig, path)


# ---------------------------------------------------------------------------
# Stakeholder tables
# ---------------------------------------------------------------------------
def archetype_table(run: RunResult, min_homes: int = 30) -> pd.DataFrame:
    """Risk by housing archetype (type x build era x fuel), on the test set."""
    df = run.test_frame.copy()
    df["predicted_risk"] = run.prob_test
    df["build_era"] = age_band_label(df["construction_year"])
    grp = (df.groupby(["property_type", "build_era", "fuel_group"], dropna=False)
           .agg(homes=("below_c", "size"), actual_below_c=("below_c", "mean"),
                predicted_risk=("predicted_risk", "mean"))
           .reset_index())
    grp = grp[grp["homes"] >= min_homes].sort_values("predicted_risk", ascending=False)
    return grp.round({"actual_below_c": 3, "predicted_risk": 3})


def local_authority_table(run: RunResult, min_homes: int = 50) -> pd.DataFrame:
    """Observed vs expected below-C share by area.

    'expected' is what the model predicts from the stock mix alone, so the gap shows
    areas doing better or worse than their housing stock would suggest.
    """
    df = run.test_frame.copy()
    df["predicted_risk"] = run.prob_test
    label = "local_authority_label" if "local_authority_label" in df else config.GROUP_COLUMN
    grp = (df.groupby(label, dropna=False)
           .agg(homes=("below_c", "size"), actual_below_c=("below_c", "mean"),
                expected_below_c=("predicted_risk", "mean"))
           .reset_index())
    grp["gap"] = grp["actual_below_c"] - grp["expected_below_c"]
    grp = grp[grp["homes"] >= min_homes].sort_values("gap")
    return grp.round(3)

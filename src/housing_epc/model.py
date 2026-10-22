"""Model training and evaluation.

Two models per feature set:
  * a logistic regression baseline (transparent, easy to explain to stakeholders)
  * a histogram gradient boosting classifier (handles missing values and
    categories natively, captures interactions such as "old AND solid wall")

Evaluation answers two questions:
  1. How well does it rank homes by risk on unseen homes? (held-out test set)
  2. Does it transfer to areas it has never seen? (leave-local-authorities-out CV)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


# ---------------------------------------------------------------------------
# Encoding for the boosting model
# ---------------------------------------------------------------------------
@dataclass
class Encoder:
    """Turns a feature DataFrame into a float matrix with integer-coded categories.

    Keeping categories as integer codes (with a mask telling the model which
    columns are categorical) lets the gradient boosting model split on categories
    natively, and keeps the matrix numeric for SHAP.
    """

    categorical: list[str]
    numeric: list[str]
    categories: dict[str, list[str]] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return self.categorical + self.numeric

    @property
    def categorical_mask(self) -> np.ndarray:
        return np.array([True] * len(self.categorical) + [False] * len(self.numeric))

    def fit(self, df: pd.DataFrame) -> "Encoder":
        for col in self.categorical:
            self.categories[col] = sorted(df[col].dropna().astype(str).unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for col in self.categorical:
            values = df[col].map(lambda v: str(v) if pd.notna(v) else None)
            cat = pd.Categorical(values, categories=self.categories[col])
            codes = cat.codes.astype(float)
            codes[codes < 0] = np.nan  # unseen or missing
            parts.append(codes)
        for col in self.numeric:
            parts.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
        return np.column_stack(parts)

    def decode(self, col: str, codes: np.ndarray) -> np.ndarray:
        labels = np.array(self.categories[col] + ["(missing)"], dtype=object)
        idx = np.where(np.isnan(codes), len(self.categories[col]), codes).astype(int)
        return labels[idx]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_baseline(categorical: list[str], numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="(missing)")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]), categorical),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])


def make_booster(categorical_mask: np.ndarray) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        categorical_features=categorical_mask,
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=config.RANDOM_STATE,
    )


def _baseline_frame(df: pd.DataFrame, categorical: list[str], numeric: list[str]) -> pd.DataFrame:
    X = df[categorical + numeric].copy()
    for col in categorical:
        X[col] = X[col].astype("object").where(X[col].notna(), np.nan)
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def score(y_true, prob, threshold: float = 0.5) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, (prob >= threshold).astype(int))),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Training run for one feature set
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    name: str
    encoder: Encoder
    booster: HistGradientBoostingClassifier
    baseline: Pipeline
    X_test: np.ndarray
    y_test: np.ndarray
    test_frame: pd.DataFrame
    prob_test: np.ndarray
    metrics: dict


def drop_features_with_no_data(df: pd.DataFrame, categorical: list[str], numeric: list[str]):
    """Some downloads lack the description fields; drop features that are entirely empty."""
    keep_cat = [c for c in categorical if df[c].notna().any()]
    keep_num = [c for c in numeric if df[c].notna().any()]
    dropped = sorted(set(categorical + numeric) - set(keep_cat + keep_num))
    if dropped:
        print(f"  dropping empty features: {dropped}")
    return keep_cat, keep_num


def train_feature_set(df: pd.DataFrame, name: str, test_size: float = 0.2) -> RunResult:
    spec = config.FEATURE_SETS[name]
    categorical, numeric = drop_features_with_no_data(df, spec["categorical"], spec["numeric"])

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["below_c"], random_state=config.RANDOM_STATE
    )
    y_train = train_df["below_c"].to_numpy()
    y_test = test_df["below_c"].to_numpy()

    # Baseline
    baseline = make_baseline(categorical, numeric)
    baseline.fit(_baseline_frame(train_df, categorical, numeric), y_train)
    prob_base = baseline.predict_proba(_baseline_frame(test_df, categorical, numeric))[:, 1]

    # Gradient boosting
    encoder = Encoder(categorical, numeric).fit(train_df)
    X_train, X_test = encoder.transform(train_df), encoder.transform(test_df)
    booster = make_booster(encoder.categorical_mask)
    booster.fit(X_train, y_train)
    prob_boost = booster.predict_proba(X_test)[:, 1]

    metrics = {
        "features": encoder.feature_names,
        "logistic_regression": score(y_test, prob_base),
        "gradient_boosting": score(y_test, prob_boost),
        "boosting_iterations": int(booster.n_iter_),
    }
    return RunResult(name, encoder, booster, baseline, X_test, y_test,
                     test_df.reset_index(drop=True), prob_boost, metrics)


def leave_areas_out_cv(df: pd.DataFrame, name: str, n_splits: int = 5) -> dict | None:
    """Train on some local authorities, test on others. Returns None if too few areas."""
    groups = df[config.GROUP_COLUMN].fillna("unknown").to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 3:
        print(f"  only {n_groups} local authorities: skipping leave-areas-out validation")
        return None
    spec = config.FEATURE_SETS[name]
    categorical, numeric = drop_features_with_no_data(df, spec["categorical"], spec["numeric"])

    label_col = "local_authority_label" if "local_authority_label" in df else config.GROUP_COLUMN
    aucs, held_out = [], []
    for train_idx, test_idx in GroupKFold(n_splits=min(n_splits, n_groups)).split(df, groups=groups):
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        if te["below_c"].nunique() < 2:
            continue
        enc = Encoder(categorical, numeric).fit(tr)
        model = make_booster(enc.categorical_mask).fit(enc.transform(tr), tr["below_c"])
        aucs.append(float(roc_auc_score(te["below_c"], model.predict_proba(enc.transform(te))[:, 1])))
        held_out.append(sorted(te[label_col].dropna().astype(str).unique().tolist()))
    return {"fold_roc_auc": aucs, "held_out_areas": held_out,
            "mean_roc_auc": float(np.mean(aucs)), "std_roc_auc": float(np.std(aucs))}

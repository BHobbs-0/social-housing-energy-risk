import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from housing_epc import forecast

ROOT = Path(__file__).resolve().parents[1]


def certificates(rows):
    """Minimal certificate table in the shape load_history returns."""
    df = pd.DataFrame(rows, columns=["home_id", "date", "sap", "below_c", "works_triggered", "total_floor_area"])
    df["date"] = pd.to_datetime(df["date"])
    for c in forecast.CATEGORICAL:
        df[c] = "x"
    return df


def simulated_pairs(n=6000, seed=0):
    """Pairs where the chance of ending below C falls with the starting SAP score and in later periods."""
    rng = np.random.default_rng(seed)
    sap = rng.uniform(45, 85, n)
    period = rng.choice(["2021", "2022", forecast.RDSAP10_PERIOD], n)
    shift = pd.Series(period).map({"2021": 0.5, "2022": 0.0, forecast.RDSAP10_PERIOD: -1.0}).to_numpy()
    p = 1 / (1 + np.exp(-(-(sap - 66) / 3 + shift)))
    df = pd.DataFrame({"sap": sap, "gap": rng.uniform(0.5, 10, n), "start_works": 0, "period": period,
                       "end_below": (rng.random(n) < p).astype(int), "home_id": np.arange(n).astype(str)})
    for c in forecast.CATEGORICAL:
        df[c] = rng.choice(["a", "b"], n)
    return df


def test_period_labels_split_at_rdsap10():
    dates = pd.Series(pd.to_datetime(["2024-12-31", "2025-06-14", "2025-06-15", "2026-03-01"]))
    assert forecast.period_of(dates).tolist() == ["2024", "2025", "RdSAP 10", "RdSAP 10"]


def test_make_pairs_links_consecutive_certificates_and_drops_mismatches():
    certs = certificates([
        ("a", "2016-01-01", 60, 1, False, 70), ("a", "2016-02-01", 61, 1, False, 70),  # one month: dropped
        ("a", "2020-01-01", 72, 0, True, 71),
        ("b", "2018-01-01", 70, 0, False, 35), ("b", "2022-01-01", 80, 0, False, 12),  # different dwelling
        ("c", "2024-01-01", 70, 0, False, 50), ("c", "2025-09-01", 66, 1, False, 52),
    ])
    pairs = forecast.make_pairs(certs)
    assert len(pairs) == 2
    a, c = pairs.set_index("home_id").loc["a"], pairs.set_index("home_id").loc["c"]
    assert (a["start_below"], a["end_below"], bool(a["end_works"])) == (1, 0, True)
    assert a["period"] == "2020" and c["period"] == "RdSAP 10"
    assert len(forecast.routine(pairs)) == 1  # the works-triggered end certificate is excluded from fitting


def test_model_learns_sap_and_period_effects():
    pairs = simulated_pairs()
    model = forecast.NextCertificateModel().fit(pairs)
    probe = pairs.head(2).copy()
    probe["sap"], probe["period"] = [55, 75], "2022"
    p = model.predict(probe)
    assert p[0] > 0.8 and p[1] < 0.2
    eff = model.period_effects()
    assert eff["2021"] > eff["2022"] > eff[forecast.RDSAP10_PERIOD]


def test_shift_lowers_risk():
    pairs = simulated_pairs(2000)
    model = forecast.NextCertificateModel().fit(pairs)
    assert np.all(model.predict(pairs, shift=-0.5) < model.predict(pairs))


def test_trend_uses_recent_years_only():
    eff = pd.Series({"2015": 5.0, "2019": 0.0, "2020": -0.2, "2021": -0.4, forecast.RDSAP10_PERIOD: -9.0})
    mids = pd.Series({"2015": 2015.5, "2019": 2019.5, "2020": 2020.5, "2021": 2021.5, forecast.RDSAP10_PERIOD: 2026.0})
    w = pd.Series(1, index=eff.index)
    assert np.isclose(forecast.trend_per_year(eff, mids, w), -0.2)


def test_history_extract_is_anonymous():
    spec = importlib.util.spec_from_file_location("extract_subset", ROOT / "scripts" / "extract_subset.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    raw = pd.DataFrame({
        "uprn": ["100", "100", None], "address1": ["1 High St", "1 High St", "2 Low Rd"],
        "postcode": ["BS1 1AA", "BS1 1AA", "BS2 2BB"], "current_energy_rating": ["D", "C", "E"],
        "lodgement_date": ["2016-01-01", "2021-01-01", "2019-05-05"],
    })
    out1, out2 = ex.all_certificates(raw), ex.all_certificates(raw)
    assert not {"uprn", "address1", "postcode"} & set(out1.columns)
    assert out1["home_id"].nunique() == 2
    assert set(out1["home_id"]).isdisjoint(out2["home_id"])  # new salt every run

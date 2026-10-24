import numpy as np
import pandas as pd

from housing_epc import columns, config, features


def s(*values):
    return pd.Series(list(values))


def test_normalise_column_names():
    assert columns.normalise_name("CURRENT_ENERGY_RATING") == "current_energy_rating"
    assert columns.normalise_name("current-energy-rating ") == "current_energy_rating"
    assert columns.build_rename_map(["LA_CODE"]) == {"LA_CODE": "local_authority"}


def test_construction_year():
    out = features.parse_construction_year(s(
        "England and Wales: 1930-1949", "England and Wales: before 1900",
        "England and Wales: 2012 onwards", "2019", None, "NO DATA!"))
    assert out.tolist()[:4] == [1939.5, 1890, 2015, 2019]
    assert out.iloc[4:].isna().all()


def test_fuel_groups_handle_not_community():
    out = features.group_fuel(s("mains gas (not community)", "mains gas (community)",
                                "electricity (not community)", "oil (not community)", None))
    assert out.tolist()[:4] == ["mains_gas", "communal", "electricity", "oil"]
    assert pd.isna(out.iloc[4])


def test_walls_and_insulation():
    desc = s("Cavity wall, filled cavity", "Solid brick, as built, no insulation (assumed)",
             "Cavity wall, as built, partial insulation (assumed)", "Timber frame, as built, insulated (assumed)",
             "Average thermal transmittance 0.28 W/m-¦K")
    assert features.group_wall_type(desc).tolist() == ["cavity", "solid", "cavity", "timber_frame", "new_build_spec"]
    assert features.group_wall_insulation(desc).tolist() == ["insulated", "uninsulated", "partial",
                                                             "insulated", "insulated"]


def test_roof_and_loft():
    desc = s("Pitched, 270 mm loft insulation", "Pitched, no insulation", "(another dwelling above)",
             "Pitched, 400+ mm loft insulation", "Flat, limited insulation")
    assert features.group_roof_type(desc).tolist() == ["pitched", "pitched", "dwelling_above", "pitched", "flat"]
    loft = features.parse_loft_insulation_mm(desc)
    assert loft.iloc[:2].tolist() == [270, 0] and loft.iloc[3] == 400
    assert np.isnan(loft.iloc[2])


def test_heating():
    desc = s("Boiler and radiators, mains gas", "Electric storage heaters", "Air source heat pump, radiators, electric",
             "Community scheme", "Room heaters, electric")
    assert features.group_heating(desc).tolist() == ["boiler", "storage_heaters", "heat_pump", "communal", "room_heaters"]


def test_keep_latest_certificate():
    df = pd.DataFrame({
        "uprn": ["1", "1", None, None],
        "address1": ["a", "a", "5 High St", "5 high st"],
        "postcode": ["X", "X", "BS1 1AA", "BS11AA"],
        "lodgement_date": ["2016-01-01", "2022-01-01", "2017-01-01", "2023-01-01"],
        "current_energy_rating": ["E", "C", "D", "C"],
    })
    out = features.keep_latest_certificate(df)
    assert len(out) == 2
    assert sorted(out["current_energy_rating"]) == ["C", "C"]


def test_build_features_target_and_no_leakage():
    df = pd.DataFrame({
        "current_energy_rating": ["C", "D", "g", "X"],
        "current_energy_efficiency": ["70", "60", "10", "0"],
        "property_type": ["Flat", "House", "House", "Flat"],
        "built_form": ["Mid-Terrace"] * 4,
        "construction_age_band": ["England and Wales: 1967-1975"] * 4,
        "total_floor_area": ["50", "80", "5", "60"],
        "main_fuel": ["mains gas (not community)"] * 4,
        "lodgement_date": ["2020-01-01"] * 4,
        "local_authority": ["E06000023"] * 4,
    })
    out = features.build_features(df)
    assert out["below_c"].tolist() == [0, 1, 1]  # invalid rating 'X' dropped
    assert np.isnan(out["total_floor_area"].iloc[2])  # 5 m² is implausible
    assert not set(config.LEAKAGE_COLUMNS) & set(out.columns)


def test_new_service_value_formats():
    assert features.clean_category(s("Not Recorded", "Mid-Terrace")).isna().tolist() == [True, False]
    fuel = features.group_fuel(s("Gas: mains gas", "Electricity: electricity, unspecified tariff",
                                 "To be used only when there is no heating/hot-water system or data is from a community network",
                                 "biomass (community)", "Gas: bulk LPG"))
    assert fuel.iloc[[0, 1, 3, 4]].tolist() == ["mains_gas", "electricity", "communal", "lpg"]
    assert pd.isna(fuel.iloc[2])


def test_rdsap10_flag():
    df = pd.DataFrame({
        "current_energy_rating": ["C", "D"], "property_type": ["Flat"] * 2, "built_form": ["Mid-Terrace"] * 2,
        "construction_age_band": ["1970"] * 2, "total_floor_area": ["50"] * 2,
        "main_fuel": ["mains gas (not community)"] * 2, "lodgement_date": ["2025-06-14", "2025-06-15"],
        "local_authority": ["E1"] * 2,
    })
    assert features.build_features(df)["rdsap10"].tolist() == [0.0, 1.0]


def test_compact_concat_keeps_values_and_categories():
    from housing_epc import load
    a = load.compact(pd.DataFrame({"home_id": ["x", "y"], "property_type": ["Flat", "House"]}))
    b = load.compact(pd.DataFrame({"home_id": ["z"], "property_type": ["Bungalow"]}))
    out = load.concat_compact([a, b])
    assert isinstance(out["property_type"].dtype, pd.CategoricalDtype)
    assert out["property_type"].astype(str).tolist() == ["Flat", "House", "Bungalow"]
    assert out["home_id"].tolist() == ["x", "y", "z"]


def test_latest_certificate_uses_home_id_when_present():
    df = pd.DataFrame({"home_id": ["a", "a", "b"], "uprn": ["1", "2", "1"],
                       "lodgement_date": ["2016-01-01", "2020-01-01", "2018-01-01"]})
    out = features.keep_latest_certificate(df)
    assert sorted(out["lodgement_date"]) == ["2018-01-01", "2020-01-01"]


def test_add_region_from_lookup(tmp_path):
    path = tmp_path / "lu.csv"
    path.write_text("﻿LAD24CD,LAD24NM,RGN24CD,RGN24NM\nE09000028,Southwark,E12000007,London\n", encoding="utf-8")
    lookup = features.load_region_lookup(path)
    df = features.add_region(pd.DataFrame({"local_authority": ["E09000028", "E99999999"]}), lookup)
    assert df["region"].astype(str).tolist() == ["London", "Unknown"]

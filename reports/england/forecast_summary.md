# 2030 forecast

- Homes: **3,030,188** social rented homes, each scored from its latest certificate
- Consecutive certificate pairs used to fit the model: **765,677** (later certificate from 2013, not triggered by works)
- Homes whose latest SAP score is within 3 points of the C line (66 to 71): **34%**

| Position | Expected homes below C | Share | 90% interval |
|---|---|---|---|
| Latest certificates, taken at face value | 1,028,056 | 33.9% |  |
| Estimated today (2026-10-24) | 607,325 | 20.0% | 605,544 to 610,938 |
| 1 April 2030, conditions as now | 670,500 | 22.1% | 667,759 to 675,554 |
| 1 April 2030, recent trend continues | 310,393 | 10.2% | 306,834 to 316,933 |

Getting every home to C by the deadline means about **176,753 fewer below-C homes each year** over the 3.4 years left. If the recent trend continues the fall is about **86,418 a year**.

The intervals cover uncertainty in the fitted model only. Which scenario holds is a bigger uncertainty than either interval.

## Checks

- **Stable years: fit to 2020, test 2021-22:** fitted on 184,760 pairs, tested on 123,769, with the 2020 period effect carried forward. Observed 42.4% below C, predicted 50.8%. Brier score 0.199 against 0.340 for assuming each home keeps its band (lower is better).
- **Across the 2023 shift: fit to 2022, test 2023 to RdSAP 10:** fitted on 308,529 pairs, tested on 275,413, with the 2022 period effect carried forward. Observed 28.4% below C, predicted 35.6%. Brier score 0.183 against 0.388 for assuming each home keeps its band (lower is better).
- **Recent reassessments, homes not seen in training** (5-fold, split by home): 457,148 pairs, observed 27.5%, predicted 27.5%. Brier 0.167 against 0.397 for assuming the band stays put.

## Period effects (log-odds of being below C at the next certificate)

| Period | Pairs | Effect |
|---|---|---|
| 2013 | 5,194 | +0.66 |
| 2014 | 20,017 | +0.56 |
| 2015 | 20,253 | +0.50 |
| 2016 | 19,214 | +0.22 |
| 2017 | 33,352 | +0.40 |
| 2018 | 30,909 | +0.35 |
| 2019 | 29,301 | +0.27 |
| 2020 | 26,520 | +0.36 |
| 2021 | 45,895 | +0.08 |
| 2022 | 77,874 | -0.17 |
| 2023 | 85,575 | -0.48 |
| 2024 | 100,295 | -1.00 |
| 2025 | 89,543 | -0.93 |
| RdSAP 10 | 181,735 | -0.91 |

Recent trend in the period effect: -0.267 log-odds a year (2019 to the start of RdSAP 10).

Tables: `forecast_by_archetype.csv`, `forecast_by_area.csv`, `forecast_by_region.csv`. Everything else: `forecast_2030.json`.

# 2030 forecast

- Homes: **156,256** social rented homes, each scored from its latest certificate
- Consecutive certificate pairs used to fit the model: **37,613** (later certificate from 2013, not triggered by works)
- Homes whose latest SAP score is within 3 points of the C line (66 to 71): **35%**

| Position | Expected homes below C | Share | 90% interval |
|---|---|---|---|
| Latest certificates, taken at face value | 51,413 | 32.9% |  |
| Estimated today (2026-10-24) | 33,623 | 21.5% | 32,546 to 34,608 |
| 1 April 2030, conditions as now | 37,547 | 24.0% | 36,435 to 38,388 |
| 1 April 2030, recent trend continues | 19,176 | 12.3% | 17,833 to 20,202 |

Getting every home to C by the deadline means about **9,786 fewer below-C homes each year** over the 3.4 years left. If the recent trend continues the fall is about **4,205 a year**.

The intervals cover uncertainty in the fitted model only. Which scenario holds is a bigger uncertainty than either interval.

## Checks

- **Stable years: fit to 2020, test 2021-22:** fitted on 5,098 pairs, tested on 7,009, with the 2020 period effect carried forward. Observed 41.6% below C, predicted 53.3%. Brier score 0.221 against 0.354 for assuming each home keeps its band (lower is better).
- **Across the 2023 shift: fit to 2022, test 2023 to RdSAP 10:** fitted on 12,107 pairs, tested on 15,555, with the 2022 period effect carried forward. Observed 23.9% below C, predicted 29.6%. Brier score 0.165 against 0.345 for assuming each home keeps its band (lower is better).
- **Recent reassessments, homes not seen in training** (5-fold, split by home): 25,506 pairs, observed 25.7%, predicted 25.8%. Brier 0.168 against 0.366 for assuming the band stays put.

## Period effects (log-odds of being below C at the next certificate)

| Period | Pairs | Effect |
|---|---|---|
| 2013 | 132 | +1.00 |
| 2014 | 472 | +0.92 |
| 2015 | 688 | +0.59 |
| 2016 | 660 | +0.08 |
| 2017 | 1,236 | +0.77 |
| 2018 | 733 | +0.02 |
| 2019 | 648 | -0.22 |
| 2020 | 529 | +0.22 |
| 2021 | 3,803 | +0.06 |
| 2022 | 3,206 | -0.14 |
| 2023 | 4,624 | -0.58 |
| 2024 | 3,832 | -1.16 |
| 2025 | 7,099 | -0.85 |
| RdSAP 10 | 9,951 | -0.73 |

Recent trend in the period effect: -0.237 log-odds a year (2019 to the start of RdSAP 10).

Tables: `forecast_by_archetype.csv`, `forecast_by_area.csv`. Everything else: `forecast_2030.json`.

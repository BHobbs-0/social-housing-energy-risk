# Which social homes will miss EPC C? Predicting and explaining energy efficiency risk in England's social housing

Social rented homes in England must reach **EPC band C by 1 April 2030**. Landlords need to know which homes are most at risk, but many do not hold up-to-date survey data for every property.

This project uses **136,379 Energy Performance Certificates for social rented homes** in five local authorities (Manchester, Sheffield, Cornwall, Newcastle upon Tyne and Southwark) to answer three questions:

1. **Prediction:** using only what a typical asset register holds (property type, built form, build year, floor area, fuel), how well can we identify homes likely to be below band C? This is the **core** model.
2. **Explanation:** what actually drives the risk, and how much better do we do when we also know about walls, roofs, insulation and heating? This is the **extended** model.
3. **Forecast:** using homes that have been assessed more than once, how many are likely to be below C today, and on 1 April 2030? See [Forecasting the position in 2030](#forecasting-the-position-in-2030).

## Headline results

**31% of the social homes in the data (42,646) were rated below band C on their latest certificate (lodged between 2015 and 2026).**

### How well does it predict?

![Model performance by feature set](reports/figures/01_feature_set_comparison.png)

*ROC AUC measures how well a model ranks homes by risk: 0.5 is no better than chance and 1.0 is a perfect ranking. Scores are for homes the model did not see during training.*

- **Asset-register data alone gets you a useful ranking.** The core model reaches a ROC AUC of **0.82**.
- **Survey data adds a lot.** The extended model lifts this to **0.90**, roughly halving the gap to a perfect ranking. This quantifies the value of a stock condition survey.
- **It transfers to areas the model has never seen.** Training on four areas and testing on the fifth gives **0.76** (core) and **0.87** (extended) across all five areas. These are the realistic figures for a landlord applying the model to new stock.
- **Gradient boosting beats logistic regression in every comparison** (0.82 vs 0.76 core, 0.90 vs 0.86 extended). Combinations of features matter, which a linear model cannot fully capture.

![Calibration](reports/figures/02_calibration_extended.png)

The predicted probabilities are also **well calibrated**: they track observed rates closely, so a predicted 40% risk really does mean roughly 40 in 100 such homes are below C. That makes the scores usable for planning, not just ranking.

### What drives the risk?

![Permutation importance, extended model](reports/figures/03_importance_extended.png)

Wall insulation is the single strongest signal, followed by property type, loft insulation and build year. The charts below show the model's predicted risk for each group, with other factors held constant.

| Wall insulation | Main fuel | Build year |
|---|---|---|
| ![Wall insulation](reports/figures/04_pdp_extended_wall_insulation.png) | ![Main fuel](reports/figures/04_pdp_core_fuel_group.png) | ![Build year](reports/figures/04_pdp_core_construction_year.png) |

- **Uninsulated walls:** 64% of homes with uninsulated walls are below C, against 23% with insulated walls. Once other factors are held constant, the model's predicted risk is 69% against 26%.
- **Off-gas homes:** homes heated by oil (61%), electricity (53%) or communal systems (51%) have around double the predicted risk of mains gas homes (27%), other factors held constant.
- **Build era:** predicted risk falls steadily from about 50% for pre-1920 homes to almost zero for homes built after 2003, other factors held constant.
- **Houses and bungalows** (39% and 42% below C) are far more likely to be below C than flats (22%), which lose less heat through shared walls, floors and ceilings.

**Highest-risk archetypes with at least 150 homes in the test set:**

| Property type | Build era | Main fuel | Homes | Actual below C | Predicted |
|---|---|---|---|---|---|
| House | 1983-2002 | Electricity | 180 | 79% | 77% |
| House | 1965-82 | Communal | 188 | 62% | 58% |
| Flat | 1945-64 | Electricity | 373 | 58% | 60% |
| House | pre-1919 | Mains gas | 1,584 | 51% | 52% |
| Flat | 1965-82 | Electricity | 830 | 48% | 48% |
| House | 1919-44 | Mains gas | 4,156 | 47% | 46% |

The interwar gas-heated house is not the riskiest archetype, but it is by far the **largest** at-risk group: **9,305 of them are below C** across the full dataset, more than any other archetype. That makes it a natural target for a retrofit programme.

### A methodology change shows up in the data

![Share below C over time](reports/figures/00_below_c_over_time.png)

The share of assessed homes below C fell from around 40% in 2015 to 20% in 2024, then rose again after the assessment method changed to **RdSAP 10 on 15 June 2025**. Holding everything else constant, the model estimates that being assessed under RdSAP 10 adds about **3.4 percentage points** to a home's risk of being below C. Comparing ratings across this change without adjusting for it could make a landlord's stock look worse than it is, so the models include the assessment year and an RdSAP 10 flag as controls.

This chart should not be read as a trend for the whole stock. Each quarter shows the homes that happened to be assessed then (mostly homes being re-let), not a consistent picture of all homes, so extending the line forward would be misleading.

Full numbers are in [`reports/summary.md`](reports/summary.md) and [`reports/metrics.json`](reports/metrics.json).

## Forecasting the position in 2030

The models above are a snapshot. To look forward, the forecast uses homes that have been assessed more than once: **208,473 certificates for 156,256 homes**, of which 41,521 have two or more, giving 37,613 usable pairs of consecutive certificates on the same home.

### What repeat certificates show

![Homes just below C at the next certificate](reports/figures/07_near_line_by_year.png)

- **A single certificate is a noisy reading.** Low scores tend to rise and high scores tend to fall at the next certificate, whatever the gap: partly assessor variation, partly "before" certificates lodged ahead of retrofit work. **35% of homes have a latest SAP score within 3 points of the C line** (66 to 71), where one reassessment can tip them either way.
- **When a home is reassessed matters more than how long since its last certificate.** For homes rated just below C, the share still below C at the next certificate was between about 40% and 70% up to 2022, then fell to 24% for certificates lodged in 2024. It is 35% under RdSAP 10, which scores the same home a little lower.
- **The national picture shows the same shift.** The [English Housing Survey](https://www.gov.uk/government/statistics/chapters-for-english-housing-survey-2024-to-2025-headline-findings-on-housing-quality-and-energy-efficiency/introduction-and-key-findings) found band D fell from 43% to 35% of all homes in England between 2022 and 2024.

### The forecast

![Forecast to 2030](reports/figures/08_forecast_2030.png)

| Position | Expected homes below C | Share |
|---|---|---|
| Latest certificates, taken at face value | 51,413 | 32.9% |
| Estimated today (October 2026) | 33,623 | 21.5% (20.8% to 22.1%) |
| 1 April 2030, conditions as now | 37,547 | 24.0% (23.3% to 24.6%) |
| 1 April 2030, recent trend continues | 19,176 | 12.3% (11.4% to 12.9%) |

*Ranges are 90% intervals from refitting the model on resampled homes. They cover uncertainty in the fit, not in which scenario holds.*

- **Latest certificates overstate the problem.** Many are a decade old. Scored as if reassessed today, about one in five homes is likely to be below C, not one in three.
- **On current evidence, the deadline will be missed.** Clearing every below-C home by April 2030 means about **9,800 fewer below-C homes a year**. Even if the recent improvement trend continues, the fall is about **4,200 a year**, leaving around 12% of homes below C.
- **The pace matches national data.** The English Housing Survey shows social rented homes below C falling by about 2 percentage points a year ([30% in 2022, 26% in 2024](https://www.gov.uk/government/statistics/chapters-for-english-housing-survey-2024-to-2025-headline-findings-on-housing-quality-and-energy-efficiency/chapter-2-energy-efficiency)), close to the "trend continues" scenario.
- **The biggest group is unchanged.** Interwar gas-heated houses are still the largest at-risk archetype: about 7,000 likely to be below C today, and about 4,100 in 2030 even if the trend continues.

| Area | Homes | Below C today | 2030, trend continues |
|---|---|---|---|
| Manchester | 51,989 | 18.2% | 9.9% |
| Southwark | 19,175 | 21.2% | 11.6% |
| Sheffield | 42,904 | 22.5% | 13.1% |
| Cornwall | 21,505 | 23.0% | 14.0% |
| Newcastle upon Tyne | 20,683 | 26.6% | 15.5% |

Full numbers are in [`reports/forecast_summary.md`](reports/forecast_summary.md), with tables by area and archetype alongside.

### How it works

For each pair of consecutive certificates, a logistic regression predicts whether the later one is below C from the earlier certificate's **SAP score** (a D at 68 is a near miss, a D at 56 is not), the gap between them, whether the earlier certificate was lodged because of works, property type, build era, fuel, area, and the **period** the later certificate was lodged in (each year, with RdSAP 10 as a period of its own). Pairs whose later certificate was itself lodged because of works (ECO, grant schemes) are left out, so the model describes a routine reassessment.

Every home is then scored from its latest certificate as if reassessed under current conditions. "Conditions as now" holds the period effect at its RdSAP 10 level until April 2030; "recent trend continues" lets it keep falling at its 2019 to 2025 pace.

A first version assumed homes improve steadily over time (a two-state Markov chain). It failed its backtest because, as above, the time between certificates barely matters while the calendar year matters a lot.

![Forecast calibration](reports/figures/09_forecast_calibration.png)

**Checks.** On recent reassessments of homes the model never saw, it predicts 25.8% below C against 25.7% observed, and its Brier score is 0.168 against 0.366 for assuming each home keeps its band. Fitted only on data to 2022, it would have predicted 29.6% below C for 2023 to mid-2025 against 23.9% observed: it could not have seen the 2023 shift coming, which is why 2030 is shown as scenarios rather than one number.

### Caveats

- **The "today" figure may be optimistic.** Landlords often lodge a routine-looking certificate after works, so recently reassessed homes may be better than the stock as a whole. The model shows a sign of this: at the same starting score, a home reassessed soon after a low rating is less likely to still be below C than one reassessed years later, which is also why "conditions as now" sits a little above "today". The 21.5% estimate is below the national 26% for 2024, though these five areas may simply be ahead.
- **Real improvement and assessment practice cannot be separated.** The period effects capture both.
- **The trend scenario is an extrapolation.** Nothing in the data says the 2019 to 2025 pace will hold.
- **Compliance is not just the current rating.** Under the [confirmed policy](https://www.elmhurstenergy.co.uk/blog/2026/02/03/social-rented-homes-to-reach-epc-c-by-1st-april-2030/), homes can comply through new fabric, heating or smart-readiness metrics, and a £10,000 spend cap allows deferral. This forecast is about the current EPC rating only.

## What the model is for (and what it isn't)

### How a landlord could use it

- **Find at-risk homes without an up-to-date rating.** Score every home from the asset register, including homes with no EPC or an old one, and survey the highest-risk homes first. The core model is the realistic choice here because it only needs data landlords already hold: it scores 0.82 on held-out homes and 0.76 in areas it has never seen.
- **Size the problem today.** Because the predicted probabilities are well calibrated, adding them up gives an expected count. If 1,000 unrated homes have an average predicted risk of 35%, roughly 350 of them are likely to be below C. That gives a planning and budgeting figure without surveying every home.
- **Target retrofit spending.** Wall insulation, loft insulation and off-gas heating are the strongest drivers of risk, which points to the interventions most likely to move homes over the line.
- **Decide whether a stock condition survey is worth it.** The jump from 0.82 (core) to 0.90 (extended) puts a number on how much better a landlord can target work once they have survey data.

### What it does not do

**The classifier does not forecast.** It estimates how likely a home is to be below C at the time it was assessed, based on its characteristics, and knows nothing about how quickly homes are being improved. Projecting forward needs ratings measured on the same homes over time, which is what the [2030 forecast](#forecasting-the-position-in-2030) uses.

## Data

**Source:** [Get energy performance of buildings data](https://get-energy-performance-data.communities.gov.uk/) (MHCLG). This GOV.UK service replaced the old Open Data Communities EPC portal in 2026.

**Areas used:** Manchester (46,296 homes), Sheffield (38,913), Cornwall (19,863), Newcastle upon Tyne (17,800) and Southwark (13,507). Together they mix large northern cities, an inner London borough and a rural, largely off-gas county. Certificates lodged from January 2015 to September 2026, latest certificate per home.

**Reproducing it**

1. Sign in with GOV.UK One Login on the service above and download domestic EPC certificate data.
2. The downloads are large (over 1GB for a couple of local authorities), so shrink them first:
   ```bash
   python scripts/extract_subset.py path/to/download.csv          # default areas
   python scripts/extract_subset.py path/to/download.zip --areas E08000003 E09000028
   ```
   This keeps social rented homes since 2015, keeps the latest certificate per home, drops addresses and writes a small `epc_social_subset.csv.gz`.
3. Put that file (or the raw CSVs) anywhere under `data/raw/` and run the pipeline.
4. For the 2030 forecast, run the same script with `--history`. This keeps every social rented certificate for each home, plus the SAP score and the reason the certificate was lodged, and links them with an anonymous `home_id` (a hash with a random salt that is never saved, so it cannot be traced back to an address):
   ```bash
   python scripts/extract_subset.py path/to/download.csv --history --areas E08000003 E08000039 E06000052 E08000021 E09000028
   ```
   Sheffield's code is E08000039 in current downloads (it was E08000019 before a 2019 boundary change).

Raw data is git-ignored and never committed. Check the service's licensing guidance before redistributing any derived data.

**If a column name has changed:** the pipeline normalises column names automatically. If it reports a missing column, add the new name to `ALIASES` in `src/housing_epc/columns.py`.

## Method

| Step | What happens | Where |
|---|---|---|
| Load | Stream CSVs in chunks, keep social rented tenure and certificates lodged since 2015 | `load.py` |
| Deduplicate | Keep the latest certificate per home (UPRN, falling back to address and postcode) | `features.py` |
| Features | Parse age bands, fuel, wall type and insulation, roof and loft depth, heating system, glazing | `features.py` |
| Target | `below_c` = current rating D, E, F or G | `config.py` |
| Models | Logistic regression baseline and histogram gradient boosting, on **core** and **extended** feature sets | `model.py` |
| Validation | Stratified 80/20 held-out test set, plus leave-one-local-authority-out cross-validation | `model.py` |
| Explanation | Permutation importance, partial dependence, SHAP values (optional), archetype and area tables | `explain.py` |
| Forecast | Pairs of consecutive certificates per home, logistic regression with period effects, scenarios to April 2030 | `forecast.py` |

### Feature sets

- **Core (asset register):** property type, built form, build year, floor area, main fuel, mains gas availability
- **Extended (stock survey):** core plus wall type, wall insulation, roof type, loft insulation depth, heating system, multiple glazing percentage
- **Controls (both sets):** year the EPC was lodged and whether it was assessed under RdSAP 10. To score a home as if assessed today, set these to the current date.

### Avoiding leakage

An EPC rating is calculated from the certificate's own inputs. Feeding the model fields such as the SAP score, running costs, CO₂ or the per-element efficiency ratings would let it reverse-engineer the formula and report a meaningless near-perfect score. These are listed in `LEAKAGE_COLUMNS` and excluded, and a test checks they never reach the feature table.

### Checking the controls are not a back door

Landlords often commission EPCs in large batches, so assessment dates could act as a proxy for area and inflate the scores. To test this, the models were rerun without the date controls. The controls improve performance on unseen areas too (core: 0.740 without, 0.757 with), so they capture genuine time effects rather than identifying the area.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[shap,dev]"

python -m housing_epc.pipeline                    # full run on data/raw (about 2 minutes for 136k homes)
python -m housing_epc.pipeline --sample 200000    # random subsample for very large inputs
python -m housing_epc.pipeline --theme light      # light charts instead of the default dark
python -m housing_epc.forecast --history data/history   # 2030 forecast from a --history extract
pytest                                            # unit tests
```

Charts default to a dark theme that matches GitHub's dark mode.

**No data yet?** Generate a synthetic file to check everything runs (its numbers are invented, never publish them):

```bash
python scripts/make_synthetic_data.py --rows 30000
python -m housing_epc.pipeline --raw data/synthetic

python scripts/make_synthetic_data.py --history --rows 20000
python scripts/extract_subset.py data/synthetic_history --history --all-areas --out data/synthetic_history.csv.gz
python -m housing_epc.forecast --history data/synthetic_history.csv.gz --reports data/synthetic_reports
```

## Outputs

| File | Contents |
|---|---|
| `reports/summary.md` | Plain-English results summary |
| `reports/metrics.json` | All metrics, including each leave-one-area-out fold |
| `reports/figures/00_below_c_over_time.png` | Share below C by quarter, with RdSAP 10 marked |
| `reports/figures/01_feature_set_comparison.png` | Core vs extended, logistic vs boosting |
| `reports/figures/02_calibration_extended.png` | Are predicted probabilities trustworthy? |
| `reports/figures/03_importance_*.png` | Permutation importance |
| `reports/figures/04_pdp_*.png` | How risk changes with build year, fuel, insulation and more |
| `reports/figures/05_*`, `06_*` | SHAP importance and dependence (generated when `shap` is installed) |
| `reports/archetype_risk.csv` | Risk by property type, build era and fuel |
| `reports/local_authority_gap.csv` | Observed vs expected below-C share by area |
| `reports/forecast_summary.md` | 2030 forecast: scenarios, checks and period effects |
| `reports/forecast_2030.json` | All forecast numbers, including backtests |
| `reports/forecast_by_area.csv`, `forecast_by_archetype.csv` | Expected homes below C today and in 2030 |
| `reports/figures/07_*`, `08_*`, `09_*` | Repeat certificates by year, forecast scenarios, forecast calibration |

## Project structure

```
├── data/
│   ├── raw/                    # EPC downloads go here (git-ignored)
│   ├── history/                # --history extracts for the forecast (git-ignored)
│   └── processed/              # cleaned modelling table (git-ignored)
├── notebooks/
│   └── 01_explore.ipynb        # exploratory analysis of the cleaned table
├── reports/                    # metrics, tables and figures
├── scripts/
│   ├── extract_subset.py       # shrink large downloads to an address-free subset
│   └── make_synthetic_data.py  # synthetic data for testing
├── src/housing_epc/
│   ├── config.py               # settings, feature sets, leakage list
│   ├── columns.py              # column-name normalisation
│   ├── load.py                 # chunked loading and tenure filter
│   ├── features.py             # cleaning and feature engineering
│   ├── model.py                # training and validation
│   ├── explain.py              # importance, partial dependence, SHAP, tables
│   ├── forecast.py             # 2030 forecast from repeat certificates
│   └── pipeline.py             # end-to-end command line entry point
└── tests/
```

## Limitations

- **Five areas, not all of England:** results describe these five local authorities. They span very different stock, and the leave-one-area-out results suggest the model transfers reasonably, but national figures would need the full dataset.
- **Coverage:** only homes with a certificate are included. Most social homes have one because an EPC is needed to let a property, but recently let or recently surveyed homes may be over-represented.
- **Certificates age:** a certificate describes the home when it was assessed. Improvements made since then are not captured.
- **"Assumed" values:** where an assessor cannot see insulation, RdSAP assumes it from the build era. Some wall and roof insulation values are therefore defaults rather than observations.
- **Explanation is not causation:** importance and partial dependence describe what the model learned. They point to where to look, not to proof that changing one feature will change a rating.

## Ideas for next steps

- Add a "planned retrofit programme" scenario to the forecast, for example a landlord's own works list scored home by home.
- Scale up to the full national dataset and compare regions.
- Join deprivation data (Index of Multiple Deprivation via postcode to LSOA): are the highest-risk homes concentrated in the most deprived areas?
- Use the recommendations file to estimate the cost of bringing each archetype up to band C.
- Predict the SAP score directly (regression) to rank homes by how far they are from C.
- Link to the English Housing Survey to explore whether low energy efficiency is associated with damp and mould.

## Licence

Code: MIT (see `LICENSE`). EPC data: subject to MHCLG's licence terms.

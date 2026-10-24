# Which social homes will miss EPC C? Predicting, explaining and forecasting energy efficiency risk in England's social housing

Social rented homes in England must reach **EPC band C by 1 April 2030**. Landlords need to know which homes are most at risk, but many do not hold up-to-date survey data for every property.

This project uses Energy Performance Certificates for **3 million social rented homes across England** to answer three questions:

1. **Prediction:** using only what a typical asset register holds (property type, built form, build year, floor area, fuel), how well can we identify homes likely to be below band C? This is the **core** model.
2. **Explanation:** what actually drives the risk, and how much better do we do when we also know about walls, roofs, insulation and heating? This is the **extended** model.
3. **Forecast:** using homes that have been assessed more than once, how many are likely to be below C today, and on 1 April 2030?

The method was built and tested on five contrasting local authorities first, then run unchanged on the whole country. See [Five areas first, then England](#five-areas-first-then-england) for why, and how the results compare.

## At a glance

- **Prediction:** asset-register data alone ranks homes by risk with a ROC AUC of **0.80** (1.0 is a perfect ranking). Adding stock survey data lifts this to **0.90**. Both hold up in regions the model never saw.
- **Explanation:** uninsulated walls, off-gas heating, older homes and houses rather than flats drive the risk. Wall insulation is the strongest single signal.
- **Today:** scored as if reassessed now, about **one in five** social homes is likely to be below C, not the one in three their latest certificates suggest. That is about **600,000 homes**.
- **2030:** on current evidence the deadline will be missed. Even if the recent improvement trend continues, about **310,000 homes (10%)** would still be below C in April 2030.

## Predicting and explaining the risk

**33% of English social homes with a certificate lodged since 2015 (871,418 of 2,668,893) were rated below band C on their latest certificate.**

### How well does it predict?

![Model performance by feature set](reports/england/figures/01_feature_set_comparison.png)

*ROC AUC measures how well a model ranks homes by risk: 0.5 is no better than chance and 1.0 is a perfect ranking. Scores are for homes the model did not see during training.*

- **Asset-register data alone gets you a useful ranking.** The core model reaches a ROC AUC of **0.80**.
- **Survey data adds a lot.** The extended model lifts this to **0.90**, roughly halving the gap to a perfect ranking. This quantifies the value of a stock condition survey.
- **It transfers to regions the model has never seen.** Training on eight regions and testing on the ninth gives **0.79** (core) and **0.89** (extended) on average, and at least 0.77 and 0.85 in every region. These are the realistic figures for a landlord applying the model to its own stock.
- **Gradient boosting beats logistic regression in every comparison** (0.80 vs 0.77 core, 0.90 vs 0.86 extended). Combinations of features matter, which a linear model cannot fully capture.

![Calibration](reports/england/figures/02_calibration_extended.png)

The predicted probabilities are also **well calibrated**: they track observed rates closely, so a predicted 40% risk really does mean roughly 40 in 100 such homes are below C. That makes the scores usable for planning, not just ranking.

### What drives the risk?

![Permutation importance, extended model](reports/england/figures/03_importance_extended.png)

Wall insulation is the single strongest signal, followed by property type, loft insulation and build year. The charts below show the model's predicted risk for each group, with other factors held constant.

| Wall insulation | Main fuel | Build year |
|---|---|---|
| ![Wall insulation](reports/england/figures/04_pdp_extended_wall_insulation.png) | ![Main fuel](reports/england/figures/04_pdp_core_fuel_group.png) | ![Build year](reports/england/figures/04_pdp_core_construction_year.png) |

- **Uninsulated walls:** 65% of homes with uninsulated walls are below C, against 23% with insulated walls. Once other factors are held constant, the model's predicted risk is 67% against 23%.
- **Off-gas homes:** homes heated by oil (70%) or electricity (61%) have about double the predicted risk of mains gas homes (29%), and communal heating (47%) is not far behind, other factors held constant.
- **Build era:** predicted risk falls steadily from almost 60% for pre-1920 homes to about 2% for homes built after 2003, other factors held constant.
- **Bungalows and houses** (46% and 38% below C) are far more likely to be below C than flats (24%), which lose less heat through shared walls, floors and ceilings.

**Highest-risk archetypes with at least 500 homes in the test set (a fifth of all homes), with the two largest groups for comparison:**

| Property type | Build era | Main fuel | Homes | Actual below C | Predicted |
|---|---|---|---|---|---|
| House | 1919-44 | Oil | 608 | 92% | 90% |
| House | 1945-64 | Oil | 820 | 86% | 85% |
| House | pre-1919 | Electricity | 738 | 78% | 76% |
| Flat | pre-1919 | Electricity | 2,713 | 72% | 71% |
| Bungalow | pre-1919 | Mains gas | 682 | 71% | 70% |
| Maisonette | 1965-82 | Electricity | 557 | 73% | 70% |
| House | 1919-44 | Mains gas | 48,545 | 49% | 49% |
| House | 1945-64 | Mains gas | 65,482 | 37% | 37% |

Neither of the last two is among the riskiest archetypes, but by number they are the **largest** at-risk groups. Across all homes, **122,517 post-war (1945-64) and 118,501 interwar (1919-44) gas-heated houses are below C** on their latest certificate, more than a quarter of all homes below C between them. Interwar houses are more likely to be below C; there are simply more post-war ones. That makes both natural targets for retrofit programmes.

### A methodology change shows up in the data

![Share below C over time](reports/england/figures/00_below_c_over_time.png)

The share of assessed homes below C fell from 45% in 2015 to 25% in 2024, then edged up after the assessment method changed to **RdSAP 10 on 15 June 2025**. Holding everything else constant, the model estimates that being assessed under RdSAP 10 adds about **1.7 percentage points** to a home's risk of being below C. Comparing ratings across this change without adjusting for it could make a landlord's stock look worse than it is, so the models include the assessment year and an RdSAP 10 flag as controls.

This chart should not be read as a trend for the whole stock. Each quarter shows the homes that happened to be assessed then (mostly homes being re-let), not a consistent picture of all homes, so extending the line forward would be misleading. The [forecast](#forecasting-the-position-in-2030) looks forward properly, by following the same homes from one certificate to the next.

Full numbers are in [`reports/england/summary.md`](reports/england/summary.md) and [`reports/england/metrics.json`](reports/england/metrics.json).

## Forecasting the position in 2030

The models above are a snapshot. To look forward, the forecast uses homes that have been assessed more than once: **4.1 million certificates lodged since 2012 for 3.0 million homes**, of which 846,282 have two or more, giving 765,677 usable pairs of consecutive certificates on the same home.

### What repeat certificates show

![Homes just below C at the next certificate](reports/england/figures/07_near_line_by_year.png)

- **A single certificate is a noisy reading.** Low scores tend to rise and high scores tend to fall at the next certificate, whatever the gap: partly assessor variation, partly "before" certificates lodged ahead of retrofit work. **About a third of homes have a latest SAP score within 3 points of the C line** (66 to 71), where one reassessment can tip them either way.
- **When a home is reassessed matters more than how long since its last certificate.** For homes rated just below C, the share still below C at the next certificate was between about 50% and 70% every year to 2022, then fell to 32% for certificates lodged in 2024. It is 33% under RdSAP 10.
- **The national survey shows the same shift.** The [English Housing Survey](https://www.gov.uk/government/statistics/chapters-for-english-housing-survey-2024-to-2025-headline-findings-on-housing-quality-and-energy-efficiency/introduction-and-key-findings) found band D fell from 43% to 35% of all homes in England between 2022 and 2024.

### The forecast

![Forecast to 2030](reports/england/figures/08_forecast_2030.png)

| Position | Expected homes below C | Share |
|---|---|---|
| Latest certificates, taken at face value | 1,028,056 | 33.9% |
| Estimated today (October 2026) | 607,325 | 20.0% (20.0% to 20.2%) |
| 1 April 2030, conditions as now | 670,500 | 22.1% (22.0% to 22.3%) |
| 1 April 2030, recent trend continues | 310,393 | 10.2% (10.1% to 10.5%) |

*Ranges are 90% intervals from refitting the model on resampled homes. They cover uncertainty in the fit, not in which scenario holds. The latest-certificate share here (34%) is a little above the 33% earlier because it also counts homes whose latest certificate is from 2012 to 2014.*

- **Latest certificates overstate the problem.** Many are a decade old. Scored as if reassessed today, about one in five homes is likely to be below C, not one in three.
- **On current evidence, the deadline will be missed.** Clearing every below-C home by April 2030 means about **177,000 fewer below-C homes a year**. Even if the recent improvement trend continues, the fall is about **86,000 a year**, half the pace needed, leaving around 310,000 homes below C.
- **It is consistent with the national survey.** The English Housing Survey found social rented homes below C falling by about 2 percentage points a year ([30% in 2022, 26% in 2024](https://www.gov.uk/government/statistics/chapters-for-english-housing-survey-2024-to-2025-headline-findings-on-housing-quality-and-energy-efficiency/chapter-2-energy-efficiency)), which would put 2026 at roughly 22%. This estimate of 20% is slightly lower, in line with the first caveat below.
- **The same two groups dominate.** Scored as if reassessed today, interwar gas-heated houses are the largest group likely to be below C (about 84,000), just ahead of post-war ones (about 79,000). Even if the trend continues, about 43,000 and 38,000 would still be below C in 2030.

Full numbers are in [`reports/england/forecast_summary.md`](reports/england/forecast_summary.md), with tables by region, local authority and archetype alongside.

### How it works

For each pair of consecutive certificates, a logistic regression predicts whether the later one is below C from the earlier certificate's **SAP score** (a D at 68 is a near miss, a D at 56 is not), the gap between them, whether the earlier certificate was lodged because of works, property type, build era, fuel, local authority, and the **period** the later certificate was lodged in (each year, with RdSAP 10 as a period of its own). Pairs whose later certificate was itself lodged because of works (ECO, grant schemes) are left out, so the model describes a routine reassessment.

Every home is then scored from its latest certificate as if reassessed under current conditions. "Conditions as now" holds the period effect at its RdSAP 10 level until April 2030; "recent trend continues" lets it keep falling at its 2019 to 2025 pace.

A first version assumed homes improve steadily over time (a two-state Markov chain). It failed its backtest because, as above, the time between certificates barely matters while the calendar year matters a lot.

![Forecast calibration](reports/england/figures/09_forecast_calibration.png)

**Checks.** On 457,000 recent reassessments of homes the model never saw, it predicts 27.5% below C against 27.5% observed. Its Brier score (the average squared error of its probabilities, lower is better) is 0.167, against 0.397 for assuming each home keeps its band. Fitted only on data to 2022, it would have predicted 35.6% below C for 2023 to mid-2025 against 28.4% observed: it could not have seen the 2023 shift coming, which is why 2030 is shown as scenarios rather than one number.

### Caveats

- **The "today" figure may be optimistic.** Landlords often lodge a routine-looking certificate after works, so recently reassessed homes may be better than the stock as a whole. The model shows a sign of this: at the same starting score, a home reassessed soon after a low rating is less likely to still be below C than one reassessed years later, which is also why "conditions as now" sits a little above "today".
- **Real improvement and assessment practice cannot be separated.** The period effects capture both.
- **The trend scenario is an extrapolation.** Nothing in the data says the 2019 to 2025 pace will hold.
- **Compliance is not just the current rating.** Under the [confirmed policy](https://www.elmhurstenergy.co.uk/blog/2026/02/03/social-rented-homes-to-reach-epc-c-by-1st-april-2030/), homes can comply through new fabric, heating or smart-readiness metrics, and a £10,000 spend cap allows deferral. This forecast is about the current EPC rating only.

## Regions

![Share below C by region](reports/england/figures/10_region_gap.png)

Comparing each region's share below C with what its housing stock alone would predict shows where homes do better or worse than their age, type and fuel suggest. **London's social homes are more likely to be below C than its stock mix predicts (37% against 31%)**, as are the West Midlands and Yorkshire and The Humber. The North West does best against expectation (28% against 32%). These shares are from each home's latest certificate in the held-out test set, so, like any latest-certificate figure, they overstate the position today.

![Forecast by region](reports/england/figures/11_forecast_by_region.png)

Scored as if reassessed today, **Yorkshire and The Humber (26%) and London (23%)** have the highest shares of homes likely to be below C, and the **North West and South East (17%)** the lowest. If the recent trend continues, every region roughly halves its share by 2030, but none reaches zero.

## Five areas first, then England

The method was developed on five contrasting local authorities: **Manchester, Sheffield and Newcastle upon Tyne** (large northern cities), **Southwark** (an inner London borough) and **Cornwall** (rural and largely off-gas). With 136,000 homes, it was quick to iterate on cleaning, features and validation, and training on four authorities and testing on the fifth was a demanding check that the model transfers to stock it has never seen. The same code then ran unchanged on all of England, which is what the results above report.

The two agree closely, which suggests the findings are not an artefact of the areas chosen:

| | Five areas | England |
|---|---|---|
| Homes (latest certificate since 2015) | 136,379 | 2,668,893 |
| Below C on latest certificate | 31% | 33% |
| Core model ROC AUC, held-out homes | 0.82 | 0.80 |
| Extended model ROC AUC, held-out homes | 0.90 | 0.90 |
| Core / extended, area never seen in training | 0.76 / 0.87 | 0.79 / 0.89 |
| Likely below C today | 21.5% | 20.0% |
| 2030, recent trend continues | 12.3% | 10.2% |

*For the five areas, "area never seen" means leaving out one local authority at a time; for England, one region at a time.*

The drivers are the same in both, and so is the 2023 shift in ratings. One figure differs: RdSAP 10 adds about 3 percentage points to the risk of being below C in the five areas, but under 2 nationally.

Full five-area results are in [`reports/summary.md`](reports/summary.md) and [`reports/forecast_summary.md`](reports/forecast_summary.md). Results for every English local authority are in [`reports/england/forecast_by_area.csv`](reports/england/forecast_by_area.csv).

## What the project is for (and what it isn't)

### How a landlord could use it

- **Find at-risk homes without an up-to-date rating.** Score every home from the asset register, including homes with no EPC or an old one, and survey the highest-risk homes first. The core model is the realistic choice here because it only needs data landlords already hold: it scores 0.80 on held-out homes and 0.79 in regions it has never seen.
- **Size the problem today.** For homes with no certificate, the classifier's probabilities are well calibrated, so adding them up gives an expected count: if 1,000 unrated homes have an average predicted risk of 35%, roughly 350 are likely to be below C. For homes with an old certificate, the forecast's "estimated today" is the better figure, because it allows for improvements since the home was last assessed. That is why it finds one in five homes below C rather than one in three.
- **Prioritise reassessment.** About a third of homes have a latest SAP score within 3 points of the C line. A fresh assessment, or small works such as topping up loft insulation, could move many of them over it, so they are a quick way to cut the number below C.
- **Target retrofit spending.** Wall insulation, loft insulation and off-gas heating are the strongest drivers of risk, which points to the interventions most likely to move homes over the line.
- **Decide whether a stock condition survey is worth it.** The jump from 0.80 (core) to 0.90 (extended) puts a number on how much better a landlord can target work once they have survey data.
- **Plan the programme to 2030.** The forecast gives an expected number of homes below C today and in April 2030 for each region, local authority and archetype, under "conditions as now" and "recent trend continues", which brackets the size of the job.

### What it does not do

- **It does not decide compliance.** The confirmed policy allows other routes to compliance and a spend cap (see [Caveats](#caveats)). This project models the current EPC rating only.
- **It does not predict 2030.** The forecast shows two scenarios. The data cannot say whether the recent pace of improvement will hold, slow or speed up.
- **It cannot separate real improvement from changes in assessment.** Ratings shifted sharply around 2023; retrofit programmes, changes in assessment practice and the move to RdSAP 10 all show up together.
- **It is for ranking and counting, not labelling one home.** Two assessments of the same home can differ by several SAP points, so for a home near the line a single prediction is close to a coin toss. The scores are most useful added up across a stock, or used to decide which homes to look at first.

## Data

**Source:** [Get energy performance of buildings data](https://get-energy-performance-data.communities.gov.uk/) (MHCLG). This GOV.UK service replaced the old Open Data Communities EPC portal in 2026.

**Coverage:** all 296 English local authorities. The classifier uses certificates lodged from January 2015 to September 2026, latest certificate per home; the forecast also uses earlier certificates, back to 2012. Welsh authorities are in the download but left out, because Wales sets its own standard for social housing.

**Development areas:** Manchester (46,296 homes), Sheffield (38,913), Cornwall (19,863), Newcastle upon Tyne (17,800) and Southwark (13,507), latest certificate since 2015.

**Reproducing it**

1. Sign in with GOV.UK One Login on the service above and download domestic EPC certificate data.
2. The downloads are large (over 1GB for a couple of local authorities), so shrink them first:
   ```bash
   python scripts/extract_subset.py path/to/download.csv          # eight large areas by default (DEFAULT_AREAS)
   python scripts/extract_subset.py path/to/download.zip --areas E08000003 E09000028
   ```
   This keeps social rented homes since 2015, keeps the latest certificate per home, drops addresses and writes a small `epc_social_subset.csv.gz`.
3. Put that file (or the raw CSVs) anywhere under `data/raw/` and run the pipeline.
4. For the 2030 forecast, run the same script with `--history`. This keeps every social rented certificate for each home, plus the SAP score and the reason the certificate was lodged, and links them with an anonymous `home_id` (a hash with a random salt that is never saved, so it cannot be traced back to an address):
   ```bash
   python scripts/extract_subset.py path/to/download.csv --history --areas E08000003 E08000039 E06000052 E08000021 E09000028
   ```
   Put the output in `data/history/`. Current EPC downloads use E08000039 for Sheffield (ONS lookups use E08000019).
5. For England, run it on the national download (an 8GB zip) with `--all-areas`. In history mode it writes as it reads, so it needs little memory, and the output (about 190MB) feeds both the classifier and the forecast:
   ```bash
   python scripts/extract_subset.py all-domestic-csv.zip --history --all-areas --out epc_social_history_england.csv.gz
   ```
   Put the output in `data/england/`. For results by region, save the ONS [Local Authority District to Region lookup](https://www.data.gov.uk/dataset/3fbe9109-f329-4c4a-85cb-3b78ea4fbce3/local-authority-district-to-region-december-2024-lookup-in-en) as `data/lookups/lad_to_region.csv` (a copy is included; Open Government Licence).

Raw data is git-ignored and never committed. Check the service's licensing guidance before redistributing any derived data.

**If a column name has changed:** the pipeline normalises column names automatically. If it reports a missing column, add the new name to `ALIASES` in `src/housing_epc/columns.py`.

## Method

| Step | What happens | Where |
|---|---|---|
| Load | Stream CSVs in chunks, keep social rented tenure and certificates lodged since 2015 | `load.py` |
| Deduplicate | Keep the latest certificate per home (the anonymous `home_id` of a `--history` extract, or UPRN, falling back to address and postcode) | `features.py` |
| Features | Parse age bands, fuel, wall type and insulation, roof and loft depth, heating system, glazing | `features.py` |
| Target | `below_c` = current rating D, E, F or G | `config.py` |
| Models | Logistic regression baseline and histogram gradient boosting, on **core** and **extended** feature sets | `model.py` |
| Validation | Stratified 80/20 held-out test set, plus leave-one-region-out cross-validation (leave-one-local-authority-out for the five development areas) | `model.py` |
| Explanation | Permutation importance, partial dependence, SHAP values (optional), archetype and area tables | `explain.py` |
| Forecast | Pairs of consecutive certificates per home, logistic regression with period effects, scenarios to April 2030 | `forecast.py` |

### Feature sets

- **Core (asset register):** property type, built form, build year, floor area, main fuel, mains gas availability
- **Extended (stock survey):** core plus wall type, wall insulation, roof type, loft insulation depth, heating system, multiple glazing percentage
- **Controls (both sets):** year the EPC was lodged and whether it was assessed under RdSAP 10. To score a home as if assessed today, set these to the current date.

### Avoiding leakage

An EPC rating is calculated from the certificate's own inputs. Feeding the model fields such as the SAP score, running costs, CO₂ or the per-element efficiency ratings would let it reverse-engineer the formula and report a meaningless near-perfect score. These are listed in `LEAKAGE_COLUMNS` and excluded, and a test checks they never reach the feature table.

### Checking the controls are not a back door

Landlords often commission EPCs in large batches, so assessment dates could act as a proxy for area and inflate the scores. To test this, the models were rerun without the date controls. The controls improve performance on unseen areas too (five areas, core model: 0.740 without, 0.757 with), so they capture genuine time effects rather than identifying the area.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[shap,dev]"

# England: put the national --history extract in data/england (about 45 minutes and 4GB of memory in total)
python -m housing_epc.pipeline --raw data/england --england-only --no-shap --reports reports/england
python -m housing_epc.forecast --history data/england --england-only --bootstrap 10 --reports reports/england

# Five development areas: extracts in data/raw and data/history (a few minutes)
python -m housing_epc.pipeline                    # classifier, writes to reports/
python -m housing_epc.forecast --history data/history   # forecast, writes to reports/

python -m housing_epc.pipeline --sample 200000    # random subsample for very large inputs
python -m housing_epc.pipeline --theme light      # light charts instead of the default dark
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

England results are in `reports/england/`; the five development areas are in `reports/`. Both folders hold:

| File | Contents |
|---|---|
| `summary.md` | Plain-English results summary for the classifier |
| `metrics.json` | All classifier metrics, including each held-out region or area |
| `figures/00_below_c_over_time.png` | Share below C by quarter, with RdSAP 10 marked |
| `figures/01_feature_set_comparison.png` | Core vs extended, logistic vs boosting |
| `figures/02_calibration_extended.png` | Are predicted probabilities trustworthy? |
| `figures/03_importance_*.png` | Permutation importance |
| `figures/04_pdp_*.png` | How risk changes with build year, fuel, insulation and more |
| `archetype_risk.csv` | Risk by property type, build era and fuel |
| `local_authority_gap.csv` | Observed vs expected below-C share by local authority |
| `forecast_summary.md` | 2030 forecast: scenarios, checks and period effects |
| `forecast_2030.json` | All forecast numbers, including backtests |
| `forecast_by_area.csv`, `forecast_by_archetype.csv` | Expected homes below C today and in 2030 |
| `figures/07_*`, `08_*`, `09_*` | Repeat certificates by year, forecast scenarios, forecast calibration |

`reports/england/` also has the regional results: `region_gap.csv`, `forecast_by_region.csv` and charts `10_*` and `11_*`. With `shap` installed, the classifier also writes SHAP charts (`05_*`, `06_*`).

## Project structure

```
├── data/
│   ├── raw/                    # EPC downloads go here (git-ignored)
│   ├── history/                # --history extracts for the forecast (git-ignored)
│   ├── england/                # national --history extract (git-ignored)
│   ├── lookups/                # ONS local authority to region lookup
│   └── processed/              # cleaned modelling table (git-ignored)
├── notebooks/
│   └── 01_explore.ipynb        # exploratory analysis of the cleaned table
├── reports/                    # five development areas: metrics, tables and figures
│   └── england/                # all of England, plus regions (the main results)
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

- **Coverage:** only homes with a certificate are included: 3.0 million English social homes with one lodged since 2012. Most social homes have one because an EPC is needed to let a property, but recently let or recently surveyed homes may be over-represented. Welsh homes are excluded.
- **Reassessed homes are not a random sample.** The forecast learns how ratings change from homes that were assessed again, and some of those were reassessed because work had just been done. This may make the "today" figure optimistic (see [Caveats](#caveats)).
- **Certificates age:** the classifier describes each home as it was when assessed. The forecast adjusts for change since then, but only through the year effects it learns from reassessed homes.
- **"Assumed" values:** where an assessor cannot see insulation, RdSAP assumes it from the build era. Some wall and roof insulation values are therefore defaults rather than observations.
- **Regional gaps are not a league table.** The gap between a region's share below C and what its stock mix predicts can reflect things the model does not see, such as how recently and how often landlords re-certify their homes.
- **Explanation is not causation:** importance and partial dependence describe what the model learned. They point to where to look, not to proof that changing one feature will change a rating.

## Ideas for next steps

- Add a "planned retrofit programme" scenario to the forecast, for example a landlord's own works list scored home by home.
- Join deprivation data (Index of Multiple Deprivation via postcode to LSOA): are the highest-risk homes concentrated in the most deprived areas?
- Use the recommendations file to estimate the cost of bringing each archetype up to band C.
- Predict the SAP score directly (regression) to rank homes by how far they are from C.
- Link to the English Housing Survey to explore whether low energy efficiency is associated with damp and mould.

## Licence

Code: MIT (see `LICENSE`). EPC data: subject to MHCLG's licence terms. Region lookup: ONS, Open Government Licence.

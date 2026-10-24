# Results summary

- Homes modelled: **2,668,893** social rented homes, latest EPC per home
- Share below band C: **32.7%**
- Local authorities covered: **296**

## How well can we predict which homes are below C?

| Feature set | Model | ROC AUC | PR AUC | Brier |
|---|---|---|---|---|
| core | logistic regression | 0.772 | 0.606 | 0.177 |
| core | gradient boosting | 0.798 | 0.655 | 0.166 |
| extended | logistic regression | 0.861 | 0.771 | 0.137 |
| extended | gradient boosting | 0.896 | 0.834 | 0.117 |

Adding survey-level fabric and heating detail moves gradient boosting ROC AUC from 0.798 (asset-register data only) to 0.896.

## Does it transfer to unseen areas?

- core: mean ROC AUC 0.789 (± 0.014) across 9 leave-areas-out folds
  - trained without London, tested on it: 0.794
  - trained without North West, tested on it: 0.774
  - trained without South East, tested on it: 0.804
  - trained without West Midlands, tested on it: 0.780
  - trained without Yorkshire and The Humber, tested on it: 0.780
  - trained without East of England, tested on it: 0.797
  - trained without East Midlands, tested on it: 0.795
  - trained without South West, tested on it: 0.811
  - trained without North East, tested on it: 0.767
- extended: mean ROC AUC 0.887 (± 0.016) across 9 leave-areas-out folds
  - trained without London, tested on it: 0.910
  - trained without North West, tested on it: 0.880
  - trained without South East, tested on it: 0.897
  - trained without West Midlands, tested on it: 0.894
  - trained without Yorkshire and The Humber, tested on it: 0.874
  - trained without East of England, tested on it: 0.898
  - trained without East Midlands, tested on it: 0.882
  - trained without South West, tested on it: 0.899
  - trained without North East, tested on it: 0.854

## Most important drivers (extended model, permutation importance)

- Wall insulation: ROC AUC drops by 0.122 when shuffled
- Property type: ROC AUC drops by 0.085 when shuffled
- Loft insulation (mm): ROC AUC drops by 0.084 when shuffled
- Build year: ROC AUC drops by 0.063 when shuffled
- Main fuel: ROC AUC drops by 0.050 when shuffled
- Roof type: ROC AUC drops by 0.038 when shuffled

## Highest-risk archetypes (core model)

Homes and the share below C count every home. Predicted is the model's average risk for the archetype's homes in the held-out test set.

| Property type | Build era | Main fuel | Homes | Below C | Share below C | Predicted |
|---|---|---|---|---|---|---|
| House | 1930-49 | oil | 2,894 | 2,628 | 91% | 90% |
| House | 1950-66 | oil | 4,014 | 3,392 | 84% | 85% |
| House | pre-1930 | electricity | 3,492 | 2,693 | 77% | 76% |
| Flat | pre-1930 | electricity | 13,662 | 9,846 | 72% | 71% |
| Bungalow | pre-1930 | mains_gas | 3,332 | 2,334 | 70% | 70% |
| Maisonette | 1967-82 | electricity | 2,814 | 2,039 | 72% | 70% |
| Flat | 1930-49 | electricity | 3,620 | 2,542 | 70% | 69% |
| House | 1967-82 | electricity | 8,306 | 5,661 | 68% | 66% |
| House | 1983-2002 | electricity | 13,387 | 8,757 | 65% | 65% |
| House | 1930-49 | electricity | 7,750 | 4,884 | 63% | 64% |

Figures are in `reports/england/figures/`. Archetype and area tables are in `reports/england/`.

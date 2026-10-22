# Results summary

- Homes modelled: **136,379** social rented homes, latest EPC per home
- Share below band C: **31.3%**
- Local authorities covered: **5**

## How well can we predict which homes are below C?

| Feature set | Model | ROC AUC | PR AUC | Brier |
|---|---|---|---|---|
| core | logistic regression | 0.758 | 0.560 | 0.179 |
| core | gradient boosting | 0.817 | 0.677 | 0.156 |
| extended | logistic regression | 0.857 | 0.752 | 0.137 |
| extended | gradient boosting | 0.904 | 0.838 | 0.111 |

Adding survey-level fabric and heating detail moves gradient boosting ROC AUC from 0.817 (asset-register data only) to 0.904.

## Does it transfer to unseen areas?

- core: mean ROC AUC 0.757 (± 0.025) across 5 leave-areas-out folds
  - trained without Manchester, tested on it: 0.732
  - trained without Sheffield, tested on it: 0.741
  - trained without Cornwall, tested on it: 0.794
  - trained without Newcastle upon Tyne, tested on it: 0.739
  - trained without Southwark, tested on it: 0.779
- extended: mean ROC AUC 0.871 (± 0.015) across 5 leave-areas-out folds
  - trained without Manchester, tested on it: 0.864
  - trained without Sheffield, tested on it: 0.868
  - trained without Cornwall, tested on it: 0.865
  - trained without Newcastle upon Tyne, tested on it: 0.859
  - trained without Southwark, tested on it: 0.900

## Most important drivers (extended model, permutation importance)

- Wall insulation: ROC AUC drops by 0.103 when shuffled
- Property type: ROC AUC drops by 0.088 when shuffled
- Loft insulation (mm): ROC AUC drops by 0.087 when shuffled
- Build year: ROC AUC drops by 0.070 when shuffled
- Main fuel: ROC AUC drops by 0.049 when shuffled
- Roof type: ROC AUC drops by 0.033 when shuffled

## Highest-risk archetypes (core model, test set)

| Property type | Build era | Main fuel | Homes | Actual below C | Predicted |
|---|---|---|---|---|---|
| House | 1919-44 | oil | 30 | 100% | 94% |
| House | pre-1919 | electricity | 34 | 74% | 79% |
| House | 1945-64 | oil | 45 | 73% | 79% |
| House | 1983-2002 | electricity | 180 | 79% | 77% |
| Flat | pre-1919 | electricity | 141 | 77% | 71% |
| Bungalow | 1983-2002 | electricity | 52 | 73% | 71% |
| House | 1919-44 | electricity | 89 | 61% | 64% |
| Bungalow | 1945-64 | electricity | 71 | 70% | 62% |
| Flat | 1945-64 | electricity | 373 | 58% | 60% |
| House | 1965-82 | communal | 188 | 62% | 58% |

Figures are in `reports/figures/`. Archetype and area tables are in `reports/`.

# Combiner results (held-out validation set)

- Rows: 160765 total, 128612 train / 32153 validation (stratified on target, seed 42)
- Validation class counts: white=26028, asian=4341, black=1006, hispanic=778
- Per-class accuracy = recall on that true class; macro avg = mean of the 4 per-class accuracies
- **Default model (NEW LightGBM, sqrt-balanced weights (DEFAULT), saved as combiner_model.joblib)**: LightGBM on ethnicolr2 + SigLIP2 with sqrt-balanced class weights white=0.556, asian=1.361, black=2.826, hispanic=3.215
- LightGBM was chosen over random forest (higher macro avg, unweighted vs unweighted); sqrt-balanced weights were chosen over unweighted / fully balanced on this validation set (RESULTS_class_weights.md), so the default's validation numbers are slightly optimistic
- Default LightGBM stopped at 179 trees (early stopping on 10% of train, not on validation)

## Accuracy

| model                                         |   n_val | white   | asian   | black   | hispanic   | overall   | macro_avg   |
|:----------------------------------------------|--------:|:--------|:--------|:--------|:-----------|:----------|:------------|
| NEW LightGBM, sqrt-balanced weights (DEFAULT) |   32153 | 93.68%  | 85.58%  | 92.25%  | 67.35%     | 91.90%    | 84.71%      |
| NEW LightGBM, unweighted                      |   32153 | 97.09%  | 83.60%  | 88.87%  | 31.88%     | 93.43%    | 75.36%      |
| NEW random forest, unweighted                 |   32153 | 97.11%  | 83.18%  | 88.47%  | 29.43%     | 93.32%    | 74.55%      |
| OLD baseline RF (DeepFace + census_ln)        |   32153 | 96.90%  | 76.89%  | 65.21%  | 33.16%     | 91.67%    | 68.04%      |

## Old vs new (NEW LightGBM, sqrt-balanced weights (DEFAULT))

|           | old    | new    | change_pp   |
|:----------|:-------|:-------|:------------|
| white     | 96.90% | 93.68% | -3.22 pp    |
| asian     | 76.89% | 85.58% | +8.68 pp    |
| black     | 65.21% | 92.25% | +27.04 pp   |
| hispanic  | 33.16% | 67.35% | +34.19 pp   |
| overall   | 91.67% | 91.90% | +0.24 pp    |
| macro_avg | 68.04% | 84.71% | +16.67 pp   |

**Black accuracy: old 65.21% -> new 92.25% (+27.04 pp)**

**Hispanic accuracy: old 33.16% -> new 67.35% (+34.19 pp)**

## Caveat: class weights trade precision for recall

The weighted default assigns more rows to the minority classes than truly belong to them (mostly White founders labelled Hispanic), and its probabilities are shifted toward minority classes. Use it for per-person labels; for estimating group counts/shares across the population, prefer combiner_model_unweighted.joblib.

Rows assigned to each class:

|                                               |   white |   asian |   black |   hispanic |
|:----------------------------------------------|--------:|--------:|--------:|-----------:|
| NEW LightGBM, sqrt-balanced weights (DEFAULT) |   25107 |    4465 |    1123 |       1458 |
| NEW LightGBM, unweighted                      |   26475 |    4221 |    1000 |        457 |
| NEW random forest, unweighted                 |   26519 |    4207 |     990 |        437 |
| OLD baseline RF (DeepFace + census_ln)        |   26905 |    3988 |     770 |        490 |
| TRUE count (validation)                       |   26028 |    4341 |    1006 |        778 |

Precision (share of rows assigned to a class that truly belong to it):

|                                               | white   | asian   | black   | hispanic   |
|:----------------------------------------------|:--------|:--------|:--------|:-----------|
| NEW LightGBM, sqrt-balanced weights (DEFAULT) | 97.12%  | 83.20%  | 82.64%  | 35.94%     |
| NEW LightGBM, unweighted                      | 95.45%  | 85.97%  | 89.40%  | 54.27%     |
| NEW random forest, unweighted                 | 95.31%  | 85.83%  | 89.90%  | 52.40%     |
| OLD baseline RF (DeepFace + census_ln)        | 93.74%  | 83.70%  | 85.19%  | 52.65%     |

## Feature importances (default model)

rf_impurity (unweighted RF) / lgbm_gain are normalised to sum to 1. lgbm_shap_<class> = mean |SHAP contribution| to that class's raw score on validation rows.

|                      |   rf_impurity |   lgbm_gain |   lgbm_shap_white |   lgbm_shap_asian |   lgbm_shap_black |   lgbm_shap_hispanic |
|:---------------------|--------------:|------------:|------------------:|------------------:|------------------:|---------------------:|
| eth_white            |        0.1234 |      0.078  |            0.3308 |            0.0461 |            0.0635 |               0.0757 |
| eth_asian            |        0.2314 |      0.195  |            0.0525 |            0.4816 |            0.0631 |               0.0615 |
| eth_black            |        0.0348 |      0.0429 |            0.0435 |            0.0254 |            0.3686 |               0.1581 |
| eth_hispanic         |        0.0492 |      0.1306 |            0.0928 |            0.116  |            0.0466 |               0.5413 |
| eth_other_or_missing |        0.0001 |      0.0003 |            0.0001 |            0.0005 |            0.0001 |               0.001  |
| sig_white            |        0.2537 |      0.2193 |            0.6509 |            0.0461 |            0.1204 |               0.0647 |
| sig_asian            |        0.1684 |      0.0832 |            0.2217 |            0.4192 |            0.0971 |               0.031  |
| sig_black            |        0.1093 |      0.2323 |            0.0468 |            0.0903 |            0.6106 |               0.0761 |
| sig_hispanic         |        0.0297 |      0.0184 |            0.0285 |            0.0387 |            0.0431 |               0.2613 |

Share of importance by source:

|                      |   rf_impurity |   lgbm_gain |   lgbm_shap_white |   lgbm_shap_asian |   lgbm_shap_black |   lgbm_shap_hispanic |
|:---------------------|--------------:|------------:|------------------:|------------------:|------------------:|---------------------:|
| SigLIP2 (4 feats)    |         0.561 |       0.553 |             0.646 |              0.47 |             0.617 |                0.341 |
| ethnicolr2 (5 feats) |         0.439 |       0.447 |             0.354 |              0.53 |             0.383 |                0.659 |

## Confusion matrix: new (NEW LightGBM, sqrt-balanced weights (DEFAULT))

|               |   pred_white |   pred_asian |   pred_black |   pred_hispanic |
|:--------------|-------------:|-------------:|-------------:|----------------:|
| true_white    |        24383 |          666 |          154 |             825 |
| true_asian    |          502 |         3715 |           26 |              98 |
| true_black    |           45 |           22 |          928 |              11 |
| true_hispanic |          177 |           62 |           15 |             524 |

## Confusion matrix: old baseline

|               |   pred_white |   pred_asian |   pred_black |   pred_hispanic |
|:--------------|-------------:|-------------:|-------------:|----------------:|
| true_white    |        25222 |          546 |           80 |             180 |
| true_asian    |          931 |         3338 |           28 |              44 |
| true_black    |          294 |           48 |          656 |               8 |
| true_hispanic |          458 |           56 |            6 |             258 |

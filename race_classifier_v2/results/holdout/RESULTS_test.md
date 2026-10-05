# Held-out TEST results

- Split: train 102889 / val 32153 (unchanged from RESULTS.md, used for model + weighting choice) / **test 25723** (carved from the original train rows, stratified, seed 42)
- Test class counts: white=20823, asian=3473, black=805, hispanic=622
- Test rows were never in the validation set used for any choice. Weighting fixed in advance: sqrt-balanced (white=0.556, asian=1.361, black=2.826, hispanic=3.215)
- All models retrained on the 102889 remaining train rows (20% fewer than in RESULTS.md); sqrt-balanced LightGBM stopped at 186 trees
- Per-class accuracy = recall on that true class; macro avg = mean of the 4 per-class accuracies

## Test accuracy

| model                                         |     n | white   | asian   | black   | hispanic   | overall   | macro_avg   |
|:----------------------------------------------|------:|:--------|:--------|:--------|:-----------|:----------|:------------|
| NEW LightGBM, sqrt-balanced weights (DEFAULT) | 25723 | 93.93%  | 85.43%  | 93.29%  | 70.42%     | 92.20%    | 85.77%      |
| NEW LightGBM, unweighted                      | 25723 | 97.19%  | 83.24%  | 90.43%  | 29.90%     | 93.47%    | 75.19%      |
| OLD baseline RF (DeepFace + census_ln)        | 25723 | 97.01%  | 75.96%  | 66.34%  | 31.03%     | 91.61%    | 67.58%      |

## Old vs new on test (NEW LightGBM, sqrt-balanced weights (DEFAULT))

|           | old    | new    | change_pp   |
|:----------|:-------|:-------|:------------|
| white     | 97.01% | 93.93% | -3.07 pp    |
| asian     | 75.96% | 85.43% | +9.47 pp    |
| black     | 66.34% | 93.29% | +26.96 pp   |
| hispanic  | 31.03% | 70.42% | +39.39 pp   |
| overall   | 91.61% | 92.20% | +0.59 pp    |
| macro_avg | 67.58% | 85.77% | +18.19 pp   |

**Black: old 66.34% -> new 93.29% (+26.96 pp)**

**Hispanic: old 31.03% -> new 70.42% (+39.39 pp)**

## Rows assigned to each class (test)

|                                               |   white |   asian |   black |   hispanic |
|:----------------------------------------------|--------:|--------:|--------:|-----------:|
| NEW LightGBM, sqrt-balanced weights (DEFAULT) |   20133 |    3548 |     896 |       1146 |
| NEW LightGBM, unweighted                      |   21223 |    3357 |     813 |        330 |
| OLD baseline RF (DeepFace + census_ln)        |   21598 |    3134 |     635 |        356 |
| TRUE count (test)                             |   20823 |    3473 |     805 |        622 |

## Precision (test)

|                                               | white   | asian   | black   | hispanic   |
|:----------------------------------------------|:--------|:--------|:--------|:-----------|
| NEW LightGBM, sqrt-balanced weights (DEFAULT) | 97.15%  | 83.62%  | 83.82%  | 38.22%     |
| NEW LightGBM, unweighted                      | 95.36%  | 86.12%  | 89.54%  | 56.36%     |
| OLD baseline RF (DeepFace + census_ln)        | 93.53%  | 84.17%  | 84.09%  | 54.21%     |

## Confusion matrix on test: NEW LightGBM, sqrt-balanced weights (DEFAULT)

|               |   pred_white |   pred_asian |   pred_black |   pred_hispanic |
|:--------------|-------------:|-------------:|-------------:|----------------:|
| true_white    |        19560 |          522 |          113 |             628 |
| true_asian    |          407 |         2967 |           30 |              69 |
| true_black    |           31 |           12 |          751 |              11 |
| true_hispanic |          135 |           47 |            2 |             438 |

## Reference: same retrained models on the original validation rows

Not a clean estimate (these rows informed the weighting choice); shown only to compare with RESULTS.md.

| model                                         |     n | white   | asian   | black   | hispanic   | overall   | macro_avg   |
|:----------------------------------------------|------:|:--------|:--------|:--------|:-----------|:----------|:------------|
| NEW LightGBM, sqrt-balanced weights (DEFAULT) | 32153 | 93.73%  | 85.53%  | 91.95%  | 67.87%     | 91.94%    | 84.77%      |
| NEW LightGBM, unweighted                      | 32153 | 97.17%  | 83.25%  | 88.97%  | 30.46%     | 93.42%    | 74.96%      |
| OLD baseline RF (DeepFace + census_ln)        | 32153 | 96.97%  | 76.30%  | 65.31%  | 34.06%     | 91.66%    | 68.16%      |

# Class-weighted LightGBM combiner (held-out validation set)

Same split, features and hyperparameters as RESULTS.md; only per-class sample weights differ.
Weights shown are per-row multipliers by true class (balanced = n / (4 * class count) on train).

| weights       |   w_white |   w_asian |   w_black |   w_hispanic |   trees | white   | asian   | black   | hispanic   | overall   | macro_avg   |
|:--------------|----------:|----------:|----------:|-------------:|--------:|:--------|:--------|:--------|:-----------|:----------|:------------|
| unweighted    |     1     |     1     |     1     |        1     |     208 | 97.09%  | 83.60%  | 88.87%  | 31.88%     | 93.43%    | 75.36%      |
| sqrt-balanced |     0.556 |     1.361 |     2.826 |        3.215 |     179 | 93.68%  | 85.58%  | 92.25%  | 67.35%     | 91.90%    | 84.71%      |
| balanced      |     0.309 |     1.852 |     7.988 |       10.335 |     155 | 89.76%  | 86.09%  | 94.14%  | 76.35%     | 89.07%    | 86.58%      |

Old baseline for reference:

|                                        | white   | asian   | black   | hispanic   | overall   | macro_avg   |
|:---------------------------------------|:--------|:--------|:--------|:-----------|:----------|:------------|
| OLD baseline RF (DeepFace + census_ln) | 96.90%  | 76.89%  | 65.21%  | 33.16%     | 91.67%    | 68.04%      |

## Confusion matrix: unweighted

|               |   pred_white |   pred_asian |   pred_black |   pred_hispanic |
|:--------------|-------------:|-------------:|-------------:|----------------:|
| true_white    |        25271 |          514 |           77 |             166 |
| true_asian    |          657 |         3629 |           19 |              36 |
| true_black    |           83 |           22 |          894 |               7 |
| true_hispanic |          464 |           56 |           10 |             248 |

## Confusion matrix: sqrt-balanced

|               |   pred_white |   pred_asian |   pred_black |   pred_hispanic |
|:--------------|-------------:|-------------:|-------------:|----------------:|
| true_white    |        24383 |          666 |          154 |             825 |
| true_asian    |          502 |         3715 |           26 |              98 |
| true_black    |           45 |           22 |          928 |              11 |
| true_hispanic |          177 |           62 |           15 |             524 |

## Confusion matrix: balanced

|               |   pred_white |   pred_asian |   pred_black |   pred_hispanic |
|:--------------|-------------:|-------------:|-------------:|----------------:|
| true_white    |        23362 |          897 |          329 |            1440 |
| true_asian    |          414 |         3737 |           43 |             147 |
| true_black    |           28 |           17 |          947 |              14 |
| true_hispanic |          107 |           56 |           21 |             594 |

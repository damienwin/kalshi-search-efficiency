# Criterion-chosen fixed-size training sets

Gains: positive = two-stage better (C, B gaps) or criterion better than random (A).
p: Holm-adjusted DM p, side in parentheses when < 0.05.

## select B — articles, balanced

| FLAT | metric | gap random | gap criterion | change | p |
|---|---|---|---|---|---|
| 0.50 | diag_logloss | +0.0193 | -0.6121 | -0.6314 | 0.000 (3c) |
| 0.60 | diag_logloss | +0.0332 | -0.7455 | -0.7787 | 0.000 (3c) |
| 0.70 | diag_logloss | +0.0397 | -0.8329 | -0.8727 | 0.000 (3c) |
| 0.80 | diag_logloss | +0.0411 | -0.9139 | -0.9549 | 0.000 (3c) |
| 0.85 | diag_logloss | +0.0348 | -0.9189 | -0.9537 | 0.000 (3c) |
| 0.90 | diag_logloss | +0.0221 | -0.8821 | -0.9042 | 0.000 (3c) |
| 0.95 | diag_logloss | +0.0056 | -0.7863 | -0.7919 | 0.000 (3c) |
| trend | diag_logloss | | | -0.4735 | 0.000 |
| 0.50 | logloss_adj | -0.0124 | -1.4606 | -1.4481 | 0.000 (3c) |
| 0.60 | logloss_adj | -0.0060 | -1.4523 | -1.4463 | 0.000 (3c) |
| 0.70 | logloss_adj | +0.0018 | -1.3943 | -1.3961 | 0.000 (3c) |
| 0.80 | logloss_adj | +0.0157 | -1.3222 | -1.3379 | 0.000 (3c) |
| 0.85 | logloss_adj | +0.0266 | -1.2427 | -1.2693 | 0.000 (3c) |
| 0.90 | logloss_adj | +0.0516 | -1.1227 | -1.1743 | 0.000 (3c) |
| 0.95 | logloss_adj | +0.1016 | -0.9091 | -1.0107 | 0.000 (3c) |
| trend | logloss_adj | | | +0.8730 | 0.000 |
| 0.50 | dir_logloss | -0.0009 | -0.0009 | +0.0000 | 0.76 |
| 0.60 | dir_logloss | +0.0010 | +0.0001 | -0.0009 | 0.76 |
| 0.70 | dir_logloss | +0.0083 | +0.0058 | -0.0025 | 0.76 |
| 0.80 | dir_logloss | +0.0240 | +0.0187 | -0.0053 | 0.13 |
| 0.85 | dir_logloss | +0.0401 | +0.0372 | -0.0029 | 0.76 |
| 0.90 | dir_logloss | +0.0886 | +0.0779 | -0.0107 | 0.09 |
| 0.95 | dir_logloss | +0.1888 | +0.1686 | -0.0202 | 0.09 |
| trend | dir_logloss | | | -0.0349 | 0.034 |

## select A — articles, balanced (random minus criterion)

| model | FLAT | prior-corrected log loss | p | direction log loss | p |
|---|---|---|---|---|---|
| 3class | 0.50 | -2.0434 | 0.000 (3c) | -0.0000 | 1.00 |
| 3class | 0.60 | -1.9485 | 0.000 (3c) | +0.0009 | 1.00 |
| 3class | 0.70 | -1.8534 | 0.000 (3c) | +0.0025 | 1.00 |
| 3class | 0.80 | -1.7296 | 0.000 (3c) | +0.0053 | 0.30 |
| 3class | 0.85 | -1.6709 | 0.000 (3c) | +0.0029 | 1.00 |
| 3class | 0.90 | -1.5875 | 0.000 (3c) | +0.0107 | 0.19 |
| 3class | 0.95 | -1.3538 | 0.000 (3c) | +0.0202 | 0.18 |
| two_stage | 0.50 | -3.4916 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.60 | -3.3948 | 0.000 (3c) | -0.0000 | 1.00 |
| two_stage | 0.70 | -3.2494 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.80 | -3.0675 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.85 | -2.9402 | 0.000 (3c) | -0.0000 | 1.00 |
| two_stage | 0.90 | -2.7619 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.95 | -2.3645 | 0.000 (3c) | +0.0000 | 1.00 |

## select B — articles, none

| FLAT | metric | gap random | gap criterion | change | p |
|---|---|---|---|---|---|
| 0.50 | diag_logloss | -0.0138 | -0.6729 | -0.6591 | 0.000 (3c) |
| 0.60 | diag_logloss | -0.0123 | -0.7986 | -0.7863 | 0.000 (3c) |
| 0.70 | diag_logloss | -0.0095 | -0.8260 | -0.8165 | 0.000 (3c) |
| 0.80 | diag_logloss | -0.0039 | -0.8898 | -0.8859 | 0.000 (3c) |
| 0.85 | diag_logloss | -0.0010 | -0.8722 | -0.8712 | 0.000 (3c) |
| 0.90 | diag_logloss | +0.0034 | -0.8202 | -0.8236 | 0.000 (3c) |
| 0.95 | diag_logloss | +0.0060 | -0.6177 | -0.6237 | 0.000 (3c) |
| trend | diag_logloss | | | -0.1054 | 0.222 |
| 0.50 | logloss_adj | -0.0095 | -1.1576 | -1.1482 | 0.000 (3c) |
| 0.60 | logloss_adj | -0.0079 | -1.1483 | -1.1404 | 0.000 (3c) |
| 0.70 | logloss_adj | -0.0063 | -1.0238 | -1.0175 | 0.000 (3c) |
| 0.80 | logloss_adj | -0.0026 | -0.9662 | -0.9637 | 0.000 (3c) |
| 0.85 | logloss_adj | +0.0000 | -0.8909 | -0.8910 | 0.000 (3c) |
| 0.90 | logloss_adj | +0.0065 | -0.7869 | -0.7933 | 0.000 (3c) |
| 0.95 | logloss_adj | +0.0199 | -0.5462 | -0.5661 | 0.000 (3c) |
| trend | logloss_adj | | | +1.1475 | 0.000 |
| 0.50 | dir_logloss | -0.0072 | -0.0047 | +0.0025 | 1.00 |
| 0.60 | dir_logloss | -0.0090 | -0.0057 | +0.0033 | 1.00 |
| 0.70 | dir_logloss | -0.0073 | -0.0058 | +0.0015 | 1.00 |
| 0.80 | dir_logloss | -0.0049 | -0.0046 | +0.0002 | 1.00 |
| 0.85 | dir_logloss | +0.0034 | -0.0025 | -0.0059 | 0.12 |
| 0.90 | dir_logloss | +0.0257 | +0.0141 | -0.0116 | 0.000 (3c) |
| 0.95 | dir_logloss | +0.0756 | +0.0618 | -0.0138 | 0.008 (3c) |
| trend | dir_logloss | | | -0.0376 | 0.002 |

## select A — articles, none (random minus criterion)

| model | FLAT | prior-corrected log loss | p | direction log loss | p |
|---|---|---|---|---|---|
| 3class | 0.50 | -2.3297 | 0.000 (3c) | -0.0025 | 1.00 |
| 3class | 0.60 | -2.3997 | 0.000 (3c) | -0.0033 | 1.00 |
| 3class | 0.70 | -2.4835 | 0.000 (3c) | -0.0015 | 1.00 |
| 3class | 0.80 | -2.5652 | 0.000 (3c) | -0.0002 | 1.00 |
| 3class | 0.85 | -2.5956 | 0.000 (3c) | +0.0059 | 0.29 |
| 3class | 0.90 | -2.6783 | 0.000 (3c) | +0.0116 | 0.001 (2s) |
| 3class | 0.95 | -2.5961 | 0.000 (3c) | +0.0138 | 0.018 (2s) |
| two_stage | 0.50 | -3.4779 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.60 | -3.5401 | 0.000 (3c) | -0.0000 | 1.00 |
| two_stage | 0.70 | -3.5010 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.80 | -3.5288 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.85 | -3.4866 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.90 | -3.4716 | 0.000 (3c) | -0.0000 | 1.00 |
| two_stage | 0.95 | -3.1622 | 0.000 (3c) | -0.0000 | 1.00 |

## select B — vol, balanced

| FLAT | metric | gap random | gap criterion | change | p |
|---|---|---|---|---|---|
| 0.50 | diag_logloss | +0.0193 | -0.0113 | -0.0306 | 0.83 |
| 0.60 | diag_logloss | +0.0332 | -0.0110 | -0.0442 | 0.27 |
| 0.70 | diag_logloss | +0.0397 | +0.0294 | -0.0103 | 0.83 |
| 0.80 | diag_logloss | +0.0411 | +0.0258 | -0.0153 | 0.83 |
| 0.85 | diag_logloss | +0.0348 | +0.0138 | -0.0210 | 0.83 |
| 0.90 | diag_logloss | +0.0221 | -0.0459 | -0.0680 | 0.15 |
| 0.95 | diag_logloss | +0.0056 | -0.1443 | -0.1499 | 0.000 (3c) |
| trend | diag_logloss | | | -0.1593 | 0.100 |
| 0.50 | logloss_adj | -0.0124 | -0.3661 | -0.3536 | 0.000 (3c) |
| 0.60 | logloss_adj | -0.0060 | -0.3740 | -0.3680 | 0.000 (3c) |
| 0.70 | logloss_adj | +0.0018 | -0.3152 | -0.3170 | 0.000 (3c) |
| 0.80 | logloss_adj | +0.0157 | -0.2943 | -0.3101 | 0.000 (3c) |
| 0.85 | logloss_adj | +0.0266 | -0.2670 | -0.2936 | 0.000 (3c) |
| 0.90 | logloss_adj | +0.0516 | -0.2471 | -0.2988 | 0.000 (3c) |
| 0.95 | logloss_adj | +0.1016 | -0.1638 | -0.2654 | 0.000 (3c) |
| trend | logloss_adj | | | +0.2003 | 0.013 |
| 0.50 | dir_logloss | -0.0009 | -0.0046 | -0.0037 | 0.10 |
| 0.60 | dir_logloss | +0.0010 | -0.0051 | -0.0061 | 0.046 (3c) |
| 0.70 | dir_logloss | +0.0083 | +0.0028 | -0.0055 | 0.043 (3c) |
| 0.80 | dir_logloss | +0.0240 | +0.0156 | -0.0084 | 0.010 (3c) |
| 0.85 | dir_logloss | +0.0401 | +0.0298 | -0.0103 | 0.000 (3c) |
| 0.90 | dir_logloss | +0.0886 | +0.0645 | -0.0241 | 0.000 (3c) |
| 0.95 | dir_logloss | +0.1888 | +0.1586 | -0.0302 | 0.000 (3c) |
| trend | dir_logloss | | | -0.0517 | 0.000 |

## select A — vol, balanced (random minus criterion)

| model | FLAT | prior-corrected log loss | p | direction log loss | p |
|---|---|---|---|---|---|
| 3class | 0.50 | -1.4708 | 0.000 (3c) | +0.0037 | 0.76 |
| 3class | 0.60 | -1.3531 | 0.000 (3c) | +0.0061 | 0.21 |
| 3class | 0.70 | -1.1840 | 0.000 (3c) | +0.0055 | 0.14 |
| 3class | 0.80 | -1.0141 | 0.000 (3c) | +0.0084 | 0.028 (2s) |
| 3class | 0.85 | -0.8980 | 0.000 (3c) | +0.0103 | 0.001 (2s) |
| 3class | 0.90 | -0.7568 | 0.000 (3c) | +0.0241 | 0.000 (2s) |
| 3class | 0.95 | -0.5211 | 0.000 (3c) | +0.0302 | 0.000 (2s) |
| two_stage | 0.50 | -1.8245 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.60 | -1.7211 | 0.000 (3c) | -0.0000 | 1.00 |
| two_stage | 0.70 | -1.5010 | 0.000 (3c) | -0.0000 | 1.00 |
| two_stage | 0.80 | -1.3242 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.85 | -1.1916 | 0.000 (3c) | -0.0000 | 1.00 |
| two_stage | 0.90 | -1.0556 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.95 | -0.7865 | 0.000 (3c) | +0.0000 | 1.00 |

## select B — vol, none

| FLAT | metric | gap random | gap criterion | change | p |
|---|---|---|---|---|---|
| 0.50 | diag_logloss | -0.0138 | -0.0644 | -0.0506 | 0.21 |
| 0.60 | diag_logloss | -0.0123 | -0.0727 | -0.0603 | 0.013 (3c) |
| 0.70 | diag_logloss | -0.0095 | -0.0240 | -0.0145 | 1.00 |
| 0.80 | diag_logloss | -0.0039 | +0.0558 | +0.0597 | 0.05 |
| 0.85 | diag_logloss | -0.0010 | +0.1282 | +0.1292 | 0.002 (2s) |
| 0.90 | diag_logloss | +0.0034 | +0.1862 | +0.1828 | 0.000 (2s) |
| 0.95 | diag_logloss | +0.0060 | +0.2309 | +0.2250 | 0.000 (2s) |
| trend | diag_logloss | | | +0.6623 | 0.000 |
| 0.50 | logloss_adj | -0.0095 | -0.1004 | -0.0910 | 0.12 |
| 0.60 | logloss_adj | -0.0079 | -0.0974 | -0.0895 | 0.004 (3c) |
| 0.70 | logloss_adj | -0.0063 | -0.0279 | -0.0216 | 0.68 |
| 0.80 | logloss_adj | -0.0026 | +0.0596 | +0.0622 | 0.06 |
| 0.85 | logloss_adj | +0.0000 | +0.1347 | +0.1346 | 0.003 (2s) |
| 0.90 | logloss_adj | +0.0065 | +0.1987 | +0.1922 | 0.000 (2s) |
| 0.95 | logloss_adj | +0.0199 | +0.3293 | +0.3095 | 0.000 (2s) |
| trend | logloss_adj | | | +0.8696 | 0.000 |
| 0.50 | dir_logloss | -0.0072 | -0.0100 | -0.0028 | 0.56 |
| 0.60 | dir_logloss | -0.0090 | -0.0134 | -0.0044 | 0.48 |
| 0.70 | dir_logloss | -0.0073 | -0.0087 | -0.0014 | 0.66 |
| 0.80 | dir_logloss | -0.0049 | -0.0033 | +0.0016 | 0.68 |
| 0.85 | dir_logloss | +0.0034 | -0.0010 | -0.0044 | 0.31 |
| 0.90 | dir_logloss | +0.0257 | +0.0127 | -0.0129 | 0.000 (3c) |
| 0.95 | dir_logloss | +0.0756 | +0.0675 | -0.0081 | 0.07 |
| trend | dir_logloss | | | -0.0139 | 0.150 |

## select A — vol, none (random minus criterion)

| model | FLAT | prior-corrected log loss | p | direction log loss | p |
|---|---|---|---|---|---|
| 3class | 0.50 | -1.7344 | 0.000 (3c) | +0.0028 | 1.00 |
| 3class | 0.60 | -1.7671 | 0.000 (3c) | +0.0044 | 1.00 |
| 3class | 0.70 | -1.7401 | 0.000 (3c) | +0.0014 | 1.00 |
| 3class | 0.80 | -1.7431 | 0.000 (3c) | -0.0016 | 1.00 |
| 3class | 0.85 | -1.7095 | 0.000 (3c) | +0.0044 | 0.75 |
| 3class | 0.90 | -1.6093 | 0.000 (3c) | +0.0129 | 0.000 (2s) |
| 3class | 0.95 | -1.2467 | 0.000 (3c) | +0.0081 | 0.15 |
| two_stage | 0.50 | -1.8254 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.60 | -1.8566 | 0.000 (3c) | -0.0000 | 1.00 |
| two_stage | 0.70 | -1.7617 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.80 | -1.6810 | 0.000 (3c) | +0.0000 | 1.00 |
| two_stage | 0.85 | -1.5749 | 0.000 (3c) | -0.0000 | 1.00 |
| two_stage | 0.90 | -1.4171 | 0.000 (3c) | -0.0000 | 1.00 |
| two_stage | 0.95 | -0.9373 | 0.000 (3c) | +0.0000 | 1.00 |

## filter C — articles, balanced

| filter | train FLAT | test FLAT | log loss gain | p | prior-corr gain | p | dir log loss gain | p | dir AUC 3c / 2s |
|---|---|---|---|---|---|---|---|---|---|
| all_0 | 0.872 | 0.851 | +0.0252 | 0.000 (2s) | +0.0427 | 0.003 (2s) | +0.0812 | 0.000 (2s) | 0.514 / 0.509 |
| articles_1 | 0.727 | 0.706 | +0.0253 | 0.000 (2s) | +0.0213 | 0.012 (2s) | +0.0133 | 0.38 | 0.518 / 0.511 |
| articles_2 | 0.711 | 0.689 | +0.0256 | 0.000 (2s) | +0.0198 | 0.022 (2s) | +0.0102 | 0.43 | 0.512 / 0.507 |
| articles_5 | 0.674 | 0.656 | +0.0243 | 0.000 (2s) | +0.0135 | 0.06 | +0.0034 | 0.62 | 0.518 / 0.515 |
| articles_10 | 0.640 | 0.618 | +0.0205 | 0.000 (2s) | +0.0022 | 0.41 | -0.0019 | 0.62 | 0.513 / 0.505 |
| trend | | | +0.0135 | 0.582 | +0.1636 | 0.001 | +0.3730 | 0.000 | |

## filter C — articles, none

| filter | train FLAT | test FLAT | log loss gain | p | prior-corr gain | p | dir log loss gain | p | dir AUC 3c / 2s |
|---|---|---|---|---|---|---|---|---|---|
| all_0 | 0.872 | 0.851 | +0.0075 | 0.12 | +0.0053 | 0.34 | +0.0247 | 0.06 | 0.512 / 0.507 |
| articles_1 | 0.727 | 0.706 | -0.0046 | 0.11 | -0.0055 | 0.06 | -0.0112 | 0.049 (3c) | 0.518 / 0.510 |
| articles_2 | 0.711 | 0.689 | -0.0054 | 0.11 | -0.0065 | 0.06 | -0.0141 | 0.048 (3c) | 0.514 / 0.504 |
| articles_5 | 0.674 | 0.656 | -0.0079 | 0.012 (3c) | -0.0082 | 0.014 (3c) | -0.0134 | 0.041 (3c) | 0.520 / 0.515 |
| articles_10 | 0.640 | 0.618 | -0.0134 | 0.003 (3c) | -0.0138 | 0.002 (3c) | -0.0158 | 0.028 (3c) | 0.515 / 0.507 |
| trend | | | +0.0852 | 0.002 | +0.0772 | 0.002 | +0.1838 | 0.000 | |

## filter C — vol, balanced

| filter | train FLAT | test FLAT | log loss gain | p | prior-corr gain | p | dir log loss gain | p | dir AUC 3c / 2s |
|---|---|---|---|---|---|---|---|---|---|
| all_0 | 0.872 | 0.851 | +0.0252 | 0.000 (2s) | +0.0427 | 0.002 (2s) | +0.0812 | 0.000 (2s) | 0.514 / 0.509 |
| vol_0.25 | 0.848 | 0.795 | +0.0318 | 0.000 (2s) | +0.0373 | 0.001 (2s) | +0.0550 | 0.008 (2s) | 0.514 / 0.505 |
| vol_0.5 | 0.813 | 0.759 | +0.0339 | 0.000 (2s) | +0.0310 | 0.002 (2s) | +0.0415 | 0.017 (2s) | 0.517 / 0.509 |
| vol_0.75 | 0.673 | 0.611 | +0.0305 | 0.000 (2s) | +0.0053 | 0.22 | +0.0098 | 0.18 | 0.517 / 0.513 |
| trend | | | -0.0118 | 0.596 | +0.1859 | 0.000 | +0.3157 | 0.000 | |

## filter C — vol, none

| filter | train FLAT | test FLAT | log loss gain | p | prior-corr gain | p | dir log loss gain | p | dir AUC 3c / 2s |
|---|---|---|---|---|---|---|---|---|---|
| all_0 | 0.872 | 0.851 | +0.0075 | 0.10 | +0.0053 | 0.27 | +0.0247 | 0.050 (2s) | 0.512 / 0.507 |
| vol_0.25 | 0.848 | 0.795 | +0.0057 | 0.28 | -0.0001 | 0.98 | +0.0091 | 0.62 | 0.513 / 0.504 |
| vol_0.5 | 0.813 | 0.759 | +0.0032 | 0.45 | -0.0028 | 0.61 | +0.0038 | 0.68 | 0.516 / 0.509 |
| vol_0.75 | 0.673 | 0.611 | -0.0059 | 0.31 | -0.0121 | 0.002 (3c) | -0.0061 | 0.80 | 0.516 / 0.511 |
| trend | | | +0.0668 | 0.001 | +0.0796 | 0.001 | +0.1263 | 0.000 | |

## Decision

- **select**: not reported, the criterion makes models significantly worse in 56 of 56 cells (prior-corrected log loss; worst A|articles|none|logloss_adj @ 0.60 two_stage: -3.540)
- **filter**: reported (32 significant new tests)
  - C|articles|balanced|logloss_adj @ all_0: +0.0427
  - C|articles|balanced|logloss_adj @ articles_1: +0.0213
  - C|articles|balanced|logloss_adj @ articles_2: +0.0198
  - C|articles|balanced|logloss_adj slope +0.1636 p=0.00074
  - C|articles|balanced|dir_logloss @ all_0: +0.0812
  - C|articles|balanced|dir_logloss slope +0.3730 p=4.29e-09
  - C|vol|balanced|logloss_adj @ all_0: +0.0427
  - C|vol|balanced|logloss_adj @ vol_0.25: +0.0373
  - C|vol|balanced|logloss_adj @ vol_0.5: +0.0310
  - C|vol|balanced|logloss_adj slope +0.1859 p=0.000436
  - C|vol|balanced|dir_logloss @ all_0: +0.0812
  - C|vol|balanced|dir_logloss @ vol_0.25: +0.0550
  - C|vol|balanced|dir_logloss @ vol_0.5: +0.0415
  - C|vol|balanced|dir_logloss slope +0.3157 p=5.79e-10
  - C|vol|balanced|dir_auc @ vol_0.25: -0.0087
  - C|articles|none|logloss @ articles_5: -0.0079
  - C|articles|none|logloss @ articles_10: -0.0134
  - C|articles|none|logloss slope +0.0852 p=0.00197
  - C|articles|none|logloss_adj @ articles_5: -0.0082
  - C|articles|none|logloss_adj @ articles_10: -0.0138
  - C|articles|none|logloss_adj slope +0.0772 p=0.00245
  - C|articles|none|dir_logloss @ articles_1: -0.0112
  - C|articles|none|dir_logloss @ articles_2: -0.0141
  - C|articles|none|dir_logloss @ articles_5: -0.0134
  - C|articles|none|dir_logloss @ articles_10: -0.0158
  - C|articles|none|dir_logloss slope +0.1838 p=8.65e-05
  - C|vol|none|logloss slope +0.0668 p=0.000714
  - C|vol|none|logloss_adj @ vol_0.75: -0.0121
  - C|vol|none|logloss_adj slope +0.0796 p=0.000887
  - C|vol|none|dir_logloss @ all_0: +0.0247
  - C|vol|none|dir_logloss slope +0.1263 p=0.000248
  - C|vol|none|dir_auc @ vol_0.25: -0.0087

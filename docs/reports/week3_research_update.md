# Kalshi Prediction Market Model — Week 3 Research Update

## v2 results: current dataset

**Change to the standard.** I propose replacing the +0.005 macro F1 keep
rule with a significance test on log loss: a feature is kept if adding it
lowers log loss with a one-sided p < 0.05 on a Diebold–Mariano test (details
in the significance standard section below). Macro F1 is still reported. The
tables therefore show log loss alongside macro F1.

Macro F1 from walk-forward CV on the current (post-Spring) development
events, with 5 training seeds. Majority and no change both always predict
FLAT, so they share a row. Log loss (lower is better) is the mean ± sd over
the same 5 seeds. Baselines are scored on probabilistic forms and have no
seeds, so no sd: majority, no change and random use the training fold's class
rates, and momentum uses the training rates of each label given the market's
previous label (smoothed toward the class rates).

**All events (10,371)**

| Predictor | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± sd | Log loss (mean ± sd) |
|---|---|---|---|---|---|---|---|
| Majority / no change | 0.307 | 0.307 | 0.307 | 0.307 | 0.307 | 0.307 | 0.531 |
| Random | 0.336 | 0.334 | 0.330 | 0.331 | 0.325 | 0.331 ± 0.004 | 0.531 |
| Momentum | 0.461 | 0.461 | 0.461 | 0.461 | 0.461 | 0.461 | 0.462 |
| Price | 0.504 | 0.501 | 0.505 | 0.501 | 0.503 | 0.503 ± 0.002 | 0.606 ± 0.003 |
| Price + sentiment | 0.513 | 0.515 | 0.513 | 0.518 | 0.519 | 0.516 ± 0.003 | 0.587 ± 0.003 |

**News events: ≥ 2 articles in the 6 hours before prediction time (4,559)**

| Predictor | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± sd | Log loss (mean ± sd) |
|---|---|---|---|---|---|---|---|
| Majority / no change | 0.280 | 0.280 | 0.280 | 0.280 | 0.280 | 0.280 | 0.786 |
| Random | 0.342 | 0.330 | 0.335 | 0.331 | 0.324 | 0.332 ± 0.007 | 0.786 |
| Momentum | 0.451 | 0.451 | 0.451 | 0.451 | 0.451 | 0.451 | 0.707 |
| Sentiment alone | 0.390 | 0.397 | 0.400 | 0.406 | 0.401 | 0.399 ± 0.006 | 0.927 ± 0.001 |
| Price | 0.514 | 0.520 | 0.519 | 0.514 | 0.521 | 0.518 ± 0.003 | 0.703 ± 0.002 |
| Price + sentiment | 0.541 | 0.544 | 0.536 | 0.536 | 0.535 | 0.538 ± 0.004 | 0.673 ± 0.001 |

## Finding: price + sentiment differs from price on few events

On the 4,559 news events, the two models make the same call on 86.5%. Where
they differ, price + sentiment is right more often: 261 events against 173,
a net of +88 correct calls (1.9% of events). The probability on the true class
moves by about 0.16 either way, so the fixes and breaks are equally confident.
The full breakdown is at the bottom of the doc.

## Finding: the matched articles are mostly unrelated to the market

| Measure | Value |
|---|---|
| Articles per news event | Mean 14.4, median 12 |
| Articles containing the market's exact query phrase | 3.4% (0.50 per event) |
| Articles with any query word in the headline | 5.5% (0.80 per event) |
| Events with ≥ 2 articles containing the query phrase | 233 of 4,559 (5.1%) |
| Events with ≥ 2 articles with a query word in the headline | 865 of 4,559 (19.0%) |
| Article overlap between unrelated markets in the same hour | Mean Jaccard 0.32 |

The search queries are built from overlapping word pairs in the market
title, such as `high temp temp Miami Miami Nov`, and the search returns
general news for them.

**Proposed fix, before the search arms start**

1. Rebuild the queries from the market's topic and keep only articles that mention it.
2. Add hour, weekday and article volume to the price baseline, so sentiment has to beat them.
3. Use different-market sentiment as the placebo feature in the standard below.
4. Re-run price + sentiment vs price.

A strict relevance filter leaves 233 to 865 news events at current coverage,
so step 1 likely also needs a longer lookback than 6 hours or a wider news
source.

## Proposed next step: significance standard

I propose replacing the +0.005 macro F1 keep rule with a significance test:
a feature is kept if adding it lowers log loss with a one-sided p < 0.05 on
a Diebold–Mariano test.

**Diebold–Mariano test.** The standard test for whether two forecasters
differ in accuracy. It takes the difference in loss between them on each
observation and tests whether the mean difference is zero, using a standard
error that allows the differences to be correlated over time.

**Event group.** All the markets Kalshi lists under one event: the same
question at different strikes, such as every "S&P 500 closes between X and Y
on Feb 23" market. Their prices move together, so each event group counts as
one observation, not each market.

| Step | |
|---|---|
| 1. Predict | Average each event's probabilities over 5 training seeds, with and without the feature |
| 2. Difference | d = log loss(without) − log loss(with) per event, summed within each event group |
| 3. Test | Diebold–Mariano on the event-group sums in time order; keep if one-sided p < 0.05 |
| 4. During search | The gain must also beat the 95th percentile of placebo features, with Benjamini–Hochberg correction across candidates |
| 5. Final claim | LLM vs TPE and vs random on the sealed slice, scored once, Holm-corrected |

Macro F1 is still reported, with an event-group bootstrap p-value.

**Dropped**

| Dropped | Why |
|---|---|
| +0.005 macro F1 threshold | Below the noise: CIs are about ±0.011, so it never decides the outcome |
| At least 4 of 5 folds won | Too coarse with 5 folds; can reject a feature whose CI excludes zero |
| Single training seed | Verdict flips by seed: sentiment vs price is significant on 3 of 5 seeds |
| Macro F1 as the deciding metric | Ignores predicted probabilities and is weak when most events are FLAT |
| Bootstrap CI as the deciding test | Treats event groups as independent; groups open at the same time are correlated, so its p-values are too small |

**Next moves**

1. Save each event's predicted probabilities from CV (only scores are saved today).
2. Implement the test and the placebo threshold.
3. Re-run price + sentiment vs price under the new standard.
4. Use it as the keep rule for the search arms.

## Additional stats: price vs price + sentiment, where they disagree

News events on the current dataset (4,559), predictions averaged over the 5
seeds. The two models make the same call on 86.5% of events.

| Outcome | Events |
|---|---|
| Both right | 2,994 |
| Both wrong | 1,131 |
| Only price + sentiment right | 261 |
| Only price right | 173 |

Sentiment nets +88 correct calls. When it changes the outcome, the
probability on the true class moves by about 0.16 either way.

Sentiment helped (261 events, price wrong and price + sentiment right):

| What price got wrong | Events |
|---|---|
| Predicted a move, but the market stayed FLAT | 99 |
| Predicted the wrong direction (UP vs DOWN) | 151 |
| Predicted FLAT, but the market moved | 11 |

Sentiment hurt (173 events, price right and price + sentiment wrong):

| What price + sentiment got wrong | Events |
|---|---|
| Predicted a move, but the market stayed FLAT | 38 |
| Predicted the wrong direction (UP vs DOWN) | 115 |
| Predicted FLAT, but the market moved | 20 |

## Where the code is

Repository: https://github.com/damienwin/kalshi-search-efficiency

The feature search for Workstream C is built and tested; no search has been run yet.

| Part | Status |
|---|---|
| Feature grammar | Candidate features are expression trees over 28 operators and the 14 price columns; no generated code runs, and every operator is tested for look-ahead |
| Search arms | Random, TPE, and LLM full / no-history / blind (C1–C3), drawing from the same grammar under the same trial budget |
| Significance gate | The test above, with a variance that also allows for correlation between markets open on the same days |
| Search loop | Logs every trial; LLM calls are recorded so a run can be replayed without API access |
| C4 | LLM prompts contain no dates, tickers or event names; gains can be split before and after the model's training cutoff |
| Tests | 561 pass, including end-to-end runs of all five arms on synthetic data |

Next: one live test of the LLM arm, then 5 seeds per arm.


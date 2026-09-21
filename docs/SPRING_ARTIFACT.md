# Spring 2026 artifact

The CSC 4444 result this project corrects lives in
[`damienwin/kalshi-sentiment-predictor`](https://github.com/damienwin/kalshi-sentiment-predictor),
frozen at two local tags. Nothing from that repo's label pipeline is ported
here; this file records what it produces so the paper's correction is
checkable.

| Tag | Commit | What it is |
|---|---|---|
| `v-spring2026-published` | `99aa2a6` (2026-07-02) | State behind the CSC 4444 paper (test Macro F1 0.4179) |
| `v-spring2026-honest` | `99eb974` (2026-07-06) | After fixing the `market_cat_*` leak (train-only, KX-normalized encoder) |

## Reproduction (run 2026-09-21)

Each tag was checked out as a detached `git worktree`, then run with the old
repo's Python 3.9.6 venv on the feature CSVs and models committed at that tag.
No raw data is needed.

```bash
git worktree add --detach /tmp/spring/<tag> <tag>
cd /tmp/spring/<tag>
python scripts/train_model.py --model finbert --classifier xgboost
python scripts/run_backtest.py --output-dir /tmp/spring/results_<tag>
```

### `v-spring2026-honest` — reproduces

| | Test Macro F1 |
|---|---|
| finbert_xgboost | **0.390** |
| momentum baseline | **0.395** |
| majority (FLAT) | 0.224 |

This matches `best_features.json` at the tag (test 0.3902, 95% CI [0.3535, 0.4223]):
the model does not beat momentum.

Feature CSV sha256 at this tag:
```
86ebe108c61771c7ddf246857a2a36933b58950dc7fd5fdc6cf20362124a4ffe  train_finbert.csv
2df51ecd7402cdf719e59d334e96bc0da8faefae602c46e78409ea68e78a6b63  val_finbert.csv
ebccb711d343176855f12b84ea662c22b89f3655f784ab52a1e5e067d4d11f11  test_finbert.csv
```

### `v-spring2026-published` — does not reproduce from its own commit

`run_backtest.py` fails on `finbert_xgboost`:

```
ValueError: X has 21 features, but StandardScaler is expecting 22 features as input.
```

The committed model was trained on 22 features. The committed feature CSVs
carry 21. At this commit features were selected by column position, not
name, so the 0.4179 cannot be regenerated from the artifacts at the tag. The
number is on record in the paper and in the tag's `automl/state/best_features.json`
(CV 0.3816 ± 0.0426), but it has no runnable provenance. Per plan, this was
time-boxed and recorded rather than debugged.

Feature CSV sha256 at this tag:
```
fea1e5bf4175106c6fe39eec358b9fdc1f89e6d06ada527bb78e5062bddf6f94  train_finbert.csv
27b7725e8ca24fc4820db00f86f54672b7e7710eae1217b4bc4e5c740368ff86  val_finbert.csv
e377fcef621db77a99bb77fd4805a415d00fe13099b861a6cd6c284a44fa8f2c  test_finbert.csv
```

## Why the Spring labels are not reused

Both tags share one label pipeline, with three defects found during this rebuild
(details and counts in [DATA_CARD.md](DATA_CARD.md)):

1. **Null price read as $0.00.** `_parse_dollar_str(None)` returned `0.0`
   (`src/data/kalshi.py:13-14`). Kalshi's candle `price.close` is null in any
   hour without trades. 69.5% of UP/DOWN labels (1,723 of 2,480) touch a
   $0.00 endpoint.
2. **"4 hours" was 4 candles.** `find_price_events` used `candles[i + 4]`
   (`src/data/dataset.py:59`). Candles are only emitted for active hours, so
   only 29.7% of windows span 4h. The median is 10h, and 26.8% exceed 24h.
3. **Live-schema quotes read as $0.00.** After the historical cutoff, the
   API renamed `yes_bid.close` to `yes_bid.close_dollars`. The old parser
   read the old key and got `None` → `0.0`, so every post-cutoff quote was $0.00.
   (Not triggered by the Spring data, which predates the cutoff, but it
   would have broken the extension to 2026.)

Also, the momentum baseline (`src/backtest/engine.py:87`) predicts the previous
*row's* label. The feature CSVs are sorted by market, then time, so this is
same-market lag-1 except at each market's first event, where the label comes
from a different market.

These defects invalidate the specific Spring numbers. They do not show that
the sentiment signal is absent; that question is re-asked on v1 labels.

"""Sentiment features, ported verbatim from Spring (src/model/features.py).

compute_event_features / compute_temporal_features / compute_cross_event_features
are unchanged. The market-category encoder is not ported here: it was the Spring
leak and returns only as a FoldScoped transform inside CV.
"""

import math
from collections import Counter

import numpy as np
import pandas as pd

from ksearch.features._time import from_iso

EVENT_FEATURES = [
    "mean_positive", "mean_negative", "mean_neutral", "mean_sentiment", "pct_positive",
    "pct_negative", "non_neutral_ratio", "sentiment_abs_mean", "sentiment_variance",
    "sentiment_range", "avg_confidence", "article_count", "log_article_count",
    "agreement_ratio", "entropy", "sentiment_kurtosis", "dual_polarity_ratio", "polarity_strength",
]
TEMPORAL_FEATURES = [
    "recency_weighted_sentiment", "early_vs_late_sentiment", "sentiment_momentum", "freshness", "event_hour",
]
CROSS_EVENT_FEATURES = ["sentiment_shock"]
SENTIMENT_FEATURES = EVENT_FEATURES + TEMPORAL_FEATURES + CROSS_EVENT_FEATURES


class SentimentFeatureEngineer:
    def compute_event_features(self, article_scores: list[dict]) -> dict:
        """
        Compute the EVENT_FEATURES registry entries from scored articles.

        Each article_score dict has: {label, positive, negative, neutral, confidence}.
        Returns dict with all feature names as keys. Empty input → all zeros.
        """
        if not article_scores:
            return self._zero_event_features()

        n = len(article_scores)
        positives = [s["positive"] for s in article_scores]
        negatives = [s["negative"] for s in article_scores]
        neutrals = [s["neutral"] for s in article_scores]
        confidences = [s["confidence"] for s in article_scores]
        labels = [s["label"] for s in article_scores]
        sentiments = [p - neg for p, neg in zip(positives, negatives)]

        # Basic sentiment
        mean_positive = np.mean(positives)
        mean_negative = np.mean(negatives)
        mean_neutral = np.mean(neutrals)
        mean_sentiment = np.mean(sentiments)
        pct_positive = sum(1 for l in labels if l == "positive") / n
        pct_negative = sum(1 for l in labels if l == "negative") / n

        # Extra features motivated by RoBERTa neutral skew
        non_neutral_ratio = sum(1 for l in labels if l != "neutral") / n
        sentiment_abs_mean = np.mean([abs(s) for s in sentiments])

        # Distribution
        sentiment_variance = float(np.var(sentiments))
        sentiment_range = max(sentiments) - min(sentiments)
        avg_confidence = np.mean(confidences)

        # Volume
        article_count = n
        log_article_count = math.log(1 + n)

        # Agreement
        label_counts = Counter(labels)
        agreement_ratio = max(label_counts.values()) / n

        # Entropy
        proportions = [count / n for count in label_counts.values()]
        entropy = -sum(p * math.log2(p) for p in proportions if p > 0)

        # Kurtosis: heavy-tailed distribution detects single-catalyst events
        if n >= 4:
            s_mean = float(np.mean(sentiments))
            s_std = float(np.std(sentiments, ddof=0))
            if s_std >= 1e-6:
                sentiment_kurtosis = float(np.mean([(s - s_mean) ** 4 for s in sentiments]) / s_std ** 4) - 3.0
            else:
                sentiment_kurtosis = 0.0
        else:
            sentiment_kurtosis = 0.0

        # Fraction of non-neutral FinBERT probability mass that is positive
        dual_polarity_ratio = float(mean_positive / (mean_positive + float(np.mean(negatives)) + 1e-6))

        # Directional label vote imbalance: |pct_positive - pct_negative|
        polarity_strength = abs(pct_positive - pct_negative)

        return {
            "mean_positive": float(mean_positive),
            "mean_negative": float(mean_negative),
            "mean_neutral": float(mean_neutral),
            "mean_sentiment": float(mean_sentiment),
            "pct_positive": pct_positive,
            "pct_negative": pct_negative,
            "non_neutral_ratio": non_neutral_ratio,
            "sentiment_abs_mean": float(sentiment_abs_mean),
            "sentiment_variance": sentiment_variance,
            "sentiment_range": sentiment_range,
            "avg_confidence": float(avg_confidence),
            "article_count": article_count,
            "log_article_count": log_article_count,
            "agreement_ratio": agreement_ratio,
            "entropy": entropy,
            "sentiment_kurtosis": sentiment_kurtosis,
            "dual_polarity_ratio": dual_polarity_ratio,
            "polarity_strength": polarity_strength,
        }

    def compute_temporal_features(
        self, article_scores: list[dict], event_time: str
    ) -> dict:
        """
        Compute the TEMPORAL_FEATURES registry entries from article publish times relative to the event.

        Each article_score dict must include a "date" key (ISO string).
        """
        if not article_scores:
            return self._zero_temporal_features()

        event_dt = from_iso(event_time)
        hours_before = []
        sentiments = []

        for s in article_scores:
            article_dt = from_iso(s["date"])
            delta_hours = (event_dt - article_dt).total_seconds() / 3600
            hours_before.append(max(delta_hours, 0.0))
            sentiments.append(s["positive"] - s["negative"])

        # Recency-weighted sentiment: weight = 0.94^hours_before
        weights = [0.94 ** h for h in hours_before]
        total_weight = sum(weights)
        if total_weight > 0:
            recency_weighted_sentiment = sum(
                w * s for w, s in zip(weights, sentiments)
            ) / total_weight
        else:
            recency_weighted_sentiment = 0.0

        # Early vs late: split at per-event median so both halves are always non-empty
        h_median = float(np.median(hours_before))
        early_sents = [s for s, h in zip(sentiments, hours_before) if h > h_median]
        late_sents = [s for s, h in zip(sentiments, hours_before) if h <= h_median]
        if early_sents and late_sents:
            early_vs_late_sentiment = np.mean(late_sents) - np.mean(early_sents)
        else:
            early_vs_late_sentiment = 0.0

        # Sentiment momentum: slope of sentiment over time
        # Positive slope = sentiment improving as time approaches event
        # We negate hours_before so that "closer to event" = higher x value
        # Require at least 0.1h (~6min) spread to avoid explosion on same-second articles
        time_spread = max(hours_before) - min(hours_before)
        if len(sentiments) >= 2 and time_spread >= 0.1:
            x = [-h for h in hours_before]  # negate so closer-to-event is higher
            coeffs = np.polyfit(x, sentiments, 1)
            sentiment_momentum = float(coeffs[0])
        else:
            sentiment_momentum = 0.0

        # Freshness: hours since most recent article
        freshness = min(hours_before)

        # Hour of event in UTC — proxy for market type (economic releases cluster at 8:30, 14:00, 18:00 UTC)
        event_hour = float(event_dt.hour)

        return {
            "recency_weighted_sentiment": float(recency_weighted_sentiment),
            "early_vs_late_sentiment": float(early_vs_late_sentiment),
            "sentiment_momentum": sentiment_momentum,
            "freshness": freshness,
            "event_hour": event_hour,
        }

    def compute_cross_event_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features that require per-market event ordering.

        Called AFTER extract_features_for_dataset. Groups by market_ticker, sorts
        chronologically, then computes rolling statistics. Uses shift(1) throughout
        so the current event is never included in its own rolling window.

        Returns a new DataFrame with additional cross-event columns appended.
        """
        df = feature_df.copy().sort_values(
            ["market_ticker", "event_timestamp"]
        ).reset_index(drop=True)

        df["sentiment_shock"] = (
            df.groupby("market_ticker", group_keys=False)["mean_sentiment"]
            .transform(lambda s: (
                (s - s.shift(1).rolling(20, min_periods=3).mean()) /
                s.shift(1).rolling(20, min_periods=3).std().clip(lower=1e-6)
            ).fillna(0.0))
        )
        return df

    @staticmethod
    def _zero_event_features() -> dict:
        feats = {k: 0.0 for k in EVENT_FEATURES}
        feats["article_count"] = 0
        return feats

    @staticmethod
    def _zero_temporal_features() -> dict:
        return {k: 0.0 for k in TEMPORAL_FEATURES}

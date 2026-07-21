import re
import logging
from typing import List

logger = logging.getLogger(__name__)

class NLPFraudDetector:
    """
    Multi-signal NLP fraud detector combining:
      1. Heuristic keyword scoring (always active, no dependencies)
      2. BART zero-shot classification for listing text (optional, use_transformers=True)
      3. RoBERTa-based sentiment analysis for review authenticity (optional, use_transformers=True)
    """

    # HuggingFace model identifiers
    _ZS_MODEL   = "facebook/bart-large-mnli"
    _SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(self, use_transformers=False):
        self.use_transformers = use_transformers
        self.zs_classifier   = None   # Zero-shot: listing text fraud detection
        self.sent_pipeline   = None   # Sentiment: review authenticity detection

        self.fraud_labels = [
            "authentic genuine product listing",
            "deceptive counterfeit fake scam listing"
        ]

        # High-risk red flag keywords common in e-commerce fraud/counterfeit listings
        self.red_flag_keywords = [
            "100% original", "guaranteed authentic", "factory price",
            "replica", "aaa quality", "same as original", "free shipping no return",
            "limited time act now", "unclaimed box", "super cheap discount",
            "oem copy", "high clone", "no warranty refund", "urgent sale",
            "overstock sale", "mystery box", "fake", "counterfeit"
        ]

        # Patterns that indicate a review is likely fabricated or incentivised
        self.fake_review_patterns = [
            r"\b(five|5)\s*stars?\b",
            r"\bperfect\b.{0,20}\bperfect\b",
            r"\bgood\s+product\b",
            r"\brecommend\b.{0,30}\beveryone\b",
            r"\b(arrived|came).{0,20}(fast|quick|quickly)\b",
            r"\bexactly\s+as\s+(described|advertised)\b",
        ]

        if self.use_transformers:
            self._load_transformers()

    # ------------------------------------------------------------------
    # Private: Model Loading
    # ------------------------------------------------------------------

    def _load_transformers(self):
        try:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading zero-shot classifier: {self._ZS_MODEL}")
            self.zs_classifier = hf_pipeline(
                "zero-shot-classification",
                model=self._ZS_MODEL
            )
            logger.info("Zero-shot classifier loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load zero-shot classifier ({self._ZS_MODEL}): {e}. "
                "Falling back to rule-based engine for listing scoring."
            )

        try:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading RoBERTa sentiment pipeline: {self._SENT_MODEL}")
            self.sent_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self._SENT_MODEL,
                top_k=None          # Return scores for all labels
            )
            logger.info("RoBERTa sentiment pipeline loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load RoBERTa pipeline ({self._SENT_MODEL}): {e}. "
                "Review authenticity scoring will use heuristic fallback."
            )

    # ------------------------------------------------------------------
    # Public: Listing Text Fraud Score
    # ------------------------------------------------------------------

    def score(self, title: str, description: str = "") -> float:
        """
        Analyses the listing title and description for deceptive signals.
        Returns a fraud probability between 0.0 (clean) and 1.0 (fraudulent).
        """
        text = f"{title}. {description or ''}".lower()

        # 1. Keyword heuristic
        hits = sum(1 for kw in self.red_flag_keywords if kw.lower() in text)
        base_keyword_score = min(hits * 0.15, 0.85)

        # 2. Linguistic features (ALL-CAPS ratio, excessive exclamation marks)
        caps_count  = sum(1 for c in title if c.isupper())
        caps_ratio  = caps_count / max(len(title), 1)
        excl_count  = text.count("!")

        linguistic_boost = 0.0
        if caps_ratio > 0.4:
            linguistic_boost += 0.1
        if excl_count >= 3:
            linguistic_boost += 0.1

        # 3. BART zero-shot transformer score (optional)
        transformer_score = 0.0
        if self.zs_classifier:
            try:
                res = self.zs_classifier(text, self.fraud_labels)
                idx = res['labels'].index("deceptive counterfeit fake scam listing")
                transformer_score = res['scores'][idx]
            except Exception as e:
                logger.error(f"Zero-shot scoring error: {e}")
                transformer_score = base_keyword_score

        # 4. Combine signals
        if self.zs_classifier and transformer_score > 0:
            final_score = (0.6 * transformer_score) + (0.4 * base_keyword_score) + linguistic_boost
        else:
            final_score = base_keyword_score + linguistic_boost

        return round(min(max(final_score, 0.0), 1.0), 3)

    # ------------------------------------------------------------------
    # Public: Review Authenticity Score (RoBERTa)
    # ------------------------------------------------------------------

    def review_score(self, reviews: List[str]) -> float:
        """
        Analyses a list of product reviews to estimate the probability that
        they are fake/manipulated.

        Strategy:
          - Genuine negative reviews are a healthy signal (products with ONLY
            glowing 5-star reviews are suspicious).
          - Repetitive phrasing patterns are flagged by regex heuristics.
          - RoBERTa maps each review into POSITIVE / NEUTRAL / NEGATIVE sentiment.
            A listing where >85 % of reviews are uniformly POSITIVE with no variance
            is flagged as suspicious (review bombing / paid reviews).

        Returns a fake-review probability between 0.0 (legitimate) and 1.0 (suspicious).
        """
        if not reviews:
            return 0.0

        clean_reviews = [r.strip() for r in reviews if r and r.strip()]
        if not clean_reviews:
            return 0.0

        # 1. Heuristic pattern scan
        pattern_hits = 0
        for review in clean_reviews:
            for pat in self.fake_review_patterns:
                if re.search(pat, review.lower()):
                    pattern_hits += 1
                    break   # Count each review at most once

        heuristic_score = min(pattern_hits / max(len(clean_reviews), 1), 1.0)

        # 2. Sentiment distribution via RoBERTa
        roberta_score = self._roberta_review_score(clean_reviews)

        # 3. Blend scores (RoBERTa weighted heavier when available)
        if self.sent_pipeline:
            final = (0.65 * roberta_score) + (0.35 * heuristic_score)
        else:
            final = heuristic_score

        return round(min(max(final, 0.0), 1.0), 3)

    # ------------------------------------------------------------------
    # Private: RoBERTa Sentiment Distribution Analysis
    # ------------------------------------------------------------------

    def _roberta_review_score(self, reviews: List[str]) -> float:
        """
        Uses RoBERTa to analyse sentiment distribution across reviews.
        Returns a suspicion score based on unnatural uniformity.
        """
        if not self.sent_pipeline:
            return 0.0

        positive_count  = 0
        negative_count  = 0
        neutral_count   = 0
        total           = len(reviews)

        for review in reviews:
            try:
                # Truncate long reviews to 512 tokens (RoBERTa limit)
                truncated = review[:512]
                results   = self.sent_pipeline(truncated)[0]

                # results is a list of {label, score} dicts when top_k=None
                label_scores = {r['label'].upper(): r['score'] for r in results}
                dominant     = max(label_scores, key=label_scores.get)

                if dominant == "POSITIVE":
                    positive_count += 1
                elif dominant == "NEGATIVE":
                    negative_count += 1
                else:
                    neutral_count += 1

            except Exception as e:
                logger.warning(f"RoBERTa inference error on review: {e}")
                neutral_count += 1

        if total == 0:
            return 0.0

        positive_ratio = positive_count / total
        negative_ratio = negative_count / total

        # Suspicion signals:
        #   • Overwhelmingly positive (>90%) with near-zero negatives → paid reviews
        #   • Overwhelmingly negative (>90%) → review-bombing campaign
        #   • Perfect uniformity in either direction is unnatural
        if positive_ratio >= 0.90 and negative_ratio <= 0.02:
            # Near-perfect positivity — high suspicion of fake/paid reviews
            suspicion = 0.75 + (positive_ratio - 0.90) * 2.5
        elif negative_ratio >= 0.90 and positive_ratio <= 0.02:
            # Near-total negativity — high suspicion of coordinated attack
            suspicion = 0.65 + (negative_ratio - 0.90) * 2.0
        elif positive_ratio >= 0.75 and negative_ratio <= 0.05:
            # Strongly skewed positive — moderate suspicion
            suspicion = 0.40 + (positive_ratio - 0.75) * 1.4
        else:
            # Healthy sentiment mix — low suspicion
            suspicion = max(0.0, positive_ratio - 0.60) * 0.5

        return round(min(max(suspicion, 0.0), 1.0), 3)

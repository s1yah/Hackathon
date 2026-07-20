import re
import logging

logger = logging.getLogger(__name__)

class NLPFraudDetector:
    def __init__(self, use_transformers=False):
        self.use_transformers = use_transformers
        self.classifier = None
        
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
        
        if self.use_transformers:
            try:
                from transformers import pipeline
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
            except Exception as e:
                logger.warning(f"Could not load HuggingFace pipeline: {e}. Falling back to rule-based engine.")

    def score(self, title: str, description: str = "") -> float:
        """
        Returns a fraud probability score between 0.0 (safe) and 1.0 (fraudulent/deceptive).
        """
        text = f"{title}. {description or ''}".lower()
        
        # 1. Base score calculated via heuristic keywords
        hits = 0
        for kw in self.red_flag_keywords:
            if kw.lower() in text:
                hits += 1
                
        # Base keyword score: 0.15 per hit up to 0.85
        base_keyword_score = min(hits * 0.15, 0.85)
        
        # Additional linguistic features (all-caps ratio, excessive exclamation marks)
        caps_count = sum(1 for c in title if c.isupper())
        title_len = max(len(title), 1)
        caps_ratio = caps_count / title_len
        exclamation_count = text.count("!")
        
        linguistic_boost = 0.0
        if caps_ratio > 0.4:
            linguistic_boost += 0.1
        if exclamation_count >= 3:
            linguistic_boost += 0.1
            
        transformer_score = 0.0
        if self.classifier:
            try:
                res = self.classifier(text, self.fraud_labels)
                idx = res['labels'].index("deceptive counterfeit fake scam listing")
                transformer_score = res['scores'][idx]
            except Exception as e:
                logger.error(f"Error during transformer scoring: {e}")
                transformer_score = base_keyword_score

        # Combine scores
        if self.classifier and transformer_score > 0:
            final_score = (0.6 * transformer_score) + (0.4 * base_keyword_score) + linguistic_boost
        else:
            final_score = base_keyword_score + linguistic_boost

        return round(min(max(final_score, 0.0), 1.0), 3)

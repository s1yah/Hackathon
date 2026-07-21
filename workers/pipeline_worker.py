import os
import logging
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session

from ml.nlp_model import NLPFraudDetector
from ml.pricing_model import PricingAnomalyDetector
from ml.trust_score import SellerTrustScorer
from workers.decision_worker import make_decision
from db.models import Listing, Seller

logger = logging.getLogger(__name__)

# Singleton instances for pipeline execution.
# use_transformers=True activates both BART (listing text) and
# RoBERTa (review authenticity) HuggingFace pipelines.
nlp_detector  = NLPFraudDetector(use_transformers=True)
price_detector = PricingAnomalyDetector()
trust_scorer  = SellerTrustScorer()

# Fit pricing baselines on startup using dataset files if available
MATERIALS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Plan", "materials")
PRICES_CSV     = os.path.join(MATERIALS_DIR, "product_prices.csv")
DEMO_PRICES_CSV = os.path.join(MATERIALS_DIR, "product_prices_demo.csv")

if os.path.exists(PRICES_CSV):
    price_detector.fit_from_csv(PRICES_CSV)
elif os.path.exists(DEMO_PRICES_CSV):
    price_detector.fit_from_csv(DEMO_PRICES_CSV)


def process_listing_sync(listing_id: str, db: Session, reviews: Optional[List[str]] = None):
    """
    Executes full AI inspection on a pending listing:

    1. Score listing text for deceptive language (NLP — keyword + BART)
    2. Score product reviews for authenticity (RoBERTa)
    3. Score price for anomalies (Pricing model)
    4. Calculate Seller Trust & Dynamic Thresholds
    5. Execute Decision Engine (combines all signals)
    6. Persist results to database
    """
    listing = db.query(Listing).filter(Listing.id == listing_id).first()
    if not listing:
        logger.error(f"Listing {listing_id} not found.")
        return None

    seller = db.query(Seller).filter(Seller.id == listing.seller_id).first()
    seller_dict = {
        'account_age_days':    seller.account_age_days    if seller else 0,
        'verification_status': seller.verification_status if seller else 'unverified',
        'total_listings':      seller.total_listings      if seller else 1,
        'flagged_listings':    seller.flagged_listings     if seller else 0,
        'successful_sales':    seller.successful_sales    if seller else 0,
    }

    # 1. NLP Fraud Score — listing title + description
    nlp_score = nlp_detector.score(listing.title, listing.description)

    # 2. RoBERTa Review Authenticity Score
    #    Falls back to 0.0 (neutral) if no reviews were provided.
    review_list  = reviews or []
    review_score = nlp_detector.review_score(review_list)

    # 3. Pricing Anomaly Score
    price_score = price_detector.score(listing.price, listing.category)

    # 4. Seller Trust Score & Dynamic Thresholds
    trust_score = trust_scorer.calculate(seller_dict)
    thresholds  = trust_scorer.get_thresholds(trust_score)

    if seller:
        seller.trust_score = trust_score

    # 5. Make Final Decision — now includes review_score as a signal
    decision = make_decision(nlp_score, price_score, trust_score, thresholds, review_score)

    # 6. Persist all scores and decision to DB
    listing.nlp_flag_score      = nlp_score
    listing.review_score        = review_score
    listing.price_anomaly_score = price_score
    listing.status              = decision
    listing.final_decision      = decision
    listing.decided_at          = datetime.utcnow()

    db.commit()
    db.refresh(listing)

    logger.info(
        f"Processed Listing {listing_id}: "
        f"NLP={nlp_score}, Review={review_score}, Price={price_score}, "
        f"Trust={trust_score} → Decision={decision}"
    )
    return listing

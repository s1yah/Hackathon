import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.nlp_model import NLPFraudDetector
from ml.pricing_model import PricingAnomalyDetector
from ml.trust_score import SellerTrustScorer
from workers.decision_worker import make_decision

def run_inspection_examples():
    print("==========================================================")
    print("🛡️  SHOPEE SENTINEL - AI FRAUD DETECTION EXAMPLE RUNNER  🛡️")
    print("==========================================================")
    print()

    # 1. Initialize models
    # Note: To enable HuggingFace BART Transformer model, set use_transformers=True
    nlp = NLPFraudDetector(use_transformers=False)
    pricing = PricingAnomalyDetector()
    trust = SellerTrustScorer()

    # Fit pricing baselines on dataset file
    materials_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Plan", "materials")
    prices_csv = os.path.join(materials_dir, "product_prices.csv")
    if os.path.exists(prices_csv):
        pricing.fit_from_csv(prices_csv)

    # Test cases
    test_cases = [
        {
            "name": "Case 1: Obvious Counterfeit / Fake Item",
            "title": "100% Original AAA Quality iPhone 15 Pro Max Super Cheap Factory Price No Return",
            "description": "Brand new genuine iPhone 15 pro max copy super cheap factory clearance act now no warranty no refund!",
            "price": 19.99,
            "category": "electronics",
            "seller": {"account_age_days": 2, "verification_status": "new", "total_listings": 3, "flagged_listings": 2, "successful_sales": 0}
        },
        {
            "name": "Case 2: Ambiguous / Suspicious Listing",
            "title": "Refurbished Wireless Noise-Canceling Headphones - Overstock Sale",
            "description": "Good working condition overstock. Minor packaging box damage. Fast delivery guaranteed.",
            "price": 65.00,
            "category": "electronics",
            "seller": {"account_age_days": 45, "verification_status": "unverified", "total_listings": 4, "flagged_listings": 0, "successful_sales": 3}
        },
        {
            "name": "Case 3: Authentic / Legitimate Listing",
            "title": "Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Headphones",
            "description": "Official Sony Flagship Store. Full 1-year official local manufacturer warranty included. Sealed original box.",
            "price": 398.00,
            "category": "electronics",
            "seller": {"account_age_days": 500, "verification_status": "verified", "total_listings": 50, "flagged_listings": 0, "successful_sales": 480}
        }
    ]

    for tc in test_cases:
        print(f"📌 {tc['name']}")
        print(f"   Title:       {tc['title']}")
        print(f"   Price:       ${tc['price']:.2f} (Category: {tc['category']})")
        
        # Calculations
        nlp_score = nlp.score(tc['title'], tc['description'])
        price_score = pricing.score(tc['price'], tc['category'])
        trust_score = trust.calculate(tc['seller'])
        thresholds = trust.get_thresholds(trust_score)
        decision = make_decision(nlp_score, price_score, trust_score, thresholds)

        verdict_badge = "🔴 AUTO-REJECTED" if decision == "rejected" else ("🟡 SUSPENDED (MODERATOR QUEUE)" if decision == "suspended" else "🟢 AUTO-APPROVED")

        print(f"   [NLP Score]:    {nlp_score * 100:.1f}% risk flag")
        print(f"   [Price Anomaly]: {price_score * 100:.1f}% deviation")
        print(f"   [Seller Trust]:  {trust_score:.1f} / 100 ({thresholds['tier']})")
        print(f"   ➡️  VERDICT:     {verdict_badge}")
        print("-" * 58)
        print()

if __name__ == "__main__":
    run_inspection_examples()

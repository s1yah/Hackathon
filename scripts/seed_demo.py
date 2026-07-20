import sys
import os

# Add root project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db.database import init_db, SessionLocal
from db.models import Seller, Listing
from workers.pipeline_worker import process_listing_sync, price_detector

def seed_database():
    print("[INFO] Initializing database schema...")
    init_db()
    db = SessionLocal()

    # Clear existing demo listings if any
    db.query(Listing).delete()
    db.query(Seller).delete()
    db.commit()

    print("[INFO] Fitting pricing model from dataset files in Plan/materials...")
    materials_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Plan", "materials")
    prices_csv = os.path.join(materials_dir, "product_prices.csv")
    demo_csv = os.path.join(materials_dir, "product_prices_demo.csv")

    if os.path.exists(prices_csv):
        price_detector.fit_from_csv(prices_csv)
    elif os.path.exists(demo_csv):
        price_detector.fit_from_csv(demo_csv)

    print("[INFO] Creating demo seller profiles...")
    sellers = [
        Seller(name="Shopee Official Tech Store", verification_status="verified", trust_score=95.0, account_age_days=730, successful_sales=850, total_listings=12),
        Seller(name="Global Gadgets Reseller", verification_status="unverified", trust_score=52.0, account_age_days=90, successful_sales=18, total_listings=5),
        Seller(name="Flash Sale Bargains 99", verification_status="new", trust_score=10.0, account_age_days=1, successful_sales=0, total_listings=3),
    ]
    db.add_all(sellers)
    db.commit()

    official_seller = sellers[0]
    standard_seller = sellers[1]
    scammer_seller = sellers[2]

    print("[INFO] Creating pre-seeded demo listings...")
    sample_listings = [
        # 1. Legitimate Listing -> Auto-Approved
        Listing(
            seller_id=official_seller.id,
            title="Sony WH-1000XM5 Wireless Noise-Canceling Headphones",
            description="Official Sony Flagship Store. 1-Year Local Warranty. Includes carrying case and charging cable.",
            price=398.00,
            category="electronics",
            status="pending"
        ),
        # 2. Ambiguous Listing -> Suspended for Moderator Review
        Listing(
            seller_id=standard_seller.id,
            title="Refurbished Wireless Noise-Canceling Headphones - Overstock Sale",
            description="Good working condition overstock. Minor packaging box damage. Fast 2-day delivery guaranteed.",
            price=65.00,
            category="electronics",
            status="pending"
        ),
        # 3. Obvious Counterfeit/Fake -> Auto-Rejected
        Listing(
            seller_id=scammer_seller.id,
            title="100% Original AAA Quality iPhone 15 Pro Max Super Cheap Factory Price No Return",
            description="Brand new genuine iPhone 15 pro max copy super cheap factory clearance act now no warranty no refund!",
            price=14.99,
            category="electronics",
            status="pending"
        ),
        # 4. Another Ambiguous Listing for Moderator Queue
        Listing(
            seller_id=standard_seller.id,
            title="Designer Brand Luxury Leather Handbag Replica AAA Quality",
            description="Same as original high clone luxury handbag. Soft genuine feeling leather. Limited stock act fast!",
            price=49.99,
            category="fashion",
            status="pending"
        ),
        # 5. Legitimate Fashion Item -> Auto-Approved
        Listing(
            seller_id=official_seller.id,
            title="Uniqlo AIRism Cotton Crew Neck T-Shirt (Unisex)",
            description="100% Authentic Uniqlo AIRism fabric. Smooth cotton exterior with moisture-wicking technology.",
            price=19.90,
            category="fashion",
            status="pending"
        )
    ]

    db.add_all(sample_listings)
    db.commit()

    print("[INFO] Processing initial AI safety inspections...")
    for listing in sample_listings:
        process_listing_sync(listing.id, db)

    db.close()
    print("[SUCCESS] Seed completed successfully! Database ready for demo.")

if __name__ == "__main__":
    seed_database()

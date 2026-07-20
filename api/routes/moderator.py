from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

from db.database import get_db
from db.models import Listing, Seller, ModeratorAction
from api.models import ModeratorDecisionSchema
from workers.pipeline_worker import process_listing_sync, price_detector

router = APIRouter(prefix="/api/moderator", tags=["Moderator"])

@router.get("/queue")
def get_moderator_queue(db: Session = Depends(get_db)):
    """Returns all listings currently suspended for human moderator review."""
    listings = db.query(Listing).filter(Listing.status == "suspended").order_by(Listing.submitted_at.desc()).all()
    
    result = []
    for l in listings:
        seller = db.query(Seller).filter(Seller.id == l.seller_id).first()
        price_stats = price_detector.get_category_stats(l.category)
        
        result.append({
            "id": l.id,
            "seller_id": l.seller_id,
            "seller_name": seller.name if seller else "Unknown Seller",
            "seller_verification": seller.verification_status if seller else "unverified",
            "seller_trust_score": seller.trust_score if seller else 50.0,
            "title": l.title,
            "description": l.description,
            "price": l.price,
            "category": l.category,
            "status": l.status,
            "nlp_flag_score": l.nlp_flag_score,
            "price_anomaly_score": l.price_anomaly_score,
            "category_baseline_price": price_stats.get("median", price_stats.get("mean", 0.0)),
            "submitted_at": l.submitted_at
        })
    return result

@router.get("/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    total = db.query(Listing).count()
    approved = db.query(Listing).filter(Listing.status == "approved").count()
    suspended = db.query(Listing).filter(Listing.status == "suspended").count()
    rejected = db.query(Listing).filter(Listing.status == "rejected").count()
    total_sellers = db.query(Seller).count()
    
    return {
        "total_scanned": total,
        "approved_count": approved,
        "suspended_count": suspended,
        "rejected_count": rejected,
        "total_sellers": total_sellers,
        "approval_rate": round((approved / max(total, 1)) * 100, 1),
        "rejection_rate": round((rejected / max(total, 1)) * 100, 1),
        "suspension_rate": round((suspended / max(total, 1)) * 100, 1)
    }

@router.post("/decision/{listing_id}")
def submit_moderator_decision(listing_id: str, payload: ModeratorDecisionSchema, db: Session = Depends(get_db)):
    listing = db.query(Listing).filter(Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
        
    if payload.action not in ["approved", "rejected"]:
        raise HTTPException(status_code=400, detail="Action must be 'approved' or 'rejected'")

    # Update listing status
    listing.status = payload.action
    listing.final_decision = payload.action
    listing.moderator_reviewed = True
    listing.decided_at = datetime.utcnow()

    # Log moderator action
    mod_action = ModeratorAction(
        listing_id=listing.id,
        moderator_id=payload.moderator_id or "mod_admin",
        action=payload.action,
        notes=payload.notes or ""
    )
    db.add(mod_action)

    # Adjust seller trust score dynamically based on human verdict
    seller = db.query(Seller).filter(Seller.id == listing.seller_id).first()
    if seller:
        if payload.action == "rejected":
            seller.flagged_listings = (seller.flagged_listings or 0) + 1
            seller.trust_score = max(seller.trust_score - 15.0, 0.0)
        elif payload.action == "approved":
            seller.successful_sales = (seller.successful_sales or 0) + 1
            seller.trust_score = min(seller.trust_score + 5.0, 100.0)

    db.commit()
    return {
        "listing_id": listing_id,
        "final_status": payload.action,
        "message": f"Moderator decision recorded: {payload.action.upper()}"
    }

@router.post("/trigger-scenario/{scenario}")
def trigger_demo_scenario(scenario: str, db: Session = Depends(get_db)):
    """
    Creates and processes test listings for Hackathon Demo:
    - scenario 'obvious_fake': Counterfeit keywords, $15 price for electronics, unverified seller.
    - scenario 'ambiguous': Mid-level description, $85 price, standard seller.
    - scenario 'legitimate': Genuine description, $299 price, verified seller.
    """
    scenario = scenario.lower()
    
    if scenario == "obvious_fake":
        seller = db.query(Seller).filter(Seller.name == "Demo_Scammer_Store").first()
        if not seller:
            seller = Seller(name="Demo_Scammer_Store", verification_status="new", trust_score=15.0, account_age_days=2)
            db.add(seller)
            db.commit()
            
        listing = Listing(
            seller_id=seller.id,
            title="100% Original AAA Quality iPhone 15 Pro Max Super Cheap Factory Price No Return",
            description="Brand new genuine iPhone 15 pro max copy super cheap factory clearance act now no warranty no refund!",
            price=19.99,
            category="electronics",
            status="pending"
        )

    elif scenario == "ambiguous":
        seller = db.query(Seller).filter(Seller.name == "Demo_Standard_Seller").first()
        if not seller:
            seller = Seller(name="Demo_Standard_Seller", verification_status="unverified", trust_score=45.0, account_age_days=60)
            db.add(seller)
            db.commit()
            
        listing = Listing(
            seller_id=seller.id,
            title="Refurbished Wireless Noise-Canceling Headphones - Overstock Sale",
            description="Good condition wireless headphones. Minor cosmetic scratches on box. Fast shipping.",
            price=45.00,
            category="electronics",
            status="pending"
        )

    elif scenario == "legitimate":
        seller = db.query(Seller).filter(Seller.name == "Demo_Official_Store").first()
        if not seller:
            seller = Seller(name="Demo_Official_Store", verification_status="verified", trust_score=95.0, account_age_days=500, successful_sales=120)
            db.add(seller)
            db.commit()

        listing = Listing(
            seller_id=seller.id,
            title="Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Headphones",
            description="Official Sony Store. Full 1 year official manufacturer warranty included. Sealed original box.",
            price=398.00,
            category="electronics",
            status="pending"
        )
    else:
        raise HTTPException(status_code=400, detail="Unknown scenario. Choose 'obvious_fake', 'ambiguous', or 'legitimate'")

    db.add(listing)
    seller.total_listings = (seller.total_listings or 0) + 1
    db.commit()
    db.refresh(listing)

    # Process AI inspection sync
    updated_listing = process_listing_sync(listing.id, db)

    return {
        "scenario": scenario,
        "listing_id": updated_listing.id,
        "title": updated_listing.title,
        "price": updated_listing.price,
        "decision": updated_listing.final_decision,
        "nlp_score": updated_listing.nlp_flag_score,
        "price_anomaly_score": updated_listing.price_anomaly_score
    }

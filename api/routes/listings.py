from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import Listing, Seller
from api.models import ListingCreateSchema
from workers.pipeline_worker import process_listing_sync

router = APIRouter(prefix="/api", tags=["Listings"])

@router.post("/submit-listing")
def submit_listing(payload: ListingCreateSchema, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # Verify or auto-create seller if missing
    seller = db.query(Seller).filter(Seller.id == payload.seller_id).first()
    if not seller:
        seller = Seller(
            id=payload.seller_id,
            name=f"Seller_{payload.seller_id[:6]}",
            verification_status="unverified",
            trust_score=50.0,
            account_age_days=15
        )
        db.add(seller)
        db.commit()

    # Create listing entry with status='pending'
    listing = Listing(
        seller_id=payload.seller_id,
        title=payload.title,
        description=payload.description,
        price=payload.price,
        category=payload.category,
        status="pending"
    )
    db.add(listing)
    seller.total_listings = (seller.total_listings or 0) + 1
    db.commit()
    db.refresh(listing)

    # Process inspection in background to avoid blocking API
    background_tasks.add_task(process_listing_sync, listing.id, db)

    return {
        "listing_id": listing.id,
        "status": "pending",
        "message": "Listing submitted successfully and sent for async AI safety inspection."
    }

@router.get("/listings/{listing_id}")
def get_listing(listing_id: str, db: Session = Depends(get_db)):
    listing = db.query(Listing).filter(Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    seller = db.query(Seller).filter(Seller.id == listing.seller_id).first()
    
    return {
        "id": listing.id,
        "seller_id": listing.seller_id,
        "seller_name": seller.name if seller else "Unknown",
        "seller_trust_score": seller.trust_score if seller else 50.0,
        "title": listing.title,
        "description": listing.description,
        "price": listing.price,
        "category": listing.category,
        "status": listing.status,
        "nlp_flag_score": listing.nlp_flag_score,
        "price_anomaly_score": listing.price_anomaly_score,
        "final_decision": listing.final_decision,
        "moderator_reviewed": listing.moderator_reviewed,
        "submitted_at": listing.submitted_at,
        "decided_at": listing.decided_at
    }

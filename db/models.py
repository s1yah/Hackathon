import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from db.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Seller(Base):
    __tablename__ = "sellers"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    verification_status = Column(String(50), default="unverified")  # 'verified', 'unverified', 'new'
    trust_score = Column(Float, default=50.0)
    total_listings = Column(Integer, default=0)
    flagged_listings = Column(Integer, default=0)
    successful_sales = Column(Integer, default=0)
    account_age_days = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    listings = relationship("Listing", back_populates="seller")

class Listing(Base):
    __tablename__ = "listings"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    seller_id = Column(String(36), ForeignKey("sellers.id"), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    price = Column(Float, nullable=False)
    category = Column(String(100), nullable=False)
    status = Column(String(50), default="pending")  # 'pending', 'approved', 'suspended', 'rejected'
    
    nlp_flag_score = Column(Float, nullable=True)
    price_anomaly_score = Column(Float, nullable=True)
    review_score = Column(Float, nullable=True)          # RoBERTa fake-review probability
    final_decision = Column(String(50), nullable=True)
    moderator_reviewed = Column(Boolean, default=False)
    
    submitted_at = Column(DateTime, default=datetime.utcnow)
    decided_at = Column(DateTime, nullable=True)

    seller = relationship("Seller", back_populates="listings")
    moderator_actions = relationship("ModeratorAction", back_populates="listing")

class ModeratorAction(Base):
    __tablename__ = "moderator_actions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    listing_id = Column(String(36), ForeignKey("listings.id"), nullable=False)
    moderator_id = Column(String(100), default="mod_admin")
    action = Column(String(50), nullable=False)  # 'approved', 'rejected'
    notes = Column(Text, nullable=True)
    actioned_at = Column(DateTime, default=datetime.utcnow)

    listing = relationship("Listing", back_populates="moderator_actions")

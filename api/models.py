from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ListingCreateSchema(BaseModel):
    seller_id: str = Field(..., description="ID of seller submitting the listing")
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field("", description="Product description")
    price: float = Field(..., gt=0, description="Listing price in USD or local currency")
    category: str = Field(..., description="Product category")
    reviews: Optional[List[str]] = Field(
        default=[],
        description="Optional list of existing product reviews to analyse for authenticity"
    )

class ModeratorDecisionSchema(BaseModel):
    action: str = Field(..., description="Decision: 'approved' or 'rejected'")
    notes: Optional[str] = Field("", description="Reason or notes for decision")
    moderator_id: Optional[str] = Field("mod_admin", description="ID of the reviewing moderator")

class SellerResponseSchema(BaseModel):
    id: str
    name: str
    verification_status: str
    trust_score: float
    total_listings: int
    flagged_listings: int
    successful_sales: int
    account_age_days: int

class ListingResponseSchema(BaseModel):
    id: str
    seller_id: str
    title: str
    description: Optional[str]
    price: float
    category: str
    status: str
    nlp_flag_score: Optional[float]
    price_anomaly_score: Optional[float]
    review_score: Optional[float]
    final_decision: Optional[str]
    moderator_reviewed: bool
    submitted_at: datetime
    decided_at: Optional[datetime]
    seller: Optional[SellerResponseSchema] = None

    class Config:
        from_attributes = True
